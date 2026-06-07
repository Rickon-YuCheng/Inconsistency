'''
f1 0.74 classic
uv run src/Inconsistency/models/Stage1Tr_quick_bin.py --d_model 256 --enc_layers 1 --lr 1e-4 --weight_decay 1e-4 --dropout 0.3 --epochs 100
f1 0.67 Valence / Arousal
uv run src/Inconsistency/models/Stage1Tr_quick_bin.py --d_model 256 --enc_layers 1 --lr 1e-4 --weight_decay 1e-4 --dropout 0.3 --epochs 100
'''
"""
Stage1Tr-quick_bin.py — segment-level ATEI training (use HuBERT pooled feature)

跟原 Stage1Tr_v1.py 差別:
- audio path 從 HuBERT 改成 HuBERT pooled
- audio 每句只有 [1, 1024],沒有 frame 維度
- ATEI 的 cross-attention 退化成 segment vs token
- 不用 gradient checkpoint / chunk_split (序列短了沒必要)

二元分類版本 (bin):
- import 改 inconsistentLabel_bin
- split 只有 tr/test (官方 dev 當 test),用 test fold 當驗證做 model selection
- feature 讀 HuBERT_pooled_bin / RoBerTa_full_bin
- PseudoLabel / 輸出檔名統一加 _bin
"""
import torch.nn as nn
import wandb
from datetime import datetime
from sklearn.metrics import f1_score
import argparse
from pathlib import Path
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
import numpy as np
from Inconsistency.utils import Timer, set_seed, numpy_random_init
from Inconsistency.datasets.inconsistentLabel_bin import get_stage1_kfold
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)

D_MODEL = 128
NHEAD = 8
LR = 5e-5
EPOCHS = 30
TRANSFORMER_ENC_LAYERS = 1


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--nhead", type=int, default=NHEAD)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--enc_layers", type=int, default=TRANSFORMER_ENC_LAYERS)

    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_batches", type=int, default=None)

    parser.add_argument("--save_dir", type=str, default="weights/stage1-quick-bin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Emotion inconsistency - Stage1 quick bin")
    parser.add_argument("--wandb_name", type=str, default=None)

    parser.add_argument("--kfold", type=int, default=3,
                        help="StratifiedKFold over official train. <=1 disables.")

    return parser.parse_args()


# ============================================================
# Dataset
# ============================================================
class daicwoz_dataset(Dataset):
    """
    每個 patient:
      xa_list: list of [1024]  (HuBERT pooled, pool 過, 每句一個 vector)
      xt_list: list of [L_i, 1024]  (RoBERTa, token-level, 每句長度不同)
    """
    def __init__(self, patient_Idx, depMap):
        self.ds = []
        a_root = Path("datasets/Feature/HuBERT_pooled_bin")     # pooled
        t_root = Path("datasets/Feature/RoBerTa_full_bin")      # token-level

        PseudoLabel = np.load("PseudoLabel_all_distilbert_zdist_q30_70_bin.npz")
        # PseudoLabel = np.load("PseudoLabel_all_va_eci_q30_70_bin.npz")
        # PseudoLabel = np.load("PseudoLabel_all_contrastive_q30_70_bin.npz")
        patientIdx = PseudoLabel["patientIdx"]
        atei_label_arr = PseudoLabel["label"]
        PseudoMap = {int(x): int(y) for x, y in zip(patientIdx, atei_label_arr)}

        for p in patient_Idx:
            if p not in PseudoMap:
                continue
            a_path = a_root / f"{p}_acoustic.pt"
            t_path = t_root / f"{p}_text.pt"
            assert a_path.exists() and t_path.exists(), f"ds error: {p}"

            dep_label = depMap[p]
            atei_label = PseudoMap[p]
            self.ds.append((p, atei_label, dep_label, a_path, t_path))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        Patient, PseudoL, DepL, a_path, t_path = self.ds[index]

        xa = torch.load(str(a_path), map_location="cpu")
        xt = torch.load(str(t_path), map_location="cpu")

        # xa 每個 element 是 [1, 1024],squeeze 成 [1024]
        xa_list = [x.squeeze(0) for x in xa]
        # xt 每個 element 是 [1, L, 1024],squeeze 成 [L, 1024]
        xt_list = [x.squeeze(0) for x in xt]

        atei_label = torch.tensor(PseudoL, dtype=torch.long)
        dep_label = torch.tensor(DepL, dtype=torch.long)
        return xa_list, xt_list, atei_label, dep_label, Patient


def collate_fn(batch):
    """
    Stage1 collate (batch_size=1 per patient):
    - xa: [num_seg, 1024]                  ← 每句一個 vector (segment-level)
    - xt: [num_seg, max_L_t, 1024]         ← 每句多個 token (token-level)
    - aMask: None  (audio 沒有 padding,每句固定 1 個 vector)
    - tMask: [num_seg, max_L_t]
    """
    batch = [item for item in batch if item is not None]
    xa_list, xt_list, pseudoL, dep_label, Patient = batch[0]

    # audio: 每句 [1024] -> stack 成 [num_seg, 1024]
    xa = torch.stack(xa_list, dim=0)  # [num_seg, 1024]

    # text: pad 不同長度的句子 -> [num_seg, max_L_t, 1024]
    xt = pad_sequence(xt_list, batch_first=True)
    tMask = (xt.sum(dim=-1) == 0)

    return xa, xt, tMask, pseudoL, dep_label, Patient


# ============================================================
# Model
# ============================================================
class atei(nn.Module):
    """
    Segment-level ATEI:
    - audio: 每句一個 [D] vector,直接當 query
    - text:  每句多個 token [L, D],當 K/V
    - cross-attn: audio (Q, seq_len=1) attend to text (K/V, seq_len=L)
    """
    def __init__(self, embd_size, nheads, inp_dim=1024, dropout=0.4,
                 enc_layers=1):
        super().__init__()
        assert embd_size % nheads == 0

        self.a_in_proj = nn.Sequential(nn.Linear(inp_dim, embd_size),
                                       nn.LayerNorm(embd_size))
        self.t_in_proj = nn.Sequential(nn.Linear(inp_dim, embd_size),
                                       nn.LayerNorm(embd_size))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embd_size, nhead=nheads, batch_first=True,
            dim_feedforward=4 * embd_size, dropout=dropout,
        )
        # 注意:audio 是 segment-level,自己的 Transformer 跑在「num_seg 維度」上
        # (整個 patient 的所有句子當一個序列來看)
        self.a_transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        # text 的 Transformer 跑在「token 維度」上(每句獨立做 self-attn)
        t_enc_layer = nn.TransformerEncoderLayer(
            d_model=embd_size, nhead=nheads, batch_first=True,
            dim_feedforward=4 * embd_size, dropout=dropout,
        )
        self.t_transformer_enc = nn.TransformerEncoder(t_enc_layer, num_layers=enc_layers)

        self.Cross_Attn = at_cross_attn(embd_size)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(4 * embd_size, embd_size)
        self.fc2 = nn.Linear(embd_size, embd_size)
        self.fc3 = nn.Linear(embd_size, embd_size)
        self.oup = nn.Linear(embd_size, 2)
        self.patient_oup = nn.Linear(embd_size, 2)

    def forward(self, xa, xt, tMask=None):
        """
        xa: [num_seg, 1024]              ← segment-level audio
        xt: [num_seg, max_L_t, 1024]     ← token-level text
        tMask: [num_seg, max_L_t]
        """
        xa = self.a_in_proj(xa)   # [num_seg, D]
        xt = self.t_in_proj(xt)   # [num_seg, max_L_t, D]

        # --- audio self-attention on segment dimension ---
        # 整個 patient 所有句子當一個序列: add batch dim -> [1, num_seg, D]
        XprimeA = self.a_transformer_enc(xa.unsqueeze(0)).squeeze(0)  # [num_seg, D]

        # --- text self-attention on token dimension ---
        # 每句獨立: [num_seg, max_L_t, D],把 num_seg 當 batch
        XprimeT = self.t_transformer_enc(xt, src_key_padding_mask=tMask)  # [num_seg, max_L_t, D]

        # --- cross-attention ---
        # Xat: audio attend to text per sentence -> [num_seg, D]
        # Xta: text attend to audio per sentence -> [num_seg, max_L_t, D]
        Xat, Xta = self.Cross_Attn(XprimeA, XprimeT, tMask)

        # --- pooling 成 segment-level feature ---
        avgXprimeA = XprimeA                                 # [num_seg, D]
        avgXat = Xat                                         # [num_seg, D]
        avgXta = self.maskMean(Xta, tMask)                   # [num_seg, D]
        avgXprimeT = self.maskMean(XprimeT, tMask)           # [num_seg, D]

        hE = torch.cat((avgXprimeA, avgXat, avgXta, avgXprimeT), dim=1)  # [num_seg, 4D]

        Fc1 = self.dropout(F.relu(self.fc1(hE)))
        Fc2 = self.dropout(F.relu(self.fc2(Fc1)))
        Fc3 = self.fc3(Fc2)
        Oup = self.oup(Fc3)                                  # [num_seg, 2]

        return Fc3, Oup  # Fc3: [num_seg, D], Oup: [num_seg, 2]

    def maskMean(self, inp, mask):
        if mask is None:
            return inp.mean(dim=1)
        valid = (~mask).unsqueeze(-1).float()
        s = (inp * valid).sum(dim=1)
        Len = valid.sum(dim=1).clamp(min=1.0)
        return s / Len


class at_cross_attn(nn.Module):
    """
    audio (Q, seq=1) attend to text tokens (K/V, seq=L)
    text  (Q, seq=L) attend to audio segment (K/V, seq=1)
    """
    def __init__(self, embd_size):
        super().__init__()
        self.at_Q = nn.Linear(embd_size, embd_size)
        self.at_K = nn.Linear(embd_size, embd_size)
        self.at_V = nn.Linear(embd_size, embd_size)
        self.ta_Q = nn.Linear(embd_size, embd_size)
        self.ta_K = nn.Linear(embd_size, embd_size)
        self.ta_V = nn.Linear(embd_size, embd_size)

    def forward(self, XprimeA, XprimeT, tMask=None):
        """
        XprimeA: [num_seg, D]              ← audio,每句一個 vector
        XprimeT: [num_seg, max_L_t, D]     ← text,每句多個 token
        tMask:   [num_seg, max_L_t]
        """
        # audio 加一個序列維度,變 [num_seg, 1, D]
        XA_q = XprimeA.unsqueeze(1)  # [num_seg, 1, D]

        # at: audio attend to text
        Qa = self.at_Q(XA_q)       # [num_seg, 1, D]
        Kt = self.at_K(XprimeT)    # [num_seg, L, D]
        Vt = self.at_V(XprimeT)
        Xat = cross_attn(Qa, Kt, Vt, tMask).squeeze(1)  # [num_seg, D]

        # ta: text attend to audio
        Qt = self.ta_Q(XprimeT)    # [num_seg, L, D]
        Ka = self.ta_K(XA_q)       # [num_seg, 1, D]
        Va = self.ta_V(XA_q)
        Xta = cross_attn(Qt, Ka, Va, mask=None)  # [num_seg, L, D]   audio 沒 padding

        return Xat, Xta


def cross_attn(Q, K, V, mask=None):
    """
    Q: [N, Lq, D]
    K: [N, Lk, D]
    V: [N, Lk, D]
    mask: [N, Lk]  (True = padding)
    """
    Q = Q.unsqueeze(1)  # [N, 1, Lq, D]
    K = K.unsqueeze(1)
    V = V.unsqueeze(1)

    attn_mask = None
    if mask is not None:
        attn_mask = (~mask).view(mask.size(0), 1, 1, mask.size(1))

    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
    return out.squeeze(1)  # [N, Lq, D]


# ============================================================
# Train / Val
# ============================================================
def run_one_fold(fold_id, train_ids, val_ids, depMap, run_id, device):
    set_seed(ARGS.seed + fold_id)

    run_name = (ARGS.wandb_name + f"_fold{fold_id}") if ARGS.wandb_name else (
        f"stage1_quick_bin_seed{ARGS.seed}_lr{LR:.0e}_wd{ARGS.weight_decay:.0e}"
        f"_do{ARGS.dropout:.2f}_d{D_MODEL}_l{TRANSFORMER_ENC_LAYERS}_fold{fold_id}_{run_id}"
    )

    save_dir = Path(ARGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    g = torch.Generator()
    g.manual_seed(ARGS.seed + fold_id)

    best_val_f1 = -1.0
    patience = 50
    no_improve = 0

    trDS = daicwoz_dataset(train_ids, depMap)
    valDS = daicwoz_dataset(val_ids, depMap)

    print(f"\n{'='*60}\nFOLD {fold_id}\n{'='*60}")
    print(f"[Train] {len(trDS)} samples")
    print(f"[Val]   {len(valDS)} samples")

    if ARGS.use_wandb:
        wandb.init(
            project=ARGS.wandb_project,
            name=run_name,
            reinit=True,
            config={
                "seed": ARGS.seed, "fold": fold_id,
                "d_model": D_MODEL, "nhead": NHEAD,
                "lr": LR, "epochs": EPOCHS, "enc_layers": TRANSFORMER_ENC_LAYERS,
                "dropout": ARGS.dropout, "weight_decay": ARGS.weight_decay,
                "label_smoothing": ARGS.label_smoothing,
                "train_samples": len(trDS), "val_samples": len(valDS),
                "audio_feature": "HuBERT_pooled_bin (pooled)",
                "text_feature": "RoBerTa_full_bin (token-level)",
                "atei_level": "segment-level",
            },
        )

    tr_loader = DataLoader(trDS, collate_fn=collate_fn, shuffle=True,
                           generator=g, worker_init_fn=numpy_random_init,
                           num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(valDS, collate_fn=collate_fn, shuffle=False,
                            generator=g, worker_init_fn=numpy_random_init,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    model = atei(D_MODEL, NHEAD, dropout=ARGS.dropout,
                 enc_layers=TRANSFORMER_ENC_LAYERS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR,
                           weight_decay=ARGS.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=ARGS.label_smoothing)
    scaler = torch.GradScaler('cuda')

    history = []

    for epoch in range(EPOCHS):
        model.train()
        totLoss, correct, n = 0.0, 0, 0

        pbar = tqdm(tr_loader, f"Fold {fold_id} Train Epoch {epoch+1}/{EPOCHS}",
                    unit="patient", leave=False)

        for batch_idx, data in enumerate(pbar):
            if ARGS.max_batches is not None and batch_idx >= ARGS.max_batches:
                break
            xa, xt, tMask, atei_label, dep_label, Patient = data
            xa, xt, tMask = xa.to(device), xt.to(device), tMask.to(device)
            atei_label = atei_label.to(device)

            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                                dtype=torch.bfloat16):
                feat, logits = model(xa, xt, tMask)
                # patient-level: 把所有 segment 的 feature mean 起來
                patient_feat = feat.mean(dim=0)              # [D]
                patient_logit = model.patient_oup(patient_feat)
                loss = criterion(patient_logit.unsqueeze(0),
                                 atei_label.unsqueeze(0))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pred = patient_logit.argmax(dim=-1)
            correct += int(pred.item() == atei_label.item())
            totLoss += loss.item()
            n += 1

            pbar.set_postfix({
                "loss": totLoss / max(n, 1),
                "acc": correct / max(n, 1),
                "patient": Patient,
            })

        train_loss = totLoss / max(n, 1)
        train_acc = correct / max(n, 1)
        val_result = validate(model, val_loader, criterion, device)
        history.append(train_loss)

        print(
            f"[Fold {fold_id}] Epoch [{epoch+1}/{EPOCHS}] | "
            f"Tr Loss: {train_loss:.4f} | Tr Acc: {train_acc:.4f} | "
            f"Val Loss: {val_result['loss']:.4f} | Val Acc: {val_result['acc']:.4f} | "
            f"Val MacroF1: {val_result['macro_f1']:.4f}"
        )
        print("Val Label counts:",
              np.bincount(val_result["y_true"], minlength=2))
        print("Val Pred counts :",
              np.bincount(val_result["y_pred"], minlength=2))
        print(val_result["cm"])
        print(classification_report(
            val_result["y_true"], val_result["y_pred"],
            labels=[0, 1],
            target_names=["inconsistency(0)", "consistency(1)"],
            digits=4, zero_division=0,
        ))

        if val_result["macro_f1"] > best_val_f1:
            best_val_f1 = val_result["macro_f1"]
            no_improve = 0
            ckpt_name = (
                f"stage1-quick-bin_{run_id}_seed{ARGS.seed}_fold{fold_id}_"
                f"f1{best_val_f1:.4f}_ep{epoch+1:03d}_"
                f"lr{LR:.0e}_wd{ARGS.weight_decay:.0e}_"
                f"d{D_MODEL}_l{TRANSFORMER_ENC_LAYERS}.pt"
            )
            ckpt_path = save_dir / ckpt_name
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "fold": fold_id,
                "best_val_f1": best_val_f1,
                "val_acc": val_result["acc"],
                "args": vars(ARGS),
                "d_model": D_MODEL, "nhead": NHEAD,
                "enc_layers": TRANSFORMER_ENC_LAYERS,
                "dropout": ARGS.dropout,
                "audio_feature": "HuBERT_pooled_bin",
                "text_feature": "RoBerTa_full_bin",
                "atei_level": "segment",
            }, ckpt_path)
            print(f"[Save Best] Fold {fold_id} Val MacroF1: {best_val_f1:.4f} -> {ckpt_path}")
        else:
            no_improve += 1
            print(f"[EarlyStopping] no improvement: {no_improve}/{patience}")

        if ARGS.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss, "train/acc": train_acc,
                "val/loss": val_result["loss"], "val/acc": val_result["acc"],
                "val/macro_f1": val_result["macro_f1"],
                "best/val_macro_f1": best_val_f1,
                "no_improve": no_improve,
            })

        if no_improve >= patience:
            print(f"[EarlyStopping] Fold {fold_id} stop at epoch {epoch+1}. "
                  f"Best Val MacroF1: {best_val_f1:.4f}")
            break

    imsave(history, out_path=f"stage1-quick_tr_loss_bin_fold{fold_id}.jpg")
    if ARGS.use_wandb:
        wandb.finish()

    return best_val_f1


def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_timer = Timer()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    depMap, folds = get_stage1_kfold(n_splits=max(ARGS.kfold, 2), seed=ARGS.seed)
    if ARGS.kfold <= 1:
        folds = folds[:1]   # 只跑第一折,等同單次

    fold_f1s = []
    for f in folds:
        f1 = run_one_fold(f["fold"], f["train"], f["val"], depMap, run_id, device)
        fold_f1s.append(f1)
        print(f"\n>>> Fold {f['fold']} best Val MacroF1: {f1:.4f}")

    print("\n" + "=" * 60)
    print("K-FOLD RESULT (Stage1)")
    print("=" * 60)
    for i, f1 in enumerate(fold_f1s):
        print(f"Fold {i}: {f1:.4f}")
    print(f"\nMean MacroF1: {np.mean(fold_f1s):.4f}")
    print(f"Std  MacroF1: {np.std(fold_f1s):.4f}")
    print(f"Total time: {total_timer}")


@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()
    totLoss, correct, n = 0.0, 0, 0
    all_y_true, all_y_pred = [], []

    pbar = tqdm(loader, desc="Validation", unit="patient", leave=False)
    for data in pbar:
        xa, xt, tMask, atei_label, dep_label, Patient = data
        xa, xt, tMask = xa.to(device), xt.to(device), tMask.to(device)
        atei_label = atei_label.to(device)

        with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                            dtype=torch.bfloat16):
            feat, logits = model(xa, xt, tMask)
            patient_feat = feat.mean(dim=0)
            patient_logit = model.patient_oup(patient_feat)
            loss = criterion(patient_logit.unsqueeze(0),
                             atei_label.unsqueeze(0))

        pred = patient_logit.argmax(dim=-1)
        correct += int(pred.item() == atei_label.item())
        totLoss += loss.item()
        n += 1
        all_y_true.append(int(atei_label.item()))
        all_y_pred.append(int(pred.item()))

    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    macro_f1 = f1_score(all_y_true, all_y_pred, labels=[0, 1],
                        average="macro", zero_division=0)

    return {
        "loss": totLoss / max(n, 1),
        "acc": correct / max(n, 1),
        "macro_f1": macro_f1,
        "y_true": np.array(all_y_true),
        "y_pred": np.array(all_y_pred),
        "cm": cm,
    }


def imsave(history, out_path="stage1-quick_tr_loss_bin.jpg"):
    fig, ax = plt.subplots()
    ax.plot(range(1, len(history) + 1), history)
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropyLoss')
    plt.title('Training Loss (Stage1-quick-bin)')
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    ARGS = parse_args()
    D_MODEL = ARGS.d_model
    NHEAD = ARGS.nhead
    LR = ARGS.lr
    EPOCHS = ARGS.epochs
    TRANSFORMER_ENC_LAYERS = ARGS.enc_layers
    main()