''' f1 mean 0.69 std 0.02
uv run src/Inconsistency/models/Stage1_daic_eatd.py --d_model 256 --enc_layers 1 --lr 1e-4 --weight_decay 1e-4 --dropout 0.3 --epochs 100'''
'''
Stage1_daic_eatd.py — joint DAIC-WOZ + EATD Stage1 ATEI training

原版 Stage1Tr_quick_bin.py 的差異:
- feature 路徑: datasets/Feat_daic_eatd/
- pseudo label: datasets/Feat_daic_eatd/PseudoLabel_joint_train_q30_70.npz
- id 格式: daic_{pid} (int) / eatd_{vol} (str, e.g. t_1)
- kfold: 只在 DAIC-train 內部切，EATD-train 全部固定在 train pool
- val: 永遠是純 DAIC kfold val（不混 EATD），metrics 可比
- ckpt save_dir: weights/stage1-daic-eatd
'''
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
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Inconsistency.utils import Timer, set_seed, numpy_random_init
from Inconsistency.datasets.inconsistentLabel_bin import get_stage1_kfold, get_Split_and_GroundTrue
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)

FEAT_DIR   = Path("datasets/Feat_daic_eatd")
PSEUDO_NPZ = FEAT_DIR / "PseudoLabel_joint_train_q30_70.npz"

D_MODEL = 128
NHEAD = 8
LR = 5e-5
EPOCHS = 30
TRANSFORMER_ENC_LAYERS = 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model",      type=int,   default=D_MODEL)
    parser.add_argument("--nhead",        type=int,   default=NHEAD)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--epochs",       type=int,   default=EPOCHS)
    parser.add_argument("--enc_layers",   type=int,   default=TRANSFORMER_ENC_LAYERS)
    parser.add_argument("--dropout",      type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_batches",  type=int,   default=None)
    parser.add_argument("--save_dir",     type=str,   default="weights/stage1-daic-eatd")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--use_wandb",    action="store_true")
    parser.add_argument("--wandb_project", type=str,  default="Stage1 daic+eatd")
    parser.add_argument("--wandb_name",   type=str,   default=None)
    parser.add_argument("--kfold",        type=int,   default=3,
                        help="StratifiedKFold over DAIC-train only. <=1 disables.")
    parser.add_argument("--low_q",        type=float, default=30)
    parser.add_argument("--high_q",       type=float, default=70)
    return parser.parse_args()


# ============================================================
# Split helpers
# ============================================================
def get_joint_stage1_folds(n_splits: int = 3, seed: int = 42, low_q=30, high_q=70):
    """
    DAIC-train 內部做 StratifiedKFold。
    EATD-train 全部固定加入每個 fold 的 train set。
    Val 永遠是純 DAIC kfold val。

    Returns
    -------
    daic_depMap : {int pid: dep_label}
    eatd_depMap : {str vol: dep_label}  (from pseudo label npz)
    eatd_train_ids : list of str  (e.g. ['eatd_t_1', ...])
    folds : list of {"fold", "daic_train", "daic_val"}
    """
    # DAIC kfold (官方 train 內部)
    daic_depMap, daic_folds = get_stage1_kfold(n_splits=n_splits, seed=seed)

    # EATD train ids from joint pseudo label npz
    pseudo = np.load(PSEUDO_NPZ, allow_pickle=True)
    all_ids = pseudo["patientIdx"]  # e.g. ['daic_300', 'eatd_t_1', ...]

    eatd_train_ids = [str(i) for i in all_ids if str(i).startswith("eatd_")]

    # dep label for EATD from npz
    dep_arr = pseudo["dep_label"]
    eatd_depMap = {str(idx): int(dep)
                   for idx, dep in zip(all_ids, dep_arr)
                   if str(idx).startswith("eatd_")}

    return daic_depMap, eatd_depMap, eatd_train_ids, daic_folds


# ============================================================
# Dataset
# ============================================================
class joint_dataset(Dataset):
    """
    吃 DAIC + EATD 混合樣本。
    id 格式:
      DAIC -> "daic_{pid}"  (pid is int, feature file: {pid}_acoustic.pt)
      EATD -> "eatd_{vol}"  (vol is str like t_1, feature file: t_1_acoustic.pt)
    """
    def __init__(self, joint_ids: list, daic_depMap: dict, eatd_depMap: dict):
        self.ds = []

        pseudo   = np.load(PSEUDO_NPZ, allow_pickle=True)
        PseudoMap = {str(k): int(v)
                     for k, v in zip(pseudo["patientIdx"], pseudo["label"])}

        for jid in joint_ids:
            if jid not in PseudoMap:
                continue

            if jid.startswith("daic_"):
                pid      = int(jid.split("_", 1)[1])
                dep      = daic_depMap[pid]
                a_path   = FEAT_DIR / f"{pid}_acoustic.pt"
                t_path   = FEAT_DIR / f"{pid}_text.pt"
            else:                          # eatd_t_1 / eatd_v_2
                vol      = jid.split("_", 1)[1]   # e.g. "t_1"
                dep      = eatd_depMap[jid]
                a_path   = FEAT_DIR / f"{vol}_acoustic.pt"
                t_path   = FEAT_DIR / f"{vol}_text.pt"

            if not (a_path.exists() and t_path.exists()):
                print(f"[Dataset] feature missing for {jid}, skip")
                continue

            self.ds.append((jid, PseudoMap[jid], dep, a_path, t_path))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        jid, atei_l, dep_l, a_path, t_path = self.ds[index]

        xa = torch.load(str(a_path), map_location="cpu")
        xt = torch.load(str(t_path), map_location="cpu")

        xa_list = [x.squeeze(0) for x in xa]   # list of [1024]
        xt_list = [x.squeeze(0) for x in xt]   # list of [L_i, 1024]

        return xa_list, xt_list, \
               torch.tensor(atei_l, dtype=torch.long), \
               torch.tensor(dep_l,  dtype=torch.long), \
               jid


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    xa_list, xt_list, pseudoL, dep_label, jid = batch[0]

    xa   = torch.stack(xa_list, dim=0)          # [num_seg, 1024]
    xt   = pad_sequence(xt_list, batch_first=True)
    tMask = (xt.sum(dim=-1) == 0)

    return xa, xt, tMask, pseudoL, dep_label, jid


# ============================================================
# Model  (identical to Stage1Tr_quick_bin)
# ============================================================
class atei(nn.Module):
    def __init__(self, embd_size, nheads, inp_dim=1024, dropout=0.4, enc_layers=1):
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
        self.a_transformer_enc = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        t_enc_layer = nn.TransformerEncoderLayer(
            d_model=embd_size, nhead=nheads, batch_first=True,
            dim_feedforward=4 * embd_size, dropout=dropout,
        )
        self.t_transformer_enc = nn.TransformerEncoder(t_enc_layer, num_layers=enc_layers)

        self.Cross_Attn  = at_cross_attn(embd_size)
        self.dropout     = nn.Dropout(dropout)
        self.fc1         = nn.Linear(4 * embd_size, embd_size)
        self.fc2         = nn.Linear(embd_size, embd_size)
        self.fc3         = nn.Linear(embd_size, embd_size)
        self.oup         = nn.Linear(embd_size, 2)
        self.patient_oup = nn.Linear(embd_size, 2)

    def forward(self, xa, xt, tMask=None):
        xa = self.a_in_proj(xa)
        xt = self.t_in_proj(xt)

        XprimeA = self.a_transformer_enc(xa.unsqueeze(0)).squeeze(0)
        XprimeT = self.t_transformer_enc(xt, src_key_padding_mask=tMask)

        Xat, Xta = self.Cross_Attn(XprimeA, XprimeT, tMask)

        avgXprimeA = XprimeA
        avgXat     = Xat
        avgXta     = self.maskMean(Xta, tMask)
        avgXprimeT = self.maskMean(XprimeT, tMask)

        hE   = torch.cat((avgXprimeA, avgXat, avgXta, avgXprimeT), dim=1)
        Fc1  = self.dropout(F.relu(self.fc1(hE)))
        Fc2  = self.dropout(F.relu(self.fc2(Fc1)))
        Fc3  = self.fc3(Fc2)
        Oup  = self.oup(Fc3)
        return Fc3, Oup

    def maskMean(self, inp, mask):
        if mask is None:
            return inp.mean(dim=1)
        valid = (~mask).unsqueeze(-1).float()
        s     = (inp * valid).sum(dim=1)
        Len   = valid.sum(dim=1).clamp(min=1.0)
        return s / Len


class at_cross_attn(nn.Module):
    def __init__(self, embd_size):
        super().__init__()
        self.at_Q = nn.Linear(embd_size, embd_size)
        self.at_K = nn.Linear(embd_size, embd_size)
        self.at_V = nn.Linear(embd_size, embd_size)
        self.ta_Q = nn.Linear(embd_size, embd_size)
        self.ta_K = nn.Linear(embd_size, embd_size)
        self.ta_V = nn.Linear(embd_size, embd_size)

    def forward(self, XprimeA, XprimeT, tMask=None):
        XA_q = XprimeA.unsqueeze(1)
        Qa   = self.at_Q(XA_q)
        Kt   = self.at_K(XprimeT)
        Vt   = self.at_V(XprimeT)
        Xat  = cross_attn(Qa, Kt, Vt, tMask).squeeze(1)

        Qt   = self.ta_Q(XprimeT)
        Ka   = self.ta_K(XA_q)
        Va   = self.ta_V(XA_q)
        Xta  = cross_attn(Qt, Ka, Va, mask=None)
        return Xat, Xta


def cross_attn(Q, K, V, mask=None):
    Q = Q.unsqueeze(1)
    K = K.unsqueeze(1)
    V = V.unsqueeze(1)
    attn_mask = None
    if mask is not None:
        attn_mask = (~mask).view(mask.size(0), 1, 1, mask.size(1))
    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
    return out.squeeze(1)


# ============================================================
# Train / Val
# ============================================================
def run_one_fold(fold_id, daic_train_ids, daic_val_ids, eatd_train_ids,
                 daic_depMap, eatd_depMap, run_id, device):
    set_seed(ARGS.seed + fold_id)

    # joint train = DAIC-train-fold + all EATD-train
    joint_train_ids = [f"daic_{p}" for p in daic_train_ids] + eatd_train_ids
    daic_val_jids   = [f"daic_{p}" for p in daic_val_ids]

    trDS  = joint_dataset(joint_train_ids, daic_depMap, eatd_depMap)
    valDS = joint_dataset(daic_val_jids,   daic_depMap, eatd_depMap)

    print(f"\n{'='*60}\nFOLD {fold_id}\n{'='*60}")
    print(f"[Train] {len(trDS)} samples  "
          f"(DAIC={sum(1 for d in trDS.ds if d[0].startswith('daic_'))} "
          f"EATD={sum(1 for d in trDS.ds if d[0].startswith('eatd_'))})")
    print(f"[Val]   {len(valDS)} samples  (DAIC only)")

    save_dir = Path(ARGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    g = torch.Generator()
    g.manual_seed(ARGS.seed + fold_id)

    tr_loader  = DataLoader(trDS,  collate_fn=collate_fn, shuffle=True,
                            generator=g, worker_init_fn=numpy_random_init,
                            num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(valDS, collate_fn=collate_fn, shuffle=False,
                            generator=g, worker_init_fn=numpy_random_init,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    run_name = (ARGS.wandb_name + f"_fold{fold_id}") if ARGS.wandb_name else (
        f"stage1_daic_eatd_seed{ARGS.seed}_lr{LR:.0e}_wd{ARGS.weight_decay:.0e}"
        f"_do{ARGS.dropout:.2f}_d{D_MODEL}_l{TRANSFORMER_ENC_LAYERS}_fold{fold_id}_{run_id}"
    )
    if ARGS.use_wandb:
        wandb.init(
            project=ARGS.wandb_project, name=run_name, reinit=True,
            config={
                "seed": ARGS.seed, "fold": fold_id,
                "d_model": D_MODEL, "nhead": NHEAD,
                "lr": LR, "epochs": EPOCHS, "enc_layers": TRANSFORMER_ENC_LAYERS,
                "dropout": ARGS.dropout, "weight_decay": ARGS.weight_decay,
                "label_smoothing": ARGS.label_smoothing,
                "train_samples": len(trDS), "val_samples": len(valDS),
                "audio_feature": "wav2vec2-xls-r-300m (pooled)",
                "text_feature": "xlm-roberta-large (token-level)",
            },
        )

    model     = atei(D_MODEL, NHEAD, dropout=ARGS.dropout,
                     enc_layers=TRANSFORMER_ENC_LAYERS).to(device)
    opt       = torch.optim.Adam(model.parameters(), lr=LR,
                                 weight_decay=ARGS.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=ARGS.label_smoothing)
    scaler    = torch.GradScaler('cuda')

    best_val_f1 = -1.0
    patience    = 50
    no_improve  = 0
    history     = []

    for epoch in range(EPOCHS):
        model.train()
        totLoss, correct, n = 0.0, 0, 0

        pbar = tqdm(tr_loader, f"Fold {fold_id} Epoch {epoch+1}/{EPOCHS}",
                    unit="sample", leave=False)
        for batch_idx, data in enumerate(pbar):
            if ARGS.max_batches is not None and batch_idx >= ARGS.max_batches:
                break
            xa, xt, tMask, atei_label, dep_label, jid = data
            xa, xt, tMask = xa.to(device), xt.to(device), tMask.to(device)
            atei_label    = atei_label.to(device)

            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                                dtype=torch.bfloat16):
                feat, _        = model(xa, xt, tMask)
                patient_feat   = feat.mean(dim=0)
                patient_logit  = model.patient_oup(patient_feat)
                loss           = criterion(patient_logit.unsqueeze(0),
                                           atei_label.unsqueeze(0))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            pred     = patient_logit.argmax(dim=-1)
            correct += int(pred.item() == atei_label.item())
            totLoss += loss.item()
            n       += 1
            pbar.set_postfix({"loss": totLoss/max(n,1), "acc": correct/max(n,1)})

        train_loss = totLoss / max(n, 1)
        train_acc  = correct / max(n, 1)
        val_result = validate(model, val_loader, criterion, device)
        history.append(train_loss)

        print(f"[Fold {fold_id}] Ep {epoch+1}/{EPOCHS} | "
              f"Tr Loss {train_loss:.4f} Acc {train_acc:.4f} | "
              f"Val Loss {val_result['loss']:.4f} Acc {val_result['acc']:.4f} "
              f"MacroF1 {val_result['macro_f1']:.4f}")
        print(classification_report(
            val_result["y_true"], val_result["y_pred"],
            labels=[0, 1], target_names=["inconsistency(0)", "consistency(1)"],
            digits=4, zero_division=0,
        ))

        if val_result["macro_f1"] > best_val_f1:
            best_val_f1 = val_result["macro_f1"]
            no_improve  = 0
            ckpt_name   = (
                f"stage1-daic-eatd_{run_id}_seed{ARGS.seed}_fold{fold_id}_"
                f"f1{best_val_f1:.4f}_ep{epoch+1:03d}_"
                f"lr{LR:.0e}_wd{ARGS.weight_decay:.0e}_"
                f"d{D_MODEL}_l{TRANSFORMER_ENC_LAYERS}.pt"
            )
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1, "fold": fold_id,
                "best_val_f1": best_val_f1,
                "args": vars(ARGS),
                "d_model": D_MODEL, "nhead": NHEAD,
                "enc_layers": TRANSFORMER_ENC_LAYERS,
                "audio_feature": "wav2vec2-xls-r-300m",
                "text_feature":  "xlm-roberta-large",
            }, save_dir / ckpt_name)
            print(f"[Save] fold{fold_id} f1={best_val_f1:.4f} -> {ckpt_name}")
        else:
            no_improve += 1
            print(f"[EarlyStopping] {no_improve}/{patience}")

        if ARGS.use_wandb:
            wandb.log({"epoch": epoch+1,
                       "train/loss": train_loss, "train/acc": train_acc,
                       "val/loss": val_result["loss"], "val/acc": val_result["acc"],
                       "val/macro_f1": val_result["macro_f1"],
                       "best/val_macro_f1": best_val_f1})

        if no_improve >= patience:
            print(f"[EarlyStopping] stop at ep {epoch+1}, best f1={best_val_f1:.4f}")
            break

    imsave(history, f"stage1-daic-eatd_loss_fold{fold_id}.jpg")
    if ARGS.use_wandb:
        wandb.finish()
    return best_val_f1


@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()
    totLoss, correct, n = 0.0, 0, 0
    all_y_true, all_y_pred = [], []

    for data in tqdm(loader, desc="Val", unit="sample", leave=False):
        xa, xt, tMask, atei_label, dep_label, jid = data
        xa, xt, tMask = xa.to(device), xt.to(device), tMask.to(device)
        atei_label    = atei_label.to(device)

        with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                            dtype=torch.bfloat16):
            feat, _       = model(xa, xt, tMask)
            patient_feat  = feat.mean(dim=0)
            patient_logit = model.patient_oup(patient_feat)
            loss          = criterion(patient_logit.unsqueeze(0),
                                      atei_label.unsqueeze(0))

        pred     = patient_logit.argmax(dim=-1)
        correct += int(pred.item() == atei_label.item())
        totLoss += loss.item()
        n       += 1
        all_y_true.append(int(atei_label.item()))
        all_y_pred.append(int(pred.item()))

    macro_f1 = f1_score(all_y_true, all_y_pred, labels=[0,1],
                        average="macro", zero_division=0)
    return {
        "loss":     totLoss / max(n, 1),
        "acc":      correct / max(n, 1),
        "macro_f1": macro_f1,
        "y_true":   np.array(all_y_true),
        "y_pred":   np.array(all_y_pred),
        "cm":       confusion_matrix(all_y_true, all_y_pred, labels=[0,1]),
    }


def imsave(history, out_path):
    fig, ax = plt.subplots()
    ax.plot(range(1, len(history)+1), history)
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Stage1 daic+eatd training loss")
    plt.savefig(out_path); plt.close()


# ============================================================
# Main
# ============================================================
def main():
    run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_timer = Timer()
    device      = "cuda" if torch.cuda.is_available() else "cpu"

    daic_depMap, eatd_depMap, eatd_train_ids, daic_folds = get_joint_stage1_folds(
        n_splits=max(ARGS.kfold, 2), seed=ARGS.seed,
        low_q=ARGS.low_q, high_q=ARGS.high_q,
    )
    if ARGS.kfold <= 1:
        daic_folds = daic_folds[:1]

    fold_f1s = []
    for f in daic_folds:
        f1 = run_one_fold(
            f["fold"], f["train"], f["val"], eatd_train_ids,
            daic_depMap, eatd_depMap, run_id, device,
        )
        fold_f1s.append(f1)
        print(f"\n>>> Fold {f['fold']} best Val MacroF1: {f1:.4f}")

    print("\n" + "="*60)
    print("K-FOLD RESULT (Stage1 daic+eatd)")
    print("="*60)
    for i, f1 in enumerate(fold_f1s):
        print(f"Fold {i}: {f1:.4f}")
    print(f"Mean: {np.mean(fold_f1s):.4f}  Std: {np.std(fold_f1s):.4f}")
    print(f"Total time: {total_timer}")


if __name__ == "__main__":
    ARGS = parse_args()
    D_MODEL = ARGS.d_model
    NHEAD   = ARGS.nhead
    LR      = ARGS.lr
    EPOCHS  = ARGS.epochs
    TRANSFORMER_ENC_LAYERS = ARGS.enc_layers
    main()