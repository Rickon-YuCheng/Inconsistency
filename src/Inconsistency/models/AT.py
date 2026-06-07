"""
f1 0.65 ep059
uv run src/Inconsistency/models/AT.py  --epochs 3000 --enc_layers 1 --d_model 256 --dropout 0.3 --weight_decay 0 --lr 5e-4 --batch_size 64 --patience 500
"""
"""
AT.py — Stage2 baseline: Audio + Text only (NO ATEI)

純粹用 audio + text fusion 做 depression 3-class 分類,不含 ATEI 分支。
作為 ablation 對照組 —— 比較 ATEI 有無對最終 F1 的影響。

跟 Stage2Main_quick.py 差別:
- 沒有 self.atei,沒有 self.atei_proj,沒有 self.alpha
- forward 沒有 ATEI 分支,fusion 只有 [eA, eT]
- collate 不處理 xa_seg_list / xt_seg_list
- loss 只算 L_Depression
"""
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch
from sklearn.model_selection import StratifiedKFold
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime
import argparse
from Inconsistency.utils import Timer, set_seed, numpy_random_init
import torch.nn as nn
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from Inconsistency.datasets.inconsistentLabel import get_Split_and_GroundTrue
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

D_MODEL = 128
NHEAD = 8
LR = 1e-5
EPOCHS = 50
TRANSFORMER_ENC_LAYERS = 1
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
PATIENCE = 50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--nhead", type=int, default=NHEAD)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--enc_layers", type=int, default=TRANSFORMER_ENC_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--save_dir", type=str, default="weights/AT")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Emotion inconsistency - AT baseline")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--kfold", type=int, default=0)
    return parser.parse_args()


def build_kfold_splits(n_splits=5, seed=42):
    depMap, train_Idx, val_Idx, test_Idx = get_Split_and_GroundTrue()
    patient_ids = train_Idx + val_Idx
    labels = [depMap[p] for p in patient_ids]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(patient_ids, labels)):
        splits.append({
            "fold": fold_idx,
            "train": [patient_ids[i] for i in tr_idx],
            "val": [patient_ids[i] for i in val_idx],
        })
    return splits


# ============================================================
# Model — A+T only (no ATEI)
# ============================================================
class whole_model(nn.Module):
    def __init__(self, embd_size=D_MODEL, nheads=NHEAD):
        super().__init__()
        self.a_in_proj = nn.Sequential(
            nn.Linear(1024, embd_size),
            nn.LayerNorm(embd_size),
        )
        self.t_in_proj = nn.Sequential(
            nn.Linear(1024, embd_size),
            nn.LayerNorm(embd_size),
        )

        self.a_post_norm = nn.LayerNorm(embd_size)
        self.t_post_norm = nn.LayerNorm(embd_size)

        # Audio encoder
        a_enc_layer = nn.TransformerEncoderLayer(
            d_model=embd_size, dropout=ARGS.dropout,
            dim_feedforward=4 * embd_size,
            nhead=nheads, batch_first=True, norm_first=True,
        )
        self.a_transformer_enc = nn.TransformerEncoder(
            a_enc_layer, num_layers=TRANSFORMER_ENC_LAYERS,
            enable_nested_tensor=False,
        )

        # Text encoder
        t_enc_layer = nn.TransformerEncoderLayer(
            d_model=embd_size, dropout=ARGS.dropout,
            dim_feedforward=4 * embd_size,
            nhead=nheads, batch_first=True, norm_first=True,
        )
        self.t_transformer_enc = nn.TransformerEncoder(
            t_enc_layer, num_layers=TRANSFORMER_ENC_LAYERS,
            enable_nested_tensor=False,
        )

        self.dropout = nn.Dropout(ARGS.dropout)
        self.fc1 = nn.Linear(2 * embd_size, embd_size)  # 2D (no ATEI)
        self.fc2 = nn.Linear(embd_size, embd_size)
        self.fc3 = nn.Linear(embd_size, embd_size)
        self.oup = nn.Linear(embd_size, 3)

    def forward(self, XA, XT, aMask=None, tMask=None, return_feature=False):
        """
        XA: [B, num_seg, 1024]
        XT: [B, num_seg, 1024]
        """
        XA_proj = self.a_in_proj(XA)
        XT_proj = self.t_in_proj(XT)

        HA = self.a_transformer_enc(XA_proj, src_key_padding_mask=aMask)
        HT = self.t_transformer_enc(XT_proj, src_key_padding_mask=tMask)

        eA = self.masked_max(HA, aMask)
        eT = self.masked_max(HT, tMask)

        eA=self.a_post_norm(eA)
        eT=self.t_post_norm(eT)

        # 確保不會只學到單一模態
        if self.training:
            print(f"eA norm: {eA.norm(dim=-1).mean().item():.4f}, "
                f"eT norm: {eT.norm(dim=-1).mean().item():.4f}")

        # Fusion: only A + T
        eFusion = torch.cat((eA, eT), dim=1)  # [B, 2D]
        Fc1 = self.dropout(F.relu(self.fc1(eFusion)))
        Fc2 = self.dropout(F.relu(self.fc2(Fc1)))
        Fc3 = self.dropout(F.relu(self.fc3(Fc2)))
        dep_logits = self.oup(Fc3)

        if return_feature:
            return dep_logits, Fc3
        return dep_logits

    def masked_max(self, x, mask):
        if mask is None:
            return x.max(dim=1)[0]
        x = x.masked_fill(mask.unsqueeze(-1), float('-inf'))
        return x.max(dim=1)[0]


# ============================================================
# Train / Val
# ============================================================
def train_one_epoch(model, tr_loader, loss_dep, opt, device,
                    cur_epoch, tot_epochs, scaler):
    model.train()
    totDepLoss = 0.0
    correct_dep = valid_batches = total_samples = 0
    train_true_arr, train_pred_arr = [], []

    pbar = tqdm(tr_loader, desc=f"Training epoch {cur_epoch}/{tot_epochs}",
                leave=False, unit='batch')

    for data in pbar:
        xa, xt, aMask, tMask, dep_label, Patient = data

        xa = xa.to(device, non_blocking=True)
        xt = xt.to(device, non_blocking=True)
        aMask = aMask.to(device, non_blocking=True)
        tMask = tMask.to(device, non_blocking=True)
        dep_label = dep_label.to(device, non_blocking=True)

        opt.zero_grad()
        with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                            dtype=torch.bfloat16):
            dep_logits = model(xa, xt, aMask, tMask)
            L_Depression = loss_dep(dep_logits, dep_label)

        L_Depression.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        totDepLoss += L_Depression.item()
        dep_pred = dep_logits.argmax(dim=-1)
        correct_dep += (dep_pred == dep_label).sum().item()
        valid_batches += 1
        total_samples += dep_label.size(0)

        pbar.set_postfix({
            "dep": totDepLoss / valid_batches,
            "dep_acc": correct_dep / total_samples,
        })

        train_true_arr.extend(dep_label.cpu().tolist())
        train_pred_arr.extend(dep_pred.cpu().tolist())

    print("Train true dist:", Counter(train_true_arr))
    print("Train pred dist:", Counter(train_pred_arr))

    return {
        "dep_loss": totDepLoss / valid_batches,
        "cur_dep_acc": correct_dep / total_samples,
    }


def val(model, val_loader, loss_dep, device, cur_epoch, tot_epochs):
    model.eval()
    totDepLoss = 0.0
    valid_batches = 0
    true_arr, pred_arr = [], []

    pbar = tqdm(val_loader, desc=f"Validation epoch {cur_epoch}/{tot_epochs}",
                leave=False, unit="batch")

    with torch.inference_mode():
        for data in pbar:
            if data is None:
                continue
            xa, xt, aMask, tMask, dep_label, Patient = data

            xa = xa.to(device, non_blocking=True)
            xt = xt.to(device, non_blocking=True)
            aMask = aMask.to(device, non_blocking=True)
            tMask = tMask.to(device, non_blocking=True)
            dep_label = dep_label.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                                dtype=torch.bfloat16):
                dep_logits = model(xa, xt, aMask, tMask)
                patient_dep = dep_logits.squeeze(0)
                L_Depression = loss_dep(patient_dep.unsqueeze(0), dep_label)

            dep_pred = patient_dep.argmax(dim=-1)
            true_arr.append(int(dep_label.item()))
            pred_arr.append(int(dep_pred.item()))
            totDepLoss += L_Depression.item()
            valid_batches += 1

            pbar.set_postfix({"dep_loss": totDepLoss / valid_batches})

    metrics = get_metrics(true_arr, pred_arr)
    print("Val true dist:", Counter(true_arr))
    print("Val  pred dist:", Counter(pred_arr))
    print("Confusion matrix:")
    print(confusion_matrix(true_arr, pred_arr, labels=[0, 1, 2]))
    print(classification_report(true_arr, pred_arr, labels=[0, 1, 2],
                                digits=4, zero_division=0))

    return {
        "dep_loss": totDepLoss / max(valid_batches, 1),
        "acc": metrics["acc"],
        "pre": metrics["pre"],
        "rec": metrics["rec"],
        "f1": metrics["f1"],
        "labels": true_arr,
        "preds": pred_arr,
    }


# ============================================================
# Dataset / Collate
# ============================================================
class stage2_dataset(Dataset):
    def __init__(self, fold: str = "tr", cv_split=None):
        self.ds = []
        a_root = Path("datasets/Feature/HuBERT_quick")     # ← 新 path
        t_root = Path("datasets/Feature/RoBerTa_slow")     # ← 新 path

        depMap, train_Idx, val_Idx, test_Idx = get_Split_and_GroundTrue()
        if cv_split is not None:
            patient_Idx = cv_split[{"tr": "train", "val": "val", "test": "test"}[fold]]
        else:
            patient_Idx = {"tr": train_Idx, "val": val_Idx, "test": test_Idx}[fold]

        for p in patient_Idx:
            a_path = a_root / f"{p}_acoustic.pt"
            t_path = t_root / f"{p}_text.pt"
            assert a_path.exists() and t_path.exists(), f"ds error: {p}"
            dep_label = depMap[p]
            self.ds.append((p, dep_label, a_path, t_path))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        Patient, DepL, a_path, t_path = self.ds[index]
        xa = torch.load(str(a_path), map_location="cpu")
        xt = torch.load(str(t_path), map_location="cpu")

        # xa: list of [1, 1024] → list of [1024]
        # xt: list of [1, L_i, 1024] → list of [L_i, 1024]
        xa_list = [x.squeeze(0) for x in xa]
        xt_list = [x.squeeze(0) for x in xt]

        dep_label = torch.tensor(DepL, dtype=torch.long)
        return xa_list, xt_list, dep_label, Patient


def stage2_collate_fn(batch):
    """
    純 A+T,不處理 ATEI:
    - xa: segment-level [B, max_num_seg, 1024]
    - xt: segment-level [B, max_num_seg, 1024]  (token mean)
    """
    xa_pool_list = []
    xt_pool_list = []
    dep_labels, patients = [], []

    for xa_i, xt_i, dep_label, patient in batch:
        # audio: 已是 segment-level
        xa_pool_list.append(torch.stack(xa_i, dim=0))
        # text: 每句 token mean
        xt_pool_list.append(torch.stack([x.mean(dim=0) for x in xt_i], dim=0))
        dep_labels.append(dep_label)
        patients.append(patient)

    xa_pool = pad_sequence(xa_pool_list, batch_first=True)
    xt_pool = pad_sequence(xt_pool_list, batch_first=True)
    aMask = (xa_pool.sum(dim=-1) == 0)
    tMask = (xt_pool.sum(dim=-1) == 0)
    dep_labels = torch.stack(dep_labels)

    return xa_pool, xt_pool, aMask, tMask, dep_labels, patients


def get_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "pre": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "rec": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


# ============================================================
# Main
# ============================================================
def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_timer = Timer()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = Path(ARGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if ARGS.kfold <= 1:
        splits = [{"fold": 0, "train": None, "val": None}]
    else:
        splits = build_kfold_splits(n_splits=ARGS.kfold, seed=ARGS.seed)

    all_fold_results = []

    for split in splits:
        fold_id = split["fold"]
        set_seed(ARGS.seed + fold_id)

        run_name = ARGS.wandb_name + f"_fold{fold_id}" if ARGS.wandb_name else (
            f"AT_seed{ARGS.seed}_lr{LR:.0e}_wd{ARGS.weight_decay:.0e}_"
            f"do{ARGS.dropout:.2f}_d{D_MODEL}_l{TRANSFORMER_ENC_LAYERS}_{run_id}"
        )

        if ARGS.use_wandb:
            wandb.init(
                project=ARGS.wandb_project,
                name=run_name,
                config={
                    "seed": ARGS.seed, "d_model": D_MODEL, "nhead": NHEAD,
                    "lr": LR, "epochs": EPOCHS, "enc_layers": TRANSFORMER_ENC_LAYERS,
                    "dropout": ARGS.dropout, "weight_decay": ARGS.weight_decay,
                    "patience": PATIENCE,
                    "loss_total": "L_Depression only",
                    "audio_feature": "HuBERT_quick (pooled)",
                    "text_feature": "RoBerTa_slow (token-level mean pool)",
                    "model_type": "AT_baseline_no_ATEI",
                },
                save_code=True,
            )

        print("\n" + "=" * 100)
        print(f"FOLD {fold_id} (A+T baseline, NO ATEI)")
        print("=" * 100)
        best_val_f1 = -1.0
        bad_epochs = 0

        if split["train"] is None:
            trDS = stage2_dataset(fold="tr")
            valDS = stage2_dataset(fold="val")
        else:
            trDS = stage2_dataset(fold="tr", cv_split=split)
            valDS = stage2_dataset(fold="val", cv_split=split)

        tr_loader = DataLoader(
            trDS, collate_fn=stage2_collate_fn,
            batch_size=ARGS.batch_size,
            shuffle=True, worker_init_fn=numpy_random_init,
            num_workers=2, pin_memory=True, persistent_workers=True,
        )
        val_loader = DataLoader(
            valDS, collate_fn=stage2_collate_fn,
            shuffle=False, batch_size=1, worker_init_fn=numpy_random_init,
            num_workers=2, pin_memory=True, persistent_workers=True,
        )

        if ARGS.use_wandb:
            wandb.config.update({
                "train_samples": len(trDS),
                "val_samples": len(valDS),
            })

        model = whole_model(D_MODEL, NHEAD).to(device)
        print("*" * 10)

        opt = torch.optim.Adam(
            model.parameters(), lr=LR,
            weight_decay=ARGS.weight_decay,
        )
        scaler = torch.GradScaler('cuda')

        # Class weights
        train_ds_records = trDS.ds
        dep_counter = Counter([int(x[1]) for x in train_ds_records])
        total = sum(dep_counter.values())
        n_classes = 3
        weights = torch.tensor([
            total / (n_classes * dep_counter[i]) for i in range(n_classes)
        ], dtype=torch.float, device=device)

        print("Train dep dist:", dep_counter)
        print("Class weights:", weights)
        print("Val dep dist:", Counter([int(x[1]) for x in valDS.ds]))

        loss_dep = nn.CrossEntropyLoss(weight=weights)

        for epoch in range(1, EPOCHS + 1):
            print("=" * 80)
            print(f"Epoch [{epoch}/{EPOCHS}]")

            tr_result = train_one_epoch(model, tr_loader, loss_dep,
                                        opt, device, epoch, EPOCHS, scaler)
            val_result = val(model, val_loader, loss_dep, device, epoch, EPOCHS)

            print(
                f"[Train] Dep: {tr_result['dep_loss']:.4f} | "
                f"Dep Acc: {tr_result['cur_dep_acc']:.4f}"
            )
            print(
                f"[Val] Dep Loss: {val_result['dep_loss']:.4f} | "
                f"Acc: {val_result['acc']:.4f} | "
                f"F1: {val_result['f1']:.4f}"
            )

            if val_result["f1"] > best_val_f1:
                best_val_f1 = val_result["f1"]
                bad_epochs = 0

                if best_val_f1 > 0.40:
                    ckpt_name = (
                        f"AT_{run_id}_seed{ARGS.seed}_"
                        f"f1{best_val_f1:.4f}_ep{epoch:03d}_"
                        f"lr{LR:.0e}_wd{ARGS.weight_decay:.0e}_"
                        f"d{D_MODEL}_l{TRANSFORMER_ENC_LAYERS}.pt"
                    )
                    ckpt_path = save_dir / ckpt_name

                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "best_val_f1": best_val_f1,
                        "val_acc": val_result["acc"],
                        "val_pre": val_result["pre"],
                        "val_rec": val_result["rec"],
                        "val_f1": val_result["f1"],
                        "args": vars(ARGS),
                        "d_model": D_MODEL, "nhead": NHEAD,
                        "enc_layers": TRANSFORMER_ENC_LAYERS,
                        "lr": LR,
                        "weight_decay": ARGS.weight_decay,
                        "dropout": ARGS.dropout,
                        "model_type": "AT_baseline_no_ATEI",
                    }, ckpt_path)

                    if ARGS.use_wandb:
                        wandb.run.summary["best_val_f1"] = best_val_f1
                    print(f"[Save Best] Val F1: {best_val_f1:.4f} -> {ckpt_path}")
                else:
                    print(f"[Skip Save] Val F1: {best_val_f1:.4f} (< 0.40)")
            else:
                bad_epochs += 1
                print(f"[EarlyStop] bad_epochs: {bad_epochs}/{PATIENCE}")

            if ARGS.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/dep_loss": tr_result["dep_loss"],
                    "train/dep_acc": tr_result["cur_dep_acc"],
                    "val/dep_loss": val_result["dep_loss"],
                    "val/acc": val_result["acc"],
                    "val/pre": val_result["pre"],
                    "val/rec": val_result["rec"],
                    "val/f1": val_result["f1"],
                    "best/val_f1": best_val_f1,
                    "no_improve": bad_epochs,
                    "lr": opt.param_groups[0]["lr"],
                })

            if bad_epochs >= PATIENCE:
                print(f"[EarlyStop] Stop at epoch {epoch}, best val F1: {best_val_f1:.4f}")
                break

        print(f"Total time: {total_timer}")
        all_fold_results.append(best_val_f1)
        print(f"\nFold {fold_id} Best F1: {best_val_f1:.4f}")
        print(f"Current Mean F1: {np.mean(all_fold_results):.4f}")

        if ARGS.use_wandb:
            wandb.finish()

    print("\n" + "=" * 100)
    print("K-FOLD RESULT (A+T baseline, NO ATEI)")
    print("=" * 100)
    for i, f1 in enumerate(all_fold_results):
        print(f"Fold {i}: {f1:.4f}")
    print(f"\nMean F1: {np.mean(all_fold_results):.4f}")
    print(f"Std  F1: {np.std(all_fold_results):.4f}")


if __name__ == "__main__":
    ARGS = parse_args()
    D_MODEL = ARGS.d_model
    NHEAD = ARGS.nhead
    LR = ARGS.lr
    EPOCHS = ARGS.epochs
    TRANSFORMER_ENC_LAYERS = ARGS.enc_layers
    PATIENCE = ARGS.patience

    print("** A+T baseline Training (NO ATEI) **")
    main()