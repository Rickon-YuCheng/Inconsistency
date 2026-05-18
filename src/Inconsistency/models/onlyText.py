"""
f1 0.54 
(inconsistency) root@ca751ba5a895:/workspace# uv run src/Inconsistency/models/onlyText.py  --epochs 3000 --enc_layers 10 --d_model 512 --dropout 0.3 --weight_decay 0 --lr 1e-3 --batch_size 32 --patience 500
f1 0.57
d_model=256
"""
"""
Stage2 Training Script — Text Modality Only

只使用 RoBERTa text feature 訓練 depression 3-class 分類。
完全不載入 / 計算 acoustic 與 ATEI 分支。

跟原版差別:
- whole_model 只保留 text encoder + fc1/2/3 + oup
- dataset 只讀 t_path (text feature)
- collate 只處理 text
- loss 只算 L_Depression,不算 L_Atei
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

# ============================================================
# Hyperparameters
# ============================================================
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

    parser.add_argument("--save_dir", type=str, default="weights/stage2_text_only")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Emotion inconsistency - Stage2 TextOnly")
    parser.add_argument("--wandb_name", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=2, help="batch size for DataLoader")
    parser.add_argument("--kfold", type=int, default=0)

    return parser.parse_args()


# ============================================================
# K-Fold split
# ============================================================
def build_kfold_splits(n_splits=5, seed=42):
    depMap, train_Idx, val_Idx, test_Idx = get_Split_and_GroundTrue()

    patient_ids = train_Idx + val_Idx
    labels = [depMap[p] for p in patient_ids]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    splits = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(patient_ids, labels)):
        tr_patients = [patient_ids[i] for i in tr_idx]
        val_patients = [patient_ids[i] for i in val_idx]
        splits.append({
            "fold": fold_idx,
            "train": tr_patients,
            "val": val_patients,
        })

    return splits


# ============================================================
# Model — Text Only
# ============================================================
class whole_model(nn.Module):
    """
    Text-only depression classifier.

    Stage3: text encoder (Transformer) -> masked_max pool -> eT
    Stage4: fc1 -> fc2 -> fc3 -> oup (3-class)
    """
    def __init__(self, embd_size=D_MODEL, nheads=NHEAD):
        super().__init__()
        # RoBERTa output dim = 1024
        self.in_proj = nn.Linear(1024, embd_size)

        t_enc_layer = nn.TransformerEncoderLayer(
            d_model=embd_size,
            dropout=ARGS.dropout,
            dim_feedforward=4 * embd_size,
            nhead=nheads,
            batch_first=True,
            norm_first=True,
        )
        self.t_transformer_enc = nn.TransformerEncoder(
            t_enc_layer,
            num_layers=TRANSFORMER_ENC_LAYERS,
            enable_nested_tensor=False,
        )

        self.dropout = nn.Dropout(ARGS.dropout)
        self.fc1 = nn.Linear(embd_size, embd_size)
        self.fc2 = nn.Linear(embd_size, embd_size)
        self.fc3 = nn.Linear(embd_size, embd_size)
        self.oup = nn.Linear(embd_size, 3)

    def forward(self, XT, tMask=None, return_feature=False):
        # XT: [B, num_seg, 1024]
        XT_proj = self.in_proj(XT)
        HT = self.t_transformer_enc(XT_proj, src_key_padding_mask=tMask)

        eT = self.masked_max(HT, tMask)  # [B, D]

        Fc1 = self.dropout(F.relu(self.fc1(eT)))
        Fc2 = self.dropout(F.relu(self.fc2(Fc1)))
        Fc3 = self.dropout(F.relu(self.fc3(Fc2)))
        dep_logits = self.oup(Fc3)  # [B, 3]

        if return_feature:
            return dep_logits, Fc3
        return dep_logits

    def masked_max(self, x, mask):
        if mask is None:
            return x.max(dim=1)[0]
        x = x.masked_fill(mask.unsqueeze(-1), float('-inf'))
        return x.max(dim=1)[0]

    def masked_mean(self, x, mask):
        if mask is None:
            return x.mean(dim=1)
        valid = (~mask).unsqueeze(-1)
        x = x * valid
        denom = valid.sum(dim=1).clamp(min=1)
        return x.sum(dim=1) / denom


# ============================================================
# Train / Val
# ============================================================
def train_one_epoch(model, tr_loader, loss_dep, opt, device, cur_epoch, tot_epochs, scaler):
    model.train()
    totDepLoss = 0.0
    correct_dep = valid_batches = total_samples = 0
    train_true_arr = []
    train_pred_arr = []

    pbar = tqdm(tr_loader, desc=f"Training epoch {cur_epoch}/{tot_epochs}",
                leave=False, unit='batch')

    for data in pbar:
        xt, tMask, dep_label, Patient = data

        xt = xt.to(device)
        tMask = tMask.to(device)
        dep_label = dep_label.to(device)

        opt.zero_grad()
        with torch.autocast(device_type="cuda", enabled=False):
            dep_logits = model(xt, tMask)             # [B, 3]
            L_Depression = loss_dep(dep_logits, dep_label)
            L_Total = L_Depression

        scaler.scale(L_Total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()

        # Loss / Acc
        totDepLoss += L_Depression.item()
        dep_pred = dep_logits.argmax(dim=-1)
        correct_dep += (dep_pred == dep_label).sum().item()
        valid_batches += 1
        total_samples += dep_label.size(0)

        pbar.set_postfix({
            "dep loss": totDepLoss / valid_batches,
            "cur dep acc": correct_dep / total_samples,
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
    true_arr = []
    pred_arr = []

    pbar = tqdm(val_loader, desc=f"Validation epoch {cur_epoch}/{tot_epochs}",
                leave=False, unit="batch")

    with torch.inference_mode():
        for data in pbar:
            if data is None:
                continue

            xt, tMask, dep_label, Patient = data

            xt = xt.to(device)
            tMask = tMask.to(device)
            dep_label = dep_label.to(device)

            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                dep_logits = model(xt, tMask)                     # [1, 3]
                L_Depression = loss_dep(dep_logits, dep_label)    # 直接餵,batch=1

            dep_pred = dep_logits.argmax(dim=-1)                  # [1]

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
    """
    Text-only dataset. 只讀 RoBERTa text feature。
    """
    def __init__(self, fold: str = "tr", cv_split=None):
        self.ds = []
        t_root = Path("datasets/Feature/RoBerTa")

        depMap, train_Idx, val_Idx, test_Idx = get_Split_and_GroundTrue()

        if cv_split is not None:
            if fold == "tr":
                patient_Idx = cv_split["train"]
            elif fold == "val":
                patient_Idx = cv_split["val"]
            elif fold == "test":
                patient_Idx = cv_split["test"]
        else:
            if fold == "tr":
                patient_Idx = train_Idx
            elif fold == "val":
                patient_Idx = val_Idx
            elif fold == "test":
                patient_Idx = test_Idx

        for p in patient_Idx:
            t_path = t_root / f"{p}_text.pt"
            assert t_path.exists(), f"text feature not found: {t_path}"

            dep_label = depMap[p]
            # (patient_id, dep_label, text_path)
            self.ds.append((p, dep_label, t_path))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        Patient, DepL, t_path = self.ds[index]
        xt = torch.load(str(t_path))
        xt_list = [x.squeeze(0) for x in xt]
        dep_label = torch.tensor(DepL, dtype=torch.long)
        return xt_list, dep_label, Patient


def stage2_collate_fn(batch):
    """
    每個 patient 的每句 mean-pool 成 segment-level [num_seg, 1024],
    然後 pad 成 [B, max_num_seg, 1024]。
    """
    xt_pool_list = []
    dep_labels = []
    patients = []

    for xt_i, dep_label, patient in batch:
        # 每句 frame-level mean -> [num_seg, 1024]
        xt_pool_list.append(torch.stack([x.mean(dim=0) for x in xt_i], dim=0))
        dep_labels.append(dep_label)
        patients.append(patient)

    xt_pool = pad_sequence(xt_pool_list, batch_first=True)   # [B, max_num_seg, 1024]
    tMask = (xt_pool.sum(dim=-1) == 0)
    dep_labels = torch.stack(dep_labels)

    return xt_pool, tMask, dep_labels, patients


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

        if ARGS.wandb_name is not None:
            run_name = f"{ARGS.wandb_name}_fold{fold_id}"
        else:
            run_name = (
                f"stage2_textonly_"
                f"seed{ARGS.seed}_"
                f"lr{LR:.0e}_"
                f"wd{ARGS.weight_decay:.0e}_"
                f"do{ARGS.dropout:.2f}_"
                f"d{D_MODEL}_"
                f"l{TRANSFORMER_ENC_LAYERS}_"
                f"{run_id}"
            )

        if ARGS.use_wandb:
            wandb.init(
                project=ARGS.wandb_project,
                name=run_name,
                config={
                    "seed": ARGS.seed,
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "lr": LR,
                    "epochs": EPOCHS,
                    "enc_layers": TRANSFORMER_ENC_LAYERS,
                    "dropout": ARGS.dropout,
                    "weight_decay": ARGS.weight_decay,
                    "patience": PATIENCE,
                    "loss_total": "L_Depression (text only)",
                    "modality": "text_only",
                },
                save_code=True,
            )

        print("\n" + "=" * 100)
        print(f"FOLD {fold_id}  (TEXT-ONLY)")
        print("=" * 100)
        best_val_f1 = -1.0
        bad_epochs = 0

        # 1. Dataset
        if split["train"] is None:
            trDS = stage2_dataset(fold="tr")
            valDS = stage2_dataset(fold="val")
        else:
            trDS = stage2_dataset(fold="tr", cv_split=split)
            valDS = stage2_dataset(fold="val", cv_split=split)

        tr_loader = DataLoader(
            trDS,
            collate_fn=stage2_collate_fn,
            batch_size=ARGS.batch_size,
            shuffle=True,
            worker_init_fn=numpy_random_init,
            num_workers=4,
            pin_memory=True,             # ✅ 加這行(CPU→GPU 傳輸快)
            persistent_workers=True,     # ✅ 加這行(避免每 epoch 重啟 worker)
            prefetch_factor=2,           # ✅ 加這行
        )
        val_loader = DataLoader(
            valDS,
            collate_fn=stage2_collate_fn,
            shuffle=False,
            batch_size=1,
            worker_init_fn=numpy_random_init,
            num_workers=2,               # ✅ val 用少一點
            pin_memory=True,
            persistent_workers=True,
        )

        if ARGS.use_wandb:
            wandb.config.update({
                "train_samples": len(trDS),
                "val_samples": len(valDS),
            })

        # 2. Model
        model = whole_model(D_MODEL, NHEAD).to(device)
        print(model)
        print("*" * 10)

        opt = torch.optim.Adam(
            model.parameters(),
            lr=LR,
            weight_decay=ARGS.weight_decay,
        )
        scaler = torch.GradScaler('cuda')

        # Class weights (用 dep_label 算)
        train_ds_records = trDS.ds
        dep_counter = Counter([int(x[1]) for x in train_ds_records])  # x = (p, dep, t_path)
        total = sum(dep_counter.values())
        n_classes = 3
        weights = torch.tensor([
            total / (n_classes * dep_counter[i]) for i in range(n_classes)
        ], dtype=torch.float, device=device)

        print("Train dep dist:", dep_counter)
        print("Class weights:", weights)

        val_dep_counter = Counter([int(x[1]) for x in valDS.ds])
        print("Val dep dist:", val_dep_counter)

        loss_dep = nn.CrossEntropyLoss(weight=weights)

        # 3. Train loop
        for epoch in range(1, EPOCHS + 1):
            print("=" * 80)
            print(f"Epoch [{epoch}/{EPOCHS}]")

            tr_result = train_one_epoch(model, tr_loader, loss_dep, opt,
                                        device, epoch, EPOCHS, scaler)
            val_result = val(model, val_loader, loss_dep, device, epoch, EPOCHS)

            print(
                f"[Train] "
                f"Dep Loss: {tr_result['dep_loss']:.4f} | "
                f"Dep Acc: {tr_result['cur_dep_acc']:.4f}"
            )
            print(
                f"[Val] "
                f"Dep Loss: {val_result['dep_loss']:.4f} | "
                f"Acc: {val_result['acc']:.4f} | "
                f"Pre: {val_result['pre']:.4f} | "
                f"Rec: {val_result['rec']:.4f} | "
                f"F1: {val_result['f1']:.4f}"
            )

            if val_result["f1"] > best_val_f1:
                best_val_f1 = val_result["f1"]
                bad_epochs = 0

                ckpt_name = (
                    f"stage2_textonly_"
                    f"{run_id}_"
                    f"seed{ARGS.seed}_"
                    f"f1{best_val_f1:.4f}_"
                    f"ep{epoch:03d}_"
                    f"lr{LR:.0e}_"
                    f"wd{ARGS.weight_decay:.0e}_"
                    f"d{D_MODEL}_"
                    f"l{TRANSFORMER_ENC_LAYERS}.pt"
                )
                ckpt_path = save_dir / ckpt_name

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "best_val_f1": best_val_f1,
                        "val_acc": val_result["acc"],
                        "val_pre": val_result["pre"],
                        "val_rec": val_result["rec"],
                        "val_f1": val_result["f1"],
                        "val_dep_loss": val_result["dep_loss"],
                        "args": vars(ARGS),
                        "d_model": D_MODEL,
                        "nhead": NHEAD,
                        "enc_layers": TRANSFORMER_ENC_LAYERS,
                        "lr": LR,
                        "weight_decay": ARGS.weight_decay,
                        "dropout": ARGS.dropout,
                        "modality": "text_only",
                    },
                    ckpt_path,
                )

                if ARGS.use_wandb:
                    wandb.run.summary["best_val_f1"] = best_val_f1

                print(f"[Save Best] Val F1: {best_val_f1:.4f} -> {ckpt_path}")
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
    print("K-FOLD RESULT (TEXT-ONLY)")
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

    print("** Text-Only Stage2 Training **")
    main()