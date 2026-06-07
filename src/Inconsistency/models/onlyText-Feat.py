"""
uv run src/Inconsistency/models/onlyText-Feat.py  --epochs 3000 --enc_layers 1 --d_model 256 --dropout 0.3 --weight_decay 0 --lr 1e-3 --batch_size 32 --patience 500 --lambda_reg 0.3
"""
"""
Stage2 Training Script — Text Modality Only with PHQ-8 multi-task

主任務: depression 3-class classification
輔助任務: PHQ-8 raw score regression (z-score normalized)
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
from Inconsistency.datasets.inconsistentLabel import get_Split_and_GroundTrue, get_PHQ8_Score
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
LAMBDA_REG = 0.3


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

    parser.add_argument("--save_dir", type=str, default="weights/stage2_text_only-Feat")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Emotion inconsistency - Stage2 TextOnly")
    parser.add_argument("--wandb_name", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=2, help="batch size for DataLoader")
    parser.add_argument("--kfold", type=int, default=0)

    parser.add_argument("--lambda_reg", type=float, default=LAMBDA_REG,
                        help="Weight of PHQ-8 regression auxiliary loss. 0 = disabled.")

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
# Model — Text Only with dual head
# ============================================================
class whole_model(nn.Module):
    """
    Text-only depression classifier with PHQ-8 regression aux head.

    Stage3: text encoder (Transformer) -> masked_max pool -> eT
    Stage4: fc1 -> fc2 -> fc3 -> {oup_cls (3-class), oup_reg (PHQ score)}
    """
    def __init__(self, embd_size=D_MODEL, nheads=NHEAD):
        super().__init__()
        # RoBERTa output dim = 1024
        self.in_proj = nn.Sequential(
            nn.Linear(1024, embd_size),
            nn.LayerNorm(embd_size),
        )

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
        self.oup_cls = nn.Linear(embd_size, 3)   # classification head
        self.oup_reg = nn.Linear(embd_size, 1)   # PHQ-8 regression head

    def forward(self, XT, tMask=None, return_feature=False):
        # XT: [B, num_seg, 1024]
        XT_proj = self.in_proj(XT)
        HT = self.t_transformer_enc(XT_proj, src_key_padding_mask=tMask)

        eT = self.masked_max(HT, tMask)  # [B, D]

        Fc1 = self.dropout(F.relu(self.fc1(eT)))
        Fc2 = self.dropout(F.relu(self.fc2(Fc1)))
        Fc3 = self.dropout(F.relu(self.fc3(Fc2)))

        dep_logits = self.oup_cls(Fc3)              # [B, 3]
        reg_score = self.oup_reg(Fc3).squeeze(-1)   # [B]

        if return_feature:
            return dep_logits, reg_score, Fc3
        return dep_logits, reg_score

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
    totDepLoss = totRegLoss = totLoss = 0.0
    correct_dep = valid_batches = total_samples = 0
    train_true_arr = []
    train_pred_arr = []

    pbar = tqdm(tr_loader, desc=f"Training epoch {cur_epoch}/{tot_epochs}",
                leave=False, unit='batch')

    for data in pbar:
        xt, tMask, dep_label, phq_score, Patient = data

        xt = xt.to(device)
        tMask = tMask.to(device)
        dep_label = dep_label.to(device)
        phq_score = phq_score.to(device)

        opt.zero_grad()
        with torch.autocast(device_type="cuda", enabled=False):
            dep_logits, reg_score = model(xt, tMask)
            L_cls = loss_dep(dep_logits, dep_label)
            L_reg = F.mse_loss(reg_score, phq_score)
            L_Total = L_cls + ARGS.lambda_reg * L_reg

        scaler.scale(L_Total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()

        totDepLoss += L_cls.item()
        totRegLoss += L_reg.item()
        totLoss += L_Total.item()

        dep_pred = dep_logits.argmax(dim=-1)
        correct_dep += (dep_pred == dep_label).sum().item()
        valid_batches += 1
        total_samples += dep_label.size(0)

        pbar.set_postfix({
            "cls": totDepLoss / valid_batches,
            "reg": totRegLoss / valid_batches,
            "tot": totLoss / valid_batches,
            "acc": correct_dep / total_samples,
        })

        train_true_arr.extend(dep_label.cpu().tolist())
        train_pred_arr.extend(dep_pred.cpu().tolist())

    print("Train true dist:", Counter(train_true_arr))
    print("Train pred dist:", Counter(train_pred_arr))

    return {
        "dep_loss": totDepLoss / valid_batches,
        "reg_loss": totRegLoss / valid_batches,
        "tot_loss": totLoss / valid_batches,
        "cur_dep_acc": correct_dep / total_samples,
    }


def val(model, val_loader, loss_dep, device, cur_epoch, tot_epochs):
    model.eval()

    totDepLoss = 0.0
    totRegLoss = 0.0
    valid_batches = 0
    true_arr = []
    pred_arr = []
    pred_score_arr = []
    true_score_arr = []

    pbar = tqdm(val_loader, desc=f"Validation epoch {cur_epoch}/{tot_epochs}",
                leave=False, unit="batch")

    with torch.inference_mode():
        for data in pbar:
            if data is None:
                continue

            xt, tMask, dep_label, phq_score, Patient = data

            xt = xt.to(device)
            tMask = tMask.to(device)
            dep_label = dep_label.to(device)
            phq_score = phq_score.to(device)

            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                dep_logits, reg_score = model(xt, tMask)
                L_cls = loss_dep(dep_logits, dep_label)
                L_reg = F.mse_loss(reg_score, phq_score)

            dep_pred = dep_logits.argmax(dim=-1)

            true_arr.append(int(dep_label.item()))
            pred_arr.append(int(dep_pred.item()))
            pred_score_arr.append(float(reg_score.item()))
            true_score_arr.append(float(phq_score.item()))

            totDepLoss += L_cls.item()
            totRegLoss += L_reg.item()
            valid_batches += 1

            pbar.set_postfix({
                "cls": totDepLoss / valid_batches,
                "reg": totRegLoss / valid_batches,
            })

    metrics = get_metrics(true_arr, pred_arr)

    print("Val true dist:", Counter(true_arr))
    print("Val  pred dist:", Counter(pred_arr))
    print("Confusion matrix:")
    print(confusion_matrix(true_arr, pred_arr, labels=[0, 1, 2]))
    print(classification_report(true_arr, pred_arr, labels=[0, 1, 2],
                                digits=4, zero_division=0))

    # 也順便看看 regression 學得怎樣
    reg_mae = float(np.mean(np.abs(np.array(pred_score_arr) - np.array(true_score_arr))))
    print(f"Val reg MAE (z-scored): {reg_mae:.4f}")

    return {
        "dep_loss": totDepLoss / max(valid_batches, 1),
        "reg_loss": totRegLoss / max(valid_batches, 1),
        "reg_mae": reg_mae,
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
    Multi-task: depression 3-class + PHQ-8 regression (z-score normalized).
    """
    def __init__(self, fold: str = "tr", cv_split=None,
                 phq_mean: float = 0.0, phq_std: float = 1.0):
        self.ds = []
        t_root = Path("datasets/Feature2/RoBerTa")

        depMap, train_Idx, val_Idx, test_Idx = get_Split_and_GroundTrue()
        phqMap = get_PHQ8_Score()

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
            assert p in phqMap, f"PHQ-8 score not found for patient {p}"

            dep_label = depMap[p]
            phq = phqMap[p]
            # (patient_id, dep_label, phq_raw, text_path)
            self.ds.append((p, dep_label, phq, t_path))

        self.phq_mean = phq_mean
        self.phq_std = phq_std

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        Patient, DepL, PHQ, t_path = self.ds[index]
        xt = torch.load(str(t_path), map_location="cpu")
        xt_list = [x.squeeze(0) for x in xt]

        phq_norm = (PHQ - self.phq_mean) / (self.phq_std + 1e-6)

        return (
            xt_list,
            torch.tensor(DepL, dtype=torch.long),
            torch.tensor(phq_norm, dtype=torch.float),
            Patient,
        )


def stage2_collate_fn(batch):
    """
    每個 patient 的每句 mean-pool 成 segment-level [num_seg, 1024],
    然後 pad 成 [B, max_num_seg, 1024]。
    """
    xt_pool_list = []
    dep_labels = []
    phq_scores = []
    patients = []

    for xt_i, dep_label, phq, patient in batch:
        xt_pool_list.append(torch.stack([x.mean(dim=0) for x in xt_i], dim=0))
        dep_labels.append(dep_label)
        phq_scores.append(phq)
        patients.append(patient)

    xt_pool = pad_sequence(xt_pool_list, batch_first=True)   # [B, max_num_seg, 1024]
    tMask = (xt_pool.sum(dim=-1) == 0)
    dep_labels = torch.stack(dep_labels)
    phq_scores = torch.stack(phq_scores)

    return xt_pool, tMask, dep_labels, phq_scores, patients


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

    # 先把 PHQ-8 全表抓出來,後面 train 切片用來算 mean/std
    phqMap = get_PHQ8_Score()
    depMap, train_Idx_full, val_Idx_full, test_Idx_full = get_Split_and_GroundTrue()

    for split in splits:
        fold_id = split["fold"]
        set_seed(ARGS.seed + fold_id)

        # 算 train set 的 PHQ-8 mean / std,用來 z-score normalize
        if split["train"] is None:
            train_pids = train_Idx_full
        else:
            train_pids = split["train"]
        train_phq_values = np.array([phqMap[p] for p in train_pids], dtype=np.float32)
        phq_mean = float(train_phq_values.mean())
        phq_std = float(train_phq_values.std())
        print(f"\n[Fold {fold_id}] PHQ-8 normalize stats: mean={phq_mean:.2f}, std={phq_std:.2f}")

        if ARGS.wandb_name is not None:
            run_name = f"{ARGS.wandb_name}_fold{fold_id}"
        else:
            run_name = (
                f"stage2_textonly_phq_"
                f"seed{ARGS.seed}_"
                f"lr{LR:.0e}_"
                f"wd{ARGS.weight_decay:.0e}_"
                f"do{ARGS.dropout:.2f}_"
                f"lr_reg{ARGS.lambda_reg:.2f}_"
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
                    "lambda_reg": ARGS.lambda_reg,
                    "phq_mean": phq_mean,
                    "phq_std": phq_std,
                    "loss_total": "L_cls + lambda_reg * L_reg",
                    "modality": "text_only_phq_multitask",
                },
                save_code=True,
            )

        print("\n" + "=" * 100)
        print(f"FOLD {fold_id}  (TEXT-ONLY + PHQ-8 MULTITASK)")
        print("=" * 100)
        best_val_f1 = -1.0
        bad_epochs = 0

        # 1. Dataset
        if split["train"] is None:
            trDS = stage2_dataset(fold="tr",
                                  phq_mean=phq_mean, phq_std=phq_std)
            valDS = stage2_dataset(fold="val",
                                   phq_mean=phq_mean, phq_std=phq_std)
        else:
            trDS = stage2_dataset(fold="tr", cv_split=split,
                                  phq_mean=phq_mean, phq_std=phq_std)
            valDS = stage2_dataset(fold="val", cv_split=split,
                                   phq_mean=phq_mean, phq_std=phq_std)

        tr_loader = DataLoader(
            trDS,
            collate_fn=stage2_collate_fn,
            batch_size=ARGS.batch_size,
            shuffle=True,
            worker_init_fn=numpy_random_init,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
        val_loader = DataLoader(
            valDS,
            collate_fn=stage2_collate_fn,
            shuffle=False,
            batch_size=1,
            worker_init_fn=numpy_random_init,
            num_workers=2,
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
        dep_counter = Counter([int(x[1]) for x in train_ds_records])  # x = (p, dep, phq, t_path)
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
                f"Cls Loss: {tr_result['dep_loss']:.4f} | "
                f"Reg Loss: {tr_result['reg_loss']:.4f} | "
                f"Tot Loss: {tr_result['tot_loss']:.4f} | "
                f"Dep Acc: {tr_result['cur_dep_acc']:.4f}"
            )
            print(
                f"[Val] "
                f"Cls Loss: {val_result['dep_loss']:.4f} | "
                f"Reg Loss: {val_result['reg_loss']:.4f} | "
                f"Reg MAE: {val_result['reg_mae']:.4f} | "
                f"Acc: {val_result['acc']:.4f} | "
                f"Pre: {val_result['pre']:.4f} | "
                f"Rec: {val_result['rec']:.4f} | "
                f"F1: {val_result['f1']:.4f}"
            )

            if val_result["f1"] > best_val_f1:
                best_val_f1 = val_result["f1"]
                bad_epochs = 0

                ckpt_name = (
                    f"stage2_textonly_phq_"
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
                        "val_reg_loss": val_result["reg_loss"],
                        "val_reg_mae": val_result["reg_mae"],
                        "args": vars(ARGS),
                        "d_model": D_MODEL,
                        "nhead": NHEAD,
                        "enc_layers": TRANSFORMER_ENC_LAYERS,
                        "lr": LR,
                        "weight_decay": ARGS.weight_decay,
                        "dropout": ARGS.dropout,
                        "lambda_reg": ARGS.lambda_reg,
                        "phq_mean": phq_mean,
                        "phq_std": phq_std,
                        "modality": "text_only_phq_multitask",
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
                    "train/reg_loss": tr_result["reg_loss"],
                    "train/tot_loss": tr_result["tot_loss"],
                    "train/dep_acc": tr_result["cur_dep_acc"],
                    "val/dep_loss": val_result["dep_loss"],
                    "val/reg_loss": val_result["reg_loss"],
                    "val/reg_mae": val_result["reg_mae"],
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
    print("K-FOLD RESULT (TEXT-ONLY + PHQ MULTITASK)")
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
    LAMBDA_REG = ARGS.lambda_reg

    print("** Text-Only Stage2 Training with PHQ-8 multi-task **")
    main()