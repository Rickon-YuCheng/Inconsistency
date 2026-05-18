"""
Stage2 depression detection (with segment-level ATEI).

跟 Stage2Main_v1.py 的差異
-------------------------
v1: 用 patient-level Stage1 ckpt
    - ATEI embd_size 寫死 256 (Stage1Tr_v1 訓出來的)
    - atei_label 來自 patient-level pseudo (PseudoLabel_all_distilbert_zdist_q30_70.npz)
    - LAMBDA_ATEI=0.1, joint loss

v2: 用 segment-level Stage1 ckpt (來自 Stage1Tr_v2)
    - ATEI embd_size 從 ckpt 自動讀, 自動對齊 atei_proj
    - LAMBDA_ATEI=0 (plan A): 不算 ATEI auxiliary loss, 只把 ATEI 當 frozen feature extractor
    - atei_label 還是會傳進來但被 LAMBDA_ATEI=0 吃掉, 之後改 plan B/C 不用動 dataset

其他完全不動: encoder (transformer 或 hope_attention)、fusion、alpha scaling、kfold。
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb

from Inconsistency.datasets.inconsistentLabel import get_Split_and_GroundTrue
from Inconsistency.models.hope_adapter import HopeEncoderBlock
from Inconsistency.models.Stage1Tr_v1 import atei
from Inconsistency.utils import Timer, numpy_random_init, set_seed

warnings.filterwarnings("ignore", category=FutureWarning)
mp.set_sharing_strategy("file_system")


# ============================================================
# Defaults
# ============================================================
# 預設用 segment-level Stage1 ckpt (改 Stage1Tr_v2 訓完後的路徑)
STAGE1_CKPT = "weights/stage1_seg/stage1seg_20260518_070629_seed42_f10.8433_ep013_lr1e-04_d128_l1.pt"

D_MODEL = 128
NHEAD = 8
LR = 1e-5
EPOCHS = 50
TRANSFORMER_ENC_LAYERS = 1
DROPOUT = 0.3
ATEI_DROPOUT = 0.4
WEIGHT_DECAY = 1e-4

# v2 預設 LAMBDA_ATEI=0 (plan A)
# 之後想試 joint loss, 改成 0.1 並改 dataset 餵正確的 segment-level atei_label
LAMBDA_ATEI = 0.0
ALPHA_INIT = 0.5
PATIENCE = 50

ENCODER_TYPE = "attn"  # "attn" or "hope_attention"

CMS_PERIODS = (1, 4)
CMS_HIDDEN_MULTIPLIER = 4
CMS_ONLINE_UPDATES = False

# 註: 還是讀這個檔 (給 dataset 一個 atei_label 欄位), 但 LAMBDA_ATEI=0 時不會真的拿來算 loss
# 之後 plan B/C 才會用到
ATEI_LABEL_FILE = "SegPseudoLabel_all_distilbert_v2_pair.npz"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stage1_ckpt", type=str, default=STAGE1_CKPT)

    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--nhead", type=int, default=NHEAD)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--enc_layers", type=int, default=TRANSFORMER_ENC_LAYERS)

    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--atei_dropout", type=float, default=ATEI_DROPOUT)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)

    parser.add_argument("--lambda_atei", type=float, default=LAMBDA_ATEI,
                        help="0 = ATEI 只當 frozen feature, 不算 aux loss (plan A)")
    parser.add_argument("--alpha_init", type=float, default=ALPHA_INIT)
    parser.add_argument("--patience", type=int, default=PATIENCE)

    parser.add_argument("--save_dir", type=str, default="weights/stage2")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Emotion inconsistency - Stage2")
    parser.add_argument("--wandb_name", type=str, default=None)

    parser.add_argument("--encoder_type", type=str, default=ENCODER_TYPE,
                        choices=["attn", "hope_attention"])
    parser.add_argument("--cms_periods", type=int, nargs="+", default=list(CMS_PERIODS))
    parser.add_argument("--cms_hidden_multiplier", type=int, default=CMS_HIDDEN_MULTIPLIER)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--kfold", type=int, default=0)

    # 額外: ATEI 是否要 freeze (Stage1 已經訓得 F1=0.84, 不想再動的話 freeze)
    parser.add_argument("--freeze_atei", action="store_true",
                        help="完全 freeze ATEI subnet, 只訓 Stage2 自己的 encoder/fusion")

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
# Model
# ============================================================
class whole_model(nn.Module):
    """
    Transformer-based depression detection with ATEI feature.

    Stage3: audio encoder + text encoder + ATEI subnet (frozen / fine-tuned)
    Stage4: concat(eA, eE, eT) -> fc1 -> fc2 -> fc3 -> oup (3-class)

    ATEI subnet 從 Stage1 v2 ckpt 載入。embd_size 自動從 ckpt 讀, 不用手動同步。
    """

    def __init__(self, args, atei_embd_size, atei_enc_layers):
        super().__init__()
        self.encoder_type = args.encoder_type
        embd_size = args.d_model
        nheads = args.nhead

        # input projection (HuBERT/RoBERTa 1024 -> embd_size)
        self.a_in_proj = nn.Sequential(
            nn.Linear(1024, embd_size),
            nn.LayerNorm(embd_size),
        )
        self.t_in_proj = nn.Sequential(
            nn.Linear(1024, embd_size),
            nn.LayerNorm(embd_size),
        )

        # ATEI subnet (從 ckpt 載權重, 維度從 ckpt 讀)
        self.atei = atei(
            embd_size=atei_embd_size,
            nheads=nheads,
            dropout=args.atei_dropout,
            TRANSFORMER_ENC_LAYERS=atei_enc_layers,
        )
        ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
        self.atei.load_state_dict(ckpt["model_state_dict"])

        if args.freeze_atei:
            for p in self.atei.parameters():
                p.requires_grad = False

        # ATEI output 從 atei_embd_size 投影到 Stage2 的 embd_size
        self.atei_proj = nn.Linear(atei_embd_size, embd_size)

        # encoder
        if self.encoder_type == "attn":
            a_enc_layer = nn.TransformerEncoderLayer(
                d_model=embd_size,
                dropout=args.dropout,
                dim_feedforward=4 * embd_size,
                nhead=nheads,
                batch_first=True,
                norm_first=True,
            )
            t_enc_layer = nn.TransformerEncoderLayer(
                d_model=embd_size,
                dropout=args.dropout,
                dim_feedforward=4 * embd_size,
                nhead=nheads,
                batch_first=True,
                norm_first=True,
            )
            self.a_transformer_enc = nn.TransformerEncoder(
                a_enc_layer,
                num_layers=args.enc_layers,
                enable_nested_tensor=False,
            )
            self.t_transformer_enc = nn.TransformerEncoder(
                t_enc_layer,
                num_layers=args.enc_layers,
                enable_nested_tensor=False,
            )
        elif self.encoder_type == "hope_attention":
            self.a_encoder = nn.ModuleList([
                HopeEncoderBlock(
                    dim=embd_size,
                    heads=nheads,
                    variant="hope_attention",
                    cms_periods=tuple(args.cms_periods),
                    hidden_multiplier=args.cms_hidden_multiplier,
                    cms_online_updates=CMS_ONLINE_UPDATES,
                )
                for _ in range(args.enc_layers)
            ])
            self.t_encoder = nn.ModuleList([
                HopeEncoderBlock(
                    dim=embd_size,
                    heads=nheads,
                    variant="hope_attention",
                    cms_periods=tuple(args.cms_periods),
                    hidden_multiplier=args.cms_hidden_multiplier,
                    cms_online_updates=CMS_ONLINE_UPDATES,
                )
                for _ in range(args.enc_layers)
            ])
        else:
            raise ValueError(f"unknown encoder_type: {self.encoder_type}")

        # fusion + classifier
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(3 * embd_size, embd_size)
        self.fc2 = nn.Linear(embd_size, embd_size)
        self.fc3 = nn.Linear(embd_size, embd_size)
        self.alpha = nn.Parameter(torch.tensor(args.alpha_init))  # scalar
        self.oup = nn.Linear(embd_size, 3)

    def forward(self, XA, XT, aMask=None, tMask=None,
                xa_seg_list=None, xt_seg_list=None, return_feature=False):
        """
        Args:
            XA, XT: [B, num_seg, 1024]  segment-level mean-pooled feature
            aMask, tMask: [B, num_seg] padding mask
            xa_seg_list, xt_seg_list: list of [num_seg, T, 1024] tensors (per-patient frame-level)
                                       for ATEI forward
        Returns:
            atei_logits: [B, 2]   per-patient mean of segment-level ATEI logits
            dep_logits:  [B, 3]
        """
        # ----- Stage3: depression encoder -----
        XA_proj = self.a_in_proj(XA)
        XT_proj = self.t_in_proj(XT)

        if self.encoder_type == "attn":
            HA = self.a_transformer_enc(XA_proj, src_key_padding_mask=aMask)
            HT = self.t_transformer_enc(XT_proj, src_key_padding_mask=tMask)
        elif self.encoder_type == "hope_attention":
            if aMask is not None:
                XA_proj = XA_proj.masked_fill(aMask.unsqueeze(-1), 0.0)
            HA = XA_proj
            for layer in self.a_encoder:
                HA = layer(HA)
            if aMask is not None:
                HA = HA.masked_fill(aMask.unsqueeze(-1), 0.0)

            if tMask is not None:
                XT_proj = XT_proj.masked_fill(tMask.unsqueeze(-1), 0.0)
            HT = XT_proj
            for layer in self.t_encoder:
                HT = layer(HT)
            if tMask is not None:
                HT = HT.masked_fill(tMask.unsqueeze(-1), 0.0)

        eA = self.masked_max(HA, aMask)  # [B, D]
        eT = self.masked_max(HT, tMask)  # [B, D]

        # ----- ATEI: batched forward, frame-level feature -----
        seg_counts = [xa_seg.size(0) for xa_seg in xa_seg_list]

        max_T_a = max(xa_seg.size(1) for xa_seg in xa_seg_list)
        max_T_t = max(xt_seg.size(1) for xt_seg in xt_seg_list)
        D_in = xa_seg_list[0].size(-1)

        device_local = xa_seg_list[0].device
        total_segs = sum(seg_counts)

        batch_a = torch.zeros(total_segs, max_T_a, D_in,
                              device=device_local, dtype=xa_seg_list[0].dtype)
        batch_t = torch.zeros(total_segs, max_T_t, D_in,
                              device=device_local, dtype=xt_seg_list[0].dtype)

        start = 0
        for xa_seg, xt_seg in zip(xa_seg_list, xt_seg_list):
            n = xa_seg.size(0)
            batch_a[start:start + n, :xa_seg.size(1)] = xa_seg
            batch_t[start:start + n, :xt_seg.size(1)] = xt_seg
            start += n

        mask_a = (batch_a.sum(dim=-1) == 0)
        mask_t = (batch_t.sum(dim=-1) == 0)

        eE_all, logits_all = self.atei(batch_a, batch_t, mask_a, mask_t)
        # eE_all: [total_segs, atei_embd_size], logits_all: [total_segs, 2]

        # per-patient aggregation (mean over patient's segments)
        eE_list = []
        atei_logits_list = []
        start = 0
        for count in seg_counts:
            eE_list.append(eE_all[start:start + count].mean(dim=0))
            atei_logits_list.append(logits_all[start:start + count].mean(dim=0))
            start += count

        eE = torch.stack(eE_list, dim=0)          # [B, atei_embd_size]
        eE = self.atei_proj(eE)                   # [B, D]
        atei_logits = torch.stack(atei_logits_list, dim=0)  # [B, 2]

        # alpha scaling
        alpha = torch.clamp(self.alpha, 0.0, 2.0)
        eE = eE * alpha

        # ----- Stage4: fusion + classifier -----
        eFusion = torch.cat((eA, eE, eT), dim=1)
        Fc1 = self.dropout(F.relu(self.fc1(eFusion)))
        Fc2 = self.dropout(F.relu(self.fc2(Fc1)))
        Fc3 = self.dropout(F.relu(self.fc3(Fc2)))
        dep_logits = self.oup(Fc3)

        if return_feature:
            return atei_logits, dep_logits, Fc3
        return atei_logits, dep_logits

    def masked_max(self, x, mask):
        if mask is None:
            return x.max(dim=1)[0]
        x = x.masked_fill(mask.unsqueeze(-1), float("-inf"))
        return x.max(dim=1)[0]


# ============================================================
# Train / Val
# ============================================================
def train_one_epoch(model, loader, loss_atei, loss_dep, opt, scaler,
                    device, epoch, tot_epochs, lambda_atei):
    model.train()
    totAteiLoss = totDepLoss = totLoss = 0.0
    correct_atei = correct_dep = valid_batches = total_samples = 0
    train_true_arr, train_pred_arr = [], []

    pbar = tqdm(loader, desc=f"Train epoch {epoch}/{tot_epochs}", leave=False, unit="batch")

    for data in pbar:
        xa, xt, aMask, tMask, atei_label, dep_label, _, xa_seg_list, xt_seg_list = data

        xa = xa.to(device, non_blocking=True)
        xt = xt.to(device, non_blocking=True)
        aMask = aMask.to(device, non_blocking=True)
        tMask = tMask.to(device, non_blocking=True)
        atei_label = atei_label.to(device, non_blocking=True)
        dep_label = dep_label.to(device, non_blocking=True)
        xa_seg_list = [x.to(device, non_blocking=True) for x in xa_seg_list]
        xt_seg_list = [x.to(device, non_blocking=True) for x in xt_seg_list]

        opt.zero_grad()

        with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                            dtype=torch.bfloat16):
            atei_logits, dep_logits = model(
                xa, xt, aMask, tMask,
                xa_seg_list=xa_seg_list,
                xt_seg_list=xt_seg_list,
            )
            L_Atei = loss_atei(atei_logits, atei_label)
            L_Dep = loss_dep(dep_logits, dep_label)
            L_Total = lambda_atei * L_Atei + L_Dep

        L_Total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        totAteiLoss += L_Atei.item()
        totDepLoss += L_Dep.item()
        totLoss += L_Total.item()

        atei_pred = atei_logits.argmax(dim=-1)
        dep_pred = dep_logits.argmax(dim=-1)
        correct_atei += (atei_pred == atei_label).sum().item()
        correct_dep += (dep_pred == dep_label).sum().item()
        valid_batches += 1
        total_samples += dep_label.size(0)

        pbar.set_postfix({
            "atei": totAteiLoss / valid_batches,
            "dep": totDepLoss / valid_batches,
            "tot": totLoss / valid_batches,
            "dep_acc": correct_dep / total_samples,
        })

        train_true_arr.extend(dep_label.cpu().tolist())
        train_pred_arr.extend(dep_pred.cpu().tolist())

    print("Train true dist:", Counter(train_true_arr))
    print("Train pred dist:", Counter(train_pred_arr))

    return {
        "atei_loss": totAteiLoss / valid_batches,
        "dep_loss": totDepLoss / valid_batches,
        "tot_loss": totLoss / valid_batches,
        "cur_atei_acc": correct_atei / total_samples,
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
            xa, xt, aMask, tMask, atei_label, dep_label, _, xa_seg_list, xt_seg_list = data

            xa = xa.to(device, non_blocking=True)
            xt = xt.to(device, non_blocking=True)
            aMask = aMask.to(device, non_blocking=True)
            tMask = tMask.to(device, non_blocking=True)
            dep_label = dep_label.to(device, non_blocking=True)
            xa_seg_list = [x.to(device, non_blocking=True) for x in xa_seg_list]
            xt_seg_list = [x.to(device, non_blocking=True) for x in xt_seg_list]

            with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                                dtype=torch.bfloat16):
                _, dep_logits = model(
                    xa, xt, aMask, tMask,
                    xa_seg_list=xa_seg_list,
                    xt_seg_list=xt_seg_list,
                )
                patient_dep = dep_logits.squeeze(0)
                L_Dep = loss_dep(patient_dep.unsqueeze(0), dep_label)

            dep_pred = patient_dep.argmax(dim=-1)
            true_arr.append(int(dep_label.item()))
            pred_arr.append(int(dep_pred.item()))
            totDepLoss += L_Dep.item()
            valid_batches += 1

            pbar.set_postfix({"dep_loss": totDepLoss / valid_batches})

    metrics = get_metrics(true_arr, pred_arr)

    print("Val true dist:", Counter(true_arr))
    print("Val pred dist:", Counter(pred_arr))
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
    Patient-level dataset.

    Item per patient:
        xa_list, xt_list : list of [T_i, 1024] tensors (per segment, frame-level)
        atei_label       : segment-majority pseudo label (見下)
        dep_label        : patient-level depression label (0/1/2)
        patient_id

    ATEI label 處理:
        v2 的 ATEI label 是 segment-level, 但 Stage2 用 patient-level forward,
        所以這裡用 segment-majority 算 patient-level ATEI label:
            - 撈出此 patient 在 SegPseudoLabel 裡的所有 kept segment labels
            - 取多數 (tie 偏 1=consistent)
            - 沒有 kept segment 的 patient: 預設給 1
        當 LAMBDA_ATEI=0 (plan A) 時這個值不會被用,只是欄位佔位。
    """

    def __init__(self, fold: str = "tr", cv_split=None,
                 atei_label_file: str = ATEI_LABEL_FILE):
        self.ds = []
        a_root = Path("datasets/Feature/HuBERT")
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

        # 建 patient-level ATEI label (segment majority)
        pl = np.load(atei_label_file)
        seg_pid = pl["patientIdx"].astype(np.int64)
        seg_lab = pl["label"].astype(np.int64)

        patient_atei_label = {}
        for p, lab in zip(seg_pid, seg_lab):
            patient_atei_label.setdefault(int(p), []).append(int(lab))

        for p in patient_Idx:
            a_path = a_root / f"{p}_acoustic.pt"
            t_path = t_root / f"{p}_text.pt"
            assert a_path.exists() and t_path.exists(), f"ds error: {p}"

            dep_label = depMap[p]

            if p in patient_atei_label:
                labs = patient_atei_label[p]
                cnt = Counter(labs)
                # tie 偏 1 (consistent)
                atei_label_patient = 1 if cnt[1] >= cnt[0] else 0
            else:
                atei_label_patient = 1

            self.ds.append((p, atei_label_patient, dep_label, a_path, t_path))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        Patient, AteiL, DepL, a_path, t_path = self.ds[index]

        xa = torch.load(str(a_path), map_location="cpu", mmap=True)
        xt = torch.load(str(t_path), map_location="cpu", mmap=True)

        xa_list = [x.squeeze(0) for x in xa]
        xt_list = [x.squeeze(0) for x in xt]

        atei_label = torch.tensor(AteiL, dtype=torch.long)
        dep_label = torch.tensor(DepL, dtype=torch.long)

        return xa_list, xt_list, atei_label, dep_label, Patient


def stage2_collate_fn(batch):
    """Patient-level collate.

    回傳:
        xa_pool, xt_pool : [B, max_num_seg, 1024]  segment mean-pool (給 encoder)
        aMask, tMask     : [B, max_num_seg]        seg-level padding mask
        atei_labels      : [B]
        dep_labels       : [B]
        patients         : list
        xa_seg_list      : list of [num_seg, T_a_max_in_patient, 1024]  per-patient frame-level (給 ATEI)
        xt_seg_list      : list of [num_seg, T_t_max_in_patient, 1024]
    """
    xa_seg_list = []
    xt_seg_list = []
    xa_pool_list = []
    xt_pool_list = []
    atei_labels = []
    dep_labels = []
    patients = []

    for xa_i, xt_i, atei_label, dep_label, patient in batch:
        # Stage3 用: per-segment mean pool over frames
        xa_pool_list.append(torch.stack([x.mean(dim=0) for x in xa_i], dim=0))
        xt_pool_list.append(torch.stack([x.mean(dim=0) for x in xt_i], dim=0))

        # ATEI 用: pad 成 [num_seg, T_max, 1024] (per patient)
        xa_seg_list.append(pad_sequence(xa_i, batch_first=True))
        xt_seg_list.append(pad_sequence(xt_i, batch_first=True))

        atei_labels.append(atei_label)
        dep_labels.append(dep_label)
        patients.append(patient)

    xa_pool = pad_sequence(xa_pool_list, batch_first=True)
    xt_pool = pad_sequence(xt_pool_list, batch_first=True)
    aMask = (xa_pool.sum(dim=-1) == 0)
    tMask = (xt_pool.sum(dim=-1) == 0)

    atei_labels = torch.stack(atei_labels)
    dep_labels = torch.stack(dep_labels)

    return (xa_pool, xt_pool, aMask, tMask, atei_labels, dep_labels,
            patients, xa_seg_list, xt_seg_list)


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
    ARGS = parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    timer = Timer()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = Path(ARGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 從 ckpt 讀 ATEI 的維度, 避免 hard-code 不匹配
    print(f"[load] reading Stage1 ckpt meta from {ARGS.stage1_ckpt}")
    ckpt = torch.load(ARGS.stage1_ckpt, map_location="cpu")
    atei_embd_size = ckpt.get("d_model", 128)
    atei_enc_layers = ckpt.get("enc_layers", 1)
    print(f"[load] atei_embd_size={atei_embd_size}, atei_enc_layers={atei_enc_layers}")
    print(f"[load] stage1 best val f1: {ckpt.get('best_val_f1', 'N/A')}")
    del ckpt  # 釋放, model 建構時會再 load 一次

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
                f"stage2v2_seed{ARGS.seed}_lr{ARGS.lr:.0e}_"
                f"wd{ARGS.weight_decay:.0e}_do{ARGS.dropout:.2f}_"
                f"la{ARGS.lambda_atei:.2f}_a{ARGS.alpha_init:.2f}_"
                f"d{ARGS.d_model}_l{ARGS.enc_layers}_enc{ARGS.encoder_type}_"
                f"frz{int(ARGS.freeze_atei)}_{run_id}"
            )

        if ARGS.use_wandb:
            wandb.init(
                project=ARGS.wandb_project,
                name=run_name,
                config={
                    "seed": ARGS.seed,
                    "d_model": ARGS.d_model,
                    "nhead": ARGS.nhead,
                    "lr": ARGS.lr,
                    "epochs": ARGS.epochs,
                    "enc_layers": ARGS.enc_layers,
                    "dropout": ARGS.dropout,
                    "atei_dropout": ARGS.atei_dropout,
                    "weight_decay": ARGS.weight_decay,
                    "lambda_atei": ARGS.lambda_atei,
                    "alpha_init": ARGS.alpha_init,
                    "patience": ARGS.patience,
                    "encoder_type": ARGS.encoder_type,
                    "stage1_ckpt": ARGS.stage1_ckpt,
                    "atei_embd_size": atei_embd_size,
                    "atei_enc_layers": atei_enc_layers,
                    "freeze_atei": ARGS.freeze_atei,
                },
                save_code=True,
            )

        print("\n" + "=" * 100)
        print(f"FOLD {fold_id}")
        print("=" * 100)

        best_val_f1 = -1.0
        bad_epochs = 0

        # --- dataset ---
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
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            valDS,
            collate_fn=stage2_collate_fn,
            shuffle=False,
            batch_size=1,
            worker_init_fn=numpy_random_init,
            num_workers=0,
            pin_memory=True,
        )

        if ARGS.use_wandb:
            wandb.config.update({
                "train_samples": len(trDS),
                "val_samples": len(valDS),
            })

        # --- model ---
        model = whole_model(ARGS,
                            atei_embd_size=atei_embd_size,
                            atei_enc_layers=atei_enc_layers).to(device)
        print(f"[model] freeze_atei={ARGS.freeze_atei}")

        # optimizer: 給 ATEI 分支較小 lr (除非完全 freeze)
        atei_params = list(model.atei.parameters())
        other_params = [
            p for name, p in model.named_parameters()
            if not name.startswith("atei.")
        ]

        if ARGS.freeze_atei:
            # ATEI freeze 時不放進 optimizer
            opt = torch.optim.Adam(other_params, lr=ARGS.lr,
                                   weight_decay=ARGS.weight_decay)
        else:
            opt = torch.optim.Adam(
                [
                    {"params": atei_params, "lr": ARGS.lr * 0.1},
                    {"params": other_params, "lr": ARGS.lr},
                ],
                weight_decay=ARGS.weight_decay,
            )

        scaler = torch.GradScaler("cuda")

        # class weight for dep loss
        train_ds_records = trDS.ds
        dep_counter = Counter([int(x[2]) for x in train_ds_records])
        atei_counter = Counter([int(x[1]) for x in train_ds_records])
        total = sum(dep_counter.values())
        n_classes = 3
        weights = torch.tensor(
            [total / (n_classes * dep_counter[i]) for i in range(n_classes)],
            dtype=torch.float, device=device,
        )

        print(f"Train dep dist: {dep_counter}")
        print(f"Train ATEI (patient-majority) dist: {atei_counter}")
        print(f"Class weights: {weights}")

        val_dep_counter = Counter([int(x[2]) for x in valDS.ds])
        val_atei_counter = Counter([int(x[1]) for x in valDS.ds])
        print(f"Val dep dist: {val_dep_counter}")
        print(f"Val ATEI dist: {val_atei_counter}")

        loss_atei = nn.CrossEntropyLoss()
        loss_dep = nn.CrossEntropyLoss(weight=weights)

        # --- train loop ---
        for epoch in range(1, ARGS.epochs + 1):
            print("=" * 80)
            print(f"Epoch [{epoch}/{ARGS.epochs}]")

            tr_result = train_one_epoch(
                model, tr_loader, loss_atei, loss_dep, opt, scaler,
                device, epoch, ARGS.epochs, ARGS.lambda_atei,
            )
            val_result = val(model, val_loader, loss_dep, device,
                             epoch, ARGS.epochs)

            print(
                f"[Train] ATEI={tr_result['atei_loss']:.4f} "
                f"Dep={tr_result['dep_loss']:.4f} "
                f"Tot={tr_result['tot_loss']:.4f} "
                f"DepAcc={tr_result['cur_dep_acc']:.4f}"
            )
            print(
                f"[Val] Loss={val_result['dep_loss']:.4f} "
                f"Acc={val_result['acc']:.4f} "
                f"Pre={val_result['pre']:.4f} "
                f"Rec={val_result['rec']:.4f} "
                f"F1={val_result['f1']:.4f}"
            )

            if val_result["f1"] > best_val_f1:
                best_val_f1 = val_result["f1"]
                bad_epochs = 0

                ckpt_name = (
                    f"stage2v2_{run_id}_seed{ARGS.seed}_"
                    f"f1{best_val_f1:.4f}_ep{epoch:03d}_"
                    f"lr{ARGS.lr:.0e}_d{ARGS.d_model}_l{ARGS.enc_layers}.pt"
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
                    "val_dep_loss": val_result["dep_loss"],
                    "args": vars(ARGS),
                    "d_model": ARGS.d_model,
                    "nhead": ARGS.nhead,
                    "enc_layers": ARGS.enc_layers,
                    "stage1_ckpt": ARGS.stage1_ckpt,
                    "atei_embd_size": atei_embd_size,
                }, ckpt_path)

                if ARGS.use_wandb:
                    wandb.run.summary["best_val_f1"] = best_val_f1

                print(f"[Save Best] F1={best_val_f1:.4f} -> {ckpt_path}")
            else:
                bad_epochs += 1
                print(f"[EarlyStop] bad_epochs: {bad_epochs}/{ARGS.patience}")

            if ARGS.use_wandb:
                log_dict = {
                    "epoch": epoch,
                    "train/atei_loss": tr_result["atei_loss"],
                    "train/dep_loss": tr_result["dep_loss"],
                    "train/tot_loss": tr_result["tot_loss"],
                    "train/atei_acc": tr_result["cur_atei_acc"],
                    "train/dep_acc": tr_result["cur_dep_acc"],
                    "val/dep_loss": val_result["dep_loss"],
                    "val/acc": val_result["acc"],
                    "val/pre": val_result["pre"],
                    "val/rec": val_result["rec"],
                    "val/f1": val_result["f1"],
                    "best/val_f1": best_val_f1,
                    "no_improve": bad_epochs,
                }
                if ARGS.freeze_atei:
                    log_dict["lr/other"] = opt.param_groups[0]["lr"]
                else:
                    log_dict["lr/atei"] = opt.param_groups[0]["lr"]
                    log_dict["lr/other"] = opt.param_groups[1]["lr"]
                wandb.log(log_dict)

            if bad_epochs >= ARGS.patience:
                print(f"[EarlyStop] Stop at epoch {epoch}, best val F1: {best_val_f1:.4f}")
                break

        print(f"Total time: {timer}")
        all_fold_results.append(best_val_f1)

        print(f"\nFold {fold_id} Best F1: {best_val_f1:.4f}")
        print(f"Current Mean F1: {np.mean(all_fold_results):.4f}")

        if ARGS.use_wandb:
            wandb.finish()

    print("\n" + "=" * 100)
    print("K-FOLD RESULT")
    print("=" * 100)
    for i, f1 in enumerate(all_fold_results):
        print(f"Fold {i}: {f1:.4f}")
    print(f"\nMean F1: {np.mean(all_fold_results):.4f}")
    print(f"Std  F1: {np.std(all_fold_results):.4f}")


if __name__ == "__main__":
    main()