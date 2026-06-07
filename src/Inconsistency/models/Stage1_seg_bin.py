"""
Stage1Tr_seg_bin.py — Stage1 ATEI segment-level training (binary classification)

跟 Stage1Tr_seg.py 的差異
-------------------------
seg (原版):
    - patient-level split 從 inconsistentLabel 取 train/val/test 三 fold
    - macro-f1 當主指標
    - pseudo label: SegPseudoLabel_all_distilbert_v2_pair.npz
    - feature: datasets/Feature/HuBERT (full) + RoBerTa

seg_bin (本檔):
    - split 改用 Incon_seg_bin.get_stage1_kfold (官方 train 內做 k-fold,
      dev 留給 Stage2 當 test, 不洩漏)
    - 雙指標 best ckpt: binary-f1 (positive=cons, 主) 與 macro-f1 (輔) 各存一份
    - early stopping 用 binary-f1 觸發
    - pseudo label: SegPseudoLabel_all_distilbert_pair_bin.npz (來自 Incon2_seg_bin)
    - feature: HuBERT_full_seg_bin (新抽) + RoBerTa_full_bin (沿用 patient-level)

ATEI model (`atei` class) 維持不動, 從 Stage1Tr_v1 import。
forward 簽名: (xa, xt, aMask, tMask), 兩邊都吃 frame/token-level 序列。
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb

from Inconsistency.datasets.Incon_seg_bin import get_stage1_kfold
# ATEI model 內嵌(原本 import 自 Stage1Tr_v1, 但 v1 的 forward 寫死
# chunk_size + gradient checkpoint, 對 segment-level mini-batch 而言反而
# 把 B=256 切成 256 次 B=1 的 forward, GPU 利用率掉到 1%, 5.2h/3-fold。
# 這裡內嵌一份「無 chunk、無 checkpoint」的等價版本, forward 一次吃完整 batch。
import torch.nn.functional as F
from Inconsistency.utils import Timer, numpy_random_init, set_seed

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# Defaults
# ============================================================
D_MODEL = 128
NHEAD = 8
LR = 1e-4
EPOCHS = 30
TRANSFORMER_ENC_LAYERS = 1
BATCH_SIZE = 64

PSEUDO_LABEL_FILE = "SegPseudoLabel_all_distilbert_pair_bin.npz"
A_ROOT = "datasets/Feature/HuBERT_full_seg_bin"
T_ROOT = "datasets/Feature/RoBerTa_full_bin"   # 沿用 patient-level 那份 (內容一致)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--nhead", type=int, default=NHEAD)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--enc_layers", type=int, default=TRANSFORMER_ENC_LAYERS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)

    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    parser.add_argument("--pseudo_label_file", type=str, default=PSEUDO_LABEL_FILE)
    parser.add_argument("--a_root", type=str, default=A_ROOT)
    parser.add_argument("--t_root", type=str, default=T_ROOT)

    parser.add_argument("--save_dir", type=str, default="weights/stage1_seg_bin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4,
                        help="DataLoader prefetch (only when num_workers>0)")
    parser.add_argument("--cache_size", type=int, default=16,
                        help="per-worker LRU cache for patient .pt files")

    parser.add_argument("--use_sampler", action="store_true",
                        help="WeightedRandomSampler 動態平衡 cons/incon")
    parser.add_argument("--use_class_weight", action="store_true",
                        help="CrossEntropyLoss weight (跟 sampler 二擇一)")
    parser.add_argument("--grouped_batch", action="store_true",
                        help="Patient-grouped batch sampler (大幅提高 IO cache 命中率, "
                             "跟 --use_sampler 互斥)")

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str,
                        default="Emotion inconsistency - Stage1 Seg bin")
    parser.add_argument("--wandb_name", type=str, default=None)

    parser.add_argument("--kfold", type=int, default=3,
                        help="StratifiedKFold over official train. <=1 disables.")

    return parser.parse_args()


# ============================================================
# ATEI model (inline, 無 chunk + 無 checkpoint 版本)
# ============================================================
# 跟 Stage1Tr_v1.atei 結構等價, 差別只在:
#   - forward 一次處理整個 batch [B, T, D], 不再 torch.split(chunk_size=1)
#   - 不用 torch.utils.checkpoint, B=256 不會被切成 256 次 forward
#   - GPU kernel launch 從 256 次 / batch 變 1 次, 利用率明顯上升
#
# 介面跟 v1 完全一致: forward(xa, xt, aMask=None, tMask=None) -> (Fc3, oup_logits)
# 所以 train/val 那段不用改。
class atei(nn.Module):
    def __init__(self, embd_size, nheads, inp_dim=1024, dropout=0.4,
                 TRANSFORMER_ENC_LAYERS=1):
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
        self.transformer_enc = nn.TransformerEncoder(
            enc_layer, num_layers=TRANSFORMER_ENC_LAYERS,
        )

        self.Cross_Attn = at_cross_attn(embd_size)
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(4 * embd_size, embd_size)
        self.fc2 = nn.Linear(embd_size, embd_size)
        self.fc3 = nn.Linear(embd_size, embd_size)
        self.oup = nn.Linear(embd_size, 2)
        self.patient_oup = nn.Linear(embd_size, 2)

    def forward(self, xa, xt, aMask=None, tMask=None):
        # xa: [B, T_a, 1024], xt: [B, T_t, 1024]
        xa = self.a_in_proj(xa)
        xt = self.t_in_proj(xt)

        # 整 batch 走 transformer, 不切 chunk
        XprimeA = self.transformer_enc(xa, src_key_padding_mask=aMask)
        XprimeT = self.transformer_enc(xt, src_key_padding_mask=tMask)

        # cross-attention
        Xat, Xta = self.Cross_Attn(XprimeA, XprimeT, aMask, tMask)

        avgXprimeA = self.maskMean(XprimeA, aMask)
        avgXat     = self.maskMean(Xat, aMask)
        avgXta     = self.maskMean(Xta, tMask)
        avgXprimeT = self.maskMean(XprimeT, tMask)
        hE = torch.cat((avgXprimeA, avgXat, avgXta, avgXprimeT), dim=1)

        Fc1 = self.dropout(F.relu(self.fc1(hE)))
        Fc2 = self.dropout(F.relu(self.fc2(Fc1)))
        Fc3 = self.fc3(Fc2)
        Oup = self.oup(Fc3)
        return Fc3, Oup

    def maskMean(self, inp, mask):
        if mask is None:
            return inp.mean(dim=1)
        valid = (~mask).unsqueeze(-1).float()
        s = (inp * valid).sum(dim=1)
        Len = valid.sum(dim=1).clamp(min=1.0)
        return s / Len


class at_cross_attn(nn.Module):
    def __init__(self, embd_size=1024):
        super().__init__()
        self.at_Q = nn.Linear(embd_size, embd_size)
        self.at_K = nn.Linear(embd_size, embd_size)
        self.at_V = nn.Linear(embd_size, embd_size)
        self.ta_Q = nn.Linear(embd_size, embd_size)
        self.ta_K = nn.Linear(embd_size, embd_size)
        self.ta_V = nn.Linear(embd_size, embd_size)

    def forward(self, XprimeA, XprimeT, aMask=None, tMask=None):
        Qa = self.at_Q(XprimeA); Kt = self.at_K(XprimeT); Vt = self.at_V(XprimeT)
        Qt = self.ta_Q(XprimeT); Ka = self.ta_K(XprimeA); Va = self.ta_V(XprimeA)
        Xat = _cross_attn(Qa, Kt, Vt, tMask)
        Xta = _cross_attn(Qt, Ka, Va, aMask)
        return Xat, Xta


def _cross_attn(Q, K, V, mask=None):
    # [B, Lq, D], [B, Lk, D], [B, Lk, D]; mask [B, Lk] True=padding
    Q = Q.unsqueeze(1); K = K.unsqueeze(1); V = V.unsqueeze(1)
    attn_mask = None
    if mask is not None:
        attn_mask = (~mask).view(mask.size(0), 1, 1, mask.size(1))
    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
    return out.squeeze(1)


# ============================================================
# Dataset / Index / Collate (跟 Stage1Tr_seg.py 一致)
# ============================================================
class SegSampleIndex:
    """每個 sample = (patient_id, seg_id, list_idx, atei_label)。"""

    def __init__(self, patient_ids, pseudo_label_path, ds_root="datasets/DAICWOZ"):
        pl = np.load(pseudo_label_path)
        seg_pid = pl["patientIdx"].astype(np.int64)
        seg_sid = pl["segIdx"].astype(np.int64)
        seg_lab = pl["label"].astype(np.int64)

        label_map = {(int(p), int(s)): int(lab)
                     for p, s, lab in zip(seg_pid, seg_sid, seg_lab)}

        self.samples = []
        for pid in patient_ids:
            csv_path = Path(ds_root) / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv"
            if not csv_path.exists():
                print(f"[warn] csv not found: {csv_path}, skip patient {pid}")
                continue

            df = pd.read_csv(csv_path, sep="\t")
            df_p = df[df["speaker"] == "Participant"].dropna(subset=["value"]).copy()

            for list_idx, row in enumerate(df_p.itertuples()):
                seg_id = row.Index + 2
                key = (pid, seg_id)
                if key not in label_map:
                    continue
                self.samples.append({
                    "patient_id": pid,
                    "seg_id": seg_id,
                    "list_idx": list_idx,
                    "atei_label": label_map[key],
                })

        print(f"[SegSampleIndex] {len(self.samples)} samples from {len(patient_ids)} patients")

    def __len__(self):
        return len(self.samples)

    def get_label_counts(self):
        labels = np.array([s["atei_label"] for s in self.samples])
        return np.bincount(labels, minlength=2)


class SegDataset(Dataset):
    """每個 patient 的 .pt 用 LRU cache。"""

    def __init__(self, sample_index, a_root=A_ROOT, t_root=T_ROOT, cache_size=8):
        self.samples = sample_index.samples
        self.a_root = Path(a_root)
        self.t_root = Path(t_root)
        self._cache = {}
        self._cache_order = []
        self._cache_size = cache_size

    def _load_patient(self, pid):
        if pid in self._cache:
            self._cache_order.remove(pid)
            self._cache_order.append(pid)
            return self._cache[pid]

        a_path = self.a_root / f"{pid}_acoustic.pt"
        t_path = self.t_root / f"{pid}_text.pt"

        xa = torch.load(str(a_path), map_location="cpu", mmap=True)
        xt = torch.load(str(t_path), map_location="cpu", mmap=True)

        xa_list = [x.squeeze(0) for x in xa]
        xt_list = [x.squeeze(0) for x in xt]

        n = min(len(xa_list), len(xt_list))
        if len(xa_list) != len(xt_list):
            print(f"[warn] patient {pid}: audio {len(xa_list)} vs text {len(xt_list)}, truncate to {n}")
        xa_list = xa_list[:n]
        xt_list = xt_list[:n]

        self._cache[pid] = (xa_list, xt_list)
        self._cache_order.append(pid)
        if len(self._cache_order) > self._cache_size:
            del self._cache[self._cache_order.pop(0)]
        return self._cache[pid]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        pid = s["patient_id"]
        list_idx = s["list_idx"]
        xa_list, xt_list = self._load_patient(pid)

        if list_idx >= len(xa_list) or list_idx >= len(xt_list):
            raise IndexError(
                f"patient {pid} seg {s['seg_id']} list_idx {list_idx} "
                f"out of range (audio {len(xa_list)}, text {len(xt_list)})"
            )

        return {
            "xa": xa_list[list_idx],
            "xt": xt_list[list_idx],
            "atei_label": s["atei_label"],
            "patient_id": pid,
            "seg_id": s["seg_id"],
        }


def collate_fn(batch):
    xa_list = [item["xa"] for item in batch]
    xt_list = [item["xt"] for item in batch]
    labels = torch.tensor([item["atei_label"] for item in batch], dtype=torch.long)
    pids = [item["patient_id"] for item in batch]
    sids = [item["seg_id"] for item in batch]

    xa = pad_sequence(xa_list, batch_first=True)
    xt = pad_sequence(xt_list, batch_first=True)
    aMask = (xa.sum(dim=-1) == 0)
    tMask = (xt.sum(dim=-1) == 0)

    return {
        "xa": xa, "xt": xt, "aMask": aMask, "tMask": tMask,
        "labels": labels, "patient_ids": pids, "seg_ids": sids,
    }


# ============================================================
# Class balance helpers
# ============================================================
def build_weighted_sampler(sample_index, seed=42):
    from torch.utils.data import WeightedRandomSampler
    labels = np.array([s["atei_label"] for s in sample_index.samples])
    counts = np.bincount(labels, minlength=2)
    print(f"[sampler] label counts: {counts}")
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(f"only one class in train set: {counts}")
    sample_weights = (1.0 / counts)[labels]
    g = torch.Generator(); g.manual_seed(seed)
    return WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights),
        replacement=True, generator=g,
    )


def build_class_weight(sample_index, device):
    labels = np.array([s["atei_label"] for s in sample_index.samples])
    counts = np.bincount(labels, minlength=2)
    weights = counts.sum() / (2.0 * counts)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[class_weight] counts: {counts}, weights: {weights.tolist()}")
    return weights


class PatientGroupedBatchSampler:
    """
    Patient-grouped batch sampler (for cache locality)。

    把同一個 patient 的 sample 集中產生連續 batch, dataset 的 LRU cache
    對單一 patient 命中率接近 100%, GPU 不再等 IO。

    每個 epoch:
        1. shuffle patient 的處理順序
        2. 在每個 patient 內 shuffle samples
        3. 攤平後依 batch_size 切, 不跨 patient (最後一個 batch 可能變小)

    注意: 這跟 WeightedRandomSampler 不相容, 開了 --use_sampler 就不用這個。
    若你想配合 class imbalance, 改用 --use_class_weight 路徑。
    """

    def __init__(self, sample_index, batch_size, seed=42, drop_last=False):
        self.batch_size = batch_size
        self.seed = seed
        self.drop_last = drop_last

        # patient_id -> list of sample idx
        groups = {}
        for idx, s in enumerate(sample_index.samples):
            groups.setdefault(s["patient_id"], []).append(idx)
        self.groups = groups
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        pids = list(self.groups.keys())
        rng.shuffle(pids)

        flat = []
        for pid in pids:
            idxs = self.groups[pid][:]
            rng.shuffle(idxs)
            flat.extend(idxs)

        for i in range(0, len(flat), self.batch_size):
            batch = flat[i:i + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch

    def __len__(self):
        total = sum(len(v) for v in self.groups.values())
        if self.drop_last:
            return total // self.batch_size
        return (total + self.batch_size - 1) // self.batch_size


# ============================================================
# Train / Val
# ============================================================
def train_one_epoch(model, loader, criterion, opt, scaler, device, epoch, tot_epochs, fold_id):
    model.train()
    total_loss = correct = n = 0

    pbar = tqdm(loader, desc=f"Fold{fold_id} Train ep {epoch}/{tot_epochs}",
                unit="batch", leave=False)

    for batch in pbar:
        xa = batch["xa"].to(device, non_blocking=True)
        xt = batch["xt"].to(device, non_blocking=True)
        aMask = batch["aMask"].to(device, non_blocking=True)
        tMask = batch["tMask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        opt.zero_grad()
        with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                            dtype=torch.bfloat16):
            _, logits = model(xa, xt, aMask, tMask)   # [B, 2]
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()

        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

        pbar.set_postfix({
            "loss": total_loss / max(n, 1),
            "acc": correct / max(n, 1),
        })

    return {"loss": total_loss / max(n, 1), "acc": correct / max(n, 1)}


@torch.inference_mode()
def validate(model, loader, criterion, device, fold_id):
    model.eval()
    total_loss = correct = n = 0
    all_y_true, all_y_pred = [], []

    pbar = tqdm(loader, desc=f"Fold{fold_id} Val", unit="batch", leave=False)
    for batch in pbar:
        xa = batch["xa"].to(device, non_blocking=True)
        xt = batch["xt"].to(device, non_blocking=True)
        aMask = batch["aMask"].to(device, non_blocking=True)
        tMask = batch["tMask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                            dtype=torch.bfloat16):
            _, logits = model(xa, xt, aMask, tMask)
            loss = criterion(logits, labels)

        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        all_y_true.extend(labels.cpu().tolist())
        all_y_pred.extend(pred.cpu().tolist())

    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1],
                        average="macro", zero_division=0)
    # 主指標: positive class = 1 (consistency, 少數類)
    bin_f1 = f1_score(y_true, y_pred, average="binary",
                      pos_label=1, zero_division=0)

    return {
        "loss": total_loss / max(n, 1),
        "acc": correct / max(n, 1),
        "macro_f1": macro_f1,
        "bin_f1": bin_f1,
        "y_true": y_true, "y_pred": y_pred, "cm": cm,
    }


# ============================================================
# Per-fold runner
# ============================================================
def run_one_fold(fold_id, train_ids, val_ids, run_id, device):
    set_seed(ARGS.seed + fold_id)

    save_dir = Path(ARGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_name = (ARGS.wandb_name + f"_fold{fold_id}") if ARGS.wandb_name else (
        f"stage1seg_bin_seed{ARGS.seed}_lr{ARGS.lr:.0e}_bs{ARGS.batch_size}_"
        f"d{ARGS.d_model}_l{ARGS.enc_layers}_fold{fold_id}_{run_id}"
    )

    # --- index / dataset / loader ---
    print(f"\n{'='*60}\nFOLD {fold_id}\n{'='*60}")
    train_index = SegSampleIndex(train_ids, ARGS.pseudo_label_file)
    val_index = SegSampleIndex(val_ids, ARGS.pseudo_label_file)
    print(f"[train] {len(train_index)} samples, dist: {train_index.get_label_counts()}")
    print(f"[val]   {len(val_index)} samples, dist: {val_index.get_label_counts()}")

    train_ds = SegDataset(train_index, a_root=ARGS.a_root, t_root=ARGS.t_root,
                          cache_size=ARGS.cache_size)
    val_ds = SegDataset(val_index, a_root=ARGS.a_root, t_root=ARGS.t_root,
                        cache_size=ARGS.cache_size)

    g = torch.Generator(); g.manual_seed(ARGS.seed + fold_id)

    # 三選一: grouped / weighted / 普通 shuffle
    train_sampler = None
    train_batch_sampler = None
    use_shuffle = True

    if ARGS.grouped_batch:
        assert not ARGS.use_sampler, "grouped_batch / use_sampler 二擇一"
        train_batch_sampler = PatientGroupedBatchSampler(
            train_index, batch_size=ARGS.batch_size,
            seed=ARGS.seed + fold_id, drop_last=False,
        )
        use_shuffle = False
        print(f"[loader] using PatientGroupedBatchSampler "
              f"({len(train_batch_sampler)} batches)")
    elif ARGS.use_sampler:
        train_sampler = build_weighted_sampler(train_index,
                                               seed=ARGS.seed + fold_id)
        use_shuffle = False

    if train_batch_sampler is not None:
        train_loader = DataLoader(
            train_ds, batch_sampler=train_batch_sampler,
            collate_fn=collate_fn, num_workers=ARGS.num_workers,
            pin_memory=True, worker_init_fn=numpy_random_init, generator=g,
            persistent_workers=(ARGS.num_workers > 0),
            prefetch_factor=(ARGS.prefetch_factor if ARGS.num_workers > 0 else None),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=ARGS.batch_size,
            sampler=train_sampler, shuffle=use_shuffle,
            collate_fn=collate_fn, num_workers=ARGS.num_workers,
            pin_memory=True, worker_init_fn=numpy_random_init, generator=g,
            persistent_workers=(ARGS.num_workers > 0),
            prefetch_factor=(ARGS.prefetch_factor if ARGS.num_workers > 0 else None),
        )
    val_loader = DataLoader(
        val_ds, batch_size=ARGS.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=ARGS.num_workers, pin_memory=True,
        persistent_workers=(ARGS.num_workers > 0),
        prefetch_factor=(ARGS.prefetch_factor if ARGS.num_workers > 0 else None),
    )

    # --- model / optim / loss ---
    model = atei(
        embd_size=ARGS.d_model, nheads=ARGS.nhead,
        dropout=ARGS.dropout, TRANSFORMER_ENC_LAYERS=ARGS.enc_layers,
    ).to(device)

    class_weight = None
    if ARGS.use_class_weight:
        assert not ARGS.use_sampler, "sampler / class_weight 二擇一"
        class_weight = build_class_weight(train_index, device)

    criterion = nn.CrossEntropyLoss(weight=class_weight,
                                    label_smoothing=ARGS.label_smoothing)
    opt = torch.optim.Adam(model.parameters(), lr=ARGS.lr,
                           weight_decay=ARGS.weight_decay)
    scaler = torch.GradScaler("cuda")

    # --- wandb ---
    if ARGS.use_wandb:
        wandb.init(
            project=ARGS.wandb_project,
            name=run_name,
            reinit=True,
            config={
                "seed": ARGS.seed, "fold": fold_id,
                "d_model": ARGS.d_model, "nhead": ARGS.nhead,
                "lr": ARGS.lr, "epochs": ARGS.epochs,
                "enc_layers": ARGS.enc_layers, "batch_size": ARGS.batch_size,
                "dropout": ARGS.dropout, "weight_decay": ARGS.weight_decay,
                "label_smoothing": ARGS.label_smoothing,
                "use_sampler": ARGS.use_sampler,
                "use_class_weight": ARGS.use_class_weight,
                "pseudo_label_file": ARGS.pseudo_label_file,
                "train_samples": len(train_index),
                "val_samples": len(val_index),
                "train_label_counts": train_index.get_label_counts().tolist(),
                "val_label_counts": val_index.get_label_counts().tolist(),
                "audio_feature": ARGS.a_root,
                "text_feature": ARGS.t_root,
                "atei_level": "segment-level",
            },
        )

    # --- training loop ---
    best_bin_f1 = -1.0
    best_macro_f1 = -1.0
    no_improve = 0   # 用 bin_f1 觸發 early stopping

    for epoch in range(1, ARGS.epochs + 1):
        print("=" * 80)
        print(f"[Fold {fold_id}] Epoch [{epoch}/{ARGS.epochs}]")

        tr = train_one_epoch(model, train_loader, criterion, opt, scaler,
                             device, epoch, ARGS.epochs, fold_id)
        val_res = validate(model, val_loader, criterion, device, fold_id)

        print(f"[Train] loss={tr['loss']:.4f} acc={tr['acc']:.4f}")
        print(f"[Val]   loss={val_res['loss']:.4f} acc={val_res['acc']:.4f} "
              f"binF1(cons)={val_res['bin_f1']:.4f} macroF1={val_res['macro_f1']:.4f}")
        print(f"[Val] label counts: {np.bincount(val_res['y_true'], minlength=2)}")
        print(f"[Val] pred  counts: {np.bincount(val_res['y_pred'], minlength=2)}")
        print(f"[Val] confusion matrix:\n{val_res['cm']}")
        print(classification_report(
            val_res["y_true"], val_res["y_pred"],
            labels=[0, 1],
            target_names=["inconsistency(0)", "consistency(1)"],
            digits=4, zero_division=0,
        ))

        # 雙 best ckpt
        saved_any = False

        if val_res["bin_f1"] > best_bin_f1:
            best_bin_f1 = val_res["bin_f1"]
            no_improve = 0
            ckpt_name = (
                f"stage1seg_bin_{run_id}_seed{ARGS.seed}_fold{fold_id}_"
                f"best_binf1_{best_bin_f1:.4f}_ep{epoch:03d}_"
                f"lr{ARGS.lr:.0e}_d{ARGS.d_model}_l{ARGS.enc_layers}.pt"
            )
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "fold": fold_id,
                "best_bin_f1": best_bin_f1,
                "best_macro_f1": best_macro_f1,
                "val_acc": val_res["acc"], "val_loss": val_res["loss"],
                "val_cm": val_res["cm"],
                "args": vars(ARGS),
                "d_model": ARGS.d_model, "nhead": ARGS.nhead,
                "enc_layers": ARGS.enc_layers, "dropout": ARGS.dropout,
                "pseudo_label_file": ARGS.pseudo_label_file,
                "audio_feature": ARGS.a_root,
                "text_feature": ARGS.t_root,
                "atei_level": "segment",
                "selected_by": "binary_f1",
            }, save_dir / ckpt_name)
            print(f"[Save best-binF1] {best_bin_f1:.4f} -> {ckpt_name}")
            saved_any = True
        else:
            no_improve += 1

        if val_res["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_res["macro_f1"]
            ckpt_name = (
                f"stage1seg_bin_{run_id}_seed{ARGS.seed}_fold{fold_id}_"
                f"best_macrof1_{best_macro_f1:.4f}_ep{epoch:03d}_"
                f"lr{ARGS.lr:.0e}_d{ARGS.d_model}_l{ARGS.enc_layers}.pt"
            )
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "fold": fold_id,
                "best_bin_f1": best_bin_f1,
                "best_macro_f1": best_macro_f1,
                "val_acc": val_res["acc"], "val_loss": val_res["loss"],
                "val_cm": val_res["cm"],
                "args": vars(ARGS),
                "d_model": ARGS.d_model, "nhead": ARGS.nhead,
                "enc_layers": ARGS.enc_layers, "dropout": ARGS.dropout,
                "pseudo_label_file": ARGS.pseudo_label_file,
                "audio_feature": ARGS.a_root,
                "text_feature": ARGS.t_root,
                "atei_level": "segment",
                "selected_by": "macro_f1",
            }, save_dir / ckpt_name)
            print(f"[Save best-macroF1] {best_macro_f1:.4f} -> {ckpt_name}")
            saved_any = True

        if not saved_any:
            print(f"[EarlyStop] no improvement (bin) {no_improve}/{ARGS.patience}")

        if ARGS.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": tr["loss"], "train/acc": tr["acc"],
                "val/loss": val_res["loss"], "val/acc": val_res["acc"],
                "val/bin_f1": val_res["bin_f1"],
                "val/macro_f1": val_res["macro_f1"],
                "val/pred_0": int(np.bincount(val_res["y_pred"], minlength=2)[0]),
                "val/pred_1": int(np.bincount(val_res["y_pred"], minlength=2)[1]),
                "best/val_bin_f1": best_bin_f1,
                "best/val_macro_f1": best_macro_f1,
                "no_improve": no_improve,
                "val/conf_mat": wandb.plot.confusion_matrix(
                    y_true=val_res["y_true"], preds=val_res["y_pred"],
                    class_names=["inconsistency", "consistency"],
                ),
            })

        if no_improve >= ARGS.patience:
            print(f"[EarlyStop] Fold {fold_id} stop at ep {epoch}, "
                  f"best binF1={best_bin_f1:.4f}, best macroF1={best_macro_f1:.4f}")
            break

    if ARGS.use_wandb:
        wandb.finish()

    return {"bin_f1": best_bin_f1, "macro_f1": best_macro_f1}


# ============================================================
# Main: k-fold loop
# ============================================================
def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    timer = Timer()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    depMap, folds = get_stage1_kfold(n_splits=max(ARGS.kfold, 2),
                                     seed=ARGS.seed)
    if ARGS.kfold <= 1:
        folds = folds[:1]

    fold_results = []
    for f in folds:
        r = run_one_fold(f["fold"], f["train"], f["val"], run_id, device)
        fold_results.append(r)
        print(f"\n>>> Fold {f['fold']} best binF1={r['bin_f1']:.4f} "
              f"macroF1={r['macro_f1']:.4f}")

    print("\n" + "=" * 60)
    print("K-FOLD RESULT (Stage1 seg_bin)")
    print("=" * 60)
    for i, r in enumerate(fold_results):
        print(f"Fold {i}: binF1={r['bin_f1']:.4f}  macroF1={r['macro_f1']:.4f}")

    bin_arr = np.array([r["bin_f1"] for r in fold_results])
    macro_arr = np.array([r["macro_f1"] for r in fold_results])
    print(f"\nMean binF1 (cons)  : {bin_arr.mean():.4f} ± {bin_arr.std():.4f}")
    print(f"Mean macroF1       : {macro_arr.mean():.4f} ± {macro_arr.std():.4f}")
    print(f"\nTotal time: {timer}")


if __name__ == "__main__":
    ARGS = parse_args()
    main()