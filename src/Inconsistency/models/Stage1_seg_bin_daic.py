"""
uv run src/Inconsistency/models/Stage1_seg_bin_daic.py --use_class_weight
"""
"""
Stage1_seg_bin_daic.py
======================
Stage1 ATEI segment-level training, DAIC-WOZ only.

跟 Stage1_seg_bin_daic_eatd.py 的差別:
  移除所有 EATD 邏輯 (SegSampleIndex, DataLoader, get_eatd_train_vols)
  FEAT_DIR  -> datasets/Feat_seg_bin_daic/
  SAVE_DIR  -> weights/stage1_seg_bin_daic/
  wandb project -> Stage1 seg_bin daic

Pseudo label
------------
  SegPseudoLabel_daic_distilbert_pair_bin.npz

Feature
-------
  datasets/Feat_seg_bin_daic/  (DAIC full-frame audio + token text)

Val
---
  純 DAIC kfold val。
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
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb

from Inconsistency.datasets.Incon_seg_bin import get_stage1_kfold
from Inconsistency.utils import Timer, numpy_random_init, set_seed

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# Defaults
# ============================================================
D_MODEL = 256
NHEAD = 8
LR = 1e-4
EPOCHS = 30
TRANSFORMER_ENC_LAYERS = 1
BATCH_SIZE = 64

DAIC_PSEUDO_NPZ = "SegPseudoLabel_daic_distilbert_pair_bin.npz"
FEAT_DIR = "datasets/Feat_seg_bin_daic"
DAIC_DS_ROOT = "datasets/DAICWOZ"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--d_model", type=int, default=D_MODEL)
    p.add_argument("--nhead", type=int, default=NHEAD)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--enc_layers", type=int, default=TRANSFORMER_ENC_LAYERS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.0)
    p.add_argument("--daic_pseudo_npz", type=str, default=DAIC_PSEUDO_NPZ)
    p.add_argument("--atei_mode", type=str, default="hard",
                   choices=["hard", "soft_cosine"],
                   help="hard: binary CE; soft_cosine: MSE on cosine similarity")
    p.add_argument("--feat_dir", type=str, default=FEAT_DIR)
    p.add_argument("--save_dir", type=str, default="weights/stage1_seg_bin_daic")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--cache_size", type=int, default=16)
    p.add_argument("--use_sampler", action="store_true")
    p.add_argument("--use_class_weight", action="store_true")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="Stage1 seg_bin daic")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--kfold", type=int, default=3)
    p.add_argument("--max_audio_frames", type=int, default=500)
    return p.parse_args()


# ============================================================
# ATEI model (segment-level, 雙獨立 encoder)
# ============================================================
class atei(nn.Module):
    def __init__(self, embd_size, nheads, inp_dim=1024, dropout=0.4, enc_layers=1):
        super().__init__()
        assert embd_size % nheads == 0
        self.a_in_proj = nn.Sequential(nn.Linear(inp_dim, embd_size),
                                       nn.LayerNorm(embd_size))
        self.t_in_proj = nn.Sequential(nn.Linear(inp_dim, embd_size),
                                       nn.LayerNorm(embd_size))

        a_enc = nn.TransformerEncoderLayer(
            d_model=embd_size, nhead=nheads, batch_first=True,
            dim_feedforward=4 * embd_size, dropout=dropout)
        self.a_transformer_enc = nn.TransformerEncoder(a_enc, num_layers=enc_layers)

        t_enc = nn.TransformerEncoderLayer(
            d_model=embd_size, nhead=nheads, batch_first=True,
            dim_feedforward=4 * embd_size, dropout=dropout)
        self.t_transformer_enc = nn.TransformerEncoder(t_enc, num_layers=enc_layers)

        self.Cross_Attn = at_cross_attn(embd_size)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(4 * embd_size, embd_size)
        self.fc2 = nn.Linear(embd_size, embd_size)
        self.fc3 = nn.Linear(embd_size, embd_size)
        self.oup = nn.Linear(embd_size, 2)        # hard mode
        self.oup_soft = nn.Linear(embd_size, 1)   # soft_cosine mode
        self.patient_oup = nn.Linear(embd_size, 2)   # 保留, Stage2 state_dict 一致

    def forward(self, xa, xt, aMask=None, tMask=None):
        xa = self.a_in_proj(xa)
        xt = self.t_in_proj(xt)

        XprimeA = self.a_transformer_enc(xa, src_key_padding_mask=aMask)
        XprimeT = self.t_transformer_enc(xt, src_key_padding_mask=tMask)

        Xat, Xta = self.Cross_Attn(XprimeA, XprimeT, aMask, tMask)

        avgXprimeA = self._mask_mean(XprimeA, aMask)
        avgXat     = self._mask_mean(Xat, aMask)
        avgXta     = self._mask_mean(Xta, tMask)
        avgXprimeT = self._mask_mean(XprimeT, tMask)
        hE = torch.cat((avgXprimeA, avgXat, avgXta, avgXprimeT), dim=1)

        Fc1 = self.dropout(F.relu(self.fc1(hE)))
        Fc2 = self.dropout(F.relu(self.fc2(Fc1)))
        Fc3 = self.fc3(Fc2)
        Oup = self.oup(Fc3)           # [B, 2]  for hard CE
        Oup_soft = self.oup_soft(Fc3).squeeze(-1)  # [B]    for soft MSE
        return Fc3, Oup, Oup_soft

    @staticmethod
    def _mask_mean(x, mask):
        if mask is None:
            return x.mean(dim=1)
        valid = (~mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


class at_cross_attn(nn.Module):
    def __init__(self, embd_size):
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
    Q = Q.unsqueeze(1); K = K.unsqueeze(1); V = V.unsqueeze(1)
    attn_mask = None
    if mask is not None:
        attn_mask = (~mask).view(mask.size(0), 1, 1, mask.size(1))
    out = F.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)
    return out.squeeze(1)


# ============================================================
# Sample Index (DAIC only)
# ============================================================
class SegSampleIndex:
    def __init__(self, daic_pids, daic_npz,
                 daic_ds_root=DAIC_DS_ROOT, feat_dir=FEAT_DIR):
        self.samples = []
        feat_dir = Path(feat_dir)

        if daic_pids and daic_npz:
            dpl = np.load(daic_npz)
            _atei_mode = str(dpl["atei_mode"]) if "atei_mode" in dpl else "hard"
            d_pid = dpl["patientIdx"].astype(np.int64)
            d_sid = dpl["segIdx"].astype(np.int64)
            if _atei_mode == "soft_cosine":
                d_lab = dpl["label"].astype(np.float32)
                d_map = {(int(p), int(s)): float(l)
                         for p, s, l in zip(d_pid, d_sid, d_lab)}
            else:
                d_lab = dpl["label"].astype(np.int64)
                d_map = {(int(p), int(s)): int(l)
                         for p, s, l in zip(d_pid, d_sid, d_lab)}
            for pid in daic_pids:
                csv_path = Path(daic_ds_root) / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv"
                if not csv_path.exists():
                    continue
                a_path = feat_dir / f"{pid}_acoustic.pt"
                t_path = feat_dir / f"{pid}_text.pt"
                if not (a_path.exists() and t_path.exists()):
                    print(f"[warn] feature missing: {pid}")
                    continue
                xa = torch.load(str(a_path), map_location="cpu", mmap=True)
                xt = torch.load(str(t_path), map_location="cpu", mmap=True)
                n_valid = min(len(xa), len(xt))
                if len(xa) != len(xt):
                    print(f"[warn] DAIC {pid}: audio={len(xa)} text={len(xt)}, "
                          f"truncate to {n_valid}")

                df = pd.read_csv(csv_path, sep="\t")
                df_p = df[df.speaker == "Participant"].dropna(subset=["value"]).copy()
                for list_idx, row in enumerate(df_p.itertuples()):
                    if list_idx >= n_valid:
                        break
                    seg_id = row.Index + 2
                    key = (pid, seg_id)
                    if key not in d_map:
                        continue
                    self.samples.append({
                        "patient_id": pid,
                        "seg_id": seg_id,
                        "list_idx": list_idx,
                        "atei_label": d_map[key],  # int (hard) or float (soft)
                    })

        print(f"[SegSampleIndex] total={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def get_label_counts(self):
        labels = np.array([s["atei_label"] for s in self.samples])
        if labels.dtype == np.float32 or labels.dtype == np.float64:
            # soft mode: show histogram buckets
            return np.histogram(labels, bins=5, range=(0,1))[0]
        return np.bincount(labels.astype(np.int64), minlength=2)


# ============================================================
# Dataset
# ============================================================
class SegDataset(Dataset):
    def __init__(self, sample_index, feat_dir, cache_size=8):
        self.samples = sample_index.samples
        self.feat_dir = Path(feat_dir)
        self._cache = {}
        self._cache_order = []
        self._cache_size = cache_size

    def _load_patient(self, pid):
        if pid in self._cache:
            self._cache_order.remove(pid)
            self._cache_order.append(pid)
            return self._cache[pid]
        xa = torch.load(str(self.feat_dir / f"{pid}_acoustic.pt"),
                        map_location="cpu", mmap=True)
        xt = torch.load(str(self.feat_dir / f"{pid}_text.pt"),
                        map_location="cpu", mmap=True)
        xa_list = [x.squeeze(0) for x in xa]
        xt_list = [x.squeeze(0) for x in xt]
        self._cache[pid] = (xa_list, xt_list)
        self._cache_order.append(pid)
        if len(self._cache_order) > self._cache_size:
            del self._cache[self._cache_order.pop(0)]
        return self._cache[pid]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        xa_list, xt_list = self._load_patient(s["patient_id"])
        li = s["list_idx"]
        if li >= len(xa_list) or li >= len(xt_list):
            raise IndexError(
                f"DAIC {s['patient_id']} seg {s['seg_id']} "
                f"list_idx {li} out of range "
                f"(audio {len(xa_list)}, text {len(xt_list)})")
        return {
            "xa": xa_list[li], "xt": xt_list[li],
            "atei_label": s["atei_label"],
            "patient_id": s["patient_id"], "seg_id": s["seg_id"],
        }


def collate_fn(batch):
    max_frames = ARGS.max_audio_frames
    xa_list = [b["xa"][:max_frames] for b in batch]
    xt_list = [b["xt"] for b in batch]
    raw = [b["atei_label"] for b in batch]
    if isinstance(raw[0], float):
        labels = torch.tensor(raw, dtype=torch.float32)
    else:
        labels = torch.tensor(raw, dtype=torch.long)
    xa = pad_sequence(xa_list, batch_first=True)
    xt = pad_sequence(xt_list, batch_first=True)
    aMask = (xa.sum(dim=-1) == 0)
    tMask = (xt.sum(dim=-1) == 0)
    return {"xa": xa, "xt": xt, "aMask": aMask, "tMask": tMask, "labels": labels}


# ============================================================
# Class balance helpers
# ============================================================
def build_weighted_sampler(sample_index, seed=42):
    from torch.utils.data import WeightedRandomSampler
    labels = np.array([s["atei_label"] for s in sample_index.samples])
    counts = np.bincount(labels, minlength=2)
    print(f"[sampler] label counts: {counts}")
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(f"only one class: {counts}")
    sw = (1.0 / counts)[labels]
    g = torch.Generator(); g.manual_seed(seed)
    return WeightedRandomSampler(weights=sw, num_samples=len(sw),
                                 replacement=True, generator=g)


def build_class_weight(sample_index, device):
    labels = np.array([s["atei_label"] for s in sample_index.samples])
    counts = np.bincount(labels, minlength=2)
    weights = counts.sum() / (2.0 * counts)
    w = torch.tensor(weights, dtype=torch.float32, device=device)
    print(f"[class_weight] counts: {counts}, weights: {w.tolist()}")
    return w


# ============================================================
# Train / Val
# ============================================================
def train_one_epoch(model, loader, criterion, opt, scaler, device, epoch, tot, fold_id):
    model.train()
    total_loss = correct = n = 0
    pbar = tqdm(loader, desc=f"Fold{fold_id} Train {epoch}/{tot}", unit="batch", leave=False)
    for batch in pbar:
        xa = batch["xa"].to(device, non_blocking=True)
        xt = batch["xt"].to(device, non_blocking=True)
        aMask = batch["aMask"].to(device, non_blocking=True)
        tMask = batch["tMask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        opt.zero_grad()
        with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                            dtype=torch.bfloat16):
            _, logits, logits_soft = model(xa, xt, aMask, tMask)
            if labels.dtype == torch.float32:
                loss = criterion(logits_soft.sigmoid(), labels)
            else:
                loss = criterion(logits, labels)
                pred = logits.argmax(dim=-1)
                correct += (pred == labels).sum().item()

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)
        pbar.set_postfix({"loss": total_loss/max(n,1)})
    is_soft = (labels.dtype == torch.float32)
    acc = None if is_soft else correct / max(n, 1)
    return {"loss": total_loss / max(n, 1), "acc": acc}


@torch.inference_mode()
def validate(model, loader, criterion, device, fold_id):
    model.eval()
    total_loss = correct = n = 0
    y_true, y_pred = [], []
    for batch in tqdm(loader, desc=f"Fold{fold_id} Val", unit="batch", leave=False):
        xa = batch["xa"].to(device, non_blocking=True)
        xt = batch["xt"].to(device, non_blocking=True)
        aMask = batch["aMask"].to(device, non_blocking=True)
        tMask = batch["tMask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                            dtype=torch.bfloat16):
            _, logits, logits_soft = model(xa, xt, aMask, tMask)
            if labels.dtype == torch.float32:
                loss = criterion(logits_soft.sigmoid(), labels)
            else:
                loss = criterion(logits, labels)
                pred = logits.argmax(dim=-1)
                correct += (pred == labels).sum().item()
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(pred.cpu().tolist())

        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

    if not y_true:
        return {"loss": total_loss/max(n,1), "acc": None,
                "macro_f1": None, "bin_f1": None,
                "y_true": None, "y_pred": None, "cm": None}

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    macro_f1 = f1_score(y_true, y_pred, labels=[0, 1],
                        average="macro", zero_division=0)
    bin_f1 = f1_score(y_true, y_pred, average="binary",
                      pos_label=1, zero_division=0)
    return {"loss": total_loss/max(n,1), "acc": correct/max(n,1),
            "macro_f1": macro_f1, "bin_f1": bin_f1,
            "y_true": y_true, "y_pred": y_pred, "cm": cm}


# ============================================================
# Per-fold runner
# ============================================================
def run_one_fold(fold_id, daic_train_pids, daic_val_pids, run_id, device):
    set_seed(ARGS.seed + fold_id)
    save_dir = Path(ARGS.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nFOLD {fold_id}\n{'='*60}")

    train_idx = SegSampleIndex(daic_train_pids, ARGS.daic_pseudo_npz,
                               feat_dir=ARGS.feat_dir)
    val_idx   = SegSampleIndex(daic_val_pids,   ARGS.daic_pseudo_npz,
                               feat_dir=ARGS.feat_dir)
    print(f"[train] {len(train_idx)} samples, dist: {train_idx.get_label_counts()}")
    print(f"[val]   {len(val_idx)} samples, dist: {val_idx.get_label_counts()}")

    train_ds = SegDataset(train_idx, feat_dir=ARGS.feat_dir, cache_size=ARGS.cache_size)
    val_ds   = SegDataset(val_idx,   feat_dir=ARGS.feat_dir, cache_size=ARGS.cache_size)

    g = torch.Generator(); g.manual_seed(ARGS.seed + fold_id)
    train_sampler = None; use_shuffle = True
    if ARGS.use_sampler:
        train_sampler = build_weighted_sampler(train_idx, seed=ARGS.seed + fold_id)
        use_shuffle = False

    train_loader = DataLoader(
        train_ds, batch_size=ARGS.batch_size, sampler=train_sampler,
        shuffle=use_shuffle, collate_fn=collate_fn,
        num_workers=ARGS.num_workers, pin_memory=True,
        worker_init_fn=numpy_random_init, generator=g,
        persistent_workers=(ARGS.num_workers > 0),
        prefetch_factor=(ARGS.prefetch_factor if ARGS.num_workers > 0 else None))
    val_loader = DataLoader(
        val_ds, batch_size=ARGS.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=ARGS.num_workers, pin_memory=True,
        persistent_workers=(ARGS.num_workers > 0),
        prefetch_factor=(ARGS.prefetch_factor if ARGS.num_workers > 0 else None))

    model = atei(embd_size=ARGS.d_model, nheads=ARGS.nhead,
                 dropout=ARGS.dropout, enc_layers=ARGS.enc_layers).to(device)

    class_weight = None
    if ARGS.atei_mode == "hard":
        if ARGS.use_class_weight:
            assert not ARGS.use_sampler, "sampler / class_weight 二擇一"
            class_weight = build_class_weight(train_idx, device)
        criterion = nn.CrossEntropyLoss(weight=class_weight,
                                        label_smoothing=ARGS.label_smoothing)
    else:
        criterion = nn.MSELoss()
        print(f"[criterion] soft_cosine -> MSELoss")
    opt = torch.optim.Adam(model.parameters(), lr=ARGS.lr,
                           weight_decay=ARGS.weight_decay)
    scaler = torch.GradScaler("cuda")

    run_name = (ARGS.wandb_name + f"_fold{fold_id}") if ARGS.wandb_name else (
        f"stage1seg_d_seed{ARGS.seed}_lr{ARGS.lr:.0e}_bs{ARGS.batch_size}_"
        f"d{ARGS.d_model}_l{ARGS.enc_layers}_fold{fold_id}_{run_id}")
    if ARGS.use_wandb:
        wandb.init(project=ARGS.wandb_project, name=run_name, reinit=True,
                   config={**vars(ARGS), "fold": fold_id,
                           "train_samples": len(train_idx),
                           "val_samples": len(val_idx)})

    best_bin_f1 = best_macro_f1 = -1.0
    best_val_loss = float("inf")
    no_improve = 0
    is_soft = (ARGS.atei_mode == "soft_cosine")

    for epoch in range(1, ARGS.epochs + 1):
        print("=" * 80)
        print(f"[Fold {fold_id}] Epoch [{epoch}/{ARGS.epochs}]")

        tr = train_one_epoch(model, train_loader, criterion, opt, scaler,
                             device, epoch, ARGS.epochs, fold_id)
        vr = validate(model, val_loader, criterion, device, fold_id)

        acc_str = f"{tr['acc']:.4f}" if tr['acc'] is not None else "N/A"
        print(f"[Train] loss={tr['loss']:.4f} acc={acc_str}")

        saved = False
        if is_soft:
            print(f"[Val]   loss(MSE)={vr['loss']:.6f}")
            if vr["loss"] < best_val_loss:
                best_val_loss = vr["loss"]; no_improve = 0
                ckpt_name = (
                    f"stage1seg_d_{run_id}_seed{ARGS.seed}_fold{fold_id}_"
                    f"best_valloss_{best_val_loss:.6f}_ep{epoch:03d}_"
                    f"lr{ARGS.lr:.0e}_d{ARGS.d_model}_l{ARGS.enc_layers}.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch, "fold": fold_id,
                    "best_val_loss": best_val_loss, "args": vars(ARGS),
                    "d_model": ARGS.d_model, "nhead": ARGS.nhead,
                    "enc_layers": ARGS.enc_layers, "dropout": ARGS.dropout,
                    "selected_by": "val_loss", "atei_level": "segment",
                    "atei_mode": ARGS.atei_mode,
                }, save_dir / ckpt_name)
                print(f"[Save best-valLoss] {best_val_loss:.6f} -> {ckpt_name}")
                saved = True
            else:
                no_improve += 1
        else:
            print(f"[Val]   loss={vr['loss']:.4f} acc={vr['acc']:.4f} "
                  f"binF1(cons)={vr['bin_f1']:.4f} macroF1={vr['macro_f1']:.4f}")
            print(f"[Val] label counts: {np.bincount(vr['y_true'], minlength=2)}")
            print(f"[Val] pred  counts: {np.bincount(vr['y_pred'], minlength=2)}")
            print(f"[Val] cm:\n{vr['cm']}")

            if vr["bin_f1"] > best_bin_f1:
                best_bin_f1 = vr["bin_f1"]; no_improve = 0
                ckpt_name = (
                    f"stage1seg_d_{run_id}_seed{ARGS.seed}_fold{fold_id}_"
                    f"best_binf1_{best_bin_f1:.4f}_ep{epoch:03d}_"
                    f"lr{ARGS.lr:.0e}_d{ARGS.d_model}_l{ARGS.enc_layers}.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch, "fold": fold_id,
                    "best_bin_f1": best_bin_f1, "best_macro_f1": best_macro_f1,
                    "val_cm": vr["cm"], "args": vars(ARGS),
                    "d_model": ARGS.d_model, "nhead": ARGS.nhead,
                    "enc_layers": ARGS.enc_layers, "dropout": ARGS.dropout,
                    "selected_by": "binary_f1", "atei_level": "segment",
                    "atei_mode": ARGS.atei_mode,
                }, save_dir / ckpt_name)
                print(f"[Save best-binF1] {best_bin_f1:.4f} -> {ckpt_name}")
                saved = True
            else:
                no_improve += 1

            if vr["macro_f1"] > best_macro_f1:
                best_macro_f1 = vr["macro_f1"]
                ckpt_name = (
                    f"stage1seg_d_{run_id}_seed{ARGS.seed}_fold{fold_id}_"
                    f"best_macrof1_{best_macro_f1:.4f}_ep{epoch:03d}_"
                    f"lr{ARGS.lr:.0e}_d{ARGS.d_model}_l{ARGS.enc_layers}.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch, "fold": fold_id,
                    "best_bin_f1": best_bin_f1, "best_macro_f1": best_macro_f1,
                    "val_cm": vr["cm"], "args": vars(ARGS),
                    "d_model": ARGS.d_model, "nhead": ARGS.nhead,
                    "enc_layers": ARGS.enc_layers, "dropout": ARGS.dropout,
                    "selected_by": "macro_f1", "atei_level": "segment",
                    "atei_mode": ARGS.atei_mode,
                }, save_dir / ckpt_name)
                print(f"[Save best-macroF1] {best_macro_f1:.4f} -> {ckpt_name}")
                saved = True

        if not saved:
            print(f"[EarlyStop] {no_improve}/{ARGS.patience}")

        if ARGS.use_wandb:
            wandb.log({"epoch": epoch,
                       "train/loss": tr["loss"],
                       "val/loss": vr["loss"],
                       "no_improve": no_improve})

        if no_improve >= ARGS.patience:
            print(f"[EarlyStop] Fold {fold_id} stop ep {epoch}")
            break

    if ARGS.use_wandb:
        wandb.finish()
    if is_soft:
        return {"bin_f1": -1.0, "macro_f1": -1.0, "best_val_loss": best_val_loss}
    return {"bin_f1": best_bin_f1, "macro_f1": best_macro_f1}


# ============================================================
# Main
# ============================================================
def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    timer = Timer()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, folds = get_stage1_kfold(n_splits=max(ARGS.kfold, 2), seed=ARGS.seed)
    if ARGS.kfold <= 1:
        folds = folds[:1]

    fold_results = []
    for f in folds:
        r = run_one_fold(f["fold"], f["train"], f["val"], run_id, device)
        fold_results.append(r)
        if ARGS.atei_mode == "soft_cosine":
            print(f"\n>>> Fold {f['fold']} best_val_loss={r.get('best_val_loss',-1):.6f}")
        else:
            print(f"\n>>> Fold {f['fold']} best binF1={r['bin_f1']:.4f} "
                  f"macroF1={r['macro_f1']:.4f}")

    print("\n" + "="*60)
    print("K-FOLD RESULT (Stage1 seg_bin daic)")
    print("="*60)
    if ARGS.atei_mode == "soft_cosine":
        for i, r in enumerate(fold_results):
            print(f"Fold {i}: best_val_loss={r.get('best_val_loss',-1):.6f}")
    else:
        for i, r in enumerate(fold_results):
            print(f"Fold {i}: binF1={r['bin_f1']:.4f}  macroF1={r['macro_f1']:.4f}")
        b = np.array([r["bin_f1"] for r in fold_results])
        m = np.array([r["macro_f1"] for r in fold_results])
        print(f"\nMean binF1   : {b.mean():.4f} +/- {b.std():.4f}")
        print(f"Mean macroF1 : {m.mean():.4f} +/- {m.std():.4f}")
    print(f"\nTotal time: {timer}")


if __name__ == "__main__":
    ARGS = parse_args()
    main()