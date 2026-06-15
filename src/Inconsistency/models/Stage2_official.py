"""
Stage2_official.py
==================
Stage2 depression detection, DAIC-WOZ official train set 3-fold CV.
Train/val on official train set (3-fold), test separately via Test_official.py.
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import wandb

from Inconsistency.datasets.Incon_seg_bin import get_Split_and_GroundTrue
from Inconsistency.models.Stage1_seg_bin_daic import atei as Stage1ATEI
from Inconsistency.utils import Timer, numpy_random_init, set_seed

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# Defaults
# ============================================================
D_MODEL = 256
NHEAD = 8
LR = 5e-4
EPOCHS = 30
TRANSFORMER_ENC_LAYERS = 1
BATCH_SIZE = 64

DROPOUT = 0.3
ATEI_DROPOUT = 0.5
WEIGHT_DECAY = 0
LABEL_SMOOTHING = 0.0

LAMBDA_ATEI = 0.1
ALPHA_INIT = 0.7
LAMBDA_AUX = 0.1
N_CLASSES = 2
MIN_SAVE_F1 = 0.60

DAIC_PSEUDO = "SegPseudoLabel_daic_distilbert_pair_bin.npz"
FEAT_DIR = "datasets/Feat_seg_bin_daic"
DAIC_DS_ROOT = "datasets/DAICWOZ"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", type=str, default=None)
    p.add_argument("--d_model", type=int, default=D_MODEL)
    p.add_argument("--nhead", type=int, default=NHEAD)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--enc_layers", type=int, default=TRANSFORMER_ENC_LAYERS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--atei_dropout", type=float, default=ATEI_DROPOUT)
    p.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    p.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    p.add_argument("--lambda_atei", type=float, default=LAMBDA_ATEI)
    p.add_argument("--alpha_init", type=float, default=ALPHA_INIT)
    p.add_argument("--lambda_aux", type=float, default=LAMBDA_AUX)
    p.add_argument("--alpha_warmup", type=int, default=0)
    p.add_argument("--lambda_warmup", type=int, default=0)
    p.add_argument("--accum_steps", type=int, default=1)
    p.add_argument("--daic_pseudo", type=str, default=DAIC_PSEUDO)
    p.add_argument("--feat_dir", type=str, default=FEAT_DIR)
    p.add_argument("--save_dir", type=str, default="weights/stage2_official")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--cache_size", type=int, default=16)
    p.add_argument("--freeze_atei", action="store_true")
    p.add_argument("--atei_lr_scale", type=float, default=0.1)
    p.add_argument("--no_atei_loss", action="store_true")
    p.add_argument("--no_atei", action="store_true")
    p.add_argument("--no_text", action="store_true")
    p.add_argument("--use_sampler", action="store_true")
    p.add_argument("--use_class_weight", action="store_true")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="Stage2 official")
    p.add_argument("--wandb_name", type=str, default=None)
    p.add_argument("--kfold", type=int, default=3)
    p.add_argument("--print_norm", action="store_true")
    p.add_argument("--min_save_f1", type=float, default=MIN_SAVE_F1)
    p.add_argument("--max_audio_frames", type=int, default=500)
    p.add_argument("--aug_min_frames", type=int, default=0)
    return p.parse_args()


# ============================================================
# Model (same as Stage2_seg_bin_daic)
# ============================================================
class whole_model(nn.Module):
    def __init__(self, embd_size, nheads, atei_ckpt_path=None,
                 atei_dropout=0.3, dropout=0.3, enc_layers=1,
                 alpha_init=0.5, inp_dim=1024, freeze_atei=False,
                 print_norm=False, no_atei=False, no_text=False):
        super().__init__()
        self.print_norm = print_norm
        self.no_atei = no_atei
        self.no_text = no_text

        self.a_in_proj = nn.Sequential(nn.Linear(inp_dim, embd_size),
                                       nn.LayerNorm(embd_size))
        a_enc = nn.TransformerEncoderLayer(
            d_model=embd_size, nhead=nheads, batch_first=True,
            dim_feedforward=4 * embd_size, dropout=dropout, norm_first=True)
        self.a_transformer_enc = nn.TransformerEncoder(
            a_enc, num_layers=enc_layers, enable_nested_tensor=False)
        self.a_post_norm = nn.LayerNorm(embd_size)
        if not no_text:
            self.t_in_proj = nn.Sequential(nn.Linear(inp_dim, embd_size),
                                           nn.LayerNorm(embd_size))
            t_enc = nn.TransformerEncoderLayer(
                d_model=embd_size, nhead=nheads, batch_first=True,
                dim_feedforward=4 * embd_size, dropout=dropout, norm_first=True)
            self.t_transformer_enc = nn.TransformerEncoder(
                t_enc, num_layers=enc_layers, enable_nested_tensor=False)
            self.t_post_norm = nn.LayerNorm(embd_size)

        if not no_atei:
            ckpt = torch.load(atei_ckpt_path, map_location="cpu")
            sd = ckpt["model_state_dict"]
            atei_d_model = int(ckpt.get("d_model",
                                        sd["a_in_proj.0.weight"].shape[0]))
            atei_nhead = int(ckpt.get("nhead", nheads))
            atei_enc_layers = int(ckpt.get("enc_layers", 1))
            print(f"[ATEI init] d_model={atei_d_model}, nhead={atei_nhead}, "
                  f"enc_layers={atei_enc_layers}")
            self.atei = Stage1ATEI(embd_size=atei_d_model, nheads=atei_nhead,
                                   dropout=atei_dropout, enc_layers=atei_enc_layers)
            self.atei.load_state_dict(sd)
            self.atei_d_model = atei_d_model
            if freeze_atei:
                for p in self.atei.parameters():
                    p.requires_grad = False
                print("[ATEI] frozen")
            self.atei_proj = nn.Linear(atei_d_model, embd_size)
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
            fusion_dim = 3 * embd_size if not no_text else 2 * embd_size
        else:
            print("[Model] no_atei=True: pure A+T baseline")
            fusion_dim = 2 * embd_size if not no_text else embd_size
        if no_text:
            print("[Model] no_text=True: text branch disabled")

        self.a_attn_pool = nn.Linear(embd_size, 1)
        if not no_text:
            self.t_attn_pool = nn.Linear(embd_size, 1)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(fusion_dim, embd_size)
        self.fc2 = nn.Linear(embd_size, embd_size)
        self.fc3 = nn.Linear(embd_size, embd_size)
        self.dep_head = nn.Linear(embd_size, N_CLASSES)
        self.aux_a_head = nn.Linear(embd_size, N_CLASSES)
        self.aux_t_head = nn.Linear(embd_size, N_CLASSES) if not no_text else None
        self.aux_e_head = nn.Linear(embd_size, N_CLASSES) if not no_atei else None

    def forward(self, xa, xt, aMask=None, tMask=None, alpha_gate=1.0):
        XA = self.a_in_proj(xa)
        HA = self.a_transformer_enc(XA, src_key_padding_mask=aMask)
        eA = self._attn_pool(HA, self.a_attn_pool, aMask)
        eA = self.a_post_norm(eA)
        aux_a = self.aux_a_head(eA)

        if not self.no_text:
            XT = self.t_in_proj(xt)
            HT = self.t_transformer_enc(XT, src_key_padding_mask=tMask)
            eT = self._attn_pool(HT, self.t_attn_pool, tMask)
            eT = self.t_post_norm(eT)
            aux_t = self.aux_t_head(eT)
        else:
            eT = None
            aux_t = None

        if not self.no_atei:
            eE_raw, atei_logits, _ = self.atei(xa, xt, aMask, tMask)
            eE = self.atei_proj(eE_raw)
            alpha = torch.clamp(self.alpha, 0.0, 2.0) * alpha_gate
            eE = eE * alpha
            aux_e = self.aux_e_head(eE)
            parts = [eA, eE] + ([eT] if eT is not None else [])
            eFusion = torch.cat(parts, dim=1)
        else:
            atei_logits = None
            aux_e = None
            parts = [eA] + ([eT] if eT is not None else [])
            eFusion = torch.cat(parts, dim=1)

        h = self.dropout(F.relu(self.fc1(eFusion)))
        h = self.dropout(F.relu(self.fc2(h)))
        dep_logits = self.dep_head(h)
        return atei_logits, dep_logits, (aux_a, aux_t, aux_e)

    @staticmethod
    def _mask_mean(x, mask):
        if mask is None:
            return x.mean(dim=1)
        valid = (~mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    @staticmethod
    def _attn_pool(x, attn_head, mask):
        scores = attn_head(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


# ============================================================
# Sample Index
# ============================================================
class Stage2SegIndex:
    def __init__(self, daic_pids, daic_depMap, daic_npz,
                 feat_dir=FEAT_DIR, daic_ds_root=DAIC_DS_ROOT):
        self.samples = []
        feat_dir = Path(feat_dir)

        if daic_pids and daic_npz:
            dpl = np.load(daic_npz)
            atei_map = {(int(p), int(s)): int(l)
                        for p, s, l in zip(dpl["patientIdx"],
                                           dpl["segIdx"], dpl["label"])}
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
                    print(f"[warn] DAIC {pid}: a={len(xa)} t={len(xt)}, "
                          f"truncate to {n_valid}")
                df = pd.read_csv(csv_path, sep="\t")
                df_p = df[df.speaker == "Participant"].dropna(subset=["value"]).copy()
                for list_idx, row in enumerate(df_p.itertuples()):
                    if list_idx >= n_valid:
                        break
                    seg_id = row.Index + 2
                    self.samples.append({
                        "patient_id": pid,
                        "seg_id": seg_id,
                        "list_idx": list_idx,
                        "dep_label": daic_depMap[pid],
                        "atei_label": atei_map.get((pid, seg_id), -1),
                    })

        print(f"[Stage2SegIndex] total={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def get_dep_counts(self):
        labs = np.array([s["dep_label"] for s in self.samples])
        return np.bincount(labs, minlength=2)

    def get_atei_counts(self):
        labs = np.array([s["atei_label"] for s in self.samples])
        valid = labs[labs >= 0]
        return np.bincount(valid, minlength=2), int((labs == -1).sum())


# ============================================================
# Dataset
# ============================================================
class Stage2SegDataset(Dataset):
    def __init__(self, sample_index, feat_dir=FEAT_DIR, cache_size=16):
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
        n = min(len(xa_list), len(xt_list))
        self._cache[pid] = (xa_list[:n], xt_list[:n])
        self._cache_order.append(pid)
        if len(self._cache_order) > self._cache_size:
            del self._cache[self._cache_order.pop(0)]
        return self._cache[pid]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        pid = s["patient_id"]
        li = s["list_idx"]
        xa_list, xt_list = self._load_patient(pid)
        if li >= len(xa_list) or li >= len(xt_list):
            raise IndexError(f"DAIC {pid} seg {s['seg_id']} list_idx {li} oob")
        return {
            "xa": xa_list[li], "xt": xt_list[li],
            "dep_label": s["dep_label"], "atei_label": s["atei_label"],
            "patient_id": pid, "seg_id": s["seg_id"],
        }


def collate_fn(batch, training=False):
    max_frames = ARGS.max_audio_frames
    aug_min = ARGS.aug_min_frames if training else 0
    xa_list = []
    for b in batch:
        frames = b["xa"][:max_frames]
        if training and aug_min > 0 and len(frames) > aug_min:
            keep = torch.randint(aug_min, len(frames) + 1, (1,)).item()
            frames = frames[:keep]
        xa_list.append(frames)
    xa = pad_sequence(xa_list, batch_first=True)
    xt = pad_sequence([b["xt"] for b in batch], batch_first=True)
    aMask = (xa.sum(dim=-1) == 0)
    tMask = (xt.sum(dim=-1) == 0)
    dep = torch.tensor([b["dep_label"] for b in batch], dtype=torch.long)
    atei = torch.tensor([b["atei_label"] for b in batch], dtype=torch.long)
    return {"xa": xa, "xt": xt, "aMask": aMask, "tMask": tMask,
            "dep": dep, "atei": atei,
            "patient_ids": [b["patient_id"] for b in batch]}


# ============================================================
# Class balance helpers
# ============================================================
def build_dep_sampler(index, seed):
    labs = np.array([s["dep_label"] for s in index.samples])
    cnt = np.bincount(labs, minlength=2)
    w = (1.0 / cnt)[labs]
    g = torch.Generator(); g.manual_seed(seed)
    return WeightedRandomSampler(weights=w, num_samples=len(w),
                                 replacement=True, generator=g)


def build_class_weight(index, device):
    labs = np.array([s["dep_label"] for s in index.samples])
    cnt = np.bincount(labs, minlength=2)
    w = cnt.sum() / (2.0 * cnt)
    w = torch.tensor(w, dtype=torch.float32, device=device)
    print(f"[class_weight] dep counts: {cnt}, weights: {w.tolist()}")
    return w


# ============================================================
# Train / Val
# ============================================================
def train_one_epoch(model, loader, loss_dep_none, loss_atei, opt, scaler,
                    device, epoch, tot_epochs, cur_lambda, fold_id,
                    accum_steps=1, lambda_aux=0.1, alpha_gate=1.0):
    model.train()
    tot_dep = tot_atei = tot = 0.0
    correct_dep = n = 0
    correct_atei = valid_atei_n = 0
    seg_true, seg_pred = [], []

    pbar = tqdm(loader, desc=f"Fold{fold_id} Train {epoch}/{tot_epochs}",
                unit="batch", leave=False)
    opt.zero_grad()
    for step, batch in enumerate(pbar):
        xa = batch["xa"].to(device, non_blocking=True)
        xt = batch["xt"].to(device, non_blocking=True)
        aMask = batch["aMask"].to(device, non_blocking=True)
        tMask = batch["tMask"].to(device, non_blocking=True)
        dep = batch["dep"].to(device, non_blocking=True)
        atei_lab = batch["atei"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                            dtype=torch.bfloat16):
            atei_logits, dep_logits, (aux_a, aux_t, aux_e) = model(
                xa, xt, aMask, tMask, alpha_gate=alpha_gate)
            L_dep = loss_dep_none(dep_logits, dep).mean()
            if atei_logits is not None:
                L_atei = (loss_atei(atei_logits, atei_lab)
                          if (atei_lab != -1).any()
                          else torch.tensor(0.0, device=device))
            else:
                L_atei = torch.tensor(0.0, device=device)
            aux_losses = [loss_dep_none(aux_a, dep).mean()]
            if aux_t is not None:
                aux_losses.append(loss_dep_none(aux_t, dep).mean())
            if aux_e is not None:
                aux_losses.append(loss_dep_none(aux_e, dep).mean())
            L_aux = sum(aux_losses) / len(aux_losses)
            L_total = L_dep + cur_lambda * L_atei + lambda_aux * L_aux

        scaler.scale(L_total / accum_steps).backward()
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()

        pred = dep_logits.argmax(dim=-1)
        correct_dep += (pred == dep).sum().item()
        valid_atei_mask = (atei_lab != -1)
        if atei_logits is not None:
            correct_atei += ((atei_logits.argmax(-1) == atei_lab) & valid_atei_mask).sum().item()
        valid_atei_n += valid_atei_mask.sum().item()
        tot_dep += L_dep.item() * dep.size(0)
        tot_atei += L_atei.item() * dep.size(0)
        tot += L_total.item() * dep.size(0)
        n += dep.size(0)
        seg_true.extend(dep.cpu().tolist())
        seg_pred.extend(pred.cpu().tolist())
        pbar.set_postfix({"dep": tot_dep/max(n,1), "atei": tot_atei/max(n,1),
                          "tot": tot/max(n,1), "acc": correct_dep/max(n,1)})

    if (step + 1) % accum_steps != 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad()

    return {"dep_loss": tot_dep/max(n,1), "atei_loss": tot_atei/max(n,1),
            "tot_loss": tot/max(n,1), "dep_acc": correct_dep/max(n,1),
            "atei_acc": correct_atei / max(valid_atei_n, 1)}


@torch.inference_mode()
def validate(model, loader, loss_dep, device, fold_id):
    model.eval()
    seg_true, seg_pred = [], []
    per_patient_scores = defaultdict(list)
    per_patient_true = {}
    tot_loss = n = 0

    for batch in tqdm(loader, desc=f"Fold{fold_id} Val", unit="batch", leave=False):
        xa = batch["xa"].to(device, non_blocking=True)
        xt = batch["xt"].to(device, non_blocking=True)
        aMask = batch["aMask"].to(device, non_blocking=True)
        tMask = batch["tMask"].to(device, non_blocking=True)
        dep = batch["dep"].to(device, non_blocking=True)
        pids = batch["patient_ids"]

        with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                            dtype=torch.bfloat16):
            _, dep_logits, _ = model(xa, xt, aMask, tMask)
            loss = loss_dep(dep_logits, dep)

        pred = dep_logits.argmax(dim=-1)
        score = (dep_logits[:, 1] - dep_logits[:, 0]).float()
        tot_loss += loss.item() * dep.size(0); n += dep.size(0)
        seg_true.extend(dep.cpu().tolist())
        seg_pred.extend(pred.cpu().tolist())
        for pid, s, t in zip(pids, score.cpu().tolist(), dep.cpu().tolist()):
            per_patient_scores[pid].append(s)
            per_patient_true.setdefault(pid, t)

    pids_list = list(per_patient_scores.keys())
    pat_means = np.array([np.mean(per_patient_scores[p]) for p in pids_list])
    pat_true  = np.array([per_patient_true[p] for p in pids_list])
    pat_pred  = (pat_means >= 0.0).astype(int)

    best_oracle_f1 = 0.0; best_thr = 0.0
    for thr in np.arange(pat_means.min(), pat_means.max(), 0.05):
        pred_thr = (pat_means >= thr).astype(int)
        f = f1_score(pat_true, pred_thr, average="macro",
                     labels=[0,1], zero_division=0)
        if f > best_oracle_f1:
            best_oracle_f1 = f; best_thr = thr

    seg_true = np.array(seg_true); seg_pred = np.array(seg_pred)
    return {
        "loss": tot_loss/max(n,1),
        "seg_macro_f1": f1_score(seg_true, seg_pred, average="macro",
                                 labels=[0,1], zero_division=0),
        "pat_acc": (pat_true == pat_pred).mean(),
        "pat_bin_f1": f1_score(pat_true, pat_pred, average="binary",
                               pos_label=1, zero_division=0),
        "pat_macro_f1": f1_score(pat_true, pat_pred, average="macro",
                                 labels=[0,1], zero_division=0),
        "pat_pre": precision_score(pat_true, pat_pred, average="binary",
                                   pos_label=1, zero_division=0),
        "pat_rec": recall_score(pat_true, pat_pred, average="binary",
                                pos_label=1, zero_division=0),
        "pat_true": pat_true, "pat_pred": pat_pred,
        "n_patients": len(per_patient_scores),
        "oracle_macro_f1": best_oracle_f1,
        "oracle_thr": best_thr,
    }


# ============================================================
# Per-fold runner
# ============================================================
def run_one_fold(fold_id, train_pids, val_pids, daic_depMap, run_id, device):
    set_seed(ARGS.seed + fold_id)
    save_dir = Path(ARGS.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\nFOLD {fold_id}\n{'='*60}")
    train_idx = Stage2SegIndex(train_pids, daic_depMap,
                               ARGS.daic_pseudo, feat_dir=ARGS.feat_dir)
    val_idx   = Stage2SegIndex(val_pids,   daic_depMap,
                               ARGS.daic_pseudo, feat_dir=ARGS.feat_dir)
    print(f"[train] dep dist: {train_idx.get_dep_counts()}")
    print(f"[val]   dep dist: {val_idx.get_dep_counts()}")

    train_ds = Stage2SegDataset(train_idx, feat_dir=ARGS.feat_dir,
                                cache_size=ARGS.cache_size)
    val_ds   = Stage2SegDataset(val_idx,   feat_dir=ARGS.feat_dir,
                                cache_size=ARGS.cache_size)

    g = torch.Generator(); g.manual_seed(ARGS.seed + fold_id)
    sampler = build_dep_sampler(train_idx, seed=ARGS.seed + fold_id) \
        if ARGS.use_sampler else None
    train_loader = DataLoader(
        train_ds, batch_size=ARGS.batch_size, sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=lambda b: collate_fn(b, training=True),
        num_workers=ARGS.num_workers, pin_memory=True,
        worker_init_fn=numpy_random_init, generator=g,
        persistent_workers=(ARGS.num_workers > 0),
        prefetch_factor=(ARGS.prefetch_factor if ARGS.num_workers > 0 else None))
    val_loader = DataLoader(
        val_ds, batch_size=ARGS.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=ARGS.num_workers, pin_memory=True,
        persistent_workers=(ARGS.num_workers > 0),
        prefetch_factor=(ARGS.prefetch_factor if ARGS.num_workers > 0 else None))

    model = whole_model(
        embd_size=ARGS.d_model, nheads=ARGS.nhead,
        atei_ckpt_path=ARGS.stage1_ckpt,
        atei_dropout=ARGS.atei_dropout, dropout=ARGS.dropout,
        enc_layers=ARGS.enc_layers, alpha_init=ARGS.alpha_init,
        freeze_atei=ARGS.freeze_atei, print_norm=ARGS.print_norm,
        no_atei=ARGS.no_atei, no_text=ARGS.no_text).to(device)

    if not ARGS.no_atei:
        atei_params = list(model.atei.parameters())
        other_params = [p for n, p in model.named_parameters()
                        if not n.startswith("atei.")]
        opt = torch.optim.Adam([
            {"params": atei_params, "lr": ARGS.lr * ARGS.atei_lr_scale,
             "weight_decay": ARGS.weight_decay},
            {"params": other_params, "lr": ARGS.lr,
             "weight_decay": ARGS.weight_decay}])
    else:
        opt = torch.optim.Adam(model.parameters(), lr=ARGS.lr,
                               weight_decay=ARGS.weight_decay)

    class_w = build_class_weight(train_idx, device) if ARGS.use_class_weight else None
    loss_dep = nn.CrossEntropyLoss(weight=class_w,
                                   label_smoothing=ARGS.label_smoothing)
    loss_dep_none = nn.CrossEntropyLoss(weight=class_w,
                                        label_smoothing=ARGS.label_smoothing,
                                        reduction="none")
    loss_atei = nn.CrossEntropyLoss(ignore_index=-1,
                                    label_smoothing=ARGS.label_smoothing)
    scaler = torch.GradScaler("cuda")

    run_name = (ARGS.wandb_name + f"_fold{fold_id}") if ARGS.wandb_name else (
        f"stage2_official_seed{ARGS.seed}_lr{ARGS.lr:.0e}_"
        f"la{ARGS.lambda_atei:.2f}_d{ARGS.d_model}_fold{fold_id}_{run_id}")
    if ARGS.use_wandb:
        wandb.init(project=ARGS.wandb_project, name=run_name, reinit=True,
                   config={**vars(ARGS), "fold": fold_id})

    best_pat_bin = best_pat_macro = -1.0
    best_oracle = -1.0
    no_improve = 0

    for epoch in range(1, ARGS.epochs + 1):
        alpha_gate = min(1.0, epoch / ARGS.alpha_warmup) if ARGS.alpha_warmup > 0 else 1.0
        cur_lambda = (0.0 if ARGS.no_atei_loss else
                      (ARGS.lambda_atei * min(1.0, epoch / ARGS.lambda_warmup)
                       if ARGS.lambda_warmup > 0 else ARGS.lambda_atei))

        print("=" * 80)
        alpha_str = f"α={float(model.alpha.detach()):.4f} " if not ARGS.no_atei else ""
        print(f"[Fold {fold_id}] Epoch [{epoch}/{ARGS.epochs}]  "
              f"{alpha_str}λ={cur_lambda:.4f}")

        tr = train_one_epoch(model, train_loader, loss_dep_none, loss_atei,
                             opt, scaler, device, epoch, ARGS.epochs,
                             cur_lambda, fold_id, accum_steps=ARGS.accum_steps,
                             lambda_aux=ARGS.lambda_aux, alpha_gate=alpha_gate)
        v = validate(model, val_loader, loss_dep, device, fold_id)

        print(f"[Train] dep_loss={tr['dep_loss']:.4f} tot={tr['tot_loss']:.4f} "
              f"dep_acc={tr['dep_acc']:.4f}")
        print(f"[Val ] patient(n={v['n_patients']}) acc={v['pat_acc']:.4f} "
              f"binF1={v['pat_bin_f1']:.4f} macroF1={v['pat_macro_f1']:.4f} "
              f"pre={v['pat_pre']:.4f} rec={v['pat_rec']:.4f}")
        print(f"[Val ] oracle macroF1={v['oracle_macro_f1']:.4f} "
              f"(thr={v['oracle_thr']:.3f})")
        print(confusion_matrix(v["pat_true"], v["pat_pred"], labels=[0, 1]))
        print(classification_report(v["pat_true"], v["pat_pred"],
                                    labels=[0, 1],
                                    target_names=["healthy(0)", "depressed(1)"],
                                    digits=4, zero_division=0))

        saved = False
        if v["oracle_macro_f1"] > best_oracle:
            best_oracle = v["oracle_macro_f1"]
        if v["pat_bin_f1"] > best_pat_bin:
            best_pat_bin = v["pat_bin_f1"]
            if best_pat_bin > ARGS.min_save_f1:
                ckpt = (f"stage2_official_{run_id}_seed{ARGS.seed}_fold{fold_id}_"
                        f"best_patBinF1_{best_pat_bin:.4f}_ep{epoch:03d}_"
                        f"lr{ARGS.lr:.0e}_d{ARGS.d_model}.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch, "fold": fold_id,
                    "best_pat_bin_f1": best_pat_bin,
                    "best_pat_macro_f1": best_pat_macro,
                    "args": vars(ARGS), "selected_by": "pat_bin_f1",
                }, save_dir / ckpt)
                print(f"[Save best-patBinF1] {best_pat_bin:.4f} -> {ckpt}")
            saved = True

        if v["pat_macro_f1"] > best_pat_macro:
            best_pat_macro = v["pat_macro_f1"]; no_improve = 0
            if best_pat_macro > ARGS.min_save_f1:
                ckpt = (f"stage2_official_{run_id}_seed{ARGS.seed}_fold{fold_id}_"
                        f"best_patMacroF1_{best_pat_macro:.4f}_ep{epoch:03d}_"
                        f"lr{ARGS.lr:.0e}_d{ARGS.d_model}.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch, "fold": fold_id,
                    "best_pat_bin_f1": best_pat_bin,
                    "best_pat_macro_f1": best_pat_macro,
                    "args": vars(ARGS), "selected_by": "pat_macro_f1",
                }, save_dir / ckpt)
                print(f"[Save best-patMacroF1] {best_pat_macro:.4f} -> {ckpt}")
            saved = True
        else:
            no_improve += 1

        if not saved:
            print(f"[EarlyStop] {no_improve}/{ARGS.patience}")

        if ARGS.use_wandb:
            wandb.log({
                "epoch": epoch,
                "alpha": float(model.alpha.detach()) if not ARGS.no_atei else 0.0,
                "train/dep_loss": tr["dep_loss"],
                "train/tot_loss": tr["tot_loss"],
                "train/dep_acc": tr["dep_acc"],
                "val/pat_bin_f1": v["pat_bin_f1"],
                "val/pat_macro_f1": v["pat_macro_f1"],
                "val/oracle_macro_f1": v["oracle_macro_f1"],
                "best/pat_bin_f1": best_pat_bin,
                "best/pat_macro_f1": best_pat_macro,
                "no_improve": no_improve})

        if no_improve >= ARGS.patience:
            print(f"[EarlyStop] Fold {fold_id} stop ep {epoch}, "
                  f"best patBinF1={best_pat_bin:.4f}, "
                  f"patMacroF1={best_pat_macro:.4f}")
            break

    if ARGS.use_wandb:
        wandb.finish()
    return {"pat_bin_f1": best_pat_bin, "pat_macro_f1": best_pat_macro,
            "oracle_macro_f1": best_oracle}


# ============================================================
# Main
# ============================================================
def main():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    timer = Timer()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 官方 train set，用 StratifiedKFold 切三折
    daic_depMap, train_pids, _ = get_Split_and_GroundTrue()
    train_pids = np.array(train_pids)
    labels = np.array([daic_depMap[p] for p in train_pids])

    skf = StratifiedKFold(n_splits=ARGS.kfold, shuffle=True,
                          random_state=ARGS.seed)
    folds = []
    for fold_id, (tr_idx, val_idx) in enumerate(skf.split(train_pids, labels)):
        folds.append({
            "fold": fold_id,
            "train": train_pids[tr_idx].tolist(),
            "val":   train_pids[val_idx].tolist(),
        })

    fold_results = []
    for f in folds:
        r = run_one_fold(f["fold"], f["train"], f["val"],
                         daic_depMap, run_id, device)
        fold_results.append(r)
        print(f"\n>>> Fold {f['fold']} best patBinF1={r['pat_bin_f1']:.4f} "
              f"patMacroF1={r['pat_macro_f1']:.4f} "
              f"oracleMacroF1={r['oracle_macro_f1']:.4f}")

    print("\n" + "="*60)
    print("K-FOLD RESULT (Stage2 official train set)")
    print("="*60)
    for i, r in enumerate(fold_results):
        print(f"Fold {i}: patBinF1={r['pat_bin_f1']:.4f}  "
              f"patMacroF1={r['pat_macro_f1']:.4f}  "
              f"oracleMacroF1={r['oracle_macro_f1']:.4f}")
    b = np.array([r["pat_bin_f1"] for r in fold_results])
    m = np.array([r["pat_macro_f1"] for r in fold_results])
    o = np.array([r["oracle_macro_f1"] for r in fold_results])
    print(f"\nMean patBinF1      : {b.mean():.4f} +/- {b.std():.4f}")
    print(f"Mean patMacroF1    : {m.mean():.4f} +/- {m.std():.4f}")
    print(f"Mean oracleMacroF1 : {o.mean():.4f} +/- {o.std():.4f}")
    print(f"\nTotal time: {timer}")


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ARGS = parse_args()
    main()