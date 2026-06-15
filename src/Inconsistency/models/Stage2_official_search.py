"""
uv run src/Inconsistency/models/Stage2_official_search.py     --stage1_ckpt /workspace/weights/stage1_official/stage1_official_20260609_062909_seed42_best_macrof1_0.7754_ep004_lr1e-04_d256_l1.pt     --n_seeds 10 --n_trials 1 --seed 101
"""
"""
Stage2_official_search.py
=========================
Random search on DAIC-WOZ official train set (3-fold CV).
Saves ckpt if macroF1 > 0.6 (best) or > 0.7 (always).

uv run src/Inconsistency/models/Stage2_official_search.py \
    --stage1_ckpt weights/stage1_seg_bin_daic/<ckpt>.pt \
    --n_trials 50
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import random
import warnings
from collections import defaultdict
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

from Inconsistency.datasets.Incon_seg_bin import get_Split_and_GroundTrue
from Inconsistency.models.Stage1_seg_bin_daic import atei as Stage1ATEI
from Inconsistency.models.hope_adapter import HopeEncoderBlock
from Inconsistency.utils import Timer, numpy_random_init, set_seed

warnings.filterwarnings("ignore", category=FutureWarning)

N_CLASSES = 2
FEAT_DIR = "datasets/Feat_seg_bin_daic"
DAIC_DS_ROOT = "datasets/DAICWOZ"
DAIC_PSEUDO = "SegPseudoLabel_daic_distilbert_pair_bin.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--feat_dir", type=str, default=FEAT_DIR)
    p.add_argument("--daic_pseudo", type=str, default=DAIC_PSEUDO)
    p.add_argument("--save_dir", type=str, default="weights/stage2_official_search")
    p.add_argument("--n_trials", type=int, default=50)
    # manual hparams (used when n_trials=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--atei_dropout", type=float, default=0.5)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--enc_layers", type=int, default=1)
    p.add_argument("--lambda_atei", type=float, default=0.7)
    p.add_argument("--lambda_aux", type=float, default=0.3)
    p.add_argument("--alpha_init", type=float, default=0.7)
    p.add_argument("--atei_lr_scale", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--encoder_type", type=str, default="hope_attention",
                   choices=["attn", "hope_attention"])
    p.add_argument("--cms_periods", type=int, nargs=2, default=[1, 4])
    p.add_argument("--aug_min_frames", type=int, default=0)
    p.add_argument("--no_atei", action="store_true")
    p.add_argument("--no_text", action="store_true")
    p.add_argument("--n_seeds", type=int, default=1,
                   help="number of seeds to run, seed=base_seed+i")
    p.add_argument("--kfold", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--cache_size", type=int, default=16)
    p.add_argument("--max_audio_frames", type=int, default=500)
    return p.parse_args()


# ============================================================
# Hyperparameter search space
# ============================================================
def sample_hparams(rng, seed=42):
    return {
        "lr":           rng.choice([1e-4, 3e-4, 5e-4, 1e-3]),
        "weight_decay": rng.choice([0.0, 1e-4, 1e-3]),
        "dropout":      rng.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
        "atei_dropout": rng.choice([0.1, 0.2, 0.3]),
        "d_model":      rng.choice([128, 256]),
        "enc_layers":   rng.choice([1, 2]),
        "lambda_atei":  rng.choice([0.0, 0.05, 0.1, 0.3]),
        "lambda_aux":   rng.choice([0.0, 0.1, 0.3]),
        "alpha_init":   rng.choice([0.3, 0.5, 0.7, 1.0]),
        "atei_lr_scale":rng.choice([0.0, 0.01, 0.1, 1.0]),
        "batch_size":   rng.choice([32, 64]),
        "epochs":       rng.choice([30, 50, 70]),
        "patience":     rng.choice([10, 15, 20]),
        "encoder_type": rng.choice(["attn", "hope_attention"]),
        "cms_periods":  [(1, 4), (1, 8), (2, 4)][rng.randint(0, 3)],
        "aug_min_frames": rng.choice([0, 30, 50]),
        "no_atei":      rng.choice([False, False, False, True]),  # 75% with ATEI
        "no_text":      rng.choice([False, False, False, True]),
        "seed":         seed,
    }


# ============================================================
# Model
# ============================================================
class whole_model(nn.Module):
    def __init__(self, embd_size, nheads, atei_ckpt_path,
                 atei_dropout=0.3, dropout=0.3, enc_layers=1,
                 alpha_init=0.5, inp_dim=1024,
                 encoder_type="attn", cms_periods=(1,4),
                 no_atei=False, no_text=False):
        super().__init__()
        self.no_atei = no_atei
        self.no_text = no_text
        self.encoder_type = encoder_type

        self.a_in_proj = nn.Sequential(nn.Linear(inp_dim, embd_size),
                                       nn.LayerNorm(embd_size))
        if not no_text:
            self.t_in_proj = nn.Sequential(nn.Linear(inp_dim, embd_size),
                                           nn.LayerNorm(embd_size))

        if encoder_type == "attn":
            a_enc = nn.TransformerEncoderLayer(
                d_model=embd_size, nhead=nheads, batch_first=True,
                dim_feedforward=4 * embd_size, dropout=dropout, norm_first=True)
            self.a_encoder = nn.TransformerEncoder(
                a_enc, num_layers=enc_layers, enable_nested_tensor=False)
            if not no_text:
                t_enc = nn.TransformerEncoderLayer(
                    d_model=embd_size, nhead=nheads, batch_first=True,
                    dim_feedforward=4 * embd_size, dropout=dropout, norm_first=True)
                self.t_encoder = nn.TransformerEncoder(
                    t_enc, num_layers=enc_layers, enable_nested_tensor=False)
        else:  # hope_attention
            self.a_encoder = nn.ModuleList([
                HopeEncoderBlock(
                    dim=embd_size, heads=nheads, variant="hope_attention",
                    cms_periods=tuple(cms_periods), hidden_multiplier=4,
                    cms_online_updates=False,
                ) for _ in range(enc_layers)
            ])
            if not no_text:
                self.t_encoder = nn.ModuleList([
                    HopeEncoderBlock(
                        dim=embd_size, heads=nheads, variant="hope_attention",
                        cms_periods=tuple(cms_periods), hidden_multiplier=4,
                        cms_online_updates=False,
                    ) for _ in range(enc_layers)
                ])

        self.a_post_norm = nn.LayerNorm(embd_size)
        if not no_text:
            self.t_post_norm = nn.LayerNorm(embd_size)

        if not no_atei:
            ckpt = torch.load(atei_ckpt_path, map_location="cpu")
            sd = ckpt["model_state_dict"]
            atei_d_model    = int(ckpt.get("d_model", sd["a_in_proj.0.weight"].shape[0]))
            atei_nhead      = int(ckpt.get("nhead", nheads))
            atei_enc_layers = int(ckpt.get("enc_layers", 1))
            self.atei = Stage1ATEI(embd_size=atei_d_model, nheads=atei_nhead,
                                   dropout=atei_dropout, enc_layers=atei_enc_layers)
            self.atei.load_state_dict(sd, strict=False)
            self.atei_proj = nn.Linear(atei_d_model, embd_size)
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
            fusion_dim = 3 * embd_size if not no_text else 2 * embd_size
        else:
            fusion_dim = 2 * embd_size if not no_text else embd_size

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

    def _encode(self, x, encoder, mask=None):
        if self.encoder_type == "attn":
            return encoder(x, src_key_padding_mask=mask)
        else:  # hope_attention
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            for layer in encoder:
                x = layer(x)
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            return x

    def forward(self, xa, xt, aMask=None, tMask=None, alpha_gate=1.0):
        XA = self.a_in_proj(xa)
        HA = self._encode(XA, self.a_encoder, aMask)
        eA = self._attn_pool(HA, self.a_attn_pool, aMask)
        eA = self.a_post_norm(eA)
        aux_a = self.aux_a_head(eA)

        if not self.no_text:
            XT = self.t_in_proj(xt)
            HT = self._encode(XT, self.t_encoder, tMask)
            eT = self._attn_pool(HT, self.t_attn_pool, tMask)
            eT = self.t_post_norm(eT)
            aux_t = self.aux_t_head(eT)
        else:
            eT = None; aux_t = None

        if not self.no_atei:
            eE_raw, atei_logits, _ = self.atei(xa, xt, aMask, tMask)
            eE = self.atei_proj(eE_raw)
            alpha = torch.clamp(self.alpha, 0.0, 2.0) * alpha_gate
            eE = eE * alpha
            aux_e = self.aux_e_head(eE)
            parts = [eA, eE] + ([eT] if eT is not None else [])
        else:
            atei_logits = None; aux_e = None
            parts = [eA] + ([eT] if eT is not None else [])
        eFusion = torch.cat(parts, dim=1)

        h = self.dropout(F.relu(self.fc1(eFusion)))
        h = self.dropout(F.relu(self.fc2(h)))
        h = self.dropout(F.relu(self.fc3(h)))
        dep_logits = self.dep_head(h)
        return atei_logits, dep_logits, (aux_a, aux_t, aux_e)

    @staticmethod
    def _attn_pool(x, attn_head, mask):
        scores = attn_head(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


# ============================================================
# Dataset
# ============================================================
class Stage2SegIndex:
    def __init__(self, daic_pids, daic_depMap, daic_npz,
                 feat_dir=FEAT_DIR, daic_ds_root=DAIC_DS_ROOT):
        self.samples = []
        feat_dir = Path(feat_dir)
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
                continue
            xa = torch.load(str(a_path), map_location="cpu", mmap=True)
            xt = torch.load(str(t_path), map_location="cpu", mmap=True)
            n_valid = min(len(xa), len(xt))
            df = pd.read_csv(csv_path, sep="\t")
            df_p = df[df.speaker == "Participant"].dropna(subset=["value"]).copy()
            for list_idx, row in enumerate(df_p.itertuples()):
                if list_idx >= n_valid:
                    break
                seg_id = row.Index + 2
                self.samples.append({
                    "patient_id": pid, "seg_id": seg_id,
                    "list_idx": list_idx,
                    "dep_label": daic_depMap[pid],
                    "atei_label": atei_map.get((pid, seg_id), -1),
                })

    def __len__(self):
        return len(self.samples)

    def get_dep_counts(self):
        labs = np.array([s["dep_label"] for s in self.samples])
        return np.bincount(labs, minlength=2)


class Stage2SegDataset(Dataset):
    def __init__(self, sample_index, feat_dir=FEAT_DIR, cache_size=16):
        self.samples = sample_index.samples
        self.feat_dir = Path(feat_dir)
        self._cache = {}
        self._cache_order = []
        self._cache_size = cache_size

    def _load_patient(self, pid):
        if pid in self._cache:
            self._cache_order.remove(pid); self._cache_order.append(pid)
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
        xa_list, xt_list = self._load_patient(s["patient_id"])
        li = s["list_idx"]
        return {"xa": xa_list[li], "xt": xt_list[li],
                "dep_label": s["dep_label"], "atei_label": s["atei_label"],
                "patient_id": s["patient_id"]}


def make_collate(max_audio_frames, aug_min_frames, training=False):
    def collate_fn(batch):
        xa_list = []
        for b in batch:
            frames = b["xa"][:max_audio_frames]
            if training and aug_min_frames > 0 and len(frames) > aug_min_frames:
                keep = torch.randint(aug_min_frames, len(frames)+1, (1,)).item()
                frames = frames[:keep]
            xa_list.append(frames)
        xa = pad_sequence(xa_list, batch_first=True)
        xt = pad_sequence([b["xt"] for b in batch], batch_first=True)
        aMask = (xa.sum(dim=-1) == 0)
        tMask = (xt.sum(dim=-1) == 0)
        dep  = torch.tensor([b["dep_label"]  for b in batch], dtype=torch.long)
        atei = torch.tensor([b["atei_label"] for b in batch], dtype=torch.long)
        return {"xa": xa, "xt": xt, "aMask": aMask, "tMask": tMask,
                "dep": dep, "atei": atei,
                "patient_ids": [b["patient_id"] for b in batch]}
    return collate_fn


def build_class_weight(index, device):
    labs = np.array([s["dep_label"] for s in index.samples])
    cnt = np.bincount(labs, minlength=2)
    w = cnt.sum() / (2.0 * cnt)
    return torch.tensor(w, dtype=torch.float32, device=device)


# ============================================================
# Train / Val
# ============================================================
def train_one_epoch(model, loader, loss_dep_none, loss_atei, opt, scaler,
                    device, epoch, tot, cur_lambda, lambda_aux):
    model.train()
    tot_dep = tot_loss = n = 0.0
    correct = valid_atei = correct_atei = 0
    opt.zero_grad()
    pbar = tqdm(loader, desc=f"  Train {epoch}/{tot}", leave=False)
    for step, batch in enumerate(pbar):
        xa  = batch["xa"].to(device, non_blocking=True)
        xt  = batch["xt"].to(device, non_blocking=True)
        aMask = batch["aMask"].to(device, non_blocking=True)
        tMask = batch["tMask"].to(device, non_blocking=True)
        dep = batch["dep"].to(device, non_blocking=True)
        atei_lab = batch["atei"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=False):
            atei_logits, dep_logits, (aux_a, aux_t, aux_e) = model(
                xa, xt, aMask, tMask)
            L_dep = loss_dep_none(dep_logits, dep).mean()
            if atei_logits is not None and (atei_lab != -1).any():
                L_atei = loss_atei(atei_logits, atei_lab)
            else:
                L_atei = torch.tensor(0.0, device=device)
            aux_losses = [loss_dep_none(aux_a, dep).mean()]
            if aux_t is not None:
                aux_losses.append(loss_dep_none(aux_t, dep).mean())
            if aux_e is not None:
                aux_losses.append(loss_dep_none(aux_e, dep).mean())
            L_aux = sum(aux_losses) / len(aux_losses)
            L_total = L_dep + cur_lambda * L_atei + lambda_aux * L_aux

        scaler.scale(L_total).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad()

        correct += (dep_logits.argmax(-1) == dep).sum().item()
        tot_dep  += L_dep.item() * dep.size(0)
        tot_loss += L_total.item() * dep.size(0)
        n += dep.size(0)
        pbar.set_postfix({"dep": tot_dep/max(n,1), "acc": correct/max(n,1)})

    return {"dep_loss": tot_dep/max(n,1), "dep_acc": correct/max(n,1)}


@torch.inference_mode()
def validate(model, loader, loss_dep, device):
    model.eval()
    per_patient_scores = defaultdict(list)
    per_patient_true = {}
    tot_loss = n = 0

    for batch in loader:
        xa  = batch["xa"].to(device, non_blocking=True)
        xt  = batch["xt"].to(device, non_blocking=True)
        aMask = batch["aMask"].to(device, non_blocking=True)
        tMask = batch["tMask"].to(device, non_blocking=True)
        dep = batch["dep"].to(device, non_blocking=True)
        pids = batch["patient_ids"]
        with torch.autocast(device_type="cuda", enabled=False):
            _, dep_logits, _ = model(xa, xt, aMask, tMask)
            loss = loss_dep(dep_logits, dep)
        score = (dep_logits[:,1] - dep_logits[:,0]).float()
        tot_loss += loss.item() * dep.size(0); n += dep.size(0)
        for pid, s, t in zip(pids, score.cpu().tolist(), dep.cpu().tolist()):
            per_patient_scores[pid].append(s)
            per_patient_true.setdefault(pid, t)

    pids_list = list(per_patient_scores.keys())
    pat_means = np.array([np.mean(per_patient_scores[p]) for p in pids_list])
    pat_true  = np.array([per_patient_true[p] for p in pids_list])
    pat_pred  = (pat_means >= 0.0).astype(int)
    macro_f1  = f1_score(pat_true, pat_pred, average="macro",
                         labels=[0,1], zero_division=0)
    return {"pat_macro_f1": macro_f1, "pat_true": pat_true, "pat_pred": pat_pred,
            "n_patients": len(pids_list)}


# ============================================================
# One trial
# ============================================================
def run_trial(trial_id, hp, train_pids, daic_depMap, folds, run_id, device):
    save_dir = Path(ARGS.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    # convert numpy types to python natives
    hp = {k: (int(v) if isinstance(v, np.integer) else
              float(v) if isinstance(v, np.floating) else
              bool(v) if isinstance(v, np.bool_) else
              str(v) if isinstance(v, np.str_) else v)
          for k, v in hp.items()}
    nhead = 8 if hp["d_model"] >= 256 else 4

    fold_macros = []
    for fold_id, (tr_pids, val_pids) in enumerate(folds):
        set_seed(hp["seed"] + trial_id * 100 + fold_id)

        train_idx = Stage2SegIndex(tr_pids,  daic_depMap, ARGS.daic_pseudo,
                                   feat_dir=ARGS.feat_dir)
        val_idx   = Stage2SegIndex(val_pids, daic_depMap, ARGS.daic_pseudo,
                                   feat_dir=ARGS.feat_dir)

        train_ds = Stage2SegDataset(train_idx, feat_dir=ARGS.feat_dir,
                                    cache_size=ARGS.cache_size)
        val_ds   = Stage2SegDataset(val_idx,   feat_dir=ARGS.feat_dir,
                                    cache_size=ARGS.cache_size)

        g = torch.Generator(); g.manual_seed(hp["seed"] + trial_id * 100 + fold_id)
        train_loader = DataLoader(
            train_ds, batch_size=hp["batch_size"], shuffle=True,
            collate_fn=make_collate(ARGS.max_audio_frames, hp["aug_min_frames"],
                                    training=True),
            num_workers=ARGS.num_workers, pin_memory=True,
            worker_init_fn=numpy_random_init, generator=g,
            persistent_workers=(ARGS.num_workers > 0),
            prefetch_factor=(ARGS.prefetch_factor if ARGS.num_workers > 0 else None))
        val_loader = DataLoader(
            val_ds, batch_size=hp["batch_size"], shuffle=False,
            collate_fn=make_collate(ARGS.max_audio_frames, 0, training=False),
            num_workers=ARGS.num_workers, pin_memory=True,
            persistent_workers=(ARGS.num_workers > 0),
            prefetch_factor=(ARGS.prefetch_factor if ARGS.num_workers > 0 else None))

        try:
            model = whole_model(
                embd_size=hp["d_model"], nheads=nhead,
                atei_ckpt_path=ARGS.stage1_ckpt,
                atei_dropout=hp["atei_dropout"], dropout=hp["dropout"],
                enc_layers=hp["enc_layers"], alpha_init=hp["alpha_init"],
                encoder_type=hp["encoder_type"],
                cms_periods=hp["cms_periods"],
                no_atei=hp["no_atei"], no_text=hp["no_text"]).to(device)
        except Exception as e:
            print(f"  [Trial {trial_id} Fold {fold_id}] model init error: {e}")
            return -1.0, None

        if not hp["no_atei"] and hp["atei_lr_scale"] > 0:
            atei_params  = list(model.atei.parameters())
            other_params = [p for n, p in model.named_parameters()
                            if not n.startswith("atei.")]
            opt = torch.optim.Adam([
                {"params": atei_params,  "lr": hp["lr"] * hp["atei_lr_scale"],
                 "weight_decay": hp["weight_decay"]},
                {"params": other_params, "lr": hp["lr"],
                 "weight_decay": hp["weight_decay"]}])
        else:
            if not hp["no_atei"]:
                for p in model.atei.parameters():
                    p.requires_grad = False
            opt = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=hp["lr"], weight_decay=hp["weight_decay"])

        class_w    = build_class_weight(train_idx, device)
        loss_dep   = nn.CrossEntropyLoss(weight=class_w)
        loss_dep_none = nn.CrossEntropyLoss(weight=class_w, reduction="none")
        loss_atei  = nn.CrossEntropyLoss(ignore_index=-1)
        scaler     = torch.GradScaler("cuda")

        best_macro = -1.0
        best_sd    = None
        best_epoch = -1
        no_improve = 0

        for epoch in range(1, hp["epochs"] + 1):
            train_one_epoch(model, train_loader, loss_dep_none, loss_atei,
                            opt, scaler, device, epoch, hp["epochs"],
                            hp["lambda_atei"], hp["lambda_aux"])
            v = validate(model, val_loader, loss_dep, device)
            mf1 = v["pat_macro_f1"]

            if mf1 > best_macro:
                best_macro = mf1
                best_sd    = {k: v.cpu().clone() for k, v in
                              model.state_dict().items()}
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= hp["patience"]:
                break

        # save best if > 0.6
        if best_macro > 0.6 and best_sd is not None:
            ckpt_name = (
                f"search_{run_id}_s{hp['seed']:03d}_trial{trial_id:03d}_fold{fold_id}_"
                f"best_macroF1_{best_macro:.4f}_ep{best_epoch:03d}.pt")
            torch.save({
                "model_state_dict": best_sd,
                "epoch": best_epoch, "trial": trial_id, "fold": fold_id,
                "pat_macro_f1": best_macro, "hparams": hp,
            }, save_dir / ckpt_name)
            print(f"  [>0.6 Save] best macroF1={best_macro:.4f} -> {ckpt_name}")

        fold_macros.append(best_macro)
        print(f"  Fold {fold_id}: best macroF1={best_macro:.4f}")

    mean_macro = float(np.mean(fold_macros))
    return mean_macro, hp


# ============================================================
# Main
# ============================================================
def main():
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    timer   = Timer()
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    rng     = np.random.RandomState(ARGS.seed)

    daic_depMap, train_pids, _ = get_Split_and_GroundTrue()
    train_pids = np.array(train_pids)
    labels = np.array([daic_depMap[p] for p in train_pids])

    skf = StratifiedKFold(n_splits=ARGS.kfold, shuffle=True,
                          random_state=ARGS.seed)
    folds = [(train_pids[tr].tolist(), train_pids[val].tolist())
             for tr, val in skf.split(train_pids, labels)]

    best_overall = -1.0
    best_hp      = None
    results      = []

    total_trials = ARGS.n_seeds * ARGS.n_trials
    print(f"Random search: {ARGS.n_seeds} seeds x {ARGS.n_trials} trials = {total_trials} total, {ARGS.kfold}-fold")
    global_trial = 0
    for seed_i in range(ARGS.n_seeds):
        cur_seed = ARGS.seed + seed_i
        seed_rng = np.random.RandomState(cur_seed)
        for trial_id in range(ARGS.n_trials):
            if ARGS.n_trials == 1:
                hp = {
                    "lr": ARGS.lr, "weight_decay": ARGS.weight_decay,
                    "dropout": ARGS.dropout, "atei_dropout": ARGS.atei_dropout,
                    "d_model": ARGS.d_model, "enc_layers": ARGS.enc_layers,
                    "lambda_atei": ARGS.lambda_atei, "lambda_aux": ARGS.lambda_aux,
                    "alpha_init": ARGS.alpha_init, "atei_lr_scale": ARGS.atei_lr_scale,
                    "batch_size": ARGS.batch_size, "epochs": ARGS.epochs,
                    "patience": ARGS.patience, "encoder_type": ARGS.encoder_type,
                    "cms_periods": tuple(ARGS.cms_periods),
                    "aug_min_frames": ARGS.aug_min_frames,
                    "no_atei": ARGS.no_atei, "no_text": ARGS.no_text,
                    "seed": cur_seed,
                }
            else:
                hp = sample_hparams(seed_rng, seed=cur_seed)
            global_trial += 1
            print(f"\n{'='*60}")
            print(f"Seed {seed_i+1}/{ARGS.n_seeds} Trial {trial_id+1}/{ARGS.n_trials} (global {global_trial}/{total_trials})")
            print(f"  {hp}")
            mean_macro, used_hp = run_trial(global_trial, hp, train_pids,
                                            daic_depMap, folds, run_id, device)
            results.append({"trial": global_trial, "mean_macro_f1": mean_macro, "hp": hp})
            print(f"  => mean macroF1={mean_macro:.4f}")

            if mean_macro > best_overall:
                best_overall = mean_macro
                best_hp = hp
                print(f"  ** New best! mean macroF1={best_overall:.4f}")

    print("\n" + "="*60)
    print("SEARCH COMPLETE")
    print("="*60)
    results.sort(key=lambda x: x["mean_macro_f1"], reverse=True)
    print("Top 5 trials:")
    for r in results[:5]:
        print(f"  Trial {r['trial']:3d}: mean macroF1={r['mean_macro_f1']:.4f}")
        print(f"    {r['hp']}")
    print(f"\nBest mean macroF1: {best_overall:.4f}")
    print(f"Best hparams: {best_hp}")
    print(f"\nTotal time: {timer}")


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ARGS = parse_args()
    main()