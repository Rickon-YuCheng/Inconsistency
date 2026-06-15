"""
uv run src/Inconsistency/models/Stage2_2022_phase1.py --d_model 256 --enc_layers 1 --lr 1e-4 --weight_decay 0 --dropout 0.5 --batch_size 64 --epochs 30 --kfold 3 --use_class_weight --n_runs 1

Stage2_2022_phase1.py
=====================
仿照 ICASSP2022 audio_gru_whole.py 的第一階段：
  跑 n_runs 次 KFold，把 val f1 好的 train_pids 存成 .json 供第二階段使用。

2022 原始做法：
  kf = KFold(n_splits=3, shuffle=True)
  for train_idxs_tmp, test_idxs_tmp in kf.split(audio_features):
      train(...); evaluate(...)
      if f1 > best: save(train_idxs_{f1}_{fold}.npy)

這裡改成 patient ID（.json）而非 numpy index（.npy），其餘邏輯相同。
產出：
  phase1_train_pids_{f1:.2f}_{fold}.json  （每折最好的 train_pids）
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
os.environ.setdefault("PYTORCH_SDP_BACKEND", "math")

import argparse
import json
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from Inconsistency.datasets.Incon_seg_bin import get_Split_and_GroundTrue
from Inconsistency.models.Stage1_seg_bin_daic import atei as Stage1ATEI
from Inconsistency.models.hope_adapter import HopeEncoderBlock
from Inconsistency.utils import Timer, numpy_random_init, set_seed

warnings.filterwarnings("ignore", category=FutureWarning)

N_CLASSES    = 2
FEAT_DIR     = "datasets/Feat_seg_bin_daic"
DAIC_DS_ROOT = "datasets/DAICWOZ"
DAIC_PSEUDO  = "SegPseudoLabel_daic_distilbert_pair_bin.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage1_ckpt",      type=str,
                   default="/workspace/weights/stage1_official/stage1_official_20260609_062909_seed42_best_macrof1_0.7754_ep004_lr1e-04_d256_l1.pt")
    p.add_argument("--d_model",          type=int,   default=256)
    p.add_argument("--nhead",            type=int,   default=8)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--enc_layers",       type=int,   default=1)
    p.add_argument("--batch_size",       type=int,   default=64)
    p.add_argument("--dropout",          type=float, default=0.5)
    p.add_argument("--atei_dropout",     type=float, default=0.3)
    p.add_argument("--weight_decay",     type=float, default=0)
    p.add_argument("--label_smoothing",  type=float, default=0.0)
    p.add_argument("--lambda_atei",      type=float, default=0.1)
    p.add_argument("--alpha_init",       type=float, default=0.5)
    p.add_argument("--lambda_aux",       type=float, default=0.1)
    p.add_argument("--daic_pseudo",      type=str,   default=DAIC_PSEUDO)
    p.add_argument("--feat_dir",         type=str,   default=FEAT_DIR)
    p.add_argument("--save_dir",         type=str,   default="weights/stage2_2022_phase1")
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--kfold",            type=int,   default=3)
    p.add_argument("--n_runs",           type=int,   default=5,
                   help="跑幾次 KFold（不同 seed），仿照 2022 找最好的 train_idxs")
    p.add_argument("--min_f1",           type=float, default=0.5,
                   help="val f1 超過此值才存（仿照 2022 f1_score > 0.5）")
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--prefetch_factor",  type=int,   default=2)
    p.add_argument("--cache_size",       type=int,   default=16)
    p.add_argument("--no_atei",          action="store_true")
    p.add_argument("--no_text",          action="store_true")
    p.add_argument("--use_class_weight", action="store_true")
    p.add_argument("--encoder_type",     type=str,   default="attn",
                   choices=["attn", "hope_attention"])
    p.add_argument("--cms_periods",      type=int,   nargs=2, default=[1, 4])
    p.add_argument("--max_audio_frames", type=int,   default=500)
    p.add_argument("--atei_lr_scale",    type=float, default=0.1)
    p.add_argument("--freeze_atei",      action="store_true")
    p.add_argument("--accum_steps",      type=int,   default=1)
    p.add_argument("--no_atei_loss",     action="store_true")
    return p.parse_args()


# ============================================================
# Model (same as Stage2_2022.py)
# ============================================================
class whole_model(nn.Module):
    def __init__(self, embd_size, nheads, atei_ckpt_path=None,
                 atei_dropout=0.3, dropout=0.3, enc_layers=1,
                 alpha_init=0.5, inp_dim=1024, freeze_atei=False,
                 no_atei=False, no_text=False,
                 encoder_type="attn", cms_periods=(1, 4)):
        super().__init__()
        self.no_atei      = no_atei
        self.no_text      = no_text
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
        else:
            self.a_encoder = nn.ModuleList([
                HopeEncoderBlock(dim=embd_size, heads=nheads,
                                 variant="hope_attention",
                                 cms_periods=tuple(cms_periods),
                                 hidden_multiplier=4,
                                 cms_online_updates=False)
                for _ in range(enc_layers)])
            if not no_text:
                self.t_encoder = nn.ModuleList([
                    HopeEncoderBlock(dim=embd_size, heads=nheads,
                                     variant="hope_attention",
                                     cms_periods=tuple(cms_periods),
                                     hidden_multiplier=4,
                                     cms_online_updates=False)
                    for _ in range(enc_layers)])

        self.a_post_norm = nn.LayerNorm(embd_size)
        if not no_text:
            self.t_post_norm = nn.LayerNorm(embd_size)

        if not no_atei:
            ckpt = torch.load(atei_ckpt_path, map_location="cpu")
            sd   = ckpt["model_state_dict"]
            atei_d_model    = int(ckpt.get("d_model",
                                  sd["a_in_proj.0.weight"].shape[0]))
            atei_nhead      = int(ckpt.get("nhead", nheads))
            atei_enc_layers = int(ckpt.get("enc_layers", 1))
            self.atei = Stage1ATEI(embd_size=atei_d_model, nheads=atei_nhead,
                                   dropout=atei_dropout,
                                   enc_layers=atei_enc_layers)
            self.atei.load_state_dict(sd, strict=False)
            if freeze_atei:
                for param in self.atei.parameters():
                    param.requires_grad = False
            self.atei_proj = nn.Linear(atei_d_model, embd_size)
            self.alpha     = nn.Parameter(torch.tensor(float(alpha_init)))
            fusion_dim = 3 * embd_size if not no_text else 2 * embd_size
        else:
            fusion_dim = 2 * embd_size if not no_text else embd_size

        self.a_attn_pool = nn.Linear(embd_size, 1)
        if not no_text:
            self.t_attn_pool = nn.Linear(embd_size, 1)

        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(fusion_dim, embd_size)
        self.fc2        = nn.Linear(embd_size,  embd_size)
        self.fc3        = nn.Linear(embd_size,  embd_size)
        self.dep_head   = nn.Linear(embd_size,  N_CLASSES)
        self.aux_a_head = nn.Linear(embd_size,  N_CLASSES)
        self.aux_t_head = nn.Linear(embd_size,  N_CLASSES) if not no_text else None
        self.aux_e_head = nn.Linear(embd_size,  N_CLASSES) if not no_atei else None

    def _encode(self, x, encoder, mask=None):
        if self.encoder_type == "attn":
            return encoder(x, src_key_padding_mask=mask)
        else:
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
            eE    = self.atei_proj(eE_raw)
            alpha = torch.clamp(self.alpha, 0.0, 2.0) * alpha_gate
            eE    = eE * alpha
            aux_e = self.aux_e_head(eE)
            parts = [eA, eE] + ([eT] if eT is not None else [])
            eFusion = torch.cat(parts, dim=1)
        else:
            atei_logits = None; aux_e = None
            parts = [eA] + ([eT] if eT is not None else [])
            eFusion = torch.cat(parts, dim=1)

        h = self.dropout(F.relu(self.fc1(eFusion)))
        h = self.dropout(F.relu(self.fc2(h)))
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
# Dataset (same as Stage2_2022.py)
# ============================================================
class Stage2SegIndex:
    def __init__(self, daic_pids, daic_depMap, daic_npz,
                 feat_dir=FEAT_DIR, daic_ds_root=DAIC_DS_ROOT):
        self.samples = []
        feat_dir = Path(feat_dir)
        dpl      = np.load(daic_npz)
        atei_map = {(int(p), int(s)): int(l)
                    for p, s, l in zip(dpl["patientIdx"],
                                       dpl["segIdx"], dpl["label"])}
        for pid in daic_pids:
            csv_path = (Path(daic_ds_root) / f"{pid}_P"
                        / f"{pid}_TRANSCRIPT.csv")
            if not csv_path.exists():
                continue
            a_path = feat_dir / f"{pid}_acoustic.pt"
            t_path = feat_dir / f"{pid}_text.pt"
            if not (a_path.exists() and t_path.exists()):
                continue
            xa = torch.load(str(a_path), map_location="cpu", mmap=True)
            xt = torch.load(str(t_path), map_location="cpu", mmap=True)
            n_valid = min(len(xa), len(xt))
            df   = pd.read_csv(csv_path, sep="\t")
            df_p = df[df.speaker == "Participant"].dropna(
                subset=["value"]).copy()
            for list_idx, row in enumerate(df_p.itertuples()):
                if list_idx >= n_valid:
                    break
                seg_id = row.Index + 2
                self.samples.append({
                    "patient_id": pid, "seg_id": seg_id,
                    "list_idx":   list_idx,
                    "dep_label":  daic_depMap[pid],
                    "atei_label": atei_map.get((pid, seg_id), -1),
                })

    def __len__(self):
        return len(self.samples)

    def get_dep_counts(self):
        return np.bincount(
            np.array([s["dep_label"] for s in self.samples]), minlength=2)


class Stage2SegDataset(Dataset):
    def __init__(self, sample_index, feat_dir=FEAT_DIR, cache_size=16):
        self.samples      = sample_index.samples
        self.feat_dir     = Path(feat_dir)
        self._cache       = {}
        self._cache_order = []
        self._cache_size  = cache_size

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
        xa_list, xt_list = self._load_patient(s["patient_id"])
        li = s["list_idx"]
        return {"xa": xa_list[li], "xt": xt_list[li],
                "dep_label": s["dep_label"], "atei_label": s["atei_label"],
                "patient_id": s["patient_id"], "seg_id": s["seg_id"]}


def collate_fn(batch):
    max_frames = ARGS.max_audio_frames
    xa    = pad_sequence([b["xa"][:max_frames] for b in batch], batch_first=True)
    xt    = pad_sequence([b["xt"] for b in batch], batch_first=True)
    aMask = (xa.sum(dim=-1) == 0)
    tMask = (xt.sum(dim=-1) == 0)
    dep   = torch.tensor([b["dep_label"]  for b in batch], dtype=torch.long)
    atei  = torch.tensor([b["atei_label"] for b in batch], dtype=torch.long)
    return {"xa": xa, "xt": xt, "aMask": aMask, "tMask": tMask,
            "dep": dep, "atei": atei,
            "patient_ids": [b["patient_id"] for b in batch]}


def build_class_weight(index, device):
    labs = np.array([s["dep_label"] for s in index.samples])
    cnt  = np.bincount(labs, minlength=2)
    w    = cnt.sum() / (2.0 * cnt)
    return torch.tensor(w, dtype=torch.float32, device=device)


# ============================================================
# Train / Evaluate
# ============================================================
def train_one_epoch(model, loader, loss_dep_none, loss_atei, opt, scaler,
                    device, epoch, tot, cur_lambda, lambda_aux=0.1,
                    accum_steps=1):
    model.train()
    tot_dep = tot_loss = n = correct = 0
    opt.zero_grad()
    pbar = tqdm(loader, desc=f"  Train {epoch}/{tot}", leave=False)
    for step, batch in enumerate(pbar):
        xa       = batch["xa"].to(device,   non_blocking=True)
        xt       = batch["xt"].to(device,   non_blocking=True)
        aMask    = batch["aMask"].to(device, non_blocking=True)
        tMask    = batch["tMask"].to(device, non_blocking=True)
        dep      = batch["dep"].to(device,   non_blocking=True)
        atei_lab = batch["atei"].to(device,  non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=False):
            atei_logits, dep_logits, (aux_a, aux_t, aux_e) = model(
                xa, xt, aMask, tMask)
            L_dep  = loss_dep_none(dep_logits, dep).mean()
            L_atei = (loss_atei(atei_logits, atei_lab)
                      if atei_logits is not None and (atei_lab != -1).any()
                      else torch.tensor(0.0, device=device))
            aux_losses = [loss_dep_none(aux_a, dep).mean()]
            if aux_t is not None:
                aux_losses.append(loss_dep_none(aux_t, dep).mean())
            if aux_e is not None:
                aux_losses.append(loss_dep_none(aux_e, dep).mean())
            L_total = L_dep + cur_lambda * L_atei + \
                      lambda_aux * sum(aux_losses) / len(aux_losses)

        scaler.scale(L_total / accum_steps).backward()
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()

        correct  += (dep_logits.argmax(-1) == dep).sum().item()
        tot_dep  += L_dep.item() * dep.size(0)
        tot_loss += L_total.item() * dep.size(0)
        n        += dep.size(0)
        pbar.set_postfix({"dep": tot_dep/max(n,1), "acc": correct/max(n,1)})

    if (step + 1) % accum_steps != 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad()

    return correct / max(n, 1)  # train_acc，仿照 2022 的 train_acc 判斷


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    per_patient_scores = defaultdict(list)
    per_patient_true   = {}

    for batch in loader:
        xa    = batch["xa"].to(device,   non_blocking=True)
        xt    = batch["xt"].to(device,   non_blocking=True)
        aMask = batch["aMask"].to(device, non_blocking=True)
        tMask = batch["tMask"].to(device, non_blocking=True)
        dep   = batch["dep"]
        pids  = batch["patient_ids"]

        with torch.autocast(device_type="cuda", enabled=False):
            _, dep_logits, _ = model(xa, xt, aMask, tMask)

        score = (dep_logits[:, 1] - dep_logits[:, 0]).float()
        for pid, s, t in zip(pids, score.cpu().tolist(), dep.tolist()):
            per_patient_scores[pid].append(s)
            per_patient_true.setdefault(pid, t)

    pids_list = list(per_patient_scores.keys())
    pat_means = np.array([np.mean(per_patient_scores[p]) for p in pids_list])
    pat_true  = np.array([per_patient_true[p]             for p in pids_list])
    pat_pred  = (pat_means >= 0.0).astype(int)
    return f1_score(pat_true, pat_pred, average="macro",
                    labels=[0, 1], zero_division=0)


# ============================================================
# One KFold run
# 仿照 2022 第一階段：kf.split -> train/evaluate -> 存好的 train_pids
# ============================================================
def run_kfold(run_id_str, all_pids, daic_depMap, kfold_seed, device):
    save_dir = Path(ARGS.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    all_pids_arr = np.array(all_pids)
    labels       = np.array([daic_depMap[p] for p in all_pids_arr])

    skf = StratifiedKFold(n_splits=ARGS.kfold, shuffle=True,
                          random_state=kfold_seed)

    for fold_id, (tr_idx, val_idx) in enumerate(skf.split(all_pids_arr, labels)):
        set_seed(kfold_seed + fold_id)
        train_pids = all_pids_arr[tr_idx].tolist()
        val_pids   = all_pids_arr[val_idx].tolist()

        train_idx = Stage2SegIndex(train_pids, daic_depMap,
                                   ARGS.daic_pseudo, feat_dir=ARGS.feat_dir)
        val_idx_  = Stage2SegIndex(val_pids,   daic_depMap,
                                   ARGS.daic_pseudo, feat_dir=ARGS.feat_dir)

        train_ds = Stage2SegDataset(train_idx,  feat_dir=ARGS.feat_dir,
                                    cache_size=ARGS.cache_size)
        val_ds   = Stage2SegDataset(val_idx_,   feat_dir=ARGS.feat_dir,
                                    cache_size=ARGS.cache_size)

        g = torch.Generator(); g.manual_seed(kfold_seed + fold_id)
        train_loader = DataLoader(
            train_ds, batch_size=ARGS.batch_size, shuffle=True,
            collate_fn=collate_fn,
            num_workers=ARGS.num_workers, pin_memory=True,
            worker_init_fn=numpy_random_init, generator=g,
            persistent_workers=(ARGS.num_workers > 0),
            prefetch_factor=(ARGS.prefetch_factor
                             if ARGS.num_workers > 0 else None))
        val_loader = DataLoader(
            val_ds, batch_size=ARGS.batch_size, shuffle=False,
            collate_fn=collate_fn,
            num_workers=ARGS.num_workers, pin_memory=True,
            persistent_workers=(ARGS.num_workers > 0),
            prefetch_factor=(ARGS.prefetch_factor
                             if ARGS.num_workers > 0 else None))

        model = whole_model(
            embd_size=ARGS.d_model, nheads=ARGS.nhead,
            atei_ckpt_path=ARGS.stage1_ckpt,
            atei_dropout=ARGS.atei_dropout, dropout=ARGS.dropout,
            enc_layers=ARGS.enc_layers, alpha_init=ARGS.alpha_init,
            freeze_atei=ARGS.freeze_atei,
            no_atei=ARGS.no_atei, no_text=ARGS.no_text,
            encoder_type=ARGS.encoder_type,
            cms_periods=tuple(ARGS.cms_periods)).to(device)

        if not ARGS.no_atei:
            atei_params  = list(model.atei.parameters())
            other_params = [p for nm, p in model.named_parameters()
                            if not nm.startswith("atei.")]
            opt = torch.optim.Adam([
                {"params": atei_params,  "lr": ARGS.lr * ARGS.atei_lr_scale,
                 "weight_decay": ARGS.weight_decay},
                {"params": other_params, "lr": ARGS.lr,
                 "weight_decay": ARGS.weight_decay}])
        else:
            opt = torch.optim.Adam(model.parameters(), lr=ARGS.lr,
                                   weight_decay=ARGS.weight_decay)

        class_w       = build_class_weight(train_idx, device) \
            if ARGS.use_class_weight else None
        loss_dep_none = nn.CrossEntropyLoss(weight=class_w, reduction="none")
        loss_atei     = nn.CrossEntropyLoss(ignore_index=-1)
        scaler        = torch.GradScaler("cuda")

        # 仿照 2022：fold 內有自己的 max_f1
        max_f1    = -1.0
        train_acc = -1.0

        print(f"  [Run {run_id_str} Fold {fold_id}] "
              f"train={len(train_pids)}p  val={len(val_pids)}p")

        for epoch in range(1, ARGS.epochs + 1):
            cur_lambda = 0.0 if ARGS.no_atei_loss else ARGS.lambda_atei
            train_acc  = train_one_epoch(
                model, train_loader, loss_dep_none, loss_atei,
                opt, scaler, device, epoch, ARGS.epochs,
                cur_lambda, lambda_aux=ARGS.lambda_aux,
                accum_steps=ARGS.accum_steps)
            f1 = evaluate(model, val_loader, device)

            print(f"    ep{epoch:02d} train_acc={train_acc:.3f} "
                  f"val_macroF1={f1:.4f}  max_f1={max_f1:.4f}")

            # 仿照 2022：if max_f1 <= f1_score and train_acc > 0.90 and f1_score > 0.5
            n_train = len(train_idx)
            if (max_f1 <= f1 and
                    train_acc > 0.90 and
                    f1 > ARGS.min_f1):
                max_f1 = f1
                # 存 train_pids（仿照 2022 存 train_idxs_{f1}_{fold}.npy）
                out_name = (f"train_pids_{f1:.2f}_{fold_id + 1}.json")
                out_path = save_dir / out_name
                with open(out_path, "w") as fp:
                    json.dump({"train_pids": train_pids,
                               "val_pids":   val_pids,
                               "f1":         f1,
                               "fold":       fold_id,
                               "epoch":      epoch,
                               "kfold_seed": kfold_seed}, fp)
                print(f"    [Save] {out_name}  f1={f1:.4f}")


# ============================================================
# Main
# ============================================================
def main():
    timer  = Timer()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    daic_depMap, train_pids, dev_pids = get_Split_and_GroundTrue()
    all_pids = train_pids + dev_pids
    print(f"[Phase1] all_pids={len(all_pids)} "
          f"(train={len(train_pids)}, dev={len(dev_pids)})")
    print(f"[Phase1] n_runs={ARGS.n_runs}, kfold={ARGS.kfold}, "
          f"epochs={ARGS.epochs}")

    # 仿照 2022：跑 n_runs 次不同 seed 的 KFold，找最好的 train_pids
    for run_i in range(ARGS.n_runs):
        kfold_seed = ARGS.seed + run_i
        print(f"\n{'='*60}")
        print(f"RUN {run_i+1}/{ARGS.n_runs}  kfold_seed={kfold_seed}")
        print(f"{'='*60}")
        run_kfold(f"{run_i+1}", all_pids, daic_depMap, kfold_seed, device)

    print(f"\n[Phase1 done] saved to {ARGS.save_dir}")
    print(f"Total time: {timer}")
    print(f"\n下一步：把 {ARGS.save_dir}/train_pids_*.json 傳給 Stage2_2022.py")


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    ARGS = parse_args()
    main()