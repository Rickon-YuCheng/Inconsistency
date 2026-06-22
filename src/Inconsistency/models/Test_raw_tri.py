"""
Test_raw_tri.py
===============
Test on DAIC-WOZ official dev set using a trained Stage2_raw_tri ckpt.
三元分類版本: Healthy(0) / Mild(1) / Moderate+(2)

uv run src/Inconsistency/models/Test_raw_tri.py \
    --stage2_ckpt weights/stage2_raw_tri/<ckpt>.pt
"""

import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Inconsistency.datasets.Incon_seg_bin_tri import get_Split_and_GroundTrue_tri
from Inconsistency.models.Stage1_raw import atei as Stage1ATEI
from Inconsistency.utils import set_seed

warnings.filterwarnings("ignore", category=FutureWarning)

N_CLASSES = 3
FEAT_DIR     = "datasets/Feat_raw"
DAIC_DS_ROOT = "datasets/DAICWOZ"
DAIC_PSEUDO  = "SegPseudoLabel_daic_distilbert_pair_bin.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2_ckpt", type=str, required=True)
    p.add_argument("--feat_dir",    type=str, default=FEAT_DIR)
    p.add_argument("--daic_pseudo", type=str, default=DAIC_PSEUDO)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--cache_size",  type=int, default=16)
    p.add_argument("--max_audio_frames", type=int, default=500)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


# ============================================================
# Model（與 Stage2_raw_tri 完全一致）
# ============================================================
class whole_model(nn.Module):
    def __init__(self, embd_size, nheads, atei_ckpt_sd,
                 atei_d_model, atei_nhead, atei_enc_layers,
                 atei_dropout=0.3, dropout=0.3, enc_layers=1,
                 alpha_init=0.5, inp_dim=1024,
                 no_atei=False, no_text=False):
        super().__init__()
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
            self.atei = Stage1ATEI(embd_size=atei_d_model, nheads=atei_nhead,
                                   dropout=atei_dropout, enc_layers=atei_enc_layers)
            self.atei.load_state_dict(atei_ckpt_sd)
            self.atei_proj = nn.Linear(atei_d_model, embd_size)
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
            fusion_dim = 3 * embd_size if not no_text else 2 * embd_size
        else:
            fusion_dim = 2 * embd_size if not no_text else embd_size

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(fusion_dim, embd_size)
        self.fc2 = nn.Linear(embd_size, embd_size)
        self.fc3 = nn.Linear(embd_size, embd_size)
        self.dep_head   = nn.Linear(embd_size, N_CLASSES)
        self.aux_a_head = nn.Linear(embd_size, N_CLASSES)
        self.aux_t_head = nn.Linear(embd_size, N_CLASSES) if not no_text else None
        self.aux_e_head = nn.Linear(embd_size, N_CLASSES) if not no_atei else None

    def forward(self, xa, xt, aMask=None, tMask=None):
        XA = self.a_in_proj(xa)
        HA = self.a_transformer_enc(XA, src_key_padding_mask=aMask)
        eA = self._mask_mean(HA, aMask)
        eA = self.a_post_norm(eA)

        if not self.no_text:
            XT = self.t_in_proj(xt)
            HT = self.t_transformer_enc(XT, src_key_padding_mask=tMask)
            eT = self._mask_mean(HT, tMask)
            eT = self.t_post_norm(eT)
        else:
            eT = None

        if not self.no_atei:
            eE_raw, _ = self.atei(xa, xt, aMask, tMask)
            eE = self.atei_proj(eE_raw)
            alpha = torch.clamp(self.alpha, 0.0, 2.0)
            eE = eE * alpha
            parts = [eA, eE] + ([eT] if eT is not None else [])
        else:
            parts = [eA] + ([eT] if eT is not None else [])
        eFusion = torch.cat(parts, dim=1)

        h = self.dropout(F.relu(self.fc1(eFusion)))
        h = self.dropout(F.relu(self.fc2(h)))
        h = self.dropout(F.relu(self.fc3(h)))
        return self.dep_head(h)

    @staticmethod
    def _mask_mean(x, mask):
        if mask is None:
            return x.mean(dim=1)
        valid = (~mask).unsqueeze(-1).float()
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


# ============================================================
# Dataset
# ============================================================
class TestSegIndex:
    def __init__(self, dev_pids, daic_depMap, feat_dir, daic_ds_root=DAIC_DS_ROOT):
        self.samples = []
        feat_dir = Path(feat_dir)
        for pid in dev_pids:
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
            df = pd.read_csv(csv_path, sep="\t")
            df_p = df[df.speaker == "Participant"].dropna(subset=["value"]).copy()
            for list_idx, row in enumerate(df_p.itertuples()):
                if list_idx >= n_valid:
                    break
                self.samples.append({
                    "patient_id": pid,
                    "list_idx":   list_idx,
                    "dep_label":  daic_depMap[pid],
                })
        print(f"[TestSegIndex] total={len(self.samples)}")


class TestSegDataset(Dataset):
    def __init__(self, sample_index, feat_dir, cache_size=16):
        self.samples  = sample_index.samples
        self.feat_dir = Path(feat_dir)
        self._cache   = {}
        self._cache_order = []
        self._cache_size  = cache_size

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

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        xa_list, xt_list = self._load_patient(s["patient_id"])
        li = s["list_idx"]
        return {"xa": xa_list[li], "xt": xt_list[li],
                "dep_label": s["dep_label"],
                "patient_id": s["patient_id"]}


def collate_fn(batch):
    max_frames = ARGS.max_audio_frames
    xa = pad_sequence([b["xa"][:max_frames] for b in batch], batch_first=True)
    xt = pad_sequence([b["xt"] for b in batch], batch_first=True)
    aMask = (xa.sum(dim=-1) == 0)
    tMask = (xt.sum(dim=-1) == 0)
    dep = torch.tensor([b["dep_label"] for b in batch], dtype=torch.long)
    return {"xa": xa, "xt": xt, "aMask": aMask, "tMask": tMask,
            "dep": dep, "patient_ids": [b["patient_id"] for b in batch]}


# ============================================================
# Inference
# ============================================================
@torch.inference_mode()
def test(model, loader, device):
    model.eval()
    per_patient_logits = defaultdict(list)
    per_patient_true   = {}

    for batch in tqdm(loader, desc="Test", unit="batch", leave=False):
        xa    = batch["xa"].to(device, non_blocking=True)
        xt    = batch["xt"].to(device, non_blocking=True)
        aMask = batch["aMask"].to(device, non_blocking=True)
        tMask = batch["tMask"].to(device, non_blocking=True)
        dep   = batch["dep"]
        pids  = batch["patient_ids"]

        with torch.autocast(device_type="cuda", enabled=False):
            dep_logits = model(xa, xt, aMask, tMask)

        logits_f = dep_logits.float().cpu()
        for pid, logit, t in zip(pids, logits_f.tolist(), dep.tolist()):
            per_patient_logits[pid].append(logit)
            per_patient_true.setdefault(pid, t)

    pids_list       = list(per_patient_logits.keys())
    pat_mean_logits = np.array([np.mean(per_patient_logits[p], axis=0)
                                 for p in pids_list])
    pat_pred = pat_mean_logits.argmax(axis=1)
    pat_true = np.array([per_patient_true[p] for p in pids_list])

    return {
        "pat_true":     pat_true,
        "pat_pred":     pat_pred,
        "pat_acc":      accuracy_score(pat_true, pat_pred),
        "pat_macro_f1": f1_score(pat_true, pat_pred, average="macro",
                                 labels=[0, 1, 2], zero_division=0),
        "n_patients":   len(pids_list),
    }


# ============================================================
# Main
# ============================================================
def main():
    set_seed(ARGS.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ARGS.stage2_ckpt, map_location="cpu", weights_only=False)
    hp   = ckpt.get("args", {})
    sd   = ckpt["model_state_dict"]

    print(f"[Ckpt] epoch={ckpt.get('epoch','?')} fold={ckpt.get('fold','?')} "
          f"val_macro={ckpt.get('best_pat_macro_f1', -1):.4f}")

    d_model      = int(hp.get("d_model", 256))
    enc_layers   = int(hp.get("enc_layers", 1))
    dropout      = float(hp.get("dropout", 0.3))
    atei_dropout = float(hp.get("atei_dropout", 0.3))
    alpha_init   = float(hp.get("alpha_init", 0.5))
    no_atei      = bool(hp.get("no_atei", False))
    no_text      = bool(hp.get("no_text", False))
    nhead        = 8 if d_model >= 256 else 4

    if not no_atei:
        atei_d_model    = sd["atei.a_in_proj.0.weight"].shape[0]
        atei_enc_layers = enc_layers
        atei_nhead      = nhead
        # 從 stage1 ckpt 重新 load
        stage1_ckpt_path = hp.get("stage1_ckpt")
        if stage1_ckpt_path is None:
            raise ValueError("hparams 裡沒有 stage1_ckpt")
        print(f"[ATEI] loading from: {stage1_ckpt_path}")
        stage1_ckpt = torch.load(stage1_ckpt_path, map_location="cpu", weights_only=False)
        atei_sd = stage1_ckpt["model_state_dict"]
    else:
        atei_d_model = atei_enc_layers = atei_nhead = 0
        atei_sd = {}

    model = whole_model(
        embd_size=d_model, nheads=nhead,
        atei_ckpt_sd=atei_sd,
        atei_d_model=atei_d_model, atei_nhead=atei_nhead,
        atei_enc_layers=atei_enc_layers,
        atei_dropout=atei_dropout, dropout=dropout,
        enc_layers=enc_layers, alpha_init=alpha_init,
        no_atei=no_atei, no_text=no_text).to(device)

    # key rename if needed
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("atei."):
            new_sd[k] = v
            continue
        new_k = k.replace("a_transformer_enc.", "a_transformer_enc.")
        new_sd[new_k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    real_missing    = [k for k in missing    if not k.startswith("atei.")]
    real_unexpected = [k for k in unexpected if not k.startswith("atei.")]
    if real_missing:
        raise RuntimeError(f"Missing keys: {real_missing}")
    if real_unexpected:
        raise RuntimeError(f"Unexpected keys: {real_unexpected}")
    print("[Load] model weights loaded successfully")
    model.eval()

    daic_depMap, _, dev_pids = get_Split_and_GroundTrue_tri()
    print(f"[Dev] {len(dev_pids)} patients")
    cnt = np.bincount([daic_depMap[p] for p in dev_pids], minlength=3)
    print(f"[Dev] label dist: Healthy={cnt[0]} Mild={cnt[1]} Moderate+={cnt[2]}")

    dev_idx = TestSegIndex(dev_pids, daic_depMap, feat_dir=ARGS.feat_dir)
    dev_ds  = TestSegDataset(dev_idx, feat_dir=ARGS.feat_dir,
                             cache_size=ARGS.cache_size)
    dev_loader = DataLoader(
        dev_ds, batch_size=ARGS.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=ARGS.num_workers,
        pin_memory=True, persistent_workers=(ARGS.num_workers > 0),
        prefetch_factor=(ARGS.prefetch_factor if ARGS.num_workers > 0 else None))

    r = test(model, dev_loader, device)

    labels_all   = [0, 1, 2]
    target_names = ["healthy(0)", "mild(1)", "moderate+(2)"]

    print("\n" + "=" * 60)
    print("TEST RESULT (official dev set, 3-class)")
    print("=" * 60)
    print(f"n_patients : {r['n_patients']}")
    print(f"Accuracy   : {r['pat_acc']:.4f}")
    print(f"MacroF1    : {r['pat_macro_f1']:.4f}")
    print(confusion_matrix(r["pat_true"], r["pat_pred"], labels=labels_all))
    print(classification_report(r["pat_true"], r["pat_pred"],
                                labels=labels_all,
                                target_names=target_names,
                                digits=4, zero_division=0))


if __name__ == "__main__":
    ARGS = parse_args()
    main()