"""112
for ckpt in weights/stage2_official_search/*.pt; do     echo "=== $ckpt ===";     uv run src/Inconsistency/models/Test_official.py --stage2_ckpt "$ckpt" 2>/dev/null | grep -E "(MacroF1|BinF1|Accuracy|n_patients)"; done
"""
"""
Test_official.py
================
Test on DAIC-WOZ official dev set using a trained Stage2 ckpt.
Supports both attn and hope_attention encoder types.

uv run src/Inconsistency/models/Test_official.py --stage2_ckpt weights/stage2_official_search/<ckpt>.pt
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
                             f1_score, precision_score, recall_score)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from Inconsistency.datasets.Incon_seg_bin import get_Split_and_GroundTrue
from Inconsistency.models.Stage1_seg_bin_daic import atei as Stage1ATEI
from Inconsistency.models.hope_adapter import HopeEncoderBlock
from Inconsistency.utils import set_seed

warnings.filterwarnings("ignore", category=FutureWarning)

N_CLASSES = 2
FEAT_DIR = "datasets/Feat_seg_bin_daic"
DAIC_DS_ROOT = "datasets/DAICWOZ"
DAIC_PSEUDO = "SegPseudoLabel_daic_distilbert_pair_bin.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2_ckpt", type=str, required=True)
    p.add_argument("--feat_dir", type=str, default=FEAT_DIR)
    p.add_argument("--daic_pseudo", type=str, default=DAIC_PSEUDO)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--prefetch_factor", type=int, default=2)
    p.add_argument("--cache_size", type=int, default=16)
    p.add_argument("--max_audio_frames", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ============================================================
# Model (mirrors Stage2_official_search)
# ============================================================
class whole_model(nn.Module):
    def __init__(self, embd_size, nheads, atei_ckpt_sd,
                 atei_d_model, atei_nhead, atei_enc_layers,
                 atei_dropout=0.3, dropout=0.3, enc_layers=1,
                 alpha_init=0.5, inp_dim=1024,
                 encoder_type="attn", cms_periods=(1, 4),
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
        else:
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
            self.atei = Stage1ATEI(embd_size=atei_d_model, nheads=atei_nhead,
                                   dropout=atei_dropout, enc_layers=atei_enc_layers)
            self.atei.load_state_dict(atei_ckpt_sd)
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
        else:
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            for layer in encoder:
                x = layer(x)
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            return x

    def forward(self, xa, xt, aMask=None, tMask=None):
        XA = self.a_in_proj(xa)
        HA = self._encode(XA, self.a_encoder, aMask)
        eA = self._attn_pool(HA, self.a_attn_pool, aMask)
        eA = self.a_post_norm(eA)

        if not self.no_text:
            XT = self.t_in_proj(xt)
            HT = self._encode(XT, self.t_encoder, tMask)
            eT = self._attn_pool(HT, self.t_attn_pool, tMask)
            eT = self.t_post_norm(eT)
        else:
            eT = None

        if not self.no_atei:
            eE_raw, _, _ = self.atei(xa, xt, aMask, tMask)
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
    def _attn_pool(x, attn_head, mask):
        scores = attn_head(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (x * weights).sum(dim=1)


# ============================================================
# Dataset
# ============================================================
class TestSegIndex:
    def __init__(self, dev_pids, daic_depMap, daic_npz,
                 feat_dir=FEAT_DIR, daic_ds_root=DAIC_DS_ROOT):
        self.samples = []
        feat_dir = Path(feat_dir)
        dpl = np.load(daic_npz)
        atei_map = {(int(p), int(s)): int(l)
                    for p, s, l in zip(dpl["patientIdx"],
                                       dpl["segIdx"], dpl["label"])}
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
                seg_id = row.Index + 2
                self.samples.append({
                    "patient_id": pid, "seg_id": seg_id,
                    "list_idx": list_idx,
                    "dep_label": daic_depMap[pid],
                })
        print(f"[TestSegIndex] total={len(self.samples)}")


class TestSegDataset(Dataset):
    def __init__(self, sample_index, feat_dir, cache_size=16):
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
# Test
# ============================================================
@torch.inference_mode()
def test(model, loader, device):
    model.eval()
    per_patient_scores = defaultdict(list)
    per_patient_true = {}

    for batch in tqdm(loader, desc="Test", unit="batch", leave=False):
        xa   = batch["xa"].to(device, non_blocking=True)
        xt   = batch["xt"].to(device, non_blocking=True)
        aMask = batch["aMask"].to(device, non_blocking=True)
        tMask = batch["tMask"].to(device, non_blocking=True)
        dep  = batch["dep"]
        pids = batch["patient_ids"]

        with torch.autocast(device_type="cuda", enabled=False):
            dep_logits = model(xa, xt, aMask, tMask)

        score = (dep_logits[:, 1] - dep_logits[:, 0]).float()
        for pid, s, t in zip(pids, score.cpu().tolist(), dep.tolist()):
            per_patient_scores[pid].append(s)
            per_patient_true.setdefault(pid, t)

    pids_list = list(per_patient_scores.keys())
    pat_means = np.array([np.mean(per_patient_scores[p]) for p in pids_list])
    pat_true  = np.array([per_patient_true[p] for p in pids_list])
    pat_pred  = (pat_means >= 0.0).astype(int)

    return {
        "pat_true": pat_true, "pat_pred": pat_pred,
        "pat_acc":      (pat_true == pat_pred).mean(),
        "pat_macro_f1": f1_score(pat_true, pat_pred, average="macro",
                                 labels=[0,1], zero_division=0),
        "pat_pre":      precision_score(pat_true, pat_pred, average="binary",
                                        pos_label=1, zero_division=0),
        "pat_rec":      recall_score(pat_true, pat_pred, average="binary",
                                     pos_label=1, zero_division=0),
        "n_patients": len(pids_list),
    }


# ============================================================
# Main
# ============================================================
def main():
    set_seed(ARGS.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(ARGS.stage2_ckpt, map_location="cpu", weights_only=False)
    hp   = ckpt.get("hparams") or ckpt.get("args", {})
    sd   = ckpt["model_state_dict"]

    print(f"[Ckpt] epoch={ckpt.get('epoch','?')} "
          f"fold={ckpt.get('fold','?')} "
          f"pat_macro_f1={ckpt.get('pat_macro_f1', ckpt.get('best_pat_macro_f1',-1)):.4f}")
    print(f"[Hparams] {hp}")

    d_model      = int(hp.get("d_model", 256))
    enc_layers   = int(hp.get("enc_layers", 1))
    dropout      = float(hp.get("dropout", 0.3))
    atei_dropout = float(hp.get("atei_dropout", 0.3))
    alpha_init   = float(hp.get("alpha_init", 0.5))
    no_atei      = bool(hp.get("no_atei", False))
    no_text      = bool(hp.get("no_text", False))
    encoder_type = str(hp.get("encoder_type", "attn"))
    cms_periods  = tuple(hp.get("cms_periods", (1, 4)))
    nhead        = 8 if d_model >= 256 else 4

    if not no_atei:
        atei_d_model    = sd["atei.a_in_proj.0.weight"].shape[0]
        atei_enc_layers = enc_layers
        atei_nhead      = nhead
        atei_sd = {k[len("atei."):]: v for k, v in sd.items()
                   if k.startswith("atei.")}
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
        encoder_type=encoder_type, cms_periods=cms_periods,
        no_atei=no_atei, no_text=no_text).to(device)
    model.load_state_dict(sd)
    model.eval()

    daic_depMap, _, dev_pids = get_Split_and_GroundTrue()
    print(f"[Dev] {len(dev_pids)} patients")

    dev_idx = TestSegIndex(dev_pids, daic_depMap, ARGS.daic_pseudo,
                           feat_dir=ARGS.feat_dir)
    dev_ds  = TestSegDataset(dev_idx, feat_dir=ARGS.feat_dir,
                             cache_size=ARGS.cache_size)
    dev_loader = DataLoader(
        dev_ds, batch_size=ARGS.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=ARGS.num_workers,
        pin_memory=True, persistent_workers=(ARGS.num_workers > 0),
        prefetch_factor=(ARGS.prefetch_factor if ARGS.num_workers > 0 else None))

    r = test(model, dev_loader, device)

    print("\n" + "="*60)
    print("TEST RESULT (official dev set)")
    print("="*60)
    print(f"n_patients : {r['n_patients']}")
    print(f"Accuracy   : {r['pat_acc']:.4f}")
    print(f"MacroF1    : {r['pat_macro_f1']:.4f}")
    print(f"Precision  : {r['pat_pre']:.4f}")
    print(f"Recall     : {r['pat_rec']:.4f}")
    print(confusion_matrix(r["pat_true"], r["pat_pred"], labels=[0, 1]))
    print(classification_report(r["pat_true"], r["pat_pred"],
                                labels=[0, 1],
                                target_names=["healthy(0)", "depressed(1)"],
                                digits=4, zero_division=0))


if __name__ == "__main__":
    ARGS = parse_args()
    main()