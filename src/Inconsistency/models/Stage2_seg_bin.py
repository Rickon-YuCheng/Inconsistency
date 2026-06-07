"""
Stage2_seg_bin.py — paper-aligned segment-level depression detection.

對齊 paper (Su et al. 2024) 的 Stage 3-4:
  - 訓練單位 = (audio segment, text segment) pair, 跟 Stage1 一致。
  - 每個 segment 的 depression label = 該 patient 的 PHQ8_Binary
    (用 patient label 蓋給該 patient 所有 segments)。
  - Model: audio transformer + text transformer + ATEI multi-head cross-attn
           + α scaling + fusion (concat) + dep head。
  - Loss (incremental training, paper eq.16):
        L_total = L_depression + λ · L_ATEI
    ATEI 從 Stage1_seg_bin ckpt 載入當 init, 跟整個網路一起 fine-tune。
  - Evaluation: segment-level prediction → 每個 patient majority vote
                → patient-level binary F1 (positive = depression = 1)。

跟 Stage2Main_bin.py 的差別
---------------------------
原 Stage2Main_bin.py (patient-level):
    一筆樣本 = 一個 patient (整堆 segment 一起 forward), batch_size 受
    記憶體限制只能 8~64 (其實是 patient 數)。ATEI 透過 xa_seg_list /
    xt_seg_list 二次拆封, 在 model 內部攤平 → ATEI forward → 再拆回 patient。

本檔 (segment-level, paper-aligned):
    一筆樣本 = 一個 (a_seg, t_seg) pair, batch_size 直接 64+。
    Model 不再做攤平/拆回, forward 簽名乾淨。
    Patient-level f1 透過 collate 帶 patient_id, 在 val 收集所有 segment
    prediction 後依 patient majority vote 得出。

ATEI ckpt
---------
從 Stage1Tr_seg_bin 訓出來的 ckpt 載入 self.atei 那一支。Stage1_seg_bin
的 atei 內嵌一份「無 chunk + 無 checkpoint」版本, forward(xa, xt, aMask, tMask)
吃 frame-level / token-level 序列。Stage2 model 內 import 同一份。
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
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             precision_score, recall_score)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

import wandb

from Inconsistency.datasets.Incon_seg_bin import (get_Split_and_GroundTrue,
                                                   get_stage1_kfold)
# 從 Stage1_seg_bin 借: ATEI model 主體 + 共用 cross-attn
from Inconsistency.models.Stage1_seg_bin import (atei as Stage1ATEI,
                                                    at_cross_attn,
                                                    _cross_attn)
from Inconsistency.utils import Timer, numpy_random_init, set_seed

# Optional: HopeEncoderBlock 不一定每個環境都有, lazy import 避免 ModuleNotFoundError
try:
    from hope_adapter import HopeEncoderBlock
    _HAS_HOPE = True
except Exception:
    HopeEncoderBlock = None
    _HAS_HOPE = False

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

DROPOUT = 0.3
ATEI_DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05

LAMBDA_ATEI = 0.1
ALPHA_INIT = 0.5
LAMBDA_AUX = 0.1   # paper 沒寫, 你原本用 0.1, 沿用
N_CLASSES = 2

ENCODER_TYPE = "attn"        # "attn" or "hope_attention"
CMS_PERIODS = (1, 4)
CMS_HIDDEN_MULTIPLIER = 4
CMS_ONLINE_UPDATES = False

MIN_SAVE_F1 = 0.40   # 低於此 patBinF1 不存 ckpt, 避免存爛模型

STAGE1_CKPT = "weights/stage1_seg_bin/stage1seg_bin_20260530_055320_seed42_fold1_best_macrof1_0.7394_ep009_lr1e-04_d128_l1.pt"   # 必填: 從 CLI 指定 Stage1_seg_bin 訓出的 ckpt
A_ROOT = "datasets/Feature/HuBERT_full_seg_bin"
T_ROOT = "datasets/Feature/RoBerTa_full_bin"
SEG_PSEUDO = "SegPseudoLabel_all_distilbert_pair_bin.npz"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stage1_ckpt", type=str, default=STAGE1_CKPT,
                        help="Stage1_seg_bin ckpt 路徑 (.pt). ATEI 分支 init 來源。")

    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--nhead", type=int, default=NHEAD)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--enc_layers", type=int, default=TRANSFORMER_ENC_LAYERS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)

    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--atei_dropout", type=float, default=ATEI_DROPOUT)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)

    parser.add_argument("--lambda_atei", type=float, default=LAMBDA_ATEI,
                        help="paper eq.16 的 λ, L_total = L_dep + λ·L_ATEI")
    parser.add_argument("--alpha_init", type=float, default=ALPHA_INIT,
                        help="paper eq.15 的 α 初始值, learnable scaling on e^E")

    parser.add_argument("--seg_pseudo", type=str, default=SEG_PSEUDO)
    parser.add_argument("--a_root", type=str, default=A_ROOT)
    parser.add_argument("--t_root", type=str, default=T_ROOT)

    parser.add_argument("--save_dir", type=str, default="weights/stage2_seg_bin")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--cache_size", type=int, default=16)

    parser.add_argument("--freeze_atei", action="store_true",
                        help="凍結 ATEI 分支 (預設不凍, 一起 fine-tune)")
    parser.add_argument("--atei_lr_scale", type=float, default=0.1,
                        help="ATEI lr = LR * scale. 0 = freeze.")
    parser.add_argument("--no_atei_loss", action="store_true",
                        help="關掉 ATEI 輔助 loss (λ=0)")

    parser.add_argument("--use_sampler", action="store_true",
                        help="WeightedRandomSampler 平衡 dep label "
                             "(seg-level, 等價於對少數類過採樣)")
    parser.add_argument("--use_class_weight", action="store_true",
                        help="CE loss 用 inverse-frequency weight")

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str,
                        default="Emotion inconsistency - Stage2 Seg bin")
    parser.add_argument("--wandb_name", type=str, default=None)

    parser.add_argument("--kfold", type=int, default=3)

    # ----- 從 Stage2Main_bin.py 補回來的小技巧 -----
    parser.add_argument("--lambda_aux", type=float, default=LAMBDA_AUX,
                        help="auxiliary depression loss (aux_a/aux_t/aux_e) 權重")
    parser.add_argument("--alpha_warmup", type=int, default=0,
                        help="epoch 數: alpha 從 0 線性升到目標 (0 = 立刻 1.0)")
    parser.add_argument("--lambda_warmup", type=int, default=0,
                        help="epoch 數: lambda_atei 從 0 線性升到目標 (0 = 立刻)")
    parser.add_argument("--accum_steps", type=int, default=1,
                        help="gradient accumulation: 等效 batch = batch_size * accum_steps")
    parser.add_argument("--print_norm", action="store_true",
                        help="forward 印 eA/eE/eT norm + alpha (debug)")
    parser.add_argument("--min_save_f1", type=float, default=MIN_SAVE_F1,
                        help="patBinF1 低於此值不存 ckpt")

    parser.add_argument("--encoder_type", type=str, default=ENCODER_TYPE,
                        choices=["attn", "hope_attention"],
                        help="depression-side encoder 種類")
    parser.add_argument("--cms_periods", type=int, nargs="+",
                        default=list(CMS_PERIODS),
                        help="hope_attention 用的 CMS periods")
    parser.add_argument("--cms_hidden_multiplier", type=int,
                        default=CMS_HIDDEN_MULTIPLIER)

    return parser.parse_args()


# ============================================================
# Model
# ============================================================
class whole_model(nn.Module):
    """
    paper-aligned Stage2 model.

    forward(xa, xt, aMask, tMask) ->
        (atei_logits[B,2], dep_logits[B,2])

    結構:
        xa -> a_in_proj -> a_transformer_enc -> H_a -> mean -> e^A
        xt -> t_in_proj -> t_transformer_enc -> H_t -> mean -> e^T

        ATEI 分支 (self.atei, 從 Stage1 ckpt 載入):
            atei.forward(xa, xt, aMask, tMask) -> (e^E[B,D_atei], atei_logits[B,2])

        e^E = α · atei_proj(e^E)    (paper eq.15, α learnable)

        fusion = concat(e^A, e^E, e^T)    (paper eq.14)
        FC1 -> FC2 -> FC3 -> dep_head -> dep_logits
    """

    def __init__(self, embd_size, nheads, atei_ckpt_path,
                 atei_dropout=0.3, dropout=0.3,
                 enc_layers=1, alpha_init=0.5, inp_dim=1024,
                 freeze_atei=False,
                 encoder_type="attn",
                 cms_periods=(1, 4), cms_hidden_multiplier=4,
                 cms_online_updates=False,
                 print_norm=False):
        super().__init__()
        self.encoder_type = encoder_type
        self.print_norm = print_norm

        # ---------- depression-side encoders ----------
        self.a_in_proj = nn.Sequential(
            nn.Linear(inp_dim, embd_size), nn.LayerNorm(embd_size),
        )
        self.t_in_proj = nn.Sequential(
            nn.Linear(inp_dim, embd_size), nn.LayerNorm(embd_size),
        )

        if encoder_type == "attn":
            a_enc_layer = nn.TransformerEncoderLayer(
                d_model=embd_size, nhead=nheads, batch_first=True,
                dim_feedforward=4 * embd_size, dropout=dropout, norm_first=True,
            )
            t_enc_layer = nn.TransformerEncoderLayer(
                d_model=embd_size, nhead=nheads, batch_first=True,
                dim_feedforward=4 * embd_size, dropout=dropout, norm_first=True,
            )
            self.a_transformer_enc = nn.TransformerEncoder(
                a_enc_layer, num_layers=enc_layers, enable_nested_tensor=False,
            )
            self.t_transformer_enc = nn.TransformerEncoder(
                t_enc_layer, num_layers=enc_layers, enable_nested_tensor=False,
            )
        elif encoder_type == "hope_attention":
            assert _HAS_HOPE, ("hope_adapter 沒裝, 用 --encoder_type attn "
                               "或安裝 hope_adapter 套件再跑")
            self.a_encoder = nn.ModuleList([
                HopeEncoderBlock(
                    dim=embd_size, heads=nheads, variant="hope_attention",
                    cms_periods=tuple(cms_periods),
                    hidden_multiplier=cms_hidden_multiplier,
                    cms_online_updates=cms_online_updates,
                ) for _ in range(enc_layers)
            ])
            self.t_encoder = nn.ModuleList([
                HopeEncoderBlock(
                    dim=embd_size, heads=nheads, variant="hope_attention",
                    cms_periods=tuple(cms_periods),
                    hidden_multiplier=cms_hidden_multiplier,
                    cms_online_updates=cms_online_updates,
                ) for _ in range(enc_layers)
            ])
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.a_post_norm = nn.LayerNorm(embd_size)
        self.t_post_norm = nn.LayerNorm(embd_size)

        # ---------- ATEI branch (from Stage1_seg_bin) ----------
        # 載 ckpt 推斷 ATEI 的 d_model, 不寫死
        ckpt = torch.load(atei_ckpt_path, map_location="cpu")
        sd = ckpt["model_state_dict"]
        atei_d_model = int(ckpt.get("d_model",
                                    sd["a_in_proj.0.weight"].shape[0]))
        atei_nhead = int(ckpt.get("nhead", nheads))
        atei_enc_layers = int(ckpt.get("enc_layers", 1))
        print(f"[ATEI init] d_model={atei_d_model}, nhead={atei_nhead}, "
              f"enc_layers={atei_enc_layers}")

        self.atei = Stage1ATEI(
            embd_size=atei_d_model, nheads=atei_nhead,
            dropout=atei_dropout,
            TRANSFORMER_ENC_LAYERS=atei_enc_layers,
        )
        self.atei.load_state_dict(sd)
        self.atei_d_model = atei_d_model

        if freeze_atei:
            for p in self.atei.parameters():
                p.requires_grad = False
            print("[ATEI] frozen")

        # ATEI feat -> depression d_model
        self.atei_proj = nn.Linear(atei_d_model, embd_size)

        # paper eq.15: learnable α on e^E
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        # ---------- fusion + head ----------
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(3 * embd_size, embd_size)
        self.fc2 = nn.Linear(embd_size, embd_size)
        self.fc3 = nn.Linear(embd_size, embd_size)
        self.dep_head = nn.Linear(embd_size, N_CLASSES)

        # ---------- auxiliary heads (deep supervision on eA/eT/eE) ----------
        # paper 沒提, 但實務上對 multimodal fusion 有幫助:
        # 每條 modality 單獨能 predict, 避免某條 modality 被 fusion 完全淹沒。
        self.aux_a_head = nn.Linear(embd_size, N_CLASSES)
        self.aux_t_head = nn.Linear(embd_size, N_CLASSES)
        self.aux_e_head = nn.Linear(embd_size, N_CLASSES)

    def forward(self, xa, xt, aMask=None, tMask=None, alpha_gate=1.0):
        """
        xa: [B, T_a, 1024], xt: [B, T_t, 1024]
        aMask/tMask: [B, T] (True = padding)
        alpha_gate: warmup 用, alpha 實際 = clamp(self.alpha) * alpha_gate
        Returns:
            atei_logits [B,2], dep_logits [B,2], (aux_a, aux_t, aux_e) each [B,2]
        """
        # ---- depression side ----
        XA = self.a_in_proj(xa)
        XT = self.t_in_proj(xt)

        if self.encoder_type == "attn":
            HA = self.a_transformer_enc(XA, src_key_padding_mask=aMask)
            HT = self.t_transformer_enc(XT, src_key_padding_mask=tMask)
        elif self.encoder_type == "hope_attention":
            # hope encoder 沒有 src_key_padding_mask, 自己 mask out padding
            if aMask is not None:
                XA = XA.masked_fill(aMask.unsqueeze(-1), 0.0)
            HA = XA
            for layer in self.a_encoder:
                HA = layer(HA)
            if aMask is not None:
                HA = HA.masked_fill(aMask.unsqueeze(-1), 0.0)
            if tMask is not None:
                XT = XT.masked_fill(tMask.unsqueeze(-1), 0.0)
            HT = XT
            for layer in self.t_encoder:
                HT = layer(HT)
            if tMask is not None:
                HT = HT.masked_fill(tMask.unsqueeze(-1), 0.0)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        eA = self.masked_mean(HA, aMask)   # [B, D]
        eT = self.masked_mean(HT, tMask)   # [B, D]

        # ---- ATEI side ----
        eE_raw, atei_logits = self.atei(xa, xt, aMask, tMask)
        eE = self.atei_proj(eE_raw)        # [B, D]
        # paper eq.15: e^E = α · FC(h^E), α 加 warmup gate
        alpha = torch.clamp(self.alpha, 0.0, 2.0) * alpha_gate
        eE = eE * alpha

        # ---- normalize 三條 scale ----
        eA = self.a_post_norm(eA)
        eT = self.t_post_norm(eT)

        # ---- auxiliary heads (eA/eT 已 post_norm, eE 已 α-scaled) ----
        aux_a = self.aux_a_head(eA)
        aux_t = self.aux_t_head(eT)
        aux_e = self.aux_e_head(eE)

        if self.training and self.print_norm:
            print(f"eA norm: {eA.norm(dim=-1).mean().item():.4f}, "
                  f"eE norm: {eE.norm(dim=-1).mean().item():.4f}, "
                  f"eT norm: {eT.norm(dim=-1).mean().item():.4f}, "
                  f"alpha: {float(self.alpha.detach()):.4f} "
                  f"(gate={alpha_gate:.3f})")

        # ---- fusion (paper eq.14: concat) ----
        eFusion = torch.cat((eA, eE, eT), dim=1)   # [B, 3D]

        h = self.dropout(F.relu(self.fc1(eFusion)))
        h = self.dropout(F.relu(self.fc2(h)))
        h = self.dropout(F.relu(self.fc3(h)))
        dep_logits = self.dep_head(h)               # [B, 2]

        return atei_logits, dep_logits, (aux_a, aux_t, aux_e)

    @staticmethod
    def masked_mean(x, mask):
        if mask is None:
            return x.mean(dim=1)
        valid = (~mask).unsqueeze(-1).float()
        s = (x * valid).sum(dim=1)
        Len = valid.sum(dim=1).clamp(min=1.0)
        return s / Len


# ============================================================
# Dataset / Collate
# ============================================================
class Stage2SegIndex:
    """
    Stage2 segment-level sample index。

    每個 sample = {
        patient_id, seg_id, list_idx,
        dep_label   : PHQ8_Binary (從 depMap),
        atei_label  : 0/1 from SegPseudoLabel (-1 if PAIR_RULE drop),
    }

    ATEI label 跟 dep label 各自獨立, ATEI loss 用 ignore_index=-1 跳過
    沒有 pair label 的 segment。
    """

    def __init__(self, patient_ids, depMap, pseudo_label_path,
                 ds_root="datasets/DAICWOZ"):
        pl = np.load(pseudo_label_path)
        seg_pid = pl["patientIdx"].astype(np.int64)
        seg_sid = pl["segIdx"].astype(np.int64)
        seg_lab = pl["label"].astype(np.int64)
        atei_map = {(int(p), int(s)): int(l)
                    for p, s, l in zip(seg_pid, seg_sid, seg_lab)}

        self.samples = []
        for pid in patient_ids:
            csv_path = Path(ds_root) / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv"
            if not csv_path.exists():
                print(f"[warn] csv missing: {csv_path}")
                continue
            df = pd.read_csv(csv_path, sep="\t")
            df_p = df[df["speaker"] == "Participant"].dropna(subset=["value"]).copy()

            for list_idx, row in enumerate(df_p.itertuples()):
                seg_id = row.Index + 2
                self.samples.append({
                    "patient_id": pid,
                    "seg_id": seg_id,
                    "list_idx": list_idx,
                    "dep_label": depMap[pid],
                    "atei_label": atei_map.get((pid, seg_id), -1),
                })

        print(f"[Stage2SegIndex] {len(self.samples)} segs from {len(patient_ids)} patients")

    def __len__(self):
        return len(self.samples)

    def get_dep_counts(self):
        labs = np.array([s["dep_label"] for s in self.samples])
        return np.bincount(labs, minlength=2)

    def get_atei_counts(self):
        labs = np.array([s["atei_label"] for s in self.samples])
        # -1 不算進來
        valid = labs[labs >= 0]
        return np.bincount(valid, minlength=2), int((labs == -1).sum())


class Stage2SegDataset(Dataset):
    """每個 patient 的 .pt 用 LRU cache (同 Stage1_seg_bin)。"""

    def __init__(self, sample_index, a_root=A_ROOT, t_root=T_ROOT, cache_size=16):
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

        xa = torch.load(str(self.a_root / f"{pid}_acoustic.pt"),
                        map_location="cpu", mmap=True)
        xt = torch.load(str(self.t_root / f"{pid}_text.pt"),
                        map_location="cpu", mmap=True)
        xa_list = [x.squeeze(0) for x in xa]
        xt_list = [x.squeeze(0) for x in xt]

        n = min(len(xa_list), len(xt_list))
        if len(xa_list) != len(xt_list):
            print(f"[warn] {pid}: a={len(xa_list)} t={len(xt_list)}, truncate {n}")
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
        list_idx = s["list_idx"]
        xa_list, xt_list = self._load_patient(pid)
        if list_idx >= len(xa_list) or list_idx >= len(xt_list):
            raise IndexError(f"{pid} seg {s['seg_id']} list_idx {list_idx} oob")
        return {
            "xa": xa_list[list_idx],
            "xt": xt_list[list_idx],
            "dep_label": s["dep_label"],
            "atei_label": s["atei_label"],
            "patient_id": pid,
            "seg_id": s["seg_id"],
        }


def collate_fn(batch):
    xa = pad_sequence([b["xa"] for b in batch], batch_first=True)
    xt = pad_sequence([b["xt"] for b in batch], batch_first=True)
    aMask = (xa.sum(dim=-1) == 0)
    tMask = (xt.sum(dim=-1) == 0)
    dep = torch.tensor([b["dep_label"] for b in batch], dtype=torch.long)
    atei = torch.tensor([b["atei_label"] for b in batch], dtype=torch.long)
    pids = [b["patient_id"] for b in batch]
    return {"xa": xa, "xt": xt, "aMask": aMask, "tMask": tMask,
            "dep": dep, "atei": atei, "patient_ids": pids}


# ============================================================
# Train / Val
# ============================================================
def train_one_epoch(model, loader, loss_dep, loss_atei, opt, scaler,
                    device, epoch, tot_epochs, cur_lambda, fold_id,
                    accum_steps=1, lambda_aux=0.1, alpha_gate=1.0):
    model.train()
    tot_dep = tot_atei = tot_aux = tot = 0.0
    correct_dep = n = 0
    seg_true, seg_pred = [], []

    pbar = tqdm(loader, desc=f"Fold{fold_id} Train ep {epoch}/{tot_epochs}",
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
            atei_logits, dep_logits, aux_logits = model(
                xa, xt, aMask, tMask, alpha_gate=alpha_gate,
            )
            aux_a, aux_t, aux_e = aux_logits

            L_dep = loss_dep(dep_logits, dep)
            if (atei_lab != -1).any():
                L_atei = loss_atei(atei_logits, atei_lab)
            else:
                L_atei = torch.tensor(0.0, device=device)
            L_aux = (loss_dep(aux_a, dep)
                     + loss_dep(aux_t, dep)
                     + loss_dep(aux_e, dep)) / 3.0
            L_total = L_dep + cur_lambda * L_atei + lambda_aux * L_aux

        # gradient accumulation
        scaler.scale(L_total / accum_steps).backward()
        if (step + 1) % accum_steps == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

        pred = dep_logits.argmax(dim=-1)
        correct_dep += (pred == dep).sum().item()
        tot_dep += L_dep.item() * dep.size(0)
        tot_atei += L_atei.item() * dep.size(0)
        tot_aux += L_aux.item() * dep.size(0)
        tot += L_total.item() * dep.size(0)
        n += dep.size(0)

        seg_true.extend(dep.cpu().tolist())
        seg_pred.extend(pred.cpu().tolist())

        pbar.set_postfix({
            "dep": tot_dep / max(n, 1),
            "atei": tot_atei / max(n, 1),
            "aux": tot_aux / max(n, 1),
            "acc": correct_dep / max(n, 1),
        })

    # tail step (處理最後一個不滿 accum_steps 的尾巴)
    if (step + 1) % accum_steps != 0:
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()

    return {
        "dep_loss": tot_dep / max(n, 1),
        "atei_loss": tot_atei / max(n, 1),
        "aux_loss": tot_aux / max(n, 1),
        "tot_loss": tot / max(n, 1),
        "dep_acc": correct_dep / max(n, 1),
        "seg_dist_true": Counter(seg_true),
        "seg_dist_pred": Counter(seg_pred),
    }


@torch.inference_mode()
def validate(model, loader, loss_dep, device, fold_id):
    """
    paper: segment-level prediction -> 每 patient majority vote -> patient F1.
    順手也算 segment-level f1 給比較。
    """
    model.eval()
    seg_true, seg_pred = [], []
    per_patient_scores = defaultdict(list)
    per_patient_true = {}

    tot_loss = n = 0
    pbar = tqdm(loader, desc=f"Fold{fold_id} Val", unit="batch", leave=False)
    for batch in pbar:
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
        # class 1 logit - class 0 logit
        # > 0 等價於 class 1 prob > 0.5
        score = (dep_logits[:, 1] - dep_logits[:, 0]).float()
        tot_loss += loss.item() * dep.size(0)
        n += dep.size(0)

        seg_true.extend(dep.cpu().tolist())
        seg_pred.extend(pred.cpu().tolist())

        for pid, s, t in zip(pids, score.cpu().tolist(), dep.cpu().tolist()):
            per_patient_scores[pid].append(s)
            per_patient_true.setdefault(pid, t)

    # patient-level majority vote
    pat_true, pat_pred = [], []
    TH = 0.0  # 先用 0；之後可以 sweep

    for pid, scores in per_patient_scores.items():
        patient_score = np.mean(scores)   # 先用 mean logit
        pat_pred.append(int(patient_score >= TH))
        pat_true.append(per_patient_true[pid])

    seg_true = np.array(seg_true); seg_pred = np.array(seg_pred)
    pat_true = np.array(pat_true); pat_pred = np.array(pat_pred)

    out = {
        "loss": tot_loss / max(n, 1),
        "seg_acc": (seg_true == seg_pred).mean(),
        "seg_bin_f1": f1_score(seg_true, seg_pred, average="binary",
                               pos_label=1, zero_division=0),
        "seg_macro_f1": f1_score(seg_true, seg_pred, average="macro",
                                 labels=[0, 1], zero_division=0),
        "pat_acc": (pat_true == pat_pred).mean(),
        "pat_bin_f1": f1_score(pat_true, pat_pred, average="binary",
                               pos_label=1, zero_division=0),
        "pat_macro_f1": f1_score(pat_true, pat_pred, average="macro",
                                 labels=[0, 1], zero_division=0),
        "pat_pre": precision_score(pat_true, pat_pred, average="binary",
                                   pos_label=1, zero_division=0),
        "pat_rec": recall_score(pat_true, pat_pred, average="binary",
                                pos_label=1, zero_division=0),
        "pat_true": pat_true, "pat_pred": pat_pred,
        "seg_true": seg_true, "seg_pred": seg_pred,
        "n_patients": len(per_patient_scores),
    }
    return out


# ============================================================
# Class balance helpers
# ============================================================
def build_dep_sampler(index, seed):
    labs = np.array([s["dep_label"] for s in index.samples])
    cnt = np.bincount(labs, minlength=2)
    print(f"[sampler] dep counts: {cnt}")
    if cnt[0] == 0 or cnt[1] == 0:
        raise ValueError(f"only one dep class: {cnt}")
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
# Per-fold runner
# ============================================================
def run_one_fold(fold_id, train_ids, val_ids, depMap, run_id, device):
    set_seed(ARGS.seed + fold_id)
    save_dir = Path(ARGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    run_name = (ARGS.wandb_name + f"_fold{fold_id}") if ARGS.wandb_name else (
        f"stage2seg_bin_seed{ARGS.seed}_lr{ARGS.lr:.0e}_"
        f"la{ARGS.lambda_atei:.2f}_a{ARGS.alpha_init:.2f}_"
        f"d{ARGS.d_model}_l{ARGS.enc_layers}_fold{fold_id}_{run_id}"
    )

    # --- indices / datasets ---
    print(f"\n{'='*60}\nFOLD {fold_id}\n{'='*60}")
    train_index = Stage2SegIndex(train_ids, depMap, ARGS.seg_pseudo)
    val_index = Stage2SegIndex(val_ids, depMap, ARGS.seg_pseudo)
    print(f"[train] dep dist: {train_index.get_dep_counts()}")
    print(f"[val]   dep dist: {val_index.get_dep_counts()}")
    a_cnt, a_drop = train_index.get_atei_counts()
    print(f"[train] atei dist (excl -1): {a_cnt}, dropped (-1): {a_drop}")

    train_ds = Stage2SegDataset(train_index, ARGS.a_root, ARGS.t_root,
                                cache_size=ARGS.cache_size)
    val_ds = Stage2SegDataset(val_index, ARGS.a_root, ARGS.t_root,
                              cache_size=ARGS.cache_size)

    g = torch.Generator(); g.manual_seed(ARGS.seed + fold_id)

    sampler = None
    if ARGS.use_sampler:
        sampler = build_dep_sampler(train_index, seed=ARGS.seed + fold_id)

    train_loader = DataLoader(
        train_ds, batch_size=ARGS.batch_size,
        sampler=sampler, shuffle=(sampler is None),
        collate_fn=collate_fn, num_workers=ARGS.num_workers,
        pin_memory=True, worker_init_fn=numpy_random_init, generator=g,
        persistent_workers=(ARGS.num_workers > 0),
        prefetch_factor=(ARGS.prefetch_factor if ARGS.num_workers > 0 else None),
    )
    val_loader = DataLoader(
        val_ds, batch_size=ARGS.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=ARGS.num_workers,
        pin_memory=True,
        persistent_workers=(ARGS.num_workers > 0),
        prefetch_factor=(ARGS.prefetch_factor if ARGS.num_workers > 0 else None),
    )

    # --- model ---
    assert ARGS.stage1_ckpt, "--stage1_ckpt 必填, 指向 Stage1_seg_bin ckpt"
    model = whole_model(
        embd_size=ARGS.d_model, nheads=ARGS.nhead,
        atei_ckpt_path=ARGS.stage1_ckpt,
        atei_dropout=ARGS.atei_dropout, dropout=ARGS.dropout,
        enc_layers=ARGS.enc_layers, alpha_init=ARGS.alpha_init,
        freeze_atei=ARGS.freeze_atei,
    ).to(device)

    # --- optimizer: ATEI 用較小 lr (paper incremental training 精神) ---
    atei_params = list(model.atei.parameters())
    other_params = [p for n, p in model.named_parameters()
                    if not n.startswith("atei.")]
    opt = torch.optim.Adam(
        [
            {"params": atei_params,
             "lr": ARGS.lr * ARGS.atei_lr_scale,
             "weight_decay": ARGS.weight_decay},
            {"params": other_params,
             "lr": ARGS.lr,
             "weight_decay": ARGS.weight_decay},
        ],
    )
    print(f"[opt] ATEI lr={ARGS.lr * ARGS.atei_lr_scale:.2e}, "
          f"other lr={ARGS.lr:.2e}")

    # --- losses ---
    class_w = build_class_weight(train_index, device) if ARGS.use_class_weight else None
    loss_dep = nn.CrossEntropyLoss(weight=class_w,
                                   label_smoothing=ARGS.label_smoothing)
    loss_atei = nn.CrossEntropyLoss(ignore_index=-1,
                                    label_smoothing=ARGS.label_smoothing)
    scaler = torch.GradScaler("cuda")

    cur_lambda = 0.0 if ARGS.no_atei_loss else ARGS.lambda_atei

    # --- wandb ---
    if ARGS.use_wandb:
        wandb.init(
            project=ARGS.wandb_project, name=run_name, reinit=True,
            config={
                "seed": ARGS.seed, "fold": fold_id,
                "d_model": ARGS.d_model, "nhead": ARGS.nhead,
                "lr": ARGS.lr, "epochs": ARGS.epochs,
                "enc_layers": ARGS.enc_layers, "batch_size": ARGS.batch_size,
                "dropout": ARGS.dropout, "atei_dropout": ARGS.atei_dropout,
                "weight_decay": ARGS.weight_decay,
                "label_smoothing": ARGS.label_smoothing,
                "lambda_atei": cur_lambda, "alpha_init": ARGS.alpha_init,
                "freeze_atei": ARGS.freeze_atei,
                "atei_lr_scale": ARGS.atei_lr_scale,
                "use_sampler": ARGS.use_sampler,
                "use_class_weight": ARGS.use_class_weight,
                "stage1_ckpt": ARGS.stage1_ckpt,
                "a_root": ARGS.a_root, "t_root": ARGS.t_root,
                "seg_pseudo": ARGS.seg_pseudo,
                "n_classes": N_CLASSES,
            },
        )

    # --- training loop ---
    best_pat_bin = -1.0
    best_pat_macro = -1.0
    no_improve = 0

    for epoch in range(1, ARGS.epochs + 1):
        print("=" * 80)
        print(f"[Fold {fold_id}] Epoch [{epoch}/{ARGS.epochs}]  "
              f"α={float(model.alpha.detach()):.4f} λ={cur_lambda:.4f}")

        tr = train_one_epoch(
            model, train_loader, loss_dep, loss_atei,
            opt, scaler, device, epoch, ARGS.epochs,
            cur_lambda, fold_id,
            accum_steps=ARGS.accum_steps,
            lambda_aux=ARGS.lambda_aux,
        )
        v = validate(model, val_loader, loss_dep, device, fold_id)

        print(f"[Train] dep_loss={tr['dep_loss']:.4f} "
              f"atei_loss={tr['atei_loss']:.4f} "
              f"tot={tr['tot_loss']:.4f} dep_acc={tr['dep_acc']:.4f}")
        print(f"[Train] seg true: {dict(tr['seg_dist_true'])} "
              f"pred: {dict(tr['seg_dist_pred'])}")
        print(f"[Val ] seg  acc={v['seg_acc']:.4f}  "
              f"binF1={v['seg_bin_f1']:.4f}  macroF1={v['seg_macro_f1']:.4f}")
        print(f"[Val ] patient(n={v['n_patients']})  "
              f"acc={v['pat_acc']:.4f}  binF1={v['pat_bin_f1']:.4f}  "
              f"macroF1={v['pat_macro_f1']:.4f}  "
              f"pre={v['pat_pre']:.4f}  rec={v['pat_rec']:.4f}")
        print("[Val ] patient confusion matrix:")
        print(confusion_matrix(v["pat_true"], v["pat_pred"], labels=[0, 1]))
        print(classification_report(v["pat_true"], v["pat_pred"],
                                    labels=[0, 1],
                                    target_names=["healthy(0)", "depressed(1)"],
                                    digits=4, zero_division=0))

        # 雙 best ckpt (主指標 = patient binary f1)
        saved_any = False
        if v["pat_bin_f1"] > best_pat_bin:
            best_pat_bin = v["pat_bin_f1"]
            no_improve = 0
            ckpt = (f"stage2seg_bin_{run_id}_seed{ARGS.seed}_fold{fold_id}_"
                    f"best_patBinF1_{best_pat_bin:.4f}_ep{epoch:03d}_"
                    f"lr{ARGS.lr:.0e}_d{ARGS.d_model}_l{ARGS.enc_layers}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "fold": fold_id,
                "best_pat_bin_f1": best_pat_bin,
                "best_pat_macro_f1": best_pat_macro,
                "val": {k: v[k] for k in
                        ["seg_acc", "seg_bin_f1", "seg_macro_f1",
                         "pat_acc", "pat_bin_f1", "pat_macro_f1",
                         "pat_pre", "pat_rec"]},
                "args": vars(ARGS),
                "selected_by": "pat_bin_f1",
            }, save_dir / ckpt)
            print(f"[Save best-patBinF1] {best_pat_bin:.4f} -> {ckpt}")
            saved_any = True
        else:
            no_improve += 1

        if v["pat_macro_f1"] > best_pat_macro:
            best_pat_macro = v["pat_macro_f1"]
            ckpt = (f"stage2seg_bin_{run_id}_seed{ARGS.seed}_fold{fold_id}_"
                    f"best_patMacroF1_{best_pat_macro:.4f}_ep{epoch:03d}_"
                    f"lr{ARGS.lr:.0e}_d{ARGS.d_model}_l{ARGS.enc_layers}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch, "fold": fold_id,
                "best_pat_bin_f1": best_pat_bin,
                "best_pat_macro_f1": best_pat_macro,
                "val": {k: v[k] for k in
                        ["seg_acc", "seg_bin_f1", "seg_macro_f1",
                         "pat_acc", "pat_bin_f1", "pat_macro_f1",
                         "pat_pre", "pat_rec"]},
                "args": vars(ARGS),
                "selected_by": "pat_macro_f1",
            }, save_dir / ckpt)
            print(f"[Save best-patMacroF1] {best_pat_macro:.4f} -> {ckpt}")
            saved_any = True

        if not saved_any:
            print(f"[EarlyStop] no improvement (patBinF1) "
                  f"{no_improve}/{ARGS.patience}")

        if ARGS.use_wandb:
            wandb.log({
                "epoch": epoch,
                "alpha": float(model.alpha.detach()),
                "train/dep_loss": tr["dep_loss"],
                "train/atei_loss": tr["atei_loss"],
                "train/tot_loss": tr["tot_loss"],
                "train/dep_acc": tr["dep_acc"],
                "val/loss": v["loss"],
                "val/seg_bin_f1": v["seg_bin_f1"],
                "val/seg_macro_f1": v["seg_macro_f1"],
                "val/pat_bin_f1": v["pat_bin_f1"],
                "val/pat_macro_f1": v["pat_macro_f1"],
                "val/pat_acc": v["pat_acc"],
                "val/pat_pre": v["pat_pre"],
                "val/pat_rec": v["pat_rec"],
                "best/pat_bin_f1": best_pat_bin,
                "best/pat_macro_f1": best_pat_macro,
                "no_improve": no_improve,
                "lr/atei": opt.param_groups[0]["lr"],
                "lr/other": opt.param_groups[1]["lr"],
            })

        if no_improve >= ARGS.patience:
            print(f"[EarlyStop] Fold {fold_id} stop at ep {epoch}, "
                  f"best patBinF1={best_pat_bin:.4f}, "
                  f"patMacroF1={best_pat_macro:.4f}")
            break

    if ARGS.use_wandb:
        wandb.finish()

    return {"pat_bin_f1": best_pat_bin, "pat_macro_f1": best_pat_macro}


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
        r = run_one_fold(f["fold"], f["train"], f["val"], depMap, run_id, device)
        fold_results.append(r)
        print(f"\n>>> Fold {f['fold']} best patBinF1={r['pat_bin_f1']:.4f} "
              f"patMacroF1={r['pat_macro_f1']:.4f}")

    print("\n" + "=" * 60)
    print("K-FOLD RESULT (Stage2 seg_bin)")
    print("=" * 60)
    for i, r in enumerate(fold_results):
        print(f"Fold {i}: patBinF1={r['pat_bin_f1']:.4f}  "
              f"patMacroF1={r['pat_macro_f1']:.4f}")

    bin_arr = np.array([r["pat_bin_f1"] for r in fold_results])
    macro_arr = np.array([r["pat_macro_f1"] for r in fold_results])
    print(f"\nMean patBinF1   : {bin_arr.mean():.4f} ± {bin_arr.std():.4f}")
    print(f"Mean patMacroF1 : {macro_arr.mean():.4f} ± {macro_arr.std():.4f}")
    print(f"\nTotal time: {timer}")


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ARGS = parse_args()
    main()