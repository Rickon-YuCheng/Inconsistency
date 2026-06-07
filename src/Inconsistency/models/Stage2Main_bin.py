"""
f10.60
uv run src/Inconsistency/models/Stage2Main_bin.py --epochs 300 --enc_layers 1 --d_model 256 --dropout 0.3 --atei_dropout 0.3 --weight_decay 0 --lr 1e-3 --lambda_atei 0.1 --alpha_init 0.5 --batch_size
 8 --patience 150

f10.66 ep113
uv run src/Inconsistency/models/Stage2Main_bin.py --epochs 300 --enc_layers 1 --d_model 256 --dropout 0.3 --atei_dropout 0.3 --weight_decay 0 --lr 5e-4 --lambda_atei 0.1 --alpha_init 0.5 --batch_size 8 --accum_steps 8 --patience 150

tot batch64 比 32 好
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
from hope_adapter import HopeEncoderBlock
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from collections import Counter
import torch
from sklearn.model_selection import StratifiedKFold
from Stage1Tr_bin import atei   # ← 注意:從 _quick_bin 載入
from torch.nn.utils.rnn import pad_sequence
from datetime import datetime
import argparse
from Inconsistency.utils import Timer, set_seed, numpy_random_init
import torch.nn as nn
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from Inconsistency.datasets.inconsistentLabel_bin import get_Split_and_GroundTrue
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

# depression 二元分類 (bin): n_classes = 2
N_CLASSES = 2

# 改成你新訓出來的 Stage1-quick-bin ckpt 路徑
STAGE1_CKPT = "weights/stage1-quick-bin/stage1-quick-bin_20260528_062718_seed42_fold1_f10.6971_ep047_lr1e-04_wd1e-04_d256_l1.pt"  # ← 改這行
# STAGE1_CKPT = "weights/stage1-quick-bin/stage1-quick-bin_20260521_083251_seed42_f10.6758_ep023_lr1e-04_wd1e-04_d256_l1.pt" # av
D_MODEL = 256
NHEAD = 8
LR = 5e-4
EPOCHS = 3000
TRANSFORMER_ENC_LAYERS = 1
DROPOUT = 0.3
ATEI_DROPOUT = 0.3
WEIGHT_DECAY = 0
LAMBDA_ATEI = 0.1
ALPHA_INIT = 0.5
PATIENCE = 500
ACCUM_STEPS = 1   # 等效 batch = batch_size * ACCUM_STEPS
ENCODER_TYPE = "attn"          # "attn" or "hope_attention"
CMS_PERIODS = (1, 4)
CMS_HIDDEN_MULTIPLIER = 4
CMS_ONLINE_UPDATES = False

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

    parser.add_argument("--lambda_atei", type=float, default=LAMBDA_ATEI)
    parser.add_argument("--alpha_init", type=float, default=ALPHA_INIT)
    parser.add_argument("--patience", type=int, default=PATIENCE)

    parser.add_argument("--save_dir", type=str, default="weights/stage2_quick_bin")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Emotion inconsistency _ Stage2 quick bin")
    parser.add_argument("--wandb_name", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--kfold", type=int, default=0)

    parser.add_argument("--freeze_atei", action="store_true",
                        help="Freeze ATEI parameters during Stage2 training.")
    parser.add_argument("--no_atei_loss", action="store_true",
                        help="Disable ATEI auxiliary loss (set lambda_atei to 0).")
    parser.add_argument("--atei_lr_scale", type=float, default=0.1,
                        help="ATEI lr = LR * scale. 0 = freeze, 1 = same as other.")
    parser.add_argument("--atei_wd", type=float, default=None,
                        help="ATEI weight_decay (None = same as --weight_decay)")
    parser.add_argument("--print_norm", action="store_true",
                        help="Print eA/eE/eT norm in forward (debug)")
    
    parser.add_argument("--accum_steps", type=int, default=1,
                        help="Gradient accumulation steps. effective batch = batch_size * accum_steps")

    parser.add_argument("--alpha_warmup", type=int, default=0,
                    help="Epochs to warmup alpha from 0 to target")
    parser.add_argument("--lambda_warmup", type=int, default=0,
                    help="Epochs to warmup lambda_atei from 0 to target")
    parser.add_argument("--encoder_type", type=str, default=ENCODER_TYPE,
                        choices=["attn", "hope_attention"])
    parser.add_argument("--cms_periods", type=int, nargs="+", default=list(CMS_PERIODS))
    parser.add_argument("--cms_hidden_multiplier", type=int, default=CMS_HIDDEN_MULTIPLIER)

    return parser.parse_args()


def build_kfold_splits(n_splits=5, seed=42):
    depMap, train_Idx, test_Idx = get_Split_and_GroundTrue()
    patient_ids = train_Idx + test_Idx
    labels = [depMap[p] for p in patient_ids]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(patient_ids, labels)):
        splits.append({
            "fold": fold_idx,
            "train": [patient_ids[i] for i in tr_idx],
            "val": [patient_ids[i] for i in val_idx],
        })
    return splits


# ============================================================
# Model
# ============================================================
class whole_model(nn.Module):
    def __init__(self, embd_size=D_MODEL, nheads=NHEAD):
        super().__init__()
        self.a_in_proj = nn.Sequential(
            nn.Linear(1024, embd_size),
            nn.LayerNorm(embd_size),
        )
        self.t_in_proj = nn.Sequential(
            nn.Linear(1024, embd_size),
            nn.LayerNorm(embd_size),
        )

        # === Stage1 ATEI (segment-level version, embd_size=256) ===
        # 注意:Stage1Tr_quick_bin 訓的時候是 d_model=256
        self.atei = atei(
            embd_size=256, nheads=nheads,
            dropout=ARGS.atei_dropout,
            enc_layers=1,   # ← Stage1_quick_bin 是 enc_layers 不是 TRANSFORMER_ENC_LAYERS
        )
        ckpt = torch.load(ARGS.stage1_ckpt, map_location="cpu")
        self.atei.load_state_dict(ckpt["model_state_dict"])

        if ARGS.freeze_atei:
            for p in self.atei.parameters():
                p.requires_grad = False
            print("[ATEI] frozen")

        # Stage3 主體 (depression 自己的 Transformer encoder)
        self.encoder_type = ARGS.encoder_type
        if self.encoder_type == "attn":
            a_enc_layer = nn.TransformerEncoderLayer(
                d_model=embd_size, dropout=ARGS.dropout,
                dim_feedforward=4 * embd_size,
                nhead=nheads, batch_first=True, norm_first=True,
            )
            t_enc_layer = nn.TransformerEncoderLayer(
                d_model=embd_size, dropout=ARGS.dropout,
                dim_feedforward=4 * embd_size,
                nhead=nheads, batch_first=True, norm_first=True,
            )
            self.a_transformer_enc = nn.TransformerEncoder(
                a_enc_layer, num_layers=TRANSFORMER_ENC_LAYERS,
                enable_nested_tensor=False,
            )
            self.t_transformer_enc = nn.TransformerEncoder(
                t_enc_layer, num_layers=TRANSFORMER_ENC_LAYERS,
                enable_nested_tensor=False,
            )
        elif self.encoder_type == "hope_attention":
            self.a_encoder = nn.ModuleList([
                HopeEncoderBlock(
                    dim=embd_size, heads=nheads, variant="hope_attention",
                    cms_periods=tuple(ARGS.cms_periods),
                    hidden_multiplier=ARGS.cms_hidden_multiplier,
                    cms_online_updates=CMS_ONLINE_UPDATES,
                ) for _ in range(TRANSFORMER_ENC_LAYERS)
            ])
            self.t_encoder = nn.ModuleList([
                HopeEncoderBlock(
                    dim=embd_size, heads=nheads, variant="hope_attention",
                    cms_periods=tuple(ARGS.cms_periods),
                    hidden_multiplier=ARGS.cms_hidden_multiplier,
                    cms_online_updates=CMS_ONLINE_UPDATES,
                ) for _ in range(TRANSFORMER_ENC_LAYERS)
            ])
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        self.atei_proj = nn.Linear(256, embd_size)  # ATEI feat 256 -> embd_size

        self.a_post_norm = nn.LayerNorm(embd_size)
        self.t_post_norm = nn.LayerNorm(embd_size)

        # self.mod_drop_p = 0.15

        # self.a_attn_pool = nn.Linear(embd_size, 1)
        # self.t_attn_pool = nn.Linear(embd_size, 1)

        # bottleneck_dim=96
        # self.fc_bottleneck=nn.Linear(3* embd_size, bottleneck_dim)
        # self.fc1 = nn.Linear(bottleneck_dim, embd_size)   # bottleneck -> embd_size
        self.dropout = nn.Dropout(ARGS.dropout)
        self.fc1 = nn.Linear(3 * embd_size, embd_size)
        # self.fc1 = nn.Linear(embd_size, embd_size)
        self.fc2 = nn.Linear(embd_size, embd_size)
        self.fc3 = nn.Linear(embd_size, embd_size)
        self.alpha = nn.Parameter(torch.tensor(ALPHA_INIT))
        self.oup = nn.Linear(embd_size, N_CLASSES)

        # self.fuse_norm_a = nn.LayerNorm(embd_size)
        # self.fuse_norm_e = nn.LayerNorm(embd_size)
        # self.fuse_norm_t = nn.LayerNorm(embd_size)

        # self.gate = nn.Linear(3 * embd_size, 3)   # 輸入三條 concat,輸出三個權重
        # self.fc1 = nn.Linear(embd_size, embd_size)   # 原本是 3*embd_size,改成 embd_size

        self.aux_a_head = nn.Linear(embd_size, N_CLASSES)
        self.aux_t_head = nn.Linear(embd_size, N_CLASSES)
        self.aux_e_head = nn.Linear(embd_size, N_CLASSES)
        self.lambda_aux = 0.3   # auxiliary loss 權重,先設 0.3

    def forward(self, XA, XT, aMask=None, tMask=None,
                xa_seg_list=None, xt_seg_list=None,alpha_gate=1.0,  return_feature=False):
        """
        XA: [B, num_seg, 1024]   ← segment-level (已 pool)
        XT: [B, num_seg, 1024]   ← segment-level (token mean)
        xa_seg_list: list of [num_seg_i, 1024]      ← segment-level audio for ATEI
        xt_seg_list: list of [num_seg_i, L_i, 1024] ← token-level text for ATEI
        """
        # ---------- Stage3: depression feature ----------
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
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")

        eA = self.masked_mean(HA, aMask)   # ← 保留你現在的 masked_mean
        eT = self.masked_mean(HT, tMask)

        # === ATEI batched forward ===
        seg_counts = [xa_seg.size(0) for xa_seg in xa_seg_list]
        max_L_t = max(xt_seg.size(1) for xt_seg in xt_seg_list)
        D_in = xa_seg_list[0].size(-1)

        device_local = xa_seg_list[0].device
        total_segs = sum(seg_counts)
        batch_a = torch.zeros(total_segs, D_in,
                              device=device_local, dtype=xa_seg_list[0].dtype)
        batch_t = torch.zeros(total_segs, max_L_t, D_in,
                              device=device_local, dtype=xt_seg_list[0].dtype)
        start = 0
        for xa_seg, xt_seg in zip(xa_seg_list, xt_seg_list):
            n = xa_seg.size(0)
            batch_a[start:start+n] = xa_seg                     # [n, 1024]
            batch_t[start:start+n, :xt_seg.size(1)] = xt_seg    # [n, L_i, 1024]
            start += n

        # text padding mask
        mask_t = (batch_t.sum(dim=-1) == 0)

        # ATEI forward (segment-level version)
        eE_all, logits_all = self.atei(batch_a, batch_t, mask_t)
        # eE_all: [total_segs, 256], logits_all: [total_segs, 2]

        # 拆回每個 patient,取 mean
        eE_list = []
        atei_logits_list = []
        start = 0
        for count in seg_counts:
            eE_list.append(eE_all[start:start+count].mean(dim=0))
            atei_logits_list.append(logits_all[start:start+count].mean(dim=0))
            start += count

        eE = torch.stack(eE_list, dim=0)        # [B, 256]
        eE = self.atei_proj(eE)                  # [B, embd_size]
        atei_logits = torch.stack(atei_logits_list, dim=0)  # [B, 2]

        # ---------- Scaling ----------
        alpha = torch.clamp(self.alpha, 0.0, 2.0)* alpha_gate
        eE = eE * alpha

        # === 新增:fusion 前各自 LayerNorm,強制 scale 一致 ===
        eA = self.a_post_norm(eA)
        eT = self.t_post_norm(eT)

        # # === modality dropout (train only, 保證至少留一條) ===
        # if self.training:
        #     drop_a = torch.rand(1).item() < self.mod_drop_p
        #     drop_t = torch.rand(1).item() < self.mod_drop_p
        #     if drop_a and drop_t:          # 不能兩條都丟(eE 不參與,留它沒問題)
        #         drop_a = drop_t = False
        #     if drop_a:
        #         eA = torch.zeros_like(eA)
        #     if drop_t:
        #         eT = torch.zeros_like(eT)

        # eA, eT 已經過 a_post_norm/t_post_norm;eE 已經過 alpha
        aux_a = self.aux_a_head(eA)
        aux_t = self.aux_t_head(eT)
        aux_e = self.aux_e_head(eE)

        if self.training and ARGS.print_norm:
            print(f"eA norm: {eA.norm(dim=-1).mean().item():.4f}, "
                f"eE norm: {eE.norm(dim=-1).mean().item():.4f}, "
                f"eT norm: {eT.norm(dim=-1).mean().item():.4f}, "
                f"alpha: {self.alpha.item():.4f}")
            
        # ---------- Fusion ----------
        eFusion = torch.cat((eA, eE, eT), dim=1)  # [B, 3D]
        Fc1 = self.dropout(F.relu(self.fc1(eFusion)))
        # ---------- Add Fusion ----------
        # eFusion = eA + eE + eT                          # [B, D] 逐元素相加
        # Fc1 = self.dropout(F.relu(self.fc1(eFusion)))
        # ---------- Mult Fusion (norm 後再相乘,避免尺度/符號亂跳) ----------
        # na = self.fuse_norm_a(eA)
        # ne = self.fuse_norm_e(eE)      # 這裡的 eE 已經過 alpha,再 norm 把 scale 拉回
        # nt = self.fuse_norm_t(eT)
        # eFusion = na * ne * nt
        # Fc1 = self.dropout(F.relu(self.fc1(eFusion)))

        # ---------- Bottleneck Fusion ----------
        # eFusion = torch.cat((eA, eE, eT), dim=1)              # [B, 3D]
        # bottleneck = self.dropout(F.relu(self.fc_bottleneck(eFusion)))  # [B, 64] 壓縮即融合
        # Fc1 = self.dropout(F.relu(self.fc1(bottleneck)))      # [B, embd_size]

        # # ---------- Gated Fusion ----------
        # gate_in = torch.cat((eA, eE, eT), dim=1)        # [B, 3D] 只拿來算 gate
        # g = torch.sigmoid(self.gate(gate_in))   # [B, 3] 三條的權重,和為1
        # ga, ge, gt = g[:, 0:1], g[:, 1:2], g[:, 2:3]    # 各 [B, 1]
        # eFusion = ga * eA + ge * eE + gt * eT           # [B, D] 加權相加
        # Fc1 = self.dropout(F.relu(self.fc1(eFusion)))

        Fc2 = self.dropout(F.relu(self.fc2(Fc1)))
        Fc3 = self.dropout(F.relu(self.fc3(Fc2)))
        dep_logits = self.oup(Fc3)

        if return_feature:
            return atei_logits, dep_logits, Fc3
        return atei_logits, dep_logits, (aux_a, aux_t, aux_e)

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
    
    def masked_attn_pool(self, x, mask, attn_layer):
        # x: [B, T, D], mask: [B, T] (True = padding)
        scores = attn_layer(x).squeeze(-1)          # [B, T]
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [B, T, 1]
        return (x * weights).sum(dim=1)             # [B, D]

# ============================================================
# Train / Val
# ============================================================
def train_one_epoch(model, tr_loader, loss_atei, loss_dep, opt, device,
                    cur_epoch, tot_epochs, scaler, accum_steps=1,alpha_gate=1.0, cur_lambda=LAMBDA_ATEI):
    model.train()
    totAteiLoss = totDepLoss = totLoss = 0.0
    correct_atei = correct_dep = valid_batches = total_samples = 0
    valid_atei_samples = 0
    train_true_arr, train_pred_arr = [], []
    

    pbar = tqdm(tr_loader, desc=f"Training epoch {cur_epoch}/{tot_epochs}",
                leave=False, unit='batch')
    opt.zero_grad()
    for step, data in enumerate(pbar):
        xa, xt, aMask, tMask, atei_label, dep_label, Patient, xa_seg_list, xt_seg_list = data

        xa = xa.to(device, non_blocking=True)
        xt = xt.to(device, non_blocking=True)
        aMask = aMask.to(device, non_blocking=True)
        tMask = tMask.to(device, non_blocking=True)
        atei_label = atei_label.to(device, non_blocking=True)
        dep_label = dep_label.to(device, non_blocking=True)
        xa_seg_list = [x.to(device, non_blocking=True) for x in xa_seg_list]
        xt_seg_list = [x.to(device, non_blocking=True) for x in xt_seg_list]

        
        # with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
        #                     dtype=torch.float16):
        with torch.autocast(device_type="cuda", enabled=False):   # ← fp32,enabled=False
            atei_logits, dep_logits, aux_logits = model(
                xa, xt, aMask, tMask,
                xa_seg_list=xa_seg_list, xt_seg_list=xt_seg_list, alpha_gate=alpha_gate,
            )
            aux_a, aux_t, aux_e = aux_logits

            if (atei_label != -1).any():
                L_Atei = loss_atei(atei_logits, atei_label)
            else:
                L_Atei = torch.tensor(0.0, device=device)
            L_Depression = loss_dep(dep_logits, dep_label)
            L_aux = (loss_dep(aux_a, dep_label)
                     + loss_dep(aux_t, dep_label)
                     + loss_dep(aux_e, dep_label)) / 3
            L_Total = cur_lambda * L_Atei + L_Depression + 0.1 * L_aux

        (L_Total / accum_steps).backward()        # L_Total.backward()
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()

        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # opt.step()

        totAteiLoss += L_Atei.item()
        totDepLoss += L_Depression.item()
        totLoss += L_Total.item()

        atei_pred = atei_logits.argmax(dim=-1)
        dep_pred = dep_logits.argmax(dim=-1)

        valid_atei_mask = atei_label != -1

        correct_atei += ((atei_pred == atei_label) & valid_atei_mask).sum().item()
        valid_atei_samples += valid_atei_mask.sum().item()

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
    
    if (step + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad()

    print("Train true dist:", Counter(train_true_arr))
    print("Train pred dist:", Counter(train_pred_arr))

    return {
        "atei_loss": totAteiLoss / valid_batches,
        "dep_loss": totDepLoss / valid_batches,
        "tot_loss": totLoss / valid_batches,
        "cur_atei_acc": correct_atei / max(valid_atei_samples, 1),
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
            xa, xt, aMask, tMask, atei_label, dep_label, Patient, xa_seg_list, xt_seg_list = data

            xa = xa.to(device, non_blocking=True)
            xt = xt.to(device, non_blocking=True)
            aMask = aMask.to(device, non_blocking=True)
            tMask = tMask.to(device, non_blocking=True)
            dep_label = dep_label.to(device, non_blocking=True)
            xa_seg_list = [x.to(device, non_blocking=True) for x in xa_seg_list]
            xt_seg_list = [x.to(device, non_blocking=True) for x in xt_seg_list]

            with torch.autocast(device_type="cuda", enabled=False):   # ← fp32,enabled=False
            # with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                                # dtype=torch.float16):
                _, dep_logits, _ = model(
                    xa, xt, aMask, tMask,
                    xa_seg_list=xa_seg_list, xt_seg_list=xt_seg_list
                )
                patient_dep = dep_logits.squeeze(0)
                L_Depression = loss_dep(patient_dep.unsqueeze(0), dep_label)

            dep_pred = patient_dep.argmax(dim=-1)
            true_arr.append(int(dep_label.item()))
            pred_arr.append(int(dep_pred.item()))
            totDepLoss += L_Depression.item()
            valid_batches += 1

            pbar.set_postfix({"dep_loss": totDepLoss / valid_batches})

    metrics = get_metrics(true_arr, pred_arr)
    print("Val true dist:", Counter(true_arr))
    print("Val  pred dist:", Counter(pred_arr))
    print("Confusion matrix:")
    print(confusion_matrix(true_arr, pred_arr, labels=list(range(N_CLASSES))))
    print(classification_report(true_arr, pred_arr, labels=list(range(N_CLASSES)),
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
    def __init__(self, fold: str = "tr", cv_split=None):
        self.ds = []
        a_root = Path("datasets/Feature/HuBERT_pooled_bin")     # ← pooled
        t_root = Path("datasets/Feature/RoBerTa_full_bin")      # ← token-level

        depMap, train_Idx, test_Idx = get_Split_and_GroundTrue()
        if cv_split is not None:
            patient_Idx = cv_split[{"tr": "train", "val": "val", "test": "val"}[fold]]
        else:
            patient_Idx = {"tr": train_Idx, "val": test_Idx, "test": test_Idx}[fold]

        PseudoLabel = np.load("PseudoLabel_all_distilbert_zdist_q30_70_bin.npz")
        # PseudoLabel = np.load("PseudoLabel_all_contrastive_q30_70_bin.npz")
        patientIdx = PseudoLabel["patientIdx"]
        pseudo_label = PseudoLabel["label"]
        PseudoMap = {int(x): int(y) for x, y in zip(patientIdx, pseudo_label)}

        for p in patient_Idx:
            a_path = a_root / f"{p}_acoustic.pt"
            t_path = t_root / f"{p}_text.pt"
            assert a_path.exists() and t_path.exists(), f"ds error: {p}"

            dep_label = depMap[p]
            atei_label = PseudoMap[p] if p in PseudoMap else -1

            self.ds.append((p, atei_label, dep_label, a_path, t_path))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        Patient, PseudoL, DepL, a_path, t_path = self.ds[index]
        xa = torch.load(str(a_path), map_location="cpu")
        xt = torch.load(str(t_path), map_location="cpu")

        # xa: list of [1, 1024] (pooled,每句一個 vector)
        # xt: list of [1, L_i, 1024] (token-level,每句多 token)
        xa_list = [x.squeeze(0) for x in xa]   # list of [1024]
        xt_list = [x.squeeze(0) for x in xt]   # list of [L_i, 1024]

        atei_label = torch.tensor(PseudoL, dtype=torch.long)
        dep_label = torch.tensor(DepL, dtype=torch.long)
        return xa_list, xt_list, atei_label, dep_label, Patient


def stage2_collate_fn(batch):
    """
    每個 sample:
      xa_i: list of [1024]      ← segment-level audio (HuBERT pooled)
      xt_i: list of [L_i, 1024] ← token-level text
    """
    xa_seg_list = []   # ATEI 用,audio: [num_seg, 1024]
    xt_seg_list = []   # ATEI 用,text:  [num_seg, max_L_i, 1024]
    xa_pool_list = []  # Stage3 主體用,audio: [num_seg, 1024]
    xt_pool_list = []  # Stage3 主體用,text 也要 segment-level: token mean -> [num_seg, 1024]
    atei_labels, dep_labels, patients = [], [], []

    for xa_i, xt_i, atei_label, dep_label, patient in batch:
        # Stage3 主體:
        #   audio 已是 segment-level,stack 即可 [num_seg, 1024]
        xa_pool_list.append(torch.stack(xa_i, dim=0))
        #   text 每句 token mean -> [num_seg, 1024]
        xt_pool_list.append(torch.stack([x.mean(dim=0) for x in xt_i], dim=0))

        # ATEI:
        #   audio 同上 [num_seg, 1024]
        xa_seg_list.append(torch.stack(xa_i, dim=0))
        #   text 保留 token-level pad -> [num_seg, max_L_i, 1024]
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

    return xa_pool, xt_pool, aMask, tMask, atei_labels, dep_labels, patients, xa_seg_list, xt_seg_list

# def get_metrics(y_true, y_pred):
#     return {
#         "acc": accuracy_score(y_true, y_pred),
#         "pre": precision_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
#         "rec": recall_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
#         "f1": f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0),
#     }
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

    for split in splits:
        fold_id = split["fold"]
        set_seed(ARGS.seed + fold_id)
        g = torch.Generator()
        g.manual_seed(ARGS.seed + fold_id)

        run_name = ARGS.wandb_name + f"_fold{fold_id}" if ARGS.wandb_name else (
            f"stage2-quick-bin_seed{ARGS.seed}_lr{LR:.0e}_wd{ARGS.weight_decay:.0e}_"
            f"do{ARGS.dropout:.2f}_la{LAMBDA_ATEI:.2f}_a{ALPHA_INIT:.2f}_"
            f"d{D_MODEL}_l{TRANSFORMER_ENC_LAYERS}_{run_id}"
        )

        if ARGS.use_wandb:
            wandb.init(
                project=ARGS.wandb_project,
                name=run_name,
                config={
                    "seed": ARGS.seed, "d_model": D_MODEL, "nhead": NHEAD,
                    "lr": LR, "epochs": EPOCHS, "enc_layers": TRANSFORMER_ENC_LAYERS,
                    "dropout": ARGS.dropout, "atei_dropout": ARGS.atei_dropout,
                    "weight_decay": ARGS.weight_decay,
                    "lambda_atei": LAMBDA_ATEI, "alpha_init": ALPHA_INIT,
                    "patience": PATIENCE,
                    "loss_total": "LAMBDA_ATEI * L_Atei + L_Depression",
                    "stage1_ckpt": ARGS.stage1_ckpt,
                    "audio_feature": "HuBERT_pooled_bin (pooled)",
                    "text_feature": "RoBerTa_full_bin (token-level)",
                    "atei_level": "segment-level",
                    "n_classes": N_CLASSES,
                },
                save_code=True,
            )

        print("\n" + "=" * 100)
        print(f"FOLD {fold_id}")
        print("=" * 100)
        best_val_f1 = -1.0
        bad_epochs = 0

        # 1. Dataset
        if split["train"] is None:
            trDS = stage2_dataset(fold="tr")
            valDS = stage2_dataset(fold="val")
        else:
            trDS = stage2_dataset(fold="tr", cv_split=split)
            valDS = stage2_dataset(fold="val", cv_split=split)

        tr_loader = DataLoader(
            trDS, collate_fn=stage2_collate_fn,
            batch_size=ARGS.batch_size,
            shuffle=True, worker_init_fn=numpy_random_init,
            num_workers=0, pin_memory=True, generator=g,
        )
        val_loader = DataLoader(
            valDS, collate_fn=stage2_collate_fn,
            shuffle=False, batch_size=1, num_workers=0, pin_memory=True,
        )

        if ARGS.use_wandb:
            wandb.config.update({
                "train_samples": len(trDS),
                "val_samples": len(valDS),
            })

        # 2. Model
        model = whole_model(D_MODEL, NHEAD).to(device)
        print("*" * 10)

        atei_params = list(model.atei.parameters())
        other_params = [p for name, p in model.named_parameters()
                        if not name.startswith("atei.")]
        atei_wd = ARGS.atei_wd if ARGS.atei_wd is not None else ARGS.weight_decay
        opt = torch.optim.Adam(
            [
                {"params": atei_params, "lr": LR * ARGS.atei_lr_scale,"weight_decay": atei_wd},
                {"params": other_params, "lr": LR,"weight_decay": ARGS.weight_decay},
            ],
        )
        print(f"[Optimizer] ATEI lr={LR * ARGS.atei_lr_scale:.2e}, other lr={LR:.2e}")
        print(f"[Optimizer] ATEI wd={atei_wd:.2e}, other wd={ARGS.weight_decay:.2e}")
        scaler = torch.GradScaler('cuda')

        # Class weights
        train_ds_records = trDS.ds
        dep_counter = Counter([int(x[2]) for x in train_ds_records])
        atei_counter = Counter([int(x[1]) for x in train_ds_records])
        total = sum(dep_counter.values())
        n_classes = N_CLASSES
        weights = torch.tensor([
            total / (n_classes * dep_counter[i]) for i in range(n_classes)
        ], dtype=torch.float, device=device)

        print("Train dep dist:", dep_counter)
        print("Train ATEI dist:", atei_counter)
        print("Class weights:", weights)
        print("Val dep dist:", Counter([int(x[2]) for x in valDS.ds]))
        print("Val ATEI dist:", Counter([int(x[1]) for x in valDS.ds]))

        loss_atei = nn.CrossEntropyLoss(ignore_index=-1)
        loss_dep = nn.CrossEntropyLoss(weight=weights)

        # 3. Train
        for epoch in range(1, EPOCHS + 1):
            # === ATEI warmup schedule ===
            alpha_gate = min(1.0, epoch / ARGS.alpha_warmup) if ARGS.alpha_warmup > 0 else 1.0
            if ARGS.no_atei_loss:
                cur_lambda_atei=0.0
            else:
                cur_lambda_atei = LAMBDA_ATEI * min(1.0, epoch / ARGS.lambda_warmup) if ARGS.lambda_warmup > 0 else LAMBDA_ATEI
            print("=" * 80)            
            print(f"[Epoch {epoch}] alpha_gate={alpha_gate:.3f}, "
                f"cur_lambda={cur_lambda_atei:.4f}")

            tr_result = train_one_epoch(model, tr_loader, loss_atei, loss_dep,
                                        opt, device, epoch, EPOCHS, scaler, accum_steps=ARGS.accum_steps,alpha_gate=alpha_gate,cur_lambda=cur_lambda_atei)
            val_result = val(model, val_loader, loss_dep, device, epoch, EPOCHS)

            print(
                f"[Train] ATEI: {tr_result['atei_loss']:.4f} | "
                f"Dep: {tr_result['dep_loss']:.4f} | "
                f"Total: {tr_result['tot_loss']:.4f} | "
                f"ATEI Acc: {tr_result['cur_atei_acc']:.4f} | "
                f"Dep Acc: {tr_result['cur_dep_acc']:.4f}"
            )
            print(
                f"[Val] Dep Loss: {val_result['dep_loss']:.4f} | "
                f"Acc: {val_result['acc']:.4f} | "
                f"F1: {val_result['f1']:.4f}"
            )

            if val_result["f1"] > best_val_f1:
                best_val_f1 = val_result["f1"]
                bad_epochs = 0

                # 只在 F1 > 0.40 時才存檔
                if best_val_f1 > 0.40:
                    ckpt_name = (
                        f"stage2-quick-bin_{run_id}_seed{ARGS.seed}_"
                        f"f1{best_val_f1:.4f}_ep{epoch:03d}_"
                        f"lr{LR:.0e}_wd{ARGS.weight_decay:.0e}_"
                        f"d{D_MODEL}_l{TRANSFORMER_ENC_LAYERS}_fold{fold_id}.pt"
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
                        "args": vars(ARGS),
                        "d_model": D_MODEL, "nhead": NHEAD,
                        "enc_layers": TRANSFORMER_ENC_LAYERS,
                        "base_lr": LR,
                        "atei_lr": opt.param_groups[0]["lr"],
                        "other_lr": opt.param_groups[1]["lr"],
                        "weight_decay": ARGS.weight_decay,
                        "dropout": ARGS.dropout,
                        "atei_dropout": ARGS.atei_dropout,
                        "lambda_atei": LAMBDA_ATEI,
                        "alpha_init": ALPHA_INIT,
                        "stage1_ckpt": ARGS.stage1_ckpt,
                        "n_classes": N_CLASSES,
                    }, ckpt_path)

                    if ARGS.use_wandb:
                        wandb.run.summary["best_val_f1"] = best_val_f1
                    print(f"[Save Best] Val F1: {best_val_f1:.4f} -> {ckpt_path}")
                else:
                    print(f"[Skip Save] Val F1: {best_val_f1:.4f} (< 0.40, not saved)")
            else:
                bad_epochs += 1
                print(f"[EarlyStop] bad_epochs: {bad_epochs}/{PATIENCE}")

            # wandb log (每個 epoch 都要 log)
            if ARGS.use_wandb:
                wandb.log({
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
                    "lr/atei": opt.param_groups[0]["lr"],
                    "lr/other": opt.param_groups[1]["lr"],
                    "train/cur_lambda": cur_lambda_atei
                })

            # ← 關鍵:加這個 break
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
    print("K-FOLD RESULT")
    print("=" * 100)
    for i, f1 in enumerate(all_fold_results):
        print(f"Fold {i}: {f1:.4f}")
    print(f"\nMean F1: {np.mean(all_fold_results):.4f}")
    print(f"Std  F1: {np.std(all_fold_results):.4f}")


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ARGS = parse_args()

    STAGE1_CKPT = ARGS.stage1_ckpt
    D_MODEL = ARGS.d_model
    NHEAD = ARGS.nhead
    LR = ARGS.lr
    EPOCHS = ARGS.epochs
    TRANSFORMER_ENC_LAYERS = ARGS.enc_layers
    LAMBDA_ATEI = ARGS.lambda_atei
    ALPHA_INIT = ARGS.alpha_init
    PATIENCE = ARGS.patience

    print("** Stage2-Quick-bin Training **")
    main()