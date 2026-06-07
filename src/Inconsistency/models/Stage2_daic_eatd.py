""" 可以嘗試加深audio branch 或者用 Conv1d 去強化聲學特徵 """
"""
Stage2_daic_eatd.py — joint DAIC-WOZ + EATD Stage2 depression detection

原版 Stage2Main_bin.py 的差異:
- Stage1 ckpt: Stage1_daic_eatd 訓出來的
- feature 路徑: datasets/Feat_daic_eatd/
- train = DAIC-train + EATD-train
- val   = DAIC-val (官方 dev) + EATD-val (分開桶)
- model selection: DAIC val macro F1 (B1)
- 額外記錄 EATD val macro F1 (B2)
- pseudo label: datasets/Feat_daic_eatd/PseudoLabel_joint_train_q30_70.npz (train only)
                datasets/Feat_daic_eatd/PseudoLabel_daic_zdist_q30_70.npz  (daic val)
                datasets/Feat_daic_eatd/PseudoLabel_eatd_val_zdist_q30_70.npz (eatd val)
- save_dir: weights/stage2-daic-eatd
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import torch
from Stage1_daic_eatd import atei
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

N_CLASSES  = 2
FEAT_DIR   = Path("datasets/Feat_daic_eatd")
EATD_DIR   = Path("datasets/EATD")

STAGE1_CKPT = "weights/stage1-daic-eatd/stage1-daic-eatd_20260601_161859_seed42_fold0_f10.7246_ep079_lr1e-04_wd1e-04_d256_l1.pt"
D_MODEL    = 256
NHEAD      = 8
LR         = 5e-4
EPOCHS     = 3000
TRANSFORMER_ENC_LAYERS = 1
DROPOUT    = 0.3
ATEI_DROPOUT = 0.3
WEIGHT_DECAY = 0
LAMBDA_ATEI  = 0.1
ALPHA_INIT   = 0.5
PATIENCE     = 500
ACCUM_STEPS  = 1
ENCODER_TYPE = "attn"
CMS_PERIODS  = (1, 4)
CMS_HIDDEN_MULTIPLIER = 4
CMS_ONLINE_UPDATES = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1_ckpt",  type=str,   default=STAGE1_CKPT)
    parser.add_argument("--d_model",      type=int,   default=D_MODEL)
    parser.add_argument("--nhead",        type=int,   default=NHEAD)
    parser.add_argument("--lr",           type=float, default=LR)
    parser.add_argument("--epochs",       type=int,   default=EPOCHS)
    parser.add_argument("--enc_layers",   type=int,   default=TRANSFORMER_ENC_LAYERS)
    parser.add_argument("--dropout",      type=float, default=DROPOUT)
    parser.add_argument("--atei_dropout", type=float, default=ATEI_DROPOUT)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--lambda_atei",  type=float, default=LAMBDA_ATEI)
    parser.add_argument("--alpha_init",   type=float, default=ALPHA_INIT)
    parser.add_argument("--patience",     type=int,   default=PATIENCE)
    parser.add_argument("--save_dir",     type=str,   default="weights/stage2-daic-eatd")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--use_wandb",    action="store_true")
    parser.add_argument("--wandb_project", type=str,  default="Stage2 daic+eatd")
    parser.add_argument("--wandb_name",   type=str,   default=None)
    parser.add_argument("--batch_size",   type=int,   default=64)
    parser.add_argument("--freeze_atei",  action="store_true")
    parser.add_argument("--no_atei_loss", action="store_true")
    parser.add_argument("--atei_lr_scale", type=float, default=0.1)
    parser.add_argument("--atei_wd",      type=float, default=None)
    parser.add_argument("--print_norm",   action="store_true")
    parser.add_argument("--accum_steps",  type=int,   default=1)
    parser.add_argument("--alpha_warmup", type=int,   default=0)
    parser.add_argument("--lambda_warmup", type=int,  default=0)
    parser.add_argument("--low_q",        type=float, default=30)
    parser.add_argument("--high_q",       type=float, default=70)
    parser.add_argument("--encoder_type",  type=str,   default=ENCODER_TYPE,
                        choices=["attn", "hope_attention"])
    parser.add_argument("--cms_periods",  type=int, nargs="+", default=list(CMS_PERIODS))
    parser.add_argument("--cms_hidden_multiplier", type=int, default=CMS_HIDDEN_MULTIPLIER)
    parser.add_argument("--eatd_loss_scale", type=float, default=0.5,
                        help="Loss weight for EATD samples (dep+aux only). 1.0=equal, 0=ignore.")
    parser.add_argument("--kfold",        type=int,   default=0,
                        help="StratifiedKFold over DAIC-train only. <=1 = single run.")
    return parser.parse_args()


# ============================================================
# Dataset
# ============================================================
def get_daic_train_folds(n_splits: int = 3, seed: int = 42):
    """DAIC-train 內部 StratifiedKFold，official dev 固定當 eval。"""
    from sklearn.model_selection import StratifiedKFold
    depMap, train_ids, _ = get_Split_and_GroundTrue()
    labels = [depMap[p] for p in train_ids]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for i, (tr_i, val_i) in enumerate(skf.split(train_ids, labels)):
        folds.append({"fold": i,
                      "train": [train_ids[j] for j in tr_i],
                      "val":   [train_ids[j] for j in val_i]})
    return depMap, folds


def _load_pseudo_map(npz_path: Path) -> dict:
    """Load pseudo label npz -> {id_str: atei_label}. Returns {} if not found."""
    if not npz_path.exists():
        print(f"[PseudoLabel] {npz_path} not found, all atei_label = -1")
        return {}
    d = np.load(npz_path, allow_pickle=True)
    return {str(k): int(v) for k, v in zip(d["patientIdx"], d["label"])}


def _read_eatd_dep(vol_dir: Path, cutoff: float = 53.0):
    lbl = vol_dir / "new_label.txt"
    if not lbl.exists():
        return None
    try:
        return 1 if float(lbl.read_text().strip()) >= cutoff else 0
    except ValueError:
        return None


class stage2_dataset(Dataset):
    """
    fold: "tr"   -> DAIC-train + EATD-train (混合)
          "daic" -> DAIC-val (官方 dev, 純 DAIC)
          "eatd" -> EATD-val
    """
    def __init__(self, fold: str, daic_train_ids: list = None):
        self.ds   = []
        self.fold = fold

        q_tag = f"q{ARGS.low_q:.0f}_{ARGS.high_q:.0f}"

        if fold == "tr":
            self._build_train(q_tag, daic_train_ids)
        elif fold == "daic":
            self._build_daic_val(q_tag)
        elif fold == "eatd":
            self._build_eatd_val(q_tag)
        else:
            raise ValueError(f"unknown fold: {fold}")

    # ── train ──
    def _build_train(self, q_tag, daic_train_ids=None):
        joint_npz  = FEAT_DIR / f"PseudoLabel_joint_train_{q_tag}.npz"
        PseudoMap  = _load_pseudo_map(joint_npz)

        depMap, official_train_ids, _ = get_Split_and_GroundTrue()
        train_ids = daic_train_ids if daic_train_ids is not None else official_train_ids

        # DAIC train
        for pid in train_ids:
            jid    = f"daic_{pid}"
            a_path = FEAT_DIR / f"{pid}_acoustic.pt"
            t_path = FEAT_DIR / f"{pid}_text.pt"
            if not (a_path.exists() and t_path.exists()):
                continue
            atei_l = PseudoMap.get(jid, -1)
            self.ds.append((jid, atei_l, depMap[pid], a_path, t_path))

        # EATD train (t_*)
        eatd_npz  = FEAT_DIR / f"PseudoLabel_eatd_train_zdist_{q_tag}.npz"
        eatd_map  = _load_pseudo_map(eatd_npz)   # key: "t_1", "t_2", ...
        for vol_dir in sorted(EATD_DIR.iterdir(),
                              key=lambda d: int(d.name.split("_")[1])
                              if d.is_dir() and d.name.startswith("t_") else -1):
            if not (vol_dir.is_dir() and vol_dir.name.startswith("t_")):
                continue
            vol    = vol_dir.name
            dep    = _read_eatd_dep(vol_dir)
            if dep is None:
                continue
            a_path = FEAT_DIR / f"{vol}_acoustic.pt"
            t_path = FEAT_DIR / f"{vol}_text.pt"
            if not (a_path.exists() and t_path.exists()):
                continue
            atei_l = eatd_map.get(vol, -1)
            self.ds.append((f"eatd_{vol}", atei_l, dep, a_path, t_path))

    # ── DAIC val ──
    def _build_daic_val(self, q_tag):
        daic_npz  = FEAT_DIR / f"PseudoLabel_daic_zdist_{q_tag}.npz"
        PseudoMap = _load_pseudo_map(daic_npz)
        depMap, _, val_ids = get_Split_and_GroundTrue()
        for pid in val_ids:
            a_path = FEAT_DIR / f"{pid}_acoustic.pt"
            t_path = FEAT_DIR / f"{pid}_text.pt"
            if not (a_path.exists() and t_path.exists()):
                continue
            atei_l = PseudoMap.get(f"daic_{pid}",
                     PseudoMap.get(str(pid), -1))
            self.ds.append((f"daic_{pid}", atei_l, depMap[pid], a_path, t_path))

    # ── EATD val ──
    def _build_eatd_val(self, q_tag):
        eatd_npz  = FEAT_DIR / f"PseudoLabel_eatd_val_zdist_{q_tag}.npz"
        eatd_map  = _load_pseudo_map(eatd_npz)
        for vol_dir in sorted(EATD_DIR.iterdir(),
                              key=lambda d: int(d.name.split("_")[1])
                              if d.is_dir() and d.name.startswith("v_") else -1):
            if not (vol_dir.is_dir() and vol_dir.name.startswith("v_")):
                continue
            vol    = vol_dir.name
            dep    = _read_eatd_dep(vol_dir)
            if dep is None:
                continue
            a_path = FEAT_DIR / f"{vol}_acoustic.pt"
            t_path = FEAT_DIR / f"{vol}_text.pt"
            if not (a_path.exists() and t_path.exists()):
                continue
            atei_l = eatd_map.get(vol, -1)
            self.ds.append((f"eatd_{vol}", atei_l, dep, a_path, t_path))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        jid, atei_l, dep_l, a_path, t_path = self.ds[index]
        xa = torch.load(str(a_path), map_location="cpu")
        xt = torch.load(str(t_path), map_location="cpu")
        xa_list = [x.squeeze(0) for x in xa]
        xt_list = [x.squeeze(0) for x in xt]
        return xa_list, xt_list, \
               torch.tensor(atei_l, dtype=torch.long), \
               torch.tensor(dep_l,  dtype=torch.long), \
               jid


def stage2_collate_fn(batch):
    xa_seg_list, xt_seg_list = [], []
    xa_pool_list, xt_pool_list = [], []
    atei_labels, dep_labels, patients = [], [], []
    seg_lens = []   # actual number of segments per patient

    for xa_i, xt_i, atei_label, dep_label, patient in batch:
        xa_pool_list.append(torch.stack(xa_i, dim=0))
        xt_pool_list.append(torch.stack([x.mean(dim=0) for x in xt_i], dim=0))
        xa_seg_list.append(torch.stack(xa_i, dim=0))
        xt_seg_list.append(pad_sequence(xt_i, batch_first=True))
        atei_labels.append(atei_label)
        dep_labels.append(dep_label)
        patients.append(patient)
        seg_lens.append(len(xa_i))

    xa_pool  = pad_sequence(xa_pool_list, batch_first=True)
    xt_pool  = pad_sequence(xt_pool_list, batch_first=True)
    max_segs = xa_pool.size(1)

    # build mask from actual segment lengths (not feature sum)
    aMask = torch.ones(len(batch), max_segs, dtype=torch.bool)
    tMask = torch.ones(len(batch), max_segs, dtype=torch.bool)
    for i, l in enumerate(seg_lens):
        aMask[i, :l] = False
        tMask[i, :l] = False

    return (xa_pool, xt_pool, aMask, tMask,
            torch.stack(atei_labels), torch.stack(dep_labels),
            patients, xa_seg_list, xt_seg_list)


# ============================================================
# Model  (identical to Stage2Main_bin whole_model)
# ============================================================
class whole_model(nn.Module):
    def __init__(self, embd_size=D_MODEL, nheads=NHEAD):
        super().__init__()
        self.a_in_proj = nn.Sequential(nn.Linear(1024, embd_size),
                                       nn.LayerNorm(embd_size))
        self.t_in_proj = nn.Sequential(nn.Linear(1024, embd_size),
                                       nn.LayerNorm(embd_size))

        self.atei = atei(embd_size=256, nheads=nheads,
                         dropout=ARGS.atei_dropout, enc_layers=1)
        ckpt = torch.load(ARGS.stage1_ckpt, map_location="cpu")
        self.atei.load_state_dict(ckpt["model_state_dict"])
        if ARGS.freeze_atei:
            for p in self.atei.parameters():
                p.requires_grad = False
            print("[ATEI] frozen")

        self.encoder_type = ARGS.encoder_type
        if self.encoder_type == "attn":
            self.a_transformer_enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embd_size, dropout=ARGS.dropout,
                    dim_feedforward=4*embd_size,
                    nhead=nheads, batch_first=True, norm_first=True),
                num_layers=TRANSFORMER_ENC_LAYERS, enable_nested_tensor=False)
            self.t_transformer_enc = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=embd_size, dropout=ARGS.dropout,
                    dim_feedforward=4*embd_size,
                    nhead=nheads, batch_first=True, norm_first=True),
                num_layers=TRANSFORMER_ENC_LAYERS, enable_nested_tensor=False)
        elif self.encoder_type == "hope_attention":
            from hope_adapter import HopeEncoderBlock
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

        self.atei_proj  = nn.Linear(256, embd_size)
        self.a_post_norm = nn.LayerNorm(embd_size)
        self.t_post_norm = nn.LayerNorm(embd_size)
        self.dropout    = nn.Dropout(ARGS.dropout)
        self.fc1        = nn.Linear(3 * embd_size, embd_size)
        self.fc2        = nn.Linear(embd_size, embd_size)
        self.fc3        = nn.Linear(embd_size, embd_size)
        self.alpha      = nn.Parameter(torch.tensor(ALPHA_INIT))
        self.oup        = nn.Linear(embd_size, N_CLASSES)
        self.aux_a_head = nn.Linear(embd_size, N_CLASSES)
        self.aux_t_head = nn.Linear(embd_size, N_CLASSES)
        self.aux_e_head = nn.Linear(embd_size, N_CLASSES)

    def forward(self, XA, XT, aMask=None, tMask=None,
                xa_seg_list=None, xt_seg_list=None,
                alpha_gate=1.0, return_feature=False):
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
        eA = self.masked_mean(HA, aMask)
        eT = self.masked_mean(HT, tMask)

        # ATEI batched forward
        seg_counts = [x.size(0) for x in xa_seg_list]
        max_L_t    = max(x.size(1) for x in xt_seg_list)
        D_in       = xa_seg_list[0].size(-1)
        dev        = xa_seg_list[0].device
        total_segs = sum(seg_counts)
        batch_a    = torch.zeros(total_segs, D_in, device=dev,
                                 dtype=xa_seg_list[0].dtype)
        batch_t    = torch.zeros(total_segs, max_L_t, D_in, device=dev,
                                 dtype=xt_seg_list[0].dtype)
        start = 0
        for xa_seg, xt_seg in zip(xa_seg_list, xt_seg_list):
            n = xa_seg.size(0)
            batch_a[start:start+n] = xa_seg
            batch_t[start:start+n, :xt_seg.size(1)] = xt_seg
            start += n
        mask_t = (batch_t.sum(dim=-1) == 0)
        eE_all, logits_all = self.atei(batch_a, batch_t, mask_t)

        eE_list, atei_logits_list = [], []
        start = 0
        for count in seg_counts:
            eE_list.append(eE_all[start:start+count].mean(dim=0))
            atei_logits_list.append(logits_all[start:start+count].mean(dim=0))
            start += count
        eE          = self.atei_proj(torch.stack(eE_list, dim=0))
        atei_logits = torch.stack(atei_logits_list, dim=0)

        alpha = torch.clamp(self.alpha, 0.0, 2.0) * alpha_gate
        eE    = eE * alpha
        eA    = self.a_post_norm(eA)
        eT    = self.t_post_norm(eT)

        aux_a = self.aux_a_head(eA)
        aux_t = self.aux_t_head(eT)
        aux_e = self.aux_e_head(eE)

        if self.training and ARGS.print_norm:
            print(f"eA={eA.norm(dim=-1).mean():.4f} "
                  f"eE={eE.norm(dim=-1).mean():.4f} "
                  f"eT={eT.norm(dim=-1).mean():.4f} "
                  f"alpha={self.alpha.item():.4f}")

        eFusion    = torch.cat((eA, eE, eT), dim=1)
        Fc1        = self.dropout(F.relu(self.fc1(eFusion)))
        Fc2        = self.dropout(F.relu(self.fc2(Fc1)))
        Fc3        = self.dropout(F.relu(self.fc3(Fc2)))
        dep_logits = self.oup(Fc3)

        if return_feature:
            return atei_logits, dep_logits, Fc3
        return atei_logits, dep_logits, (aux_a, aux_t, aux_e)

    def masked_mean(self, x, mask):
        if mask is None:
            return x.mean(dim=1)
        valid = (~mask).unsqueeze(-1)
        return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)


# ============================================================
# Train / Val
# ============================================================
def get_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "pre": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "rec": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1":  f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_one_epoch(model, tr_loader, loss_atei, loss_dep, loss_dep_none, opt,
                    device, cur_epoch, tot_epochs, accum_steps, alpha_gate, cur_lambda):
    model.train()
    totAtei = totDep = totLoss = 0.0
    correct_dep = valid_batches = total_samples = 0
    valid_atei_samples = correct_atei = 0
    true_arr, pred_arr = [], []

    pbar = tqdm(tr_loader, desc=f"Train {cur_epoch}/{tot_epochs}",
                leave=False, unit="batch")
    opt.zero_grad()
    for step, data in enumerate(pbar):
        (xa, xt, aMask, tMask, atei_label, dep_label,
         Patient, xa_seg_list, xt_seg_list) = data

        xa         = xa.to(device, non_blocking=True)
        xt         = xt.to(device, non_blocking=True)
        aMask      = aMask.to(device, non_blocking=True)
        tMask      = tMask.to(device, non_blocking=True)
        atei_label = atei_label.to(device, non_blocking=True)
        dep_label  = dep_label.to(device, non_blocking=True)
        xa_seg_list = [x.to(device, non_blocking=True) for x in xa_seg_list]
        xt_seg_list = [x.to(device, non_blocking=True) for x in xt_seg_list]

        # per-sample dep loss scale: EATD samples get eatd_loss_scale
        sample_scale = torch.tensor(
            [ARGS.eatd_loss_scale if str(p).startswith("eatd_") else 1.0
             for p in Patient],
            dtype=torch.float, device=device
        )  # [B]

        with torch.autocast(device_type="cuda", enabled=False):
            atei_logits, dep_logits, aux_logits = model(
                xa, xt, aMask, tMask,
                xa_seg_list=xa_seg_list, xt_seg_list=xt_seg_list,
                alpha_gate=alpha_gate)
            aux_a, aux_t, aux_e = aux_logits

            L_Atei = (loss_atei(atei_logits, atei_label)
                      if (atei_label != -1).any()
                      else torch.tensor(0.0, device=device))
            # per-sample weighted dep loss
            L_Dep  = (loss_dep_none(dep_logits, dep_label) * sample_scale).mean()
            L_aux  = ((loss_dep_none(aux_a, dep_label) * sample_scale).mean()
                      + (loss_dep_none(aux_t, dep_label) * sample_scale).mean()
                      + (loss_dep_none(aux_e, dep_label) * sample_scale).mean()) / 3
            L_Total = cur_lambda * L_Atei + L_Dep + 0.3 * L_aux

        (L_Total / accum_steps).backward()
        if (step + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step(); opt.zero_grad()

        valid_atei_mask = (atei_label != -1)
        correct_atei   += ((atei_logits.argmax(-1) == atei_label) & valid_atei_mask).sum().item()
        valid_atei_samples += valid_atei_mask.sum().item()

        dep_pred     = dep_logits.argmax(-1)
        correct_dep += (dep_pred == dep_label).sum().item()
        total_samples += dep_label.size(0)
        valid_batches += 1
        totAtei += L_Atei.item(); totDep += L_Dep.item(); totLoss += L_Total.item()
        true_arr.extend(dep_label.cpu().tolist())
        pred_arr.extend(dep_pred.cpu().tolist())
        pbar.set_postfix({"dep": totDep/valid_batches, "tot": totLoss/valid_batches, "dep_acc": correct_dep/max(total_samples,1), "atei_acc": correct_atei/max(valid_atei_samples,1)})

    if (step + 1) % accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step(); opt.zero_grad()

    print("Train true dist:", Counter(true_arr))
    print("Train pred dist:", Counter(pred_arr))
    return {"atei_loss": totAtei/valid_batches, "dep_loss": totDep/valid_batches,
            "tot_loss": totLoss/valid_batches,
            "cur_atei_acc": correct_atei/max(valid_atei_samples,1),
            "cur_dep_acc": correct_dep/total_samples}


@torch.inference_mode()
def run_val(model, loader, loss_dep, device, cur_epoch, tot_epochs, tag="Val"):
    model.eval()
    totLoss, valid_batches = 0.0, 0
    true_arr, pred_arr = [], []

    for data in tqdm(loader, desc=f"{tag} {cur_epoch}/{tot_epochs}",
                     leave=False, unit="batch"):
        if data is None:
            continue
        (xa, xt, aMask, tMask, atei_label, dep_label,
         Patient, xa_seg_list, xt_seg_list) = data

        xa         = xa.to(device, non_blocking=True)
        xt         = xt.to(device, non_blocking=True)
        aMask      = aMask.to(device, non_blocking=True)
        tMask      = tMask.to(device, non_blocking=True)
        dep_label  = dep_label.to(device, non_blocking=True)
        xa_seg_list = [x.to(device, non_blocking=True) for x in xa_seg_list]
        xt_seg_list = [x.to(device, non_blocking=True) for x in xt_seg_list]

        with torch.autocast(device_type="cuda", enabled=False):
            _, dep_logits, _ = model(xa, xt, aMask, tMask,
                                     xa_seg_list=xa_seg_list,
                                     xt_seg_list=xt_seg_list)
            dep_logits = dep_logits.squeeze(0)
            totLoss   += loss_dep(dep_logits.unsqueeze(0), dep_label).item()

        true_arr.append(int(dep_label.item()))
        pred_arr.append(int(dep_logits.argmax().item()))
        valid_batches += 1

    m = get_metrics(true_arr, pred_arr)
    print(f"\n[{tag}] true dist: {Counter(true_arr)}")
    print(f"[{tag}] pred dist: {Counter(pred_arr)}")
    print(confusion_matrix(true_arr, pred_arr, labels=list(range(N_CLASSES))))
    print(classification_report(true_arr, pred_arr,
                                labels=list(range(N_CLASSES)),
                                digits=4, zero_division=0))
    return {"dep_loss": totLoss/max(valid_batches,1), **m,
            "labels": true_arr, "preds": pred_arr}


# ============================================================
# Main
# ============================================================
def main():
    run_id      = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_timer = Timer()
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir    = Path(ARGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── kfold splits ──
    if ARGS.kfold >= 2:
        _, daic_folds = get_daic_train_folds(n_splits=ARGS.kfold, seed=ARGS.seed)
    else:
        _, official_train_ids, _ = get_Split_and_GroundTrue()
        daic_folds = [{"fold": 0, "train": official_train_ids, "val": None}]

    all_fold_daic_f1 = []

    for fold_info in daic_folds:
        fold_id        = fold_info["fold"]
        fold_train_ids = fold_info["train"]   # DAIC-train subset for this fold

        set_seed(ARGS.seed + fold_id)
        g = torch.Generator(); g.manual_seed(ARGS.seed + fold_id)

        print(f"\n{'='*80}\nFOLD {fold_id}\n{'='*80}")

        # ── Dataset ──
        trDS   = stage2_dataset("tr",   daic_train_ids=fold_train_ids)
        daicDS = stage2_dataset("daic")

        print(f"[Dataset] train={len(trDS)} "
              f"(DAIC={sum(1 for d in trDS.ds if d[0].startswith('daic_'))} "
              f"EATD={sum(1 for d in trDS.ds if d[0].startswith('eatd_'))})")
        print(f"[Dataset] DAIC-val={len(daicDS)}")

        tr_loader   = DataLoader(trDS,   collate_fn=stage2_collate_fn,
                                 batch_size=ARGS.batch_size, shuffle=True,
                                 worker_init_fn=numpy_random_init,
                                 num_workers=0, pin_memory=True, generator=g)
        daic_loader = DataLoader(daicDS, collate_fn=stage2_collate_fn,
                                 batch_size=1, shuffle=False,
                                 num_workers=0, pin_memory=True)


        # ── Model ──
        model = whole_model(D_MODEL, NHEAD).to(device)

        atei_params  = list(model.atei.parameters())
        other_params = [p for n, p in model.named_parameters()
                        if not n.startswith("atei.")]
        atei_wd = ARGS.atei_wd if ARGS.atei_wd is not None else ARGS.weight_decay
        opt = torch.optim.Adam([
            {"params": atei_params,  "lr": LR * ARGS.atei_lr_scale, "weight_decay": atei_wd},
            {"params": other_params, "lr": LR,                       "weight_decay": ARGS.weight_decay},
        ])
        print(f"[Opt] ATEI lr={LR*ARGS.atei_lr_scale:.2e}  other lr={LR:.2e}")
    
        # Class weights (train dep dist)
        dep_counter = Counter(int(d[2]) for d in trDS.ds)
        total       = sum(dep_counter.values())
        weights     = torch.tensor([total / (N_CLASSES * dep_counter[i])
                                     for i in range(N_CLASSES)],
                                    dtype=torch.float, device=device)
        print(f"[Train dep dist] {dep_counter}  weights={weights}")
    
        loss_atei     = nn.CrossEntropyLoss(ignore_index=-1)
        loss_dep      = nn.CrossEntropyLoss(weight=weights)
        loss_dep_none = nn.CrossEntropyLoss(weight=weights, reduction="none")
    
        if ARGS.use_wandb:
            run_name = ARGS.wandb_name or (
                f"stage2-daic-eatd_seed{ARGS.seed}_fold{fold_id}_lr{LR:.0e}_"
                f"la{LAMBDA_ATEI:.2f}_a{ALPHA_INIT:.2f}_{run_id}")
            wandb.init(project=ARGS.wandb_project, name=run_name,
                       config=vars(ARGS), reinit=True, save_code=True)
    
        best_daic_f1      = -1.0
        bad_epochs        = 0
    
        for epoch in range(1, EPOCHS + 1):
            alpha_gate    = min(1.0, epoch/ARGS.alpha_warmup) if ARGS.alpha_warmup > 0 else 1.0
            cur_lambda    = (0.0 if ARGS.no_atei_loss else
                             LAMBDA_ATEI * min(1.0, epoch/ARGS.lambda_warmup)
                             if ARGS.lambda_warmup > 0 else LAMBDA_ATEI)
    
            print("=" * 80)
            print(f"[Epoch {epoch}] alpha_gate={alpha_gate:.3f} lambda={cur_lambda:.4f}")
    
            tr_r   = train_one_epoch(model, tr_loader, loss_atei, loss_dep, loss_dep_none,
                                     opt, device, epoch, EPOCHS,
                                     ARGS.accum_steps, alpha_gate, cur_lambda)
            daic_r = run_val(model, daic_loader, loss_dep, device, epoch, EPOCHS, "DAIC-val")
    
            print(f"[Train] atei_loss={tr_r['atei_loss']:.4f} dep_loss={tr_r['dep_loss']:.4f} tot={tr_r['tot_loss']:.4f} | atei_acc={tr_r['cur_atei_acc']:.4f} dep_acc={tr_r['cur_dep_acc']:.4f}")
            print(f"[DAIC-val] F1={daic_r['f1']:.4f} Acc={daic_r['acc']:.4f}")

            # model selection on DAIC val F1 (B1)
            if daic_r["f1"] > best_daic_f1:
                best_daic_f1         = daic_r["f1"]
                bad_epochs           = 0
                if best_daic_f1 > 0.40:
                    ckpt_name = (
                        f"stage2-daic-eatd_{run_id}_seed{ARGS.seed}_fold{fold_id}_"
                        f"daicF1_{best_daic_f1:.4f}_"
                        f"ep{epoch:03d}_lr{LR:.0e}_d{D_MODEL}_l{TRANSFORMER_ENC_LAYERS}.pt"
                    )
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "best_daic_f1": best_daic_f1,
                        "args": vars(ARGS),
                        "d_model": D_MODEL, "nhead": NHEAD,
                        "enc_layers": TRANSFORMER_ENC_LAYERS,
                        "stage1_ckpt": ARGS.stage1_ckpt,
                        "n_classes": N_CLASSES,
                    }, save_dir / ckpt_name)
                    print(f"[Save] DAIC F1={best_daic_f1:.4f} -> {ckpt_name}")
            else:
                bad_epochs += 1
                print(f"[EarlyStop] {bad_epochs}/{PATIENCE}")
    
            if ARGS.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train/atei_loss": tr_r["atei_loss"],
                    "train/dep_loss":  tr_r["dep_loss"],
                    "train/tot_loss":  tr_r["tot_loss"],
                    "train/atei_acc":  tr_r["cur_atei_acc"],
                    "train/dep_acc":   tr_r["cur_dep_acc"],
                    "daic_val/f1":     daic_r["f1"],
                    "daic_val/acc":    daic_r["acc"],
                    "daic_val/pre":    daic_r["pre"],
                    "daic_val/rec":    daic_r["rec"],
                    "best/daic_f1":    best_daic_f1,
                    "no_improve":      bad_epochs,
                    "train/cur_lambda": cur_lambda,
                })
    
            if bad_epochs >= PATIENCE:
                print(f"[EarlyStop] stop ep {epoch}, best DAIC F1={best_daic_f1:.4f}")
                break
    
        # ── end of epoch loop ──
        print(f"\n[Fold {fold_id}] best DAIC F1={best_daic_f1:.4f}")
        all_fold_daic_f1.append(best_daic_f1)
        if ARGS.use_wandb:
            wandb.finish()

    print("\n" + "="*80)
    print("K-FOLD RESULT (Stage2 daic+eatd)")
    print("="*80)
    for i, f1 in enumerate(all_fold_daic_f1):
        print(f"Fold {i}: DAIC F1={f1:.4f}")
    print(f"\nDAIC Mean={np.mean(all_fold_daic_f1):.4f} Std={np.std(all_fold_daic_f1):.4f}")
    print(f"Total time: {total_timer}")


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ARGS = parse_args()

    STAGE1_CKPT = ARGS.stage1_ckpt
    D_MODEL     = ARGS.d_model
    NHEAD       = ARGS.nhead
    LR          = ARGS.lr
    EPOCHS      = ARGS.epochs
    TRANSFORMER_ENC_LAYERS = ARGS.enc_layers
    LAMBDA_ATEI = ARGS.lambda_atei
    ALPHA_INIT  = ARGS.alpha_init
    PATIENCE    = ARGS.patience

    print("** Stage2 daic+eatd **")
    main()