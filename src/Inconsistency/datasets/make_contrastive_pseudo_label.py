"""
uv run src/Inconsistency/datasets/make_contrastive_pseudo_label.py

用 cross-modal contrastive 取代 emotion-recognizer-based pseudo label。

動機
----
原版 pseudo label 用 DistilBERT(text emotion)+ Wav2Vec2-IEMOCAP(audio emotion)
數 pos/neg/neu,再算 z-score 距離。問題是這兩個 emotion recognizer 跟 depression
任務無關(一個是多語 sentiment,一個是 acted IEMOCAP),產生的 inconsistency 不一定
跟 depression 相關。

本方法改用「資料自身」學 audio/text 對齊,不依賴外部 emotion recognizer:
  1. 訓一個獨立的 contrastive encoder,讓同一 segment 的 audio/text 靠近 (InfoNCE)
  2. 訓完後,每個 patient 算「audio/text 平均對齊度」
  3. 對齊度低 = inconsistent (label 0),對齊度高 = consistent (label 1)
  4. 沿用 q30/q70 切法,只保留兩端高信心樣本

這個 contrastive encoder 跟 paper 的 ATEI module 完全獨立、無關。
ATEI 架構 (Stage1Tr_quick) 一行都不動,只是換一個 pseudo label 來源。

輸出
----
PseudoLabel_all_contrastive_q30_70.npz
  格式跟原版完全相同 (patientIdx, label, score, all_patientIdx, all_score, low_th, high_th)
  → 可直接餵給 Stage1Tr_quick / Stage2Main_quick,只改 np.load 檔名
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from Inconsistency.utils import set_seed
from Inconsistency.datasets.inconsistentLabel import get_Split_and_GroundTrue

# ============================================================
# Config
# ============================================================
SPLIT = "all"
A_ROOT = Path("datasets/Feature/HuBERT_quick")
T_ROOT = Path("datasets/Feature/RoBerTa_slow")
D_MODEL = 256
NHEAD = 8
EPOCHS = 12
LR = 1e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.5
TEMPERATURE = 0.07
SEED = 42
LOW_Q = 30
HIGH_Q = 70
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 獨立的 contrastive encoder (跟 ATEI 無關)
# ============================================================
class ContrastiveEncoder(nn.Module):
    """
    純粹學 audio/text segment 對齊。
    跟 paper ATEI 完全無關,只是用來算 inconsistency score。
    """
    def __init__(self, embd_size=D_MODEL, nheads=NHEAD, inp_dim=1024, dropout=0.3):
        super().__init__()
        self.a_in_proj = nn.Sequential(nn.Linear(inp_dim, embd_size),
                                       nn.LayerNorm(embd_size))
        self.t_in_proj = nn.Sequential(nn.Linear(inp_dim, embd_size),
                                       nn.LayerNorm(embd_size))

        a_enc = nn.TransformerEncoderLayer(
            d_model=embd_size, nhead=nheads, batch_first=True,
            dim_feedforward=4 * embd_size, dropout=dropout)
        self.a_enc = nn.TransformerEncoder(a_enc, num_layers=1)

        t_enc = nn.TransformerEncoderLayer(
            d_model=embd_size, nhead=nheads, batch_first=True,
            dim_feedforward=4 * embd_size, dropout=dropout)
        self.t_enc = nn.TransformerEncoder(t_enc, num_layers=1)

        self.a_proj = nn.Linear(embd_size, embd_size)
        self.t_proj = nn.Linear(embd_size, embd_size)

    def forward(self, xa, xt, tMask=None):
        """
        xa: [num_seg, 1024]
        xt: [num_seg, max_L, 1024]
        tMask: [num_seg, max_L]
        return:
          za, zt: [num_seg, D]  L2 normalized
        """
        xa = self.a_in_proj(xa)
        xt = self.t_in_proj(xt)

        ha = self.a_enc(xa.unsqueeze(0)).squeeze(0)        # [num_seg, D]
        ht = self.t_enc(xt, src_key_padding_mask=tMask)    # [num_seg, max_L, D]
        ht = self.mask_mean(ht, tMask)                     # [num_seg, D]

        za = F.normalize(self.a_proj(ha), dim=-1)
        zt = F.normalize(self.t_proj(ht), dim=-1)
        return za, zt

    def mask_mean(self, inp, mask):
        if mask is None:
            return inp.mean(dim=1)
        valid = (~mask).unsqueeze(-1).float()
        return (inp * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)


def info_nce(za, zt, temperature=0.07):
    if za.size(0) < 2:
        return torch.tensor(0.0, device=za.device, requires_grad=True)
    logits = za @ zt.t() / temperature
    labels = torch.arange(za.size(0), device=za.device)
    return (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.t(), labels)) / 2


# ============================================================
# 載入單一 patient 的 feature
# ============================================================
def load_patient(p):
    xa = torch.load(str(A_ROOT / f"{p}_acoustic.pt"), map_location="cpu")
    xt = torch.load(str(T_ROOT / f"{p}_text.pt"), map_location="cpu")
    xa = torch.stack([x.squeeze(0) for x in xa], dim=0)           # [num_seg, 1024]
    xt_list = [x.squeeze(0) for x in xt]                          # list of [L, 1024]
    xt = pad_sequence(xt_list, batch_first=True)                 # [num_seg, max_L, 1024]
    tMask = (xt.sum(dim=-1) == 0)
    return xa, xt, tMask


# ============================================================
# Step 1: 訓 contrastive encoder (只用 train patient,避免 leakage)
# ============================================================
def train_contrastive_encoder(train_ids):
    set_seed(SEED)
    model = ContrastiveEncoder(D_MODEL, NHEAD, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    print(f"\n[Train contrastive encoder] {len(train_ids)} train patients")
    for epoch in range(EPOCHS):
        model.train()
        np.random.shuffle(train_ids)
        totLoss, n = 0.0, 0
        correct, total = 0, 0

        pbar = tqdm(train_ids, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for p in pbar:
            xa, xt, tMask = load_patient(p)
            xa, xt, tMask = xa.to(DEVICE), xt.to(DEVICE), tMask.to(DEVICE)

            opt.zero_grad()
            za, zt = model(xa, xt, tMask)
            loss = info_nce(za, zt, TEMPERATURE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            totLoss += loss.item(); n += 1
            if za.size(0) >= 2:
                with torch.no_grad():
                    pred = (za @ zt.t()).argmax(dim=1)
                    labels = torch.arange(za.size(0), device=DEVICE)
                    correct += (pred == labels).sum().item()
                    total += za.size(0)
            pbar.set_postfix({"loss": totLoss/max(n,1),
                              "align": correct/max(total,1)})

        print(f"Epoch [{epoch+1}/{EPOCHS}] loss={totLoss/max(n,1):.4f} "
              f"align_acc={correct/max(total,1):.4f}")

    return model


# ============================================================
# Step 2: 算每個 patient 的 inconsistency score
# ============================================================
@torch.inference_mode()
def compute_scores(model, all_ids):
    """
    每個 patient 的對齊度 = 對角線 cosine similarity 的平均
    (za[i] 跟 zt[i] 的相似度)
    inconsistency score = 1 - 平均對齊度  → 越大越不一致
    """
    model.eval()
    pids, scores = [], []
    for p in all_ids:
        xa, xt, tMask = load_patient(p)
        xa, xt, tMask = xa.to(DEVICE), xt.to(DEVICE), tMask.to(DEVICE)
        za, zt = model(xa, xt, tMask)            # [num_seg, D]
        # 每個 segment 的 audio/text cosine sim(已 normalize,點積=cosine)
        seg_sim = (za * zt).sum(dim=-1)          # [num_seg]
        align = seg_sim.mean().item()            # patient 平均對齊度
        score = 1.0 - align                      # 越大越不一致
        pids.append(int(p))
        scores.append(float(score))
        print(f"patient{p}: align={align:.4f} score={score:.4f} (n_seg={za.size(0)})")
    return np.array(pids, dtype=np.int64), np.array(scores, dtype=np.float32)

def check_split_distribution(pids, scores, low_th, high_th):
    """檢查 train/val/test 各自的 score 落點,看 val 是不是全擠一端"""
    depMap, train_ids, val_ids, test_ids = get_Split_and_GroundTrue()
    score_map = {int(p): float(s) for p, s in zip(pids, scores)}

    for name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        ss = np.array([score_map[int(p)] for p in ids if int(p) in score_map])
        n_con = int((ss <= low_th).sum())     # label 1
        n_incon = int((ss >= high_th).sum())  # label 0
        n_mid = len(ss) - n_con - n_incon     # 中間,被丟掉
        print(f"\n[{name}] n={len(ss)}  "
              f"score range: {ss.min():.4f} ~ {ss.max():.4f}")
        print(f"  consistency(1, <={low_th:.4f}): {n_con}")
        print(f"  inconsistency(0, >={high_th:.4f}): {n_incon}")
        print(f"  middle(dropped): {n_mid}")
# ============================================================
# Step 3: q30/q70 切法產生 pseudo label (格式同原版)
# ============================================================
def make_pseudo_label(pids, scores, low_q=LOW_Q, high_q=HIGH_Q):
    low_th = np.percentile(scores, low_q)
    high_th = np.percentile(scores, high_q)

    labels = np.full(len(scores), -1, dtype=np.int64)
    labels[scores <= low_th] = 1   # 對齊好 = consistency
    labels[scores >= high_th] = 0  # 對齊差 = inconsistency
    keep = labels != -1

    out_path = f"PseudoLabel_{SPLIT}_contrastive_q{LOW_Q}_{HIGH_Q}.npz"
    np.savez(
        out_path,
        patientIdx=pids[keep],
        label=labels[keep],
        score=scores[keep],
        all_patientIdx=pids,
        all_score=scores,
        low_th=low_th, high_th=high_th,
    )
    print("\nsaved:", out_path)
    print("low_th :", low_th, "high_th:", high_th)
    print("kept   :", int(keep.sum()), "/", len(scores))
    print("label counts (0=incon, 1=con):", np.bincount(labels[keep], minlength=2))
    return out_path, labels, keep


# ============================================================
# 診斷:inconsistent 端是不是 depressed?(關鍵 sanity check)
# ============================================================
def diagnose(pids, scores, depMap):
    order = np.argsort(scores)
    print("\n=== Most consistent (對齊好, 預期=健康) ===")
    for k in order[:10]:
        p = int(pids[k])
        print(f"  p{p} score={scores[k]:.4f} dep_label={depMap.get(p, '?')}")
    print("\n=== Most inconsistent (對齊差, 預期=depressed) ===")
    for k in order[-10:][::-1]:
        p = int(pids[k])
        print(f"  p{p} score={scores[k]:.4f} dep_label={depMap.get(p, '?')}")

    # 量化:inconsistent 端的平均 dep_label vs consistent 端
    n = len(scores) // 3
    con_deps = [depMap[int(pids[k])] for k in order[:n] if int(pids[k]) in depMap]
    incon_deps = [depMap[int(pids[k])] for k in order[-n:] if int(pids[k]) in depMap]
    print(f"\n[診斷] consistent 端平均 dep_label: {np.mean(con_deps):.3f}")
    print(f"[診斷] inconsistent 端平均 dep_label: {np.mean(incon_deps):.3f}")
    print("如果 inconsistent 端的 dep_label 明顯較高,代表 contrastive 抓對了 ECI 信號")

if __name__ == "__main__":
    depMap, train_ids, val_ids, test_ids = get_Split_and_GroundTrue()
    all_ids = train_ids + val_ids + test_ids

    model = train_contrastive_encoder(list(train_ids))

    print("\n[Compute scores for all patients]")
    pids, scores = compute_scores(model, all_ids)

    out_path, labels, keep = make_pseudo_label(pids, scores)

    # ↓ 換成這個有意義的診斷(放在 make_pseudo_label 之後,才有 th)
    low_th = np.percentile(scores, LOW_Q)
    high_th = np.percentile(scores, HIGH_Q)
    check_split_distribution(pids, scores, low_th, high_th)