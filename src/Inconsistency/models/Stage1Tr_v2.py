"""
Stage1 ATEI training (segment-level).

跟 Stage1Tr_v1.py 的差異
------------------------
v1: patient-level
    一個 patient 的所有 segments 一起做 forward, 取 mean logit 後算 CE,
    一個 patient 一個 label (來自 PseudoLabel_all_distilbert_zdist_q30_70.npz)。

v2: segment-level (對齊 Inconsistency paper)
    每個 (audio_segment, text_segment) pair 是獨立 sample,
    一個 segment 一個 label (來自 SegPseudoLabel_all_distilbert_v2_pair.npz)。

    結果:
        - 訓練資料從 ~107 patients → ~6204 segments
        - 可以真正 batch (B=64), 不用 chunk + checkpoint
        - patient-level split 防 leakage, segment-level loss

Pseudo label 來源:
    SegPseudoLabel_all_distilbert_v2_pair.npz, 來自 inconsistentLabel2_v2_1.py
    內容: patientIdx, segIdx, label (0=incon, 1=cons), a_label, t_label

ATEI model (`atei` class) 不動, 從 Stage1Tr_v1 import。
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb

from Inconsistency.datasets.inconsistentLabel import get_Split_and_GroundTrue
from Inconsistency.models.Stage1Tr_v1 import atei
from Inconsistency.utils import Timer, numpy_random_init, set_seed

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# Defaults (CLI 可覆蓋)
# ============================================================
D_MODEL = 128
NHEAD = 8
LR = 1e-4
EPOCHS = 30
TRANSFORMER_ENC_LAYERS = 1
BATCH_SIZE = 64

PSEUDO_LABEL_FILE = "SegPseudoLabel_all_distilbert_v2_pair.npz"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--nhead", type=int, default=NHEAD)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--enc_layers", type=int, default=TRANSFORMER_ENC_LAYERS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)

    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    parser.add_argument("--pseudo_label_file", type=str, default=PSEUDO_LABEL_FILE)
    parser.add_argument("--save_dir", type=str, default="weights/stage1_seg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--num_workers", type=int, default=4)

    # WeightedRandomSampler 處理 class imbalance (cons:incon ≈ 1:3.3)
    parser.add_argument("--use_sampler", action="store_true",
                        help="WeightedRandomSampler 動態平衡 cons/incon")
    parser.add_argument("--use_class_weight", action="store_true",
                        help="CrossEntropyLoss weight (跟 sampler 二擇一, 不要同時開)")

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str,
                        default="Emotion inconsistency - Stage1 Seg")
    parser.add_argument("--wandb_name", type=str, default=None)

    return parser.parse_args()


# ============================================================
# Dataset
# ============================================================
class SegSampleIndex:
    """
    建立 segment-level sample 列表 (memory-light: 只存 metadata, 不存 feature)。

    每個 sample = (patient_id, segIdx, list_idx_a, list_idx_t, atei_label)
        list_idx_a = audio .pt 裡這個 seg 在 list 的 position
        list_idx_t = text  .pt 裡這個 seg 在 list 的 position

    對應關係要在這裡建好,因為:
        - .pt 內的 list 順序 = csv 內 Participant + dropna 後的順序
        - segIdx = csv 的 row.Index + 2 (跟 audioPreprosessing 一致)
        - list_idx 跟 segIdx 不直接相等,但同一個 patient 內順序是嚴格遞增的
    """

    def __init__(self, patient_ids, pseudo_label_path, ds_root="datasets/DAICWOZ"):
        # 讀 segment-level pseudo label
        pl = np.load(pseudo_label_path)
        seg_pid = pl["patientIdx"].astype(np.int64)
        seg_sid = pl["segIdx"].astype(np.int64)
        seg_lab = pl["label"].astype(np.int64)

        # 為了快速查表, 建 (pid, sid) -> label
        label_map = {}
        for p, s, lab in zip(seg_pid, seg_sid, seg_lab):
            label_map[(int(p), int(s))] = int(lab)

        # 對每個 patient 讀 csv 還原 list 順序
        self.samples = []

        for pid in patient_ids:
            csv_path = Path(ds_root) / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv"
            if not csv_path.exists():
                print(f"[warn] csv not found: {csv_path}, skip patient {pid}")
                continue

            df = pd.read_csv(csv_path, sep="\t")
            df_p = df[df["speaker"] == "Participant"].dropna(subset=["value"]).copy()

            # 依 FeatureExtraction 的順序: csv row 順序 (因為 sorted by row.Index)
            # audio 端: sorted by stem int (= row.Index + 2 = segIdx), 跟 csv 順序一致
            # text 端: 依 df_p iteration 順序 = csv row 順序
            # 所以 audio 和 text 的 list_idx 對同一個 segIdx 是相同的
            for list_idx, row in enumerate(df_p.itertuples()):
                seg_id = row.Index + 2
                key = (pid, seg_id)
                if key not in label_map:
                    continue  # 此 seg 被 PAIR_RULE drop, 不訓練
                lab = label_map[key]

                self.samples.append({
                    "patient_id": pid,
                    "seg_id": seg_id,
                    "list_idx": list_idx,
                    "atei_label": lab,
                })

        print(f"[SegSampleIndex] built {len(self.samples)} samples from {len(patient_ids)} patients")

    def __len__(self):
        return len(self.samples)

    def get_label_counts(self):
        labels = np.array([s["atei_label"] for s in self.samples])
        return np.bincount(labels, minlength=2)


class SegDataset(Dataset):
    """
    Segment-level dataset.

    每個 patient 的 .pt 用 LRU cache, 避免每個 sample 都重讀整個 patient 的 feature。
    在 num_workers>0 + shuffle=True 下, cache 命中率取決於 batch 內 patient 多樣性,
    但因為一個 patient 有幾十到幾百 segments, 同一個 patient 在多個 worker / 多個 batch
    被反覆訪問的機率很高, cache 仍有效。
    """

    def __init__(self, sample_index, a_root="datasets/Feature/HuBERT",
                 t_root="datasets/Feature/RoBerTa", cache_size=8):
        self.samples = sample_index.samples
        self.a_root = Path(a_root)
        self.t_root = Path(t_root)

        # 簡易 LRU: dict + 訪問順序追蹤
        # 不用 functools.lru_cache 因為要在 worker 內部各自維護
        self._cache = {}
        self._cache_order = []
        self._cache_size = cache_size

    def _load_patient(self, pid):
        if pid in self._cache:
            # 更新 LRU 順序
            self._cache_order.remove(pid)
            self._cache_order.append(pid)
            return self._cache[pid]

        a_path = self.a_root / f"{pid}_acoustic.pt"
        t_path = self.t_root / f"{pid}_text.pt"

        xa = torch.load(str(a_path), map_location="cpu", mmap=True)
        xt = torch.load(str(t_path), map_location="cpu", mmap=True)

        # xa[i] shape: [1, T_i, 1024], squeeze batch dim
        xa_list = [x.squeeze(0) for x in xa]
        xt_list = [x.squeeze(0) for x in xt]

        # 容錯: audio 和 text list 長度若不同, 用較短的
        # (理論上應該一樣長, 但有 dropna 等步驟, 保險起見)
        n = min(len(xa_list), len(xt_list))
        if len(xa_list) != len(xt_list):
            print(f"[warn] patient {pid}: audio {len(xa_list)} vs text {len(xt_list)} mismatch, truncate to {n}")
        xa_list = xa_list[:n]
        xt_list = xt_list[:n]

        self._cache[pid] = (xa_list, xt_list)
        self._cache_order.append(pid)

        if len(self._cache_order) > self._cache_size:
            evict_pid = self._cache_order.pop(0)
            del self._cache[evict_pid]

        return self._cache[pid]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        pid = s["patient_id"]
        list_idx = s["list_idx"]

        xa_list, xt_list = self._load_patient(pid)

        # list_idx 應該在範圍內. 若超出 (極少數 csv 有問題), 跳過該 sample
        if list_idx >= len(xa_list) or list_idx >= len(xt_list):
            # 用 idx+1 重試會破壞 dataloader, 直接返回第一個 sample 比較安全
            # 但這應該不會發生, 真發生了 dataset 有 bug
            raise IndexError(
                f"patient {pid} seg {s['seg_id']} list_idx {list_idx} "
                f"out of range (audio {len(xa_list)}, text {len(xt_list)})"
            )

        xa = xa_list[list_idx]   # [T_a, 1024]
        xt = xt_list[list_idx]   # [T_t, 1024]

        return {
            "xa": xa,
            "xt": xt,
            "atei_label": s["atei_label"],
            "patient_id": pid,
            "seg_id": s["seg_id"],
        }


def collate_fn(batch):
    """把一個 batch 的不等長 segments pad 成 [B, T_max, 1024]。"""
    xa_list = [item["xa"] for item in batch]
    xt_list = [item["xt"] for item in batch]
    labels = torch.tensor([item["atei_label"] for item in batch], dtype=torch.long)
    pids = [item["patient_id"] for item in batch]
    sids = [item["seg_id"] for item in batch]

    xa = pad_sequence(xa_list, batch_first=True)  # [B, T_a_max, 1024]
    xt = pad_sequence(xt_list, batch_first=True)  # [B, T_t_max, 1024]

    aMask = (xa.sum(dim=-1) == 0)
    tMask = (xt.sum(dim=-1) == 0)

    return {
        "xa": xa,
        "xt": xt,
        "aMask": aMask,
        "tMask": tMask,
        "labels": labels,
        "patient_ids": pids,
        "seg_ids": sids,
    }


# ============================================================
# Train / Val
# ============================================================
def train_one_epoch(model, loader, criterion, opt, scaler, device, epoch, tot_epochs):
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0

    pbar = tqdm(loader, desc=f"Train epoch {epoch}/{tot_epochs}", unit="batch", leave=False)

    for batch in pbar:
        xa = batch["xa"].to(device, non_blocking=True)
        xt = batch["xt"].to(device, non_blocking=True)
        aMask = batch["aMask"].to(device, non_blocking=True)
        tMask = batch["tMask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        opt.zero_grad()

        with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                            dtype=torch.bfloat16):
            # atei.forward 回傳 (Fc3, oup_logits), 兩個都是 [B, ?]
            _, logits = model(xa, xt, aMask, tMask)  # logits: [B, 2]
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(opt)
        scaler.update()

        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

        pbar.set_postfix({
            "loss": total_loss / max(n, 1),
            "acc": correct / max(n, 1),
        })

    return {
        "loss": total_loss / max(n, 1),
        "acc": correct / max(n, 1),
    }


@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    all_y_true = []
    all_y_pred = []

    pbar = tqdm(loader, desc="Validation", unit="batch", leave=False)

    for batch in pbar:
        xa = batch["xa"].to(device, non_blocking=True)
        xt = batch["xt"].to(device, non_blocking=True)
        aMask = batch["aMask"].to(device, non_blocking=True)
        tMask = batch["tMask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", enabled=(device == "cuda"),
                            dtype=torch.bfloat16):
            _, logits = model(xa, xt, aMask, tMask)
            loss = criterion(logits, labels)

        pred = logits.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total_loss += loss.item() * labels.size(0)
        n += labels.size(0)

        all_y_true.extend(labels.cpu().tolist())
        all_y_pred.extend(pred.cpu().tolist())

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    macro_f1 = f1_score(all_y_true, all_y_pred, labels=[0, 1],
                        average="macro", zero_division=0)

    return {
        "loss": total_loss / max(n, 1),
        "acc": correct / max(n, 1),
        "macro_f1": macro_f1,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "cm": cm,
    }


# ============================================================
# Class balance helpers
# ============================================================
def build_weighted_sampler(sample_index, seed=42):
    """根據 atei_label 的 inverse frequency 做 WeightedRandomSampler。"""
    from torch.utils.data import WeightedRandomSampler

    labels = np.array([s["atei_label"] for s in sample_index.samples])
    counts = np.bincount(labels, minlength=2)

    print(f"[sampler] label counts: {counts}")

    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(f"only one class in train set: {counts}")

    weights_per_class = 1.0 / counts
    sample_weights = weights_per_class[labels]

    g = torch.Generator()
    g.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=g,
    )
    return sampler


def build_class_weight(sample_index, device):
    """根據 atei_label 的 inverse frequency 算 class weight。"""
    labels = np.array([s["atei_label"] for s in sample_index.samples])
    counts = np.bincount(labels, minlength=2)
    n_total = counts.sum()

    # n_total / (n_class * count_per_class), 兩個 class
    weights = n_total / (2.0 * counts)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print(f"[class_weight] counts: {counts}, weights: {weights.tolist()}")
    return weights


# ============================================================
# Main
# ============================================================
def main():
    ARGS = parse_args()

    set_seed(ARGS.seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = Path(ARGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- split ---
    _, train_idx, val_idx, test_idx = get_Split_and_GroundTrue()
    print(f"[split] train patients: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    # --- build sample index ---
    print("\n[1/3] Building sample index...")
    train_index = SegSampleIndex(train_idx, ARGS.pseudo_label_file)
    val_index = SegSampleIndex(val_idx, ARGS.pseudo_label_file)

    print(f"[train] {len(train_index)} samples, label counts: {train_index.get_label_counts()}")
    print(f"[val]   {len(val_index)} samples, label counts: {val_index.get_label_counts()}")

    # --- dataset / loader ---
    print("\n[2/3] Building DataLoader...")
    train_ds = SegDataset(train_index)
    val_ds = SegDataset(val_index)

    g = torch.Generator()
    g.manual_seed(ARGS.seed)

    # class imbalance: 二擇一
    train_sampler = None
    if ARGS.use_sampler:
        train_sampler = build_weighted_sampler(train_index, seed=ARGS.seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=ARGS.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn,
        num_workers=ARGS.num_workers,
        pin_memory=True,
        worker_init_fn=numpy_random_init,
        generator=g,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=ARGS.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=ARGS.num_workers,
        pin_memory=True,
    )

    # --- model ---
    print("\n[3/3] Building model...")
    model = atei(
        embd_size=ARGS.d_model,
        nheads=ARGS.nhead,
        dropout=ARGS.dropout,
        TRANSFORMER_ENC_LAYERS=ARGS.enc_layers,
    ).to(device)

    # CrossEntropy loss
    class_weight = None
    if ARGS.use_class_weight:
        assert not ARGS.use_sampler, "use_sampler 和 use_class_weight 不要同時開"
        class_weight = build_class_weight(train_index, device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weight,
        label_smoothing=ARGS.label_smoothing,
    )

    opt = torch.optim.Adam(
        model.parameters(),
        lr=ARGS.lr,
        weight_decay=ARGS.weight_decay,
    )
    scaler = torch.GradScaler("cuda")

    # --- wandb ---
    if ARGS.use_wandb:
        run_name = ARGS.wandb_name or (
            f"stage1seg_seed{ARGS.seed}_lr{ARGS.lr:.0e}_bs{ARGS.batch_size}_"
            f"d{ARGS.d_model}_l{ARGS.enc_layers}_{run_id}"
        )
        wandb.init(
            project=ARGS.wandb_project,
            name=run_name,
            config={
                "seed": ARGS.seed,
                "d_model": ARGS.d_model,
                "nhead": ARGS.nhead,
                "lr": ARGS.lr,
                "epochs": ARGS.epochs,
                "enc_layers": ARGS.enc_layers,
                "batch_size": ARGS.batch_size,
                "dropout": ARGS.dropout,
                "weight_decay": ARGS.weight_decay,
                "label_smoothing": ARGS.label_smoothing,
                "use_sampler": ARGS.use_sampler,
                "use_class_weight": ARGS.use_class_weight,
                "pseudo_label_file": ARGS.pseudo_label_file,
                "train_samples": len(train_index),
                "val_samples": len(val_index),
                "train_label_counts": train_index.get_label_counts().tolist(),
                "val_label_counts": val_index.get_label_counts().tolist(),
            },
            save_code=True,
        )

    # --- training loop ---
    timer = Timer()
    best_val_f1 = -1.0
    no_improve = 0

    for epoch in range(1, ARGS.epochs + 1):
        print("=" * 80)
        print(f"Epoch [{epoch}/{ARGS.epochs}]")

        tr = train_one_epoch(
            model, train_loader, criterion, opt, scaler, device,
            epoch, ARGS.epochs,
        )
        val_res = validate(model, val_loader, criterion, device)

        print(
            f"[Train] loss={tr['loss']:.4f} acc={tr['acc']:.4f}"
        )
        print(
            f"[Val]   loss={val_res['loss']:.4f} acc={val_res['acc']:.4f} "
            f"macroF1={val_res['macro_f1']:.4f}"
        )
        print(f"[Val] label counts: {np.bincount(val_res['y_true'], minlength=2)}")
        print(f"[Val] pred  counts: {np.bincount(val_res['y_pred'], minlength=2)}")
        print(f"[Val] confusion matrix:\n{val_res['cm']}")
        print(classification_report(
            val_res["y_true"], val_res["y_pred"],
            labels=[0, 1],
            target_names=["inconsistency(0)", "consistency(1)"],
            digits=4, zero_division=0,
        ))

        # save best
        if val_res["macro_f1"] > best_val_f1:
            best_val_f1 = val_res["macro_f1"]
            no_improve = 0

            ckpt_name = (
                f"stage1seg_{run_id}_seed{ARGS.seed}_"
                f"f1{best_val_f1:.4f}_ep{epoch:03d}_"
                f"lr{ARGS.lr:.0e}_d{ARGS.d_model}_l{ARGS.enc_layers}.pt"
            )
            ckpt_path = save_dir / ckpt_name

            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "best_val_f1": best_val_f1,
                "val_acc": val_res["acc"],
                "val_loss": val_res["loss"],
                "val_cm": val_res["cm"],
                "args": vars(ARGS),
                "d_model": ARGS.d_model,
                "nhead": ARGS.nhead,
                "enc_layers": ARGS.enc_layers,
                "lr": ARGS.lr,
                "weight_decay": ARGS.weight_decay,
                "dropout": ARGS.dropout,
                "pseudo_label_file": ARGS.pseudo_label_file,
            }, ckpt_path)

            if ARGS.use_wandb:
                wandb.run.summary["best_val_f1"] = best_val_f1

            print(f"[Save] {best_val_f1:.4f} -> {ckpt_path}")
        else:
            no_improve += 1
            print(f"[EarlyStop] no improvement {no_improve}/{ARGS.patience}")

        if ARGS.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train/loss": tr["loss"],
                "train/acc": tr["acc"],
                "val/loss": val_res["loss"],
                "val/acc": val_res["acc"],
                "val/macro_f1": val_res["macro_f1"],
                "val/pred_0": int(np.bincount(val_res["y_pred"], minlength=2)[0]),
                "val/pred_1": int(np.bincount(val_res["y_pred"], minlength=2)[1]),
                "best/val_macro_f1": best_val_f1,
                "no_improve": no_improve,
                "val/conf_mat": wandb.plot.confusion_matrix(
                    y_true=val_res["y_true"],
                    preds=val_res["y_pred"],
                    class_names=["inconsistency", "consistency"],
                ),
            })

        if no_improve >= ARGS.patience:
            print(f"[EarlyStop] stop at epoch {epoch}, best f1={best_val_f1:.4f}")
            break

    print(f"\nTotal time: {timer}")
    print(f"Best val macro F1: {best_val_f1:.4f}")

    if ARGS.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()