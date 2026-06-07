"""
Stage2 dataset / collate (可重用 module)

用法:
    from stage2_data import stage2_dataset, stage2_collate_fn, build_pool_arrays

    # 1) 原本 Stage2 用的 PyTorch Dataset
    ds = stage2_dataset(fold="tr")  # 或 fold="val"/"test"
    # 或自訂 patient id list (給 kfold / LR baseline 用)
    ds = stage2_dataset(patient_ids=[303, 304, ...])

    # 2) 給 sklearn / logistic regression 用的 numpy array
    X, y, pids = build_pool_arrays(patient_ids=[...])
    # X: [N, 2048] (audio pooled mean + text pooled mean concat)
    # y: [N]      (PHQ8_Binary)
"""
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from Inconsistency.datasets.inconsistentLabel_bin import get_Split_and_GroundTrue


# === 路徑常數 ===
A_ROOT = Path("datasets/Feature/HuBERT_pooled_bin")     # audio pooled
T_ROOT = Path("datasets/Feature/RoBerTa_full_bin")      # text token-level
PSEUDO_PATH = "PseudoLabel_all_distilbert_zdist_q30_70_bin.npz"


def _load_pseudo_map(path: str = PSEUDO_PATH):
    P = np.load(path)
    return {int(i): int(l) for i, l in zip(P["patientIdx"], P["label"])}


def _resolve_ids(fold=None, patient_ids=None, cv_split=None):
    """三選一決定要哪些 patient。"""
    if patient_ids is not None:
        return list(patient_ids)
    if cv_split is not None:
        key = {"tr": "train", "val": "val", "test": "val"}[fold]
        return cv_split[key]
    _, tr, te = get_Split_and_GroundTrue()
    return {"tr": tr, "val": te, "test": te}[fold]


class stage2_dataset(Dataset):
    """
    回傳每個 patient:
        xa_list: list of [1024]       (audio pooled, 每句一個 vector)
        xt_list: list of [L_i, 1024]  (text token-level, 每句不同長)
        atei_label: long
        dep_label:  long (PHQ8_Binary)
        patient_id: int

    參數 (擇一):
        fold: "tr" / "val" / "test"  搭配官方 split
        patient_ids: 自訂 patient id 清單 (給 kfold / baseline 用)
        cv_split: dict {"train": [...], "val": [...]} 搭配 fold 取對應那組
    """
    def __init__(self, fold: str = None, patient_ids=None, cv_split=None):
        self.ds = []

        depMap, _, _ = get_Split_and_GroundTrue()
        ids = _resolve_ids(fold=fold, patient_ids=patient_ids, cv_split=cv_split)
        pseudo = _load_pseudo_map()

        for p in ids:
            a_path = A_ROOT / f"{p}_acoustic.pt"
            t_path = T_ROOT / f"{p}_text.pt"
            assert a_path.exists() and t_path.exists(), f"ds error: {p}"

            self.ds.append((p,
                            pseudo.get(p, -1),
                            depMap[p],
                            a_path, t_path))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        patient, pseudo_l, dep_l, a_path, t_path = self.ds[index]
        xa = torch.load(str(a_path), map_location="cpu")
        xt = torch.load(str(t_path), map_location="cpu")

        xa_list = [x.squeeze(0) for x in xa]           # list of [1024]
        xt_list = [x.squeeze(0) for x in xt]           # list of [L_i, 1024]

        return (xa_list, xt_list,
                torch.tensor(pseudo_l, dtype=torch.long),
                torch.tensor(dep_l, dtype=torch.long),
                patient)


def stage2_collate_fn(batch):
    """
    回傳 9 個東西 (跟原 Stage2 完全一致):
        xa_pool:  [B, num_seg, 1024]
        xt_pool:  [B, num_seg, 1024]   (token mean -> segment-level)
        aMask:    [B, num_seg]
        tMask:    [B, num_seg]
        atei_labels: [B]
        dep_labels:  [B]
        patients:   list[int]
        xa_seg_list: list of [num_seg_i, 1024]       (ATEI 用)
        xt_seg_list: list of [num_seg_i, max_L_i, 1024] (ATEI 用,token-level)
    """
    xa_seg_list, xt_seg_list = [], []
    xa_pool_list, xt_pool_list = [], []
    atei_labels, dep_labels, patients = [], [], []

    for xa_i, xt_i, atei_l, dep_l, p in batch:
        xa_pool_list.append(torch.stack(xa_i, dim=0))
        xt_pool_list.append(torch.stack([x.mean(dim=0) for x in xt_i], dim=0))

        xa_seg_list.append(torch.stack(xa_i, dim=0))
        xt_seg_list.append(pad_sequence(xt_i, batch_first=True))

        atei_labels.append(atei_l)
        dep_labels.append(dep_l)
        patients.append(p)

    xa_pool = pad_sequence(xa_pool_list, batch_first=True)
    xt_pool = pad_sequence(xt_pool_list, batch_first=True)
    aMask = (xa_pool.sum(dim=-1) == 0)
    tMask = (xt_pool.sum(dim=-1) == 0)

    return (xa_pool, xt_pool, aMask, tMask,
            torch.stack(atei_labels), torch.stack(dep_labels),
            patients, xa_seg_list, xt_seg_list)


# ============================================================
# 給 sklearn / LR baseline 用: 直接吐 numpy
# ============================================================
def build_pool_arrays(fold: str = None, patient_ids=None):
    """
    把每個 patient 壓成單一向量,給 logistic regression 之類用。

    每個 patient 的特徵 = audio_pool_mean (1024) ++ text_pool_mean (1024) = 2048

    Returns:
        X:    np.ndarray [N, 2048]
        y:    np.ndarray [N]    PHQ8_Binary
        pids: np.ndarray [N]
    """
    depMap, _, _ = get_Split_and_GroundTrue()
    ids = _resolve_ids(fold=fold, patient_ids=patient_ids)

    X, y, pids = [], [], []
    for p in ids:
        a_path = A_ROOT / f"{p}_acoustic.pt"
        t_path = T_ROOT / f"{p}_text.pt"
        if not (a_path.exists() and t_path.exists()):
            continue

        xa = torch.load(str(a_path), map_location="cpu")  # list of [1, 1024]
        xt = torch.load(str(t_path), map_location="cpu")  # list of [1, L, 1024]

        a_mean = torch.stack([x.squeeze(0) for x in xa], dim=0).mean(dim=0)
        # text 先每句 token mean,再 segment mean
        t_mean = torch.stack([x.squeeze(0).mean(dim=0) for x in xt], dim=0).mean(dim=0)

        X.append(torch.cat([a_mean, t_mean], dim=0).numpy())
        y.append(depMap[p])
        pids.append(p)

    return np.array(X), np.array(y), np.array(pids)


if __name__ == "__main__":
    # smoke test
    X, y, pids = build_pool_arrays(fold="tr")
    print("X:", X.shape, "y dist:", Counter(y.tolist()), "pids:", len(pids))
    X, y, pids = build_pool_arrays(fold="test")
    print("X:", X.shape, "y dist:", Counter(y.tolist()), "pids:", len(pids))