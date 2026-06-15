"""
Incon_seg_bin_tri.py
====================
在 Incon_seg_bin.py 的基礎上，新增三元分類的 split functions。

切法（對應 DAIC-WOZ PHQ-8）：
  0 = Healthy   : PHQ8_Score < 10
  1 = Mild      : PHQ8_Score 10-14
  2 = Moderate+ : PHQ8_Score >= 15

使用方式：
  from Inconsistency.datasets.Incon_seg_bin_tri import (
      get_Split_and_GroundTrue_tri,
      get_stage1_kfold_tri,
  )
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold

TRAIN_CSV = "datasets/DAICWOZ/train_split_Depression_AVEC2017.csv"
VAL_CSV   = "datasets/DAICWOZ/dev_split_Depression_AVEC2017.csv"


def _phq8_to_tri(score: int) -> int:
    if score < 10:
        return 0   # Healthy
    elif score < 15:
        return 1   # Mild
    else:
        return 2   # Moderate+


def get_Split_and_GroundTrue_tri():
    """
    三元分類版本。
    Returns:
        depMap    : dict {pid -> label}  0=Healthy / 1=Mild / 2=Moderate+
        train_idx : List[int]  官方 train set PIDs
        test_idx  : List[int]  官方 dev set PIDs
    """
    tr  = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)

    depMap = {}
    for _, row in pd.concat([tr, val], ignore_index=True).iterrows():
        depMap[int(row["Participant_ID"])] = _phq8_to_tri(int(row["PHQ8_Score"]))

    train_idx = tr["Participant_ID"].astype(int).tolist()
    test_idx  = val["Participant_ID"].astype(int).tolist()
    return depMap, train_idx, test_idx


def get_stage1_kfold_tri(n_splits: int = 3, seed: int = 42):
    """
    三元 kfold，用於 Stage2_seg_bin_daic_tri.py。
    StratifiedKFold 在三元 label 上做分層，確保每個 fold 的三類比例一致。
    Returns:
        depMap : dict {pid -> tri_label}
        folds  : List[{"fold": int, "train": List[int], "val": List[int]}]
    """
    tr  = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)

    depMap = {}
    for _, row in pd.concat([tr, val], ignore_index=True).iterrows():
        depMap[int(row["Participant_ID"])] = _phq8_to_tri(int(row["PHQ8_Score"]))

    ids    = tr["Participant_ID"].astype(int).tolist()
    labels = [depMap[i] for i in ids]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for i, (tr_i, val_i) in enumerate(skf.split(ids, labels)):
        folds.append({
            "fold":  i,
            "train": [ids[j] for j in tr_i],
            "val":   [ids[j] for j in val_i],
        })
    return depMap, folds