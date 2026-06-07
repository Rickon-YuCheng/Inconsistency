import numpy as np
import pandas as pd

TRAIN_CSV = "datasets/DAICWOZ/train_split_Depression_AVEC2017.csv"
VAL_CSV   = "datasets/DAICWOZ/dev_split_Depression_AVEC2017.csv"

tr  = pd.read_csv(TRAIN_CSV)
val = pd.read_csv(VAL_CSV)
train_idx = tr["Participant_ID"].astype(int).tolist()
test_idx  = val["Participant_ID"].astype(int).tolist()   # 官方 dev 當 test

P = np.load("PseudoLabel_all_distilbert_zdist_q30_70_bin.npz")
PMap = {int(i): int(l) for i, l in zip(P["patientIdx"], P["label"])}

def dist(name, ids):
    kept = [PMap[p] for p in ids if p in PMap]
    cnt = np.bincount(kept, minlength=2) if kept else np.array([0, 0])
    print(f"{name}: 總 patient={len(ids)}, 有pseudo label(kept)={len(kept)}, "
          f"label[0,1]={cnt.tolist()}")

print("=== bin 版 fold 分配 (tr=官方train, 驗證=官方dev) ===")
dist("TR  (官方train)", train_idx)
dist("VAL (官方dev) ", test_idx)

# 對照: 如果像舊版那樣把 tr+val 打散切, 86 個 kept 會怎麼分
print("\n=== 參考: kept 總數 ===", len(PMap), " label分布:", np.bincount(P["label"], minlength=2).tolist())