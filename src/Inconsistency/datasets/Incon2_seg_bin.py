"""
Segment-level ATEI pseudo label (pair-matching, paper-aligned).

按 paper (Su et al. 2024) 第 5 頁的定義:
    當 audio segment 和對應的 text segment 的 sentiment label 相同
        -> consistency  (label = 1)
    當兩者 sentiment label 不同
        -> inconsistency (label = 0)

跟 inconsistentLabel2_bin.py 的差別
-----------------------------------
原版 (patient-level):
    對每位 patient 的 [pos, neg, neu] 計數做 z-score normalization,
    用兩個 modality 的 z-score 向量距離做 inconsistency score,
    再以 q30/70 過濾出高信心 patient。

本檔 (segment-level):
    直接逐 segment 比對 audio_label 和 text_label 是否相同。
    不做 z-score, 不做百分位過濾——paper 沒做這些, 且 segment-level
    比的是 argmax label 是否相等, bias 已內建在 emotion recognizer
    的決策邊界, 不需要 modality-level 校正。

輸出
----
    SegPseudoLabel_{split}_{text_source}_pair_bin.npz
    欄位:
        patientIdx : [N]  segment 所屬 patient id
        segIdx     : [N]  segment id (跟 audioPreprosessing 一致)
        label      : [N]  0 = inconsistent, 1 = consistent
        a_label    : [N]  audio  side argmax emotion (0=pos/1=neg/2=neu)
        t_label    : [N]  text   side argmax emotion (0=pos/1=neg/2=neu)
"""
import numpy as np
from collections import Counter


SPLIT = "all"
TEXT_SOURCE = "distilbert"   # "distilbert" or "hownet"


def load_seg_labels(path: str):
    """讀 Incon_seg_bin.py 產的 segment-level emotion npz。"""
    x = np.load(path)
    return {
        "patientIdx": x["patientIdx"].astype(np.int64),
        "segIdx": x["segIdx"].astype(np.int64),
        "label": x["label"].astype(np.int64),
    }


def make_pair_pseudo_label(split: str = SPLIT,
                           text_source: str = TEXT_SOURCE):
    if text_source == "hownet":
        T = load_seg_labels(f"HowNet_{split}_seg_bin.npz")
    elif text_source == "distilbert":
        T = load_seg_labels(f"DistilBert_{split}_seg_bin.npz")
    else:
        raise ValueError(f"unknown text_source: {text_source}")

    A = load_seg_labels(f"Wav2Vec2_{split}_seg_bin.npz")

    # 用 (patient, seg) 當 key 配對
    A_map = {(int(p), int(s)): int(l)
             for p, s, l in zip(A["patientIdx"], A["segIdx"], A["label"])}
    T_map = {(int(p), int(s)): int(l)
             for p, s, l in zip(T["patientIdx"], T["segIdx"], T["label"])}

    keys_a = set(A_map.keys())
    keys_t = set(T_map.keys())
    keys_paired = keys_a & keys_t
    only_a = keys_a - keys_t
    only_t = keys_t - keys_a

    print(f"[pair] audio-only segments: {len(only_a)}")
    print(f"[pair] text-only  segments: {len(only_t)}")
    print(f"[pair] paired     segments: {len(keys_paired)}")

    pid_arr, sid_arr, lab_arr, a_arr, t_arr = [], [], [], [], []

    # 排序讓輸出順序穩定 (依 patient, 然後 seg)
    for key in sorted(keys_paired):
        pid, sid = key
        a_lab = A_map[key]
        t_lab = T_map[key]
        cons = int(a_lab == t_lab)   # 1 = consistent, 0 = inconsistent

        pid_arr.append(pid)
        sid_arr.append(sid)
        lab_arr.append(cons)
        a_arr.append(a_lab)
        t_arr.append(t_lab)

    pid_arr = np.array(pid_arr, dtype=np.int64)
    sid_arr = np.array(sid_arr, dtype=np.int64)
    lab_arr = np.array(lab_arr, dtype=np.int64)
    a_arr = np.array(a_arr, dtype=np.int64)
    t_arr = np.array(t_arr, dtype=np.int64)

    out_path = f"SegPseudoLabel_{split}_{text_source}_pair_bin.npz"
    np.savez(
        out_path,
        patientIdx=pid_arr,
        segIdx=sid_arr,
        label=lab_arr,
        a_label=a_arr,
        t_label=t_arr,
    )

    # 統計
    print(f"\nsaved: {out_path}")
    print(f"total paired segments: {len(lab_arr)}")
    print(f"label counts (0=incon, 1=cons): {np.bincount(lab_arr, minlength=2).tolist()}")
    cons_ratio = lab_arr.mean() if len(lab_arr) else 0.0
    print(f"consistency ratio: {cons_ratio:.4f}")

    n_patients = len(set(pid_arr.tolist()))
    print(f"patients covered: {n_patients}")

    # audio / text 各自的 label 分布
    print(f"\naudio emotion dist (0=pos/1=neg/2=neu): "
          f"{np.bincount(a_arr, minlength=3).tolist()}")
    print(f"text  emotion dist (0=pos/1=neg/2=neu): "
          f"{np.bincount(t_arr, minlength=3).tolist()}")

    # (a_label, t_label) 共現矩陣, 對角 = 一致
    print("\nco-occurrence (rows=audio, cols=text):")
    co = np.zeros((3, 3), dtype=np.int64)
    for a, t in zip(a_arr, t_arr):
        co[a, t] += 1
    print("        pos     neg     neu")
    names = ["pos", "neg", "neu"]
    for r in range(3):
        print(f"{names[r]:>4}  " + "  ".join(f"{co[r, c]:>6d}" for c in range(3)))


if __name__ == "__main__":
    make_pair_pseudo_label()