"""
Incon_seg_bin_daic_eatd.py  — 簡化版
====================================
讀 Incon_seg_bin.py 產的「z-score 校正後」segment-level emotion label,
做 pair matching 出 ATEI label (consistent=1, inconsistent=0)。
分開存 DAIC / EATD 兩個 npz。

z-score 校正在 Incon_seg_bin.py 已完成,本檔不再做任何 normalization。
"""
import numpy as np
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text_source", type=str, default="distilbert",
                   choices=["distilbert", "hownet"])
    p.add_argument("--skip_daic", action="store_true")
    p.add_argument("--skip_eatd", action="store_true")
    return p.parse_args()


def load_seg_labels(path: str):
    x = np.load(path, allow_pickle=True)
    return {
        "patientIdx": x["patientIdx"],
        "segIdx": x["segIdx"].astype(np.int64),
        "label": x["label"].astype(np.int64),
    }


def pair_match(A, T):
    """Pair by (patientIdx, segIdx). patientIdx 可 int 或 str。"""
    A_map = {(p.item() if hasattr(p, "item") else p, int(s)): int(l)
             for p, s, l in zip(A["patientIdx"], A["segIdx"], A["label"])}
    T_map = {(p.item() if hasattr(p, "item") else p, int(s)): int(l)
             for p, s, l in zip(T["patientIdx"], T["segIdx"], T["label"])}
    keys = set(A_map) & set(T_map)

    pid, sid, cons, a, t = [], [], [], [], []
    for k in sorted(keys, key=lambda x: (str(x[0]), x[1])):
        a_l = A_map[k]; t_l = T_map[k]
        pid.append(k[0]); sid.append(k[1])
        cons.append(int(a_l == t_l))
        a.append(a_l); t.append(t_l)
    return (np.array(pid, dtype=object),
            np.array(sid), np.array(cons), np.array(a), np.array(t))


def save_pair(corpus: str, text_source: str, pid, sid, cons, a, t):
    out = f"SegPseudoLabel_{corpus}_{text_source}_pair_bin.npz"
    if corpus == "daic":
        pid_save = np.array([int(p) for p in pid], dtype=np.int64)
    else:
        pid_save = pid.astype(object)
    np.savez(out,
             patientIdx=pid_save,
             segIdx=sid.astype(np.int64),
             label=cons.astype(np.int64),
             a_label=a.astype(np.int64),
             t_label=t.astype(np.int64),
             corpus=np.array([corpus] * len(pid), dtype=object))
    print(f"\n[{corpus.upper()}] saved -> {out}")
    print(f"  total: {len(pid)} segments")
    print(f"  cons dist (0=incon, 1=cons): {np.bincount(cons, minlength=2).tolist()}")
    print(f"  audio dist (pos/neg/neu): {np.bincount(a, minlength=3).tolist()}")
    print(f"  text  dist (pos/neg/neu): {np.bincount(t, minlength=3).tolist()}")


def make_daic(text_source):
    print("\n" + "="*60); print("DAIC pair matching"); print("="*60)
    t_path = ("HowNet_all_seg_bin.npz" if text_source == "hownet"
              else "DistilBert_all_seg_bin.npz")
    T = load_seg_labels(t_path)
    A = load_seg_labels("Wav2Vec2_all_seg_bin.npz")
    pid, sid, cons, a, t = pair_match(A, T)
    save_pair("daic", text_source, pid, sid, cons, a, t)


def make_eatd(text_source):
    print("\n" + "="*60); print("EATD pair matching"); print("="*60)
    if text_source == "hownet":
        print("[EATD] hownet 不支援 EATD,改用 distilbert")
        text_source = "distilbert"
    T = load_seg_labels("DistilBert_eatd_seg_bin.npz")
    A = load_seg_labels("Wav2Vec2_eatd_seg_bin.npz")
    pid, sid, cons, a, t = pair_match(A, T)
    save_pair("eatd", text_source, pid, sid, cons, a, t)


def main():
    args = parse_args()
    if not args.skip_daic:
        make_daic(args.text_source)
    if not args.skip_eatd:
        make_eatd(args.text_source)
    print("\n✓ Incon_seg_bin_daic_eatd.py done")


if __name__ == "__main__":
    main()