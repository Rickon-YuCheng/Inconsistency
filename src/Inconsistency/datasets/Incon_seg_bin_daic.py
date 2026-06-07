"""
Incon_seg_bin_daic.py
=====================
讀 Incon_seg_bin.py 產的 segment-level emotion label/prob,
做 pair matching 出 ATEI label。只處理 DAIC-WOZ。

--atei_mode hard        (預設) consistent=1/inconsistent=0，hard binary CE
--atei_mode soft_cosine  cosine similarity of soft prob → float 0~1，BCE/MSE
"""
import numpy as np
import argparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--text_source", type=str, default="distilbert",
                   choices=["distilbert", "hownet"])
    p.add_argument("--atei_mode", type=str, default="hard",
                   choices=["hard", "soft_cosine"])
    return p.parse_args()


def load_seg_data(path: str):
    x = np.load(path, allow_pickle=True)
    data = {
        "patientIdx": x["patientIdx"],
        "segIdx": x["segIdx"].astype(np.int64),
        "label": x["label"].astype(np.int64),
    }
    if "soft_prob" in x:
        data["soft_prob"] = x["soft_prob"].astype(np.float32)
    return data


def cosine_similarity_mapped(a: np.ndarray, b: np.ndarray) -> float:
    """cosine similarity, mapped from [-1,1] to [0,1]."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-8:
        return 0.5
    sim = float(np.dot(a, b) / denom)
    return float(np.clip((sim + 1.0) / 2.0, 0.0, 1.0))


def pair_match_hard(A, T):
    A_map = {(p.item() if hasattr(p, "item") else p, int(s)): int(l)
             for p, s, l in zip(A["patientIdx"], A["segIdx"], A["label"])}
    T_map = {(p.item() if hasattr(p, "item") else p, int(s)): int(l)
             for p, s, l in zip(T["patientIdx"], T["segIdx"], T["label"])}
    keys = set(A_map) & set(T_map)

    pid, sid, cons, a_lab, t_lab = [], [], [], [], []
    for k in sorted(keys, key=lambda x: (str(x[0]), x[1])):
        a_l = A_map[k]; t_l = T_map[k]
        pid.append(k[0]); sid.append(k[1])
        cons.append(int(a_l == t_l))
        a_lab.append(a_l); t_lab.append(t_l)
    return (np.array(pid, dtype=object),
            np.array(sid), np.array(cons, dtype=np.int64),
            np.array(a_lab), np.array(t_lab))


def pair_match_soft_cosine(A, T):
    assert "soft_prob" in A and "soft_prob" in T, \
        "soft_prob not found — re-run Incon_seg_bin.py"

    A_label_map = {(p.item() if hasattr(p, "item") else p, int(s)): int(l)
                   for p, s, l in zip(A["patientIdx"], A["segIdx"], A["label"])}
    T_label_map = {(p.item() if hasattr(p, "item") else p, int(s)): int(l)
                   for p, s, l in zip(T["patientIdx"], T["segIdx"], T["label"])}
    A_prob_map = {(p.item() if hasattr(p, "item") else p, int(s)): prob
                  for p, s, prob in zip(A["patientIdx"], A["segIdx"], A["soft_prob"])}
    T_prob_map = {(p.item() if hasattr(p, "item") else p, int(s)): prob
                  for p, s, prob in zip(T["patientIdx"], T["segIdx"], T["soft_prob"])}

    keys = set(A_prob_map) & set(T_prob_map)
    pid, sid, scores, a_lab, t_lab = [], [], [], [], []
    for k in sorted(keys, key=lambda x: (str(x[0]), x[1])):
        sim = cosine_similarity_mapped(A_prob_map[k], T_prob_map[k])
        pid.append(k[0]); sid.append(k[1]); scores.append(sim)
        a_lab.append(A_label_map.get(k, -1))
        t_lab.append(T_label_map.get(k, -1))

    return (np.array(pid, dtype=object),
            np.array(sid), np.array(scores, dtype=np.float32),
            np.array(a_lab), np.array(t_lab))


def make_daic(text_source: str, atei_mode: str):
    print("\n" + "="*60)
    print(f"DAIC pair matching  [mode={atei_mode}]")
    print("="*60)
    t_path = ("HowNet_all_seg_bin.npz" if text_source == "hownet"
              else "DistilBert_all_seg_bin.npz")
    T = load_seg_data(t_path)
    A = load_seg_data("Wav2Vec2_all_seg_bin.npz")

    out = (f"SegPseudoLabel_daic_{text_source}_pair_bin.npz" if atei_mode == "hard"
           else f"SegPseudoLabel_daic_{text_source}_pair_bin_{atei_mode}.npz")

    if atei_mode == "hard":
        pid, sid, label, a_lab, t_lab = pair_match_hard(A, T)
        pid_save = np.array([int(p) for p in pid], dtype=np.int64)
        np.savez(out,
                 patientIdx=pid_save,
                 segIdx=sid.astype(np.int64),
                 label=label,
                 a_label=a_lab.astype(np.int64),
                 t_label=t_lab.astype(np.int64),
                 corpus=np.array(["daic"] * len(pid), dtype=object),
                 atei_mode=np.array(atei_mode))
        print(f"\n[DAIC] saved -> {out}")
        print(f"  total: {len(pid)} segments")
        print(f"  cons dist (0=incon, 1=cons): {np.bincount(label, minlength=2).tolist()}")
        print(f"  audio dist: {np.bincount(a_lab, minlength=3).tolist()}")
        print(f"  text  dist: {np.bincount(t_lab, minlength=3).tolist()}")

    elif atei_mode == "soft_cosine":
        pid, sid, scores, a_lab, t_lab = pair_match_soft_cosine(A, T)
        pid_save = np.array([int(p) for p in pid], dtype=np.int64)
        np.savez(out,
                 patientIdx=pid_save,
                 segIdx=sid.astype(np.int64),
                 label=scores,
                 a_label=a_lab.astype(np.int64),
                 t_label=t_lab.astype(np.int64),
                 corpus=np.array(["daic"] * len(pid), dtype=object),
                 atei_mode=np.array(atei_mode))
        print(f"\n[DAIC] saved -> {out}")
        print(f"  total: {len(pid)} segments")
        print(f"  score stats: min={scores.min():.4f} max={scores.max():.4f} "
              f"mean={scores.mean():.4f} std={scores.std():.4f}")
        print(f"  audio dist: {np.bincount(a_lab[a_lab>=0], minlength=3).tolist()}")
        print(f"  text  dist: {np.bincount(t_lab[t_lab>=0], minlength=3).tolist()}")


def main():
    args = parse_args()
    make_daic(args.text_source, args.atei_mode)
    print("\n✓ Incon_seg_bin_daic.py done")


if __name__ == "__main__":
    main()