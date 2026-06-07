"""
make_va_pseudo_label.py — 用連續 V/A + ECI 公式產生 pseudo label

取代原本的「z-score count 距離」。

ECI 理論:depressed 患者「文字內容負面 + 語音平淡(低 arousal)」
         → 該有情緒卻平淡 = inconsistency

兩個 inconsistency 成分:
  1. valence 不一致:文字與語音的正負向不符
     score_v = |text_valence - audio_valence|
  2. ECI 平淡:文字越負面,語音 arousal 卻越低 → 越異常
     score_eci = max(0, -text_valence) * (1 - audio_arousal_norm)

綜合 score = score_v + score_eci (可調權重)
score 越大 → 越 inconsistent

沿用 q30/q70 切法。
"""
import numpy as np

SPLIT = "all"
W_V = 1.0     # valence 不一致的權重
W_ECI = 1.0   # ECI 平淡的權重


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def make_va_pseudo_label(split=SPLIT, low_q=30, high_q=70,
                          w_v=W_V, w_eci=W_ECI):
    A = np.load(f"VA_audio_{split}.npz")
    T = np.load(f"VA_text_{split}.npz")

    # 對齊 patient id(兩邊可能順序/數量不同)
    a_map = {int(p): k for k, p in enumerate(A["patientIdx"])}
    t_map = {int(p): k for k, p in enumerate(T["patientIdx"])}
    common = sorted(set(a_map) & set(t_map))
    print(f"common patients: {len(common)}")

    audio_arousal = np.array([A["arousal"][a_map[p]] for p in common])
    audio_valence = np.array([A["valence"][a_map[p]] for p in common])
    text_valence  = np.array([T["valence"][t_map[p]] for p in common])
    pids = np.array(common, dtype=np.int64)

    # --- normalize 到可比較的尺度 ---
    # audio_valence 來自 audeering 大約 0..1,text_valence 是 -1..1
    # 各自 z-score 後比較,扣掉 model bias(跟你原本 z-score 精神一致)
    av_z = zscore(audio_valence)
    tv_z = zscore(text_valence)
    ar_z = zscore(audio_arousal)

    # --- 成分 1: valence 不一致 ---
    score_v = np.abs(tv_z - av_z)

    # --- 成分 2: ECI 平淡 ---
    # 文字越負面(tv_z 越小/負)→ -tv_z 越大
    # 語音 arousal 越低(ar_z 越小)→ (max(ar_z) - ar_z) 越大
    neg_text = np.clip(-tv_z, 0, None)          # 只取文字負面的部分
    low_arousal = ar_z.max() - ar_z             # arousal 越低越大
    low_arousal = low_arousal / (low_arousal.max() + 1e-8)  # 0..1
    score_eci = neg_text * low_arousal

    # --- 綜合 ---
    scores = w_v * score_v + w_eci * score_eci

    low_th = np.percentile(scores, low_q)
    high_th = np.percentile(scores, high_q)

    labels = np.full(len(scores), -1, dtype=np.int64)
    labels[scores <= low_th] = 1   # consistency
    labels[scores >= high_th] = 0  # inconsistency
    keep = labels != -1

    out_path = f"PseudoLabel_{split}_va_eci_q{low_q}_{high_q}.npz"
    np.savez(
        out_path,
        patientIdx=pids[keep],
        label=labels[keep],
        score=scores[keep],
        all_patientIdx=pids,
        all_score=scores,
        low_th=low_th, high_th=high_th,
    )

    print("saved:", out_path)
    print("kept:", int(keep.sum()), "/", len(scores))
    print("label counts:", np.bincount(labels[keep], minlength=2))

    # 診斷:看 consistent/inconsistent 兩端
    order = np.argsort(scores)
    print("\nMost consistent (low score):")
    for k in order[:8]:
        print(f"  p{pids[k]} score={scores[k]:.3f} "
              f"tv={text_valence[k]:.2f} av={audio_valence[k]:.2f} ar={audio_arousal[k]:.2f}")
    print("\nMost inconsistent (high score):")
    for k in order[-8:][::-1]:
        print(f"  p{pids[k]} score={scores[k]:.3f} "
              f"tv={text_valence[k]:.2f} av={audio_valence[k]:.2f} ar={audio_arousal[k]:.2f}")


if __name__ == "__main__":
    make_va_pseudo_label()