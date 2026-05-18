"""
產生 DAIC-WOZ 的 Segment-level ATEI pseudo label (v2.1)。

對齊 Inconsistency paper (Su et al. 2024) 4.3.2 節：
    segment-level: (audio_seg sentiment == text_seg sentiment) -> 1, else 0

為什麼不直接用 raw t_label == a_label
-----------------------------------
DAIC-WOZ 上的 emotion recognizer 有兩個現實問題:
    1. DistilBERT (text)   prior ≈ [neg=0.27, neu=0.10, pos=0.63]
    2. Wav2Vec2/IEMOCAP    prior ≈ [neg=0.03, neu=0.87, pos=0.10]

也就是 audio classifier 幾乎 90% 都輸出 neutral,
等於這個 modality 在大多 segment 上根本沒給訊號。

如果直接用 raw t==a:
    - consistent 集合裡 (neu, neu) 佔 50% (1464 / 2893):
        雙方都「沒判斷出情緒」, 拿來當 consistency 證據訊息量太低。
    - inconsistent 集合裡 (neu, pos) 佔 66% (9227 / 14002):
        audio 「沒情緒」+ text 「positive」這種日常 small talk 也被當不一致,
        但雙方其實沒有真的衝突, 都只是 baseline 行為。

之前版本 (inconsistentLabel2_v2.py) 用 surprise correction + percentile,
但 score 只有 9 個 distinct 值, percentile 閥值會崩塌(low_th == high_th),
等於沒過濾。

這個版本 (v2.1) 改成明確列出每個 emotion pair 怎麼處理,
保留的都是 affective 上真的有意義的 pair, 丟掉的是 baseline noise。

Pair 規則
---------
保留 consistent (label=1):
    (pos, pos)  兩邊都同意 positive, 雙方都罕見地對齊到 prior 第二名
    (neg, neg)  兩邊都同意 negative, 雙方都罕見地對齊到 prior 最低名

保留 inconsistent (label=0):
    (a=neu, t=neg)   audio 平淡 + text 負向: 典型「我說難過, 但口氣很平」
    (a=neu, t=pos)   -> 丟棄, 太 baseline (見下)
    (a=pos, t=neg)   雙方矛盾
    (a=pos, t=neu)   audio 高情緒 + text 平淡
    (a=neg, t=pos)   雙方矛盾
    (a=neg, t=neu)   audio 高情緒 + text 平淡

丟棄:
    (neu, neu)       雙方都 baseline, 訊息量低
    (neu, pos)       兩個都是各自 prior 高位, 太常見, 不算真衝突

最終分布 (基於 16895 segments):
    consistent      = 1258 + 171  = 1429
    inconsistent    = 3986+323+241+149+76 = 4775
    discarded       = 1464 + 9227 = 10691
    ratio consistent:inconsistent ≈ 1 : 3.3

輸出
----
SegPseudoLabel_<split>_<text_source>_v2_pair.npz:
    patientIdx     : (N_kept,) int64
    segIdx         : (N_kept,) int64
    label          : (N_kept,) int64  1=consistent, 0=inconsistent
    a_label        : (N_kept,) int64  (0=neg, 1=neu, 2=pos)
    t_label        : (N_kept,) int64

    all_patientIdx : (N_all,) int64   過濾前的全部 segment, 給診斷
    all_segIdx     : (N_all,) int64
    all_a_label    : (N_all,) int64
    all_t_label    : (N_all,) int64
    all_kept       : (N_all,) bool    哪些 segment 被保留

    pair_rule      : (3, 3) int8      -1=discard, 0=inconsistent, 1=consistent
                                       index [a_label, t_label]
"""

import numpy as np
from collections import Counter

SPLIT = "all"
TEXT_SOURCE = "distilbert"  # "distilbert" or "hownet"

LABEL_NEG = 0
LABEL_NEU = 1
LABEL_POS = 2

# Pair 規則表:
#   行 = audio label, 列 = text label
#   -1 = discard, 0 = inconsistent, 1 = consistent
PAIR_RULE = np.array(
    [
        # text:  neg, neu, pos
        [1, 0, 0],   # audio=neg
        [0, -1, -1], # audio=neu  -> (neu, neu) 和 (neu, pos) 都丟棄
        [0, 0, 1],   # audio=pos
    ],
    dtype=np.int8,
)


def load_seg_npz(path: str) -> dict:
    x = np.load(path)
    return {
        "patientIdx": x["patientIdx"].astype(np.int64),
        "segIdx": x["segIdx"].astype(np.int64),
        "label": x["label"].astype(np.int64),
    }


def align_audio_text(audio: dict, text: dict) -> dict:
    """用 (patient, segIdx) 做 inner join 對齊 audio 和 text 的 segment label。"""

    def make_key(pid, sid):
        return pid.astype(np.int64) * 1_000_000 + sid.astype(np.int64)

    a_key = make_key(audio["patientIdx"], audio["segIdx"])
    t_key = make_key(text["patientIdx"], text["segIdx"])

    a_map = {int(k): i for i, k in enumerate(a_key)}
    t_map = {int(k): i for i, k in enumerate(t_key)}

    common_keys = sorted(set(a_map.keys()) & set(t_map.keys()))

    n = len(common_keys)
    aligned = {
        "patientIdx": np.empty(n, dtype=np.int64),
        "segIdx": np.empty(n, dtype=np.int64),
        "a_label": np.empty(n, dtype=np.int64),
        "t_label": np.empty(n, dtype=np.int64),
    }

    for i, k in enumerate(common_keys):
        ai = a_map[k]
        ti = t_map[k]
        aligned["patientIdx"][i] = audio["patientIdx"][ai]
        aligned["segIdx"][i] = audio["segIdx"][ai]
        aligned["a_label"][i] = audio["label"][ai]
        aligned["t_label"][i] = text["label"][ti]

    print(f"[align] audio segs: {len(a_key)}, text segs: {len(t_key)}, common: {n}")
    return aligned


def compute_prior(labels: np.ndarray, n_classes: int = 3) -> np.ndarray:
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    return counts / max(counts.sum(), 1.0)


def print_pair_table(a_label: np.ndarray, t_label: np.ndarray, title: str):
    names = ["neg", "neu", "pos"]
    print(f"\n{title}:")
    print("              " + "  ".join(f"{n:>5s}" for n in names))
    for i, na in enumerate(names):
        row = []
        for j in range(3):
            c = int(((a_label == i) & (t_label == j)).sum())
            row.append(f"{c:>5d}")
        print(f"  audio={na}:   " + "  ".join(row))


def apply_pair_rule(a_label: np.ndarray, t_label: np.ndarray):
    """根據 PAIR_RULE 把每個 segment 標成 -1/0/1。"""
    rule_label = PAIR_RULE[a_label, t_label]  # vectorized lookup
    return rule_label


def make_seg_pseudo_label(
    split: str = SPLIT,
    text_source: str = TEXT_SOURCE,
):
    # 1. load
    audio_path = f"Wav2Vec2_seg_{split}_v2.npz"
    if text_source == "distilbert":
        text_path = f"DistilBert_seg_{split}_v2.npz"
    elif text_source == "hownet":
        text_path = f"HowNet_seg_{split}_v2.npz"
    else:
        raise ValueError(f"unknown text_source: {text_source}")

    print(f"[load] audio: {audio_path}")
    print(f"[load] text : {text_path}")
    audio = load_seg_npz(audio_path)
    text = load_seg_npz(text_path)

    # 2. align
    aligned = align_audio_text(audio, text)
    a_label = aligned["a_label"]
    t_label = aligned["t_label"]
    patientIdx = aligned["patientIdx"]
    segIdx = aligned["segIdx"]
    n_total = len(a_label)

    # 3. modality prior (診斷)
    audio_prior = compute_prior(a_label)
    text_prior = compute_prior(t_label)
    print("\n[prior] (0=neg, 1=neu, 2=pos)")
    print(f"  audio: {audio_prior}")
    print(f"  text : {text_prior}")

    # 4. raw stats
    raw_label = (a_label == t_label).astype(np.int64)
    raw_counts = np.bincount(raw_label, minlength=2)
    print(f"\n[raw] consistent   (a==t): {raw_counts[1]} / {n_total} = {raw_counts[1] / n_total:.3f}")
    print(f"[raw] inconsistent (a!=t): {raw_counts[0]} / {n_total} = {raw_counts[0] / n_total:.3f}")

    neu_neu = int(((a_label == 1) & (t_label == 1)).sum())
    print(f"[raw] (neu, neu) pair: {neu_neu} / {raw_counts[1]} of consistent ({neu_neu / max(raw_counts[1], 1):.3f})")

    print_pair_table(a_label, t_label, "[raw] full emotion pair table")

    # 5. apply pair rule
    rule = apply_pair_rule(a_label, t_label)
    keep = rule != -1
    labels = rule[keep].astype(np.int64)

    print("\n[rule] applied PAIR_RULE:")
    print("              text=neg  text=neu  text=pos")
    rule_names = {-1: "  drop", 0: "  incon", 1: "  cons"}
    for i, na in enumerate(["neg", "neu", "pos"]):
        cells = []
        for j in range(3):
            r = int(PAIR_RULE[i, j])
            cells.append(rule_names[r])
        print(f"  audio={na}: " + " ".join(f"{c:>9s}" for c in cells))

    n_kept = int(keep.sum())
    kept_counts = np.bincount(labels, minlength=2)
    print(f"\n[kept] {n_kept} / {n_total} = {n_kept / n_total:.3f}")
    print(f"[kept] inconsistent (label=0): {kept_counts[0]}")
    print(f"[kept] consistent   (label=1): {kept_counts[1]}")
    print(f"[kept] ratio cons : incon = 1 : {kept_counts[0] / max(kept_counts[1], 1):.2f}")

    print_pair_table(a_label[keep], t_label[keep], "[kept] emotion pair table")

    # 6. per-patient stats (確認 Stage1 改 dataset 後每個 patient 都還有資料)
    kept_pid = patientIdx[keep]
    per_patient = Counter(kept_pid.tolist())
    n_patient_kept = len(per_patient)
    n_patient_total = len(set(patientIdx.tolist()))
    print(f"\n[kept] patients with at least 1 kept segment: {n_patient_kept} / {n_patient_total}")
    if n_patient_kept > 0:
        print(f"[kept] avg kept segs / patient: {n_kept / n_patient_kept:.2f}")
        print(f"[kept] min / max kept segs / patient: {min(per_patient.values())} / {max(per_patient.values())}")

    # 6.1 每 patient 的 consistent / inconsistent 分布
    kept_label = labels
    pid_to_cnt = {}
    for pid, lab in zip(kept_pid, kept_label):
        if pid not in pid_to_cnt:
            pid_to_cnt[pid] = [0, 0]
        pid_to_cnt[pid][int(lab)] += 1

    only_inconsistent = sum(1 for v in pid_to_cnt.values() if v[1] == 0)
    only_consistent = sum(1 for v in pid_to_cnt.values() if v[0] == 0)
    print(f"[kept] patients with ONLY inconsistent (no consistent): {only_inconsistent}")
    print(f"[kept] patients with ONLY consistent (no inconsistent): {only_consistent}")

    # 7. save
    out_path = f"SegPseudoLabel_{split}_{text_source}_v2_pair.npz"
    np.savez(
        out_path,
        patientIdx=patientIdx[keep],
        segIdx=segIdx[keep],
        label=labels,
        a_label=a_label[keep],
        t_label=t_label[keep],
        all_patientIdx=patientIdx,
        all_segIdx=segIdx,
        all_a_label=a_label,
        all_t_label=t_label,
        all_kept=keep,
        pair_rule=PAIR_RULE,
    )
    print(f"\n[save] -> {out_path}")


if __name__ == "__main__":
    make_seg_pseudo_label(split=SPLIT, text_source=TEXT_SOURCE)