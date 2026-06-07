"""比原版還好"""

"""
產生 DAIC-WOZ 的 Acoustic-Textual Emotional Inconsistency, ATEI,
高信心 pseudo labels。

這個檔案的目標是根據 text modality 與 audio modality 的情緒型態差異，
產生 patient-level 的一致 / 不一致 pseudo label，供 Stage1Tr 訓練 ATEI
classifier 使用。

背景
----
text 與 audio 使用的情緒辨識器本身有不同偏好。例如 DistilBERT 在文字情緒
上常偏向 positive，而 Wav2Vec2/SpeechBrain 在語音情緒上常偏向 neutral。
因此，若直接比較 raw emotion counts 或直接比較 argmax emotion label，
很容易把模型本身的偏差誤判成 acoustic-textual inconsistency。

為了降低這個問題，本檔案先對每個 modality 各自做 z-score normalization，
讓每位 patient 的情緒分布變成「相對於該 modality 整體分布的偏移」。

方法
----
每位 patient 會有兩組情緒數量向量：

    Text  emotion counts: [pos, neg, neu]
    Audio emotion counts: [pos, neg, neu]

接著分別在 text modality 與 audio modality 內做 z-score：

    T_z = zscore(text emotion counts)
    A_z = zscore(audio emotion counts)

然後計算兩個 z-score emotion vector 的距離：

    score = ||T_z - A_z||_2

score 的意義如下：

    score 越小 -> text/audio 情緒型態越一致
    score 越大 -> text/audio 情緒型態越不一致

Pseudo label 規則
-----------------
本檔案不會強迫所有 patient 都產生 pseudo label，而是只保留 score 分布兩端的
高信心樣本。

以 low_q=30, high_q=70 為例：

    score <= 第 30 百分位數 -> consistency label   = 1
    score >= 第 70 百分位數 -> inconsistency label = 0
    中間 40%                -> 視為模糊樣本，不給 label，不用於 Stage1Tr

這樣可以避免把中間不明確的 patient 硬分成一致或不一致，降低 pseudo label
noise。

輸出
----
輸出的 .npz 檔案只包含被保留下來的高信心 patient：

    patientIdx : 被保留的 patient ids
    label      : pseudo labels，1 = consistency，0 = inconsistency
    score      : 被保留 patient 的 inconsistency scores

另外也會保存診斷資訊：

    all_patientIdx : 過濾前的全部 patient ids
    all_score      : 全部 patient 的 inconsistency scores
    low_th         : consistency threshold
    high_th        : inconsistency threshold

檔名範例
--------
    PseudoLabel_all_distilbert_zdist_q30_70_bin.npz

意思是：

    all        : 使用 all split 產生 pseudo label
    distilbert : text emotion source 使用 DistilBERT
    zdist      : 使用 z-score vector distance
    q30_70     : 使用第 30 與第 70 百分位數作為門檻
    bin        : 二元分類版本
"""
from sklearn.preprocessing import StandardScaler
import numpy as np

SPLIT = "all"
TEXT_SOURCE = "distilbert"  # "distilbert" or "hownet"


def load_emotion_counts(path: str):
    x = np.load(path)
    patient_idx = x["patientIdx"].astype(np.int64)

    # a, b, c = pos, neg, neu
    feats = np.column_stack((x["a"], x["b"], x["c"])).astype(np.float32)
    feats = np.nan_to_num(feats, nan=0.0)

    return feats, patient_idx


def zscore_each_modality(feats: np.ndarray):
    """
    對單一 modality 的 [pos, neg, neu] count 做 z-score。
    目的：扣掉該 emotion recognizer 自己的整體偏好。
    例如 DistilBERT 整體偏 pos、Wav2Vec2 整體偏 neu。
    """
    scaler = StandardScaler()
    z = scaler.fit_transform(feats)

    print("mean:", scaler.mean_)
    print("scale:", scaler.scale_)

    return z


def make_zdist_pseudo_label(
    split: str = SPLIT,
    text_source: str = TEXT_SOURCE,
    low_q: float = 30,
    high_q: float = 70,
):
    if text_source == "hownet":
        T_counts, TpatientIdx = load_emotion_counts(f"HowNet_{split}_bin.npz")
    elif text_source == "distilbert":
        T_counts, TpatientIdx = load_emotion_counts(f"DistilBert_{split}_bin.npz")
    else:
        raise ValueError(f"unknown text_source: {text_source}")

    A_counts, ApatientIdx = load_emotion_counts(f"Wav2Vec2_{split}_bin.npz")

    assert np.array_equal(TpatientIdx, ApatientIdx), "T and A id error"

    print("\n[Text modality]")
    T_z = zscore_each_modality(T_counts)

    print("\n[Audio modality]")
    A_z = zscore_each_modality(A_counts)

    # z-score vector distance
    # score 越大：越不一致
    # score 越小：越一致
    scores = np.linalg.norm(T_z - A_z, axis=1)

    low_th = np.percentile(scores, low_q)
    high_th = np.percentile(scores, high_q)

    labels = np.full(len(scores), -1, dtype=np.int64)

    # low distance = consistency
    labels[scores <= low_th] = 1

    # high distance = inconsistency
    labels[scores >= high_th] = 0

    keep = labels != -1

    out_path = f"PseudoLabel_{split}_{text_source}_zdist_q{low_q}_{high_q}_bin.npz"

    np.savez(
        out_path,
        patientIdx=TpatientIdx[keep],
        label=labels[keep],
        score=scores[keep],
        all_patientIdx=TpatientIdx,
        all_score=scores,
        low_th=low_th,
        high_th=high_th,
    )

    print("\nsaved:", out_path)
    print("low_th :", low_th)
    print("high_th:", high_th)
    print("kept   :", int(keep.sum()), "/", len(scores))
    print("label counts:", np.bincount(labels[keep], minlength=2))

    # 額外印出前幾個檢查
    order = np.argsort(scores)
    print("\nMost consistent patients:")
    for idx in order[:10]:
        print(int(TpatientIdx[idx]), "score:", float(scores[idx]), "label:", int(labels[idx]))

    print("\nMost inconsistent patients:")
    for idx in order[-10:][::-1]:
        print(int(TpatientIdx[idx]), "score:", float(scores[idx]), "label:", int(labels[idx]))


if __name__ == "__main__":
    make_zdist_pseudo_label()