"""
提取每句對話的 segment-level 情緒標籤 (pos/neg/neu)。

與原版的差異
------------
原版:每個 patient 把所有句子的情緒投票,輸出 patient-level 的
        (pos_count, neg_count, neu_count) 三個 count。
新版:每句話 (segment) 各自輸出一個情緒 label,輸出 segment-level
        的逐句 label。同時也保留 patient-level 的 count,方便向下
        相容到舊的 z-score pipeline。

對齊 Inconsistency paper (Su et al. 2024) 4.3.2 節:
    "when an acoustic segment and its corresponding textual segment share
    the same sentiment label, the supervised label for this acoustic-textual
    segment pair is set to '1'. Conversely, ... it is assigned '0'."
也就是說 ATEI 的 supervised label 是 segment-level 的。

Segment 索引對齊
----------------
與 audioPreprosessing 一致,segment id 使用 `row.Index + 2` (csv 內每行對應的
wav 檔名為 `{row.Index+2}_Participant.wav`),這樣 text/audio 兩個 modality 的
segment 一定對得起來。

Label 編碼
----------
    0 = neg
    1 = neu
    2 = pos

輸出
----
1. DistilBert_{split}.npz       patient-level count (向下相容)
2. Wav2Vec2_{split}.npz         patient-level count (向下相容)
3. HowNet_{split}.npz           patient-level count (向下相容)
4. DistilBert_seg_{split}_v2.npz  segment-level label (新增)
5. Wav2Vec2_seg_{split}_v2.npz    segment-level label (新增)
6. HowNet_seg_{split}_v2.npz      segment-level label (新增)

segment-level npz 內容
----------------------
    patientIdx : (N_seg,) int64  每個 segment 所屬的 patient id
    segIdx     : (N_seg,) int64  segment 在該 patient 內的 index (= row.Index+2)
    label      : (N_seg,) int64  該 segment 的情緒 label, 0=neg,1=neu,2=pos
"""

import torch
from transformers import pipeline
import yaml
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import torchaudio
from pathlib import Path
from speechbrain.inference.interfaces import foreign_class
import numpy as np
import OpenHowNet
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

SPLIT = "all"  # all,train,val,test
CFG_PATH = "configs/inconsistentLabel.yaml"
TRAIN_CSV = "datasets/DAICWOZ/train_split_Depression_AVEC2017.csv"
VAL_CSV = "datasets/DAICWOZ/dev_split_Depression_AVEC2017.csv"

# label 編碼常數
LABEL_NEG = 0
LABEL_NEU = 1
LABEL_POS = 2


def parse_args():
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.set_defaults(**cfg)
    parser.add_argument("--ds", type=str, help="upper case")
    parser.add_argument(
        "--split",
        type=str,
        default=SPLIT,
        choices=["train", "val", "test", "all"],
    )
    args = parser.parse_args()

    assert args.ds in ["DAICWOZ", "MOSI"], f"Invalid ds name: {args.ds}"

    return args


def get_Split_and_GroundTrue():
    """
    Split tr/val/test and get label(ground truth). 7:2:1

    ### PHQ8:
        **No depression**: 0-4
        **Slight depression**: 5-9
        **Severe depression**: 10-24

    ### Returns:
        **depMap (dict)**: {patient id, PHQ8 label}
        **train_idx**: train patient id, total patient(train_idx)=98
        **val_idx**: val patient id, total patient(val_idx)=29
        **test_idx** test patient id, total patient(test_idx)=15
    """

    def score_to_label(score: int) -> int:
        if 0 <= score <= 4:
            return 0
        elif 5 <= score <= 9:
            return 1
        elif 10 <= score <= 24:
            return 2
        else:
            raise ValueError(f"Unexpected PHQ8 score: {score}")

    df = pd.read_csv(TRAIN_CSV)
    depMap = {}

    for _, row in df.iterrows():
        pid = int(row["Participant_ID"])
        score = int(row["PHQ8_Score"])
        depMap[pid] = score_to_label(score)

    patient_df = df[["Participant_ID", "PHQ8_Score"]].copy()
    patient_df["label"] = patient_df["PHQ8_Score"].apply(score_to_label)

    # 7:2:1
    tr_val_df, test_df = train_test_split(
        patient_df, test_size=0.1, random_state=42, stratify=patient_df["label"]
    )
    tr_df, val_df = train_test_split(
        tr_val_df, test_size=2 / 9, random_state=42, stratify=tr_val_df["label"]
    )

    train_idx = tr_df["Participant_ID"].astype(int).tolist()
    val_idx = val_df["Participant_ID"].astype(int).tolist()
    test_idx = test_df["Participant_ID"].astype(int).tolist()
    return depMap, train_idx, val_idx, test_idx


def get_patient_ids(split: str):
    _, train_idx, val_idx, test_idx = get_Split_and_GroundTrue()

    if split == "train":
        return train_idx
    elif split == "val":
        return val_idx
    elif split == "test":
        return test_idx
    elif split == "all":
        return train_idx + val_idx + test_idx
    else:
        raise ValueError(f"unknown split: {split}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_seg_npz(
    out_path: str,
    seg_patientIdx: list,
    seg_segIdx: list,
    seg_label: list,
):
    """把 segment-level label 存成 npz"""
    np.savez(
        out_path,
        patientIdx=np.array(seg_patientIdx, dtype=np.int64),
        segIdx=np.array(seg_segIdx, dtype=np.int64),
        label=np.array(seg_label, dtype=np.int64),
    )


def save_patient_npz(
    out_path: str,
    idx: list,
    poslist: list,
    neglist: list,
    neulist: list,
):
    """把 patient-level pos/neg/neu count 存成 npz (向下相容舊 pipeline)"""
    np.savez(
        out_path,
        a=poslist,
        b=neglist,
        c=neulist,
        patientIdx=np.array(idx, dtype=np.int64),
    )


def draw(idx, poslist, neglist, neulist, out_path):
    """draw figure"""
    x = idx
    plt.plot(x, poslist, "-", label="pos samples")
    plt.fill_between(x, poslist, alpha=0.8)
    plt.plot(x, neglist, "-", label="neg samples")
    plt.fill_between(x, neglist, alpha=0.8)
    plt.plot(x, neulist, "-", label="neu samples")
    plt.fill_between(x, neulist, alpha=0.8)
    plt.legend(loc="best")
    plt.grid()
    plt.title("fig1, 300~492", fontsize=24)
    plt.xlabel("patient")
    plt.ylabel("emotion distribution")
    plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------------
# DistilBERT (text)
# ---------------------------------------------------------------------------
def DISTILBERT(ds: str, ds_dir: str, device: str, split: str) -> None:
    """
    每句話跑一次 DistilBERT,輸出該句的 pos/neg/neu label。
    同時保留 patient-level 的 pos/neg/neu count。
    """
    print("\n**DistilBert**")

    classifier = pipeline(
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )
    poslist, neglist, neulist, idx = [], [], [], []
    seg_patientIdx, seg_segIdx, seg_label = [], [], []

    patient_ids = get_patient_ids(split)

    for i in patient_ids:
        filePath = f"{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv"

        if not os.path.exists(filePath):
            print(f"PATH: {filePath} does not exist")
            continue

        df = pd.read_csv(filePath, sep="\t")
        df_p = df[df["speaker"] == "Participant"].dropna(subset=["value"]).copy()

        pos = neg = neu = 0
        for row in df_p.itertuples():
            # segment id 與 audioPreprosessing 對齊: row.Index + 2
            seg_id = row.Index + 2
            sentence = row.value

            try:
                Sentence = classifier(sentence, batch_size=24)
                label_str = Sentence[0]["label"]
            except Exception as e:
                # 萬一某句太長或解析錯誤,給 neu 保底
                print(f"(DB) patient{i} seg{seg_id} classify error: {e}, fallback neu")
                label_str = "neutral"

            if label_str == "positive":
                pos += 1
                lab = LABEL_POS
            elif label_str == "negative":
                neg += 1
                lab = LABEL_NEG
            elif label_str == "neutral":
                neu += 1
                lab = LABEL_NEU
            else:
                print(f"(DB) patient{i} seg{seg_id} unknown label: {label_str}, fallback neu")
                neu += 1
                lab = LABEL_NEU

            seg_patientIdx.append(int(i))
            seg_segIdx.append(int(seg_id))
            seg_label.append(int(lab))

        Dict = {"pos": pos, "neg": neg, "neu": neu}
        key = max(Dict, key=Dict.get)

        print(f"=== (DB)patient{i} success -> label: {key}, votes: {Dict}")
        idx.append(i)
        poslist.append(Dict["pos"])
        neglist.append(Dict["neg"])
        neulist.append(Dict["neu"])

    draw(idx, poslist, neglist, neulist, f"DistilBert_{split}.jpg")
    save_patient_npz(f"DistilBert_{split}", idx, poslist, neglist, neulist)
    save_seg_npz(f"DistilBert_seg_{split}_v2", seg_patientIdx, seg_segIdx, seg_label)
    print(f"DistilBert segment-level total: {len(seg_label)} segments")


# ---------------------------------------------------------------------------
# Wav2Vec2 (audio)
# ---------------------------------------------------------------------------
def WAV2VEC2(ds: str, ds_dir: str, device: str, split: str) -> None:
    """
    每個切好的 wav 跑一次 IEMOCAP classifier,輸出該句的 pos/neg/neu label。
        hap        -> pos
        sad / ang  -> neg
        neu        -> neu
    """
    print("\n**WAV2VEC2**")
    sb_Path = Path(".sb_cache")
    sb_Path.mkdir(parents=True, exist_ok=True)
    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        savedir=sb_Path,
        run_opts={"device": device},
    )

    poslist, neglist, neulist, idx = [], [], [], []
    seg_patientIdx, seg_segIdx, seg_label = [], [], []

    patient_ids = get_patient_ids(split)

    for i in patient_ids:
        p_path = Path(f"datasets/DAICWOZ/{i}_P/{i}_aSplits")
        wavFiles = list(p_path.glob("*.wav"))
        if len(wavFiles) == 0:
            print(f"patient{i} no wav splits")
            continue

        pos = neg = neu = 0
        for j in wavFiles:
            # 檔名格式: {seg_id}_Participant.wav (來自 audioPreprosessing)
            stem = j.stem  # e.g. "5_Participant"
            try:
                seg_id = int(stem.split("_")[0])
            except Exception as e:
                print(f"(WV) patient{i} cannot parse seg_id from {stem}: {e}, skip")
                continue

            waveform, sr = torchaudio.load(str(j))
            with torch.no_grad():
                _, _, _, text_lab = classifier.classify_batch(waveform.to(device))

            if text_lab == ["hap"]:
                pos += 1
                lab = LABEL_POS
            elif text_lab in [["sad"], ["ang"]]:
                neg += 1
                lab = LABEL_NEG
            elif text_lab == ["neu"]:
                neu += 1
                lab = LABEL_NEU
            else:
                print(f"(WV) patient{i} seg{seg_id} unknown lab: {text_lab}, fallback neu")
                neu += 1
                lab = LABEL_NEU

            seg_patientIdx.append(int(i))
            seg_segIdx.append(int(seg_id))
            seg_label.append(int(lab))

        Dict = {"pos": pos, "neg": neg, "neu": neu}
        key = max(Dict, key=Dict.get)

        print(f"=== (WV)patient{i} success -> label: {key}, votes: {Dict}")

        idx.append(i)
        poslist.append(Dict["pos"])
        neglist.append(Dict["neg"])
        neulist.append(Dict["neu"])

    draw(idx, poslist, neglist, neulist, f"Wav2Vec2_{split}.jpg")
    save_patient_npz(f"Wav2Vec2_{split}", idx, poslist, neglist, neulist)
    save_seg_npz(f"Wav2Vec2_seg_{split}_v2", seg_patientIdx, seg_segIdx, seg_label)
    print(f"Wav2Vec2 segment-level total: {len(seg_label)} segments")


# ---------------------------------------------------------------------------
# Audio preprocessing (unchanged)
# ---------------------------------------------------------------------------
def audioPreprosessing(ds: str, ds_dir: str, device: str, split: str):
    print("\n**audioPreprocessing**")
    patient_ids = get_patient_ids(split)

    for i in patient_ids:
        csvfilePath = f"{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv"
        audiofilePath = f"{ds_dir}/{i}_P/{i}_AUDIO.wav"

        if not os.path.exists(audiofilePath):
            print(f"PATH: {audiofilePath} does not exist")
            continue

        x = pd.read_csv(csvfilePath, sep="\t")
        x = x[x["speaker"] == "Participant"].dropna(subset=["value"]).copy()

        _, sr = torchaudio.load(audiofilePath)
        fpath = Path("/workspace/datasets/DAICWOZ") / f"{i}_P" / f"{i}_aSplits"
        fpath.mkdir(parents=True, exist_ok=True)

        for row in x.itertuples():
            p = fpath / f"{row.Index + 2}_{row.speaker}.wav"
            if p.exists():
                continue

            s_frame = int(row.start_time * sr)
            n_frame = int((row.stop_time - row.start_time) * sr)

            waveform, _ = torchaudio.load(
                audiofilePath, frame_offset=s_frame, num_frames=n_frame
            )
            torchaudio.save(p, waveform, sr)

        print(f"(aP)patient{i} finish")


# ---------------------------------------------------------------------------
# HowNet (text) - 對齊 paper 規則
# ---------------------------------------------------------------------------
def HOWNET_api(ds: str, ds_dir: str, device: str):
    """Isn't work, because not have emotion"""
    OpenHowNet.download()
    hownet_dict = OpenHowNet.HowNetDict(init_sim=False)
    poslist, neglist, neulist = [], [], []
    _, trDS, _, _ = get_Split_and_GroundTrue()

    for i in trDS:
        filePath = f"{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv"

        if not os.path.exists(filePath):
            print(f"PATH: {filePath} does not exist")
            poslist.append(float("nan"))
            neglist.append(float("nan"))
            neulist.append(float("nan"))
            continue

        x = pd.read_csv(filePath, sep="\t")
        x = x[x.speaker == "Participant"]
        x = x["value"].dropna().tolist()
        breakpoint()
        for j in x:
            x_word = j.split()
            for k in x_word:
                result_list = hownet_dict.get_sense(k)  # noqa
                breakpoint()


def HOWNET_txt():
    HNdict = {}
    curWord = None
    with open("/workspace/datasets/HowNetDict/HowNet.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("W_E="):
                curWord = line.split("=")[1].lower()
                if curWord not in HNdict:
                    HNdict[curWord] = []
            elif line.startswith("S_E=") and curWord:
                sentiment = line.split("=")[1]
                if sentiment:
                    sentiment = sentiment.split("|")[0]
                    HNdict[curWord] = sentiment
    return HNdict


def HOWNET(ds: str, ds_dir: str, device: str, split: str):
    """
    對齊 Inconsistency paper 4.1:
        "If the aggregated sentiment score of a text segment was greater than
        zero, it was classified as positive; if less than zero, it was deemed
        negative. In cases where the total sentiment score of a text segment
        was zero, the sentiment label was assigned as neutral."

    每句話獨立判一個 label。
    Score 計算: pos 字 +1, neg 字 -1, 其他 0;sum > 0 -> pos, < 0 -> neg, == 0 -> neu。
    """
    print("\n**HOWNET**")
    HNdict = HOWNET_txt()

    poslist, neglist, neulist, idx = [], [], [], []
    seg_patientIdx, seg_segIdx, seg_label = [], [], []

    patient_ids = get_patient_ids(split)

    for i in patient_ids:
        filePath = f"{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv"

        if not os.path.exists(filePath):
            print(f"PATH: {filePath} does not exist")
            continue

        df = pd.read_csv(filePath, sep="\t")
        df_p = df[df["speaker"] == "Participant"].dropna(subset=["value"]).copy()

        pos = neg = neu = 0
        for row in df_p.itertuples():
            seg_id = row.Index + 2
            sentence = row.value

            # 對齊 paper: aggregated sentiment score per sentence
            score = 0
            for k in sentence.lower().split():
                if k not in HNdict:
                    continue  # 不在詞典 -> 不貢獻分數
                sentiment = HNdict[k]
                if not sentiment:
                    continue
                if "Plus" in sentiment:
                    score += 1
                elif "Minus" in sentiment:
                    score -= 1
                # else: 不貢獻分數

            if score > 0:
                pos += 1
                lab = LABEL_POS
            elif score < 0:
                neg += 1
                lab = LABEL_NEG
            else:
                neu += 1
                lab = LABEL_NEU

            seg_patientIdx.append(int(i))
            seg_segIdx.append(int(seg_id))
            seg_label.append(int(lab))

        Dict = {"pos": pos, "neg": neg, "neu": neu}
        key = max(Dict, key=Dict.get)
        print(f"=== (HN)patient{i} success -> label: {key}, votes: {Dict}")

        idx.append(i)
        poslist.append(Dict["pos"])
        neglist.append(Dict["neg"])
        neulist.append(Dict["neu"])

    draw(idx, poslist, neglist, neulist, f"HowNet_{split}.jpg")
    save_patient_npz(f"HowNet_{split}", idx, poslist, neglist, neulist)
    save_seg_npz(f"HowNet_seg_{split}_v2", seg_patientIdx, seg_segIdx, seg_label)
    print(f"HowNet segment-level total: {len(seg_label)} segments")


if __name__ == "__main__":
    args = parse_args()
    args.ds_dir = os.path.join(args.ds_dir, args.ds)
    DISTILBERT(**vars(args))
    audioPreprosessing(**vars(args))
    WAV2VEC2(**vars(args))
    HOWNET(**vars(args))