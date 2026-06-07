# 提取正負中性標籤
# import pdb
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

SPLIT = "all" # all,train,test
CFG_PATH = "configs/inconsistentLabel.yaml"
TRAIN_CSV="datasets/DAICWOZ/train_split_Depression_AVEC2017.csv"
VAL_CSV="datasets/DAICWOZ/dev_split_Depression_AVEC2017.csv"

# start=300
# end=302
# start = 300
# end = 493  # +1


def parse_args():
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.set_defaults(**cfg)
    parser.add_argument("--ds", type=str, help="upper case")
    parser.add_argument("--split",type=str,default=SPLIT,choices=["train", "test", "all"],
)
    args = parser.parse_args()

    assert args.ds in ["DAICWOZ", "MOSI"], f"Invalid ds name: {args.ds}"

    return args


def get_Split_and_GroundTrue():
    """ 
    Split train/test and get label(ground truth).
    train = official train split, test = official val(dev) split.

    ### PHQ8_Binary (二元分類):
        **No depression**: 0
        **Depression**: 1

    ### Returns:
        **depMap (dict)**: {patient id, PHQ8_Binary label}
        **train_idx**: train patient id
        **test_idx**: test patient id (use val/dev split as test)
    """
    tr = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)

    depMap = {} # Dict: tr + test, [id: gt_label]

    for _, row in pd.concat([tr, val], ignore_index=True).iterrows():
        pid = int(row["Participant_ID"])
        depMap[pid] = int(row["PHQ8_Binary"])

    train_idx = tr["Participant_ID"].astype(int).tolist()
    test_idx = val["Participant_ID"].astype(int).tolist()

    # === 原本的 tr+val 再切割 (7:2:1) ===
    # df=pd.concat([tr,val], ignore_index=True)
    # patient_df = df[["Participant_ID", "PHQ8_Binary"]]
    # patient_df = patient_df.copy()
    # patient_df["label"] = patient_df["PHQ8_Binary"]
    # tr_val_df, test_df= train_test_split(patient_df, test_size=0.1, random_state=24,stratify=patient_df["label"])
    # tr_df, val_df= train_test_split(tr_val_df, test_size=2/9, random_state=24,stratify=tr_val_df["label"])
    # train_idx = tr_df["Participant_ID"].astype(int).tolist()
    # val_idx = val_df["Participant_ID"].astype(int).tolist()
    # test_idx = test_df["Participant_ID"].astype(int).tolist()
    # =====================================

    return depMap, train_idx, test_idx


def get_stage1_kfold(n_splits: int = 3, seed: int = 42):
    """
    Stage1 專用 k-fold:只在「官方 train」內部做 StratifiedKFold,
    完全不碰官方 dev(留給 Stage2 當 test)。

    小資料量下單一 val 的 f1 抖動極大,用 k-fold 看 mean±std 才可信。

    ### Returns:
        **depMap (dict)**: {patient id, PHQ8_Binary label}
        **folds (list)**: 每個元素 {"fold": i, "train": [...], "val": [...]}
    """
    from sklearn.model_selection import StratifiedKFold

    tr = pd.read_csv(TRAIN_CSV)

    depMap = {}
    for _, row in pd.concat([pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)],
                            ignore_index=True).iterrows():
        depMap[int(row["Participant_ID"])] = int(row["PHQ8_Binary"])

    ids = tr["Participant_ID"].astype(int).tolist()
    labels = tr["PHQ8_Binary"].astype(int).tolist()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for i, (tr_i, val_i) in enumerate(skf.split(ids, labels)):
        folds.append({
            "fold": i,
            "train": [ids[j] for j in tr_i],
            "val": [ids[j] for j in val_i],
        })
    return depMap, folds


def get_stage1_split(seed: int = 42):
    """
    Stage1 專用切分:只在「官方 train」內部做 stratify 8:2,
    切成 Stage1 的 train / val。完全不碰官方 dev。

    目的:官方 dev 完整保留給 Stage2 當 test,避免 Stage1 看過 dev
    造成 Stage2 評估洩漏。同時用 stratify 讓 Stage1 的 train/val
    正負比平衡,避免官方切分本身的偏斜。

    ### Returns:
        **depMap (dict)**: {patient id, PHQ8_Binary label}
        **s1_train_idx**: Stage1 train patient id (官方 train 的 80%)
        **s1_val_idx**:   Stage1 val   patient id (官方 train 的 20%)
    """
    tr = pd.read_csv(TRAIN_CSV)

    depMap = {}
    for _, row in pd.concat([pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)],
                            ignore_index=True).iterrows():
        depMap[int(row["Participant_ID"])] = int(row["PHQ8_Binary"])

    patient_df = tr[["Participant_ID", "PHQ8_Binary"]].copy()
    s1_tr_df, s1_val_df = train_test_split(
        patient_df, test_size=0.2, random_state=seed,
        stratify=patient_df["PHQ8_Binary"],
    )

    s1_train_idx = s1_tr_df["Participant_ID"].astype(int).tolist()
    s1_val_idx = s1_val_df["Participant_ID"].astype(int).tolist()
    return depMap, s1_train_idx, s1_val_idx


def get_patient_ids(split: str):
    _, train_idx, test_idx = get_Split_and_GroundTrue()

    if split == "train":
        return train_idx
    elif split == "test":
        return test_idx
    elif split == "all":
        return train_idx + test_idx
    else:
        raise ValueError(f"unknown split: {split}")




def DISTILBERT(ds: str, ds_dir: str, device: str, split: str) -> None:
    print("\n**DistilBert**")
    
    classifier = pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    poslist, neglist, neulist, idx = [], [], [], []
    patient_ids = get_patient_ids(split)

    # for i in trDS:
    for i in patient_ids:
        filePath = f"{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv"

        if not os.path.exists(filePath):
            print(f"PATH: {filePath} does not exist")
            continue

        x = pd.read_csv(f"{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv", sep="\t")
        x = x[x.speaker == "Participant"]
        x = x["value"].dropna().tolist()
        j = 0
        pos = neg = neu = 0
        for j in x:
            Sentence = classifier(j, batch_size=24)
            if Sentence[0]["label"] == "positive":
                pos += 1
            elif Sentence[0]["label"] == "negative":
                neg += 1
            elif Sentence[0]["label"] == "neutral":
                neu += 1
            else:
                print("Sentence emotion error")

        Dict = {"pos": pos, "neg": neg, "neu": neu}
        key = max(Dict, key=Dict.get)

        print(f"=== (DB)patient{i} success -> label: {key}, votes: {Dict}")
        idx.append(i)
        poslist.append(Dict["pos"])
        neglist.append(Dict["neg"])
        neulist.append(Dict["neu"])
    draw(idx, poslist, neglist, neulist, f"DistilBert_{split}_bin.jpg")
    np.savez(f"DistilBert_{split}_bin", a=poslist, b=neglist, c=neulist, patientIdx=np.array(idx, dtype=np.int64))


def draw(idx, poslist, neglist, neulist, out_path):
    """draw figure"""
    # x = [i for i in trDS]
    x=idx
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


def WAV2VEC2(ds: str, ds_dir: str, device: str, split: str) -> None:
    """sb -> Speech brain"""
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
    # totDict = {}
    poslist, neglist, neulist, idx = [], [], [], []
    patient_ids = get_patient_ids(split)
    # tr_val_DS=trDS+valDS

    for i in patient_ids:
        p_path = Path(f"datasets/DAICWOZ/{i}_P/{i}_aSplits")
        wavFiles = list(p_path.glob("*.wav"))
        if len(wavFiles) == 0:
            print(f"patient{i} no wav splits")
            continue
        pos = neg = neu = 0
        for j in wavFiles:
            waveform, sr = torchaudio.load(str(j))
            with torch.no_grad():
                _, _, _, text_lab = classifier.classify_batch(waveform.to(device))
            if text_lab == ["hap"]:
                pos += 1
            elif text_lab in [["sad"], ["ang"]]:
                neg += 1
            elif text_lab == ["neu"]:
                neu += 1
            else:
                print(f"something error {text_lab}")
        Dict = {"pos": pos, "neg": neg, "neu": neu}
        key = max(Dict, key=Dict.get)

        print(f"=== (WV)patient{i} success -> label: {key}, votes: {Dict}")

        # totDict[str(i)] = file_p(str(i), Dict["pos"], Dict["neg"], Dict["neu"])
        idx.append(i)
        poslist.append(Dict["pos"])
        neglist.append(Dict["neg"])
        neulist.append(Dict["neu"])
    draw(idx, poslist, neglist, neulist, f"Wav2Vec2_{split}_bin.jpg")
    np.savez(f"Wav2Vec2_{split}_bin", a=poslist, b=neglist, c=neulist, patientIdx=np.array(idx, dtype=np.int64))
    # breakpoint()

def audioPreprosessing(ds: str, ds_dir: str, device: str, split: str):
    print("\n**audioPreprocessing**")
    patient_ids = get_patient_ids(split)

    for i in patient_ids:
        csvfilePath = f"{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv"
        audiofilePath = f"{ds_dir}/{i}_P/{i}_AUDIO.wav"

        if not os.path.exists(audiofilePath):
            print(f"PATH: {audiofilePath} does not exist")
            continue

        # csv processing
        x = pd.read_csv(csvfilePath, sep="\t")
        x = x[x["speaker"] == "Participant"].dropna(subset=["value"]).copy()

        # audio processing
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
            torchaudio.save(p, waveform, sr)  # 1024 dim

        print(f"(aP)patient{i} finish")


def HOWNET_api(ds: str, ds_dir: str, device: str):
    """Isn't work, because not have emotion"""
    OpenHowNet.download()
    hownet_dict = OpenHowNet.HowNetDict(init_sim=False)
    poslist, neglist, neulist = [], [], []
    _,trDS,_=get_Split_and_GroundTrue()

    for i in trDS:
        filePath = f"{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv"

        if not os.path.exists(filePath):
            print(f"PATH: {filePath} does not exist")
            poslist.append(float("nan"))
            neglist.append(float("nan"))
            neulist.append(float("nan"))
            continue

        x = pd.read_csv(f"{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv", sep="\t")
        x = x[x.speaker == "Participant"]
        x = x["value"].dropna().tolist()
        breakpoint()
        j = 0
        pos = neg = neu = 0  # noqa
        for j in x:
            x_word = j.split()
            # breakpoint()
            for k in x_word:
                result_list = hownet_dict.get_sense(k)  # noqa
                breakpoint()


def HOWNET_txt():
    HNdict = {}
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
    print("\n**HOWNET**")
    HNdict = HOWNET_txt()
    result_list = []
    poslist, neglist, neulist = [], [], []
    idx=[]
    patient_ids = get_patient_ids(split)

    for i in patient_ids:
        filePath = f"{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv"
        pos = neg = neu = 0
        if not os.path.exists(filePath):
            print(f"PATH: {filePath} does not exist")
            continue

        x = pd.read_csv(f"{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv", sep="\t")
        x = x[x.speaker == "Participant"]
        x = x["value"].dropna().tolist()
        j = 0
        for j in x:
            x_word = j.lower().split()
            for k in x_word:
                if k not in HNdict:
                    neu += 1
                elif "Plus" in HNdict[k]:
                    pos += 1
                elif "Minus" in HNdict[k]:
                    neg += 1
                else:
                    neu += 1

        if pos > neg:
            emoLabel = 0
        elif pos < neg:
            emoLabel = 1
        else:
            emoLabel = 2

        Dict = {"pos": pos, "neg": neg, "neu": neu}
        key = max(Dict, key=Dict.get)
        print(f"=== (HN)patient{i} success -> label: {key}, votes: {Dict}")

        idx.append(i)
        poslist.append(Dict["pos"])
        neglist.append(Dict["neg"])
        neulist.append(Dict["neu"])
        result_list.append(emoLabel)
    # print(result_list)
    # breakpoint()
    draw(idx, poslist, neglist, neulist, f"HowNet_{split}_bin.jpg")
    np.savez(f"HowNet_{split}_bin", a=poslist, b=neglist, c=neulist, patientIdx=np.array(idx,dtype=np.int64))


if __name__ == "__main__":
    args = parse_args()
    args.ds_dir = os.path.join(args.ds_dir, args.ds)
    DISTILBERT(**vars(args))
    audioPreprosessing(**vars(args))
    WAV2VEC2(**vars(args))

    HOWNET(**vars(args))