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

CFG_PATH = "configs/inconsistentLabel.yaml"
TRAIN_CSV="datasets/DAICWOZ/train_split_Depression_AVEC2017.csv"
VAL_CSV="datasets/DAICWOZ/dev_split_Depression_AVEC2017.csv"

# start=300
# end=302
# start = 300
# end = 493  # +1

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
        if 0 <= score <= 4: return 0
        elif 5 <= score <= 9: return 1
        elif 10 <= score <= 24: return 2
        else: raise ValueError(f"Unexpected PHQ8 score: {score}")

    tr = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    df=pd.concat([tr,val], ignore_index=True)
    depMap = {} # Dict: tr + test, [id: gt_label]

    for _, row in df.iterrows():
        pid = int(row["Participant_ID"])
        score = int(row["PHQ8_Score"])
        depMap[pid] = score_to_label(score) # [303: 0, .., 491: 1, 302: 0, .., 492: 0]
    
    patient_df = df[["Participant_ID", "PHQ8_Score"]]
    # 7:2:1
    tr_val_df, test_df= train_test_split(patient_df, test_size=0.1, random_state=42)
    tr_df, val_df= train_test_split(tr_val_df, test_size=2/9, random_state=42)

    train_idx = tr_df["Participant_ID"].astype(int).tolist() # len: 107, [303,304,..]
    val_idx = val_df["Participant_ID"].astype(int).tolist()
    test_idx = test_df["Participant_ID"].astype(int).tolist() # len: 35 [302,307,..]
    return depMap, train_idx, val_idx, test_idx

def parse_args():
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="upper case")
    parser.set_defaults(**cfg)
    args = parser.parse_args()

    assert args.ds in ["DAICWOZ", "MOSI"], f"Invalid ds name: {args.ds}"

    return args


def DISTILBERT(ds: str, ds_dir: str, device: str) -> None:
    print("\n**DistilBert**")
    
    classifier = pipeline(model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")
    poslist, neglist, neulist, idx = [], [], [], []
    _,trDS,_,_=get_Split_and_GroundTrue()

    # for i in trDS:
    for i in trDS:
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
    draw(idx, poslist, neglist, neulist)
    np.savez("DistilBert", a=poslist, b=neglist, c=neulist, patientIdx=np.array(idx, dtype=np.int64))


def draw(idx, poslist, neglist, neulist):
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
    plt.savefig("test.jpg")
    plt.close()


def WAV2VEC2(ds: str, ds_dir: str, device: str) -> None:
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
    _,trDS,_,_=get_Split_and_GroundTrue()
    # tr_val_DS=trDS+valDS

    for i in trDS:
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
    draw(idx, poslist, neglist, neulist)
    np.savez("Wav2Vec2", a=poslist, b=neglist, c=neulist, patientIdx=np.array(idx, dtype=np.int64))
    breakpoint()

def audioPreprosessing(ds: str, ds_dir: str, device: str):
    print("\n**audioPreprocessing**")
    _,trDS,_,_=get_Split_and_GroundTrue()

    for i in trDS:
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
    _,trDS,_,_=get_Split_and_GroundTrue()

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


def HOWNET(ds: str, ds_dir: str, device: str):
    print("\n**HOWNET**")
    HNdict = HOWNET_txt()
    result_list = []
    poslist, neglist, neulist = [], [], []
    idx=[]
    _,trDS,_,_=get_Split_and_GroundTrue()

    for i in trDS:
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
    np.savez("HowNet", a=poslist, b=neglist, c=neulist, patientIdx=np.array(idx,dtype=np.int64))


if __name__ == "__main__":
    args = parse_args()
    args.ds_dir = os.path.join(args.ds_dir, args.ds)
    DISTILBERT(**vars(args))
    audioPreprosessing(**vars(args))
    WAV2VEC2(**vars(args))

    HOWNET(**vars(args))
