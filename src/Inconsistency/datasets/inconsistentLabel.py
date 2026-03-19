# 提取正負中性標籤
# import pdb
import torch
from transformers import pipeline
import yaml
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import torchaudio
from pathlib import Path
from speechbrain.inference.interfaces import foreign_class
import numpy as np
import OpenHowNet

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

CFG_PATH = "configs/inconsistentLabel.yaml"

# start=300
# end=302
start = 300
end = 493  # +1


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
    classifier = pipeline(
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
    )
    # totDict = {}
    poslist, neglist, neulist = [], [], []
    for i in range(start, end):
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

        # totDict[str(i)] = file_p(str(i), Dict["pos"], Dict["neg"], Dict["neu"])
        poslist.append(Dict["pos"])
        neglist.append(Dict["neg"])
        neulist.append(Dict["neu"])
    draw(poslist, neglist, neulist)
    np.savez("DistilBert", a=poslist, b=neglist, c=neulist)


# class file_p:
#     """patient file"""

#     def __init__(self, patient: str, pos, neg, neu):
#         self.patient = patient
#         self.pos = pos
#         self.neg = neg
#         self.neu = neu

#     def __repr__(self):
#         return f"{self.pos}(pos) {self.neg}(neg) {self.neu}(neu)"


def draw(poslist, neglist, neulist):
    """draw figure"""
    x = [i for i in range(start, end)]
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
    poslist, neglist, neulist = [], [], []

    for i in range(start, end):
        p_path = Path(f"datasets/DAICWOZ/{i}_P/{i}_aSplits")
        pos = neg = neu = 0
        for j in p_path.glob("*.wav"):
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
        poslist.append(Dict["pos"])
        neglist.append(Dict["neg"])
        neulist.append(Dict["neu"])
    draw(poslist, neglist, neulist)
    np.savez("Wav2Vec2", a=poslist, b=neglist, c=neulist)


def audioPreprosessing(ds: str, ds_dir: str, device: str):
    print("\n**audioPreprocessing**")
    for i in range(start, end):
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
    for i in range(start, end):
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
    HNdict = HOWNET_txt()
    result_list = []
    poslist, neglist, neulist = [], [], []
    for i in range(start, end):
        filePath = f"{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv"
        pos = neg = neu = 0
        if not os.path.exists(filePath):
            print(f"PATH: {filePath} does not exist")
            poslist.append(float("nan"))
            neglist.append(float("nan"))
            neulist.append(float("nan"))
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

        poslist.append(Dict["pos"])
        neglist.append(Dict["neg"])
        neulist.append(Dict["neu"])
        result_list.append(emoLabel)
    print(result_list)
    breakpoint()
    np.savez("HowNet", a=poslist, b=neglist, c=neulist)


if __name__ == "__main__":
    args = parse_args()
    args.ds_dir = os.path.join(args.ds_dir, args.ds)
    # DISTILBERT(**vars(args))
    # audioPreprosessing(**vars(args))
    # WAV2VEC2(**vars(args))

    HOWNET(**vars(args))
