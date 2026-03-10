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

CFG_PATH = "configs/inconsistentLabel.yaml"

# start=313
# end=324
start = 300
end = 492


def parse_args():
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help="upper case")
    parser.set_defaults(**cfg)
    args = parser.parse_args()

    assert args.ds in ["DAICWOZ", "MOSI"], f"Invalid ds name: {args.ds}"

    return args


def DISTELBERT(ds: str, ds_dir: str, device: str) -> None:
    classifier = pipeline(
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        return_all_scores=True,
        dtype=torch.float16,
        device=0,
    )
    totDict = {}
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
        # breakpoint()
        x = x["value"].dropna().tolist()
        # print(x)
        j = 0
        # breakpoint()
        pos = neg = neu = 0
        # breakpoint()
        for j in x:
            # print(f'cur sentence: {j}')
            Sentence = classifier(j)
            # breakpoint()
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

        print(f"=== patient{i} success -> label: {key}, votes: {Dict}")

        totDict[str(i)] = file_p(str(i), Dict["pos"], Dict["neg"], Dict["neu"])
        poslist.append(Dict["pos"])
        neglist.append(Dict["neg"])
        neulist.append(Dict["neu"])
    draw(totDict, poslist, neglist, neulist)

    # breakpoint()
    # with open(f'{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv')
    #     result = classifier("I love using Hugging Face Transformers!")
    # print(result)
    # print(f"polarity: {result[0]['label']}")


class file_p:
    """patient file"""

    def __init__(self, patient: str, pos, neg, neu):
        self.patient = patient
        self.pos = pos
        self.neg = neg
        self.neu = neu

    def __repr__(self):
        return f"{self.pos}(pos) {self.neg}(neg) {self.neu}(neu)"


def draw(totDict, poslist, neglist, neulist):
    """draw figure"""
    x = [i for i in range(start, end+1)]
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
    '''sb -> Speech brain'''
    sb_Path=Path(".sb_cache")
    sb_Path.mkdir(parents=True,exist_ok=True)
    classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                               pymodule_file="custom_interface.py", 
                               classname="CustomEncoderWav2vec2Classifier",
                               savedir=sb_Path,
                               run_opts={'device':device})
    totDict = {}
    poslist, neglist, neulist = [], [], []

    for i in range(start,end+1):
        p_path=Path(f"datasets/DAICWOZ/{i}_P/{i}_aSplits")
        pos=neg=neu=0
        for j in p_path.glob("*.wav"):
            waveform,sr=torchaudio.load(str(j))
            with torch.no_grad():
                _,_,_, text_lab = classifier.classify_batch(waveform.to(device))
            if text_lab==['hap']:
                pos+=1
            elif text_lab in [['sad'],['ang']]:
                neg+=1
            elif text_lab==['neu']:
                neu+=1
            else:
                print(f"something error {text_lab}")
        Dict = {"pos": pos, "neg": neg, "neu": neu}
        key = max(Dict, key=Dict.get)

        print(f"=== patient{i} success -> label: {key}, votes: {Dict}")

        totDict[str(i)] = file_p(str(i), Dict["pos"], Dict["neg"], Dict["neu"])
        poslist.append(Dict["pos"])
        neglist.append(Dict["neg"])
        neulist.append(Dict["neu"])
    draw(totDict, poslist, neglist, neulist)

def audioPreprosessing(ds: str, ds_dir: str, device: str):
    for i in range(start, end+1):
        csvfilePath = f"{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv"
        audiofilePath = f"{ds_dir}/{i}_P/{i}_AUDIO.wav"

        if not os.path.exists(audiofilePath):
            print(f"PATH: {audiofilePath} does not exist")
            continue

        # csv processing
        x = pd.read_csv(csvfilePath, sep="\t")
        x = x[(x["speaker"] == "Participant") & (x["value"].notna())].copy()

        # audio processing
        _,sr=torchaudio.load(audiofilePath)
        fpath=Path("/workspace/datasets/DAICWOZ")/f"{i}_P"/f"{i}_aSplits"
        fpath.mkdir(parents=True,exist_ok=True)

        for row in x.itertuples():

            p=fpath / f"{row.Index+2}_{row.speaker}.wav"
            if p.exists(): continue

            s_frame=int(row.start_time*sr)
            n_frame=int((row.stop_time-row.start_time)*sr)

            waveform,_=torchaudio.load(audiofilePath,frame_offset=s_frame,num_frames=n_frame)
            torchaudio.save(p,waveform,sr)

        print(f"patient{i} finish")




if __name__ == "__main__":
    args = parse_args()
    args.ds_dir = os.path.join(args.ds_dir, args.ds)
    # print(args)
    # breakpoint()
    # DISTELBERT(**vars(args))
    # breakpoint()
    # draw2(totDict, poslist, neglist, neulist)
    # audioPreprosessing(**vars(args))
    WAV2VEC2(**vars(args))
