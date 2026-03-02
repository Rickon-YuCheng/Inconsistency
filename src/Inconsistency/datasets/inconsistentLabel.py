# 提取正負中性標籤
# import pdb
import torch
from transformers import pipeline
import yaml
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

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
            # breakpoint()
            # print(f'into while: {x[j]}')
        # print(pos,' ',neg.' ',neu)
        Dict = {"pos": pos, "neg": neg, "neu": neu}
        key = max(Dict, key=Dict.get)
        # breakpoint()
        print(f"=== patient{i} success -> label: {key}, votes: {Dict}")
        totDict[str(i)] = file_p(str(i), Dict["pos"], Dict["neg"], Dict["neu"])
        poslist.append(Dict["pos"])
        neglist.append(Dict["neg"])
        neulist.append(Dict["neu"])
    return totDict, poslist, neglist, neulist

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


if __name__ == "__main__":
    args = parse_args()
    args.ds_dir = os.path.join(args.ds_dir, args.ds)
    print(args)
    # breakpoint()
    totDict, poslist, neglist, neulist = DISTELBERT(**vars(args))
    # breakpoint()
    draw(totDict, poslist, neglist, neulist)
    # draw2(totDict, poslist, neglist, neulist)
    # WAV2VEC2()
