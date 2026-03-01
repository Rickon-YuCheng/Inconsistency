# 提取正負中性標籤
import pdb
import torch
from transformers import pipeline
import yaml
import argparse
import os
import pandas as pd
CFG_PATH='configs/inconsistentLabel.yaml'

def parse_args():
    with open(CFG_PATH, "r") as f:
        cfg=yaml.safe_load(f)

    parser=argparse.ArgumentParser()
    parser.add_argument("--ds", type=str, help='upper case')
    parser.set_defaults(**cfg)
    args=parser.parse_args()

    assert args.ds in ['DAICWOZ', 'MOSI'],f'Invalid ds name: {args.ds}'

    return args

def DISTELBERT(ds: str, ds_dir: str, device: str) -> None:
    classifier = pipeline(
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        return_all_scores=True,
        dtype=torch.float16,
        device=0
    )
    for i in range(300,493):
    # for i in range(313,324):
        filePath=f'{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv'

        if not os.path.exists(filePath):
            print(f'PATH: {filePath} does not exist')
            continue

        x=pd.read_csv(f'{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv',sep='\t')
        x=x[x.speaker == 'Participant']
        # breakpoint()
        x=x['value'].dropna().tolist()
        # print(x)
        j=0
        # breakpoint()
        pos=neg=neu=0
        # breakpoint()
        for j in x:
            # print(f'cur sentence: {j}')
            Sentence=classifier(j)
            # breakpoint()
            if Sentence[0]['label']=='positive': pos+=1
            elif Sentence[0]['label']=='negative': neg+=1
            elif Sentence[0]['label']=='neutral': neu+=1
            else: print('Sentence emotion error')
            # breakpoint()
            # print(f'into while: {x[j]}')
        # print(pos,' ',neg.' ',neu)
        Dict={'pos':pos, 'neg':neg, 'neu': neu}
        key=max(Dict, key=Dict.get)
        # breakpoint()
        tot={f'{i}':Dict[key]}
        print(f'=== patient{i} success -> label: {key}, votes: {Dict}')
        # breakpoint()
        # with open(f'{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv')
        #     result = classifier("I love using Hugging Face Transformers!")
        # print(result)
        # print(f"polarity: {result[0]['label']}")



if __name__ == '__main__':
    args=parse_args()
    args.ds_dir=os.path.join(args.ds_dir, args.ds)
    print(args)
    # breakpoint()
    DISTELBERT(**vars(args))
    # WAV2VEC2()