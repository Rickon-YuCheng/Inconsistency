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
        task="text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        dtype=torch.float16,
        device=0)
    for i in range(300,493):
        filePath=f'{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv'

        assert os.path.exists(filePath), f'PATH: {filePath}'

        x=pd.read_csv(f'{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv',sep='\t')
        x=x['value'].tolist()
        j=0
        pos=neg=neu=0
        while x[j]:
            Sentence=classifier(x[j])
            if Sentence[j]['label']=='POSITIVE': pos+=1
            elif Sentence[j]['label']=='NEGATIVE': neg+=1
            elif Sentence[j]['label']=='NEUTUAL': neu+=1
            else: print('Sentence emotion error, Line44')
            j+=1
        # print(pos,' ',neg.' ',neu)
        breakpoint()
        # with open(f'{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv')
        #     result = classifier("I love using Hugging Face Transformers!")
        # print(result)
        # print(f"polarity: {result[0]['label']}")



if __name__ == '__main__':
    args=parse_args()
    args.ds_dir=os.path.join(args.ds_dir, args.ds)
    print(args)
    breakpoint()
    DISTELBERT(**vars(args))
    # WAV2VEC2()