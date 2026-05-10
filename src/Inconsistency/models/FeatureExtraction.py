""" SSL-based Feature Extraction """
# HuBERT: https://huggingface.co/docs/transformers/model_doc/hubert#transformers.HubertModel
# RoBERTA:
import torchaudio
from transformers import AutoProcessor, HubertModel
from pathlib import Path
import torch
import os
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import warnings
from Inconsistency.datasets.inconsistentLabel import get_Split_and_GroundTrue



warnings.filterwarnings("ignore", category=FutureWarning)



# START = 300
# END = 493
# START=300
# END=302


def loadAudio():
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to("cuda")
    _,trDS,valDS,testDS=get_Split_and_GroundTrue()
    for dataset in [trDS,valDS,testDS]:
        for i in dataset:
            s_path = Path(f"datasets/DAICWOZ/{i}_P/{i}_aSplits")
            Personal_list = []

            if not os.path.exists(s_path):
                print(f"PATH: {s_path} does not exist")
                continue

            for j in sorted(s_path.glob("*.wav"), key=lambda p: int(p.stem.split("_")[0])):
                waveframe, sr = torchaudio.load(str(j))

                # preprocessing for HuBERT
                input_values = processor(
                    waveframe.squeeze(), sampling_rate=sr, return_tensors="pt"
                ).input_values.to("cuda")  # Batch size 1

                # run HuBERT
                with torch.no_grad():
                    X = model(input_values, output_hidden_states=True).hidden_states[
                        12
                    ]  # HuBERT 12th
                    # breakpoint()
                    Personal_list.append(X.cpu().detach())
            torch.save(
                Personal_list, f"datasets/Feature/HuBERT/{i}_acoustic.pt"
            )  # tot 17.9GB
            print(f"patient{i} successed")


def loadText():
    model = AutoModel.from_pretrained("FacebookAI/roberta-large").to("cuda")
    tokenizer=AutoTokenizer.from_pretrained("FacebookAI/roberta-large")
    _,trDS,valDS,testDS=get_Split_and_GroundTrue()
    for dataset in [trDS,valDS,testDS]:
        for i in dataset:
            t_path = Path(f"datasets/DAICWOZ/{i}_P/{i}_TRANSCRIPT.csv")
            Personal_list=[]
            if not os.path.exists(t_path):
                print(f"PATH: {t_path} does not exist")
                continue


            x = pd.read_csv(t_path, sep="\t")
            x = x[x.speaker == "Participant"]
            x = x["value"].dropna().tolist()
            j = 0
            model.eval()
            for j in x:
                inputs = tokenizer(j, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs=model(**inputs)
                    Personal_list.append(outputs.last_hidden_state)
            
            # print(f"(Test) Patient{i} csvLen: {len(Personal_list)} x[0]: {x[0]}, x[-1]: {x[-1]}")
            torch.save(
                Personal_list, f"datasets/Feature/RoBerTa/{i}_text.pt" # Xa = tot patient
            )  # tot 1.2 GB
            print(f"patient{i} successed")
            # breakpoint()




if __name__ == "__main__":
    # loadAudio()
    loadText()
