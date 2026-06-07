""" SSL-based Feature Extraction """
# HuBERT: facebook/hubert-large-ll60k (non-finetuned, 通用性較好,適合非 ASR 下游任務)
# RoBERTa: mental/mental-roberta-large (RoBERTa-large 架構,在心理健康文本上繼續預訓練)
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from pathlib import Path
import torch
import os
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import warnings
from Inconsistency.datasets.inconsistentLabel import get_Split_and_GroundTrue


warnings.filterwarnings("ignore", category=FutureWarning)


# 取第幾層 hidden state
# HuBERT-large 共 24 層 transformer (+1 embedding),hidden_states 有 25 個
# 對非 ASR 下游任務,中層 (~12) 通常包含最豐富的語意/情緒資訊
HUBERT_LAYER = 12


def loadAudio():
    MODEL_NAME = "facebook/hubert-large-ll60k"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = HubertModel.from_pretrained(MODEL_NAME).to("cuda").eval()

    _, trDS, valDS, testDS = get_Split_and_GroundTrue()
    for dataset in [trDS, valDS, testDS]:
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
                ).input_values.to("cuda")

                # run HuBERT
                with torch.inference_mode():
                    X = model(
                        input_values, output_hidden_states=True
                    ).hidden_states[HUBERT_LAYER]
                    Personal_list.append(X.cpu().detach())

            torch.save(
                Personal_list, f"datasets/Feature2/HuBERT/{i}_acoustic.pt"
            )
            print(f"patient{i} successed")


def loadText():
    MODEL_NAME = "mental/mental-roberta-large"
    model = AutoModel.from_pretrained(MODEL_NAME).to("cuda").eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    _, trDS, valDS, testDS = get_Split_and_GroundTrue()
    for dataset in [trDS, valDS, testDS]:
        for i in dataset:
            t_path = Path(f"datasets/DAICWOZ/{i}_P/{i}_TRANSCRIPT.csv")
            Personal_list = []
            if not os.path.exists(t_path):
                print(f"PATH: {t_path} does not exist")
                continue

            x = pd.read_csv(t_path, sep="\t")
            x = x[x.speaker == "Participant"]
            x = x["value"].dropna().tolist()

            for j in x:
                inputs = tokenizer(j, return_tensors="pt", truncation=True, max_length=512).to("cuda")
                with torch.inference_mode():
                    outputs = model(**inputs)
                    Personal_list.append(outputs.last_hidden_state.cpu().detach())

            torch.save(
                Personal_list, f"datasets/Feature2/RoBerTa/{i}_text.pt"
            )
            print(f"patient{i} successed")


if __name__ == "__main__":
    loadAudio()
    loadText()