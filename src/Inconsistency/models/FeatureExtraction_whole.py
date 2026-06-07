""" 
SSL-based Feature Extraction (Whole version: full + pooled)

抽 4 份 feature:
  - datasets/Feature/HuBERT_full/      frame-level audio,每句 [1, T, 1024]
  - datasets/Feature/HuBERT_pooled/    pool 過,每句 [1, 1024]
  - datasets/Feature/RoBerTa_full/     token-level text,每句 [1, L, 1024]
  - datasets/Feature/RoBerTa_pooled/   pool 過,每句 [1, 1024]

跑一次同時抽完,以後不用重抽。
"""
import torchaudio
from transformers import AutoProcessor, HubertModel
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import torch
import os
import pandas as pd
import warnings
from Inconsistency.datasets.inconsistentLabel import get_Split_and_GroundTrue

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# Audio: HuBERT
# ============================================================
def loadAudio():
    MODEL_NAME = "facebook/hubert-large-ls960-ft"
    HUBERT_LAYER = 12

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = HubertModel.from_pretrained(MODEL_NAME).to("cuda").eval()

    out_full = Path("datasets/Feature/HuBERT_slow")
    out_pooled = Path("datasets/Feature/HuBERT_quick")
    out_full.mkdir(parents=True, exist_ok=True)
    out_pooled.mkdir(parents=True, exist_ok=True)

    _, trDS, valDS, testDS = get_Split_and_GroundTrue()
    for dataset in [trDS, valDS, testDS]:
        for i in dataset:
            s_path = Path(f"datasets/DAICWOZ/{i}_P/{i}_aSplits")
            if not os.path.exists(s_path):
                print(f"PATH: {s_path} does not exist")
                continue

            Full_list = []
            Pooled_list = []

            for j in sorted(s_path.glob("*.wav"),
                            key=lambda p: int(p.stem.split("_")[0])):
                waveframe, sr = torchaudio.load(str(j))

                input_values = processor(
                    waveframe.squeeze(), sampling_rate=sr, return_tensors="pt"
                ).input_values.to("cuda")

                with torch.inference_mode():
                    X = model(
                        input_values, output_hidden_states=True
                    ).hidden_states[HUBERT_LAYER]  # [1, T, 1024]

                    Full_list.append(X.cpu().detach())                 # [1, T, 1024]
                    Pooled_list.append(X.mean(dim=1).cpu().detach())   # [1, 1024]

            torch.save(Full_list,   out_full   / f"{i}_acoustic.pt")
            torch.save(Pooled_list, out_pooled / f"{i}_acoustic.pt")
            print(f"[Audio] patient{i} done ({len(Full_list)} segs)")


# ============================================================
# Text: RoBERTa
# ============================================================
def loadText():
    MODEL_NAME = "FacebookAI/roberta-large"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to("cuda").eval()

    out_full = Path("datasets/Feature/RoBerTa_full")
    out_pooled = Path("datasets/Feature/RoBerTa_pooled")
    out_full.mkdir(parents=True, exist_ok=True)
    out_pooled.mkdir(parents=True, exist_ok=True)

    _, trDS, valDS, testDS = get_Split_and_GroundTrue()
    for dataset in [trDS, valDS, testDS]:
        for i in dataset:
            t_path = Path(f"datasets/DAICWOZ/{i}_P/{i}_TRANSCRIPT.csv")
            if not os.path.exists(t_path):
                print(f"PATH: {t_path} does not exist")
                continue

            x = pd.read_csv(t_path, sep="\t")
            x = x[x.speaker == "Participant"]
            x = x["value"].dropna().tolist()

            Full_list = []
            Pooled_list = []

            for sent in x:
                inputs = tokenizer(sent, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    outputs = model(**inputs)
                    # last_hidden_state: [1, L, 1024]
                    H = outputs.last_hidden_state

                    Full_list.append(H.cpu().detach())                # [1, L, 1024]
                    Pooled_list.append(H.mean(dim=1).cpu().detach())  # [1, 1024]

            torch.save(Full_list,   out_full   / f"{i}_text.pt")
            torch.save(Pooled_list, out_pooled / f"{i}_text.pt")
            print(f"[Text]  patient{i} done ({len(Full_list)} sents)")


if __name__ == "__main__":
    loadAudio()
    loadText()