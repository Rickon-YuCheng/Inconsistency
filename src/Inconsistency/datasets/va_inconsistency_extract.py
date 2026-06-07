"""
extract_va.py — 抽取連續 valence / arousal,取代原本的 pos/neg/neu count

Audio: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
       → 每段 audio 輸出 [arousal, dominance, valence] (連續值 ~0..1)
Text:  cardiffnlp/twitter-roberta-base-sentiment-latest
       → 每句輸出 P(neg), P(neu), P(pos);valence = P(pos) - P(neg) (連續 -1..1)

每個 patient:
  audio_arousal_mean, audio_valence_mean
  text_valence_mean

存成:
  VA_audio_{split}.npz : patientIdx, arousal, valence
  VA_text_{split}.npz  : patientIdx, valence
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import pandas as pd
from pathlib import Path
from transformers import (
    Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2PreTrainedModel,
    AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
)
import warnings
warnings.filterwarnings("ignore")

# 沿用你原本的 split 函式
from Inconsistency.datasets.inconsistentLabel import get_patient_ids

DS_DIR = "datasets/DAICWOZ"
SPLIT = "all"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Audio V/A model (audeering)
# ============================================================
# audeering model 需要自訂 head,官方給的結構如下
class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden = outputs[0]
        hidden = torch.mean(hidden, dim=1)   # pooling
        logits = self.classifier(hidden)     # [B, 3] = arousal, dominance, valence
        return logits


def extract_audio_va(split: str):
    print("\n** Audio V/A extraction (audeering) **")
    model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name).to(DEVICE).eval()

    patient_ids = get_patient_ids(split)
    idx, arousal_list, valence_list = [], [], []

    for i in patient_ids:
        p_path = Path(f"{DS_DIR}/{i}_P/{i}_aSplits")
        wavFiles = list(p_path.glob("*.wav"))
        if len(wavFiles) == 0:
            print(f"patient{i} no wav splits")
            continue

        a_vals, v_vals = [], []
        for j in wavFiles:
            waveform, sr = torchaudio.load(str(j))
            # audeering model 要 16kHz mono
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            inputs = processor(waveform.squeeze(0).numpy(),
                               sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                out = model(inputs.input_values.to(DEVICE))  # [1, 3]
            arousal, dominance, valence = out[0].cpu().numpy()
            a_vals.append(float(arousal))
            v_vals.append(float(valence))

        idx.append(i)
        arousal_list.append(np.mean(a_vals))
        valence_list.append(np.mean(v_vals))
        print(f"(audio) patient{i} -> arousal={np.mean(a_vals):.3f}, "
              f"valence={np.mean(v_vals):.3f}")

    np.savez(f"VA_audio_{split}",
             patientIdx=np.array(idx, dtype=np.int64),
             arousal=np.array(arousal_list, dtype=np.float32),
             valence=np.array(valence_list, dtype=np.float32))
    print(f"saved VA_audio_{split}.npz")


# ============================================================
# Text valence model (cardiffnlp)
# ============================================================
def extract_text_valence(split: str):
    print("\n** Text valence extraction (cardiffnlp) **")
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(DEVICE).eval()
    # label order: 0=negative, 1=neutral, 2=positive

    patient_ids = get_patient_ids(split)
    idx, valence_list = [], []

    for i in patient_ids:
        filePath = f"{DS_DIR}/{i}_P/{i}_TRANSCRIPT.csv"
        if not os.path.exists(filePath):
            print(f"patient{i} no transcript")
            continue

        x = pd.read_csv(filePath, sep="\t")
        x = x[x.speaker == "Participant"]["value"].dropna().tolist()
        if len(x) == 0:
            continue

        v_vals = []
        for sent in x:
            enc = tokenizer(sent, return_tensors="pt", truncation=True,
                            max_length=128).to(DEVICE)
            with torch.no_grad():
                logits = model(**enc).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            # valence = P(pos) - P(neg)
            valence = float(probs[2] - probs[0])
            v_vals.append(valence)

        idx.append(i)
        valence_list.append(np.mean(v_vals))
        print(f"(text) patient{i} -> valence={np.mean(v_vals):.3f}")

    np.savez(f"VA_text_{split}",
             patientIdx=np.array(idx, dtype=np.int64),
             valence=np.array(valence_list, dtype=np.float32))
    print(f"saved VA_text_{split}.npz")


if __name__ == "__main__":
    extract_audio_va(SPLIT)
    extract_text_valence(SPLIT)