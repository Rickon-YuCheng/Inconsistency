""" SSL-based Feature Extraction — Audio with mean pool """
# HuBERT: facebook/hubert-large-ll60k (non-finetuned)
# 抽完 frame-level feature 直接 mean pool over time,每 segment 只存 [1024] 向量
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from pathlib import Path
import torch
import os
from Inconsistency.datasets.inconsistentLabel import get_Split_and_GroundTrue

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

HUBERT_LAYER = 12


def loadAudio():
    MODEL_NAME = "facebook/hubert-large-ll60k"
    processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = HubertModel.from_pretrained(MODEL_NAME).to("cuda").eval()

    # 確保輸出目錄存在
    out_root = Path("datasets/Feature/HuBERT2")
    out_root.mkdir(parents=True, exist_ok=True)

    _, trDS, valDS, testDS = get_Split_and_GroundTrue()
    for dataset in [trDS, valDS, testDS]:
        for i in dataset:
            s_path = Path(f"datasets/DAICWOZ/{i}_P/{i}_aSplits")
            Personal_list = []

            if not os.path.exists(s_path):
                print(f"PATH: {s_path} does not exist")
                continue

            for j in sorted(s_path.glob("*.wav"),
                            key=lambda p: int(p.stem.split("_")[0])):
                waveframe, sr = torchaudio.load(str(j))

                input_values = processor(
                    waveframe.squeeze(), sampling_rate=sr, return_tensors="pt"
                ).input_values.to("cuda")

                with torch.inference_mode():
                    X = model(
                        input_values, output_hidden_states=True
                    ).hidden_states[HUBERT_LAYER]
                    # X: [1, T, 1024]
                    X_pooled = X.mean(dim=1).cpu()  # [1, 1024]
                    Personal_list.append(X_pooled)

            torch.save(
                Personal_list,
                out_root / f"{i}_acoustic.pt"
            )
            print(f"patient{i} successed (pooled, {len(Personal_list)} segs)")


if __name__ == "__main__":
    loadAudio()