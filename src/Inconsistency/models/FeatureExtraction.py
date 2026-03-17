# HuBERT: https://huggingface.co/docs/transformers/model_doc/hubert#transformers.HubertModel
# RoBERTA:
import torchaudio
from transformers import AutoProcessor, HubertModel
from pathlib import Path
import torch
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

START = 300
END = 493
# START=300
# END=302


def loadAudio():
    processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to("cuda")
    for i in range(START, END):
        s_path = Path(f"datasets/DAICWOZ/{i}_P/{i}_aSplits")
        Personal_list = []

        if not os.path.exists(s_path):
            print(f"PATH: {s_path} does not exist")
            continue

        for j in s_path.glob("*.wav"):
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
                breakpoint()
                Personal_list.append(X.cpu().detach())
        torch.save(
            Personal_list, f"datasets/Feature/HuBERT/{i}_acoustic.pt"
        )  # tot 17.9GB
        print(f"patient{i} successed")


if __name__ == "__main__":
    loadAudio()
