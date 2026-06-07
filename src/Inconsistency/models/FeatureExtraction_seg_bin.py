"""
SSL-based Feature Extraction (segment-level pipeline 用)

抽 2 份 feature:
  - datasets/Feature/HuBERT_full_seg_bin/    frame-level audio, 每句 [1, T, 1024]
  - datasets/Feature/RoBerTa_full_seg_bin/   token-level  text,  每句 [1, L, 1024]

跟 FeatureExtraction_bin.py (patient-level pipeline) 的差別
-----------------------------------------------------------
patient-level:
  audio 是 pooled  -> [1, 1024]    (segment 內 mean over time)
  text  是 full    -> [1, L, 1024]
  Stage2 主體吃 pooled audio + token-level text。

segment-level:
  audio 必須是 full -> [1, T, 1024]
  text  必須是 full -> [1, L, 1024]
  Stage1Tr_seg 的 SegDataset 預期兩邊都是序列, 在 model 內各自走
  self-attention + cross-attention。pooled audio 沒有 T 維度,
  segment-level cross-attention 會退化。

備註
----
text 端的內容 (token-level RoBERTa) 其實跟 patient-level 那份 (RoBerTa_full_bin)
一模一樣。如果你想省時間, 可以直接 symlink 或在 Stage1Tr_seg 把路徑指過去,
就不用重跑 loadText。本檔為了讓兩條 pipeline 解耦, 預設仍另存一份。
"""
import torchaudio
from transformers import AutoProcessor, HubertModel
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import torch
import os
import pandas as pd
import warnings
from Inconsistency.datasets.Incon_seg_bin import get_Split_and_GroundTrue

warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================
# Audio: HuBERT (full / frame-level)
# ============================================================
def loadAudio():
    MODEL_NAME = "facebook/hubert-large-ls960-ft"
    HUBERT_LAYER = 12

    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = HubertModel.from_pretrained(MODEL_NAME).to("cuda").eval()

    out_full = Path("datasets/Feature/HuBERT_full_seg_bin")
    out_full.mkdir(parents=True, exist_ok=True)

    _, trDS, testDS = get_Split_and_GroundTrue()
    for dataset in [trDS, testDS]:
        for i in dataset:
            s_path = Path(f"datasets/DAICWOZ/{i}_P/{i}_aSplits")
            if not os.path.exists(s_path):
                print(f"PATH: {s_path} does not exist")
                continue

            Full_list = []

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

                    Full_list.append(X.cpu().detach())   # [1, T, 1024]

            torch.save(Full_list, out_full / f"{i}_acoustic.pt")
            print(f"[Audio] patient{i} done ({len(Full_list)} segs)")


# ============================================================
# Text: RoBERTa (full / token-level)
# ============================================================
def loadText():
    MODEL_NAME = "FacebookAI/roberta-large"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to("cuda").eval()

    out_full = Path("datasets/Feature/RoBerTa_full_seg_bin")
    out_full.mkdir(parents=True, exist_ok=True)

    _, trDS, testDS = get_Split_and_GroundTrue()
    for dataset in [trDS, testDS]:
        for i in dataset:
            t_path = Path(f"datasets/DAICWOZ/{i}_P/{i}_TRANSCRIPT.csv")
            if not os.path.exists(t_path):
                print(f"PATH: {t_path} does not exist")
                continue

            x = pd.read_csv(t_path, sep="\t")
            x = x[x.speaker == "Participant"]
            x = x["value"].dropna().tolist()

            Full_list = []

            for sent in x:
                inputs = tokenizer(sent, return_tensors="pt").to("cuda")
                with torch.inference_mode():
                    outputs = model(**inputs)
                    # last_hidden_state: [1, L, 1024]
                    H = outputs.last_hidden_state

                    Full_list.append(H.cpu().detach())   # [1, L, 1024]

            torch.save(Full_list, out_full / f"{i}_text.pt")
            print(f"[Text]  patient{i} done ({len(Full_list)} sents)")


if __name__ == "__main__":
    loadAudio()
    # loadText()