"""
Feat_seg_bin_daic.py
====================
SSL-based **segment-level** feature extraction for DAIC-WOZ only.

跟 Feat_seg_bin_daic_eatd.py 的差別:
  audio : facebook/wav2vec2-xls-r-300m  ->  facebook/hubert-large-ls960-ft
  text  : FacebookAI/xlm-roberta-large  ->  roberta-large
  移除所有 EATD 邏輯

Models
------
  audio : facebook/hubert-large-ls960-ft
  text  : roberta-large

Output layout  (datasets/Feat_seg_bin_daic/)
--------------------------------------------
  {pid}_acoustic.pt   list of [1, T, 1024]
  {pid}_text.pt       list of [1, L, 1024]

Resume: 若 {id}_acoustic.pt 已存在就 skip。
"""
import argparse
import warnings
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    AutoTokenizer,
    HubertModel,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────
DAIC_DIR  = Path("datasets/DAICWOZ")
OUT_DIR   = Path("datasets/Feat_raw")

TRAIN_CSV = DAIC_DIR / "train_split_Depression_AVEC2017.csv"
VAL_CSV   = DAIC_DIR / "dev_split_Depression_AVEC2017.csv"

AUDIO_MODEL = "facebook/hubert-large-ls960-ft"
TEXT_MODEL  = "roberta-large"
AUDIO_LAYER = 12
TARGET_SR   = 16_000
MIN_WAV_LEN = 1600   # < 0.1 sec at 16kHz -> skip


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ─────────────────────────────────────────────
# Model loaders (lazy)
# ─────────────────────────────────────────────
_audio_model = _audio_extractor = None
_text_model = _text_tokenizer = None


def get_audio_model(device: str):
    global _audio_model, _audio_extractor
    if _audio_model is None:
        print(f"[Audio] loading {AUDIO_MODEL} ...")
        _audio_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
        _audio_model = HubertModel.from_pretrained(AUDIO_MODEL).to(device).eval()
        print("[Audio] ready")
    return _audio_extractor, _audio_model


def get_text_model(device: str):
    global _text_model, _text_tokenizer
    if _text_model is None:
        print(f"[Text] loading {TEXT_MODEL} ...")
        _text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
        _text_model = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()
        print("[Text] ready")
    return _text_tokenizer, _text_model


# ─────────────────────────────────────────────
# Extraction helpers (segment-level, full features)
# ─────────────────────────────────────────────
def extract_audio_full(wav_path: Path, extractor, model, device: str):
    """Return [1, T, 1024] (layer-12, NO pooling). None if too short."""
    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.shape[0] > 1:   # stereo -> mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    waveform = waveform.squeeze()   # [T]

    if waveform.numel() < MIN_WAV_LEN:
        return None

    inputs = extractor(waveform, sampling_rate=TARGET_SR,
                       return_tensors="pt").input_values.to(device)
    with torch.inference_mode():
        H = model(inputs, output_hidden_states=True).hidden_states[AUDIO_LAYER]
    return H.cpu()   # [1, T', 1024]


def extract_text_full(text: str, tokenizer, model, device: str):
    """Return [1, L, 1024]."""
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512).to(device)
    with torch.inference_mode():
        H = model(**inputs).last_hidden_state
    return H.cpu()   # [1, L, 1024]


# ─────────────────────────────────────────────
# DAIC-WOZ
# ─────────────────────────────────────────────
def get_daic_ids():
    tr = pd.read_csv(TRAIN_CSV)["Participant_ID"].astype(int).tolist()
    val = pd.read_csv(VAL_CSV)["Participant_ID"].astype(int).tolist()
    return tr + val


def extract_daic_audio(device: str):
    extractor, model = get_audio_model(device)
    for pid in get_daic_ids():
        out_path = OUT_DIR / f"{pid}_acoustic.pt"
        if out_path.exists():
            print(f"[DAIC audio] {pid}: exists, skip")
            continue
        splits_dir = DAIC_DIR / f"{pid}_P" / f"{pid}_aSplits"
        if not splits_dir.exists():
            print(f"[DAIC audio] {pid}: no aSplits, skip")
            continue
        wav_files = sorted(splits_dir.glob("*.wav"),
                           key=lambda p: int(p.stem.split("_")[0]))
        full_list = []
        for wav in wav_files:
            try:
                feat = extract_audio_full(wav, extractor, model, device)
                if feat is None:
                    print(f"  [skip] {wav.name}: too short")
                    continue
                full_list.append(feat)
            except Exception as e:
                print(f"  [error] {wav.name}: {e}")
        if full_list:
            torch.save(full_list, out_path)
            print(f"[DAIC audio] {pid}: done ({len(full_list)} segs)")


def extract_daic_text(device: str):
    tokenizer, model = get_text_model(device)
    for pid in get_daic_ids():
        out_path = OUT_DIR / f"{pid}_text.pt"
        if out_path.exists():
            print(f"[DAIC text] {pid}: exists, skip")
            continue
        t_path = DAIC_DIR / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv"
        if not t_path.exists():
            print(f"[DAIC text] {pid}: no transcript, skip")
            continue
        df = pd.read_csv(t_path, sep="\t")
        sents = df[df.speaker == "Participant"]["value"].dropna().tolist()
        full_list = []
        for sent in sents:
            try:
                full_list.append(extract_text_full(sent, tokenizer, model, device))
            except Exception as e:
                print(f"  [error] '{sent[:30]}...': {e}")
        if full_list:
            torch.save(full_list, out_path)
            print(f"[DAIC text] {pid}: done ({len(full_list)} sents)")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60); print("DAIC-WOZ"); print("="*60)
    extract_daic_audio(args.device)
    extract_daic_text(args.device)

    print("\n✓ Feat_seg_bin_daic.py done")


if __name__ == "__main__":
    main()