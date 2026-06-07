"""
Feat_seg_bin_daic_eatd.py
==========================
SSL-based **segment-level** feature extraction for DAIC-WOZ + EATD.

跟 Feat_daic_eatd.py 的差別:
  audio: pooled [1, 1024]      ->  full [1, T, 1024]   (frame-level)
  text : token-level [1, L, 1024] (一樣)

跟 FeatureExtraction_seg_bin.py 的差別:
  monolingual (HuBERT + RoBERTa)  ->  multilingual (XLS-R + XLM-RoBERTa)
  DAIC only                       ->  DAIC + EATD

Models
------
  audio : facebook/wav2vec2-xls-r-300m
  text  : FacebookAI/xlm-roberta-large

Output layout  (datasets/Feat_seg_bin_daic_eatd/)
--------------------------------------------------
  DAIC-WOZ
    {pid}_acoustic.pt   list of [1, T, 1024]
    {pid}_text.pt       list of [1, L, 1024]

  EATD
    {vol}_acoustic.pt   list of [1, T, 1024]   (最多 3 segments)
    {vol}_text.pt       list of [1, L, 1024]   (最多 3 segments)

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
    Wav2Vec2Model,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────
# Paths & constants
# ─────────────────────────────────────────────
DAIC_DIR  = Path("datasets/DAICWOZ")
EATD_DIR  = Path("datasets/EATD")
OUT_DIR   = Path("datasets/Feat_seg_bin_daic_eatd")

TRAIN_CSV = DAIC_DIR / "train_split_Depression_AVEC2017.csv"
VAL_CSV   = DAIC_DIR / "dev_split_Depression_AVEC2017.csv"

AUDIO_MODEL = "facebook/wav2vec2-xls-r-300m"
TEXT_MODEL  = "FacebookAI/xlm-roberta-large"
AUDIO_LAYER = 12
TARGET_SR   = 16_000
MIN_WAV_LEN = 1600   # < 0.1 sec at 16kHz -> skip

EATD_PROMPTS = ["positive", "negative", "neutral"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_daic", action="store_true")
    p.add_argument("--skip_eatd", action="store_true")
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
        _audio_model = Wav2Vec2Model.from_pretrained(AUDIO_MODEL).to(device).eval()
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
# EATD
# ─────────────────────────────────────────────
def get_eatd_vol_dirs():
    dirs = [d for d in EATD_DIR.iterdir()
            if d.is_dir() and (d.name.startswith("t_") or d.name.startswith("v_"))]
    return sorted(dirs, key=lambda d: (d.name[0], int(d.name.split("_")[1])))


def extract_eatd_audio(device: str):
    extractor, model = get_audio_model(device)
    for vol_dir in get_eatd_vol_dirs():
        vol = vol_dir.name
        out_path = OUT_DIR / f"{vol}_acoustic.pt"
        if out_path.exists():
            print(f"[EATD audio] {vol}: exists, skip")
            continue
        full_list = []
        all_ok = True
        for prompt in EATD_PROMPTS:
            wav = vol_dir / f"{prompt}_out.wav"
            if not wav.exists():
                wav = vol_dir / f"{prompt}.wav"
            if not wav.exists():
                print(f"  [zero] {vol}/{prompt}: no wav")
                full_list.append(torch.zeros(1, 1, 1024))   # 占位 [1, T=1, 1024]
                continue
            try:
                feat = extract_audio_full(wav, extractor, model, device)
                if feat is None:
                    print(f"  [zero] {vol}/{prompt}: too short")
                    full_list.append(torch.zeros(1, 1, 1024))
                    continue
                full_list.append(feat)
            except Exception as e:
                print(f"  [zero] {vol}/{prompt}: {e}")
                full_list.append(torch.zeros(1, 1, 1024))
        torch.save(full_list, out_path)   # always 3 elements
        print(f"[EATD audio] {vol}: done (3 segs)")

def extract_eatd_text(device: str):
    tokenizer, model = get_text_model(device)
    for vol_dir in get_eatd_vol_dirs():
        vol = vol_dir.name
        out_path = OUT_DIR / f"{vol}_text.pt"
        if out_path.exists():
            print(f"[EATD text] {vol}: exists, skip")
            continue
        full_list = []
        for prompt in EATD_PROMPTS:
            txt_path = vol_dir / f"{prompt}.txt"
            if not txt_path.exists():
                print(f"  [zero] {vol}/{prompt}.txt: missing")
                full_list.append(torch.zeros(1, 1, 1024))
                continue
            text = txt_path.read_text(encoding="utf-8").strip()
            if not text:
                print(f"  [zero] {vol}/{prompt}.txt: empty")
                full_list.append(torch.zeros(1, 1, 1024))
                continue
            try:
                full_list.append(extract_text_full(text, tokenizer, model, device))
            except Exception as e:
                print(f"  [zero] {vol}/{prompt}: {e}")
                full_list.append(torch.zeros(1, 1, 1024))
        torch.save(full_list, out_path)   # always 3 elements
        print(f"[EATD text] {vol}: done (3 segs)")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_daic:
        print("\n" + "="*60); print("DAIC-WOZ"); print("="*60)
        extract_daic_audio(args.device)
        extract_daic_text(args.device)

    if not args.skip_eatd:
        if not EATD_DIR.exists():
            print(f"[EATD] {EATD_DIR} not found")
        else:
            print("\n" + "="*60); print("EATD"); print("="*60)
            extract_eatd_audio(args.device)
            extract_eatd_text(args.device)

    print("\n✓ Feat_seg_bin_daic_eatd.py done")


if __name__ == "__main__":
    main()