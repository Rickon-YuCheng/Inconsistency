"""
Feat_daic_eatd.py
=================
SSL-based feature extraction for DAIC-WOZ + EATD (joint training).

Models
------
  audio : facebook/wav2vec2-xls-r-300m   (XLS-R, multilingual wav2vec2, 1024-dim)
  text  : FacebookAI/xlm-roberta-large   (XLM-RoBERTa, multilingual RoBERTa, 1024-dim)

維度與原版 (HuBERT / RoBERTa-large) 相同，model code 不需要改動。

Output layout  (datasets/Feat_daic_eatd/)
-----------------------------------------
  DAIC-WOZ
    {pid}_acoustic.pt   list of [1, 1024]   (segment-level pooled, one per sentence)
    {pid}_text.pt       list of [1, L, 1024] (token-level, one per sentence)

  EATD
    {vol}_acoustic.pt   list of [1, 1024]   (3 elements: positive/negative/neutral 順序)
    {vol}_text.pt       list of [1, L, 1024] (3 elements, same order)

Resume
------
  若 {id}_acoustic.pt 已存在就 skip，不重抽。

Usage
-----
  uv run Feat_daic_eatd.py                # 兩個 corpus 都抽
  uv run Feat_daic_eatd.py --skip_daic    # 只抽 EATD
  uv run Feat_daic_eatd.py --skip_eatd    # 只抽 DAIC
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
OUT_DIR   = Path("datasets/Feat_daic_eatd")

TRAIN_CSV = DAIC_DIR / "train_split_Depression_AVEC2017.csv"
VAL_CSV   = DAIC_DIR / "dev_split_Depression_AVEC2017.csv"

AUDIO_MODEL = "facebook/wav2vec2-xls-r-300m"
TEXT_MODEL  = "FacebookAI/xlm-roberta-large"
AUDIO_LAYER = 12    # same convention as original HuBERT extraction

EATD_PROMPTS = ["positive", "negative", "neutral"]   # fixed order -> 3 segments


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--skip_daic", action="store_true")
    p.add_argument("--skip_eatd", action="store_true")
    p.add_argument("--device",   type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ─────────────────────────────────────────────
# Model loader (lazy, shared)
# ─────────────────────────────────────────────
_audio_model     = None
_audio_extractor = None
_text_model      = None
_text_tokenizer  = None


def get_audio_model(device: str):
    global _audio_model, _audio_extractor
    if _audio_model is None:
        print(f"[Audio] loading {AUDIO_MODEL} ...")
        _audio_extractor = AutoFeatureExtractor.from_pretrained(AUDIO_MODEL)
        _audio_model     = Wav2Vec2Model.from_pretrained(AUDIO_MODEL).to(device).eval()
        print("[Audio] model ready")
    return _audio_extractor, _audio_model


def get_text_model(device: str):
    global _text_model, _text_tokenizer
    if _text_model is None:
        print(f"[Text] loading {TEXT_MODEL} ...")
        _text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
        _text_model     = AutoModel.from_pretrained(TEXT_MODEL).to(device).eval()
        print("[Text] model ready")
    return _text_tokenizer, _text_model


# ─────────────────────────────────────────────
# Core extraction helpers
# ─────────────────────────────────────────────
TARGET_SR = 16_000   # XLS-R expects 16 kHz


def extract_audio_segment(wav_path: Path, extractor, model, device: str) -> torch.Tensor:
    """
    Load one wav file, run XLS-R, return [1, 1024] (layer-12 mean pooled).
    """
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
    waveform = waveform.squeeze()   # [T]

    inputs = extractor(waveform, sampling_rate=TARGET_SR,
                       return_tensors="pt").input_values.to(device)

    with torch.inference_mode():
        hidden = model(inputs, output_hidden_states=True).hidden_states[AUDIO_LAYER]
        # hidden: [1, T', 1024] -> mean pool -> [1, 1024]
        pooled = hidden.mean(dim=1).cpu()

    return pooled   # [1, 1024]


def extract_text_segment(text: str, tokenizer, model, device: str) -> torch.Tensor:
    """
    Tokenize one sentence, run XLM-RoBERTa, return [1, L, 1024].
    """
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=512).to(device)

    with torch.inference_mode():
        H = model(**inputs).last_hidden_state   # [1, L, 1024]

    return H.cpu()   # [1, L, 1024]


# ─────────────────────────────────────────────
# DAIC-WOZ
# ─────────────────────────────────────────────
def get_daic_ids():
    import pandas as pd
    tr  = pd.read_csv(TRAIN_CSV)["Participant_ID"].astype(int).tolist()
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
            print(f"[DAIC audio] {pid}: aSplits not found, skip")
            continue

        wav_files = sorted(splits_dir.glob("*.wav"),
                           key=lambda p: int(p.stem.split("_")[0]))
        if not wav_files:
            print(f"[DAIC audio] {pid}: no wav files, skip")
            continue

        pooled_list = []
        for wav in wav_files:
            try:
                pooled_list.append(extract_audio_segment(wav, extractor, model, device))
            except Exception as e:
                print(f"  [error] {wav.name}: {e}, skip segment")

        if pooled_list:
            torch.save(pooled_list, out_path)
            print(f"[DAIC audio] {pid}: done ({len(pooled_list)} segs)")
        else:
            print(f"[DAIC audio] {pid}: no valid segments, skip")


def extract_daic_text(device: str):
    tokenizer, model = get_text_model(device)

    for pid in get_daic_ids():
        out_path = OUT_DIR / f"{pid}_text.pt"
        if out_path.exists():
            print(f"[DAIC text] {pid}: exists, skip")
            continue

        t_path = DAIC_DIR / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv"
        if not t_path.exists():
            print(f"[DAIC text] {pid}: transcript not found, skip")
            continue

        df    = pd.read_csv(t_path, sep="\t")
        sents = df[df.speaker == "Participant"]["value"].dropna().tolist()

        full_list = []
        for sent in sents:
            try:
                full_list.append(extract_text_segment(sent, tokenizer, model, device))
            except Exception as e:
                print(f"  [error] sent '{sent[:30]}...': {e}, skip")

        if full_list:
            torch.save(full_list, out_path)
            print(f"[DAIC text] {pid}: done ({len(full_list)} sents)")
        else:
            print(f"[DAIC text] {pid}: no valid sents, skip")


# ─────────────────────────────────────────────
# EATD
# ─────────────────────────────────────────────
def get_eatd_vol_dirs():
    """Return all t_* and v_* volunteer dirs, sorted numerically."""
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

        pooled_list = []
        all_ok = True
        for prompt in EATD_PROMPTS:
            # prefer *_out.wav, fallback to *.wav
            wav = vol_dir / f"{prompt}_out.wav"
            if not wav.exists():
                wav = vol_dir / f"{prompt}.wav"
            if not wav.exists():
                print(f"[EATD audio] {vol}/{prompt}: no wav, skip volunteer")
                all_ok = False
                break
            try:
                pooled_list.append(extract_audio_segment(wav, extractor, model, device))
            except Exception as e:
                print(f"  [error] {vol}/{prompt}: {e}, use zeros")
                pooled_list.append(torch.zeros(1, 1024))

        if all_ok:
            torch.save(pooled_list, out_path)   # always 3 elements
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
        all_ok = True
        for prompt in EATD_PROMPTS:
            txt_path = vol_dir / f"{prompt}.txt"
            if not txt_path.exists():
                print(f"[EATD text] {vol}/{prompt}.txt not found, skip volunteer")
                all_ok = False
                break
            text = txt_path.read_text(encoding="utf-8").strip()
            if not text:
                print(f"[EATD text] {vol}/{prompt}.txt empty, skip volunteer")
                all_ok = False
                break
            try:
                full_list.append(extract_text_segment(text, tokenizer, model, device))
            except Exception as e:
                print(f"  [error] {vol}/{prompt}: {e}, use zeros")
                full_list.append(torch.zeros(1, 1, 1024))

        if all_ok:
            torch.save(full_list, out_path)   # always 3 elements
            print(f"[EATD text] {vol}: done (3 sents)")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not args.skip_daic:
        print("\n" + "=" * 60)
        print("DAIC-WOZ feature extraction")
        print("=" * 60)
        extract_daic_audio(args.device)
        extract_daic_text(args.device)

    if not args.skip_eatd:
        if not EATD_DIR.exists():
            print(f"[EATD] {EATD_DIR} not found — re-run after dataset downloaded")
        else:
            print("\n" + "=" * 60)
            print("EATD feature extraction")
            print("=" * 60)
            extract_eatd_audio(args.device)
            extract_eatd_text(args.device)

    print("\n✓ Feat_daic_eatd.py done")


if __name__ == "__main__":
    main()