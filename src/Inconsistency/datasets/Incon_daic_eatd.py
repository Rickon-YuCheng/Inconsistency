"""
incon_daic_eatd.py
==================
Joint emotion extraction + pseudo label generation for DAIC-WOZ and EATD.

DAIC-WOZ
--------
  text  : DistilBERT multilingual (lxyuan/distilbert-base-multilingual-cased-sentiments-student)
  audio : SpeechBrain wav2vec2-IEMOCAP (hap->pos, sad/ang->neg, neu->neu)
  split : official train / dev CSVs

EATD
----
  text  : prompt label (資料夾名稱 positive/negative/neutral 直接對應 pos/neg/neu)
  audio : SpeechBrain wav2vec2-IEMOCAP, 優先用 *_out.wav, fallback 到 *.wav
  split : t_* -> train, v_* -> val
  label : new_label.txt (raw SDS × 1.25), cutoff >= 53 -> depressed (1)

Pseudo label strategy
---------------------
  每個 corpus (DAIC-all, EATD-train, EATD-val) 各自做
  z-score normalization + percentile filtering (q30/q70)。
  EATD text emotion 是 deterministic (prompt label),z-score 主要
  normalize audio modality bias。

Outputs (全部放在 OUT_DIR = datasets/Feat_daic_eatd/)
------------------------------------------------------
  emotion_daic_all.npz
  emotion_eatd_train.npz
  emotion_eatd_val.npz
  PseudoLabel_daic_zdist_q30_70.npz
  PseudoLabel_eatd_train_zdist_q30_70.npz
  PseudoLabel_eatd_val_zdist_q30_70.npz
  PseudoLabel_joint_train_q30_70.npz   <- DAIC train + EATD train 合併, Stage1 用

Resume: 若 emotion_*.npz 已存在就 skip extraction,直接讀取。

Usage
-----
  uv run incon_daic_eatd.py
  uv run incon_daic_eatd.py --low_q 25 --high_q 75
  uv run incon_daic_eatd.py --skip_daic   # 只跑 EATD
  uv run incon_daic_eatd.py --skip_eatd   # 只跑 DAIC
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from sklearn.preprocessing import StandardScaler
from speechbrain.inference.interfaces import foreign_class
from transformers import pipeline

warnings.filterwarnings("ignore", category=FutureWarning)

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
DAIC_DIR  = Path("datasets/DAICWOZ")
EATD_DIR  = Path("datasets/EATD")
OUT_DIR   = Path("datasets/Feat_daic_eatd")
SB_CACHE  = Path(".sb_cache")

TRAIN_CSV = DAIC_DIR / "train_split_Depression_AVEC2017.csv"
VAL_CSV   = DAIC_DIR / "dev_split_Depression_AVEC2017.csv"

EATD_SDS_CUTOFF = 53   # new_label.txt >= 53 -> depressed


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--low_q",     type=float, default=30)
    p.add_argument("--high_q",    type=float, default=70)
    p.add_argument("--skip_daic", action="store_true")
    p.add_argument("--skip_eatd", action="store_true")
    p.add_argument("--device",    type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ─────────────────────────────────────────────
# Shared models (lazy init)
# ─────────────────────────────────────────────
_sb_classifier = None
_distilbert    = None


def get_sb_classifier(device: str):
    global _sb_classifier
    if _sb_classifier is None:
        SB_CACHE.mkdir(parents=True, exist_ok=True)
        _sb_classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            savedir=SB_CACHE,
            run_opts={"device": device},
        )
        print("[SpeechBrain] model loaded")
    return _sb_classifier


def get_distilbert():
    global _distilbert
    if _distilbert is None:
        _distilbert = pipeline(
            model="lxyuan/distilbert-base-multilingual-cased-sentiments-student"
        )
        print("[DistilBERT] model loaded")
    return _distilbert


# ─────────────────────────────────────────────
# Audio emotion helpers
# ─────────────────────────────────────────────
def sb_label_to_pnn(label: str) -> str:
    """SpeechBrain IEMOCAP label -> 'pos' / 'neg' / 'neu'."""
    if label == "hap":
        return "pos"
    if label in ("sad", "ang"):
        return "neg"
    return "neu"


def audio_emotion_counts(wav_paths: list, classifier, device: str) -> dict:
    """
    Run SpeechBrain on a list of wav paths.
    Returns dict {pos, neg, neu}.
    Silently skips files that cannot be loaded.
    """
    pos = neg = neu = 0
    for wav_path in wav_paths:
        wav_path = Path(wav_path)
        if not wav_path.exists():
            print(f"  [skip] {wav_path} not found")
            continue
        try:
            waveform, sr = torchaudio.load(str(wav_path))
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            with torch.no_grad():
                _, _, _, labels = classifier.classify_batch(waveform.to(device))
            pnn = sb_label_to_pnn(labels[0])
        except Exception as e:
            print(f"  [error] {wav_path}: {e}")
            pnn = "neu"
        if pnn == "pos":
            pos += 1
        elif pnn == "neg":
            neg += 1
        else:
            neu += 1
    return {"pos": pos, "neg": neg, "neu": neu}


# ─────────────────────────────────────────────
# DAIC-WOZ
# ─────────────────────────────────────────────
def get_daic_splits():
    """Returns depMap, train_ids, val_ids."""
    tr  = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    depMap = {}
    for _, row in pd.concat([tr, val], ignore_index=True).iterrows():
        depMap[int(row["Participant_ID"])] = int(row["PHQ8_Binary"])
    return (depMap,
            tr["Participant_ID"].astype(int).tolist(),
            val["Participant_ID"].astype(int).tolist())


def extract_daic_emotions(patient_ids: list, device: str):
    classifier = get_sb_classifier(device)
    distilbert = get_distilbert()

    idx = []
    t_pos_l, t_neg_l, t_neu_l = [], [], []
    a_pos_l, a_neg_l, a_neu_l = [], [], []

    for pid in patient_ids:
        transcript = DAIC_DIR / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv"
        if not transcript.exists():
            print(f"[DAIC] patient {pid}: transcript not found, skip")
            continue

        df    = pd.read_csv(transcript, sep="\t")
        sents = df[df.speaker == "Participant"]["value"].dropna().tolist()
        tp = tn_neg = tn_neu = 0
        for sent in sents:
            result = distilbert(sent, batch_size=24)
            lbl    = result[0]["label"]
            if lbl == "positive":
                tp += 1
            elif lbl == "negative":
                tn_neg += 1
            else:
                tn_neu += 1

        splits_dir = DAIC_DIR / f"{pid}_P" / f"{pid}_aSplits"
        wav_files  = (sorted(splits_dir.glob("*.wav"),
                             key=lambda p: int(p.stem.split("_")[0]))
                      if splits_dir.exists() else [])
        if not wav_files:
            print(f"[DAIC] patient {pid}: no aSplits, skip")
            continue
        ac = audio_emotion_counts(wav_files, classifier, device)

        idx.append(pid)
        t_pos_l.append(tp);       t_neg_l.append(tn_neg); t_neu_l.append(tn_neu)
        a_pos_l.append(ac["pos"]); a_neg_l.append(ac["neg"]); a_neu_l.append(ac["neu"])

        print(f"[DAIC] patient {pid} | text: pos={tp} neg={tn_neg} neu={tn_neu} "
              f"| audio: pos={ac['pos']} neg={ac['neg']} neu={ac['neu']}")

    return (np.array(idx, dtype=np.int64),
            np.array(t_pos_l), np.array(t_neg_l), np.array(t_neu_l),
            np.array(a_pos_l), np.array(a_neg_l), np.array(a_neu_l))


def run_daic(device: str):
    out_path = OUT_DIR / "emotion_daic_all.npz"
    if out_path.exists():
        print(f"[DAIC] {out_path} exists, loading from cache")
        return np.load(out_path)

    # ── 快速路徑：若舊 npz 已存在，直接合成，不重跑 extraction ──
    text_npz  = Path("DistilBert_all_bin.npz")
    audio_npz = Path("Wav2Vec2_all_bin.npz")
    if text_npz.exists() and audio_npz.exists():
        print(f"[DAIC] 從現有 npz 合成 (skip extraction)")
        T = np.load(text_npz)
        A = np.load(audio_npz)
        assert np.array_equal(T["patientIdx"], A["patientIdx"]), \
            "DistilBert 和 Wav2Vec2 的 patientIdx 不一致"
        idx = T["patientIdx"].astype(np.int64)
        depMap, train_ids, _ = get_daic_splits()
        dep_labels = np.array([depMap[int(i)] for i in idx], dtype=np.int64)
        in_train   = np.array([1 if int(i) in set(train_ids) else 0 for i in idx],
                              dtype=np.int64)
        np.savez(out_path,
                 patientIdx=idx,
                 text_pos=T["a"],  text_neg=T["b"],  text_neu=T["c"],
                 audio_pos=A["a"], audio_neg=A["b"], audio_neu=A["c"],
                 dep_label=dep_labels, in_train=in_train)
        print(f"[DAIC] saved -> {out_path}  ({len(idx)} patients)")
        return np.load(out_path)

    # ── 完整 extraction（未來重現用）──
    print("\n" + "=" * 60)
    print("DAIC-WOZ emotion extraction")
    print("=" * 60)
    depMap, train_ids, val_ids = get_daic_splits()
    all_ids = train_ids + val_ids
    idx, tp, tn, tne, ap, an, ane = extract_daic_emotions(all_ids, device)
    dep_labels = np.array([depMap[i] for i in idx], dtype=np.int64)
    in_train   = np.array([1 if i in set(train_ids) else 0 for i in idx],
                          dtype=np.int64)
    np.savez(out_path,
             patientIdx=idx,
             text_pos=tp,  text_neg=tn,  text_neu=tne,
             audio_pos=ap, audio_neg=an, audio_neu=ane,
             dep_label=dep_labels, in_train=in_train)
    print(f"[DAIC] saved -> {out_path}  ({len(idx)} patients)")
    return np.load(out_path)


# ─────────────────────────────────────────────
# EATD
# ─────────────────────────────────────────────
EATD_PROMPT_MAP = {"positive": "pos", "negative": "neg", "neutral": "neu"}
EATD_PROMPTS    = list(EATD_PROMPT_MAP.keys())


def read_eatd_dep_label(vol_dir: Path):
    """Returns 0/1, or None if new_label.txt missing/invalid."""
    lbl_path = vol_dir / "new_label.txt"
    if not lbl_path.exists():
        return None
    try:
        val = float(lbl_path.read_text().strip())
        return 1 if val >= EATD_SDS_CUTOFF else 0
    except ValueError:
        return None


def extract_eatd_emotions(vol_dirs: list, device: str):
    classifier = get_sb_classifier(device)

    idx      = []
    dep_lbls = []
    t_pos_l, t_neg_l, t_neu_l = [], [], []
    a_pos_l, a_neg_l, a_neu_l = [], [], []

    for vol_dir in vol_dirs:
        dep = read_eatd_dep_label(vol_dir)
        if dep is None:
            print(f"[EATD] {vol_dir.name}: missing/invalid new_label.txt, skip")
            continue

        t_pos = t_neg = t_neu = 0
        a_pos = a_neg = a_neu = 0
        all_ok = True

        for prompt in EATD_PROMPTS:
            # text: 1 deterministic vote from prompt name
            pnn = EATD_PROMPT_MAP[prompt]
            if pnn == "pos":   t_pos += 1
            elif pnn == "neg": t_neg += 1
            else:              t_neu += 1

            # audio: prefer *_out.wav, fallback to *.wav
            wav = vol_dir / f"{prompt}_out.wav"
            if not wav.exists():
                wav = vol_dir / f"{prompt}.wav"
            if not wav.exists():
                print(f"[EATD] {vol_dir.name}/{prompt}: no wav, skip volunteer")
                all_ok = False
                break
            ac = audio_emotion_counts([wav], classifier, device)
            a_pos += ac["pos"]; a_neg += ac["neg"]; a_neu += ac["neu"]

        if not all_ok:
            continue

        idx.append(vol_dir.name)
        dep_lbls.append(dep)
        t_pos_l.append(t_pos); t_neg_l.append(t_neg); t_neu_l.append(t_neu)
        a_pos_l.append(a_pos); a_neg_l.append(a_neg); a_neu_l.append(a_neu)

        print(f"[EATD] {vol_dir.name} dep={dep} "
              f"| text: pos={t_pos} neg={t_neg} neu={t_neu} "
              f"| audio: pos={a_pos} neg={a_neg} neu={a_neu}")

    return (np.array(idx, dtype=object),
            np.array(dep_lbls, dtype=np.int64),
            np.array(t_pos_l), np.array(t_neg_l), np.array(t_neu_l),
            np.array(a_pos_l), np.array(a_neg_l), np.array(a_neu_l))


def run_eatd(device: str) -> dict:
    results = {}
    for split_prefix, split_name in [("t", "train"), ("v", "val")]:
        out_path = OUT_DIR / f"emotion_eatd_{split_name}.npz"

        if out_path.exists():
            print(f"[EATD-{split_name}] {out_path} exists, loading from cache")
            results[split_name] = np.load(out_path, allow_pickle=True)
            continue

        print("\n" + "=" * 60)
        print(f"EATD {split_name} emotion extraction")
        print("=" * 60)

        vol_dirs = sorted(
            [d for d in EATD_DIR.iterdir()
             if d.is_dir() and d.name.startswith(f"{split_prefix}_")],
            key=lambda d: int(d.name.split("_")[1]),
        )
        if not vol_dirs:
            print(f"[EATD-{split_name}] no {split_prefix}_* dirs in {EATD_DIR}, skip")
            continue

        idx, dep, tp, tn, tne, ap, an, ane = extract_eatd_emotions(vol_dirs, device)

        np.savez(out_path,
                 patientIdx=idx,
                 text_pos=tp,  text_neg=tn,  text_neu=tne,
                 audio_pos=ap, audio_neg=an, audio_neu=ane,
                 dep_label=dep)
        print(f"[EATD-{split_name}] saved -> {out_path}  ({len(idx)} volunteers)")
        results[split_name] = np.load(out_path, allow_pickle=True)

    return results


# ─────────────────────────────────────────────
# Pseudo label
# ─────────────────────────────────────────────
def make_pseudo_label(
    patientIdx,
    text_counts,
    audio_counts,
    dep_labels,
    out_path: Path,
    low_q: float = 30,
    high_q: float = 70,
    tag: str = "",
):
    T = np.nan_to_num(text_counts.astype(np.float32),  nan=0.0)
    A = np.nan_to_num(audio_counts.astype(np.float32), nan=0.0)

    def _zscore(X):
        return StandardScaler().fit_transform(X)

    T_z = _zscore(T)
    A_z = _zscore(A)
    scores = np.linalg.norm(T_z - A_z, axis=1)

    low_th  = np.percentile(scores, low_q)
    high_th = np.percentile(scores, high_q)

    labels = np.full(len(scores), -1, dtype=np.int64)
    labels[scores <= low_th]  = 1   # consistent
    labels[scores >= high_th] = 0   # inconsistent
    keep = labels != -1

    np.savez(
        out_path,
        patientIdx     = patientIdx[keep],
        label          = labels[keep],
        score          = scores[keep],
        dep_label      = dep_labels[keep],
        all_patientIdx = patientIdx,
        all_score      = scores,
        all_dep_label  = dep_labels,
        low_th         = low_th,
        high_th        = high_th,
    )
    print(f"\n[PseudoLabel{tag}] -> {out_path}")
    print(f"  low_th={low_th:.4f}  high_th={high_th:.4f}")
    print(f"  kept {keep.sum()} / {len(scores)}")
    print(f"  atei label dist: {np.bincount(labels[keep], minlength=2)}")
    print(f"  dep  dist kept:  {np.bincount(dep_labels[keep].astype(int), minlength=2)}")


def run_pseudo_labels(daic_data, eatd_data: dict, low_q: float, high_q: float):
    q_tag = f"q{low_q:.0f}_{high_q:.0f}"

    # ── DAIC ──
    if daic_data is not None:
        d = daic_data
        make_pseudo_label(
            patientIdx   = d["patientIdx"],
            text_counts  = np.column_stack([d["text_pos"],  d["text_neg"],  d["text_neu"]]),
            audio_counts = np.column_stack([d["audio_pos"], d["audio_neg"], d["audio_neu"]]),
            dep_labels   = d["dep_label"],
            out_path     = OUT_DIR / f"PseudoLabel_daic_zdist_{q_tag}.npz",
            low_q=low_q, high_q=high_q, tag=" DAIC-all",
        )

    # ── EATD ──
    for split_name, data in eatd_data.items():
        make_pseudo_label(
            patientIdx   = data["patientIdx"],
            text_counts  = np.column_stack([data["text_pos"],  data["text_neg"],  data["text_neu"]]),
            audio_counts = np.column_stack([data["audio_pos"], data["audio_neg"], data["audio_neu"]]),
            dep_labels   = data["dep_label"],
            out_path     = OUT_DIR / f"PseudoLabel_eatd_{split_name}_zdist_{q_tag}.npz",
            low_q=low_q, high_q=high_q, tag=f" EATD-{split_name}",
        )

    # ── Joint train (DAIC-train + EATD-train) ──
    daic_pl_path = OUT_DIR / f"PseudoLabel_daic_zdist_{q_tag}.npz"
    eatd_pl_path = OUT_DIR / f"PseudoLabel_eatd_train_zdist_{q_tag}.npz"
    if not (daic_pl_path.exists() and eatd_pl_path.exists()):
        print("[Joint] skipping joint train npz (daic or eatd pseudo label missing)")
        return

    _, train_ids, _ = get_daic_splits()
    train_set = set(train_ids)

    dp = np.load(daic_pl_path)
    ep = np.load(eatd_pl_path, allow_pickle=True)

    # DAIC: keep train split only, prefix id with "daic_"
    daic_mask  = np.array([int(i) in train_set for i in dp["patientIdx"]])
    daic_idx   = np.array([f"daic_{i}" for i in dp["patientIdx"][daic_mask]], dtype=object)
    daic_label = dp["label"][daic_mask]
    daic_dep   = dp["dep_label"][daic_mask]

    # EATD: prefix id with "eatd_"
    eatd_idx   = np.array([f"eatd_{i}" for i in ep["patientIdx"]], dtype=object)
    eatd_label = ep["label"]
    eatd_dep   = ep["dep_label"]

    joint_idx   = np.concatenate([daic_idx,   eatd_idx])
    joint_label = np.concatenate([daic_label, eatd_label])
    joint_dep   = np.concatenate([daic_dep,   eatd_dep])

    out_joint = OUT_DIR / f"PseudoLabel_joint_train_{q_tag}.npz"
    np.savez(out_joint,
             patientIdx=joint_idx,
             label=joint_label,
             dep_label=joint_dep)

    print(f"\n[Joint train PseudoLabel] -> {out_joint}")
    print(f"  DAIC train kept: {daic_mask.sum()}  EATD train kept: {len(eatd_idx)}")
    print(f"  total: {len(joint_idx)}")
    print(f"  atei label dist: {np.bincount(joint_label, minlength=2)}")
    print(f"  dep  label dist: {np.bincount(joint_dep.astype(int), minlength=2)}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    daic_data = None
    eatd_data = {}

    if not args.skip_daic:
        daic_data = run_daic(args.device)

    if not args.skip_eatd:
        if not EATD_DIR.exists():
            print(f"[EATD] {EATD_DIR} not found — re-run after dataset downloaded")
        else:
            eatd_data = run_eatd(args.device)

    run_pseudo_labels(daic_data, eatd_data, args.low_q, args.high_q)
    print("\n✓ incon_daic_eatd.py done")


if __name__ == "__main__":
    main()