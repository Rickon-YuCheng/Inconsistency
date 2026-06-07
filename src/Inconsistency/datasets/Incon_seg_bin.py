# Segment-level 情緒抽取 (二元 ATEI pipeline 用) — 加入 z-score 校正版本
#
# 跟舊版的差別:
#   1. 三個 extraction 函數 (DISTILBERT / WAV2VEC2 / HOWNET) 改成:
#      raw prob → per-class z-score 校正 → argmax → 存校正後 label
#      同時存 z-score 後接 softmax 的 soft prob [N, 3]
#   2. 新增 EATD 處理 (audio + text)
#   3. EATD text 用 {prompt}.txt 內容跑 DistilBERT,不是用檔名當 label
import torch
from transformers import pipeline
import yaml
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchaudio
from pathlib import Path
from speechbrain.inference.interfaces import foreign_class
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

SPLIT = "all"
CFG_PATH = "configs/inconsistentLabel.yaml"
TRAIN_CSV = "datasets/DAICWOZ/train_split_Depression_AVEC2017.csv"
VAL_CSV = "datasets/DAICWOZ/dev_split_Depression_AVEC2017.csv"
EATD_DIR = Path("datasets/EATD")

EMO2ID = {"positive": 0, "negative": 1, "neutral": 2}
EATD_PROMPTS = ["positive", "negative", "neutral"]


def parse_args():
    with open(CFG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**cfg)
    parser.add_argument("--ds", type=str, help="upper case")
    parser.add_argument("--split", type=str, default=SPLIT,
                        choices=["train", "test", "all"])
    parser.add_argument("--run_eatd", action="store_true",
                        help="also process EATD")
    parser.add_argument("--run_hownet", action="store_true",
                        help="also run HowNet (default off)")
    args = parser.parse_args()
    assert args.ds in ["DAICWOZ", "MOSI"], f"Invalid ds name: {args.ds}"
    return args


# ============================================================
# Split utils (跟原版完全一致,給其他模組 import)
# ============================================================
def get_Split_and_GroundTrue():
    tr = pd.read_csv(TRAIN_CSV)
    val = pd.read_csv(VAL_CSV)
    depMap = {}
    for _, row in pd.concat([tr, val], ignore_index=True).iterrows():
        pid = int(row["Participant_ID"])
        depMap[pid] = int(row["PHQ8_Binary"])
    train_idx = tr["Participant_ID"].astype(int).tolist()
    test_idx = val["Participant_ID"].astype(int).tolist()
    return depMap, train_idx, test_idx


def get_stage1_kfold(n_splits: int = 3, seed: int = 42):
    from sklearn.model_selection import StratifiedKFold
    tr = pd.read_csv(TRAIN_CSV)
    depMap = {}
    for _, row in pd.concat([pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)],
                            ignore_index=True).iterrows():
        depMap[int(row["Participant_ID"])] = int(row["PHQ8_Binary"])
    ids = tr["Participant_ID"].astype(int).tolist()
    labels = tr["PHQ8_Binary"].astype(int).tolist()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for i, (tr_i, val_i) in enumerate(skf.split(ids, labels)):
        folds.append({"fold": i,
                      "train": [ids[j] for j in tr_i],
                      "val": [ids[j] for j in val_i]})
    return depMap, folds


def get_stage1_split(seed: int = 42):
    tr = pd.read_csv(TRAIN_CSV)
    depMap = {}
    for _, row in pd.concat([pd.read_csv(TRAIN_CSV), pd.read_csv(VAL_CSV)],
                            ignore_index=True).iterrows():
        depMap[int(row["Participant_ID"])] = int(row["PHQ8_Binary"])
    patient_df = tr[["Participant_ID", "PHQ8_Binary"]].copy()
    s1_tr_df, s1_val_df = train_test_split(
        patient_df, test_size=0.2, random_state=seed,
        stratify=patient_df["PHQ8_Binary"])
    return (depMap,
            s1_tr_df["Participant_ID"].astype(int).tolist(),
            s1_val_df["Participant_ID"].astype(int).tolist())


def get_patient_ids(split: str):
    _, train_idx, test_idx = get_Split_and_GroundTrue()
    if split == "train":
        return train_idx
    elif split == "test":
        return test_idx
    elif split == "all":
        return train_idx + test_idx
    else:
        raise ValueError(f"unknown split: {split}")


# ============================================================
# Z-score helper: 給定 [N, 3] raw prob
#   回傳 hard label [N] 和 soft prob [N, 3]
#   流程: raw prob → per-class z-score → softmax → argmax
# ============================================================
def zscore_and_argmax(probs: np.ndarray):
    """
    probs: [N, 3]  raw probability for (pos, neg, neu)
    回傳:
      hard_label: [N]    z-score+softmax 後 argmax
      soft_prob:  [N, 3] z-score 後接 softmax 的合法分布
    """
    z = StandardScaler().fit_transform(probs.astype(np.float32))
    # softmax over z to get valid probability distribution
    e = np.exp(z - z.max(axis=1, keepdims=True))
    soft_prob = (e / e.sum(axis=1, keepdims=True)).astype(np.float32)
    hard_label = soft_prob.argmax(axis=1).astype(np.int64)
    return hard_label, soft_prob


# ============================================================
# Audio preprocessing (原版,不動)
# ============================================================
def audioPreprosessing(ds: str, ds_dir: str, device: str, split: str, **kw):
    print("\n**audioPreprocessing**")
    patient_ids = get_patient_ids(split)
    for i in patient_ids:
        csvfilePath = f"{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv"
        audiofilePath = f"{ds_dir}/{i}_P/{i}_AUDIO.wav"
        if not os.path.exists(audiofilePath):
            print(f"PATH: {audiofilePath} does not exist")
            continue
        x = pd.read_csv(csvfilePath, sep="\t")
        x = x[x["speaker"] == "Participant"].dropna(subset=["value"]).copy()
        _, sr = torchaudio.load(audiofilePath)
        fpath = Path("/workspace/datasets/DAICWOZ") / f"{i}_P" / f"{i}_aSplits"
        fpath.mkdir(parents=True, exist_ok=True)
        for row in x.itertuples():
            p = fpath / f"{row.Index + 2}_{row.speaker}.wav"
            if p.exists():
                continue
            s_frame = int(row.start_time * sr)
            n_frame = int((row.stop_time - row.start_time) * sr)
            waveform, _ = torchaudio.load(audiofilePath, frame_offset=s_frame,
                                          num_frames=n_frame)
            torchaudio.save(p, waveform, sr)
        print(f"(aP)patient{i} finish")


# ============================================================
# DistilBERT (text, segment-level) — 加入 z-score 校正 + soft prob
# ============================================================
def _distilbert_probs(classifier, text: str) -> np.ndarray:
    result = classifier(text, top_k=None, batch_size=24)
    d = {r["label"]: r["score"] for r in result}
    return np.array([d.get("positive", 0.0),
                     d.get("negative", 0.0),
                     d.get("neutral",  0.0)], dtype=np.float32)


def DISTILBERT(ds: str, ds_dir: str, device: str, split: str, **kw) -> None:
    print("\n**DistilBert (segment-level, z-score 校正 + soft prob)**")
    classifier = pipeline(
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

    patient_ids = get_patient_ids(split)
    pid_arr, sid_arr, prob_arr = [], [], []

    for i in patient_ids:
        filePath = f"{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv"
        if not os.path.exists(filePath):
            print(f"PATH: {filePath} does not exist")
            continue
        x = pd.read_csv(filePath, sep="\t")
        x = x[x["speaker"] == "Participant"].dropna(subset=["value"]).copy()
        cnt = 0
        for row in x.itertuples():
            seg_id = row.Index + 2
            probs = _distilbert_probs(classifier, row.value)
            pid_arr.append(i)
            sid_arr.append(seg_id)
            prob_arr.append(probs)
            cnt += 1
        print(f"=== (DB) patient {i} done -> {cnt} segments")

    prob_arr = np.stack(prob_arr, axis=0)
    label_arr, soft_prob_arr = zscore_and_argmax(prob_arr)

    out_path = f"DistilBert_{split}_seg_bin"
    np.savez(out_path,
             patientIdx=np.array(pid_arr, dtype=np.int64),
             segIdx=np.array(sid_arr, dtype=np.int64),
             label=label_arr,
             soft_prob=soft_prob_arr)
    print(f"saved: {out_path}.npz, total: {len(label_arr)}")
    print(f"  label dist (pos/neg/neu): {np.bincount(label_arr, minlength=3).tolist()}")


# ============================================================
# Wav2Vec2 (audio, segment-level) — 加入 z-score 校正 + soft prob
# ============================================================
def _sb_classify_probs(classifier, waveform, device) -> np.ndarray:
    with torch.no_grad():
        out_prob, _, _, _ = classifier.classify_batch(waveform.to(device))
    p = out_prob.cpu().numpy().squeeze()
    if p.ndim > 1:
        p = p.mean(axis=tuple(range(p.ndim - 1)))
    if p.min() < 0:
        p = np.exp(p)

    try:
        ind2lab = classifier.hparams.label_encoder.ind2lab
    except AttributeError:
        ind2lab = {0: 'neu', 1: 'ang', 2: 'hap', 3: 'sad'}

    name2p = {ind2lab[i]: float(p[i]) for i in range(len(p))}
    return np.array([name2p.get("hap", 0.0),
                     name2p.get("sad", 0.0) + name2p.get("ang", 0.0),
                     name2p.get("neu", 0.0)], dtype=np.float32)


def WAV2VEC2(ds: str, ds_dir: str, device: str, split: str, **kw) -> None:
    print("\n**WAV2VEC2 (segment-level, z-score 校正 + soft prob)**")
    sb_Path = Path(".sb_cache")
    sb_Path.mkdir(parents=True, exist_ok=True)
    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        savedir=sb_Path,
        run_opts={"device": device})

    patient_ids = get_patient_ids(split)
    pid_arr, sid_arr, prob_arr = [], [], []

    for i in patient_ids:
        p_path = Path(f"datasets/DAICWOZ/{i}_P/{i}_aSplits")
        wavFiles = sorted(p_path.glob("*.wav"),
                          key=lambda p: int(p.stem.split("_")[0]))
        if len(wavFiles) == 0:
            print(f"patient{i} no wav splits")
            continue
        cnt = 0
        for wf in wavFiles:
            seg_id = int(wf.stem.split("_")[0])
            waveform, _ = torchaudio.load(str(wf))
            probs = _sb_classify_probs(classifier, waveform, device)
            pid_arr.append(i)
            sid_arr.append(seg_id)
            prob_arr.append(probs)
            cnt += 1
        print(f"=== (WV) patient {i} done -> {cnt} segments")

    prob_arr = np.stack(prob_arr, axis=0)
    label_arr, soft_prob_arr = zscore_and_argmax(prob_arr)

    out_path = f"Wav2Vec2_{split}_seg_bin"
    np.savez(out_path,
             patientIdx=np.array(pid_arr, dtype=np.int64),
             segIdx=np.array(sid_arr, dtype=np.int64),
             label=label_arr,
             soft_prob=soft_prob_arr)
    print(f"saved: {out_path}.npz, total: {len(label_arr)}")
    print(f"  label dist (pos/neg/neu): {np.bincount(label_arr, minlength=3).tolist()}")


# ============================================================
# HowNet (text lexicon, segment-level) — count-based,做 z-score
# ============================================================
def HOWNET_txt():
    HNdict = {}
    curWord = None
    with open("/workspace/datasets/HowNetDict/HowNet.txt", "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("W_E="):
                curWord = line.split("=")[1].lower()
                if curWord not in HNdict:
                    HNdict[curWord] = []
            elif line.startswith("S_E=") and curWord:
                sentiment = line.split("=")[1]
                if sentiment:
                    sentiment = sentiment.split("|")[0]
                    HNdict[curWord] = sentiment
    return HNdict


def HOWNET(ds: str, ds_dir: str, device: str, split: str, **kw):
    print("\n**HOWNET (segment-level, z-score 校正)**")
    HNdict = HOWNET_txt()
    patient_ids = get_patient_ids(split)
    pid_arr, sid_arr, cnt_arr = [], [], []

    for i in patient_ids:
        filePath = f"{ds_dir}/{i}_P/{i}_TRANSCRIPT.csv"
        if not os.path.exists(filePath):
            print(f"PATH: {filePath} does not exist")
            continue
        x = pd.read_csv(filePath, sep="\t")
        x = x[x["speaker"] == "Participant"].dropna(subset=["value"]).copy()
        cnt = 0
        for row in x.itertuples():
            seg_id = row.Index + 2
            pos = neg = neu = 0
            for k in row.value.lower().split():
                if k not in HNdict:
                    neu += 1
                elif "Plus" in HNdict[k]:
                    pos += 1
                elif "Minus" in HNdict[k]:
                    neg += 1
                else:
                    neu += 1
            pid_arr.append(i)
            sid_arr.append(seg_id)
            cnt_arr.append([pos, neg, neu])
            cnt += 1
        print(f"=== (HN) patient {i} done -> {cnt} segments")

    cnt_arr = np.array(cnt_arr, dtype=np.float32)
    label_arr, soft_prob_arr = zscore_and_argmax(cnt_arr)

    out_path = f"HowNet_{split}_seg_bin"
    np.savez(out_path,
             patientIdx=np.array(pid_arr, dtype=np.int64),
             segIdx=np.array(sid_arr, dtype=np.int64),
             label=label_arr,
             soft_prob=soft_prob_arr)
    print(f"saved: {out_path}.npz, total: {len(label_arr)}")
    print(f"  label dist (pos/neg/neu): {np.bincount(label_arr, minlength=3).tolist()}")


# ============================================================
# EATD: audio + text segment-level emotion
# ============================================================
def _read_eatd_dep(vol_dir: Path, cutoff: float = 53.0):
    lbl = vol_dir / "new_label.txt"
    if not lbl.exists():
        return None
    try:
        return 1 if float(lbl.read_text().strip()) >= cutoff else 0
    except ValueError:
        return None


def _get_eatd_vol_dirs():
    return sorted(
        [d for d in EATD_DIR.iterdir()
         if d.is_dir() and (d.name.startswith("t_") or d.name.startswith("v_"))],
        key=lambda d: (d.name[0], int(d.name.split("_")[1])))


def EATD_DISTILBERT(device: str):
    print("\n**EATD DistilBert (z-score 校正 + soft prob)**")
    classifier = pipeline(
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student")

    pid_arr, sid_arr, prob_arr = [], [], []
    for vol_dir in _get_eatd_vol_dirs():
        if _read_eatd_dep(vol_dir) is None:
            continue
        vol = vol_dir.name
        for seg_id, prompt in enumerate(EATD_PROMPTS):
            txt_path = vol_dir / f"{prompt}.txt"
            if not txt_path.exists():
                print(f"  [skip] {vol}/{prompt}.txt missing")
                continue
            text = txt_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            probs = _distilbert_probs(classifier, text)
            pid_arr.append(vol)
            sid_arr.append(seg_id)
            prob_arr.append(probs)
        print(f"=== (DB-EATD) {vol} done")

    prob_arr = np.stack(prob_arr, axis=0)
    label_arr, soft_prob_arr = zscore_and_argmax(prob_arr)

    out_path = "DistilBert_eatd_seg_bin"
    np.savez(out_path,
             patientIdx=np.array(pid_arr, dtype=object),
             segIdx=np.array(sid_arr, dtype=np.int64),
             label=label_arr,
             soft_prob=soft_prob_arr)
    print(f"saved: {out_path}.npz, total: {len(label_arr)}")
    print(f"  label dist (pos/neg/neu): {np.bincount(label_arr, minlength=3).tolist()}")


def EATD_WAV2VEC2(device: str):
    print("\n**EATD WAV2VEC2 (z-score 校正 + soft prob)**")
    sb_Path = Path(".sb_cache")
    sb_Path.mkdir(parents=True, exist_ok=True)
    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        savedir=sb_Path,
        run_opts={"device": device})

    pid_arr, sid_arr, prob_arr = [], [], []
    for vol_dir in _get_eatd_vol_dirs():
        if _read_eatd_dep(vol_dir) is None:
            continue
        vol = vol_dir.name
        for seg_id, prompt in enumerate(EATD_PROMPTS):
            wav = vol_dir / f"{prompt}_out.wav"
            if not wav.exists():
                wav = vol_dir / f"{prompt}.wav"
            if not wav.exists():
                print(f"  [skip seg] {vol}/{prompt}: no wav")
                continue
            waveform, _ = torchaudio.load(str(wav))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if waveform.shape[-1] < 1600:
                print(f"  [skip seg] {vol}/{prompt}: too short")
                continue
            try:
                probs = _sb_classify_probs(classifier, waveform, device)
            except Exception as e:
                print(f"  [skip seg] {vol}/{prompt}: {e}")
                continue
            pid_arr.append(vol)
            sid_arr.append(seg_id)
            prob_arr.append(probs)
        print(f"=== (WV-EATD) {vol} done")

    prob_arr = np.stack(prob_arr, axis=0)
    label_arr, soft_prob_arr = zscore_and_argmax(prob_arr)

    out_path = "Wav2Vec2_eatd_seg_bin"
    np.savez(out_path,
             patientIdx=np.array(pid_arr, dtype=object),
             segIdx=np.array(sid_arr, dtype=np.int64),
             label=label_arr,
             soft_prob=soft_prob_arr)
    print(f"saved: {out_path}.npz, total: {len(label_arr)}")
    print(f"  label dist (pos/neg/neu): {np.bincount(label_arr, minlength=3).tolist()}")


if __name__ == "__main__":
    args = parse_args()
    args.ds_dir = os.path.join(args.ds_dir, args.ds)

    DISTILBERT(**vars(args))
    audioPreprosessing(**vars(args))
    WAV2VEC2(**vars(args))
    if args.run_hownet:
        HOWNET(**vars(args))

    if args.run_eatd:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        EATD_DISTILBERT(device)
        EATD_WAV2VEC2(device)