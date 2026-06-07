"""
XGBoost baseline (3-fold)

跟 lr_baseline_bin.py 同條件:
  - 同 seed (42)
  - 同 pool (官方 tr+dev 混切)
  - 同特徵 (audio pooled mean + text pooled mean = 2048)
  - 同 StratifiedKFold

scale_pos_weight 用 train fold 的 neg/pos 比, 處理類別不平衡。
"""
import numpy as np
from collections import Counter
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from stage2_data import build_pool_arrays
from Inconsistency.datasets.inconsistentLabel_bin import get_Split_and_GroundTrue


SEED = 42
N_SPLITS = 3


def main():
    _, tr, te = get_Split_and_GroundTrue()
    all_ids = tr + te

    X, y, pids = build_pool_arrays(patient_ids=all_ids)
    print(f"Total: {len(y)} patients, label dist: {Counter(y.tolist())}")
    print(f"Feature dim: {X.shape[1]}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_f1s, fold_accs = [], []

    for fold, (tr_i, val_i) in enumerate(skf.split(X, y)):
        Xtr, Xval = X[tr_i], X[val_i]
        ytr, yval = y[tr_i], y[val_i]

        # 類別不平衡: scale_pos_weight = neg / pos
        n_pos = int((ytr == 1).sum())
        n_neg = int((ytr == 0).sum())
        spw = n_neg / max(n_pos, 1)

        clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            scale_pos_weight=spw,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
            tree_method="hist",
        ).fit(Xtr, ytr)

        pred = clf.predict(Xval)
        f1 = f1_score(yval, pred, average="binary", pos_label=1, zero_division=0)
        acc = (pred == yval).mean()
        fold_f1s.append(f1)
        fold_accs.append(acc)

        print(f"\n--- Fold {fold} ---")
        print(f"train dist: {Counter(ytr.tolist())}, val dist: {Counter(yval.tolist())}")
        print(f"scale_pos_weight: {spw:.4f}")
        print(f"Acc: {acc:.4f} | MacroF1: {f1:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(yval, pred, labels=[0, 1]))
        print(classification_report(yval, pred, labels=[0, 1], digits=4,
                                    zero_division=0))

    print("\n" + "=" * 50)
    print("XGBoost BASELINE (3-fold)")
    print("=" * 50)
    for i, (f1, acc) in enumerate(zip(fold_f1s, fold_accs)):
        print(f"Fold {i}: F1={f1:.4f}  Acc={acc:.4f}")
    print(f"\nMean F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"Mean Acc: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")


if __name__ == "__main__":
    main()