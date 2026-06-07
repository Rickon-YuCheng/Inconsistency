"""
Logistic regression baseline (3-fold)

跟 Stage2 一樣的 CV pool (官方 tr+dev 混切),用最簡單的 LR 看看
單純拿 HuBERT/RoBERTa pooled feature 能跑到多少。
"""
import numpy as np
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold

from stage2_data import build_pool_arrays
from Inconsistency.datasets.inconsistentLabel_bin import get_Split_and_GroundTrue


SEED = 42
N_SPLITS = 3


def main():
    depMap, tr, te = get_Split_and_GroundTrue()
    all_ids = tr + te

    X, y, pids = build_pool_arrays(patient_ids=all_ids)
    print(f"Total: {len(y)} patients, label dist: {Counter(y.tolist())}")
    print(f"Feature dim: {X.shape[1]}")

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_f1s, fold_accs = [], []

    for fold, (tr_i, val_i) in enumerate(skf.split(X, y)):
        Xtr, Xval = X[tr_i], X[val_i]
        ytr, yval = y[tr_i], y[val_i]

        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xval_s = sc.transform(Xtr), sc.transform(Xval)

        clf = LogisticRegression(
            max_iter=2000, class_weight="balanced",
            C=1.0, random_state=SEED,
        ).fit(Xtr_s, ytr)

        pred = clf.predict(Xval_s)
        f1 = f1_score(yval, pred, average="binary", pos_label=1, zero_division=0)
        acc = (pred == yval).mean()
        fold_f1s.append(f1)
        fold_accs.append(acc)

        print(f"\n--- Fold {fold} ---")
        print(f"train dist: {Counter(ytr.tolist())}, val dist: {Counter(yval.tolist())}")
        print(f"Acc: {acc:.4f} | MacroF1: {f1:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(yval, pred, labels=[0, 1]))
        print(classification_report(yval, pred, labels=[0, 1], digits=4,
                                    zero_division=0))

    print("\n" + "=" * 50)
    print("LR BASELINE (3-fold)")
    print("=" * 50)
    for i, (f1, acc) in enumerate(zip(fold_f1s, fold_accs)):
        print(f"Fold {i}: F1={f1:.4f}  Acc={acc:.4f}")
    print(f"\nMean F1: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")
    print(f"Mean Acc: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")


if __name__ == "__main__":
    main()