"""load npz to calculate z-score"""

from sklearn.preprocessing import StandardScaler
import numpy as np


def T_HN_zscores():
    x = np.load("HowNet.npz")
    x = np.column_stack((x["a"], x["b"], x["c"]))
    X = np.nan_to_num(x, nan=0)

    scaler = StandardScaler()
    scaler.fit(X)
    X_zscores = scaler.transform(X)
    t_labels = np.argmax(X_zscores, axis=1)

    print(f"平均: {scaler.mean_}")
    print(t_labels)
    breakpoint()
    return t_labels


def T_zscores():
    x = np.load("DistilBert.npz")
    x = np.column_stack((x["a"], x["b"], x["c"]))
    X = np.nan_to_num(x, nan=0)

    scaler = StandardScaler()
    scaler.fit(X)
    X_zscores = scaler.transform(X)
    t_labels = np.argmax(X_zscores, axis=1)

    print(f"平均: {scaler.mean_}")
    print(t_labels)
    breakpoint()
    return t_labels


def A_zscores():
    x = np.load("Wav2Vec2.npz")
    x = np.column_stack((x["a"], x["b"], x["c"]))
    X = np.nan_to_num(x, nan=0)
    scaler = StandardScaler()
    scaler.fit(X)
    X_zscores = scaler.transform(X)
    print(f"平均: {scaler.mean_}")
    a_labels = np.argmax(X_zscores, axis=1)
    print(a_labels)
    breakpoint()
    return a_labels


def INCONSISTENCY():
    # T = T_zscores()
    T = T_HN_zscores()
    A = A_zscores()
    result = []
    Con = Incon = 0
    for i in range(0, len(T)):
        if T[i] == A[i]:
            result.append(0)  # 一致
            Con += 1
        elif T[i] != A[i]:
            result.append(1)  # 不一致
            Incon += 1
    print(f"result: {result}")
    print(f"一致: {Con} 不一致:{Incon}")


if __name__ == "__main__":
    INCONSISTENCY()
