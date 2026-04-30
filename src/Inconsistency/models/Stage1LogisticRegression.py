''' 重現paper說單一ATEI資訊無法作為模型，作者提到acc僅40% '''

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from Stage1Tr import atei, ateiDataset, collate_fn, D_MODEL, NHEAD
from Inconsistency.datasets.inconsistentLabel import get_Split_and_GroundTrue
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def build_ds():
    depMap, tr_idx, test_idx = get_Split_and_GroundTrue()
    tr_map = {int(x): int(depMap[x]) for x in tr_idx}
    te_map = {int(x): int(depMap[x]) for x in test_idx}

    train_ds = ateiDataset(pseudoMap=tr_map)
    test_ds = ateiDataset(pseudoMap=te_map)
    return train_ds, test_ds


def extract_embedding(ds, model, device):
    loader = DataLoader(ds, collate_fn=collate_fn)
    X, y = [], []

    model.eval()
    with torch.no_grad():
        for data in loader:
            xa, xt, aMask, tMask, label, dep_label = [d.to(device) for d in data]

            emb, _ = model(xa, xt, aMask, tMask)   # emb: ATEI embedding
            patient_emb = emb.mean(dim=0).cpu().numpy()

            X.append(patient_emb)
            y.append(label.item())

    return np.array(X), np.array(y)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_ds, test_ds = build_ds()

    model = atei(D_MODEL, NHEAD).to(device)
    model.load_state_dict(torch.load("stage1Weights.pth", map_location=device))
    model.eval()

    X_train, y_train = extract_embedding(train_ds, model, device)
    X_test, y_test = extract_embedding(test_ds, model, device)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"Test Accuracy : {accuracy_score(y_test, y_pred)}")


if __name__ == "__main__":
    main()