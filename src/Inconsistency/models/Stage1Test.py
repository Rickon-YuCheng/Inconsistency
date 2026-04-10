import numpy as np
import torch
from Stage1Tr import atei, ateiDataset, collate_fn, D_MODEL, NHEAD, TINYTEST
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, Subset
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def testDS():
    patientIdx=np.load("PseudoLabel.npz")["patientIdx"]
    PseudoL=np.load("PseudoLabel.npz")["label"]
    PseudoMap = {int(x): int(y) for x, y in zip(patientIdx, PseudoL)}

    ds=ateiDataset(pseudoMap=PseudoMap) 
    testIdx=np.load("stage1Split.npz")["testIdx"]
    testDS=Subset(ds,testIdx)
    return testDS

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    TestDS=testDS()
    TestLoader=DataLoader(TestDS, batch_size=1, collate_fn=collate_fn, shuffle=False)
    model = atei(D_MODEL, NHEAD).to(device)
    model.load_state_dict(torch.load("atei_stage1_weights.pth", map_location=device))
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in TestLoader:
            xa, xt, aMask, tMask, label = [d.to(device) for d in data]
            _, logits = model(xa, xt, aMask, tMask)   # [N_seg, 2]

            seg_pred = torch.argmax(logits, dim=-1)   # [N_seg]
            votes = torch.bincount(seg_pred, minlength=2)
            patient_pred = torch.argmax(votes).item()

            y_true.append(label.item())
            y_pred.append(patient_pred)

    print(f"Test Accuracy : {accuracy_score(y_true, y_pred)}")
    print(f"Test Precision: {precision_score(y_true, y_pred, average='binary', zero_division=0)}")
    print(f"Test Recall   : {recall_score(y_true, y_pred, average='binary', zero_division=0)}")
    print(f"Test F1       : {f1_score(y_true, y_pred, average='binary', zero_division=0)}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()