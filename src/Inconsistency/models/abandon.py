'''
! 棄用，需要建立一個 classification 並將 ATEI embd 當作 inp 傳入該 classification 並進行 pred on test_GroundTrue
  所以要建一個 LogisticRegression model 做三元分類

ATEI output -> test_GroundTrue 
'''
import numpy as np
import torch
from Stage1Tr import atei, ateiDataset, collate_fn, D_MODEL, NHEAD, TINYTEST
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, Subset
from Inconsistency.datasets.inconsistentLabel import get_Split_and_GroundTrue
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def testDS():
    depMap,_,test_idx=get_Split_and_GroundTrue()
    testMap={int(x): int(y) for x,y in zip(test_idx, depMap[test_idx])}

    # patientIdx, PseudoL=np.load("PseudoLabel.npz")["patientIdx"], np.load("PseudoLabel.npz")["label"]
    # PseudoMap = {int(x): int(y) for x, y in zip(patientIdx, PseudoL)}

    ds=ateiDataset(pseudoMap=testMap) 
    # testIdx=np.load("stage1Split.npz")["testIdx"]
    # testDS=Subset(ds,testIdx)
    # return testDS
    return ds
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    TestDS=testDS()
    TestLoader=DataLoader(TestDS, collate_fn=collate_fn)
    model = atei(D_MODEL, NHEAD).to(device)
    model.load_state_dict(torch.load("stage1Weights.pth", map_location=device))
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in TestLoader:
            xa, xt, aMask, tMask, label = [d.to(device) for d in data]
            _, logits = model(xa, xt, aMask, tMask)   # [N_seg, 2]

            patient_logit = logits.mean(dim=0)        # [2]
            patient_pred = torch.argmax(patient_logit).item()

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