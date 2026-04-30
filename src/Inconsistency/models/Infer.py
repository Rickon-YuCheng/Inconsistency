import numpy as np
import torch
from Stage1Tr import atei, ateiDataset, collate_fn, TINYTEST
from Stage2Tr import whole_model
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

D_MODEL=128
NHEAD=8
LR=1e-5
EPOCHS=2
TRANSFORMER_ENC_LAYERS=1

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # For atei pseudo label
    # PseudoLabel=np.load("PseudoLabel.npz")
    # patientIdx, PseudoL = PseudoLabel["patientIdx"], PseudoLabel["label"]
    # pseudoMap = {int(x): int(y) for x, y in zip(patientIdx, PseudoL)}

    # For tot ground true lable
    depMap,_,testIdx=get_Split_and_GroundTrue()
    # testPseudoMap = {int(pid): int(pseudoMap[int(pid)]) for pid in testIdx if int(pid) in pseudoMap}
    testDepMap = {int(pid): int(depMap[int(pid)]) for pid in testIdx if int(pid) in depMap}
    # depMap = {int(x): int(depMap[int(x)]) for x in testIdx}

    # 1. Dataset
    ds=ateiDataset(pseudoMap=testPseudoMap,depMap=testDepMap, TINYTEST=TINYTEST)
    dataLoader=DataLoader(ds,collate_fn=collate_fn)

    # 2. Model parameter setting
    model=whole_model(D_MODEL,NHEAD).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    criterion_atei=nn.CrossEntropyLoss()
    criterion_dep=nn.CrossEntropyLoss()
    scaler = torch.GradScaler('cuda')
    

    # 3. Train
    model.train()

    arr_atei_loss = []
    arr_dep_loss = []
    arr_tot_loss = []
    arr_atei_acc = []
    arr_dep_acc = []

    for epoch in range(EPOCHS):
        totAteiLoss=totDepLoss=totLoss=0.0
        correct_atei=correct_dep=valid_batches=0

        for data in dataLoader:
            if data is None: continue
            xa, xt, aMask, tMask, atei_label, dep_label = [d.to(device) for d in data]
            opt.zero_grad()
            with torch.autocast('cuda'):
                atei_logits, dep_logits=model(xa,xt,aMask,tMask) # logits:　[LenFeat,2] eg: [89,2]
                
                patient_atei=atei_logits.mean(dim=0) # torch.Size([2])
                patient_dep=dep_logits.mean(dim=0)
                
                L_Atei = criterion_atei(patient_atei.unsqueeze(0), atei_label.unsqueeze(0)) # 加batch
                L_Depression = criterion_dep(patient_dep.unsqueeze(0), dep_label.unsqueeze(0)) # 加batch
                L_Total=L_Atei+L_Depression

            scaler.scale(L_Total).backward()
            scaler.step(opt)
            scaler.update()

            # Loss
            totAteiLoss += L_Atei.item()
            totDepLoss += L_Depression.item()
            totLoss += L_Total.item()

            # Acc       # argmax -> return idx
            atei_pred = patient_atei.argmax(dim=-1)  # return tensor([ 0.2683, -0.1693]
            dep_pred = patient_dep.argmax(dim=-1) # return tensor([-0.0456,  0.0038,  0.0627]
            correct_atei += int(atei_pred.item() == atei_label.item())
            correct_dep += int(dep_pred.item() == dep_label.item())
            valid_batches += 1

        avg_atei_loss = totAteiLoss / max(valid_batches, 1)
        avg_dep_loss = totDepLoss / max(valid_batches, 1)
        avg_total_loss = totLoss / max(valid_batches, 1)

        atei_acc = correct_atei / max(valid_batches, 1)
        dep_acc = correct_dep / max(valid_batches, 1)

        arr_atei_loss.append(avg_atei_loss)
        arr_dep_loss.append(avg_dep_loss)
        arr_tot_loss.append(avg_total_loss)
        arr_atei_acc.append(atei_acc)
        arr_dep_acc.append(dep_acc)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"ATEI Loss: {avg_atei_loss:.4f} | "
            f"Dep Loss: {avg_dep_loss:.4f} | "
            f"Total Loss: {avg_total_loss:.4f} | "
            f"ATEI Acc: {atei_acc:.4f} | "
            f"Dep Acc: {dep_acc:.4f}"
        )

        # history.append(totLoss/len(dataLoader))
        # print(f"Epoch [{epoch+1}/{EPOCHS}], Avg Loss: {totLoss/len(dataLoader):.4f}")
    print("=" * 80)
    print(f"arr_atei_loss: {arr_atei_loss}")
    print(f"arr_dep_loss: {arr_dep_loss}")
    print(f"arr_tot_loss: {arr_tot_loss}")
    print(f"arr_atei_acc: {arr_atei_acc}")
    print(f"arr_dep_acc: {arr_dep_acc}")
    torch.save(model.state_dict(), "stage2Weights.pth")

if __name__=="__main__":
    main()