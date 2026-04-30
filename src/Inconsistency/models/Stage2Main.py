import numpy as np
import torch
from Stage1Tr import atei, ateiDataset, collate_fn, TINYTEST
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from Inconsistency.datasets.inconsistentLabel import get_Split_and_GroundTrue
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

D_MODEL=128
NHEAD=8
LR=1e-5
EPOCHS=5
TRANSFORMER_ENC_LAYERS=2

class Stage2ValDataset(Dataset):
    """
    只需要 depression label
    """

    def __init__(self, depMap, TINYTEST=None):
        self.samples = []

        self.a_root = Path("datasets/Feature/HuBERT")
        self.t_root = Path("datasets/Feature/RoBerTa")

        if TINYTEST is not None: patientList = sorted(TINYTEST)
        else: patientList = sorted(depMap.keys())

        for patient in patientList:
            patient = int(patient)

            a_path = self.a_root / f"{patient}_acoustic.pt"
            t_path = self.t_root / f"{patient}_text.pt"

            if not a_path.exists():
                print(f"[Val Skip] missing acoustic: {a_path}")
                continue

            if not t_path.exists():
                print(f"[Val Skip] missing text: {t_path}")
                continue

            if patient not in depMap:
                print(f"[Val Skip] missing dep label: {patient}")
                continue
            self.samples.append(patient)
        self.depMap = depMap

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        patient = self.samples[idx]

        a_path = self.a_root / f"{patient}_acoustic.pt"
        t_path = self.t_root / f"{patient}_text.pt"

        xa = torch.load(a_path).float()
        xt = torch.load(t_path).float()

        if xa.dim() == 3 and xa.size(0) == 1:
            xa = xa.squeeze(0)

        if xt.dim() == 3 and xt.size(0) == 1:
            xt = xt.squeeze(0)

        xa = xa.unsqueeze(0)
        xt = xt.unsqueeze(0)

        aMask = torch.zeros((1, xa.size(1)), dtype=torch.bool)
        tMask = torch.zeros((1, xt.size(1)), dtype=torch.bool)

        dep_label = torch.tensor(self.depMap[patient], dtype=torch.long)

        return patient, xa, xt, aMask, tMask, dep_label

class whole_model(nn.Module):
    '''
    Transformer: (S: source sequence length, T: tgt seq_len, N: batch size, E: feature number)
    '''
    def __init__(self,embd_size=D_MODEL,nheads=NHEAD):
        super().__init__()
        self.in_proj=nn.Linear(1024, embd_size) # Because HuBERT and Wav2Vec2 oup are 1024 dim
        self.atei=atei(embd_size=embd_size,nheads=nheads)
        self.atei.load_state_dict(torch.load("stage1Weights.pth"))
        a_enc_layer=nn.TransformerEncoderLayer(d_model=embd_size, nhead=nheads,batch_first=True) # # (N, T, E)
        t_enc_layer=nn.TransformerEncoderLayer(d_model=embd_size, nhead=nheads,batch_first=True) # # (N, T, E)
        self.a_transformer_enc=nn.TransformerEncoder(a_enc_layer,num_layers=TRANSFORMER_ENC_LAYERS) #12
        self.t_transformer_enc=nn.TransformerEncoder(t_enc_layer,num_layers=TRANSFORMER_ENC_LAYERS) #12
        self.fc1=nn.Linear(3*embd_size,embd_size)
        self.fc2=nn.Linear(embd_size,embd_size)
        self.fc3=nn.Linear(embd_size, embd_size)
        self.oup=nn.Linear(embd_size,3)
    def forward(self, XA, XT, aMask=None, tMask=None):
        XA_raw = XA
        XT_raw = XT

        XA = self.in_proj(XA_raw)
        XT = self.in_proj(XT_raw)


        # Stage 3: Depression-related Feature Extraction
        def run_a_encoder(x):
            return self.a_transformer_enc(x, src_key_padding_mask=aMask)
        def run_t_encoder(x):
            return self.t_transformer_enc(x, src_key_padding_mask=tMask)
        def run_atei(xa_raw, xt_raw):
            eE, atei_logit = self.atei(xa_raw, xt_raw, aMask, tMask)
            return eE, atei_logit
        
        
        # *** XA to eA ***
        # Transformer-based Acoustic Feature Aggregation & Textual
        HA = checkpoint(run_a_encoder, XA, use_reentrant=False)
        HT = checkpoint(run_t_encoder, XT, use_reentrant=False)
        eA=self.masked_mean(HA, aMask)
        eT=self.masked_mean(HT, tMask)

        # *** XA & eA to eE ***
        eE, atei_logit = checkpoint(run_atei,XA_raw,XT_raw,use_reentrant=False)
        

        # Stage4: Fusion and Depression Detection
        eFusion=torch.cat((eA,eE,eT),dim=1)
        Fc1=F.relu(self.fc1(eFusion))
        Fc2=F.relu(self.fc2(Fc1))
        Fc3=F.relu(self.fc3(Fc2))
        Oup=self.oup(Fc3)
        return atei_logit, Oup
    def masked_mean(self, x, mask):
        if mask is None: return x.mean(dim=1)

        valid = (~mask).unsqueeze(-1)          # [B, T, 1]
        x = x * valid
        denom = valid.sum(dim=1).clamp(min=1)  # [B, 1]
        return x.sum(dim=1) / denom
    

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # For atei pseudo label
    PseudoLabel=np.load("PseudoLabel.npz")
    patientIdx, PseudoL = PseudoLabel["patientIdx"], PseudoLabel["label"]
    pseudoMap = {int(x): int(y) for x, y in zip(patientIdx, PseudoL)}

    # For tot ground true lable
    depMap,trIdx,valIdx=get_Split_and_GroundTrue()
    trainPseudoMap = {int(pid): int(pseudoMap[int(pid)]) for pid in trIdx if int(pid) in pseudoMap}
    trainDepMap = {int(pid): int(depMap[int(pid)]) for pid in trIdx if int(pid) in depMap}
    valPseudoMap = {int(pid): int(pseudoMap[int(pid)]) for pid in valIdx if int(pid) in pseudoMap}
    valDepMap = {int(pid): int(depMap[int(pid)]) for pid in valIdx if int(pid) in depMap}
    # depMap = {int(x): int(depMap[int(x)]) for x in trIdx}

    # 1. Dataset
    ds=ateiDataset(pseudoMap=trainPseudoMap,depMap=trainDepMap, TINYTEST=TINYTEST)
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


if __name__ == "__main__":
    main()