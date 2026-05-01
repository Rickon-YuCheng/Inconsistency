import numpy as np
import torch
from Stage1Tr import atei, daicwoz_dataset, collate_fn
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
from tqdm import tqdm
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
TRANSFORMER_ENC_LAYERS=1


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
        self.alpha = nn.Parameter(torch.ones(embd_size))
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
        
        # Formula 15: apply learnable alpha to ATEI feature
        eE = eE * self.alpha.unsqueeze(0)

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
    

def train_one_epoch(model, tr_loader, loss_atei, loss_dep, opt, device, cur_epoch, tot_epochs,scaler):
    model.train()
    totAteiLoss=totDepLoss=totLoss=0.0
    correct_atei=correct_dep=valid_batches=0
    pbar= tqdm(tr_loader, desc=f"Training epoch {cur_epoch}/{tot_epochs}",leave=False, unit='batch')
    
    for data in pbar:
        
        xa, xt, aMask, tMask, atei_label, dep_label, Patient = data

        xa = xa.to(device)
        xt = xt.to(device)
        aMask = aMask.to(device)
        tMask = tMask.to(device)
        atei_label = atei_label.to(device)
        dep_label = dep_label.to(device)
        opt.zero_grad()
        with torch.autocast('cuda'):
            atei_logits, dep_logits=model(xa,xt,aMask,tMask) # logits:　[LenFeat,2] eg: [89,2]
        
            patient_atei=atei_logits.mean(dim=0) # torch.Size([2])
            patient_dep=dep_logits.mean(dim=0)
                    
            L_Atei = loss_atei(patient_atei.unsqueeze(0), atei_label.unsqueeze(0)) # 加batch
            L_Depression = loss_dep(patient_dep.unsqueeze(0), dep_label.unsqueeze(0)) # 加batch
            L_Total=L_Atei+L_Depression
                
        scaler.scale(L_Total).backward()
        scaler.step(opt)
        scaler.update()

        # Loss
        totAteiLoss += L_Atei.item()
        totDepLoss += L_Depression.item()
        totLoss += L_Total.item()

        # Acc       # argmax -> return idx
        atei_pred = patient_atei.argmax()  # return tensor([ 0.2683, -0.1693]
        dep_pred = patient_dep.argmax() # return tensor([-0.0456,  0.0038,  0.0627]
        correct_atei += int(atei_pred.item() == atei_label.item())
        correct_dep += int(dep_pred.item() == dep_label.item())
        valid_batches += 1

        pbar.set_postfix({
            "atei loss": totAteiLoss/valid_batches,
            "dep loss": totDepLoss/valid_batches,
            "tot loss": totLoss/valid_batches,
            "cur atei acc": correct_atei/valid_batches,
            "cur dep acc": correct_dep/valid_batches
        })
    return {"atei_loss": totAteiLoss/valid_batches,
            "dep_loss": totDepLoss/valid_batches,
            "tot_loss": totLoss/valid_batches,
            "cur_atei_acc": correct_atei/valid_batches,
            "cur_dep_acc": correct_dep/valid_batches}

def val(model, val_loader, loss_dep, device, cur_epoch, tot_epochs):
    model.eval()

    totDepLoss = 0.0
    correct_dep = 0
    valid_batches = 0

    pbar = tqdm(val_loader,desc=f"Validation epoch {cur_epoch}/{tot_epochs}",leave=False,unit="batch",)

    with torch.inference_mode():
        for data in pbar:
            if data is None: continue

            xa, xt, aMask, tMask, atei_label, dep_label, Patient = data

            xa = xa.to(device)
            xt = xt.to(device)
            aMask = aMask.to(device)
            tMask = tMask.to(device)
            dep_label = dep_label.to(device)

            with torch.autocast(device_type="cuda",enabled=(device == "cuda"),):
                _, dep_logits = model(xa, xt, aMask, tMask)

                patient_dep = dep_logits.mean(dim=0)

                L_Depression = loss_dep(patient_dep.unsqueeze(0),dep_label.unsqueeze(0))

            dep_pred = patient_dep.argmax(dim=-1)
            totDepLoss += L_Depression.item()
            correct_dep += int(dep_pred.item() == dep_label.item())
            valid_batches += 1

            pbar.set_postfix({
                "dep_loss": totDepLoss / valid_batches,
                "dep_acc": correct_dep / valid_batches,
            })

    return {
        "dep_loss": totDepLoss / max(valid_batches, 1),
        "dep_acc": correct_dep / max(valid_batches, 1),
    }

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Dataset
    trDS=daicwoz_dataset(fold="tr")
    tr_loader=DataLoader(trDS,collate_fn=collate_fn)
    valDS=daicwoz_dataset(fold="val")
    val_loader=DataLoader(valDS,collate_fn=collate_fn)

    # 2. Model parameter setting
    model=whole_model(D_MODEL,NHEAD).to(device)
    loss_atei=nn.CrossEntropyLoss()
    loss_dep=nn.CrossEntropyLoss()
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    scaler = torch.GradScaler('cuda')
    best_val_acc=-1.0

    tr_history=[]
    val_history=[]
    # 3. Train
    for epoch in range(1,EPOCHS+1):
        tr_result=train_one_epoch(model, tr_loader, loss_atei, loss_dep, opt, device, epoch, EPOCHS,scaler)
        val_result=val(model, val_loader, loss_dep, device, epoch, EPOCHS)

        tr_history.append(tr_result)
        val_history.append(val_result)

        print("=" * 80)
        print(f"Epoch [{epoch}/{EPOCHS}]")

        print(
            f"[Train] "
            f"ATEI Loss: {tr_result['atei_loss']:.4f} | "
            f"Dep Loss: {tr_result['dep_loss']:.4f} | "
            f"Total Loss: {tr_result['tot_loss']:.4f} | "
            f"ATEI Acc: {tr_result['cur_atei_acc']:.4f} | "
            f"Dep Acc: {tr_result['cur_dep_acc']:.4f}"
        )

        print(
            f"[Val]   "
            f"Dep Loss: {val_result['dep_loss']:.4f} | "
            f"Dep Acc: {val_result['dep_acc']:.4f}"
        )

        if val_result["dep_acc"] > best_val_acc:
            best_val_acc = val_result["dep_acc"]
            torch.save(model.state_dict(),"stage2BestWeights.pth")
            print(f"[Save Best] Val Dep Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()