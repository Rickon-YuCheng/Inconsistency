import torch.nn as nn
from pathlib import Path
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torch.utils.checkpoint import checkpoint
from Inconsistency.datasets.inconsistentLabel import get_Split_and_GroundTrue
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)

TINYTEST=None 
# TINYTEST=[300,301,302,303,304,305] # 小範圍測試, 要先刪掉stage1Split

D_MODEL=128
NHEAD=8
LR=1e-5
EPOCHS=5
TRANSFORMER_ENC_LAYERS=2
CHUNK_SIZE=1
# RANDOM_SEED=42

class ateiDataset(Dataset):
    def __init__(self, pseudoMap, depMap=None, TINYTEST=None):
        self.samples = []
        self.pseudoMap=pseudoMap
        self.depMap = depMap
        # self.dep_label,_,_=get_Split_and_GroundTrue()
        self.a_root = Path("datasets/Feature/HuBERT")
        self.t_root = Path("datasets/Feature/RoBerTa")

        if TINYTEST is None: patientList=sorted(pseudoMap.keys())
        else: patientList=sorted(TINYTEST)

        for patient in patientList:
            a_path = self.a_root / f"{patient}_acoustic.pt"
            t_path = self.t_root / f"{patient}_text.pt"

            if patient not in pseudoMap: continue
            if depMap is not None and patient not in depMap: continue
            if not a_path.exists() or not t_path.exists(): continue
            pseudo_label = int(pseudoMap[patient])
            if depMap is None: dep_label=pseudo_label
            else: dep_label=int(depMap[patient])
            self.samples.append((patient,pseudo_label,dep_label,a_path,t_path))
            
            # if a_path.exists() and t_path.exists() and patient in pseudoMap:
            #     self.samples.append((patient, pseudoMap[patient]))
        # breakpoint()
        

    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        Patient, PseudoL, DepL, a_path, t_path=self.samples[index]
        # dep_label = torch.tensor(int(self.dep_label[Patient]), dtype=torch.long)

        # a_path = self.a_root / f"{Patient}_acoustic.pt"
        # t_path = self.t_root / f"{Patient}_text.pt"

        # xa type:List(tensor)
        xa = torch.load(str(a_path))
        xt = torch.load(str(t_path))

        xa_list=[x.squeeze(0) for x in xa] # 這個人的每句話
        xt_list=[x.squeeze(0) for x in xt]

        atei_label = torch.tensor(PseudoL, dtype=torch.long)
        dep_label = torch.tensor(DepL, dtype=torch.long)

        return xa_list, xt_list, atei_label, dep_label, Patient#, a_path


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    
    xa_list,xt_list,pseudoL,dep_label,Patient=batch[0]
    print(f"==patient{Patient}")
    xa=pad_sequence(xa_list,batch_first=True) # (Pdb) p xa_padded[0,:,:] eg:這位病人的第0句話的長度進行填充
    xt=pad_sequence(xt_list,batch_first=True)

    # aMask = [N, seq_len]
    aMask=(xa.sum(dim=-1)==0) # (Pdb) p aMask[0,:] eg:若1024維特徵總和為0，則為Padding(True)
    tMask=(xt.sum(dim=-1)==0)

    return xa,xt,aMask,tMask,pseudoL,dep_label

# def get_split(ds):
#     # 切 tr 與 test
#     if not os.path.exists("stage1Split.npz"):
#         indices=np.arange(len(ds))
#         if len(indices)>1:
#             # rng=np.random.RandomState(RANDOM_SEED)
#             # rng.shuffle(indices)

#             split=int(len(indices)*0.8) # train_test_split
#             trIdx, testIdx = indices[:split], indices[split:]
#             np.savez("stage1Split", trIdx=trIdx, testIdx=testIdx)
#         else:
#             trIdx=indices
#             testIdx=indices
#     else:
#         splitDS=np.load("stage1Split.npz")
#         trIdx=splitDS["trIdx"]
#     return Subset(ds, trIdx)

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    PseudoLabel=np.load("PseudoLabel.npz")
    patientIdx, PseudoL = PseudoLabel["patientIdx"], PseudoLabel["label"]
    PseudoMap = {int(x): int(y) for x, y in zip(patientIdx, PseudoL)}

    # pseudoMap is built from the training split, so no additional split is needed here.
    ds=ateiDataset(pseudoMap=PseudoMap, TINYTEST=TINYTEST) 
    dataLoader=DataLoader(ds,collate_fn=collate_fn)#,shuffle=True)

    model=atei(D_MODEL,NHEAD).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    criterion=nn.CrossEntropyLoss()
    scaler = torch.GradScaler('cuda')
    history=[]

    model.train()
    for epoch in range(EPOCHS):
        totLoss=0

        for data in dataLoader:
            if data is None: continue
            xa, xt, aMask, tMask, atei_label, dep_label = [d.to(device) for d in data]
            opt.zero_grad()
            with torch.autocast('cuda'):
                _,logits=model(xa,xt,aMask,tMask) # logits:　[LenFeat,2] eg: [89,2]
                patient_logit=logits.mean(dim=0) # torch.Size([2])
                loss = criterion(patient_logit.unsqueeze(0), atei_label.unsqueeze(0)) # 加batch

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            totLoss += loss.item()
        history.append(totLoss/len(dataLoader))
        print(f"Epoch [{epoch+1}/{EPOCHS}], Avg Loss: {totLoss/len(dataLoader):.4f}")

    torch.save(model.state_dict(), "stage1Weights.pth")
    print("saved stage1Weights.pth")
    imsave(EPOCHS, history)

def imsave(EPOCHS, history):
    fig,ax=plt.subplots()
    ax.plot(range(EPOCHS),history)
    plt.xlabel('Epoch')
    plt.ylabel('BCELoss')
    plt.title('Training Loss')
    plt.savefig("stage1_tr_BCEloss.jpg")

class atei(nn.Module):
    def __init__(self,embd_size,nheads,inp_dim=1024):
        # super(atei,self).__init__()
        super().__init__()
        assert embd_size % nheads == 0, "Embedding size must be divisible by number of heads"
        self.in_proj=nn.Linear(inp_dim,embd_size) # Dynamic projection, Hubert and Wav2Vec2 oup are 1024 dim
        enc_layer=nn.TransformerEncoderLayer(d_model=embd_size, nhead=nheads,batch_first=True)
        self.transformer_enc=nn.TransformerEncoder(enc_layer,num_layers=TRANSFORMER_ENC_LAYERS) #12

        self.Cross_Attn=at_cross_attn(embd_size)

        self.fc1=nn.Linear(4*embd_size,embd_size)
        self.fc2=nn.Linear(embd_size,embd_size)
        self.fc3=nn.Linear(embd_size,embd_size)
        self.oup=nn.Linear(embd_size,2)
        

    def forward(self,xa, xt, aMask=None, tMask=None):
        xa=self.in_proj(xa)
        xt=self.in_proj(xt)
        XprimeA,XprimeT=[],[]

        def run_transformer(x, mask):
            return self.transformer_enc(x, src_key_padding_mask=mask)

        chunk_size = CHUNK_SIZE # split batch [87, 386, 1024]->[5, 386, 1024],[5, 386, 1024]..

        # Audio
        xa_chunks = torch.split(xa, chunk_size, dim=0)
        ma_chunks = torch.split(aMask, chunk_size, dim=0)
        
        XprimeA_list = []
        for x, m in zip(xa_chunks, ma_chunks):
            chunk_out = checkpoint(run_transformer, x, m, use_reentrant=False)
            XprimeA_list.append(chunk_out)
        XprimeA = torch.cat(XprimeA_list, dim=0)

        # Text
        xt_chunks = torch.split(xt, chunk_size, dim=0)
        mt_chunks = torch.split(tMask, chunk_size, dim=0)
        
        XprimeT_list = []
        for x, m in zip(xt_chunks, mt_chunks):
            chunk_out = checkpoint(run_transformer, x, m, use_reentrant=False)
            XprimeT_list.append(chunk_out)
        XprimeT = torch.cat(XprimeT_list, dim=0)
        
        # Formula 8. X^(AT)
        Xat,Xta=self.Cross_Attn(XprimeA,XprimeT,aMask,tMask)

        avgXprimeA=self.maskMean(XprimeA, aMask) # XprimeA: torch.Size([87,386,1024])
        avgXat=self.maskMean(Xat,aMask) # Xat: torch.Size([87,386,1024])
        avgXta=self.maskMean(Xta,tMask) # Xta: torch.Size([87,23,1024])
        avgXprimeT=self.maskMean(XprimeT, tMask) # XprimeT: torch.Size([87,23,1024])
        hE=torch.cat((avgXprimeA,avgXat,avgXta,avgXprimeT),dim=1)
        # Formula 5. FFN(Z)=ReLU(ZW_1+b_1)W_2+b_2
        Fc1=F.relu(self.fc1(hE))
        Fc2=F.relu(self.fc2(Fc1))
        Fc3=self.fc3(Fc2)
        Oup=self.oup(Fc3)

        return Fc3,Oup
    

    def maskMean(self, inp, mask):
        if mask is None:
            return inp.mean(dim=1)

        valid = (~mask).unsqueeze(-1).float()   # [B, L, 1]
        s = (inp * valid).sum(dim=1)            # 只加有效位置
        Len = valid.sum(dim=1).clamp(min=1.0)   # [B, 1]
        return s / Len
    # def maskMean(self, inp, mask):
    #     '''
    #     mask.shape: torch.Size([87, 386])
    #     tensor([[False, False, False,  ...,  True,  True,  True],
    #             ...,
    #             [False, False, False,  ...,  True,  True,  True],
    #     '''
    #     if mask is None:
    #         return torch.mean(inp,dim=1)

    #     Len = (~mask).sum(dim=1).clamp(min=1) # for all patient, real length
    #     s = inp.sum(dim=1) # s: (Batch, Dim) -> for all patient, tot feat
    #     return s / Len.unsqueeze(-1)

class at_cross_attn(nn.Module):
    def __init__(self,embd_size=1024):
        super(at_cross_attn,self).__init__()
        self.at_Q = nn.Linear(embd_size, embd_size)
        self.at_K = nn.Linear(embd_size, embd_size)
        self.at_V = nn.Linear(embd_size, embd_size)

        self.ta_Q = nn.Linear(embd_size, embd_size)
        self.ta_K = nn.Linear(embd_size, embd_size)
        self.ta_V = nn.Linear(embd_size, embd_size)
    def forward(self,XprimeA,XprimeT,aMask=None,tMask=None):
        Qa = self.at_Q(XprimeA)
        Kt = self.at_K(XprimeT)
        Vt = self.at_V(XprimeT)

        Qt = self.ta_Q(XprimeT)
        Ka = self.ta_K(XprimeA)
        Va = self.ta_V(XprimeA)

        Xat = cross_attn(Qa,Kt,Vt,tMask)
        Xta = cross_attn(Qt,Ka,Va,aMask)

        return Xat,Xta


def cross_attn(Q,K,V,mask=None):
    '''
    patient300
    mask: [87, 23]
    Q: [87, 386, 1024] [N, Lq, E] SDPA require (N,..,Hq,L,E) Hq: Number of heads of Q
    K: [87, 23, 1024] [N, Lk, E] SDPA require (N,..,H,S,E) H: Number of heads of K&V
    V: [87, 23, 1024] [N, Lv, E] SDPA require (N,..,H,S,Ev) H: Number of heads of K&V
        - L: Target sequence length
        - S: Source sequence length
        - E: Embedding dim of the q and k
        - Ev: Embedding dim of the v
    '''
    Q = Q.unsqueeze(1) # [87, 1, 386, 1024] add a head dimension
    K = K.unsqueeze(1) # [87, 1, 23, 1024] 
    V = V.unsqueeze(1) # [87, 1, 23, 1024]

    attn_mask = None
    if mask is not None:
        attn_mask = (~mask).view(mask.size(0), 1, 1, mask.size(1)) # attn_mask: [87, 1, 1, 23] for broadcase
    
    output = F.scaled_dot_product_attention(Q, K, V,attn_mask=attn_mask) # [N, Hq: 1, L, Ev]

    return output.squeeze(1)

if __name__=="__main__":
    main()
