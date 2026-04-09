# 0.7501 0.6920 0.7231 0.6843 0.6927 0.6934 0.6868 0.6854 0.6900 0.6948
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
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

START=300
END=493
# END=302

D_MODEL=1024
NHEAD=8
LR=1e-5
EPOCHS=10

class ateiDataset(Dataset):
    def __init__(self, START, END, psuedoLabel):
        allIdx=list(range(START,END))
        self.idx=[]
        a_root=Path("datasets/Feature/HuBERT")
        t_root=Path("datasets/Feature/RoBerTa")

        for x in allIdx:
            a_path=a_root/f"{x}_acoustic.pt"
            t_path=t_root/f"{x}_text.pt"
            if a_path.exists() and t_path.exists(): self.idx.append(x)
        
        self.psuedoL=psuedoLabel
        self.a_root=a_root
        self.t_root=t_root
        
    def __len__(self):
        return len(self.idx)
    def __getitem__(self, index):
        Patient=self.idx[index]
        PsuedoL=self.psuedoL[index]

        a_path = self.a_root / f"{Patient}_acoustic.pt"
        t_path = self.t_root / f"{Patient}_text.pt"

        # xa type:List(tensor)
        xa = torch.load(str(a_path))
        xt = torch.load(str(t_path))

        xa_list=[x.squeeze(0) for x in xa] # 這個人的每句話
        xt_list=[x.squeeze(0) for x in xt]

        return xa_list, xt_list, torch.tensor(PsuedoL), Patient#, a_path


def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    
    xa_list,xt_list,pseudoL,Patient=batch[0]
    print(f"==patient{Patient}")
    xa=pad_sequence(xa_list,batch_first=True) # (Pdb) p xa_padded[0,:,:] eg:這位病人的第0句話的長度進行填充
    xt=pad_sequence(xt_list,batch_first=True)

    # aMask = [N, seq_len]
    aMask=(xa.sum(dim=-1)==0) # (Pdb) p aMask[0,:] eg:若1024維特徵總和為0，則為Padding(True)
    tMask=(xt.sum(dim=-1)==0)

    return xa,xt,aMask,tMask,pseudoL

def train_stage1():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    PsuedoL=np.load("PseudoLabel.npz")["a"]

    ds=ateiDataset(START=START,END=END, psuedoLabel=PsuedoL) #(Pdb) p ds[0][2]->tensor(0) (Pdb) p ds[0][3]->\300
    dataLoader=DataLoader(ds,batch_size=1,collate_fn=collate_fn,shuffle=True)
    model=atei(D_MODEL,NHEAD).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    criterion=nn.CrossEntropyLoss()

    scaler = torch.GradScaler('cuda')
    model.train()
    for epoch in range(EPOCHS):
        totLoss=0
        for key, data in enumerate(dataLoader):
            if data is None: continue
            xa, xt, aMask, tMask, label = [d.to(device) for d in data]
            opt.zero_grad()
            with torch.autocast('cuda'):
                _,logits=model(xa,xt,aMask,tMask)
                patient_logit=logits.mean(dim=0,keepdim=True)
                breakpoint() # watch patient_logit and label
                loss = criterion(patient_logit, label.unsqueeze(0)) # label 需要擴展一維
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()


            totLoss += loss.item()
            del xa, xt, aMask, tMask, logits, patient_logit, loss
        print(f"Epoch [{epoch+1}/{EPOCHS}], Avg Loss: {totLoss/len(dataLoader):.4f}")

    # 儲存權重檔
    torch.save(model.state_dict(), "atei_stage1_weights.pth")
    print("權重已保存至 atei_stage1_weights.pth")

            



def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    PsuedoL=np.load("PseudoLabel.npz")
    PsuedoL=PsuedoL["a"]
    Atei=atei(D_MODEL,NHEAD).to(device)
    Atei.eval()

    for i in range(START, END):
        a_path = Path("datasets/Feature/HuBERT")
        a_path = a_path / f"{i}_acoustic.pt"
        t_path = Path("datasets/Feature/RoBerTa")
        t_path = t_path / f"{i}_text.pt"

        if not os.path.exists(t_path):
            print(f"PATH: {t_path} does not exist")
            continue

        # xa type:List(tensor)
        xa = torch.load(str(a_path))
        xt = torch.load(str(t_path))

        xa_list=[x.squeeze(0) for x in xa] # 這個人的每句話
        xt_list=[x.squeeze(0) for x in xt]
        
        # xa = [N, seq_len, 1024] -> [N, Max_seq_len, 1024]
        xa=pad_sequence(xa_list,batch_first=True).to(device) # (Pdb) p xa_padded[0,:,:] eg:這位病人的第0句話的長度進行填充
        xt=pad_sequence(xt_list,batch_first=True).to(device)

        # aMask = [N, seq_len]
        aMask=(xa.sum(dim=-1)==0).to(device) # (Pdb) p aMask[0,:] eg:若1024維特徵總和為0，則為Padding(True)
        tMask=(xt.sum(dim=-1)==0).to(device)

        with torch.no_grad():
            embdRep, probRep=Atei(xa,xt,aMask,tMask)
        breakpoint()
        binLabel=torch.argmax(probRep,dim=-1)
        breakpoint()


class atei(nn.Module):
    def __init__(self,embd_size,nheads):
        super(atei,self).__init__()
        assert embd_size % nheads == 0, "Embedding size must be divisible by number of heads"
        enc_layer=nn.TransformerEncoderLayer(d_model=embd_size, nhead=nheads,batch_first=True)
        self.transformer_enc=nn.TransformerEncoder(enc_layer,num_layers=4) #12

        self.Cross_Attn=at_cross_attn(embd_size)

        self.fc1=nn.Linear(4*embd_size,embd_size)
        self.fc2=nn.Linear(embd_size,embd_size)
        self.fc3=nn.Linear(embd_size,1024)
        self.oup=nn.Linear(1024,2)
        

    def forward(self,xa, xt, aMask=None, tMask=None):
        XprimeA,XprimeT=[],[]
        
        # XprimeA=self.transformer_enc(xa, src_key_padding_mask=aMask)
        # XprimeT=self.transformer_enc(xt, src_key_padding_mask=tMask)
        def run_transformer(x, mask):
            return self.transformer_enc(x, src_key_padding_mask=mask)

        chunk_size = 5

        # Audio
        xa_chunks = torch.split(xa, chunk_size, dim=0)
        ma_chunks = torch.split(aMask, chunk_size, dim=0)
        
        XprimeA_list = []
        for x, m in zip(xa_chunks, ma_chunks):
            # --- 方案 1：Checkpointing (不存中間層) ---
            # use_reentrant=False 是為了更好的相容性
            chunk_out = checkpoint(run_transformer, x, m, use_reentrant=False)
            XprimeA_list.append(chunk_out)
        XprimeA = torch.cat(XprimeA_list, dim=0)

        # 同理處理文本特徵 (xt)
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
        '''
        mask.shape: torch.Size([87, 386])
        tensor([[False, False, False,  ...,  True,  True,  True],
                ...,
                [False, False, False,  ...,  True,  True,  True],
        '''
        if mask is None:
            return torch.mean(inp,dim=1)

        Len = (~mask).sum(dim=1).clamp(min=1) # for all patient, real length
        s = inp.sum(dim=1) # s: (Batch, Dim) -> for all patient, tot feat
        return s / Len.unsqueeze(-1)

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

        # aMask = aMask.unsqueeze(1) if aMask is not None else None
        # tMask = tMask.unsqueeze(1) if tMask is not None else None
        Xat, at_attn_w=cross_attn(Qa,Kt,Vt,tMask)
        Xta, ta_attn_w=cross_attn(Qt,Ka,Va,aMask)

        return Xat,Xta


def cross_attn(Q,K,V,mask=None):
    Q = Q.unsqueeze(1)
    K = K.unsqueeze(1)
    V = V.unsqueeze(1)

    attn_mask = None
    if mask is not None:
        # 2. Mask 也要對應變成 4D: [N, 1, 1, Lk]
        # ~mask 是因為 SDPA 的布林遮罩中 True 代表「可看」，False 代表「遮掉」
        attn_mask = (~mask).view(mask.size(0), 1, 1, mask.size(1))

    # 3. 執行運算
    # 此時輸出的 shape 會是 [N, 1, Lq, E]
    output = F.scaled_dot_product_attention(
        Q, K, V, 
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=False
    )

    # 4. 運算完再把 Head 維度壓掉，回到 [N, Lq, E]
    return output.squeeze(1), None
    # # Compute the dot products between Q and K, then scale
    # d_k = Q.size(-1)
    # scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # # Apply mask if provided
    # if mask is not None:
    #     scores = scores.masked_fill(mask, float('-inf'))
    
    # # Softmax to normalize scores and get attention weights
    # attention_weights = F.softmax(scores, dim=-1)
     
    # # Weighted sum of values
    # output = torch.matmul(attention_weights, V)
    # return output, attention_weights


if __name__=="__main__":
    train_stage1()
    # main()