import torch.nn as nn
from pathlib import Path
import os
import torch
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

START=300
END=493
END=302


def main():
    Xa=[]
    Xt=[]
    
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
        Xa.append(xa) # len(Xa[:])=189 tot patient
        Xt.append(xt) # len(Xt[:])=189 tot patient
        
        Atei=atei()
        Atei(xa,xt)


        for i in range(len(xa)):
            XprimeA.append(transformer_enc(xa[i].to("cuda")))
            XprimeT.append(transformer_enc(xt[i].to("cuda")))
        avgXprimeA=nn.AvgPool1d(XprimeA)
        avgXprimeT=nn.AvgPool1d(XprimeT)
        
        breakpoint()
        
class atei(nn.Module):
    def __init__(self,embd_size,num_heads):
        super(atei,self).__init__()
        enc_layer=nn.TransformerEncoderLayer(d_model=1024, nhead=8,batch_first=True)
        self.transformer_enc=nn.TransformerEncoder(enc_layer,num_layers=12)

        self.Cross_Attn=at_cross_attn(embd_size)

        

    def forward(self,xa, xt):
        XprimeA,XprimeT=[],[]
        

        for i in range(len(xa)):
            XprimeA.append(transformer_enc(xa[i].to("cuda")))
            XprimeT.append(transformer_enc(xt[i].to("cuda")))

        avgXprimeA=torch.mean(XprimeA, dim=1)
        avgXprimeT=torch.mean(XprimeT, dim=1)
        
        Xat,Xta=at_cross_attn()(XprimeA,XprimeT)

        avgXat=torch.mean(Xat,dim=1)
        avgXta=torch.mean(Xta,dim=1)

        hE=torch.cat(avgXprimeA,Xat,Xta,avgXprimeT)
        fc1=nn.Linear(hE[-1],hE[-1])(hE)
        fc2=nn.Linear(hE[-1],hE[-1])(fc1)
        fc3=nn.Linear(hE[-1],hE[-1])(fc2)
        eE=nn.Softmax(dim=1)(fc3)
        return eE

class at_cross_attn(nn.modules):
    def __init__(self,embd_size=1024):
        super(at_cross_attn,self).__init__()
        self.at_Q = nn.Linear(embd_size, embd_size)
        self.at_K = nn.Linear(embd_size, embd_size)
        self.at_V = nn.Linear(embd_size, embd_size)

        self.ta_Q = nn.Linear(embd_size, embd_size)
        self.ta_K = nn.Linear(embd_size, embd_size)
        self.ta_V = nn.Linear(embd_size, embd_size)
    def forward(self,XprimeA,XprimeT):
        Qa = self.at_Q(XprimeA)
        Kt = self.at_K(XprimeT)
        Vt = self.at_V(XprimeT)

        Qt = self.ta_Q(XprimeA)
        Ka = self.ta_K(XprimeA)
        Va = self.ta_V(XprimeT)

        Xat, at_attn_w=cross_attn(Qa,Kt,Vt)
        Xta, ta_attn_w=cross_attn(Qt,Ka,Va)

        return Xat,Xta


def cross_attn(Q,K,V,mask=None):
    # Compute the dot products between Q and K, then scale
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    
    # Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Softmax to normalize scores and get attention weights
    attention_weights = F.softmax(scores, dim=-1)
     
    # Weighted sum of values
    output = torch.matmul(attention_weights, V)
    return output, attention_weights


if __name__=="__main__":
    main()