import torch.nn as nn
import wandb
from datetime import datetime
from sklearn.metrics import f1_score
import argparse
from pathlib import Path
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import Counter
import numpy as np
from Inconsistency.utils import Timer, set_seed, numpy_random_init
from torch.utils.checkpoint import checkpoint
from Inconsistency.datasets.inconsistentLabel import get_Split_and_GroundTrue
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import torch
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=FutureWarning)

D_MODEL=128
NHEAD=8
LR=5e-5
EPOCHS=30
TRANSFORMER_ENC_LAYERS=1
CHUNK_SIZE=1
# RANDOM_SEED=42

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--nhead", type=int, default=NHEAD)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--enc_layers", type=int, default=TRANSFORMER_ENC_LAYERS)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)

    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_batches", type=int, default=None)

    parser.add_argument("--save_dir", type=str, default="weights/stage1")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--label_smoothing", type=float, default=0.05)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Emotion inconsistency - Stage1")
    parser.add_argument("--wandb_name", type=str, default=None)

    return parser.parse_args()

class daicwoz_dataset(Dataset):
    '''
    fold = "val" -> 給 Stage2Main 用的
    '''
    def __init__(self, fold: str="tr"):
        self.ds=[]
        a_root = Path("datasets/Feature/HuBERT")
        t_root = Path("datasets/Feature/RoBerTa")
        depMap, train_Idx,val_Idx,test_Idx=get_Split_and_GroundTrue()
        if fold=='tr':
            patient_Idx=train_Idx
        elif fold =='val':
            patient_Idx=val_Idx     
        elif fold =='test':
            patient_Idx=test_Idx          
        else: raise Exception('fold error')

        PseudoLabel = np.load("PseudoLabel_all_distilbert_zdist_q30_70.npz")
        patientIdx = PseudoLabel["patientIdx"]
        atei_label = PseudoLabel["label"]
        PseudoMap = {int(x): int(y) for x, y in zip(patientIdx, atei_label)}


        for p in patient_Idx:
            if p not in PseudoMap: continue
            a_path=a_root / f"{p}_acoustic.pt"
            t_path=t_root / f"{p}_text.pt"

            assert a_path.exists() and t_path.exists(), "ds error"

            dep_label = depMap[p]
            atei_label = PseudoMap[p]
            
            self.ds.append((p, atei_label, dep_label, a_path, t_path))
        

    def __len__(self):
        return len(self.ds)
    def __getitem__(self, index):
        Patient, PseudoL, DepL, a_path, t_path=self.ds[index]
        
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
    # print(f"==patient{Patient}")
    xa=pad_sequence(xa_list,batch_first=True) # (Pdb) p xa_padded[0,:,:] eg:這位病人的第0句話的長度進行填充
    xt=pad_sequence(xt_list,batch_first=True)

    # aMask = [N, seq_len]
    aMask=(xa.sum(dim=-1)==0) # (Pdb) p aMask[0,:] eg:若1024維特徵總和為0，則為Padding(True)
    tMask=(xt.sum(dim=-1)==0)

    return xa,xt,aMask,tMask,pseudoL,dep_label,Patient
def get_class_weight(ds, device):
    labels = []

    for item in ds.ds:
        p, atei_label, dep_label, a_path, t_path = item
        labels.append(atei_label)

    labels = np.array(labels)
    counts = np.bincount(labels, minlength=2)

    weights = len(labels) / (2.0 * counts)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    print("Train label counts:", counts)
    print("Class weights:", weights)

    return weights
def print_ds_label_dist(name, ds):
    labels = []
    patients = []

    for item in ds.ds:
        p, atei_label, dep_label, a_path, t_path = item
        labels.append(atei_label)
        patients.append(p)

    labels = np.array(labels)

    print(f"\n[{name}]")
    print("num samples:", len(labels))
    print("label counts:", np.bincount(labels, minlength=2))
    print("label ratio :", np.bincount(labels, minlength=2) / len(labels))
    print("patients:", patients)
def build_pseudo_balanced_sampler(ds, seed=42):
    """
    建立 Stage1Tr 訓練用的 WeightedRandomSampler。

    Stage1Tr 使用 patient-level pseudo label 作為 ATEI 任務的監督標籤。
    這個 sampler 會根據 train set 中 pseudo label 的類別分布調整抽樣機率，
    讓少數類 pseudo label 的病患在訓練時有較高機率被抽到。

    注意：
        - 這是 DataLoader 層級的 resampling。
        - 不會修改原始 dataset，也不會真的複製 ds.ds 裡的資料。
        - 只能用在 training set。
        - validation/test set 必須維持原始 label 分布，不能使用這個 sampler。

    Args:
        ds: daicwoz_dataset(fold="tr")。
            ds.ds 的每個 item 應為：
            (patient_id, atei_label, dep_label, a_path, t_path)
        seed: 控制 weighted sampling 的隨機種子，方便重現實驗。

    Returns:
        WeightedRandomSampler:
            訓練時用來近似平衡 pseudo-label 類別分布的 sampler。
    """

    labels = [int(item[1]) for item in ds.ds]  # item = (p, atei_label, dep_label, a_path, t_path)
    label_count = Counter(labels)

    print("\n[Stage1 Train Sampler]")
    print("original pseudo label count:", label_count)

    # 若某一類完全不存在，sampler 也救不了，直接報錯比較乾淨
    if len(label_count) < 2:
        raise ValueError(f"Train set only has one pseudo label class: {label_count}")

    sample_weights = [
        1.0 / label_count[label]
        for label in labels
    ]

    g = torch.Generator()
    g.manual_seed(seed)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=g,
    )

    return sampler
def main():
    set_seed(ARGS.seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


    run_name = ARGS.wandb_name
    if run_name is None:
        run_name = (f"stage1_seed{ARGS.seed}_lr{LR:.0e}_wd{ARGS.weight_decay:.0e}_do{ARGS.dropout:.2f}_ls{ARGS.label_smoothing:.2f}_d{D_MODEL}_l{TRANSFORMER_ENC_LAYERS}_{run_id}"
        )



    total_timer = Timer()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    save_dir = Path(ARGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    g = torch.Generator()
    g.manual_seed(ARGS.seed)

    best_val_f1 = -1.0
    patience = 50
    no_improve = 0
    # PseudoLabel=np.load("PseudoLabel.npz")
    # patientIdx, PseudoL = PseudoLabel["patientIdx"], PseudoLabel["label"]
    # PseudoMap = {int(x): int(y) for x, y in zip(patientIdx, PseudoL)}


    # from torch.utils.data import Subset

    # base_trDS = daicwoz_dataset(fold="tr")

    # # 先印前幾個確認 label
    # for i in range(10):
    #     print(i, base_trDS.ds[i][0], base_trDS.ds[i][1])  # index, patient, atei_label

    # trDS = Subset(base_trDS, [0, 2])
    # valDS = trDS

    # tr_loader = DataLoader(trDS, collate_fn=collate_fn, shuffle=False)
    # val_loader = DataLoader(valDS, collate_fn=collate_fn, shuffle=False)
    trDS = daicwoz_dataset(fold="tr")
    valDS = daicwoz_dataset(fold="val")

    print_ds_label_dist("Stage1 Train", trDS)
    print_ds_label_dist("Stage1 Val", valDS)

    if ARGS.use_wandb:
        wandb.init(
            project=ARGS.wandb_project,
            name=run_name,
            config={
                "seed": ARGS.seed,
                "d_model": D_MODEL,
                "nhead": NHEAD,
                "lr": LR,
                "epochs": EPOCHS,
                "enc_layers": TRANSFORMER_ENC_LAYERS,
                "chunk_size": CHUNK_SIZE,
                "dropout": ARGS.dropout,
                "weight_decay": ARGS.weight_decay,
                "label_smoothing": ARGS.label_smoothing,
                "pseudo_label_file": "PseudoLabel_all_distilbert_zdist_q30_70.npz",
                "train_samples": len(trDS),
                "val_samples": len(valDS),
            },
        )

    if ARGS.use_wandb:
        wandb.config.update({
            "train_samples": len(trDS),
            "val_samples": len(valDS),
        })

    # sampler = build_pseudo_balanced_sampler(trDS, seed=ARGS.seed)

    tr_loader = DataLoader(trDS, collate_fn=collate_fn, shuffle=True,generator=g, worker_init_fn=numpy_random_init)
    val_loader = DataLoader(valDS, collate_fn=collate_fn, shuffle=False,generator=g, worker_init_fn=numpy_random_init)

    model=atei(D_MODEL,NHEAD).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=ARGS.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=ARGS.label_smoothing)
    scaler = torch.GradScaler('cuda')
    history=[]

    model.train()
    for epoch in range(EPOCHS):
        model.train()
        totLoss = 0.0
        correct = 0
        n = 0

        pbar = tqdm(
            tr_loader,
            f"Train Epoch {epoch+1}/{EPOCHS}",
            unit="patient",
            leave=False,
        )

        for batch_idx, data in enumerate(pbar):
            if ARGS.max_batches is not None and batch_idx >= ARGS.max_batches:
                break
            xa, xt, aMask, tMask, atei_label, dep_label, Patient = data

            xa = xa.to(device)
            xt = xt.to(device)
            aMask = aMask.to(device)
            tMask = tMask.to(device)
            atei_label = atei_label.to(device)

            opt.zero_grad()

            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                feat, logits = model(xa, xt, aMask, tMask)
                patient_feat = feat.mean(dim=0)
                patient_logit = model.patient_oup(patient_feat)
                loss = criterion(patient_logit.unsqueeze(0), atei_label.unsqueeze(0))
                # _,logits=model(xa,xt,aMask,tMask) # logits:　[LenFeat,2] eg: [89,2]
                # patient_logit=logits.mean(dim=0) # torch.Size([2])
                # loss = criterion(patient_logit.unsqueeze(0), atei_label.unsqueeze(0))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            pred = patient_logit.argmax(dim=-1)
            correct += int(pred.item() == atei_label.item())
            totLoss += loss.item()
            n += 1

            pbar.set_postfix({
                "loss": totLoss / max(n, 1),
                "acc": correct / max(n, 1),
                "patient": Patient,
            })
        train_loss = totLoss / max(n, 1)
        train_acc = correct / max(n, 1)

        val_result = validate(model, val_loader, criterion, device)

        history.append(train_loss)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_result['loss']:.4f} | "
            f"Val Acc: {val_result['acc']:.4f} | "
            f"Val MacroF1: {val_result['macro_f1']:.4f}"
        )

        print("Val Label counts:", np.bincount(val_result["y_true"], minlength=2))
        print("Val Pred counts :", np.bincount(val_result["y_pred"], minlength=2))
        print("Val Confusion Matrix:")
        print(val_result["cm"])

        print(classification_report(
            val_result["y_true"],
            val_result["y_pred"],
            labels=[0, 1],
            target_names=["inconsistency(0)", "consistency(1)"],
            digits=4,
            zero_division=0,
        ))

        if ARGS.use_wandb:
            wandb.log({
                "epoch": epoch + 1,

                "train/loss": train_loss,
                "train/acc": train_acc,

                "val/loss": val_result["loss"],
                "val/acc": val_result["acc"],
                "val/macro_f1": val_result["macro_f1"],

                "val/pred_0": int(np.bincount(val_result["y_pred"], minlength=2)[0]),
                "val/pred_1": int(np.bincount(val_result["y_pred"], minlength=2)[1]),
                "val/label_0": int(np.bincount(val_result["y_true"], minlength=2)[0]),
                "val/label_1": int(np.bincount(val_result["y_true"], minlength=2)[1]),

                "best/val_macro_f1": best_val_f1,
                "no_improve": no_improve,

                "val/conf_mat": wandb.plot.confusion_matrix(
                    y_true=val_result["y_true"],
                    preds=val_result["y_pred"],
                    class_names=["inconsistency", "consistency"],
                ),
            })

        if val_result["macro_f1"] > best_val_f1:
            best_val_f1 = val_result["macro_f1"]
            no_improve = 0
            ckpt_name = (
                f"stage1_"
                f"{run_id}_"
                f"seed{ARGS.seed}_"
                f"f1{best_val_f1:.4f}_"
                f"ep{epoch+1:03d}_"
                f"lr{LR:.0e}_"
                f"wd{ARGS.weight_decay:.0e}_"
                f"d{D_MODEL}_"
                f"l{TRANSFORMER_ENC_LAYERS}.pt"
            )

            ckpt_path = save_dir / ckpt_name

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_f1": best_val_f1,
                    "val_acc": val_result["acc"],
                    "val_loss": val_result["loss"],
                    "val_cm": val_result["cm"],
                    "args": vars(ARGS),
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "enc_layers": TRANSFORMER_ENC_LAYERS,
                    "lr": LR,
                    "weight_decay": ARGS.weight_decay,
                    "dropout": ARGS.dropout,
                    "pseudo_label_file": "PseudoLabel_all_distilbert_zdist_q30_70.npz",
                },
                ckpt_path,
            )

            print(f"[Save Best] Val MacroF1: {best_val_f1:.4f} -> {ckpt_path}")
        else:
            no_improve += 1
            print(f"[EarlyStopping] no improvement: {no_improve}/{patience}")

            if no_improve >= patience:
                print(f"[EarlyStopping] Stop at epoch {epoch+1}. Best Val MacroF1: {best_val_f1:.4f}")
                break
        # if val_result["macro_f1"] > best_val_f1:
        #     best_val_f1 = val_result["macro_f1"]
        #     torch.save(model.state_dict(), ARGS.save_path)
        #     print(f"[Save Best] Val MacroF1: {best_val_f1:.4f} -> {ARGS.save_path}")
    imsave(history)
    print(f"Total time: {total_timer}")
    if ARGS.use_wandb:
        wandb.finish()


@torch.inference_mode()
def validate(model, loader, criterion, device):
    model.eval()
    totLoss = 0.0
    correct = 0
    n = 0

    all_y_true = []
    all_y_pred = []
    all_patient = []
    pbar = tqdm(loader, desc="Validation", unit="patient", leave=False)

    for batch_idx, data in enumerate(pbar):

        xa, xt, aMask, tMask, atei_label, dep_label, Patient = data

        xa = xa.to(device)
        xt = xt.to(device)
        aMask = aMask.to(device)
        tMask = tMask.to(device)
        atei_label = atei_label.to(device)

        # with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
        #     _, logits = model(xa, xt, aMask, tMask)
        #     patient_logit = logits.mean(dim=0)
        #     loss = criterion(patient_logit.unsqueeze(0), atei_label.unsqueeze(0))
        #     loss = loss.mean()
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            feat, logits = model(xa, xt, aMask, tMask)
            patient_feat = feat.mean(dim=0)
            patient_logit = model.patient_oup(patient_feat)
            loss = criterion(patient_logit.unsqueeze(0), atei_label.unsqueeze(0))
        pred = patient_logit.argmax(dim=-1)
        correct += int(pred.item() == atei_label.item())
        totLoss += loss.item()
        n += 1

        all_y_true.append(int(atei_label.item()))
        all_y_pred.append(int(pred.item()))
        all_patient.append(int(Patient))

        pbar.set_postfix({
            "loss": totLoss / max(n, 1),
            "acc": correct / max(n, 1),
            "patient": Patient,
        })

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_patient = np.array(all_patient)

    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    macro_f1 = f1_score(all_y_true, all_y_pred, labels=[0, 1], average="macro", zero_division=0)

    return {
        "loss": totLoss / max(n, 1),
        "acc": correct / max(n, 1),
        "macro_f1": macro_f1,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "patient": all_patient,
        "cm": cm,
    }


def imsave(history):
    fig, ax = plt.subplots()
    ax.plot(range(1, len(history) + 1), history)
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropyLoss')
    plt.title('Training Loss')
    plt.savefig("stage1_tr_loss.jpg")

class atei(nn.Module):
    def __init__(self,embd_size,nheads,inp_dim=1024):
        # super(atei,self).__init__()
        super().__init__()
        assert embd_size % nheads == 0, "Embedding size must be divisible by number of heads"
        self.in_proj=nn.Linear(inp_dim,embd_size) # Dynamic projection, Hubert and Wav2Vec2 oup are 1024 dim
        enc_layer=nn.TransformerEncoderLayer(d_model=embd_size, nhead=nheads,batch_first=True,dim_feedforward=4 * embd_size,dropout=ARGS.dropout)
        self.transformer_enc=nn.TransformerEncoder(enc_layer,num_layers=TRANSFORMER_ENC_LAYERS) #12

        self.Cross_Attn=at_cross_attn(embd_size)

        self.dropout=nn.Dropout(ARGS.dropout)

        self.fc1=nn.Linear(4*embd_size,embd_size)
        self.fc2=nn.Linear(embd_size,embd_size)
        self.fc3=nn.Linear(embd_size,embd_size)
        self.oup=nn.Linear(embd_size,2)

        self.patient_oup = nn.Linear(embd_size, 2)
        

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
        Fc1=self.dropout(F.relu(self.fc1(hE)))
        Fc2=self.dropout(F.relu(self.fc2(Fc1)))
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
    ARGS = parse_args()

    D_MODEL = ARGS.d_model
    NHEAD = ARGS.nhead
    LR = ARGS.lr
    EPOCHS = ARGS.epochs
    TRANSFORMER_ENC_LAYERS = ARGS.enc_layers
    CHUNK_SIZE = ARGS.chunk_size

    main()
