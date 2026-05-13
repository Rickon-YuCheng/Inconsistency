import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from collections import Counter
import torch
from Stage1Tr_v1 import atei
from hope_adapter import HopeEncoderBlock
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,DataLoader
from datetime import datetime
import argparse
from Inconsistency.utils import Timer, set_seed, numpy_random_init
import torch.nn as nn
from tqdm import tqdm
import wandb
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from Inconsistency.datasets.inconsistentLabel import get_Split_and_GroundTrue
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

STAGE1_CKPT = "weights/stage1/stage1_20260510_051049_seed42_f10.7619_ep044_lr2e-05_wd5e-04_d128_l1.pt"
STAGE1_CKPT = "weights/stage1/stage1_20260513_023333_seed42_f10.6905_ep021_lr5e-05_wd0e+00_d128_l1.pt" # tr split
D_MODEL=128
NHEAD=8
LR=1e-5
EPOCHS=50
TRANSFORMER_ENC_LAYERS=1
DROPOUT = 0.3
ATEI_DROPOUT = 0.4
WEIGHT_DECAY = 1e-4
LAMBDA_ATEI = 0.1
ALPHA_INIT = 0.5
PATIENCE = 50
ENCODER_TYPE = "attn"  # "attn" or "hope_attention"

CMS_PERIODS = (1, 4)
CMS_HIDDEN_MULTIPLIER = 4
CMS_ONLINE_UPDATES = False

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--stage1_ckpt", type=str, default=STAGE1_CKPT)

    parser.add_argument("--d_model", type=int, default=D_MODEL)
    parser.add_argument("--nhead", type=int, default=NHEAD)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--enc_layers", type=int, default=TRANSFORMER_ENC_LAYERS)

    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--atei_dropout", type=float, default=ATEI_DROPOUT)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)

    parser.add_argument("--lambda_atei", type=float, default=LAMBDA_ATEI)
    parser.add_argument("--alpha_init", type=float, default=ALPHA_INIT)
    parser.add_argument("--patience", type=int, default=PATIENCE)

    parser.add_argument("--save_dir", type=str, default="weights/stage2")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Emotion inconsistency - Stage2")
    parser.add_argument("--wandb_name", type=str, default=None)

    parser.add_argument("--encoder_type",type=str,default=ENCODER_TYPE,choices=["attn", "hope_attention"],)
    parser.add_argument("--cms_periods",type=int,nargs="+",default=list(CMS_PERIODS),)
    parser.add_argument("--cms_hidden_multiplier",type=int,default=CMS_HIDDEN_MULTIPLIER,)

    parser.add_argument("--batch_size", type=int, default=2, help="batch size for DataLoader")

    return parser.parse_args()

def main():
    set_seed(ARGS.seed)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_name = ARGS.wandb_name
    if run_name is None:
        run_name = (
            f"stage2_"
            f"seed{ARGS.seed}_"
            f"lr{LR:.0e}_"
            f"wd{ARGS.weight_decay:.0e}_"
            f"do{ARGS.dropout:.2f}_"
            f"la{LAMBDA_ATEI:.2f}_"
            f"a{ALPHA_INIT:.2f}_"
            f"d{D_MODEL}_"
            f"l{TRANSFORMER_ENC_LAYERS}_"
            f"enc{ARGS.encoder_type}_"
            f"{run_id}"
        )

    total_timer = Timer()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    save_dir = Path(ARGS.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
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

                "dropout": ARGS.dropout,
                "atei_dropout": ARGS.atei_dropout,
                "weight_decay": ARGS.weight_decay,

                "lambda_atei": LAMBDA_ATEI,
                "alpha_init": ALPHA_INIT,
                "patience": PATIENCE,

                "loss_total": "LAMBDA_ATEI * L_Atei + L_Depression",
                "dep_class_weights": None,

                "stage1_ckpt": ARGS.stage1_ckpt,
            },
            save_code=True,
        )

    g = torch.Generator()
    g.manual_seed(ARGS.seed)

    best_val_f1 = -1.0
    bad_epochs = 0

    
    # 1. Dataset
    # trDS=stage2_dataset(fold="tr")
    # tr_loader=DataLoader(trDS,collate_fn=stage2_collate_fn, shuffle=True,generator=g,worker_init_fn=numpy_random_init)
    # tr_loader=DataLoader(trDS,collate_fn=collate_fn)
    # gpt
    from torch.utils.data import Subset

    trDS = stage2_dataset(fold="tr")
    sampler = build_stage2_dep_balanced_sampler(trDS, seed=ARGS.seed)

    tr_loader = DataLoader(
        trDS,
        collate_fn=stage2_collate_fn,
        batch_size=ARGS.batch_size,
        sampler=sampler,   # 使用完整 dataset 的 pseudo-label resample
        worker_init_fn=numpy_random_init,
    )
    # 只取前 10 個病人快速測試
    # trDS_full = stage2_dataset(fold="tr")
    # trDS = Subset(trDS_full, list(range(10)))

    # tr_loader = DataLoader(
    #     trDS,   # ✅ 這裡用 Subset
    #     collate_fn=stage2_collate_fn,
    #     shuffle=False,    # 快速測試不用 sampler
    #     worker_init_fn=numpy_random_init,
    # )
    # ===
    valDS=stage2_dataset(fold="val")
    val_loader=DataLoader(valDS,collate_fn=stage2_collate_fn, shuffle=False,batch_size=1, worker_init_fn=numpy_random_init)
    if ARGS.use_wandb:
        wandb.config.update({
            "train_samples": len(trDS),
            "val_samples": len(valDS),
        })


    # 2. Model parameter setting
    model=whole_model(D_MODEL,NHEAD).to(device)
    # opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=ARGS.weight_decay)
    atei_params = list(model.atei.parameters())
    other_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("atei.")
    ]

    opt = torch.optim.Adam(
        [
            {"params": atei_params, "lr": LR * 0.1},
            {"params": other_params, "lr": LR},
        ],
        weight_decay=ARGS.weight_decay,
    )
    scaler = torch.GradScaler('cuda')
    

    # gpt
    from collections import Counter
    # dep_counter = Counter([int(x[2]) for x in trDS.ds])
    # atei_counter = Counter([int(x[1]) for x in trDS.ds])
    if isinstance(trDS, torch.utils.data.Subset):
        train_ds_records = [trDS.dataset.ds[i] for i in trDS.indices]
    else:
        train_ds_records = trDS.ds

    dep_counter = Counter([int(x[2]) for x in train_ds_records])
    atei_counter = Counter([int(x[1]) for x in train_ds_records])
    # weights = torch.tensor([1.0, 1.3, 1.3], dtype=torch.float).to(device)
    # weights = None
    total = sum(dep_counter.values())
    n_classes = 3
    weights = torch.tensor([
        total / (n_classes * dep_counter[i]) for i in range(n_classes)
    ], dtype=torch.float, device=device)

    print("Train dep dist:", dep_counter)
    print("Train ATEI dist:", atei_counter)
    print("Class weights:", weights)

    val_dep_counter = Counter([int(x[2]) for x in valDS.ds])
    val_atei_counter = Counter([int(x[1]) for x in valDS.ds])

    print("Val dep dist:", val_dep_counter)
    print("Val ATEI dist:", val_atei_counter)

    loss_atei = nn.CrossEntropyLoss()
    # loss_dep = nn.CrossEntropyLoss()
    loss_dep = nn.CrossEntropyLoss(weight=weights)
    # ====




    # 3. Train
    for epoch in range(1,EPOCHS+1):
        print("=" * 80)
        print(f"Epoch [{epoch}/{EPOCHS}]")

        tr_result=train_one_epoch(model, tr_loader, loss_atei, loss_dep, opt, device, epoch, EPOCHS,scaler)
        val_result=val(model, val_loader, loss_dep, device, epoch, EPOCHS)


        print(
            f"[Train] "
            f"ATEI Loss: {tr_result['atei_loss']:.4f} | "
            f"Dep Loss: {tr_result['dep_loss']:.4f} | "
            f"Total Loss: {tr_result['tot_loss']:.4f} | "
            f"ATEI Acc: {tr_result['cur_atei_acc']:.4f} | "
            f"Dep Acc: {tr_result['cur_dep_acc']:.4f}"
        )

        print(
            f"[Val] "
            f"Dep Loss: {val_result['dep_loss']:.4f} | "
            f"Acc: {val_result['acc']:.4f} | "
            f"Pre: {val_result['pre']:.4f} | "
            f"Rec: {val_result['rec']:.4f} | "
            f"F1: {val_result['f1']:.4f}"
        )




        if val_result["f1"] > best_val_f1:
            best_val_f1 = val_result["f1"]
            bad_epochs = 0

            ckpt_name = (
                f"stage2_"
                f"{run_id}_"
                f"seed{ARGS.seed}_"
                f"f1{best_val_f1:.4f}_"
                f"ep{epoch:03d}_"
                f"lr{LR:.0e}_"
                f"wd{ARGS.weight_decay:.0e}_"
                f"d{D_MODEL}_"
                f"l{TRANSFORMER_ENC_LAYERS}.pt"
            )

            ckpt_path = save_dir / ckpt_name

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_f1": best_val_f1,
                    "val_acc": val_result["acc"],
                    "val_pre": val_result["pre"],
                    "val_rec": val_result["rec"],
                    "val_f1": val_result["f1"],
                    "val_dep_loss": val_result["dep_loss"],

                    "args": vars(ARGS),

                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "enc_layers": TRANSFORMER_ENC_LAYERS,
                    # "lr": LR,
                    "base_lr": LR,
                    "atei_lr": opt.param_groups[0]["lr"],
                    "other_lr": opt.param_groups[1]["lr"],
                    "weight_decay": ARGS.weight_decay,
                    "dropout": ARGS.dropout,
                    "atei_dropout": ARGS.atei_dropout,
                    "lambda_atei": LAMBDA_ATEI,
                    "alpha_init": ALPHA_INIT,

                    "stage1_ckpt": ARGS.stage1_ckpt,
                },
                ckpt_path,
            )

            if ARGS.use_wandb:
                wandb.run.summary["best_val_f1"] = best_val_f1

            print(f"[Save Best] Val F1: {best_val_f1:.4f} -> {ckpt_path}")
        else:
            bad_epochs += 1
            print(f"[EarlyStop] bad_epochs: {bad_epochs}/{PATIENCE}")


        if ARGS.use_wandb:
            wandb.log({
                "epoch": epoch,

                "train/atei_loss": tr_result["atei_loss"],
                "train/dep_loss": tr_result["dep_loss"],
                "train/tot_loss": tr_result["tot_loss"],
                "train/atei_acc": tr_result["cur_atei_acc"],
                "train/dep_acc": tr_result["cur_dep_acc"],

                "val/dep_loss": val_result["dep_loss"],
                "val/acc": val_result["acc"],
                "val/pre": val_result["pre"],
                "val/rec": val_result["rec"],
                "val/f1": val_result["f1"],

                "best/val_f1": best_val_f1,
                "no_improve": bad_epochs,
                "lr/atei": opt.param_groups[0]["lr"],
                "lr/other": opt.param_groups[1]["lr"],
                # "lr": opt.param_groups[0]["lr"],
            })
        if bad_epochs >= PATIENCE:
            print(f"[EarlyStop] Stop at epoch {epoch}, best val F1: {best_val_f1:.4f}")
            break

    print(f"Total time: {total_timer}")

    if ARGS.use_wandb:
        wandb.finish()


class whole_model(nn.Module):
    '''
    Transformer: (S: source sequence length, T: tgt seq_len, N: batch size, E: feature number)
    '''
    def __init__(self,embd_size=D_MODEL,nheads=NHEAD):
        super().__init__()
        self.in_proj=nn.Linear(1024, embd_size) # Because HuBERT and Wav2Vec2 oup are 1024 dim
        self.atei=atei(embd_size=embd_size,nheads=nheads, dropout=ARGS.atei_dropout)
        ckpt = torch.load(ARGS.stage1_ckpt, map_location="cpu")
        self.atei.load_state_dict(ckpt["model_state_dict"])
        # self.atei.load_state_dict(torch.load("stage1Weights.pth"))
        # for p in self.atei.parameters():
        #     p.requires_grad = False
        self.encoder_type = ARGS.encoder_type
        if self.encoder_type == "attn":
            a_enc_layer=nn.TransformerEncoderLayer(d_model=embd_size, dropout=ARGS.dropout, dim_feedforward=4*embd_size, nhead=nheads,batch_first=True) # # (N, T, E)
            t_enc_layer=nn.TransformerEncoderLayer(d_model=embd_size, dropout=ARGS.dropout, dim_feedforward=4*embd_size, nhead=nheads,batch_first=True) # # (N, T, E)
            self.a_transformer_enc=nn.TransformerEncoder(a_enc_layer,num_layers=TRANSFORMER_ENC_LAYERS) #12
            self.t_transformer_enc=nn.TransformerEncoder(t_enc_layer,num_layers=TRANSFORMER_ENC_LAYERS) #12
        elif self.encoder_type == "hope_attention":
            self.a_encoder = HopeEncoderBlock(dim=embd_size,heads=nheads,variant="hope_attention",cms_periods=tuple(ARGS.cms_periods),hidden_multiplier=ARGS.cms_hidden_multiplier,cms_online_updates=CMS_ONLINE_UPDATES,)
            self.t_encoder = HopeEncoderBlock(dim=embd_size,heads=nheads,variant="hope_attention",cms_periods=tuple(ARGS.cms_periods),hidden_multiplier=ARGS.cms_hidden_multiplier,cms_online_updates=CMS_ONLINE_UPDATES,)
        else:
            raise ValueError(f"Unknown encoder_type: {self.encoder_type}")
        self.dropout=nn.Dropout(ARGS.dropout)
        self.fc1=nn.Linear(3*embd_size,embd_size)
        self.fc2=nn.Linear(embd_size,embd_size)
        self.fc3=nn.Linear(embd_size, embd_size)
        # self.alpha = nn.Parameter(torch.ones(embd_size)*ALPHA_INIT)
        self.alpha = nn.Parameter(torch.tensor(ALPHA_INIT))  # scalar
        self.oup=nn.Linear(embd_size,3)
        
    def forward(self, XA, XT, aMask=None, tMask=None, 
        xa_seg_list=None, xt_seg_list=None, return_feature=False):
    
        # ---------- Stage3: Depression-related Feature Extraction ----------
        # XA, XT: [B, num_seg, 1024]，segment-level pooled
        XA_proj = self.in_proj(XA)  # [B, num_seg, D]
        XT_proj = self.in_proj(XT)

        if self.encoder_type == "attn":
            HA = self.a_transformer_enc(XA_proj, src_key_padding_mask=aMask)
            HT = self.t_transformer_enc(XT_proj, src_key_padding_mask=tMask)
        elif self.encoder_type == "hope_attention":  # hope_attention
            if aMask is not None: XA_proj = XA_proj.masked_fill(aMask.unsqueeze(-1), 0.0)
            HA = self.a_encoder(XA_proj)
            if aMask is not None: HA = HA.masked_fill(aMask.unsqueeze(-1), 0.0)

            if tMask is not None: XT_proj = XT_proj.masked_fill(tMask.unsqueeze(-1), 0.0)
            HT = self.t_encoder(XT_proj)
            if tMask is not None: HT = HT.masked_fill(tMask.unsqueeze(-1), 0.0)
        else: raise "encoder_type error(transformer or hope)"

        eA = self.masked_mean(HA, aMask)  # [B, D]
        eT = self.masked_mean(HT, tMask)  # [B, D]

        # 在 forward 裡 masked_mean 之後
        # print(f"aMask padding ratio: {aMask.float().mean().item():.4f}")
        # print(f"eA norm after masked_mean: {eA.norm(dim=-1).mean().item():.4f}")

        # ---------- ATEI: 永遠用 frame-level feature ----------
        # 不受 encoder_type 影響，ATEI 內部自己有 Transformer
        eE_list = []
        atei_logits_list = []
        for xa_seg, xt_seg in zip(xa_seg_list, xt_seg_list):
            # xa_seg: [num_seg, max_T, 1024]
            xa_seg = xa_seg.to(XA.device)
            xt_seg = xt_seg.to(XA.device)
            seg_mask_a = (xa_seg.sum(dim=-1) == 0)
            seg_mask_t = (xt_seg.sum(dim=-1) == 0)
            
            eE_i, logits_i = self.atei(xa_seg, xt_seg, seg_mask_a, seg_mask_t)
            eE_list.append(eE_i.mean(dim=0))           # [D]
            atei_logits_list.append(logits_i.mean(dim=0))  # [2]

        eE = torch.stack(eE_list, dim=0)               # [B, D]
        atei_logits = torch.stack(atei_logits_list, dim=0)  # [B, 2]
        # forward 裡暫時跳過 ATEI，直接用零向量
        # eE = torch.zeros_like(eA)
        # atei_logits = torch.zeros(eA.size(0), 2, device=eA.device)

        # ---------- Scaling ----------
        # alpha_norm = torch.softmax(self.alpha, dim=0)
        # eE = eE * alpha_norm.unsqueeze(0)
        eE=eE*self.alpha

        # print(f"eA norm: {eA.norm(dim=-1).mean().item():.4f}")
        # print(f"eT norm: {eT.norm(dim=-1).mean().item():.4f}")
        # print(f"eE norm: {eE.norm(dim=-1).mean().item():.4f}")

        # ---------- Stage4: Fusion ----------
        eFusion = torch.cat((eA, eE, eT), dim=1)       # [B, 3D]
        Fc1 = self.dropout(F.relu(self.fc1(eFusion)))
        Fc2 = self.dropout(F.relu(self.fc2(Fc1)))
        Fc3 = self.dropout(F.relu(self.fc3(Fc2)))
        dep_logits = self.oup(Fc3)                      # [B, 3]

        if return_feature:
            return atei_logits, dep_logits, Fc3
        return atei_logits, dep_logits
    
    def masked_mean(self, x, mask):
        if mask is None: return x.mean(dim=1)

        valid = (~mask).unsqueeze(-1)          # [B, T, 1]
        x = x * valid
        denom = valid.sum(dim=1).clamp(min=1)  # [B, 1]
        return x.sum(dim=1) / denom
    

def train_one_epoch(model, tr_loader, loss_atei, loss_dep, opt, device, cur_epoch, tot_epochs,scaler):
    model.train()
    totAteiLoss=totDepLoss=totLoss=0.0
    correct_atei=correct_dep=valid_batches=total_samples=0
    train_true_arr = []
    train_pred_arr = []
    pbar= tqdm(tr_loader, desc=f"Training epoch {cur_epoch}/{tot_epochs}",leave=False, unit='batch')
    
    for data in pbar:
        
        xa, xt, aMask, tMask, atei_label, dep_label, Patient, xa_seg_list, xt_seg_list = data

        xa = xa.to(device)
        xt = xt.to(device)
        aMask = aMask.to(device)
        tMask = tMask.to(device)
        atei_label = atei_label.to(device)
        dep_label = dep_label.to(device)

        opt.zero_grad()
        with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
            atei_logits, dep_logits=model(xa,xt,aMask,tMask, xa_seg_list=xa_seg_list,xt_seg_list=xt_seg_list) # logits:　[LenFeat,2] eg: [89,2]
        
            # patient_atei=atei_logits.mean(dim=1) # torch.Size([2])
            # patient_dep=dep_logits.mean(dim=1)
                    

            # L_Atei = loss_atei(patient_atei.unsqueeze(0), atei_label.unsqueeze(0)) # 加batch
            # L_Depression = loss_dep(patient_dep.unsqueeze(0), dep_label.unsqueeze(0)) # 加batch
            L_Atei = loss_atei(atei_logits, atei_label)   # [B, 2], [B]
            L_Depression = loss_dep(dep_logits, dep_label) # [B, 3], [B]
            L_Total=LAMBDA_ATEI*L_Atei+L_Depression
            # ===

        # gpt
        scaler.scale(L_Total).backward()

        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(opt)
        scaler.update()
        # ===
        # scaler.scale(L_Total).backward()
        # scaler.step(opt)
        # scaler.update()

        # Loss
        totAteiLoss += L_Atei.item()
        totDepLoss += L_Depression.item()
        totLoss += L_Total.item()

        # Acc       # argmax -> return idx
        atei_pred = atei_logits.argmax(dim=-1)  # return tensor([ 0.2683, -0.1693]
        dep_pred = dep_logits.argmax(dim=-1) # return tensor([-0.0456,  0.0038,  0.0627]
        correct_atei += (atei_pred == atei_label).sum().item()
        correct_dep += (dep_pred == dep_label).sum().item()
        valid_batches += 1
        total_samples += dep_label.size(0)

        pbar.set_postfix({
            "atei loss": totAteiLoss/valid_batches,
            "dep loss": totDepLoss/valid_batches,
            "tot loss": totLoss/valid_batches,
            "cur atei acc": correct_atei/total_samples,
            "cur dep acc": correct_dep/total_samples
        })
        # gpt
        # train_true_arr.append(int(dep_label))
        # train_pred_arr.append(int(dep_pred))
        train_true_arr.extend(dep_label.cpu().tolist())
        train_pred_arr.extend(dep_pred.cpu().tolist())
        # ===
    # gpt
    from collections import Counter
    print("Train true dist:", Counter(train_true_arr))
    print("Train pred dist:", Counter(train_pred_arr))
    # ===
    return {"atei_loss": totAteiLoss/valid_batches,
            "dep_loss": totDepLoss/valid_batches,
            "tot_loss": totLoss/valid_batches,
            "cur_atei_acc": correct_atei/total_samples,
            "cur_dep_acc": correct_dep/total_samples}


def val(model, val_loader, loss_dep, device, cur_epoch, tot_epochs):
    model.eval()

    totDepLoss = 0.0
    valid_batches = 0

    true_arr = []
    pred_arr = []

    pbar = tqdm(val_loader,desc=f"Validation epoch {cur_epoch}/{tot_epochs}",leave=False,unit="batch",)

    with torch.inference_mode():
        for data in pbar:
            if data is None: continue

            xa, xt, aMask, tMask, atei_label, dep_label, Patient, xa_seg_list, xt_seg_list = data

            xa = xa.to(device)
            xt = xt.to(device)
            aMask = aMask.to(device)
            tMask = tMask.to(device)
            dep_label = dep_label.to(device)

            with torch.autocast(device_type="cuda",enabled=(device == "cuda"),):
                _, dep_logits= model(xa, xt, aMask, tMask, xa_seg_list=xa_seg_list,xt_seg_list=xt_seg_list)

                patient_dep = dep_logits.squeeze(0)

                L_Depression = loss_dep(patient_dep.unsqueeze(0),dep_label)


            # prediction 1: mean logits
            dep_pred = patient_dep.argmax(dim=-1)


            true_arr.append(int(dep_label.item()))
            pred_arr.append(int(dep_pred.item()))

            totDepLoss += L_Depression.item()
            valid_batches += 1


            pbar.set_postfix({
                "dep_loss": totDepLoss / valid_batches,
            })
    metrics = get_metrics(true_arr, pred_arr)

    # gpt
    from sklearn.metrics import classification_report, confusion_matrix

    from collections import Counter
    print("Val true dist:", Counter(true_arr))
    print("Val  pred dist:", Counter(pred_arr))

    print("Confusion matrix:")
    print(confusion_matrix(true_arr, pred_arr, labels=[0, 1, 2]))

    print(classification_report(
        true_arr,
        pred_arr,
        labels=[0, 1, 2],
        digits=4,
        zero_division=0
    ))
    # ===

    return {
        "dep_loss": totDepLoss / max(valid_batches, 1),

        "acc": metrics["acc"],
        "pre": metrics["pre"],
        "rec": metrics["rec"],
        "f1": metrics["f1"],

        "labels": true_arr,
        "preds": pred_arr,
    }


class stage2_dataset(Dataset):
    def __init__(self, fold: str = "tr"):
        self.ds = []

        a_root = Path("datasets/Feature/HuBERT")
        t_root = Path("datasets/Feature/RoBerTa")

        depMap, train_Idx, val_Idx, test_Idx = get_Split_and_GroundTrue()

        if fold == "tr":
            patient_Idx = train_Idx
        elif fold == "val":
            patient_Idx = val_Idx
        elif fold == "test":
            patient_Idx = test_Idx
        else:
            raise Exception("fold error")

        PseudoLabel = np.load("PseudoLabel_all_distilbert_zdist_q30_70.npz")
        patientIdx = PseudoLabel["patientIdx"]
        pseudo_label = PseudoLabel["label"]
        PseudoMap = {int(x): int(y) for x, y in zip(patientIdx, pseudo_label)}

        for p in patient_Idx:
            a_path = a_root / f"{p}_acoustic.pt"
            t_path = t_root / f"{p}_text.pt"

            assert a_path.exists() and t_path.exists(), "ds error"

            dep_label = depMap[p]

            # q30_70 檔裡沒有的中間 40%，在 Stage2 補成 consistency(1)
            atei_label = PseudoMap[p] if p in PseudoMap else 1

            self.ds.append((p, atei_label, dep_label, a_path, t_path))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        Patient, PseudoL, DepL, a_path, t_path = self.ds[index]

        xa = torch.load(str(a_path))
        xt = torch.load(str(t_path))

        xa_list = [x.squeeze(0) for x in xa]
        xt_list = [x.squeeze(0) for x in xt]

        atei_label = torch.tensor(PseudoL, dtype=torch.long)
        dep_label = torch.tensor(DepL, dtype=torch.long)


        return xa_list, xt_list, atei_label, dep_label, Patient
    


def stage2_collate_fn(batch):
    xa_seg_list = []   # 給 ATEI 用：保留 frame-level
    xt_seg_list = []
    xa_pool_list = []  # 給 Stage3 Transformer 用：segment-level pooled
    xt_pool_list = []
    atei_labels = []
    dep_labels = []
    patients = []

    for xa_i, xt_i, atei_label, dep_label, patient in batch:
        # Stage3 用：每句話 mean pool -> [num_seg, 1024]
        xa_pool_list.append(torch.stack([x.mean(dim=0) for x in xa_i], dim=0))
        xt_pool_list.append(torch.stack([x.mean(dim=0) for x in xt_i], dim=0))

        # ATEI 用：保留 frame-level，pad 成 [num_seg, max_T, 1024]
        xa_seg_list.append(pad_sequence(xa_i, batch_first=True))
        xt_seg_list.append(pad_sequence(xt_i, batch_first=True))

        atei_labels.append(atei_label)
        dep_labels.append(dep_label)
        patients.append(patient)

    # Stage3 input: [B, max_num_seg, 1024]
    xa_pool = pad_sequence(xa_pool_list, batch_first=True)
    xt_pool = pad_sequence(xt_pool_list, batch_first=True)
    aMask = (xa_pool.sum(dim=-1) == 0)
    tMask = (xt_pool.sum(dim=-1) == 0)
    # 在 collate_fn 裡加這行
    # print(f"aMask any padding: {aMask.any().item()}, ratio: {aMask.float().mean().item():.4f}")

    atei_labels = torch.stack(atei_labels)
    dep_labels = torch.stack(dep_labels)

    return xa_pool, xt_pool, aMask, tMask, atei_labels, dep_labels, patients, xa_seg_list, xt_seg_list

# def stage2_collate_fn(batch):
#     xa_list = []
#     xt_list = []
#     atei_labels = []
#     dep_labels = []
#     patients = []

#     for xa_i, xt_i, atei_label, dep_label, patient in batch:
#         xa_list.append(torch.cat(xa_i, dim=0))
#         xt_list.append(torch.cat(xt_i, dim=0))
#         atei_labels.append(atei_label)
#         dep_labels.append(dep_label)
#         patients.append(patient)

#     xa = pad_sequence(xa_list, batch_first=True)
#     xt = pad_sequence(xt_list, batch_first=True)

#     aMask = (xa.sum(dim=-1) == 0)
#     tMask = (xt.sum(dim=-1) == 0)

#     atei_labels = torch.stack(atei_labels)
#     dep_labels = torch.stack(dep_labels)

#     return xa, xt, aMask, tMask, atei_labels, dep_labels, patients

def build_stage2_dep_balanced_sampler(ds, seed=42):
    """
    Stage2 正式訓練用：根據 depression label 做 patient-level resampling。

    注意：
    - Stage2 主任務是 depression 3-class，所以正式訓練應該優先平衡 dep_label，不是 pseudo ATEI label。
    - 只用在 train set。
    - val/test 不要用 sampler。
    - 支援 stage2_dataset，也支援 Subset。
    """

    if isinstance(ds, torch.utils.data.Subset):
        records = [ds.dataset.ds[i] for i in ds.indices]
    else:
        records = ds.ds

    labels = [int(item[2]) for item in records]  # item = (p, atei_label, dep_label, a_path, t_path)
    label_count = Counter(labels)

    print("\n[Stage2 Train Sampler]")
    print("original dep label count:", label_count)

    if len(label_count) < 2:
        raise ValueError(f"Train set only has one dep class: {label_count}")

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

def get_metrics(y_true, y_pred):
    return {
        "acc": accuracy_score(y_true, y_pred),
        "pre": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "rec": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


if __name__ == "__main__":
    ARGS = parse_args()

    STAGE1_CKPT = ARGS.stage1_ckpt
    D_MODEL = ARGS.d_model
    NHEAD = ARGS.nhead
    LR = ARGS.lr
    EPOCHS = ARGS.epochs
    TRANSFORMER_ENC_LAYERS = ARGS.enc_layers
    LAMBDA_ATEI = ARGS.lambda_atei
    ALPHA_INIT = ARGS.alpha_init
    PATIENCE = ARGS.patience
    ENCODER_TYPE = ARGS.encoder_type

    print(f"** Use type: {ENCODER_TYPE}**")

    main()