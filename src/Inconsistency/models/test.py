import torch
from pathlib import Path
import numpy as np

feat_dir = Path("datasets/Feat_daic_eatd")

# 檢查所有 EATD feature
for pt in sorted(feat_dir.glob("t_*_acoustic.pt")) + sorted(feat_dir.glob("v_*_acoustic.pt")):
    data = torch.load(pt, weights_only=False)
    for i, x in enumerate(data):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"BAD {pt.name} seg{i}")
            
for pt in sorted(feat_dir.glob("t_*_text.pt")) + sorted(feat_dir.glob("v_*_text.pt")):
    data = torch.load(pt, weights_only=False)
    for i, x in enumerate(data):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"BAD {pt.name} seg{i}")

# 檢查 DAIC feature
for pt in sorted(feat_dir.glob("[0-9]*_acoustic.pt")):
    data = torch.load(pt, weights_only=False)
    for i, x in enumerate(data):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"BAD {pt.name} seg{i}")

print("check done")