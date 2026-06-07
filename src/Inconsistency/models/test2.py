import torch
old = torch.load("datasets/Feature/RoBerTa/302_text.pt", map_location="cpu")
new = torch.load("datasets/Feature2/RoBerTa/302_text.pt", map_location="cpu")
print(f"Old: n={len(old)}, shape={old[0].shape}, dtype={old[0].dtype}")
print(f"New: n={len(new)}, shape={new[0].shape}, dtype={new[0].dtype}")

# 看前 5 個 segment 的 token 數
print("Old T:", [x.shape[1] for x in old[:5]])
print("New T:", [x.shape[1] for x in new[:5]])