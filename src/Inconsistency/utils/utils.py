import random
import time

import torch
import numpy as np
import wandb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # sets the seed for cpu
    torch.cuda.manual_seed(seed)  # Sets the seed for the current GPU.
    torch.cuda.manual_seed_all(seed)  #  Sets the seed for the all GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def numpy_random_init(worker_id):
    process_seed = torch.initial_seed()
    base_seed = process_seed - worker_id
    ss = np.random.SeedSequence([worker_id, base_seed])
    np.random.seed(ss.generate_state(4))


def numpy_fix_init(worker_id):
    np.random.seed(2 << 16 + worker_id)


numpy_init_dict = {
    "train": numpy_random_init,
    "val": numpy_fix_init,
    "test": numpy_fix_init,
}


class Averager:
    """用於計算一個 Epoch 內 Loss 的平均值"""

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer:
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return "{:.1f}h".format(t / 3600)
    elif t >= 60:
        return "{:.1f}m".format(t / 60)
    else:
        return "{:.1f}s".format(t)


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return "{:.1f}M".format(tot / 1e6)
        else:
            return "{:.1f}K".format(tot / 1e3)
    else:
        return tot


def init_wandb(config, model=None):
    """
    初始化 WandB 並自動紀錄配置與參數
    """
    # 啟動 WandB
    run = wandb.init(
        project=config.get("project_name", "my_project"),
        name=config.get("run_name", None),
        config=config,
        reinit=True,
    )

    # 如果有傳入模型，自動紀錄參數量到 WandB Config
    if model is not None:
        params_text = compute_num_params(model, text=True)
        wandb.config.update({"total_params": params_text})
        # wandb.watch(model) # 視需求開啟，可追蹤梯度

    return run
