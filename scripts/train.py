import argparse
import yaml
import wandb
from temp.utils import set_seed, init_wandb, Timer, Averager  # noqa: F401


def train(config, model, dataloader):
    """這是一個通用的訓練存根"""
    # 這裡放最基礎的 Torch 邏輯
    # optimizer = ...
    # for epoch in ...
    pass


if __name__ == "__main__":
    # 1. 參數解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()

    # 2. 配置加載
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 3. 環境初始化
    set_seed(42)

    # 4. 組件構建 (此處為 Stub 邏輯)
    # model = build_model(config)
    # dataloader = build_dataloader(config)

    # 5. 實驗紀錄
    if args.wandb:
        init_wandb(config["project"], config["name"], config)

    # 6. 執行
    try:
        # train(config, model, dataloader)
        pass
    finally:
        if args.wandb:
            wandb.finish()
