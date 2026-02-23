# Temp: A Universal AI Research Template

[![python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

這是一個基於 **uv** 與 **WandB** 構建的輕量化、模組化深度學習研究範本。旨在提供一個乾淨的「骨架」，讓研究者能專注於模型架構（如 Mamba）與數據處理，而無需重複撰寫實驗管理代碼。

> [!CAUTION]
> **🚧 Work In Progress (WIP)**
> 本專案目前正在開發中，程式碼可能隨時變動。

---

## 📂 Project Structure

```text
.
├── configs/            # YAML 配置文件 (Hyperparameters)
├── datasets/           # 原始數據存放路徑 (Git ignored)
├── scripts/            # 實驗入口腳本 (Training/Inference)
├── src/temp/           # 核心原始碼包 (Importable package)
│   ├── datasets/       # 數據加載邏輯 (DataLoaders)
│   ├── models/         # 模型架構定義 (NN Modules)
│   └── utils/          # 核心工具 (Timer, Logger, etc.)
├── tests/              # 單元測試 (Shape & Hardware checks)
└── pyproject.toml      # uv 專案定義與依賴管理
```

## 🚀 Quick Start

1. 環境初始化
本專案使用 `uv` 進行套件管理。請確保已安裝 uv。
```bash
# 同步虛擬環境與依賴
uv sync
```

2. 執行實驗
使用指定的設定檔啟動訓練：
```bash
# 預設執行 (不開啟 WandB)
uv run scripts/train.py --config configs/cfg.yaml

# 開啟 WandB 紀錄
uv run scripts/train.py --config configs/cfg.yaml --wandb
```

## ⚙️ Configuration
所有的實驗參數都定義在 `configs/*.yaml` 中。

- Model: 定義輸入輸出維度、隱藏層參數等。

- Train: 定義 Learning Rate, Batch Size, Optimizer 等。

你可以透過命令行輕鬆覆蓋 YAML 設定：
```bash
uv run scripts/train.py --config configs/cfg.yaml --bs 64 --lr 0.0005
```

## 🧪 Reproducibility (實驗重現)
為了確保實驗結果的可重現性，請遵循以下規範：

1. 隨機種子: 預設種子已在 cfg.yaml 中設定為 `0`，並透過 utils.set_seed() 統一控制。

2. 環境鎖定: 始終透過 uv.lock 確保依賴版本一致。

3. 硬體紀錄: 本範本預設紀錄 CUDA 狀態。建議在論文中註明使用的 GPU 型號與驅動版本。

## 🛠 Development Workflow
當你想開始一個新研究時，請按以下順序操作：

1. Define Model: 在 src/temp/models/ 建立新的模型檔案。

2. Setup Data: 在 src/temp/datasets/ 撰寫數據處理邏輯。

3. Modify Loop: 根據需求微調 scripts/train.py 中的訓練迴圈。

4. Update Config: 建立新的 YAML 檔案並執行。

## 📊 Logging & Visualization
本專案原生整合 Weights & Biases (WandB)。

- 實驗開始前會自動上傳當前的 config 快照。

- 訓練過程中會記錄 loss、epoch 以及系統資源佔用。

- 即使實驗崩潰，try...finally 機制也會確保數據正確上傳並關閉任務。

## 📝 License
Distributed under the MIT License. See LICENSE for more information.