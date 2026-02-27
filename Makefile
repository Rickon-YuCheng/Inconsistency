# 不要管資料夾裡有沒有同名的檔案，只要我下指令，你就給我跑後面的腳本就對了
# uv add transformers uv remove transformers
.PHONY: main main-wandb lint format clean test check tree tree2

# 啟動實驗
train:
	uv run scripts/train.py

# 開啟 WandB 的實驗
train-wandb:
	uv run scripts/train.py --wandb

# 靜態檢查 (Ruff)
lint:
	uv run ruff check .

# 自動格式化 (Ruff)
format:
	uv run ruff format .

# 單元測試
test:
	uv run pytest tests/

# 清理快取
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf .ruff_cache

# tree
tree:
	uv run tree -L 2
# tree2:
	@git ls-tree -r --name-only HEAD | tree --fromfile .
	rm -rf .ruff_cache

# 環境檢查
check:
	@uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); torch.cuda.is_available() and print(f'GPU: {torch.cuda.get_device_name(0)}')"