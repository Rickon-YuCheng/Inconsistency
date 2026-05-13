#!/usr/bin/env bash
set -e

cd /workspace

echo "============================================================"
echo "RunPod container started"
echo "Working directory: $(pwd)"
echo "============================================================"

echo "[Python]"
python --version

echo ""
echo "[uv]"
uv --version

echo ""
echo "[CUDA / PyTorch check]"
python - <<'PY'
import torch

print("torch version       :", torch.__version__)
print("cuda available     :", torch.cuda.is_available())
print("cuda device count  :", torch.cuda.device_count())

if torch.cuda.is_available():
    print("current device     :", torch.cuda.current_device())
    print("device name        :", torch.cuda.get_device_name(0))
PY

echo ""
echo "[Path check]"

if [ ! -d "/workspace/datasets" ]; then
    echo "WARNING: /workspace/datasets does not exist. Creating..."
    mkdir -p /workspace/datasets/Feature/HuBERT
    mkdir -p /workspace/datasets/Feature/RoBerTa
    mkdir -p /workspace/weights/stage1
    mkdir -p /workspace/weights/stage2
else
    echo "datasets/ found."
fi

if [ ! -d "/workspace/src/Inconsistency" ]; then
    echo "ERROR: /workspace/src/Inconsistency not found."
    exit 1
else
    echo "src/Inconsistency found."
fi

echo ""
echo "============================================================"
echo "Executing command:"
echo "$@"
echo "============================================================"

exec "$@"