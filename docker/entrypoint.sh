#!/usr/bin/env bash
set -e

# 確保 PATH 包含 uv 和 venv
export PATH="/root/.local/bin:/workspace/.venv/bin:${PATH}"
export PYTHONPATH="/workspace/nested_learning/src:${PYTHONPATH}"

cd /workspace

# SSH key 注入(RunPod 會傳入 $PUBLIC_KEY)
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p /root/.ssh
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
fi

# 健康檢查(失敗不致命)
set +e
echo "============================================================"
echo "RunPod container started | $(date)"
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
[ -d "/workspace/datasets" ] && echo "datasets/ found." || echo "WARNING: datasets/ not found"
[ -d "/workspace/src/Inconsistency" ] && echo "src/Inconsistency found." || echo "WARNING: src/Inconsistency not found"
set -e

echo ""
echo "============================================================"

# 決定主 process
if [ "$#" -eq 0 ] || [ "$1" = "bash" ]; then
    # 預設:跑 sshd 在前景當主 process
    echo "Starting sshd in foreground (main process)..."
    echo "============================================================"
    exec /usr/sbin/sshd -D -e
else
    # 使用者指定了其他指令:背景啟動 sshd,然後 exec 指令
    echo "Starting sshd in background, then executing: $@"
    echo "============================================================"
    /usr/sbin/sshd
    exec "$@"
fi