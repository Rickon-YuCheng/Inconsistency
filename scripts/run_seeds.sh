#!/bin/bash
# run_seeds.sh — 跑 5 個種子 (43~47)

set -e   # 任何一個 seed 失敗就停下來
cd /workspace   # 確保在對的目錄

for SEED in 63 64 65 66 67 68 69 70 71 72 73 74 75; do
    echo "================================"
    echo "Running seed=$SEED"
    echo "================================"
    
    uv run src/Inconsistency/models/Stage2Main_quick.py \
        --epochs 300 \
        --enc_layers 1 \
        --d_model 256 \
        --dropout 0.3 \
        --atei_dropout 0.3 \
        --weight_decay 0 \
        --lr 1e-3 \
        --lambda_atei 0.1 \
        --alpha_init 0.5 \
        --batch_size 8 \
        --patience 70 \
        --seed $SEED \
        --use_wandb \
        --wandb_name "seed${SEED}"
done

echo "All seeds done."