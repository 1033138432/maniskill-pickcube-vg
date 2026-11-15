#!/usr/bin/env bash
# 06 训练基线模型（固定相机，repo_id=pickcube_ind）
# 用法: bash 06_train_baseline.sh
# 预期: 12,000 步 / batch 8 / 1.7 秒每步 / 5.7 小时 / 显存 34.7 GB / 可训练 50 M
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

req OPENPI_ROOT "/path/to/openpi" || exit 1
req LEROBOT_HOME "/path/to/lerobot" || exit 1

cd "$OPENPI_ROOT" || exit 1
source ./my_env.sh 2>/dev/null || true
export HF_LEROBOT_HOME="$LEROBOT_HOME"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
exec uv run scripts/train.py pi05_pickcube_lora \
    --exp-name pickcube_lora \
    --batch-size 8 \
    --num-train-steps 12000 \
    --save-interval 6000 \
    --checkpoint-base-dir "$CKPT_ROOT/checkpoints_pickcube" \
    --resume \
    --no-wandb-enabled
