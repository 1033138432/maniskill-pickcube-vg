#!/usr/bin/env bash
# 07 训练相机随机化模型（repo_id=pickcube_ind_randcam）
# 用法: bash 07_train_randcam.sh
# 与 06 唯一的区别是配置名与 repo_id；超参完全相同（对比才公平）
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

req OPENPI_ROOT "/path/to/openpi" || exit 1
req LEROBOT_HOME "/path/to/lerobot" || exit 1

cd "$OPENPI_ROOT" || exit 1
source ./my_env.sh 2>/dev/null || true
export HF_LEROBOT_HOME="$LEROBOT_HOME"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
exec uv run scripts/train.py pi05_pickcube_randcam \
    --exp-name pickcube_randcam \
    --batch-size 8 \
    --num-train-steps 12000 \
    --save-interval 6000 \
    --checkpoint-base-dir "$CKPT_ROOT/checkpoints_randcam" \
    --no-wandb-enabled
