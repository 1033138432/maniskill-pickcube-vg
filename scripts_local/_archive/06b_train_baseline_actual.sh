#!/usr/bin/env bash
# PickCube LoRA 训练启动脚本
set -uo pipefail
cd /root/gpufree-data/openpi
source ./my_env.sh
export HF_LEROBOT_HOME=/root/gpufree-data/maniskill_proj/lerobot
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
exec uv run scripts/train.py pi05_pickcube_lora \
  --exp-name pickcube_lora \
  --batch-size 8 \
  --num-train-steps 12000 \
  --save-interval 6000 \
  --checkpoint-base-dir /root/gpufree-data/openpi/checkpoints_pickcube \
  --resume \
  --no-wandb-enabled
