#!/usr/bin/env bash
set -uo pipefail
cd /root/gpufree-data/openpi
source ./my_env.sh
export HF_LEROBOT_HOME=/root/gpufree-data/maniskill_proj/lerobot
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
exec uv run scripts/train.py pi05_pickcube_randcam \
  --exp-name pickcube_randcam \
  --batch-size 8 \
  --num-train-steps 12000 \
  --save-interval 6000 \
  --checkpoint-base-dir /root/gpufree-data/openpi/checkpoints_randcam \
  --no-wandb-enabled
