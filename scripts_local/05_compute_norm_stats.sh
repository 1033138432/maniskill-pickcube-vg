#!/usr/bin/env bash
# 05 计算归一化统计量（openpi 训练前必须做）
# 用法: bash 05_compute_norm_stats.sh [配置名=pi05_pickcube_lora]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG=${1:-pi05_pickcube_lora}
req OPENPI_ROOT "/path/to/openpi" || exit 1
req LEROBOT_HOME "/path/to/lerobot" || exit 1

cd "$OPENPI_ROOT" || exit 1
HF_LEROBOT_HOME="$LEROBOT_HOME" uv run scripts/compute_norm_stats.py --config-name "$CONFIG"
echo "输出: $OPENPI_ROOT/assets/$CONFIG/<repo_id>/norm_stats.json"
