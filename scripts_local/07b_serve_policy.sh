#!/usr/bin/env bash
# 07b 启动策略推理服务（一个终端），评测脚本从另一个终端连 8000 端口
# 用法: bash 07b_serve_policy.sh <配置名> <断点目录>
#   例: bash 07b_serve_policy.sh pi05_pickcube_lora  $CKPT_ROOT/checkpoints_pickcube/pi05_pickcube_lora/pickcube_lora/11999
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG=${1:-pi05_pickcube_lora}
CKPT=${2:?用法: $0 <配置名> <断点目录>}
[ -d "$CKPT" ] || { echo "错误：断点目录不存在 $CKPT" >&2; exit 1; }

req OPENPI_ROOT "/path/to/openpi" || exit 1
cd "$OPENPI_ROOT" || exit 1
export CUDA_VISIBLE_DEVICES="$GPU"
exec uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config="$CONFIG" --policy.dir="$CKPT"
