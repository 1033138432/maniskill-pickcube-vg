#!/usr/bin/env bash
# 01 下载官方演示（只有 state+action，无图像）
# 用法: bash 01_download_demos.sh [目标目录]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

DEST=${1:-${DEMO_ROOT:-$REPO_ROOT/demos}}
mkdir -p "$DEST"
python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('haosulab/ManiSkill_Demonstrations', 'demos/PickCube-v1.zip',
                    repo_type='dataset', local_dir='$DEST')
print('下载完成:', p)
"
cd "$DEST" && unzip -o -q PickCube-v1.zip && echo "解压完成: $(pwd)/PickCube-v1/motionplanning/trajectory.h5"
echo "提示：02/03 默认去 \$DEMO_ROOT/extracted 找 h5，若解压到别处请 export DEMO_ROOT=<解压目录>"
