#!/usr/bin/env bash
# 00 环境自检：8 项。不通过不影响继续，但每一条 ❌ 都会在后面的步骤报错。
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ok=0; bad=0
chk() {  # chk <名称> <命令...>
    local name=$1; shift
    if "$@" >/dev/null 2>&1; then echo "  ✅ $name"; ok=$((ok+1)); else echo "  ❌ $name"; bad=$((bad+1)); fi
}

echo "=== 1. Python 与环境 ==="
echo "  PY=$PY"
"$PY" -c 'import sys; print(f"  Python {sys.version.split()[0]}")' 2>/dev/null || echo "  ❌ $PY 不可用"

echo "=== 2. ManiSkill ==="
chk "import mani_skill" "$PY" -c "import mani_skill"
chk "PickCube-v1 已注册" "$PY" -c "import gymnasium as gym, mani_skill; assert 'PickCube-v1' in gym.registry"

echo "=== 3. 渲染后端（Vulkan）==="
if [ -n "${VK_ICD_FILENAMES:-}" ]; then echo "  ✅ VK_ICD_FILENAMES=$VK_ICD_FILENAMES"; else echo "  ⚠️  VK_ICD_FILENAMES 未设置（无显示环境时通常必须设置）"; fi
chk "import sapien" "$PY" -c "import sapien"

echo "=== 4. 仿真资产 ==="
if [ -n "$MS_ASSET_DIR" ]; then echo "  ✅ MS_ASSET_DIR=$MS_ASSET_DIR"; else echo "  ⚠️  MS_ASSET_DIR 未设置，将使用 mani_skill 默认缓存"; fi

echo "=== 5. openpi（训练/服务侧，Python 3.11）==="
chk "OPENPI_ROOT 已设置" test -n "$OPENPI_ROOT"
if [ -n "$OPENPI_ROOT" ]; then
    chk "openpi 目录存在" test -d "$OPENPI_ROOT"
    chk "openpi_client 可 import" "$PY" -c "import openpi_client"
fi

echo "=== 6. π0.5 base 权重 ==="
if [ -n "${PI05_BASE:-}" ]; then
    "$PY" -c "
import os, numpy as np, jax
from openpi.models import model as m
p = m.restore_params(os.environ['PI05_BASE'], restore_type=np.ndarray)
flat = jax.tree_util.tree_leaves(p)
print(f'  张量 {len(flat)} / 参数 {sum(x.size for x in flat)/1e9:.3f} B   （期望 51 / 3.353 B）')
" 2>&1 | tail -2
else
    echo "  ⚠️  PI05_BASE 未设置，跳过（需要 12.44 GB，见 docs/03_runbook.md 第 1 步）"
fi

echo "=== 7. 官方演示 ==="
H5="${DEMO_ROOT:-$REPO_ROOT/demos/extracted}/PickCube-v1/motionplanning/trajectory.h5"
if [ -f "$H5" ]; then echo "  ✅ $H5"; else echo "  ❌ 缺少 $H5（跑 01_download_demos.sh）"; fi

echo "=== 8. 磁盘与 GPU ==="
echo "  可用磁盘: $(df -h "$REPO_ROOT" 2>/dev/null | awk 'NR==2{print $4}')（训练需 ≥120 GB）"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  GPU: /' || echo "  ⚠️  nvidia-smi 不可用"

echo
echo "结果：$ok 通过 / $bad 未通过"
