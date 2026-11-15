#!/usr/bin/env bash
# 02b 批量生成三套数据（IND / OOD-颜色 / OOD-相机）
# 用法: bash 02b_gen_all_splits.sh [每色轨迹数=600]
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

N=${1:-600}
S="$REPO_ROOT/scripts_local/02_build_dataset.py"
[ -f "$S" ] || { echo "错误：找不到 $S" >&2; exit 1; }

for split in ind ood_color ood_cam; do
    echo "===== $split ====="
    "$PY" "$S" --split "$split" --n "$N" || echo "❌ $split 失败"
done
echo "=== 全部完成 ==="
du -sh "$DATA_ROOT"/* 2>/dev/null
