#!/usr/bin/env bash
# 09 完整 12 格评测协议 —— 产出 README 主表的那一套。
#    主实验 3 格 × 100 集 + 视角扫描 6 档 × 50 集 + 三条正交相机轴 × 50 集
#
# 用法:
#   bash 09_eval_full_protocol.sh <tag> <out_dir> [主实验集数=100] [单格集数=50]
#   例: bash 09_eval_full_protocol.sh baseline results/baseline
#
# 前置: 07b_serve_policy.sh 已在 8000 端口起好服务（评测哪个模型由服务端决定）
#
# 环境变量: 见 common.sh（至少需要 PY 指向含 mani_skill 的解释器）
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TAG=${1:?用法: $0 <tag> <out_dir> [主实验集数] [单格集数]}
OUT=${2:?用法: $0 <tag> <out_dir> [主实验集数] [单格集数]}
MAIN_EPS=${3:-100}
CELL_EPS=${4:-50}

SCRIPT="$REPO_ROOT/scripts_local/08_eval_single_split.py"
[ -f "$SCRIPT" ] || { echo "错误：找不到 $SCRIPT" >&2; exit 1; }

mkdir -p "$OUT" "$LOG_ROOT"
echo "===== $TAG 完整评测 $(date '+%F %H:%M') ====="
echo "  主实验 ${MAIN_EPS} 集/格，扫描与正交轴 ${CELL_EPS} 集/格，输出 → $OUT"

wait_policy 8000 60 5 || exit 1

run() {  # run <名字> <split> <集数> [额外参数...]
    local name=$1 split=$2 eps=$3
    shift 3
    "$PY" "$SCRIPT" --split "$split" --episodes "$eps" "$@" \
        --out "$OUT/$name.json" > "$LOG_ROOT/${TAG}_${name}.log" 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        "$PY" - "$OUT/$name.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
s = d["summary"]
print(f"  [OK]   {sys.argv[1].rsplit('/', 1)[-1][:-5]:14s} {s['success']:3d}/{s['episodes']:<3d} = {s['rate']*100:5.1f}%")
PY
    else
        echo "  [FAIL] $name 失败 rc=$rc（见 $LOG_ROOT/${TAG}_${name}.log）"
    fi
}

# 1) 三套主实验
run ind        ind        "$MAIN_EPS"
run ood_color  ood_color  "$MAIN_EPS"
run ood_cam    ood_cam    "$MAIN_EPS"

# 2) 训练轴正向扫描：6 档。eye = base + t*delta，t 由角度换算
for spec in 0:0.45,0.30,0.38 5:0.474,0.325,0.402 10:0.497,0.350,0.424 \
            14:0.516,0.371,0.443 20:0.58,0.42,0.50 25:0.615,0.45,0.53; do
    ang=${spec%%:*}
    eye=${spec#*:}
    run "cam_${ang}deg" ind "$CELL_EPS" --cam-eye "$eye"
done

# 3) 三条正交相机轴（训练完全没覆盖，真正的泛化判据）
run cam_reverse  ind "$CELL_EPS" --cam-eye 0.32,0.18,0.26   # 反向：往里收
run cam_lateral  ind "$CELL_EPS" --cam-eye 0.45,0.48,0.42   # 侧向：左右偏
run cam_vertical ind "$CELL_EPS" --cam-eye 0.45,0.30,0.52   # 垂直：纯抬高

echo "===== $TAG 评测完成 $(date '+%F %H:%M') ====="
