#!/usr/bin/env bash
# 所有脚本共用的环境变量与工具函数。
# 用法：在脚本开头 source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
#
# 必须自己 export 的（没有默认值）：
#   OPENPI_ROOT    openpi 仓库根目录
#   LEROBOT_HOME   LeRobot 数据集根目录
# 建议 export 的：
#   PY             含 mani_skill / gymnasium 的 Python 解释器（默认 python）
#   MS_ASSET_DIR   ManiSkill 仿真资产目录（8.2 GB，不设则用 mani_skill 默认缓存）
#   GPU            CUDA 可见卡号（默认 0）
#
# 其余全部有默认值，指向仓库内部，不需要设置。

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

export PY="${PY:-python}"
export GPU="${GPU:-0}"
export OPENPI_ROOT="${OPENPI_ROOT:-}"
export LEROBOT_HOME="${LEROBOT_HOME:-}"
export MS_ASSET_DIR="${MS_ASSET_DIR:-}"

# 数据/结果/日志都默认落在仓库内部，可用环境变量覆盖
export DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/dataset}"
export RESULTS_ROOT="${RESULTS_ROOT:-$REPO_ROOT/results}"
export LOG_ROOT="${LOG_ROOT:-$REPO_ROOT/logs}"
export CKPT_ROOT="${CKPT_ROOT:-$REPO_ROOT/checkpoints}"

# 让评测进程能 import openpi_client（纯 Python 包，跨 venv 可用）
if [ -n "$OPENPI_ROOT" ]; then
    export PYTHONPATH="$OPENPI_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
fi

# 检查必填变量：req OPENPI_ROOT "/path/to/openpi" || exit 1
req() {
    local var="$1" hint="${2:-}"
    if [ -z "${!var:-}" ]; then
        echo "错误：环境变量 $var 未设置。" >&2
        [ -n "$hint" ] && echo "  建议：export $var=$hint" >&2
        return 1
    fi
}

# 策略服务（默认 8000）是否就绪。用 python socket，避免依赖 ss/netstat。
port_open() {
    local port="${1:-8000}"
    "$PY" - "$port" <<'PY' >/dev/null 2>&1
import socket, sys
s = socket.socket()
s.settimeout(1.0)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
}

# 等待策略服务就绪：wait_policy [端口] [最多等几轮] [每轮间隔秒]
wait_policy() {
    local port="${1:-8000}" tries="${2:-120}" gap="${3:-5}" i
    for ((i = 1; i <= tries; i++)); do
        port_open "$port" && return 0
        sleep "$gap"
    done
    echo "错误：等了 $((tries * gap)) 秒，端口 $port 仍不可用（策略服务没起？见 07b_serve_policy.sh）" >&2
    return 1
}

mkdir -p "$LOG_ROOT" 2>/dev/null || true
