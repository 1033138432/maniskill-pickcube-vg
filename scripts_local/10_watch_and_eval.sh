#!/usr/bin/env bash
# 10 看门狗：等断点出现 → 自动起策略服务 → 自动跑完整评测 → 落盘。
#    不依赖任何人在线，夜里训练跑完也能自动接上评测。
#
# 用法: bash 10_watch_and_eval.sh <配置名> <断点目录> [tag]
#   例: bash 10_watch_and_eval.sh pi05_pickcube_randcam \
#         "$CKPT_ROOT/checkpoints_randcam/pi05_pickcube_randcam/pickcube_randcam/11999" randcam
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

CONFIG=${1:?用法: $0 <配置名> <断点目录> [tag]}
CKPT=${2:?用法: $0 <配置名> <断点目录> [tag]}
TAG=${3:-$CONFIG}
OUT="$RESULTS_ROOT/$TAG"
LOG="$LOG_ROOT/watch_${TAG}.log"

log() { echo "$(date '+%F %H:%M:%S') $*" | tee -a "$LOG"; }

log "看门狗启动，等待断点 $CKPT"
for i in $(seq 1 240); do          # 最多等 2 小时
    [ -f "$CKPT/_CHECKPOINT_METADATA" ] && break
    sleep 30
done
[ -f "$CKPT/_CHECKPOINT_METADATA" ] || { log "❌ 超时，断点未出现"; exit 1; }
log "✅ 断点就绪 → 起策略服务"

req OPENPI_ROOT "/path/to/openpi" || exit 1
(
    cd "$OPENPI_ROOT" || exit 1
    CUDA_VISIBLE_DEVICES="$GPU" setsid nohup uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config="$CONFIG" --policy.dir="$CKPT" \
        > "$LOG_ROOT/serve_${TAG}.log" 2>&1 < /dev/null &
)

wait_policy 8000 60 5 || { log "❌ 策略服务未就绪，见 $LOG_ROOT/serve_${TAG}.log"; exit 1; }
log "✅ 服务就绪 → 完整 12 格评测"

bash "$REPO_ROOT/scripts_local/09_eval_full_protocol.sh" "$TAG" "$OUT" >> "$LOG" 2>&1
log "🎉 $TAG 评测完成 → $OUT"
