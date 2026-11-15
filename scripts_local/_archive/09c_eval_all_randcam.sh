#!/usr/bin/env bash
# 三套评测：用 dsynth 环境（有 mani_skill）+ openpi_client 连本地服务
set -uo pipefail
O=/root/gpufree-data/openpi
M=/root/gpufree-data/maniskill_proj
PY=$O/.cache/rbm/.dsynth-env/bin/python
export PYTHONPATH=$O/packages/openpi-client/src:$O/.cache/rbm/.dsynth-env/lib/python3.10/site-packages
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export MS_ASSET_DIR=/root/gpufree-data/rbm_eval_assets/ms_assets
mkdir -p $M/results_randcam $M/logs
for split in ind ood_color ood_cam; do
  echo "===== $split 开始 $(date +%H:%M) ====="
  $PY $M/scripts/eval_pickcube.py --config pi05_pickcube_randcam --ckpt /root/gpufree-data/openpi/checkpoints_randcam/pi05_pickcube_randcam/pickcube_randcam/11999 --split "$split" \
      --episodes "${EPISODES:-20}" --out "$M/results_randcam/$split.json" \
      > "$M/logs/eval_$split.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "❌ $split 失败 rc=$rc"; tail -4 "$M/logs/eval_$split.log" | cut -c1-140
  else
    python3 -c "
import json
d=json.load(open('$M/results_randcam/$split.json')); s=d['summary']
print(f\"✅ $split: {s['success']}/{s['episodes']} = {s['rate']*100:.1f}%\")"
  fi
done
echo "=== 全部完成 $(date +%H:%M) ==="
