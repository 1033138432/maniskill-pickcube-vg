#!/usr/bin/env bash
# 一键：启动服务 → 三套评测 → 视角扫描 → 反向相机对照
set -uo pipefail
O=/root/gpufree-data/openpi; M=/root/gpufree-data/maniskill_proj
PY=$O/.cache/rbm/.dsynth-env/bin/python
export PYTHONPATH=$O/packages/openpi-client/src:$O/.cache/rbm/.dsynth-env/lib/python3.10/site-packages
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export MS_ASSET_DIR=/root/gpufree-data/rbm_eval_assets/ms_assets
mkdir -p $M/results_randcam $M/logs
# 等服务就绪
for i in $(seq 1 60); do ss -ltn 2>/dev/null | grep -q ":8000 " && break; sleep 10; done
echo "服务就绪 $(date +%H:%M)"
# 1) 三套主实验
bash $M/scripts/eval_all_randcam.sh
# 2) 视角扫描
bash $M/scripts/sweep_randcam.sh
# 3) 反向相机（训练完全没见过的方向）
$PY $M/scripts/eval_pickcube.py --config pi05_pickcube_randcam --ckpt /root/gpufree-data/openpi/checkpoints_randcam/pi05_pickcube_randcam/pickcube_randcam/11999 --split ind --episodes 30 \
    --cam-eye 0.32,0.18,0.26 --out $M/results_randcam/cam_reverse.json \
    > $M/logs/cam_reverse.log 2>&1
echo "=== 全部评测完成 $(date +%H:%M) ==="
