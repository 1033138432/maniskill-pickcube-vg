#!/usr/bin/env bash
set -uo pipefail
O=/root/gpufree-data/openpi; M=/root/gpufree-data/maniskill_proj
PY=$O/.cache/rbm/.dsynth-env/bin/python
export PYTHONPATH=$O/packages/openpi-client/src:$O/.cache/rbm/.dsynth-env/lib/python3.10/site-packages
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export MS_ASSET_DIR=/root/gpufree-data/rbm_eval_assets/ms_assets
E=30
$PY $M/scripts/eval_pickcube.py --config x --ckpt x --split ind --episodes $E \
    --cam-eye 0.45,0.30,0.38 --out $M/results/cam_0deg.json > $M/logs/cam_0deg.log 2>&1
$PY $M/scripts/eval_pickcube.py --config x --ckpt x --split ind --episodes $E \
    --cam-eye 0.4833,0.33,0.412 --out $M/results/cam_7deg.json > $M/logs/cam_7deg.log 2>&1
$PY $M/scripts/eval_pickcube.py --config x --ckpt x --split ind --episodes $E \
    --cam-eye 0.5167,0.36,0.444 --out $M/results/cam_14deg.json > $M/logs/cam_14deg.log 2>&1
$PY $M/scripts/eval_pickcube.py --config x --ckpt x --split ind --episodes $E \
    --cam-eye 0.58,0.42,0.50 --out $M/results/cam_20deg.json > $M/logs/cam_20deg.log 2>&1
echo "=== 视角扫描完成 ==="
python3 -c "
import json,glob
for f in sorted(glob.glob('$M/results/cam_*.json')):
    d=json.load(open(f)); s=d['summary']
    print(f\"{f.split('/')[-1]:16s} {s['success']:2d}/{s['episodes']} = {s['rate']*100:5.1f}%\")"
