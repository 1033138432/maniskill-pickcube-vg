# 评测原始数据

`baseline/` 与 `randcam/` 各 12 个 JSON，与根目录 [README](../README.md) 的主表一一对应，
合计 **24 格 / 1500 集**（每模型 750 集）。

## 文件名含义

| 文件 | 相机 eye | 集数 | 说明 |
|---|---|---|---|
| `ind.json` | `[0.45, 0.30, 0.38]` | 100 | 分布内：红/绿方块 + 训练用相机 |
| `ood_color.json` | 同上 | 100 | 未见颜色：蓝/黄 |
| `ood_cam.json` | `[0.58, 0.42, 0.50]` | 100 | 训练轴外移 ≈20° |
| `cam_0deg` … `cam_25deg` | `base + t·Δ` | 50 | 训练轴正向扫描 6 档 |
| `cam_reverse.json` | `[0.32, 0.18, 0.26]` | 50 | **正交轴**：反向（往里收） |
| `cam_lateral.json` | `[0.45, 0.48, 0.42]` | 50 | **正交轴**：侧向 |
| `cam_vertical.json` | `[0.45, 0.30, 0.52]` | 50 | **正交轴**：垂直抬高 |

三条正交轴是训练时**从未覆盖**的方向，也是「随机化是否学到真泛化」的判据。

## 为什么 `cam_*.json` 里写着 `"split": "ind"`

视角扫描沿用 `--split ind` 的颜色与种子规则，只用 `--cam-eye` 覆盖相机位置，
所以 `split` 字段保持 `ind`。真实相机位姿以本文上表为准。

## 字段

```jsonc
{
  "summary": {
    "split": "ind", "episodes": 50, "success": 45, "rate": 0.9,
    "cam_eye": [...], "fov": 1.05, "size": 256    // 早期 JSON 无后三个字段
  },
  "episodes": [
    {"episode": 0, "seed": 1000, "success": true, "steps": 62, "color": [1.0, 0.0, 0.0, 1.0]}
  ]
}
```

> 早期版本的 JSON 只有 `summary` 的 `split/episodes/success/rate` 与逐集
> `episode/success/steps/color`，缺 `seed`、`cam_eye`、`fov`、`size`（当时脚本还没加这些字段）。
> 这不影响成功率统计——种子固定为 `1000 + episode`，两个模型完全一致。

## 复现

```bash
# 终端 1：起服务（决定评测的是哪个模型）
bash scripts_local/07b_serve_policy.sh pi05_pickcube_lora <断点目录>/11999

# 终端 2：跑完整 12 格
bash scripts_local/09_eval_full_protocol.sh baseline results/baseline
```

## 一致性说明

根 README 主表的每一个数字都与这里的 JSON 逐格核对过（24/24 一致）。
若要复核：

```bash
python - <<'PY'
import json, glob, os
for m in ("baseline", "randcam"):
    for f in sorted(glob.glob(f"results/{m}/*.json")):
        s = json.load(open(f))["summary"]
        print(f"{m:9s} {os.path.basename(f)[:-5]:14s} {s['success']:3d}/{s['episodes']:<3d} = {s['rate']*100:5.1f}%")
PY
```
