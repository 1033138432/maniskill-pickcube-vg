# 评测录像与对照动画

渲染 256×256（与评测协议一致），20 fps，8 段 mp4 + 2 段并排对照 GIF。

## 对照动画（README 内嵌的就是这两个）

GitHub 的 Markdown 不会内联播放 mp4，所以关键对照做成了 GIF。

| 文件 | 左 | 右 | 说明 |
|---|---|---|---|
| [`comparison_baseline_vs_randcam_cam20.gif`](comparison_baseline_vs_randcam_cam20.gif) | 基线 @20° → 失败 | 随机化 @20° → 成功 | **同种子 1000、同相机**，训练轴上的差距 |
| [`comparison_randcam_cam20_vs_reverse.gif`](comparison_randcam_cam20_vs_reverse.gif) | 随机化 @20°（覆盖）→ 成功 | 随机化 反向（未覆盖）→ 失败 | **同一套权重**，正交方向零迁移 |

两段都是 605×348 / 12 fps / 13 s（右侧先结束，之后冻结并压暗标注结束时刻），
合计约 3.4 MB，由 ffmpeg 的 `palettegen` + `paletteuse` 编码。

重新生成：

```bash
pip install imageio imageio-ffmpeg      # 只在做 GIF 时需要，会自带 ffmpeg
python scripts_local/12_make_comparison_gifs.py
python scripts_local/12_make_comparison_gifs.py --panel 240 --fps 10   # 想要更小
```

## mp4 清单

| 文件 | 模型 | 相机 | 种子 | 结果 | 步数 |
|---|---|---|---|---|---|
| `baseline_cam0_seed1000_success_64steps.mp4` | 基线 | 0° | 1000 | ✅ | 64 |
| `baseline_cam0_seed1001_fail_200steps.mp4` | 基线 | 0° | 1001 | ❌ | 200 |
| `baseline_cam20_seed1000_fail_200steps.mp4` | 基线 | 20° | 1000 | ❌ | 200 |
| `baseline_cam20_seed1001_fail_200steps.mp4` | 基线 | 20° | 1001 | ❌ | 200 |
| `randcam_cam0_seed1000_success_68steps.mp4` | 随机化 | 0° | 1000 | ✅ | 68 |
| `randcam_cam20_seed1000_success_62steps.mp4` | 随机化 | 20° | 1000 | ✅ | 62 |
| `randcam_cam20_seed1001_success_75steps.mp4` | 随机化 | 20° | 1001 | ✅ | 75 |
| `randcam_reverse_seed1000_fail_200steps.mp4` | 随机化 | 反向（未见） | 1000 | ❌ | 200 |

## 最能说明问题的两对

**第一对**（同种子 1000，同相机 20°）：

```
baseline_cam20_seed1000_fail_200steps.mp4    跑满 200 步未完成
randcam_cam20_seed1000_success_62steps.mp4   62 步完成
```

**第二对**（随机化模型自己）：

```
randcam_cam20_seed1000_success_62steps.mp4   20°（训练覆盖）→ 成功
randcam_reverse_seed1000_fail_200steps.mp4   反向（未见）    → 失败
```

第二对是核心证据：**同一套权重，训练覆盖的方向成功，未覆盖的方向完全失败。**

## 一个录制时的意外发现

首次录制用 512×512 渲染（评测是 256×256），基线在 0° 下 5/5 全败；
改回 256 后恢复正常（种子 1000 成功，64 步）。
说明该策略对**渲染分辨率**同样敏感——比相机扰动更隐蔽的分布偏移。
