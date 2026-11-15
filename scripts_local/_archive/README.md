# 历史脚本存档

这里放的是**当时在训练机上实际执行的原始脚本**，原样保留（含机器上的绝对路径），
仅为留证。它们不是本仓库推荐的工作流，**直接运行会失败**。

| 存档文件 | 现在用什么 |
|---|---|
| `06b_train_baseline_actual.sh` | [`06_train_baseline.sh`](../06_train_baseline.sh) |
| `07c_train_randcam_actual.sh` | [`07_train_randcam.sh`](../07_train_randcam.sh) |
| `09b_eval_all_splits.sh` | [`09_eval_full_protocol.sh`](../09_eval_full_protocol.sh)（只跑 3 格主实验就调小集数） |
| `09c_eval_all_randcam.sh` | [`09_eval_full_protocol.sh`](../09_eval_full_protocol.sh) |
| `09d_sweep_camera_baseline.sh` | [`09_eval_full_protocol.sh`](../09_eval_full_protocol.sh) 的视角扫描段 |
| `09e_sweep_camera_randcam.sh` | [`09_eval_full_protocol.sh`](../09_eval_full_protocol.sh) 的视角扫描段 |
| `10b_run_all_eval_pipeline.sh` | [`09_eval_full_protocol.sh`](../09_eval_full_protocol.sh) + [`10_watch_and_eval.sh`](../10_watch_and_eval.sh) |

## 为什么归档而不删

1. 里面写死了 `/root/gpufree-data/...` 这类训练机路径，换台机器就跑不了；
2. 调用的是机器上的旧文件名（`eval_pickcube.py`、`build_dataset.py`、`run_full_eval.sh`），
   与仓库里重整后的 `08_eval_single_split.py`、`02_build_dataset.py`、`09_eval_full_protocol.sh` 对不上；
3. 部分已被后面扩展到 12 格的版本取代——视角档位从 0/7/14/20 扩到 0/5/10/14/20/25，
   反向相机从 30 集扩到 50 集，并新增侧向、垂直两条正交轴。

保留它们是为了能配合 [`docs/04_deviations_and_limitations.md`](../../docs/04_deviations_and_limitations.md)
回溯「当时到底跑了什么、哪一步被改过」。这也是本仓库可信度的一部分：
**能查到原始命令行，而不是只给一份整理过的漂亮脚本。**
