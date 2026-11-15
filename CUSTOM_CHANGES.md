# 项目自定义改动索引

本文区分 **ManiSkill / openpi 上游原有能力** 与本项目新增工作，便于代码审查时快速定位实现证据。

---

## 1. 数据管线（全部新增）

| 文件 | 作用 |
|---|---|
| `scripts_local/01_download_demos.sh` | 下载官方演示（只有 state + action，无图像）|
| `scripts_local/02_build_dataset.py` | 回放渲染 + 三套数据（IND / OOD-颜色 / OOD-相机）|
| `scripts_local/03_build_dataset_randcam.py` | 5 档随机相机的训练数据 |
| `scripts_local/04_convert_to_lerobot.py` | npz → LeRobot v2.1（含 HF 元数据）|

**核心方法**：官方演示存了每步完整 `env_states`，恢复后重新渲染，
吞吐 **228 帧/秒**（1000 条轨迹 5 分钟）。
改变外观（颜色/相机）不影响动力学，因此原始动作序列在新视角下**仍然有效**——
这是用回放做数据增广的前提。

**LeRobot 格式的三个硬性要求**（否则加载失败）：
1. 图像列必须是 `struct<bytes, path>`
2. parquet 内必须内嵌 `huggingface` 元数据声明 `{"_type": "Image"}`
3. `chunks_size` 必须大于总 episode 数

---

## 2. openpi 配置适配（补丁）

**文件**：`openpi_patch/pickcube_lora_config.patch`（3 处改动）

### 2.1 新增 `_SliceActions`

openpi 的 `LiberoOutputs` 把动作硬编码截成 **7 维**（LIBERO 的维度），
但 ManiSkill PickCube 是 **8 维**（7 关节 + 1 夹爪）。
新增一个可配置的截断类：

```python
class _SliceActions:
    """把模型输出（padding 到 32 维）截回数据集真实动作维度。"""
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
    def __call__(self, data: dict) -> dict:
        return {"actions": data["actions"][..., : self.action_dim]}
```

> 坑：**不能**继承 `transforms.DataTransformFn` 再用 `@dataclass(frozen=True)`，
> 因为它是个 `Protocol`，会报 `Cannot overwrite attribute __setattr__`。

### 2.2 `LeRobotLiberoDataConfig` 加字段

```python
extra_delta_transform: bool = False
output_action_dim: int = 7          # ← 新增
...
outputs=[_SliceActions(self.output_action_dim)],   # ← 原为 LiberoOutputs()
```

### 2.3 两个训练配置

- `pi05_pickcube_lora`：基线（固定相机），`repo_id="pickcube_ind"`
- `pi05_pickcube_randcam`：随机化，`repo_id="pickcube_ind_randcam"`

两者都是 `paligemma_variant="gemma_2b_lora"` + `action_expert_variant="gemma_300m_lora"`，
**必须带 `freeze_filter`**（漏了会变成全参数微调 → 46 GB 显存不够 → OOM）。

---

## 3. ManiSkill 环境适配（补丁）

**文件**：`maniskill_patch/pick_cube_color.patch`（2 处改动）

`pick_cube.py` 原来把方块颜色硬编码为红色，无法做 OOD 颜色实验：

```python
# 改动 1：__init__ 中
self.cube_color = cfg.get("cube_color", [1, 0, 0, 1])

# 改动 2：_load_scene 中
color=self.cube_color,      # 原：color=[1, 0, 0, 1],
```

用法：构造环境**之前**修改配置字典

```python
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
PICK_CUBE_CONFIGS["panda"]["cube_color"] = [0.15, 0.35, 0.85, 1.0]  # 蓝
env = gym.make("PickCube-v1", ...)
```

**另外两处必须知道的隐式要求**（无需改源码，但评测脚本必须写对）：

```python
# 1. 控制模式必须与演示一致（绝对关节角），否则全 0%
env = gym.make("PickCube-v1", ..., control_mode="pd_joint_pos")

# 2. 目标球默认被隐藏，必须显示
env.unwrapped.goal_site.show_visual()
env.unwrapped._hidden_objects = []
```

---

## 4. 评测系统（全部新增）

| 文件 | 作用 |
|---|---|
| `scripts_local/08_eval_single_split.py` | 单格评测（WebSocket 客户端）|
| `scripts_local/09_eval_full_protocol.sh` | 完整 12 格协议 |
| `scripts_local/10_watch_and_eval.sh` | 看门狗：等断点 → 自动起服务 → 自动评测 |
| `scripts_local/11_record_videos.py` | 录制评测视频 |

**公共层**：`scripts_local/common.sh` 统一管理 `$OPENPI_ROOT` / `$LEROBOT_HOME` / `$PY` 等环境变量，
所有脚本 `source` 它，仓库内**没有硬编码的机器路径**，换机器只改环境变量。

**历史脚本**：`scripts_local/_archive/` 保留 7 个训练机上的原始脚本（含机器绝对路径），
仅用于回溯「当时到底跑了什么」，直接运行会失败。见 [`scripts_local/_archive/README.md`](scripts_local/_archive/README.md)。

**双进程架构**（因 Python 环境无法合并）：

```
openpi venv (Python 3.11, JAX)  ──WebSocket──  ManiSkill venv (Python 3.10, SAPIEN)
   策略推理服务                                  环境交互 + 观测
```

`openpi_client` 是纯 Python 包，用 `PYTHONPATH` 引入 ManiSkill 环境即可。

**12 格评测矩阵**：

| 组别 | 格子 | 集数 |
|---|---|---|
| 主实验 | ind / ood_color / ood_cam | 100 |
| 正向轴扫描 | 0° / 5° / 10° / 14° / 20° / 25° | 50 |
| **正交轴**（真泛化判据）| 反向 / 侧向 / 垂直 | 50 |

---

## 5. 实验设计与分析（全部新增）

| 文件 | 内容 |
|---|---|
| `docs/01_experiment_design_and_results.md` | 实验设计、12 格结果、三层解读 |
| `docs/02_pitfalls_and_failures.md` | **12 个实际踩过的坑**：现象 → 原因 → 解法 |
| `docs/03_runbook.md` | 从零复现的 12 步 + 8 项检查清单 |
| `docs/04_deviations_and_limitations.md` | 所有被简化的操作与无法消除的偏差 |
| `media/COMPARISON.png` | 首末帧对比图 |
| `media/*.mp4` | 8 段录像（含同种子 A/B）|

---

## 6. 公开仓库边界

本仓库**不包含**：

```text
模型权重（π0.5 base 12.44 GB、训练断点 15 GB/个）
LeRobot 数据集（约 25 GB）
ManiSkill 仿真资产（8.2 GB）
虚拟环境
训练日志
录像原始帧
```

补丁基于本项目使用的版本生成：

| 上游 | 版本 |
|---|---|
| openpi | 见 `pyproject.toml`（JAX 0.5.3 / Flax 0.10.2）|
| ManiSkill | 3.0.1 |
| SAPIEN | 3.0.3 |

应用补丁前请先 `--dry-run` 检查；`weight_loader` 路径是占位符，需改成自己的。
