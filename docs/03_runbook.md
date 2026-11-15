# 运行手册

> 从零复现的完整步骤。每一步都给了可直接执行的命令与需要修改的代码位置。

### 7.1 环境要求

| 项 | 要求 |
|---|---|
| GPU | ≥ 40 GB 显存（L40 / A100 / H100）|
| 磁盘 | ≥ 120 GB（权重 13 GB + 数据 25 GB + 断点 30 GB + 缓冲）|
| Python | openpi 需 3.11；ManiSkill 需 3.10 |
| 长期依赖 | openpi（JAX）、ManiSkill 3.0.1 + SAPIEN 3.0.3、Vulkan |

### 7.2 第一步：下载 π0.5 base 权重

```bash
# 官方源（GCS，国内慢）：gs://openpi-assets/checkpoints/pi05_base
# 国内镜像（推荐）：
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('hairuoliu/pi05_base', local_dir='<你的路径>/pi05_base')
"
```

**⚠️ 必须校验**：`params/ocdbt.process_0/d/` 下应有 12 个文件，
总大小 12.44 GB。逐块对比官方 API 的 Size 字段（见坑 9）。

最终目录结构：

```
pi05_base/
└── params/
    ├── _CHECKPOINT_METADATA
    ├── _METADATA
    ├── manifest.ocdbt
    └── ocdbt.process_0/d/   ← 12 个数据块，共 12.44 GB
```

验证：

```bash
cd <openpi根目录>
uv run python -c "
import numpy as np, jax
from openpi.models import model as m
p = m.restore_params('<路径>/pi05_base/params', restore_type=np.ndarray)
flat = jax.tree_util.tree_leaves(p)
print('张量:', len(flat), '参数:', round(sum(x.size for x in flat)/1e9, 3), 'B')
"   # 期望：51 张量，3.353 B
```

### 7.3 第二步：下载官方演示

```bash
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('haosulab/ManiSkill_Demonstrations', 'demos/PickCube-v1.zip',
                repo_type='dataset', local_dir='<路径>/demos')
"
cd <路径>/demos && unzip PickCube-v1.zip
# 得到 PickCube-v1/motionplanning/trajectory.h5（1000 条轨迹）
```

### 7.4 第三步：改 ManiSkill 源码（1 处）

**文件**：`<site-packages>/mani_skill/envs/tasks/tabletop/pick_cube.py`

```python
# 改动 1：__init__ 里，在 self.cube_half_size = cfg["cube_half_size"] 之后加一行
self.cube_color = cfg.get("cube_color", [1, 0, 0, 1])

# 改动 2：_load_scene 里，把
#   color=[1, 0, 0, 1],
# 改成
#   color=self.cube_color,
```

**用法**：

```python
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
PICK_CUBE_CONFIGS["panda"]["cube_color"] = [0.15, 0.35, 0.85, 1.0]   # 蓝色
env = gym.make("PickCube-v1", ...)   # 构造前设置才生效
```

### 7.5 第四步：生成数据

```bash
# 三套基线数据（IND / OOD-相机 / OOD-颜色）
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
MS_ASSET_DIR=<ms_assets路径> \
python 02_build_dataset.py --split ind --n 600
python 02_build_dataset.py --split ood_color --n 600
python 02_build_dataset.py --split ood_cam --n 600
# 或者一条命令：bash 02b_gen_all_splits.sh 600

# 随机化相机数据（官方演示只有 1000 条轨迹）
python 03_build_dataset_randcam.py --n 1000
```

**脚本里的关键参数**：

```python
# build_dataset.py
TASKS = {"PickCube-v1": ("PickCube-v1/motionplanning/trajectory.h5",
                         "pick up the cube and move it to the goal")}
SPLITS = {
    "ind":       {"colors": [[1,0,0,1],[0.1,0.7,0.15,1]],        # 红/绿
                  "cam": ([0.45,0.30,0.38], [0.02,0.0,0.03])},
    "ood_color": {"colors": [[0.15,0.35,0.85,1],[0.85,0.65,0.1,1]],  # 蓝/黄
                  "cam": ([0.45,0.30,0.38], [0.02,0.0,0.03])},
    "ood_cam":   {"colors": [[1,0,0,1],[0.1,0.7,0.15,1]],
                  "cam": ([0.58,0.42,0.50], [0.02,0.0,0.03])},   # 外移+抬高 ~20°
}

# build_dataset_randcam.py
BASE  = np.array([0.45, 0.30, 0.38])
DELTA = np.array([0.13, 0.12, 0.12])     # t=1 即 20°
TS = [0.0, 0.25, 0.5, 0.75, 1.0]         # 5 档
# 颜色按块轮换，与相机解耦：
color = COLORS[(idx // len(TS)) % len(COLORS)]
```

### 7.6 第五步：转 LeRobot

```bash
python 04_convert_to_lerobot.py \
  --src dataset/ind \
  --root lerobot \
  --repo-id pickcube_ind \
  --fps 20
```

**脚本里三个必须注意的点**（对应坑 3）：

```python
# 1. 图像必须是 struct<bytes,path>
img_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])

# 2. parquet 内嵌 HF 元数据
hf_meta = {"info": {"features": {"image": {"_type": "Image"}, ...}}}
schema = schema.with_metadata({b"huggingface": json.dumps(hf_meta).encode()})

# 3. chunks_size 大于总 episode 数
info = {"chunks_size": 10000, ...}
```

### 7.7 第六步：改 openpi 源码（3 处）

**改动 A：加 `_SliceActions` 类**

文件：`src/openpi/training/config.py`，放在 `class LeRobotLiberoDataConfig` **之前**。

```python
class _SliceActions:
    """把模型输出（padding 到 32 维）截回数据集真实动作维度。"""
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
    def __call__(self, data: dict) -> dict:
        return {"actions": data["actions"][..., : self.action_dim]}
```

**改动 B：`LeRobotLiberoDataConfig` 加字段**

```python
@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    extra_delta_transform: bool = False
    output_action_dim: int = 7          # ← 新增

    def create(self, ...):
        ...
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[_SliceActions(self.output_action_dim)],   # ← 改这里
        )
```

**改动 C：加训练配置**

文件：`src/openpi/training/config.py`，加到 `_CONFIGS` 列表里。

```python
TrainConfig(
    name="pi05_pickcube_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_horizon=50,
        discrete_state_input=False,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotLiberoDataConfig(
        repo_id="pickcube_ind",
        base_config=DataConfig(prompt_from_task=True),
        extra_delta_transform=False,
        output_action_dim=8,          # ← 关键：PickCube 是 8 维
    ),
    batch_size=8,
    num_workers=8,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=500, peak_lr=5e-5, decay_steps=12_000, decay_lr=1e-5),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "<你的路径>/pi05_base/params"),
    freeze_filter=pi0_config.Pi0Config(     # ← 漏了这行会 OOM（坑 4）
        pi05=True, action_horizon=50, discrete_state_input=False,
        paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    num_train_steps=12_000,
),
```

> 随机化版本复制一份，改 `name`、`repo_id="pickcube_ind_randcam"` 即可。

### 7.8 第七步：算归一化统计量

```bash
cd <openpi根目录>
HF_LEROBOT_HOME=<lerobot路径> uv run scripts/compute_norm_stats.py \
  --config-name pi05_pickcube_lora
# 输出：assets/pi05_pickcube_lora/pickcube_ind/norm_stats.json
```

### 7.9 第八步：训练

```bash
cd <openpi根目录>
source ./my_env.sh          # 设置 HF_HOME / HF_LEROBOT_HOME / VK_ICD 等
export HF_LEROBOT_HOME=<lerobot路径>
export CUDA_VISIBLE_DEVICES=0

uv run scripts/train.py pi05_pickcube_lora \
  --exp-name pickcube_lora \
  --batch-size 8 \
  --num-train-steps 12000 \
  --save-interval 6000 \
  --checkpoint-base-dir <断点目录> \
  --no-wandb-enabled
```

**预期**：1.7 秒/步，5.7 小时，显存 34.7 GB，loss 从 0.0677 降到 0.0044。

**断点续训**（崩溃后）：加 `--resume`，去掉 `--overwrite`。

### 7.10 第九步：起服务

```bash
cd <openpi根目录>
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_pickcube_lora \
  --policy.dir=<断点目录>/11999
# INFO:websockets.server:server listening on 0.0.0.0:8000
```

### 7.11 第十步：评测

```bash
export OPENPI_ROOT=<openpi根目录>
export PY=<含 mani_skill 的 python>
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export MS_ASSET_DIR=<ms_assets路径>

# 评测脚本是 WebSocket 客户端，不加载权重：测哪个模型由服务端决定
$PY 08_eval_single_split.py \
  --split ind \
  --episodes 100 \
  --out results/ind.json

# 或者直接跑完整 12 格（推荐）：
bash 09_eval_full_protocol.sh baseline results/baseline
```

**评测脚本里两个关键点**（对应坑 5、坑 2）：

```python
# 1. 必须用 pd_joint_pos，与演示数据一致
env = gym.make("PickCube-v1", obs_mode="rgb", render_mode="rgb_array",
               sensor_configs=cfg, num_envs=1,
               control_mode="pd_joint_pos")      # ← 漏了会全 0%

# 2. 目标球必须显示
env.reset(seed=0)
env.goal_site.show_visual(); env._hidden_objects = []
```

### 7.12 复现检查清单

| # | 检查项 | 期望 |
|---|---|---|
| 1 | base 权重 | 51 张量 / 3.353 B |
| 2 | 数据规模 | 每套 1000+ 条 / 7.8 万+ 帧 |
| 3 | LeRobot 可加载 | `ds[0]["image"].shape == (3,256,256)` |
| 4 | 训练显存 | 34–35 GB（若 >40 GB 说明 freeze_filter 没生效）|
| 5 | 训练速度 | 1.7 秒/步 |
| 6 | loss 起点 | 0.06–0.10 |
| 7 | 服务返回动作形状 | `(50, 8)`（若是 7 说明 `_SliceActions` 没接上）|
| 8 | 评测非零 | IND 应 > 40%（若 0% 检查 control_mode）|

---

