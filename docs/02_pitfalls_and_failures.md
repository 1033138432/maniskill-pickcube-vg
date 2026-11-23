# 踩坑与失败分析

> 12 个实际踩过的问题，每个都给出：现象 → 原因 → 解法。
> 这些坑在复现时**大概率会再遇到**。

### 坑 1：官方演示没有图像

**现象**：下载的 `trajectory.h5` 里 `obs_mode = "none"`，只有 state 和 action。

**原因**：官方演示是给 RL 训练用的，不含视觉观测。

**解法**：从 `env_states` 恢复仿真状态后重新渲染。

```python
state = {"actors": {n: torch.tensor(np.asarray(t[f"env_states/actors/{n}"][s]))
                    for n in t["env_states/actors"]},
         "articulations": {...}}
env.unwrapped.set_state_dict(state)
rgb = env.unwrapped.get_obs()["sensor_data"]["base_camera"]["rgb"][0]
```

---

### 坑 2：目标球默认隐藏

**现象**：渲染出来的画面里看不到绿色目标点，模型无法知道该把方块放到哪。

**原因**：PickCube 把 `goal_site` 放进了 `_hidden_objects`（RL 训练时不需要看见目标）。

**解法**：

```python
env.reset(seed=0)
env.unwrapped.goal_site.show_visual()
env.unwrapped._hidden_objects = []
```

---

### 坑 3：LeRobot 图像格式（最坑的一个）

**现象**：加载数据集时报 `RuntimeError: Could not infer dtype of dict`。

**原因**：图像列必须写成 HF 标准的 `struct<bytes, path>`，
且 parquet 文件里必须内嵌 `huggingface` 元数据，声明 `{"_type": "Image"}`。
否则 `datasets` 把图像当普通结构体，不解码。

**解法**：

```python
img_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
schema = pa.schema([("image", img_type), ...])
hf_meta = {"info": {"features": {
    "image": {"_type": "Image"},
    "state": {"feature": {"dtype": "float32", "_type": "Value"},
              "length": 9, "_type": "Sequence"},
    ...
}}}
schema = schema.with_metadata({b"huggingface": json.dumps(hf_meta).encode()})
pq.write_table(pa.Table.from_pydict(rows, schema=schema), path)
```

**另外两个相关坑**：
- `info.json` 里的 `names` 字段不能是 `null`，要写 `["height","width","channel"]`
- `chunks_size` 必须**大于总 episode 数**，否则 episode 1000+ 会去找 `chunk-001`

---

### 坑 4：训练配置漏了 freeze_filter → 全参数微调 → OOM

**现象**：`RESOURCE_EXHAUSTED: Out of memory while trying to allocate 2.25GiB`

**原因**：配置里只写了 `paligemma_variant="gemma_2b_lora"`，
但**没有** `freeze_filter`，openpi 仍然训练全部 3.353 B 参数。
AdamW 优化器状态 26.8 GB + 参数 6.7 GB + 梯度 6.7 GB → 超过 46 GB。

**解法**：

```python
freeze_filter=pi0_config.Pi0Config(
    pi05=True, action_horizon=50, discrete_state_input=False,
    paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora",
).get_freeze_filter(),
```

加上后显存降到 34.7 GB，可训练参数只有 50 M。

---

### 坑 5：评测全部 0%（控制模式不匹配）

**现象**：三套评测都 0/20，包括分布内的 IND。

**原因**：演示数据用的是 `pd_joint_pos`（**绝对关节角**，范围如 -2.9 ~ 2.9 弧度），
而 `gym.make("PickCube-v1")` 默认是归一化增量控制（动作范围 -1 ~ 1）。
模型输出的绝对角度被当作归一化动作，机械臂完全乱动。

**诊断方法**：

```python
# 打印演示动作范围
print(np.round(actions.min(0), 2), "~", np.round(actions.max(0), 2))   # -1.96 ~ 2.62
# 对比两种控制模式的动作空间
gym.make("PickCube-v1")                      # Box(-1, 1, (8,))
gym.make("PickCube-v1", control_mode="pd_joint_pos")   # Box([-2.9,...], [2.9,...])
```

**解法**：

```python
env = gym.make("PickCube-v1", ..., control_mode="pd_joint_pos")
```

---

### 坑 6：动作维度被截断成 7 维

**现象**：`AssertionError: Received action of shape torch.Size([1, 7]) but expected (1, 8)`

**原因**：openpi 的 `LiberoOutputs` 把动作硬编码截成 7 维（LIBERO 的动作维度），
ManiSkill PickCube 是 8 维（7 关节 + 1 夹爪）。

**解法**：加一个可配置的截断类。

```python
# src/openpi/training/config.py
class _SliceActions:
    """把模型输出（padding 到 32 维）截回数据集真实动作维度。"""
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
    def __call__(self, data: dict) -> dict:
        return {"actions": data["actions"][..., : self.action_dim]}

# LeRobotLiberoDataConfig 里加字段并替换 outputs
output_action_dim: int = 7
...
outputs=[_SliceActions(self.output_action_dim)],
```

> 注意：**不能**继承 `transforms.DataTransformFn` 再用 dataclass，
> 因为它是 `Protocol`，会导致 `Cannot overwrite attribute __setattr__`。

---

### 坑 7：两个 Python 环境无法合并

**现象**：`ModuleNotFoundError: No module named 'mani_skill'`（openpi venv）

**原因**：openpi 用 Python 3.11，ManiSkill 环境是 3.10，依赖冲突无法合并。

**解法**：服务端 + 客户端双进程，通过 `openpi_client` 通信（见 `CUSTOM_CHANGES.md` §4）。

---

### 坑 8：磁盘写满两次

**事故 A（数据盘）**：训练中磁盘 100%，训练崩溃。
原因是每个 checkpoint **13 GB**（参数 6.7 + 训练状态 6.7），
而 `--save-interval 2000` 意味着每小时要吃掉 13 GB 临时空间。

**解法**：
- `--save-interval 6000`（只在 6000 和 12000 存）
- 训练前先算：`13 GB × 断点数 + 数据集 + 权重 < 可用空间`
- 加磁盘守护脚本，低于阈值自动删已完结项目

**事故 B（系统盘）**：干跑训练的断点写到了 `/tmp/ckpt_dry`（一个断点 13 GB），
系统盘只有 30 GB，直接写满。

**解法**：断点永远写到数据盘；干跑配置用完立即删除。

---

### 坑 9：π0.5 base 权重下载损坏

**现象**：`OUT_OF_RANGE: Error reading ... Requested byte range [1120149583, 3077149148)
is not valid for value of size 3075764224`

**原因**：从 ModelScope 下载 12.44 GB / 12 个数据块时，
curl 没有校验文件大小，其中 **3 个块被截断**（各少 1.3~1.4 MB）。

**解法**：逐块对比官方尺寸并重下。

```bash
# 用官方 API 拿准确尺寸，与本地对比
curl -s "https://www.modelscope.cn/api/v1/models/hairuoliu/pi05_base/repo/files?Revision=master&Root=params/ocdbt.process_0/d" \
  | python3 -c "import json,sys; [print(f['Name'], f['Size']) for f in json.load(sys.stdin)['Data']['Files']]"
# 对每个不一致的块：8 路并行下载 + 逐段校验大小 + 合并
```

**教训**：大文件分块下载**必须逐块校验字节数**，不能只看"下载完成"。

---

### 坑 10：颜色与相机变量混淆

**现象**：生成随机化训练数据时，第一版脚本按 `idx % 5` 分相机、按 `idx % 2` 分颜色，
导致 **t=0 的相机永远是红方块、t=0.25 永远是绿方块**。

**后果**：颜色与相机共线，无法判断性能变化是哪个因素引起的。

**解法**：颜色改为按块轮换 `COLORS[(idx // len(TS)) % len(COLORS)]`，与相机解耦。
**重新生成了整批数据。**

---

### 坑 11：评测录像用错分辨率

**现象**：录制视频时用 512×512 渲染，基线模型在 0° 下 5/5 全部失败——
但正式评测（256×256）里基线 IND 是 51%。

**原因**：策略对**渲染分辨率**也敏感，512×512 是一种训练时没见过的分布偏移。

**解法**：录制统一用 256×256（与评测协议一致）。

**副产品**：这条意外发现说明该策略的视觉脆弱性不止限于相机位姿。

---

### 坑 12：训练结束了却没人接着评测

**现象**：训练 03:51 就完成了，评测直到 04:12 才开始，白等 21 分钟。

**原因**：训练结束不会自动触发评测，靠人去看日志才发现训练已经停了。

**解法**：改为**看门狗脚本**——每 30 秒检查断点是否就绪，
一旦出现就自动启动服务并跑完评测，不再依赖人守着。

```bash
# 核心逻辑（完整可运行版见 scripts_local/10_watch_and_eval.sh）
for i in $(seq 1 240); do
  [ -f "$CKPT/_CHECKPOINT_METADATA" ] && break
  sleep 30
done
setsid nohup ... serve_policy.py ... &
wait_policy 8000 60 5        # common.sh 里用 python socket 探活，不依赖 ss/netstat
bash 09_eval_full_protocol.sh randcam "$RESULTS_ROOT/randcam"
```

---

