"""把 ManiSkill 官方演示回放成 VLA 数据集（图像 + 状态 + 动作 + 语言指令）。

原理：官方演示（trajectory.h5）只存了每步完整 env_states 和 actions，没有图像。
      恢复 env_states 后重新渲染即可得到任意外观/相机视角下的 RGB，
      而动作序列因为是「物理状态 + 动作」不会被外观改变，所以标签仍然有效。
      这是用回放做视觉数据增广的前提。

支持按「外观/相机」生成 IND 与 OOD 变体。

用法:
  python 02_build_dataset.py --split ind --n 600
  python 02_build_dataset.py --split ood_color --n 600
  python 02_build_dataset.py --split ood_cam --n 600

环境变量（都可被命令行参数覆盖）:
  DEMO_ROOT   演示解压目录，默认 <仓库>/demos/extracted
  DATA_ROOT   数据集输出目录，默认 <仓库>/dataset

⚠️ FOV 注意：本脚本默认 --fov 1.1，与 03/08/11 用的 1.05 不同。
   基线数据历史上就是用 1.1 渲染的，为保证可复现性保留该默认值。
   若要严格单变量对比，请把所有脚本统一到同一个 --fov。
"""
import argparse
import io
import json
import os
import pathlib

import numpy as np
import torch
import gymnasium as gym
import h5py
import mani_skill
from PIL import Image
from mani_skill.utils import sapien_utils

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEMO_ROOT = pathlib.Path(os.environ.get("DEMO_ROOT", REPO_ROOT / "demos" / "extracted"))
DATA_ROOT = pathlib.Path(os.environ.get("DATA_ROOT", REPO_ROOT / "dataset"))

# 任务 → (h5 相对路径, 语言指令)
TASKS = {
    "PickCube-v1": ("PickCube-v1/motionplanning/trajectory.h5",
                   "pick up the cube and move it to the goal"),
}

# IND：见过的颜色 + 标准相机；OOD：未见颜色 / 偏移相机
SPLITS = {
    "ind": {
        # 见过的颜色：红、绿
        "colors": {"PickCube-v1": [[1.0, 0.0, 0.0, 1.0], [0.1, 0.7, 0.15, 1.0]]},
        "cam": ([0.45, 0.30, 0.38], [0.02, 0.0, 0.03]),
    },
    "ood_color": {
        # 未见颜色：蓝、黄
        "colors": {"PickCube-v1": [[0.15, 0.35, 0.85, 1.0], [0.85, 0.65, 0.1, 1.0]]},
        "cam": ([0.45, 0.30, 0.38], [0.02, 0.0, 0.03]),
    },
    "ood_cam": {
        # 同 IND 颜色，但相机位姿偏移 ≈20°
        "colors": {"PickCube-v1": [[1.0, 0.0, 0.0, 1.0], [0.1, 0.7, 0.15, 1.0]]},
        "cam": ([0.58, 0.42, 0.50], [0.02, 0.0, 0.03]),
    },
}


def encode(img):
    """RGB → JPEG 字节（质量 95）。"""
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def make_env(task, color, cam, size=256, fov=1.1):
    # 颜色通过改配置字典注入（需要 maniskill_patch/pick_cube_color.patch 已应用）
    if color is not None and task in ("PickCube-v1", "PushCube-v1"):
        from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
        PICK_CUBE_CONFIGS["panda"]["cube_color"] = color
        try:
            from mani_skill.envs.tasks.tabletop.push_cube_cfgs import PUSH_CUBE_CONFIGS
            PUSH_CUBE_CONFIGS["panda"]["cube_color"] = color
        except Exception:
            pass
    pose = sapien_utils.look_at(cam[0], cam[1])
    cfg = {"base_camera": {"width": size, "height": size, "pose": pose, "fov": fov}}
    env = gym.make(task, obs_mode="rgb", render_mode="rgb_array", sensor_configs=cfg,
                   num_envs=1, control_mode="pd_joint_pos")
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=list(SPLITS))
    ap.add_argument("--n", type=int, default=600, help="每个颜色取多少条轨迹")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--fov", type=float, default=1.1, help="相机视场角，见文件头注意事项")
    ap.add_argument("--demo-root", type=pathlib.Path, default=DEMO_ROOT)
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help=f"输出目录，默认 {DATA_ROOT}/<split>")
    a = ap.parse_args()

    split = SPLITS[a.split]
    out_root = a.out or (DATA_ROOT / a.split)
    meta = []
    for task, (rel, instr) in TASKS.items():
        h5p = a.demo_root / rel
        if not h5p.exists():
            print(f"跳过 {task}: 缺少 {h5p}（先跑 01_download_demos.sh）")
            continue
        colors = split["colors"][task] or [None]
        with h5py.File(h5p, "r") as f:
            keys = sorted(f.keys(), key=lambda x: int(x.split("_")[1]))[: a.n]
            for ci, color in enumerate(colors):
                env = make_env(task, color, split["cam"], a.size, a.fov)
                env.reset(seed=0)
                # 目标球默认在 _hidden_objects 里（RL 训练不需要看见），评测时必须显示
                try:
                    env.unwrapped.goal_site.show_visual()
                    env.unwrapped._hidden_objects = []
                except Exception as e:
                    print("goal 显示失败:", e)
                tag = f"{task}_{a.split}_c{ci}"
                outdir = out_root / tag
                outdir.mkdir(parents=True, exist_ok=True)
                for k in keys:
                    t = f[k]
                    T = t["actions"].shape[0]
                    arts = list(t["env_states/articulations"].keys())
                    state = {
                        "actors": {n: torch.tensor(np.asarray(t["env_states/actors"][n]))
                                   for n in t["env_states/actors"]},
                        "articulations": {n: torch.tensor(np.asarray(t["env_states/articulations"][n]))
                                          for n in arts},
                    }
                    imgs, qpos, succ = [], [], []
                    for s in range(T):
                        env.unwrapped.set_state_dict(
                            {kk: {n: v[s:s + 1] for n, v in vv.items()} for kk, vv in state.items()})
                        rgb = env.unwrapped.get_obs()["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
                        imgs.append(encode(rgb))
                        qpos.append(env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy().astype(np.float32))
                        succ.append(bool(np.asarray(t["success"])[s]))
                    np.savez(outdir / f"{k}.npz",
                             images=np.array(imgs, dtype=object),
                             qpos=np.stack(qpos),
                             actions=np.asarray(t["actions"], np.float32),
                             success=np.array(succ),
                             instruction=instr)
                    meta.append({"task": task, "shard": tag, "episode": k, "frames": int(T),
                                 "instruction": instr, "color": color})
                env.close()
                print(f"✅ {tag}: {len(keys)} 条 (fov={a.fov}, size={a.size})", flush=True)
    (out_root / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"总计 {len(meta)} 条 → {out_root}")


if __name__ == "__main__":
    main()
