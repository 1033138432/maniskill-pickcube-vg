"""生成「相机随机化」训练集：每条演示分配一个相机位姿。

相机轴：eye = base + t * delta，t ∈ {0, 0.25, 0.5, 0.75, 1.0}
        base = [0.45, 0.30, 0.38]，delta = [0.13, 0.12, 0.12]（t=1 即原 OOD-相机 ≈20°）
颜色：红/绿（与 IND 相同），按块轮换，与相机解耦（见 docs/02_pitfalls_and_failures.md 坑 10）

用法:
  python 03_build_dataset_randcam.py --n 1000

环境变量:
  DEMO_ROOT   演示解压目录，默认 <仓库>/demos/extracted
  DATA_ROOT   数据集输出目录，默认 <仓库>/dataset
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

DEMO_H5 = "PickCube-v1/motionplanning/trajectory.h5"
INSTR = "pick up the cube and move it to the goal"
BASE = np.array([0.45, 0.30, 0.38])
DELTA = np.array([0.13, 0.12, 0.12])
TS = [0.0, 0.25, 0.5, 0.75, 1.0]
COLORS = [[1.0, 0.0, 0.0, 1.0], [0.1, 0.7, 0.15, 1.0]]


def encode(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000, help="拿多少条演示（官方只有 1000 条）")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--fov", type=float, default=1.05)
    ap.add_argument("--demo-root", type=pathlib.Path, default=DEMO_ROOT)
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help=f"输出目录，默认 {DATA_ROOT}/ind_randcam")
    a = ap.parse_args()

    h5p = a.demo_root / DEMO_H5
    if not h5p.exists():
        raise SystemExit(f"缺少演示文件 {h5p}（先跑 01_download_demos.sh）")

    out = a.out or (DATA_ROOT / "ind_randcam")
    out.mkdir(parents=True, exist_ok=True)
    meta = []
    env = None
    cur_t = None
    with h5py.File(h5p, "r") as f:
        keys = sorted(f.keys(), key=lambda x: int(x.split("_")[1]))[: a.n]
        for idx, k in enumerate(keys):
            t = TS[idx % len(TS)]                          # 轮流分配相机
            color = COLORS[(idx // len(TS)) % len(COLORS)]  # 颜色按块轮换，与相机解耦
            if env is None or t != cur_t:
                if env is not None:
                    env.close()
                from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
                PICK_CUBE_CONFIGS["panda"]["cube_color"] = color
                eye = (BASE + t * DELTA).tolist()
                cfg = {"base_camera": {"width": a.size, "height": a.size,
                                       "pose": sapien_utils.look_at(eye, [0.02, 0.0, 0.03]),
                                       "fov": a.fov}}
                env = gym.make("PickCube-v1", obs_mode="rgb", render_mode="rgb_array",
                               sensor_configs=cfg, num_envs=1, control_mode="pd_joint_pos")
                env.reset(seed=0)
                try:
                    env.unwrapped.goal_site.show_visual()
                    env.unwrapped._hidden_objects = []
                except Exception:
                    pass
                cur_t = t
                print(f"  相机 t={t} (eye={[round(x, 2) for x in eye]})", flush=True)

            tt = f[k]
            T = tt["actions"].shape[0]
            arts = list(tt["env_states/articulations"].keys())
            state = {
                "actors": {n: torch.tensor(np.asarray(tt["env_states/actors"][n]))
                           for n in tt["env_states/actors"]},
                "articulations": {n: torch.tensor(np.asarray(tt["env_states/articulations"][n]))
                                  for n in arts},
            }
            imgs, qpos, succ = [], [], []
            for s in range(T):
                env.unwrapped.set_state_dict(
                    {kk: {n: v[s:s + 1] for n, v in vv.items()} for kk, vv in state.items()})
                rgb = env.unwrapped.get_obs()["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
                imgs.append(encode(rgb))
                qpos.append(env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy().astype(np.float32))
                succ.append(bool(np.asarray(tt["success"])[s]))
            np.savez(out / f"{k}.npz",
                     images=np.array(imgs, dtype=object),
                     qpos=np.stack(qpos),
                     actions=np.asarray(tt["actions"], np.float32),
                     success=np.array(succ),
                     instruction=INSTR)
            meta.append({"episode": k, "cam_t": t, "frames": int(T)})
            if (idx + 1) % 200 == 0:
                print(f"  {idx + 1}/{len(keys)} 条", flush=True)
    env.close()
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"完成 {len(meta)} 条 → {out}")


if __name__ == "__main__":
    main()
