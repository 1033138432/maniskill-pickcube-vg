"""评测 PickCube 单格成功率（IND / OOD-颜色 / OOD-相机 / 任意相机位姿）。

架构说明：本脚本是 **WebSocket 客户端**，不加载模型权重。
          策略由 07b_serve_policy.sh 起在 127.0.0.1:8000 的服务提供。
          因此「评测哪个模型」由服务端决定，不要在这里传 --ckpt。

用法:
  python 08_eval_single_split.py --split ind --episodes 100 --out results/ind.json
  python 08_eval_single_split.py --split ind --episodes 50 --cam-eye 0.58,0.42,0.50 \
      --out results/cam_20deg.json

环境变量:
  MS_ASSET_DIR   ManiSkill 仿真资产目录（不设则用 mani_skill 默认缓存）
"""
import argparse
import json
import pathlib

import numpy as np
import gymnasium as gym
import torch
import mani_skill
from mani_skill.utils import sapien_utils

CAM_LOOK_AT = [0.02, 0.0, 0.03]
CAM_IND = ([0.45, 0.30, 0.38], CAM_LOOK_AT)
CAM_OOD = ([0.58, 0.42, 0.50], CAM_LOOK_AT)
COLORS = {
    "ind": [[1.0, 0.0, 0.0, 1.0], [0.1, 0.7, 0.15, 1.0]],           # 红 / 绿
    "ood_color": [[0.15, 0.35, 0.85, 1.0], [0.85, 0.65, 0.1, 1.0]],  # 蓝 / 黄
}
INSTR = "pick up the cube and move it to the goal"


def make_env(color, cam, size=256, fov=1.05):
    from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
    if color is not None:
        PICK_CUBE_CONFIGS["panda"]["cube_color"] = color
    cfg = {"base_camera": {"width": size, "height": size,
                           "pose": sapien_utils.look_at(cam[0], cam[1]), "fov": fov}}
    env = gym.make("PickCube-v1", obs_mode="rgb", render_mode="rgb_array",
                   sensor_configs=cfg, num_envs=1, control_mode="pd_joint_pos")
    env.reset(seed=0)
    try:
        env.unwrapped.goal_site.show_visual()
        env.unwrapped._hidden_objects = []
    except Exception:
        pass
    return env


def obs_to_inputs(env):
    o = env.unwrapped.get_obs()
    img = o["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
    qpos = env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy().astype(np.float32)
    return img, qpos


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=list(COLORS) + ["ood_cam"],
                    help="ind / ood_color / ood_cam；配合 --cam-eye 时只用其颜色规则")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--exec-horizon", type=int, default=10,
                    help="每次推理后执行多少步（action_horizon=50，这里只取前 N 步）")
    ap.add_argument("--max-steps", type=int, default=200, help="单集步数上限")
    ap.add_argument("--seed0", type=int, default=1000, help="首集种子；两个模型用同一套种子")
    ap.add_argument("--cam-eye", default=None, help="x,y,z 覆盖相机位置，用于视角扫描")
    ap.add_argument("--fov", type=float, default=1.05)
    ap.add_argument("--size", type=int, default=256, help="渲染分辨率，必须与训练一致")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from openpi_client import websocket_client_policy
    policy = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=a.port)

    if a.cam_eye:
        eye = [float(x) for x in a.cam_eye.split(",")]
        cam = (eye, CAM_LOOK_AT)
    else:
        cam = CAM_OOD if a.split == "ood_cam" else CAM_IND
    colors = COLORS.get(a.split, COLORS["ind"])

    results = []
    for ep in range(a.episodes):
        color = colors[ep % len(colors)]
        env = make_env(color, cam, a.size, a.fov)
        env.reset(seed=a.seed0 + ep)
        try:
            env.unwrapped.goal_site.show_visual()
            env.unwrapped._hidden_objects = []
        except Exception:
            pass
        success = False
        steps = 0
        while steps < a.max_steps and not success:
            img, qpos = obs_to_inputs(env)
            out = policy.infer({
                "observation/image": img,
                "observation/wrist_image": img,   # 单相机方案：同一张图喂两个槽
                "observation/state": qpos.astype(np.float32),
                "prompt": INSTR,
            })
            acts = np.asarray(out["actions"])[: a.exec_horizon]
            for k in range(len(acts)):
                env.step(torch.tensor(acts[k], dtype=torch.float32).unsqueeze(0))
                steps += 1
                if bool(env.unwrapped.evaluate().get("success", torch.tensor(False))[0]):
                    success = True
                    break
                if steps >= a.max_steps:
                    break
        results.append({"episode": ep, "seed": a.seed0 + ep, "success": success,
                        "steps": steps, "color": color})
        env.close()
        print(f"  ep{ep:02d} color={color[:3]} success={success} steps={steps}", flush=True)

    n = len(results)
    s = sum(r["success"] for r in results)
    summary = {"split": a.split, "episodes": n, "success": s, "rate": s / n if n else 0,
               "cam_eye": cam[0], "fov": a.fov, "size": a.size}
    print(json.dumps(summary))
    if a.out:
        pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(a.out).write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))


if __name__ == "__main__":
    main()
