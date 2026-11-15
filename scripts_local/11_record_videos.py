"""录制评测视频：指定相机/颜色/种子跑若干集，把每集存成 mp4。

用法:
  python 11_record_videos.py --tag randcam_cam20 --cam-t 1.0 --color red \
      --seeds 1000,1001 --out media

注意：渲染分辨率必须与评测协议一致（256×256）。
      用 512×512 录制时基线模型会 5/5 全败 —— 分辨率本身也是一种分布偏移。
"""
import argparse
import json
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np
import gymnasium as gym
import torch
import mani_skill
from PIL import Image
from mani_skill.utils import sapien_utils

BASE = np.array([0.45, 0.30, 0.38])
DELTA = np.array([0.13, 0.12, 0.12])
LOOK_AT = [0.02, 0.0, 0.03]
INSTR = "pick up the cube and move it to the goal"
COLORS = {"red": [1.0, 0.0, 0.0, 1.0], "green": [0.1, 0.7, 0.15, 1.0],
          "blue": [0.15, 0.35, 0.85, 1.0], "yellow": [0.85, 0.65, 0.1, 1.0]}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cam-t", type=float, default=0.0, help="0.0=原相机，1.0≈20°")
    ap.add_argument("--cam-eye", default=None, help="直接指定 x,y,z（优先于 --cam-t）")
    ap.add_argument("--color", default="red", choices=list(COLORS))
    ap.add_argument("--seeds", default="1000,1001,1002")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--exec-horizon", type=int, default=10)
    ap.add_argument("--fov", type=float, default=1.05)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from openpi_client import websocket_client_policy
    policy = websocket_client_policy.WebsocketClientPolicy(host="127.0.0.1", port=a.port)

    from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
    PICK_CUBE_CONFIGS["panda"]["cube_color"] = COLORS[a.color]
    if a.cam_eye:
        eye = [float(x) for x in a.cam_eye.split(",")]
    else:
        eye = (BASE + a.cam_t * DELTA).tolist()
    cfg = {"base_camera": {"width": a.size, "height": a.size,
                           "pose": sapien_utils.look_at(eye, LOOK_AT), "fov": a.fov}}

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        env = gym.make("PickCube-v1", obs_mode="rgb", render_mode="rgb_array",
                       sensor_configs=cfg, num_envs=1, control_mode="pd_joint_pos")
        env.reset(seed=seed)
        u = env.unwrapped
        try:
            u.goal_site.show_visual()
            u._hidden_objects = []
        except Exception:
            pass
        frames, success, steps = [], False, 0
        while steps < a.max_steps and not success:
            o = u.get_obs()
            img = o["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
            frames.append(img)
            qpos = u.agent.robot.get_qpos()[0].cpu().numpy().astype(np.float32)
            res = policy.infer({"observation/image": img, "observation/wrist_image": img,
                                "observation/state": qpos, "prompt": INSTR})
            acts = np.asarray(res["actions"])[: a.exec_horizon]
            for k in range(len(acts)):
                env.step(torch.tensor(acts[k], dtype=torch.float32).unsqueeze(0))
                steps += 1
                frames.append(u.get_obs()["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy())
                if bool(u.evaluate().get("success", torch.tensor(False))[0]):
                    success = True
                    break
                if steps >= a.max_steps:
                    break
        env.close()

        name = f"{a.tag}_seed{seed}_{'success' if success else 'fail'}_{steps}steps.mp4"
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="rec_"))
        try:
            for i, fr in enumerate(frames):
                Image.fromarray(fr).save(tmp / f"{i:05d}.png")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
                            "-i", str(tmp / "%05d.png"), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-vf", "scale=384:384",
                            str(out / name)], check=False)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        results.append({"seed": seed, "success": success, "steps": steps, "file": name})
        print(f"  {name} ({len(frames)} 帧)", flush=True)

    (out / f"{a.tag}_summary.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results))


if __name__ == "__main__":
    main()
