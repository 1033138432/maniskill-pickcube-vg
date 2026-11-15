"""把 ManiSkill 官方演示（只有状态+动作）回放成「图像+动作+状态」数据集。

用法:
    python replay_render.py --h5 <trajectory.h5> --out <dir> --n 10 --size 256
"""
import argparse, io, json, pathlib, time
import numpy as np, torch, gymnasium as gym, h5py, mani_skill
from PIL import Image

CAM_POSE = [0.5, 0.0, 0.55, 0.0, -0.4, 0.0, 0.9165]   # 稍远一点，能看到整个桌面
CAM_FOV = 1.15


def make_env(size: int, seed: int = 0):
    cfg = {"base_camera": {"width": size, "height": size, "pose": CAM_POSE, "fov": CAM_FOV}}
    env = gym.make("PickCube-v1", obs_mode="rgb", render_mode="rgb_array",
                   sensor_configs=cfg, num_envs=1)
    env.reset(seed=seed)
    return env


def encode(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--start", type=int, default=0)
    a = ap.parse_args()

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = h5py.File(a.h5, "r")
    keys = sorted(f.keys(), key=lambda x: int(x.split("_")[1]))[a.start:a.start + a.n]
    env = make_env(a.size)

    t0 = time.time(); frames = 0
    for ti, k in enumerate(keys):
        t = f[k]
        T = t["actions"].shape[0]
        state = {
            "actors": {n: torch.tensor(np.asarray(t["env_states/actors"][n])) for n in t["env_states/actors"]},
            "articulations": {n: torch.tensor(np.asarray(t["env_states/articulations"][n])) for n in t["env_states/articulations"]},
        }
        imgs, qpos, succ = [], [], []
        for s in range(T):
            env.unwrapped.set_state_dict({kk: {n: v[s:s+1] for n, v in vv.items()} for kk, vv in state.items()})
            obs = env.unwrapped.get_obs()
            rgb = obs["sensor_data"]["base_camera"]["rgb"][0].cpu().numpy()
            imgs.append(encode(rgb))
            qpos.append(env.unwrapped.agent.robot.get_qpos()[0].cpu().numpy().astype(np.float32))
            succ.append(bool(np.asarray(t["success"])[s]))
            frames += 1
        np.savez_compressed(
            out / f"{k}.npz",
            images_rgb=np.array(imgs, dtype=object),
            qpos=np.stack(qpos),
            actions=np.asarray(t["actions"], dtype=np.float32),
            success=np.array(succ),
        )
        if ti == 0 or (ti + 1) % 5 == 0:
            dt = time.time() - t0
            print(f"  {ti+1}/{len(keys)} 条, {frames} 帧, {dt:.1f}s, {frames/dt:.1f} 帧/秒", flush=True)
    env.close()
    dt = time.time() - t0
    print(f"完成: {len(keys)} 条 / {frames} 帧 / {dt:.1f}s → {frames/dt:.1f} 帧/秒")
    print(f"推算 1000 条 × 74 帧 = {1000*74/(frames/dt)/3600:.1f} 小时")


if __name__ == "__main__":
    main()
