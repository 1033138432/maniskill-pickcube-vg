"""把回放生成的 npz 数据集转成 LeRobot v2.1（openpi 可读）。

单相机方案：同一张图同时写入 image 与 wrist_image，
这样可以直接复用 openpi 的 LeRobotLiberoDataConfig。
"""
import argparse, json, pathlib
import io
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from datasets import Features, Value, Image as HFImage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset/<split> 目录")
    ap.add_argument("--root", required=True, help="LeRobot 输出根目录")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--fps", type=int, default=20)
    a = ap.parse_args()

    src = pathlib.Path(a.src); out = pathlib.Path(a.root) / a.repo_id
    (out / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    shards = sorted([d for d in src.iterdir() if d.is_dir()])
    if not shards:            # 扁平结构：npz 直接放在 src 下
        shards = [src]
    tasks, task_to_idx = [], {}
    ep_idx = 0
    total_frames = 0
    ep_meta, ep_stats = [], []

    for shard in shards:
        for f in sorted(shard.glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            imgs = z["images"]; qpos = np.asarray(z["qpos"], np.float32)
            acts = np.asarray(z["actions"], np.float32)
            instr = str(z["instruction"])
            if instr not in task_to_idx:
                task_to_idx[instr] = len(tasks); tasks.append(instr)
            T = len(imgs)
            # 用 pyarrow 直接写，并注入 HF feature 元数据（与 datasets 写出的格式一致）
            # 图像必须是 struct<bytes, path>，否则 datasets 不会把它当 Image 解码（坑 3）
            img_struct = lambda b: {"bytes": bytes(b), "path": None}
            img_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
            schema = pa.schema([
                ("image", img_type), ("wrist_image", img_type),
                ("state", pa.list_(pa.float32(), list_size=qpos.shape[1])),
                ("actions", pa.list_(pa.float32(), list_size=acts.shape[1])),
                ("timestamp", pa.float32()), ("frame_index", pa.int64()),
                ("episode_index", pa.int64()), ("index", pa.int64()), ("task_index", pa.int64()),
            ])
            hf_meta = {"info": {"features": {
                "image": {"_type": "Image"},
                "wrist_image": {"_type": "Image"},
                "state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": int(qpos.shape[1]), "_type": "Sequence"},
                "actions": {"feature": {"dtype": "float32", "_type": "Value"}, "length": int(acts.shape[1]), "_type": "Sequence"},
                "timestamp": {"dtype": "float32", "_type": "Value"},
                "frame_index": {"dtype": "int64", "_type": "Value"},
                "episode_index": {"dtype": "int64", "_type": "Value"},
                "index": {"dtype": "int64", "_type": "Value"},
                "task_index": {"dtype": "int64", "_type": "Value"},
            }}}
            schema = schema.with_metadata({b"huggingface": json.dumps(hf_meta).encode()})
            rows = {
                "image": [img_struct(b) for b in imgs],
                "wrist_image": [img_struct(b) for b in imgs],
                "state": [qpos[i].tolist() for i in range(T)],
                "actions": [acts[i].tolist() for i in range(T)],
                "timestamp": np.float32(np.arange(T) / a.fps),
                "frame_index": np.arange(T, dtype=np.int64),
                "episode_index": np.full(T, ep_idx, np.int64),
                "index": np.arange(total_frames, total_frames + T, dtype=np.int64),
                "task_index": np.full(T, task_to_idx[instr], np.int64),
            }
            pq.write_table(pa.Table.from_pydict(rows, schema=schema),
                           out / f"data/chunk-000/episode_{ep_idx:06d}.parquet")
            ep_meta.append({"episode_index": ep_idx, "tasks": [instr], "length": int(T)})
            ep_stats.append({"episode_index": ep_idx, "stats": {
                "state": {"count": [int(T)], "mean": qpos.mean(0).tolist(), "std": qpos.std(0).tolist(),
                          "min": qpos.min(0).tolist(), "max": qpos.max(0).tolist()},
                "actions": {"count": [int(T)], "mean": acts.mean(0).tolist(), "std": acts.std(0).tolist(),
                            "min": acts.min(0).tolist(), "max": acts.max(0).tolist()},
            }})
            total_frames += T; ep_idx += 1
            if ep_idx % 300 == 0:
                print(f"  {ep_idx} 条 / {total_frames} 帧", flush=True)

    for name, data in (("episodes.jsonl", ep_meta), ("episodes_stats.jsonl", ep_stats), ("tasks.jsonl", [{"task_index": i, "task": t} for i, t in enumerate(tasks)])):
        with open(out / "meta" / name, "w") as fh:
            for rec in data: fh.write(json.dumps(rec) + "\n")

    info = {
        "codebase_version": "v2.1", "robot_type": "panda", "total_episodes": ep_idx,
        "total_frames": total_frames, "total_tasks": len(tasks), "total_videos": 0,
        "total_chunks": 1, "chunks_size": 10000, "fps": a.fps, "splits": {"train": f"0:{ep_idx}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "image": {"dtype": "image", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": [256, 256, 3], "names": ["height", "width", "channel"]},
            "state": {"dtype": "float32", "shape": [qpos.shape[1]], "names": ["state"]},
            "actions": {"dtype": "float32", "shape": [acts.shape[1]], "names": ["actions"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (out / "meta/info.json").write_text(json.dumps(info, indent=1))
    print(f"完成: {a.repo_id}  {ep_idx} 条 / {total_frames} 帧 / {len(tasks)} 任务 → {out}")


if __name__ == "__main__":
    main()
