"""把评测录像拼成「并排 A/B 对照」GIF，供 README 内嵌展示。

GitHub 的 Markdown 不会内联播放 mp4，所以把关键对照做成 GIF 才能直接看到动作。

生成两个文件（输出到 media/）：
  comparison_baseline_vs_randcam_cam20.gif   同种子 1000、同相机 20°：基线失败 vs 随机化成功
  comparison_randcam_cam20_vs_reverse.gif    同一套权重：训练覆盖方向成功 vs 正交方向失败

用法:
  python 12_make_comparison_gifs.py
  python 12_make_comparison_gifs.py --panel 240 --fps 10 --colors 96

额外依赖（不在主 requirements 里，只在做 GIF 时需要）:
  pip install imageio imageio-ffmpeg
  # imageio-ffmpeg 会自带一个 ffmpeg 可执行文件，无需系统安装

原理：PIL 合成画面（缩放 + 标签栏 + 播完压暗），再交给 ffmpeg 的
      palettegen/paletteuse 编码成 GIF —— 比 PIL 直接存 GIF 小 3~4 倍。
"""
import argparse
import os
import pathlib
import shutil
import subprocess
import tempfile

import numpy as np

try:
    import imageio
    import imageio_ffmpeg
except ImportError as e:  # pragma: no cover
    raise SystemExit(f"缺少依赖：{e}\n请先 pip install imageio imageio-ffmpeg") from e

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MEDIA = REPO_ROOT / "media"

# Pillow 9.1+ 把重采样枚举挪到了 Image.Resampling，做个兼容
RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

GAP = 5          # 两个面板之间的间距
BAR = 48         # 顶部标签栏高度
BG = (18, 18, 20)
BAR_BG = (32, 32, 37)
FG = (238, 238, 240)
DIM = (150, 150, 158)
FAIL = (228, 78, 78)
OK = (74, 202, 124)


def get_font(size):
    """优先用 Arial Bold；找不到就退回 PIL 内置字体。"""
    for p in (r"C:\Windows\Fonts\arialbd.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def load_frames(path):
    """读整段视频为 numpy 帧列表 + fps。"""
    reader = imageio.get_reader(str(path), "ffmpeg")
    fps = float(reader.get_meta_data().get("fps", 20.0))
    frames = [np.asarray(f) for f in reader]
    reader.close()
    return frames, fps


def compose_pair(a, b, out_path, panel, fps_out, colors):
    """a/b = (frames, fps, 第一行文字, 第二行文字, 强调色)。"""
    width, height = panel * 2 + GAP, panel + BAR
    f1, f2 = get_font(max(11, panel // 20)), get_font(max(9, panel // 25))
    dur_a, dur_b = len(a[0]) / a[1], len(b[0]) / b[1]
    n = int(round(max(dur_a, dur_b) * fps_out))

    tmp = pathlib.Path(tempfile.mkdtemp(prefix="gif_"))
    try:
        for i in range(n):
            t = i / fps_out
            canvas = Image.new("RGB", (width, height), BG)
            d = ImageDraw.Draw(canvas)
            for k, (frames, fps_, line1, line2, col) in enumerate((a, b)):
                x = k * (panel + GAP)
                # 标签栏：左侧强调色条 + 名称 + 结果
                d.rectangle([x, 0, x + panel - 1, BAR - 1], fill=BAR_BG)
                d.rectangle([x, 0, x + 4, BAR - 1], fill=col)
                d.text((x + 14, 5), line1, font=f1, fill=FG)
                d.text((x + 14, BAR - f2.size - 7), line2, font=f2, fill=col)
                # 视频帧
                idx = min(len(frames) - 1, int(t * fps_))
                canvas.paste(Image.fromarray(frames[idx]).resize((panel, panel), RESAMPLE),
                             (x, BAR))
                # 片段播完后压暗并标注，便于看出「这边早就结束了」
                if t > len(frames) / fps_:
                    y0 = BAR + panel - 22
                    canvas.paste(canvas.crop((x, y0, x + panel, y0 + 22))
                                 .point(lambda v: int(v * 0.30)), (x, y0))
                    d.text((x + 10, y0 + 4), f"clip ended @ {len(frames)/fps_:.2f}s", font=f2, fill=DIM)
            canvas.save(tmp / f"f{i:04d}.png")

        vf = (f"split[s0][s1];[s0]palettegen=max_colors={colors}:stats_mode=diff[p];"
              f"[s1][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle")
        r = subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
             "-framerate", str(fps_out), "-i", str(tmp / "f%04d.png"),
             "-vf", vf, "-loop", "0", str(out_path)],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"ffmpeg 失败：\n{r.stderr[-600:]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # 注意：这里只用 ASCII，避免 Windows 的 GBK 控制台打印 emoji 时抛 UnicodeEncodeError
    print(f"  [OK] {out_path.name}: {width}x{height} {n} frames @{fps_out}fps "
          f"-> {out_path.stat().st_size / 1048576:.2f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", type=int, default=300, help="单侧面板边长（像素）")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--colors", type=int, default=128, help="GIF 调色板颜色数")
    ap.add_argument("--media", type=pathlib.Path, default=MEDIA)
    a = ap.parse_args()

    def read(name):
        p = a.media / name
        if not p.exists():
            raise SystemExit(f"缺少录像：{p}")
        return load_frames(p)

    print("读取录像…")
    base, base_fps = read("baseline_cam20_seed1000_fail_200steps.mp4")
    rand, rand_fps = read("randcam_cam20_seed1000_success_62steps.mp4")
    rev, rev_fps = read("randcam_reverse_seed1000_fail_200steps.mp4")
    print(f"  基线 {len(base)} 帧 / 随机化 {len(rand)} 帧 / 反向 {len(rev)} 帧")

    print("合成 GIF…")
    compose_pair(
        (base, base_fps, "BASELINE  |  fixed camera  |  20 deg",
         "FAILED  |  200 steps, never succeeded", FAIL),
        (rand, rand_fps, "RANDOMIZED  |  5 cameras  |  20 deg",
         "SUCCESS  |  62 steps", OK),
        a.media / "comparison_baseline_vs_randcam_cam20.gif", a.panel, a.fps, a.colors)

    compose_pair(
        (rand, rand_fps, "RANDOMIZED  |  20 deg  (covered axis)",
         "SUCCESS  |  62 steps", OK),
        (rev, rev_fps, "RANDOMIZED  |  reverse  (unseen axis)",
         "FAILED  |  200 steps, never succeeded", FAIL),
        a.media / "comparison_randcam_cam20_vs_reverse.gif", a.panel, a.fps, a.colors)


if __name__ == "__main__":
    main()
