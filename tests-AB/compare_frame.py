"""Same SOURCE frame from several arms, stacked and labelled, for a like-for-like look.

The auto-saved PNGs each arm writes are at fixed INDICES (0, mid, last), and the arms cover
different source windows, so those stills are never the same moment. This pulls one chosen
source frame out of each arm and stacks them in one image.

Two outputs, because they answer different questions:

  _full   whole frames, half scale, stacked -- composition and any gross difference
  _crop   1:1 pixels from the SAME region of every arm -- the detail comparison, which is
          the only way to judge sharpness honestly. The region is chosen automatically as
          the highest high-frequency-energy window in the first arm, so it lands on real
          texture rather than flat sky.

    <venv-python> tests-AB/compare_frame.py --source 100 --arms chunkA chunk2frz22 chunk2frz39
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-H3-seam")
W, H = 2688, 1536
CROP_W, CROP_H = 1100, 700

# arm -> first source frame it covers
ARM_START = {"whole": 102, "chunkA": 47, "chunkC": 90, "chunkE": 35, "chunk1": 38,
             "chunk2": 93, "chunk2kf": 93, "chunk1_d16": 38, "chunk2kf_d16": 93,
             "chunk2kf_vl": 93, "chunk2kf_ref": 93, "chunk2kf_refp": 93,
             "chunk2kf_refp22": 93, "chunk2frz": 89, "chunk2frzn": 89, "chunk2frznk": 89,
             "chunk2frznc": 89, "chunk2frzng": 89, "chunk2frz22": 72, "chunk2frz39": 55}


def frame_at(arm, source):
    start = ARM_START[arm]
    index = source - start
    if index < 0:
        raise SystemExit(f"{arm} starts at source {start}; cannot supply {source}")
    path = OUTPUT_DIR / f"SEAM_{arm}.mp4"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    n, got = W * H * 3, None
    for i in range(index + 1):
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        got = buf if i == index else None
    proc.stdout.close()
    proc.kill()
    if got is None:
        raise SystemExit(f"{arm} has no frame at index {index} (source {source})")
    return np.frombuffer(got, np.uint8).reshape(H, W, 3)


def busiest_window(img):
    """Top-left of the CROP_W x CROP_H window with the most high-frequency energy."""
    g = img.astype(np.float32).mean(axis=2)
    d = np.abs(np.diff(g, axis=0)[:, :-1]) + np.abs(np.diff(g, axis=1)[:-1, :])
    # integral image -> exact window sums without a python loop over every offset
    ii = np.pad(d, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    ys = np.arange(0, H - CROP_H, 40)
    xs = np.arange(0, W - CROP_W, 40)
    best, best_xy = -1.0, (0, 0)
    for y in ys:
        for x in xs:
            s = (ii[y + CROP_H, x + CROP_W] - ii[y, x + CROP_W]
                 - ii[y + CROP_H, x] + ii[y, x])
            if s > best:
                best, best_xy = float(s), (x, y)
    return best_xy


def stack(images, labels, scale=1.0):
    w = int(images[0].shape[1] * scale)
    h = int(images[0].shape[0] * scale)
    band = 42
    out = Image.new("RGB", (w, (h + band) * len(images)), (16, 16, 16))
    draw = ImageDraw.Draw(out)
    for i, (img, label) in enumerate(zip(images, labels, strict=True)):
        tile = Image.fromarray(img)
        if scale != 1.0:
            tile = tile.resize((w, h), Image.LANCZOS)
        y = i * (h + band)
        draw.text((12, y + 12), label, fill=(255, 255, 255))
        out.paste(tile, (0, y + band))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=int, default=100)
    parser.add_argument("--arms", nargs="+",
                        default=["chunkA", "chunk2frz22", "chunk2frz39"])
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    frames = [frame_at(a, args.source) for a in args.arms]
    labels = [f"{a}   source frame {args.source}   (index {args.source - ARM_START[a]} in its chunk)"
              for a in args.arms]
    labels[0] += "   <- no boundary" if args.arms[0] == "chunkA" else ""

    x, y = busiest_window(frames[0])
    print(f"source frame {args.source}; 1:1 crop at ({x},{y}) {CROP_W}x{CROP_H} "
          "(highest-detail window)")
    tag = args.tag or "_".join(a.replace("chunk2", "") for a in args.arms[1:])

    full = stack(frames, labels, scale=0.5)
    full_path = OUTPUT_DIR / f"COMPARE_f{args.source}_{tag}_full.png"
    full.save(full_path)
    print(f"  wrote {full_path.name}  ({full.size[0]}x{full.size[1]})")

    crops = [f[y:y + CROP_H, x:x + CROP_W] for f in frames]
    crop = stack(crops, [f"{lbl}   1:1 pixels" for lbl in labels], scale=1.0)
    crop_path = OUTPUT_DIR / f"COMPARE_f{args.source}_{tag}_crop.png"
    crop.save(crop_path)
    print(f"  wrote {crop_path.name}  ({crop.size[0]}x{crop.size[1]})")

    # One FULL-RESOLUTION file per arm, unscaled and unlabelled. Stacked sheets are useless
    # for small differences: the eye needs to flip between two images pinned at the same
    # position and size, and any burned-in label draws attention away from the pixels.
    # Identical filename stem so a viewer sorts them adjacently.
    for arm, img in zip(args.arms, frames, strict=True):
        path = OUTPUT_DIR / f"FRAME_f{args.source}__{arm}.png"
        Image.fromarray(img).save(path)
        print(f"  wrote {path.name}  ({W}x{H}, full resolution, no label)")


if __name__ == "__main__":
    main()
