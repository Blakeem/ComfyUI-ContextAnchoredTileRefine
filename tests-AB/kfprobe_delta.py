"""Did the keyframe cond block do anything? kfprobe vs base, same seed, same content.

The `kfprobe` arm is `base` plus one shipped whole-crop keyframe cond block per tile
(see `_KeyframeProbe` in run_ab_h3seam.py). Everything else — noise, conditioning slices,
denoise mask, composite — is identical, so `kfprobe - base` is exactly the block's effect.

Three signatures separate "the cond block steered the tile" from "something drifted":

  per TILE      tile (0,0) has no already-processed neighbour, so its keyframe (the LIVE
                canvas) equals its own latent and it should barely move. Tiles (1,0),
                (0,1), (1,1) get a keyframe that disagrees with their latent across the
                cross-dissolve band, and should move more. A uniform delta on all four is
                a global perturbation, not a seam lock.
  over TIME     the block is pinned to frame 0's t coordinate (`resolved_frame_index: 0`,
                comfy/ldm/minimax/model.py:308-309). A real effect is strongest at frame 0
                and decays. A flat-in-time delta is not the keyframe acting positionally.
  over SPACE    within a tile, the keyframe differs from the tile's own latent only where
                an earlier tile pasted — i.e. the band. Delta concentrated near the seam
                columns is the wanted behaviour; delta spread over the whole tile is the
                block rewriting content it should have agreed with.

A near-zero delta everywhere is the decisive NEGATIVE: at denoise 0.22 a clean,
target-registered cond block does not move this pipeline, and every packed-cond-row shape
in the determination dies with it.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-H3-seam")
W, H = 2688, 1536
SEAM_X, SEAM_Y = 1344, 768


def frames(path):
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    n = W * H * 3
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32)
    proc.stdout.close()
    proc.wait()


def main():
    a = OUTPUT_DIR / f"SEAM_{sys.argv[1] if len(sys.argv) > 1 else 'kfprobe'}.mp4"
    b = OUTPUT_DIR / f"SEAM_{sys.argv[2] if len(sys.argv) > 2 else 'base'}.mp4"
    if not (a.is_file() and b.is_file()):
        raise SystemExit(f"need both {a.name} and {b.name}")
    quad = {"T(0,0) no neighbour": (0, SEAM_Y, 0, SEAM_X),
            "T(1,0) kept_left": (0, SEAM_Y, SEAM_X, W),
            "T(0,1) kept_top": (SEAM_Y, H, 0, SEAM_X),
            "T(1,1) kept both": (SEAM_Y, H, SEAM_X, W)}
    per_frame, per_quad = [], {k: [] for k in quad}
    band, away = [], []
    for fa, fb in zip(frames(a), frames(b), strict=True):
        d = np.abs(fa - fb).mean(axis=2)
        per_frame.append(float(d.mean()))
        for k, (y0, y1, x0, x1) in quad.items():
            per_quad[k].append(float(d[y0:y1, x0:x1].mean()))
        band.append(float(d[:, SEAM_X - 32:SEAM_X].mean()))
        away.append(float(d[:, SEAM_X - 400:SEAM_X - 368].mean()))
    per_frame = np.array(per_frame)
    print(f"\n{a.name} vs {b.name}: |delta| over {per_frame.size} frames\n")
    print(f"  overall mean {per_frame.mean():.3f}/255   max {per_frame.max():.3f}/255")
    if per_frame.mean() < 0.05:
        print("  => NEGATIVE: the cond block changed essentially nothing.")
    print("\n  per tile quadrant (tile (0,0) is the control — its keyframe agrees):")
    for k, v in per_quad.items():
        print(f"    {k:22s} {np.mean(v):.3f}/255")
    print("\n  over time (the block is pinned to frame 0's t):")
    idx = [0, 1, 2, 4, 8, 16, 32, per_frame.size - 1]
    print("    frame " + " ".join(f"{i:6d}" for i in idx if i < per_frame.size))
    print("    delta " + " ".join(f"{per_frame[i]:6.3f}" for i in idx if i < per_frame.size))
    print(f"\n  at the seam band [{SEAM_X - 32},{SEAM_X}): {np.mean(band):.3f}/255"
          f"   vs a same-width strip 400px away: {np.mean(away):.3f}/255")


if __name__ == "__main__":
    main()
