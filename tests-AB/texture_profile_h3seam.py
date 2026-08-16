"""Local grain amplitude across a tile seam, per column, from the arms' lossless PNGs.

Hypothesis this tests
---------------------
The probe says the two tiles agree on LEVEL (residual DC <= 0.5/255) but disagree on
TEXTURE (band RMS 3-5/255). Where they disagree, the directional feather cross-dissolves
them, and a weighted average of two INDEPENDENT textures has lower amplitude than either
one: at blend weight a, an independent component of std s becomes s*sqrt(a^2+(1-a)^2),
which bottoms out at 0.707*s where a = 0.5. So the feather band should read as a strip
of REDUCED GRAIN — invisible against detailed content, but against a smooth dark sky the
grain IS the content, and a ~30% dip in it over ~10-15 px is a soft vertical line.

Measurement
-----------
    hp      = img - gaussian(img, GRAIN_SIGMA)      isolate the grain
    e(x)    = sqrt(mean over rows of hp^2)          grain amplitude per column

e(x) still carries the scene (a cloud edge raises it), so it is only read as a RATIO
against a reference that has the same scene and no seam:

  --vs whole   e_arm(x) / e_whole(x). The `whole` arm is one tile over the whole canvas:
               same content, same settings, no seam anywhere. A dip localised at the
               band is then unambiguous.
  (no --vs)    e_arm(x) alone, with control columns for scale.

    <venv-python> tests-AB/texture_profile_h3seam.py --arms base ov0 --vs whole
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-H3-seam")
SEAM_X = 1344
GRAIN_SIGMA = 1.5


def gaussian_1d(sigma):
    radius = max(1, int(sigma * 3))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def blur(img, sigma):
    k = gaussian_1d(sigma)
    r = (k.size - 1) // 2
    out = img
    for axis in (0, 1):
        pad = [(0, 0)] * img.ndim
        pad[axis] = (r, r)
        p = np.pad(out, pad, mode="edge")
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            sl = [slice(None)] * img.ndim
            sl[axis] = slice(i, i + out.shape[axis])
            acc += w * p[tuple(sl)]
        out = acc
    return out


def grain_profile(path, rows):
    img = np.array(Image.open(path).convert("RGB")).astype(np.float64)[rows[0]:rows[1]]
    hp = img - blur(img, GRAIN_SIGMA)
    return np.sqrt((hp ** 2).mean(axis=(0, 2)))          # [W] grain amplitude per column


def band_dip(profile, seam, band, wide=96):
    """How far the profile inside the feather band sits below its two flanks, in %."""
    inside = profile[seam - band:seam].mean()
    flank = np.concatenate([profile[seam - band - wide:seam - band],
                            profile[seam:seam + wide]]).mean()
    return 100.0 * (inside / flank - 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", nargs="+", default=["base"])
    parser.add_argument("--vs", default=None, help="reference arm, e.g. whole")
    parser.add_argument("--frame", type=int, default=55)
    parser.add_argument("--rows", type=int, nargs=2, default=(120, 620))
    parser.add_argument("--band", type=int, default=32)
    args = parser.parse_args()

    rows = tuple(args.rows)
    ref = None
    if args.vs:
        ref_path = OUTPUT_DIR / f"SEAM_{args.vs}_f{args.frame:03d}.png"
        if not ref_path.is_file():
            raise SystemExit(f"reference arm missing: {ref_path}")
        ref = grain_profile(ref_path, rows)

    print(f"frame {args.frame}, rows {rows}, grain sigma {GRAIN_SIGMA}, "
          f"reference = {args.vs or 'none'}")
    xs = list(range(SEAM_X - 96, SEAM_X + 97, 16))
    print("    x      " + " ".join(f"{x:6d}" for x in xs))
    for arm in args.arms:
        path = OUTPUT_DIR / f"SEAM_{arm}_f{args.frame:03d}.png"
        if not path.is_file():
            print(f"  {arm}: missing {path.name}")
            continue
        prof = grain_profile(path, rows)
        shown = prof / ref if ref is not None else prof
        print(f"  {arm:9s} " + " ".join(f"{shown[x]:6.3f}" for x in xs))
        for band in (args.band, 64, 128):
            print(f"      dip over the last {band:3d}px before the seam: "
                  f"{band_dip(prof, SEAM_X, band):+6.2f}%    "
                  f"control x=1024 {band_dip(prof, 1024, band):+6.2f}%    "
                  f"x=1664 {band_dip(prof, 1664, band):+6.2f}%")


if __name__ == "__main__":
    main()
