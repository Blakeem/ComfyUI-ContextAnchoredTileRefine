"""Is the seam a GRAIN NOTCH rather than a level offset? Per-column grain amplitude, all frames.

Hypothesis
----------
The probe says the two tiles agree on LEVEL to within 0.2-1.5/255 after DC match, yet the
owner still sees a line, and only when the camera moves. A level offset that small in a dark
sky is near threshold; a *stationary* strip of REDUCED GRAIN in a field of shimmering grain
is not, because motion segmentation locks onto anything that fails to move with the scene.

The composite makes such a strip by construction. Across the cross-dissolve band the two
tiles hold INDEPENDENT realisations of the same grain statistics. A weighted average at
blend weight a turns an independent component of std s into s*sqrt(a^2+(1-a)^2), which
bottoms out at 0.707*s at a=0.5. So the band should read ~30% low in grain amplitude while
reading correct in mean level -- invisible to a DC probe, invisible in a still against
detailed content, and a soft vertical line against a smooth dark sky where the grain IS the
content.

Measurement
-----------
    luma    = mean over RGB                          (what the eye segments on)
    hp      = luma - gaussian(luma, GRAIN_SIGMA)     isolate the grain
    e(x)    = sqrt(mean over rows AND FRAMES of hp^2)

e(x) still carries the scene: a cloud edge raises it far more than a seam lowers it. So it
is only ever read as a RATIO against an arm with the same content and no seam anywhere --
the `whole` arm, one tile over the whole canvas, which the owner confirms is clean to his
eye. Scene cancels, and what survives at the seam columns is the seam.

Two orthogonal seams are measured from the same frames. The vertical seam at x=1344 and the
horizontal seam at y=768 are produced by the same feather code on different axes, so a notch
that appears at BOTH is a property of the composite, while one that appears at only one is
more likely a coincidence of scene layout.

    <venv-python> tests-AB/grain_notch.py --arms base ov0 ov128 an128
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-H3-seam")
W, H = 2688, 1536
SEAM_X, SEAM_Y = 1344, 768
GRAIN_SIGMA = 1.5
# Sky only. The grain notch argument is a Weber argument: it needs a smooth dark field where
# grain is the entire signal. Rows below the skyline carry buildings, whose edges swamp e(x).
SKY_ROWS = (0, 620)
# Columns for the horizontal-seam profile, chosen to sit in the same sky band and to avoid
# the vertical seam so the two measurements stay independent.
SKY_COLS = (1600, 2600)


def gaussian_1d(sigma):
    radius = max(1, int(sigma * 3))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()).astype(np.float32)


def blur(img, k):
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


def frames(path):
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    n = W * H * 3
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        yield np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32).mean(axis=2)
    proc.stdout.close()
    proc.wait()


def profiles(path):
    """(e(x) over SKY_ROWS, e(y) over SKY_COLS, frame count) -- one pass, both axes."""
    k = gaussian_1d(GRAIN_SIGMA)
    acc_x = np.zeros(W, dtype=np.float64)
    acc_y = np.zeros(H, dtype=np.float64)
    count = 0
    for luma in frames(path):
        hp = luma - blur(luma, k)
        sq = hp * hp
        acc_x += sq[SKY_ROWS[0]:SKY_ROWS[1]].mean(axis=0)
        acc_y += sq[:, SKY_COLS[0]:SKY_COLS[1]].mean(axis=1)
        count += 1
    if not count:
        raise SystemExit(f"no frames decoded from {path}")
    return np.sqrt(acc_x / count), np.sqrt(acc_y / count), count


def report(name, ratio, seam, band, controls):
    """Mean ratio in the feather band, at the seam column, and in matched control windows."""
    in_band = float(ratio[seam - band:seam].mean())
    at_seam = float(ratio[seam - 2:seam + 2].mean())
    ctrl = {c: float(ratio[c - band:c].mean()) for c in controls}
    spread = float(np.std(list(ctrl.values())))
    print(f"    {name:18s} band[{seam - band},{seam}) {in_band:6.3f}   "
          f"seam+-2 {at_seam:6.3f}   controls " +
          " ".join(f"{v:.3f}" for v in ctrl.values()) +
          f"   ctrl sd {spread:.3f}")
    return {"in_band": in_band, "at_seam": at_seam, "controls": ctrl, "control_sd": spread}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", nargs="+", default=["base"])
    parser.add_argument("--vs", default="whole", help="seamless reference arm")
    parser.add_argument("--band", type=int, default=32)
    args = parser.parse_args()

    ref_path = OUTPUT_DIR / f"SEAM_{args.vs}.mp4"
    if not ref_path.is_file():
        raise SystemExit(f"reference arm missing: {ref_path}")
    ref_x, ref_y, n_ref = profiles(ref_path)
    print(f"\nreference {args.vs}: {n_ref} frames, grain sigma {GRAIN_SIGMA}")
    print(f"sky rows {SKY_ROWS} for the x-profile, sky cols {SKY_COLS} for the y-profile")
    print(f"absolute grain amplitude at x=1024 in the reference: {ref_x[1024]:.3f}/255\n")

    xs = list(range(SEAM_X - 64, SEAM_X + 65, 8))
    out = {}
    for arm in args.arms:
        path = OUTPUT_DIR / f"SEAM_{arm}.mp4"
        if not path.is_file():
            print(f"  {arm}: missing {path.name}")
            continue
        arm_x, arm_y, n = profiles(path)
        if n != n_ref:
            print(f"  {arm}: {n} frames vs reference {n_ref} -- skipped")
            continue
        rx, ry = arm_x / ref_x, arm_y / ref_y
        print(f"  {arm}  (grain amplitude relative to {args.vs}; 1.000 = identical)")
        print("    x     " + " ".join(f"{x:6d}" for x in xs))
        print("    ratio " + " ".join(f"{rx[x]:6.3f}" for x in xs))
        out[arm] = {
            "vertical": report("VERTICAL x=1344", rx, SEAM_X, args.band, (896, 1024, 1664, 1792)),
            "horizontal": report("HORIZONTAL y=768", ry, SEAM_Y, args.band, (448, 576, 960, 1088)),
            "profile_x": rx[SEAM_X - 96:SEAM_X + 96].tolist(),
        }
        print()

    (OUTPUT_DIR / "SEAM_grain_notch.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("A notch is real when the band ratio sits below the controls by more than the "
          "control sd,\non BOTH axes. ov0 is the positive control for a HARD seam: an "
          "unblended step raises\nthe high-pass, so it should read ABOVE 1, not below.")


if __name__ == "__main__":
    main()
