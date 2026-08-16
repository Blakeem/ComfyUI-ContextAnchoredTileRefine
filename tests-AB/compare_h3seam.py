"""Compare run_ab_h3seam.py arms: probe tables, and a 1:1 seam strip per arm.

Two views, both fed by the arms' lossless outputs (never the mp4s — h264 noise sits at
the same 1-2/255 as the effect being measured):

  --table   the recorded band error per arm, side by side. `offset` is the static step
            the feather must hide, `swing` is how much that step moves over the clip.
  --strips  a stack of 1:1 crops centred on the vertical seam column (x=1344), one row
            per arm, from the same frame. Written twice: as-is, and with a strong
            local contrast stretch that makes a sub-1/255 step visible.

    <venv-python> tests-AB/compare_h3seam.py --table
    <venv-python> tests-AB/compare_h3seam.py --strips --frame 55
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-H3-seam")
SEAM_X = 1344
ORDER = ("base", "ov0", "ov64", "an64ov64", "ov128", "an128", "whole")


def load_probes():
    found = {}
    for label in ORDER:
        path = OUTPUT_DIR / f"SEAM_{label}_probe.json"
        if path.is_file():
            found[label] = json.loads(path.read_text(encoding="utf-8"))
    return found


def table(probes):
    print(f"{'arm':10s} {'anch':>4s} {'ovl':>4s} {'tile/side':11s} {'band':>4s} "
          f"{'|offset|':>8s} {'|swing|':>8s} {'peak':>6s} {'rms':>6s} {'dc removed':>11s}")
    for label, data in probes.items():
        s = data["settings"]
        for row in data["rows"]:
            dc = row["dc_removed_255"]
            dc_s = "-" if dc is None else "/".join(f"{v * 255:+.2f}" for v in dc)
            print(f"{label:10s} {s['context_anchor']:4d} {s['context_overlap']:4d} "
                  f"{row['tile'] + ' ' + row['side']:11s} {row['band_px']:4d} "
                  f"{max(abs(v) for v in row['offset_255']):8.2f} "
                  f"{max(row['swing_255']):8.2f} {max(row['peak_255']):6.2f} "
                  f"{row['rms_255']:6.2f} {dc_s:>11s}")
    print(f"\n{'arm':10s} {'r':>4s} {'WORST offset':>13s} {'WORST swing':>12s} {'WORST rms':>10s}")
    for label, data in probes.items():
        rows = data["rows"]
        if not rows:
            print(f"{label:10s} {'-':>4s} {'(no seam: single tile)':>38s}")
            continue
        s = data["settings"]
        print(f"{label:10s} {s['context_anchor'] + s['context_overlap']:4d} "
              f"{max(max(abs(v) for v in r['offset_255']) for r in rows):13.2f} "
              f"{max(max(r['swing_255']) for r in rows):12.2f} "
              f"{max(r['rms_255'] for r in rows):10.2f}")


def local_stretch(crop, sigma_px=64):
    """Remove a heavy horizontal low-pass and rescale, per channel.

    The seam is a step ALONG x; the scene gradient across a 400px sky crop is smooth on
    a much longer scale. Subtracting a wide horizontal box average therefore keeps the
    step and drops the gradient, and the fixed gain makes the result comparable between
    arms (never a per-crop min/max, which would rescale each arm differently)."""
    f = crop.astype(np.float64)
    pad = np.pad(f, ((0, 0), (sigma_px, sigma_px), (0, 0)), mode="edge")
    kernel = np.ones(2 * sigma_px + 1) / (2 * sigma_px + 1)
    low = np.stack([np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 1, pad[:, :, c])
                    for c in range(3)], axis=2)
    return np.clip((f - low) * 16.0 + 128.0, 0, 255).astype(np.uint8)


def strips(probes, frame, half, rows):
    panels_raw, panels_hp, labels = [], [], []
    for label in probes:
        path = OUTPUT_DIR / f"SEAM_{label}_f{frame:03d}.png"
        if not path.is_file():
            print(f"  (no {path.name})")
            continue
        img = np.array(Image.open(path).convert("RGB"))
        crop = img[rows[0]:rows[1], SEAM_X - half:SEAM_X + half, :]
        panels_raw.append(crop)
        panels_hp.append(local_stretch(crop))
        labels.append(label)
    if not panels_raw:
        raise SystemExit("no arm PNGs found for that frame")
    sep = np.full((6, 2 * half, 3), 255, np.uint8)
    for name, panels in (("raw", panels_raw), ("stretch", panels_hp)):
        stack = []
        for p in panels:
            marked = p.copy()
            marked[:3, half - 1:half + 1] = (255, 0, 255)
            marked[-3:, half - 1:half + 1] = (255, 0, 255)
            stack += [marked, sep]
        out = OUTPUT_DIR / f"SEAM_COMPARE_{name}_f{frame:03d}.png"
        Image.fromarray(np.concatenate(stack[:-1], axis=0)).save(out)
        print(f"  wrote {out}  rows top-to-bottom: {', '.join(labels)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--strips", action="store_true")
    parser.add_argument("--frame", type=int, default=55)
    parser.add_argument("--half", type=int, default=200, help="crop half-width in px")
    parser.add_argument("--rows", type=int, nargs=2, default=(200, 560))
    args = parser.parse_args()

    probes = load_probes()
    if not probes:
        raise SystemExit(f"no probe JSONs in {OUTPUT_DIR}")
    print(f"arms found: {', '.join(probes)}\n")
    if args.table or not args.strips:
        table(probes)
    if args.strips:
        strips(probes, args.frame, args.half, tuple(args.rows))


if __name__ == "__main__":
    main()
