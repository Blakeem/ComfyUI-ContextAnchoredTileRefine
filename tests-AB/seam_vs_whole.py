"""Measure each tiled arm's seam step against the SINGLE-TILE arm, over the whole clip.

The `whole` arm refines the same clip, same seed, same settings, as ONE tile — so it has
no seam anywhere. Differencing a tiled arm against it cancels the scene and leaves only
what tiling did. That difference is not zero away from the seam (each tile refines from
its own crop, with its own conditioning slice, so whole cores shift a little), which is
why the statistic is the STEP ACROSS THE SEAM and not the difference itself:

    d(x)   = row-mean of (arm - whole) for one frame
    step   = mean d over [seam, seam+FIT)  -  mean d over [seam-FIT, seam)

reported as its mean and std over frames, against the same estimator at control columns
where no seam exists. Control columns set the noise floor: the arms differ everywhere,
so only a step that beats the controls is a seam.

    <venv-python> tests-AB/seam_vs_whole.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-H3-seam")
W, H = 2688, 1536
SEAM = 1344
CONTROLS = (896, 1024, 1664, 1792)
FIT = 96
# Columns skipped on the LEFT of the seam. The later tile's feather band occupies
# [seam-overlap, seam), so a left window that reached the seam would average the
# cross-dissolve into the "pure neighbour" side and understate the core-to-core step.
# 128 clears the widest overlap any arm uses. The right window starts at the seam
# because the seam column is already pure this-tile content (alpha = 1 there).
GAP = 128


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


def step(d, col):
    return float(d[col:col + FIT].mean() - d[col - GAP - FIT:col - GAP].mean())


def run(arm, reference="whole"):
    ref_path = OUTPUT_DIR / f"SEAM_{reference}.mp4"
    arm_path = OUTPUT_DIR / f"SEAM_{arm}.mp4"
    if not arm_path.is_file():
        return None
    steps = {SEAM: [], **{c: [] for c in CONTROLS}}
    # strict: both arms refine the same window, so a length mismatch means one mp4 is
    # truncated and every number below would be computed against the wrong frames.
    for fa, fw in zip(frames(arm_path), frames(ref_path), strict=True):
        d = (fa - fw).mean(axis=2).mean(axis=0)
        for col in steps:
            steps[col].append(step(d, col))
    seam = np.array(steps[SEAM])
    ctrl = np.concatenate([np.array(steps[c]) for c in CONTROLS])
    # The SIGNED mean is the discriminating statistic, not the magnitude. Both arms and
    # the reference refine every core slightly differently, so |step| is well above zero
    # at any column; but only a real seam puts a step of a CONSISTENT SIGN there. A
    # control column's step is content, so it averages toward zero over the clip.
    # t is the signed mean in units of its own standard error — how many sigma the
    # systematic part stands off zero.
    t = seam.mean() / max(1e-6, seam.std() / np.sqrt(seam.size))
    print(f"  {arm:9s} seam {seam.mean():+6.2f} +- {seam.std():4.2f} ({t:+5.1f} sigma)   "
          f"controls {ctrl.mean():+6.2f} +- {ctrl.std():4.2f}   "
          f"|seam| {np.abs(seam).mean():4.2f} vs |ctrl| {np.abs(ctrl).mean():4.2f}")
    return seam.mean(), ctrl.mean()


if __name__ == "__main__":
    print(f"Step across x={SEAM} relative to the single-tile arm, all frames, units /255.")
    print("ratio > 1 means the seam column carries a step the rest of the canvas does not.")
    for arm in ("ov0", "base", "ov64", "an64ov64", "ov128", "an128"):
        run(arm)
    sys.stdout.flush()
