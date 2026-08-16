"""Does output frame k actually render source frame start+k? Checks for drift, offset, or rate change.

Why this exists
---------------
The owner's report is not a seam description: *"all the ships in the background speed up as
well"*, and it happens identically with no conditioning, with a keyframe, with a 5-frame
reference and with a 22-frame reference. A conditioning lever that changed nothing across
that range is evidence the problem is not conditioning.

The alternative is that the refine does not preserve the 1:1 frame mapping we assume. H3
packs 17 image frames into 5 latent frames with a 5-frame head into 2, so the temporal
mapping is not uniform, and nothing so far has verified that decoded frame k corresponds to
input frame k. If a chunk's output runs even slightly ahead of its input, then:

  - joining two chunks would jump, because chunk 2 resumes at the wrong point in time
  - the scene would appear to SPEED UP across the boundary
  - no amount of position/motion conditioning would fix it, because the conditioning is
    correct and the indexing is not

Method
------
Each refined frame is box-downscaled back to source resolution and compared against a window
of SOURCE frames; the best match is the source frame it actually rendered. Reported as
`best - expected`:

    all zeros            -> mapping is 1:1, the speed-up is a rendering effect, not indexing
    constant non-zero    -> fixed temporal OFFSET; the fix is an index shift, not conditioning
    growing with k       -> RATE error; the chunk renders faster/slower than its input
                            (this is what "everything speeds up" looks like)

A refine at denoise 0.22 changes pixels, so the match is never exact -- what matters is which
source frame is CLOSEST, and whether the argmin sits where it should.

    <venv-python> tests-AB/temporal_alignment.py
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-H3-seam")
SOURCE_MP4 = Path(r"C:\Users\Blake\Documents\ComfyUI\output\video\MiniMax_H3_00020_.mp4")
SRC_W, SRC_H = 1344, 768
W, H = 2688, 1536
# arm -> first SOURCE frame the arm was given
ARMS = {"chunk1": 38, "chunk2": 93, "chunk2kf": 93, "chunk2kf_ref": 93,
        "chunk2kf_refp": 93, "chunk2kf_refp22": 93}


def source_frames():
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(SOURCE_MP4), "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, SRC_H, SRC_W, 3).astype(np.float32)


def refined_frames(path, count):
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    n = W * H * 3
    out = []
    while len(out) < count:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        f = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32)
        # box-downscale 2x back to source resolution so the comparison is like-for-like
        out.append(f.reshape(SRC_H, 2, SRC_W, 2, 3).mean(axis=(1, 3)))
    proc.stdout.close()
    proc.kill()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--frames", type=int, default=12, help="output frames to check")
    parser.add_argument("--window", type=int, default=6, help="+/- source frames to search")
    args = parser.parse_args()

    src = source_frames()
    print(f"source: {len(src)} frames\n")
    for arm in args.arms:
        path = OUTPUT_DIR / f"SEAM_{arm}.mp4"
        if not path.is_file():
            print(f"  {arm}: missing {path.name}")
            continue
        start = ARMS[arm]
        refined = refined_frames(path, args.frames)
        deltas, rows = [], []
        for k, img in enumerate(refined):
            expected = start + k
            lo = max(0, expected - args.window)
            hi = min(len(src), expected + args.window + 1)
            errs = [(j, float(np.abs(img - src[j]).mean())) for j in range(lo, hi)]
            best, best_err = min(errs, key=lambda t: t[1])
            exp_err = dict(errs).get(expected, float("nan"))
            deltas.append(best - expected)
            rows.append((k, expected, best, best - expected, best_err, exp_err))
        print(f"  {arm}  (given source {start}..)")
        print(f"    {'out':>4s} {'expected':>9s} {'best':>5s} {'delta':>6s} "
              f"{'err@best':>9s} {'err@expected':>13s}")
        for k, expected, best, delta, be, ee in rows:
            mark = "" if delta == 0 else "   <-- MISMATCH"
            print(f"    {k:4d} {expected:9d} {best:5d} {delta:+6d} {be:9.3f} {ee:13.3f}{mark}")
        d = np.array(deltas)
        if np.all(d == 0):
            verdict = "1:1 mapping intact"
        elif len(set(deltas)) == 1:
            verdict = f"CONSTANT OFFSET of {deltas[0]:+d} frames -- fix the index, not the conditioning"
        else:
            slope = np.polyfit(np.arange(len(d)), d, 1)[0]
            verdict = (f"DRIFT: delta grows {slope:+.2f} frames per output frame -- "
                       f"the chunk renders at ~{1 + slope:.2f}x the input rate")
        print(f"    => {verdict}\n")


if __name__ == "__main__":
    main()
