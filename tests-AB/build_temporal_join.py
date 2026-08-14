"""Assemble the temporally-chunked clip so the join can be judged by eye, naive vs DC-matched.

What this builds
----------------
`chunkA` covers source frames 47..102 and `whole` covers 102..157, both refined whole-canvas
with no spatial tiling. Delivering a chunked refine means cutting at the shared frame and
concatenating:

    delivered = chunkA[0..54]  ++  whole[0..55]        source 47..157, 111 frames
                            ^^^^  ^^
                             the join, between source 101 and 102

Two versions are written so the fix can be judged against the thing it fixes:

  naive    straight concatenation. Whatever the two chunks disagree about lands here.
  dcmatch  `whole` shifted by ONE constant per-channel offset, measured as the mean
           difference between the two arms' refinements of the SHARED source frame
           (chunkA[55] vs whole[0]). Constant, not per-frame: a per-frame correction
           would track content and inject its own drift, while a constant offset is a
           single global level match -- arithmetic, not diffusion, so it re-diffuses no
           finished pixels and cannot compound grit (prime directive 2).

Also written: a stress loop that repeats the join several times so a one-frame event gets
enough screen time to register, and a x16 local-contrast stretch of the shared frame's
disagreement, which is the amplification that made the 1/255 spatial seam visible.

    <venv-python> tests-AB/build_temporal_join.py
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-H3-seam")
W, H = 2688, 1536
FPS = 24


def frames(path):
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    n = W * H * 3
    out = []
    while True:
        buf = proc.stdout.read(n)
        if len(buf) < n:
            break
        out.append(np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32))
    proc.stdout.close()
    proc.wait()
    return out


def write_mp4(path, seq, crf=10):
    import av
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.options = {"crf": str(crf), "preset": "slow"}
    for f in seq:
        arr = np.clip(f, 0, 255).astype(np.uint8)
        container.mux(stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")))
    container.mux(stream.encode())
    container.close()
    print(f"  wrote {path.name}  ({len(seq)} frames, crf {crf})")


def local_stretch(img, gain=16.0, sigma_px=64):
    """Remove a wide box average, then amplify. The spatial-seam work used exactly this."""
    k = np.ones(sigma_px) / sigma_px
    pad = np.pad(img.mean(axis=2), ((0, 0), (sigma_px // 2, sigma_px // 2)), mode="edge")
    base = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, pad)
    base = base[:, sigma_px // 2:sigma_px // 2 + img.shape[1]]
    return np.clip(128.0 + (img.mean(axis=2) - base) * gain, 0, 255)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", default="chunkA")
    parser.add_argument("--b", default="whole")
    parser.add_argument("--loops", type=int, default=6, help="join repeats in the stress clip")
    args = parser.parse_args()

    pa, pb = OUTPUT_DIR / f"SEAM_{args.a}.mp4", OUTPUT_DIR / f"SEAM_{args.b}.mp4"
    for p in (pa, pb):
        if not p.is_file():
            raise SystemExit(f"missing {p}")
    fa, fb = frames(pa), frames(pb)
    print(f"{args.a}: {len(fa)} frames    {args.b}: {len(fb)} frames")

    # The shared source frame, refined once by each chunk. This is the only place the two
    # arms can be compared with zero scene change, so it is where the offset is measured.
    shared_a, shared_b = fa[-1], fb[0]
    offset = (shared_a - shared_b).mean(axis=(0, 1))
    print(f"  shared-frame per-channel offset (A - B): "
          f"{offset[0]:+.3f}/{offset[1]:+.3f}/{offset[2]:+.3f} /255")

    naive = fa[:-1] + fb
    matched = fa[:-1] + [f + offset for f in fb]
    write_mp4(OUTPUT_DIR / "JOIN_naive.mp4", naive)
    write_mp4(OUTPUT_DIR / "JOIN_dcmatch.mp4", matched)

    # A one-frame event is hard to see once. Hold 8 frames either side and repeat, so the
    # eye gets the transition several times at playback speed.
    stress = []
    for _ in range(args.loops):
        stress += fa[-9:-1] + fb[:8]
    write_mp4(OUTPUT_DIR / "JOIN_stress_naive.mp4", stress)
    stress_m = []
    for _ in range(args.loops):
        stress_m += fa[-9:-1] + [f + offset for f in fb[:8]]
    write_mp4(OUTPUT_DIR / "JOIN_stress_dcmatch.mp4", stress_m)

    # The disagreement itself, amplified the way the spatial seam had to be to be seen.
    from PIL import Image
    diff = np.abs(shared_a - shared_b).mean(axis=2)
    print(f"  shared-frame |A-B|: mean {diff.mean():.3f}/255  max {diff.max():.3f}/255")
    Image.fromarray(local_stretch(shared_a).astype(np.uint8)).save(
        OUTPUT_DIR / "JOIN_shared_A_stretch.png")
    Image.fromarray(local_stretch(shared_b).astype(np.uint8)).save(
        OUTPUT_DIR / "JOIN_shared_B_stretch.png")
    Image.fromarray(np.clip(128 + (shared_a - shared_b).mean(axis=2) * 16, 0, 255).astype(np.uint8)).save(
        OUTPUT_DIR / "JOIN_shared_diff_x16.png")
    print("  wrote JOIN_shared_A_stretch.png / _B_stretch.png / _diff_x16.png")
    print("\n  JOIN_shared_diff_x16.png is flat grey if the two chunks agree. Structure in it "
          "is\n  what the chunks disagree about, at 16x. Watch JOIN_stress_*.mp4 for the "
          "moving judgement.")


if __name__ == "__main__":
    main()
