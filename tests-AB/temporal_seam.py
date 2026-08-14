"""How big is a TEMPORAL chunk boundary, against the only scale that matters: normal motion.

Why this measurement exists
---------------------------
Spatial tiling puts 4 seams on every one of 158 frames. Temporal chunking puts 2 seams on
one frame each, and costs FEWER total rows (205,632 vs 215,072 at 56-frame chunks). The
owner cannot find a seam in `whole` -- one 56-frame chunk of this canvas refined with no
spatial tiling at all -- so the only thing standing between that arm and a 158-frame clip is
what happens where two chunks meet.

The geometry, which is what makes the number clean:

    source frame   ... 100  101  102  103 ...
    chunkA (start 47, 56f)  |----------- ends at 102
    whole  (start 102, 56f)            102 -----------|
                                        ^
                                 refined TWICE, independently

So `chunkA[55]` and `whole[0]` are two independent refinements of the SAME source frame.
Their difference is the chunk-to-chunk disagreement with zero scene change between them --
the temporal analogue of the overlap-band probe, where content cancels exactly.

Three numbers decide it
-----------------------
  d_shared    |chunkA[55] - whole[0]|      the disagreement itself
  d_natural   |whole[k+1] - whole[k]|      what one frame of normal motion already changes
  dc_step     mean(whole[0]) - mean(chunkA[55])  vs the same step within a chunk

d_shared is a JOIN artefact only insofar as it exceeds d_natural. If a chunk boundary moves
the picture less than an ordinary frame advance already does, there is nothing there to see:
the eye has no stationary reference to compare against, unlike a spatial seam, which is a
fixed line the scene slides past for the whole clip.

dc_step is the separate risk, and the one that actually bites video upscalers -- a uniform
level step between chunks reads as "pumping" or "breathing" across the cut even when every
individual frame is fine. It is also the cheap one to fix: a global offset is not diffusion,
so matching it re-diffuses nothing.

    <venv-python> tests-AB/temporal_seam.py
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-H3-seam")
W, H = 2688, 1536
# The sky is where the spatial seam was visible, so it is the fair field to judge the
# temporal boundary on too: smooth, dark, and where Weber makes small steps loudest.
SKY_ROWS = (0, 620)


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


def stats(a, b):
    """(mean abs delta, mean abs delta in sky, signed DC step in sky) between two frames."""
    d = a - b
    sky = d[SKY_ROWS[0]:SKY_ROWS[1]]
    return float(np.abs(d).mean()), float(np.abs(sky).mean()), float(sky.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", default="chunkA", help="arm whose LAST frame is the join")
    parser.add_argument("--b", default="whole", help="arm whose FIRST frame is the join")
    args = parser.parse_args()

    pa, pb = OUTPUT_DIR / f"SEAM_{args.a}.mp4", OUTPUT_DIR / f"SEAM_{args.b}.mp4"
    for p in (pa, pb):
        if not p.is_file():
            raise SystemExit(f"missing {p}")
    fa, fb = frames(pa), frames(pb)
    print(f"\n{args.a}: {len(fa)} frames    {args.b}: {len(fb)} frames")

    # The join: same source frame, refined once in each chunk.
    j_all, j_sky, j_dc = stats(fa[-1], fb[0])
    print("\n  THE JOIN — one source frame refined independently by two chunks")
    print(f"    |{args.a}[-1] - {args.b}[0]|   whole frame {j_all:6.3f}/255   "
          f"sky {j_sky:6.3f}/255   sky DC step {j_dc:+6.3f}/255")

    # The scale to judge it against: what one ordinary frame advance already changes.
    # This window contains a SHOT CUT around source frame 108 (whole[6]), and a cut is a
    # huge frame-to-frame delta. Averaging it in would inflate "normal motion" and flatter
    # the join for free, so normal motion is summarised by the MEDIAN (one cut cannot move
    # it) and separately by the NEAR-JOIN pairs, which are the frames the eye actually has
    # on screen either side of the boundary.
    nat_all, nat_sky, nat_dc = [], [], []
    for k in range(len(fb) - 1):
        a, s, d = stats(fb[k], fb[k + 1])
        nat_all.append(a)
        nat_sky.append(s)
        nat_dc.append(d)
    near = slice(0, 5)                                   # whole[0..5], all before the cut
    print("\n  NORMAL MOTION — consecutive frames inside one chunk (the scale to beat)")
    print(f"    median    whole frame {np.median(nat_all):6.3f}/255   "
          f"sky {np.median(nat_sky):6.3f}/255   "
          f"|sky DC step| {np.median(np.abs(nat_dc)):6.3f}/255")
    print(f"    near join whole frame {np.mean(nat_all[near]):6.3f}/255   "
          f"sky {np.mean(nat_sky[near]):6.3f}/255   "
          f"|sky DC step| {np.mean(np.abs(nat_dc[near])):6.3f}/255")
    print(f"    quietest  whole frame {np.min(nat_all):6.3f}/255   sky {np.min(nat_sky):6.3f}/255")
    print(f"    loudest   whole frame {np.max(nat_all):6.3f}/255   sky {np.max(nat_sky):6.3f}/255"
          f"   <- the shot cut, excluded from the verdict by using the median")

    ratio = j_sky / float(np.median(nat_sky))
    dc_ratio = abs(j_dc) / float(np.median(np.abs(nat_dc)))
    near_ratio = j_sky / float(np.mean(nat_sky[near]))
    print("\n  VERDICT")
    print(f"    join / normal-motion (median), sky: {ratio:5.2f}x")
    print(f"    join / normal-motion (near join):   {near_ratio:5.2f}x")
    print(f"    join DC / normal DC step, sky:      {dc_ratio:5.2f}x")
    if ratio <= 1.0 and dc_ratio <= 2.0:
        print("    => the boundary moves the picture LESS than an ordinary frame advance.")
    elif dc_ratio > 2.0:
        print("    => DC STEP dominates: this is the 'pumping' failure mode. A global "
              "per-chunk\n       offset match fixes it without re-diffusing anything.")
    else:
        print("    => the boundary exceeds normal motion; a keyframe handoff is needed.")

    # There is exactly ONE shared frame, so the join has no second sample. The hardest
    # honest test available is the join against the quietest frame pair in the chunk.
    print(f"\n    hardest comparison — join vs the QUIETEST frame pair in the chunk: "
          f"{j_sky / float(np.min(nat_sky)):5.2f}x")

    (OUTPUT_DIR / "SEAM_temporal_seam.json").write_text(json.dumps({
        "join": {"all": j_all, "sky": j_sky, "sky_dc": j_dc},
        "natural_median": {"all": float(np.median(nat_all)), "sky": float(np.median(nat_sky)),
                           "abs_sky_dc": float(np.median(np.abs(nat_dc)))},
        "natural_near_join_sky": float(np.mean(nat_sky[near])),
        "natural_min_sky": float(np.min(nat_sky)),
        "natural_max_sky": float(np.max(nat_sky)),
        "ratio_sky_median": ratio, "ratio_sky_near": near_ratio, "ratio_dc": dc_ratio,
    }, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
