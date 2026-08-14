"""Does a frame render differently because of WHERE it sits in its chunk? 44 shared frames.

The question this settles
------------------------
Two chunks disagreed by 10.571/255 on their one shared frame, while changing only the SEED
(same window, same context) moves the picture 3.9/255 and its DC by 0.2/255 -- against a
3.0/255 DC step at the chunk join. So the disagreement is not noise realisation. The
remaining suspect is POSITION IN CHUNK: H3 packs the first 5 image frames into 2 latent
frames and every 17 after into 5, and a frame at a chunk edge has context on one side only.

`chunkC` covers source 90..145 and `whole` covers 102..157, so source 102..145 is rendered
by BOTH -- at chunk-local positions that differ by 12:

    source     102  ...  123  ...  145
    chunkC pos  12  ...   33  ...   55     <- 55 is chunkC's LAST frame (edge)
    whole  pos   0  ...   21  ...   43     <- 0 is whole's FIRST frame (edge)

Read the U
----------
  U-SHAPED (high at both ends, low in the middle) => the edge effect is real, and it is
      bounded. Then chunking is solved WITHOUT blending or conditioning: overlap the chunks
      and DISCARD the edge frames, so every delivered frame is a mid-chunk frame. Nothing is
      re-diffused, nothing is averaged, and the seam does not exist rather than being hidden.
  FLAT near 10/255 => position is not the cause either, two chunks simply do not agree, and
      temporal chunking needs conditioning (keyframe handoff) or it is not viable.
  FLAT near 3.9/255 => the single shared frame in the earlier test was a fluke of the jump,
      and naive chunking is already fine away from a shot cut.

The shot cut at source 106 is marked; frames spanning it are not evidence about position.

    <venv-python> tests-AB/chunk_position.py
"""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-H3-seam")
W, H = 2688, 1536
SKY_ROWS = (0, 620)
CUT_FRAME = 106                      # measured: source 105->106 jumps 48/255


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


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", default="chunkC")
    parser.add_argument("--a-start", type=int, default=90)
    parser.add_argument("--b", default="whole")
    parser.add_argument("--b-start", type=int, default=102)
    parser.add_argument("--noise-floor", type=float, default=3.939,
                        help="measured base vs base_s1234: seed-only disagreement")
    args = parser.parse_args()

    pa, pb = OUTPUT_DIR / f"SEAM_{args.a}.mp4", OUTPUT_DIR / f"SEAM_{args.b}.mp4"
    for p in (pa, pb):
        if not p.is_file():
            raise SystemExit(f"missing {p}")
    fa, fb = frames(pa), frames(pb)
    lo = max(args.a_start, args.b_start)
    hi = min(args.a_start + len(fa), args.b_start + len(fb))
    shared = list(range(lo, hi))
    print(f"\n{args.a} covers source {args.a_start}..{args.a_start + len(fa) - 1}   "
          f"{args.b} covers source {args.b_start}..{args.b_start + len(fb) - 1}")
    print(f"shared: {len(shared)} frames, source {shared[0]}..{shared[-1]}")
    print(f"seed-only floor (same context, different noise): {args.noise_floor:.3f}/255\n")

    print(f"  {'src':>4s} {'posA':>5s} {'posB':>5s} {'|diff|':>7s} {'sky':>7s} "
          f"{'DC R/G/B':>20s}   edge-distance")
    rows = []
    for k in shared:
        ia, ib = k - args.a_start, k - args.b_start
        d = fa[ia] - fb[ib]
        mean_all = float(np.abs(d).mean())
        mean_sky = float(np.abs(d[SKY_ROWS[0]:SKY_ROWS[1]]).mean())
        dc = d.mean(axis=(0, 1))
        # How far this frame sits from the NEAREST chunk edge, in either chunk. The edge
        # hypothesis predicts disagreement falls as this grows.
        edge = min(ia, len(fa) - 1 - ia, ib, len(fb) - 1 - ib)
        mark = "  <- SHOT CUT" if k in (CUT_FRAME, CUT_FRAME - 1) else ""
        rows.append({"src": k, "posA": ia, "posB": ib, "edge": edge,
                     "all": mean_all, "sky": mean_sky, "dc": dc.tolist()})
        if k < shared[0] + 6 or k > shared[-1] - 6 or k % 4 == 0:
            print(f"  {k:4d} {ia:5d} {ib:5d} {mean_all:7.3f} {mean_sky:7.3f} "
                  f"{dc[0]:+6.2f}/{dc[1]:+6.2f}/{dc[2]:+6.2f}   {edge:3d}{mark}")

    clean = [r for r in rows if abs(r["src"] - CUT_FRAME) > 2]
    print(f"\n  (excluding {len(rows) - len(clean)} frames within 2 of the shot cut)")
    print("\n  DISAGREEMENT vs DISTANCE FROM THE NEAREST CHUNK EDGE")
    print(f"  {'edge dist':>10s} {'n':>4s} {'|diff|':>8s} {'sky':>8s} {'|DC| max ch':>12s}")
    for lo_e, hi_e in ((0, 1), (2, 3), (4, 7), (8, 15), (16, 99)):
        sel = [r for r in clean if lo_e <= r["edge"] <= hi_e]
        if not sel:
            continue
        print(f"  {f'{lo_e}-{hi_e}':>10s} {len(sel):4d} "
              f"{np.mean([r['all'] for r in sel]):8.3f} "
              f"{np.mean([r['sky'] for r in sel]):8.3f} "
              f"{np.mean([max(abs(c) for c in r['dc']) for r in sel]):12.3f}")

    deep = [r for r in clean if r["edge"] >= 8]
    edgy = [r for r in clean if r["edge"] <= 1]
    print("\n  VERDICT")
    if deep and edgy:
        dm, em = np.mean([r["all"] for r in deep]), np.mean([r["all"] for r in edgy])
        print(f"    mid-chunk (edge>=8): {dm:6.3f}/255      chunk-edge (edge<=1): {em:6.3f}/255")
        print(f"    seed-only floor:     {args.noise_floor:6.3f}/255")
        if dm <= args.noise_floor * 1.35:
            print("    => U CONFIRMED and mid-chunk frames are at the noise floor. Overlap the\n"
                  "       chunks and DISCARD edge frames: every delivered frame is mid-chunk,\n"
                  "       nothing is blended, nothing is re-diffused, and no seam exists.")
        elif dm < em * 0.7:
            print("    => U present but mid-chunk sits ABOVE the noise floor: discarding edges\n"
                  "       reduces the artefact without eliminating it. Needs a keyframe handoff.")
        else:
            print("    => NO U: position is not the driver. Two chunks simply disagree, so\n"
                  "       temporal chunking needs conditioning to be viable.")

    (OUTPUT_DIR / "SEAM_chunk_position.json").write_text(
        json.dumps({"rows": rows, "noise_floor": args.noise_floor}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
