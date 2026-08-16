"""The blind A/B: identical source frames, one clip with a chunk boundary in it, one without.

Everything measured says a temporal chunk boundary in steady content is a ~3.8/255 one-frame
texture reseat with a ~0.15/255 DC step -- below the disagreement one chunk has with itself
under a different seed (3.939/255). Metrics bottom out around 2/255 on this material, so what
is left is not measurable, it is visual, and the project rule is that the owner's eye decides.

Construction, from arms that already exist. HEAD comes from one chunk, TAIL from another; the
control takes BOTH from a single chunk that spans the whole segment, so the two clips cover
the same source frames and differ only in whether a boundary is crossed:

    --preset true-edge   (default, the realistic worst case)
      source        77 .......... 89 | 90 .......... 102
      CONTROL       chunkA[30..42]   | chunkA[43..55]      one chunk, NO boundary
      JOIN          chunkE[42..54]   | chunkC[0..12]       boundary at 89 -> 90
      chunkE ENDS at source 90 and chunkC BEGINS there, so the join hands an edge-1 frame
      to an edge-0 frame -- exactly what a real pipeline delivers.

    --preset stand-in    (the earlier, weaker version)
      source       132 ......... 144 | 145 ......... 157
      CONTROL      whole[30..42]     | whole[43..55]
      JOIN         chunkC[42..54]    | whole[43..55]
      Weaker because `whole[43]` sits mid-chunk (edge 12) where a real next-chunk frame sits
      at edge 0; measured, that gap is worth about 12.5%, so this preset UNDERSTATES the cost.

What the two clips necessarily share and do not share: after the boundary, JOIN continues in
the new chunk's rendering while CONTROL continues in the old one. That is not a confound to
be removed -- it is what chunking delivers. The event to watch for is the transition itself.

    <venv-python> tests-AB/build_quiet_join.py [--preset true-edge|stand-in]
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-H3-seam")
W, H = 2688, 1536
FPS = 24

# arm -> first source frame it covers (mirrors ARMS[...]["start"] in run_ab_h3seam.py)
ARM_START = {"whole": 102, "chunkA": 47, "chunkC": 90, "chunkE": 35,
             "chunk1": 38, "chunk2": 93, "chunk2kf": 93,
             "chunk1_d16": 38, "chunk2kf_d16": 93, "chunk2kf_vl": 93,
             "chunk2kf_ref": 93, "chunk2kf_refp": 93, "chunk2kf_refp22": 93,
             "chunk2frz": 89, "chunk2frzn": 89, "chunk2frznk": 89, "chunk2frznc": 89, "chunk2frzng": 89, "chunk2frz22": 72, "chunk2frz39": 55, "chunk2frz22r": 72}

# The source has scene cuts at frames 37 and 106. Any window containing one lets the model
# hide a texture reseat behind content that was already discontinuous, so the handoff
# presets below are placed to keep every JUDGED frame cut-free.
PRESETS = {
    "true-edge": {"prefix": "QUIET", "suffix": "", "control": "chunkA",
                  "head": "chunkE", "tail": "chunkC", "seg0": 77, "cut": 90, "seg1": 102},
    "stand-in": {"prefix": "QUIET", "suffix": "_standin", "control": "whole",
                 "head": "chunkC", "tail": "whole", "seg0": 132, "cut": 145, "seg1": 157},
    # THE HANDOFF TEST. Boundary at source 93. head=chunk1 supplies 73..93 (its own last
    # frame), tail supplies 94..102 -- i.e. the tail chunk's frame 0 is DISCARDED, because
    # chunk1 already rendered that source frame and it is the frame chunk2kf was given as
    # its keyframe. `_naive` and `_kf` differ ONLY in whether that keyframe was supplied.
    "handoff-naive": {"prefix": "HANDOFF", "suffix": "_naive", "control": "chunkA",
                      "head": "chunk1", "tail": "chunk2", "seg0": 73, "cut": 94,
                      "seg1": 102},
    "handoff-kf": {"prefix": "HANDOFF", "suffix": "_kf", "control": "chunkA",
                   "head": "chunk1", "tail": "chunk2kf", "seg0": 73, "cut": 94,
                   "seg1": 102},
    # TEST 1: denoise 0.16 on BOTH chunks. The control stays chunkA at 0.22, so this clip
    # differs from handoff-kf in refine strength AND from the control in it too -- read it
    # against handoff-kf, not against the control, for the denoise question.
    "handoff-d16": {"prefix": "HANDOFF", "suffix": "_d16", "control": "chunkA",
                    "head": "chunk1_d16", "tail": "chunk2kf_d16", "seg0": 73,
                    "cut": 94, "seg1": 102},
    # TEST 2: VL reference extended backwards with chunk1's refined frames. Same denoise as
    # handoff-kf, so the pair isolates the VL change alone.
    "handoff-vl": {"prefix": "HANDOFF", "suffix": "_vl", "control": "chunkA",
                   "head": "chunk1", "tail": "chunk2kf_vl", "seg0": 73, "cut": 94,
                   "seg1": 102},
    # TEST 3: keyframe + 5-frame reference VIDEO. The keyframe carries position, the ref
    # video carries MOTION, which one frame cannot encode.
    "handoff-ref": {"prefix": "HANDOFF", "suffix": "_ref", "control": "chunkA",
                    "head": "chunk1", "tail": "chunk2kf_ref", "seg0": 73, "cut": 94,
                    "seg1": 102},
    # TEST 4: same as TEST 3 plus a prompt DECLARING [video continuation]. Per the ref2va
    # guide a reference video is read as "reference generation" -- a style/camera hint --
    # unless the text declares otherwise, and the node passes an empty prompt today.
    "handoff-refp": {"prefix": "HANDOFF", "suffix": "_refp", "control": "chunkA",
                     "head": "chunk1", "tail": "chunk2kf_refp", "seg0": 73, "cut": 94,
                     "seg1": 102},
    # TEST 5: the owner's pick (refp) with the reference video grown 5 -> 22 frames, the
    # next length core allows. Half resolution keeps the block CHEAPER than refp's despite
    # 4.4x the temporal extent, because refs are not registered to the target grid.
    "handoff-refp22": {"prefix": "HANDOFF", "suffix": "_refp22", "control": "chunkA",
                       "head": "chunk1", "tail": "chunk2kf_refp22", "seg0": 73, "cut": 94,
                       "seg1": 102},
    # TEST 6: chunk2 CONTAINS chunk1 rather than referencing it -- its first 5 frames are
    # chunk1 refined output, held by a zeroed denoise mask. Delivery starts after them.
    "handoff-frz": {"prefix": "HANDOFF", "suffix": "_frz", "control": "chunkA",
                    "head": "chunk1", "tail": "chunk2frz", "seg0": 73, "cut": 94,
                    "seg1": 102},
    # TEST 7: frozen head PLUS a continued global noise sequence. Isolates noise against
    # handoff-frz, which is identical in every other respect.
    "handoff-frzn": {"prefix": "HANDOFF", "suffix": "_frzn", "control": "chunkA",
                     "head": "chunk1", "tail": "chunk2frzn", "seg0": 73, "cut": 94,
                     "seg1": 102},
    # TEST 8: frozen head + continued noise, NO keyframe. The keyframe cost ~12% of the
    # high-frequency detail; the mask enforces the frozen pixels without it.
    "handoff-frznk": {"prefix": "HANDOFF", "suffix": "_frznk", "control": "chunkA",
                      "head": "chunk1", "tail": "chunk2frznk", "seg0": 73, "cut": 94,
                      "seg1": 102},
    # TEST 9: frozen head + noise + HOLD ANCHOR + CLEAN LABEL, no keyframe.
    "handoff-frznc": {"prefix": "HANDOFF", "suffix": "_frznc", "control": "chunkA",
                      "head": "chunk1", "tail": "chunk2frznc", "seg0": 73, "cut": 94,
                      "seg1": 102},
    # TEST 10 (Job 2): frznc + the whole clip at half res as frozen context rows.
    "handoff-frzng": {"prefix": "HANDOFF", "suffix": "_frzng", "control": "chunkA",
                      "head": "chunk1", "tail": "chunk2frzng", "seg0": 73, "cut": 94,
                      "seg1": 102},
    # TEST 11: 22 frozen frames instead of 5 -- the sliding window, further along the axis.
    "handoff-frz22": {"prefix": "HANDOFF", "suffix": "_frz22", "control": "chunkA",
                      "head": "chunk1", "tail": "chunk2frz22", "seg0": 73, "cut": 94,
                      "seg1": 102},
    # TEST 12: 39 frozen frames -- the practical ceiling (56 would deliver nothing).
    "handoff-frz39": {"prefix": "HANDOFF", "suffix": "_frz39", "control": "chunkA",
                      "head": "chunk1", "tail": "chunk2frz39", "seg0": 73, "cut": 94,
                      "seg1": 102},
    # TEST 13: frz22 with the anchor RELEASED below sigma 0.55 -- clean while it steers
    # motion, re-noised before detail forms.
    "handoff-frz22r": {"prefix": "HANDOFF", "suffix": "_frz22r", "control": "chunkA",
                       "head": "chunk1", "tail": "chunk2frz22r", "seg0": 73, "cut": 94,
                       "seg1": 102},
}


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
        container.mux(stream.encode(
            av.VideoFrame.from_ndarray(np.clip(f, 0, 255).astype(np.uint8), format="rgb24")))
    container.mux(stream.encode())
    container.close()
    print(f"  wrote {path.name}  ({len(seq)} frames)")


def take(cache, arm, lo, hi):
    """Source frames [lo, hi) from `arm`, failing loudly if the arm does not cover them."""
    if arm not in cache:
        path = OUTPUT_DIR / f"SEAM_{arm}.mp4"
        if not path.is_file():
            raise SystemExit(f"missing {path} -- run: run_ab_h3seam.py --only {arm}")
        cache[arm] = frames(path)
    seq, start = cache[arm], ARM_START[arm]
    idx = [s - start for s in range(lo, hi)]
    if idx[0] < 0 or idx[-1] >= len(seq):
        raise SystemExit(f"{arm} covers source {start}..{start + len(seq) - 1}, "
                         f"cannot supply {lo}..{hi - 1}")
    return [seq[i] for i in idx], idx


def edge_distance(idx, n):
    return min(idx, n - 1 - idx)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="true-edge")
    # Loops are OFF by default: they were just the same clip repeated, which any
    # player can do, and they tripled the file count for no information.
    parser.add_argument("--loops", type=int, default=0,
                        help="repeat the clip N times into an extra _loop file; 0 = skip")
    args = parser.parse_args()
    p = PRESETS[args.preset]
    ctrl_arm, head_arm, tail_arm = p["control"], p["head"], p["tail"]
    seg0, cut, seg1 = p["seg0"], p["cut"], p["seg1"]
    prefix, suffix = p["prefix"], p["suffix"]

    cache = {}
    print(f"preset {args.preset}: source {seg0}..{seg1}, boundary at {cut - 1}->{cut}")
    print(f"  CONTROL {ctrl_arm} throughout   JOIN {head_arm} -> {tail_arm}\n")

    control, _ = take(cache, ctrl_arm, seg0, seg1 + 1)
    head, head_idx = take(cache, head_arm, seg0, cut)
    tail, tail_idx = take(cache, tail_arm, cut, seg1 + 1)
    join = head + tail

    n_head, n_tail = len(cache[head_arm]), len(cache[tail_arm])
    print(f"  join boundary: {head_arm}[{head_idx[-1]}] (edge "
          f"{edge_distance(head_idx[-1], n_head)}) -> {tail_arm}[{tail_idx[0]}] (edge "
          f"{edge_distance(tail_idx[0], n_tail)})")

    write_mp4(OUTPUT_DIR / f"{prefix}_control_no_boundary{suffix}.mp4", control)
    write_mp4(OUTPUT_DIR / f"{prefix}_join_with_boundary{suffix}.mp4", join)
    if args.loops:
        write_mp4(OUTPUT_DIR / f"{prefix}_control_loop{suffix}.mp4", control * args.loops)
        write_mp4(OUTPUT_DIR / f"{prefix}_join_loop{suffix}.mp4", join * args.loops)

    k = len(head)
    ja, jb = join[k - 1], join[k]
    ca, cb = control[k - 1], control[k]
    jd, cd = (ja - jb).mean(axis=(0, 1)), (ca - cb).mean(axis=(0, 1))
    print(f"\n  transition at source {cut - 1}->{cut}")
    print(f"    JOIN    (cross-chunk) |delta| {np.abs(ja - jb).mean():6.3f}/255   "
          f"DC {jd[0]:+.2f}/{jd[1]:+.2f}/{jd[2]:+.2f}")
    print(f"    CONTROL (same chunk)  |delta| {np.abs(ca - cb).mean():6.3f}/255   "
          f"DC {cd[0]:+.2f}/{cd[1]:+.2f}/{cd[2]:+.2f}")
    print(f"    boundary cost: {np.abs(ja - jb).mean() / np.abs(ca - cb).mean():.2f}x "
          f"the same transition inside one chunk")


if __name__ == "__main__":
    main()
