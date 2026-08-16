"""Seam A/B for ContextAnchoredTileUpscaleVLVideo on MOVING, dark-sky H3 content.

Why this harness exists
-----------------------
`run_node_h3.py` refines a 124-frame clip whose sky is essentially static and produces
no seam the owner can find. The same node, same settings, on a clip with a dark sky and
a moving camera (video_minimax_h3_r2v_multiref.json -> MiniMax_H3_refined_00002_.mp4)
produces a seam he can see. Settings were checked and match, so the variable is the
CONTENT — which this harness reproduces directly by refining the frames of that very
clip, at production geometry, one arm per (context_anchor, context_overlap) pair.

Stages
------
    frames      decoded from the owner's base mp4 (h264 8-bit), a window of `--frames`
                frames starting at `--start`, on the 17k+5 grid
    upscaled    the node's own lanczos 2x, recomputed here so the seam metric has a
                float reference the node never returns
    refined     the PRODUCTION node class, owner's widget values except anchor/overlap

Deviations from the owner's workflow, both deliberate and constant across arms:
  - av_latent is None (a silent zero audio latent). The mp4 carries no latent and the
    audio stream is frozen in every tile either way; what it changes is the frozen
    cross-modal context, identically for every arm.
  - the base frames are h264-decoded rather than the sampler's float decode. Both tiles
    of a seam read the SAME frames, so nothing about tile-to-tile disagreement changes.

The measurement
---------------
`--frames`-long clips make the delivered-mp4 statistics useless (h264 noise sits at the
same 1-2/255 as the effect). So the seam is measured where it is content-free: inside
the composite, the overlap band holds TWO INDEPENDENT REFINEMENTS OF THE SAME RAW
PIXELS — this tile's in `sub`, its neighbour's in `region`. Their difference is pure
tile-to-tile disagreement with the scene cancelled exactly.

`_Probe` wraps sampling.seam_dc_offset / sampling.seam_displacements (never patching
production code) and records, per tile and PER FRAME, the band-mean difference. That
splits the seam error into the two parts the design treats differently:

    offset  = mean over frames of the band mean   -> the STATIC step. One DC offset per
              clip is subtracted, so what is left here is the residual pedestal.
    swing   = std over frames of the band mean    -> the TEMPORAL part. A static scene
              cannot produce it; a moving one can, and no single per-clip correction
              can remove it.

    <venv-python> tests-AB/run_ab_h3seam.py --only base
    <venv-python> tests-AB/run_ab_h3seam.py --only ov128

ONE GPU sampling job at a time on this machine, and ONE ARM PER PROCESS (CLAUDE.md).
"""
import argparse
import contextlib
import faulthandler
import json
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

faulthandler.enable()

sys.path.insert(0, str(Path(__file__).resolve().parent))

import spike_h3 as poc  # noqa: E402  (module import runs the ComfyUI bootstrap)

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import ab_models  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from context_anchored_tile_refine import sampling, video  # noqa: E402
from context_anchored_tile_refine.node import ContextAnchoredTileUpscaleVLVideo  # noqa: E402

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-H3-seam")
SOURCE_MP4 = Path(r"C:\Users\Blake\Documents\ComfyUI\output\video\MiniMax_H3_00020_.mp4")

# The workflow's refine model is the ref2va checkpoint (UNETLoader 127 feeds both the
# sampler AND the refine node), NOT the fl2va one spike_h3 renders its own base with.
# That is a second difference from the clean run, so the arm list can switch it.
UNET_REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"

SRC_W, SRC_H = 1344, 768
FPS = 24

# Owner's widget values (workflow node 149), everything except anchor/overlap held fixed.
SEED = 42
# The default source window. Named because the output LABEL keys off it: a non-default
# --start or --frames adds a suffix so two windows of one arm cannot overwrite each other.
DEFAULT_START, DEFAULT_FRAMES = 102, 56
SAMPLER, SCHEDULER, STEPS, DENOISE, UPSCALE_BY = "res_multistep", "simple", 8, 0.22, 2.0

# max_tile_* gate only WHICH grid the solver picks, never the crop size (crop = base + r).
# 1600x1408 keeps the 2x2 layout — cores 1344x768, crops (1344+r)x(768+r) — for every r
# up to 256, so an arm changes r alone. The owner's 1408x1408 picks the SAME layout at
# r=64, which is what makes the `base` arm an exact reproduction.
MAX_TILE_W, MAX_TILE_H = 1600, 1408

# r = anchor + overlap fixes the crop size (base + r), so arms sharing an r sample crops
# of identical geometry and differ only in how that ring is split between the frozen
# anchor and the feathered overlap. `whole` lifts the caps above the canvas so the solver
# picks n=1 on both axes: ONE tile, no seam at all — the control that says whether the
# artifact is the tiling. At 56 frames its token count (84x48x17 = 68.5k) sits just under
# the validated 1408x1408x124f tile (44x44x37 = 71.6k), which is why it fits here and
# would not at 124 frames.
# Shared by every prompted arm so a second copy can never drift from the first: two arms that
# differ in reference LENGTH must not also differ by a stray word. Declaration only — a task
# type and a retention assertion, per the ref2va guide — with no scene description, no style
# words and no object nouns, which is the axis the phantom-object A/Bs found dangerous.
ARMS_PROMPT_CONTINUATION = (
    "subject_definitions:\n"
    "<Video 1> is the immediately preceding segment of the same continuous shot.\n\n"
    "summary:\n"
    "[video continuation] The target video continues directly from <Video 1>, preserving "
    "its camera motion, speed, lighting, and colour.\n\n"
    "retention_analysis:\n"
    "<Video 1>: fully_preserved - camera movement, pan speed, lighting, and colour continue "
    "unchanged.")

ARMS = {
    "base":     {"anchor": 32, "overlap": 32},                    # r=64, owner's config
    "ov0":      {"anchor": 64, "overlap": 0},                     # r=64, hard seam, same crop
    "ov64":     {"anchor": 32, "overlap": 64},                    # r=96
    "an64ov64": {"anchor": 64, "overlap": 64},                    # r=128
    "ov128":    {"anchor": 32, "overlap": 128},                   # r=160, wide feather
    "an128":    {"anchor": 128, "overlap": 32},                   # r=160, wide anchor
    "whole":    {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536},
    # Gate test for the whole packed-cond-row family — see _KeyframeProbe.
    "kfprobe":  {"anchor": 32, "overlap": 32, "keyframe_probe": True},
    # TEMPORAL decomposition instead of spatial. `whole` proved one 56-frame chunk of this
    # canvas fits (68.5k rows, 14.61 GiB) and is seam-free to the owner's eye, so the only
    # question left for a 158-frame clip is what happens at a CHUNK BOUNDARY. `chunkA` is
    # the 56 frames ENDING where `whole` begins: start 47 covers source 47..102, `whole`
    # covers 102..157, so source frame 102 is refined independently by both arms. The delta
    # on that one shared frame is the temporal seam, measured with zero scene change —
    # exactly the way the overlap-band probe cancels content on the spatial seam.
    "chunkA":   {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 47},
    # `chunkA` vs `whole` share exactly ONE source frame, and it fell inside the jump --
    # the loudest motion in the clip -- so that single sample cannot separate "the chunks
    # disagree" from "this frame is hard". `chunkC` starts at 90 and so covers source
    # 90..145, overlapping `whole` (102..157) on 44 frames that span both the jump and the
    # quiet sky after it. Those 44 give the disagreement as a FUNCTION of scene difficulty
    # instead of one reading, and they answer the question chunking actually turns on:
    # source 102..145 sit at positions 12..55 in chunkC but 0..43 in `whole`, so the same
    # content is rendered late-in-chunk and early-in-chunk. If disagreement is flat across
    # position, chunk boundaries are safe; if it grows toward a chunk edge, they are not.
    "chunkC":   {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 90},
    # The A/B built from chunkC+whole UNDERSTATES a real boundary: the frame after a join is
    # its chunk's FIRST frame (edge 0), but `whole[43]` sits mid-chunk (edge 12), worth ~12.5%.
    # `chunkE` covers source 35..90 and so ENDS exactly where `chunkC` begins, giving the
    # true worst case -- an edge-1 frame handing off to an edge-0 frame across chunks:
    #   JOIN     chunkE[54] (source 89) -> chunkC[0] (source 90)
    #   CONTROL  chunkA[42] (source 89) -> chunkA[43] (source 90)   both from ONE chunk
    # Same two source frames either way, so the difference is the boundary and nothing else.
    # Source 89->90 also sits well before the jump and the shot cut at 106.
    "chunkE":   {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 35},
    # --- THE KEYFRAME HANDOFF TEST, on a CUT-FREE window -------------------------------
    # Every arm above except chunkA contains one of the source's two scene cuts (frames 37
    # and 106), and chunkE opens with a 2-frame orphan of a departing shot -- the pathology
    # measured at 3x worse. This trio isolates the handoff with no cut assisting it.
    #
    #   chunk1   source  38..93   fully cut-free
    #   chunk2   source  93..148  cut at 106 lands at index 13, OUTSIDE the judged range
    #   BOUNDARY at source 93;  JUDGED 73..102;  CONTROL = chunkA (47..102, cut-free)
    #
    # chunk2 vs chunk2kf is the whole experiment: identical window, identical seed, the
    # only difference being whether chunk1's last REFINED frame is supplied as a clean
    # keyframe. Naive splicing is what the owner already sees glitch.
    "chunk1":   {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 38},
    "chunk2":   {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 93},
    "chunk2kf": {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 93,
                 # chunk1's LAST frame (source 93), lossless PNG rather than the crf10 mp4.
                 "keyframe_png": "SEAM_chunk1_f055.png"},
    # --- ISOLATION ARMS: each changes ONE thing vs chunk1 + chunk2kf ---------------------
    # TEST 1, lower denoise. BOTH chunks must move together: running the tail at 0.16 while
    # the head stays at 0.22 would put a refinement-strength step at the boundary and
    # confound the very thing being measured.
    "chunk1_d16":   {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 38,
                     "denoise": 0.16},
    "chunk2kf_d16": {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 93,
                     "denoise": 0.16, "keyframe_png": "SEAM_chunk1_d16_f055.png"},
    # TEST 2, VL sees the PREVIOUS chunk. Today `sample_conditioning_frames` picks every
    # 12th frame of THIS chunk's own upscaled source, so the VL matches style to the raw
    # upscale and knows nothing about the refined neighbour — the opposite of the spatial
    # path, where each tile's anchor halo encodes from the LIVE refined canvas. This arm
    # extends the reference video BACKWARDS in time with chunk1's refined frames. Source
    # 69/81/93 are chunk1 indices 31/43/55, and 93 is shared, so its raw pick is REPLACED
    # by chunk1's refined version rather than duplicated. 5 picks -> 7.
    "chunk2kf_vl":  {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 93,
                     "keyframe_png": "SEAM_chunk1_f055.png",
                     "vl_prepend": {"mp4": "SEAM_chunk1.mp4", "arm_start": 38,
                                    "sources": (69, 81, 93)}},
    # TEST 3, MOTION. A keyframe pins position and carries no velocity, so a chunk given one
    # frame reproduces the starting image then picks its own speed — structure continuous,
    # motion not, which is what the owner sees. A reference VIDEO carries motion. Core's
    # floor is 5 frames, so this is chunk1's last five refined frames (source 89..93),
    # alongside the keyframe: refs are NOT registered to the target grid and cannot replace
    # it. Adds one 2-latent-frame ref block, ~8,064 rows (+11.8%).
    "chunk2kf_ref": {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 93,
                     "keyframe_png": "SEAM_chunk1_f055.png",
                     "ref_video": {"mp4": "SEAM_chunk1.mp4", "arm_start": 38,
                                   "sources": (89, 90, 91, 92, 93)}},
    # TEST 4, DECLARE the relationship. Same as chunk2kf_ref plus a prompt, because the
    # ref2va guide is explicit that a reference video is read as `reference generation` — a
    # style/camera hint — unless the text declares `video continuation`, and that
    # `retention_analysis` is where `fully_preserved` is asserted. The node passes "" today,
    # so the reference arrives unlabelled.
    #
    # The text is DECLARATION ONLY: no scene description, no style words, no objects. The
    # phantom-object finding (AB26-AB40) was about DESCRIPTIVE text adding content demands
    # the model then invents; this adds none, and `detailed_description` is deliberately
    # omitted for exactly that reason. Volume is the risk axis, so it is kept to 3 lines.
    "chunk2kf_refp": {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 93,
                      "keyframe_png": "SEAM_chunk1_f055.png",
                      "ref_video": {"mp4": "SEAM_chunk1.mp4", "arm_start": 38,
                                    "sources": (89, 90, 91, 92, 93)},
                      "prompt": ARMS_PROMPT_CONTINUATION},
    # TEST 5, MORE MOTION. chunk2kf_refp is the owner's pick ("only the motion is the main
    # tell, but it is better"), so this changes exactly one thing: the reference video grows
    # 5 frames -> 22, the next size core allows (it trims until n % 17 == 5, so 5 and 22 are
    # adjacent). 22 frames = 0.92 s of motion instead of 0.21 s.
    #
    # Why HALF resolution. Ref rows = (h/32)*(w/32)*latent_t, and latent_t goes 2 -> 7, so at
    # full canvas this block would be 28,224 rows; with the keyframe that is +47% on a
    # 68,544-row target, projecting to ~25-27 GiB from the measured +0.42 GiB/1000-row slope
    # — past the 24 GB card. At half resolution it is 42*24*7 = 7,056 rows, FEWER than the
    # 5-frame full-res block it replaces, so the arm buys temporal extent at no memory cost.
    # Legitimate because refs are not registered to the target grid and core resizes them
    # itself; the keyframe still supplies full-resolution spatial detail. The prompt is
    # byte-identical to chunk2kf_refp so the only variable is reference LENGTH.
    "chunk2kf_refp22": {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 93,
                        "keyframe_png": "SEAM_chunk1_f055.png",
                        "ref_video": {"mp4": "SEAM_chunk1.mp4", "arm_start": 38,
                                      "sources": tuple(range(72, 94)), "scale": 0.5},
                        "prompt": ARMS_PROMPT_CONTINUATION},
    # TEST 6, DETERMINISTIC HANDOFF. Five conditioning levers moved the boundary between
    # 0.98x and 1.06x and the owner could not tell them apart, so the residual is the
    # stochastic difference between two independent refines — not something a reference can
    # reach. This stops referencing the previous chunk and starts CONTAINING it.
    #
    # chunk2 begins 5 frames EARLIER (89 instead of 93) so it overlaps chunk1. Its first 5
    # frames are overwritten with chunk1's refined output and the denoise mask is zeroed
    # over the latent frames they occupy (0..1 — exactly 5 image frames on H3's grid), so
    # the sampler holds them fixed at every step. Delivery starts at chunk2's frame 5, i.e.
    # source 94, so no frozen pixel survives into the output and nothing is re-diffused.
    # The keyframe stays because frozen target rows carry the CURRENT sigma, not a clean
    # label — the cond block is what tells the model those pixels are trustworthy.
    "chunk2frz": {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 89,
                  "keyframe_png": "SEAM_chunk1_f055.png",
                  "frozen_head": {"mp4": "SEAM_chunk1.mp4", "arm_start": 38,
                                  "sources": (89, 90, 91, 92, 93)}},
    # TEST 7, FROZEN HEAD + CONTINUED NOISE. The frozen head is the first lever the owner
    # ranked better by eye ("the ships do not speed up the same ... properly conveyed the
    # motion"), and it is also the first that could express VELOCITY: frozen frames sit in
    # the target segment at the target's own coordinates, so a position-at-t-1 vs
    # position-at-t relationship is representable. A reference block, on its own
    # area-normalised grid, structurally cannot say that.
    #
    # What is left is texture and brightness restarting at the join, which is noise, not
    # conditioning. Same arm plus a global noise offset so chunk2's field CONTINUES chunk1's
    # rather than restarting at index 0. Isolates the noise change alone.
    "chunk2frzn": {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 89,
                   "keyframe_png": "SEAM_chunk1_f055.png",
                   "frozen_head": {"mp4": "SEAM_chunk1.mp4", "arm_start": 38,
                                   "sources": (89, 90, 91, 92, 93)},
                   "noise_offset": 15},
    # TEST 8, DROP THE KEYFRAME. Measured on source 94..102, chunks matched for content and
    # position: chunkC (no keyframe) holds 2.334 detail while every keyframe-carrying arm
    # falls to 2.03-2.10 — the cond block costs ~13% of the high-frequency detail. That is
    # the "lossy second half" the owner sees, and chunk1 does not have it because chunk1
    # carries no keyframe.
    #
    # The keyframe existed only to give the frozen pixels a CLEAN timestep label, since
    # frozen rows inside the target segment carry the current sigma
    # ([[h3-frozen-rows-lack-clean-label]]). But the denoise mask enforces those pixels every
    # step whatever the model believes about them, so the label may be redundant — and it is
    # not worth 13% of the detail if it is. Frozen head + continued noise, nothing else.
    "chunk2frznk": {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 89,
                    "frozen_head": {"mp4": "SEAM_chunk1.mp4", "arm_start": 38,
                                    "sources": (89, 90, 91, 92, 93)},
                    "noise_offset": 15},
    # TEST 9, THE TRADE-OFF BREAKER. chunk2frznk is sharp (-0.3% detail) but the owner sees
    # the motion jump; chunk2frzn keeps the motion but is 13% soft because the keyframe pulls
    # the target toward a static frame. The keyframe was never carrying the motion — the
    # frozen frames were — it was only supplying a CLEAN TIMESTEP LABEL. This arm gives that
    # label directly to the frozen rows via mod_segments, at zero added rows and with no
    # block to be pulled toward. Same as chunk2frznk otherwise.
    "chunk2frznc": {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 89,
                    "frozen_head": {"mp4": "SEAM_chunk1.mp4", "arm_start": 38,
                                    "sources": (89, 90, 91, 92, 93)},
                    "noise_offset": 15,
                    "hold_anchor": True,
                    "clean_head_label": {"frozen_latent_frames": 2}},
    # TEST 10, JOB 2: the winning chunk2frznc config PLUS the whole clip at quarter
    # resolution as frozen context rows. Isolates global context against chunk2frznc, which
    # is identical in every other respect.
    "chunk2frzng": {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 89,
                    "frozen_head": {"mp4": "SEAM_chunk1.mp4", "arm_start": 38,
                                    "sources": (89, 90, 91, 92, 93)},
                    "noise_offset": 15,
                    "hold_anchor": True,
                    "clean_head_label": {"frozen_latent_frames": 2},
                    # Half, not quarter: quarter is 320x192 for 2,820 rows (+4.1%) — nearly
                    # free but coarse enough that a null result would be uninformative. Half
                    # is 672x384 for 11,844 rows (+17%), still affordable at ~16 GiB.
                    "global_ref": {"scale": 0.5}},
    # TEST 11, THE SLIDING WINDOW. The owner's "stream frames using the whole video as the
    # block" cannot work as literally stated: H3 is bidirectional and non-causal, so there is
    # no streaming state — one frame per forward means 158 forwards AND 157 independent
    # diffusion runs, i.e. a texture reseat at every frame. And the low-res clip cannot BE
    # the block: ref rows have now failed to carry temporal context in seven variants.
    #
    # What the instinct DOES map onto: freeze MORE of the target. 5 -> 22 frames is 7 latent
    # frames of real, exact context instead of 2, delivering 34 frames per chunk instead of
    # 51 (5 chunks for 158 frames instead of 3, ~1.7x compute). Same mechanism that already
    # works, further along the same axis. chunk1 covers 38..93 so it can supply 72..93.
    "chunk2frz22": {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 72,
                    "frozen_head": {"mp4": "SEAM_chunk1.mp4", "arm_start": 38,
                                    "sources": tuple(range(72, 94))},
                    # chunk2's latent 0 now covers source 72..76, which sits near chunk1's
                    # latent 10 rather than 15 — the offset tracks the head size.
                    "noise_offset": 10,
                    "hold_anchor": True,
                    "clean_head_label": {"frozen_latent_frames": 7}},
    # TEST 12, THE CEILING. 39 frozen frames = 12 of 17 latent frames held, delivering only
    # 17 frames per chunk (9 chunks for 158 frames, 3x compute). "Freeze everything" is not
    # reachable: at 56 the chunk generates nothing. chunk1 covers 38..93 so it can supply
    # 55..93. If motion is no better than 22, the axis has saturated and 22 is the setting.
    "chunk2frz39": {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 55,
                    "frozen_head": {"mp4": "SEAM_chunk1.mp4", "arm_start": 38,
                                    "sources": tuple(range(55, 94))},
                    # chunk2's latent 0 now covers source 55..59, near chunk1's latent 6.
                    "noise_offset": 6,
                    "hold_anchor": True,
                    "clean_head_label": {"frozen_latent_frames": 12}},
    # TEST 13, BOTH CORNERS. Every arm so far sits on one curve: clean anchor -> motion,
    # ~10% softer (frz22 at 2.183); re-noised anchor -> sharp, jumps (frznk at 2.396 against
    # a no-boundary 2.403, i.e. free). The two effects land at different points in the
    # schedule, so this releases the anchor between them. Measured sigmas at denoise 0.22 /
    # 8 steps / shift 12: 0.774 0.743 0.706 0.659 0.600 | 0.522 0.414 0.255 0.000 — a 0.55
    # threshold holds the anchor clean for steps 0-4 and frees it for 5-8. Otherwise
    # identical to chunk2frz22, so it isolates the schedule alone.
    "chunk2frz22r": {"anchor": 32, "overlap": 32, "mtw": 2688, "mth": 1536, "start": 72,
                     "frozen_head": {"mp4": "SEAM_chunk1.mp4", "arm_start": 38,
                                     "sources": tuple(range(72, 94))},
                     "noise_offset": 10,
                     "hold_anchor": {"release_below": 0.55},
                     "clean_head_label": {"frozen_latent_frames": 7}},
}


# ------------------------------------------------------------------ input

def load_frames(path, start, count):
    """A window of an mp4 as [T,H,W,3] float32 in 0..1, at the file's native size."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, check=True).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, SRC_H, SRC_W, 3)
    if start + count > len(frames):
        raise SystemExit(f"{path.name} has {len(frames)} frames; need {start}+{count}")
    window = np.ascontiguousarray(frames[start:start + count])
    return torch.from_numpy(window).float().div_(255.0)


# ------------------------------------------------------------------ probe

class _Probe:
    """Records the per-tile seam error the composite is about to hide.

    Wraps the two sampling entry points video.refine_video calls per tile, so the
    production path runs untouched and the numbers are the ones it actually used.
    `seam_dc_offset` sees the tile BEFORE the DC correction, `seam_displacements`
    AFTER it — so `dc` is the removed pedestal and everything derived in
    `_record` is the residual the feather still has to absorb.
    """

    def __init__(self):
        self.rows = []
        self._pending = None
        self._real_dc = sampling.seam_dc_offset
        self._real_disp = sampling.seam_displacements

    def __enter__(self):
        sampling.seam_dc_offset = self._dc
        sampling.seam_displacements = self._disp
        return self

    def __exit__(self, *exc):
        sampling.seam_dc_offset = self._real_dc
        sampling.seam_displacements = self._real_disp
        return False

    def _dc(self, tile, sub, region, region_mask=None):
        offset = self._real_dc(tile, sub, region, region_mask)
        self._pending = None if offset is None else offset.float().cpu().tolist()
        return offset

    def _disp(self, tile, sub, region):
        self._record(tile, sub, region)
        return self._real_disp(tile, sub, region)

    def _record(self, tile, sub, region):
        paste, core = tile.paste_rect, tile.core
        top_band = core.y0 - paste.y0
        left_band = core.x0 - paste.x0
        bands = {}
        if tile.kept_top and top_band > 0:
            bands["top"] = (sub[:, :top_band, :, :] - region[:, :top_band, :, :])
        if tile.kept_left and left_band > 0:
            bands["left"] = (sub[:, :, :left_band, :] - region[:, :, :left_band, :])
        for side, diff in bands.items():
            d = diff.float()
            per_frame = d.mean(dim=(1, 2)).cpu()                  # [T,3] band DC per frame
            rms = d.pow(2).mean(dim=(1, 2, 3)).sqrt().cpu()       # [T]   total disagreement
            self.rows.append({
                "tile": f"r{tile.row}c{tile.col}", "side": side,
                "dc_removed_255": self._pending,
                "band_px": top_band if side == "top" else left_band,
                "offset_255": (per_frame.mean(dim=0) * 255).tolist(),
                "swing_255": (per_frame.std(dim=0) * 255).tolist(),
                "peak_255": (per_frame.abs().max(dim=0).values * 255).tolist(),
                "rms_255": float(rms.mean() * 255),
                "per_frame_luma_255": (per_frame.mean(dim=1) * 255).tolist(),
            })


class _KeyframeProbe:
    """Attach a whole-crop KEYFRAME cond block per tile, using core's own shipped key.

    This is a GATE TEST, not a fix. Every packed-cond-row design rests on one unattested
    precondition: at denoise 0.22, does a clean, target-registered frozen cond block change
    the tile's output AT ALL? Six independent H3 implementations agree on the consumed input
    set and none attests a partial-canvas cond block, so the precondition has to be measured
    before an afternoon is spent on the clamped-band shape.

    Why a keyframe is the right probe: it is the ONLY cond block core registers to the
    TARGET's own spatial grid (`g[:,1:] = frame`, comfy/ldm/minimax/model.py:314-316), and
    it is settable from a node with no wrapper and no core edit — `minimax_keyframes` is a
    plain conditioning key (comfy_extras/nodes_minimax_h3.py:143-148), and `convert_cond`
    copies extras straight into the cond dict that becomes `extra_conds` kwargs
    (comfy/sampler_helpers.py:59-69). Cost is one latent frame: 1,144 rows = +2.1%.

    Its CONTENT is what makes the delta meaningful. The block is the LIVE canvas crop's
    frame 0, which differs from the tile's own latent exactly where an earlier tile pasted:
    `tile_encode_input` refills `overlap_inner_rect` from the FROZEN RAW canvas
    (video.py:167-168), so across the cross-dissolve band the tile holds RAW while the live
    canvas holds the neighbour's REFINED output. A non-zero delta therefore reads as "the
    neighbour's refined band, supplied clean, moved this tile".

    Built-in control: tile (0,0) has no already-processed neighbour, so live == raw there and
    its keyframe AGREES with its own latent. A real effect shows up on tiles 1-3 and not on
    tile 0; a uniform shift on all four is the block perturbing everything, not locking a seam.

    Being whole-canvas and pinned to one latent frame, it is a poor FIX (the determination
    records it as a near miss: only 69 of 1,144 cells informative). It is a good PROBE.
    """

    def __init__(self, vae, frame_count):
        self.vae = vae
        self.frame_count = frame_count
        self.shapes = []
        self._pending = None
        from context_anchored_tile_refine import video, vl_video
        self._video = video
        self._vl_video = vl_video
        self._real_encode_input = video.tile_encode_input
        self._real_slice_rows = vl_video.slice_rows

    def __enter__(self):
        self._video.tile_encode_input = self._encode_input
        self._vl_video.slice_rows = self._slice_rows
        return self

    def __exit__(self, *exc):
        self._video.tile_encode_input = self._real_encode_input
        self._vl_video.slice_rows = self._real_slice_rows
        return False

    def _encode_input(self, raw, live, tile):
        # Encode the LIVE crop's frame 0 HERE, not in _slice_rows: this runs immediately
        # before video.py's own vae.encode of the same tile, so the VAE is about to be
        # resident either way and the probe adds no extra model swap.
        crop = tile.crop_rect
        live_frame0 = live[:1, crop.y0:crop.y1, crop.x0:crop.x1, :]
        self._pending = self.vae.encode(live_frame0)
        self.shapes.append(tuple(self._pending.shape))
        return self._real_encode_input(raw, live, tile)

    def _slice_rows(self, pack, crop, canvas_h, canvas_w):
        cond = self._real_slice_rows(pack, crop, canvas_h, canvas_w)
        if self._pending is None:
            return cond
        # convert_cond has already flattened extras into each dict, so setting the keys
        # here is exactly what MiniMaxH3ImageToVideo's conditioning_set_values produces.
        for entry in cond:
            entry["minimax_keyframes"] = [{"resolved_frame_index": 0, "latent": self._pending}]
            entry["minimax_frame_count"] = self.frame_count
        self._pending = None
        return cond


class _ChunkHandoff:
    """Condition this chunk on the PREVIOUS chunk's last refined frame, as a clean keyframe.

    This is the temporal analogue of the node's spatial `context_anchor`, but riding the one
    mechanism H3 was actually TRAINED for: `minimax_keyframes` is "first frame -> video",
    which is precisely fl2va's task. The spatial seam has no such trained path — six
    independent H3 implementations attest none — and that asymmetry is the whole reason the
    temporal split is the easier one to fix.

    Why it is not the same as freezing pixels. A keyframe is a SEPARATE packed block of cond
    rows carrying the target's own spatial grid (`g[:,1:] = frame`,
    comfy/ldm/minimax/model.py:314-316) and labelled max(t_v, 0.999) — told it is CLEAN —
    while the target rows carry the current sigma. Pixels frozen inside the target segment by
    a denoise mask get no such label; the DiT never sees the mask at all. So the keyframe is
    the only way to hand this chunk a reference it is told to trust.

    At denoise 0.22 the target already holds the upscaled source, so the keyframe does not
    replace frame 0 — it is an additional clean anchor, and the chunk still refines normally.
    That is why the delivered clip can simply DISCARD this chunk's frame 0 (the previous
    chunk already supplied that source frame) and start at frame 1.

    Known limit, stated up front: the kfprobe gate test measured a 6x effect at the frames a
    cond block occupies, decaying to the floor by frame 4. So this should SPREAD the texture
    reseat over ~4 frames rather than delete it, and it cannot touch the ~2.5% whole-chunk
    detail difference between two independently sampled chunks.
    """

    def __init__(self, vae, frame_count, png_path):
        import numpy as np
        from PIL import Image
        if not png_path.is_file():
            raise SystemExit(f"handoff keyframe missing: {png_path}\n"
                             f"run the previous chunk first (--only chunk1)")
        arr = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.float32) / 255.0
        self.vae = vae
        self.frame_count = frame_count
        self.png_path = png_path
        self.pixels = torch.from_numpy(arr)[None]          # [1,H,W,3], the refined frame
        self.latent = None
        from context_anchored_tile_refine import vl_video
        self._vl_video = vl_video
        self._real_slice_rows = vl_video.slice_rows

    def __enter__(self):
        self.latent = self.vae.encode(self.pixels)
        print(f"handoff: keyframe {self.png_path.name} -> latent {tuple(self.latent.shape)}",
              flush=True)
        self._vl_video.slice_rows = self._slice_rows
        return self

    def __exit__(self, *exc):
        self._vl_video.slice_rows = self._real_slice_rows
        return False

    def _slice_rows(self, pack, crop, canvas_h, canvas_w):
        cond = self._real_slice_rows(pack, crop, canvas_h, canvas_w)
        # convert_cond flattens extras into each cond dict, so setting the keys here is
        # exactly what MiniMaxH3ImageToVideo's conditioning_set_values produces.
        for entry in cond:
            entry["minimax_keyframes"] = [
                {"resolved_frame_index": 0, "latent": self.latent}]
            entry["minimax_frame_count"] = self.frame_count
        return cond


class _VLPrepend:
    """Extend the VL reference video BACKWARDS in time with the PREVIOUS chunk's refined frames.

    What the node does today: `vl_video.sample_conditioning_frames` picks every FPS//2-th
    frame of THIS chunk's own upscaled source and encodes them once through Qwen3-VL; that
    encode IS the positive conditioning. So a chunk's VL matches style to the RAW upscale and
    has no knowledge of the refined neighbour it must continue — the exact opposite of the
    spatial path, where each tile's `context_anchor` halo encodes from the LIVE, already
    refined canvas (video.py:166).

    What this changes: the reference video is extended backwards using the previous chunk's
    REFINED output. Because the two chunks are contiguous in time, prepending those frames at
    the same 2 fps cadence produces a presentation that is still a single continuous video —
    the reference simply starts earlier and its early frames happen to be refined. The shared
    boundary frame is REPLACED rather than duplicated, so the cadence stays uniform.

    Cost: picks 5 -> 7, so the encode grows by two 2 fps frames. `token_layout` WALKS the
    token stream rather than assuming a count (vl_video.py:107-123), so the extra vision rows
    are picked up automatically; `rows_per_block` is unchanged because every prepended frame
    is at the same canvas size as the picks it joins.

    WHERE this patches, and why not the obvious place. The node's RAM-safe stage order picks
    at SOURCE resolution and upscales only the picks afterwards (node.py:314-326):

        314  picks, stamps = sample_conditioning_frames(image)   # 768x1344, source res
        320  picks = prepare_upscaled_video(picks, ...)          # picks upscaled
        322  vl_pack = encode_global(clip, picks, stamps)        # 1536x2688, canvas res

    Refined frames are already at canvas resolution, so they must join at 322, not 314.
    Patching `sample_conditioning_frames` fails with a tensor-size mismatch (1536 vs 768) —
    and quietly downscaling them to meet the source-res picks would resample the very
    refined content the arm exists to show the encoder.
    """

    def __init__(self, mp4_path, arm_start, sources, this_start):
        self.mp4_path = mp4_path
        self.arm_start = arm_start
        self.sources = tuple(sources)
        self.this_start = this_start
        self.extra = None
        from context_anchored_tile_refine import vl_video
        self._vl_video = vl_video
        self._real_encode_global = vl_video.encode_global

    def __enter__(self):
        if not self.mp4_path.is_file():
            raise SystemExit(f"vl_prepend source missing: {self.mp4_path}")
        want = [s - self.arm_start for s in self.sources]
        wanted = set(want)
        grabbed = {}
        cmd = ["ffmpeg", "-v", "error", "-i", str(self.mp4_path),
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        index, nbytes = 0, SRC_W * 2 * SRC_H * 2 * 3
        while wanted:
            buf = proc.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            if index in wanted:
                arr = np.frombuffer(buf, np.uint8).reshape(SRC_H * 2, SRC_W * 2, 3)
                grabbed[index] = arr.astype(np.float32) / 255.0
                wanted.discard(index)
            index += 1
        proc.stdout.close()
        proc.kill()
        missing = [s for s, i in zip(self.sources, want, strict=True) if i not in grabbed]
        if missing:
            raise SystemExit(f"vl_prepend: {self.mp4_path.name} lacks source frames {missing}")
        self.extra = torch.from_numpy(np.stack([grabbed[i] for i in want]))
        print(f"vl_prepend: {len(self.sources)} refined frames from {self.mp4_path.name} "
              f"(source {self.sources}) -> {tuple(self.extra.shape)}", flush=True)
        self._vl_video.encode_global = self._encode_global
        return self

    def __exit__(self, *exc):
        self._vl_video.encode_global = self._real_encode_global
        return False

    def _encode_global(self, clip, picks, stamps):
        if tuple(self.extra.shape[1:]) != tuple(picks.shape[1:]):
            raise SystemExit(
                f"vl_prepend: refined frames are {tuple(self.extra.shape[1:])} but the "
                f"upscaled picks are {tuple(picks.shape[1:])}; they must match at canvas "
                "resolution")
        # The last prepended source frame is this chunk's own frame 0, already rendered by
        # the previous chunk. Drop the raw pick for it so the refined one is not duplicated
        # and the 2 fps cadence stays uniform.
        overlap = 1 if self.sources and self.sources[-1] == self.this_start else 0
        merged = torch.cat(
            [self.extra.to(dtype=picks.dtype, device=picks.device), picks[overlap:]], dim=0)
        merged_stamps = [i / 2.0 for i in range(int(merged.shape[0]))]
        print(f"vl_prepend: picks {int(picks.shape[0])} -> {int(merged.shape[0])} "
              f"(replaced {overlap} raw pick with its refined version)", flush=True)
        return self._real_encode_global(clip, merged, merged_stamps)


class _RefVideoContext:
    """Keyframe for POSITION plus a reference VIDEO for MOTION. Owns both patch sites itself.

    Why this exists. A single keyframe pins where things ARE and says nothing about how fast
    they are moving, so a chunk conditioned on one frame reproduces the starting image and
    then chooses its own velocity — structure continuous, speed not. Only a multi-frame
    reference carries motion.

    The two mechanisms are complementary, not alternatives:
      `minimax_keyframes`  registered to the TARGET grid (model.py:314-316), 1 frame,
                           first/last only -> exact position, no velocity.
      `minimax_refs`       its OWN area-normalised grid (model.py:77-81, 324-367), >=5
                           frames -> motion and colour over time, but NOT positionally
                           registered, so it cannot replace the keyframe.

    Core's floor is hard (nodes_minimax_h3.py:249-253): fewer than 5 frames raises, and the
    count is then trimmed until n % 17 == 5. So 5 is the smallest legal reference video.

    Both a tokenizer ITEM and a DiT BLOCK are emitted, the way core does it
    (nodes_minimax_h3.py:263-267): the item creates the <Video k> label rows in the token
    stream the model was trained to see, the block supplies the packed latent rows. Emitting
    only the block would leave the reference unannounced in the text stream.

    This class patches BOTH `encode_global` (to add the item) and `slice_rows` (to add the
    cond keys), rather than composing with _ChunkHandoff/_VLPrepend, because those two patch
    one site each and stacking them would have one silently overwrite the other.
    """

    def __init__(self, vae, frame_count, png_path, mp4_path, arm_start, sources, prompt=None,
                 ref_scale=1.0):
        self.vae = vae
        self.frame_count = frame_count
        self.png_path = png_path
        self.mp4_path = mp4_path
        self.arm_start = arm_start
        self.sources = tuple(sources)
        self.prompt = prompt
        self.ref_scale = ref_scale
        self.keyframe_latent = None
        self.ref_latent = None
        self.ref_pixels = None
        self._restore_extra_conds = None
        from context_anchored_tile_refine import vl_video
        self._vl_video = vl_video
        self._real_encode_global = vl_video.encode_global
        self._real_slice_rows = vl_video.slice_rows

    def _patch_payload(self):
        """Work around a ComfyUI 0.31.0 bug: refs OVERWRITE the keyframes' cond latents.

        comfy/model_base.py:2156-2163 assembles the payload as

            if keyframes is not None:
                payload["cond_video_latents"] = [kf["latent"] for kf in keyframes]
            if refs is not None:
                payload["cond_video_latents"] = [r["latent"] for r in refs ...]   # <-- =, not +=

        while `PackedLayout` is still built with BOTH (model_base.py:2176-2179), so
        `img_update` marks keyframe AND ref rows frozen but only the refs' values are
        supplied. The result is a hard failure at model.py:580,
        `all_video_rows[~img_update] = cond_video_rows`, shape [8064,96] into [12096,96].

        Core never trips this because its image-to-video and reference-to-video nodes each
        set only one of the two. Combining them is what this arm needs, so the list is
        rebuilt in PackedLayout's own segment order: keyframes ("cond") precede refs
        ("ref_img") (model.py:300-367).
        """
        import comfy.model_base
        cls = comfy.model_base.MiniMaxH3
        real_extra_conds = cls.extra_conds

        def extra_conds(model_self, **kwargs):
            out = real_extra_conds(model_self, **kwargs)
            payload = out.get("minimax_payload")
            if payload is None:
                return out
            data = payload.cond
            keyframes, refs = data.get("keyframes"), data.get("refs")
            if keyframes and refs:
                data["cond_video_latents"] = (
                    [kf["latent"] for kf in keyframes]
                    + [r["latent"] for r in refs if "latent" in r])
                print(f"refvideo: repaired cond_video_latents -> "
                      f"{len(data['cond_video_latents'])} blocks "
                      f"({len(keyframes)} keyframe + {len(refs)} ref)", flush=True)
            return out

        cls.extra_conds = extra_conds
        self._restore_extra_conds = (cls, real_extra_conds)

    def __enter__(self):
        import numpy as np
        from PIL import Image
        if len(self.sources) < 5:
            raise SystemExit(f"ref video needs >=5 frames, got {len(self.sources)} "
                             "(comfy_extras/nodes_minimax_h3.py:249)")
        self._patch_payload()
        arr = np.asarray(Image.open(self.png_path).convert("RGB"), dtype=np.float32) / 255.0
        self.keyframe_latent = self.vae.encode(torch.from_numpy(arr)[None])
        self.ref_pixels = _read_frames(self.mp4_path,
                                       [s - self.arm_start for s in self.sources])
        # The tokenizer keeps the FULL-resolution frames even when the latent is downscaled.
        # `token_layout` derives one `rows_per_block` from the main picks and assumes every
        # vision block matches it (vl_video.py:107-141), so a half-resolution ref emits a
        # 42x24=1008-cell block where 84x48=4032 is expected and the node's fail-fast fires:
        #   "encoded conditioning has 13222 rows, expected 16246" == 3*4032 + 1*1008 + 118.
        # Splitting them is safe because the two carry different things: the tokenizer item
        # supplies the <Video k> label and vision tokens, the latent block supplies the
        # packed rows the DiT freezes. Only the latter scales with resolution, and only the
        # latter is what made a 22-frame reference unaffordable.
        # Keep ONLY the 2 fps picks at full resolution, not the whole reference. The
        # tokenizer sees every FPS//2-th frame (nodes_minimax_h3.py:261-262), so a 22-frame
        # reference contributes just 2 frames there — holding all 22 at 2688x1536 fp32 would
        # pin 1.09 GB of host RAM for the entire sampling run to use 99 MB of it.
        self.ref_pixels_full = self.ref_pixels[::FPS // 2].clone()
        if self.ref_scale != 1.0:
            # Reference blocks carry their OWN area-normalised grid and are NOT registered
            # to the target (model.py:77-81, 324-367), and core resizes them itself via
            # adapt_canvas (nodes_minimax_h3.py:241-245). So a smaller reference is a legal
            # reference, not a degraded one. It is what makes a LONGER reference affordable:
            # rows scale with latent_t x h x w, and this arm is buying temporal extent.
            # Lossy, and on CONDITIONING only — sampled tiles are never resampled
            # (prime directive 1). Area resampling, not bilinear, for a clean downscale.
            _, h, w, _ = self.ref_pixels.shape
            new_h = round(h * self.ref_scale) // 32 * 32
            new_w = round(w * self.ref_scale) // 32 * 32
            self.ref_pixels = torch.nn.functional.interpolate(
                self.ref_pixels.movedim(-1, 1), size=(new_h, new_w), mode="area"
            ).movedim(1, -1).contiguous()
            print(f"refvideo: ref downscaled {w}x{h} -> {new_w}x{new_h} "
                  f"(scale {self.ref_scale})", flush=True)
        self.ref_latent = self.vae.encode(self.ref_pixels)
        self.ref_pixels = None          # encoded; nothing downstream needs the pixels
        print(f"refvideo: keyframe {self.png_path.name} -> {tuple(self.keyframe_latent.shape)}; "
              f"ref {len(self.sources)} frames (source {self.sources}) -> "
              f"{tuple(self.ref_latent.shape)}", flush=True)
        self._vl_video.encode_global = self._encode_global
        self._vl_video.slice_rows = self._slice_rows
        return self

    def __exit__(self, *exc):
        self._vl_video.encode_global = self._real_encode_global
        self._vl_video.slice_rows = self._real_slice_rows
        if self._restore_extra_conds is not None:
            cls, real = self._restore_extra_conds
            cls.extra_conds = real
        return False

    def _encode_global(self, clip, picks, stamps):
        # Core shows Qwen the reference video at 2 fps like any other video
        # (nodes_minimax_h3.py:261-264); at 5 frames that is a single frame.
        import types
        real_tokenize = clip.tokenize
        qwen = self.ref_pixels_full          # already the 2 fps picks, full resolution
        item = {"type": "video", "data": qwen.to(picks.dtype),
                "timestamps": [i / 2.0 for i in range(int(qwen.shape[0]))]}

        def tokenize(_self, text, **kwargs):
            items = list(kwargs.pop("minimax_ref_items", []))
            items.append(item)
            # The node always passes "" (vl_video.py:136). Per the ref2va prompt guide, a
            # reference video with no declared task type defaults to `reference generation`
            # — a style/camera hint — NOT `video continuation`. So an empty prompt leaves
            # this arm's reference unlabelled and the model has no instruction to continue
            # from it. A prompt here declares the relationship instead of describing content.
            if self.prompt is not None:
                text = self.prompt
                print(f"refvideo: prompt {len(text)} chars (was empty)", flush=True)
            print(f"refvideo: tokenizer items {len(items) - 1} -> {len(items)}", flush=True)
            return real_tokenize(text, minimax_ref_items=items, **kwargs)

        clip.tokenize = types.MethodType(tokenize, clip)
        try:
            return self._real_encode_global(clip, picks, stamps)
        finally:
            clip.tokenize = real_tokenize

    def _slice_rows(self, pack, crop, canvas_h, canvas_w):
        cond = self._real_slice_rows(pack, crop, canvas_h, canvas_w)
        z = self.ref_latent
        block = {"kind": "video", "latent_t": int(z.shape[2]),
                 "latent_h": int(z.shape[3]), "latent_w": int(z.shape[4]),
                 "ref_audio_t": 0, "latent": z, "audio_latent": None}
        for entry in cond:
            entry["minimax_keyframes"] = [
                {"resolved_frame_index": 0, "latent": self.keyframe_latent}]
            entry["minimax_frame_count"] = self.frame_count
            entry["minimax_refs"] = [block]
        return cond


class _FrozenHead:
    """Hand the next chunk the previous chunk's actual PIXELS, frozen — not a reference to them.

    Why, after five conditioning levers changed nothing. keyframe, VL-sees-previous, 5-frame
    reference, 22-frame reference and a continuation prompt all landed between 0.98x and 1.06x
    and were indistinguishable to the owner. Five independent levers with no differentiation
    says conditioning is not the binding constraint: at denoise 0.22 the target latent already
    holds most of the source, and what differs at a join is the stochastic part conditioning
    does not control — two independent refines of the same content differ by ~3.3/255 in
    texture realisation. A reference is advisory; the sampler may or may not follow it.

    This is not advisory. Chunk N+1 is started 5 frames EARLIER so it overlaps chunk N, its
    first 5 input frames are overwritten with chunk N's REFINED output, and the denoise mask
    is zeroed over the latent frames those occupy. The sampler re-applies the mask every step
    (which is why the mask is binary, see CLAUDE.md), so those frames are held EXACTLY, at the
    target's own coordinates, inside the target segment. Zero chance in the overlap.

    The 5/2 split is H3's own: latent_t(n) = ((n-5)//17)*5+2, so image frames 0..4 occupy
    latent frames 0..1 exactly — freezing 2 latent frames freezes 5 image frames with no
    partial cell. Delivery discards those 5 frames (chunk N already rendered them) and starts
    at frame 5, so nothing frozen survives into the output and nothing is re-diffused.

    Known weakness, stated because it is why the keyframe is kept alongside: frozen rows sit
    INSIDE the target segment, and the DiT never sees the denoise mask — it labels them with
    the current sigma like everything else ([[h3-frozen-rows-lack-clean-label]]). So the model
    is not TOLD they are clean. The keyframe cond block supplies that label separately.
    """

    # A frozen head must land exactly on H3's temporal grid, latent_t(n) = ((n-5)//17)*5+2:
    #   5 image frames -> 2 latent,  22 -> 7,  39 -> 12.  Anything else straddles a cell.
    # Freezing MORE is the sliding window: more real context handed to the chunk, fewer
    # frames delivered per chunk, proportionally more compute.
    GRID: ClassVar[dict[int, int]] = {5: 2, 22: 7, 39: 12, 56: 17}

    def __init__(self, mp4_path, arm_start, sources):
        self.mp4_path = mp4_path
        self.arm_start = arm_start
        self.sources = tuple(sources)
        self.head = None
        self.HEAD_IMAGE_FRAMES = 0      # set in __enter__ once the head size is validated
        self.HEAD_LATENT_FRAMES = 0
        from context_anchored_tile_refine import video
        self._video = video
        self._real_refine = video.refine_video
        self._real_mask = video.tile_denoise_mask

    def __enter__(self):
        n = len(self.sources)
        if n not in self.GRID:
            raise SystemExit(f"frozen head of {n} frames does not land on H3's temporal grid; "
                             f"legal sizes are {sorted(self.GRID)}")
        self.HEAD_IMAGE_FRAMES = n
        self.HEAD_LATENT_FRAMES = self.GRID[n]
        self.head = _read_frames(self.mp4_path,
                                 [s - self.arm_start for s in self.sources])
        print(f"frozen_head: {len(self.sources)} refined frames from {self.mp4_path.name} "
              f"(source {self.sources[0]}..{self.sources[-1]}) -> {tuple(self.head.shape)}",
              flush=True)
        self._video.refine_video = self._refine_video
        self._video.tile_denoise_mask = self._mask
        return self

    def __exit__(self, *exc):
        self._video.refine_video = self._real_refine
        self._video.tile_denoise_mask = self._real_mask
        return False

    def _refine_video(self, frames, *args, **kwargs):
        n = self.HEAD_IMAGE_FRAMES
        if tuple(frames.shape[1:]) != tuple(self.head.shape[1:]):
            raise SystemExit(
                f"frozen_head: refined head is {tuple(self.head.shape[1:])} but the upscaled "
                f"canvas is {tuple(frames.shape[1:])}; they must match")
        frames = frames.clone()
        frames[:n] = self.head.to(dtype=frames.dtype, device=frames.device)
        print(f"frozen_head: substituted the first {n} upscaled frames", flush=True)
        return self._real_refine(frames, *args, **kwargs)

    def _mask(self, tile, latent_t, audio):
        nested = self._real_mask(tile, latent_t, audio)
        video_mask, audio_mask = nested.unbind()
        video_mask = video_mask.clone()
        # 1 = diffuse, 0 = frozen (video.py:160). Zero the head so the sampler restores those
        # latent frames from the input at every step instead of denoising them.
        video_mask[:, :, :self.HEAD_LATENT_FRAMES] = 0.0
        print(f"frozen_head: denoise mask zeroed over latent frames "
              f"0..{self.HEAD_LATENT_FRAMES - 1} of {latent_t}", flush=True)
        return self._video._nested(video_mask, audio_mask)


class _GlobalContextRef:
    """Give the chunk the WHOLE clip at low resolution, as frozen context rows.

    Job 2. Every chunk sees only its own 56 frames; bidirectional attention over the packed
    sequence is what keeps motion coherent, and chunking amputates it. This hands the chunk a
    low-resolution copy of the entire clip so attention has the global trajectory to read.

    What it actually adds HERE, stated honestly: chunk2 covers source 89..144 and the clip
    ends at 157, so the FUTURE it gains is only 13 frames. The real addition is the PAST
    (source 0..88), which chunk2 otherwise has only as its 5 frozen frames. On a middle chunk
    of a longer clip the future would matter more.

    Resolution: quarter, so the block is affordable. Ref blocks carry their OWN
    area-normalised grid (`_frame_grid`, model.py:77-81) and — as the owner spotted — that
    grid's coordinate SPAN depends only on aspect ratio, not resolution, so a quarter-res
    block lands in the same normalised space as the target, just sampled coarsely.
    Rows = (h/32)*(w/32)*latent_t = 21*12*47 = 11,844, about +17%.

    NO tokenizer item is emitted, unlike core (nodes_minimax_h3.py:263-267). The layout builds
    ref segments from `payload["refs"]` alone, and a full-clip tokenizer item would be 14 qwen
    frames = 7 vision blocks = 28,224 rows at full resolution — it must be full-res to keep
    `rows_per_block` uniform (vl_video.py:107-141), which is what broke the first 22-frame
    attempt. Skipping it costs the `<Video k>` label rows and saves 2.4x the block itself.

    Composes with the frozen head rather than replacing it: this supplies global CONTEXT,
    the frozen head supplies the exact HANDOFF.
    """

    def __init__(self, vae, mp4_path, scale=0.25):
        self.vae = vae
        self.mp4_path = mp4_path
        self.scale = scale
        self.latent = None
        from context_anchored_tile_refine import vl_video
        self._vl_video = vl_video
        self._real_slice_rows = vl_video.slice_rows

    def __enter__(self):
        import numpy as np
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(self.mp4_path), "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, check=True).stdout
        clip = np.frombuffer(raw, np.uint8).reshape(-1, SRC_H, SRC_W, 3)
        n = video.align_frame_count(clip.shape[0])
        if n > clip.shape[0]:                      # repeat-pad onto H3's 17k+5 grid
            clip = np.concatenate([clip, np.repeat(clip[-1:], n - clip.shape[0], axis=0)])
        clip = clip[:n].astype(np.float32) / 255.0
        px = torch.from_numpy(clip)
        h = round(SRC_H * self.scale) // 32 * 32
        w = round(SRC_W * self.scale) // 32 * 32
        px = torch.nn.functional.interpolate(
            px.movedim(-1, 1), size=(h, w), mode="area").movedim(1, -1).contiguous()
        self.latent = self.vae.encode(px)
        rows = (self.latent.shape[3] // 2) * (self.latent.shape[4] // 2) * self.latent.shape[2]
        print(f"global_ref: whole clip {n} frames @ {w}x{h} -> latent "
              f"{tuple(self.latent.shape)} = {rows:,} frozen rows", flush=True)
        self._vl_video.slice_rows = self._slice_rows
        return self

    def __exit__(self, *exc):
        self._vl_video.slice_rows = self._real_slice_rows
        return False

    def _slice_rows(self, pack, crop, canvas_h, canvas_w):
        cond = self._real_slice_rows(pack, crop, canvas_h, canvas_w)
        z = self.latent
        block = {"kind": "video", "latent_t": int(z.shape[2]),
                 "latent_h": int(z.shape[3]), "latent_w": int(z.shape[4]),
                 "ref_audio_t": 0, "latent": z, "audio_latent": None}
        for entry in cond:
            entry["minimax_refs"] = [block]
        return cond


class _HoldAnchor:
    """Stop re-noising the frozen rows, so their content is usable from the FIRST step.

    The defect this fixes, found by review before it cost a run. The denoise mask does NOT
    hand clean values to the DiT on H3. At every step comfy re-noises the masked region back
    to the current sigma (`comfy/samplers.py:639`):

        x = x*denoise_mask + scale_latent_inpaint(x, sigma, noise, latent_image)*latent_mask

    and `MiniMaxH3` (model_base.py:2117) does NOT override `scale_latent_inpaint`, so it falls
    through to `CONST.noise_scaling` (model_sampling.py:94-97):

        sigma*noise + (1-sigma)*latent_image

    At denoise 0.22 with H3's sigma_shift_video 12, the first step's sigma is ~0.77 — the
    frozen head arrives as ~77% NOISE. Its motion information only emerges as sigma falls,
    far too late to steer the trajectory. That is why the frozen head alone "jumps", and why
    the keyframe rescued it: cond rows bypass this path entirely (injected via
    `all_video_rows[~img_update]`, model.py:580) and are clean from step 1 — at the price of
    being a STATIC frame the target is pulled toward for the whole schedule, which is the
    12-16% softening.

    Four models in core already do exactly this for their anchors — WAN21 (model_base.py:1919),
    WAN22 (:2058, "Hold anchor constant across all sigmas"), HunyuanVideo (:1391), LTXAV
    (:1237). H3 simply is not one of them. This borrows their behaviour for the frozen head.

    Pairs with `_CleanHeadLabel`: once the rows really ARE clean, the per-segment label
    (`t_v`, model.py:547) becomes wrong for them and the label patch becomes both correct and
    necessary. Applied alone, this makes the rows clean while the model still believes they
    carry the current sigma; applied alone, the label patch lies about noised rows. They are
    only meaningful together.
    """

    def __init__(self, release_below=None):
        import comfy.model_base
        self._cls = comfy.model_base.MiniMaxH3
        self._had_own = "scale_latent_inpaint" in vars(self._cls)
        self._saved = vars(self._cls).get("scale_latent_inpaint")
        # None = clean at every sigma. A float SCHEDULES the anchor: clean while sigma is
        # above it, then back to the re-noising default below.
        #
        # Why a schedule should beat either extreme. Every arm so far sits on one curve —
        # clean anchor gives motion and costs ~10% detail (frz22), re-noised anchor is
        # essentially free in detail but jumps (frznk, 2.396 vs the no-boundary 2.403). The
        # two effects happen at DIFFERENT points in the schedule: the trajectory is set early
        # at high sigma, fine detail is generated late at low sigma. Holding the anchor clean
        # only while it is steering, then releasing it before detail forms, should reach both
        # corners. The frozen frames are discarded from delivery, so letting them drift late
        # costs nothing.
        self.release_below = release_below

    def __enter__(self):
        threshold = self.release_below

        def scale_latent_inpaint(model_self, sigma, noise, latent_image, **kwargs):
            if threshold is None or float(sigma.flatten()[0]) > threshold:
                return latent_image                      # clean anchor
            return model_self.model_sampling.noise_scaling(sigma, noise, latent_image)

        self._cls.scale_latent_inpaint = scale_latent_inpaint
        if threshold is None:
            print("hold_anchor: MiniMaxH3.scale_latent_inpaint -> return latent_image "
                  "(frozen rows stay CLEAN at every sigma, as WAN22/HunyuanVideo do)",
                  flush=True)
        else:
            print(f"hold_anchor: anchor CLEAN while sigma > {threshold}, re-noised below "
                  "(clean while it steers motion, released before detail forms)", flush=True)
        return self

    def __exit__(self, *exc):
        if self._had_own:
            self._cls.scale_latent_inpaint = self._saved
        else:
            del self._cls.scale_latent_inpaint      # fall back to BaseModel's again
        return False


class _CleanHeadLabel:
    """Tell the model the frozen head rows are CLEAN, at ZERO added rows.

    The trade-off this exists to break. The owner's A/B: keyframe + frozen head gives good
    motion but ~13% softer output; frozen head alone is sharp but "everything jumps". The
    keyframe was never conveying motion — the frozen frames were. What the keyframe supplied
    is a *trust* signal, and it cost 4,032 rows of a static frame that the target then gets
    pulled toward (measured: keyframe -15.7% detail, keyframe+5-frame-ref only -12.8%, i.e.
    more temporal content in the block = less softening).

    Two independent labels ride on every row, and conflating them is what made this hard:

        AXIS A  position_ids[row] = (t, h, w)  -> RoPE -> WHO ATTENDS TO WHOM, WHEN
        AXIS B  mod_segments -> t_row[ts]*3+tag -> AdaLN -> HOW NOISY THE ROW IS BELIEVED

    The denoise mask pins the frozen rows' VALUES, but the DiT never sees the mask: timesteps
    are assigned per SEGMENT (`seg_t`, model.py:547-549), so frozen rows inside the target
    carry the current sigma. Frozen in value, announced as noisy. This patch moves them to a
    clean entry on Axis B alone — no new rows, no new block, nothing for the target to be
    pulled toward.

    Why it is expressible here and was NOT for the spatial seam: video rows are ordered
    (t, h, w), so latent frames 0..k-1 are a CONTIGUOUS PREFIX of the video segment. A spatial
    anchor ring is not contiguous in row order, which is why this lever was unavailable there.

    Hook: `patches_replace["dit"][("double_block", i)]`, which receives BOTH `t_emb` and
    `mod_segments` (model.py:630-636) and can rewrite them per block. `layout.segments` is
    left untouched, so the output extraction at model.py:643 — which finds the target by
    `k == "video"` — still sees one contiguous video segment and slices correctly.

    Mechanics: `AdalnProj(t_dim, hidden, 6, 3)` expands t_emb by the 3 modality tags, so
    APPENDING ONE ROW to t_emb creates modulation rows n*3+{0,1,2}; video is tag 0
    (`seg_tag`, model.py:553). The clean value is VISUAL_COND_TIMESTEP = 0.999, the same one
    `seg_t` pins cond/ref rows to.
    """

    CLEAN_T = 0.999

    def __init__(self, model_patcher, latent_t, frozen_latent_frames=2):
        self.model_patcher = model_patcher
        self.latent_t = int(latent_t)
        self.frozen_latent_frames = int(frozen_latent_frames)
        self._diffusion = model_patcher.model.diffusion_model
        self._emb_cache = {}
        self._reported = False

    def __enter__(self):
        n = len(self._diffusion.blocks)
        for i in range(n):
            self.model_patcher.set_model_patch_replace(self._patch, "dit", "double_block", i)
        print(f"clean_head_label: patched {n} blocks; latent frames "
              f"0..{self.frozen_latent_frames - 1} of {self.latent_t} -> t={self.CLEAN_T}",
              flush=True)
        return self

    def __exit__(self, *exc):
        # Unregister rather than relying on main() dropping the patcher: leaving 50 live
        # patches on a shared object is the kind of thing that silently contaminates a later
        # arm if this ever runs twice in one process.
        replace = self.model_patcher.model_options.get("transformer_options", {}) \
                                                  .get("patches_replace", {}).get("dit", {})
        for i in range(len(self._diffusion.blocks)):
            replace.pop(("double_block", i), None)
        return False

    def _clean_embedding(self, t_emb):
        key = (t_emb.device, t_emb.dtype)
        if key not in self._emb_cache:
            import comfy.model_management
            dm = self._diffusion
            vals = torch.tensor([self.CLEAN_T], dtype=torch.float32, device=t_emb.device)
            if getattr(dm, "use_adaln_curves", False):
                # mirrors model.py:611-616 exactly, including both clamps
                table = comfy.model_management.cast_to(dm.adaln_t_table, device=t_emb.device)
                pos = vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
                i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
                emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
            else:
                emb = dm.time_embedder(vals)
            self._emb_cache[key] = emb.to(dtype=t_emb.dtype, device=t_emb.device)
        return self._emb_cache[key]

    def _patch(self, args, extras):
        t_emb, mod = args["t_emb"], args["mod_segments"]
        # The target video segment is appended LAST (model.py:376-380), so it is the last
        # modulation entry. Guard rather than assume.
        a, b, row = mod[-1]
        span = b - a
        # latent_t is DERIVED from the run, never hand-entered: a wrong constant would still
        # divide evenly for some frame counts and silently split on the wrong rows.
        if span % self.latent_t != 0:
            raise RuntimeError(f"clean_head_label: last mod segment spans {span} rows, not a "
                               f"multiple of latent_t {self.latent_t} — it is not the video "
                               "segment and the patch would corrupt modulation")
        frozen_rows = (span // self.latent_t) * self.frozen_latent_frames
        clean_index = t_emb.shape[0]                      # appended row's timestep index
        t_ext = torch.cat([t_emb, self._clean_embedding(t_emb)], dim=0)
        # tag 0 == video (seg_tag, model.py:553). Entries stay NON-OVERLAPPING and contiguous:
        # _mod_scale_shift accumulates IN PLACE (model.py:203-207), so an overlap would apply
        # modulation twice.
        new_mod = [*mod[:-1], (a, a + frozen_rows, clean_index * 3 + 0),
                   (a + frozen_rows, b, row)]
        if not self._reported:
            print(f"clean_head_label: video segment [{a},{b}) split at {a + frozen_rows}; "
                  f"clean mod row {clean_index * 3}, normal {row}", flush=True)
            self._reported = True
        return extras["original_block"]({**args, "t_emb": t_ext, "mod_segments": new_mod})


class _NoiseOffset:
    """Continue the previous chunk's noise sequence instead of restarting it at index 0.

    `canvas_noise` draws ONE noise tensor per chunk from the seed (video.py:128-140), so two
    56-frame chunks at the same canvas and seed get IDENTICAL noise — but indexed
    chunk-locally. chunk2's latent frame 2 therefore gets the same noise chunk1 used for its
    latent frame 2, which covers completely different source frames. Across a join the noise
    field restarts, so texture realisation restarts with it.

    This draws a longer tensor and hands the chunk a SLICE at a global offset, which is what a
    single continuous render would have used:

        chunk1     source 38..93   latent 0..16  -> global noise 0..16
        chunk2frz  source 89..144  latent 0..16  -> global noise 15..31

    Offset 15 because chunk1's image frames 51..55 (source 89..93) land on latent ~15..16
    under `latent_t(n) = ((n-5)//17)*5+2`, and those are exactly chunk2's head frames 0..4.
    The shared source frames then draw the SAME noise, and the first delivered frame
    (chunk2 latent 2) continues at global 17 rather than reusing global 2.

    Audio is unaffected: its dummy is `zeros_like(audio)` regardless of latent_t, and the
    audio stream is frozen in every tile anyway (video.py:14-15), so its noise never reaches
    the output. It does come from the same generator, so extending the video draw shifts the
    audio values — harmless for that reason, and the reason this is not applied to chunk1.
    """

    def __init__(self, offset):
        self.offset = int(offset)
        from context_anchored_tile_refine import video
        self._video = video
        self._real = video.canvas_noise

    def __enter__(self):
        self._video.canvas_noise = self._canvas_noise
        return self

    def __exit__(self, *exc):
        self._video.canvas_noise = self._real
        return False

    def _canvas_noise(self, seed, latent_t, canvas_h, canvas_w, audio):
        video_noise, audio_noise = self._real(
            seed, latent_t + self.offset, canvas_h, canvas_w, audio)
        sliced = video_noise[:, :, self.offset:self.offset + latent_t].contiguous()
        if sliced.shape[2] != latent_t:
            raise SystemExit(f"noise offset {self.offset}: sliced {sliced.shape[2]} latent "
                             f"frames, need {latent_t}")
        print(f"noise_offset: drew {latent_t + self.offset} latent frames, using "
              f"[{self.offset}:{self.offset + latent_t}] -- continues the previous chunk's "
              "sequence instead of restarting", flush=True)
        return sliced, audio_noise


def _read_frames(mp4_path, indices):
    """Specific frames of an mp4 as [N,H,W,3] float32 in 0..1, at the file's native size."""
    import numpy as np
    if not mp4_path.is_file():
        raise SystemExit(f"missing {mp4_path}")
    wanted, grabbed, index = set(indices), {}, 0
    nbytes = SRC_W * 2 * SRC_H * 2 * 3
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(mp4_path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE)
    while wanted:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        if index in wanted:
            grabbed[index] = (np.frombuffer(buf, np.uint8)
                              .reshape(SRC_H * 2, SRC_W * 2, 3).astype(np.float32) / 255.0)
            wanted.discard(index)
        index += 1
    proc.stdout.close()
    proc.kill()
    if wanted:
        raise SystemExit(f"{mp4_path.name} lacks frame indices {sorted(wanted)}")
    return torch.from_numpy(np.stack([grabbed[i] for i in indices]))


def summarize(rows):
    # The `whole` arm solves to a single tile, so there is no kept side anywhere and the
    # probe records nothing. That is the arm's whole point, not a failure — report it and
    # hand back zeros rather than reducing over an empty list.
    if not rows:
        print("\n  no seam recorded: the layout solved to ONE tile, so no tile has an "
              "already-processed neighbour and no overlap band exists.")
        return 0.0, 0.0
    print("\n  seam error inside the overlap band (content cancels exactly; units /255)")
    print(f"  {'tile/side':12s} {'band':>5s} {'offset RGB':>21s} {'swing RGB':>21s} {'peak':>6s} {'rms':>6s}")
    for r in rows:
        off = "/".join(f"{v:+5.2f}" for v in r["offset_255"])
        swg = "/".join(f"{v:5.2f}" for v in r["swing_255"])
        print(f"  {r['tile'] + ' ' + r['side']:12s} {r['band_px']:5d} {off:>21s} {swg:>21s} "
              f"{max(r['peak_255']):6.2f} {r['rms_255']:6.2f}")
    worst_off = max(max(abs(v) for v in r["offset_255"]) for r in rows)
    worst_swing = max(max(r["swing_255"]) for r in rows)
    print(f"  WORST static offset {worst_off:.2f}/255   WORST temporal swing {worst_swing:.2f}/255")
    return worst_off, worst_swing


# ------------------------------------------------------------------ output

def write_mp4(path, frames, crf=10):
    """One encode of `frames` at the given CRF, from the FLOAT frames every time.

    Both CRFs are written from the master rather than by transcoding one into the
    other, because a crf10 -> crf23 transcode adds a second generation of loss and
    would overstate what SaveVideo's default costs. CRF 10 is the harness's usual
    near-transparent setting; CRF 23 is libx264's default, which is what SaveVideo
    lands on whenever its `codec` widget is left at `auto` (it passes no crf at all,
    comfy_extras/nodes_video.py SaveVideo.execute -> crf=encoding.get("crf")).
    """
    import av

    tmp = path.with_suffix(".tmp")
    container = av.open(str(tmp), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height = int(frames.shape[2]), int(frames.shape[1])
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": str(crf), "preset": "medium"}
    for frame in frames:
        picture = av.VideoFrame.from_ndarray(
            np.clip(255.0 * frame.cpu().float().numpy(), 0, 255).astype(np.uint8), format="rgb24")
        for packet in stream.encode(picture):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    Path(tmp).replace(path)
    print(f"  wrote {path}", flush=True)


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", default="base", choices=sorted(ARMS))
    parser.add_argument("--start", type=int, default=DEFAULT_START,
                        help="first source frame; 102 puts the window on the low-angle "
                             "sky shot the owner calls out, ending on the last frame. "
                             "Outputs gain an _f<start> suffix when this is non-default, so "
                             "two windows of one arm never overwrite each other.")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES,
                        help="17k+5 grid: 39, 56, 73, 90, 107, 124. Non-default adds an "
                             "n<frames> suffix to the output label.")
    parser.add_argument("--unet", default=UNET_REF2VA)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="refine seed. A second seed is not optional when comparing "
                             "context_anchor values: an earlier sweep found an anchor "
                             "winning on one lucky draw of the DC offset alone. Outputs "
                             "are suffixed _s<seed> so a second seed never overwrites "
                             "the first.")
    parser.add_argument("--no-video", action="store_true", help="skip the mp4 write")
    args = parser.parse_args()

    arm = ARMS[args.only]
    # _KeyframeProbe and _ChunkHandoff BOTH rebind vl_video.slice_rows. If an arm ever asked
    # for both, whichever entered the `with` last would win and the other would silently do
    # nothing — a null result that looks like a real measurement. No arm does today; this
    # keeps it that way.
    if arm.get("keyframe_probe") and arm.get("keyframe_png"):
        raise SystemExit(f"arm {args.only}: keyframe_probe and keyframe_png both patch "
                         "vl_video.slice_rows; one would silently win. Pick one.")
    anchor, overlap = arm["anchor"], arm["overlap"]
    max_tile_w = arm.get("mtw", MAX_TILE_W)
    max_tile_h = arm.get("mth", MAX_TILE_H)
    # An arm may pin its own source window when the window IS the experiment (chunkA).
    start = arm.get("start", args.start)
    # The label must carry EVERY axis that changes the pixels, or two runs of the same arm
    # silently overwrite each other. Temporal chunking runs one arm at several --start
    # values, so a start-blind label would have clobbered SEAM_whole.mp4 -- the reference
    # the owner judges against -- with a second window under the same name. Suffixes are
    # added only when the value is non-default, so existing filenames stay stable.
    label = args.only
    if args.seed != SEED:
        label += f"_s{args.seed}"
    if "start" not in arm and args.start != DEFAULT_START:
        label += f"_f{start}"
    if args.frames != DEFAULT_FRAMES:
        label += f"n{args.frames}"
    print(f"ComfyUI root: {poc.ROOT} version {poc.ab_env.version(poc.ROOT)}", flush=True)
    print(f"arm {label}: anchor={anchor} overlap={overlap} r={anchor + overlap} "
          f"max_tile={max_tile_w}x{max_tile_h} frames={args.frames}@{start} "
          f"unet={args.unet}", flush=True)

    frames = load_frames(SOURCE_MP4, start, args.frames)
    print(f"source {tuple(frames.shape)} from {SOURCE_MP4.name}", flush=True)

    vae = ab_models.load_vae(poc.VAE_NAME)
    model = ab_models.load_unet(args.unet)
    clip = ab_models.load_clip(poc.CLIP_NAME, poc.CLIP_TYPE)

    node = ContextAnchoredTileUpscaleVLVideo()
    # contextlib.nullcontext when the arm does not want the probe, so the base path stays
    # byte-identical: nothing is patched unless the arm asked for it.
    keyframe_probe = (_KeyframeProbe(vae, video.align_frame_count(args.frames))
                      if arm.get("keyframe_probe") else contextlib.nullcontext())
    rv = arm.get("ref_video")
    if rv:
        # _RefVideoContext owns BOTH patch sites, so it replaces _ChunkHandoff rather than
        # stacking with it — two objects patching slice_rows would have one silently win.
        handoff = _RefVideoContext(vae, video.align_frame_count(args.frames),
                                   OUTPUT_DIR / arm["keyframe_png"], OUTPUT_DIR / rv["mp4"],
                                   rv["arm_start"], rv["sources"], arm.get("prompt"),
                                   rv.get("scale", 1.0))
    elif arm.get("keyframe_png"):
        handoff = _ChunkHandoff(vae, video.align_frame_count(args.frames),
                                OUTPUT_DIR / arm["keyframe_png"])
    else:
        handoff = contextlib.nullcontext()
    noise_offset = (_NoiseOffset(arm["noise_offset"])
                    if arm.get("noise_offset") else contextlib.nullcontext())
    ha = arm.get("hold_anchor")
    hold_anchor = (_HoldAnchor(ha.get("release_below") if isinstance(ha, dict) else None)
                   if ha else contextlib.nullcontext())
    gr = arm.get("global_ref")
    global_ref = (_GlobalContextRef(vae, SOURCE_MP4, gr.get("scale", 0.25))
                  if gr else contextlib.nullcontext())
    chl = arm.get("clean_head_label")
    clean_label = (_CleanHeadLabel(
        model, video.video_latent_t(video.align_frame_count(args.frames)),
        chl["frozen_latent_frames"]) if chl else contextlib.nullcontext())
    fh = arm.get("frozen_head")
    frozen_head = (_FrozenHead(OUTPUT_DIR / fh["mp4"], fh["arm_start"], fh["sources"])
                   if fh else contextlib.nullcontext())
    vlp = arm.get("vl_prepend")
    vl_prepend = (_VLPrepend(OUTPUT_DIR / vlp["mp4"], vlp["arm_start"], vlp["sources"], start)
                  if vlp else contextlib.nullcontext())
    with _Probe() as probe, keyframe_probe as kfp, vl_prepend, handoff, frozen_head, \
            noise_offset, hold_anchor, clean_label, global_ref, \
            ab_models.VramProbe() as vram, torch.inference_mode():
        (refined,) = node.refine(
            image=frames, model=model, clip=clip, vae=vae,
            seed=args.seed, sampler_name=SAMPLER, scheduler=SCHEDULER,
            steps=STEPS, denoise=arm.get("denoise", DENOISE), upscale_by=UPSCALE_BY,
            max_tile_width=max_tile_w, max_tile_height=max_tile_h,
            context_anchor=anchor, context_overlap=overlap,
            upscale_model=None, av_latent=None)
    print(f"refine: {vram}", flush=True)
    if isinstance(kfp, _KeyframeProbe):
        print(f"keyframe probe: attached {len(kfp.shapes)} block(s), latent shapes {set(kfp.shapes)}",
              flush=True)

    del model, clip
    ab_models.free_gpu()

    settings = {
        "source": SOURCE_MP4.name, "start": start, "frames": args.frames,
        "unet": args.unet, "clip": poc.CLIP_NAME, "vae": poc.VAE_NAME,
        "seed": args.seed, "sampler": SAMPLER, "scheduler": SCHEDULER, "steps": STEPS,
        "denoise": arm.get("denoise", DENOISE), "upscale_by": UPSCALE_BY,
        "max_tile_width": max_tile_w, "max_tile_height": max_tile_h,
        "context_anchor": anchor, "context_overlap": overlap,
        "conditioning": "vl-slice", "av_latent": "none (silent)", "arm": args.only,
        "comfyui_version": poc.ab_env.version(poc.ROOT),
    }

    worst_off, worst_swing = summarize(probe.rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / f"SEAM_{label}_probe.json").write_text(
        json.dumps({"settings": settings, "rows": probe.rows}, indent=1), encoding="utf-8")

    refined = refined.to(torch.float16)
    picks = (0, refined.shape[0] // 2, refined.shape[0] - 1)
    for index in picks:
        ab_models.save_png(OUTPUT_DIR / f"SEAM_{label}_f{index:03d}.png",
                           refined[index:index + 1], settings)
    if not args.no_video:
        write_mp4(OUTPUT_DIR / f"SEAM_{label}.mp4", refined, crf=10)
        write_mp4(OUTPUT_DIR / f"SEAM_{label}_crf23.mp4", refined, crf=23)
    print(f"arm {label} done: offset {worst_off:.2f}/255 swing {worst_swing:.2f}/255",
          flush=True)


if __name__ == "__main__":
    main()
