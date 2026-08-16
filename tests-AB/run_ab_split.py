"""Caption-encode-context A/B: what should the caption tokens READ while being encoded?

PRE-PORT HARNESS, PINNED TO COMMIT 0dfd1e4. Not runnable against the current package: this
file calls `captions._encode_slice_caption`, which was DELETED on 2026-08-16 when arm 2's
text-cat surface was promoted into `captions.build_slice_caption_conds`. Kept verbatim as
the frozen record of the campaign that decided that (roadmap decision — tests-AB is a
record, not a maintained suite).

Three arms of the `vision tokens and captions` surface, identical everywhere except the
ONE thing under test — the image context present in the encoder call that produces each
tile's caption rows. Designed in docs/vl-conditioning-encode-cost.md (sections 8/10/11)
plus the tile-grounded variant proposed by the owner 2026-08-14.

    1-cap-whole-image   SHIPPED design: the caption rides inside a whole-canvas vision
                        encode, one encode per tile (captions.build_slice_caption_conds,
                        unpatched). Caption rows are globally contextualized.
    2-cap-no-image      plain split: vision rows sliced from ONE pure-vision global
                        encode; the caption encoded TEXT-ONLY (exactly CLIPTextEncode);
                        rows concatenated. Caption rows are image-blind.
    3-cap-tile-only     local-grounded split: same sliced global vision rows; the
                        caption encoded WITH the tile's own 384^2 crop attached (the
                        very pixels the caption was written from); its crop vision rows
                        dropped, caption+tail rows concatenated. Caption rows are
                        locally grounded.
    4-cap-grey-image    RoPE-position control (opus review finding): the caption encoded
                        with a canvas-sized FLAT GREY image — arm 1's exact token count
                        and stream positions, zero image content. Separates "captions
                        need the image" from "captions need to sit late in the stream"
                        when reading arm 2 against arm 1.

Round 2 (2026-08-14, after the owner judged 1-4: arm 2 wins content and is the only
moonless arm, arm 1 wins seams — same mechanism, the whole-canvas grounding leak). Three
combination arms targeting "arm 2's content with arm 1's seams":

    5-cap-edge-blend    masked dual positive per tile: the tile's EDGE BAND (within
                        EDGE_BAND px of any internal side, feathered over EDGE_FEATHER)
                        denoises under arm 1's positive, the INTERIOR under arm 2's.
                        Core's own cond-mask compositing (samplers.py get_area_and_mult /
                        calc_cond_batch: predictions are mask-weight-averaged; pixel-space
                        masks are bilinear-resized to the tile latent by
                        resolve_areas_and_cond_masks_multidim). Costs one extra cond
                        evaluation per step.
    6-cap-lerp50        caption rows = 0.5*whole-image-encoded + 0.5*text-only-encoded,
                        row-aligned (same text, same tail_len — verified by the probe),
                        cat'd after the sliced vision rows. Tests whether embedding-space
                        interpolation blends the behaviours or melts
                        (vl-strength-scaling-destructive is the adjacent caution).
    7-cap-noimg-lead1   arm 2 unchanged + the ring's FULL lead curve (lead1, no release —
                        R2's curve, previously judged zero-seam but "a little smooth").
                        Arm 2 has surplus detail-building from hungry captions, so it can
                        afford lead1's smoothness; zero new conditioning machinery.

Round 3 (2026-08-14, after the owner judged arm 5 the winner — both issues solved — but
1.6x slow). Two cost-reduction variants, both switching at the run's step midpoint via RAW
sigma thresholds ('timestep_start'/'timestep_end' cond keys, compared against sigma in
samplers.py get_area_and_mult). Raw sigmas on purpose: the percent keys address the FULL
model schedule, so a run-local 50% under denoise 0.5 would be full-schedule 75% — a cut
computed from the run's own sigmas cannot misplace the switch.

    8-cap-edge-gate50   arm 5's masked dual positive for the FIRST half of the steps
                        (structure — where both the seam and the moon are decided), then
                        arm 2 alone, unmasked, for the second half (texture). Spatial
                        split preserved where it matters; ~1.3x instead of 1.6x.
    9-cap-time-switch50 the owner's temporal-split hypothesis: arm 1's conditioning over
                        the WHOLE tile for the first half, arm 2's for the second, no
                        masks — cost 1.0x. Mechanism predicts the moon returns (it is an
                        early-step structure, laid down while arm 1 steers); one render
                        falsifies that prediction for free.

Round 4 (2026-08-14, after the owner judged round 3: arm 8 matches arm 5; arm 9's seam is
WORSE — the spatial split is load-bearing, a time split divides the axis the two problems
share). Two arms walking toward single-pass cost:

    10-cap-edge-gate25  arm 8 with the dual window at the first QUARTER of the steps.
                        The cut fraction is a continuous knob (CUT_FRACTIONS); walking it
                        earlier until the seam reappears finds the cost floor of the
                        mechanism that is known to work. ~1.12x.
    11-cap-seam-noise   the owner's filler hypothesis, single pass at 1.0x: arm 4's
                        structure (real vision rows + caption grounded in a canvas-sized
                        image) with the grey replaced by a COLLAGE — real canvas within
                        (context_anchor + context_overlap) of every internal seam line,
                        seeded gaussian noise elsewhere, assembled at ENCODE resolution
                        (resampling noise would average it back to arm 4's grey). Arm 4
                        proved filler LEAKS into the caption rows; this measures whether
                        noise's leak is more benign than grey's, and whether the real
                        seam bands carry enough shared context to hold the seam. The
                        collage is saved next to the render as *-grounding.png.

Every arm is the production path end to end: refine_image(vl_clip=..., anchor_type=
"leading", vlm_method="vision tokens and captions") — Lead ring, per-tile vision rows,
settled caption instruction (SETTLED_POSITION_INSTRUCTION, thinking=True, strip+clean).
The captions are generated ONCE and cached, then injected into all arms via a
captions.generate_tile_captions patch, so every arm conditions on byte-identical text.
Arms 2/3 patch captions.build_slice_caption_conds — the same call-time seam sampling.py
resolves (the anchor-ring A/B's pattern) — so nothing in the node package changes.

Row-layout invariant (probe_split_encode.py, 2026-08-14): a text-only encode of a
caption yields exactly the caption+template-tail rows of the in-stream layout (head
already stripped), and vision rows are caption-independent bit-for-bit at matched
length. Every arm's tile positive therefore has the same row roles:
[vis_start][tile's canvas cells][vis_end][caption rows][ONE template tail].

Scene: the 00676 woman portrait, 768x1024 base -> 3x -> 2304x3072, 2x2 grid (4 tiles),
seed 42, denoise 0.50, 28 steps, cfg 3.5, exp_heun_2_x0_sde / sgm_uniform, anchor 256,
overlap 32. Base pixels, upscale (cache shared with run_ab_krea2.py), captions, noise,
sigmas, negative and tile grid are held identical across arms.

    <venv-python> tests-AB/run_ab_split.py --list
    <venv-python> tests-AB/run_ab_split.py --only 1-cap-whole-image   # ONE arm per process

Outputs: output\\AB-Test-Images\\AB_00676-d050__{arm}.png, full image only, settings and
captions in the `catr_ab` tEXt chunk.
"""
import argparse
import contextlib
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_env
import ab_models
import run_ab_krea2 as krea2_run  # sets COMFYUI_ROOT at import, before bootstrap() runs
from run_ab_krea2 import (
    BASE_PNG,
    CLIP_NAME,
    CLIP_TYPE,
    NEGATIVE_PROMPT,
    UNET_NAME,
    UPSCALE_MODEL_NAME,
    VAE_NAME,
)

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-Images")
CACHE_DIR = Path(__file__).resolve().parent / "cache"
RUN_TAG = "00676-d050"


# run_ab_krea2's reference settings with the owner's requested denoise 0.50. ONE shared
# object, not a copy: stage_upscale/solve_tiles read krea2_run.REFINE, so a copied class
# could silently diverge on upscale_by (nothing downstream would catch that one).
REFINE = dataclasses.replace(krea2_run.REFINE, denoise=0.50)

ARMS = {
    "1-cap-whole-image": "SHIPPED: caption encoded inside the whole-canvas vision encode",
    "2-cap-no-image": "plain split: caption encoded text-only, cat'd after sliced vision rows",
    "3-cap-tile-only": "local split: caption encoded with the tile's own crop, cat'd the same way",
    "4-cap-grey-image": "control: caption encoded with a flat grey canvas — arm 1's token "
                        "count and positions, zero image content (isolates RoPE position)",
    "5-cap-edge-blend": "masked dual positive: seam edges follow arm 1's conditioning, "
                        "the tile interior follows arm 2's",
    "6-cap-lerp50": "caption rows = 50/50 lerp of the whole-image and text-only encodes",
    "7-cap-noimg-lead1": "arm 2's conditioning + the ring's full lead curve (no release)",
    "8-cap-edge-gate50": "arm 5's masked dual positive for the first half of the steps, "
                         "then arm 2 alone (spatial split kept, ~1.3x cost)",
    "9-cap-time-switch50": "arm 1's conditioning whole-tile for the first half of the "
                           "steps, then arm 2's — no masks, 1.0x cost",
    "10-cap-edge-gate25": "arm 8 with the dual window shrunk to the first QUARTER of the "
                          "steps (~1.12x cost) — walking the gate toward single-pass",
    "11-cap-seam-noise": "single pass: caption grounded in a collage — REAL canvas within "
                         "anchor+overlap of every seam, seeded noise elsewhere (owner's "
                         "filler hypothesis; arm 4 with noise instead of grey)",
}

# Edge-blend geometry, in PIXELS of the tile crop. The edge band spans the frozen anchor
# ring (256, where the cond weighting is moot — those pixels are not diffused), the
# overlap band (32, where the seam is decided) and 64px of the tile's own core, then
# feathers to the interior conditioning over another 128px. The two masks sum to exactly
# 1 everywhere, so calc_cond_batch's mult-normalized average has no coverage holes.
EDGE_BAND = 352
EDGE_FEATHER = 128


def output_path(arm):
    return OUTPUT_DIR / f"AB_{RUN_TAG}__{arm}.png"


# run_ab_krea2's helpers, shared rather than copied.
_rects = krea2_run._rects
_file_sha = krea2_run._file_sha


# ------------------------------------------------------------------ captions (shared)

def load_or_generate_captions(clip, padded, tiles, force):
    """The production caption pre-pass, run ONCE and cached: the settled instruction,
    thinking=True, strip_thinking + clean_caption — exactly what refine_image would do
    for `vision tokens and captions`. Cached so all three arms condition on the same
    byte-identical text (and reruns skip ~4 clip.generate calls)."""
    from context_anchored_tile_refine import captions

    instruction, max_length = captions.CAPTION_INSTRUCTIONS[captions.VLM_METHOD_VISION_CAPTIONS]
    key = {"base_sha": _file_sha(BASE_PNG), "rects": _rects(tiles), "instruction": instruction,
           "max_length": max_length, "clip": CLIP_NAME, "surface": "settled-position"}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    cached = CACHE_DIR / f"krea2-00676_split_captions_{digest}.json"

    if cached.is_file() and not force:
        tile_captions = json.loads(cached.read_text(encoding="utf-8"))["captions"]
        print(f"[caption] cache hit  {cached.name}")
    else:
        with torch.inference_mode():
            tile_captions = captions.generate_tile_captions(clip, padded, tiles, instruction, max_length)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(dict(key, captions=tile_captions), indent=1), encoding="utf-8")
        print(f"[caption] generated and cached  {cached.name}")
    for tile_idx, row_captions in enumerate(tile_captions):
        print(f"[caption] tile {tile_idx}: {row_captions[0]}")
    return tile_captions


@contextlib.contextmanager
def inject_captions(tile_captions, tiles):
    """Swap captions.generate_tile_captions for a hook returning the cached text, so
    every arm (including the unpatched shipped arm) reads identical captions. The tile
    layout is re-asserted — a divergence would silently misassign captions."""
    from context_anchored_tile_refine import captions

    expected = _rects(tiles)

    def hook(clip, source, hook_tiles, instruction, max_length, batch_size=1, batch_index=0):
        if _rects(hook_tiles) != expected:
            raise RuntimeError("tile layout diverged between caption pre-pass and refine")
        return tile_captions

    original = captions.generate_tile_captions
    captions.generate_tile_captions = hook
    try:
        yield
    finally:
        captions.generate_tile_captions = original


# ------------------------------------------------------------------ split cond builders

def _entry_extras(entry):
    extras = dict(entry[1])
    extras.pop("attention_mask", None)
    return extras


def _cat_rows(vision_entries, caption_entries):
    """One tile's positive: sliced global vision rows + caption rows, dim 1. Extras come
    from the vision encode (attention_mask dropped, as the shipped slice does); a caption
    encode carrying an extra key the vision encode lacks would be silently lost, so that
    is a hard error."""
    merged = []
    for vis, cap in zip(vision_entries, caption_entries, strict=True):
        vis_extras, cap_extras = _entry_extras(vis), _entry_extras(cap)
        stray = set(cap_extras) - set(vis_extras)
        if stray:
            raise RuntimeError(f"caption encode carries extras the vision encode lacks: {sorted(stray)}")
        if cap_extras.get("pooled_output") is not None:
            raise RuntimeError("caption encode has a real pooled_output; the split would drop it "
                               "(None on this CLIP when this harness was written)")
        merged.append([torch.cat([vis[0], cap[0].to(vis[0].device, vis[0].dtype)], dim=1), vis_extras])
    return merged


def _slice_vision_rows(encoded, tile, canvas_h, canvas_w, enc_h, enc_w, n_rows, offset_x, offset_y):
    """[vis_start][tile's cells][vis_end] from the pure global encode — today's slice
    minus its template tail (expected_seq = n_rows + 2 makes slice_indices' tail range
    empty; the ONE template tail comes from the caption encode instead)."""
    from context_anchored_tile_refine import vl

    indices = vl.slice_indices(tile.crop_rect, canvas_h, canvas_w, enc_h, enc_w,
                               n_rows + 2, offset_x, offset_y)
    sliced = []
    for entry in encoded:
        index = torch.tensor(indices, device=entry[0].device)
        sliced.append([entry[0].index_select(1, index), _entry_extras(entry)])
    return sliced


def _caption_tail_len(clip, caption, probe_image):
    """How many rows this caption's stream carries after vision_end (caption + template
    tail) — the layout contract every arm shares. The tokenizer only counts; the image
    just makes the stream well-formed."""
    from context_anchored_tile_refine import captions, vl

    _, tail_len = captions._tokenize_images(clip, vl.VISION_BLOCK + caption, probe_image,
                                            llama_template=vl.KREA2_TEMPLATE)
    return tail_len


def _encode_caption_text_only(clip, caption, expected_rows):
    """Arm 2's operand: the caption as CLIPTextEncode would encode it. Krea 2's template
    strip leaves exactly [caption rows][template tail] (probe_split_encode.py, E5), which
    is asserted against the in-stream layout so drift fails fast instead of scrambling."""
    encoded = clip.encode_from_tokens_scheduled(clip.tokenize(caption))
    seq = int(encoded[0][0].shape[1])
    if seq != expected_rows:
        raise RuntimeError(
            f"text-only caption encode has {seq} rows, expected {expected_rows} "
            "(caption + template tail). The text encoder's strip layout does not match "
            "the Krea 2 contract this split concatenates by.")
    return encoded


def _tail_rows(encoded, tail_len):
    """The LAST tail_len rows of an encode (caption + template tail). Correct for any
    caption-carrying stream because Krea 2's template strip removes a PREFIX only
    (comfy/text_encoders/krea2.py), so the stream always ends at the token stream's end."""
    seq = int(encoded[0][0].shape[1])
    kept = []
    for entry in encoded:
        index = torch.arange(seq - tail_len, seq, device=entry[0].device)
        kept.append([entry[0].index_select(1, index), _entry_extras(entry)])
    return kept


def _encode_caption_with_image(clip, caption, image):
    """Arms 3/4's operand: caption encoded WITH `image` in the stream, then only the rows
    AFTER its vision_end kept (caption + template tail). The stream's own vision rows are
    dropped — the kept vision rows come from the sliced global encode; the image's content
    (tile crop, or flat grey for the position control) reaches the caption rows through
    attention during this encode."""
    from context_anchored_tile_refine import captions, vl

    tokens, tail_len = captions._tokenize_images(clip, vl.VISION_BLOCK + caption, image,
                                                 llama_template=vl.KREA2_TEMPLATE)
    encoded = clip.encode_from_tokens_scheduled(tokens)
    seq = int(encoded[0][0].shape[1])
    if seq - 2 - tail_len <= 0:
        raise RuntimeError(f"caption-with-image encode has {seq} rows for tail {tail_len} — no vision cells")
    return _tail_rows(encoded, tail_len)


def _edge_mask(crop, canvas_h, canvas_w):
    """Soft 0..1 mask over the tile crop, PIXEL space [1, crop_h, crop_w]: 1 within
    EDGE_BAND px of any INTERNAL side (a side another tile continues past), ramping to 0
    over EDGE_FEATHER px. External sides (canvas borders) contribute nothing. None when
    the tile has no internal side (1x1 grid) — the caller then skips the dual positive.
    Core bilinear-resizes this to the tile latent (samplers.py
    resolve_areas_and_cond_masks_multidim)."""
    crop_h, crop_w = crop.y1 - crop.y0, crop.x1 - crop.x0
    far = 1.0e9
    xs = torch.arange(crop_w, dtype=torch.float32)
    ys = torch.arange(crop_h, dtype=torch.float32)
    dist_x = torch.full((crop_w,), far)
    dist_y = torch.full((crop_h,), far)
    if crop.x0 > 0:
        dist_x = torch.minimum(dist_x, xs)
    if crop.x1 < canvas_w:
        dist_x = torch.minimum(dist_x, crop_w - 1 - xs)
    if crop.y0 > 0:
        dist_y = torch.minimum(dist_y, ys)
    if crop.y1 < canvas_h:
        dist_y = torch.minimum(dist_y, crop_h - 1 - ys)
    dist = torch.minimum(dist_y[:, None].expand(crop_h, crop_w),
                         dist_x[None, :].expand(crop_h, crop_w))
    if float(dist.min()) >= far:
        return None
    edge = ((EDGE_BAND + EDGE_FEATHER - dist) / EDGE_FEATHER).clamp(0.0, 1.0)
    return edge[None]


def _masked(entries, mask):
    """Attach a cond mask to every entry, in place. Keys are ConditioningSetMask's;
    convert_cond copies extras verbatim so they reach the sampler."""
    for entry in entries:
        entry[1].update({"mask": mask, "mask_strength": 1.0, "set_area_to_bounds": False})
    return entries


def _seam_noise_canvas(canvas_copy, tiles, canvas_h, canvas_w, offset_x=0, offset_y=0):
    """The owner's filler hypothesis (round 4): a grounding collage at ENCODE resolution —
    REAL canvas content within (context_anchor + context_overlap) of every internal seam
    line (where tile cores meet), seeded gaussian noise everywhere else. Noise is drawn
    HERE, at encode resolution, on a dedicated generator seeded with the refine seed:
    area-resampling noise would average it to the flat grey arm 4 proved toxic, and a
    local generator cannot perturb the run's own sampling noise."""
    enc_h, enc_w = int(canvas_copy.shape[1]), int(canvas_copy.shape[2])
    band = REFINE.context_anchor + REFINE.context_overlap
    xs = torch.arange(enc_w, dtype=torch.float32) * (canvas_w / enc_w)
    ys = torch.arange(enc_h, dtype=torch.float32) * (canvas_h / enc_h)
    dist_x = torch.full((enc_w,), 1.0e9)
    dist_y = torch.full((enc_h,), 1.0e9)
    for tile in tiles:
        for edge in (tile.core.x0 + offset_x, tile.core.x1 + offset_x):
            if 0 < edge < canvas_w:
                dist_x = torch.minimum(dist_x, (xs - edge).abs())
        for edge in (tile.core.y0 + offset_y, tile.core.y1 + offset_y):
            if 0 < edge < canvas_h:
                dist_y = torch.minimum(dist_y, (ys - edge).abs())
    dist = torch.minimum(dist_y[:, None].expand(enc_h, enc_w),
                         dist_x[None, :].expand(enc_h, enc_w))
    # Draw on CPU, then move: a CUDA generator with the same seed yields DIFFERENT numbers,
    # so drawing here keeps the collage byte-identical wherever the canvas lives.
    keep = (dist <= band).to(canvas_copy.dtype)[None, :, :, None].to(canvas_copy.device)
    generator = torch.Generator().manual_seed(REFINE.seed)
    noise = (0.5 + 0.25 * torch.randn((1, enc_h, enc_w, 3), generator=generator)).clamp(0.0, 1.0)
    return canvas_copy[:1] * keep + noise.to(canvas_copy.dtype).to(canvas_copy.device) * (1.0 - keep)


def _time_gated(entries, timestep_start=None, timestep_end=None):
    """Restrict entries to part of the schedule via core's RAW sigma keys (samplers.py
    get_area_and_mult): 'timestep_end' = active while sigma >= value, 'timestep_start' =
    active while sigma <= value. calculate_start_end_timesteps only writes these when the
    percent keys are present, so raw values pass through untouched."""
    for entry in entries:
        if timestep_start is not None:
            entry[1]["timestep_start"] = timestep_start
        if timestep_end is not None:
            entry[1]["timestep_end"] = timestep_end
    return entries


def make_split_builder(mode, cut_sigma=None):
    """A drop-in replacement for captions.build_slice_caption_conds (same signature, same
    return shape). Vision rows always come from ONE pure-vision global encode, sliced per
    tile; `mode` decides the caption rows:
        text-only / tile-crop / grey-image /
        seam-noise                           one caption-row set, concatenated (seam-noise
                                             grounds the caption in the seam-band collage)
        lerp50                               0.5*whole-image rows + 0.5*text-only rows
        edge-blend                           TWO positives per tile — arm 1's full slice
                                             masked to the edge band, arm 2's cat masked
                                             to the interior; core composites predictions
        edge-gate50 / edge-gate25            edge-blend's pair active only while
                                             sigma >= cut_sigma, arm 2 alone after
        time-switch50                        arm 1's positive while sigma >= cut_sigma,
                                             arm 2's after — whole tile, no masks
    Only single-picture runs are supported — refine_image's picture loop guarantees that
    through the node, and the guard makes it explicit."""
    if mode in TIME_GATED_MODES and cut_sigma is None:
        raise RuntimeError(f"mode {mode!r} needs the run's cut_sigma")

    def builder(clip, encode_source, tiles, tile_captions, offset_x=0, offset_y=0):
        from context_anchored_tile_refine import captions, vl

        started = time.perf_counter()
        canvas_copy, enc_h, enc_w = vl.resample_for_global(encode_source)
        grid_h, grid_w = enc_h // vl.MERGED_CELL, enc_w // vl.MERGED_CELL
        n_rows = grid_h * grid_w
        encoded, _seq = vl._encode_canvas(clip, canvas_copy, grid_h, grid_w)
        canvas_h, canvas_w = int(encode_source.shape[1]), int(encode_source.shape[2])

        if any(len(row_captions) != 1 for row_captions in tile_captions):
            raise RuntimeError("split builder supports single-picture runs only")

        whole_cache = {}
        seam_canvas = [None]      # built once for mode "seam-noise", shared by every tile

        def whole_encode(caption, tail_len):
            # Arm-1-style caption-carrying whole-canvas encode, one per distinct caption,
            # with the production seq fail-fast plus the tail cross-check.
            if caption not in whole_cache:
                whole_cache[caption] = captions._encode_slice_caption(clip, canvas_copy, caption, n_rows)
            whole_encoded, whole_seq = whole_cache[caption]
            if whole_seq != n_rows + 2 + tail_len:
                raise RuntimeError(
                    f"whole-image caption encode has {whole_seq} rows, expected "
                    f"{n_rows + 2 + tail_len} for the same caption")
            return whole_encoded, whole_seq

        conds = []
        for tile, row_captions in zip(tiles, tile_captions, strict=True):
            caption = row_captions[0]
            tail_len = _caption_tail_len(clip, caption, canvas_copy)
            vision_entries = _slice_vision_rows(encoded, tile, canvas_h, canvas_w,
                                                enc_h, enc_w, n_rows, offset_x, offset_y)

            if mode == "text-only":
                cap_entries = _encode_caption_text_only(clip, caption, tail_len)
            elif mode == "tile-crop":
                # Tile rects index the tiled canvas; +offset places them in encode_source's
                # frame (the mask path's bbox origin — the same math as vl.slice_indices).
                # Offsets are 0 on this harness's whole-image path.
                crop = tile.crop_rect
                tile_crop = captions.resample_for_vl(
                    encode_source[:1, crop.y0 + offset_y:crop.y1 + offset_y,
                                  crop.x0 + offset_x:crop.x1 + offset_x, :])
                cap_entries = _encode_caption_with_image(clip, caption, tile_crop)
            elif mode == "grey-image":
                # The RoPE-position control: canvas-sized flat grey — arm 1's exact token
                # count and stream positions with zero image content.
                cap_entries = _encode_caption_with_image(clip, caption, torch.full_like(canvas_copy[:1], 0.5))
            elif mode == "seam-noise":
                # Arm 4's structure with the owner's collage instead of grey. Built once,
                # shared by every tile's caption encode; saved so the owner can inspect it.
                if seam_canvas[0] is None:
                    seam_canvas[0] = _seam_noise_canvas(canvas_copy, tiles, canvas_h, canvas_w,
                                                       offset_x, offset_y)
                    grounding_png = OUTPUT_DIR / f"AB_{RUN_TAG}__11-cap-seam-noise-grounding.png"
                    grounding_png.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(ab_models.to_uint8(seam_canvas[0])).save(grounding_png)
                    print(f"[conds]   seam-noise grounding collage saved -> {grounding_png.name}")
                cap_entries = _encode_caption_with_image(clip, caption, seam_canvas[0])
            elif mode == "lerp50":
                whole_encoded, _ = whole_encode(caption, tail_len)
                whole_cap = _tail_rows(whole_encoded, tail_len)
                blind_cap = _encode_caption_text_only(clip, caption, tail_len)
                cap_entries = []
                for whole_entry, blind_entry in zip(whole_cap, blind_cap, strict=True):
                    blend = 0.5 * whole_entry[0] + 0.5 * blind_entry[0].to(whole_entry[0].device, whole_entry[0].dtype)
                    # Norm diagnostic (review INFO-5): a 50/50 average of two distant
                    # embeddings shrinks the norm; if the render melts, this says why.
                    print(f"[conds]   lerp50 norms: whole {float(whole_entry[0].norm()):.1f}  "
                          f"blind {float(blind_entry[0].norm()):.1f}  blend {float(blend.norm()):.1f}")
                    cap_entries.append([blend, _entry_extras(whole_entry)])
            elif mode in ("edge-blend", "edge-gate50", "edge-gate25", "time-switch50"):
                whole_encoded, whole_seq = whole_encode(caption, tail_len)
                indices = vl.slice_indices(tile.crop_rect, canvas_h, canvas_w, enc_h, enc_w,
                                           whole_seq, offset_x, offset_y)
                arm1_entries = captions._slice_rows(whole_encoded, indices)
                blind_cap = _encode_caption_text_only(clip, caption, tail_len)
                arm2_entries = _cat_rows(vision_entries, blind_cap)
                if int(arm1_entries[0][0].shape[1]) != int(arm2_entries[0][0].shape[1]):
                    raise RuntimeError(
                        f"{mode} halves disagree: arm1 {int(arm1_entries[0][0].shape[1])} rows "
                        f"vs arm2 {int(arm2_entries[0][0].shape[1])} — the two slices are not the same cells")
                if mode == "time-switch50":
                    # The temporal split: one entry active per step, cost identical to a
                    # plain arm. Coverage is total at every step by construction.
                    conds.append(vl._convert(_time_gated(arm1_entries, timestep_end=cut_sigma)
                                             + _time_gated(arm2_entries, timestep_start=cut_sigma)))
                    continue
                edge = _edge_mask(tile.crop_rect, canvas_h, canvas_w)
                if edge is None:
                    conds.append(vl._convert(arm2_entries))    # 1x1 grid: no seam to guard
                    continue
                masked_pair = _masked(arm1_entries, edge) + _masked(arm2_entries, 1.0 - edge)
                if mode == "edge-blend":
                    conds.append(vl._convert(masked_pair))
                else:
                    # edge-gate50/25: the dual positive while structure forms, then a fresh
                    # UNMASKED arm-2 entry for the remaining steps — without it the edge
                    # band would have zero active conditioning after the cut.
                    late = _time_gated(_cat_rows(vision_entries, blind_cap), timestep_start=cut_sigma)
                    conds.append(vl._convert(_time_gated(masked_pair, timestep_end=cut_sigma) + late))
                continue
            else:
                raise RuntimeError(f"unknown caption-encode mode {mode!r}")

            if int(cap_entries[0][0].shape[1]) != tail_len:
                raise RuntimeError(
                    f"mode {mode!r} caption rows ({int(cap_entries[0][0].shape[1])}) disagree "
                    f"with the canvas-stream layout ({tail_len}) for the same caption")
            conds.append(vl._convert(_cat_rows(vision_entries, cap_entries)))
        print(f"[conds]   mode '{mode}': {len(tiles)} tile positives in {time.perf_counter() - started:.1f}s")
        return conds

    return builder


@contextlib.contextmanager
def caption_encode_mode(mode, cut_sigma=None):
    """Swap captions.build_slice_caption_conds for the split builder of `mode`
    (None = shipped builder, untouched)."""
    from context_anchored_tile_refine import captions

    if mode is None:
        yield
        return
    original = captions.build_slice_caption_conds
    captions.build_slice_caption_conds = make_split_builder(mode, cut_sigma)
    print(f"[render]  caption-encode mode '{mode}' overrides build_slice_caption_conds"
          + (f" (cut sigma {cut_sigma:.4f})" if cut_sigma is not None else ""))
    try:
        yield
    finally:
        captions.build_slice_caption_conds = original


ARM_MODES = {
    "1-cap-whole-image": None,
    "2-cap-no-image": "text-only",
    "3-cap-tile-only": "tile-crop",
    "4-cap-grey-image": "grey-image",
    "5-cap-edge-blend": "edge-blend",
    "6-cap-lerp50": "lerp50",
    "7-cap-noimg-lead1": "text-only",
    "8-cap-edge-gate50": "edge-gate50",
    "9-cap-time-switch50": "time-switch50",
    "10-cap-edge-gate25": "edge-gate25",
    "11-cap-seam-noise": "seam-noise",
}

# Modes whose builder needs a sigma cut, and where in the run their cut sits: the dual
# (or arm-1) phase covers the first `fraction` of the steps.
CUT_FRACTIONS = {
    "edge-gate50": 0.5,
    "time-switch50": 0.5,
    "edge-gate25": 0.25,
}
TIME_GATED_MODES = tuple(CUT_FRACTIONS)

# arm -> anchor-ring curve override. None = the SHIPPED curve (lead1r15). "lead1" is R2's
# full lead with no release, swapped at the documented single seam
# (sampling.anchor_ring_factor — the same pattern as run_ab_krea2.ring_schedule).
ARM_RINGS = {
    "7-cap-noimg-lead1": "lead1",
}


@contextlib.contextmanager
def ring_override(kind):
    from context_anchored_tile_refine import sampling

    if kind is None:
        yield
        return
    if kind != "lead1":
        raise RuntimeError(f"unknown ring curve {kind!r}")

    def lead1(r):
        return min(1.0, max(0.0, r))

    saved = sampling.anchor_ring_factor
    sampling.anchor_ring_factor = lead1
    print("[render]  ring curve 'lead1' overrides the shipped anchor_ring_factor")
    try:
        yield
    finally:
        sampling.anchor_ring_factor = saved


# ------------------------------------------------------------------ render

def build_settings(arm, tile_captions, cut_sigma=None):
    from context_anchored_tile_refine import captions

    instruction, max_length = captions.CAPTION_INSTRUCTIONS[captions.VLM_METHOD_VISION_CAPTIONS]
    payload = dataclasses.asdict(REFINE)
    payload.update({
        "run_label": arm,
        "arm": ARMS[arm],
        "caption_encode_context": ARM_MODES[arm] or "whole-image (shipped)",
        "base_png": str(BASE_PNG),
        "base_sha": _file_sha(BASE_PNG),
        "unet": UNET_NAME, "clip": CLIP_NAME, "clip_type": CLIP_TYPE, "vae": VAE_NAME,
        "upscale_model": UPSCALE_MODEL_NAME,
        "guider": "CFGGuider",
        "anchor_type": "leading",
        "ring_curve": ARM_RINGS.get(arm) or "production (lead1r15)",
        "edge_band": EDGE_BAND if ARM_MODES[arm] in ("edge-blend", "edge-gate50", "edge-gate25") else None,
        "edge_feather": EDGE_FEATHER if ARM_MODES[arm] in ("edge-blend", "edge-gate50", "edge-gate25") else None,
        "lerp_weight": 0.5 if ARM_MODES[arm] == "lerp50" else None,
        "gate_cut_sigma": cut_sigma,
        "gate_fraction": CUT_FRACTIONS.get(ARM_MODES[arm]),
        "seam_noise_band": (REFINE.context_anchor + REFINE.context_overlap)
                           if ARM_MODES[arm] == "seam-noise" else None,
        "seam_noise_seed": REFINE.seed if ARM_MODES[arm] == "seam-noise" else None,
        "vlm_method": captions.VLM_METHOD_VISION_CAPTIONS,
        "negative_prompt": NEGATIVE_PROMPT,
        "caption_instruction": instruction,
        "caption_max_length": max_length,
        "tile_captions": [row[0] for row in tile_captions],
        "harness": "tests-AB/run_ab_split.py",
        "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return payload


def render(arm, canvas, tiles, clip, model, vae, negative, empty, tile_captions):
    """One arm end to end through the production path: refine_image with the Lead ring,
    vision rows, and the settled caption surface — only the caption-encode context varies."""
    import comfy.samplers

    from context_anchored_tile_refine import captions, sampling, upscale

    guider = upscale.build_guider(model, empty, negative, REFINE.cfg)
    sigmas = upscale.build_sigmas(model, REFINE.scheduler, REFINE.steps, REFINE.denoise)
    sampler = comfy.samplers.sampler_object(REFINE.sampler)
    noise = upscale.Noise_RandomNoise(REFINE.seed)

    # Time-gated modes switch after CUT_FRACTIONS[mode] of the run's steps. The cut is the
    # midpoint of the two adjacent RUN sigmas, so exactly one side of the gate is active at
    # every model call (sigma never equals the cut) — steps 0..mid-1 early, mid..end late.
    cut_sigma = None
    if ARM_MODES[arm] in TIME_GATED_MODES:
        if len(sigmas) < 3:
            raise RuntimeError(f"time-gated modes need >= 2 steps; got {len(sigmas)} sigmas")
        steps_n = len(sigmas) - 1
        mid = min(steps_n - 1, max(1, round(steps_n * CUT_FRACTIONS[ARM_MODES[arm]])))
        cut_sigma = float((sigmas[mid - 1] + sigmas[mid]) / 2)

    with inject_captions(tile_captions, tiles), caption_encode_mode(ARM_MODES[arm], cut_sigma), \
            ring_override(ARM_RINGS.get(arm)), ab_models.VramProbe() as probe, \
            torch.inference_mode():
        result = sampling.refine_image(
            canvas, guider, sampler, sigmas, vae, noise,
            max_tile_width=REFINE.max_tile_width,
            max_tile_height=REFINE.max_tile_height,
            context_anchor=REFINE.context_anchor,
            context_overlap=REFINE.context_overlap,
            vl_clip=clip,
            anchor_type="leading",
            vlm_method=captions.VLM_METHOD_VISION_CAPTIONS,
        )

    ab_models.require_image_shape(result, canvas.shape[1], canvas.shape[2], "refine " + arm)
    destination = output_path(arm)
    written = ab_models.save_png(destination, result.cpu(), build_settings(arm, tile_captions, cut_sigma))
    print(f"[render]  {arm:<20} {probe}  -> {destination.name} {written[0]}x{written[1]}")


# ------------------------------------------------------------------ main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", action="append", default=[], metavar="ARM",
                        help="render only these arms (repeatable; ONE per process at 3x)")
    parser.add_argument("--force", action="store_true", help="re-render existing outputs")
    parser.add_argument("--force-captions", action="store_true", help="regenerate the cached captions")
    parser.add_argument("--list", action="store_true", help="show the arms and exit")
    args = parser.parse_args(argv)

    unknown = set(args.only) - set(ARMS)
    if unknown:
        raise SystemExit("unknown arm(s): {}".format(", ".join(sorted(unknown))))
    selected = [arm for arm in ARMS if not args.only or arm in args.only]

    if args.list:
        for arm in ARMS:
            marker = "*" if arm in selected else " "
            print(f"{marker} {arm:<20} {ARMS[arm]}  -> {output_path(arm).name}")
        return 0

    root, note = ab_env.bootstrap()
    print(f"[env]     ComfyUI {ab_env.version(root)} at {root}  ({note})")
    print("[env]     torch {}  cuda {}  device {}".format(
        torch.__version__, torch.version.cuda,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))

    base = krea2_run.load_base()
    canvas = krea2_run.stage_upscale(base, force=False)
    padded, tiles = krea2_run.solve_tiles(canvas)

    print(f"[clip]    loading {CLIP_NAME} ({CLIP_TYPE})")
    with ab_models.VramProbe() as probe:
        clip = ab_models.load_clip(CLIP_NAME, CLIP_TYPE)
        with torch.inference_mode():
            from context_anchored_tile_refine import upscale
            negative = ab_models.encode_prompt(clip, NEGATIVE_PROMPT)
            empty = upscale.encode_empty(clip)
        tile_captions = load_or_generate_captions(clip, padded, tiles, args.force_captions)
    print(f"[clip]    conditioning prepared  {probe}")

    print(f"[unet]    loading {UNET_NAME}")
    with ab_models.VramProbe() as probe:
        model = ab_models.load_unet(UNET_NAME)
        vae = ab_models.load_vae(VAE_NAME)
    print(f"[unet]    loaded  {probe}")

    for arm in selected:
        destination = output_path(arm)
        if destination.is_file() and not args.force:
            print(f"[render]  {arm:<20} exists, skipped (--force to redo)")
            continue
        ab_models.clear_cache()
        render(arm, canvas, tiles, clip, model, vae, negative, empty, tile_captions)

    bad = []
    for arm in selected:
        destination = output_path(arm)
        if not destination.is_file():
            bad.append(f"{destination.name} MISSING")
            continue
        width, height = ab_models.png_size(destination)
        verdict = "OK" if (width, height) == (canvas.shape[2], canvas.shape[1]) else "WRONG SIZE"
        if verdict != "OK":
            bad.append(f"{destination.name} is {width}x{height}")
        print(f"[done]    {destination.name:<44} {width:>4}x{height:<4} {verdict}")
    if bad:
        raise SystemExit("[done]    FAILED: " + "; ".join(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "run_ab_split.py: PRE-PORT CAMPAIGN, pinned to commit 0dfd1e4 (see the module docstring)."
        " refine_image no longer accepts anchor_type= and captions._encode_slice_caption was"
        " deleted in 1.6.0. Check out 0dfd1e4 to re-run. Module helpers stay importable (the"
        " active sync harnesses use them)."
    )
