"""Global-slice video conditioning: the VL Video node's per-tile positive (MiniMax H3).

The whole upscaled clip is sampled at 2 fps into core's reference-video presentation and
run ONCE through the H3 text encoder (Qwen3-VL-32B) with an EMPTY prompt. Every tile's
positive is then a row slice of that one encode: the merged-cell rows its crop covers,
plus every literal token row, with `minimax_token_tags` sliced by the same indices.

Why this shape (vl.py's image-side result, re-settled on H3 by the F0b stress run and
the 2x2 factorial):
- ONE shared encode keeps the cross-tile story coherent: every tile reads the same
  globally-informed picture of the whole clip, so structures crossing a seam continue.
- Vision rows are positionally exact and demand-free, while text object names act as
  per-tile re-instantiation demands — the text arm drifted from denoise 0.25 up, the VL
  slice stayed intact 0.18-0.32. Hence the empty prompt, and no prompt input at all.
- The tags ride the SAME indices as the rows because the DiT's adaLN modality run is
  keyed on them (comfy/text_encoders/minimax.py:71-78 token_tags_from_embeds_info):
  a row slice with unsliced tags would mark the wrong positions as video.

Module scope is torch-only; comfy is imported lazily inside functions (the same contract
as vl.py / sampling.py, pinned by a subprocess test).
"""
import math
from dataclasses import dataclass

import torch

# H3's frame rate is a model constant (comfy_extras/nodes_minimax_h3.py:29), so the 2 fps
# conditioning cadence below has no parameter to get wrong.
FPS = 24

# One merged patch covers this many encode-side pixels: Qwen3-VL patch 16 x spatial merge
# 2 (comfy/text_encoders/minimax.py:35 process_video_block defaults). The vision tower
# therefore emits one conditioning row per 32x32 px cell, in raster order per 2-frame block.
MERGED_CELL = 32

# The tokenizer's per-frame pixel cap (comfy/text_encoders/minimax.py:36). Measured on ONE
# frame's area, not the 2-frame block: minimax.py:42 unpacks height/width from the
# [T,H,W,C] frames and minimax.py:47 tests h_bar * w_bar against it.
MAX_ENCODE_PIXELS = 12845056

# VRAM reserve held during the encode; see the comment in encode_global.
ENCODE_RESERVED_VRAM = 8 * 1024 ** 3


@dataclass(frozen=True)
class GlobalEncode:
    """The ONE whole-clip encode every tile slices.

    `layout` is the token stream walked into ("text", row) / ("grid", first_row) entries;
    `enc_h`/`enc_w` are the pixel size the vision tower actually saw, which the tile rects
    are mapped through (identical to the canvas in the normal, unresampled case).
    """
    cond: torch.Tensor
    tags: torch.Tensor
    layout: list
    enc_h: int
    enc_w: int
    extras: dict


def sample_conditioning_frames(frames):
    # Core's reference-video presentation, verbatim (comfy_extras/nodes_minimax_h3.py:
    # 260-264): every FPS//2-th frame, sequential half-second stamps. An ODD pick count is
    # deliberately NOT padded here — the tokenizer repeat-pads to its 2-frame temporal
    # patch itself (comfy/text_encoders/minimax.py:168-170) and core ships odd counts (124
    # frames -> 11 picks), so padding here would hand the encoder a different presentation
    # than the model's own reference path produces.
    sample_idx = list(range(0, int(frames.shape[0]), FPS // 2))
    picks = frames[sample_idx]
    stamps = [i / 2.0 for i in range(len(sample_idx))]
    return picks, stamps


def resample_for_encode(picks):
    # Returns (frames the tower will see, enc_h, enc_w). The tokenizer snaps each frame to
    # round(dim / MERGED_CELL) * MERGED_CELL (comfy/text_encoders/minimax.py:45-46), which
    # is an IDENTITY for the /32-aligned canvas this node feeds it, so under the pixel cap
    # the picks reach the tower untouched and enc_h/enc_w are the canvas dims.
    # Over the cap — reachable only at large targets, e.g. 4x of a ~1 MP base — core would
    # bilinear-resize to a size of its own choosing (minimax.py:48-50). Resampling ONCE
    # here instead keeps the merged-cell grid exactly known and uses an area resample
    # rather than bilinear. Conditioning-side only: sampled tiles are never resampled
    # (prime directive 1).
    import comfy.utils

    height, width = int(picks.shape[1]), int(picks.shape[2])
    enc_h = round(height / MERGED_CELL) * MERGED_CELL
    enc_w = round(width / MERGED_CELL) * MERGED_CELL
    if enc_h * enc_w <= MAX_ENCODE_PIXELS:
        return picks, enc_h, enc_w

    scale = math.sqrt(MAX_ENCODE_PIXELS / (height * width))
    # Floor-snapped like core's own over-cap branch: rounding up could cross the cap again
    # and hand the resize back to the tokenizer after all.
    enc_h = max(MERGED_CELL, math.floor(height * scale / MERGED_CELL) * MERGED_CELL)
    enc_w = max(MERGED_CELL, math.floor(width * scale / MERGED_CELL) * MERGED_CELL)
    samples = picks.movedim(-1, 1)
    resampled = comfy.utils.common_upscale(samples, enc_w, enc_h, "area", "disabled")
    return resampled.movedim(1, -1)[:, :, :, :3], enc_h, enc_w


def token_layout(tokens, rows_per_block):
    # Walk the FIRST tokenizer key's stream (never hardcode "qwen3vl_32b"): a dict entry is
    # a vision block that expands to one row per merged cell, every other entry is one
    # literal token row. That interleave is what the tokenizer builds — "<Video k>: ", then
    # per 2-frame block "<T.T seconds>" + vision_start + block + vision_end
    # (comfy/text_encoders/minimax.py:141-186).
    layout = []
    pos = 0
    for entry in tokens[next(iter(tokens))][0]:
        if isinstance(entry[0], dict):
            layout.append(("grid", pos))
            pos += rows_per_block
        else:
            layout.append(("text", pos))
            pos += 1
    return layout, pos


def encode_global(clip, picks, stamps):
    # ONE video-reference encode of the whole clip, empty prompt. The row layout is derived
    # from the token stream and asserted against the encoder's output, so a core
    # presentation change fails fast instead of silently scrambling every tile's slice.
    import comfy.model_management

    encode_frames, enc_h, enc_w = resample_for_encode(picks)
    rows_per_block = (enc_h // MERGED_CELL) * (enc_w // MERGED_CELL)
    items = [{"type": "video", "data": encode_frames, "timestamps": list(stamps)}]

    try:
        tokens = clip.tokenize("", minimax_ref_items=items)
    except TypeError as error:
        raise RuntimeError(
            "VL video refine: this CLIP's tokenizer does not accept minimax_ref_items. "
            "The node needs the MiniMax H3 CLIP. ({})".format(error)) from error
    layout, total_rows = token_layout(tokens, rows_per_block)
    if not any(kind == "grid" for kind, _ in layout):
        # Core tokenizers swallow unknown kwargs (comfy/sd.py CLIP.tokenize -> **kwargs), so
        # a non-H3 CLIP gets this far and produces a text-only stream.
        raise RuntimeError(
            "VL video refine: the tokenizer produced no video tokens. The node needs the "
            "MiniMax H3 CLIP.")

    # A ~24k-token forward through the 32B encoder needs ~7.5 GiB of activations; with the
    # 14.6 GiB model loaded whole that leaves ~0.6 GiB of margin. Raising the reserve makes
    # comfy stream layers instead, trading encode time for guaranteed headroom — proven
    # necessary at this size (tests-AB/spike_h3_stress.py:216-241). Restored in finally.
    saved_reserve = comfy.model_management.EXTRA_RESERVED_VRAM
    comfy.model_management.EXTRA_RESERVED_VRAM = ENCODE_RESERVED_VRAM
    try:
        encoded = clip.encode_from_tokens_scheduled(tokens)
    finally:
        comfy.model_management.EXTRA_RESERVED_VRAM = saved_reserve

    if len(encoded) != 1:
        raise RuntimeError(
            "VL video refine: the scheduled encode returned {} sections, expected 1. A "
            "hook-scheduled CLIP cannot be sliced per tile.".format(len(encoded)))
    cond, extras = encoded[0][0], dict(encoded[0][1])
    if cond.shape[1] != total_rows:
        raise RuntimeError(
            "VL video refine: encoded conditioning has {} rows, expected {} ({} vision "
            "blocks of {}x{} cells + {} literal token rows). The tokenizer's presentation "
            "does not match the MiniMax H3 contract this node slices by.".format(
                cond.shape[1], total_rows,
                sum(1 for kind, _ in layout if kind == "grid"),
                enc_h // MERGED_CELL, enc_w // MERGED_CELL,
                sum(1 for kind, _ in layout if kind == "text")))
    tags = extras.get("minimax_token_tags")
    if tags is None or int(tags.shape[0]) != total_rows:
        raise RuntimeError(
            "VL video refine: minimax_token_tags is missing or misaligned ({} vs {} rows). "
            "The node needs the MiniMax H3 CLIP.".format(
                None if tags is None else int(tags.shape[0]), total_rows))
    return GlobalEncode(cond, tags, layout, enc_h, enc_w, extras)


def slice_indices(layout, crop, canvas_h, canvas_w, enc_h, enc_w):
    # Pure index math (stdlib only). Every literal token row is kept — the presentation's
    # labels and the vision_start/vision_end delimiters flanking each block. Grid cells are
    # kept by intersection with the tile's crop_rect, so partly-covered boundary cells stay
    # in BOTH neighbors (the row-space analogue of the overlap band). The enc/canvas ratio
    # is an identity unless resample_for_encode fired, where /32 crops give exact x // 32
    # cells. Indices come out ascending: layout is in stream order and a block's cells are
    # raster-ordered inside it.
    grid_h, grid_w = enc_h // MERGED_CELL, enc_w // MERGED_CELL
    cx0 = max(0, math.floor(crop.x0 * enc_w / canvas_w / MERGED_CELL))
    cx1 = min(grid_w, math.ceil(crop.x1 * enc_w / canvas_w / MERGED_CELL))
    cy0 = max(0, math.floor(crop.y0 * enc_h / canvas_h / MERGED_CELL))
    cy1 = min(grid_h, math.ceil(crop.y1 * enc_h / canvas_h / MERGED_CELL))

    indices = []
    for kind, start in layout:
        if kind == "text":
            indices.append(start)
            continue
        for row in range(cy0, cy1):
            base = start + row * grid_w
            indices.extend(range(base + cx0, base + cx1))
    return indices


def slice_rows(pack, crop, canvas_h, canvas_w):
    # The tile's positive: the same rows selected out of cond and tags, ready to be
    # assigned into guider.original_conds["positive"].
    indices = slice_indices(pack.layout, crop, canvas_h, canvas_w, pack.enc_h, pack.enc_w)
    index = torch.tensor(indices, dtype=torch.long)
    extras = dict(pack.extras)
    # A whole-clip attention mask is wrong for a slice; drop it (absence means "attend to
    # everything", which is exact for the rows kept).
    extras.pop("attention_mask", None)
    extras["minimax_token_tags"] = pack.tags.index_select(0, index.to(pack.tags.device))
    return _convert([[pack.cond.index_select(1, index.to(pack.cond.device)), extras]])


def _convert(cond_list):
    import comfy.sampler_helpers

    return comfy.sampler_helpers.convert_cond(cond_list)
