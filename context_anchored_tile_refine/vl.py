"""Global-slice vision conditioning: the VL node's per-tile positive.

The whole canvas is area-resampled ONCE to GLOBAL_SLICE_BUDGET pixels and run through
the workflow's vision-language text encoder (Krea 2's Qwen3-VL) as a pure vision
encode — no text. Every tile's positive is then a row slice of that one encode: the
vision-grid cells its crop covers, plus the delimiter and template-tail rows.

Why this replaces the prompt (A/B-settled, AB26-AB36):
- Vision rows are positionally exact and demand-free. Text object names act as per-tile
  re-instantiation demands — a "blood-red moon" in the prompt regrows a moon inside
  every tile — while vision rows only describe what each cell actually holds.
- Because every tile slices the SAME globally-informed encode, the cross-tile story
  stays coherent (gaze, eye color, tone, palette, large structures crossing seams)
  with no per-tile text at all. Independent per-tile encodes fixed the phantoms but
  moved the seam problem from pixels to story; text mixes re-admitted phantoms in
  proportion to text volume. The pure slice was the only variant clean on both axes.

Module scope is torch-only; comfy is imported lazily inside functions (the same
contract as sampling.py, pinned by a subprocess test).
"""
import math

import torch

# Krea 2's conditioning template (comfy/text_encoders/krea2.py KREA2_TEMPLATE), passed
# explicitly: with images attached and no template the tokenizer picks qwen3vl's IMAGE
# text-gen template, whose prefix survives Krea 2's template-strip logic and shifts the
# row layout sliced below.
KREA2_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

# Expands to one <|image_pad|> token per merged patch of the attached image.
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

# One merged patch covers this many encode-side pixels (patch 16 x spatial merge 2).
MERGED_CELL = 32

# Encode budget (total pixels, aspect preserved, snapped to /MERGED_CELL). Sized so a
# 2x2 grid keeps the per-tile row density proven sufficient in per-tile-encode A/Bs.
# The tower's 48x48 num_position_embeddings table (QWEN35_VISION_DEFAULTS: patch 16,
# merge 2, 2304 embeddings) is indexed on the UNMERGED 16px patch lattice, not the 32px
# merged-cell lattice this module reasons in, so staying inside it would mean <= 768x768
# px total; at this budget a square canvas is 28x28 merged cells = 56x56 patches and the
# tower interpolates its position embeddings (fast_pos_embed_interpolate) by design.
GLOBAL_SLICE_BUDGET = 768 * 1024


def resample_for_global(source):
    # Aspect-preserved area resample of the whole canvas to GLOBAL_SLICE_BUDGET, snapped
    # to /MERGED_CELL so process_qwen2vl_images' own rounding (factor = patch 16 x merge
    # 2) is an identity and the merged-patch grid is exactly (h/32, w/32). This copy is
    # conditioning-side only — the sampled tiles are never resampled (prime directive 1).
    import comfy.utils

    samples = source.movedim(-1, 1)
    scale = math.sqrt(GLOBAL_SLICE_BUDGET / (samples.shape[3] * samples.shape[2]))
    width = max(MERGED_CELL, round(samples.shape[3] * scale / MERGED_CELL) * MERGED_CELL)
    height = max(MERGED_CELL, round(samples.shape[2] * scale / MERGED_CELL) * MERGED_CELL)
    resampled = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    return resampled.movedim(1, -1)[:, :, :, :3], height, width


def slice_indices(crop, canvas_h, canvas_w, enc_h, enc_w, expected_seq, offset_x=0, offset_y=0):
    # Pure index math (stdlib only, unit-tested without torch). The stripped encode's
    # layout is [0]=vision_start, [1..N]=grid rows in raster order, [N+1]=vision_end,
    # [N+2..]=template tail; raster order is pinned by core's patchify permute
    # (qwen_vl.py) — cell (r, c) sits at row r*grid_w + c. The tile rect maps to the
    # merged-cell range by intersection, keeping partly-covered boundary cells so
    # neighboring tiles share them — the row-space analogue of the overlap band.
    # The offsets place a region-crop's tile rects into the encoded canvas's frame
    # (the mask path encodes the FULL image while tiles index the bbox crop).
    grid_h, grid_w = enc_h // MERGED_CELL, enc_w // MERGED_CELL
    n_rows = grid_h * grid_w
    cx0 = max(0, math.floor((crop.x0 + offset_x) * enc_w / canvas_w / MERGED_CELL))
    cx1 = min(grid_w, math.ceil((crop.x1 + offset_x) * enc_w / canvas_w / MERGED_CELL))
    cy0 = max(0, math.floor((crop.y0 + offset_y) * enc_h / canvas_h / MERGED_CELL))
    cy1 = min(grid_h, math.ceil((crop.y1 + offset_y) * enc_h / canvas_h / MERGED_CELL))
    rows = [1 + r * grid_w + c for r in range(cy0, cy1) for c in range(cx0, cx1)]
    return [0, *rows, 1 + n_rows, *range(1 + n_rows + 1, expected_seq)]


def _encode_canvas(clip, canvas_copy, grid_h, grid_w):
    # ONE pure-vision encode of the resampled canvas. The expected stripped length is
    # derived from the token stream (vision_start + N grid rows + vision_end + tail) and
    # asserted against the encoder's output, so a core layout change fails fast instead
    # of silently scrambling every slice.
    n_rows = grid_h * grid_w
    try:
        tokens = clip.tokenize(VISION_BLOCK, images=[canvas_copy], llama_template=KREA2_TEMPLATE)
    except TypeError as error:
        raise RuntimeError(
            "VL refine: this CLIP's tokenizer does not accept images. The node needs a "
            f"vision-language text encoder (Krea 2 family). ({error})") from error
    ids = [t[0] for t in tokens[next(iter(tokens))][0]]
    pad_pos = next((i for i, v in enumerate(ids) if isinstance(v, dict)), None)
    if pad_pos is None:
        raise RuntimeError(
            "VL refine: the tokenizer produced no image tokens. The node needs a "
            "vision-language text encoder (Krea 2 family).")
    tail_len = len(ids) - (pad_pos + 2)
    expected_seq = 1 + n_rows + 1 + tail_len

    encoded = clip.encode_from_tokens_scheduled(tokens)
    seq = encoded[0][0].shape[1]
    if seq != expected_seq:
        raise RuntimeError(
            f"VL refine: encoded conditioning has {seq} rows, expected {expected_seq} (vision grid {grid_h}x{grid_w} "
            f"+ template tail {tail_len}). The text encoder's template or strip layout does not "
            "match the Krea 2 contract this node slices by.")
    return encoded, expected_seq


def _encode_batch(clip, canvas_copy, grid_h, grid_w):
    # One encode PER BATCH ROW, concatenated on the batch axis. Core's tokenizer attaches
    # images[0] alone (comfy/text_encoders/qwen_vl.py process_qwen2vl_images reads only the
    # first row), so handing over a whole [B,H,W,3] canvas would condition EVERY image on
    # row 0's picture. Every row is the same size by construction, so _encode_canvas' own
    # seq fail-fast covers layout drift and the rows always cat. A batched cond then passes
    # through core's machinery unchanged (CONDRegular.process_cond -> repeat_to_batch_size
    # is an identity when the cond batch already matches the latent's). B=1 takes the single
    # unchanged encode — no cat, byte-for-byte the pre-batch path.
    batch = int(canvas_copy.shape[0])
    if batch == 1:
        return _encode_canvas(clip, canvas_copy, grid_h, grid_w)

    per_row = []
    expected_seq = None
    for b in range(batch):
        encoded, expected_seq = _encode_canvas(clip, canvas_copy[b:b + 1], grid_h, grid_w)
        per_row.append(encoded)

    merged = []
    # strict: every row went through the same CLIP on the same token layout, so the encodes
    # carry the same number of cond entries — a mismatch is a contract break, not truncation.
    for entries in zip(*per_row, strict=True):
        # Keep the first row's extras (build_global_slices drops the canvas attention_mask
        # per tile); pooled_output is the one extra that is per-image, so cat it as well.
        extras = dict(entries[0][1])
        pooled = extras.get("pooled_output")
        if isinstance(pooled, torch.Tensor):
            extras["pooled_output"] = torch.cat([entry[1]["pooled_output"] for entry in entries], dim=0)
        merged.append([torch.cat([entry[0] for entry in entries], dim=0), extras])
    return merged, expected_seq


def build_global_slices(clip, source, tiles, offset_x=0, offset_y=0):
    # Pre-pass for the tile loop: encode once per image, slice per tile. `source` is the canvas
    # the OFFSET tile rects index: the padded canvas itself on the whole-image path
    # (offset 0), or the FULL image on the mask path, where tiles index the bbox crop
    # and the offsets are the bbox origin — the region's tiles then slice their true
    # place in the whole image's encode, keeping the conditioning globally informed.
    canvas_copy, enc_h, enc_w = resample_for_global(source)
    grid_h, grid_w = enc_h // MERGED_CELL, enc_w // MERGED_CELL
    encoded, expected_seq = _encode_batch(clip, canvas_copy, grid_h, grid_w)
    canvas_h, canvas_w = int(source.shape[1]), int(source.shape[2])

    tile_positives = []
    for tile in tiles:
        indices = slice_indices(tile.crop_rect, canvas_h, canvas_w, enc_h, enc_w, expected_seq, offset_x, offset_y)
        sliced = []
        for entry in encoded:
            tensor, extras = entry[0], dict(entry[1])
            # A full-canvas attention mask is wrong for a slice; drop it (absence means
            # "attend to everything", which is exact for the rows kept).
            extras.pop("attention_mask", None)
            index = torch.tensor(indices, device=tensor.device)
            sliced.append([tensor.index_select(1, index), extras])
        tile_positives.append(_convert(sliced))
    return tile_positives


def _convert(cond_list):
    import comfy.sampler_helpers

    return comfy.sampler_helpers.convert_cond(cond_list)
