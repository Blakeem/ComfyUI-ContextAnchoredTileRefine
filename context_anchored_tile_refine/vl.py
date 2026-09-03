"""Vision conditioning: the VL nodes' per-tile positive, built from the image with no text.

Each tile's positive concatenates the rows of two pure vision encodes through the workflow's
vision-language text encoder (Krea 2's Qwen3-VL):

    crop rows    the tile's own crop, resampled to `crop_tokens` x 1024 px and encoded ALONE,
                 so every cell the tower sees is the tile's own. One encode per tile.
    canvas rows  the tile's row slice of ONE encode of the entire image, resampled so a
                 tile's share of it holds about `canvas_tokens` cells (canvas_budget_pixels),
                 capped at PICTURE_CAP_MEGAPIXELS. One encode per picture.

Both counts come from the settings file's [vision] table (captions.load_settings), so a tile
gets the same rows from each source at every image size and tile count. Settled by the
owner's block A/B of 2026-09-02 (TESTS.md test 10, `tests-AB/run_ab_tile_phantom.py`): a cell
carries the picture it was encoded in, so the canvas slice of a flat sky tile grows a copy of
the image's salient object (a tower in the clouds), larger the more canvas rows it gets; the
crop rows of that tile hold only sky and cancel the demand. About 100 crop rows do that
without redrawing a content tile, where 200 swirl the tile's own subject and 768 gouge it. A
3x3 neighborhood window was tried between the two and fails like the canvas slice.

Why vision rows replace the prompt (A/B-settled, AB26-AB36): vision rows are positionally
exact and demand-free, while text object names act as per-tile re-instantiation demands (a
"blood-red moon" in the prompt regrows a moon inside every tile). Krea 2's DiT gives every
conditioning row RoPE position 0, so the rows are a bag and the block order below carries
nothing (measured bit-exact under a shuffle).

Module scope is torch-only; comfy is imported lazily inside functions (the same contract as
sampling.py, pinned by a subprocess test).
"""
import math
from typing import NamedTuple

import torch

# Krea 2's conditioning template (comfy/text_encoders/krea2.py KREA2_TEMPLATE), passed
# explicitly: with images attached and no template the tokenizer picks qwen3vl's IMAGE
# text-gen template, whose prefix survives Krea 2's template-strip logic and shifts the
# row layout sliced below.
KREA2_TEMPLATE = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

# Expands to one <|image_pad|> token per merged patch of the attached image.
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

# One merged patch covers this many encode-side pixels (patch 16 x spatial merge 2), so a
# picture of P pixels is P / PIXELS_PER_TOKEN vision tokens at any size.
MERGED_CELL = 32
PIXELS_PER_TOKEN = MERGED_CELL * MERGED_CELL

# The most picture one encode reads. The owner measured the VLM breaking down past 2 MP
# (VLMAnchoredRemix), and the tower's own position table is native at 768x768 px with
# everything past it interpolated (fast_pos_embed_interpolate), so spatial precision softens
# as the stretch grows. The canvas budget is capped here and a crop budget cannot reach it.
PICTURE_CAP_MEGAPIXELS = 2.0
PICTURE_CAP_PIXELS = round(PICTURE_CAP_MEGAPIXELS * 1_000_000)
MAX_VISION_TOKENS = PICTURE_CAP_PIXELS // PIXELS_PER_TOKEN


class Encode(NamedTuple):
    """One picture through the tower: the encode, its merged-cell grid, the row count the
    slice layout asserts, and the resampled copy (the caption surface's tokenizer probe)."""

    encoded: list
    enc_h: int
    enc_w: int
    expected_seq: int
    copy: torch.Tensor

    @property
    def n_rows(self):
        return (self.enc_h // MERGED_CELL) * (self.enc_w // MERGED_CELL)


def canvas_budget_pixels(tiles, source_h, source_w, canvas_tokens):
    # The canvas sample size that hands a tile about `canvas_tokens` cells: a tile's slice is
    # its crop's share of the picture, so the picture is sized off the MEAN crop area and the
    # sample grows with the tile count. Rows still vary by tile (edge tiles are smaller and
    # boundary cells are shared). Past the cap a tile gets fewer rows than asked; the crop
    # rows never depend on this. Pure stdlib math, unit-tested without torch.
    if not tiles:
        raise ValueError("canvas_budget_pixels needs at least one tile")
    mean_crop = sum((tile.crop_rect.x1 - tile.crop_rect.x0) * (tile.crop_rect.y1 - tile.crop_rect.y0)
                    for tile in tiles) / len(tiles)
    budget = canvas_tokens * PIXELS_PER_TOKEN * (source_h * source_w) / mean_crop
    return min(round(budget), PICTURE_CAP_PIXELS)


def resample_picture(source, budget):
    # Aspect-preserved area resample of one picture to `budget` pixels, snapped to
    # /MERGED_CELL so process_qwen2vl_images' own rounding (factor = patch 16 x merge 2) is
    # an identity and the merged-patch grid is exactly (h/32, w/32). This copy is
    # conditioning-side only — the sampled tiles are never resampled (prime directive 1).
    import comfy.utils

    samples = source.movedim(-1, 1)
    scale = math.sqrt(budget / (samples.shape[3] * samples.shape[2]))
    width = max(MERGED_CELL, round(samples.shape[3] * scale / MERGED_CELL) * MERGED_CELL)
    height = max(MERGED_CELL, round(samples.shape[2] * scale / MERGED_CELL) * MERGED_CELL)
    resampled = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    return resampled.movedim(1, -1)[:, :, :, :3], height, width


def crop_picture(source, crop, offset_x=0, offset_y=0):
    # The tile's crop cut from the encode source in that source's frame and clamped to it:
    # on the mask path the tiles index the bbox crop while the source is the FULL image, and
    # a canvas padded to /8 lets a tile overreach the image by up to 7 px.
    source_h, source_w = int(source.shape[1]), int(source.shape[2])
    x0, y0 = max(0, crop.x0 + offset_x), max(0, crop.y0 + offset_y)
    x1, y1 = min(source_w, crop.x1 + offset_x), min(source_h, crop.y1 + offset_y)
    return source[:, y0:y1, x0:x1, :]


def slice_indices(crop, canvas_h, canvas_w, enc_h, enc_w, expected_seq, offset_x=0, offset_y=0):
    # Pure index math (stdlib only, unit-tested without torch). The stripped encode's
    # layout is [0]=vision_start, [1..N]=grid rows in raster order, [N+1]=vision_end,
    # [N+2..]=template tail; raster order is pinned by core's patchify permute
    # (qwen_vl.py) — cell (r, c) sits at row r*grid_w + c. The tile rect maps to the
    # merged-cell range by intersection, keeping partly-covered boundary cells so
    # neighboring tiles share them — the row-space analogue of the overlap band.
    # `canvas_h`/`canvas_w` are the ENCODED picture's own pixel size and the offsets shift a
    # tile rect into that picture's frame: the bbox origin on the mask path, where tiles
    # index the bbox crop and the canvas is the FULL image. `expected_seq` of n_rows + 2
    # leaves the trailing tail range empty.
    grid_h, grid_w = enc_h // MERGED_CELL, enc_w // MERGED_CELL
    n_rows = grid_h * grid_w
    cx0 = max(0, math.floor((crop.x0 + offset_x) * enc_w / canvas_w / MERGED_CELL))
    cx1 = min(grid_w, math.ceil((crop.x1 + offset_x) * enc_w / canvas_w / MERGED_CELL))
    cy0 = max(0, math.floor((crop.y0 + offset_y) * enc_h / canvas_h / MERGED_CELL))
    cy1 = min(grid_h, math.ceil((crop.y1 + offset_y) * enc_h / canvas_h / MERGED_CELL))
    rows = [1 + r * grid_w + c for r in range(cy0, cy1) for c in range(cx0, cx1)]
    return [0, *rows, 1 + n_rows, *range(1 + n_rows + 1, expected_seq)]


def _encode_one(clip, picture_copy, grid_h, grid_w):
    # ONE pure-vision encode of one resampled picture. The expected stripped length is
    # derived from the token stream (vision_start + N grid rows + vision_end + tail) and
    # asserted against the encoder's output, so a core layout change fails fast instead
    # of silently scrambling every slice.
    n_rows = grid_h * grid_w
    try:
        tokens = clip.tokenize(VISION_BLOCK, images=[picture_copy], llama_template=KREA2_TEMPLATE)
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


def _encode_batch(clip, picture_copy, grid_h, grid_w):
    # One encode PER BATCH ROW, concatenated on the batch axis. Core's tokenizer attaches
    # images[0] alone (comfy/text_encoders/qwen_vl.py process_qwen2vl_images reads only the
    # first row), so handing over a whole [B,H,W,3] picture would condition EVERY image on
    # row 0's picture. Every row is the same size by construction, so _encode_one's own
    # seq fail-fast covers layout drift and the rows always cat. A batched cond then passes
    # through core's machinery unchanged (CONDRegular.process_cond -> repeat_to_batch_size
    # is an identity when the cond batch already matches the latent's). B=1 takes the single
    # unchanged encode — no cat, byte-for-byte the pre-batch path.
    batch = int(picture_copy.shape[0])
    if batch == 1:
        return _encode_one(clip, picture_copy, grid_h, grid_w)

    per_row = []
    expected_seq = None
    for b in range(batch):
        encoded, expected_seq = _encode_one(clip, picture_copy[b:b + 1], grid_h, grid_w)
        per_row.append(encoded)

    merged = []
    # strict: every row went through the same CLIP on the same token layout, so the encodes
    # carry the same number of cond entries — a mismatch is a contract break, not truncation.
    for entries in zip(*per_row, strict=True):
        # Keep the first row's extras (slice_rows drops the attention_mask per tile);
        # pooled_output is the one extra that is per-image, so cat it as well.
        extras = dict(entries[0][1])
        pooled = extras.get("pooled_output")
        if isinstance(pooled, torch.Tensor):
            extras["pooled_output"] = torch.cat([entry[1]["pooled_output"] for entry in entries], dim=0)
        merged.append([torch.cat([entry[0] for entry in entries], dim=0), extras])
    return merged, expected_seq


def encode_picture(clip, picture, budget):
    """One tower pass over `picture` resampled to `budget` pixels. The pre-pass is one pass
    per picture and one per tile, so a cancel that arrives mid-grid must be seen before each
    pass or it waits out every remaining encode."""
    import comfy.model_management

    comfy.model_management.throw_exception_if_processing_interrupted()
    copy, enc_h, enc_w = resample_picture(picture, budget)
    encoded, expected_seq = _encode_batch(clip, copy, enc_h // MERGED_CELL, enc_w // MERGED_CELL)
    return Encode(encoded, enc_h, enc_w, expected_seq, copy)


def _entry_extras(entry):
    # A slice's extras: the encode's own, minus the full-picture attention mask (its absence
    # means "attend to everything", which is exact for the rows kept).
    extras = dict(entry[1])
    extras.pop("attention_mask", None)
    return extras


def slice_rows(encoded, indices):
    # One encode's rows at `indices`, per cond entry.
    sliced = []
    for entry in encoded:
        index = torch.tensor(indices, device=entry[0].device)
        sliced.append([entry[0].index_select(1, index), _entry_extras(entry)])
    return sliced


def cat_rows(first, second, second_name):
    """`first`'s rows then `second`'s, on the ROW axis, per cond entry. The extras kept are
    `first`'s — `second`'s are dropped — so anything the second carries that the first does
    not would vanish silently. Both are hard errors instead: a stray extra key, and a real
    pooled_output (Krea 2 produces None on every encode, measured; a CLIP that does not is
    outside what this layout settled on)."""
    merged = []
    for head, tail in zip(first, second, strict=True):
        head_extras, tail_extras = _entry_extras(head), _entry_extras(tail)
        stray = sorted(set(tail_extras) - set(head_extras))
        if stray:
            raise RuntimeError(
                f"Context-Anchored Tile Refine (VL): the {second_name} carries conditioning "
                f"extras the rows before it lack ({stray}). Concatenating the rows would drop "
                "them silently.")
        if tail_extras.get("pooled_output") is not None:
            raise RuntimeError(
                f"Context-Anchored Tile Refine (VL): the {second_name} has a real pooled_output. "
                "The concatenated positive keeps the first encode's, so this one would be "
                "dropped silently.")
        rows = torch.cat([head[0], tail[0].to(head[0].device, head[0].dtype)], dim=1)
        merged.append([rows, head_extras])
    return merged


def build_vision_rows(clip, source, tiles, vision, offset_x=0, offset_y=0, with_tail=True):
    """The vision pre-pass: every tile's rows as unconverted cond entries, plus one resampled
    copy for the caption surface's tokenizer probe.

    Per tile the rows are [vision_start][every crop cell][vision_end] out of its own crop's
    encode, then [vision_start][its slice][vision_end] out of the canvas encode, and with
    `with_tail` the template tail once, after the last block; the caption surface passes
    False and brings the tail in with its caption rows. `vision` is the settings file's
    [vision] table (captions.VisionSettings); a source at 0 tokens is skipped, its encode never
    run. `source` is the image the OFFSET tile rects index: the padded canvas itself on the
    whole-image path (offset 0), or the FULL image on the mask path, where tiles index the bbox
    crop and the offsets are the bbox origin, so a region's canvas rows are the entire image's.
    The probe is the smallest copy the run made: any resampled copy gives the same tail
    length, and the caption surface tokenizes it once per tile.
    """
    if vision.canvas_tokens <= 0 and vision.crop_tokens <= 0:
        raise ValueError("VL refine: canvas_tokens and crop_tokens are both 0, so a tile would have no vision rows")
    source_h, source_w = int(source.shape[1]), int(source.shape[2])
    canvas = None
    crop_budget = vision.crop_tokens * PIXELS_PER_TOKEN
    probe = None
    tile_rows = []

    if vision.canvas_tokens > 0:
        budget = canvas_budget_pixels(tiles, source_h, source_w, vision.canvas_tokens)
        canvas = encode_picture(clip, source, budget)

    for tile in tiles:
        blocks = []
        if vision.crop_tokens > 0:
            crop = encode_picture(clip, crop_picture(source, tile.crop_rect, offset_x, offset_y), crop_budget)
            if probe is None:
                probe = crop.copy
            blocks.append((crop, list(range(crop.n_rows + 2))))
        if canvas is not None:
            indices = slice_indices(tile.crop_rect, source_h, source_w, canvas.enc_h, canvas.enc_w,
                                    canvas.n_rows + 2, offset_x, offset_y)
            blocks.append((canvas, indices))
        if with_tail:
            last, indices = blocks[-1]
            blocks[-1] = (last, [*indices, *range(last.n_rows + 2, last.expected_seq)])
        rows = None
        for encode, indices in blocks:
            part = slice_rows(encode.encoded, indices)
            rows = part if rows is None else cat_rows(rows, part, "canvas encode")
        tile_rows.append(rows)
    if probe is None and canvas is not None:
        probe = canvas.copy
    return tile_rows, probe


def build_global_slices(clip, source, tiles, vision, offset_x=0, offset_y=0):
    # The "vision tokens" surface: build_vision_rows converted per tile, nothing else.
    tile_rows, _probe = build_vision_rows(clip, source, tiles, vision, offset_x, offset_y)
    return [_convert(rows) for rows in tile_rows]


def _convert(cond_list):
    import comfy.sampler_helpers

    return comfy.sampler_helpers.convert_cond(cond_list)
