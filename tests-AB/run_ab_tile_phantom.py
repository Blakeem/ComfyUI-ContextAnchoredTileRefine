"""Single-tile and tile-block reproduction of the sky phantom (TESTS.md tests 2 and 10).

The owner's 8K pass 2 (`output/ComfyUI-2x_00787_.png`, vision tokens only from the clean 4K
`ComfyUI-2x_00785_.png`) grew a copy of the Empire State crown in the clouds of tile r0c1.
Rendering the full 24-tile canvas costs about 70 minutes per arm, so this harness refines ONE
tile of that canvas, or the tile and its bordering tiles, through the shipped sync engine and
builds each tile's vision positive with the node's own vl.build_global_slices under a chosen
[vision] table:

  crop-global     the node's method: the tile's own crop at --crop-tokens plus its slice of
                  the entire canvas at --canvas-tokens (the settings file's values by default)
  canvas          the canvas slice alone (crop_tokens 0, the 1.6.1 conditioning)
  crop            the crop alone (canvas_tokens 0, AB26's rejected isolated encode)
  empty           no vision rows at all (the empty-prompt positive)
  text            a plain text prompt (--text)

The retired neighborhood-window arms (window, global-context, two-scale, crop-context, mixed)
and their results are recorded in TESTS.md test 10; the window design they exercised was
removed from the node on 2026-09-02.

What is faithful: the tile's crop rect, its vision rows (the node's own encode, resample and
slice, with the canvas budget derived from the FULL run's layout), the sampler, schedule,
cfg, negative, the source-image ring schedule on the anchor band (through the mask path,
which freezes the 32 px anchor band on every neighbor side), and the initial noise (the full
run's canvas-wide draw at the same seed, sliced to this tile). What a single lane lacks: the
per-step consolidation with neighbor lanes in the overlap bands, and the SDE noise field,
which is drawn at crop size. `--block` adds the consolidation back: the tile and its
bordering tiles run as one sync run over the full canvas' mask path, with the solver forced
to the full run's tile rects, so the center tile keeps its exact crop, noise slice and
neighbors (about 11 minutes per arm at 6 lanes).

Usage (one arm per process, GPU idle):
  python tests-AB/run_ab_tile_phantom.py --arm crop-global --block
  python tests-AB/run_ab_tile_phantom.py --arm crop-global --block --crop-tokens 200
  python tests-AB/run_ab_tile_phantom.py --arm canvas --tile r0c1 --seed 1234
Outputs: output/AB-Test-Images/AB_tilephantom-<tile>__<arm>-d<denoise>-s<seed>[-cfg<cfg>]
[-canvas<N>-crop<N>][-shuffled][-<tag>].png, plus <tile>__source.png (the crop before refine)
once; block mode writes AB_tilephantom-block-<tile>__<arm>-... (the block) and
<tile>__<arm>-block-... (the tile cut from it).
"""
import argparse
import dataclasses
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_env
import ab_models
import run_ab_krea2 as krea2_run  # sets COMFYUI_ROOT at import, before bootstrap()
import run_ab_matrix as matrix_run

Image.MAX_IMAGE_PIXELS = None

SOURCE_4K = Path(r"C:/Users/Blake/Documents/ComfyUI/output/ComfyUI-2x_00785_.png")
OUTPUT_DIR = krea2_run.OUTPUT_DIR
CACHE_DIR = Path(__file__).resolve().parent / "cache" / "tile_phantom"

# The owner's pass 2 widgets, read from 00787's embedded prompt.
UNET_NAME = "krea2SATDirtyrealism_krea2SAT.safetensors"
CLIP_NAME = matrix_run.CLIP_NAME
CLIP_TYPE = matrix_run.CLIP_TYPE
VAE_NAME = matrix_run.VAE_NAME
UPSCALE_MODEL_NAME = "4xFaceUpDAT.safetensors"
NEGATIVE = "blurry, jpeg compression, scribbles, AI generated, hredded cloth"
ARMS = ("crop-global", "canvas", "crop", "empty", "text")
VISION_ARMS = ("crop-global", "canvas", "crop")


@dataclasses.dataclass(frozen=True)
class Settings:
    seed: int = 42
    sampler: str = "dpmpp_2m_sde"
    scheduler: str = "sgm_uniform"
    steps: int = 20
    cfg: float = 3.5
    denoise: float = 0.35
    upscale_by: float = 2.0
    max_tile_width: int = 2048
    max_tile_height: int = 1728
    context_anchor: int = 32
    context_overlap: int = 256
    anchor_source: str = "source image"


# DAT margin around the tile's 4K footprint, so the upscaler's receptive field at the crop
# border sees real neighbors (4x output = 256 px of context).
DAT_MARGIN_4K = 64


NAME_TAG = ""
SHUFFLE = False


def output_path(tile_name, arm, settings):
    cfg = "" if settings.cfg == Settings.cfg else f"-cfg{settings.cfg:g}"
    return OUTPUT_DIR / f"AB_tilephantom-{tile_name}__{arm}-d{settings.denoise:.2f}-s{settings.seed}{cfg}{NAME_TAG}.png"


def load_source(path):
    image = Image.open(path).convert("RGB")
    return torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0)[None]


def lanczos(image, width, height):
    import comfy.utils

    return comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", "disabled").movedim(1, -1)


def solve_layout(width, height, settings):
    from context_anchored_tile_refine import grid

    sx = grid.solve_axis(width, settings.max_tile_width, settings.context_anchor, settings.context_overlap, axis="width")
    sy = grid.solve_axis(height, settings.max_tile_height, settings.context_anchor, settings.context_overlap, axis="height")
    return grid.build_layout(width, height, sx, sy, settings.context_anchor, settings.context_overlap)


# ------------------------------------------------------------------ the tile's positive

def arm_vision(arm, vision):
    """The [vision] table an arm runs under: the run's own for crop-global, one source off
    for the two single-source arms."""
    if arm == "canvas":
        return dataclasses.replace(vision, crop_tokens=0)
    if arm == "crop":
        return dataclasses.replace(vision, canvas_tokens=0)
    return vision


class FullLayoutBudget:
    """vl.canvas_budget_pixels sized off the FULL run's tiles whatever tiles the call names,
    so a lone tile or a block samples the canvas at the size the node would over the whole
    grid. Restored on exit."""

    def __init__(self, tiles):
        self.tiles = tiles
        self.saved = None

    def __enter__(self):
        from context_anchored_tile_refine import vl

        self.saved = vl.canvas_budget_pixels
        real, tiles = self.saved, self.tiles
        vl.canvas_budget_pixels = lambda _tiles, source_h, source_w, canvas_tokens: real(tiles, source_h, source_w, canvas_tokens)
        return self

    def __exit__(self, *exc):
        from context_anchored_tile_refine import vl

        vl.canvas_budget_pixels = self.saved
        return False


def encode_dims(width, height, budget):
    """vl.resample_picture's own snap, for the record without resampling a picture."""
    from context_anchored_tile_refine import vl

    scale = math.sqrt(budget / (width * height))
    snapped_w = max(vl.MERGED_CELL, round(width * scale / vl.MERGED_CELL) * vl.MERGED_CELL)
    snapped_h = max(vl.MERGED_CELL, round(height * scale / vl.MERGED_CELL) * vl.MERGED_CELL)
    return snapped_w, snapped_h


def describe_rows(canvas, tile, vision, layout_tiles, offset_x=0, offset_y=0):
    """The encode sizes and per-source cell counts the node's rows are built from, for the
    record: the same snap and slice math, run without the tower or the resample."""
    from context_anchored_tile_refine import vl

    canvas_h, canvas_w = int(canvas.shape[1]), int(canvas.shape[2])
    info = {}
    if vision.crop_tokens > 0:
        crop = vl.crop_picture(canvas, tile.crop_rect, offset_x, offset_y)
        c_w, c_h = encode_dims(int(crop.shape[2]), int(crop.shape[1]), vision.crop_tokens * vl.PIXELS_PER_TOKEN)
        info.update({"crop": [tile.crop_rect.x0 + offset_x, tile.crop_rect.y0 + offset_y,
                              tile.crop_rect.x1 + offset_x, tile.crop_rect.y1 + offset_y],
                     "crop_encode": [c_w, c_h], "crop_cells": (c_h // vl.MERGED_CELL) * (c_w // vl.MERGED_CELL)})
    if vision.canvas_tokens > 0:
        budget = vl.canvas_budget_pixels(layout_tiles, canvas_h, canvas_w, vision.canvas_tokens)
        g_w, g_h = encode_dims(canvas_w, canvas_h, budget)
        n_g = (g_h // vl.MERGED_CELL) * (g_w // vl.MERGED_CELL)
        indices = vl.slice_indices(tile.crop_rect, canvas_h, canvas_w, g_h, g_w, n_g + 2, offset_x, offset_y)
        info.update({"canvas_budget_mp": round(budget / 1e6, 3), "canvas_encode": [g_w, g_h],
                     "canvas_cells": sum(1 <= i <= n_g for i in indices)})
    return info


def shuffle_rows(positive, seed):
    """The same rows in a random order: identical output proves the DiT reads them as a bag."""
    from context_anchored_tile_refine import vl

    generator = torch.Generator().manual_seed(seed)
    shuffled = []
    for entry in positive:
        tensor = entry["cross_attn"] if isinstance(entry, dict) else entry[0]
        order = torch.randperm(tensor.shape[1], generator=generator).to(tensor.device)
        extras = {k: v for k, v in entry.items() if k != "cross_attn"} if isinstance(entry, dict) else dict(entry[1])
        shuffled.append([tensor.index_select(1, order), extras])
    return vl._convert(shuffled)


def build_positives(arm, clip, canvas, tiles, vision, layout_tiles, text="", offset_x=0, offset_y=0):
    """One positive per tile, in tile order, plus one info dict each. The vision arms run the
    node's builder ONCE over the tile list, so the canvas is encoded once per run as in the
    node; the empty and text arms share one encode across the tiles."""
    from context_anchored_tile_refine import upscale, vl

    if arm == "empty":
        # No vision rows at all: what the model does to the crop on its prior alone.
        positive = vl._convert(upscale.encode_empty(clip))
        return [positive] * len(tiles), [{"positive": "empty prompt"}] * len(tiles)
    if arm == "text":
        positive = vl._convert(ab_models.encode_prompt(clip, text))
        return [positive] * len(tiles), [{"positive": "text", "text": text}] * len(tiles)
    if arm not in VISION_ARMS:
        raise ValueError(arm)
    settings = arm_vision(arm, vision)
    with FullLayoutBudget(layout_tiles):
        positives = vl.build_global_slices(clip, canvas, tiles, settings, offset_x=offset_x, offset_y=offset_y)
    infos = []
    for tile, positive in zip(tiles, positives, strict=True):
        info = describe_rows(canvas, tile, settings, layout_tiles, offset_x, offset_y)
        entry = positive[0]
        info["rows"] = int((entry["cross_attn"] if isinstance(entry, dict) else entry[0]).shape[1])
        info["vision"] = dataclasses.asdict(settings)
        infos.append(info)
    return positives, infos


# ------------------------------------------------------------------ block mode: the tile and its neighbors

def block_tiles(layout, tile):
    return [t for t in layout.tiles if abs(t.col - tile.col) <= 1 and abs(t.row - tile.row) <= 1]


def block_rect(layout, tile):
    """The sampled region for the tile's 3x3 tile block, in canvas px. build_layout places the
    first core at the region origin, so on a left or top side with tiles beyond the block the
    region starts at the edge tile's CORE (its outer ring is dropped) and the center tile's
    rects land exactly on the full run's; the right and bottom sides keep the outer rings,
    which the last tile absorbs as core."""
    from context_anchored_tile_refine import grid

    near = block_tiles(layout, tile)
    x0 = min(t.crop_rect.x0 for t in near)
    y0 = min(t.crop_rect.y0 for t in near)
    if x0 > 0:
        x0 = min(t.core.x0 for t in near)
    if y0 > 0:
        y0 = min(t.core.y0 for t in near)
    return grid.Rect(x0, y0, max(t.crop_rect.x1 for t in near), max(t.crop_rect.y1 for t in near))


def block_layout(layout, tile, block, settings):
    """The layout the engine must build over the block so its tiles are the full run's tiles,
    rect for rect: the same bases, and the last tile absorbing the remainder."""
    from context_anchored_tile_refine import grid

    cols = [t.col for t in layout.tiles if abs(t.col - tile.col) <= 1]
    rows = [t.row for t in layout.tiles if abs(t.row - tile.row) <= 1]
    n_x, n_y = len(set(cols)), len(set(rows))
    r = settings.context_anchor + settings.context_overlap
    width, height = block.x1 - block.x0, block.y1 - block.y0
    sx = grid.AxisSolution(n=n_x, base=layout.sol_x.base, last=width - (n_x - 1) * layout.sol_x.base,
                           overhead=0 if n_x == 1 else r if n_x == 2 else 2 * r, r=r)
    sy = grid.AxisSolution(n=n_y, base=layout.sol_y.base, last=height - (n_y - 1) * layout.sol_y.base,
                           overhead=0 if n_y == 1 else r if n_y == 2 else 2 * r, r=r)
    return grid.build_layout(width, height, sx, sy, settings.context_anchor, settings.context_overlap), sx, sy


class SolverPatch:
    """grid.solve_axis returns the block's forced solutions for the block's own two axes and
    defers to the real solver for anything else. Keyed by axis AND length, so a square block
    cannot hand the row solution to the columns."""

    def __init__(self, block, sx, sy):
        self.forced = {("width", block.x1 - block.x0): sx, ("height", block.y1 - block.y0): sy}
        self.saved = None

    def __enter__(self):
        from context_anchored_tile_refine import grid

        self.saved = grid.solve_axis
        forced, real = self.forced, self.saved

        def solve_axis(L, cap, ctx, overlap=0, multiple=8, axis=None):
            return forced.get((axis, L)) or real(L, cap, ctx, overlap=overlap, multiple=multiple, axis=axis)

        grid.solve_axis = solve_axis
        return self

    def __exit__(self, *exc):
        from context_anchored_tile_refine import grid

        grid.solve_axis = self.saved
        return False


def block_mask(block, canvas_w, canvas_h, settings):
    """1 over the block, 0 on a 32 px band at every block edge that has tiles beyond it. On the
    right and bottom that band is the edge tile's own frozen anchor ring; on a left or top
    edge that starts at a core (block_rect) it freezes 32 px of that edge tile's core, the
    price of the mask path's bbox + context_anchor crop landing on the block exactly."""
    mask = torch.zeros((1, canvas_h, canvas_w), dtype=torch.float32)
    a = settings.context_anchor
    x0 = block.x0 + (a if block.x0 > 0 else 0)
    y0 = block.y0 + (a if block.y0 > 0 else 0)
    x1 = block.x1 - (a if block.x1 < canvas_w else 0)
    y1 = block.y1 - (a if block.y1 < canvas_h else 0)
    mask[:, y0:y1, x0:x1] = 1.0
    return mask


def region_pixels(source_4k, rect, cache_key, settings, use_dat):
    """The canvas pixels of `rect` the full run would have sampled: the owner's pass 2 ran
    4xFaceUpDAT then one lanczos to 2x, so the rect's 4K footprint (plus a margin) goes through
    the same stage. Cached per rect."""
    from context_anchored_tile_refine import upscale

    cached = CACHE_DIR / f"{SOURCE_4K.stem}-{cache_key}-{'dat' if use_dat else 'lanczos'}.png"
    if cached.is_file():
        return load_source(cached)
    scale = settings.upscale_by
    x0 = max(0, int(rect.x0 / scale) - DAT_MARGIN_4K)
    y0 = max(0, int(rect.y0 / scale) - DAT_MARGIN_4K)
    x1 = min(int(source_4k.shape[2]), int(rect.x1 / scale) + DAT_MARGIN_4K)
    y1 = min(int(source_4k.shape[1]), int(rect.y1 / scale) + DAT_MARGIN_4K)
    region = source_4k[:, y0:y1, x0:x1, :]
    if use_dat:
        matrix_run.UPSCALE_MODEL_NAME = UPSCALE_MODEL_NAME
        upscale_model = matrix_run._load_upscale_model()
        with torch.inference_mode():
            upscaled = upscale.prepare_upscaled(region, upscale_model, scale)
    else:
        upscaled = lanczos(region, round((x1 - x0) * scale), round((y1 - y0) * scale))
    ox, oy = rect.x0 - round(x0 * scale), rect.y0 - round(y0 * scale)
    pixels = upscaled[:, oy:oy + (rect.y1 - rect.y0), ox:ox + (rect.x1 - rect.x0), :].contiguous()
    ab_models.require_image_shape(pixels, rect.y1 - rect.y0, rect.x1 - rect.x0, f"region {cache_key}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ab_models.save_png(cached, pixels.cpu(), {"stage": "region crop", "rect": [rect.x0, rect.y0, rect.x1, rect.y1]})
    return pixels


# ------------------------------------------------------------------ the tile's pixels and noise

def region_mask(tile, settings):
    """1 on the core plus overlap, 0 on the anchor band of every neighbor side, so the mask
    path freezes exactly what the full run's lane froze."""
    crop = tile.crop_rect
    h, w = crop.y1 - crop.y0, crop.x1 - crop.x0
    mask = torch.ones((1, h, w), dtype=torch.float32)
    a = settings.context_anchor
    if tile.nb.left:
        mask[:, :, :a] = 0.0
    if tile.nb.right:
        mask[:, :, w - a:] = 0.0
    if tile.nb.top:
        mask[:, :a, :] = 0.0
    if tile.nb.bottom:
        mask[:, h - a:, :] = 0.0
    return mask


class SlicedCanvasNoise:
    """The full run's ONE canvas-wide draw (sync.build_canvas_noise's dummy at the same seed)
    sliced to this tile's latent window, so the lane starts from the noise 00787 gave it."""

    def __init__(self, vae, seed, canvas_h, canvas_w, rect):
        from context_anchored_tile_refine import sync, upscale

        self.seed = seed
        full = sync.build_canvas_noise(vae, upscale.Noise_RandomNoise(seed), canvas_h, canvas_w)
        self.slice = full[..., rect.y0 // 8:rect.y1 // 8, rect.x0 // 8:rect.x1 // 8].contiguous()

    def generate_noise(self, input_latent):
        expected = tuple(input_latent["samples"].shape)
        if tuple(self.slice.shape) != expected:
            raise RuntimeError(f"noise slice {tuple(self.slice.shape)} does not match the lane latent {expected}")
        return self.slice.clone()


# ------------------------------------------------------------------ the render

def render(arm, tile, tile_name, settings, vision, source_4k, use_dat, text=""):
    import comfy.samplers

    from context_anchored_tile_refine import sampling, sync, upscale

    timings = {}
    canvas_w, canvas_h = upscale.scale_target(int(source_4k.shape[2]), int(source_4k.shape[1]), settings.upscale_by)
    layout = solve_layout(canvas_w, canvas_h, settings)
    crop = tile.crop_rect
    print(f"[layout]  canvas {canvas_w}x{canvas_h} grid {layout.sol_x.n}x{layout.sol_y.n}  tile {tile_name} "
          f"crop {crop.x0},{crop.y0}-{crop.x1},{crop.y1}")

    print(f"[clip]    loading {CLIP_NAME} ({CLIP_TYPE})")
    clip = ab_models.load_clip(CLIP_NAME, CLIP_TYPE)
    started = time.perf_counter()
    # The VLM reads the canvas after a deep area downsample, where the DAT detail is invisible,
    # so the encode source is the plain 2x lanczos of the 4K picture.
    canvas = lanczos(source_4k, canvas_w, canvas_h)
    with torch.inference_mode():
        positives, infos = build_positives(arm, clip, canvas, [tile], vision, layout.tiles, text=text)
        positive, info = positives[0], infos[0]
        if SHUFFLE:
            positive = shuffle_rows(positive, settings.seed)
            info["rows_shuffled"] = True
        negative = ab_models.encode_prompt(clip, NEGATIVE)
        empty = upscale.encode_empty(clip)
    del canvas
    timings["conditioning"] = time.perf_counter() - started
    print(f"[arm]     {arm}: {info}")

    pixels = region_pixels(source_4k, crop, f"{tile.cls}-{tile_name}", settings, use_dat)
    source_png = OUTPUT_DIR / f"AB_tilephantom-{tile_name}__source.png"
    if not source_png.is_file():
        ab_models.save_png(source_png, pixels.cpu(), {"stage": "tile crop before refine", "tile": tile_name})

    print(f"[unet]    loading {UNET_NAME}")
    model = ab_models.load_unet(UNET_NAME)
    vae = ab_models.load_vae(VAE_NAME)
    sigmas = upscale.build_sigmas(model, settings.scheduler, settings.steps, settings.denoise)
    sampler = comfy.samplers.sampler_object(settings.sampler)
    noise = SlicedCanvasNoise(vae, settings.seed, canvas_h, canvas_w, crop)
    guider = upscale.build_guider(model, empty, negative, settings.cfg)
    mask = region_mask(tile, settings)

    saved = sync.build_tile_positives
    sync.build_tile_positives = lambda *args, **kwargs: [positive]
    started = time.perf_counter()
    try:
        with ab_models.VramProbe() as probe, torch.inference_mode():
            result = sampling.refine_image(
                pixels, guider, sampler, sigmas, vae, noise,
                settings.max_tile_width, settings.max_tile_height,
                settings.context_anchor, settings.context_overlap,
                mask=mask, vl_clip=clip, vlm_method="vision tokens",
                anchor_source=settings.anchor_source, sampler_name=settings.sampler)
    finally:
        sync.build_tile_positives = saved
    timings["refine"] = time.perf_counter() - started
    print(f"[refine]  {arm} done  {probe}")

    ab_models.require_image_shape(result, crop.y1 - crop.y0, crop.x1 - crop.x0, f"refine {arm}")
    payload = dataclasses.asdict(settings)
    payload.update({
        "run_label": arm, "tile": tile_name, "crop": [crop.x0, crop.y0, crop.x1, crop.y1],
        "conditioning": info, "source_4k": SOURCE_4K.name, "crop_upscale": "4xFaceUpDAT + lanczos" if use_dat else "lanczos",
        "unet": UNET_NAME, "clip": CLIP_NAME, "vae": VAE_NAME, "negative_prompt": NEGATIVE,
        "sigmas": [float(v) for v in sigmas],
        "timings": {name: round(seconds, 1) for name, seconds in timings.items()},
        "harness": "tests-AB/run_ab_tile_phantom.py",
        "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    destination = output_path(tile_name, arm, settings)
    written = ab_models.save_png(destination, result.cpu(), payload)
    print(f"[render]  {arm} -> {destination.name} {written[0]}x{written[1]}")
    print("[timing]  " + "  ".join(f"{name}={seconds:.1f}s" for name, seconds in timings.items()))


def render_block(arm, tile, tile_name, settings, vision, source_4k, use_dat, text=""):
    """The tile AND its bordering tiles as one sync run over the full canvas' mask path, with
    the solver forced to the full run's tile rects: every lane keeps its full-run crop, noise
    slice and neighbors, so the per-step consolidation the single-lane mode lacks is real."""
    import comfy.samplers

    from context_anchored_tile_refine import sampling, sync, upscale

    timings = {}
    canvas_w, canvas_h = upscale.scale_target(int(source_4k.shape[2]), int(source_4k.shape[1]), settings.upscale_by)
    layout = solve_layout(canvas_w, canvas_h, settings)
    block = block_rect(layout, tile)
    blk_layout, sx, sy = block_layout(layout, tile, block, settings)
    # The center tile's crop must be the full run's exactly; an edge tile's differs only by
    # the outer ring block_rect documents.
    first_col, first_row = max(tile.col - 1, 0), max(tile.row - 1, 0)
    for blk_tile in blk_layout.tiles:
        full_tile = next(t for t in layout.tiles
                         if t.col == blk_tile.col + first_col and t.row == blk_tile.row + first_row)
        shifted = (blk_tile.crop_rect.x0 + block.x0, blk_tile.crop_rect.y0 + block.y0,
                   blk_tile.crop_rect.x1 + block.x0, blk_tile.crop_rect.y1 + block.y0)
        full_rect = (full_tile.crop_rect.x0, full_tile.crop_rect.y0, full_tile.crop_rect.x1, full_tile.crop_rect.y1)
        if full_tile is tile and shifted != full_rect:
            raise RuntimeError(f"the block places {tile_name} at {shifted}, the full run at {full_rect}")
        if shifted != full_rect:
            print(f"[layout]  edge tile r{full_tile.row}c{full_tile.col} samples {shifted} (full run {full_rect})")
    print(f"[layout]  canvas {canvas_w}x{canvas_h} grid {layout.sol_x.n}x{layout.sol_y.n}  block around {tile_name} "
          f"{block.x0},{block.y0}-{block.x1},{block.y1} = {blk_layout.sol_x.n}x{blk_layout.sol_y.n} tiles")

    started = time.perf_counter()
    pixels = region_pixels(source_4k, block, f"block-{tile_name}", settings, use_dat)
    source_png = OUTPUT_DIR / f"AB_tilephantom-block-{tile_name}__source.png"
    if not source_png.is_file():
        ab_models.save_png(source_png, pixels.cpu(), {"stage": "block before refine", "block": [block.x0, block.y0, block.x1, block.y1]})
    # The engine's image AND encode source: the plain 2x lanczos canvas, with the sampled block
    # replaced by the DAT stage's pixels, which is what the node's own upscale produced there.
    canvas = lanczos(source_4k, canvas_w, canvas_h)
    canvas[:, block.y0:block.y1, block.x0:block.x1, :] = pixels
    timings["pixels"] = time.perf_counter() - started

    print(f"[clip]    loading {CLIP_NAME} ({CLIP_TYPE})")
    clip = ab_models.load_clip(CLIP_NAME, CLIP_TYPE)
    started = time.perf_counter()
    with torch.inference_mode():
        positives, infos = build_positives(arm, clip, canvas, blk_layout.tiles, vision, layout.tiles, text=text,
                                           offset_x=block.x0, offset_y=block.y0)
        if SHUFFLE:
            positives = [shuffle_rows(positive, settings.seed) for positive in positives]
            for info in infos:
                info["rows_shuffled"] = True
        negative = ab_models.encode_prompt(clip, NEGATIVE)
        empty = upscale.encode_empty(clip)
    timings["conditioning"] = time.perf_counter() - started
    for blk_tile, info in zip(blk_layout.tiles, infos, strict=True):
        print(f"[arm]     {arm} r{blk_tile.row + first_row}c{blk_tile.col + first_col}: {info}")

    mask = block_mask(block, canvas_w, canvas_h, settings)
    bbox = sampling._mask_bbox(mask >= 0.5)
    y0, y1, x0, x1 = sampling._expand_snap_clamp(bbox, settings.context_anchor, canvas_h, canvas_w)
    if (x0, y0, x1, y1) != (block.x0, block.y0, block.x1, block.y1):
        raise RuntimeError(f"mask path would crop {(x0, y0, x1, y1)}, not the block {block}")

    print(f"[unet]    loading {UNET_NAME}")
    model = ab_models.load_unet(UNET_NAME)
    vae = ab_models.load_vae(VAE_NAME)
    sigmas = upscale.build_sigmas(model, settings.scheduler, settings.steps, settings.denoise)
    sampler = comfy.samplers.sampler_object(settings.sampler)
    noise = SlicedCanvasNoise(vae, settings.seed, canvas_h, canvas_w, block)
    guider = upscale.build_guider(model, empty, negative, settings.cfg)

    saved = sync.build_tile_positives
    sync.build_tile_positives = lambda *args, **kwargs: list(positives)
    started = time.perf_counter()
    try:
        with SolverPatch(block, sx, sy), ab_models.VramProbe() as probe, torch.inference_mode():
            result = sampling.refine_image(
                canvas, guider, sampler, sigmas, vae, noise,
                settings.max_tile_width, settings.max_tile_height,
                settings.context_anchor, settings.context_overlap,
                mask=mask, vl_clip=clip, vlm_method="vision tokens",
                anchor_source=settings.anchor_source, sampler_name=settings.sampler)
    finally:
        sync.build_tile_positives = saved
    timings["refine"] = time.perf_counter() - started
    print(f"[refine]  {arm} block done  {probe}")

    ab_models.require_image_shape(result, canvas_h, canvas_w, f"refine block {arm}")
    crop = tile.crop_rect
    payload = dataclasses.asdict(settings)
    payload.update({
        "run_label": f"block-{arm}", "tile": tile_name, "block": [block.x0, block.y0, block.x1, block.y1],
        "crop": [crop.x0, crop.y0, crop.x1, crop.y1], "conditioning": infos, "source_4k": SOURCE_4K.name,
        "crop_upscale": "4xFaceUpDAT + lanczos" if use_dat else "lanczos",
        "unet": UNET_NAME, "clip": CLIP_NAME, "vae": VAE_NAME, "negative_prompt": NEGATIVE,
        "sigmas": [float(v) for v in sigmas],
        "timings": {name: round(seconds, 1) for name, seconds in timings.items()},
        "harness": "tests-AB/run_ab_tile_phantom.py --block",
        "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    block_png = output_path(f"block-{tile_name}", arm, settings)
    written = ab_models.save_png(block_png, result[:, block.y0:block.y1, block.x0:block.x1, :].contiguous().cpu(), payload)
    print(f"[render]  {arm} -> {block_png.name} {written[0]}x{written[1]}")
    tile_png = output_path(tile_name, f"{arm}-block", settings)
    ab_models.save_png(tile_png, result[:, crop.y0:crop.y1, crop.x0:crop.x1, :].contiguous().cpu(), payload)
    print(f"[render]  {arm} -> {tile_png.name} (the tile cut from the block)")
    print("[timing]  " + "  ".join(f"{name}={seconds:.1f}s" for name, seconds in timings.items()))


def main(argv=None):
    global SOURCE_4K, NAME_TAG, SHUFFLE
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--tile", default="r0c1", help="tile name as r<row>c<col> (default r0c1)")
    parser.add_argument("--denoise", type=float, default=Settings.denoise)
    parser.add_argument("--seed", type=int, default=Settings.seed)
    parser.add_argument("--steps", type=int, default=Settings.steps)
    parser.add_argument("--cfg", type=float, default=Settings.cfg)
    parser.add_argument("--text", default="dark storm clouds in a night sky", help="the text arm's prompt")
    parser.add_argument("--canvas-tokens", type=int, default=None,
                        help="[vision] canvas_tokens for this run (default: the settings file's value)")
    parser.add_argument("--crop-tokens", type=int, default=None,
                        help="[vision] crop_tokens for this run (default: the settings file's value)")
    parser.add_argument("--shuffle", action="store_true", help="permute every tile's rows (the bag test)")
    parser.add_argument("--tag", default="", help="suffix for the output name, e.g. a determinism repeat")
    parser.add_argument("--source", type=Path, default=SOURCE_4K, help="the 4K picture pass 2 upscales")
    parser.add_argument("--no-dat", action="store_true", help="lanczos-only crop instead of the DAT stage")
    parser.add_argument("--block", action="store_true",
                        help="refine the tile and its bordering tiles together, with real per-step consolidation")
    args = parser.parse_args(argv)

    SOURCE_4K = args.source
    settings = dataclasses.replace(Settings(), denoise=args.denoise, seed=args.seed, steps=args.steps, cfg=args.cfg)
    root, note = ab_env.bootstrap()
    print(f"[env]     ComfyUI {ab_env.version(root)} at {root}  ({note})")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    from context_anchored_tile_refine import captions, upscale

    vision = captions.load_settings().vision
    if args.canvas_tokens is not None:
        vision = dataclasses.replace(vision, canvas_tokens=args.canvas_tokens)
    if args.crop_tokens is not None:
        vision = dataclasses.replace(vision, crop_tokens=args.crop_tokens)
    print(f"[vision]  canvas_tokens {vision.canvas_tokens}  crop_tokens {vision.crop_tokens}")
    SHUFFLE = args.shuffle
    if args.arm in VISION_ARMS:
        NAME_TAG += f"-canvas{vision.canvas_tokens}-crop{vision.crop_tokens}"
    if args.shuffle:
        NAME_TAG += "-shuffled"
    if args.tag:
        NAME_TAG += f"-{args.tag}"

    source_4k = load_source(SOURCE_4K)
    canvas_w, canvas_h = upscale.scale_target(int(source_4k.shape[2]), int(source_4k.shape[1]), settings.upscale_by)
    layout = solve_layout(canvas_w, canvas_h, settings)
    by_name = {f"r{t.row}c{t.col}": t for t in layout.tiles}
    if args.tile not in by_name:
        raise SystemExit(f"no tile {args.tile}; the grid is {layout.sol_x.n}x{layout.sol_y.n}")
    run = render_block if args.block else render
    run(args.arm, by_name[args.tile], args.tile, settings, vision, source_4k, use_dat=not args.no_dat, text=args.text)


if __name__ == "__main__":
    main()
