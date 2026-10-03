"""A/B of the base node's per-step sampler noise on the raster path.

Arm `per-tile` turns the package's shared step noise off (per_tile_arm). Every tile hands
the sampler the same seed, so an ancestral sampler draws the same per-step noise stream for
every tile of the same crop shape (core default_noise_sampler seeds one generator per
call). Arm `canvas` draws each noise step once at canvas size and gives every tile its
crop's slice, the way the sync engine shares its field, so tiles never repeat a pattern.
The canvas arm also gives two overlapping tiles the same noise on their shared band, where
the per-tile arm gives them different crop-local draws, so a seam can differ between the
arms for that reason too.

Both arms share the upscaled input, the canvas start noise, the model and every setting,
and they differ only in the noise_sampler the tile's sampler receives. The settings follow
the base node example workflow (euler_ancestral, ddim_uniform, 9 steps, denoise 0.35,
anchor and overlap 64, empty prompt) on Krea 2, since that workflow's Z-Image model is not
installed. The tile caps are its second stage's 1280x2048, whose 4x2 grid at 4096x2304
puts two same-shaped tiles side by side in each row. One render per process (CLAUDE.md,
3x and above):

    python tests-AB/run_ab_raster_noise.py --arm per-tile --seed 42
    python tests-AB/run_ab_raster_noise.py --arm canvas --seed 42
    python tests-AB/run_ab_raster_noise.py --arm shipped --scene market --seed 42 --tag before

Arm `shipped` runs the package as it stands with no harness hook, which shares one field
built from 64-cell blocks (sampling.StepNoiseField), and `--tag` names the render, so one
command renders a scene before and after a production change.

Each run also reports the repeat metric, the correlation of the high-passed refine
residual between tiles of the same crop shape at the same crop-local cells, minus the same
correlation with one window shifted SHIFT_CONTROL px. Content tied to crop-local position
in both arms (the DiT's position ids start at each crop) raises both terms alike, so the
difference isolates a repeat.
"""
import argparse
import contextlib
import json
import time
from pathlib import Path

import ab_env
import ab_models

ARMS = ("per-tile", "canvas", "shipped")
SCENES = {"dark-city": "dark-city.webp", "city": "cyberpunk-city.webp", "market": "renaissance-market-original.png"}
UPSCALE_MODEL = "4x-UltraSharp.pth"
UPSCALE_BY = 4.0
UNET = "krea2RawBaseInt8Row_v10.safetensors"
VAE = "krea2RealVae_v10.safetensors"
CLIP = "qwen3vl_4b_fp8_scaled.safetensors"
SAMPLER = "euler_ancestral"
SCHEDULER = "ddim_uniform"
STEPS = 9
DENOISE = 0.35
# Positive and negative are both the empty prompt, so CFG above 1 returns the same
# prediction at twice the evals.
CFG = 1.0
MAX_TILE_WIDTH = 1280
MAX_TILE_HEIGHT = 2048
CONTEXT_ANCHOR = 64
CONTEXT_OVERLAP = 64
LATENT_SCALE = 8
# 9 px keeps single-pixel grain, and 33 px also keeps the 8 to 32 px band a latent-cell
# pattern would sit in.
HIGH_PASS_KERNELS = (9, 33)
SHIFT_CONTROL = 16
OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Raster-Noise")


def output_path(scene, arm, seed, tag=""):
    suffix = f"__{tag}" if tag else ""
    return OUTPUT_DIR / f"AB_raster-noise__{scene}__{arm}__s{seed}{suffix}.png"


def upscaled_input(scene="dark-city"):
    # Computed once and reused, so every arm and seed refines the identical pixels.
    import numpy as np
    import torch
    from comfy_extras.nodes_upscale_model import UpscaleModelLoader
    from PIL import Image

    from context_anchored_tile_refine import upscale

    cache = OUTPUT_DIR / f"upscaled__{scene}__{UPSCALE_MODEL}.pt"
    source = ab_env.REPO_ROOT / "samples" / SCENES[scene]
    picture = None
    image = None
    upscale_model = None
    upscaled = None

    if cache.is_file():
        return torch.load(cache)
    picture = np.asarray(Image.open(source).convert("RGB"), dtype=np.float32) / 255.0
    image = torch.from_numpy(picture)[None]
    upscale_model = UpscaleModelLoader.execute(UPSCALE_MODEL)[0]
    with torch.inference_mode():
        upscaled = upscale.prepare_upscaled(image, upscale_model, UPSCALE_BY).float().cpu().clone()
    del upscale_model
    ab_models.free_gpu()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(upscaled, cache)
    return upscaled


def empty_conditioning():
    # Cached, so later processes skip loading the text encoder to encode "".
    import torch

    from context_anchored_tile_refine import upscale

    cache = OUTPUT_DIR / f"empty__{CLIP}.pt"
    clip = None
    empty = None

    if cache.is_file():
        return torch.load(cache, weights_only=False)
    clip = ab_models.load_clip(CLIP, "krea2")
    with torch.inference_mode():
        empty = upscale.encode_empty(clip)
    del clip
    ab_models.free_gpu()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(empty, cache)
    return empty


class CanvasNoiseField:
    """One canvas-sized CPU draw per noise step, sliced to each tile's crop.

    Seeded seed + 1, core's CPU offset (default_noise_sampler), because a generator seeded
    with the bare seed would replay the canvas start noise as the first step's draw.
    Tiles run one after another, so each tile replays the draws from the start, and every
    tile gets the same canvas field at the same step.
    """

    def __init__(self, shape, seed):
        self.shape = tuple(shape)
        self.seed = int(seed)

    def for_window(self, y0, y1, x0, x1):
        import torch

        generator = torch.Generator().manual_seed(self.seed + 1)

        def noise_sampler(sigma, sigma_next):
            field = torch.randn(self.shape, generator=generator)
            return field[..., y0:y1, x0:x1].to(device=sigma.device)

        return noise_sampler


@contextlib.contextmanager
def canvas_noise_arm(seed):
    """Give every raster tile's sampler its crop's slice of one CanvasNoiseField.

    Wraps the two seams _refine_tiles calls by name. build_canvas_noise reports the canvas
    size, from which the harness solves the same grid, and sample_latent maps the Nth call
    to the Nth tile in raster order. Each tile's start noise must equal the canvas start
    noise at the harness window, which proves the mapping.
    """
    import comfy.samplers
    import torch

    from context_anchored_tile_refine import grid, sampling

    original_build = sampling.build_canvas_noise
    original_sample = sampling.sample_latent
    state = {"field": None, "canvas_noise": None, "tiles": None, "next": 0}

    def build_canvas_noise(vae, noise, canvas_h, canvas_w, *args, **kwargs):
        canvas_noise = original_build(vae, noise, canvas_h, canvas_w, *args, **kwargs)
        layout = grid.solve_layout(canvas_w, canvas_h, MAX_TILE_WIDTH, MAX_TILE_HEIGHT,
                                   CONTEXT_ANCHOR, CONTEXT_OVERLAP)
        state["field"] = CanvasNoiseField(canvas_noise.shape, seed)
        state["canvas_noise"] = canvas_noise
        state["tiles"] = layout.tiles
        state["next"] = 0
        return canvas_noise

    def sample_latent(guider, sampler, sigmas, noise_tensor, tile_seed, latent_samples, denoise_mask=None, callback=None):
        tile = state["tiles"][state["next"]]
        crop = tile.crop_rect
        y0, y1 = crop.y0 // LATENT_SCALE, crop.y1 // LATENT_SCALE
        x0, x1 = crop.x0 // LATENT_SCALE, crop.x1 // LATENT_SCALE
        if not torch.equal(noise_tensor, state["canvas_noise"][..., y0:y1, x0:x1]):
            raise RuntimeError(f"Tile {state['next']} start noise is not the canvas start noise at the harness "
                               f"window {(y0, y1, x0, x1)}. The harness geometry no longer matches the run.")
        state["next"] += 1
        tile_sampler = comfy.samplers.KSAMPLER(
            sampler.sampler_function,
            extra_options={**sampler.extra_options, "noise_sampler": state["field"].for_window(y0, y1, x0, x1)},
            inpaint_options=dict(sampler.inpaint_options),
        )
        return original_sample(guider, tile_sampler, sigmas, noise_tensor, tile_seed, latent_samples,
                               denoise_mask=denoise_mask, callback=callback)

    sampling.build_canvas_noise = build_canvas_noise
    sampling.sample_latent = sample_latent
    try:
        yield state
    finally:
        sampling.build_canvas_noise = original_build
        sampling.sample_latent = original_sample
    if state["tiles"] is not None and state["next"] != len(state["tiles"]):
        raise RuntimeError(f"The run sampled {state['next']} tiles, but the harness grid has {len(state['tiles'])}.")


@contextlib.contextmanager
def per_tile_arm():
    """Turn the shipped shared step noise off, so every tile's sampler draws its own per-step
    stream from the tile seed over its own crop."""
    from context_anchored_tile_refine import sampling

    original = sampling.build_step_noise
    sampling.build_step_noise = lambda *args, **kwargs: None
    try:
        yield
    finally:
        sampling.build_step_noise = original


def arm_scope(arm, seed):
    # One arm-to-hook map for this harness and probe_raster_noise_band.py.
    if arm == "per-tile":
        return per_tile_arm()
    if arm == "canvas":
        return canvas_noise_arm(seed)
    return contextlib.nullcontext()


def solo_cells(tile, layout, inset):
    """The core rect only this tile writes, inset by `inset`, as crop-local (y0, y1, x0, x1).

    The next row's and the next column's paste bands reach `overlap` px into a core's
    bottom and right edges, where the feather and the seam cut blend two tiles.
    """
    core, crop = tile.core, tile.crop_rect
    bottom = layout.overlap if tile.nb.bottom else 0
    right = layout.overlap if tile.nb.right else 0
    return (core.y0 - crop.y0 + inset, core.y1 - bottom - crop.y0 - inset,
            core.x0 - crop.x0 + inset, core.x1 - right - crop.x0 - inset)


def repeat_metric(result, source):
    """Per high-pass kernel, the mean aligned, shifted and aligned minus shifted correlation of
    the refine residual over every pair of same-shaped tiles."""
    import itertools

    import torch

    from context_anchored_tile_refine import grid

    residual = (result[0] - source[0]).mean(dim=-1)[None, None].float()
    layout = grid.solve_layout(int(result.shape[2]), int(result.shape[1]), MAX_TILE_WIDTH, MAX_TILE_HEIGHT,
                               CONTEXT_ANCHOR, CONTEXT_OVERLAP)
    inset = max(HIGH_PASS_KERNELS) // 2
    windows = []
    by_kernel = {}

    for (first_index, first), (second_index, second) in itertools.combinations(enumerate(layout.tiles), 2):
        a, b = first.crop_rect, second.crop_rect
        if (a.x1 - a.x0, a.y1 - a.y0) != (b.x1 - b.x0, b.y1 - b.y0):
            continue
        cells_a, cells_b = solo_cells(first, layout, inset), solo_cells(second, layout, inset)
        y0, y1 = max(cells_a[0], cells_b[0]), min(cells_a[1], cells_b[1])
        x0, x1 = max(cells_a[2], cells_b[2]), min(cells_a[3], cells_b[3]) - SHIFT_CONTROL
        if y1 > y0 and x1 > x0:
            windows.append((first_index, second_index, a, b, y0, y1, x0, x1))
    for kernel in HIGH_PASS_KERNELS:
        blurred = torch.nn.functional.avg_pool2d(residual, kernel, stride=1, padding=kernel // 2,
                                                 count_include_pad=False)
        detail = (residual - blurred)[0, 0]
        pairs = []
        for first_index, second_index, a, b, y0, y1, x0, x1 in windows:
            window_a = detail[a.y0 + y0:a.y0 + y1, a.x0 + x0:a.x0 + x1].flatten()
            window_b = detail[b.y0 + y0:b.y0 + y1, b.x0 + x0:b.x0 + x1].flatten()
            shifted_b = detail[b.y0 + y0:b.y0 + y1, b.x0 + x0 + SHIFT_CONTROL:b.x0 + x1 + SHIFT_CONTROL].flatten()
            aligned = torch.corrcoef(torch.stack([window_a, window_b]))[0, 1].item()
            shifted = torch.corrcoef(torch.stack([window_a, shifted_b]))[0, 1].item()
            pairs.append({"tiles": [first_index, second_index], "cells": int(window_a.numel()),
                          "aligned": round(aligned, 5), "shifted": round(shifted, 5),
                          "repeat": round(aligned - shifted, 5)})
        by_kernel[f"k{kernel}"] = {
            "mean_repeat": sum(p["repeat"] for p in pairs) / len(pairs) if pairs else None,
            "pairs": pairs,
        }
    return by_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scene", choices=sorted(SCENES), default="dark-city")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    ab_env.bootstrap()

    import comfy.samplers
    import torch

    from context_anchored_tile_refine import sampling, upscale

    path = output_path(args.scene, args.arm, args.seed, args.tag)
    settings = {
        "image_label": args.scene, "run_label": args.arm, "upscale_model": UPSCALE_MODEL, "upscale_by": UPSCALE_BY,
        "unet": UNET, "vae": VAE, "sampler": SAMPLER, "scheduler": SCHEDULER, "steps": STEPS, "denoise": DENOISE,
        "cfg": CFG, "seed": args.seed, "max_tile_width": MAX_TILE_WIDTH, "max_tile_height": MAX_TILE_HEIGHT,
        "context_anchor": CONTEXT_ANCHOR, "context_overlap": CONTEXT_OVERLAP,
    }
    scope = arm_scope(args.arm, args.seed)
    canvas = None
    empty = None
    result = None
    elapsed = 0.0

    canvas = upscaled_input(args.scene)
    empty = empty_conditioning()
    model = ab_models.load_unet(UNET)
    vae = ab_models.load_vae(VAE)
    with torch.inference_mode():
        guider = upscale.build_guider(model, empty, empty, CFG)
        sigmas = upscale.build_sigmas(model, SCHEDULER, STEPS, DENOISE)
        sampler = comfy.samplers.sampler_object(SAMPLER)
        started = time.perf_counter()
        with scope:
            result = sampling.refine_image(canvas, guider, sampler, sigmas, vae, upscale.Noise_RandomNoise(args.seed),
                                           MAX_TILE_WIDTH, MAX_TILE_HEIGHT, CONTEXT_ANCHOR, CONTEXT_OVERLAP)
        elapsed = time.perf_counter() - started
    result = result.float().cpu()

    report = {"settings": settings, "refine_seconds": round(elapsed, 1), "repeat": repeat_metric(result, canvas)}
    ab_models.save_png(path, result, settings)
    path.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"refine_seconds": report["refine_seconds"],
                      **{kernel: values["mean_repeat"] for kernel, values in report["repeat"].items()}}))
    print("wrote", path)


if __name__ == "__main__":
    main()
