"""VRAM and wall time of ONE VAE window, at the owner's 8K geometry.

Why this exists. run_ab_vae_window.py settles whether capping the two VAE windows changes
the picture, and at its 1536x864 scale nothing is near the card's limit, so the peak VRAM
never moves and the answer to "what does the cap buy" stays theoretical. The saving that
motivates the change lives at 8K, where comfy's reservation decides whether VAE.decode gets
the fast path or falls back to decode_tiled, and whether the DiT is evicted between stages.

What this measures. The WORST single tile of the 8K pass-2 layout, encoded and decoded at
each candidate window size, with the real VAE. Peak allocation and wall time only. The
tensors are random, so this says NOTHING about pixels. Peak VRAM and time do not depend on
content, and the picture question is run_ab_vae_window.py's job.

One window per margin is the right unit: comfy reserves per call
(comfy/sd.py memory_used_encode / memory_used_decode), so the run's peak is set by its
largest single window, not by the total.

    <venv-python> tests-AB/probe_vae_window_vram.py
    <venv-python> tests-AB/probe_vae_window_vram.py --width 4096 --height 3456
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_env
import ab_models
import run_ab_krea2 as krea2_run  # noqa: F401  sets COMFYUI_ROOT at import, before bootstrap()
import run_ab_matrix as matrix_run

# The owner's shipped 8K pass 2 (Krea 2 8k upscale.json, second node).
CANVAS_W, CANVAS_H = 8192, 6912
MAX_TILE_W, MAX_TILE_H = 2048, 1728
CONTEXT_ANCHOR, CONTEXT_OVERLAP = 32, 256
MARGINS = (None, 128, 64, 32, 0)


def worst_window(layout, stage, margin):
    from context_anchored_tile_refine import sync

    window_of = sync.encode_window if stage == "encode" else sync.decode_window
    rects = [window_of(tile, margin) for tile in layout.tiles]
    return max(rects, key=lambda r: (r.x1 - r.x0) * (r.y1 - r.y0))


def reservation_gb(stage, width, height, dtype_size=2):
    # comfy/sd.py's own Wan 2.1 formulas, the numbers load_models_gpu is actually given.
    if stage == "encode":
        return 1500 * width * height * dtype_size / 1e9
    return 2200 * (width // 8) * (height // 8) * 64 * dtype_size / 1e9


def probe(vae, stage, rect):
    from context_anchored_tile_refine import sampling

    width, height = rect.x1 - rect.x0, rect.y1 - rect.y0
    ab_models.clear_cache()
    if stage == "encode":
        pixels = torch.rand(1, height, width, 3)

        def call():
            return sampling.encode_pixels(vae, pixels)
    else:
        frames = (1,) if getattr(vae, "latent_dim", 2) == 3 else ()
        latent = torch.randn(1, vae.latent_channels, *frames, height // 8, width // 8)

        def call():
            return vae.decode(latent)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        with ab_models.VramProbe() as vram, torch.inference_mode():
            call()
        note = ""
    except torch.cuda.OutOfMemoryError as error:
        return width, height, float("nan"), f"OOM: {str(error)[:60]}"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return width, height, time.perf_counter() - started, f"{vram}{note}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--width", type=int, default=CANVAS_W)
    parser.add_argument("--height", type=int, default=CANVAS_H)
    args = parser.parse_args(argv)

    root, note = ab_env.bootstrap()
    print(f"[env]     ComfyUI {ab_env.version(root)} at {root}  ({note})")
    from context_anchored_tile_refine import grid

    sx = grid.solve_axis(args.width, MAX_TILE_W, CONTEXT_ANCHOR, CONTEXT_OVERLAP, axis="width")
    sy = grid.solve_axis(args.height, MAX_TILE_H, CONTEXT_ANCHOR, CONTEXT_OVERLAP, axis="height")
    layout = grid.build_layout(args.width, args.height, sx, sy, CONTEXT_ANCHOR, CONTEXT_OVERLAP)
    print(f"[layout]  {args.width}x{args.height}  grid {sx.n}x{sy.n} = {len(layout.tiles)} tiles, "
          f"anchor {CONTEXT_ANCHOR} overlap {CONTEXT_OVERLAP}")

    print(f"[vae]     loading {matrix_run.VAE_NAME}")
    vae = ab_models.load_vae(matrix_run.VAE_NAME)

    for stage in ("encode", "decode"):
        print(f"\n=== stage {stage[0].upper()}, worst single tile, RANDOM content "
              f"(VRAM and time only)")
        # One throwaway call first. The very first encode or decode of a process pays
        # kernel autotuning, which would otherwise land entirely on whichever margin ran
        # first and read as that window being slow.
        probe(vae, stage, worst_window(layout, stage, 0))
        print(f"{'margin':>8}{'window':>14}{'formula GB':>12}{'seconds':>10}   measured peak")
        for margin in MARGINS:
            rect = worst_window(layout, stage, margin)
            width, height, seconds, vram = probe(vae, stage, rect)
            label = "none" if margin is None else str(margin)
            print(f"{label:>8}{f'{width}x{height}':>14}"
                  f"{reservation_gb(stage, width, height):12.2f}{seconds:10.2f}   {vram}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
