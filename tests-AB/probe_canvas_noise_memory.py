"""RAM and CPU time of a canvas-wide per-step noise field for the base node's raster path.

Each case runs in its own process so its resident memory reads clean, at the Krea 2 latent
shape [1, 16, 1, H/8, W/8] for a 4K and an 8K canvas, over the grid the raster noise A/B uses:
    replay     every tile replays one canvas-sized CPU draw per noise step from the seed,
               the run_ab_raster_noise canvas arm (one field alive at a time)
    cache      every step's canvas field drawn once and held for the run
    blocks     every step's field cut into fixed blocks, each drawn from its own seed, so a
               tile draws only the blocks under its crop
    brownian   core's BrownianTreeNoiseSampler at canvas size, the field the VL nodes' sync
               engine already holds, queried once per noise step
None of the cases touches the GPU. Only a tile's slice moves there per step, the same size
as core's own per-tile draw today.

    python tests-AB/probe_canvas_noise_memory.py
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import ab_env

CASES = ("replay", "cache", "blocks", "brownian")
CANVASES = {"4K": (4096, 2304), "8K": (8192, 4608)}
LATENT_CHANNELS = 16
LATENT_SCALE = 8
MAX_TILE_WIDTH, MAX_TILE_HEIGHT, CONTEXT_ANCHOR, CONTEXT_OVERLAP = 1280, 2048, 64, 64
BLOCK_CELLS = 64
SIGMA_START = 0.6


def tile_windows(width, height):
    from context_anchored_tile_refine import grid

    layout = grid.solve_layout(width, height, MAX_TILE_WIDTH, MAX_TILE_HEIGHT, CONTEXT_ANCHOR, CONTEXT_OVERLAP)
    return [(t.crop_rect.y0 // LATENT_SCALE, t.crop_rect.y1 // LATENT_SCALE,
             t.crop_rect.x0 // LATENT_SCALE, t.crop_rect.x1 // LATENT_SCALE) for t in layout.tiles]


def resident_mb():
    import psutil

    return psutil.Process().memory_info().rss / 2**20


def block_window(seed, step, window, channels):
    """One step's noise over `window`, assembled from the fixed blocks it covers."""
    import torch

    y0, y1, x0, x1 = window
    out = torch.empty((1, channels, 1, y1 - y0, x1 - x0))

    for by in range(y0 // BLOCK_CELLS, (y1 - 1) // BLOCK_CELLS + 1):
        for bx in range(x0 // BLOCK_CELLS, (x1 - 1) // BLOCK_CELLS + 1):
            generator = torch.Generator().manual_seed(hash((seed, step, by, bx)) & 0x7FFF_FFFF_FFFF_FFFF)
            block = torch.randn((1, channels, 1, BLOCK_CELLS, BLOCK_CELLS), generator=generator)
            cy0, cy1 = max(y0, by * BLOCK_CELLS), min(y1, (by + 1) * BLOCK_CELLS)
            cx0, cx1 = max(x0, bx * BLOCK_CELLS), min(x1, (bx + 1) * BLOCK_CELLS)
            out[..., cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = block[..., cy0 - by * BLOCK_CELLS:cy1 - by * BLOCK_CELLS,
                                                                   cx0 - bx * BLOCK_CELLS:cx1 - bx * BLOCK_CELLS]
    return out


def run_case(case, canvas, steps):
    import torch

    width, height = CANVASES[canvas]
    shape = (1, LATENT_CHANNELS, 1, height // LATENT_SCALE, width // LATENT_SCALE)
    windows = tile_windows(width, height)
    noisy_steps = steps - 1
    sigmas = torch.linspace(SIGMA_START, 0.0, steps + 1)
    held = []
    peak = 0.0
    started = 0.0
    before = 0.0

    torch.set_grad_enabled(False)
    before = resident_mb()
    started = time.perf_counter()
    if case == "replay":
        for y0, y1, x0, x1 in windows:
            generator = torch.Generator().manual_seed(43)
            for _ in range(noisy_steps):
                field = torch.randn(shape, generator=generator)
                held = [field[..., y0:y1, x0:x1].clone()]
                peak = max(peak, resident_mb())
                del field
    elif case == "cache":
        generator = torch.Generator().manual_seed(43)
        held = [torch.randn(shape, generator=generator) for _ in range(noisy_steps)]
        peak = resident_mb()
        for y0, y1, x0, x1 in windows:
            for field in held:
                field[..., y0:y1, x0:x1].clone()
    elif case == "blocks":
        for window in windows:
            for step in range(noisy_steps):
                held = [block_window(42, step, window, LATENT_CHANNELS)]
                peak = max(peak, resident_mb())
    else:
        ab_env.bootstrap()
        import comfy.k_diffusion.sampling

        before = resident_mb()
        started = time.perf_counter()
        tree = comfy.k_diffusion.sampling.BrownianTreeNoiseSampler(
            torch.zeros(shape), sigmas[sigmas > 0].min(), sigmas.max(), seed=42, cpu=True)
        peak = resident_mb()
        for step in range(noisy_steps):
            held = [tree(sigmas[step], sigmas[step + 1])]
            peak = max(peak, resident_mb())
    elapsed = time.perf_counter() - started
    field_mb = LATENT_CHANNELS * shape[-2] * shape[-1] * 4 / 2**20
    return {"case": case, "canvas": canvas, "steps": steps, "tiles": len(windows), "field_mb": round(field_mb, 1),
            "peak_extra_mb": round(max(peak, resident_mb()) - before, 1), "seconds": round(elapsed, 2)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--canvas", choices=sorted(CANVASES))
    parser.add_argument("--steps", type=int, default=9)
    args = parser.parse_args()
    rows = []

    if args.case:
        sys.path.insert(0, str(ab_env.REPO_ROOT))
        print(json.dumps(run_case(args.case, args.canvas, args.steps)))
        return
    for steps in (9, 28):
        for canvas in CANVASES:
            for case in CASES:
                completed = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--case", case, "--canvas", canvas,
                     "--steps", str(steps)], capture_output=True, text=True, check=True)
                row = json.loads(completed.stdout.strip().splitlines()[-1])
                rows.append(row)
                print(f"{steps:>3} steps {canvas} {case:<9} tiles {row['tiles']:>3}  field {row['field_mb']:>6} MB  "
                      f"peak extra RAM {row['peak_extra_mb']:>7} MB  {row['seconds']:>6} s", flush=True)


if __name__ == "__main__":
    main()
