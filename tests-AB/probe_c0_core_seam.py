"""Does a capped stage E window put a STEP in C_0 where the cores butt?

The one place a small encode window can bite. C_0 is assembled from butted cores, so a
tile's kept region ends exactly on a core boundary. At margin 0 the two encodes meeting at
that boundary share no pixels at all, which is the classic tiled-VAE setup that shows tile
edges. At the shipped window they share 2 * (context_anchor + context_overlap) pixels.

run_ab_vae_window.py --stage encode answers the picture question, but its 2x1 grid has ONE
interior boundary and an 8K run has sixty. This probe is encode-only, so it can afford a
real multi-tile grid, and it measures the boundary directly instead of inferring it through
twenty diffusion steps and two feathers.

Two measurements per margin, both against the same real image:
    latent step   |C_0[.., c-1] - C_0[.., c]| across every core boundary, over the mean
                  neighbouring-cell difference nearby. 1.0 means the boundary is no more of
                  a step than the picture already has there.
    pixel step    the same ratio on vae.decode(C_0), which is what the lanes start from.

A ratio near 1.0 at every margin means the cap adds no boundary. Anything climbing with a
smaller margin is the tiled-VAE edge appearing, and it is a reason to keep a margin.

    <venv-python> tests-AB/probe_c0_core_seam.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_env
import ab_models
import run_ab_krea2 as krea2_run  # noqa: F401  sets COMFYUI_ROOT at import, before bootstrap()
import run_ab_matrix as matrix_run

CANVAS_W, CANVAS_H = 4096, 3456          # 3x3 at the production caps: 2+2 interior boundaries
MAX_TILE_W, MAX_TILE_H = 2048, 1728
CONTEXT_ANCHOR, CONTEXT_OVERLAP = 32, 256
MARGINS = (None, 128, 64, 32, 0)
SPAN = 24                                 # cells or pixels either side, for the local baseline


def step_ratio(plane, cuts, span):
    """Mean |difference| across each cut line, over the mean |difference| just beside it."""
    diff = (plane[:, 1:] - plane[:, :-1]).abs().mean(dim=0)      # index i = |col i+1 - col i|
    ratios = []
    for cut in cuts:
        if cut - span < 1 or cut + span >= diff.shape[0]:
            continue
        local = torch.cat([diff[cut - span:cut - 1], diff[cut + 1:cut + span]]).mean()
        ratios.append(float(diff[cut - 1] / local) if float(local) else float("nan"))
    return sum(ratios) / len(ratios) if ratios else float("nan")


def both_axes(plane, x_cuts, y_cuts, span):
    return (step_ratio(plane, x_cuts, span) + step_ratio(plane.T, y_cuts, span)) / 2


def main():
    root, note = ab_env.bootstrap()
    print(f"[env]     ComfyUI {ab_env.version(root)} at {root}  ({note})")
    from context_anchored_tile_refine import grid, sync

    sx = grid.solve_axis(CANVAS_W, MAX_TILE_W, CONTEXT_ANCHOR, CONTEXT_OVERLAP, axis="width")
    sy = grid.solve_axis(CANVAS_H, MAX_TILE_H, CONTEXT_ANCHOR, CONTEXT_OVERLAP, axis="height")
    layout = grid.build_layout(CANVAS_W, CANVAS_H, sx, sy, CONTEXT_ANCHOR, CONTEXT_OVERLAP)
    x_cuts = sorted({t.core.x0 for t in layout.tiles} | {t.core.x1 for t in layout.tiles})[1:-1]
    y_cuts = sorted({t.core.y0 for t in layout.tiles} | {t.core.y1 for t in layout.tiles})[1:-1]
    print(f"[layout]  {CANVAS_W}x{CANVAS_H}  grid {sx.n}x{sy.n} = {len(layout.tiles)} tiles, "
          f"interior core boundaries at x={x_cuts} y={y_cuts}")

    print(f"[vae]     loading {matrix_run.VAE_NAME}")
    vae = ab_models.load_vae(matrix_run.VAE_NAME)

    # A REAL picture, not noise: a boundary step only shows against real structure. The market
    # canvas is the busiest of the five scenes, lanczos'd up to the probe geometry.
    scene = matrix_run.SCENES_BY_KEY["market"]
    base = matrix_run.stage_base(scene, None, None, None, force=False)
    canvas = matrix_run.stage_upscale(scene, base, force=False)
    from context_anchored_tile_refine import upscale
    padded = upscale._lanczos_resize(canvas[..., :3], CANVAS_W, CANVAS_H).clamp(0.0, 1.0)
    print(f"[input]   market upscaled to {tuple(padded.shape[1:3])}")

    print(f"\n{'margin':>8}{'window':>14}{'latent step':>13}{'pixel step':>13}")
    for margin in MARGINS:
        sync.VAE_ENCODE_MARGIN = margin
        try:
            with torch.inference_mode():
                c0 = sync.encode_canvas_latent(vae, padded, layout.tiles)
                decoded = vae.decode(c0)
        finally:
            sync.VAE_ENCODE_MARGIN = None
        if decoded.ndim == 5:
            decoded = decoded.reshape(-1, *decoded.shape[-3:])
        rect = sync.encode_window(layout.tiles[0], margin)
        latent_plane = c0.float().reshape(-1, c0.shape[-2], c0.shape[-1]).mean(dim=0)
        pixel_plane = decoded.float()[0].mean(dim=-1)
        latent = both_axes(latent_plane, [c // 8 for c in x_cuts], [c // 8 for c in y_cuts], 8)
        pixel = both_axes(pixel_plane, x_cuts, y_cuts, SPAN)
        label = "none" if margin is None else str(margin)
        print(f"{label:>8}{f'{rect.x1 - rect.x0}x{rect.y1 - rect.y0}':>14}"
              f"{latent:13.3f}{pixel:13.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
