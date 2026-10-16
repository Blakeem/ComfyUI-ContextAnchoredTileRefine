"""Inside each overlap band, how far do the two tiles' textures disagree, and how much detail survives?

Runs one run_ab_raster_noise render (same input, settings and arms) and reads every seam band
twice:
    disagreement   at the composite, the later tile's decode against the earlier tile's pixels
                   on the band cells: the mean absolute difference, the correlation of their
                   high-passed textures, and the DC offset the composite then removes
    band detail    in the final image, the high-pass energy of the band against the two
                   strips beside it, divided by the same ratio in the upscaled input, so 1.0
                   means the band kept the detail level of its neighbors and below 1.0 means
                   a smudged band

    python tests-AB/probe_raster_noise_band.py --arm per-tile --seed 42
    python tests-AB/probe_raster_noise_band.py --arm shipped --scene market --seed 42 --tag before

`--tag` names the record, so one command measures a scene before and after a production change.
"""
import argparse
import contextlib
import json
import statistics

import ab_env
import ab_models
import run_ab_raster_noise as ab

HIGH_PASS_KERNEL = 9


def high_pass(luma):
    import torch

    blurred = torch.nn.functional.avg_pool2d(luma[None, None], HIGH_PASS_KERNEL, stride=1,
                                             padding=HIGH_PASS_KERNEL // 2, count_include_pad=False)
    return (luma[None, None] - blurred)[0, 0]


def band_slices(tile, overlap):
    # Paste-local band slices: the left `overlap` columns and the top `overlap` rows of paste_rect.
    bands = []
    if tile.kept_left:
        bands.append(("left", (slice(None), slice(0, overlap))))
    if tile.kept_top:
        bands.append(("top", (slice(0, overlap), slice(None))))
    return bands


@contextlib.contextmanager
def disagreement_probe(overlap, records):
    """Read every band at the composite through the seam_dc_offset seam, which _refine_tiles
    calls with the tile, its decode over paste_rect and the canvas under it."""
    import torch

    from context_anchored_tile_refine import sampling

    original = sampling.seam_dc_offset

    def seam_dc_offset(tile, sub, region, region_mask=None):
        offset = original(tile, sub, region, region_mask)
        later = sub[0].float().mean(dim=-1)
        earlier = region[0].float().mean(dim=-1)
        later_detail, earlier_detail = high_pass(later), high_pass(earlier)
        for side, cells in band_slices(tile, overlap):
            a, b = later_detail[cells].flatten(), earlier_detail[cells].flatten()
            records.append({
                "tile": [tile.row, tile.col], "side": side,
                "mean_abs_diff_255": round(255.0 * (later[cells] - earlier[cells]).abs().mean().item(), 3),
                "texture_correlation": round(torch.corrcoef(torch.stack([a, b]))[0, 1].item(), 4),
                "dc_offset_255": None if offset is None else round(255.0 * offset.float().abs().mean().item(), 3),
            })
        return offset

    sampling.seam_dc_offset = seam_dc_offset
    try:
        yield
    finally:
        sampling.seam_dc_offset = original


def band_detail(result, source):
    """Per seam band, (band / mean of its two side strips) in the result over the same in the source."""
    from context_anchored_tile_refine import grid

    overlap = ab.CONTEXT_OVERLAP
    layout = grid.solve_layout(int(result.shape[2]), int(result.shape[1]), ab.MAX_TILE_WIDTH, ab.MAX_TILE_HEIGHT,
                               ab.CONTEXT_ANCHOR, ab.CONTEXT_OVERLAP)
    details = {name: high_pass(image[0].float().mean(dim=-1)) for name, image in (("result", result), ("source", source))}
    rows = []

    for tile in layout.tiles:
        core = tile.core
        strips = []
        if tile.kept_left:
            x = core.x0
            strips.append(("left", [(slice(core.y0, core.y1), slice(x - overlap, x)),
                                    (slice(core.y0, core.y1), slice(x - 2 * overlap, x - overlap)),
                                    (slice(core.y0, core.y1), slice(x, x + overlap))]))
        if tile.kept_top:
            y = core.y0
            strips.append(("top", [(slice(y - overlap, y), slice(core.x0, core.x1)),
                                   (slice(y - 2 * overlap, y - overlap), slice(core.x0, core.x1)),
                                   (slice(y, y + overlap), slice(core.x0, core.x1))]))
        for side, (band, before, after) in strips:
            ratios = {}
            for name, detail in details.items():
                energy = [detail[cells].std().item() for cells in (band, before, after)]
                ratios[name] = energy[0] / ((energy[1] + energy[2]) / 2.0)
            rows.append({"tile": [tile.row, tile.col], "side": side,
                         "band_detail": round(ratios["result"] / ratios["source"], 4)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", choices=ab.ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--scene", choices=sorted(ab.SCENES), default="dark-city")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    ab_env.bootstrap()

    import comfy.samplers
    import torch

    from context_anchored_tile_refine import sampling, upscale

    arm_scope = ab.arm_scope(args.arm, args.seed)
    records = []
    canvas = ab.upscaled_input(args.scene)
    empty = ab.empty_conditioning()
    model = ab_models.load_unet(ab.UNET)
    vae = ab_models.load_vae(ab.VAE)
    result = None

    with torch.inference_mode():
        guider = upscale.build_guider(model, empty, empty, ab.CFG)
        sigmas = upscale.build_sigmas(model, ab.SCHEDULER, ab.STEPS, ab.DENOISE)
        sampler = comfy.samplers.sampler_object(ab.SAMPLER)
        with arm_scope, disagreement_probe(ab.CONTEXT_OVERLAP, records):
            result = sampling.refine_image(canvas, guider, sampler, sigmas, vae, upscale.Noise_RandomNoise(args.seed),
                                           ab.MAX_TILE_WIDTH, ab.MAX_TILE_HEIGHT, ab.CONTEXT_ANCHOR, ab.CONTEXT_OVERLAP)
    result = result.float().cpu()
    detail = band_detail(result, canvas)

    summary = {
        "arm": args.arm, "seed": args.seed, "bands": len(records),
        "mean_abs_diff_255": round(statistics.mean(r["mean_abs_diff_255"] for r in records), 3),
        "texture_correlation": round(statistics.mean(r["texture_correlation"] for r in records), 4),
        "dc_offset_255": round(statistics.mean(r["dc_offset_255"] or 0.0 for r in records), 3),
        "band_detail": round(statistics.mean(r["band_detail"] for r in detail), 4),
    }
    suffix = f"__{args.tag}" if args.tag else ""
    out = ab.OUTPUT_DIR / f"band__{args.scene}__{args.arm}__s{args.seed}{suffix}.json"
    out.write_text(json.dumps({"summary": summary, "disagreement": records, "band_detail": detail}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
