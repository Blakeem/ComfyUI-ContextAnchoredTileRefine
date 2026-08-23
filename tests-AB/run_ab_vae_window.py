"""VAE WINDOW sweep: does shrinking what the VAE looks at change the picture?

The question. Both of the sync engine's VAE calls are sized to `crop_rect`, which is the
tile's SAMPLED extent, and neither keeps that much. Stage E keeps `core` alone, because C_0
is assembled from butted cores. Stage D keeps `paste_rect`. At the owner's shipped
`context_anchor 32` / `context_overlap 256` that discards 53% of every encode and 32% of
every decode, and every discarded cell was already computed by a neighbour whose core covers
it. Capping each window at "what this call keeps, plus a margin" removes that work and a
large slice of the VRAM reservation comfy hands `load_models_gpu`.

It is NOT free. Both `Encoder3d.middle` and `Decoder3d.middle` run a full spatial attention
block at the /8 bottleneck (comfy/ldm/wan/vae.py), whatever the empty `attn_scales` suggests,
so every kept cell reads the whole window and the framing error has no analytic bound. That
is why this is measured rather than argued, and why a bigger margin is not assumed to be
safer.

The prior worth knowing before judging stage D. `paste_rect` already sits exactly
`context_anchor` inside `crop_rect` on the TOP and LEFT borders, so today's decode already
runs with 32 px of context there, seam band included. A cap only ever binds on the right and
bottom, and any margin above `context_anchor` gives those borders MORE context than the top
and left already get.

The design differs per stage, because the two are not equally separable.

    --stage decode   Sampling is SHARED. One refine per scene produces one canvas latent and
                     every margin decodes that same latent, so the arms differ by the decode
                     window and by nothing else. The delta printed is the whole effect.
    --stage encode   Each margin is its own full refine, because the encode window sets every
                     cell of C_0 and the sampling follows it. The delta therefore carries
                     sampling divergence too, so the question here is not whether the pixels
                     match but whether the picture is as good. That one is the owner's eye.

Geometry is the AB-Final-Matrix scenes at their small scale (512x288 base, 3x to 1536x864)
with the owner's production overlap, which solves to 2x1 and puts ONE vertical seam at
x=768. Tile 0's right border IS that seam, and at margin 0 its decode window ends exactly on
it, which is the worst case this change can produce.

    <venv-python> tests-AB/run_ab_vae_window.py
    <venv-python> tests-AB/run_ab_vae_window.py --stage encode
    <venv-python> tests-AB/run_ab_vae_window.py --only market --only portrait

Outputs (output\\AB-VAE-Window\\), per scene:
    AB_vaewin__{scene}__{arm}.png          the render, one per margin (judge these)
    AB_vaewin__{scene}__{arm}-x16.png      the delta against that stage's control, x16
    AB_vaewin__report-{stage}.json         every number the run measured
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_env
import ab_models
import run_ab_krea2 as krea2_run  # sets COMFYUI_ROOT at import, before bootstrap()
import run_ab_matrix as matrix_run

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-VAE-Window")
RUN_TAG = "vaewin"

UNET_NAME = matrix_run.UNET_NAME
CLIP_NAME = matrix_run.CLIP_NAME
CLIP_TYPE = matrix_run.CLIP_TYPE
VAE_NAME = matrix_run.VAE_NAME

# The owner's shipped 8K pass 2, at the matrix's small scale. anchor 32 / overlap 256 is the
# production pair and is what makes the discarded decode border 288 px wide. Caps 1088x896
# solve 2x1 at 1536x864, asserted before any model loads.
SETTINGS = krea2_run.Refine(
    seed=42, sampler="dpmpp_2m_sde", scheduler="sgm_uniform", steps=20, cfg=3.5,
    denoise=0.35, upscale_by=3.0, max_tile_width=1088, max_tile_height=896,
    context_anchor=32, context_overlap=256)
VLM_METHOD = "vision tokens and captions"      # the default preset, the production surface
ANCHOR_SOURCE = "source image"                 # the production ring, pass 2
EXPECTED_GRID = (2, 1)
SEAM_X = 768                                   # asserted against the solver
SEAM_HALF_WIDTH = 128                          # the band the seam metrics read

# None is the shipped window, the whole crop_rect. Every other value caps the window at the
# rect that stage keeps plus the margin. 0 is the floor, the kept rect exactly.
MARGINS = (None, 128, 64, 32, 0)
ARM_OF = {margin: ("control" if margin is None else f"m{margin}") for margin in MARGINS}
# Stage E cannot share one sampling pass. Its window sets C_0, so every cell of the canvas
# latent moves and the whole render follows, seed interaction included. Three arms only,
# because each is a full refine.
ENCODE_MARGINS = (None, 64, 0)
ENCODE_ARM_OF = {margin: ("enc-control" if margin is None else f"enc-m{margin}")
                 for margin in ENCODE_MARGINS}


def output_path(scene_key, arm, suffix=""):
    return OUTPUT_DIR / f"AB_{RUN_TAG}__{scene_key}__{arm}{suffix}.png"


def solve_tiles(canvas):
    from context_anchored_tile_refine import grid, sampling

    padded, _ = sampling.pad_image_to_multiple(canvas[..., :3])
    sx = grid.solve_axis(padded.shape[2], SETTINGS.max_tile_width,
                         SETTINGS.context_anchor, SETTINGS.context_overlap, axis="width")
    sy = grid.solve_axis(padded.shape[1], SETTINGS.max_tile_height,
                         SETTINGS.context_anchor, SETTINGS.context_overlap, axis="height")
    if (sx.n, sy.n) != EXPECTED_GRID:
        raise SystemExit(f"grid {sx.n}x{sy.n} != expected {EXPECTED_GRID} — caps drifted")
    if sx.base != SEAM_X:
        raise SystemExit(f"seam at x={sx.base}, expected {SEAM_X}")
    return grid.build_layout(padded.shape[2], padded.shape[1], sx, sy,
                             SETTINGS.context_anchor, SETTINGS.context_overlap)


def window_pixels(layout, margin, stage="decode"):
    from context_anchored_tile_refine import sync

    window_of = sync.encode_window if stage == "encode" else sync.decode_window
    total = 0
    for tile in layout.tiles:
        rect = window_of(tile, margin)
        total += (rect.x1 - rect.x0) * (rect.y1 - rect.y0)
    return total


def delta_stats(image, control):
    # Everything in /255, which is the unit every seam finding in this repo is quoted in.
    diff = (image - control).abs() * 255.0
    band = diff[:, :, max(0, SEAM_X - SEAM_HALF_WIDTH):SEAM_X + SEAM_HALF_WIDTH, :]
    return {
        "max": float(diff.max()),
        "mean": float(diff.mean()),
        "seam_band_max": float(band.max()),
        "seam_band_mean": float(band.mean()),
        "changed_px_frac": float((diff.amax(dim=-1) > 0.5).float().mean()),
    }


def refine_and_capture(scene, image, clip, model, vae, negative, empty):
    """One production refine, with the canvas latent intercepted on its way into stage D.

    sampling.refine_image is the real entry point every node uses, so what is captured here
    is the canvas the shipped engine actually built. decode_composite runs once inside (the
    control arm), and the captured arguments then drive every other margin.
    """
    from context_anchored_tile_refine import sampling, sync, upscale

    sigmas = upscale.build_sigmas(model, SETTINGS.scheduler, SETTINGS.steps, SETTINGS.denoise)
    guider = upscale.build_guider(model, empty, negative, SETTINGS.cfg)
    sampler = ab_models.build_sampler(SETTINGS.sampler)
    noise = upscale.Noise_RandomNoise(SETTINGS.seed)

    captured = {}
    real_decode = sync.decode_composite

    def capture(vae_arg, model_arg, canvas, padded, layout, progress=None):
        captured.update(vae=vae_arg, model=model_arg, canvas=canvas.clone(),
                        padded=padded.clone(), layout=layout)
        return real_decode(vae_arg, model_arg, canvas, padded, layout, progress=progress)

    sync.decode_composite = capture
    try:
        started = time.perf_counter()
        with torch.inference_mode():
            control = sampling.refine_image(
                image, guider, sampler, sigmas, vae, noise,
                SETTINGS.max_tile_width, SETTINGS.max_tile_height,
                SETTINGS.context_anchor, SETTINGS.context_overlap,
                vl_clip=clip, vlm_method=VLM_METHOD, anchor_source=ANCHOR_SOURCE)
        refine_seconds = time.perf_counter() - started
    finally:
        sync.decode_composite = real_decode
    if not captured:
        raise SystemExit("stage D never ran — the sync engine did not take the dispatch")
    return control, captured, real_decode, refine_seconds, sigmas


def decode_at(real_decode, captured, margin):
    from context_anchored_tile_refine import sampling, sync

    previous = sync.VAE_DECODE_MARGIN
    sync.VAE_DECODE_MARGIN = margin
    try:
        ab_models.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with ab_models.VramProbe() as probe, torch.inference_mode():
            decoded = real_decode(captured["vae"], captured["model"], captured["canvas"],
                                  captured["padded"], captured["layout"])
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - started
    finally:
        sync.VAE_DECODE_MARGIN = previous
    padded = captured["padded"]
    cropped = sampling.crop_image_to(decoded, int(padded.shape[1]), int(padded.shape[2]))
    return cropped, seconds, str(probe)


def refine_full(scene, image, clip, model, vae, negative, empty, encode_margin):
    """One whole production refine at a given stage E window, with that stage timed and probed.

    Stage E is not separable the way stage D was. Its window sets every cell of C_0, so each
    arm is its own render and the delta below carries the sampling divergence too, which is
    the honest question for this half: not whether the pixels match, but whether the picture
    is as good.
    """
    from context_anchored_tile_refine import sampling, sync, upscale

    sigmas = upscale.build_sigmas(model, SETTINGS.scheduler, SETTINGS.steps, SETTINGS.denoise)
    guider = upscale.build_guider(model, empty, negative, SETTINGS.cfg)
    sampler = ab_models.build_sampler(SETTINGS.sampler)
    noise = upscale.Noise_RandomNoise(SETTINGS.seed)

    stats = {}
    real_encode = sync.encode_canvas_latent

    def timed_encode(vae_arg, padded, tiles, progress=None):
        ab_models.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()
        with ab_models.VramProbe() as probe:
            latent = real_encode(vae_arg, padded, tiles, progress=progress)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stats["encode_seconds"] = time.perf_counter() - started
        stats["encode_vram"] = str(probe)
        return latent

    sync.encode_canvas_latent = timed_encode
    sync.VAE_ENCODE_MARGIN = encode_margin
    try:
        started = time.perf_counter()
        with torch.inference_mode():
            result = sampling.refine_image(
                image, guider, sampler, sigmas, vae, noise,
                SETTINGS.max_tile_width, SETTINGS.max_tile_height,
                SETTINGS.context_anchor, SETTINGS.context_overlap,
                vl_clip=clip, vlm_method=VLM_METHOD, anchor_source=ANCHOR_SOURCE)
        stats["refine_seconds"] = time.perf_counter() - started
    finally:
        sync.encode_canvas_latent = real_encode
        sync.VAE_ENCODE_MARGIN = None
    return result, stats, sigmas


def run_scene_encode(scene, clip, model, vae, empty, report):
    print(f"\n=== {scene.key}: {scene.title}   [stage E]")
    base = matrix_run.stage_base(scene, model, clip, vae, force=False)
    canvas = matrix_run.stage_upscale(scene, base, force=False)
    layout = solve_tiles(canvas)
    with torch.inference_mode():
        negative = ab_models.encode_prompt(clip, scene.negative)

    control_px = window_pixels(layout, None, stage="encode")
    control = None
    for margin in ENCODE_MARGINS:
        arm = ENCODE_ARM_OF[margin]
        image, stats, sigmas = refine_full(scene, canvas, clip, model, vae, negative, empty, margin)
        ab_models.require_image_shape(image, canvas.shape[1], canvas.shape[2], f"{scene.key} {arm}")
        if control is None:
            control = image
        delta = delta_stats(image, control)
        px = window_pixels(layout, margin, stage="encode")
        payload = settings_payload(scene, arm, margin, layout, sigmas, delta,
                                   stats["encode_seconds"], stage="encode")
        payload["encode_vram"] = stats["encode_vram"]
        payload["refine_seconds"] = round(stats["refine_seconds"], 2)
        ab_models.save_png(output_path(scene.key, arm), image.cpu(), payload)
        if margin is not None:
            stretched = ((image - control).abs() * 16.0).clamp(0.0, 1.0)
            ab_models.save_png(output_path(scene.key, arm, "-x16"), stretched.cpu(),
                               {"stage": "delta vs enc-control, stretched x16", "arm": arm})
        report.append({"scene": scene.key, "arm": arm, "margin": margin, "decode_px": px,
                       "decode_px_vs_control": round(px / control_px, 4),
                       "decode_seconds": round(stats["encode_seconds"], 3),
                       "refine_seconds": round(stats["refine_seconds"], 2),
                       "vram": stats["encode_vram"], **delta})
        print(f"[stageE]  {arm:<12} encode {stats['encode_seconds']:5.2f}s  "
              f"window {px / 1e6:5.3f} MP ({100 * px / control_px:5.1f}%)  "
              f"refine {stats['refine_seconds']:6.1f}s  max {delta['max']:6.2f}/255  "
              f"mean {delta['mean']:6.3f}  changed {100 * delta['changed_px_frac']:5.1f}%")
        print(f"[stageE]  {arm:<12} {stats['encode_vram']}")


def settings_payload(scene, arm, margin, layout, sigmas, stats, seconds, stage="decode"):
    kept = "core" if stage == "encode" else "paste_rect"
    payload = dataclasses.asdict(SETTINGS)
    payload.update({
        "run_label": f"{scene.key}__{arm}",
        "arm": (f"control: the shipped stage {stage[0].upper()} window, the whole crop_rect"
                if margin is None else
                f"stage {stage} window capped at {kept} + {margin} px, clamped inside crop_rect"),
        "vae_window_stage": stage,
        "vae_window_margin": margin,
        "method": ("synchronized latent tiling (context_anchored_tile_refine.sync). Stage D "
                   "arms share ONE canvas latent, so only the decode window differs. Stage E "
                   "arms are separate renders, because the encode window sets C_0."),
        "vlm_method": VLM_METHOD,
        "anchor_source": ANCHOR_SOURCE,
        "vae_window_px": window_pixels(layout, margin, stage=stage),
        "decode_seconds": round(seconds, 3),
        "delta_vs_control_255": stats,
        "seam_x": SEAM_X,
        "scene": scene.key,
        "gen": matrix_run.gen_key(scene),
        "sigmas": [float(v) for v in sigmas],
        "unet": UNET_NAME, "clip": CLIP_NAME, "clip_type": CLIP_TYPE, "vae": VAE_NAME,
        "upscale_model": matrix_run.UPSCALE_MODEL_NAME,
        "negative_prompt": scene.negative,
        "harness": "tests-AB/run_ab_vae_window.py",
        "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return payload


def run_scene(scene, clip, model, vae, empty, report):
    print(f"\n=== {scene.key}: {scene.title}")
    base = matrix_run.stage_base(scene, model, clip, vae, force=False)
    canvas = matrix_run.stage_upscale(scene, base, force=False)
    layout = solve_tiles(canvas)
    print(f"[layout]  {canvas.shape[2]}x{canvas.shape[1]}  {len(layout.tiles)} tiles, "
          f"seam x={SEAM_X}, anchor {SETTINGS.context_anchor} overlap {SETTINGS.context_overlap}")
    for index, tile in enumerate(layout.tiles):
        crop, paste = tile.crop_rect, tile.paste_rect
        print(f"[layout]  tile {index}: keeps {paste.x1 - paste.x0}x{paste.y1 - paste.y0}, "
              f"decodes {crop.x1 - crop.x0}x{crop.y1 - crop.y0} today")

    with torch.inference_mode():
        negative = ab_models.encode_prompt(clip, scene.negative)
    control, captured, real_decode, refine_seconds, sigmas = refine_and_capture(
        scene, canvas, clip, model, vae, negative, empty)
    print(f"[refine]  whole refine {refine_seconds:.1f}s")

    control_px = window_pixels(layout, None)
    rows = []
    for margin in MARGINS:
        arm = ARM_OF[margin]
        image, seconds, probe = decode_at(real_decode, captured, margin)
        ab_models.require_image_shape(image, canvas.shape[1], canvas.shape[2],
                                      f"{scene.key} {arm}")
        stats = ({"max": 0.0, "mean": 0.0, "seam_band_max": 0.0, "seam_band_mean": 0.0,
                  "changed_px_frac": 0.0} if margin is None else delta_stats(image, control))
        px = window_pixels(layout, margin)
        ab_models.save_png(output_path(scene.key, arm), image.cpu(),
                           settings_payload(scene, arm, margin, layout, sigmas, stats, seconds))
        if margin is not None:
            stretched = ((image - control).abs() * 16.0).clamp(0.0, 1.0)
            ab_models.save_png(output_path(scene.key, arm, "-x16"), stretched.cpu(),
                               {"stage": "delta vs control, stretched x16", "arm": arm})
        rows.append({"scene": scene.key, "arm": arm, "margin": margin, "decode_px": px,
                     "decode_px_vs_control": round(px / control_px, 4),
                     "decode_seconds": round(seconds, 3), "vram": probe, **stats})
        print(f"[decode]  {arm:<8} {seconds:6.2f}s  window {px / 1e6:5.3f} MP "
              f"({100 * px / control_px:5.1f}%)  max {stats['max']:6.2f}/255  "
              f"mean {stats['mean']:5.3f}  seam max {stats['seam_band_max']:6.2f}  "
              f"changed {100 * stats['changed_px_frac']:5.1f}%")
    report.extend(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", action="append", default=[], metavar="SCENE",
                        help="render only these scenes (repeatable)")
    parser.add_argument("--stage", choices=("decode", "encode"), default="decode",
                        help="which VAE window to sweep (default: decode)")
    parser.add_argument("--list", action="store_true", help="show the scenes and exit")
    args = parser.parse_args(argv)

    keys = [scene.key for scene in matrix_run.SCENES]
    unknown = set(args.only) - set(keys)
    if unknown:
        raise SystemExit("unknown scene(s): {}".format(", ".join(sorted(unknown))))
    selected = [scene for scene in matrix_run.SCENES if not args.only or scene.key in args.only]
    if args.list:
        for scene in matrix_run.SCENES:
            marker = "*" if scene in selected else " "
            print(f"{marker} {scene.key:<12} {scene.title}")
        return 0

    root, note = ab_env.bootstrap()
    print(f"[env]     ComfyUI {ab_env.version(root)} at {root}  ({note})")
    print(f"[env]     torch {torch.__version__}  device "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    from context_anchored_tile_refine import grid, upscale
    target_w, target_h = upscale.scale_target(matrix_run.GEN_WIDTH, matrix_run.GEN_HEIGHT,
                                              matrix_run.UPSCALE_BY)
    sx = grid.solve_axis(target_w, SETTINGS.max_tile_width, SETTINGS.context_anchor,
                         SETTINGS.context_overlap, axis="width")
    sy = grid.solve_axis(target_h, SETTINGS.max_tile_height, SETTINGS.context_anchor,
                         SETTINGS.context_overlap, axis="height")
    if (sx.n, sy.n) != EXPECTED_GRID:
        raise SystemExit(f"grid {sx.n}x{sy.n} != {EXPECTED_GRID} at {target_w}x{target_h}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[clip]    loading {CLIP_NAME} ({CLIP_TYPE})")
    clip = ab_models.load_clip(CLIP_NAME, CLIP_TYPE)
    print(f"[unet]    loading {UNET_NAME}")
    model = ab_models.load_unet(UNET_NAME)
    vae = ab_models.load_vae(VAE_NAME)
    with torch.inference_mode():
        empty = upscale.encode_empty(clip)

    report = []
    for scene in selected:
        if args.stage == "encode":
            run_scene_encode(scene, clip, model, vae, empty, report)
        else:
            run_scene(scene, clip, model, vae, empty, report)

    (OUTPUT_DIR / f"AB_{RUN_TAG}__report-{args.stage}.json").write_text(
        json.dumps({"settings": dataclasses.asdict(SETTINGS), "vlm_method": VLM_METHOD,
                    "anchor_source": ANCHOR_SOURCE, "seam_x": SEAM_X, "stage": args.stage,
                    "rows": report},
                   indent=1), encoding="utf-8")

    print("\n=== summary, delta against each scene's own control, in /255")
    print(f"{'scene':<12}{'arm':<9}{'window':>9}{'decode':>9}{'max':>8}{'mean':>8}"
          f"{'seam max':>10}{'changed':>9}")
    for row in report:
        print(f"{row['scene']:<12}{row['arm']:<9}{100 * row['decode_px_vs_control']:8.1f}%"
              f"{row['decode_seconds']:8.2f}s{row['max']:8.2f}{row['mean']:8.3f}"
              f"{row['seam_band_max']:10.2f}{100 * row['changed_px_frac']:8.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
