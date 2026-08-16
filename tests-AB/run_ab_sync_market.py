"""Sync-tiles STRESS TEST on the market scene — the workflow's original hardest case.

Owner's spec (2026-08-15): the Renaissance-market prompt (run_ab_matrix scene
"market", verbatim), base GENERATED at 1024x576 (16:9), 4x upscale -> 4096x2304,
SIX tiles (3x2), denoise 0.50 ("I was unable to get this working 100% of the time
before, so it's a real stress test"), sampler dpmpp_2m ("best speed with still good
quality"), scheduler sgm_uniform, context_overlap 32 / context_anchor 128 ("a good
balance in the past and a bit faster"). Method: the sync engine (run_ab_sync.py),
ring mode "lead" — arm 22's judged config (proper age, accurate detail, beats
9-cap-time-switch50, the previous best).

    23-sync-market-lead   the one arm. Judge: seams across SIX tile joints (incl.
                          TWO interior crossings), duplicated people/stalls/banners,
                          crowd coherence across tiles, the cathedral dome (spans
                          the top band), tone vs the upscaled canvas.

What is NEW vs run_ab_sync (and why it needed engine work, not just constants):
  * dpmpp_2m is a MULTI-STEP sampler — its 2M correction consumes the PREVIOUS
    step's denoised and sigmas[i-1]. Naive per-step calls would restart that state
    every step and silently degrade the whole run to first order. The engine's
    _dpmpp_2m_single_step carries {old_denoised, sigma_prev} per tile across the 28
    chained calls; the scratch selftest pinned chained-vs-whole BIT-IDENTICAL on a
    fake model before this harness existed. Deterministic: StepNoise is not used
    (step-0 construction noise is the canvas draw, sampler-independent).
  * The base is a NEW DRAW: same prompt/negative/sampler/steps/cfg/seed as the
    matrix chain but at 1024x576 — a different picture from the matrix's 512x288
    base (size changes the draw), same subject. Cached; the base and upscaled
    canvas PNGs are saved next to the render so structure keeps its reference.
  * Geometry solved and asserted 3x2: caps 1728x1408 with anchor 128 / overlap 32
    give cores 1368/1360 x 1152, crops <= 1688x1312 (verified via grid.solve_axis
    before this file was written; the run re-asserts).

Everything else IS run_ab_sync's engine, imported: C_0 from butted window encodes,
per-step resume identity, ring = frozen RAW + canvas epsilon presented on the
shipped lead curve (lead_ring_patch, run-global sigma_first), per-step latent
feather (seam cell pinned 1.0), stock pixel feather at decode, no DC / no cut.
Conditioning: arm 2's builder (sliced vision rows + text-only captions), the
settled instruction, captions cached per (gen digest, rects).

    <venv-python> tests-AB/run_ab_sync_market.py                  # pending arms (24)
    <venv-python> tests-AB/run_ab_sync_market.py --only 24-sync-market-o128
    <venv-python> tests-AB/run_ab_sync_market.py --list

Outputs (output\\AB-Test-Images\\):
    AB_market1024-d050__23-sync-market-lead.png   arm 23 (judged: best ever on this
                                                  scene; one weird arm IN the 32px
                                                  both-diffused band — band-map
                                                  confirmed by the owner)
    AB_market1024-d050__24-sync-market-o128.png   arm 24 (judge the arm vs 23)
    AB_market1024-d050__scene-base.png            the 1024x576 base draw
    AB_market1024-d050__scene-canvas.png          the 4x upscaled refine input
    (the 23-suffixed -base/-canvas pair from the first render stays on disk)
"""
import argparse
import dataclasses
import hashlib
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
import run_ab_split as split_run
import run_ab_sync as sync_run

OUTPUT_DIR = split_run.OUTPUT_DIR
CACHE_DIR = split_run.CACHE_DIR
RUN_TAG = "market1024-d050"

SCENE = matrix_run.SCENES_BY_KEY["market"]
UNET_NAME = matrix_run.UNET_NAME
CLIP_NAME = matrix_run.CLIP_NAME
CLIP_TYPE = matrix_run.CLIP_TYPE
VAE_NAME = matrix_run.VAE_NAME
UPSCALE_MODEL_NAME = matrix_run.UPSCALE_MODEL_NAME

# Base generation: the matrix chain verbatim except the size (owner: 1024x576).
GEN_WIDTH, GEN_HEIGHT = 1024, 576
GEN_SEED = matrix_run.GEN_SEED
GEN_STEPS = matrix_run.GEN_STEPS
GEN_SAMPLER = matrix_run.GEN_SAMPLER
GEN_SCHEDULER = matrix_run.GEN_SCHEDULER
GEN_CFG = matrix_run.CFG

# The stress refine (owner's spec). Caps verified to yield the 3x2 grid.
SETTINGS = krea2_run.Refine(
    seed=42, sampler="dpmpp_2m", scheduler="sgm_uniform", steps=28, cfg=3.5,
    denoise=0.50, upscale_by=4.0, max_tile_width=1728, max_tile_height=1408,
    context_anchor=128, context_overlap=32)
EXPECTED_GRID = (3, 2)          # (columns, rows) — asserted, never assumed
RING_MODE = "lead"              # arm 22's judged config

ARMS = {
    "23-sync-market-lead": "sync tiles on the market stress scene: 6 tiles, "
                           "dpmpp_2m, anchor 128 / overlap 32, lead raw ring — "
                           "arm 22's method at the workflow's hardest case",
    "24-sync-market-o128": "arm 23 with context_overlap 32 -> 128 (caps 1920x1472 "
                           "keep the same 3x2 grid, SAME cores/seam lines) — the "
                           "ONLY change vs 23. Mechanism target: the weird arm sits "
                           "in the 32px both-diffused band, negotiated by two tiles "
                           "whose far context is opposed raw mush; at 128 an "
                           "arm-sized structure fits INSIDE the co-owned band and "
                           "each tile sees 4x more live trajectory before its ring "
                           "begins. Also the owner's planned 8K-config knob.",
}

# arm -> refine settings. Overlap is the ONLY sampling change in 24; the caps rise
# purely so the wider crops (up to 1880x1408) still solve to the SAME 3x2 grid —
# verified: cores and seam lines are byte-identical to arm 23's, so the defect site
# is directly comparable.
ARM_SETTINGS = {
    "23-sync-market-lead": SETTINGS,
    "24-sync-market-o128": dataclasses.replace(
        SETTINGS, context_overlap=128, max_tile_width=1920, max_tile_height=1472),
}


def output_path(arm):
    return OUTPUT_DIR / f"AB_{RUN_TAG}__{arm}.png"


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ pixel stages

def gen_key():
    return {"prompt": SCENE.positive, "negative": SCENE.negative, "seed": GEN_SEED,
            "sampler": GEN_SAMPLER, "scheduler": GEN_SCHEDULER, "steps": GEN_STEPS,
            "cfg": GEN_CFG, "w": GEN_WIDTH, "h": GEN_HEIGHT, "unet": UNET_NAME}


def stage_base(model, clip, vae, force):
    """The 1024x576 market base, generated once and cached (matrix stage_base's chain
    at the owner's size — a different draw from the matrix's 512x288 base)."""
    from context_anchored_tile_refine import upscale

    cached = ab_models.cache_path(CACHE_DIR, "market1024", "base", GEN_WIDTH, GEN_HEIGHT, gen_key())
    if cached.is_file() and not force:
        print(f"[base]    cache hit  {cached.name}")
        return torch.load(cached, map_location="cpu")

    with ab_models.VramProbe() as probe, torch.inference_mode():
        positive = ab_models.encode_prompt(clip, SCENE.positive)
        negative = ab_models.encode_prompt(clip, SCENE.negative)
        guider = upscale.build_guider(model, positive, negative, GEN_CFG)
        sigmas = ab_models.build_sigmas(model, GEN_SCHEDULER, GEN_STEPS, 1.0)
        sampler = ab_models.build_sampler(GEN_SAMPLER)
        noise = ab_models.build_noise(GEN_SEED)
        latent = ab_models.empty_sd3_latent(GEN_WIDTH, GEN_HEIGHT, 1)
        out = ab_models.sample_custom_advanced(noise, guider, sampler, sigmas, latent)
        base = ab_models.vae_decode(vae, out).detach().float().cpu()
    ab_models.require_image_shape(base, GEN_HEIGHT, GEN_WIDTH, "market base gen")
    cached.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, cached)
    print(f"[base]    generated {GEN_WIDTH}x{GEN_HEIGHT}  {probe}  -> {cached.name}")
    return base


def stage_upscale(base, force):
    """4x: model pass (4xFaceUpDAT, the established chain) + lanczos to exactly 4x."""
    from context_anchored_tile_refine import upscale

    target_w, target_h = upscale.scale_target(GEN_WIDTH, GEN_HEIGHT, SETTINGS.upscale_by)
    key = {"gen": _digest(gen_key()), "model": UPSCALE_MODEL_NAME,
           "upscale_by": SETTINGS.upscale_by, "stage": "prepare_upscaled"}
    cached = ab_models.cache_path(CACHE_DIR, "market1024", "upscale", target_w, target_h, key)

    if cached.is_file() and not force:
        canvas = torch.load(cached, map_location="cpu")
        print(f"[upscale] cache hit  {cached.name}")
    else:
        print(f"[upscale] {GEN_WIDTH}x{GEN_HEIGHT} -> 4x {UPSCALE_MODEL_NAME} -> lanczos {target_w}x{target_h}")
        upscale_model = matrix_run._load_upscale_model()
        try:
            with ab_models.VramProbe() as probe, torch.inference_mode():
                canvas = upscale.prepare_upscaled(base, upscale_model, SETTINGS.upscale_by)
        finally:
            del upscale_model
            ab_models.free_gpu()
        canvas = canvas.detach().float().cpu()
        cached.parent.mkdir(parents=True, exist_ok=True)
        torch.save(canvas, cached)
        print(f"[upscale] done  {probe}  -> {cached.name}")
    ab_models.require_image_shape(canvas, target_h, target_w, "market upscaled canvas")
    return canvas


def solve_tiles(canvas, settings):
    from context_anchored_tile_refine import grid, sampling

    pixels = canvas[..., :3]
    padded, _ = sampling.pad_image_to_multiple(pixels)
    sx = grid.solve_axis(padded.shape[2], settings.max_tile_width,
                         settings.context_anchor, settings.context_overlap, axis="width")
    sy = grid.solve_axis(padded.shape[1], settings.max_tile_height,
                         settings.context_anchor, settings.context_overlap, axis="height")
    layout = grid.build_layout(padded.shape[2], padded.shape[1], sx, sy,
                               settings.context_anchor, settings.context_overlap)
    print(f"[layout]  canvas {padded.shape[2]}x{padded.shape[1]}  grid {sx.n}x{sy.n}  "
          f"({len(layout.tiles)} tiles, overlap {settings.context_overlap})")
    if (sx.n, sy.n) != EXPECTED_GRID:
        raise SystemExit(f"grid {sx.n}x{sy.n} != expected {EXPECTED_GRID} — caps drifted "
                         "from the 6-tile spec")
    return padded, layout.tiles


def load_or_generate_captions(clip, padded, tiles, force):
    """The production caption pre-pass (settled instruction), cached per
    (gen digest, rects) — split_run's pattern on this scene's own key."""
    from context_anchored_tile_refine import captions

    # Pinned to the SETTLED_POSITION wording these arms were judged with, NOT resolved
    # through captions.CAPTION_INSTRUCTIONS: that table moved VLM_METHOD_VISION_CAPTIONS to
    # RICH_GROUPED on 2026-08-16, so resolving it now would silently re-caption this campaign
    # with different text while every label and cache key still said "settled-position".
    instruction, max_length = captions.SETTLED_POSITION_INSTRUCTION, captions.SETTLED_POSITION_MAX_TOKENS
    key = {"gen": _digest(gen_key()), "rects": krea2_run._rects(tiles),
           "instruction": instruction, "max_length": max_length, "clip": CLIP_NAME,
           "surface": "settled-position"}
    cached = CACHE_DIR / f"market1024_sync_captions_{_digest(key)}.json"

    if cached.is_file() and not force:
        tile_captions = json.loads(cached.read_text(encoding="utf-8"))["captions"]
        print(f"[caption] cache hit  {cached.name}")
    else:
        with torch.inference_mode():
            tile_captions = captions.generate_tile_captions(clip, padded, tiles, instruction, max_length)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(dict(key, captions=tile_captions), indent=1), encoding="utf-8")
        print(f"[caption] generated and cached  {cached.name}")
    for tile_idx, row_captions in enumerate(tile_captions):
        print(f"[caption] tile {tile_idx}: {row_captions[0]}")
    return tile_captions


# ------------------------------------------------------------------ render

def build_settings(arm, tile_captions, sigmas, settings):
    payload = dataclasses.asdict(settings)
    payload.update({
        "run_label": arm,
        "arm": ARMS[arm],
        "method": "synchronized tiles (run_ab_sync engine): one pass, 6 tiles "
                  "stepped together per sigma on one shared canvas latent; "
                  "dpmpp_2m per-step with cross-step state carried per tile; "
                  "per-step directional feather in latent space",
        "conditioning": "arm 2: sliced vision rows + text-only captions",
        "sigmas": [float(v) for v in sigmas],
        "ring": "frozen RAW canvas on the SHIPPED lead curve (run-global "
                "sigma_first) — arm 22's judged config",
        "anchor_type": "lead (run-global sigma_first; see 'ring')",
        "seam_machinery": "none: no DC match, no min-error cut — bands couple every "
                          "step and decode one shared latent",
        "scene": "market (run_ab_matrix), base regenerated at 1024x576",
        "gen": gen_key(),
        "expected_grid": list(EXPECTED_GRID),
        "unet": UNET_NAME, "clip": CLIP_NAME, "clip_type": CLIP_TYPE, "vae": VAE_NAME,
        "upscale_model": UPSCALE_MODEL_NAME,
        "guider": "CFGGuider",
        "negative_prompt": SCENE.negative,
        "tile_captions": [row[0] for row in tile_captions],
        "harness": "tests-AB/run_ab_sync_market.py",
        "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return payload


def render(arm, padded, tiles, clip, model, vae, negative, empty, tile_captions):
    import contextlib

    from context_anchored_tile_refine import upscale

    settings = ARM_SETTINGS[arm]
    sigmas = upscale.build_sigmas(model, settings.scheduler, settings.steps, settings.denoise)
    steps = int(sigmas.shape[-1]) - 1
    sync_run.check_sync_preconditions(model, sigmas)

    canvas_h, canvas_w = int(padded.shape[1]), int(padded.shape[2])
    print(f"[render]  {arm}: {steps} synchronized steps over {len(tiles)} tiles, "
          f"sigma {float(sigmas[0]):.4f} -> 0, canvas {canvas_w}x{canvas_h}, "
          f"sampler {settings.sampler}, overlap {settings.context_overlap}, "
          f"ring '{RING_MODE}'")

    timings = {}
    total_started = time.perf_counter()

    started = time.perf_counter()
    with torch.inference_mode():
        tile_positives = split_run.make_split_builder("text-only")(
            clip, padded, tiles, tile_captions)
    timings["conds"] = time.perf_counter() - started

    started = time.perf_counter()
    ab_models.clear_cache()
    with ab_models.VramProbe() as probe, torch.inference_mode():
        canvas_latent = sync_run.encode_canvas_latent(vae, padded, tiles)
    timings["encode"] = time.perf_counter() - started
    print(f"[encode]  C_0 assembled {tuple(canvas_latent.shape)} in {timings['encode']:.1f}s  {probe}")

    latent_time = (1,) if getattr(vae, "latent_dim", 2) == 3 else ()
    dummy = torch.zeros((1, vae.latent_channels, *latent_time, canvas_h // 8, canvas_w // 8),
                        dtype=torch.float32)
    canvas_noise = upscale.Noise_RandomNoise(settings.seed).generate_noise({"samples": dummy})
    if tuple(canvas_noise.shape) != tuple(canvas_latent.shape):
        raise RuntimeError(f"canvas noise {tuple(canvas_noise.shape)} != latent canvas "
                           f"{tuple(canvas_latent.shape)}")

    raw_latent = canvas_latent.clone()
    ring_patch = (sync_run.lead_ring_patch(model, float(sigmas[0])) if RING_MODE == "lead"
                  else contextlib.nullcontext())
    guider = upscale.build_guider(model, empty, negative, settings.cfg)
    with ab_models.VramProbe() as probe, torch.inference_mode(), ring_patch:
        canvas_latent = sync_run.sync_refine(canvas_latent, canvas_noise, sigmas, tiles,
                                             guider, tile_positives, timings,
                                             ring_mode=RING_MODE, raw_latent=raw_latent,
                                             settings=settings)
    print(f"[sync]    {steps} steps x {len(tiles)} tiles in {timings['sync-steps']:.1f}s  {probe}")

    ab_models.clear_cache()
    with ab_models.VramProbe() as probe, torch.inference_mode():
        result = sync_run.decode_composite(vae, canvas_latent, padded, tiles, timings,
                                           overlap=settings.context_overlap)
    print(f"[decode]  composite in {timings['decode-composite']:.1f}s  {probe}")

    ab_models.require_image_shape(result, canvas_h, canvas_w, "market sync refine")
    raw_luma = sync_run.mean_luma(padded)
    # Delta vs THIS scene's own upscaled canvas — cross-scene luma comparison is not
    # meaningful (review F8), so no 00676 reference numbers here.
    print(f"[tone]    final luma delta vs raw canvas: "
          f"{sync_run.mean_luma(result) - raw_luma:+.2f}/255")
    destination = output_path(arm)
    written = ab_models.save_png(destination, result.cpu(),
                                 build_settings(arm, tile_captions, sigmas, settings))
    timings["total"] = time.perf_counter() - total_started
    print(f"[render]  {arm} -> {destination.name} {written[0]}x{written[1]}")
    print("[timing]  " + "  ".join(f"{name}={seconds:.1f}s" for name, seconds in timings.items()))


# ------------------------------------------------------------------ main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", action="append", default=[], metavar="ARM",
                        help="render only these arms (repeatable)")
    parser.add_argument("--force", action="store_true", help="re-render existing outputs")
    parser.add_argument("--force-base", action="store_true", help="regenerate the cached base draw")
    parser.add_argument("--force-captions", action="store_true", help="regenerate the cached captions")
    parser.add_argument("--list", action="store_true", help="show the arms and exit")
    args = parser.parse_args(argv)

    unknown = set(args.only) - set(ARMS)
    if unknown:
        raise SystemExit("unknown arm(s): {}".format(", ".join(sorted(unknown))))
    selected = [arm for arm in ARMS if not args.only or arm in args.only]

    if args.list:
        for arm, description in ARMS.items():
            marker = "*" if arm in selected else " "
            print(f"{marker} {arm:<22} {description}  -> {output_path(arm).name}")
        return 0

    root, note = ab_env.bootstrap()
    print(f"[env]     ComfyUI {ab_env.version(root)} at {root}  ({note})")
    print("[env]     torch {}  cuda {}  device {}".format(
        torch.__version__, torch.version.cuda,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))

    # Geometry pre-check BEFORE any model loads (review F2), per selected arm: the
    # target dims and the grid are pure math, so a caps drift fails here in
    # milliseconds, not after the base generation.
    from context_anchored_tile_refine import grid, upscale
    for arm in selected:
        s = ARM_SETTINGS[arm]
        # Per-arm upscale_by (review LOW-2): shared today, but a future arm varying
        # it must not silently validate against another arm's canvas size.
        target_w, target_h = upscale.scale_target(GEN_WIDTH, GEN_HEIGHT, s.upscale_by)
        sx = grid.solve_axis(target_w, s.max_tile_width,
                             s.context_anchor, s.context_overlap, axis="width")
        sy = grid.solve_axis(target_h, s.max_tile_height,
                             s.context_anchor, s.context_overlap, axis="height")
        if (sx.n, sy.n) != EXPECTED_GRID:
            raise SystemExit(f"{arm}: grid {sx.n}x{sy.n} != expected {EXPECTED_GRID} at "
                             f"{target_w}x{target_h} — caps drifted from the 6-tile spec")

    print(f"[clip]    loading {CLIP_NAME} ({CLIP_TYPE})")
    with ab_models.VramProbe() as probe:
        clip = ab_models.load_clip(CLIP_NAME, CLIP_TYPE)
    print(f"[clip]    loaded  {probe}")
    print(f"[unet]    loading {UNET_NAME}")
    with ab_models.VramProbe() as probe:
        model = ab_models.load_unet(UNET_NAME)
        vae = ab_models.load_vae(VAE_NAME)
    print(f"[unet]    loaded  {probe}")

    base = stage_base(model, clip, vae, args.force_base)
    canvas = stage_upscale(base, force=args.force_base)

    # Scene-level reference PNGs (arm-independent; the 23-suffixed pair from the
    # first render stays on disk as the owner's already-reviewed copies).
    for suffix, tensor, stage in (("scene-base", base, "base"),
                                  ("scene-canvas", canvas, "upscaled-canvas")):
        ref = OUTPUT_DIR / f"AB_{RUN_TAG}__{suffix}.png"
        if not ref.is_file():
            ab_models.save_png(ref, tensor.cpu(), dict(gen_key(), stage=stage))
            print(f"[refs]    -> {ref.name}")

    with torch.inference_mode():
        negative = ab_models.encode_prompt(clip, SCENE.negative)
        empty = upscale.encode_empty(clip)

    for arm in selected:
        destination = output_path(arm)
        if destination.is_file() and not args.force:
            print(f"[render]  {arm:<22} exists, skipped (--force to redo)")
            continue
        padded, tiles = solve_tiles(canvas, ARM_SETTINGS[arm])
        if tuple(padded.shape[1:3]) != tuple(canvas.shape[1:3]):
            raise SystemExit("canvas is not /8 — the sync loop's latent math assumes "
                             "the padded and working canvases coincide")
        tile_captions = load_or_generate_captions(clip, padded, tiles, args.force_captions)
        render(arm, padded, tiles, clip, model, vae, negative, empty, tile_captions)

    bad = []
    for arm in selected:
        destination = output_path(arm)
        if not destination.is_file():
            bad.append(f"{destination.name} MISSING")
            continue
        width, height = ab_models.png_size(destination)
        verdict = "OK" if (width, height) == (canvas.shape[2], canvas.shape[1]) else "WRONG SIZE"
        if verdict != "OK":
            bad.append(f"{destination.name} is {width}x{height}")
        print(f"[done]    {destination.name:<48} {width:>4}x{height:<4} {verdict}")
    if bad:
        raise SystemExit("[done]    FAILED: " + "; ".join(bad))
    return 0


if __name__ == "__main__":
    sys.exit(main())
