"""Sync-tiles SANITY SWEEP: the four matrix scenes, small scale, one seam, d0.50.

Owner's spec (2026-08-15, after judging the market stress "perfect... not a single
issue"): before porting sync into the VL node, run it across the AB-Final-Matrix
scenes 1-4 (face / cybercity / market / portrait) at their small scale — 512x288
matrix bases (CACHED; regenerated deterministically if absent), 3x upscale ->
1536x864, a SINGLE vertical seam at x=768 (2x1 grid), denoise 0.50. "It will test
various scenes and styles more so than how well it fixes seams" — though scene 4
(portrait) was designed with a face straddling x=768, so the band mechanism gets
one more stress for free.

Config = the VALIDATED winner (market arm 24): dpmpp_2m / sgm_uniform / 28 steps /
cfg 3.5 / seed 42, context_anchor 128 / context_overlap 128, ring "lead" (frozen
RAW on the shipped lead curve, run-global sigma_first). Caps 1088x896 solve the
2x1 grid (crops 1024x864, verified via grid.solve_axis before this file existed;
re-asserted per run).

    25-face        dark-fantasy close-up character (the 00676-style prompt)
    26-cybercity   distant city, night
    27-market      the Renaissance market at SMALL scale (same prompt as the 4k
                   stress test, different base draw size)
    28-portrait    documentary face CENTERED ON the x=768 seam, detailed background
    2x-*-live      each of the four with ring "live" — "anchor to what is created"
                   (arm 20's mode) instead of the frozen raw. Owner-requested
                   (2026-08-15) after judging the lead sweep: the portrait's
                   mangled background tools should get REPAIRED (captions name
                   them; the model rebuilds what the names say), which the raw
                   anchor prevents. The node's anchor_type option test, across
                   styles. Lead sweep judged: seams/styles all good.

All four render in ONE process — matrix precedent at exactly this canvas scale
(the 3x8 matrix ran 24+ cells per process at 1536x864 without the 3x-scale
silent-death mode); the engine clears the allocator every outer step regardless.

    <venv-python> tests-AB/run_ab_sync_scenes.py                # all four
    <venv-python> tests-AB/run_ab_sync_scenes.py --only 28-portrait

Outputs (output\\AB-Test-Images\\), per arm:
    AB_scene288-d050__{arm}.png          the render (judge this)
    AB_scene288-d050__{arm}-canvas.png   the 3x upscaled refine input (reference)
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
RUN_TAG = "scene288-d050"

UNET_NAME = matrix_run.UNET_NAME
CLIP_NAME = matrix_run.CLIP_NAME
CLIP_TYPE = matrix_run.CLIP_TYPE
VAE_NAME = matrix_run.VAE_NAME

# The validated market-arm-24 config at the matrix's small scale. upscale_by 3.0
# is informational here — the canvas comes from matrix_run.stage_upscale, which
# owns that constant (and its cache).
SETTINGS = krea2_run.Refine(
    seed=42, sampler="dpmpp_2m", scheduler="sgm_uniform", steps=28, cfg=3.5,
    denoise=0.50, upscale_by=3.0, max_tile_width=1088, max_tile_height=896,
    context_anchor=128, context_overlap=128)
EXPECTED_GRID = (2, 1)          # one vertical seam at x=768 — asserted, never assumed

ARMS = {
    "25-face": "dark-fantasy close-up character",
    "26-cybercity": "distant city at night",
    "27-market": "the Renaissance market at small scale",
    "28-portrait": "documentary face CENTERED ON the x=768 seam (the base node's "
                   "designed stress case), detailed background",
    # ---- the LIVE-ring variants (owner, 2026-08-15, after judging the lead sweep):
    # ring = "what is created" — the live same-sigma trajectory (arm 20's mode) —
    # instead of the frozen raw. The portrait scene is WHY anchor_type became a node
    # option in the first place: its source background (the wall of tools) is
    # mangled, and anchoring to it keeps the tools mangled ("it stays having no
    # idea what the tools are"); unanchored, the refine REPAIRS them (the captions
    # name the tools, and the model rebuilds what the names say). Same judged
    # tradeoff as 00676 arm 20 vs 21/22: live = more repair/detail, raw+lead = more
    # fidelity/tone-hold. This is the option test across all four styles.
    "25-face-live": "arm 25 with ring 'live' — the ONLY change",
    "26-cybercity-live": "arm 26 with ring 'live' — the ONLY change",
    "27-market-live": "arm 27 with ring 'live' — the ONLY change",
    "28-portrait-live": "arm 28 with ring 'live' — the ONLY change (judged: tools "
                        "NOT fixed, image changed very little — the ring is the "
                        "WEAKEST of the three source anchors; the d0.5 trajectory "
                        "and the vision rows both still depict the mangled source)",
    # ---- the SURFACE test (owner, 2026-08-15, after the live verdict): the old
    # tools repair came from the RICH captions (they NAME the tools; the settled
    # position captions say "tools on pegboard" and name nothing). The repair dial
    # is the CONDITIONING SURFACE, not the ring. Two shipped-surface variants, ring
    # back on "lead" (the default candidate; the owner's old repair worked even
    # against a raw-adjacent ring, so rich text should out-pull the lead too — if
    # tools do NOT repair under -cap, one live+cap arm isolates ring suppression):
    #   -cap    RICH captions ONLY (vlm_method "captions", RICH_GROUPED, text-only
    #           positives) — max repair, the owner's old winning surface
    #   -vlcap  RICH captions + sliced VISION ROWS (arm-2 cat, rich text) — the
    #           win+win candidate for the node DEFAULT if it repairs WITHOUT the
    #           historic overcook (overexposure/artifacts when long captions
    #           overlapped VL — the owner will read it off the market scene)
    # Both share ONE byte-identical rich-caption cache, so the VL rows are the only
    # difference inside the pair.
    "25-face-cap": "rich captions only",
    "26-cybercity-cap": "rich captions only",
    "27-market-cap": "rich captions only (the owner's overcook detector scene)",
    "28-portrait-cap": "rich captions only (the tools-repair case)",
    "25-face-vlcap": "rich captions + vision rows",
    "26-cybercity-vlcap": "rich captions + vision rows",
    "27-market-vlcap": "rich captions + vision rows (overcook detector)",
    "28-portrait-vlcap": "rich captions + vision rows (repair vs depiction fight)",
}
_SCENE_OF = {"25": "face", "26": "cybercity", "27": "market", "28": "portrait"}
ARM_SCENE = {arm: _SCENE_OF[arm.split("-")[0]] for arm in ARMS}
# arm -> ring presentation (run_ab_sync.ARM_RING semantics): "lead" = frozen RAW on
# the shipped lead curve (the validated market config); "live" = the same-sigma
# trajectory, no clean signal, no lead (arm 20's mode — "anchor to what is created").
ARM_RING = {arm: ("live" if arm.endswith("-live") else "lead") for arm in ARMS}
# arm -> conditioning surface: "pos" = sliced vision rows + settled POSITION captions
# (the validated sweep config); "cap" = RICH captions text-only (shipped vlm_method
# "captions"); "vlcap" = sliced vision rows + RICH captions (arm-2 cat, rich text).
ARM_SURFACE = {arm: ("cap" if arm.endswith("-cap") else
                     "vlcap" if arm.endswith("-vlcap") else "pos") for arm in ARMS}


def output_path(arm, suffix=""):
    return OUTPUT_DIR / f"AB_{RUN_TAG}__{arm}{suffix}.png"


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _arm_instruction(method):
    """This campaign's instruction table, pinned — NOT captions.CAPTION_INSTRUCTIONS.

    That table moved VLM_METHOD_VISION_CAPTIONS from the SETTLED_POSITION wording to
    RICH_GROUPED on 2026-08-16. Resolving through it now would silently re-caption the "pos"
    arms with the "vlcap" arms' text, making the two render identically while every label,
    PNG tEXt stamp and cache key still said "settled-position". The "pos" arms stay pinned to
    the instruction they were judged with."""
    from context_anchored_tile_refine import captions

    if method == captions.VLM_METHOD_VISION_CAPTIONS:
        return captions.SETTLED_POSITION_INSTRUCTION, captions.SETTLED_POSITION_MAX_TOKENS
    return captions.CAPTION_INSTRUCTIONS[method]


def load_or_generate_captions(scene, clip, padded, tiles, force, method=None):
    """The production caption pre-pass per scene, cached per (scene gen digest,
    rects, instruction) — the market harness's pattern. `method` picks the shipped
    instruction: VLM_METHOD_VISION_CAPTIONS (settled position, the default) or
    VLM_METHOD_CAPTIONS (rich grouped — the -cap/-vlcap arms; those two SHARE one
    cache entry, byte-identical text, so the VL rows are their only difference).
    The "surface" label keeps the pos arms' existing cache entries valid; the
    instruction text in the key is what actually separates the methods."""
    from context_anchored_tile_refine import captions

    method = captions.VLM_METHOD_VISION_CAPTIONS if method is None else method
    instruction, max_length = _arm_instruction(method)
    key = {"gen": _digest(matrix_run.gen_key(scene)), "rects": krea2_run._rects(tiles),
           "instruction": instruction, "max_length": max_length, "clip": CLIP_NAME,
           "surface": ("settled-position" if method == captions.VLM_METHOD_VISION_CAPTIONS
                       else "rich-grouped")}
    cached = CACHE_DIR / f"scene288_{scene.key}_sync_captions_{_digest(key)}.json"

    if cached.is_file() and not force:
        tile_captions = json.loads(cached.read_text(encoding="utf-8"))["captions"]
        print(f"[caption] {scene.key:<10} cache hit  {cached.name}")
    else:
        with torch.inference_mode():
            tile_captions = captions.generate_tile_captions(clip, padded, tiles, instruction, max_length)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(dict(key, captions=tile_captions), indent=1), encoding="utf-8")
        print(f"[caption] {scene.key:<10} generated and cached  {cached.name}")
    for tile_idx, row_captions in enumerate(tile_captions):
        print(f"[caption] tile {tile_idx}: {row_captions[0]}")
    return tile_captions


def solve_tiles(canvas):
    from context_anchored_tile_refine import grid, sampling

    pixels = canvas[..., :3]
    padded, _ = sampling.pad_image_to_multiple(pixels)
    sx = grid.solve_axis(padded.shape[2], SETTINGS.max_tile_width,
                         SETTINGS.context_anchor, SETTINGS.context_overlap, axis="width")
    sy = grid.solve_axis(padded.shape[1], SETTINGS.max_tile_height,
                         SETTINGS.context_anchor, SETTINGS.context_overlap, axis="height")
    layout = grid.build_layout(padded.shape[2], padded.shape[1], sx, sy,
                               SETTINGS.context_anchor, SETTINGS.context_overlap)
    print(f"[layout]  canvas {padded.shape[2]}x{padded.shape[1]}  grid {sx.n}x{sy.n}  "
          f"({len(layout.tiles)} tiles, overlap {SETTINGS.context_overlap})")
    if (sx.n, sy.n) != EXPECTED_GRID:
        raise SystemExit(f"grid {sx.n}x{sy.n} != expected {EXPECTED_GRID} — caps drifted "
                         "from the single-seam spec")
    return padded, layout.tiles


def build_settings(arm, scene, tile_captions, sigmas):
    from context_anchored_tile_refine import captions as captions_mod

    caption_method = (captions_mod.VLM_METHOD_VISION_CAPTIONS
                      if ARM_SURFACE[arm] == "pos" else captions_mod.VLM_METHOD_CAPTIONS)
    instruction, max_length = _arm_instruction(caption_method)
    ring_desc = {
        "lead": "frozen RAW canvas on the SHIPPED lead curve (run-global "
                "sigma_first) — the market arm-24 validated config",
        "live": "live same-sigma trajectory — 'anchor to what is created' (arm "
                "20's mode): no clean signal, no lead; mangled source content "
                "gets REPAIRED instead of adhered-to",
    }[ARM_RING[arm]]
    payload = dataclasses.asdict(SETTINGS)
    payload.update({
        "run_label": arm,
        "arm": ARMS[arm],
        "method": "synchronized tiles (run_ab_sync engine): one pass, 2 tiles "
                  "stepped together per sigma; dpmpp_2m per-step with cross-step "
                  "state carried per tile; per-step directional feather in latent",
        "conditioning": {
            "pos": "sliced vision rows + settled POSITION captions (text-cat, arm-2 "
                   "builder) — the validated sweep surface",
            "cap": "RICH captions ONLY (vlm_method 'captions', RICH_GROUPED "
                   "instruction, text-only positives, no vision rows)",
            "vlcap": "sliced vision rows + RICH captions (text-cat, arm-2 builder, "
                     "RICH_GROUPED instruction) — the default-candidate combo",
        }[ARM_SURFACE[arm]],
        # Greppable surface keys (review INFO): the campaign compares PNGs by tEXt.
        "vlm_method": caption_method,
        "caption_instruction": instruction,
        "caption_max_length": max_length,
        "sigmas": [float(v) for v in sigmas],
        "ring": ring_desc,
        "anchor_type": ("lead (run-global sigma_first; see 'ring')"
                        if ARM_RING[arm] == "lead" else "live (see 'ring')"),
        "seam_machinery": "none: no DC match, no min-error cut — the band couples "
                          "every step and decodes one shared latent",
        "scene": scene.key,
        "gen": matrix_run.gen_key(scene),
        "expected_grid": list(EXPECTED_GRID),
        "unet": UNET_NAME, "clip": CLIP_NAME, "clip_type": CLIP_TYPE, "vae": VAE_NAME,
        "upscale_model": matrix_run.UPSCALE_MODEL_NAME,
        "guider": "CFGGuider",
        "negative_prompt": scene.negative,
        "tile_captions": [row[0] for row in tile_captions],
        "harness": "tests-AB/run_ab_sync_scenes.py",
        "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return payload


def render(arm, scene, padded, tiles, clip, model, vae, negative, empty, tile_captions):
    import contextlib

    from context_anchored_tile_refine import upscale

    ring_mode = ARM_RING[arm]
    sigmas = upscale.build_sigmas(model, SETTINGS.scheduler, SETTINGS.steps, SETTINGS.denoise)
    steps = int(sigmas.shape[-1]) - 1
    sync_run.check_sync_preconditions(model, sigmas)

    canvas_h, canvas_w = int(padded.shape[1]), int(padded.shape[2])
    print(f"[render]  {arm}: {steps} synchronized steps over {len(tiles)} tiles, "
          f"sigma {float(sigmas[0]):.4f} -> 0, canvas {canvas_w}x{canvas_h}, "
          f"sampler {SETTINGS.sampler}, overlap {SETTINGS.context_overlap}, "
          f"ring '{ring_mode}'")

    timings = {}
    total_started = time.perf_counter()

    started = time.perf_counter()
    surface = ARM_SURFACE[arm]
    with torch.inference_mode():
        if surface == "cap":
            # Rich captions ONLY — the shipped vlm_method "captions" surface: each
            # tile's positive is its caption encoded as plain text, no vision rows.
            from context_anchored_tile_refine import captions as captions_mod
            tile_positives = captions_mod.build_caption_conds(clip, tile_captions)
        else:
            # "pos" and "vlcap": sliced vision rows + the captions text-cat'd (the
            # arm-2 builder; caption-length agnostic, so rich text rides the same way).
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
    canvas_noise = upscale.Noise_RandomNoise(SETTINGS.seed).generate_noise({"samples": dummy})
    if tuple(canvas_noise.shape) != tuple(canvas_latent.shape):
        raise RuntimeError(f"canvas noise {tuple(canvas_noise.shape)} != latent canvas "
                           f"{tuple(canvas_latent.shape)}")

    raw_latent = canvas_latent.clone() if ring_mode in ("raw", "lead") else None
    ring_patch = (sync_run.lead_ring_patch(model, float(sigmas[0])) if ring_mode == "lead"
                  else contextlib.nullcontext())
    guider = upscale.build_guider(model, empty, negative, SETTINGS.cfg)
    with ab_models.VramProbe() as probe, torch.inference_mode(), ring_patch:
        canvas_latent = sync_run.sync_refine(canvas_latent, canvas_noise, sigmas, tiles,
                                             guider, tile_positives, timings,
                                             ring_mode=ring_mode, raw_latent=raw_latent,
                                             settings=SETTINGS)
    print(f"[sync]    {steps} steps x {len(tiles)} tiles in {timings['sync-steps']:.1f}s  {probe}")

    ab_models.clear_cache()
    with ab_models.VramProbe() as probe, torch.inference_mode():
        result = sync_run.decode_composite(vae, canvas_latent, padded, tiles, timings,
                                           overlap=SETTINGS.context_overlap)
    print(f"[decode]  composite in {timings['decode-composite']:.1f}s  {probe}")

    ab_models.require_image_shape(result, canvas_h, canvas_w, f"sync refine {scene.key}")
    raw_luma = sync_run.mean_luma(padded)
    print(f"[tone]    final luma delta vs raw canvas: "
          f"{sync_run.mean_luma(result) - raw_luma:+.2f}/255")
    destination = output_path(arm)
    written = ab_models.save_png(destination, result.cpu(),
                                 build_settings(arm, scene, tile_captions, sigmas))
    timings["total"] = time.perf_counter() - total_started
    print(f"[render]  {arm} -> {destination.name} {written[0]}x{written[1]}")
    print("[timing]  " + "  ".join(f"{name}={seconds:.1f}s" for name, seconds in timings.items()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", action="append", default=[], metavar="ARM",
                        help="render only these arms (repeatable)")
    parser.add_argument("--force", action="store_true", help="re-render existing outputs")
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
            print(f"{marker} {arm:<14} {description}  -> {output_path(arm).name}")
        return 0

    root, note = ab_env.bootstrap()
    print(f"[env]     ComfyUI {ab_env.version(root)} at {root}  ({note})")
    print("[env]     torch {}  cuda {}  device {}".format(
        torch.__version__, torch.version.cuda,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))

    # Geometry pre-check before any model loads (pure math; one config, all arms).
    from context_anchored_tile_refine import grid, upscale
    target_w, target_h = upscale.scale_target(matrix_run.GEN_WIDTH, matrix_run.GEN_HEIGHT,
                                              matrix_run.UPSCALE_BY)
    sx = grid.solve_axis(target_w, SETTINGS.max_tile_width,
                         SETTINGS.context_anchor, SETTINGS.context_overlap, axis="width")
    sy = grid.solve_axis(target_h, SETTINGS.max_tile_height,
                         SETTINGS.context_anchor, SETTINGS.context_overlap, axis="height")
    if (sx.n, sy.n) != EXPECTED_GRID:
        raise SystemExit(f"grid {sx.n}x{sy.n} != expected {EXPECTED_GRID} at "
                         f"{target_w}x{target_h} — caps drifted from the single-seam spec")

    print(f"[clip]    loading {CLIP_NAME} ({CLIP_TYPE})")
    with ab_models.VramProbe() as probe:
        clip = ab_models.load_clip(CLIP_NAME, CLIP_TYPE)
    print(f"[clip]    loaded  {probe}")
    print(f"[unet]    loading {UNET_NAME}")
    with ab_models.VramProbe() as probe:
        model = ab_models.load_unet(UNET_NAME)
        vae = ab_models.load_vae(VAE_NAME)
    print(f"[unet]    loaded  {probe}")

    with torch.inference_mode():
        empty = upscale.encode_empty(clip)

    for arm in selected:
        destination = output_path(arm)
        if destination.is_file() and not args.force:
            print(f"[render]  {arm:<14} exists, skipped (--force to redo)")
            continue
        scene = matrix_run.SCENES_BY_KEY[ARM_SCENE[arm]]
        base = matrix_run.stage_base(scene, model, clip, vae, force=False)
        canvas = matrix_run.stage_upscale(scene, base, force=False)
        padded, tiles = solve_tiles(canvas)
        if tuple(padded.shape[1:3]) != tuple(canvas.shape[1:3]):
            raise SystemExit("canvas is not /8 — the sync loop's latent math assumes "
                             "the padded and working canvases coincide")
        # Variant arms share the pos/lead arm's canvas byte-for-byte; only the
        # original four write refs — no duplicates in the judging folder.
        ref = output_path(arm, "-canvas")
        if ARM_SURFACE[arm] == "pos" and ARM_RING[arm] == "lead" and not ref.is_file():
            ab_models.save_png(ref, canvas.cpu(),
                               dict(matrix_run.gen_key(scene), stage="upscaled-canvas"))
            print(f"[refs]    -> {ref.name}")
        with torch.inference_mode():
            negative = ab_models.encode_prompt(clip, scene.negative)
        from context_anchored_tile_refine import captions as captions_mod
        caption_method = (captions_mod.VLM_METHOD_VISION_CAPTIONS
                          if ARM_SURFACE[arm] == "pos" else captions_mod.VLM_METHOD_CAPTIONS)
        tile_captions = load_or_generate_captions(scene, clip, padded, tiles,
                                                  args.force_captions, method=caption_method)
        render(arm, scene, padded, tiles, clip, model, vae, negative, empty, tile_captions)

    bad = []
    for arm in selected:
        destination = output_path(arm)
        if not destination.is_file():
            bad.append(f"{destination.name} MISSING")
            continue
        width, height = ab_models.png_size(destination)
        verdict = "OK" if (width, height) == (target_w, target_h) else "WRONG SIZE"
        if verdict != "OK":
            bad.append(f"{destination.name} is {width}x{height}")
        print(f"[done]    {destination.name:<44} {width:>4}x{height:<4} {verdict}")
    if bad:
        raise SystemExit("[done]    FAILED: " + "; ".join(bad))
    return 0


if __name__ == "__main__":
    sys.exit(main())
