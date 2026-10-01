"""VLM INPUT BUDGET sweep on the portrait scene: how large a picture can Qwen3-VL read?

Owner's question (2026-09-01): the caption pass reads a 384x384 thumbnail of a crop that is
3 MP at the production tile size (2048x1728 caps), and the vision encode reads the entire
canvas at 768x1024. Does the VLM stay coherent up to the tile size, and what does a caption
gain or lose as the picture it reads grows? Two sweeps, one arm per budget, on the portrait
scene (the tools-on-pegboard background is where captions help most):

    cap-<N>mp   vlm_method "captions" (the standard preset's wording), the caption pass
                reading each tile's crop resampled to N megapixels. 0.15 MP is the shipped
                384x384. 4 MP is LARGER than the crop, so that arm is a bicubic upsample
                (production uses "area", which replicates pixels when upsampling); it tests
                token count, not real detail.
    vl-<N>mp    vlm_method "vision tokens", the entire canvas encoded at N megapixels.
                0.79 MP is the shipped 768x1024 budget. The canvas is 5.3 MP, so every
                budget here is a downsample.

Scene: a matrix scene (--scene portrait, the default, or market or face), base regenerated
at 1024x576 (run_ab_sync_market's pattern, a new draw at that size), 3x upscale (4xFaceUpDAT
+ lanczos) -> 3072x1728. The portrait ran first at 0.15/1/2/3/4 MP (captions) and
0.79/1/2/3/4 MP (vision); the owner's read of it set the four-budget ladder below, which
the other scenes run to verify it.
Refine: the owner's 8K workflow widgets (caps 2048x1728, anchor 32, overlap 256,
dpmpp_2m_sde, sgm_uniform, 28 steps, cfg 3.5, denoise 0.5, seed 42, anchor_source "source
image"), which solve to a 2x1 grid with 1824x1728 crops (3.15 MP), one seam at x=1536.
Engine: the SHIPPED path, sampling.refine_image with the VL clip, so what renders is what the
node renders. The only harness-side changes are the two budget injections (a Preset with the
arm's megapixels via captions.resolve_method, and vl.canvas_budget_pixels), plus a recorder
around captions.generate_tile_captions so each arm's caption text lands next to its PNG.

One arm per process (the 3x-scale rule):

    <venv-python> tests-AB/run_ab_vlm_budget.py --list
    <venv-python> tests-AB/run_ab_vlm_budget.py --scene market --only cap-2mp

Outputs (output\\AB-Test-Images\\):
    AB_vlmbudget-<scene>__<arm>.png             the render (judge this)
    AB_vlmbudget-<scene>__<arm>-captions.txt    the tile captions that arm rendered with
    AB_vlmbudget-<scene>__scene-base.png        the 1024x576 base draw
    AB_vlmbudget-<scene>__scene-canvas.png      the 3x upscaled refine input
"""
import argparse
import dataclasses
import hashlib
import json
import math
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

OUTPUT_DIR = split_run.OUTPUT_DIR
CACHE_DIR = split_run.CACHE_DIR
SCENE_KEYS = ("portrait", "market", "face")

# Set by configure(): the scene under test and the labels derived from it.
SCENE = matrix_run.SCENES_BY_KEY["portrait"]
RUN_TAG = "vlmbudget-portrait"
SCENE_LABEL = "portrait1024"


def configure(scene_key):
    global SCENE, RUN_TAG, SCENE_LABEL
    if scene_key not in SCENE_KEYS:
        raise SystemExit(f"unknown scene {scene_key!r}, expected one of {SCENE_KEYS}")
    SCENE = matrix_run.SCENES_BY_KEY[scene_key]
    RUN_TAG = f"vlmbudget-{scene_key}"
    SCENE_LABEL = f"{scene_key}1024"


UNET_NAME = matrix_run.UNET_NAME
CLIP_NAME = matrix_run.CLIP_NAME
CLIP_TYPE = matrix_run.CLIP_TYPE
VAE_NAME = matrix_run.VAE_NAME
UPSCALE_MODEL_NAME = matrix_run.UPSCALE_MODEL_NAME

# Base generation: the matrix chain verbatim except the size.
GEN_WIDTH, GEN_HEIGHT = 1024, 576
GEN_SEED = matrix_run.GEN_SEED
GEN_STEPS = matrix_run.GEN_STEPS
GEN_SAMPLER = matrix_run.GEN_SAMPLER
GEN_SCHEDULER = matrix_run.GEN_SCHEDULER
GEN_CFG = matrix_run.CFG

# The owner's 8K workflow widgets (Krea 2 8k upscale.json), verbatim. upscale_by 3.0 takes
# the 1024x576 base to 3072x1728, where these caps solve to 2x1 with 1824x1728 crops.
SETTINGS = krea2_run.Refine(
    seed=42, sampler="dpmpp_2m_sde", scheduler="sgm_uniform", steps=28, cfg=3.5,
    denoise=0.50, upscale_by=3.0, max_tile_width=2048, max_tile_height=1728,
    context_anchor=32, context_overlap=256)
ANCHOR_SOURCE = "source image"
EXPECTED_GRID = (2, 1)          # (columns, rows), asserted before any model loads

SHIPPED_CAPTION_MP = 384 * 384 / 1_000_000
SHIPPED_VISION_MP = 768 * 1024 / 1_000_000

# arm -> (surface, megapixels). The same four-budget ladder on both sweeps, so the two
# shipped budgets (0.15 for captions, 0.79 for vision) each appear on the other sweep too.
# 4 MP is above the crop's 3.15 MP, so cap-4mp is a bicubic upsample of the crop.
ARMS = {
    "cap-0.15mp": ("captions", SHIPPED_CAPTION_MP),
    "cap-0.79mp": ("captions", SHIPPED_VISION_MP),
    "cap-2mp": ("captions", 2.0),
    "cap-4mp": ("captions", 4.0),
    "vl-0.15mp": ("vision", SHIPPED_CAPTION_MP),
    "vl-0.79mp": ("vision", SHIPPED_VISION_MP),
    "vl-2mp": ("vision", 2.0),
    "vl-4mp": ("vision", 4.0),
    # The portrait's first ladder, kept so its PNGs stay reproducible by name.
    "cap-1mp": ("captions", 1.0),
    "cap-3mp": ("captions", 3.0),
    "vl-1mp": ("vision", 1.0),
    "vl-3mp": ("vision", 3.0),
}
DEFAULT_ARMS = ("cap-0.15mp", "cap-0.79mp", "cap-2mp", "cap-4mp",
                "vl-0.15mp", "vl-0.79mp", "vl-2mp", "vl-4mp")

# The two counts move together on this geometry (the tile covers 60% of the canvas), so
# the ladder above cannot say which one the DiT objects to: the size of the picture the
# VLM ENCODED, or the number of rows INJECTED into the tile. These arms hold the encode
# and thin the slice to every other cell on both axes (a quarter of the rows):
#   vl-2mp-half     2 MP encode, ~297 rows injected (vl-2mp injects 1188 and was judged bad).
#                   Clean here => the injected count was the problem, not the encode size.
#   vl-0.79mp-half  0.79 MP encode, ~116 rows injected (vl-0.79mp injects 462 and won).
#                   Bad here => too few rows starves a tile even from the winning encode.
ARMS.update({
    "vl-2mp-half": ("vision", 2.0),
    "vl-0.79mp-half": ("vision", SHIPPED_VISION_MP),
})
THINNED_ARMS = ("vl-2mp-half", "vl-0.79mp-half")


def thinned_slice_indices(real):
    """vl.slice_indices keeping only the grid cells at even row AND even column, the
    delimiters and the tail untouched. Same encode, same region, a quarter of the rows."""
    from context_anchored_tile_refine import vl

    def slice_indices(crop, canvas_h, canvas_w, enc_h, enc_w, expected_seq, offset_x=0, offset_y=0):
        indices = real(crop, canvas_h, canvas_w, enc_h, enc_w, expected_seq, offset_x, offset_y)
        grid_w = enc_w // vl.MERGED_CELL
        n_rows = (enc_h // vl.MERGED_CELL) * grid_w
        kept = []
        for index in indices:
            if 1 <= index <= n_rows:
                row, column = divmod(index - 1, grid_w)
                if row % 2 or column % 2:
                    continue
            kept.append(index)
        return kept

    return slice_indices


def output_path(arm, suffix="", extension="png"):
    return OUTPUT_DIR / f"AB_{RUN_TAG}__{arm}{suffix}.{extension}"


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


# ------------------------------------------------------------------ pixel stages

def gen_key():
    return {"prompt": SCENE.positive, "negative": SCENE.negative, "seed": GEN_SEED,
            "sampler": GEN_SAMPLER, "scheduler": GEN_SCHEDULER, "steps": GEN_STEPS,
            "cfg": GEN_CFG, "w": GEN_WIDTH, "h": GEN_HEIGHT, "unet": UNET_NAME}


def stage_base(model, clip, vae, force):
    """The 1024x576 portrait base, generated once and cached."""
    from context_anchored_tile_refine import upscale

    cached = ab_models.cache_path(CACHE_DIR, SCENE_LABEL, "base", GEN_WIDTH, GEN_HEIGHT, gen_key())
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
    ab_models.require_image_shape(base, GEN_HEIGHT, GEN_WIDTH, "portrait base gen")
    cached.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, cached)
    print(f"[base]    generated {GEN_WIDTH}x{GEN_HEIGHT}  {probe}  -> {cached.name}")
    return base


def stage_upscale(base, force):
    """3x: the node's own upscale stage (4xFaceUpDAT, then lanczos to exactly 3x), cached."""
    from context_anchored_tile_refine import upscale

    target_w, target_h = upscale.scale_target(GEN_WIDTH, GEN_HEIGHT, SETTINGS.upscale_by)
    key = {"gen": _digest(gen_key()), "model": UPSCALE_MODEL_NAME,
           "upscale_by": SETTINGS.upscale_by, "stage": "prepare_upscaled"}
    cached = ab_models.cache_path(CACHE_DIR, SCENE_LABEL, "upscale", target_w, target_h, key)

    if cached.is_file() and not force:
        canvas = torch.load(cached, map_location="cpu")
        print(f"[upscale] cache hit  {cached.name}")
    else:
        print(f"[upscale] {GEN_WIDTH}x{GEN_HEIGHT} -> {UPSCALE_MODEL_NAME} -> lanczos {target_w}x{target_h}")
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
    ab_models.require_image_shape(canvas, target_h, target_w, "portrait upscaled canvas")
    return canvas


def solve_layout(canvas_w, canvas_h):
    """The grid the engine will solve for this canvas, asserted against the spec."""
    from context_anchored_tile_refine import grid

    sx = grid.solve_axis(canvas_w, SETTINGS.max_tile_width, SETTINGS.context_anchor,
                         SETTINGS.context_overlap, axis="width")
    sy = grid.solve_axis(canvas_h, SETTINGS.max_tile_height, SETTINGS.context_anchor,
                         SETTINGS.context_overlap, axis="height")
    if (sx.n, sy.n) != EXPECTED_GRID:
        raise SystemExit(f"grid {sx.n}x{sy.n} != expected {EXPECTED_GRID} at {canvas_w}x{canvas_h}: "
                         "the caps no longer solve to one seam")
    return grid.build_layout(canvas_w, canvas_h, sx, sy, SETTINGS.context_anchor, SETTINGS.context_overlap)


# ------------------------------------------------------------------ budget injection

def vision_encode_dims(canvas_w, canvas_h, budget):
    """vl.resample_picture's own snap, for the record without running the encode."""
    from context_anchored_tile_refine import vl

    scale = math.sqrt(budget / (canvas_w * canvas_h))
    width = max(vl.MERGED_CELL, round(canvas_w * scale / vl.MERGED_CELL) * vl.MERGED_CELL)
    height = max(vl.MERGED_CELL, round(canvas_h * scale / vl.MERGED_CELL) * vl.MERGED_CELL)
    return width, height


def vision_rows_per_tile(layout, enc_w, enc_h):
    from context_anchored_tile_refine import vl

    grid_w, grid_h = enc_w // vl.MERGED_CELL, enc_h // vl.MERGED_CELL
    tail = 5
    rows = []
    for tile in layout.tiles:
        indices = vl.slice_indices(tile.crop_rect, layout.h, layout.w, enc_h, enc_w,
                                   1 + grid_w * grid_h + 1 + tail)
        rows.append(len(indices) - 2 - tail)
    return rows


def resample_for_vl_bicubic_up(tile_pixels, budget=None):
    """captions.resample_for_vl with ONE change: a budget above the crop's own area
    upsamples with bicubic instead of "area" (which replicates pixels when enlarging).
    Identical to production whenever the budget is a downsample."""
    import comfy.utils

    from context_anchored_tile_refine import captions

    samples = tile_pixels.movedim(-1, 1)
    pixels = captions.VL_INPUT_BUDGET if budget is None else budget
    scale_by = math.sqrt(pixels / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    method = "bicubic" if scale_by > 1.0 else "area"
    resampled = comfy.utils.common_upscale(samples, width, height, method, "disabled")
    return resampled.movedim(1, -1)[:, :, :, :3]


class ArmPatches:
    """The arm's budget, injected where the shipped engine reads it, and a recorder around
    the caption pass. Restores everything on exit."""

    def __init__(self, arm, layout):
        from context_anchored_tile_refine import captions, vl

        self.arm = arm
        self.surface, self.megapixels = ARMS[arm]
        self.layout = layout
        self.captions_mod = captions
        self.vl_mod = vl
        self.tile_captions = None
        self.caption_seconds = 0.0
        self._saved = {}

    def vlm_method(self):
        return (self.captions_mod.VLM_METHOD_CAPTIONS if self.surface == "captions"
                else self.captions_mod.VLM_METHOD_VISION)

    def preset(self):
        """The standard preset's wording (RICH_GROUPED, 768 tokens, no style caption) at the
        arm's megapixels. Built directly so the 2 MP ceiling settings.toml enforces does not
        apply: the sweep exists to look past it."""
        standard = self.captions_mod.resolve_method(f"{self.captions_mod.VLM_METHOD_CAPTIONS} (standard)")
        return dataclasses.replace(standard, label=f"budget-sweep-{self.arm}",
                                   vision=dataclasses.replace(standard.vision, caption_megapixels=self.megapixels))

    def __enter__(self):
        captions, vl = self.captions_mod, self.vl_mod
        self._saved = {
            "resolve_method": captions.resolve_method,
            "generate_tile_captions": captions.generate_tile_captions,
            "resample_for_vl": captions.resample_for_vl,
            "canvas_budget_pixels": vl.canvas_budget_pixels,
            "slice_indices": vl.slice_indices,
        }
        if self.arm in THINNED_ARMS:
            vl.slice_indices = thinned_slice_indices(vl.slice_indices)
        if self.surface == "captions":
            preset = self.preset()
            real_generate = captions.generate_tile_captions

            def resolve_method(vlm_method):
                if captions.method_surface(vlm_method) != captions.VLM_METHOD_CAPTIONS:
                    raise RuntimeError(f"caption arm asked to resolve {vlm_method!r}")
                return preset

            def generate_tile_captions(*args, **kwargs):
                started = time.perf_counter()
                result = real_generate(*args, **kwargs)
                self.caption_seconds = time.perf_counter() - started
                self.tile_captions = [row[0] for row in result]
                return result

            captions.resolve_method = resolve_method
            captions.generate_tile_captions = generate_tile_captions
            captions.resample_for_vl = resample_for_vl_bicubic_up
        else:
            # The sweep's vision arms are the canvas slice alone at a FIXED sample size: the
            # crop rows are turned off and the canvas budget pinned whatever the tile count.
            budget = round(self.megapixels * 1_000_000)
            real_resolve = captions.resolve_method

            def resolve_method(vlm_method):
                preset = real_resolve(vlm_method)
                return dataclasses.replace(preset, vision=dataclasses.replace(preset.vision, crop_tokens=0))

            captions.resolve_method = resolve_method
            vl.canvas_budget_pixels = lambda tiles, source_h, source_w, canvas_tokens: budget
        return self

    def __exit__(self, *_):
        captions, vl = self.captions_mod, self.vl_mod
        captions.resolve_method = self._saved["resolve_method"]
        captions.generate_tile_captions = self._saved["generate_tile_captions"]
        captions.resample_for_vl = self._saved["resample_for_vl"]
        vl.canvas_budget_pixels = self._saved["canvas_budget_pixels"]
        vl.slice_indices = self._saved["slice_indices"]
        return False


# ------------------------------------------------------------------ the render

def build_settings(arm, patches, sigmas, layout, timings, extra):
    surface, megapixels = ARMS[arm]
    payload = dataclasses.asdict(SETTINGS)
    payload.update({
        "run_label": arm,
        "sweep": surface,
        "megapixels": megapixels,
        "vlm_method": patches.vlm_method(),
        "anchor_source": ANCHOR_SOURCE,
        "engine": "shipped sync engine (sampling.refine_image with vl_clip)",
        "scene": "portrait (run_ab_matrix prompt), base regenerated at 1024x576, 3x upscale",
        "gen": gen_key(),
        "grid": list(EXPECTED_GRID),
        "crops": krea2_run._rects(layout.tiles),
        "sigmas": [float(v) for v in sigmas],
        "unet": UNET_NAME, "clip": CLIP_NAME, "clip_type": CLIP_TYPE, "vae": VAE_NAME,
        "upscale_model": UPSCALE_MODEL_NAME,
        "guider": "CFGGuider",
        "negative_prompt": SCENE.negative,
        "timings": {name: round(seconds, 1) for name, seconds in timings.items()},
        "harness": "tests-AB/run_ab_vlm_budget.py",
        "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    payload.update(extra)
    return payload


def render(arm, canvas, layout, clip, model, vae, negative, empty):
    import comfy.samplers

    from context_anchored_tile_refine import sampling, upscale

    surface, megapixels = ARMS[arm]
    canvas_h, canvas_w = int(canvas.shape[1]), int(canvas.shape[2])
    crop = layout.tiles[0]
    crop_mp = crop.sampled_w * crop.sampled_h / 1_000_000
    sigmas = upscale.build_sigmas(model, SETTINGS.scheduler, SETTINGS.steps, SETTINGS.denoise)
    sampler = comfy.samplers.sampler_object(SETTINGS.sampler)
    noise = upscale.Noise_RandomNoise(SETTINGS.seed)
    guider = upscale.build_guider(model, empty, negative, SETTINGS.cfg)
    extra = {}
    timings = {}

    if surface == "captions":
        upsample = megapixels > crop_mp
        print(f"[arm]     {arm}: captions read each {crop.sampled_w}x{crop.sampled_h} crop "
              f"({crop_mp:.2f} MP) at {megapixels:.3f} MP"
              + (", a BICUBIC UPSAMPLE" if upsample else ""))
        extra.update({"caption_input_megapixels": megapixels, "crop_megapixels": round(crop_mp, 3),
                      "caption_input_is_upsample": upsample,
                      "caption_instruction": "standard preset (RICH_GROUPED), 768 tokens"})
    total_started = time.perf_counter()
    with ArmPatches(arm, layout) as patches, ab_models.VramProbe() as probe, torch.inference_mode():
        if surface != "captions":
            # Inside the patch scope, so a thinned arm reports the rows it injects.
            enc_w, enc_h = vision_encode_dims(canvas_w, canvas_h, round(megapixels * 1_000_000))
            rows = vision_rows_per_tile(layout, enc_w, enc_h)
            thinned = arm in THINNED_ARMS
            print(f"[arm]     {arm}: entire canvas encoded at {enc_w}x{enc_h} "
                  f"({enc_w * enc_h / 1_000_000:.2f} MP, {canvas_w / (enc_w // 32):.0f} px per cell), "
                  f"vision rows per tile {rows}" + (", every other cell on both axes" if thinned else ""))
            extra.update({"vision_encode": [enc_w, enc_h], "px_per_cell": round(canvas_w / (enc_w // 32), 1),
                          "vision_rows_per_tile": rows, "slice_thinned_to_every_other_cell": thinned})
        result = sampling.refine_image(
            canvas, guider, sampler, sigmas, vae, noise,
            SETTINGS.max_tile_width, SETTINGS.max_tile_height,
            SETTINGS.context_anchor, SETTINGS.context_overlap,
            mask=None, vl_clip=clip, vlm_method=patches.vlm_method(),
            anchor_source=ANCHOR_SOURCE, sampler_name=SETTINGS.sampler)
    timings["total"] = time.perf_counter() - total_started
    timings["captions"] = patches.caption_seconds
    print(f"[refine]  {arm} done  {probe}")

    ab_models.require_image_shape(result, canvas_h, canvas_w, f"refine {arm}")
    if patches.tile_captions is not None:
        extra["tile_captions"] = patches.tile_captions
        sidecar = output_path(arm, "-captions", "txt")
        lines = [f"{arm}: captions read the crop at {megapixels:.3f} MP "
                 f"(crop {crop.sampled_w}x{crop.sampled_h} = {crop_mp:.2f} MP)", ""]
        for index, text in enumerate(patches.tile_captions):
            lines += [f"--- tile {index} ---", text, ""]
        sidecar.write_text("\n".join(lines), encoding="utf-8")
        for index, text in enumerate(patches.tile_captions):
            print(f"[caption] tile {index}: {text}")
    destination = output_path(arm)
    written = ab_models.save_png(destination, result.cpu(),
                                 build_settings(arm, patches, sigmas, layout, timings, extra))
    print(f"[render]  {arm} -> {destination.name} {written[0]}x{written[1]}")
    print("[timing]  " + "  ".join(f"{name}={seconds:.1f}s" for name, seconds in timings.items()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", default="portrait", choices=SCENE_KEYS,
                        help="which matrix scene to render (default portrait)")
    parser.add_argument("--only", action="append", default=[], metavar="ARM",
                        help="render only these arms (repeatable); default is the four-budget ladder")
    parser.add_argument("--force", action="store_true", help="re-render existing outputs")
    parser.add_argument("--force-base", action="store_true", help="regenerate the cached base draw")
    parser.add_argument("--list", action="store_true", help="show the arms and exit")
    args = parser.parse_args(argv)
    configure(args.scene)

    unknown = set(args.only) - set(ARMS)
    if unknown:
        raise SystemExit("unknown arm(s): {}".format(", ".join(sorted(unknown))))
    selected = [arm for arm in ARMS if (arm in args.only if args.only else arm in DEFAULT_ARMS)]

    if args.list:
        for arm, (surface, megapixels) in ARMS.items():
            marker = "*" if arm in selected else " "
            print(f"{marker} {arm:<11} {surface:<9} {megapixels:.3f} MP  -> {output_path(arm).name}")
        return 0

    root, note = ab_env.bootstrap()
    print(f"[env]     ComfyUI {ab_env.version(root)} at {root}  ({note})")
    print("[env]     torch {}  cuda {}  device {}".format(
        torch.__version__, torch.version.cuda,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))

    from context_anchored_tile_refine import upscale

    target_w, target_h = upscale.scale_target(GEN_WIDTH, GEN_HEIGHT, SETTINGS.upscale_by)
    layout = solve_layout(target_w, target_h)
    crop = layout.tiles[0]
    print(f"[layout]  canvas {target_w}x{target_h}  grid {EXPECTED_GRID[0]}x{EXPECTED_GRID[1]}  "
          f"crops {crop.sampled_w}x{crop.sampled_h} ({crop.sampled_w * crop.sampled_h / 1e6:.2f} MP)")

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
    canvas = canvas[..., :3].contiguous()
    if tuple(canvas.shape[1:3]) != (target_h, target_w):
        raise SystemExit(f"canvas is {tuple(canvas.shape[1:3])}, expected {(target_h, target_w)}")
    for suffix, picture in (("scene-base", base), ("scene-canvas", canvas)):
        reference = output_path(suffix)
        if not reference.is_file():
            ab_models.save_png(reference, picture.cpu(), dict(gen_key(), stage=suffix))
            print(f"[refs]    -> {reference.name}")

    with torch.inference_mode():
        negative = ab_models.encode_prompt(clip, SCENE.negative)
        empty = upscale.encode_empty(clip)

    for arm in selected:
        destination = output_path(arm)
        if destination.is_file() and not args.force:
            print(f"[render]  {arm:<11} exists, skipped (--force to redo)")
            continue
        ab_models.clear_cache()
        render(arm, canvas, layout, clip, model, vae, negative, empty)

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
