"""Anchor-mechanism A/B on the owner's 2026-08-10 two-tile sweep recipe (Krea 2).

PRE-PORT HARNESS, PINNED TO COMMIT 0dfd1e4: drove the raster VL branch removed in 1.6.0
(the VL path is the sync engine now). __main__ is guarded; kept as the campaign record.

Reproduces the sweep workflow exactly — base gen 512x288 "Cyberpunk cityscape at
night" (seed 16928, dpmpp_2m / sgm_uniform 52 steps, cfg 3.5), node upscale 3x via
4xFaceUpDAT + lanczos to 1536x864, ContextAnchoredTileUpscaleVL refine (seed 42,
dpmpp_2m_sde / sgm_uniform 28 steps, cfg 3.5, denoise 0.35, tiles 1024x1024 → 2x1
grid, seam at x=768) — then renders arms that vary ONE mechanism each:

    ring       what tile 1's frozen context_anchor ring encodes: "live" (production —
               tile 0's refined output), "raw" (the un-refined upscale), "grey" (0.5
               flat). raw/grey are the direct test of "Krea 2 ignores the anchor".
    gradring   fractional (linear-ramp) denoise mask over the ring instead of binary.
    vl_scale   multiply the whole-canvas VL encode rows by this before slicing
               (the TBG-ETUR developer's x0.5 recommendation).
    overlap    0 = the owner's sweep (no band, no feather, NO DC match);
               32 = the production seam machinery.
    denoise    0.35 = the sweep; 0.90 = the high-denoise regime the TBG developer
               reasons from (first sigma ~0.90 vs ~0.38 — see the printed sigmas).

Hooks reach production code only through call-time seams (sampling.encode_pixels,
sampling.tile_gradient, vl._encode_batch), restored in finally; production files are
untouched.

    <venv-python> tests-AB/run_ab_anchor.py                 # all arms
    <venv-python> tests-AB/run_ab_anchor.py --only a64-o0 --force
    <venv-python> tests-AB/run_ab_anchor.py --list

Outputs: output\\AB-Test-Images\\AB_anchor__{arm}.png, settings in a `catr_ab` chunk.
"""
import argparse
import contextlib
import dataclasses
import os
import sys
import time
from pathlib import Path

import torch

# ------------------------------------------------------------------ CONFIG

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-Images")
CACHE_DIR = Path(__file__).resolve().parent / "cache"

KREA2_ROOT = Path(r"C:\Users\Blake\ComfyUI-Installs\ComfyUI\ComfyUI")

UNET_NAME = "krea2Center_v408bitFp8Int8.safetensors"
CLIP_NAME = "qwen3-vl-4b-heretic_int8.safetensors"
CLIP_TYPE = "krea2"
VAE_NAME = "krea2RealVae_v10.safetensors"
UPSCALE_MODEL_NAME = "4xFaceUpDAT.safetensors"

# The sweep workflow's base-generation chain, verbatim (nodes 309/310/315/316/317/
# 318/341/343/344/346 of context_anchor_64.png's prompt chunk).
GEN_WIDTH, GEN_HEIGHT = 512, 288
GEN_SEED = 16928
GEN_STEPS = 52
GEN_SAMPLER = "dpmpp_2m"
GEN_SCHEDULER = "sgm_uniform"
CFG = 3.5
POSITIVE_PROMPT = "Cyberpunk cityscape at night"
# "hredded" [sic] — the sweep's negative, kept verbatim so the base reproduces.
NEGATIVE_PROMPT = "jpeg compression, scribbles, AI generated, hredded cloth"

# Node 364's fixed widgets (the refine seed lives on Arm; the sweep's value is 42).
REFINE_SAMPLER = "dpmpp_2m_sde"
REFINE_SCHEDULER = "sgm_uniform"
REFINE_STEPS = 28
UPSCALE_BY = 3.0
MAX_TILE = 1024

RUN_TAG = "anchor"


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    anchor: int
    overlap: int
    denoise: float = 0.35
    ring: str = "live"          # live | raw | grey  (tile 1's frozen ring content)
    gradring: bool = False      # fractional ramp ring instead of binary frozen
    vl_scale: float = 1.0       # VL encode multiplier before slicing
    vl: bool = True             # False = base-node conditioning (text prompt, no slices)
    vl_inner: bool = False      # True = slice from overlap_inner_rect (no ring cells)
    seed: int = 42              # refine noise seed (canvas noise draw)
    hold_anchor: bool = False   # stop re-noising the frozen ring; see hold_anchor_hook
    anchor_lead: float = 0.0    # with hold_anchor: ring DENOISES FASTER than the core.
    #                             0 = off (use anchor_sigma). >0 = exponent on sigma/sigma0,
    #                             so the ring starts MATCHED and pulls ahead. 1 = quadratic.
    anchor_sigma: float = 0.0   # with hold_anchor: sigma MULTIPLIER for the ring.
    #                             0.0 = fully clean, 0.5 = half the normal noise,
    #                             1.0 = identical to the default (a no-op).
    note: str = ""


# Seed x anchor matrix (overlap 0): separates a SYSTEMATIC anchor-size effect on the
# butt-joint DC step from a per-configuration stochastic draw. Same geometry within a
# column; only the canvas noise draw changes down a column.
_DC_MATRIX = [
    Arm(f"a{a}-o0" + (f"-s{s}" if s != 42 else ""), a, 0, seed=s,
        note=f"DC-draw matrix: anchor {a}, seed {s}")
    for a in (32, 64, 128) for s in (42, 43, 44)
    if not (a == 64 and s == 42)   # a64-o0 (seed 42) already exists as the baseline arm
]

ARMS = [
    # --- HOLD ANCHOR at denoise 0.5, overlap 0 -----------------------------------------
    # overlap 0 means NO feather and NO DC match, so the butt joint is held by the frozen
    # context_anchor ring and nothing else — the cleanest possible test of whether the ring
    # is doing its job. Pair differs in one bit: whether the ring is re-noised every step.
    Arm("a64-o0-d050", 64, 0, denoise=0.50,
        note="CONTROL: VL slices, overlap 0, denoise 0.5, ring re-noised (current behaviour)"),
    Arm("a64-o0-d050-hold", 64, 0, denoise=0.50, hold_anchor=True,
        note="ring held CLEAN at every sigma — the H3 finding applied to Krea 2"),
    # anchor 256 is the LARGEST ring that still solves to 2 tiles: crop = base + r = 768+256
    # = 1024, exactly MAX_TILE. 320 would collapse the layout to 4 tiles and change the
    # experiment. A wider ring gives more samples of the double-exposure the owner sees.
    Arm("a256-o0-d050", 256, 0, denoise=0.50,
        note="CONTROL at anchor 256: ring re-noised (current behaviour)"),
    Arm("a256-o0-d050-hold", 256, 0, denoise=0.50, hold_anchor=True,
        note="anchor 256, ring held CLEAN"),
    # DIAGNOSTIC for "is it anchoring at all?". ring="grey" replaces tile 1's frozen ring
    # with flat grey in the ENCODE input only. Previously this probe was confounded: the ring
    # was ~76% noise at step 0, so the model could not read it whether it wanted to. With the
    # ring held clean the probe is finally meaningful — if this renders the same as
    # a256-o0-d050-hold, the model is ignoring the ring content entirely and the "double
    # exposure" cannot be the anchor being followed. If it renders differently, the ring IS
    # being read and the ghost is the model trying to continue content its own latent
    # disagrees with.
    Arm("a256-o0-d050-hold-grey", 256, 0, denoise=0.50, hold_anchor=True, ring="grey",
        note="anchor 256, ring held clean but filled FLAT GREY — does the model read it?"),
    # SECOND SEED. On this scene the seed moved the seam ~3x while anchor moved it ~0.05, and
    # an earlier "anchor 128 wins" result did not replicate. A one-seed improvement is not a
    # finding, so the pair is repeated at 1234 before anything is recommended.
    Arm("a256-o0-d050-s1234", 256, 0, denoise=0.50, seed=1234,
        note="CONTROL at anchor 256, seed 1234"),
    Arm("a256-o0-d050-hold-s1234", 256, 0, denoise=0.50, hold_anchor=True, seed=1234,
        note="anchor 256 ring held CLEAN, seed 1234 — replication"),
    Arm("a256-o0-d050-s7", 256, 0, denoise=0.50, seed=7,
        note="CONTROL at anchor 256, third seed"),
    Arm("a256-o0-d050-hold-s7", 256, 0, denoise=0.50, hold_anchor=True, seed=7,
        note="anchor 256 ring held CLEAN, third seed"),
    # PARTIAL HOLD. A fully clean ring beside a sigma-noised core is a noise-level
    # discontinuity that never occurs in training, which is the likely cause of the ~8px
    # ghost the owner sees at the ring edge. Half sigma keeps the ring far more legible than
    # the default while leaving a much smaller step at the boundary.
    Arm("a256-o0-d050-hold50", 256, 0, denoise=0.50, hold_anchor=True, anchor_sigma=0.5,
        note="anchor 256, ring at HALF the normal sigma — partial hold"),
    Arm("a256-o0-d050-hold25", 256, 0, denoise=0.50, hold_anchor=True, anchor_sigma=0.25,
        note="anchor 256, ring at QUARTER sigma — between full hold and half"),
    # Is the sky BANDING in hold50 the method or the draw? Seam quality and sky banding are
    # separate axes — hold50 has the best seam of any arm while showing the worst banding —
    # so this repeats hold50 on the other two seeds. Banding on all three = the method;
    # banding only on 42 = the draw.
    Arm("a256-o0-d050-hold50-s1234", 256, 0, denoise=0.50, hold_anchor=True,
        anchor_sigma=0.5, seed=1234, note="hold50 replication, seed 1234"),
    Arm("a256-o0-d050-hold50-s7", 256, 0, denoise=0.50, hold_anchor=True,
        anchor_sigma=0.5, seed=7, note="hold50 replication, seed 7"),
    # LEADING ring: starts matched to the core, then denoises faster so it resolves first and
    # the core follows it. Seed 7 because that is the arm the owner rated best.
    Arm("a256-o0-d050-lead1-s7", 256, 0, denoise=0.50, hold_anchor=True, anchor_lead=1.0,
        seed=7, note="ring leads the core, quadratic (sigma^2/sigma0)"),
    Arm("a256-o0-d050-lead2-s7", 256, 0, denoise=0.50, hold_anchor=True, anchor_lead=2.0,
        seed=7, note="ring leads harder (sigma^3/sigma0^2)"),
    Arm("a64-o0", 64, 0, note="baseline: replicates the owner's context_anchor_64 render"),
    Arm("a64-o0-ringraw", 64, 0, ring="raw", note="tile 1's ring holds the RAW upscale, not tile 0's output"),
    Arm("a64-o0-ringgrey", 64, 0, ring="grey", note="tile 1's ring flattened to 0.5 grey"),
    Arm("a64-o0-gradring", 64, 0, gradring=True, note="linear-ramp denoise mask over the ring (DD-style)"),
    Arm("a32-o32", 32, 32, note="production default geometry: band + feather + DC match on"),
    Arm("a64-o32", 64, 32, note="anchor 64 with the seam machinery on"),
    Arm("a64-o32-vl05", 64, 32, vl_scale=0.5, note="TBG's x0.5 on the VL rows, vs a64-o32"),
    Arm("a64-o0-d090", 64, 0, denoise=0.90, note="high-denoise regime, anchor intact"),
    Arm("a64-o0-d090-ringraw", 64, 0, denoise=0.90, ring="raw", note="high-denoise anchor-readability probe"),
    *_DC_MATRIX,
    # VL x0.5 in the TBG developer's own probing regime: does the melt seen at d0.35
    # disappear once the model has full generative freedom?
    Arm("a64-o32-d090", 64, 32, denoise=0.90, note="control: VL 1.0 at denoise 0.90"),
    Arm("a64-o32-d090-vl05", 64, 32, denoise=0.90, vl_scale=0.5, note="VL x0.5 at denoise 0.90"),
    Arm("a64-o32-d100", 64, 32, denoise=1.00, note="control: VL 1.0 at denoise 1.00"),
    Arm("a64-o32-d100-vl05", 64, 32, denoise=1.00, vl_scale=0.5, note="VL x0.5 at denoise 1.00"),
    # TBG claim "ring pickup is even weaker when VL tokens are added": the same ring
    # pair under BASE-node conditioning (text prompt positive, no slices). If VL
    # suppresses latent-context pickup, this pair's live-vs-raw delta must exceed
    # the VL pair's (a64-o0 vs a64-o0-ringraw).
    Arm("a64-o0-novl", 64, 0, vl=False, note="base-node conditioning: text prompt, ring live"),
    Arm("a64-o0-novl-ringraw", 64, 0, vl=False, ring="raw", note="base-node conditioning, ring raw"),
    # TBG's clarified construction, in-framework analogue: slice from overlap_inner_rect
    # (ring cell columns EXCLUDED from the tokens), everything else identical. Tests
    # whether ring pickup is driven by the tokens' ring coverage.
    Arm("a64-o0-vlinner", 64, 0, vl_inner=True, note="VL slices exclude the ring cells, ring live"),
    Arm("a64-o0-vlinner-ringraw", 64, 0, vl_inner=True, ring="raw", note="VL slices exclude the ring cells, ring raw"),
]

# ------------------------------------------------------------------ bootstrap

sys.path.insert(0, str(Path(__file__).resolve().parent))
if "COMFYUI_ROOT" not in os.environ:
    if not (KREA2_ROOT / "comfy" / "text_encoders" / "krea2.py").is_file():
        raise SystemExit(f"No Krea 2 support at {KREA2_ROOT} — set COMFYUI_ROOT")
    os.environ["COMFYUI_ROOT"] = str(KREA2_ROOT)
import ab_env  # noqa: E402  (must precede every comfy import)
import ab_models  # noqa: E402


def output_path(arm_name):
    return OUTPUT_DIR / f"AB_{RUN_TAG}__{arm_name}.png"


# ------------------------------------------------------------------ stages

def stage_base(model, clip, vae, force):
    """The sweep's own base generation, cached: 512x288, seed 16928."""
    from context_anchored_tile_refine import upscale

    key = {"prompt": POSITIVE_PROMPT, "negative": NEGATIVE_PROMPT, "seed": GEN_SEED,
           "sampler": GEN_SAMPLER, "scheduler": GEN_SCHEDULER, "steps": GEN_STEPS,
           "cfg": CFG, "w": GEN_WIDTH, "h": GEN_HEIGHT, "unet": UNET_NAME}
    cached = ab_models.cache_path(CACHE_DIR, "cybercity", "base", GEN_WIDTH, GEN_HEIGHT, key)
    if cached.is_file() and not force:
        base = torch.load(cached, map_location="cpu")
        print(f"[base]    cache hit  {cached.name}")
        return base

    with ab_models.VramProbe() as probe, torch.inference_mode():
        positive = ab_models.encode_prompt(clip, POSITIVE_PROMPT)
        negative = ab_models.encode_prompt(clip, NEGATIVE_PROMPT)
        guider = upscale.build_guider(model, positive, negative, CFG)
        sigmas = ab_models.build_sigmas(model, GEN_SCHEDULER, GEN_STEPS, 1.0)
        sampler = ab_models.build_sampler(GEN_SAMPLER)
        noise = ab_models.build_noise(GEN_SEED)
        latent = ab_models.empty_sd3_latent(GEN_WIDTH, GEN_HEIGHT, 1)
        out = ab_models.sample_custom_advanced(noise, guider, sampler, sigmas, latent)
        base = ab_models.vae_decode(vae, out).detach().float().cpu()
    ab_models.require_image_shape(base, GEN_HEIGHT, GEN_WIDTH, "base gen")
    cached.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, cached)
    print(f"[base]    generated  {probe}  -> {cached.name}")
    return base


def stage_upscale(base, force):
    from context_anchored_tile_refine import upscale

    target_w, target_h = upscale.scale_target(GEN_WIDTH, GEN_HEIGHT, UPSCALE_BY)
    key = {"gen_seed": GEN_SEED, "model": UPSCALE_MODEL_NAME, "upscale_by": UPSCALE_BY,
           "stage": "prepare_upscaled", "unet": UNET_NAME}
    cached = ab_models.cache_path(CACHE_DIR, "cybercity", "upscale", target_w, target_h, key)
    if cached.is_file() and not force:
        canvas = torch.load(cached, map_location="cpu")
        print(f"[upscale] cache hit  {cached.name}")
    else:
        upscale_model = _load_upscale_model()
        try:
            with ab_models.VramProbe() as probe, torch.inference_mode():
                canvas = upscale.prepare_upscaled(base, upscale_model, UPSCALE_BY)
        finally:
            del upscale_model
            ab_models.free_gpu()
        canvas = canvas.detach().float().cpu()
        cached.parent.mkdir(parents=True, exist_ok=True)
        torch.save(canvas, cached)
        print(f"[upscale] done  {probe}  -> {cached.name}")
    ab_models.require_image_shape(canvas, target_h, target_w, "upscaled canvas")
    return canvas


def _load_upscale_model():
    """UpscaleModelLoader body incl. the .patcher wrap (see run_ab_krea2.py)."""
    import comfy.model_management
    import comfy.model_patcher
    import comfy.utils
    import folder_paths
    from comfy_extras.nodes_upscale_model import ImageModelDescriptor, ModelLoader

    model_path = folder_paths.get_full_path_or_raise("upscale_models", UPSCALE_MODEL_NAME)
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
    out = ModelLoader().load_from_state_dict(sd).eval()
    if not isinstance(out, ImageModelDescriptor):
        raise RuntimeError("Upscale model must be a single-image model.")
    out.patcher = comfy.model_patcher.CoreModelPatcher(
        out.model, load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device())
    return out


def solve_layout(canvas, anchor, overlap):
    from context_anchored_tile_refine import grid, sampling

    padded, _ = sampling.pad_image_to_multiple(canvas[..., :3])
    sx = grid.solve_axis(padded.shape[2], MAX_TILE, anchor, overlap)
    sy = grid.solve_axis(padded.shape[1], MAX_TILE, anchor, overlap)
    return grid.build_layout(padded.shape[2], padded.shape[1], sx, sy, anchor, overlap)


# ------------------------------------------------------------------ hooks

@contextlib.contextmanager
def hold_anchor_hook(arm):
    """Stop ComfyUI re-noising the frozen context_anchor ring at every sampling step.

    The ring is frozen by a binary denoise mask, but the mask does NOT hand clean pixels to
    the model. `comfy/samplers.py:639` replaces the masked region at the MODEL INPUT with

        scale_latent_inpaint(x, sigma, noise, latent_image)

    and `Krea2` (comfy/model_base.py:2456) does not override it, so it falls through to
    `CONST.noise_scaling` (model_sampling.py:94-97) = `sigma*noise + (1-sigma)*latent_image`.
    At denoise 0.5 with Krea 2's shift 1.15 the first step therefore hands the ring ~53%
    NOISE. The neighbour's refined pixels only become legible late in the schedule — after
    the tile's structure is already committed — which is exactly the recorded behaviour that
    ring influence survives at high denoise while seam-HOLDING does not, and why anchor width
    moved the seam ~0.05/255 while the seed moved it 3x.

    WAN21 (model_base.py:1919), WAN22 (:2058, "Hold anchor constant across all sigmas"),
    HunyuanVideo (:1391) and LTXAV (:1237) all override this to `return latent_image`. Krea 2
    and MiniMax H3 are the ones that do not. Borrowing it made the frozen head usable from
    step 1 on H3 (docs/h3-video-chunking-findings.md); this is the same one-line change.

    Caveat carried over from H3: a clean anchor cost ~9% of the high-frequency detail there.
    It is a trade to measure, not a free win.
    """
    if not arm.hold_anchor:
        yield
        return
    import comfy.model_base
    cls = comfy.model_base.Krea2
    had_own = "scale_latent_inpaint" in vars(cls)
    saved = vars(cls).get("scale_latent_inpaint")

    factor = float(arm.anchor_sigma)
    lead = float(arm.anchor_lead)
    first = {}

    def scale_latent_inpaint(model_self, sigma, noise, latent_image, **kwargs):
        s = sigma
        if lead > 0.0:
            # LEADING ring (owner's idea). A flat multiplier puts its biggest noise-level
            # discontinuity at step 0, exactly when structure is decided. This instead
            # MATCHES the core at step 0 and then denoises the ring faster, so the ring
            # resolves ahead and the core follows it:
            #     ring_sigma = sigma * (sigma / sigma_first) ** lead
            # At sigma == sigma_first the factor is 1 (no discontinuity at all); as sigma
            # falls the factor falls with it, so the gap opens gradually and closes again as
            # both reach 0. lead=1 is quadratic, higher leads pull further ahead.
            key = float(sigma.flatten()[0])
            first.setdefault("s0", key)
            s = sigma * (key / first["s0"]) ** lead
        else:
            # factor 0 collapses noise_scaling to latent_image exactly (0*noise + 1*latent),
            # so one expression covers "fully clean" and every flat partial hold.
            s = sigma * factor
        return model_self.model_sampling.noise_scaling(s, noise, latent_image)

    cls.scale_latent_inpaint = scale_latent_inpaint
    if lead > 0.0:
        print(f"[render]  hold_anchor: ring LEADS the core, exponent {lead:g} — matched at "
              "step 0 (no initial discontinuity), then denoises faster so it resolves first")
    elif factor == 0.0:
        print("[render]  hold_anchor: ring CLEAN at every sigma (as WAN22/HunyuanVideo do). "
              "NOTE this deliberately breaks the single-noise-level assumption the denoiser "
              "is trained on — a clean strip beside a noisy region is a discontinuity the "
              "model can read as a real edge.")
    else:
        print(f"[render]  hold_anchor: ring at {factor:g}x the normal sigma — partial hold, "
              "trading some legibility for a smaller noise-level discontinuity at the ring "
              "boundary")
    try:
        yield
    finally:
        if had_own:
            cls.scale_latent_inpaint = saved
        else:
            del cls.scale_latent_inpaint


@contextlib.contextmanager
def ring_hook(arm, layout, raw_canvas):
    """Overwrite tile 1's frozen context_anchor ring in the ENCODE input only.

    _refine_tiles calls sampling.encode_pixels once per tile in raster order; at this
    layout that is exactly [tile 0, tile 1]. Tile 1's enc columns [0, anchor) hold the
    live canvas (tile 0's pasted output) — the anchor ring; everything from column
    `anchor` on was already overwritten from the raw source by production code. The
    hook replaces those ring columns with the raw upscale ("raw") or flat 0.5 grey
    ("grey"), so the ONLY change is what the frozen ring shows the model."""
    from context_anchored_tile_refine import sampling

    if arm.ring == "live":
        yield
        return

    tiles = layout.tiles
    if len(tiles) != 2 or tiles[1].crop_rect.x0 != tiles[1].core.x0 - layout.r:
        raise RuntimeError(f"ring hook expects the 2x1 sweep layout, got {len(tiles)} tiles")
    crop = tiles[1].crop_rect
    ring_w = arm.anchor
    original = sampling.encode_pixels
    calls = {"n": 0}

    def hook(vae, pixels):
        calls["n"] += 1
        if calls["n"] == 2:
            replaced = pixels.clone()
            if arm.ring == "raw":
                replaced[:, :, :ring_w, :] = raw_canvas[:, crop.y0:crop.y1, crop.x0:crop.x0 + ring_w, :]
            else:
                replaced[:, :, :ring_w, :] = 0.5
            pixels = replaced
        return original(vae, pixels)

    sampling.encode_pixels = hook
    try:
        yield
    finally:
        sampling.encode_pixels = original
    if calls["n"] != 2:
        raise RuntimeError(f"ring hook saw {calls['n']} encode calls, expected 2")


@contextlib.contextmanager
def gradring_hook(arm):
    """Fractional ring: replace the binary denoise-mask indicator with a linear ramp
    1 -> 0 across the context_anchor ring (in latent cells). KSamplerX0Inpaint then
    partially re-diffuses the ring every step instead of hard-freezing it."""
    from context_anchored_tile_refine import sampling

    if not arm.gradring:
        yield
        return

    original = sampling.tile_gradient
    ring_cells = arm.anchor / 8.0

    def hook(crop, core, scale=1):
        w = (crop.x1 - crop.x0) // scale
        h = (crop.y1 - crop.y0) // scale
        xs = torch.arange(w, dtype=torch.float32) + 0.5
        ys = torch.arange(h, dtype=torch.float32) + 0.5
        dx = ((core.x0 - crop.x0) / scale - xs).clamp(min=0) + (xs - (core.x1 - crop.x0) / scale).clamp(min=0)
        dy = ((core.y0 - crop.y0) / scale - ys).clamp(min=0) + (ys - (core.y1 - crop.y0) / scale).clamp(min=0)
        d = torch.sqrt(dx[None, :] ** 2 + dy[:, None] ** 2)
        return (1.0 - d / ring_cells).clamp(0.0, 1.0)

    sampling.tile_gradient = hook
    try:
        yield
    finally:
        sampling.tile_gradient = original


@contextlib.contextmanager
def vl_inner_hook(arm):
    """Slice each tile's positive from overlap_inner_rect instead of crop_rect, so the
    tokens stop covering the frozen ring — the in-framework analogue of TBG's
    'local per inner tile' VL encode. build_global_slices reads only tile.crop_rect,
    so stand-ins carrying the inner rect under that name reproduce it exactly."""
    import types

    from context_anchored_tile_refine import vl

    if not arm.vl_inner:
        yield
        return

    original = vl.build_global_slices

    def hook(clip, source, tiles, offset_x=0, offset_y=0):
        stand_ins = [types.SimpleNamespace(crop_rect=t.overlap_inner_rect) for t in tiles]
        return original(clip, source, stand_ins, offset_x, offset_y)

    vl.build_global_slices = hook
    try:
        yield
    finally:
        vl.build_global_slices = original


@contextlib.contextmanager
def vl_scale_hook(arm):
    """Multiply the whole-canvas VL encode by vl_scale before slicing — the sequence
    tensor only (what the DiT cross-attends); extras are left alone."""
    from context_anchored_tile_refine import vl

    if arm.vl_scale == 1.0:
        yield
        return

    original = vl._encode_batch

    def hook(clip, canvas_copy, grid_h, grid_w):
        encoded, expected_seq = original(clip, canvas_copy, grid_h, grid_w)
        scaled = [[entry[0] * arm.vl_scale, entry[1]] for entry in encoded]
        return scaled, expected_seq

    vl._encode_batch = hook
    try:
        yield
    finally:
        vl._encode_batch = original


# ------------------------------------------------------------------ render

def build_settings(arm, sigma0):
    payload = dataclasses.asdict(arm)
    payload.update({
        "run_label": arm.name,
        "seed": arm.seed, "sampler": REFINE_SAMPLER, "scheduler": REFINE_SCHEDULER,
        "steps": REFINE_STEPS, "cfg": CFG, "upscale_by": UPSCALE_BY,
        "max_tile_width": MAX_TILE, "max_tile_height": MAX_TILE,
        "context_anchor": arm.anchor, "context_overlap": arm.overlap,
        "sigma_start": sigma0,
        "gen": {"seed": GEN_SEED, "steps": GEN_STEPS, "sampler": GEN_SAMPLER,
                "prompt": POSITIVE_PROMPT, "negative": NEGATIVE_PROMPT,
                "width": GEN_WIDTH, "height": GEN_HEIGHT},
        "unet": UNET_NAME, "clip": CLIP_NAME, "clip_type": CLIP_TYPE, "vae": VAE_NAME,
        "upscale_model": UPSCALE_MODEL_NAME, "guider": "CFGGuider",
        "harness": "tests-AB/run_ab_anchor.py",
        "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return payload


def render(arm, canvas, clip, model, vae, negative, empty, pos_text):
    import comfy.samplers

    from context_anchored_tile_refine import sampling, upscale

    # VL arms: empty placeholder positive, overwritten per tile by the slices.
    # Base-node arms (vl=False): the text prompt is the positive for every tile.
    guider = upscale.build_guider(model, empty if arm.vl else pos_text, negative, CFG)
    sigmas = upscale.build_sigmas(model, REFINE_SCHEDULER, REFINE_STEPS, arm.denoise)
    sampler = comfy.samplers.sampler_object(REFINE_SAMPLER)
    noise = upscale.Noise_RandomNoise(arm.seed)
    sigma0 = float(sigmas[0])
    print(f"[render]  {arm.name:<22} denoise {arm.denoise}  sigma[0] {sigma0:.4f}  ({arm.note})")

    layout = solve_layout(canvas, arm.anchor, arm.overlap)
    with ring_hook(arm, layout, canvas), gradring_hook(arm), vl_scale_hook(arm), \
            vl_inner_hook(arm), hold_anchor_hook(arm), ab_models.VramProbe() as probe, \
            torch.inference_mode():
        result = sampling.refine_image(
            canvas, guider, sampler, sigmas, vae, noise,
            max_tile_width=MAX_TILE, max_tile_height=MAX_TILE,
            context_anchor=arm.anchor, context_overlap=arm.overlap,
            vl_clip=clip if arm.vl else None,
        )

    ab_models.require_image_shape(result, canvas.shape[1], canvas.shape[2], "refine " + arm.name)
    destination = output_path(arm.name)
    written = ab_models.save_png(destination, result.cpu(), build_settings(arm, sigma0))
    print(f"[render]  {arm.name:<22} {probe}  -> {destination.name} {written[0]}x{written[1]}")


# ------------------------------------------------------------------ main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", action="append", default=[], metavar="ARM")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-base", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    names = [a.name for a in ARMS]
    unknown = set(args.only) - set(names)
    if unknown:
        raise SystemExit("unknown arm(s): {}".format(", ".join(sorted(unknown))))
    selected = [a for a in ARMS if not args.only or a.name in args.only]

    if args.list:
        for arm in ARMS:
            marker = "*" if arm in selected else " "
            print(f"{marker} {arm.name:<22} {arm.note}")
        return 0

    root, note = ab_env.bootstrap()
    print(f"[env]     ComfyUI {ab_env.version(root)} at {root}  ({note})")

    from context_anchored_tile_refine import upscale

    print(f"[clip]    loading {CLIP_NAME} ({CLIP_TYPE})")
    clip = ab_models.load_clip(CLIP_NAME, CLIP_TYPE)
    with torch.inference_mode():
        negative = ab_models.encode_prompt(clip, NEGATIVE_PROMPT)
        empty = upscale.encode_empty(clip)
        pos_text = ab_models.encode_prompt(clip, POSITIVE_PROMPT)

    print(f"[unet]    loading {UNET_NAME}")
    model = ab_models.load_unet(UNET_NAME)
    vae = ab_models.load_vae(VAE_NAME)

    base = stage_base(model, clip, vae, args.force_base)
    canvas = stage_upscale(base, args.force_base)

    for arm in selected:
        destination = output_path(arm.name)
        if destination.is_file() and not args.force:
            print(f"[render]  {arm.name:<22} exists, skipped (--force to redo)")
            continue
        ab_models.clear_cache()
        render(arm, canvas, clip, model, vae, negative, empty, pos_text)

    bad = []
    for arm in selected:
        destination = output_path(arm.name)
        if not destination.is_file():
            bad.append(f"{destination.name} MISSING")
            continue
        width, height = ab_models.png_size(destination)
        print(f"[done]    {destination.name:<40} {width}x{height}")
        if (width, height) != (canvas.shape[2], canvas.shape[1]):
            bad.append(f"{destination.name} is {width}x{height}")
    if bad:
        raise SystemExit("[done]    FAILED: " + "; ".join(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "run_ab_anchor.py: PRE-PORT CAMPAIGN, pinned to commit 0dfd1e4. Since 1.6.0 refine_image"
        " routes the VL path to the sync engine, so these arms would silently render on a"
        " different engine than the one they were judged on. Check out 0dfd1e4 to re-run."
    )
