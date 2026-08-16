"""Synchronized tiles: ONE pass, every tile at the SAME sigma at every step.

The confluence campaign's remaining defects (run_ab_confluence.py, STATUS block) all
trace to its two-pass COMMIT: a region meets its neighbours' content at a DIFFERENT
denoising stage (frozen refined rings, a committed posterior-mean draft, re-noised
residue). This harness removes stage mismatch BY CONSTRUCTION instead of managing it
(exactly so for single-step samplers; a MULTISTEP sampler's per-tile history
reintroduces a small band-confined residual — see _dpmpp_2m_single_step):
the schedule is stepped OUTSIDE the tiles, and no tile advances to sigma[i+1] until
every tile has taken its sigma[i] step. There is no draft, no cut, no strip, no
handoff — one trajectory, held in one shared canvas latent, sampled through the
stock tile windows.

    20-sync-tiles     the method's first render: stock geometry (2x2, anchor 256,
                      overlap 32), arm-2 conditioning (sliced vision rows + text-only
                      captions — the campaign's content winner), seed 42, d0.50.
                      JUDGED (2026-08-15): "FAR better... so far, this is a win" —
                      zero visible seams, no dupe moon, tree uncut, crisp colours,
                      much better skin detail, eyes stable, far more realistic. Open:
                      a bit brighter (measured +5.98 luma; owner: crisp, NOT
                      over-exposed — mean luma cannot tell re-exposure from wash) and
                      the woman AGED 20+ years. Hypothesis: the ring carries ZERO
                      advance information in sync mode (see THE RING below), and the
                      lead ring's judged effects were "freckles gone" + "a little
                      smooth" — so no-clean-ring = texture amplification unchecked:
                      the detail AND the aging move on the same knob.
    21-sync-raw-ring  arm 20 with ONE change: the anchor ring's CONTENT is restored
                      to a clean signal — the frozen RAW canvas re-noised to the
                      current sigma (stock's exact construction for a not-yet-refined
                      neighbour, here on ALL sides). Sync stepping, masks, feather,
                      shared SDE noise, conditioning byte-identical to 20. Per tile
                      step i>0 the ring cells of the latent input come from the
                      preserved C_0 raw latent and the ring cells of the noise tensor
                      from the seed-42 canvas draw (the SAME epsilon every step —
                      stock's own within-call behaviour), so KSamplerX0Inpaint
                      presents sigma*eps + (1-sigma)*raw at every eval. Step 0 is
                      already exactly that construction and needs no branch. If
                      aging/brightness retreat here, the clean-signal ring is the
                      knob (arm 22 = add the lead analog to dial it); if they
                      persist, sync stepping itself is the cause.
                      READ THE VERDICT CAREFULLY (review): this change also DELETES
                      arm 20's inter-tile content coupling — the 32->256px ring band
                      lies entirely inside the neighbouring tile's core, so in 20 it
                      carried that neighbour's in-progress refinement; in 21 the only
                      coupling left is the 32px write-back feather + the shared SDE
                      noise, and the never-advancing raw band butts fully-refined
                      content on ALL sides as sigma -> 0 (stock only ever does that
                      on its not-yet-refined bottom/right sides). If seams or
                      incoherence RETURN here, blame coupling-loss, not the clean
                      signal.
                      JUDGED (2026-08-15): "aging did get better and not as bright —
                      it did what we hoped." Measured tone +4.53 vs 20's +5.98: the
                      clean ring recovered ~27% of the excess over the +0.6
                      single-pass band, so it is A cause, not THE cause. Next suspect:
                      the missing LEAD (arm 2 stock's +0.67 was measured WITH the
                      leading schedule; 21 presents the ring adjacent-style — mostly
                      noise exactly while structure is decided).
    22-sync-lead-ring arm 21 with ONE change: the ring is presented on the SHIPPED
                      lead curve instead of adjacent — sigma_ring = sigma *
                      anchor_ring_factor(sigma / sigma_first) with sigma_first = the
                      RUN's sigmas[0]. Ring latent/noise inputs are byte-identical to
                      21 (raw + canvas epsilon); the only delta is an instance patch
                      on scale_latent_inpaint mirroring the shipped
                      anchor_ring_schedule's body. The shipped context manager cannot
                      be reused here because it normalizes sigma_first PER CALL, and
                      a per-step call has sigmas[0] == the current sigma, so r == 1
                      at every eval 1 and the lead never engages; the patch fixes the
                      NORMALIZATION while the CURVE still resolves through
                      sampling.anchor_ring_factor — the module's own A/B swap point.
                      Exact at both evals (the factor is recomputed at each eval's
                      sigma, like production). Step 0 is automatically matched
                      (r = 1 -> factor 1); the release lands at sigma < 0.114 (steps
                      26-28), mirroring lead1r15 in production. This is the closest
                      sync analog of the SHIPPING config: the remaining differences
                      vs stock are sync stepping itself, the per-step latent feather,
                      and raw (never refined-neighbour) ring content.

STRUCTURE (C = one canvas-sized latent, canonical latent space, CPU fp32):
  stage E   per-tile vae.encode of the RAW crop windows at stock tile sizes (never a
            whole-canvas VAE call — its ~21 GiB spike is why the engine tiles at
            all); the butted CORES assemble C_0. Window-framing differences between
            neighbouring encodes exist only where cores butt, are sub-1/255, and are
            then diffused for the full 28 steps.
  stage S   28 outer steps. Per step i, from ONE snapshot of C at sigma[i]: each
            tile runs a single sampler step [sigma[i], sigma[i+1]] over its crop
            window (core+overlap diffused under the stock binary mask, the anchor
            ring frozen context), then writes core+overlap back through the
            engine's own directional feather (feather_alpha at latent scale,
            straight ramps) in raster order. Every (1-alpha)-weighted read is a
            cell an earlier tile already wrote THIS step (bands extend only into
            earlier cores; tile 0 is all-ones), so after the loop every cell of C
            is at sigma[i+1] — no stale-sigma residue survives a step.
  stage D   per-tile vae.decode of the final windows, composited by the stock pixel
            feather. Both sides of an overlap band decode the SAME latent, so DC
            match and the min-error cut have nothing to correct and are deliberately
            absent; the band delta is printed as evidence.

COORDINATE CONTRACT (verified against the installed comfy 0.32.0 source):
  * C is stored in the canonical form process_latent_out(x / (1 - sigma)). A step
    call passes it as the latent with a ZERO noise tensor: comfy applies
    process_latent_in (exact affine inverse, Wan21) and CONST.noise_scaling(sigma,
    0, z) = (1 - sigma) * z, reproducing the trajectory state x bit-exactly; the
    return is process_latent_out(inverse_noise_scaling(sigma[i+1], x')) — already
    the storage form at sigma[i+1]. This is arm 15's truncate/resume identity
    applied per step. Step 0 instead passes the raw window latent plus the seed-42
    canvas-noise slice, so sigma0*eps + (1-sigma0)*raw is byte-identical to a stock
    tile's first step (same dummy draw, same slicing). offset_first_sigma_for_snr
    rewrites only sigmas[0] >= 1; sigma[0] is 0.7596 here, asserted < 1.
  * Blending in the canonical form is exact: process_latent_out is per-channel
    affine and the feather weights sum to 1, so the blend commutes with it.
  * THE RING: anchor_ring=False — the lead schedule is inapplicable (it exists to
    bridge a FROZEN REFINED ring to a noisy core; here nothing refined exists
    mid-run). With the per-step noise tensor zero, KSamplerX0Inpaint's ring input
    at the step's first model eval is (1-sigma[i]) * z_ring = the LIVE trajectory
    at exactly sigma[i]; at the second eval the same state rescaled to sigma[i+1].
    Step 0 re-noises the raw ring from the canvas-noise slice — the cores' own
    construction. Exactness (review finding 5): the same-sigma claim is EXACT at
    each step's first model eval (the mask blend is a numerical no-op there); at
    the second eval the ring's noise content sits ~1.13x the core's fresh
    sigma[i+1] — the same within-step deviation a stock frozen-ring run carries,
    not a harness regression. Requires the default CONST scale_latent_inpaint,
    asserted via sampling's own _model_re_noises_anchor. This bullet describes
    ARM_RING "live" (arm 20); "raw" (arm 21) swaps the ring cells of the latent and
    noise inputs so the same machinery presents sigma*eps + (1-sigma)*raw instead;
    "lead" (arm 22) keeps "raw"'s inputs and rescales the presentation sigma via
    lead_ring_patch — the SHIPPED comfy-side schedule stays inapplicable (per-call
    normalization), the shipped CURVE does not.
  * SDE NOISE: exp_heun_2_x0_sde injects 2 stochastic draws per non-final step. A
    noise_sampler on KSAMPLER.extra_options slices per-step CANVAS-shaped draws, so
    neighbouring tiles receive IDENTICAL noise where windows overlap — band
    divergence is model-context only, never stochastic (the shared-noise
    requirement from the MIT interpolation paper). One generator, seeded
    REFINE.seed; two fields per step; a third draw raises.

WHAT THIS REMOVES STRUCTURALLY (why it is worth a render):
  no draft/commit/handoff -> speckle generator 1 cannot exist; no arm-1 surface ->
  generator 2 absent; no freeze at a cut, no mean-draft rings -> the colour/tone
  drift causes absent (expected tone: the single-pass +0.6 band); model EVALS are
  the STOCK count (55 x 4 tiles — the final step is single-eval), under arm 8's
  1.33x. Wall clock runs somewhat above 1.0x: 112 guider.sample invocations pay
  prepare_sampling / cond re-upload where stock pays 4 (review finding 6).
UNKNOWNS (why it is only a render that can judge it):
  (a) per-step blending in the band is MultiDiffusion's known softness site — the
      directional feather (later tile owns the seam line) narrows but does not
      remove it; (b) the stock ring pulls a tile toward FINISHED neighbour detail,
      which is replaced here by same-sigma coupling + VL rows — whether seams and
      coherence hold to the owner's eye is exactly the A/B question.

    <venv-python> tests-AB/run_ab_sync.py                    # renders pending arms (22)
    <venv-python> tests-AB/run_ab_sync.py --only 22-sync-lead-ring
    <venv-python> tests-AB/run_ab_sync.py --list

Outputs (output\\AB-Test-Images\\), one PNG per arm — no draft/stage files exist,
the method is single-pass:
    AB_00676-d050__20-sync-tiles.png      arm 20 (judged: a win; brighter, aged her)
    AB_00676-d050__21-sync-raw-ring.png   arm 21 (judged: aging/brightness better)
    AB_00676-d050__22-sync-lead-ring.png  arm 22 (judge against 21: tone gap closed?)
"""
import argparse
import contextlib
import dataclasses
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_env
import ab_models
import run_ab_confluence as confluence_run  # sets COMFYUI_ROOT via its own import chain
import run_ab_krea2 as krea2_run
import run_ab_split as split_run
from run_ab_krea2 import (
    BASE_PNG,
    CLIP_NAME,
    CLIP_TYPE,
    NEGATIVE_PROMPT,
    UNET_NAME,
    UPSCALE_MODEL_NAME,
    VAE_NAME,
)

OUTPUT_DIR = split_run.OUTPUT_DIR
RUN_TAG = split_run.RUN_TAG
REFINE = split_run.REFINE          # the d0.50 settings every campaign arm shares
mean_luma = confluence_run.mean_luma
_file_sha = krea2_run._file_sha

ARMS = {
    "20-sync-tiles": "synchronized tiles: single pass, all tiles stepped together "
                     "sigma by sigma on one shared canvas latent — no draft, no "
                     "commit, no strips (arm-2 conditioning, stock geometry) "
                     "(judged: a win — zero seams, crisp, detailed; brighter, aged her)",
    "21-sync-raw-ring": "arm 20 with the ring content restored to a CLEAN signal — "
                        "frozen raw canvas re-noised to the current sigma on all "
                        "sides; the ONLY change vs 20 (judged: aging and brightness "
                        "both better — the clean ring is a confirmed knob)",
    "22-sync-lead-ring": "arm 21 with the ring presented on the SHIPPED lead curve "
                         "(run-global sigma_first) instead of adjacent — the ONLY "
                         "change vs 21 (does the lead close the remaining tone gap?)",
}

# arm -> what the frozen anchor ring PRESENTS to the model at each eval.
# "live": the same-sigma trajectory — pure context, zero advance information (arm 20's
#         judged detail win AND its aging/brightness suspect).
# "raw":  sigma*eps + (1-sigma)*raw_canvas — the adjacent-style clean signal (stock's
#         construction for a not-yet-refined neighbour, applied on all sides).
# "lead": "raw" inputs + sigma rescaled to sigma*anchor_ring_factor(sigma/sigma_first)
#         inside scale_latent_inpaint (lead_ring_patch) — the shipped lead1r15 curve
#         with RUN-global normalization.
ARM_RING = {
    "20-sync-tiles": "live",
    "21-sync-raw-ring": "raw",
    "22-sync-lead-ring": "lead",
}


def output_path(arm):
    return OUTPUT_DIR / f"AB_{RUN_TAG}__{arm}.png"


# ------------------------------------------------------------------ latent geometry

def _latent_rect(rect):
    """A grid rect scaled to the /8 latent grid. Every grid coordinate is a multiple
    of 8 by construction (solve_axis multiple=8), asserted so a future geometry change
    fails here instead of shearing the feather."""
    from context_anchored_tile_refine import grid

    coords = (rect.x0, rect.y0, rect.x1, rect.y1)
    if any(c % 8 for c in coords):
        raise RuntimeError(f"tile rect {coords} is not /8 — latent scaling would shear")
    return grid.Rect(rect.x0 // 8, rect.y0 // 8, rect.x1 // 8, rect.y1 // 8)


def encode_canvas_latent(vae, padded, tiles):
    """C_0: per-tile vae.encode of the raw crop windows (stock tile sizes), butted
    CORES pasted into one canvas latent. Coverage is asserted — the cores tile the
    canvas exactly, so every cell (ring regions included) comes from exactly one
    encode."""
    from context_anchored_tile_refine import sampling

    h8, w8 = padded.shape[1] // 8, padded.shape[2] // 8
    canvas = None
    covered = torch.zeros((h8, w8), dtype=torch.bool)
    for tile in tiles:
        crop, core = tile.crop_rect, tile.core
        latent = sampling.encode_pixels(vae, padded[:, crop.y0:crop.y1, crop.x0:crop.x1, :])
        latent = latent.detach().to("cpu", torch.float32)
        if latent.shape[-2:] != ((crop.y1 - crop.y0) // 8, (crop.x1 - crop.x0) // 8):
            raise RuntimeError(
                f"window encode shape {tuple(latent.shape[-2:])} != crop {crop} at /8 — "
                "this VAE's spatial compression is not the /8 the grid math models")
        if canvas is None:
            canvas = torch.zeros((*latent.shape[:-2], h8, w8), dtype=torch.float32)
        cy0, cy1 = (core.y0 - crop.y0) // 8, (core.y1 - crop.y0) // 8
        cx0, cx1 = (core.x0 - crop.x0) // 8, (core.x1 - crop.x0) // 8
        canvas[..., core.y0 // 8:core.y1 // 8, core.x0 // 8:core.x1 // 8] = latent[..., cy0:cy1, cx0:cx1]
        covered[core.y0 // 8:core.y1 // 8, core.x0 // 8:core.x1 // 8] = True
    if not bool(covered.all()):
        raise RuntimeError("tile cores do not tile the latent canvas — coverage hole")
    return canvas


# ------------------------------------------------------------------ shared SDE noise

class StepNoise:
    """Canvas-shaped SDE noise for one outer step, sliced per tile window. The
    sampler (exp_heun_2_x0_sde, r=1) calls its noise_sampler exactly twice per
    non-final step and never on the final one (sigma_next == 0 short-circuits), so
    two fields are drawn per step from ONE run-long generator and a third draw
    raises. At r=1 the SECOND draw's integrator weight is exactly zero
    (segment_factor = (r-1)*h*eta = 0 in sample_seeds_2), so all acting
    stochasticity is field 0 — both are still served because the sampler requests
    both. Fields live on CPU; slices move to the sampler's device at call time
    (sigma rides on the load device inside the k-diffusion loop)."""

    def __init__(self, seed, shape):
        # seed + 1, comfy's own CPU convention (default_noise_sampler): seeding with
        # the bare refine seed makes this generator's FIRST draw bit-identical to
        # prepare_noise's canvas draw (same engine, seed, shape, dtype, device), and
        # the step-0 SDE injection would then re-apply the construction epsilon — a
        # coherent 1.30x over-noise at sigma[1] on every tile that would be misread
        # as sync-tiling fidelity loss (review finding 1, empirically torch.equal).
        self.generator = torch.Generator().manual_seed(seed + 1)
        self.shape = tuple(shape)
        self.fields = []

    def begin_step(self, final_step):
        self.fields = [] if final_step else [
            torch.randn(self.shape, generator=self.generator) for _ in range(2)]

    def sampler_for(self, wy0, wy1, wx0, wx1):
        fields = self.fields
        state = {"draws": 0}

        def noise_sampler(sigma, sigma_next):
            index = state["draws"]
            state["draws"] += 1
            if index >= len(fields):
                raise RuntimeError(
                    f"sampler requested SDE draw {index + 1} of {len(fields)} this step — "
                    "the two-draws-per-step contract (seeds_2, r=1) no longer holds")
            return fields[index][..., wy0:wy1, wx0:wx1].to(device=sigma.device, dtype=torch.float32)

        return noise_sampler


# ------------------------------------------------------------------ lead ring (arm 22)

@contextlib.contextmanager
def lead_ring_patch(model, sigma_first):
    """Arm 22: the shipped lead curve on the sync raw ring. The shipped
    anchor_ring_schedule cannot be reused: it normalizes sigma_first PER CALL, and a
    per-step call has sigmas[0] == the current sigma, so r == 1 at eval 1 and the
    lead never engages. This mirrors its instance-patch body exactly but pins
    sigma_first to the RUN's sigmas[0]; the CURVE still resolves through
    sampling.anchor_ring_factor at call time — the module's own A/B swap point, so a
    curve experiment swaps one function and this patch follows. Deliberate deviation
    from the harness convention "override the factor, never patch comfy": here the
    factor is untouched and only its NORMALIZATION is fixed, which no module-level
    override can express. Patches the model INSTANCE (core caches model objects
    session-wide — a class patch would leak into the app) and restores in finally,
    exactly like the shipped context manager. sample_latent runs with
    anchor_ring=False, so the shipped schedule never shadows this patch."""
    from context_anchored_tile_refine import sampling

    if sigma_first <= 0.0:
        raise RuntimeError(f"lead ring needs sigma_first > 0, got {sigma_first}")
    base_model = model.model

    def scale_latent_inpaint(sigma, noise, latent_image, **kwargs):
        factor = sampling.anchor_ring_factor(float(sigma.reshape(-1)[0]) / sigma_first)
        scaled = sigma * factor
        # core's own reshape (model_base.py:409): sigma is per-batch-item and
        # broadcasts over the latent's trailing dims.
        shape = [scaled.shape[0]] + [1] * (len(noise.shape) - 1)
        return base_model.model_sampling.noise_scaling(scaled.reshape(shape), noise, latent_image)

    had_own = "scale_latent_inpaint" in vars(base_model)
    saved = vars(base_model).get("scale_latent_inpaint")
    base_model.scale_latent_inpaint = scale_latent_inpaint
    try:
        yield
    finally:
        if had_own:
            base_model.scale_latent_inpaint = saved
        else:
            del base_model.scale_latent_inpaint


# ------------------------------------------------------------------ dpmpp_2m stepping

def _dpmpp_2m_single_step(model, x, sigmas, extra_args=None, callback=None, disable=None,
                          state=None):
    """ONE dpmpp_2m step — comfy's sample_dpmpp_2m body (k_diffusion/sampling.py:796-818)
    verbatim, with the loop unrolled to a single [sigma_i, sigma_i+1] pair and the
    CROSS-STEP state (old_denoised, the previous sigma) carried through `state`, a
    per-tile dict the engine threads into every call. dpmpp_2m is a MULTI-STEP method:
    its 2M correction needs the previous step's denoised AND sigmas[i-1]; a naive
    per-step call would restart with old_denoised=None every step and silently degrade
    the whole run to first order. The final step (sigmas[1] == 0) takes the first-order
    branch exactly as the stock loop does (t_next = +inf -> sigma ratio 0, expm1 -> -1,
    x = denoised — same float semantics, same result). Deterministic: no noise_sampler,
    so the engine's StepNoise machinery is not used with this sampler.
    KNOWN RESIDUAL (review F1): the multistep memory is PER-TILE — old_denoised is
    this tile's pre-blend prediction, while the x it corrects was feather-blended
    with the neighbour's write in the 4-cell overlap bands, so a 0.5-weighted stale
    term enters those bands each step. Second-order in inter-tile disagreement and
    band-confined; single-step samplers carry none of it. If a band artifact appears
    under this sampler that arms 20-22 lacked, suspect this before geometry."""
    if state is None:
        raise RuntimeError("dpmpp_2m per-step needs the engine's per-tile state dict — "
                           "without it every step silently degrades to first order")
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()       # stock body kept verbatim
    t_fn = lambda sigma: sigma.log().neg()   # (incl. these lambda names)
    denoised = model(x, sigmas[0] * s_in, **extra_args)
    if callback is not None:
        callback({'x': x, 'i': 0, 'sigma': sigmas[0], 'sigma_hat': sigmas[0],
                  'denoised': denoised})
    t, t_next = t_fn(sigmas[0]), t_fn(sigmas[1])
    h = t_next - t
    old_denoised = state.get("old_denoised")
    sigma_prev = state.get("sigma_prev")
    if old_denoised is None or sigmas[1] == 0:
        x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
    else:
        h_last = t - t_fn(sigma_prev)
        r = h_last / h
        denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
        x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
    state["old_denoised"] = denoised.detach()
    state["sigma_prev"] = sigmas[0].detach()
    return x


# ------------------------------------------------------------------ the sync loop

@contextlib.contextmanager
def _quiet_progress_bars():
    """112 one-step sampler calls would print 112 tqdm bars; the per-step [sync]
    lines are the progress readout instead. PROGRESS_BAR_ENABLED is read at each
    sample_latent call, so flipping it here silences exactly this loop."""
    import comfy.utils

    saved = comfy.utils.PROGRESS_BAR_ENABLED
    comfy.utils.PROGRESS_BAR_ENABLED = False
    try:
        yield
    finally:
        comfy.utils.PROGRESS_BAR_ENABLED = saved


def _noop_callback(step, x0, x, total_steps):
    return None


def sync_refine(canvas_latent, canvas_noise, sigmas, tiles, guider, tile_positives,
                timings, ring_mode="live", raw_latent=None, settings=None):
    """Stage S: the synchronized stepping loop. canvas_latent is C_0 (raw window
    encodes, canonical space); returns C at sigma 0 (clean canonical latent).
    ring_mode (ARM_RING): "live" leaves the ring cells on the trajectory snapshot;
    "raw" swaps them for the preserved raw latent + the canvas epsilon, so the ring
    presents the adjacent-style clean signal at every eval; "lead" uses "raw"'s
    inputs unchanged — its whole delta is the lead_ring_patch render() holds around
    this loop.
    settings (default REFINE) supplies seed / context_overlap / sampler, so other
    harnesses (run_ab_sync_market.py) can drive this engine with their own scene
    config. Samplers: "exp_heun_2_x0_sde" (SDE — per-step canvas-shaped shared noise
    via StepNoise) and "dpmpp_2m" (deterministic multi-step — per-tile cross-step
    state via _dpmpp_2m_single_step). Anything else fails fast: every sampler's
    per-step call contract must be derived before it is trusted here."""
    import comfy.samplers

    from context_anchored_tile_refine import sampling

    settings = REFINE if settings is None else settings
    if ring_mode not in ("live", "raw", "lead"):
        raise RuntimeError(f"unknown ring mode {ring_mode!r}")
    if ring_mode in ("raw", "lead") and raw_latent is None:
        raise RuntimeError(f"ring mode {ring_mode!r} needs the preserved raw canvas latent")
    if settings.sampler not in ("exp_heun_2_x0_sde", "dpmpp_2m"):
        raise RuntimeError(
            f"sampler {settings.sampler!r} has no derived per-step contract — SDE draw "
            "count / cross-step state must be established before syncing with it")
    use_sde = settings.sampler == "exp_heun_2_x0_sde"
    steps = int(sigmas.shape[-1]) - 1
    step_noise = StepNoise(settings.seed, canvas_latent.shape) if use_sde else None
    sampler_states = None if use_sde else [{} for _ in tiles]
    pristine_conds = guider.original_conds

    # Per-tile constants: latent window, binary denoise mask (stock), latent feather
    # (stock curve over the 4-cell band; alpha broadcast to the latent's trailing dims).
    windows = []
    overlap_cells = settings.context_overlap // 8
    # The pixel curve's plateau (10% of a 32px band) pins the seam-most ~3 pixels at
    # exactly 1.0; at 4 latent cells 10% pins NOTHING (0.4 cell) and the seam-most
    # cell would sit at 0.945 — 5.5% of the neighbour bleeding into the seam line
    # every step (review finding 3). 0.5/band is the smallest plateau whose clamp
    # lands the seam-most cell CENTER exactly on 1.0; the falloff stays stock.
    latent_plateau = (max(sampling.FEATHER_PLATEAU, 0.5 / overlap_cells)
                      if overlap_cells else sampling.FEATHER_PLATEAU)
    for tile in tiles:
        crop = _latent_rect(tile.crop_rect)   # the /8 guard, not bare //8 (review 4)
        wy0, wy1, wx0, wx1 = crop.y0, crop.y1, crop.x0, crop.x1
        mask = sampling.tile_gradient(tile.crop_rect, tile.overlap_inner_rect, scale=8)
        alpha = sampling.feather_alpha(
            _latent_rect(tile.paste_rect), _latent_rect(tile.core),
            overlap_cells, tile.kept_top, tile.kept_left, plateau=latent_plateau)
        alpha = alpha.reshape((1, 1) + (1,) * (canvas_latent.ndim - 4) + alpha.shape)
        paste = _latent_rect(tile.paste_rect)
        # The ring indicator (mask == 0), broadcastable over the latent — the swap
        # site for ring_mode "raw". Complementary to the denoise mask by construction.
        ring_gate = (mask < 0.5).reshape((1, 1) + (1,) * (canvas_latent.ndim - 4) + mask.shape)
        windows.append((wy0, wy1, wx0, wx1, mask, alpha, paste, ring_gate))

    canvas = canvas_latent
    started = time.perf_counter()
    with _quiet_progress_bars():
        for i in range(steps):
            step_started = time.perf_counter()
            sigma_pair = sigmas[i:i + 2]
            if use_sde:
                step_noise.begin_step(final_step=(i == steps - 1))
            snapshot = canvas
            canvas = canvas.clone()
            ab_models.clear_cache()   # allocator reset once per outer step (machine notes)
            for tile_idx in range(len(tiles)):
                wy0, wy1, wx0, wx1, mask, alpha, paste, ring_gate = windows[tile_idx]
                latent_in = snapshot[..., wy0:wy1, wx0:wx1].contiguous()
                # Step 0 is the stock construction (seed-42 canvas slice); every later
                # step resumes with zero noise so the (1-sigma)*z identity is exact.
                noise_in = (canvas_noise[..., wy0:wy1, wx0:wx1].contiguous() if i == 0
                            else torch.zeros_like(latent_in))
                if ring_mode in ("raw", "lead") and i > 0:
                    # The arm-21 delta: ring cells present the CLEAN raw signal.
                    # KSamplerX0Inpaint re-noises them from these two tensors at every
                    # eval — sigma*eps + (1-sigma)*raw, the same epsilon all steps
                    # (stock's within-call behaviour). Step 0 is already exactly this
                    # everywhere; the diffused region's exact-resume inputs untouched.
                    # Ring mode "lead" uses the SAME inputs — its whole delta is the
                    # lead_ring_patch render() holds around this loop.
                    latent_in = torch.where(ring_gate, raw_latent[..., wy0:wy1, wx0:wx1], latent_in)
                    noise_in = torch.where(ring_gate, canvas_noise[..., wy0:wy1, wx0:wx1], noise_in)
                if use_sde:
                    sampler = comfy.samplers.ksampler(
                        settings.sampler,
                        extra_options={"noise_sampler": step_noise.sampler_for(wy0, wy1, wx0, wx1)})
                else:
                    # Deterministic multi-step sampler: the per-tile state dict carries
                    # old_denoised + sigma_prev across the 28 single-step calls.
                    sampler = comfy.samplers.KSAMPLER(
                        _dpmpp_2m_single_step,
                        extra_options={"state": sampler_states[tile_idx]})
                swapped = dict(pristine_conds)
                swapped["positive"] = tile_positives[tile_idx]
                guider.original_conds = swapped
                try:
                    out = sampling.sample_latent(
                        guider, sampler, sigma_pair, noise_in, settings.seed, latent_in,
                        denoise_mask=mask, callback=_noop_callback, anchor_ring=False)
                finally:
                    guider.original_conds = pristine_conds
                out = out.detach().to("cpu", torch.float32)
                # Directional write-back: paste_rect only (core + kept top/left bands;
                # the ring and non-kept overlap were context, discarded exactly as the
                # stock composite discards them).
                sub = out[..., paste.y0 - wy0:paste.y1 - wy0, paste.x0 - wx0:paste.x1 - wx0]
                region = canvas[..., paste.y0:paste.y1, paste.x0:paste.x1]
                canvas[..., paste.y0:paste.y1, paste.x0:paste.x1] = alpha * sub + (1.0 - alpha) * region
            print(f"[sync]    step {i + 1:>2}/{steps}  sigma {float(sigma_pair[0]):.4f} -> "
                  f"{float(sigma_pair[1]):.4f}  {time.perf_counter() - step_started:.1f}s")
    timings["sync-steps"] = time.perf_counter() - started
    return canvas


def decode_composite(vae, canvas_latent, padded, tiles, timings, overlap=None):
    """Stage D: per-tile decode of the final windows, stock pixel feather, raster
    order. Both sides of a band decode the SAME latent, so the only band delta is
    VAE window framing — measured and printed, deliberately not corrected.
    overlap defaults to this scene's REFINE.context_overlap; other harnesses pass
    their own."""
    from context_anchored_tile_refine import sampling

    overlap = REFINE.context_overlap if overlap is None else overlap
    started = time.perf_counter()
    result = padded.clone()
    for tile in tiles:
        crop, paste, core = tile.crop_rect, tile.paste_rect, tile.core
        wy0, wx0 = crop.y0 // 8, crop.x0 // 8
        window = canvas_latent[..., wy0:crop.y1 // 8, wx0:crop.x1 // 8]
        decoded = vae.decode(window)
        if decoded.ndim == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        decoded = decoded.detach().to("cpu", torch.float32)
        sub = decoded[:, paste.y0 - crop.y0:paste.y1 - crop.y0, paste.x0 - crop.x0:paste.x1 - crop.x0, :]
        region = result[:, paste.y0:paste.y1, paste.x0:paste.x1, :]
        alpha = sampling.feather_alpha(paste, core, overlap,
                                       tile.kept_top, tile.kept_left)[..., None]
        band = (alpha[..., 0] > 0.0) & (alpha[..., 0] < 1.0)
        if bool(band.any()):
            delta = float((sub - region)[0][band].abs().mean()) * 255.0
            print(f"[decode]  tile r{tile.row}c{tile.col}: band framing delta "
                  f"{delta:.3f}/255 (same latent both sides; no DC step applied)")
        result[:, paste.y0:paste.y1, paste.x0:paste.x1, :] = alpha * sub + (1.0 - alpha) * region
    timings["decode-composite"] = time.perf_counter() - started
    return result


# ------------------------------------------------------------------ settings / render

def build_settings(arm, tile_captions, sigmas):
    ring_desc = {
        "live": "live same-sigma canvas — pure context, zero advance information, "
                "no lead schedule",
        "raw": "frozen RAW canvas re-noised to the current sigma (adjacent-style "
               "clean signal on all sides — stock's construction for a "
               "not-yet-refined neighbour). NOTE: this also deletes arm 20's "
               "inter-tile content coupling (the ring band lies inside the "
               "neighbour's core and carried its in-progress refinement in 20); "
               "remaining coupling = 32px write-back feather + shared SDE noise, "
               "and the never-advancing raw band butts refined content on all "
               "sides at low sigma — a second variable riding with 'clean signal'",
        "lead": "frozen RAW canvas presented on the SHIPPED lead curve — sigma_ring "
                "= sigma * anchor_ring_factor(sigma / sigma_first) with sigma_first "
                "= the run's sigmas[0] (per-call normalization would never engage); "
                "ring inputs byte-identical to arm 21, the lead patch is the only "
                "delta. Same coupling caveat as 'raw': no inter-tile content "
                "coupling beyond the 32px feather + shared noise",
    }[ARM_RING[arm]]
    payload = dataclasses.asdict(REFINE)
    payload.update({
        "run_label": arm,
        "arm": ARMS[arm],
        "method": "synchronized tiles: one pass, all tiles stepped together per sigma "
                  "on one shared canvas latent; per-step directional feather in latent "
                  "space; SDE noise shared canvas-wide per step",
        "conditioning": "arm 2: sliced vision rows + text-only captions",
        "sigmas": [float(v) for v in sigmas],
        "ring": ring_desc,
        # anchor_type stays greppable across the campaign's PNGs: "lead" must not
        # bucket as ringless (review finding 17).
        "anchor_type": ("lead (run-global sigma_first; see 'ring')"
                        if ARM_RING[arm] == "lead"
                        else "none (sync mode; ring presentation per 'ring')"),
        "seam_machinery": "none: no DC match, no min-error cut, no strips — the band "
                          "is coupled every step and decodes one shared latent",
        # Review finding 7: a verdict on a sync arm is a verdict on a BUNDLE. Each
        # rung narrows it by one component; state the residue per ring mode.
        "differs_from_arm2": (
            "synchronized stepping + ring content RAW on all sides (arm 2's ring = "
            "refined top/left neighbours) under the SAME lead curve (run-global "
            "normalization) + per-step latent feather replacing the pixel "
            "feather/DC/min-error cut"
            if ARM_RING[arm] == "lead" else
            "synchronized stepping + ring per 'ring' above (arm 2 ran the LEADING "
            "schedule on refined/raw neighbours) + per-step latent feather "
            "replacing the pixel feather/DC/min-error cut"),
        "base_png": str(BASE_PNG),
        "base_sha": _file_sha(BASE_PNG),
        "unet": UNET_NAME, "clip": CLIP_NAME, "clip_type": CLIP_TYPE, "vae": VAE_NAME,
        "upscale_model": UPSCALE_MODEL_NAME,
        "guider": "CFGGuider",
        "negative_prompt": NEGATIVE_PROMPT,
        "tile_captions": [row[0] for row in tile_captions],
        "harness": "tests-AB/run_ab_sync.py",
        "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return payload


def check_sync_preconditions(model, sigmas):
    """Every assumption the per-step resume identity and the ring construction rest
    on, checked before any GPU time. Shared with run_ab_sync_market.py."""
    import comfy.model_sampling

    from context_anchored_tile_refine import sampling

    if not torch.all(sigmas[1:] < sigmas[:-1]):
        raise RuntimeError("sigmas are not strictly decreasing")
    if float(sigmas[-1]) != 0.0:
        raise RuntimeError("schedule does not end at sigma 0")
    if float(sigmas[0]) >= 1.0:
        raise RuntimeError("sigma[0] >= 1 — offset_first_sigma_for_snr would rewrite it "
                           "and break the per-step resume identity")
    model_sampling = model.get_model_object("model_sampling")
    if not isinstance(model_sampling, comfy.model_sampling.CONST):
        raise RuntimeError("sync stepping's resume identity is derived for CONST "
                           f"(flow) models; got {type(model_sampling).__name__}")
    # model.model: the comfy BaseModel — _model_re_noises_anchor walks the BASE model's
    # __mro__ (the patcher's has no scale_latent_inpaint and would always return False).
    if not sampling._model_re_noises_anchor(model.model):
        raise RuntimeError("model overrides scale_latent_inpaint — the same-sigma ring "
                           "construction does not hold; sync mode is untested there")


def render(arm, padded, tiles, clip, model, vae, negative, empty, tile_captions):
    from context_anchored_tile_refine import upscale

    sigmas = upscale.build_sigmas(model, REFINE.scheduler, REFINE.steps, REFINE.denoise)
    steps = int(sigmas.shape[-1]) - 1
    check_sync_preconditions(model, sigmas)

    ring_mode = ARM_RING[arm]
    canvas_h, canvas_w = int(padded.shape[1]), int(padded.shape[2])
    print(f"[render]  {arm}: {steps} synchronized steps over {len(tiles)} tiles, "
          f"sigma {float(sigmas[0]):.4f} -> 0, canvas {canvas_w}x{canvas_h}, "
          f"ring '{ring_mode}'")

    timings = {}
    total_started = time.perf_counter()

    # ---- conditioning: arm 2's builder, one canvas encode + per-tile text captions
    started = time.perf_counter()
    with torch.inference_mode():
        tile_positives = split_run.make_split_builder("text-only")(
            clip, padded, tiles, tile_captions)
    timings["conds"] = time.perf_counter() - started

    # ---- stage E: C_0 from raw window encodes; the stock canvas-noise draw
    started = time.perf_counter()
    ab_models.clear_cache()
    with ab_models.VramProbe() as probe, torch.inference_mode():
        canvas_latent = encode_canvas_latent(vae, padded, tiles)
    timings["encode"] = time.perf_counter() - started
    print(f"[encode]  C_0 assembled {tuple(canvas_latent.shape)} in {timings['encode']:.1f}s  {probe}")

    latent_time = (1,) if getattr(vae, "latent_dim", 2) == 3 else ()
    dummy = torch.zeros((1, vae.latent_channels, *latent_time, canvas_h // 8, canvas_w // 8),
                        dtype=torch.float32)
    canvas_noise = upscale.Noise_RandomNoise(REFINE.seed).generate_noise({"samples": dummy})
    if tuple(canvas_noise.shape) != tuple(canvas_latent.shape):
        raise RuntimeError(f"canvas noise {tuple(canvas_noise.shape)} != latent canvas "
                           f"{tuple(canvas_latent.shape)}")

    # ---- stage S. Ring modes "raw"/"lead" preserve C_0 before the trajectory
    # overwrites it: the ring swap reads the clean raw latent for the whole run.
    # "lead" additionally holds the run-global lead patch around the loop.
    raw_latent = canvas_latent.clone() if ring_mode in ("raw", "lead") else None
    ring_patch = (lead_ring_patch(model, float(sigmas[0])) if ring_mode == "lead"
                  else contextlib.nullcontext())
    guider = upscale.build_guider(model, empty, negative, REFINE.cfg)
    with ab_models.VramProbe() as probe, torch.inference_mode(), ring_patch:
        canvas_latent = sync_refine(canvas_latent, canvas_noise, sigmas, tiles, guider,
                                    tile_positives, timings,
                                    ring_mode=ring_mode, raw_latent=raw_latent)
    print(f"[sync]    {steps} steps x {len(tiles)} tiles in {timings['sync-steps']:.1f}s  {probe}")

    # ---- stage D
    ab_models.clear_cache()
    with ab_models.VramProbe() as probe, torch.inference_mode():
        result = decode_composite(vae, canvas_latent, padded, tiles, timings)
    print(f"[decode]  composite in {timings['decode-composite']:.1f}s  {probe}")

    ab_models.require_image_shape(result, canvas_h, canvas_w, "sync refine")
    raw_luma = mean_luma(padded)
    print(f"[tone]    final luma delta vs raw: {mean_luma(result) - raw_luma:+.2f}/255"
          "  (single-pass band: +0.6; confluence arm 17: +3.62)")
    destination = output_path(arm)
    written = ab_models.save_png(destination, result.cpu(),
                                 build_settings(arm, tile_captions, sigmas))
    timings["total"] = time.perf_counter() - total_started
    print(f"[render]  {arm} -> {destination.name} {written[0]}x{written[1]}")
    print("[timing]  " + "  ".join(f"{name}={seconds:.1f}s" for name, seconds in timings.items()))


# ------------------------------------------------------------------ main

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
        for arm in ARMS:
            marker = "*" if arm in selected else " "
            print(f"{marker} {arm:<20} {ARMS[arm]}  -> {output_path(arm).name}")
        return 0

    root, note = ab_env.bootstrap()
    print(f"[env]     ComfyUI {ab_env.version(root)} at {root}  ({note})")
    print("[env]     torch {}  cuda {}  device {}".format(
        torch.__version__, torch.version.cuda,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))

    base = krea2_run.load_base()
    canvas = krea2_run.stage_upscale(base, force=False)
    padded, tiles = krea2_run.solve_tiles(canvas)
    if tuple(padded.shape[1:3]) != tuple(canvas.shape[1:3]):
        raise SystemExit("canvas is not /8 — the sync loop's latent math assumes the "
                         "padded and working canvases coincide")

    print(f"[clip]    loading {CLIP_NAME} ({CLIP_TYPE})")
    with ab_models.VramProbe() as probe:
        clip = ab_models.load_clip(CLIP_NAME, CLIP_TYPE)
        with torch.inference_mode():
            from context_anchored_tile_refine import upscale
            negative = ab_models.encode_prompt(clip, NEGATIVE_PROMPT)
            empty = upscale.encode_empty(clip)
        tile_captions = split_run.load_or_generate_captions(clip, padded, tiles, args.force_captions)
    print(f"[clip]    conditioning prepared  {probe}")

    print(f"[unet]    loading {UNET_NAME}")
    with ab_models.VramProbe() as probe:
        model = ab_models.load_unet(UNET_NAME)
        vae = ab_models.load_vae(VAE_NAME)
    print(f"[unet]    loaded  {probe}")

    for arm in selected:
        destination = output_path(arm)
        if destination.is_file() and not args.force:
            print(f"[render]  {arm:<20} exists, skipped (--force to redo)")
            continue
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
        print(f"[done]    {destination.name:<40} {width:>4}x{height:<4} {verdict}")
    if bad:
        raise SystemExit("[done]    FAILED: " + "; ".join(bad))
    return 0


if __name__ == "__main__":
    sys.exit(main())
