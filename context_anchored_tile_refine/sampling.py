import logging
import math

import torch

from . import captions, conds, grid

# How a seam is hidden is not configurable, by design: the node exposes only the geometry
# that genuinely depends on the image (the tile caps, context_anchor, context_overlap) and
# bakes in everything that A/B testing settled. Four alternatives were built, compared on a
# night sky and a close-up face, and removed once they lost:
#   a straight transition          an unbroken line is easy to spot even when it is faint
#   a coherent-noise meander       organic, but reads as random next to a routed cut
#   a ramp landing exactly on the  measured identical to the shipped feather; the routed
#     routed cut                     path's benefit does not depend on landing precisely
#   a hard cut, no blending        tore at a high-contrast silhouette edge
# The last two matter for anyone tempted to re-add them: all placements measured the same
# tile-level DC step to within 0.05/255, because that step spans the WHOLE tile and no
# handover position inside a 32px band can move it. Only the hard cut differed visually,
# and it lost.

# Inter-tile feather curve, settled by visual A/B in ComfyUI (the only judge that counts
# here) across character/landscape scenes and a deliberate worst case: a night sky with a
# radiating moon, where a smooth luminance gradient crosses the seam and hides nothing.
# Both were node widgets during tuning and are now baked in, because both optima were
# one-sided rather than scene-dependent:
#   plateau 10% — the plateau only has to cover the seam-adjacent pixels; past that it
#     spends band on a hard-edged solid region instead of on the fall-off that hides the DC
#     step, so wider strictly wastes the band. At every /8 overlap 10% pins the seam-adjacent
#     pixel of the UNDISPLACED ramp to exactly 1.0, which is the whole job. (A seam mode can
#     displace it slightly below 1.0 — at overlap 8, to ~0.95 worst case; harmless, because
#     the band holds the neighbor's already-refined output, never raw canvas.)
#   falloff 2.0 — the lowest exponent that still meets 0 with zero slope (no slope
#     discontinuity against the neighbor); higher made the fall-off itself visible.
# Scene difficulty is absorbed by context_overlap instead (32 normally, up to 128 for a
# large smooth gradient), which widens the band without changing the curve's shape.
FEATHER_PLATEAU = 0.10
FEATHER_FALLOFF = 2.0

# The frozen context_anchor ring is re-presented to the model at EVERY step: comfy
# samplers.py:639 rebuilds the model's input as
#     x*denoise_mask + scale_latent_inpaint(x, sigma, noise, latent_image)*(1 - denoise_mask)
# and core's default scale_latent_inpaint (comfy model_base.py:408) re-noises the frozen
# region to the CURRENT sigma. At denoise 0.35 the first sigma is 0.63, so the ring reaches
# the model as ~63% noise exactly while composition is being decided — the anchor is barely
# legible when it matters most. That default is principled (one consistent noise level across
# the latent), not a bug; the video families deliberately trade it away instead, overriding
# scale_latent_inpaint to return the clean latent ("Hold anchor constant across all sigmas",
# comfy model_base.py:2058 WAN22, also WAN21 :1919 / HunyuanVideo :1391 / LTXAV :1237).
# Those models are LEFT ALONE — holding clean is the endpoint this schedule approaches.
#
# For every model on core's default, the ring is instead presented at
# `sigma * anchor_ring_factor(r)`, r = sigma/sigma_first:
#
#   r  1.00 ...... 0.33  0.28  0.23  0.18  0.12  0.06  0.00
#   f  1.00 ...... 0.33  0.28  0.23  0.18  0.21  0.65  1.00
#      \___ the ring LEADS the core ___/    \__ released __/
#
# Down to r == RELEASE the ring leads: matched to the core at step 0, so there is no noise
# discontinuity while structure forms, then denoising faster so it resolves FIRST and the core
# follows it. Below RELEASE it smoothstep-rejoins the core (continuous by construction: the
# blend weight is 0 at r == RELEASE), so the last steps generate their own texture instead of
# copying the neighbour's.
#
# Settled by owner A/B at production scale (2304x3072, 4 tiles, context_anchor 256,
# context_overlap 32, denoise 0.35, 28 steps) against core's plain re-noised default and five
# other curves. Against the default this removes freckle amplification and a white spotting
# artifact, improves foliage texture, and holds the joint at 0.17x the seam-free control's own
# column-to-column variation. Two neighbours measured worse and are not shipped: releasing at
# 0.30 (drifts, because the reference is withdrawn while structure is still settling) and a
# gentler sqrt lead across the whole schedule (moves the subject 4x as much for less texture —
# coherence is bought in the EARLY steps). 0.15 is where fine detail starts arriving on a
# 28-step schedule.
#
# This changes only what the model SEES. comfy samplers.py:642 restores the frozen pixels in
# the sampler's OUTPUT unconditionally, so no finished pixel is ever re-diffused.
ANCHOR_RING_RELEASE = 0.15


def anchor_ring_factor(r):
    # Ring sigma as a fraction of the core's, at denoising progress r = sigma/sigma_first.
    # 1.0 == the ring carries the same noise as the core (core's own default); 0.0 == clean.
    # Clamped because an SDE/ancestral sampler may evaluate at a sigma_hat above sigma_first,
    # and the ring must never be noisier than the core it is anchoring.
    # Blending the still-falling lead against 1.0 with a smoothstep (zero slope at the
    # handover) makes the factor dip fractionally below the lead line for the first part of the
    # release window before climbing — 0.00224 at worst, 1.5% of RELEASE, and part of the curve
    # the owner's A/B picked, so it stands. tests/test_anchor_ring.py pins its size.
    r = min(1.0, max(0.0, r))
    if r > ANCHOR_RING_RELEASE:
        return r
    u = 1.0 - r / ANCHOR_RING_RELEASE
    return r + (1.0 - r) * (u * u * (3.0 - 2.0 * u))     # smoothstep: flat slope at both ends


def _model_re_noises_anchor(model):
    # True only for models on core's DEFAULT scale_latent_inpaint. Walking __mro__ keeps this
    # comfy-free (module scope stays torch-only) and needs no list of model classes: anything
    # that overrides the method already manages the frozen region itself, and anything without
    # the method has no frozen-region behaviour to schedule.
    for cls in type(model).__mro__:
        if "scale_latent_inpaint" in vars(cls):
            return cls.__name__ == "BaseModel"
    return False


def anchor_ring_schedule(guider, sigmas, enabled=True):
    # Applies anchor_ring_factor for the duration of ONE sampler run. Patches the model
    # INSTANCE, never the class: core caches model objects for the whole session, so a class
    # patch would follow the model into every other node in the graph. Restored in `finally`.
    # `enabled` is the per-path gate: False from the raster path (_refine_tiles, the base
    # node), and on the sync path whatever anchor_source resolves to (sync._refine_canvas).
    import contextlib

    @contextlib.contextmanager
    def noop():
        yield

    model = getattr(getattr(guider, "model_patcher", None), "model", None)
    if not enabled or model is None or sigmas.numel() == 0 or not _model_re_noises_anchor(model):
        return noop()
    sigma_first = float(sigmas.reshape(-1)[0])
    if sigma_first <= 0.0:
        return noop()

    @contextlib.contextmanager
    def patched():
        def scale_latent_inpaint(sigma, noise, latent_image, **kwargs):
            # anchor_ring_factor is resolved through the module at call time, so an A/B
            # harness can swap the curve at one point without patching comfy at all.
            factor = anchor_ring_factor(float(sigma.reshape(-1)[0]) / sigma_first)
            scaled = sigma * factor
            # core's own reshape (comfy model_base.py:409): sigma is per-batch-item and has to
            # broadcast over the latent's trailing dims.
            shape = [scaled.shape[0]] + [1] * (len(noise.shape) - 1)
            return model.model_sampling.noise_scaling(scaled.reshape(shape), noise, latent_image)

        had_own = "scale_latent_inpaint" in vars(model)
        saved = vars(model).get("scale_latent_inpaint")
        model.scale_latent_inpaint = scale_latent_inpaint
        try:
            yield
        finally:
            if had_own:
                model.scale_latent_inpaint = saved
            else:
                del model.scale_latent_inpaint

    return patched()


def pad_image_to_multiple(image, multiple=8):
    # [B,H,W,C] → reflect-pad right/bottom onto the /multiple grid; returns the image
    # unchanged (no copy) when already aligned, plus the original (H, W) for crop-back.
    height = image.shape[1]
    width = image.shape[2]
    pad_h = (-height) % multiple
    pad_w = (-width) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, (height, width)
    # Reflect is always valid: pad <= multiple - 1 < multiple <= dim (node validates H,W >= 8).
    padded = torch.nn.functional.pad(image.movedim(-1, 1), (0, pad_w, 0, pad_h), mode="reflect")
    return padded.movedim(1, -1), (height, width)


def crop_image_to(image, height, width):
    return image[:, :height, :width, :]


def tile_gradient(crop, core, scale=1):
    # Binary denoise-mask indicator: 1 for
    # every cell whose center lies inside the `core` rect, 0 outside. Cell centers
    # (k+0.5) so one formula works at any grid resolution via `scale`; the pipeline uses
    # scale=8 and passes overlap_inner_rect as `core` (core+overlap diffused at full
    # strength, context_anchor ring frozen). The mask stays binary because ComfyUI's static
    # inpaint blend (KSamplerX0Inpaint) re-applies denoise_mask on every step, so a
    # fractional cell is only ever partially denoised — at low step counts (z-image turbo
    # ~8 steps) that leaves an under-refined halo along the seam. A binary {0,1} mask
    # fully diffuses every released cell and hard-freezes the rest, with no partial band.
    w = (crop.x1 - crop.x0) // scale
    h = (crop.y1 - crop.y0) // scale
    xs = torch.arange(w, dtype=torch.float32) + 0.5
    ys = torch.arange(h, dtype=torch.float32) + 0.5
    dx = ((core.x0 - crop.x0) / scale - xs).clamp(min=0) + (xs - (core.x1 - crop.x0) / scale).clamp(min=0)
    dy = ((core.y0 - crop.y0) / scale - ys).clamp(min=0) + (ys - (core.y1 - crop.y0) / scale).clamp(min=0)
    d = torch.sqrt(dx[None, :] ** 2 + dy[:, None] ** 2)
    return (d == 0).to(torch.float32)


def _feather_ramp(band, plateau, k):
    # Weight of THIS tile across one kept overlap band, cell-centered: index 0 = the outer
    # edge (against the already-processed neighbor), index band-1 = the seam-adjacent pixel.
    # Shape: a SOLID plateau over the seam-most `plateau` fraction (held at 1.0), then a fast
    # base^k fall-off to a soft feather at the outer edge. For k > 1 the tail meets 0 with
    # zero slope, so there is no slope discontinuity against the neighbor's pure content (the
    # junction that turns a DC mismatch into a visible seam line). `base` is the pre-exponent
    # tile weight: u / (1 - plateau) clamped to 1, where u = (i+0.5)/band rises 0->1 outer->
    # seam and reaches 1.0 once inside the plateau (u >= 1 - plateau). Normalizing by the
    # ACTUAL band width keeps the plateau/fall-off proportional even where the band is clamped
    # narrower than the nominal `overlap`. plateau=0, k=1 gives base = u, so that pair
    # reproduces the original linear ramp (i+0.5)/band bit for bit at any band width (no
    # 1-(1-u) round-off — division by 1.0 and clamp are exact for u in (0, 1)).
    u = (torch.arange(band, dtype=torch.float32) + 0.5) / band   # 0..1, outer -> seam
    return _ramp_curve(u, plateau, k)


def _ramp_curve(u, plateau, k):
    # The feather curve evaluated at arbitrary band coordinates `u` (any shape). Split out
    # of _feather_ramp so the displaced (2-D) seam modes reuse the exact same curve — only
    # WHERE it is sampled changes, never its shape. The clamp to [0,1] is what makes every
    # seam mode safe: it bounds alpha, and it guarantees the base is never negative, so a
    # fractional `k` can never produce NaN no matter how far a displacement pushes `u`.
    denom = max(1e-6, 1.0 - plateau)                             # guard plateau -> 1
    base = (u / denom).clamp(0.0, 1.0)                           # 1.0 through the plateau
    return base ** k


def _displaced_ramp(band, length, plateau, k, disp):
    # The band ramp as a 2-D field [band, length]: the same curve, but its position slides
    # along the seam. `disp` is a [length] displacement in band-normalized units (+ pushes
    # the transition toward the seam, - toward the neighbor), so the straight iso-alpha
    # contour becomes a curve the eye cannot lock onto.
    #
    # The displacement is tapered by sin(pi*u), which vanishes at BOTH band ends and peaks
    # mid-band (u=0.5) — near, but not exactly at, the baked curve's 50% crossing (u~0.64).
    # That keeps the transition from being shoved hard against either boundary (which would
    # re-create a step at the core edge or at the outer edge of paste_rect) while leaving the
    # perceptual midpoint largely free to wander. The cost is that a requested displacement is
    # attenuated, and more so the closer it aims to a band end; see _path_displacement.
    # It is a continuity aid, not a safety device: alpha is exactly 1.0 through
    # the core because the ramp is only ever written into the band slice (see feather_alpha).
    u = ((torch.arange(band, dtype=torch.float32) + 0.5) / band)[:, None]   # [band,1]
    t = u + disp[None, :] * torch.sin(u * math.pi)                          # [band,length]
    return _ramp_curve(t, plateau, k)


def feather_alpha(paste_rect, core, overlap, kept_top, kept_left, plateau=FEATHER_PLATEAU, k=FEATHER_FALLOFF, disp_top=None, disp_left=None):
    # Directional-feather alpha at NATIVE pixel resolution (no resize): an [h, w]
    # float32 weight over paste_rect (h/w = paste_rect dims), for the post-composite
    # cross-dissolve of this tile into its already-processed top/left neighbor.
    #
    # Two independent per-axis ramps (see _feather_ramp), cell-centered. The top overlap
    # band is the rows of paste_rect above the core (present iff kept_top; `overlap` px, or
    # fewer if clamped at the canvas edge); over it ay holds 1.0 across the seam-most
    # `plateau` fraction and then falls off to ~0 at the outer edge, and ay = 1 through the
    # core itself. So the SEAM pixel (the core edge) is 100% this tile — the already-refined
    # edge stays crisp — while the neighbor fades back in across the band. The left band is
    # analogous on x. alpha = ay (x) ax; the corner is the product of the two ramps.
    #
    # With no kept side (overlap == 0, or a top-left tile) both ramps are all-ones, so the
    # composite reduces to the fe6f6a7 hard core paste, byte-identical, regardless of
    # plateau/k. plateau=0, k=1 reproduces the original linear-ramp feather exactly.
    #
    # disp_top ([w]) / disp_left ([h]) optionally slide the transition along each seam (the
    # baked-in min_error routing) so it stops reading as a straight line; None keeps the
    # straight 1-D ramp, bit-for-bit. A displaced side becomes a 2-D field, but the
    # result is still the product of a vertical and a horizontal factor, so the corner stays
    # the product of the two seams' curves. CRITICAL: the ramp is only ever written into the
    # band slice, so every core row/column stays exactly 1.0 under any displacement — the
    # core must never blend, because `region` there is RAW un-refined canvas, not a
    # neighbor's output (no tile has pasted into this tile's core yet).
    h = paste_rect.y1 - paste_rect.y0
    w = paste_rect.x1 - paste_rect.x0
    top_band = core.y0 - paste_rect.y0   # rows of paste_rect above the core
    left_band = core.x0 - paste_rect.x0  # cols of paste_rect left of the core
    use_top = kept_top and overlap > 0 and top_band > 0
    use_left = kept_left and overlap > 0 and left_band > 0

    if use_top and disp_top is not None:
        ay = torch.ones(h, w, dtype=torch.float32)
        ay[:top_band] = _displaced_ramp(top_band, w, plateau, k, disp_top)
    else:
        ay = torch.ones(h, dtype=torch.float32)
        if use_top:
            ay[:top_band] = _feather_ramp(top_band, plateau, k)
        ay = ay[:, None]

    if use_left and disp_left is not None:
        ax = torch.ones(h, w, dtype=torch.float32)
        # _displaced_ramp lays the band out along dim 0; the left band spans columns.
        ax[:, :left_band] = _displaced_ramp(left_band, h, plateau, k, disp_left).T
    else:
        ax = torch.ones(w, dtype=torch.float32)
        if use_left:
            ax[:left_band] = _feather_ramp(left_band, plateau, k)
        ax = ax[None, :]

    return (ay * ax).clamp(0.0, 1.0)


def _min_error_path(err):
    # Efros-Freeman minimum-error boundary cut (Image Quilting, SIGGRAPH 2001), 1-D DP.
    # err: [band, length] squared difference between the two tiles across the overlap band.
    # Returns [length] int64 band indices — the row (per column) where the handoff should
    # happen — constrained to move at most one step per column, so the cut is connected.
    # Routing the seam through pixels where the two refinements ALREADY agree beats hiding a
    # mismatch: where err ~ 0 there is nothing left to see across the handoff.
    # The scan is inherently sequential along the seam, so it is a Python loop over columns.
    # It runs in numpy on preallocated buffers, not torch: the per-column arrays are tiny
    # (3 x band) and torch's per-op dispatch overhead dominates there — numpy is ~10x faster
    # for the same arithmetic. Imported lazily to keep the module scope torch-only.
    import numpy as np

    e = err.detach().cpu().numpy().astype(np.float32, copy=False)
    band, length = e.shape
    cum = np.empty((band, length), dtype=np.float32)
    back = np.zeros((band, length), dtype=np.int8)
    cum[:, 0] = e[:, 0]
    tri = np.empty((3, band), dtype=np.float32)   # [toward-outer, same-row, toward-seam]
    inf = np.float32(np.inf)                      # off-band predecessors are unreachable
    for j in range(1, length):
        prev = cum[:, j - 1]
        tri[0, 0] = inf
        tri[0, 1:] = prev[:-1]
        tri[1] = prev
        tri[2, :-1] = prev[1:]
        tri[2, -1] = inf
        cum[:, j] = e[:, j] + tri.min(axis=0)
        back[:, j] = tri.argmin(axis=0) - 1       # 0/1/2 -> predecessor offset -1/0/+1
    path = np.empty(length, dtype=np.int64)
    path[-1] = int(cum[:, -1].argmin())
    for j in range(length - 1, 0, -1):
        path[j - 1] = path[j] + back[path[j], j]
    return torch.from_numpy(path)


def _path_displacement(err, band, plateau, k):
    # Turn a minimum-error path into a band-normalized displacement for _displaced_ramp.
    # u50 is where the undisplaced curve crosses alpha = 0.5, so (u50 - u_path) is the shift
    # that WOULD put that perceptual midpoint of the feather on the routed cut.
    #
    # It BIASES the blend toward the cut rather than landing on it. _displaced_ramp tapers
    # every displacement by sin(pi*u), which vanishes at both band ends, so the realized shift
    # is only a fraction of this one — most of it near mid-band, none at the edges. The
    # response is monotone (a cut nearer the seam always pulls the blend nearer the seam) but
    # compressed: at band 32 the reachable 50%-crossings are rows 7..26, so a cut requested at
    # row 0 lands near row 7. That compression is deliberate — an untapered shift would shove
    # the transition hard against the core edge or the outer edge and re-create exactly the
    # step the feather exists to hide. A variant that landed exactly on the cut was built and
    # compared: it measured the same and looked the same, so the attenuation costs nothing.
    path = _min_error_path(err)
    u50 = (0.5 ** (1.0 / k)) * (1.0 - plateau)
    u_path = (path.to(torch.float32) + 0.5) / band
    return u50 - u_path


def _normalize_denoise_mask(denoise_mask, latent_samples, load_device):
    # The canonical broadcastable denoise mask — [B,1,h,w] for a 4-D latent, [B,1,1,h,w]
    # for a 5-D video-family latent — float32 on the guider's load device: the FIXED
    # POINT of comfy.sampler_helpers.prepare_mask (whose full output is the
    # channel-expanded mask). A well-behaved guider re-runs prepare_mask over this form
    # and gets value-identical results; a guider pack whose copied sample() predates core
    # moving that prep out of outer_sample would otherwise receive a CPU tensor and crash
    # against a CUDA latent in KSamplerX0Inpaint. The single channel — never the
    # channel-expanded form — is what survives re-preparation at every batch size:
    # prepare_mask flattens leading dims to (-1,1,h,w) before re-expanding, which
    # batch-scrambles a channel-expanded mask at B>1. The mask's ndim must match the
    # latent's for the same reason: reshape_mask folds any sub-5-D mask headed for a 5-D
    # latent to (1,1,-1,h,w), pooling the batch into the temporal axis at B>1. Spatial
    # size always matches the tile latent by construction, so this is a pure view plus a
    # device/dtype move that no-ops on the CPU path — binary values untouched.
    # Both engines normalize HERE: the raster path through sample_latent below, and the
    # sync engine (sync.py), whose lanes are handed to the stepper pre-normalized because
    # it calls guider.sample itself. One implementation, so the two can never diverge.
    mask_shape = (-1, 1) + (1,) * (latent_samples.ndim - 4) + denoise_mask.shape[-2:]
    return denoise_mask.reshape(mask_shape).to(device=load_device, dtype=torch.float32)


def sample_latent(guider, sampler, sigmas, noise_tensor, seed, latent_samples, denoise_mask=None, callback=None, anchor_ring=True):
    # The per-tile reuse seam. Mirrors SamplerCustomAdvanced.sample with three deliberate
    # deviations: no fix_empty_latent_channels (the latent always comes from a real
    # vae.encode), no x0_output dict (the denoised output is unused; previews work
    # without it), and the denoise mask is handed over pre-normalized (below).
    # denoise_mask/callback are injectable for the later tiling feature.
    import comfy.model_management
    import comfy.utils
    import latent_preview

    if callback is None:
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1)
    if denoise_mask is not None:
        denoise_mask = _normalize_denoise_mask(denoise_mask, latent_samples, guider.model_patcher.load_device)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    # anchor_ring is False from _refine_tiles (the base node's raster path); the VL path's
    # lead schedule is entered by the sync engine itself, once around the whole run.
    with anchor_ring_schedule(guider, sigmas, enabled=anchor_ring):
        samples = guider.sample(noise_tensor, latent_samples, sampler, sigmas, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    return samples.to(comfy.model_management.intermediate_device())


def make_tile_progress(model_patcher, steps, n_tiles, batch_size=1, batch_index=0):
    # latent_preview.prepare_callback replicated at aggregate scale: one ProgressBar
    # spanning steps * n_tiles * batch_size so the UI shows a single run across every tile of
    # every picture. batch_size/batch_index come from refine_image's picture loop: every
    # picture reports on the SAME total and starts where the previous picture ended, so the
    # bar advances once across the whole run instead of restarting at each picture.
    import comfy.utils
    import latent_preview

    previewer = latent_preview.get_previewer(model_patcher.load_device, model_patcher.model.latent_format)
    per_picture = steps * n_tiles
    total = per_picture * batch_size
    offset = per_picture * batch_index
    pbar = comfy.utils.ProgressBar(total)

    def for_tile(tile_idx):
        def callback(step, x0, x, total_steps):
            preview = None
            if previewer:
                # Core wraps a nested-latent callback and hands x0 back as a NestedTensor
                # (comfy/samplers.py:1289-1295), which no previewer can decode; core's own
                # callback previews the first stream (latent_preview.py:126-127). MiniMax H3
                # HAS a previewer (latent_formats.py MiniMaxH3AV defines latent_rgb_factors),
                # so without this every video tile preview would crash. No-op for images.
                if x0.is_nested:
                    x0 = x0.tensors[0]
                preview = previewer.decode_latent_to_preview_image("JPEG", x0)
            pbar.update_absolute(offset + tile_idx * steps + step + 1, total, preview)
        return callback

    return for_tile


def _mask_bbox(mask_bin):
    # Union bounding box over the whole batch of a [B,H,W] boolean mask, as
    # (y0, y1, x0, x1) with EXCLUSIVE y1/x1 (max + 1); None when the mask is empty.
    # The union still runs, but no node reaches it with more than one row: refine_image's
    # picture loop hands each picture its OWN mask row, so every picture gets its own crop
    # and none is widened to cover another picture's region. The row extent and the column
    # extent are INDEPENDENT reductions, so each axis is collapsed to a [H] / [W] boolean
    # first: a 2-D torch.nonzero would materialize an [N,2] int64 row per masked pixel
    # (~200 MB on half a 4096x6144 canvas) to yield the same four integers.
    any_mask = mask_bin.any(dim=0)
    rows = any_mask.any(dim=1)
    cols = any_mask.any(dim=0)
    if not bool(rows.any()):
        return None
    ry = torch.nonzero(rows)
    rx = torch.nonzero(cols)
    y0 = int(ry.min())
    y1 = int(ry.max()) + 1
    x0 = int(rx.min())
    x1 = int(rx.max()) + 1
    return (y0, y1, x0, x1)


def _expand_snap_clamp(bbox, anchor, H, W):
    # Expand the bbox by `anchor` (the frozen-background halo the subject conditions
    # against), snap x0/y0 DOWN and x1/y1 UP to /8, then clamp to [0,H]/[0,W] and floor the
    # result at 8px. Snapping before clamping keeps interior edges on the /8 grid; a mask
    # touching a non-/8 image border yields a clamped, non-/8 crop that _refine_tiles'
    # internal pad then handles. That pad is reflect-mode, so it needs pad < dim: with
    # anchor=0 the clamp can leave fewer than 8px against a non-/8 border, hence the floor
    # (node.py guarantees H,W >= 8, so widening back always fits).
    y0, y1, x0, x1 = bbox
    y0 = ((y0 - anchor) // 8) * 8
    x0 = ((x0 - anchor) // 8) * 8
    y1 = ((y1 + anchor + 7) // 8) * 8
    x1 = ((x1 + anchor + 7) // 8) * 8
    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(H, y1)
    x1 = min(W, x1)
    if y1 - y0 < 8:
        y0 = max(0, y1 - 8)
        y1 = min(H, y0 + 8)
    if x1 - x0 < 8:
        x0 = max(0, x1 - 8)
        x1 = min(W, x0 + 8)
    return y0, y1, x0, x1


def _aa_alpha(sub_mask):
    # 1px anti-alias of a binary [B,ch,cw] mask: separable [0.25,0.5,0.25] blur (H then
    # W) with REPLICATE padding → [B,ch,cw] float in [0,1]. A binary step edge yields
    # ...,0,0.25,0.75,1,... (dyadic → exact f32, bitwise-pinnable). Replicate padding keeps
    # an all-ones mask all-ones, so a full-image / border-touching mask gets no dark seam.
    kernel = torch.tensor([0.25, 0.5, 0.25], dtype=sub_mask.dtype, device=sub_mask.device)
    x = sub_mask[:, None]  # [B,1,ch,cw]
    x = torch.nn.functional.pad(x, (0, 0, 1, 1), mode="replicate")
    x = torch.nn.functional.conv2d(x, kernel.view(1, 1, 3, 1))
    x = torch.nn.functional.pad(x, (1, 1, 0, 0), mode="replicate")
    x = torch.nn.functional.conv2d(x, kernel.view(1, 1, 1, 3))
    return x[:, 0]


def seam_displacements(tile, sub, region):
    # Per-kept-seam displacement, in band-normalized units: [w] for the top seam, [h] for the
    # left seam, None where that side has no already-processed neighbor. This is what bends
    # the feather's transition off a straight line and onto the minimum-error path.
    paste, core = tile.paste_rect, tile.core
    top_band = core.y0 - paste.y0
    left_band = core.x0 - paste.x0
    disp_top = None
    disp_left = None

    # Squared difference between this tile's refinement and the already-pasted neighbor,
    # averaged over batch and channels. Only the BAND is a valid error surface: inside the
    # core, `region` is raw un-refined canvas, so the surface is taken from each kept band
    # directly and the core is never computed. The reduction is elementwise per spatial
    # position, so band-slicing first yields the same values as slicing a full-tile surface.
    # The DP is small and strictly sequential, so it runs on the CPU.
    if tile.kept_top and top_band > 0:
        err_top = ((sub[:, :top_band, :, :] - region[:, :top_band, :, :]) ** 2).mean(dim=(0, 3)).float().cpu()   # [top_band, w]
        disp_top = _path_displacement(err_top, top_band, FEATHER_PLATEAU, FEATHER_FALLOFF)
    if tile.kept_left and left_band > 0:
        err_left = ((sub[:, :, :left_band, :] - region[:, :, :left_band, :]) ** 2).mean(dim=(0, 3)).float().cpu()   # [h, left_band]
        disp_left = _path_displacement(err_left.T.contiguous(), left_band, FEATHER_PLATEAU, FEATHER_FALLOFF)

    return disp_top, disp_left


# Per-tile DC match, always on (seam-placement-cannot-fix-tile-dc). Two tiles that diffused
# the SAME raw band land at slightly different DC levels, and that step spans the whole tile,
# so no feather placement can move it. Measured contributions: 0.03-0.74/255 from VAE crop
# framing alone (encode+decode of one strip inside two different crops, no sampler involved),
# plus a larger diffusion-framing term.
#
# The overlap band is the only place two tiles have refined the SAME raw pixels, so content
# cancels there exactly and what is left is pure tile-to-tile disagreement. Measuring a
# tile's core against the raw source instead does NOT work; it was built as a control arm and
# refuted. That difference is dominated by content (a detailed tile round-trips with a
# different bias than a flat one): VAE-only, on a face, the core-referenced estimate differed
# between tiles by 0.9-1.9/255 where the tiles actually agreed to within 0.27/255, so a
# correction built on it injects seam error rather than removing it.
#
# The reduction is the MEDIAN, picked over the mean in a blind flip-comparison. Not on shade —
# there the two are indistinguishable (0.20 vs 0.23/255 on the cleanest case) — but on SEAM
# ROUTING: the offset is subtracted before the min-error surface is built, and subtracting the
# median leaves that surface describing texture instead of a constant pedestal, so the DP
# discriminates between paths instead of tie-breaking on noise.
#
# The correction is additive and per-channel: the offset is partly chroma (per-channel
# peak-to-peak R 1.18 / G 1.25 / B 1.64 on a night sky, with differing signs), and the
# magnitude does not scale with luminance the way a multiplicative gain would.

# Safety rail only. Every measurement so far is ~1/255; a correction near this bound means
# the estimate is pathological (a decode failure, a mismatched neighbor) and clamping keeps
# it from painting a visible block, rather than being a tuning knob.
DC_MATCH_CLAMP = 4.0 / 255.0

# Floor on how many band samples the median may be taken over. It only ever binds on the mask
# path, where the band is gated to the masked region and can shrink to a sliver: the median's
# standard error is ~1.25*sigma/sqrt(N), and against locally large texture disagreement a
# handful of pixels can leave it sitting on an outlier — which would then shift a WHOLE tile.
# Below the floor the tile takes no correction at all, the safe direction; DC_MATCH_CLAMP is
# the backstop above it. The no-mask path never reaches this: its bands are ~36k samples.
DC_MATCH_MIN_SAMPLES = 256


def _band_samples(diff, keep, channels):
    # One band's [B,h,w,C] difference flattened to [N,C]. `keep` is that band's [B,h,w] binary
    # region gate, or None to take every pixel.
    if keep is None:
        return diff.reshape(-1, channels)
    return diff[keep > 0]


def seam_dc_offset(tile, sub, region, region_mask=None):
    # Per-channel additive offset that lands this tile's DC on its ALREADY-PLACED
    # neighbors', measured over the shared overlap band(s). Returns [C], or None when the
    # tile has no processed neighbor (the first tile, which therefore sets the gauge), when
    # there is no overlap band at all (context_overlap=0 — a silent no-op, not an error), or
    # when too few samples survive the region gate.
    #
    # sub and region both cover paste_rect. On a kept side its first `band` rows/columns
    # hold two INDEPENDENT refinements of the same raw pixels — this tile's in `sub`, the
    # neighbor's in `region` — which is what makes the difference content-free. These are
    # the same slices seam_displacements reduces as a squared error.
    #
    # region_mask is the paste_rect slice of the binary region mask at PIXEL resolution on the
    # mask path, None otherwise. Where it is given, only band pixels inside the region count:
    # those are the only ones both tiles actually diffused. Band pixels outside it are frozen
    # (denoise 0) and are discarded at composite time, so pooling them would derive the offset
    # partly from VAE crop-framing disagreement on pixels the output never uses and then apply
    # it to the foreground that it does.
    paste, core = tile.paste_rect, tile.core
    top_band = core.y0 - paste.y0
    left_band = core.x0 - paste.x0
    take_top = tile.kept_top and top_band > 0
    take_left = tile.kept_left and left_band > 0
    channels = sub.shape[-1]
    samples = []

    # The two bands are made DISJOINT — the left band starts below the top band — so the
    # band x band corner belongs to exactly one of them and is counted exactly once. Pooling
    # the samples before reducing (rather than combining two per-band medians) is what keeps
    # that true; under a median it also means the LARGER band's value wins outright when the
    # two neighbors disagree, which is the wanted behaviour — the wider shared edge is the
    # better-supported measurement, and an edge crossing the narrower seam cannot drag it.
    if take_top:
        samples.append(_band_samples(
            sub[:, :top_band, :, :] - region[:, :top_band, :, :],
            None if region_mask is None else region_mask[:, :top_band, :], channels))
    if take_left:
        y0 = top_band if take_top else 0
        samples.append(_band_samples(
            sub[:, y0:, :left_band, :] - region[:, y0:, :left_band, :],
            None if region_mask is None else region_mask[:, y0:, :left_band], channels))
    if not samples:
        return None

    flat = torch.cat(samples, dim=0).float()
    if flat.shape[0] < DC_MATCH_MIN_SAMPLES:
        return None
    # The median is robust to a high-contrast edge crossing the band, where the two
    # refinements genuinely disagree by a lot over a few pixels — that is texture
    # disagreement, not a DC offset, and a mean lets it drag the whole tile.
    offset = flat.median(dim=0).values
    return offset.clamp(-DC_MATCH_CLAMP, DC_MATCH_CLAMP).to(sub.dtype)


def _debug_dir():
    # Opt-in seam debug. When the CATR_DEBUG_DIR env var is set, _refine_tiles writes the
    # per-tile intermediates (decoded crop, the raw band it encoded, the feather alpha, and
    # the canvas region before/after the composite) plus a manifest there as PNGs, so the
    # seam can be inspected tile by tile. Unset (the default) → a single env lookup and no
    # other work, so the byte-for-byte output path is untouched. Read lazily (not at module
    # scope) to keep sampling.py's module scope torch-only.
    import os
    return os.environ.get("CATR_DEBUG_DIR") or None


def _dump_png(path, tensor):
    # Save a debug tensor as PNG, batch 0 only. [B,H,W,C]/[H,W,C] → RGB (first 3 channels);
    # [H,W] (a feather alpha or mask) → grayscale. Values are clamped to [0,1] and quantized
    # for viewing only — this never feeds back into the pipeline. PIL is imported lazily so
    # the module scope stays torch-only (a subprocess test pins that).
    from PIL import Image

    t = tensor.detach().to("cpu", torch.float32)
    if t.ndim == 4:
        t = t[0]
    if t.ndim == 3:
        arr = (t[..., :3].clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).contiguous().numpy()
        Image.fromarray(arr, mode="RGB").save(path)
    else:
        arr = (t.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).contiguous().numpy()
        Image.fromarray(arr, mode="L").save(path)


def _manifest_line(tile_idx, tile, dc_offset=None):
    # One human-readable row per tile for the seam-debug manifest: the geometry needed to
    # line the dumped PNGs up in canvas space (core is the hard-owned region; paste is the
    # feathered top/left band; crop is the full sampled extent incl. the frozen anchor ring).
    # dc_offset is the applied per-channel DC correction — the number the DC match exists to
    # produce, and it is invisible in the dumped PNGs (which hold the UNcorrected decode), so
    # it has to be recorded here or it cannot be checked at all.
    def rc(r):
        return f"({r.x0},{r.y0})-({r.x1},{r.y1})"

    dc = "" if dc_offset is None else " dc=[{}]".format(
        ",".join(f"{v:+.6f}" for v in dc_offset.tolist()))
    return f"t{tile_idx:02d} r{tile.row}c{tile.col} cls={tile.cls} kept_top={tile.kept_top} kept_left={tile.kept_left} core={rc(tile.core)} paste={rc(tile.paste_rect)} crop={rc(tile.crop_rect)}{dc}"


def encode_pixels(vae, pixels):
    # The tile encode, batch-safe. A video-family VAE (latent_dim 3 — the Wan family, used
    # by e.g. Krea 2) reshapes a 4-D image batch to [1,C,B,H,W] (comfy/sd.py VAE.encode):
    # the batch becomes FRAMES of one clip and temporal compression merges them into a
    # single latent frame, cross-contaminating the images. Encoding one row at a time keeps
    # the [1,C,1,h,w] layout every single-image run already produces, concatenated back to
    # the [B,C,1,h,w] the noise dummy mirrors. Any B=1 run, and every 4-D VAE, take the
    # single unchanged call — byte-for-byte the pre-helper path.
    if getattr(vae, "latent_dim", 2) == 3 and pixels.shape[0] > 1:
        return torch.cat([vae.encode(pixels[b:b + 1]) for b in range(pixels.shape[0])], dim=0)
    return vae.encode(pixels)


def _refine_tiles(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, region_pixel=None, *, hint_canvas=None, batch_size=1, batch_index=0):
    # Tiled img2img: pad to /8 → solve the grid → encode/sample/decode each tile in
    # raster order from a live canvas → DC-match onto the already-placed neighbors →
    # composite by a directional feather → crop back.
    # The BASE node's path only: since 1.6.0 the VL nodes tile through the sync engine
    # (sync.py), so nothing VL — no CLIP, no per-tile positive, no conditioning-surface
    # select — reaches this function at all.
    # A 1x1 grid (or overlap=ctx=0) reproduces the feature-2 whole-image path bit for bit.
    # region_pixel (optional [B,H,W] binary float at input pixel size) gates each tile's
    # denoise_mask to the masked region so only it diffuses; the background stays frozen,
    # and it gates the DC measurement to the same region (see seam_dc_offset).
    # hint_canvas (optional) is refine_image's prepared ControlNet hint map, already in THIS
    # call's pixel space; None — the no-ControlNet case — leaves every step below untouched.
    # batch_size/batch_index name this call's picture inside refine_image's picture loop
    # (batch == 1 there). They reach the noise draw and the progress bar only; 1/0 — a
    # direct call, or a single-picture IMAGE — changes nothing.
    import comfy.model_management

    # Opt-in seam debug (CATR_DEBUG_DIR). Every _save call site sits inside its own
    # `if debug_dir is not None:` guard, which is what keeps _save undefined and unreachable
    # when the variable is unset. Nothing below changes the output, only side-effect PNGs and
    # a manifest are written.
    debug_dir = _debug_dir()
    debug_manifest = []
    if debug_dir is not None:
        import os
        os.makedirs(debug_dir, exist_ok=True)

        def _save(name, tensor):
            _dump_png(os.path.join(debug_dir, name), tensor)

    pixels = image[..., :3]
    padded, (height, width) = pad_image_to_multiple(pixels)
    batch, canvas_h, canvas_w = padded.shape[0], padded.shape[1], padded.shape[2]

    # Ride the hints onto the same padded canvas as the pixels — same reflect pad, same
    # amounts — so a tile's crop_rect indexes hint and canvas identically.
    if hint_canvas is not None:
        hint_canvas = conds.pad_hint_canvas(hint_canvas, (height, width), (canvas_h, canvas_w))

    # Region gate at latent resolution. Pad the pixel mask to the /8 canvas with constant
    # 0 (the padded strip is always cropped away), then cover-downsample by 8. max_pool2d
    # (NOT avg+threshold) so every pixel with region==1 lands in a DIFFUSED latent cell —
    # else a subject-edge pixel would be frozen yet pasted. Over-cover is harmless: extra
    # diffused cells fall outside the mask and are discarded by the AA composite.
    # region_padded (the PIXEL-resolution gate) is kept as well, because the DC measurement
    # has to be gated at pixel resolution: region_latent over-covers by up to 8px, and those
    # extra pixels are exactly the frozen ones that must not enter the estimate.
    region_padded = None
    region_latent = None
    if region_pixel is not None:
        region_padded = torch.nn.functional.pad(region_pixel, (0, canvas_w - width, 0, canvas_h - height))
        region_latent = (torch.nn.functional.max_pool2d(region_padded[:, None].float(), 8) > 0).float()[:, 0]

    sx = grid.solve_axis(canvas_w, max_tile_width, context_anchor, context_overlap, axis="width")
    sy = grid.solve_axis(canvas_h, max_tile_height, context_anchor, context_overlap, axis="height")
    layout = grid.build_layout(canvas_w, canvas_h, sx, sy, context_anchor, context_overlap)

    # One noise draw for the whole canvas, then sliced per tile: per-tile draws would
    # give every same-shaped tile identical noise, and slices are spatially anchored
    # so overlapping crop regions of fade-expanded tiles agree on noise. prepare_noise
    # reads only size/dtype/layout, so the zeros dummy (encode outputs float32) keeps
    # a 1x1 grid bit-identical to the whole-image {"samples": latent} draw.
    # The dummy mirrors vae.encode's latent layout exactly: a video-family VAE
    # (latent_dim 3 — Wan/Qwen-Image, used by e.g. Krea 2) encodes an image batch to a
    # 5-D [B,C,1,h,w] latent, and a 4-D draw against that 5-D tile latent would
    # broadcast the sampler's sigma*noise + latent mix into a fake C-frame temporal
    # axis, which temporal models then fold into the batch (a batch-mismatch crash).
    latent_time = (1,) if getattr(vae, "latent_dim", 2) == 3 else ()
    # Rows the draw covers. Under refine_image's picture loop this call holds ONE picture of a
    # batch_size-picture run, and the dummy is still built at the FULL batch so picture b gets
    # the row it would have drawn in one shared call: prepare_noise draws a SINGLE torch.randn
    # over the whole latent size, so a [1,...] dummy would hand every picture the same noise.
    # batch_size == 1 (no loop, or a direct _refine_tiles call) draws at `batch` and takes no
    # slice — byte-for-byte the pre-loop path.
    draw_batch = batch_size if batch_size > 1 else batch
    dummy = torch.zeros((draw_batch, vae.latent_channels, *latent_time, canvas_h // 8, canvas_w // 8), dtype=torch.float32)
    canvas_noise = noise.generate_noise({"samples": dummy})
    if batch_size > 1:
        canvas_noise = canvas_noise[batch_index:batch_index + 1]

    steps = sigmas.shape[-1] - 1
    for_tile = make_tile_progress(guider.model_patcher, steps, len(layout.tiles), batch_size, batch_index)
    # Live canvas: tiles paste into it in raster order, so a later tile's frozen
    # context_anchor ring encodes its already-refined neighbors (seam conditioning).
    # The clone is mandatory — pad_image_to_multiple returns the caller's tensor (a
    # view) when already /8-aligned, so pasting in place would mutate the node input.
    canvas = padded.clone()
    # Frozen raw source, read-only (padded is never written — no clone needed). The
    # DIFFUSED core + context_overlap band of every tile encodes from here rather than
    # the live canvas, so the directional feather cross-dissolves two INDEPENDENT
    # refinements of the same raw band, never a serial re-diffusion of a neighbor's
    # already-decoded output (see the encode site for the full argument).
    source = padded
    for tile_idx, tile in enumerate(layout.tiles):
        comfy.model_management.throw_exception_if_processing_interrupted()
        crop = tile.crop_rect
        core = tile.core
        expanded = (crop.x0, crop.y0, crop.x1, crop.y1) != (core.x0, core.y0, core.x1, core.y1)
        if expanded:
            # Split the encode source by band. The DIFFUSED region — core +
            # context_overlap == tile.overlap_inner_rect, where the binary denoise
            # mask is 1 — encodes from the FROZEN RAW source, not the live canvas: on a
            # side bordering an already-processed neighbor (top/left in raster order)
            # the live canvas already holds that neighbor's decoded output there, so
            # re-diffusing it would run a SECOND diffusion + VAE round-trip in series,
            # and the directional feather would then cross-dissolve that serial
            # re-diffusion against the neighbor — compounding artifacts on the seam.
            # Taking the diffused band from raw makes the feather cross-dissolve two
            # INDEPENDENT refinements of the same raw content instead. The FROZEN
            # context_anchor ring (crop \ overlap_inner_rect, denoise mask 0) stays on
            # the LIVE canvas — it carries the neighbor's already-refined pixels as
            # conditioning, which is the anchor's whole purpose. At context_overlap=0,
            # overlap_inner_rect == core and canvas[core] is still raw (no tile pastes
            # into a later tile's core), so the source overwrite is a byte-for-byte
            # no-op → enc equals canvas[crop] → bit-identical to the pre-split encode.
            inner = tile.overlap_inner_rect
            enc = canvas[:, crop.y0:crop.y1, crop.x0:crop.x1, :].clone()
            iy0, iy1 = inner.y0 - crop.y0, inner.y1 - crop.y0
            ix0, ix1 = inner.x0 - crop.x0, inner.x1 - crop.x0
            enc[:, iy0:iy1, ix0:ix1, :] = source[:, inner.y0:inner.y1, inner.x0:inner.x1, :]
            if debug_dir is not None:
                _save(f"t{tile_idx:02d}_r{tile.row}c{tile.col}_enc.png", enc)
            tile_latent = encode_pixels(vae, enc)
        else:
            # Whole-image / n=1 path: crop == core, no anchor ring and no already-
            # processed neighbor, so encode the live crop directly (feature-2, byte-
            # for-byte). Kept exactly as before to preserve the whole-image path.
            tile_latent = encode_pixels(vae, canvas[:, crop.y0:crop.y1, crop.x0:crop.x1, :])
        # .contiguous() because downstream sampler code may .view() the slice; a 1x1
        # grid slices the full tensor, so it returns self unchanged. The ellipsis
        # slices the trailing spatial dims of both the 4-D and the 5-D noise layout.
        tile_noise = canvas_noise[..., crop.y0 // 8:crop.y1 // 8, crop.x0 // 8:crop.x1 // 8].contiguous()
        # Fail fast on any encode/noise layout divergence encode_pixels does not already
        # resolve — a VAE folding an image batch outside the video family it handles, or a
        # spatial compression the /8 grid math does not model. A mismatch here would
        # otherwise broadcast into a cryptic shape error deep inside the sampler or model.
        if tile_latent.shape != tile_noise.shape:
            raise RuntimeError(
                f"VAE encoded a {crop.x1 - crop.x0}x{crop.y1 - crop.y0} px tile to latent shape {tuple(tile_latent.shape)}, but the tile's noise slice is {tuple(tile_noise.shape)}. "
                "This VAE's latent layout is not supported (e.g. a video VAE folding an image batch "
                "into frames, or a non-8x spatial compression).")
        # Binary latent denoise_mask: full-strength denoise over core+overlap
        # (== tile.overlap_inner_rect), frozen only through the context_anchor ring.
        # Passing overlap_inner_rect as the "core" arg makes tile_gradient return its
        # hard indicator (1 inside, 0 outside) — deliberately NOT a fractional ramp: the
        # static inpaint blend re-applies this mask every step, so a fractional cell would
        # only partly denoise (an under-refined seam halo). At context_overlap=0,
        # overlap_inner_rect == core, bit-identical to the prior binary core indicator.
        # With a region mask the gradient is intersected with the tile's region_latent
        # slice ([B,th,tw], still binary) so only the masked region diffuses and the
        # surrounding background stays frozen context; the n=1 (unexpanded) masked case
        # yields denoise_mask = region_latent (NOT None), anchoring the in-crop background.
        if region_pixel is None:
            denoise_mask = tile_gradient(crop, tile.overlap_inner_rect, scale=8) if expanded else None
        else:
            reg = region_latent[:, crop.y0 // 8:crop.y1 // 8, crop.x0 // 8:crop.x1 // 8]
            # .to(reg.device): tile_gradient builds its indicator from torch.arange, so it is
            # always CPU, while region_latent follows the MASK/IMAGE device (CUDA under
            # --gpu-only). The multiply below is elementwise, so it would raise across devices
            # on every multi-tile masked refine. A no-op when both are already CPU.
            grad = tile_gradient(crop, tile.overlap_inner_rect, scale=8).to(reg.device) if expanded else 1.0
            denoise_mask = grad * reg
        # Per-tile ControlNet swap: the swapped map's control chains are fresh copies whose
        # hints are this tile's crop_rect slice, so core's rescale is an identity and the hint
        # lands aligned.
        # CFGGuider.sample() re-copies original_conds on every call, so the swap takes effect
        # per tile. The pristine map — the SAME object the caller handed in — goes back in
        # `finally`, so an interrupt or a sampler raise cannot leave the guider swapped.
        pristine_conds = None
        if hint_canvas is not None:
            pristine_conds = guider.original_conds
            guider.original_conds = conds.crop_tile_conds(pristine_conds, hint_canvas, crop)
        try:
            # anchor_ring=False, always: the lead schedule is the VL path's, and the VL path
            # IS the sync engine since 1.6.0 (it enters anchor_ring_schedule itself, once for
            # the whole run). On this node's plain conditioning the schedule was consistently
            # worse in the owner's visual A/B (2026-08-13, tests-AB/run_ab_matrix.py, scene
            # `portrait`): a distorted ear, duller colour and texture at the seam. The
            # mechanism fits — the ring resolves ahead of the core so the core follows it, and
            # only a VL positive tells a tile what its neighbourhood contains, so without one
            # the tile is pulled toward a ring it cannot interpret.
            samples = sample_latent(guider, sampler, sigmas, tile_noise, noise.seed, tile_latent, denoise_mask=denoise_mask, callback=for_tile(tile_idx), anchor_ring=False)
        finally:
            if pristine_conds is not None:
                guider.original_conds = pristine_conds
        decoded = vae.decode(samples)
        # A video-family VAE decodes the 5-D latent to [B,T,H,W,C]; fold T into the
        # batch exactly as core's VAEDecode node does. T is 1 by construction here
        # (the fail-fast noise/latent check above pins one latent row per image), so
        # this is a pure view and the composite below sees the same [B,H,W,C] contract
        # as a 2-D VAE's output.
        if decoded.ndim == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        if expanded:
            # Directional feather (docs/CLAUDE.md prime directive 2): write paste_rect
            # — the core plus the context_overlap band ONLY on sides bordering an
            # already-processed neighbor (top/left in raster order) — cross-dissolving
            # this tile into that neighbor. feather_alpha is 1 through the core (so the
            # seam pixel is 100% this tile) and ramps to ~0 across the top/left overlap
            # bands, so the neighbor fades back in over the band; each interior seam is
            # feathered exactly once, by the later tile. Right/bottom overlap and the
            # whole context_anchor ring were sampled for context only and are cropped
            # away (paste_rect excludes them), so no strip is hard-processed twice.
            # This is a thin cross-dissolve of two conditioned, mutually-coherent edges
            # (this tile encoded the neighbor's finished pixels from the live canvas),
            # never a wide/divergent blend. With no kept side (top-left tile, or
            # context_overlap=0) paste_rect == core and alpha is all-ones, so this
            # reduces to the fe6f6a7 hard core paste, byte-identical.
            # .to() aligns the VAE output to the canvas device (under --gpu-only decode
            # lands on the GPU intermediate device while the cloned canvas stays on the
            # input device); a no-op when they already match (CPU tests bit-identical).
            decoded = decoded.to(canvas.device)
            paste = tile.paste_rect
            sub = decoded[:, paste.y0 - crop.y0:paste.y1 - crop.y0, paste.x0 - crop.x0:paste.x1 - crop.x0, :]
            region = canvas[:, paste.y0:paste.y1, paste.x0:paste.x1, :]
            # DC match, BEFORE the displacement and the composite. Before the composite
            # because a per-tile constant applied afterwards would re-introduce a hard step
            # at the tile boundary and undo the feather; doing it here means feather_alpha
            # cross-dissolves two already-matched tiles and needs no new machinery. Before
            # the displacement because removing a constant offset leaves the error surface
            # describing TEXTURE disagreement rather than being dominated by that constant,
            # which is what the routed cut should follow.
            region_mask = None if region_padded is None else region_padded[:, paste.y0:paste.y1, paste.x0:paste.x1]
            dc_offset = seam_dc_offset(tile, sub, region, region_mask)
            if dc_offset is not None:
                # `sub - dc_offset` is a copy; `decoded` and the canvas are untouched. The
                # correction must never push a pixel OUT of the displayable gamut: standard
                # VAEs clamp their decode to [0,1] in process_output, and subtracting the
                # offset from a pixel already at a rail (0.0 minus a positive offset) would
                # leak past that contract into the node's IMAGE output. The clamp is gated on
                # the pixel having been in gamut BEFORE correction, because clamping is not
                # free: a VAE whose process_output is the identity (core has several) emits
                # out-of-range values legitimately, and flattening those would be a lossy op
                # on the tile. Doing this before seam_displacements keeps the error surface
                # describing the values the composite actually writes.
                corrected = sub - dc_offset
                in_gamut = (sub >= 0.0) & (sub <= 1.0)
                sub = torch.where(in_gamut, corrected.clamp(0.0, 1.0), corrected)
            disp_top, disp_left = seam_displacements(tile, sub, region)
            alpha = feather_alpha(paste, core, layout.overlap, tile.kept_top, tile.kept_left, disp_top=disp_top, disp_left=disp_left).to(canvas.device)[..., None]
            if debug_dir is not None:
                # region is a VIEW into canvas, so dump it BEFORE the composite overwrites it.
                tag = f"t{tile_idx:02d}_r{tile.row}c{tile.col}"
                _save(tag + "_decoded.png", decoded)
                _save(tag + "_alpha.png", alpha[..., 0])
                _save(tag + "_region_before.png", region)
            canvas[:, paste.y0:paste.y1, paste.x0:paste.x1, :] = alpha * sub + (1.0 - alpha) * region
            if debug_dir is not None:
                _save(tag + "_region_after.png", canvas[:, paste.y0:paste.y1, paste.x0:paste.x1, :])
                debug_manifest.append(_manifest_line(tile_idx, tile, dc_offset))
        else:
            if debug_dir is not None:
                _save(f"t{tile_idx:02d}_r{tile.row}c{tile.col}_decoded.png", decoded)
                debug_manifest.append(_manifest_line(tile_idx, tile))
            canvas[:, crop.y0:crop.y1, crop.x0:crop.x1, :] = decoded
    if debug_dir is not None:
        _save("canvas_final.png", crop_image_to(canvas, height, width))
        with open(os.path.join(debug_dir, "manifest.txt"), "w", encoding="utf-8") as manifest_file:
            manifest_file.write("\n".join(debug_manifest) + "\n")
    return crop_image_to(canvas, height, width)


def _mask_row(mask, batch_index):
    # One picture's row of the mask, for the picture loop below. A single-row mask applies to
    # every picture — node.py already expands one to the image batch, but refine_image is also
    # called directly (the A/B harnesses) and must broadcast the same way — which is what the
    # clamp keeps: row batch_index of a B-row mask, row 0 of a single-row one.
    if mask is None:
        return None
    row = min(batch_index, mask.shape[0] - 1)
    return mask[row:row + 1]


def _check_sync_intake(sampler, sigmas, sampler_name=None):
    # The VL dispatch's fail-fast, run BEFORE any VAE encode or VL encode — both of which
    # cost minutes and GPU memory the user would spend only to hit the same rejection deep
    # inside the engine. Two things the sync engine cannot work around:
    #   * the sampler must be one the stepper can time (its EVALS_PER_STEP table). When the
    #     caller knows the NAME (the all-in-one node's widget), it is checked first: core's
    #     sampler_object wraps several names in a private function (dpm_fast ->
    #     dpm_fast_function), so the object alone would name something no widget offers.
    #   * every lane steps ONE shared schedule, so the schedule must be strictly decreasing
    #     and end at 0 — true of simple/normal/beta/sgm_uniform, so this needs no
    #     per-scheduler knowledge. The engine's own preconditions re-check it.
    from . import stepper

    if sampler_name is not None and sampler_name not in stepper.EVALS_PER_STEP:
        raise ValueError(
            f"Context-Anchored Tile Refine (VL): sampler '{sampler_name}' is not supported by "
            f"the synchronized tile engine. Supported: {', '.join(stepper.SUPPORTED_SAMPLERS)}. "
            f"{', '.join(stepper.UNSUPPORTED_BY_DESIGN)} are unsupported BY DESIGN. They own "
            "their own schedule or internal history, so no evals-per-step entry can time them.")
    stepper._resolve_sampler_name(sampler)
    if not bool(torch.all(sigmas[1:] < sigmas[:-1])) or float(sigmas[-1]) != 0.0:
        raise ValueError(
            "Context-Anchored Tile Refine (VL): every tile of a synchronized run steps ONE "
            "shared schedule, so sigmas must be strictly decreasing and end at 0.0. Got a "
            f"schedule of {int(sigmas.numel())} sigmas from {float(sigmas.reshape(-1)[0])} to "
            f"{float(sigmas.reshape(-1)[-1])}.")


def refine_image(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=None, vl_clip=None, vlm_method=captions.VLM_METHOD_VISION, anchor_source=None, sampler_name=None, batch_size=1, batch_index=0, progress=None):
    # Entry point, and the ONE place all three nodes meet. With a vl_clip the whole refine is
    # handed to the sync engine (sync.py) — mask or no mask — and everything below it is the
    # BASE node's raster path.
    # Without a mask this is the whole-image / multi-tile refine, byte-for-
    # byte the extracted _refine_tiles body. With a mask: harden (>=0.5) → bbox of THIS
    # picture's mask row → crop to bbox + context_anchor (frozen-background halo) → tile-refine the
    # crop with every tile's denoise_mask gated to the masked region → composite the
    # refined crop back through a 1px anti-aliased edge. The mask=0 background survives from
    # the ORIGINAL pixels (never the VAE-decoded crop), so it is never re-diffused.
    # The minimum-error routing and the per-tile DC match shape the inter-TILE directional
    # feather only; neither touches the mask boundary, which stays a hard binary gate plus a
    # 1px anti-alias (mask-boundary-conditioning-not-feather). The DC measurement is gated to
    # the masked region for the same reason — it is a tile-to-tile correction, and the
    # background outside the region is never shaded at all.
    # anchor_source is the VL nodes' select, resolved against sync.py's own default when a
    # caller leaves it None; sampler_name is the all-in-one node's widget, carried only so a
    # rejected sampler is named as the user picked it. Both are inert without vl_clip.
    # batch_size/batch_index name this call's picture inside the picture loop below; a caller
    # always starts at the 1/0 defaults.
    # progress is the VL run's progress ledger (progress.py), created by node.py and only ever
    # PASSED THROUGH here — including across the picture loop, so a batch shares ONE bar. It is
    # inert without vl_clip: the base raster path keeps its own per-tile bar untouched.
    if sigmas.numel() < 2:
        # Zero steps: nothing to sample, and the lossy VAE roundtrip would degrade the image.
        # Narrowed to RGB like every sampled path, so the output channel count never depends
        # on a widget value; on a 3-channel input the slice is the whole tensor. Batch-safe
        # already, so it sits ABOVE the picture loop and stays one pure clone.
        return image[..., :3].clone()

    # PICTURE loop: a batched IMAGE is refined ONE PICTURE AT A TIME, then concatenated back.
    # Everything below is a per-picture quantity that a shared call silently pools across
    # unrelated pictures: seam_dc_offset's median over [B,h,w,C] band pixels yields one offset
    # correct for none of them, seam_displacements' mean over dim 0 routes one cut around
    # averaged content, and _mask_bbox unions every row's region into one oversized crop. Peak
    # VRAM stops scaling with the batch as well (a tile latent is [1,C,h,w], not [B,C,h,w]).
    # Recursion, so there is exactly ONE implementation of everything downstream; a
    # single-picture IMAGE never enters the loop, so its path is untouched.
    if image.shape[0] > 1:
        pictures = [
            refine_image(image[b:b + 1], guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=_mask_row(mask, b), vl_clip=vl_clip, vlm_method=vlm_method, anchor_source=anchor_source, sampler_name=sampler_name, batch_size=image.shape[0], batch_index=b, progress=progress)
            for b in range(image.shape[0])
        ]
        return torch.cat(pictures, dim=0)

    # ControlNet prep happens HERE because this is the last place the FULL input size is in
    # hand: the hints are validated against it, and on the mask path they take the same bbox
    # slice the image does. _refine_tiles pads them and cuts them per tile. Without a control
    # object in the conds this is a single dict walk and hint_canvas stays None, leaving every
    # path below byte-identical to a run without the feature.
    original_conds = getattr(guider, "original_conds", None)
    control_active = conds.has_spatial_conds(original_conds)
    if vl_clip is not None:
        # The per-tile positive swap needs the CFGGuider conds convention; anything else
        # would silently sample with the swap never taking effect.
        if not isinstance(original_conds, dict) or "positive" not in original_conds:
            raise ValueError(
                "VL refine needs a guider that keeps 'positive' in original_conds "
                f"(the CFGGuider convention). Got {sorted(original_conds) if isinstance(original_conds, dict) else type(original_conds).__name__}")
        # ControlNet is a BASE-node feature only. Each tile's positive is replaced outright
        # by a fresh vision-slice cond that carries no control chain, so a cropped hint would
        # reach the negative branch alone — asymmetric CFG, with the hint steering only the
        # unconditional side — and every per-tile positive control copy would be built and
        # thrown away. Drop the whole hint pipeline here and say so once; the sync engine
        # then strips the `control` key itself (sync.build_lane_guiders → conds.strip_control),
        # so the hints are never validated NOR consulted.
        if control_active:
            logging.warning(
                "Context-Anchored Tile Refine (VL): ControlNet conditioning is ignored by the "
                "VL nodes, because every tile's positive is replaced by its vision slice. Use "
                "the base Context-Anchored Tile Refine node for control.")
        # THE VL PATH IS THE SYNC ENGINE, for the masked refine as much as the whole-image
        # one: refine_sync owns the bbox crop, the region gate and the anti-aliased composite
        # itself, and it is what builds the (full image, bbox x0, bbox y0) triple the
        # conditioning pre-pass reads — so a masked VL refine still encodes the WHOLE image
        # and offsets each region tile into its frame, exactly as the raster path did.
        # LAZY import, and the one direction that can be: sync.py imports this module at its
        # own module scope, so the pair is acyclic only while this stays inside the function.
        from . import sync

        _check_sync_intake(sampler, sigmas, sampler_name)
        return sync.refine_sync(
            image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height,
            context_anchor, context_overlap, vl_clip, mask=mask, vlm_method=vlm_method,
            anchor_source=sync.ANCHOR_SOURCE_IMAGE if anchor_source is None else anchor_source,
            batch_size=batch_size, batch_index=batch_index, progress=progress)
    if mask is None:
        hint_canvas = conds.prepare_hint_canvas(original_conds, (image.shape[1], image.shape[2])) if control_active else None
        # Picture b must be conditioned on hint row b — see conds.slice_hint_row. Skipped
        # entirely for a single-picture run, which keeps the caller's own hint tensors.
        if batch_size > 1 and hint_canvas is not None:
            hint_canvas = conds.slice_hint_row(hint_canvas, batch_index)
        return _refine_tiles(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, region_pixel=None, hint_canvas=hint_canvas, batch_size=batch_size, batch_index=batch_index)

    # Region-mask path. Harden a soft input at 0.5 — a fractional denoise mask would leave
    # the under-refined halo we reject for turbo (finding-dd-fade-artifacts-turbo).
    mask_bin = mask >= 0.5
    bbox = _mask_bbox(mask_bin)
    if bbox is None:
        # Empty mask: nothing to refine, and (unlike the input) a clone is a safe no-op.
        # RGB-narrowed for the same reason as the zero-step return above.
        return image[..., :3].clone()
    height, width = image.shape[1], image.shape[2]
    y0, y1, x0, x1 = _expand_snap_clamp(bbox, context_anchor, height, width)
    sub_image = image[:, y0:y1, x0:x1, :]
    sub_mask = mask_bin[:, y0:y1, x0:x1].to(image.dtype)
    hint_canvas = conds.prepare_hint_canvas(original_conds, (height, width), (y0, y1, x0, x1)) if control_active else None
    if batch_size > 1 and hint_canvas is not None:
        hint_canvas = conds.slice_hint_row(hint_canvas, batch_index)
    refined_sub = _refine_tiles(sub_image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, region_pixel=sub_mask, hint_canvas=hint_canvas, batch_size=batch_size, batch_index=batch_index)
    # Composite the refined crop back through the anti-aliased mask edge. Outside the crop,
    # and wherever aa == 0 inside it, the output is the byte-identical original image.
    # Narrow to RGB: _refine_tiles decodes a 3-channel refined_sub (as the no-mask path does,
    # dropping a 4-channel input's alpha), so the composite and the output stay 3-channel.
    rgb = image[..., :3]
    out = rgb.clone()
    aa = _aa_alpha(sub_mask)[..., None]
    out[:, y0:y1, x0:x1, :] = aa * refined_sub + (1.0 - aa) * rgb[:, y0:y1, x0:x1, :]
    return out
