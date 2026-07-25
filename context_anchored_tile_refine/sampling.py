import math

import torch

from . import grid

# seam_mode values (how the feather's transition is positioned along a seam).
SEAM_STRAIGHT = "straight"     # no displacement — the transition is a straight line
SEAM_WARP = "warp"             # displaced by low-frequency seeded coherent noise
SEAM_MIN_ERROR = "min_error"   # displaced toward the Efros-Freeman minimum-error path
# A list, NOT a tuple: ComfyUI only range-checks a COMBO when the schema entry is a list
# (execution.py gates the membership test on isinstance(input_type, list)), and the datatype
# reference specifies COMBO as list[str]. As a tuple an API-submitted typo would pass
# validation, match no branch in _seam_displacements, and silently run as `straight`.
SEAM_MODES = [SEAM_STRAIGHT, SEAM_WARP, SEAM_MIN_ERROR]

# How WIDE the handover is, once seam_mode has POSITIONED it, is not configurable: the
# full-band feather below is the only method. Two alternatives were built and A/B'd against
# it on a night sky and a face (a narrow ramp landing exactly on the routed cut, and a binary
# cut with no blending at all). All three measured the same tile-level DC step to within
# 0.05/255, because that step spans the whole tile and no choice of handover position inside
# a 32px band can move it. Visually the binary cut lost outright, tearing at a high-contrast
# silhouette edge, so the feather stays and the axis was removed rather than left as dead
# configuration.

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


def tile_gradient(crop, core, fade, scale=1):
    # Binary denoise-mask indicator: 1 for
    # every cell whose center lies inside the `core` rect, 0 outside. Cell centers
    # (k+0.5) so one formula works at any grid resolution via `scale`; the pipeline uses
    # scale=8 and passes overlap_inner_rect as `core` (core+overlap diffused at full
    # strength, context_anchor ring frozen). `fade` is retained for call-site stability
    # and must be 0: the fractional 1->0 ramp was removed with the pixel-space blend and
    # the directional-feather rewrite. The mask stays binary because ComfyUI's static
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
    # `warp` and `min_error` seam modes) so it stops reading as a straight line; None keeps
    # the straight 1-D ramp, bit-for-bit. A displaced side becomes a 2-D field, but the
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



def _warp_field(canvas_h, canvas_w, scale, seed):
    # Low-resolution seeded grid, one cell per `scale` canvas px. Sampling it bicubically
    # (see _warp_slice) IS coherent value noise with a feature size of ~`scale` px — the
    # cheap, grain-free way to get a smooth field. Generating it small is the point: white
    # noise at pixel scale would read as grit, and the whole reason this hides a seam
    # instead of adding texture is that the field varies far more slowly than a pixel.
    # Kept at low res (a few KB) rather than materialized canvas-sized.
    grid_h = max(2, math.ceil(canvas_h / scale) + 1)
    grid_w = max(2, math.ceil(canvas_w / scale) + 1)
    generator = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)
    low = torch.randn(1, 1, grid_h, grid_w, generator=generator, dtype=torch.float32)
    return low / (low.std() + 1e-6)


def _warp_slice(low, scale, ys, xs):
    # Sample the warp field at ABSOLUTE canvas coords (ys, xs, both [n] float32) -> [n] in
    # [-1,1]. Absolute coords are what make the warp canvas-anchored rather than tile-local:
    # two tiles that meet along a shared seam sample the same coordinates and therefore agree
    # on the displacement, so the meander runs continuously across the whole canvas instead
    # of resetting at every tile boundary. Same draw-once/slice-per-tile discipline as
    # canvas_noise. Bicubic keeps the field smooth (bilinear would kink at cell edges).
    grid_h, grid_w = low.shape[2], low.shape[3]
    # Cell j spans canvas [j*scale, (j+1)*scale); grid_sample(align_corners=False) maps
    # normalized g to index (g+1)/2*size - 0.5, so index c/scale - 0.5 <=> g = 2c/(scale*size) - 1.
    gx = xs / scale * 2.0 / grid_w - 1.0
    gy = ys / scale * 2.0 / grid_h - 1.0
    sample_grid = torch.stack([gx, gy], dim=-1)[None, None]  # [1,1,n,2], last dim is (x,y)
    out = torch.nn.functional.grid_sample(low, sample_grid, mode="bicubic", align_corners=False, padding_mode="border")
    return out[0, 0, 0].clamp(-1.0, 1.0)


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
    # step the feather exists to hide — and the seam_mode A/B was judged on this behavior.
    # Making it land exactly is a real visual change, not a bug fix: re-run the A/B first.
    path = _min_error_path(err)
    u50 = (0.5 ** (1.0 / k)) * (1.0 - plateau)
    u_path = (path.to(torch.float32) + 0.5) / band
    return u50 - u_path


def sample_latent(guider, sampler, sigmas, noise_tensor, seed, latent_samples, denoise_mask=None, callback=None):
    # The per-tile reuse seam. Mirrors SamplerCustomAdvanced.sample with two deliberate
    # deviations: no fix_empty_latent_channels (the latent always comes from a real
    # vae.encode) and no x0_output dict (the denoised output is unused; previews work
    # without it). denoise_mask/callback are injectable for the later tiling feature.
    import comfy.model_management
    import comfy.utils
    import latent_preview

    if callback is None:
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    samples = guider.sample(noise_tensor, latent_samples, sampler, sigmas, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
    return samples.to(comfy.model_management.intermediate_device())


def make_tile_progress(model_patcher, steps, n_tiles):
    # latent_preview.prepare_callback replicated at aggregate scale: one ProgressBar
    # spanning steps * n_tiles so the UI shows a single run across every tile.
    import comfy.utils
    import latent_preview

    previewer = latent_preview.get_previewer(model_patcher.load_device, model_patcher.model.latent_format)
    total = steps * n_tiles
    pbar = comfy.utils.ProgressBar(total)

    def for_tile(tile_idx):
        def callback(step, x0, x, total_steps):
            preview = None
            if previewer:
                preview = previewer.decode_latent_to_preview_image("JPEG", x0)
            pbar.update_absolute(tile_idx * steps + step + 1, total, preview)
        return callback

    return for_tile


def _mask_bbox(mask_bin):
    # Union bounding box over the whole batch of a [B,H,W] boolean mask, as
    # (y0, y1, x0, x1) with EXCLUSIVE y1/x1 (max + 1); None when the mask is empty.
    # torch.nonzero on the batch-union keeps one shared crop for every row.
    any_mask = mask_bin.any(dim=0)
    nz = torch.nonzero(any_mask)
    if nz.numel() == 0:
        return None
    y0 = int(nz[:, 0].min())
    y1 = int(nz[:, 0].max()) + 1
    x0 = int(nz[:, 1].min())
    x1 = int(nz[:, 1].max()) + 1
    return (y0, y1, x0, x1)


def _expand_snap_clamp(bbox, anchor, H, W):
    # Expand the bbox by `anchor` (the frozen-background halo the subject conditions
    # against), snap x0/y0 DOWN and x1/y1 UP to /8, then clamp to [0,H]/[0,W]. Snapping
    # before clamping keeps interior edges on the /8 grid; a mask touching a non-/8 image
    # border yields a clamped, non-/8 crop that _refine_tiles' internal pad then handles.
    y0, y1, x0, x1 = bbox
    y0 = ((y0 - anchor) // 8) * 8
    x0 = ((x0 - anchor) // 8) * 8
    y1 = ((y1 + anchor + 7) // 8) * 8
    x1 = ((x1 + anchor + 7) // 8) * 8
    y0 = max(0, y0)
    x0 = max(0, x0)
    y1 = min(H, y1)
    x1 = min(W, x1)
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


def _seam_displacements(seam_mode, tile, sub, region, warp_low, warp_scale, warp_amount):
    # Per-tile displacement for each kept seam, in band-normalized units: [w] for the top
    # seam, [h] for the left seam (None = leave that seam straight). This is the ONLY thing
    # that differs between seam modes — every mode then feeds the same feather curve — so an
    # A/B comparison isolates the boundary shape and nothing else.
    paste, core = tile.paste_rect, tile.core
    top_band = core.y0 - paste.y0
    left_band = core.x0 - paste.x0
    want_top = tile.kept_top and top_band > 0
    want_left = tile.kept_left and left_band > 0
    disp_top = None
    disp_left = None

    if seam_mode == SEAM_WARP:
        # Sampled at absolute canvas coords along each seam line, so adjacent tiles agree.
        if want_top:
            xs = torch.arange(paste.x0, paste.x1, dtype=torch.float32)
            ys = torch.full_like(xs, float(core.y0))
            disp_top = warp_amount * _warp_slice(warp_low, warp_scale, ys, xs)
        if want_left:
            ys = torch.arange(paste.y0, paste.y1, dtype=torch.float32)
            xs = torch.full_like(ys, float(core.x0))
            disp_left = warp_amount * _warp_slice(warp_low, warp_scale, ys, xs)
    elif seam_mode == SEAM_MIN_ERROR:
        # Squared difference between this tile's refinement and the already-pasted neighbor,
        # averaged over batch and channels. Only the BAND is a valid error surface: inside
        # the core, `region` is raw un-refined canvas, so those rows/cols are excluded.
        # The DP is small and strictly sequential, so it runs on the CPU.
        err = ((sub - region) ** 2).mean(dim=(0, 3)).float().cpu()   # [h, w]
        if want_top:
            disp_top = _path_displacement(err[:top_band, :], top_band, FEATHER_PLATEAU, FEATHER_FALLOFF)
        if want_left:
            disp_left = _path_displacement(err[:, :left_band].T.contiguous(), left_band, FEATHER_PLATEAU, FEATHER_FALLOFF)

    return disp_top, disp_left


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


def _manifest_line(tile_idx, tile):
    # One human-readable row per tile for the seam-debug manifest: the geometry needed to
    # line the dumped PNGs up in canvas space (core is the hard-owned region; paste is the
    # feathered top/left band; crop is the full sampled extent incl. the frozen anchor ring).
    def rc(r):
        return "({},{})-({},{})".format(r.x0, r.y0, r.x1, r.y1)

    return "t{:02d} r{}c{} cls={} kept_top={} kept_left={} core={} paste={} crop={}".format(
        tile_idx, tile.row, tile.col, tile.cls, tile.kept_top, tile.kept_left,
        rc(tile.core), rc(tile.paste_rect), rc(tile.crop_rect),
    )


def _refine_tiles(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, seam_mode=SEAM_MIN_ERROR, warp_amount=0.5, warp_scale=64, region_pixel=None):
    # Tiled img2img: pad to /8 → solve the grid → encode/sample/decode each tile in
    # raster order from a live canvas → composite by a directional feather → crop back.
    # A 1x1 grid (or overlap=ctx=0) reproduces the feature-2 whole-image path bit for bit.
    # region_pixel (optional [B,H,W] binary float at input pixel size) gates each tile's
    # denoise_mask to the masked region so only it diffuses; the background stays frozen.
    import comfy.model_management

    # Opt-in seam debug (CATR_DEBUG_DIR). `_save(name, tensor)` is a no-op when unset, so
    # nothing below changes the output — only side-effect PNGs and a manifest are written.
    debug_dir = _debug_dir()
    debug_manifest = []
    if debug_dir is not None:
        import os
        os.makedirs(debug_dir, exist_ok=True)

        def _save(name, tensor):
            _dump_png(os.path.join(debug_dir, name), tensor)
    else:
        def _save(name, tensor):
            return None

    pixels = image[..., :3]
    padded, (height, width) = pad_image_to_multiple(pixels)
    batch, canvas_h, canvas_w = padded.shape[0], padded.shape[1], padded.shape[2]

    # Region gate at latent resolution. Pad the pixel mask to the /8 canvas with constant
    # 0 (the padded strip is always cropped away), then cover-downsample by 8. max_pool2d
    # (NOT avg+threshold) so every pixel with region==1 lands in a DIFFUSED latent cell —
    # else a subject-edge pixel would be frozen yet pasted. Over-cover is harmless: extra
    # diffused cells fall outside the mask and are discarded by the AA composite.
    region_latent = None
    if region_pixel is not None:
        region_padded = torch.nn.functional.pad(region_pixel, (0, canvas_w - width, 0, canvas_h - height))
        region_latent = (torch.nn.functional.max_pool2d(region_padded[:, None].float(), 8) > 0).float()[:, 0]

    sx = grid.solve_axis(canvas_w, max_tile_width, context_anchor, context_overlap)
    sy = grid.solve_axis(canvas_h, max_tile_height, context_anchor, context_overlap)
    layout = grid.build_layout(canvas_w, canvas_h, sx, sy, context_anchor, context_overlap)

    # One noise draw for the whole canvas, then sliced per tile: per-tile draws would
    # give every same-shaped tile identical noise, and slices are spatially anchored
    # so overlapping crop regions of fade-expanded tiles agree on noise. prepare_noise
    # reads only size/dtype/layout, so the zeros dummy (encode outputs float32) keeps
    # a 1x1 grid bit-identical to the whole-image {"samples": latent} draw.
    dummy = torch.zeros((batch, vae.latent_channels, canvas_h // 8, canvas_w // 8), dtype=torch.float32)
    canvas_noise = noise.generate_noise({"samples": dummy})

    # Seeded warp field for the `warp` seam mode, drawn once per run and sampled per seam in
    # absolute canvas coords (same discipline as canvas_noise). Seeding off noise.seed keeps
    # a run reproducible: same seed -> same meander.
    warp_low = _warp_field(canvas_h, canvas_w, max(8, warp_scale), noise.seed) if seam_mode == SEAM_WARP else None

    steps = sigmas.shape[-1] - 1
    for_tile = make_tile_progress(guider.model_patcher, steps, len(layout.tiles))
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
                _save("t{:02d}_r{}c{}_enc.png".format(tile_idx, tile.row, tile.col), enc)
            tile_latent = vae.encode(enc)
        else:
            # Whole-image / n=1 path: crop == core, no anchor ring and no already-
            # processed neighbor, so encode the live crop directly (feature-2, byte-
            # for-byte). Kept exactly as before to preserve the whole-image path.
            tile_latent = vae.encode(canvas[:, crop.y0:crop.y1, crop.x0:crop.x1, :])
        # .contiguous() because downstream sampler code may .view() the slice; a 1x1
        # grid slices the full tensor, so it returns self unchanged.
        tile_noise = canvas_noise[:, :, crop.y0 // 8:crop.y1 // 8, crop.x0 // 8:crop.x1 // 8].contiguous()
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
            denoise_mask = tile_gradient(crop, tile.overlap_inner_rect, 0, scale=8) if expanded else None
        else:
            reg = region_latent[:, crop.y0 // 8:crop.y1 // 8, crop.x0 // 8:crop.x1 // 8]
            grad = tile_gradient(crop, tile.overlap_inner_rect, 0, scale=8) if expanded else 1.0
            denoise_mask = grad * reg
        samples = sample_latent(guider, sampler, sigmas, tile_noise, noise.seed, tile_latent, denoise_mask=denoise_mask, callback=for_tile(tile_idx))
        decoded = vae.decode(samples)
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
            # Seam-mode displacement: straight (None), a coherent-noise meander, or the
            # minimum-error cut. Computed from sub/region, so it must follow both.
            disp_top, disp_left = _seam_displacements(seam_mode, tile, sub, region, warp_low, max(8, warp_scale), warp_amount)
            alpha = feather_alpha(paste, core, layout.overlap, tile.kept_top, tile.kept_left, disp_top=disp_top, disp_left=disp_left).to(canvas.device)[..., None]
            if debug_dir is not None:
                # region is a VIEW into canvas, so dump it BEFORE the composite overwrites it.
                tag = "t{:02d}_r{}c{}".format(tile_idx, tile.row, tile.col)
                _save(tag + "_decoded.png", decoded)
                _save(tag + "_alpha.png", alpha[..., 0])
                _save(tag + "_region_before.png", region)
            canvas[:, paste.y0:paste.y1, paste.x0:paste.x1, :] = alpha * sub + (1.0 - alpha) * region
            if debug_dir is not None:
                _save(tag + "_region_after.png", canvas[:, paste.y0:paste.y1, paste.x0:paste.x1, :])
                debug_manifest.append(_manifest_line(tile_idx, tile))
        else:
            if debug_dir is not None:
                _save("t{:02d}_r{}c{}_decoded.png".format(tile_idx, tile.row, tile.col), decoded)
                debug_manifest.append(_manifest_line(tile_idx, tile))
            canvas[:, crop.y0:crop.y1, crop.x0:crop.x1, :] = decoded
    if debug_dir is not None:
        import os
        _save("canvas_final.png", crop_image_to(canvas, height, width))
        with open(os.path.join(debug_dir, "manifest.txt"), "w", encoding="utf-8") as manifest_file:
            manifest_file.write("\n".join(debug_manifest) + "\n")
    return crop_image_to(canvas, height, width)


def refine_image(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, seam_mode=SEAM_MIN_ERROR, warp_amount=0.5, warp_scale=64, mask=None):
    # Entry point. Without a mask this is the whole-image / multi-tile refine, byte-for-
    # byte the extracted _refine_tiles body. With a mask: harden (>=0.5) → union bbox over
    # the batch → crop to bbox + context_anchor (frozen-background halo) → tile-refine the
    # crop with every tile's denoise_mask gated to the masked region → composite the
    # refined crop back through a 1px anti-aliased edge. The mask=0 background survives from
    # the ORIGINAL pixels (never the VAE-decoded crop), so it is never re-diffused.
    # seam_mode (and the warp_* knobs it may use) positions the inter-tile directional
    # feather only; it never touches the mask-boundary composite, which stays a hard binary
    # gate plus a 1px anti-alias (mask-boundary-conditioning-not-feather).
    if sigmas.numel() < 2:
        # Zero steps: nothing to sample, and the lossy VAE roundtrip would degrade the image.
        return image.clone()
    if mask is None:
        return _refine_tiles(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, seam_mode, warp_amount, warp_scale, region_pixel=None)

    # Region-mask path. Harden a soft input at 0.5 — a fractional denoise mask would leave
    # the under-refined halo we reject for turbo (finding-dd-fade-artifacts-turbo).
    mask_bin = mask >= 0.5
    bbox = _mask_bbox(mask_bin)
    if bbox is None:
        # Empty mask: nothing to refine, and (unlike the input) a clone is a safe no-op.
        return image.clone()
    height, width = image.shape[1], image.shape[2]
    y0, y1, x0, x1 = _expand_snap_clamp(bbox, context_anchor, height, width)
    sub_image = image[:, y0:y1, x0:x1, :]
    sub_mask = mask_bin[:, y0:y1, x0:x1].to(image.dtype)
    refined_sub = _refine_tiles(sub_image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, seam_mode, warp_amount, warp_scale, region_pixel=sub_mask)
    # Composite the refined crop back through the anti-aliased mask edge. Outside the crop,
    # and wherever aa == 0 inside it, the output is the byte-identical original image.
    # Narrow to RGB: _refine_tiles decodes a 3-channel refined_sub (as the no-mask path does,
    # dropping a 4-channel input's alpha), so the composite and the output stay 3-channel.
    rgb = image[..., :3]
    out = rgb.clone()
    aa = _aa_alpha(sub_mask)[..., None]
    out[:, y0:y1, x0:x1, :] = aa * refined_sub + (1.0 - aa) * rgb[:, y0:y1, x0:x1, :]
    return out
