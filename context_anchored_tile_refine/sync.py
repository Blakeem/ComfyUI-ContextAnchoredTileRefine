"""Synchronized tiles: the engine's components, its run loop, and the region path.

Every tile is a LANE — one ordinary full-length `guider.sample()` call, held at each sigma
step by stepper.py's barrier so no tile ever advances past a neighbour. This module builds
everything a run needs BEFORE the first eval, then drives the run (`refine_sync`, which
takes the optional region MASK and owns the bbox crop and the anti-aliased composite back).

Semantics are ported from the owner-judged harness tests-AB/run_ab_sync.py, re-based for
full-length lane calls: the harness's per-step resume identity, its per-sampler state dicts
and its lead_ring_patch were workarounds for chaining one step at a time and have no
counterpart here (stepper.py runs the stock sampler function end to end).

    stage L   pad to /8, solve the grid ONCE, and run the VL conditioning pre-pass over
              THAT layout's tiles — one solve, so the layout that slices a tile's positive
              is by construction the layout the engine tiles with.
    stage E   per-tile `vae.encode` of the RAW crop windows (never a whole-canvas VAE call —
              its ~21 GiB spike is why the engine tiles at all), the butted CORES assembled
              into ONE canvas latent C_0, with full coverage asserted.
    stage N   ONE canvas-shaped noise draw, sliced per lane window — the identical contract
              as the raster path (sampling._refine_tiles), so a lane keeps the noise its
              tile would have had.
    stage P   per lane: its RAW C_0 window (latent), its noise window CLONE, its binary
              denoise mask in the canonical form, and its OWN guider carrying that tile's
              positive.
    stage S   ONE stepper run over every lane at once. At each sigma step the surgery hook
              consolidates the lanes into the ONE maintained canvas (directional feather at
              LATENT scale, raster order) and scatters every window back, so no two lanes
              disagree on a shared cell at a step start; the frozen ring is presented on the
              settled lead schedule by ONE instance patch held around the whole run.
    stage D   per-tile `vae.decode` of the CANVAS windows — never a lane's own `x`, whose
              ring cells carry the unrefined source — composited by the stock pixel feather.
              No DC match and no min-error cut: both sides of every band decoded ONE latent,
              so neither has anything left to correct.

REGION (mask) PATH: `refine_sync(mask=...)` crops to the mask bbox plus `context_anchor`
(sampling's own `_mask_bbox` / `_expand_snap_clamp`), runs stages L..D over THAT crop with
every lane's denoise mask intersected with the region gate, and composites the crop back
through the shipped 1px anti-aliased edge (`sampling._aa_alpha`) — no feather at the mask
boundary (CLAUDE.md prime directive 3), and every pixel the edge weights at 0 stays
byte-identical to the input. The conditioning still reads the WHOLE image: the pre-pass gets
the pre-crop image plus the bbox origin as offsets, so a region tile slices its true place in
the full encode.

COORDINATE CONTRACT (roadmap CANVAS SPACE decision, verified against comfy 0.33.0):
  * A lane's `latent_image` is handed over in RAW space — comfy applies `process_latent_in`
    itself inside guider.sample (samplers.py:1226). EVERY lane's latent is a SLICE of the
    ONE assembled C_0, never its own window encode, so two overlapping lanes hold
    torch.equal values on every shared cell at step 0 and each lane's frozen ring shows the
    canvas (the neighbour's core framing) — which is what anchor_source "source image"
    anchors to.
  * The canvas this engine MAINTAINS across the run lives in PROCESS space
    (`model.process_latent_in(C_0)`), because the lanes' live `.x` tensors the surgery hook
    blends are already process-space. The two spaces are never mixed; a window headed for
    `vae.decode` is converted back with `process_latent_out` (the surgery feature).

torch + stdlib at module scope; `comfy` is imported lazily inside functions and stepper.py
is imported lazily too (a subprocess test pins the comfy half).
"""
import copy
from dataclasses import dataclass

import torch

from . import captions, conds, grid, sampling, vl
from .progress import (
    CANVAS_ENCODE,
    CAPTION_ENCODE,
    CAPTIONS,
    DECODE,
    K_CAPTION,
    SAMPLING,
    VISION_ENCODE,
    W_DECODE_TILE,
    W_ENCODE,
    W_ENCODE_CAPTION_TEXT,
    W_ENCODE_TILE,
)

# What the frozen context_anchor ring around each tile SHOWS the model. "source image": the
# unmodified input, presented on the settled lead schedule (the shipped default — the
# campaign's four-scene sweep ran on it). "live canvas": the in-progress result itself,
# presented by `present_live_ring` (which is why the schedule is NOT entered in that mode —
# nothing frozen-refined is left to lead). The two values are the node's anchor_source
# widget; they are defined here because the preconditions below are mode-scoped.
ANCHOR_SOURCE_IMAGE = "source image"
ANCHOR_SOURCE_CANVAS = "live canvas"
ANCHOR_SOURCES = (ANCHOR_SOURCE_IMAGE, ANCHOR_SOURCE_CANVAS)


@dataclass(frozen=True)
class SyncRun:
    """Everything `_prepare_run` assembles, in the two spaces kept deliberately apart.

    padded/size are the /8 canvas the tile rects index and the unpadded input size the run's
    output is cropped back to; layout is the ONE grid solve; raw_latent is C_0 (RAW space,
    what the lanes slice); canvas is the maintained PROCESS-space canvas; canvas_noise is the
    single draw the lane windows are cloned from; lanes are stepper.LaneSpec objects, in
    layout.tiles order, and ring_gates their per-lane live-canvas ring indicators in the same
    order (the complement of the denoise mask INSIDE the region — see `_prepare_run`).
    """

    padded: torch.Tensor
    size: tuple
    layout: grid.Layout
    raw_latent: torch.Tensor
    canvas: torch.Tensor
    canvas_noise: torch.Tensor
    lanes: tuple
    ring_gates: tuple


def _latent_rect(rect):
    # A grid rect scaled to the /8 latent grid. Every grid coordinate is a multiple of 8 by
    # construction (solve_axis multiple=8), asserted so a future geometry change fails HERE
    # instead of shearing every window, mask and feather by a fraction of a cell.
    coords = (rect.x0, rect.y0, rect.x1, rect.y1)
    if any(c % 8 for c in coords):
        raise RuntimeError(f"tile rect {coords} is not /8 — latent scaling would shear")
    return grid.Rect(rect.x0 // 8, rect.y0 // 8, rect.x1 // 8, rect.y1 // 8)


def encode_canvas_latent(vae, padded, tiles, progress=None):
    # C_0: per-tile vae.encode of the raw crop windows at the stock tile sizes, the butted
    # CORES pasted into one canvas latent. Coverage is asserted — the cores tile the canvas
    # exactly, so every cell (ring regions included) comes from exactly one encode, and each
    # lane's ring therefore shows its neighbour's own framing rather than a second opinion of
    # its own. Window-framing differences where cores butt are sub-1/255 and are then diffused
    # for the whole schedule.
    h8, w8 = padded.shape[1] // 8, padded.shape[2] // 8
    canvas = None
    covered = torch.zeros((h8, w8), dtype=torch.bool)
    for index, tile in enumerate(tiles):
        crop, core = tile.crop_rect, tile.core
        latent = sampling.encode_pixels(vae, padded[:, crop.y0:crop.y1, crop.x0:crop.x1, :]).detach()
        if latent.shape[-2:] != ((crop.y1 - crop.y0) // 8, (crop.x1 - crop.x0) // 8):
            raise RuntimeError(
                f"window encode shape {tuple(latent.shape[-2:])} != crop {crop} at /8 — "
                "this VAE's spatial compression is not the /8 the grid math models")
        if canvas is None:
            canvas = torch.zeros((*latent.shape[:-2], h8, w8), dtype=latent.dtype, device=latent.device)
        cy0, cy1 = (core.y0 - crop.y0) // 8, (core.y1 - crop.y0) // 8
        cx0, cx1 = (core.x0 - crop.x0) // 8, (core.x1 - crop.x0) // 8
        canvas[..., core.y0 // 8:core.y1 // 8, core.x0 // 8:core.x1 // 8] = latent[..., cy0:cy1, cx0:cx1]
        covered[core.y0 // 8:core.y1 // 8, core.x0 // 8:core.x1 // 8] = True
        if progress is not None:
            # The ledger's canvas-encode segment is budgeted per tile, so it advances from
            # OUR count. The production VAE (latent_dim 3) reaches tiled_scale_multidim with
            # no pbar at all, so nothing else here would move it.
            progress.advance((index + 1) * W_ENCODE_TILE)
    if canvas is None or not bool(covered.all()):
        raise RuntimeError("tile cores do not tile the latent canvas — coverage hole")
    return canvas


def check_preconditions(model_patcher, sigmas, anchor_source):
    # Every assumption the sync construction rests on, checked before any GPU time (ported
    # from run_ab_sync.py:635-658). Four are unconditional; the fifth is mode-scoped.
    #
    # The harness ALSO rejected sigmas[0] >= 1 unconditionally, but that guarded its per-step
    # resume identity, which this port deliberately does not have — and denoise 1.0 is a
    # working widget value that yields sigmas[0] == 1.0 exactly under sgm_uniform/normal,
    # which comfy's samplers handle themselves (k_diffusion/sampling.py:168-176). It is kept
    # only where the algebra genuinely breaks: the live-canvas ring divides by (1 - sigma).
    import comfy.model_sampling

    if not torch.all(sigmas[1:] < sigmas[:-1]):
        raise RuntimeError("sigmas are not strictly decreasing — every lane steps ONE shared schedule")
    if float(sigmas[-1]) != 0.0:
        raise RuntimeError("schedule does not end at sigma 0")
    model_sampling = model_patcher.get_model_object("model_sampling")
    if not isinstance(model_sampling, comfy.model_sampling.CONST):
        # The ring algebra IS comfy's CONST noise_scaling inverted (sigma*noise +
        # (1-sigma)*latent). On an EPS/V_PREDICTION model the same expression mis-scales
        # silently, so this is a hard error rather than a fallback: from 1.6.0 sync is the
        # only VL path, and the base node is the raster path for every other model family.
        raise RuntimeError("sync stepping is derived for CONST (flow) models; got "
                           f"{type(model_sampling).__name__}")
    # model_patcher.model is the comfy BaseModel — _model_re_noises_anchor walks the BASE
    # model's __mro__ (the patcher's has no scale_latent_inpaint and would always answer False).
    if not sampling._model_re_noises_anchor(model_patcher.model):
        raise RuntimeError("model overrides scale_latent_inpaint — the frozen-ring construction "
                           "does not hold; sync mode is untested there")
    if anchor_source == ANCHOR_SOURCE_CANVAS and float(sigmas[0]) >= 1.0:
        raise RuntimeError(
            f"anchor_source '{ANCHOR_SOURCE_CANVAS}' presents the ring as x / (1 - sigma), which "
            f"is undefined at sigma 1.0; this schedule starts at {float(sigmas[0])}. Lower denoise, "
            f"or use anchor_source '{ANCHOR_SOURCE_IMAGE}'.")


def build_canvas_noise(vae, noise, canvas_h, canvas_w, batch=1, batch_size=1, batch_index=0):
    # ONE canvas-wide draw, sliced per lane: per-lane draws would give every same-shaped tile
    # identical noise, and slices are spatially anchored so overlapping windows agree. The
    # dummy mirrors vae.encode's latent layout exactly — a video-family VAE (latent_dim 3,
    # e.g. Krea 2's Wan VAE) encodes an image batch to a 5-D [B,C,1,h,w] latent — and is drawn
    # at the FULL batch with row batch_index selected, so a picture inside refine_image's
    # picture loop keeps the noise it would have drawn in one shared call (prepare_noise draws
    # a SINGLE randn over the whole latent size). The identical contract as
    # sampling._refine_tiles:751-762, deliberately: a picture must not change noise because it
    # was refined by a different engine.
    latent_time = (1,) if getattr(vae, "latent_dim", 2) == 3 else ()
    draw_batch = batch_size if batch_size > 1 else batch
    dummy = torch.zeros((draw_batch, vae.latent_channels, *latent_time, canvas_h // 8, canvas_w // 8), dtype=torch.float32)
    canvas_noise = noise.generate_noise({"samples": dummy})
    if batch_size > 1:
        canvas_noise = canvas_noise[batch_index:batch_index + 1]
    return canvas_noise


def build_tile_positives(vl_clip, source, tiles, preset, batch_size=1, batch_index=0,
                         vl_context=None, progress=None):
    # The VL conditioning pre-pass, run by the ENGINE over the tiles it is about to sample —
    # the same three surfaces the raster branch built. It runs before the first lane so the
    # text encoder is resident exactly once, before the diffusion model loads.
    # `preset` is the run's resolved vlm_method (captions.resolve_method), so the settings
    # file is read ONCE per run by the caller and every count below reads the same block.
    # Cost differs per surface: "vision tokens" is ONE whole-canvas encode for the run,
    # "captions" is one clip.generate plus one cheap text encode per tile, and
    # "vision tokens and captions" is that one shared encode plus both per tile. A preset
    # carrying a style instruction adds one whole-image clip.generate per row on both caption
    # surfaces. Only the clip.generate pre-pass scales with tile count.
    # captions.generate_tile_captions owns the interrupt check and the ProgressBar that
    # keep it cancellable.
    # vl_context (region path only) is (full_image, offset_x, offset_y): the VISION encode
    # reads the FULL image and each region tile's rect is offset into its frame, so a masked
    # refine stays globally informed instead of only seeing its own crop. CAPTIONS always
    # describe THIS run's own canvas — the region crop the tiles actually sample — which is
    # why `source` is kept separate from the encode source. The STYLE caption is the one
    # exception and reads encode_source, so a masked refine's style stays global.
    # PROGRESS: this function owns the pre-pass's ledger segments, and their ORDER is the
    # code's own — the captions are written FIRST and the conditioning built after, so the
    # bar never claims an encode that has not started. "vision tokens" opens one segment,
    # the two caption surfaces open two.
    encode_source, offset_x, offset_y = (source, 0, 0) if vl_context is None else vl_context
    n_tiles, rows = len(tiles), int(source.shape[0])
    if preset.surface == captions.VLM_METHOD_VISION:
        if progress is not None:
            progress.open(VISION_ENCODE, W_ENCODE)
        return vl.build_global_slices(vl_clip, encode_source, tiles, offset_x=offset_x, offset_y=offset_y)
    n_captions = (n_tiles + (1 if preset.style_instruction else 0)) * rows
    if progress is not None:
        progress.open(CAPTIONS, n_captions * K_CAPTION, chunks=n_captions)
    tile_captions = captions.generate_tile_captions(vl_clip, source, tiles, preset,
                                                    batch_size, batch_index, progress=progress,
                                                    style_source=encode_source)
    if preset.surface == captions.VLM_METHOD_VISION_CAPTIONS:
        # ONE whole-canvas vision encode plus one caption TEXT encode per tile — the text
        # half scales with the grid, so it is budgeted rather than folded into W_ENCODE.
        if progress is not None:
            progress.open(VISION_ENCODE, W_ENCODE + n_tiles * W_ENCODE_CAPTION_TEXT)
        return captions.build_slice_caption_conds(vl_clip, encode_source, tiles, tile_captions, offset_x=offset_x, offset_y=offset_y)
    if progress is not None:
        # Captions ONLY: no canvas encode at all, but still one text-encoder pass per tile.
        progress.open(CAPTION_ENCODE, n_tiles * W_ENCODE_CAPTION_TEXT)
    return captions.build_caption_conds(vl_clip, tile_captions)


def build_lane_guiders(guider, tile_positives):
    # One guider PER LANE: comfy's CFGGuider stores per-run state on itself (inner_model,
    # conds, loaded_models) and tears it down in outer_sample's finally, so N in-flight
    # sample() calls on ONE guider clobber each other. A shallow copy shares the model patcher
    # (deliberately — one model, one GPU stream) and gets its OWN original_conds map.
    #
    # strip_control is load-bearing, not tidiness: each tile's positive is a fresh vision
    # slice that carries no control chain, so a control left on the caller's NEGATIVE would
    # steer the uncond branch alone against an uncropped hint (conds.py:129-137).
    # The caller's guider is never touched — no swap, no restore, nothing to leak on a raise.
    pristine_conds = getattr(guider, "original_conds", None)
    if not isinstance(pristine_conds, dict) or "positive" not in pristine_conds:
        raise ValueError(
            "Context-Anchored Tile Refine (sync): the guider must keep 'positive' in "
            "original_conds (the CFGGuider convention); got "
            f"{sorted(pristine_conds) if isinstance(pristine_conds, dict) else type(pristine_conds).__name__}")
    stripped_conds = conds.strip_control(pristine_conds)

    lane_guiders = []
    for positive in tile_positives:
        lane_guider = copy.copy(guider)
        if not isinstance(getattr(lane_guider, "original_conds", None), dict):
            # A third-party guider pack whose __copy__ drops or reshapes original_conds would
            # otherwise sample every lane with the caller's untouched conds — every tile
            # conditioned on the placeholder positive, silently.
            raise ValueError(
                "Context-Anchored Tile Refine (sync): copying this guider did not carry a dict "
                "'original_conds', so its lanes cannot be given their own tile conditioning. "
                "The sync engine needs a CFGGuider-compatible guider.")
        swapped_conds = dict(stripped_conds)
        swapped_conds["positive"] = positive
        lane_guider.original_conds = swapped_conds
        lane_guiders.append(lane_guider)
    return lane_guiders


def _prepare_run(image, guider, sigmas, vae, noise, max_tile_width, max_tile_height,
                 context_anchor, context_overlap, vl_clip, vlm_method=captions.VLM_METHOD_VISION,
                 anchor_source=ANCHOR_SOURCE_IMAGE, batch_size=1, batch_index=0,
                 region_pixel=None, vl_context=None, progress=None, sampler=None):
    # Everything a run needs, in one place, before any sampling: ONE picture in
    # ([1,H,W,C] — the caller owns the batch loop), one grid solve, one conditioning
    # pre-pass, one C_0, one noise draw, N lanes out.
    # region_pixel (region path, [B,H,W] binary float at THIS call's pixel size) gates every
    # lane's denoise mask to the masked region; vl_context is the (full_image, offset_x,
    # offset_y) triple the conditioning pre-pass reads. Both are None on the whole-image path
    # and leave every step below untouched.
    from . import stepper

    # ---- inputs
    if image.shape[0] != 1:
        raise ValueError(
            f"Context-Anchored Tile Refine (sync): the engine refines ONE picture at a time, got "
            f"a batch of {image.shape[0]}. The caller's picture loop is what keeps peak VRAM off "
            "the batch size.")
    if anchor_source not in ANCHOR_SOURCES:
        raise ValueError(f"anchor_source must be one of {list(ANCHOR_SOURCES)}, got {anchor_source!r}")
    if vl_clip is None:
        raise ValueError(
            "Context-Anchored Tile Refine (sync): the sync engine is the VL path — every lane's "
            "positive is its tile's slice of the VL encode, so it needs the VL CLIP.")
    # ONE settings read for the whole picture, before any GPU time: the ledger's caption
    # count and the pre-pass's own must come from the same block, and a file edited mid-run
    # would otherwise give the two different answers.
    preset = captions.resolve_method(vlm_method)
    check_preconditions(guider.model_patcher, sigmas, anchor_source)

    pixels = image[..., :3]
    padded, (height, width) = sampling.pad_image_to_multiple(pixels)
    batch, canvas_h, canvas_w = int(padded.shape[0]), int(padded.shape[1]), int(padded.shape[2])

    # ---- process. Region gate at LATENT resolution, the raster path's rule verbatim
    # (sampling._refine_tiles): pad the pixel mask onto the /8 canvas with constant 0 (the
    # padded strip is always cropped away), then cover-downsample by 8 with max_pool2d — NOT
    # avg+threshold — so every pixel with region == 1 lands in a DIFFUSED latent cell; else a
    # subject-edge pixel would be frozen yet composited back. Over-cover is harmless: the
    # extra diffused cells fall outside the mask and the anti-aliased composite discards them.
    # No pixel-resolution copy is kept, unlike the raster path: this engine has no DC match to
    # gate, because both sides of every band decode ONE latent.
    region_latent = None
    if region_pixel is not None:
        region_padded = torch.nn.functional.pad(region_pixel, (0, canvas_w - width, 0, canvas_h - height))
        region_latent = (torch.nn.functional.max_pool2d(region_padded[:, None].float(), 8) > 0).float()[:, 0]

    # The grid is solved ONCE and layout.tiles is what BOTH the conditioning pre-pass and the
    # lanes below read, so a tile's positive can never be sliced for a different rect than the
    # one it samples.
    sx = grid.solve_axis(canvas_w, max_tile_width, context_anchor, context_overlap, axis="width")
    sy = grid.solve_axis(canvas_h, max_tile_height, context_anchor, context_overlap, axis="height")
    layout = grid.build_layout(canvas_w, canvas_h, sx, sy, context_anchor, context_overlap)
    tiles = layout.tiles

    # This picture's whole plan block at TRUE sizes, before any pre-pass work fills: the
    # grid and the eval plan are both known here, and presetting now is what keeps the
    # displayed fraction honest — sized only at each segment's open, the total was tiny
    # through the captions (bar ~full) and then ~n_tiles-fold larger at sampling (bar
    # visibly dropping), the value itself monotone the whole time.
    if progress is not None and sampler is not None:
        total_evals, _hook_step_at = stepper.plan_evals(sampler, sigmas)
        # The style caption count mirrors build_tile_positives' own arithmetic, off the ONE
        # preset resolved above, so the segment the ledger sizes is the segment the pre-pass
        # then opens.
        style_rows = batch if preset.style_instruction else 0
        progress.preset_picture(vlm_method, len(tiles), batch, total_evals, style_rows=style_rows)

    tile_positives = build_tile_positives(vl_clip, padded, tiles, preset, batch_size, batch_index,
                                          vl_context=vl_context, progress=progress)
    if len(tile_positives) != len(tiles):
        raise RuntimeError(
            f"the VL pre-pass built {len(tile_positives)} positives for {len(tiles)} tiles — the "
            "conditioning and the tiling disagree about the layout")
    # The per-tile VAE window encodes are their own phase: one vae.encode per tile, minutes on
    # a large grid, and until now the only phase with no bar of any kind. Closed before the
    # noise draw, which is a single cheap call.
    if progress is not None:
        progress.open(CANVAS_ENCODE, len(tiles) * W_ENCODE_TILE)
    raw_latent = encode_canvas_latent(vae, padded, tiles, progress=progress)
    if progress is not None:
        progress.close()
    canvas_noise = build_canvas_noise(vae, noise, canvas_h, canvas_w, batch, batch_size, batch_index)
    if tuple(canvas_noise.shape) != tuple(raw_latent.shape):
        raise RuntimeError(
            f"canvas noise {tuple(canvas_noise.shape)} != canvas latent {tuple(raw_latent.shape)}. "
            "This VAE's latent layout is not supported (e.g. a video VAE folding an image batch "
            "into frames, or a non-8x spatial compression).")

    model = guider.model_patcher.model
    canvas = model.process_latent_in(raw_latent)
    if canvas.data_ptr() == raw_latent.data_ptr():
        # The maintained canvas must never alias C_0: the lanes' frozen rings are restored
        # from their RAW latent at every eval, so a canvas write would rewrite what they are
        # anchored to. A real latent_format always allocates; a pass-through one would not.
        canvas = canvas.clone()

    lane_guiders = build_lane_guiders(guider, tile_positives)
    load_device = guider.model_patcher.load_device
    lanes = []
    ring_gates = []
    for tile, lane_guider in zip(tiles, lane_guiders, strict=True):
        rect = _latent_rect(tile.crop_rect)
        # (y0, y1, x0, x1) in LATENT CELLS — the stepper's shared-noise providers read exactly
        # this shape, and the surgery hook gets it back untouched.
        window = (rect.y0, rect.y1, rect.x0, rect.x1)
        # .contiguous() because downstream sampler code may .view() the slice; on a 1x1 grid
        # the window IS the whole canvas, so it returns the tensor unchanged.
        latent = raw_latent[..., rect.y0:rect.y1, rect.x0:rect.x1].contiguous()
        # .clone(), NOT .contiguous(): on a 1x1 grid a contiguous() window is the canvas draw
        # ITSELF, and comfy binds the tensor as model_k.noise, which `present_live_ring` zeroes
        # IN PLACE at the ring cells — through a view that would zero a neighbour's core cells.
        # A clone never aliases.
        lane_noise = canvas_noise[..., rect.y0:rect.y1, rect.x0:rect.x1].clone()
        # Binary denoise mask: full-strength denoise over core + context_overlap (==
        # overlap_inner_rect), frozen only through the context_anchor ring. Binary because
        # KSamplerX0Inpaint re-applies it every step, so a fractional cell is only ever partly
        # denoised (an under-refined seam halo at low step counts). Normalized HERE because
        # the stepper calls guider.sample directly — sample_latent, which normalizes on the
        # raster path, is not on this path at all.
        gradient = sampling.tile_gradient(tile.crop_rect, tile.overlap_inner_rect, scale=8)
        if region_latent is None:
            denoise_mask = gradient
            ring = 1.0 - gradient
        else:
            # Region path: intersect with the region gate, so only the masked region diffuses
            # and the surrounding background stays frozen source context. .to(reg.device)
            # because tile_gradient builds its indicator from torch.arange (always CPU) while
            # region_latent follows the MASK/IMAGE device (CUDA under --gpu-only) — the
            # elementwise multiply would raise across devices otherwise.
            reg = region_latent[:, rect.y0:rect.y1, rect.x0:rect.x1]
            denoise_mask = gradient.to(reg.device) * reg
            # The RING is what a NEIGHBOUR diffuses and this lane freezes: mask-0 cells INSIDE
            # the region. Mask-0 cells outside it are the frozen background, which no lane ever
            # diffuses — its context is the source image under EITHER anchor_source, so the
            # live-canvas rewrite must leave it alone. Both terms are binary and denoise_mask
            # is `region AND gradient`, so the subtraction is exactly `region AND NOT gradient`.
            ring = reg - denoise_mask
        denoise_mask = sampling._normalize_denoise_mask(denoise_mask, latent, load_device)
        # The ring gate rides the SAME canonical broadcastable form as the denoise mask it
        # complements — [B,1,h,w], or [B,1,1,h,w] for a video-family latent, on the guider's
        # load device — so the live-canvas surgery can gate a lane's latent window with it
        # directly. Built for every run and consulted only by `present_live_ring`.
        ring_gates.append(sampling._normalize_denoise_mask(ring, latent, load_device))
        lanes.append(stepper.LaneSpec(
            guider=lane_guider, sigmas=sigmas, noise=lane_noise, latent=latent,
            denoise_mask=denoise_mask, seed=noise.seed, window=window))

    # ---- output
    return SyncRun(
        padded=padded, size=(height, width), layout=layout, raw_latent=raw_latent,
        canvas=canvas, canvas_noise=canvas_noise, lanes=tuple(lanes),
        ring_gates=tuple(ring_gates),
    )


def latent_plateau(band_cells):
    # The feather's plateau fraction, re-derived for a LATENT-scale band. The shipped 10% is
    # tuned for a PIXEL band — 10% of 32px pins the seam-most ~3 pixels at exactly 1.0 — and
    # the same 10% of that band's FOUR latent cells pins nothing (0.4 of a cell), leaving the
    # seam-most cell at 0.945: 5.5% of the neighbour bleeding across the seam line at EVERY
    # step, which per-step blending then compounds. 0.5/band is the smallest plateau whose
    # clamp lands the seam-most cell CENTRE exactly on 1.0 (its u is 1 - 0.5/band, so
    # u / (1 - plateau) == 1); the falloff stays stock. context_overlap 0 is a legal widget
    # value and leaves no band at all — the zero guard, verbatim from
    # tests-AB/run_ab_sync.py:464.
    if not band_cells:
        return sampling.FEATHER_PLATEAU
    return max(sampling.FEATHER_PLATEAU, 0.5 / band_cells)


def build_blends(layout, canvas):
    # Per tile, in raster order: (paste rect in LATENT CELLS, feather alpha over it), the two
    # constants every consolidation pass reuses. paste_rect is the core plus the
    # context_overlap band on the ALREADY-BLENDED sides only (top/left in raster order), so
    # each interior seam is cross-dissolved exactly once, by the later tile — grid.py's
    # raster-order invariant, applied here per STEP instead of once at the end.
    # alpha follows the canvas's device: it is built from torch.arange (always CPU) while the
    # lanes' live tensors reach the hook on the guider's load device.
    band_cells = layout.overlap // 8
    plateau = latent_plateau(band_cells)
    blends = []
    for tile in layout.tiles:
        paste = _latent_rect(tile.paste_rect)
        alpha = sampling.feather_alpha(paste, _latent_rect(tile.core), band_cells,
                                       tile.kept_top, tile.kept_left, plateau=plateau)
        # [h,w] -> broadcastable over a [B,C,h,w] latent, or a [B,C,T,h,w] video-family one.
        alpha = alpha.reshape((1, 1) + (1,) * (canvas.ndim - 4) + alpha.shape)
        blends.append((paste, alpha.to(canvas.device)))
    return tuple(blends)


def blend_lanes(canvas, blends, windows, values):
    # ONE consolidation pass: each tile's paste region cross-dissolved into the canvas, in
    # raster order, so afterwards every cell shared by two lane windows holds ONE value —
    # which is what makes both sides of a band decode one latent at the end.
    # RASTER INDUCTION: a tile's feathered band lies inside an EARLIER tile's core, and that
    # earlier tile wrote its whole core in this same pass, so the (1 - alpha) term always
    # reads this pass's value and never a stale one. Tile 0's alpha is all-ones.
    # `windows` are the lanes' (y0, y1, x0, x1) canvas rects and `values` their process-space
    # window tensors, both in tile order.
    for (paste, alpha), window, value in zip(blends, windows, values, strict=True):
        wy0, wx0 = window[0], window[2]
        sub = value[..., paste.y0 - wy0:paste.y1 - wy0, paste.x0 - wx0:paste.x1 - wx0]
        region = canvas[..., paste.y0:paste.y1, paste.x0:paste.x1]
        canvas[..., paste.y0:paste.y1, paste.x0:paste.x1] = alpha * sub + (1.0 - alpha) * region


def make_sync_progress(model_patcher, steps, batch_size=1, batch_index=0, progress=None):
    # latent_preview.prepare_callback at ENGINE scale: ONE progress report over the run's sigma
    # steps (every lane advances together, so there is nothing per-tile to count) and one
    # WHOLE-IMAGE preview per step — something the raster path could never show, because it
    # only ever holds one tile's latent at a time. batch_size/batch_index come from
    # refine_image's picture loop: every picture reports on the SAME total and starts where
    # the previous one ended, so the bar advances once across the whole run.
    # WITH a ledger (`progress`) no standalone bar is constructed at all, and the hook does
    # NOT move the fill either — the stepper's per-eval on_eval tick owns it exactly (one
    # unit per completed eval; the old per-step advance claimed a step's evals at its START
    # and jumped ahead of the ticks). advance(0.0, ...) is fill-preserving: it only carries
    # the preview. The preview rides along either way; it must not regress.
    import comfy.utils
    import latent_preview

    previewer = latent_preview.get_previewer(model_patcher.load_device, model_patcher.model.latent_format)
    total = steps * batch_size
    offset = steps * batch_index
    pbar = None if progress is not None else comfy.utils.ProgressBar(total)

    def update(step_index, canvas, lanes):
        preview = None
        if previewer is not None and step_index > 0:
            # DENOISED, never the trajectory: `x` is noisy at the current sigma and previews
            # as noise, while `denoised` is the model's x0 prediction — the same tensor kind
            # (and space) the raster path's per-tile callback previews. Whole windows pasted
            # in raster order, no feather: this is the preview image, not the result. Step 0
            # has no prediction at all (a lane's `denoised` is None until its first eval
            # returns), so the bar advances there with no image.
            preview_canvas = torch.zeros_like(canvas)
            for lane in lanes:
                y0, y1, x0, x1 = lane.window
                preview_canvas[..., y0:y1, x0:x1] = lane.denoised
            preview = previewer.decode_latent_to_preview_image("JPEG", preview_canvas)
        if pbar is None:
            progress.advance(0.0, preview)
        else:
            pbar.update_absolute(offset + step_index + 1, total, preview)

    return update


def present_live_ring(lanes, ring_gates, sigma, zero_noise=False):
    """anchor_source "live canvas": present every lane's frozen ring as its NEIGHBOUR's LIVE
    trajectory at this step's sigma, instead of the unmodified source.

    THE ALGEBRA. comfy rebuilds the model's input at every eval as
        x * denoise_mask + scale_latent_inpaint(sigma, noise, latent_image) * (1 - denoise_mask)
    (samplers.py:634-643), and core's default scale_latent_inpaint is CONST's noise_scaling,
    `sigma * noise + (1 - sigma) * latent_image` (model_base.py:408 — the precondition check
    rejects every model that overrides it). So with a lane's `noise` ring cells zeroed ONCE and
    its `latent_image` ring cells rewritten each step to `x_ring / (1 - sigma_step)`, the
    presented ring collapses to exactly `x_ring` — and `x` is the consolidated canvas, because
    the surgery hook has just scattered it into every lane. Two consequences, both deliberate:
      * the equality is EXACT at each step's FIRST eval, which is where the hook runs; a
        2-eval sampler's intra-step eval sees the same state rescaled by
        (1 - sigma_mid) / (1 - sigma_step) — the same within-step deviation a stock
        frozen-ring run carries (tests-AB/run_ab_sync.py:117-121), not a defect.
      * `1 - sigma_step` must stay positive. check_preconditions already rejects a schedule
        starting at sigma >= 1 in this mode and sigmas are strictly decreasing, so this guard
        can only fire if that intake is bypassed.
    `latent_image` is the process_latent_in'd tensor comfy binds in KSAMPLER.sample
    (samplers.py:986) and a lane's `x` is process-space too, so the rewrite stays inside ONE
    space (the roadmap CANVAS SPACE decision).
    """
    scale = 1.0 - float(sigma)
    if scale <= 0.0:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (sync): anchor_source {ANCHOR_SOURCE_CANVAS!r} presents "
            f"the ring as x / (1 - sigma), which is undefined at sigma {float(sigma)}.")
    for lane, gate in zip(lanes, ring_gates, strict=True):
        model_k = lane.model_k
        if zero_noise:
            # ONCE, from the step-0 hook: AFTER KSAMPLER bound this tensor (samplers.py:991)
            # and consumed it for the initial noise_scaling (:993), BEFORE the first eval — so
            # the lane's step-0 trajectory still carries the noise its tile would have had.
            # In place, which is safe exactly because every lane's noise window is a CLONE
            # (_prepare_run): through a view this would zero a neighbour's core cells.
            model_k.noise.mul_(1.0 - gate)
        # REBIND, not an in-place write: on a degenerate geometry that tensor can still be the
        # caller's own latent (a pass-through process_latent_in over a 1x1-grid window, where
        # the window IS the whole canvas), and KSamplerX0Inpaint reads the attribute fresh at
        # every eval anyway.
        model_k.latent_image = torch.where(gate > 0.0, lane.x / scale, model_k.latent_image)


def _make_surgery_hook(canvas, blends, progress, ring_gates=None):
    # The stepper's per-step surgery, run with the whole fleet parked at the barrier and the
    # step not yet begun: consolidate the lanes into the canvas, hand every lane the
    # consolidated view of its own window back, present the live-canvas ring if that is the
    # mode, then report progress.
    # At step 0 nothing has been evaluated yet and every lane holds the SAME construction over
    # the same canvas slices (sigma0 * noise + (1 - sigma0) * latent, KSAMPLER.sample:992), so
    # that pass only seeds the canvas with the trajectory; from step 1 it is the real
    # consolidation of N independent refinements.
    def hook(step_index, sigma, lanes):
        import comfy.model_management

        # BEFORE the release, and before any work: the stepper routes whatever the hook
        # raises through its abort flag, so a cancel here stops the whole fleet within one
        # eval instead of leaving N full-length sampler calls running to the end.
        comfy.model_management.throw_exception_if_processing_interrupted()
        windows = [lane.window for lane in lanes]
        blend_lanes(canvas, blends, windows, [lane.x for lane in lanes])
        for lane, window in zip(lanes, windows, strict=True):
            y0, y1, x0, x1 = window
            # copy_, never rebinding: the sampler holds its own reference to this tensor, so
            # only an in-place edit reaches the next eval. Scattering the WHOLE window (not
            # just the feathered band) is what makes a lane's right/bottom overlap — cells
            # its neighbour owns — agree with that neighbour at the step start; the frozen
            # ring cells ride along inertly, since KSamplerX0Inpaint rebuilds the ring the
            # model sees from `latent_image`/`noise` and never from `x`.
            lane.x.copy_(canvas[..., y0:y1, x0:x1])
        if ring_gates is not None:
            # AFTER the scatter, never before: the cells the ring presents are the ones just
            # written from the consolidated canvas, i.e. the NEIGHBOUR's live trajectory.
            present_live_ring(lanes, ring_gates, sigma, zero_noise=step_index == 0)
        progress(step_index, canvas, lanes)

    return hook


def decode_composite(vae, model, canvas, padded, layout, progress=None):
    # Stage D: per-tile decode of the CANVAS windows, composited by the stock pixel feather in
    # raster order. Reading the canvas rather than a lane's own `x` is load-bearing twice
    # over: a lane window's frozen ring is restored to its unrefined `latent_image` at every
    # eval (KSamplerX0Inpaint's out-blend), so lane windows would paint an unrefined halo
    # around every tile, and the last step's bands would decode two-valued. NO seam_dc_offset
    # and NO seam_displacements here: both sides of every band decoded ONE latent, so the
    # only band delta left is VAE window framing, which the campaign measured at the floor.
    result = padded.clone()
    for index, tile in enumerate(layout.tiles):
        crop, paste, core = tile.crop_rect, tile.paste_rect, tile.core
        rect = _latent_rect(crop)
        # RAW space for the VAE: the canvas is maintained in the model's process space (the
        # lanes' `x` lives there), and process_latent_out is the exact inverse comfy applies
        # to every sampler result on the way out (samplers.py:1238).
        raw_window = model.process_latent_out(canvas[..., rect.y0:rect.y1, rect.x0:rect.x1])
        decoded = vae.decode(raw_window)
        if decoded.ndim == 5:
            # A video-family VAE decodes the 5-D latent to [B,T,H,W,C]; fold T into the batch
            # exactly as core's VAEDecode node does. T is 1 by construction (the run's
            # noise/latent shape check pins one latent row per image), so this is a view.
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        # .to(): under --gpu-only the decode lands on the GPU intermediate device while the
        # canvas stays on the input device; a no-op when they already match.
        decoded = decoded.to(result.device)
        sub = decoded[:, paste.y0 - crop.y0:paste.y1 - crop.y0, paste.x0 - crop.x0:paste.x1 - crop.x0, :]
        region = result[:, paste.y0:paste.y1, paste.x0:paste.x1, :]
        alpha = sampling.feather_alpha(paste, core, layout.overlap, tile.kept_top,
                                       tile.kept_left).to(result.device)[..., None]
        result[:, paste.y0:paste.y1, paste.x0:paste.x1, :] = alpha * sub + (1.0 - alpha) * region
        if progress is not None:
            progress.advance((index + 1) * W_DECODE_TILE)
    return result


def _refine_canvas(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height,
                   context_anchor, context_overlap, vl_clip, vlm_method=captions.VLM_METHOD_VISION,
                   anchor_source=ANCHOR_SOURCE_IMAGE, batch_size=1, batch_index=0,
                   region_pixel=None, vl_context=None, progress=None):
    # The run loop over ONE canvas — the whole picture on the no-mask path, the bbox crop on
    # the region path (region_pixel/vl_context set). Every tile samples its own full-length
    # schedule on its own thread, held at each sigma step by stepper.py's barrier; between
    # steps the surgery hook consolidates them into one canvas.
    from . import stepper

    # ---- inputs
    run = _prepare_run(image, guider, sigmas, vae, noise, max_tile_width, max_tile_height,
                       context_anchor, context_overlap, vl_clip, vlm_method=vlm_method,
                       anchor_source=anchor_source, batch_size=batch_size, batch_index=batch_index,
                       region_pixel=region_pixel, vl_context=vl_context, progress=progress,
                       sampler=sampler)
    model = guider.model_patcher.model
    steps = int(sigmas.shape[-1]) - 1
    # The sampling budget is EXACT and known here: the stepper's own eval plan, which
    # run_lanes then enforces at every lane's eval — so the ledger and the engine cannot
    # disagree about how long the run is. Sized at stepper intake, never re-derived.
    on_eval = None
    if progress is not None:
        total_evals, _hook_step_at = stepper.plan_evals(sampler, sigmas)
        progress.open(SAMPLING, total_evals * len(run.lanes))

        evals_done = [0]

        def on_eval(evals_done=evals_done, progress=progress):
            # One unit per COMPLETED eval: n_lanes ticks per sigma step, so the bar and
            # the "sampling {pct}%" line walk in ~1% increments instead of whole-step
            # jumps (5% per hook firing at 20 steps, with all 30 tiles' work invisible
            # in between). The hook still lands each step exactly on its target and
            # carries the preview; advance() is monotone, so the two never fight.
            evals_done[0] += 1
            progress.advance(evals_done[0])
    # The maintained canvas follows the LANES' device: comfy moves every lane's noise/latent
    # to the guider's load device inside outer_sample (samplers.py:1254-1255), so `x` reaches
    # the hook there, while the canvas comes off vae.encode on the VAE's output device — the
    # CPU intermediate device in a default install. Blending the two would raise on step 0.
    # A no-op when they already match, which is what keeps the CPU tests bit-identical.
    canvas = run.canvas.to(guider.model_patcher.load_device)
    windows = [lane.window for lane in run.lanes]

    # ---- process
    blends = build_blends(run.layout, canvas)
    # "live canvas" replaces the frozen ring's CONTENT per step; "source image" leaves the ring
    # on the unmodified input and presents it on the lead schedule below. Exactly one of the
    # two is ever active — the schedule exists to bridge a frozen REFINED ring to a noisy core,
    # and in live-canvas mode nothing frozen-refined is left to lead.
    ring_gates = run.ring_gates if anchor_source == ANCHOR_SOURCE_CANVAS else None
    hook = _make_surgery_hook(canvas, blends,
                              make_sync_progress(guider.model_patcher, steps, batch_size, batch_index,
                                                 progress=progress),
                              ring_gates=ring_gates)
    # The shared canvas-wide SDE field, or None for a deterministic sampler — both are
    # rejected the other way round by the stepper, and a per-lane field would put two
    # different noises in every overlap band.
    noise_fields = stepper.build_noise_fields(sampler, canvas.shape, noise.seed, sigmas)
    # ONE instance patch for the WHOLE run, never one per lane: the lanes share the model, and
    # anchor_ring_schedule normalizes against sigmas[0] — which is the run's first sigma here
    # exactly because the schedule handed over is the full one.
    with sampling.anchor_ring_schedule(guider, sigmas, enabled=anchor_source == ANCHOR_SOURCE_IMAGE):
        samples = stepper.run_lanes(run.lanes, sampler, hook, noise_fields=noise_fields, on_eval=on_eval)
    # The hook is PRE-step, so the FINAL step's results are still only in the lanes when the
    # stepper returns. comfy hands each sample back in RAW space (process_latent_out,
    # samplers.py:1238); process_latent_in is the exact inverse it applies on the way in, and
    # puts them back in the canvas's space for one last consolidation pass.
    blend_lanes(canvas, blends, windows, [model.process_latent_in(sample) for sample in samples])

    # ---- output
    if progress is not None:
        progress.open(DECODE, len(run.layout.tiles) * W_DECODE_TILE)
    decoded = decode_composite(vae, model, canvas, run.padded, run.layout, progress=progress)
    return sampling.crop_image_to(decoded, *run.size)


def refine_sync(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height,
                context_anchor, context_overlap, vl_clip, mask=None,
                vlm_method=captions.VLM_METHOD_VISION, anchor_source=ANCHOR_SOURCE_IMAGE,
                batch_size=1, batch_index=0, progress=None):
    """The sync engine: ONE picture in, that picture refined tile-synchronously out.

    Without a mask the whole picture is tiled. With one, only the masked region is refined:
    crop to the mask bbox plus `context_anchor` (the frozen-background halo the subject
    conditions against), tile the crop with every lane's denoise mask gated to the region, and
    composite the refined crop back through a 1px anti-aliased edge — everything the edge
    weights at 0 stays byte-identical to the input, and the mask=0 background survives from the
    ORIGINAL pixels rather than the VAE-decoded crop, so it is never re-diffused.
    NO feather at the mask boundary (CLAUDE.md prime directive 3): an inward feather would
    under-process a ring around the subject and an outward one would re-diffuse finished
    background. The inter-tile feather inside the crop is untouched by this — it is a
    tile-to-tile handover, not a boundary.
    The caller owns the picture loop (batch_size/batch_index name this picture inside it), and
    hands over that picture's own mask row.
    `progress` is the VL run's ledger (progress.py), created by the node and only ever
    RECEIVED here — with None (direct callers, the tests-AB harnesses) every bar and every
    value below is exactly what it was before the ledger existed.
    """
    # ---- inputs
    if mask is None:
        return _refine_canvas(image, guider, sampler, sigmas, vae, noise, max_tile_width,
                              max_tile_height, context_anchor, context_overlap, vl_clip,
                              vlm_method=vlm_method, anchor_source=anchor_source,
                              batch_size=batch_size, batch_index=batch_index, progress=progress)

    # Region path. Harden a soft input at 0.5 — a fractional denoise mask would leave the
    # under-refined halo we reject for turbo (finding-dd-fade-artifacts-turbo).
    mask_bin = mask >= 0.5
    bbox = sampling._mask_bbox(mask_bin)
    if bbox is None:
        # Empty mask: nothing to refine, and (unlike the input) a clone is a safe no-op. RGB-
        # narrowed like every sampled path, so the output channel count never depends on the
        # mask's content.
        return image[..., :3].clone()
    height, width = int(image.shape[1]), int(image.shape[2])
    y0, y1, x0, x1 = sampling._expand_snap_clamp(bbox, context_anchor, height, width)

    # ---- process
    sub_image = image[:, y0:y1, x0:x1, :]
    sub_mask = mask_bin[:, y0:y1, x0:x1].to(image.dtype)
    # The conditioning still reads the WHOLE image: the pre-pass encodes `image` and offsets
    # every region tile's rect by the bbox origin, so a masked refine stays globally informed
    # instead of only seeing its own crop.
    refined_sub = _refine_canvas(sub_image, guider, sampler, sigmas, vae, noise, max_tile_width,
                                 max_tile_height, context_anchor, context_overlap, vl_clip,
                                 vlm_method=vlm_method, anchor_source=anchor_source,
                                 batch_size=batch_size, batch_index=batch_index,
                                 region_pixel=sub_mask, vl_context=(image, x0, y0),
                                 progress=progress)

    # ---- output. Narrow to RGB: the crop comes back 3-channel (a 4-channel input's alpha is
    # dropped exactly as on the no-mask path), so the composite and the output stay 3-channel.
    rgb = image[..., :3]
    out = rgb.clone()
    aa = sampling._aa_alpha(sub_mask)[..., None]
    out[:, y0:y1, x0:x1, :] = aa * refined_sub + (1.0 - aa) * rgb[:, y0:y1, x0:x1, :]
    return out
