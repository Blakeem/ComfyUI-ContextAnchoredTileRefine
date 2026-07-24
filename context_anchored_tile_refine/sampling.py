import torch

from . import grid


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


def feather_alpha(paste_rect, core, overlap, kept_top, kept_left):
    # Directional-feather alpha at NATIVE pixel resolution (no resize): an [h, w]
    # float32 weight over paste_rect (h/w = paste_rect dims), for the post-composite
    # cross-dissolve of this tile into its already-processed top/left neighbor.
    #
    # Two independent per-axis linear ramps, cell-centered. The top overlap band is the
    # rows of paste_rect above the core (present iff kept_top; `overlap` px, or fewer if
    # clamped at the canvas edge); over it ay rises (k+0.5)/band from ~0 at the outer
    # edge to 1 - 0.5/band immediately adjacent to the core, and ay = 1 through the core
    # itself. So the SEAM pixel (the core edge) is 100% this tile — no hard edge shows —
    # while the neighbor fades back in across the band. The left band is analogous on x.
    # alpha = ay (x) ax; the corner is the product of the two ramps.
    #
    # Normalizing by the actual band width (not the nominal `overlap`) keeps the ramp at
    # ~1 adjacent to the core even where the band is clamped narrower than `overlap`.
    # With no kept side (overlap == 0, or a top-left tile) both ramps are all-ones, so
    # the composite reduces to the fe6f6a7 hard core paste, byte-identical.
    h = paste_rect.y1 - paste_rect.y0
    w = paste_rect.x1 - paste_rect.x0
    ay = torch.ones(h, dtype=torch.float32)
    ax = torch.ones(w, dtype=torch.float32)

    top_band = core.y0 - paste_rect.y0  # rows of paste_rect above the core
    if kept_top and overlap > 0:
        ay[:top_band] = (torch.arange(top_band, dtype=torch.float32) + 0.5) / top_band
    left_band = core.x0 - paste_rect.x0  # cols of paste_rect left of the core
    if kept_left and overlap > 0:
        ax[:left_band] = (torch.arange(left_band, dtype=torch.float32) + 0.5) / left_band

    return (ay[:, None] * ax[None, :]).clamp(0.0, 1.0)


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


def _refine_tiles(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, region_pixel=None):
    # Tiled img2img: pad to /8 → solve the grid → encode/sample/decode each tile in
    # raster order from a live canvas → composite by a directional feather → crop back.
    # A 1x1 grid (or overlap=ctx=0) reproduces the feature-2 whole-image path bit for bit.
    # region_pixel (optional [B,H,W] binary float at input pixel size) gates each tile's
    # denoise_mask to the masked region so only it diffuses; the background stays frozen.
    import comfy.model_management

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
            alpha = feather_alpha(paste, core, layout.overlap, tile.kept_top, tile.kept_left).to(canvas.device)[..., None]
            canvas[:, paste.y0:paste.y1, paste.x0:paste.x1, :] = alpha * sub + (1.0 - alpha) * region
        else:
            canvas[:, crop.y0:crop.y1, crop.x0:crop.x1, :] = decoded
    return crop_image_to(canvas, height, width)


def refine_image(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=None):
    # Entry point. Without a mask this is the whole-image / multi-tile refine, byte-for-
    # byte the extracted _refine_tiles body. With a mask: harden (>=0.5) → union bbox over
    # the batch → crop to bbox + context_anchor (frozen-background halo) → tile-refine the
    # crop with every tile's denoise_mask gated to the masked region → composite the
    # refined crop back through a 1px anti-aliased edge. The mask=0 background survives from
    # the ORIGINAL pixels (never the VAE-decoded crop), so it is never re-diffused.
    if sigmas.numel() < 2:
        # Zero steps: nothing to sample, and the lossy VAE roundtrip would degrade the image.
        return image.clone()
    if mask is None:
        return _refine_tiles(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, region_pixel=None)

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
    refined_sub = _refine_tiles(sub_image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, region_pixel=sub_mask)
    # Composite the refined crop back through the anti-aliased mask edge. Outside the crop,
    # and wherever aa == 0 inside it, the output is the byte-identical original image.
    # Narrow to RGB: _refine_tiles decodes a 3-channel refined_sub (as the no-mask path does,
    # dropping a 4-channel input's alpha), so the composite and the output stay 3-channel.
    rgb = image[..., :3]
    out = rgb.clone()
    aa = _aa_alpha(sub_mask)[..., None]
    out[:, y0:y1, x0:x1, :] = aa * refined_sub + (1.0 - aa) * rgb[:, y0:y1, x0:x1, :]
    return out
