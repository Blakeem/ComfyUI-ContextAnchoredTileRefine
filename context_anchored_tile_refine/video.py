"""Tiled MiniMax H3 video refine: the video analogue of sampling._refine_tiles.

`refine_video` takes an ALREADY-UPSCALED clip and refines it tile by tile on a /32 grid
(H3's VAE spatial factor 16 x the DiT's patch 2), reusing the image path's settled seam
design unchanged: the frozen context_anchor ring encodes from the LIVE canvas, the
context_overlap bands diffuse from the FROZEN RAW source in both neighbours, noise is
drawn ONCE for the canvas and sliced per tile, and the composite is DC match -> min-error
seam routing -> directional feather.

What is video-specific here, and why:
- Latents are NestedTensor pairs (video [1,24,T,H/16,W/16], audio [1,32,2,Ta]). Tiling is
  SPATIAL only — every tile samples the whole clip at once, so a seam is a single decision
  for all frames and cannot flicker.
- The audio stream rides along FROZEN: the nested denoise mask is (video mask,
  zeros_like(audio)). A stream with no mask defaults to ones (comfy/samplers.py:1304-1305),
  which would regenerate the audio once per tile.
- Frame counts live on the model's 17k+5 grid, so an off-grid clip is repeat-padded before
  tiling and the output trimmed back to the caller's frame count.
- Canvases are fp16 and the blend runs fp32 in frame chunks: two fp32 full-clip canvases do
  not fit beside the DiT's CPU-side staging on a 32 GB machine, and fp16's step is below the
  8-bit output floor, so this is a memory decision, not a quality one
  (tests-AB/spike_h3_stress.py:426-430).
- Conditioning is SLICED here, never encoded: `vl_pack` arrives already encoded because the
  32B text encoder must never be resident while the full-clip canvases exist (two spike runs
  died at exactly that bracket).

Module scope is torch-only; comfy is imported lazily inside functions (the same contract as
sampling.py / vl.py, pinned by a subprocess test).
"""
import torch

from . import grid, sampling, upscale, vl_video

# Model constants mirrored from comfy_extras/nodes_minimax_h3.py rather than imported: that
# module pulls in the whole comfy_api node stack, and this module's scope stays torch-only.
FPS = 24                        # :29
AUDIO_LATENT_FPS = 40           # :30
FRAME_GRID_PERIOD = 17          # frame counts satisfy n % 17 == 5 (:33-36 align_frame_count)
FRAME_GRID_PHASE = 5
VIDEO_LATENT_CHANNELS = 24      # :72
AUDIO_LATENT_CHANNELS = 32      # :74
AUDIO_LATENT_ROWS = 2           # :74
VAE_SPATIAL = 16                # video latent cell in px (:72 height // 16)
CANVAS_MULTIPLE = 32            # :25 — VAE 16 x DiT patch 2; every sampled crop lands on it

# Frames per fp32 blend chunk. Per-frame blend math is independent, so chunking is
# bit-identical; it only caps the fp32 transient, which at full clip size is several GB
# (tests-AB/spike_h3_stress.py:505-513).
BLEND_FRAME_CHUNK = 16

# Frame stride for the DC ESTIMATE only. A 32px band over 124 frames pools ~10.7M rows
# (~170 MB fp32 plus the median's sort workspace); every 4th frame keeps ~2.7M, still four
# orders of magnitude above sampling.DC_MATCH_MIN_SAMPLES. The offset is a global statistic
# of two tiles' disagreement over the same raw pixels, so dropping frames costs a little
# precision and introduces no bias. The seam routing and the composite see every frame.
DC_FRAME_STRIDE = 4


def align_frame_count(length):
    # comfy_extras/nodes_minimax_h3.py:33-36 align_frame_count, floored at 5 the way
    # temporal_shape:44 floors it: the next frame count on the model's 17k+5 grid.
    count = max(FRAME_GRID_PHASE, int(length))
    while count % FRAME_GRID_PERIOD != FRAME_GRID_PHASE:
        count += 1
    return count


def video_latent_t(frame_count):
    # comfy_extras/nodes_minimax_h3.py:39-40: the VAE's temporal compression, which is what
    # the canvas noise draw and the per-tile shape guard are built on.
    if frame_count <= FRAME_GRID_PHASE:
        return 2
    return ((frame_count - FRAME_GRID_PHASE) // FRAME_GRID_PERIOD) * 5 + 2


def audio_latent_length(frame_count):
    # comfy_extras/nodes_minimax_h3.py:43-46 temporal_shape: round(duration *
    # AUDIO_LATENT_FPS) with duration = frame_count / FPS.
    return round(frame_count / FPS * AUDIO_LATENT_FPS)


def pad_frames_to_grid(frames):
    # Returns (frames on the 17k+5 grid, that frame count). The pad repeats the LAST frame:
    # the padded tail is trimmed off the output, and a repeated frame keeps the tail static
    # rather than inventing motion the model would then refine into the frames we keep.
    # An on-grid clip comes back untouched (no copy).
    count = int(frames.shape[0])
    target = align_frame_count(count)
    if target == count:
        return frames, count
    tail = frames[-1:].expand(target - count, *frames.shape[1:])
    return torch.cat([frames, tail], dim=0), target


def _nested(video_tensor, audio_tensor):
    # The (video, audio) pair every H3 sampling object is: latent, noise and denoise mask.
    import comfy.nested_tensor

    return comfy.nested_tensor.NestedTensor((video_tensor, audio_tensor))


def audio_stream(audio_latent, frame_count):
    # The frozen audio context every tile samples against. Unconnected, a silent zero latent
    # stands in. Connected, its length must match the frame count on the grid: the H3 sampler
    # emits round(frames / 24 * 40) audio rows, so a mismatch means the clip was trimmed (or
    # repeat-padded) after the latent was made, and the two streams no longer describe the
    # same footage. Fail fast rather than sample against a misaligned soundtrack.
    import comfy.model_management

    expected = audio_latent_length(frame_count)
    if audio_latent is None:
        # intermediate_device(), because that is where BOTH a real H3 AV latent
        # (comfy_extras/nodes_minimax_h3.py _empty_av_latent) and vae.encode's output
        # (comfy/sd.py:1027 VAE.output_device) land — the GPU under --gpu-only
        # (comfy/model_management.py:1227-1231). A CPU-pinned stream would straddle devices
        # against the encoded video stream; see the pairing in the tile loop.
        return torch.zeros([1, AUDIO_LATENT_CHANNELS, AUDIO_LATENT_ROWS, expected],
                           device=comfy.model_management.intermediate_device())
    length = int(audio_latent.shape[-1])
    if length != expected:
        raise ValueError(
            f"AV latent audio length {length} does not match this clip: {frame_count} frames need {expected} audio "
            f"latent rows (round(frames / {FPS} * {AUDIO_LATENT_FPS})). Connect the AV latent from the same H3 "
            "sampler run as the frames.")
    return audio_latent


def canvas_noise(seed, latent_t, canvas_h, canvas_w, audio):
    # ONE draw at canvas latent size, sliced per tile below — the production convention, so
    # two tiles sharing an overlap band see correlated noise instead of independent redraws.
    # prepare_noise reads only size/dtype/layout, so a zeros dummy is enough, and the nested
    # dummy makes it draw both streams from one generator exactly as a real H3 run does
    # (comfy/sample.py:29-34).
    # The dummy pair is built on ONE device (prepare_noise draws on the CPU either way,
    # comfy/sample.py:11), so it never mirrors a caller's off-device audio latent.
    video_dummy = torch.zeros([1, VIDEO_LATENT_CHANNELS, latent_t,
                               canvas_h // VAE_SPATIAL, canvas_w // VAE_SPATIAL])
    audio_dummy = torch.zeros_like(audio, device=video_dummy.device)
    dummy = {"samples": _nested(video_dummy, audio_dummy)}
    return upscale.Noise_RandomNoise(seed).generate_noise(dummy).unbind()


def tile_denoise_mask(tile, latent_t, audio):
    # Binary video mask broadcast over the tile's latent frames, paired with an all-zero
    # audio mask. tile_gradient at scale 16 gives the hard indicator of overlap_inner_rect
    # (core + context_overlap diffuse at full strength; the context_anchor ring is frozen at
    # 0, which is what carries the neighbour's already-refined pixels as conditioning).
    # The nested pair goes to guider.sample AS IS — sampling.sample_latent's normalization is
    # for a single tensor only. Core unbinds a nested mask and runs prepare_mask per stream
    # against that stream's own latent shape (comfy/samplers.py:1297-1314); the single-channel
    # [1,1,T,h,w] handed over here is what survives that prep value-unchanged, exactly as
    # [B,1,h,w] does on the image path.
    gradient = sampling.tile_gradient(tile.crop_rect, tile.overlap_inner_rect, scale=VAE_SPATIAL)
    video_mask = gradient[None, None, None].expand(1, 1, latent_t, *gradient.shape).contiguous()
    return _nested(video_mask, torch.zeros_like(audio))


def tile_encode_input(raw, live, tile):
    # The tile's encode source, split by band exactly as sampling._refine_tiles splits it:
    # the DIFFUSED region (overlap_inner_rect, denoise mask 1) comes from the FROZEN RAW
    # canvas so the feather cross-dissolves two INDEPENDENT refinements of the same raw
    # pixels, never a serial re-diffusion of a neighbour's decoded output; the frozen
    # context_anchor ring around it stays on the LIVE canvas, carrying the already-refined
    # neighbours into this tile's conditioning.
    crop, inner = tile.crop_rect, tile.overlap_inner_rect
    enc = live[:, crop.y0:crop.y1, crop.x0:crop.x1, :].clone()
    enc[:, inner.y0 - crop.y0:inner.y1 - crop.y0, inner.x0 - crop.x0:inner.x1 - crop.x0, :] = (
        raw[:, inner.y0:inner.y1, inner.x0:inner.x1, :])
    return enc


def apply_dc_offset(sub, dc_offset):
    # sampling._refine_tiles' DC correction, formula for formula, but applied IN PLACE to the
    # tile's own decode — never to the canvas — and frame-chunked. The image path builds a
    # corrected copy; at clip scale that is a second full-tile allocation (~0.9 GB fp16 at
    # 124 frames), and the tile decode is a private buffer, so writing through it is free.
    # The gamut gate is verbatim: a pixel already outside [0,1] came from a VAE whose
    # process_output is the identity, and flattening it would be a lossy op on the tile.
    for f0 in range(0, sub.shape[0], BLEND_FRAME_CHUNK):
        chunk = sub[f0:f0 + BLEND_FRAME_CHUNK]
        corrected = chunk - dc_offset
        in_gamut = (chunk >= 0.0) & (chunk <= 1.0)
        chunk.copy_(torch.where(in_gamut, corrected.clamp(0.0, 1.0), corrected))


def blend_into_canvas(live, paste, sub, alpha):
    # The directional-feather composite over paste_rect, in fp32 per frame chunk and stored
    # back fp16 — one rounding at the end, below the 8-bit step. Per-frame math is
    # independent of every other frame, so chunking is bit-identical to a whole-clip blend.
    blend = alpha.view(1, alpha.shape[0], alpha.shape[1], 1)
    for f0 in range(0, sub.shape[0], BLEND_FRAME_CHUNK):
        f1 = min(f0 + BLEND_FRAME_CHUNK, sub.shape[0])
        region = live[f0:f1, paste.y0:paste.y1, paste.x0:paste.x1, :]
        live[f0:f1, paste.y0:paste.y1, paste.x0:paste.x1, :] = (
            sub[f0:f1].float() * blend + region.float() * (1.0 - blend)).to(live.dtype)


def refine_video(frames, guider, sampler, sigmas, vae, seed, max_tile_width, max_tile_height,
                 context_anchor, context_overlap, vl_pack, audio_latent=None):
    # Entry point: pad the clip onto the frame grid and the canvas to /32 -> solve the tile
    # grid -> encode/sample/decode each tile from a live canvas in raster order -> DC-match
    # onto the already-placed neighbours -> composite by the directional feather -> crop and
    # trim back. `vl_pack` is vl_video.encode_global's ALREADY-COMPUTED result; this function
    # only slices it, and never touches a text encoder.
    import comfy.model_management
    import comfy.utils

    if sigmas.numel() < 2:
        # Zero steps: nothing to sample, and the lossy VAE roundtrip would degrade the clip.
        return frames.clone()

    # fp16 before either pad, so the pad copies (a full clip each) are never fp32.
    pixels = frames[..., :3].to(torch.float16)
    frame_count = int(pixels.shape[0])
    on_grid, grid_frames = pad_frames_to_grid(pixels)
    raw, (height, width) = sampling.pad_image_to_multiple(on_grid, CANVAS_MULTIPLE)
    canvas_h, canvas_w = int(raw.shape[1]), int(raw.shape[2])
    latent_t = video_latent_t(grid_frames)
    audio = audio_stream(audio_latent, grid_frames)

    sx = grid.solve_axis(canvas_w, max_tile_width, context_anchor, context_overlap, multiple=CANVAS_MULTIPLE)
    sy = grid.solve_axis(canvas_h, max_tile_height, context_anchor, context_overlap, multiple=CANVAS_MULTIPLE)
    layout = grid.build_layout(canvas_w, canvas_h, sx, sy, context_anchor, context_overlap)
    # Row slices of the one whole-clip encode: every tile reads the same globally-informed
    # picture, so structures crossing a seam continue (see vl_video.py).
    tile_positives = [vl_video.slice_rows(vl_pack, tile.crop_rect, canvas_h, canvas_w)
                      for tile in layout.tiles]

    video_noise, audio_noise = canvas_noise(seed, latent_t, canvas_h, canvas_w, audio)
    steps = sigmas.shape[-1] - 1
    for_tile = sampling.make_tile_progress(guider.model_patcher, steps, len(layout.tiles))
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    # Live canvas: tiles paste into it in raster order, so a later tile's frozen anchor ring
    # encodes its already-refined neighbours. `raw` is never written.
    live = raw.clone()

    for tile_idx, tile in enumerate(layout.tiles):
        comfy.model_management.throw_exception_if_processing_interrupted()
        crop = tile.crop_rect
        tile_latent = vae.encode(tile_encode_input(raw, live, tile))
        # Spatially anchored noise slice: overlapping crops agree on the noise their shared
        # band sees. .contiguous() because downstream sampler code may .view() the slice.
        tile_noise = video_noise[..., crop.y0 // VAE_SPATIAL:crop.y1 // VAE_SPATIAL,
                                 crop.x0 // VAE_SPATIAL:crop.x1 // VAE_SPATIAL].contiguous()
        if tile_latent.shape != tile_noise.shape:
            raise RuntimeError(
                f"VAE encoded a {crop.x1 - crop.x0}x{crop.y1 - crop.y0} px x {grid_frames} frame tile to latent shape {tuple(tile_latent.shape)}, but the tile's "
                f"noise slice is {tuple(tile_noise.shape)}. This VAE's latent layout is not the MiniMax H3 one "
                f"(video [1,{VIDEO_LATENT_CHANNELS},T,H/{VAE_SPATIAL},W/{VAE_SPATIAL}] on the {FRAME_GRID_PERIOD}k+{FRAME_GRID_PHASE} frame grid).")
        # The audio latent and its noise are COPIED per tile: they are the sampler's input
        # buffers, and one shared tensor would let a sampler's in-place write leak from tile
        # to tile (and back into the caller's AV latent). The latent's copy lands on the
        # ENCODE's device: guider.sample cats the nested streams (comfy/samplers.py:1281-1283
        # -> comfy/utils.py:1374 pack_latents) BEFORE outer_sample normalizes their device
        # (comfy/samplers.py:1254-1255), so a caller's off-device AV latent would raise there.
        latent = _nested(tile_latent, audio.to(device=tile_latent.device, copy=True))
        noise = _nested(tile_noise, audio_noise.clone())
        denoise_mask = tile_denoise_mask(tile, latent_t, audio)

        # Per-tile positive swap. Guider_Basic keeps `positive` in original_conds and
        # re-copies that map on every sample() call, so the swap takes effect per tile; the
        # pristine map — the SAME object the caller handed in — goes back in `finally`, so an
        # interrupt or a sampler raise cannot leave the guider swapped.
        pristine_conds = guider.original_conds
        swapped_conds = dict(pristine_conds)
        swapped_conds["positive"] = tile_positives[tile_idx]
        guider.original_conds = swapped_conds
        try:
            samples = guider.sample(noise, latent, sampler, sigmas, denoise_mask=denoise_mask,
                                    callback=for_tile(tile_idx), disable_pbar=disable_pbar,
                                    seed=seed)
        finally:
            guider.original_conds = pristine_conds

        # SamplerCustomAdvanced moves the sampler output to the intermediate device; only the
        # video stream survives here (the audio rode along frozen), so only it is moved.
        video_samples = samples.unbind()[0].to(comfy.model_management.intermediate_device())
        decoded = vae.decode(video_samples)
        # A video VAE decodes to [B,T,H,W,C]; fold T into the batch exactly as core's
        # VAEDecode does, which for B=1 gives this pipeline's [frames,H,W,C] contract.
        if decoded.ndim == 5:
            decoded = decoded.reshape(-1, decoded.shape[-3], decoded.shape[-2], decoded.shape[-1])
        decoded = decoded.to(device=live.device, dtype=live.dtype)

        paste = tile.paste_rect
        sub = decoded[:, paste.y0 - crop.y0:paste.y1 - crop.y0,
                      paste.x0 - crop.x0:paste.x1 - crop.x0, :]
        region = live[:, paste.y0:paste.y1, paste.x0:paste.x1, :]
        # DC match BEFORE the routing and the composite, for the image path's two reasons: a
        # per-tile constant applied afterwards would re-introduce a hard step at the tile
        # boundary, and removing it first leaves the error surface describing TEXTURE
        # disagreement instead of a constant pedestal the DP would tie-break on.
        dc_offset = sampling.seam_dc_offset(tile, sub[::DC_FRAME_STRIDE], region[::DC_FRAME_STRIDE])
        if dc_offset is not None:
            apply_dc_offset(sub, dc_offset)
        disp_top, disp_left = sampling.seam_displacements(tile, sub, region)
        alpha = sampling.feather_alpha(paste, tile.core, layout.overlap, tile.kept_top,
                                       tile.kept_left, disp_top=disp_top,
                                       disp_left=disp_left).to(live.device)
        blend_into_canvas(live, paste, sub, alpha)
        del decoded, sub    # ~0.9 GB at clip scale, freed before the next tile encodes

    del raw                 # last read was tile_encode_input, inside the loop
    refined = sampling.crop_image_to(live, height, width)[:frame_count]
    return refined.to(torch.float32)
