
# Mirrors core nodes.MAX_RESOLUTION (importing it would pull comfy in at module scope).
MAX_RESOLUTION = 16384


def _validate_image(image):
    if image.ndim != 4:
        raise ValueError(f"image must be a [B,H,W,C] IMAGE tensor, got {image.ndim} dimensions")
    if image.shape[1] < 8 or image.shape[2] < 8:
        raise ValueError(f"image must be at least 8x8 pixels, got {image.shape[1]}x{image.shape[2]}")


def _unpack_av_latent(av_latent):
    # The audio stream of a MiniMax H3 AV LATENT, duck-typed the way conds.py duck-types
    # control objects (comfy is never imported to check a type). The video stream is
    # discarded: this node re-encodes the video from the upscaled pixels, and only the
    # soundtrack rides along frozen as sampling context.
    if av_latent is None:
        return None

    samples = av_latent.get("samples") if isinstance(av_latent, dict) else None
    unbind = getattr(samples, "unbind", None)
    if unbind is None:
        raise ValueError("av_latent must be the MiniMax H3 sampler's AV LATENT — a LATENT whose samples are a nested (video, audio) pair. Connect the sampler's LATENT output, or leave this input unconnected.")
    streams = list(unbind())
    if len(streams) != 2:
        raise ValueError(f"av_latent holds {len(streams)} latent stream(s); the MiniMax H3 sampler's AV LATENT holds exactly 2 (video, audio). Connect that output, or leave this input unconnected.")
    # .cpu() because the frozen soundtrack is only ever re-uploaded per tile (video.py pairs
    # it onto the tile encode's device): parking it in VRAM would hold ~megabytes hostage for
    # the whole run beside the canvases.
    return streams[1].cpu()


def _normalize_mask(mask, image):
    # Normalize a 2D [H,W] mask to [1,H,W]; reject any other rank.
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if mask.ndim != 3:
        raise ValueError(f"mask must be a [H,W] or [B,H,W] MASK tensor, got {mask.ndim} dimensions")
    # Strict spatial match (no resample — a resized mask would misalign the region).
    if mask.shape[1] != image.shape[1] or mask.shape[2] != image.shape[2]:
        raise ValueError(f"mask size {mask.shape[1]}x{mask.shape[2]} must match image size {image.shape[1]}x{image.shape[2]}")
    # Batch must be 1 (broadcast to every image) or exactly the image batch.
    if mask.shape[0] not in (1, image.shape[0]):
        raise ValueError(f"mask batch {mask.shape[0]} must be 1 or match image batch {image.shape[0]}")
    if mask.shape[0] == 1 and image.shape[0] != 1:
        mask = mask.expand(image.shape[0], -1, -1)
    return mask


class ContextAnchoredTileRefine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The upscaled image to refine tile by tile."}),
                "guider": ("GUIDER", {"tooltip": "The guider (e.g. from CFGGuider) that denoises each tile."}),
                "sampler": ("SAMPLER", {"tooltip": "The sampler used to denoise each tile."}),
                "sigmas": ("SIGMAS", {"tooltip": "The sigma schedule used when sampling each tile."}),
                "vae": ("VAE", {"tooltip": "The VAE used to encode each tile for sampling and decode the result."}),
                "noise": ("NOISE", {"tooltip": "The noise source (e.g. from RandomNoise) used when sampling each tile."}),
                "max_tile_width": ("INT", {"default": 1024, "min": 256, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Hard cap on the width the model ever sees per sampled crop, including the context_overlap and context_anchor rings. Set to the largest width the model supports."}),
                "max_tile_height": ("INT", {"default": 1024, "min": 256, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Hard cap on the height the model ever sees per sampled crop, including the context_overlap and context_anchor rings. Set to the largest height the model supports."}),
                "context_anchor": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Width of the fully-frozen, visible-only context ring sampled beyond the overlap on every edge (including up against a mask) then cropped away. With a mask it also sets the frozen-background halo the masked region conditions against — keep it > 0. Always additive: the ring outside a tile core is context_overlap + context_anchor. 32-256 is the useful range; 32 suits most scenes."}),
                "context_overlap": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Inter-tile directional feather width (multi-tile runs only; never applied at a mask boundary). Each tile is sampled oversized; on sides bordering an already-processed neighbor (top/left) this band is fully diffused and feathered into that neighbor (100% at the seam → 0% over the band); elsewhere it's context, then cropped. 32-256 is the useful range: DETAILED scenes need LESS (32 is invisible even when you know where the seam is) because the seam has texture to hide in and to route through, while a large smooth gradient (an open night sky) is the hard case and wants 128+. 0 = hard seams."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional region mask: only the masked region is refined (hardened at 0.5), cropped to the mask plus context_anchor, with a 1px anti-aliased edge; the unmasked background is left untouched. Feed an inverted mask for a second pass."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "refine"
    CATEGORY = "image/upscaling"

    @classmethod
    def VALIDATE_INPUTS(s, max_tile_width=None, max_tile_height=None, context_anchor=None, context_overlap=None):
        # Naming widgets here disables ComfyUI's default min/max validation for them,
        # so ranges are re-checked alongside the /8 constraint (widget step is UI-only;
        # API-submitted workflows can send arbitrary INTs, and all derived geometry
        # must stay on the /8 grid the VAE requires).
        checks = (
            ("max_tile_width", max_tile_width, 256, MAX_RESOLUTION),
            ("max_tile_height", max_tile_height, 256, MAX_RESOLUTION),
            ("context_anchor", context_anchor, 0, 512),
            ("context_overlap", context_overlap, 0, 512),
        )
        for name, value, minimum, maximum in checks:
            if value is None:
                continue
            if value % 8 != 0:
                return f"{name} must be a multiple of 8, got {value}"
            if value < minimum or value > maximum:
                return f"{name} must be between {minimum} and {maximum}, got {value}"
        return True

    def refine(self, image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=None):
        _validate_image(image)
        if mask is not None:
            mask = _normalize_mask(mask, image)
        from . import sampling
        return (sampling.refine_image(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=mask),)


class ContextAnchoredTileRefineVL(ContextAnchoredTileRefine):
    """The vision-conditioned variant for VLM-encoder models (Krea 2 family).

    Same tiling engine as the base node, but every tile's positive conditioning is
    replaced by its slice of ONE whole-image vision encode from the required CLIP:
    positionally exact, free of text demands, and globally informed, so tiles neither
    re-instantiate prompt objects they don't contain nor drift apart in story
    (gaze, tone, palette). The guider's positive text is ignored by construction;
    its negative still applies. No prompt input exists because none is needed.
    ControlNet is ignored on this node (the per-tile positive carries no control chain);
    use the base Context-Anchored Tile Refine node for control.
    With a mask, the WHOLE image is still encoded and the region's tiles slice their
    true place in it, so a masked refine stays aware of the image around the region.
    """

    @classmethod
    def INPUT_TYPES(s):
        input_types = ContextAnchoredTileRefine.INPUT_TYPES()
        input_types["required"]["clip"] = ("CLIP", {"tooltip": "The workflow's CLIP — must be a vision-language text encoder (Krea 2 family). The whole image is encoded once through its vision path and each tile's positive conditioning becomes its slice of that encode; the guider's positive prompt is ignored, the negative still applies."})
        return input_types

    def refine(self, image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, clip, mask=None):
        _validate_image(image)
        if mask is not None:
            mask = _normalize_mask(mask, image)
        from . import sampling
        return (sampling.refine_image(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=mask, vl_clip=clip),)


class ContextAnchoredTileUpscaleVL(ContextAnchoredTileRefine):
    """Upscale + VL tile refine in one node: image in, refined image out.

    Same engine as ContextAnchoredTileRefineVL, with the four custom-sampling inputs
    (NOISE / SAMPLER / SIGMAS / GUIDER) built in-process from widgets by upscale.py, and
    the whole-image upscale stage run first so no tile is ever resampled. No mask input
    (use ContextAnchoredTileRefineVL for a region pass) and no positive prompt input —
    the positive is a placeholder that every tile's vision slice replaces, so a prompt
    here would only re-admit the phantom objects the VL path exists to remove. The
    optional negative is the one text channel that still applies.
    """

    @classmethod
    def INPUT_TYPES(s):
        # Lazy import: node.py's module scope stays comfy-free (pinned by a subprocess test).
        import comfy.samplers

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image to upscale and then refine tile by tile."}),
                "model": ("MODEL", {"tooltip": "The diffusion model used to denoise each tile."}),
                "clip": ("CLIP", {"tooltip": "The workflow's CLIP — must be a vision-language text encoder (Krea 2 family). The whole upscaled image is encoded once through its vision path and each tile's positive conditioning becomes its slice of that encode."}),
                "vae": ("VAE", {"tooltip": "The VAE used to encode each tile for sampling and decode the result."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Seed for the noise. Noise is drawn once for the whole image and sliced per tile, so a tile's noise does not change when the grid does."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m", "tooltip": "The sampler used to denoise each tile."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "sgm_uniform", "tooltip": "The sigma schedule used when sampling each tile."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Sampling steps per tile. As in BasicScheduler, the schedule is built for steps/denoise steps and only the last steps+1 sigmas are used."}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "Classifier-free guidance scale applied against the negative conditioning."}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much of the schedule to run per tile: 1.0 rewrites the tile, lower values keep more of the upscaled pixels. 0.0 skips diffusion entirely and returns the upscale alone."}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 8.0, "step": 0.01, "tooltip": "Target scale for the whole-image upscale stage that runs before any tiling. The optional upscale_model runs first and this factor sets the final size; 1.0 with no model leaves the input pixels untouched."}),
                "max_tile_width": ("INT", {"default": 1536, "min": 256, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Hard cap on the width the model ever sees per sampled crop, including the context_overlap and context_anchor rings. Set to the largest width the model supports."}),
                "max_tile_height": ("INT", {"default": 2048, "min": 256, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Hard cap on the height the model ever sees per sampled crop, including the context_overlap and context_anchor rings. Set to the largest height the model supports."}),
                "context_anchor": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Width of the fully-frozen, visible-only context ring sampled beyond the overlap on every edge then cropped away. Always additive: the ring outside a tile core is context_overlap + context_anchor. 32-256 is the useful range; 32 suits most scenes."}),
                "context_overlap": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Inter-tile directional feather width (multi-tile runs only). Each tile is sampled oversized; on sides bordering an already-processed neighbor (top/left) this band is fully diffused and feathered into that neighbor (100% at the seam → 0% over the band); elsewhere it's context, then cropped. 32-256 is the useful range: DETAILED scenes need LESS (32 is invisible even when you know where the seam is), while a large smooth gradient (an open night sky) wants 128+. 0 = hard seams."}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Optional upscale model run over the whole image before tiling. Its fixed integer scale rarely lands on upscale_by, so a single lanczos pass then takes the result to the exact target."}),
                "negative": ("CONDITIONING", {"tooltip": "Optional negative conditioning. Left unconnected it is an empty encode of this node's CLIP, which is what cfg needs to be meaningful at all."}),
            },
        }

    def refine(self, image, model, clip, vae, seed, sampler_name, scheduler, steps, cfg, denoise, upscale_by, max_tile_width, max_tile_height, context_anchor, context_overlap, upscale_model=None, negative=None):
        # Lazy import: node.py's module scope stays comfy-free (pinned by a subprocess test).
        import comfy.samplers

        from . import sampling, upscale

        empty_cond = None
        upscaled = None
        guider = None
        sigmas = None
        sampler = None
        noise = None

        _validate_image(image)

        # Everything that resamples happens here, on the whole image, before any tiling.
        upscaled = upscale.prepare_upscaled(image, upscale_model, upscale_by)
        # The upscale REPLACES the validated input, and upscale_by goes down to 0.01, so a
        # small image can leave here below 8px on an axis — where the /8 reflect pad (which
        # needs pad < dim) would raise naming neither this node nor the widget that did it.
        if upscaled.shape[1] < 8 or upscaled.shape[2] < 8:
            raise ValueError(f"upscale_by {upscale_by} takes the {image.shape[1]}x{image.shape[2]} input to {upscaled.shape[1]}x{upscaled.shape[2]}; the upscaled image must be at least 8x8 pixels")

        # One empty encode serves as the positive placeholder (vl.py replaces every tile's
        # positive with its slice of the whole-image vision encode) and, unless the optional
        # input is connected, as the negative.
        empty_cond = upscale.encode_empty(clip)
        guider = upscale.build_guider(model, empty_cond, empty_cond if negative is None else negative, cfg)
        sigmas = upscale.build_sigmas(model, scheduler, steps, denoise)
        sampler = comfy.samplers.sampler_object(sampler_name)
        noise = upscale.Noise_RandomNoise(seed)

        return (sampling.refine_image(upscaled, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=None, vl_clip=clip),)


class ContextAnchoredTileUpscaleVLVideo(ContextAnchoredTileRefine):
    """Upscale + VL tile refine for MiniMax H3 video: frames in, refined frames out.

    ContextAnchoredTileUpscaleVL's design carried onto H3's audio-video latents. Tiling is
    SPATIAL only — every tile refines the whole clip at once, so a seam is one decision for
    all frames and cannot flicker — and the audio stream rides along frozen so a refine pass
    never rewrites the soundtrack. The conditioning is one 2 fps vision encode of the whole
    upscaled clip that each tile slices its own rows out of: globally informed, positionally
    exact, and demand-free, which is why there is no prompt input (text re-admits the phantom
    objects the VL path exists to remove) and no cfg or negative (H3 is CFG-free).

    Stage order inside refine() is load-bearing, not incidental: the 32B text encoder runs
    on ~11 upscaled 2 fps picks and is gone BEFORE the full-clip canvases are allocated.
    """

    @classmethod
    def INPUT_TYPES(s):
        # Lazy import: node.py's module scope stays comfy-free (pinned by a subprocess test).
        import comfy.samplers

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The clip to upscale and then refine, as the frames of ONE video at H3's native 24 fps (from VAE Decode, or Get Video Components). Frame counts off the model's 17k+5 grid are repeat-padded internally and trimmed back, so the output frame count always matches this input."}),
                "model": ("MODEL", {"tooltip": "The MiniMax H3 diffusion model used to denoise each tile."}),
                "clip": ("CLIP", {"tooltip": "The workflow's CLIP — must be the MiniMax H3 text encoder. The whole upscaled clip is sampled at 2 fps and encoded ONCE through its vision path, and each tile's positive conditioning becomes its slice of that encode. There is no prompt input: the encode runs on an empty prompt because any text re-admits phantom objects."}),
                "vae": ("VAE", {"tooltip": "The MiniMax H3 VAE used to encode each tile for sampling and decode the result."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Seed for the noise. Noise is drawn once for the whole canvas and sliced per tile, so a tile's noise does not change when the grid does."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep", "tooltip": "The sampler used to denoise each tile. res_multistep is the A/B-settled default for H3."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple", "tooltip": "The sigma schedule used when sampling each tile."}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 10000, "tooltip": "Sampling steps per tile. As in BasicScheduler, the schedule is built for steps/denoise steps and only the last steps+1 sigmas are used. 8 is settled independently of denoise: fewer steps inside the window visibly under-integrate at low denoise."}),
                "denoise": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much of the schedule to run per tile: higher values rewrite more of the clip, lower values keep more of the upscaled pixels. 0.22 is the owner-approved working point. 0.0 skips diffusion entirely and returns the upscale alone."}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 8.0, "step": 0.01, "tooltip": "Target scale for the whole-clip upscale stage that runs before any tiling. The optional upscale_model runs first and this factor sets the final size; 1.0 with no model leaves the input pixels untouched."}),
                "max_tile_width": ("INT", {"default": 1408, "min": 256, "max": MAX_RESOLUTION, "step": 32, "tooltip": "Hard cap on the width the model ever sees per sampled crop, including the context_overlap and context_anchor rings. A crop this wide holds the WHOLE clip, so this is a VRAM ceiling, not a resolution one: 1408x1408 crops of a 124-frame clip are the validated ceiling on a 24 GB card."}),
                "max_tile_height": ("INT", {"default": 1408, "min": 256, "max": MAX_RESOLUTION, "step": 32, "tooltip": "Hard cap on the height the model ever sees per sampled crop, including the context_overlap and context_anchor rings. A crop this tall holds the WHOLE clip, so this is a VRAM ceiling, not a resolution one: 1408x1408 crops of a 124-frame clip are the validated ceiling on a 24 GB card."}),
                "context_anchor": ("INT", {"default": 32, "min": 0, "max": 512, "step": 32, "tooltip": "Width of the fully-frozen, visible-only context ring sampled beyond the overlap on every edge then cropped away. It carries the already-refined neighbours into this tile's conditioning. Always additive: the ring outside a tile core is context_overlap + context_anchor."}),
                "context_overlap": ("INT", {"default": 32, "min": 0, "max": 512, "step": 32, "tooltip": "Inter-tile directional feather width (multi-tile runs only). Each tile is sampled oversized; on sides bordering an already-processed neighbour (top/left) this band is fully diffused from the same raw pixels by both tiles and feathered into that neighbour, elsewhere it is context, then cropped. Detailed scenes need LESS; large smooth gradients want more. 0 = hard seams."}),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Optional upscale model run over the whole clip before tiling. Its fixed integer scale rarely lands on upscale_by, so a single lanczos pass then takes the result to the exact target."}),
                "av_latent": ("LATENT", {"tooltip": "The MiniMax H3 sampler's AV LATENT for THIS clip. Connected, its audio stream is the frozen soundtrack every tile samples against — the configuration every full-video test validated, and recommended whenever the frames came from the H3 sampler. Unconnected, a silent zero audio latent stands in: clean in the single-frame runs, but never validated at full video scale. Its length must match the frame count."}),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(s, max_tile_width=None, max_tile_height=None, context_anchor=None, context_overlap=None):
        # Same reasoning as the image node's /8 check (widget steps are UI-only, an
        # API-submitted workflow can send arbitrary INTs), on H3's coarser grid: every
        # sampled crop must be a multiple of 32 (VAE spatial factor 16 x DiT patch 2), and
        # each of these four widths feeds a crop dimension directly.
        checks = (
            ("max_tile_width", max_tile_width, 256, MAX_RESOLUTION),
            ("max_tile_height", max_tile_height, 256, MAX_RESOLUTION),
            ("context_anchor", context_anchor, 0, 512),
            ("context_overlap", context_overlap, 0, 512),
        )
        for name, value, minimum, maximum in checks:
            if value is None:
                continue
            if value % 32 != 0:
                return f"{name} must be a multiple of 32, got {value}"
            if value < minimum or value > maximum:
                return f"{name} must be between {minimum} and {maximum}, got {value}"
        return True

    def refine(self, image, model, clip, vae, seed, sampler_name, scheduler, steps, denoise, upscale_by, max_tile_width, max_tile_height, context_anchor, context_overlap, upscale_model=None, av_latent=None):
        # Lazy import: node.py's module scope stays comfy-free (pinned by a subprocess test).
        import comfy.samplers

        from . import sampling, upscale, video, vl_video

        audio_latent = None
        picks = None
        stamps = None
        vl_pack = None
        upscaled = None
        placeholder = None
        guider = None
        sigmas = None
        sampler = None

        _validate_image(image)
        # The target size is known before any pixel moves (prepare_upscaled always lands on
        # scale_target), so the too-small case is rejected here rather than after a full-clip
        # upscale. Below 32px an axis cannot carry even one /32 crop, and the reflect pad
        # (which needs pad < dim) would raise naming neither this node nor upscale_by.
        target_width, target_height = upscale.scale_target(int(image.shape[2]), int(image.shape[1]), upscale_by)
        if target_height < video.CANVAS_MULTIPLE or target_width < video.CANVAS_MULTIPLE:
            raise ValueError(f"upscale_by {upscale_by} takes the {image.shape[1]}x{image.shape[2]} input to {target_height}x{target_width}; the upscaled clip must be at least {video.CANVAS_MULTIPLE}x{video.CANVAS_MULTIPLE} pixels")
        audio_latent = _unpack_av_latent(av_latent)

        # STAGE ORDER IS THE HARD CONSTRAINT HERE, not a preference: the 32B text encoder
        # must never be resident while the full-clip canvases exist (two spike runs died at
        # exactly that RAM bracket — tests-AB/spike_h3_stress.py:437-440). So the conditioning
        # is encoded from the ~11 2 fps picks ALONE, brought to canvas geometry by the same
        # model + lanczos + reflect pad the canvas will get. Every one of those ops is
        # per-frame, which is what makes pick-then-upscale identical to upscale-then-pick and
        # lets the encode grid map onto rects the tiles have not been cut from yet.
        picks, stamps = vl_video.sample_conditioning_frames(image)
        picks = upscale.prepare_upscaled(picks, upscale_model, upscale_by)
        picks, _ = sampling.pad_image_to_multiple(picks, video.CANVAS_MULTIPLE)
        vl_pack = vl_video.encode_global(clip, picks, stamps)
        del picks           # dropped before the canvases claim the RAM the picks are holding

        # Only now do the full-clip canvases exist.
        upscaled = upscale.prepare_upscaled_video(image, upscale_model, upscale_by)
        # The positive placeholder is the whole-clip encode itself, not a second CLIP load:
        # video.py replaces it with the tile's own row slice on every sample() call, and the
        # guider only ever needs `positive` to BE in original_conds.
        placeholder = [[vl_pack.cond, {"minimax_token_tags": vl_pack.tags}]]
        guider = upscale.build_basic_guider(model, placeholder)
        sigmas = upscale.build_sigmas(model, scheduler, steps, denoise)
        sampler = comfy.samplers.sampler_object(sampler_name)

        return (video.refine_video(upscaled, guider, sampler, sigmas, vae, seed, max_tile_width, max_tile_height, context_anchor, context_overlap, vl_pack, audio_latent=audio_latent),)
