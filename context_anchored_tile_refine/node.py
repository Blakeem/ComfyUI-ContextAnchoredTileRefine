
# Mirrors core nodes.MAX_RESOLUTION (importing it would pull comfy in at module scope).
MAX_RESOLUTION = 16384


def _validate_image(image):
    if image.ndim != 4:
        raise ValueError(f"image must be a [B,H,W,C] IMAGE tensor, got {image.ndim} dimensions")
    if image.shape[1] < 8 or image.shape[2] < 8:
        raise ValueError(f"image must be at least 8x8 pixels, got {image.shape[1]}x{image.shape[2]}")


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
    # Onto the IMAGE's device: every tensor sampling.py derives from the mask (the region
    # gate, the AA alpha) is combined with canvas-device tensors, and under --gpu-only the
    # VAE emits a CUDA IMAGE while the mask nodes emit a CPU MASK. A no-op when they match.
    return mask.to(image.device)


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
