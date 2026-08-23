
# Mirrors core nodes.MAX_RESOLUTION (importing it would pull comfy in at module scope).
MAX_RESOLUTION = 16384


def _anchor_source():
    # WHAT the frozen context_anchor ring shows the model, defined once so both VL nodes'
    # wording cannot drift. The options come from sync.py, so the strings the widget offers
    # and the strings the engine branches on cannot diverge; the list is copied per call like
    # every other widget here, so no caller can poison a shared object. Lazy import:
    # node.py's module scope stays comfy-free (pinned by a subprocess test).
    # VL nodes ONLY: the base node is the raster path, whose ring is always the source.
    from . import sync

    return (list(sync.ANCHOR_SOURCES), {"default": sync.ANCHOR_SOURCE_IMAGE, "tooltip": "What fills context_anchor. source image keeps the result true to the input. live canvas adds more detail and drifts further from the input."})


def _vlm_method():
    # The conditioning-surface select, defined once so both VL nodes' wording cannot drift.
    # The options come from captions.py, so the strings the widget offers and the strings the
    # engine branches on cannot diverge; the list is copied per call like every other widget
    # here, so no caller can poison a shared object. captions.vlm_methods() is built once per
    # session, which is what keeps this list identical to the one the frontend cached at
    # startup. Lazy import: node.py's module scope stays comfy-free (pinned by a subprocess
    # test).
    from . import captions

    return (list(captions.vlm_methods()), {"default": captions.default_vlm_method(), "tooltip": "Whether each tile is conditioned on a caption of itself, on its slice of the entire image's vision encode, or on both. The name in parentheses is the caption preset it asks. Copy settings.toml to settings.user.toml to write your own tile prompts."})


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
                "image": ("IMAGE", {"tooltip": "The image to refine. Upscale it before this node."}),
                "guider": ("GUIDER", {"tooltip": "The guider that denoises each tile."}),
                "sampler": ("SAMPLER", {"tooltip": "The sampler used to denoise each tile."}),
                "sigmas": ("SIGMAS", {"tooltip": "The sigma schedule used when sampling each tile."}),
                "vae": ("VAE", {"tooltip": "The VAE that encodes and decodes each tile."}),
                "noise": ("NOISE", {"tooltip": "Noise is drawn once for the entire image and then sliced for each tile."}),
                "max_tile_width": ("INT", {"default": 1024, "min": 256, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Hard cap on the width the model ever sees per sampled crop, including the context_overlap and context_anchor rings. Set to the largest width the model supports."}),
                "max_tile_height": ("INT", {"default": 1024, "min": 256, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Hard cap on the height the model ever sees per sampled crop, including the context_overlap and context_anchor rings. Set to the largest height the model supports."}),
                "context_anchor": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Pixels around each tile that are frozen and shown to the model as context, then cropped away. With a mask it is also the frozen background the region is refined against, so keep it above 0."}),
                "context_overlap": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Overlapped context that is diffused from both sides and then blended. It anchors the tiles to each other, like context_anchor anchors each tile to its surroundings."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Only the masked region is refined and the rest is left untouched. Feed an inverted mask for a second pass."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "refine"
    CATEGORY = "image/upscaling"

    @classmethod
    def VALIDATE_INPUTS(s, max_tile_width=None, max_tile_height=None, context_anchor=None, context_overlap=None, vlm_method=None):
        # Naming widgets here disables ComfyUI's default min/max validation for them,
        # so ranges are re-checked alongside the /8 constraint (widget step is UI-only;
        # API-submitted workflows can send arbitrary INTs, and all derived geometry
        # must stay on the /8 grid the VAE requires).
        # vlm_method is named for the OTHER validation the same bypass covers: core rejects
        # a combo value that is not in the widget's own list (execution.py:1047, inside the
        # `x not in validate_function_inputs` guard). Every caption option gained a preset
        # label in 2026-08-22, so a workflow saved before that carries a bare
        # "vision tokens and captions" the current list no longer holds and would fail to
        # queue. captions.method_surface accepts it and resolve_method routes it to the
        # first preset. The SURFACE is all that is checked here — a label naming an absent
        # preset is named by resolve_method, which reads the file the run will actually use.
        # This is None on the base node, which offers no such widget.
        if vlm_method is not None:
            from . import captions

            try:
                captions.method_surface(vlm_method)
            except ValueError as error:
                return str(error)
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

    The tiles are stepped TOGETHER as lanes of one synchronized run (sync.py), and every
    tile's positive conditioning is replaced by its slice of ONE whole-image vision encode
    from the required CLIP:
    positionally exact, free of text demands, and globally informed, so tiles neither
    re-instantiate prompt objects they don't contain nor drift apart in story
    (gaze, tone, palette). The guider's positive text is ignored by construction;
    its negative still applies. No prompt input exists because none is needed.
    ControlNet is ignored on this node (the per-tile positive carries no control chain);
    use the base Context-Anchored Tile Refine node for control.
    vlm_method picks WHICH surface fills that positive: the vision slice (default), a
    per-tile VLM caption of the tile's own crop, or both (see captions.py).
    With a mask, the WHOLE image is still encoded and the region's tiles slice their
    true place in it, so a masked refine stays aware of the image around the region.
    """

    @classmethod
    def INPUT_TYPES(s):
        input_types = ContextAnchoredTileRefine.INPUT_TYPES()
        # Appended past context_overlap, never spliced in beside the ring it describes: dict
        # order IS widget order, and the frontend restores a saved workflow's widgets_values
        # POSITIONALLY, so a widget added mid-list shifts every value after it (the shipped
        # Krea 2 workflows would load context_anchor 256 into this combo and drop to 32).
        # Past the end of a legacy array the restore loop stops, leaving these widgets at
        # their defaults — which is the pre-select behaviour.
        input_types["required"]["anchor_source"] = _anchor_source()
        input_types["required"]["vlm_method"] = _vlm_method()
        input_types["required"]["clip"] = ("CLIP", {"tooltip": "Must be a vision-language text encoder (Krea 2 family). The guider's positive prompt is ignored and its negative still applies."})
        # The node's own id, so the ledger can write the live phase line under its progress
        # bar. Hidden inputs create no socket and no widget and never enter widgets_values,
        # so this is invisible to the append-only widget rule above. ComfyUI passes it to
        # FUNCTION as a keyword (execution.py:218), hence the refine parameter below.
        input_types["hidden"] = {"unique_id": "UNIQUE_ID"}
        return input_types

    def refine(self, image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, anchor_source, vlm_method, clip, mask=None, unique_id=None):
        _validate_image(image)
        if mask is not None:
            mask = _normalize_mask(mask, image)
        from . import progress, sampling

        # THE LEDGER IS CREATED HERE, on the VL nodes only: this is the one place the whole
        # run's shape is in hand, and one owner means the engine never has to decide whether
        # to make a bar. `with` is what scopes the comfy.utils.ProgressBar shim to this run;
        # finish() lands the bar on its total, including the zero-step early return.
        ledger = progress.build_ledger(vlm_method, int(sigmas.numel()) - 1, batch=int(image.shape[0]),
                                       unique_id=unique_id)
        with ledger:
            refined = sampling.refine_image(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=mask, vl_clip=clip, vlm_method=vlm_method, anchor_source=anchor_source, progress=ledger)
            ledger.finish()
        return (refined,)


class ContextAnchoredTileUpscaleVL(ContextAnchoredTileRefine):
    """Upscale + VL tile refine in one node: image in, refined image out.

    Same engine as ContextAnchoredTileRefineVL, with the four custom-sampling inputs
    (NOISE / SAMPLER / SIGMAS / GUIDER) built in-process from widgets by upscale.py, and
    the whole-image upscale stage run first so no tile is ever resampled. No mask input
    (use ContextAnchoredTileRefineVL for a region pass) and no positive prompt input —
    the positive is a placeholder that every tile's vision slice replaces, so a prompt
    here would only re-admit the phantom objects the VL path exists to remove. The
    optional negative is the one text channel that still applies.
    vlm_method picks WHICH surface fills that positive: the vision slice (default), a
    per-tile VLM caption of the tile's own crop, or both (see captions.py).
    """

    @classmethod
    def INPUT_TYPES(s):
        # Lazy import: node.py's module scope stays comfy-free (pinned by a subprocess test).
        import comfy.samplers

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The image to upscale and then refine."}),
                "model": ("MODEL", {"tooltip": "The diffusion model that denoises each tile."}),
                "clip": ("CLIP", {"tooltip": "Must be a vision-language text encoder (Krea 2 family). There is no positive prompt input, since each tile is conditioned on the image itself."}),
                "vae": ("VAE", {"tooltip": "The VAE that encodes and decodes each tile."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Noise is drawn once for the entire image and then sliced for each tile."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "sgm_uniform"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 8.0, "step": 0.01, "tooltip": "The upscale multiplier. The optional upscale_model runs first when one is connected."}),
                "max_tile_width": ("INT", {"default": 1536, "min": 256, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Hard cap on the width the model ever sees per sampled crop, including the context_overlap and context_anchor rings. Set to the largest width the model supports."}),
                "max_tile_height": ("INT", {"default": 2048, "min": 256, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Hard cap on the height the model ever sees per sampled crop, including the context_overlap and context_anchor rings. Set to the largest height the model supports."}),
                "context_anchor": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Pixels around each tile that are frozen and shown to the model as context, then cropped away."}),
                "context_overlap": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Overlapped context that is diffused from both sides and then blended. It anchors the tiles to each other, like context_anchor anchors each tile to its surroundings."}),
                # Last, never beside the ring it describes: the frontend restores a saved
                # workflow's widgets_values POSITIONALLY, so a widget added mid-list shifts
                # every value after it. Past the end of a legacy array the restore loop stops
                # and these keep their defaults — the pre-select behaviour.
                "anchor_source": _anchor_source(),
                "vlm_method": _vlm_method(),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Optional upscale model, run over the entire image before any tiling."}),
                "negative": ("CONDITIONING", {"tooltip": "Optional negative conditioning. Unconnected it is an empty encode of this node's CLIP."}),
            },
            # The node's own id, so the ledger can write the live phase line under its
            # progress bar. Hidden inputs create no socket and no widget and never enter
            # widgets_values, so the positional restore rule above is untouched. ComfyUI
            # passes it to FUNCTION as a keyword (execution.py:218).
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    def refine(self, image, model, clip, vae, seed, sampler_name, scheduler, steps, cfg, denoise, upscale_by, max_tile_width, max_tile_height, context_anchor, context_overlap, anchor_source, vlm_method, upscale_model=None, negative=None, unique_id=None):
        # Lazy import: node.py's module scope stays comfy-free (pinned by a subprocess test).
        import comfy.samplers

        from . import captions, progress, sampling, upscale

        empty_cond = None
        upscaled = None
        guider = None
        sigmas = None
        sampler = None
        noise = None

        _validate_image(image)
        # The settings file is read FIRST on the caption surfaces: this node runs the
        # upscale-model pass and the text-encoder load before the engine's own read in
        # sync._prepare_run, so a typo in the file would otherwise cost minutes of GPU
        # time to reach. "vision tokens" never reads it.
        captions.resolve_method(vlm_method)

        # THE LEDGER IS CREATED HERE, before the first phase it covers (the upscale model
        # pass), and `with` scopes the comfy.utils.ProgressBar shim to the whole run.
        ledger = progress.build_ledger(vlm_method, steps, batch=int(image.shape[0]),
                                       upscale_model=upscale_model is not None, clip_load=True,
                                       unique_id=unique_id)
        with ledger:
            # Everything that resamples happens here, on the whole image, before any tiling.
            upscaled = upscale.prepare_upscaled(image, upscale_model, upscale_by, progress=ledger)
            # The upscale REPLACES the validated input, and upscale_by goes down to 0.01, so a
            # small image can leave here below 8px on an axis — where the /8 reflect pad (which
            # needs pad < dim) would raise naming neither this node nor the widget that did it.
            if upscaled.shape[1] < 8 or upscaled.shape[2] < 8:
                raise ValueError(f"upscale_by {upscale_by} takes the {image.shape[1]}x{image.shape[2]} input to {upscaled.shape[1]}x{upscaled.shape[2]}. The upscaled image must be at least 8x8 pixels")

            # One empty encode serves as the positive placeholder (vl.py replaces every tile's
            # positive with its slice of the whole-image vision encode) and, unless the optional
            # input is connected, as the negative.
            # It is also the run's FIRST CLIP call, which pays CLIP.load_model and moves the
            # text encoder onto the GPU — minutes on a cold cache, with nothing else covering
            # it. Its own segment buys it an honest share of the total and its own status
            # line ("loading text encoder"); the bar itself sits at the segment boundary for
            # the load's duration, since nothing inside the load reports progress.
            ledger.open(progress.CLIP_LOAD)
            empty_cond = upscale.encode_empty(clip)
            guider = upscale.build_guider(model, empty_cond, empty_cond if negative is None else negative, cfg)
            sigmas = upscale.build_sigmas(model, scheduler, steps, denoise)
            sampler = comfy.samplers.sampler_object(sampler_name)
            noise = upscale.Noise_RandomNoise(seed)

            # sampler_name rides along beside the built SAMPLER: the sync engine rejects an
            # unsupported sampler before any encode, and core's sampler_object wraps several names
            # in a private function (dpm_fast -> dpm_fast_function), so resolving the rejection off
            # the OBJECT alone would name something this node's widget never offered.
            refined = sampling.refine_image(upscaled, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=None, vl_clip=clip, vlm_method=vlm_method, anchor_source=anchor_source, sampler_name=sampler_name, progress=ledger)
            # denoise 0.0 is a legitimate "upscale only" setting: build_sigmas returns an empty
            # schedule and refine_image returns before the picture loop, so most of the plan is
            # never opened. finish() is what stops the bar freezing mid-run on it.
            ledger.finish()
        return (refined,)
