
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

    return (list(sync.ANCHOR_SOURCES), {"default": sync.ANCHOR_SOURCE_IMAGE, "tooltip": "What the frozen context ring around each tile shows the model. source image shows the unmodified input, presented on the settled lead schedule. That is maximum fidelity, so placement, style and objects stay locked to the input, including its flaws. live canvas shows the in-progress result itself. The refine may reinterpret or repair damaged content the captions describe. Expect more invention and slightly brighter output."})


def _vlm_method():
    # The conditioning-surface select, defined once so both VL nodes' wording cannot drift.
    # The options come from captions.py, so the strings the widget offers and the strings the
    # engine branches on cannot diverge; the list is copied per call like every other widget
    # here, so no caller can poison a shared object. captions.vlm_methods() is built once per
    # session, which is what keeps this list identical to the one the frontend cached at
    # startup. Lazy import: node.py's module scope stays comfy-free (pinned by a subprocess
    # test).
    from . import captions

    return (list(captions.vlm_methods()), {"default": captions.default_vlm_method(), "tooltip": "What the model is told about each tile. vision tokens slices one whole-image vision encode per tile. The slice carries the original style, what is in that tile, and global coherence, and invents nothing. captions writes a short description of each tile with the same VL model and uses it as that tile's prompt. That is more creative, and it can repair messy backgrounds and hallucinations in the source by steering the tile toward something coherent. vision tokens and captions carries both. Each tile caption is written from that tile's pixels alone. The name in parentheses is a caption preset from settings.toml in the node's folder, which holds the prompts every caption option asks. An option with no name in parentheses uses the first preset in that file, which is the default one. Edit a preset there to change its wording, and edits apply on the next run. Add a preset to add its own pair of options, which needs a ComfyUI restart. When a preset sets global_style_instruction, one style caption of the whole image is written per picture and placed on top of every tile caption so that all tiles follow one style description. vision tokens is fastest, since the whole-image encode is built once and every tile takes a slice of it. captions writes one caption per tile and builds no whole-image encode. vision tokens and captions costs the two added together. Writing captions is what scales with tile count, and the whole-image encode does not."})


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
                "context_anchor": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Width of the fully-frozen, visible-only context ring sampled beyond the overlap on every edge, including up against a mask, then cropped away. With a mask it also sets the frozen-background halo the masked region conditions against, so keep it above 0. It is always additive, so the ring outside a tile core is context_overlap + context_anchor. The useful range is 32 to 256 and 32 suits most scenes."}),
                "context_overlap": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Inter-tile directional feather width, on multi-tile runs only. It is never applied at a mask boundary. Each tile is sampled oversized. On sides bordering an already-processed neighbor (top and left) this band is fully diffused and feathered into that neighbor, at 100% on the seam falling to 0% over the band. Elsewhere it is context, then cropped. The useful range is 32 to 256. DETAILED scenes need LESS, and 32 is invisible even when you know where the seam is, because the seam has texture to hide in and to route through. A large smooth gradient such as an open night sky is the hard case and wants 128 or more. 0 gives hard seams."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional region mask. Only the masked region is refined, hardened at 0.5, cropped to the mask plus context_anchor, with a 1px anti-aliased edge. The unmasked background is left untouched. Feed an inverted mask for a second pass."}),
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
        input_types["required"]["clip"] = ("CLIP", {"tooltip": "The workflow's CLIP, which must be a vision-language text encoder (Krea 2 family). The whole image is encoded once through its vision path and each tile's positive conditioning becomes its slice of that encode. The guider's positive prompt is ignored and the negative still applies."})
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
                "image": ("IMAGE", {"tooltip": "The image to upscale and then refine tile by tile."}),
                "model": ("MODEL", {"tooltip": "The diffusion model used to denoise each tile."}),
                "clip": ("CLIP", {"tooltip": "The workflow's CLIP, which must be a vision-language text encoder (Krea 2 family). The whole upscaled image is encoded once through its vision path and each tile's positive conditioning becomes its slice of that encode."}),
                "vae": ("VAE", {"tooltip": "The VAE used to encode each tile for sampling and decode the result."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Seed for the noise. Noise is drawn once for the whole image and sliced per tile, so a tile's noise does not change when the grid does."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m", "tooltip": "The sampler used to denoise each tile."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "sgm_uniform", "tooltip": "The sigma schedule used when sampling each tile."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Sampling steps per tile. As in BasicScheduler, the schedule is built for steps/denoise steps and only the last steps+1 sigmas are used."}),
                "cfg": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01, "tooltip": "Classifier-free guidance scale applied against the negative conditioning."}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "How much of the schedule to run per tile: 1.0 rewrites the tile, lower values keep more of the upscaled pixels. 0.0 skips diffusion entirely and returns the upscale alone."}),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 0.01, "max": 8.0, "step": 0.01, "tooltip": "Target scale for the whole-image upscale stage that runs before any tiling. The optional upscale_model runs first and this factor sets the final size. 1.0 with no model leaves the input pixels untouched."}),
                "max_tile_width": ("INT", {"default": 1536, "min": 256, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Hard cap on the width the model ever sees per sampled crop, including the context_overlap and context_anchor rings. Set to the largest width the model supports."}),
                "max_tile_height": ("INT", {"default": 2048, "min": 256, "max": MAX_RESOLUTION, "step": 8, "tooltip": "Hard cap on the height the model ever sees per sampled crop, including the context_overlap and context_anchor rings. Set to the largest height the model supports."}),
                "context_anchor": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Width of the fully-frozen, visible-only context ring sampled beyond the overlap on every edge, then cropped away. It is always additive, so the ring outside a tile core is context_overlap + context_anchor. The useful range is 32 to 256 and 32 suits most scenes."}),
                "context_overlap": ("INT", {"default": 32, "min": 0, "max": 512, "step": 8, "tooltip": "Inter-tile directional feather width, on multi-tile runs only. Each tile is sampled oversized. On sides bordering an already-processed neighbor (top and left) this band is fully diffused and feathered into that neighbor, at 100% on the seam falling to 0% over the band. Elsewhere it is context, then cropped. The useful range is 32 to 256. DETAILED scenes need LESS, and 32 is invisible even when you know where the seam is. A large smooth gradient such as an open night sky wants 128 or more. 0 gives hard seams."}),
                # Last, never beside the ring it describes: the frontend restores a saved
                # workflow's widgets_values POSITIONALLY, so a widget added mid-list shifts
                # every value after it. Past the end of a legacy array the restore loop stops
                # and these keep their defaults — the pre-select behaviour.
                "anchor_source": _anchor_source(),
                "vlm_method": _vlm_method(),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL", {"tooltip": "Optional upscale model run over the whole image before tiling. Its fixed integer scale rarely lands on upscale_by, so a single lanczos pass then takes the result to the exact target."}),
                "negative": ("CONDITIONING", {"tooltip": "Optional negative conditioning. Left unconnected it is an empty encode of this node's CLIP, which is what cfg needs to be meaningful at all."}),
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
