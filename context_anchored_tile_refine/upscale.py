"""Whole-image upscale stage, plus the sampling plumbing the all-in-one node internalizes.

`prepare_upscaled` is the upscale entry point. It brings the input to
`scale_target(width, height, upscale_by)` in one pass over the WHOLE image — the optional
upscale model first, then at most one lanczos resize onto the exact target — so the tile
pipeline downstream never resamples a tile (prime directive 1).

The non-obvious rule: a resize that would NOT change the size is skipped, not run.
comfy.utils.common_upscale's "lanczos" branch round-trips through PIL on uint8
(comfy/utils.py `lanczos`), so a same-size lanczos is an 8-bit quantization loss rather
than an identity.

`Noise_RandomNoise` / `build_sigmas` / `build_guider` / `encode_empty` are the NOISE,
SIGMAS, GUIDER and CONDITIONING that `ContextAnchoredTileRefineVL` takes as inputs, built
in-process from widgets instead. Each mirrors the core node that produces that type, so an
all-in-one run and the equivalent hand-wired graph sample identically.

`prepare_upscaled_video` and `build_basic_guider` are the video node's two variants of
that pair: the same upscale rules applied in frame chunks, and the CFG-FREE guider MiniMax
H3 runs.

Module scope is torch-only; comfy is imported lazily inside functions (the same contract
as sampling.py / vl.py, pinned by a subprocess test).
"""
import torch

# Core's ImageUpscaleWithModel tiling (comfy_extras/nodes_upscale_model.py): the model runs
# at tile 512 / overlap 32, halves the tile on an OOM, and gives up below 128.
MODEL_TILE = 512
MODEL_TILE_OVERLAP = 32
MODEL_TILE_MIN = 128

# Frames per whole-clip upscale chunk; see prepare_upscaled_video.
UPSCALE_FRAME_CHUNK = 8


def scale_target(width, height, upscale_by):
    # Target size as (width, height). Rounding matches core's ImageScaleBy (nodes.py:
    # round(dim * scale_by)); the 1px floor only binds where core would ask for a 0-px
    # axis (a tiny image at a small upscale_by).
    return max(1, round(width * upscale_by)), max(1, round(height * upscale_by))


def _lanczos_resize(image, width, height):
    # ImageScaleBy's body with the method pinned to lanczos: NHWC -> NCHW -> NHWC.
    import comfy.utils

    samples = image.movedim(-1, 1)
    resized = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    return resized.movedim(1, -1)


def _upscale_with_model(upscale_model, image):
    # comfy_extras/nodes_upscale_model.py ImageUpscaleWithModel.execute, with model
    # residency made version-defensive: pyproject only requires ComfyUI >= 0.3.45, and the
    # UPSCALE_MODEL object gained its `.patcher` after that floor. WITH a patcher the model
    # is a managed model and load_models_gpu owns residency and eviction. WITHOUT one (the
    # older loader) the pre-patcher contract applies: reserve the memory by hand, move the
    # module onto the device, and move it back off in `finally` so a raise cannot strand it
    # on the GPU.
    import comfy.model_management
    import comfy.utils

    patcher = getattr(upscale_model, "patcher", None)
    # Core's estimate verbatim; the 384.0 is core's own guess at the per-pixel working set.
    memory_required = (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0
    memory_required += image.nelement() * image.element_size()
    tile = MODEL_TILE
    oom = True
    upscaled = None

    if patcher is not None:
        device = patcher.load_device
        comfy.model_management.load_models_gpu([patcher], memory_required=memory_required)
    else:
        device = comfy.model_management.get_torch_device()
        comfy.model_management.free_memory(memory_required, device)
        upscale_model.to(device)

    # The try opens on the statement AFTER the device move: the large VRAM copy below is the
    # most likely thing here to OOM, and outside the try it would strand the module on the GPU.
    try:
        in_img = image.movedim(-1, -3).to(device)
        output_device = comfy.model_management.intermediate_device()
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=MODEL_TILE_OVERLAP)
                pbar = comfy.utils.ProgressBar(steps)
                upscaled = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a.float()), tile_x=tile, tile_y=tile, overlap=MODEL_TILE_OVERLAP, upscale_amount=upscale_model.scale, pbar=pbar, output_device=output_device)
                oom = False
            except Exception as error:
                # Only an out-of-memory condition is retryable; raise_non_oom re-raises
                # everything else rather than letting it drive the tile down to the floor.
                comfy.model_management.raise_non_oom(error)
                tile //= 2
                if tile < MODEL_TILE_MIN:
                    raise
    finally:
        if patcher is None:
            upscale_model.to("cpu")
    return torch.clamp(upscaled.movedim(-3, -1), min=0, max=1.0).to(comfy.model_management.intermediate_dtype())


def prepare_upscaled(image, upscale_model, upscale_by):
    # The pre-tiling resize stage: [B,H,W,C] in, [B,target_h,target_w,C] out.
    # Model first, lanczos second, because a model's fixed integer scale rarely lands on
    # upscale_by — the lanczos pass then takes that output the rest of the way to the exact
    # target. Either stage is skipped when it would not change the size (see the module
    # docstring on why a same-size lanczos is a loss, not a no-op), so at upscale_by == 1.0
    # with no model the caller's own tensor comes back untouched.
    target_width, target_height = scale_target(int(image.shape[2]), int(image.shape[1]), upscale_by)
    current = image

    if upscale_model is not None:
        current = _upscale_with_model(upscale_model, current)
    if int(current.shape[2]) == target_width and int(current.shape[1]) == target_height:
        return current
    return _lanczos_resize(current, target_width, target_height)


def prepare_upscaled_video(frames, upscale_model, upscale_by):
    # prepare_upscaled's rules applied to a whole clip: [T,H,W,C] in, [T,target_h,target_w,C]
    # fp16 out. Chunked because core's lanczos builds an fp32 list plus a stack of the WHOLE
    # batch it is handed (comfy/utils.py lanczos) — ~15 GB at 2688x1536x124 on a 32 GB
    # machine — while 8 frames cap that transient near 1 GB, and writing into a preallocated
    # canvas avoids a second full-clip allocation (tests-AB/spike_h3_stress.py:460-470).
    # fp16 is the canvas dtype video.py works in; the outputs are 8-bit, so the step is below
    # the quantization floor.
    target_width, target_height = scale_target(int(frames.shape[2]), int(frames.shape[1]), upscale_by)
    frame_count = int(frames.shape[0])
    canvas = None

    if upscale_model is None and int(frames.shape[2]) == target_width and int(frames.shape[1]) == target_height:
        # Same rule as prepare_upscaled: a resize that would not change the size is a loss,
        # not an identity (see the module docstring), so the caller's tensor comes back.
        return frames

    canvas = torch.empty([frame_count, target_height, target_width, int(frames.shape[3])],
                         dtype=torch.float16)
    for f0 in range(0, frame_count, UPSCALE_FRAME_CHUNK):
        f1 = min(f0 + UPSCALE_FRAME_CHUNK, frame_count)
        canvas[f0:f1] = prepare_upscaled(frames[f0:f1], upscale_model, upscale_by)
    return canvas


class Noise_RandomNoise:
    # comfy_extras/nodes_custom_sampler.py Noise_RandomNoise: the object behind the NOISE
    # type, reproduced here because the all-in-one node takes a seed widget rather than a
    # NOISE input. sampling.py calls generate_noise once, on a dummy latent that mirrors the
    # VAE's layout, so the seed→noise mapping is the whole image's, not a per-tile one.
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        import comfy.sample

        latent_image = input_latent["samples"]
        batch_inds = input_latent.get("batch_index", None)
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)


def build_sigmas(model, scheduler, steps, denoise):
    # comfy_extras/nodes_custom_sampler.py BasicScheduler.execute: schedule `total_steps`
    # then keep only the last steps+1 sigmas, which is what makes `denoise` a partial-noise
    # start point rather than a step count.
    # denoise <= 0.0 yields an EMPTY tensor. sampling.refine_image reads sigmas.numel() < 2
    # as "nothing to sample" and returns the image unchanged, so denoise 0 on this node
    # means "upscale only, no diffusion" — a legitimate setting, not an error.
    import comfy.samplers

    total_steps = steps
    if denoise < 1.0:
        if denoise <= 0.0:
            return torch.FloatTensor([])
        total_steps = int(steps / denoise)

    sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
    return sigmas[-(steps + 1):]


def build_guider(model, positive, negative, cfg):
    # comfy_extras/nodes_custom_sampler.py CFGGuider.execute. It must be core's CFGGuider
    # specifically: the VL path in sampling.refine_image swaps each tile's positive inside
    # `original_conds`, which is that class's convention.
    import comfy.samplers

    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)
    return guider


def build_basic_guider(model, positive):
    # comfy_extras/nodes_custom_sampler.py:805-827 BasicGuider.execute, whose Guider_Basic is
    # a CFGGuider subclass carrying ONLY a positive. MiniMax H3 is CFG-free, so this is the
    # guider its templates run: a CFGGuider with a dummy negative would double the DiT cost
    # for a channel the model never weighs. The subclass is declared here, inside the
    # function, so the module scope stays torch-only; `positive` still lands in
    # `original_conds` (comfy/samplers.py:1201-1205 inner_set_conds), which is the seam
    # video.py's per-tile positive swap keys on.
    import comfy.samplers

    class Guider_Basic(comfy.samplers.CFGGuider):
        def set_conds(self, positive):
            self.inner_set_conds({"positive": positive})

    guider = Guider_Basic(model)
    guider.set_conds(positive)
    return guider


def encode_empty(clip):
    # nodes.py CLIPTextEncode.encode with an empty prompt. The VL node has no prompt input:
    # this is the positive placeholder vl.py overwrites per tile, and the negative whenever
    # the optional negative input is unconnected.
    return clip.encode_from_tokens_scheduled(clip.tokenize(""))
