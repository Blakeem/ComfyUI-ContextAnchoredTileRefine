"""ComfyUI-facing plumbing for the A/B harness: image IO, the upscale stage, and
the model/guider/sigmas construction that mirrors the workflow's refine stage.

Every function here replicates one ComfyUI node body (named in its comment) so the
harness produces what the graph produced. `ab_env.bootstrap()` must have run first;
all comfy imports are therefore function-local.
"""
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo


# ---------------------------------------------------------------- image IO

# Deliberately no load_image(): nothing in this harness may take PIXELS from a PNG.
# The graph hands a float VAEDecode tensor to the upscaler, and 8-bit quantization of a
# very dark frame is large relative to the signal the A/B measures. Prompts still come
# from a PNG's `prompt` chunk (read_prompts below) — text, not pixels.


def require_image_shape(image, height, width, what):
    """Fail loudly on a wrong-sized IMAGE tensor.

    A shadowed variable once produced an 8px-wide output that looked plausible in a
    directory listing, so every stage boundary is checked explicitly now."""
    actual = tuple(image.shape)
    if image.ndim != 4 or actual[1:] != (height, width, 3):
        raise RuntimeError("{}: expected [B,{},{},3], got {}".format(what, height, width, actual))
    return image


def png_size(path):
    """(width, height) of a PNG on disk, read back from the file itself."""
    with Image.open(path) as handle:
        return handle.size


def to_uint8(image):
    """SaveImage's quantization, exactly: truncate, do NOT round.
    (nodes.py SaveImage: np.clip(255. * image, 0, 255).astype(np.uint8).)
    Matching it matters — the self-check diffs our PNG against ComfyUI's."""
    # .float() after .cpu(): fp16 canvases would otherwise scale IN fp16 (spacing 0.125
    # near 255), rounding before the truncation and biasing ~3% of pixels +1 vs SaveImage.
    scaled = 255.0 * image[0].detach().cpu().float().numpy()
    return np.clip(scaled, 0, 255).astype(np.uint8)


def save_png(path, image, settings):
    """Write [1,H,W,3] to PNG with the run settings in a tEXt chunk.

    Returns the (width, height) read back OFF DISK and raises if it disagrees with the
    tensor — the only check that cannot be fooled by a wrong-shaped tensor upstream."""
    info = PngInfo()
    info.add_text("catr_ab", json.dumps(settings, indent=1, sort_keys=True))
    info.add_text("catr_ab_summary", summarize(settings))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(image)).save(path, pnginfo=info, compress_level=4)

    written = png_size(path)
    expected = (int(image.shape[2]), int(image.shape[1]))
    if written != expected:
        raise RuntimeError("{}: wrote {}x{} but the tensor is {}x{}".format(
            path, written[0], written[1], expected[0], expected[1]))
    return written


def summarize(settings):
    """One-line human-readable digest of a settings dict, for the second tEXt key."""
    keys = ("image_label", "run_label", "seam_mode", "seam_blend", "context_anchor",
            "context_overlap", "max_tile_width", "max_tile_height", "sampler",
            "scheduler", "steps", "denoise", "cfg", "seed")
    return " ".join("{}={}".format(k, settings[k]) for k in keys if k in settings)


def read_prompts(path, positive_node, negative_node):
    """Pull the positive/negative prompt strings out of a ComfyUI PNG's `prompt`
    chunk. Returns (positive, negative) or (None, None) when unavailable."""
    try:
        with Image.open(path) as handle:
            raw = handle.text.get("prompt")
        if not raw:
            return None, None
        graph = json.loads(raw)
        return (graph[positive_node]["inputs"]["value"],
                graph[negative_node]["inputs"]["value"])
    except (KeyError, ValueError, OSError):
        return None, None


# ---------------------------------------------------------------- float caches

def cache_path(cache_dir, label, kind, width, height, key):
    """Cache target for one float intermediate.

    `key` is the full set of settings that produced it (any JSON-serializable value);
    the upscale key nests the base key, so any base-affecting change renames both files
    instead of silently reusing a stale one. Nothing here hashes file mtimes any more —
    no stage reads pixels from disk."""
    payload = json.dumps(key, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return Path(cache_dir) / "{}_{}_{}x{}_{}.pt".format(label, kind, width, height, digest)


# ---------------------------------------------------------------- upscale stage


def upscale_to(image, upscale_model_name, target_w, target_h):
    """The workflow's node 223 -> 224 chain.

    223 ImageUpscaleWithModel: the real node body is called, so the 512px
    comfy.utils.tiled_scale path with its halving OOM retry is reused verbatim
    rather than re-implemented (a 4x of 768x1024 is 3072x4096 and will not fit
    otherwise).

    224 ResizeImageMaskNode(resize_type="scale dimensions", crop="disabled",
    scale_method="lanczos"): NOT a deviation. That core node's scale_dimensions()
    is exactly movedim(-1,1) -> comfy.utils.common_upscale(w, h, method, crop) ->
    movedim(1,-1) (comfy_extras/nodes_post_processing.py), which is what runs here.
    Note common_upscale's lanczos goes through uint8 PIL internally.
    """
    import comfy.utils
    from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel, UpscaleModelLoader

    upscale_model = UpscaleModelLoader.execute(upscale_model_name)[0]
    try:
        with torch.inference_mode():
            big = ImageUpscaleWithModel.execute(upscale_model, image)[0]
    finally:
        del upscale_model
        free_gpu()
    resized = comfy.utils.common_upscale(
        big.movedim(-1, 1), target_w, target_h, "lanczos", "disabled"
    ).movedim(1, -1)
    del big
    free_gpu()
    return resized.contiguous()


# ------------------------------------------------- models and sampling (both stages)

def load_clip(clip_name, clip_type):
    """CLIPLoader.load_clip with device="default"."""
    import comfy.sd
    import folder_paths

    clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    return comfy.sd.load_clip(
        ckpt_paths=[clip_path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=getattr(comfy.sd.CLIPType, clip_type.upper()),
        model_options={},
    )


def encode_prompt(clip, text):
    """CLIPTextEncode.encode."""
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))


def load_unet(unet_name):
    """UNETLoader.load_unet with weight_dtype="default" (no dtype override)."""
    import comfy.sd
    import folder_paths

    unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
    return comfy.sd.load_diffusion_model(unet_path, model_options={})


def load_vae(vae_name):
    """VAELoader.load_vae for a real .safetensors VAE (no taesd branch)."""
    import comfy.sd
    import comfy.utils
    import folder_paths

    vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
    state_dict, metadata = comfy.utils.load_torch_file(vae_path, return_metadata=True)
    vae = comfy.sd.VAE(sd=state_dict, metadata=metadata)
    vae.throw_exception_if_invalid()
    return vae


def build_guider(model, positive, negative, empty, cfg, neg_scale):
    """CFGGuider.execute — the real comfy.samplers.CFGGuider.

    Replaces the workflow's PerpNegGuider on purpose. PerpNeg evaluates positive, negative
    AND empty conditioning every step; CFGGuider at cfg 1.0 evaluates ONE, because
    comfy/samplers.py sampling_function() sets uncond_ = None when isclose(cond_scale, 1.0).
    Three model calls per step become one, and the seam behaviour under test is unaffected —
    this harness exists to compare seam methods, not to reproduce a guider.

    `negative` and `empty` are accepted and ignored so the call sites and the recorded
    settings stay stable; at cfg 1.0 neither would reach the model anyway. `neg_scale` is
    PerpNeg-only and likewise ignored."""
    import comfy.samplers

    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(cfg)
    return guider


def build_sigmas(model, scheduler, steps, denoise):
    """BasicScheduler.get_sigmas."""
    import comfy.samplers

    if denoise <= 0.0:
        return torch.FloatTensor([])
    total_steps = steps if denoise >= 1.0 else int(steps / denoise)
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), scheduler, total_steps
    ).cpu()
    return sigmas[-(steps + 1):]


def build_sampler(sampler_name):
    """KSamplerSelect.get_sampler."""
    import comfy.samplers

    return comfy.samplers.sampler_object(sampler_name)


def build_noise(seed):
    """RandomNoise.get_noise."""
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

    return Noise_RandomNoise(seed)


# ---------------------------------------------------------------- base generation

def load_lora_model_only(model, lora_name, strength_model):
    """LoraLoaderModelOnly.load_lora_model_only, i.e. LoraLoader.load_lora(model, None, ...).

    Returns a ModelPatcher CLONE that shares the base nn.Module, so the 11.5 GB DiT is read
    from disk once and the caller's unpatched patcher stays usable for the refine stage.
    That is not an optimization, it is the workflow: node 185 feeds BOTH the LoRA branch
    (node 65 -> base generation) and the unpatched branch (nodes 170/183 -> refine)."""
    import comfy.sd
    import comfy.utils
    import folder_paths

    lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
    lora, lora_metadata = comfy.utils.load_torch_file(lora_path, safe_load=True, return_metadata=True)
    patched, _ = comfy.sd.load_lora_for_models(
        model, None, lora, strength_model, 0, lora_metadata=lora_metadata
    )
    return patched


def empty_sd3_latent(width, height, batch_size):
    """EmptySD3LatentImage.execute. The `downscale_ratio_spacial` key (comfy's spelling)
    is part of the LATENT contract SamplerCustomAdvanced reads — do not drop it."""
    import comfy.model_management

    latent = torch.zeros(
        [batch_size, 16, height // 8, width // 8],
        device=comfy.model_management.intermediate_device(),
        dtype=comfy.model_management.intermediate_dtype(),
    )
    return {"samples": latent, "downscale_ratio_spacial": 8}


def sample_custom_advanced(noise, guider, sampler, sigmas, latent):
    """SamplerCustomAdvanced.execute, output slot 0 ("output").

    Slot 0, not slot 1: node 112 VAEDecode is wired to ["38", 0], the sampler result, not
    the denoised_output preview. The real node body runs, so the latent_preview callback
    and fix_empty_latent_channels behave exactly as they do in the app."""
    from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced

    return SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, latent)[0]


def vae_decode(vae, latent):
    """VAEDecode.decode. Returns the float tensor the graph feeds straight to the
    upscaler — NOT clamped or quantized to 8 bits."""
    samples = latent["samples"]
    if samples.is_nested:
        samples = samples.unbind()[0]
    images = vae.decode(samples)
    if len(images.shape) == 5:            # combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def weight_fingerprint(model, samples=4):
    """Checksum of a few of the DiT's own weights, to prove the LoRA clone never wrote
    through to the shared nn.Module. Compared before the patch and after free_gpu()
    unpatches; a change would mean the refine stage is running a LoRA'd model.

    Accumulated in float64 on purpose. A float32 sum over a 10M-element weight tensor
    drifts by ~1e-4 purely from reduction order, which would force a tolerance loose
    enough to hide a real patch; in float64 an unchanged weight reproduces exactly."""
    state = model.model.state_dict()
    total = 0.0
    for name in sorted(state):
        tensor = state[name]
        if tensor.numel() < 4096:
            continue
        # One float64 copy, abs in place on it: .double().abs() keeps both alive at once.
        # Cast first, never abs() first — these state_dicts carry int8/float8 weights.
        total += float(tensor.detach().to(torch.float64).abs_().sum().item())
        samples -= 1
        if samples <= 0:
            break
    return total


# ---------------------------------------------------------------- resources

def free_gpu():
    """Drop every ComfyUI-managed model off the GPU and empty the caching allocator.
    `del`eting a patcher is not enough — model_management keeps its own reference.

    Use this only when SWAPPING model sets (upscaler -> CLIP -> UNET). Between runs that
    share the same models use clear_cache() instead: unload_all_models() costs a full
    re-upload of the 11.5 GB UNET across PCIe on the next sample, per run."""
    import comfy.model_management

    # Cache-empty LAST: the collect that reaps the cycle-held patchers unload_all_models()
    # dropped has to run before the blocks they hold can be returned to the allocator.
    gc.collect()
    comfy.model_management.unload_all_models()
    gc.collect()
    comfy.model_management.soft_empty_cache()


def clear_cache():
    """Release cached allocator blocks WITHOUT unloading models, so consecutive runs
    start from a defragmented heap but the resident UNET is not re-uploaded."""
    import comfy.model_management

    gc.collect()
    comfy.model_management.soft_empty_cache()


class VramProbe:
    """Peak torch allocator usage across a block, in GiB. Reserved is the number
    that matters against the card's capacity; allocated is live tensor bytes."""

    def __init__(self):
        self.available = torch.cuda.is_available()
        self.allocated = 0.0
        self.reserved = 0.0
        self.started = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        if self.available:
            torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self.started
        if self.available:
            self.allocated = torch.cuda.max_memory_allocated() / 1024 ** 3
            self.reserved = torch.cuda.max_memory_reserved() / 1024 ** 3
        return False

    def __str__(self):
        if not self.available:
            return "{:.1f}s".format(self.elapsed)
        return "{:.1f}s  peak VRAM {:.2f} GiB alloc / {:.2f} GiB reserved".format(
            self.elapsed, self.allocated, self.reserved
        )
