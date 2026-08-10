"""F0 proof of concept: small MiniMax H3 video, 2x pixel upscale, partial-denoise refine.

Gates the H3 upscale-node feature cycle. One small base render feeds five artifacts:

    base            t2va at 704x384x124f (template sampling: res_multistep, simple/20)
    refine_pure     partial denoise of the base latent at base res, no VAE round trip:
                    isolates denoise tolerance from everything else
    upscaled        the base frames lanczos'd 2x to 1408x768 — the "before" reference
                    (just above the owner's usual 1376x768, still ~1MP in-distribution)
    up_roundtrip    encode -> decode of the upscaled frames, NO sampling: the video
                    VAE's fidelity floor, which bounds what refine can preserve
    refine_up       encode(upscaled) -> partial denoise -> decode: the production
                    path minus tiling. THE proof-of-concept artifact.

Audio is frozen in every refine run (nested denoise mask: video ones, audio zeros —
the same full-shape convention core sample() builds for absent stream masks).
Audio is never decoded; the A/B is visual.

Outputs: one mp4 (crf 10) + first/middle/last PNG frames per artifact, in OUTPUT_DIR.
The base AV latent pair and the t2va conditioning are cached as .pt files, so
refine-strength sweeps re-run without the 20-step base render or the 32B CLIP load.

    <venv-python> tests-AB/spike_h3.py                 # all five artifacts
    <venv-python> tests-AB/spike_h3.py --denoise 0.25  # sweep refine strength
    <venv-python> tests-AB/spike_h3.py --force         # ignore caches

Verdict is the owner's visual A/B. Grit or content destruction in refine_pure kills
the feature; refine_up vs upscaled is the proof of concept; up_roundtrip soft spots
are VAE-floor, not sampling, artifacts.
"""
import argparse
import faulthandler
import json
import os
import sys
from pathlib import Path

faulthandler.enable()  # native-crash stack into the log (two runs died with SIGSEGV)

# The only H3-capable tree on this machine (comfy/ldm/minimax). ab_env's candidate
# list predates it, so pin the root before bootstrap unless the caller overrode it.
H3_ROOT = Path(r"C:\Users\Blake\ComfyUI-Installs\ComfyUI\ComfyUI")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import ab_env  # noqa: E402

if "COMFYUI_ROOT" not in os.environ:
    if not (H3_ROOT / "comfy" / "ldm" / "minimax" / "model.py").is_file():
        raise SystemExit(f"No MiniMax H3 support at {H3_ROOT} — set COMFYUI_ROOT")
    os.environ["COMFYUI_ROOT"] = str(H3_ROOT)

ROOT, ROOT_NOTE = ab_env.bootstrap()

import ab_models  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# ------------------------------------------------------------------ CONFIG

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-H3")
CACHE_DIR = Path(__file__).resolve().parent / "cache"

UNET_NAME = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP_NAME = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
CLIP_TYPE = "minimax"
VAE_NAME = "minimax_h3_video_vae_fp16.safetensors"

# Base small on purpose; 2x lands at 1408x768 — just above the owner's usual
# 1376x768 and inside the model's ~768*1344 trained canvas. Both /32.
WIDTH, HEIGHT = 704, 384
UP_W, UP_H = 1408, 768
LENGTH = 124                                     # 17k+5 grid, trained-range floor
FPS = 24
SAMPLER, SCHEDULER, STEPS = "res_multistep", "simple", 20
BASE_SEED, REFINE_SEED = 71, 72

PROMPT = (
    "Handheld documentary footage of a weathered fisherman mending a net on a wooden "
    "dock at golden hour. Deep facial wrinkles, individual grey beard hairs, knit "
    "sweater texture, frayed rope fibers. Behind him moored boats bob slowly, gulls "
    "wheel overhead, small waves lap the pilings. Warm rim light, shallow depth of "
    "field, gentle camera sway."
)

BASE_KEY = {
    "unet": UNET_NAME, "clip": CLIP_NAME, "vae": VAE_NAME, "prompt": PROMPT,
    "width": WIDTH, "height": HEIGHT, "length": LENGTH, "sampler": SAMPLER,
    "scheduler": SCHEDULER, "steps": STEPS, "seed": BASE_SEED,
}


# ------------------------------------------------------------------ helpers

def build_basic_guider(model, positive):
    """BasicGuider.execute: the CFG-free guider the H3 templates run."""
    from comfy_extras.nodes_custom_sampler import Guider_Basic

    guider = Guider_Basic(model)
    guider.set_conds(positive)
    return guider


def frozen_audio_mask(video, audio):
    """Nested denoise mask: refine the video stream, freeze the audio stream."""
    import comfy.nested_tensor

    return comfy.nested_tensor.NestedTensor(
        (torch.ones_like(video, device="cpu"), torch.zeros_like(audio, device="cpu")))


def upscale_frames(frames):
    """Per-frame 2x lanczos to (UP_W, UP_H) — core's common_upscale, the same op the
    production upscale stage ends with. [T,H,W,C] in and out."""
    import comfy.utils

    # Frame-chunked: core's lanczos builds an fp32 tensor per frame for the WHOLE batch
    # and stacks it while the list is still live; 8-frame chunks written into a
    # preallocated output cap that transient and drop the whole-batch .contiguous() copy.
    out = torch.empty([frames.shape[0], UP_H, UP_W, 3], dtype=frames.dtype)
    for i0 in range(0, frames.shape[0], 8):
        i1 = min(i0 + 8, frames.shape[0])
        out[i0:i1] = comfy.utils.common_upscale(
            frames[i0:i1].movedim(-1, 1), UP_W, UP_H, "lanczos", "disabled").movedim(1, -1)
    return out


def save_outputs(label, frames, settings):
    """One mp4 + first/middle/last PNGs for artifact `label`."""
    import av

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for index in (0, frames.shape[0] // 2, frames.shape[0] - 1):
        ab_models.save_png(
            OUTPUT_DIR / f"H3_{label}_f{index:03d}.png",
            frames[index:index + 1], settings)

    path = OUTPUT_DIR / f"H3_{label}.mp4"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=FPS)
    stream.width, stream.height = int(frames.shape[2]), int(frames.shape[1])
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "10", "preset": "medium"}
    for frame in frames:
        picture = av.VideoFrame.from_ndarray(
            np.clip(255.0 * frame.cpu().float().numpy(), 0, 255).astype(np.uint8),
            format="rgb24")
        for packet in stream.encode(picture):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    print(f"  wrote {path}", flush=True)


def latent_stats(tag, reference, candidate):
    diff = (candidate.float() - reference.float()).abs()
    print(f"  {tag}: mean|d|={diff.mean().item():.5f} max|d|={diff.max().item():.3f} ref_std={reference.float().std().item():.3f}",
        flush=True)


# ------------------------------------------------------------------ stages

def get_positive(force):
    """t2va conditioning, cached: [[tensor, extras]] straight from the 32B encoder.
    Text-only, so it is canvas-size independent and shared by every refine run."""
    cache = ab_models.cache_path(CACHE_DIR, "h3spike", "cond", WIDTH, HEIGHT, BASE_KEY)
    if cache.is_file() and not force:
        print(f"conditioning from cache: {cache.name}", flush=True)
        return torch.load(cache, weights_only=True)

    clip = ab_models.load_clip(CLIP_NAME, CLIP_TYPE)
    with ab_models.VramProbe() as probe, torch.inference_mode():
        positive = ab_models.encode_prompt(clip, PROMPT)
    print(f"conditioning: {probe}", flush=True)
    del clip
    ab_models.free_gpu()

    positive = [[entry[0].cpu().clone(), {
        k: (v.cpu().clone() if isinstance(v, torch.Tensor) else v)
        for k, v in entry[1].items()}] for entry in positive]
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(positive, cache)
    return positive


def empty_av_latent():
    """EmptyMiniMaxH3LatentAV via the real node internals."""
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent

    latent, frame_count = _empty_av_latent(WIDTH, HEIGHT, LENGTH)
    if frame_count != LENGTH:
        raise RuntimeError(f"LENGTH {LENGTH} not on the 17k+5 grid (snapped to {frame_count})")
    return latent


def render_base(positive, force):
    """Base t2va render, cached: returns (video_latent, audio_latent) on CPU."""
    cache = ab_models.cache_path(CACHE_DIR, "h3spike", "base", WIDTH, HEIGHT, BASE_KEY)
    if cache.is_file() and not force:
        pair = torch.load(cache, weights_only=True)
        print(f"base latents from cache: {cache.name}", flush=True)
        return pair["video"], pair["audio"]

    model = ab_models.load_unet(UNET_NAME)
    guider = build_basic_guider(model, positive)
    sigmas = ab_models.build_sigmas(model, SCHEDULER, STEPS, 1.0)
    sampler = ab_models.build_sampler(SAMPLER)
    noise = ab_models.build_noise(BASE_SEED)
    with ab_models.VramProbe() as probe, torch.inference_mode():
        out = ab_models.sample_custom_advanced(
            noise, guider, sampler, sigmas, empty_av_latent())
    print(f"base sample ({STEPS} steps): {probe}", flush=True)
    del guider, model
    ab_models.free_gpu()

    video, audio = out["samples"].unbind()
    video, audio = video.cpu(), audio.cpu()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"video": video, "audio": audio}, cache)
    print(f"cached base latents: {cache.name}", flush=True)
    return video, audio


def decode_video(vae, video_latent):
    with ab_models.VramProbe() as probe, torch.inference_mode():
        frames = ab_models.vae_decode(vae, {"samples": video_latent})
    print(f"decode: {probe}  -> frames {tuple(frames.shape)}", flush=True)
    return frames


def encode_video(vae, frames):
    with ab_models.VramProbe() as probe, torch.inference_mode():
        latent = vae.encode(frames)
    print(f"encode: {probe}  -> latent {tuple(latent.shape)}", flush=True)
    return latent


def refine(model, positive, video_latent, audio_latent, denoise):
    """Partial denoise of the AV pair with the audio stream frozen."""
    import comfy.nested_tensor

    steps = max(1, round(STEPS * denoise))
    guider = build_basic_guider(model, positive)
    sigmas = ab_models.build_sigmas(model, SCHEDULER, steps, denoise)
    sampler = ab_models.build_sampler(SAMPLER)
    noise = ab_models.build_noise(REFINE_SEED)
    latent = {
        "samples": comfy.nested_tensor.NestedTensor(
            (video_latent.clone(), audio_latent.clone())),
        "noise_mask": frozen_audio_mask(video_latent, audio_latent),
    }
    with ab_models.VramProbe() as probe, torch.inference_mode():
        out = ab_models.sample_custom_advanced(noise, guider, sampler, sigmas, latent)
    print(f"refine d={denoise} ({steps} steps): {probe}", flush=True)
    del guider
    video, _ = out["samples"].unbind()
    return video.cpu()


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--denoise", type=float, default=0.4)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    print(f"ComfyUI root: {ROOT} ({ROOT_NOTE}) version {ab_env.version(ROOT)}", flush=True)
    tag = f"d{args.denoise * 100:02.0f}"
    settings = dict(BASE_KEY, denoise=args.denoise, refine_seed=REFINE_SEED,
                    up_w=UP_W, up_h=UP_H)
    # base/upscaled/up_roundtrip run no refine: they must not carry args.denoise or
    # refine_seed, or a --denoise sweep re-stamps byte-identical files with a new value.
    nosample_settings = dict(BASE_KEY, up_w=UP_W, up_h=UP_H)

    positive = get_positive(args.force)
    base_video, base_audio = render_base(positive, args.force)

    # ---- VAE stages first, with no DiT resident
    vae = ab_models.load_vae(VAE_NAME)
    base_frames = decode_video(vae, base_video)
    save_outputs("base", base_frames, dict(nosample_settings, run_label="base",
                                           denoise=1.0))

    up_frames = upscale_frames(base_frames)
    del base_frames
    save_outputs("upscaled", up_frames, dict(nosample_settings, run_label="upscaled"))

    up_latent = encode_video(vae, up_frames)
    expected = (1, 24, base_video.shape[2], UP_H // 16, UP_W // 16)
    if tuple(up_latent.shape) != expected:
        raise RuntimeError(f"upscaled latent {tuple(up_latent.shape)} != expected {expected}")
    up_latent = up_latent.cpu()
    del up_frames
    ab_models.clear_cache()

    save_outputs("up_roundtrip", decode_video(vae, up_latent),
                 dict(nosample_settings, run_label="up_roundtrip"))

    # ---- both refine variants on one resident DiT
    ab_models.free_gpu()
    model = ab_models.load_unet(UNET_NAME)
    pure_video = refine(model, positive, base_video, base_audio, args.denoise)
    up_video = refine(model, positive, up_latent, base_audio, args.denoise)
    del model
    ab_models.free_gpu()

    latent_stats("refine_up latent vs its input", up_latent, up_video)
    save_outputs(f"refine_pure_{tag}", decode_video(vae, pure_video),
                 dict(settings, run_label="refine_pure"))
    save_outputs(f"refine_up_{tag}", decode_video(vae, up_video),
                 dict(settings, run_label="refine_up"))

    print(f"done. A/B in {OUTPUT_DIR}", flush=True)
    print(json.dumps(settings, indent=1, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
