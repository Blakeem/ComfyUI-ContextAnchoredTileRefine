"""Standalone GPU A/B render harness for Context-Anchored Tile Refine.

Renders the refine stage end to end without a ComfyUI server: models are loaded
directly, the guider/sampler/sigmas are built from the real ComfyUI node bodies, and
sampling.refine_image is called directly. Judging seams needs comparable images, not a
metric, so this exists to produce them cheaply: a full sweep over one image is minutes,
and a code change needs no ComfyUI restart.

Four stages, in dependency order:

    1. conditioning   CLIP -> positive/negative/empty per image, then CLIP freed
    2. base           DiT -> float 768x1024 (cached)
    3. upscale        4x model + lanczos -> float 2304x3072 (cached)
    4. render         one output per RunConfig

Everything runs inside torch.inference_mode(), as ComfyUI does (execution.py). Without
it autograd retains each op's inputs for a backward graph that is never used, which put
the VAE round trip at ~21 GiB and forced ComfyUI's tiled-VAE fallback on every tile.

    <venv-python> tests-AB/run_ab.py            # render every run x every image
    <venv-python> tests-AB/run_ab.py --list
    <venv-python> tests-AB/run_ab.py --image face --force

Outputs go to OUTPUT_DIR as AB_{image_label}__{run_label}.png, each carrying its
full settings in a PNG tEXt chunk. Existing files are left alone unless --force.

=========================== EDIT BELOW TO ADD RUNS ===========================
"""
import argparse
import contextlib
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ------------------------------------------------------------------ CONFIG

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-Images")
CACHE_DIR = Path(__file__).resolve().parent / "cache"

# Models, as named by the workflow's loader nodes. Resolved through folder_paths
# against C:\Users\Blake\Documents\ComfyUI\models (see ab_env.MODEL_BASE_DIR).
UNET_NAME = "ungloryhailZImage_bf16.safetensors"      # node 185, weight_dtype "default"
CLIP_NAME = "qwen_3_4b.safetensors"                   # node 184
CLIP_TYPE = "lumina2"                                 # node 184
VAE_NAME = "ae.safetensors"                           # node 17
UPSCALE_MODEL_NAME = "4xFaceUpDAT.safetensors"        # node 35

# Upscale stage: 4x with the model, then lanczos down to exactly 3x the base.
UPSCALE_MULTIPLIER = 3                                # node 232


@dataclasses.dataclass(frozen=True)
class BaseGen:
    """The BASE generation: nodes 110 -> 38 -> 112.

    Shares nothing with RunConfig on purpose. Every field below differs from the refine
    stage or is unique to it — different seed, sampler, scheduler, denoise. Mixing the two
    silently produces a plausible-looking wrong answer, so they stay in separate dataclasses.

    DELIBERATELY NOT the recorded workflow any more. The owner's graph was thrown together
    for this test (their real pipeline is Chroma -> z-image turbo) and is not what the A/B
    should optimise for; what matters is that it runs fast and consistently, since the seam
    behaviour under test does not depend on the generator. Three changes from the recording:
      scheduler normal -> simple   the correct pairing for z-image turbo (normal suits Chroma)
      PerpNeg -> CFGGuider         3 model calls per step -> 1 at cfg 1.0 (see build_guider)
      LoRA removed                 chroma-flash-heun is a FLUX-1 LoRA (ss_network_module:
                                   networks.lora_flux); on this Z-Image DiT 690 of its 693
                                   tensors fail to bind and 1 of 231 modules applies, so it
                                   only cost load time. Set lora_name to re-enable.
    """
    width: int = 768                 # node 173
    height: int = 1024               # node 174
    batch_size: int = 1              # node 175
    lora_name: str = ""              # was node 65; "" = no LoRA
    lora_strength: float = 1.0       # node 65 strength_model
    seed: int = 19578                # node 39 RandomNoise
    sampler: str = "euler"           # node 55 KSamplerSelect
    scheduler: str = "simple"        # was node 93 "normal"
    steps: int = 9                   # node 123
    denoise: float = 1.0             # node 107
    cfg: float = 1.0                 # node 113
    neg_scale: float = 1.0           # unused by CFGGuider; kept so cache keys stay stable


BASE = BaseGen()


@dataclasses.dataclass(frozen=True)
class ImageSource:
    """One base generation to produce and then refine.

    `filename` is now only a PROMPT source: the positive/negative strings are read out
    of that PNG's `prompt` chunk so the harness renders what the app rendered. Its pixels
    are never used — stage 2 regenerates the base in float from BASE + those prompts."""
    label: str                  # goes in the output filename
    filename: str               # under OUTPUT_DIR; read for its `prompt` chunk only
    positive: str               # fallback; the PNG's `prompt` chunk wins
    negative: str               # fallback; the PNG's `prompt` chunk wins
    reference: str = ""         # existing ComfyUI output to self-check against
    positive_node: str = "160"  # PrimitiveStringMultiline holding the positive prompt
    negative_node: str = "192"  # PrimitiveStringMultiline holding the negative prompt


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """One A/B configuration of the REFINE stage. Defaults are the workflow's node-228
    values, so a new run only spells out what it changes. The model here is the
    UNPATCHED node 185 — see BaseGen for the LoRA'd base-generation settings."""
    label: str                       # goes in the output filename
    context_anchor: int = 32
    context_overlap: int = 32
    max_tile_width: int = 1536       # node 234: base width x2
    max_tile_height: int = 2048      # node 235: base height x2
    seed: int = 42                   # node 229 RandomNoise
    # euler/simple, not the recording's heun/beta: heun costs TWO model calls per step for a
    # marginal quality gain that is irrelevant here — euler leaves comparable grit, which is
    # what makes the seam legible in the first place.
    sampler: str = "euler"           # was node 188 "heun"
    scheduler: str = "simple"        # was node 170 "beta"
    steps: int = 9                   # node 170
    denoise: float = 0.35            # node 170
    cfg: float = 1.0                 # node 183 PerpNegGuider
    neg_scale: float = 1.0           # node 183


IMAGES = [
    # No `reference=`: the ComfyUI renders that used to serve as a fidelity baseline were
    # made with PerpNeg + heun/beta + the chroma-flash LoRA. This harness now runs CFGGuider
    # + euler/simple with no LoRA, so diffing against them would report a large number that
    # means nothing. Set reference= again only if BaseGen/RunConfig are restored to match a
    # specific recorded render.
    ImageSource(
        label="nightsky",
        filename="AB-Test_00001_.png",
        positive="Clear night sky, small full moon with failt nebula and star clusters, high ISO.",
        negative="cartoon, cgi, disproportionate body, jpeg compression, scribbles, AI generated, shredded cloth",
    ),
    ImageSource(
        label="face",
        filename="AB-Test_00002_.png",
        positive="Woman's face close-up, blush, foundation, studio lighting",
        negative="cartoon, cgi, disproportionate body, jpeg compression, scribbles, AI generated, shredded cloth",
    ),
]

RUNS = [
    # The shipped configuration. Add entries to compare against it; each renders one image
    # per IMAGES entry as AB_{image_label}__{run_label}.png.
    RunConfig(label="baseline"),
]

# The run whose settings match what was already rendered in ComfyUI, so its output
# can be diffed against ImageSource.reference as a fidelity check.
REFERENCE_RUN = "baseline"

# =========================== EDIT ABOVE TO ADD RUNS ===========================

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_env  # noqa: E402  (must precede every comfy import)
import ab_models  # noqa: E402


def output_path(image, run):
    return OUTPUT_DIR / "AB_{}__{}.png".format(image.label, run.label)


def upscale_size():
    """The exact refine input size: nodes 225/226, base dimensions x node 232."""
    return BASE.width * UPSCALE_MULTIPLIER, BASE.height * UPSCALE_MULTIPLIER


def cache_paths(image, prompts):
    """(base, upscale) float-cache targets for one image.

    The upscale key NESTS the base key, so anything that changes the base — a prompt
    edit, a seed, a LoRA — renames both files rather than pairing a fresh base with a
    stale upscale."""
    positive, negative, _ = prompts[image.label]
    base_key = dict(dataclasses.asdict(BASE),
                    unet=UNET_NAME, vae=VAE_NAME, clip=CLIP_NAME, clip_type=CLIP_TYPE,
                    guider="PerpNegGuider", positive=positive, negative=negative, empty="")
    upscale_key = {"base": base_key, "upscale_model": UPSCALE_MODEL_NAME,
                   "multiplier": UPSCALE_MULTIPLIER}
    target_w, target_h = upscale_size()
    return (
        ab_models.cache_path(CACHE_DIR, image.label, "base", BASE.width, BASE.height, base_key),
        ab_models.cache_path(CACHE_DIR, image.label, "upscale", target_w, target_h, upscale_key),
    )


class StageClock:
    """Wall clock per stage, collected for the end-of-run summary."""

    def __init__(self):
        self.entries = []

    @contextlib.contextmanager
    def stage(self, label):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.entries.append((label, time.perf_counter() - started))

    def report(self):
        for label, seconds in self.entries:
            print("[time]    {:<14} {:7.1f}s".format(label, seconds))
        print("[time]    {:<14} {:7.1f}s".format("TOTAL", sum(s for _, s in self.entries)))


def patch_broken_encode_tiled():
    """Make ComfyUI's VAE.encode_tiled_ actually run. Upstream bug, comfy/sd.py.

    It builds `samples` from tiled_scale (an inference-mode tensor) and then does
    `samples += tiled_scale(...)`, which raises "Inplace update to inference tensor outside
    InferenceMode". decode_tiled_ sums out-of-place and so survives — which is why only the
    ENCODE fallback ever kills a run.

    We need the fallback to work, not to avoid it. Measured on this card: the VAE round trip
    at 1216x1600 reaches ~21 GiB of live allocation against 24 GB total, so ComfyUI takes the
    tiled path here — and takes it in the desktop app too, at the same tile size on the same
    GPU. Suppressing it would make the A/B measure something the production pipeline never
    does. Rewrites the three accumulations out-of-place, changing nothing else.
    """
    import comfy.sd
    import comfy.utils

    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        steps = pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap)
        steps += pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x * 2, tile_y // 2, overlap)
        pbar = comfy.utils.ProgressBar(steps)

        encode_fn = lambda a: self.first_stage_model.encode((self.process_input(a)).to(self.vae_dtype).to(self.device)).to(dtype=self.vae_output_dtype())  # noqa: E731
        common = dict(upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels,
                      output_device=self.output_device, pbar=pbar)
        samples = comfy.utils.tiled_scale(pixel_samples, encode_fn, tile_x, tile_y, overlap, **common)
        samples = samples + comfy.utils.tiled_scale(pixel_samples, encode_fn, tile_x * 2, tile_y // 2, overlap, **common)
        samples = samples + comfy.utils.tiled_scale(pixel_samples, encode_fn, tile_x // 2, tile_y * 2, overlap, **common)
        return samples / 3.0

    comfy.sd.VAE.encode_tiled_ = encode_tiled_
    print("[env]     patched VAE.encode_tiled_ (upstream inplace-on-inference-tensor bug)")


# ------------------------------------------------------------------ stages

def stage_conditioning(images):
    """Stage 1. Encode every prompt once, then get the 8 GB text encoder off the card
    before any diffusion model is loaded. Each image needs positive/negative; the empty
    conditioning PerpNeg wants is prompt-independent and is encoded once.

    Runs FIRST now, because the base generation (stage 2) consumes these conds."""
    print("[clip]    loading {} ({})".format(CLIP_NAME, CLIP_TYPE))
    with ab_models.VramProbe() as probe:
        clip = ab_models.load_clip(CLIP_NAME, CLIP_TYPE)
        empty = ab_models.encode_prompt(clip, "")
        conds = {}
        prompts = {}
        for image in images:
            source = OUTPUT_DIR / image.filename
            positive, negative = ab_models.read_prompts(source, image.positive_node, image.negative_node)
            origin = "PNG prompt chunk"
            if positive is None or negative is None:
                positive, negative = image.positive, image.negative
                origin = "hardcoded fallback"
            prompts[image.label] = (positive, negative, origin)
            conds[image.label] = (
                ab_models.encode_prompt(clip, positive),
                ab_models.encode_prompt(clip, negative),
            )
            print("[clip]    {:<9} prompts from {}: {!r}".format(image.label, origin, positive[:60]))
        del clip
    ab_models.free_gpu()
    print("[clip]    encoded and released  {}".format(probe))
    return conds, empty, prompts


def generate_base(model, vae, conds, empty):
    """Nodes 110 -> 38 -> 112 for one image: EmptySD3LatentImage, SamplerCustomAdvanced,
    VAEDecode. `model` must be the LoRA-PATCHED clone (node 65), and the sigmas come off
    that same patched model because node 93's model input is node 65, not node 185."""
    positive, negative = conds
    guider = ab_models.build_guider(model, positive, negative, empty, BASE.cfg, BASE.neg_scale)
    sampler = ab_models.build_sampler(BASE.sampler)
    sigmas = ab_models.build_sigmas(model, BASE.scheduler, BASE.steps, BASE.denoise)
    noise = ab_models.build_noise(BASE.seed)
    latent = ab_models.empty_sd3_latent(BASE.width, BASE.height, BASE.batch_size)

    sampled = ab_models.sample_custom_advanced(noise, guider, sampler, sigmas, latent)
    return ab_models.vae_decode(vae, sampled)


def stage_base(images, conds, empty, prompts, force):
    """Stage 2. Reproduce the base generation in FLOAT and cache it.

    Returns (bases, model, vae). `model` is the UNPATCHED node-185 patcher, handed back so
    the render stage does not re-read 11.5 GB from disk; it is None when every base was a
    cache hit and no DiT was needed at all."""
    bases = {}
    pending = []
    for image in images:
        cached, _ = cache_paths(image, prompts)
        if cached.is_file() and not force:
            tensor = torch.load(cached, map_location="cpu")
            ab_models.require_image_shape(tensor, BASE.height, BASE.width,
                                          "cached base {}".format(image.label))
            bases[image.label] = tensor
            print("[base]    {:<9} cache hit  {}".format(image.label, cached.name))
        else:
            pending.append((image, cached))

    if not pending:
        print("[base]    every base cached; DiT not loaded here")
        return bases, None, None

    print("[base]    loading {}{}".format(
        UNET_NAME, " + LoRA {} @ {}".format(BASE.lora_name, BASE.lora_strength) if BASE.lora_name else " (no LoRA)"))
    with ab_models.VramProbe() as probe:
        model = ab_models.load_unet(UNET_NAME)
        vae = ab_models.load_vae(VAE_NAME)
        # No LoRA -> generate from the same patcher the refine uses.
        patched = ab_models.load_lora_model_only(model, BASE.lora_name, BASE.lora_strength) if BASE.lora_name else model
    print("[base]    loaded  {}".format(probe))
    fingerprint_before = ab_models.weight_fingerprint(model)

    for image, cached in pending:
        print("[base]    {:<9} {}x{} seed {} {}/{} steps {} denoise {}".format(
            image.label, BASE.width, BASE.height, BASE.seed,
            BASE.sampler, BASE.scheduler, BASE.steps, BASE.denoise))
        with ab_models.VramProbe() as probe:
            with torch.inference_mode():
                pixels = generate_base(patched, vae, conds[image.label], empty)
        ab_models.require_image_shape(pixels, BASE.height, BASE.width, "base {}".format(image.label))
        # detach() changes no values. The app runs the whole graph inside
        # torch.inference_mode() (execution.py), so its VAEDecode output carries no
        # autograd graph; this harness does not, and caching a grad-tracking tensor would
        # make the upscale model build a backward graph over a 3072x4096 intermediate.
        pixels = pixels.detach().cpu()
        stats = pixels.float()
        print("[base]    {:<9} done  {}  dtype {}  min {:.4f} max {:.4f} mean {:.4f}".format(
            image.label, probe, pixels.dtype,
            float(stats.min()), float(stats.max()), float(stats.mean())))
        if not torch.isfinite(stats).all():
            raise RuntimeError("base {}: non-finite pixels".format(image.label))
        cached.parent.mkdir(parents=True, exist_ok=True)
        torch.save(pixels, cached)
        bases[image.label] = pixels
        print("[base]    {:<9} cached  {}".format(image.label, cached.name))

    # The LoRA'd clone shares node 185's nn.Module; dropping it and unloading restores the
    # base weights, exactly as ComfyUI does between the two branches of the graph.
    if patched is not model:
        del patched
    ab_models.free_gpu()
    fingerprint_after = ab_models.weight_fingerprint(model)
    drift = abs(fingerprint_after - fingerprint_before)
    print("[base]    LoRA released; DiT weight fingerprint {:.6e} -> {:.6e} (drift {:.3e})".format(
        fingerprint_before, fingerprint_after, drift))
    if drift > 1e-6 * max(abs(fingerprint_before), 1.0):
        raise RuntimeError("the LoRA wrote through to the shared DiT weights; refine would "
                           "run a patched model (fingerprint drift {:.3e})".format(drift))
    return bases, model, vae


def stage_upscale(images, bases, prompts, force, env_note):
    """Stage 3. Float base -> float 2304x3072, cached, with the upscale model released
    afterwards. It peaks at a 3072x4096 intermediate, so it must not share VRAM with the
    DiT/CLIP; the cache also makes seam iteration cheap, since the upscale is identical
    for every run."""
    target_w, target_h = upscale_size()
    upscaled = {}
    for image in images:
        _, cached = cache_paths(image, prompts)
        if cached.is_file() and not force:
            result = torch.load(cached, map_location="cpu")
            print("[upscale] {:<9} cache hit  {}".format(image.label, cached.name))
        else:
            base = bases[image.label]
            print("[upscale] {:<9} {}x{} -> 4x {} -> lanczos {}x{}".format(
                image.label, base.shape[2], base.shape[1], UPSCALE_MODEL_NAME, target_w, target_h))
            with ab_models.VramProbe() as probe:
                result = ab_models.upscale_to(base, UPSCALE_MODEL_NAME, target_w, target_h).cpu()
            cached.parent.mkdir(parents=True, exist_ok=True)
            torch.save(result, cached)
            print("[upscale] {:<9} done  {}  -> {}".format(image.label, probe, cached.name))
        ab_models.require_image_shape(result, target_h, target_w, "upscale {}".format(image.label))
        upscaled[image.label] = result

    ab_models.free_gpu()
    print("[upscale] upscale model released ({})".format(env_note))
    return upscaled


def render(image, run, pixels, conds, empty, model, vae, settings):
    """Stage 4. One config against one image: build the guider/sampler/sigmas exactly as
    the workflow's nodes do, then call refine_image directly (the node class has no
    seam_blend widget). `model` is the UNPATCHED node-185 patcher."""
    from context_anchored_tile_refine import sampling

    target_w, target_h = upscale_size()
    positive, negative = conds[image.label]
    guider = ab_models.build_guider(model, positive, negative, empty, run.cfg, run.neg_scale)
    sampler = ab_models.build_sampler(run.sampler)
    sigmas = ab_models.build_sigmas(model, run.scheduler, run.steps, run.denoise)
    noise = ab_models.build_noise(run.seed)

    # inference_mode mirrors ComfyUI (execution.py wraps every prompt in it) and is not
    # cosmetic: without it autograd retains each op's inputs for a backward graph that is
    # never used, which is why the VAE round trip measured ~21 GiB here and is comfortable
    # in the desktop app at the identical tile size. It also makes the upstream
    # encode_tiled_ bug moot, since in-place on an inference tensor is legal INSIDE
    # inference mode — that patch stays as a belt-and-braces guard.
    with ab_models.VramProbe() as probe, torch.inference_mode():
        result = sampling.refine_image(
            pixels, guider, sampler, sigmas, vae, noise,
            max_tile_width=run.max_tile_width,
            max_tile_height=run.max_tile_height,
            context_anchor=run.context_anchor,
            context_overlap=run.context_overlap,
        )

    # refine_image must be size-preserving; verify before the tensor reaches disk.
    ab_models.require_image_shape(result, target_h, target_w,
                                  "refine {} {}".format(image.label, run.label))
    destination = output_path(image, run)
    written = ab_models.save_png(destination, result.cpu(), settings)
    print("[render]  {:<9} {:<20} {}  -> {} {}x{}".format(
        image.label, run.label, probe, destination.name, written[0], written[1]))
    return probe


def build_settings(image, run, prompts, pixels, env):
    positive, negative, origin = prompts[image.label]
    payload = dataclasses.asdict(run)
    payload.update({
        "image_label": image.label,
        "run_label": run.label,
        "prompt_source_png": image.filename,
        "base_generation": dataclasses.asdict(BASE),
        "base_note": "regenerated in float from nodes 110/38/112 with the LoRA-patched "
                     "model; no 8-bit PNG round trip between base and upscale",
        "base_size": "{}x{}".format(BASE.width, BASE.height),
        "input_size": "{}x{}".format(pixels.shape[2], pixels.shape[1]),
        "upscale_model": UPSCALE_MODEL_NAME,
        "upscale_multiplier": UPSCALE_MULTIPLIER,
        "upscale_note": "4x with model (tiled 512/32), then lanczos common_upscale to {}x".format(UPSCALE_MULTIPLIER),
        "unet": UNET_NAME,
        "clip": CLIP_NAME,
        "clip_type": CLIP_TYPE,
        "vae": VAE_NAME,
        "guider": "PerpNegGuider",
        "positive_prompt": positive,
        "negative_prompt": negative,
        "empty_conditioning": "",
        "prompt_source": origin,
        "comfyui_root": str(env[0]),
        "comfyui_version": env[1],
        "harness": "tests-AB/run_ab.py",
        "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return payload


# ------------------------------------------------------------------ self-check

def self_check(images, runs):
    """Diff the REFERENCE_RUN outputs against the PNGs ComfyUI already produced with the
    same settings.

    Now that the base is regenerated in float, every stage of the graph is reproduced and
    the remaining difference is only non-determinism ComfyUI itself has: cuDNN/cuBLAS
    kernel selection, the tiled-VAE path's memory-dependent batch_number, and allocator
    state. Still a closeness check rather than equality — but a much tighter one, and a
    regression here means a real divergence, not quantization."""
    if not any(run.label == REFERENCE_RUN for run in runs):
        print("[check]   {} not in this render set — skipped".format(REFERENCE_RUN))
        return
    run = next(r for r in runs if r.label == REFERENCE_RUN)
    print("\n[check]   {} vs the existing ComfyUI outputs".format(REFERENCE_RUN))
    for image in images:
        if not image.reference:
            continue
        reference = OUTPUT_DIR / image.reference
        produced = output_path(image, run)
        if not reference.is_file() or not produced.is_file():
            print("[check]   {:<9} missing ({} / {})".format(image.label, reference.name, produced.name))
            continue
        left = np.asarray(Image.open(reference).convert("RGB")).astype(np.int16)
        right = np.asarray(Image.open(produced).convert("RGB")).astype(np.int16)
        if left.shape != right.shape:
            print("[check]   {:<9} shape mismatch {} vs {}".format(image.label, left.shape, right.shape))
            continue
        diff = np.abs(left - right)
        print("[check]   {:<9} vs {}  size {}x{}".format(image.label, reference.name, left.shape[1], left.shape[0]))
        for index, channel in enumerate("RGB"):
            plane = diff[..., index]
            print("[check]     {}  mean {:7.4f}/255  max {:3d}/255  p99 {:3.0f}  pct>2 {:5.2f}%".format(
                channel, plane.mean(), int(plane.max()), np.percentile(plane, 99),
                100.0 * (plane > 2).mean()))
        print("[check]     all mean {:.4f}/255 ({:.6f} normalized)  max {}/255".format(
            diff.mean(), diff.mean() / 255.0, int(diff.max())))


# ------------------------------------------------------------------ main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", action="append", default=[], metavar="RUN_LABEL",
                        help="render only these run labels (repeatable)")
    parser.add_argument("--image", action="append", default=[], metavar="IMAGE_LABEL",
                        help="render only these image labels (repeatable)")
    parser.add_argument("--force", action="store_true",
                        help="re-render outputs that already exist")
    parser.add_argument("--force-upscale", action="store_true",
                        help="rebuild the cached upscale even if it is present")
    parser.add_argument("--force-base", action="store_true",
                        help="rebuild the cached float base generation (implies --force-upscale)")
    parser.add_argument("--list", action="store_true", help="show the configured runs and exit")
    args = parser.parse_args(argv)

    runs = [run for run in RUNS if not args.only or run.label in args.only]
    images = [image for image in IMAGES if not args.image or image.label in args.image]
    unknown = set(args.only) - {run.label for run in RUNS}
    if unknown:
        raise SystemExit("unknown run label(s): {}".format(", ".join(sorted(unknown))))
    if not runs or not images:
        raise SystemExit("nothing selected")

    if args.list:
        for run in runs:
            print(json.dumps(dataclasses.asdict(run), sort_keys=True))
        for image in images:
            print(image.label, "->", ", ".join(output_path(image, run).name for run in runs))
        return 0

    root, note = ab_env.bootstrap()
    version = ab_env.version(root)
    print("[env]     ComfyUI {} at {}  ({})".format(version, root, note))
    patch_broken_encode_tiled()
    print("[env]     torch {}  cuda {}  device {}".format(
        torch.__version__, torch.version.cuda,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))

    clock = StageClock()
    target_w, target_h = upscale_size()

    with clock.stage("conditioning"):
        conds, empty, prompts = stage_conditioning(images)
    with clock.stage("base"):
        bases, model, vae = stage_base(images, conds, empty, prompts, args.force_base)
    with clock.stage("upscale"):
        # A forced base rebuild invalidates the upscale even when the key is unchanged.
        upscaled = stage_upscale(images, bases, prompts,
                                 args.force_upscale or args.force_base, note)
    del bases

    if model is None:
        print("[unet]    loading {}".format(UNET_NAME))
        with clock.stage("unet load"), ab_models.VramProbe() as probe:
            model = ab_models.load_unet(UNET_NAME)
            vae = ab_models.load_vae(VAE_NAME)
        print("[unet]    loaded  {}".format(probe))
    else:
        print("[unet]    reusing the base stage's UNPATCHED patcher (node 185 feeds both branches)")

    # Load once, render every config against it: the pipeline is the only thing
    # under test, so the models must not be rebuilt between runs.
    rendered = []
    with clock.stage("render"):
        for image in images:
            pixels = upscaled[image.label]
            for run in runs:
                destination = output_path(image, run)
                if destination.is_file() and not args.force:
                    print("[render]  {:<9} {:<20} exists, skipped (--force to redo)".format(image.label, run.label))
                    rendered.append((image, run, None))
                    continue
                settings = build_settings(image, run, prompts, pixels, (root, version))
                # Defragment the allocator but KEEP the UNET resident: unloading it here
                # would re-upload 11.5 GB across PCIe before every single run, which is
                # pure overhead when consecutive runs share the same model set.
                ab_models.clear_cache()
                probe = render(image, run, pixels, conds, empty, model, vae, settings)
                rendered.append((image, run, probe))

    # Every output is re-measured OFF DISK against the one size the refine may produce.
    print("\n[done]    {} output(s)".format(len(rendered)))
    bad = []
    for image, run, probe in rendered:
        destination = output_path(image, run)
        if not destination.is_file():
            bad.append("{} MISSING".format(destination.name))
            print("[done]    {:<60} {:<12} {}".format(str(destination), "MISSING", probe or "(skipped)"))
            continue
        width, height = ab_models.png_size(destination)
        verdict = "OK" if (width, height) == (target_w, target_h) else "WRONG SIZE"
        if verdict != "OK":
            bad.append("{} is {}x{}, expected {}x{}".format(destination.name, width, height, target_w, target_h))
        print("[done]    {:<60} {:>4}x{:<4} {:<10} {}".format(
            str(destination), width, height, verdict, probe or "(skipped)"))

    clock.report()
    self_check(images, runs)
    if bad:
        raise SystemExit("[done]    FAILED size verification: " + "; ".join(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
