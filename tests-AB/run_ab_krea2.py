"""Krea 2 conditioning A/B at production settings: does low-denoise refine need text?

Renders FOUR arms of the refine stage from ONE frozen base, all at the exact settings
of output\\ComfyUI-2x_00676_.png (node 364: seed 42, exp_heun_2_x0_sde / sgm_uniform,
28 steps, cfg 3.5, denoise 0.35, upscale 3x via 4xFaceUpDAT + lanczos, tile caps
1536x2048, anchor 256, overlap 32). The ONLY variable across arms is each tile's
positive conditioning:

    1-text-prompt       the pre-VL standard: every tile shares the full original
                        prompt (base ContextAnchoredTileRefine usage; AB01 design)
    2-vl-slices         the production VL path untouched: per-tile row slices of one
                        whole-canvas vision encode, no text (AB29; ships in vl.py)
    3-tile-captions     no slices: each tile's positive is a same-encoder caption of
                        that tile's crop, re-encoded as plain text (AB27 e_caption)
    4-vl-plus-captions  slices AND the same caption riding as text rows after each
                        tile's slice (AB32 f_gs_caption design)

Arms 3 and 4 share IDENTICAL captions (one greedy generation per tile, cached), so
the four arms form a clean 2x2 factorial — {vision slices on/off} x {caption text
on/off} — with base pixels, upscale, noise, sigmas, sampler, negative, and tile grid
held identical. Deviation from the historical AB32, deliberate: captions use
CAPTION_INSTRUCTION (AB27's instruction, Krea 2's own conditioning register) rather
than DETAIL_INSTRUCTION, so the caption text is the same object in both caption arms.

The base is tests-AB/inputs/krea2-00676-base-768x1024.png — the app's own saved base
(rescued from the volatile temp previews; see inputs/README.md). A fresh run of the
gen chain produces a DIFFERENT composition (reconfirmed 2026-08-05), so nothing here
re-runs it. The upscale stage runs once through upscale.prepare_upscaled and is
cached; every arm refines the same float canvas.

    <venv-python> tests-AB/run_ab_krea2.py                    # all four arms
    <venv-python> tests-AB/run_ab_krea2.py --list
    <venv-python> tests-AB/run_ab_krea2.py --only 2-vl-slices --force

Outputs: output\\AB-Test-Images\\AB_00676-d035__{arm}.png plus a labeled contact
sheet, each PNG carrying its full settings (and captions) in a `catr_ab` tEXt chunk.
"""
import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ------------------------------------------------------------------ CONFIG

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Test-Images")
CACHE_DIR = Path(__file__).resolve().parent / "cache"
INPUTS_DIR = Path(__file__).resolve().parent / "inputs"

BASE_PNG = INPUTS_DIR / "krea2-00676-base-768x1024.png"
REFERENCE_OUTPUT = Path(r"C:\Users\Blake\Documents\ComfyUI\output\ComfyUI-2x_00676_.png")
UPSCALE_SNAPSHOT = Path(r"C:\Users\Blake\Documents\ComfyUI\input\AB-refine-input.png")

# The one live Krea2-capable tree (ab_env's desktop candidates were deleted). Set
# before the ab_env import so bootstrap resolves here; spike_h3.py's convention.
KREA2_ROOT = Path(r"C:\Users\Blake\ComfyUI-Installs\ComfyUI\ComfyUI")

UNET_NAME = "krea2Center_v408bitFp8Int8.safetensors"   # node 283
CLIP_NAME = "qwen3-vl-4b-heretic_int8.safetensors"     # node 294
CLIP_TYPE = "krea2"                                    # node 294
VAE_NAME = "krea2RealVae_v10.safetensors"              # node 295
UPSCALE_MODEL_NAME = "4xFaceUpDAT.safetensors"         # node 325

# Nodes 344 / 368 of the reference PNG, verbatim. The negative is the REFINE node's
# (it appends ", freckles" to the base generation's negative).
POSITIVE_PROMPT = (
    "breathtaking, striking cinematic dark fantasy, low angle perspective, dream girl "
    "fantasy character, strangely attractive face, striking otherworldly beauty, long "
    "hair in loose waves, dense enchanted woodland filled with towering twisted oaks "
    "and thick undergrowth, forest floor covered in blood red flowers, ancient "
    "crumbling stone ruins half-swallowed by ivy, night sky dominated by a massive "
    "blood-red full moon with faint nebula streaks visible through the forest canopy, "
    "light mist, dramatic chiaroscuro lighting, deep crimson, obsidian black, rich "
    "violet, gold accents, ethereal, ominous, mesmerizing, intensely atmospheric."
)
NEGATIVE_PROMPT = ("undead, zombie, jpeg compression, scribbles, AI generated, "
                   "plastic toy doll, shredded cloth, freckles")


@dataclasses.dataclass(frozen=True)
class Refine:
    """Node 364's widgets in the reference PNG. Shared by every arm."""
    seed: int = 42
    sampler: str = "exp_heun_2_x0_sde"
    scheduler: str = "sgm_uniform"
    steps: int = 28
    cfg: float = 3.5
    denoise: float = 0.35
    upscale_by: float = 3.0
    max_tile_width: int = 1536
    max_tile_height: int = 2048
    context_anchor: int = 256
    context_overlap: int = 32


REFINE = Refine()

# AB27's caption instruction: Krea 2's own conditioning-template register, so the
# generated text lands in the description style the DiT was trained to read.
CAPTION_INSTRUCTION = ("Describe the image by detailing the color, shape, size, texture, "
                       "quantity, text, spatial relationships of the objects and background.")

ARMS = {
    "1-text-prompt": "full original prompt shared by every tile; no VL slices",
    "2-vl-slices": "production VL path: per-tile vision slices, no text",
    "3-tile-captions": "per-tile VLM caption as plain-text positive; no VL slices",
    "4-vl-plus-captions": "per-tile vision slices + the same caption as text rows",
}

RUN_TAG = "00676-d035"

# ------------------------------------------------------------------ bootstrap

sys.path.insert(0, str(Path(__file__).resolve().parent))
if "COMFYUI_ROOT" not in os.environ:
    if not (KREA2_ROOT / "comfy" / "text_encoders" / "krea2.py").is_file():
        raise SystemExit("No Krea 2 support at {} — set COMFYUI_ROOT".format(KREA2_ROOT))
    os.environ["COMFYUI_ROOT"] = str(KREA2_ROOT)
import ab_env  # noqa: E402  (must precede every comfy import)
import ab_models  # noqa: E402


def _file_sha(path, chars=12):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:chars]


def output_path(arm):
    return OUTPUT_DIR / "AB_{}__{}.png".format(RUN_TAG, arm)


# ------------------------------------------------------------------ stages

def load_base():
    """The app's saved base, exactly as LoadImage would hand it over: RGB / 255."""
    with Image.open(BASE_PNG) as handle:
        arr = np.asarray(handle.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)[None]
    ab_models.require_image_shape(tensor, 1024, 768, "base png")
    return tensor


def load_upscale_model():
    """UpscaleModelLoader.execute body (a V3 io node now, so its NodeOutput return is
    not indexable; the body is reproduced instead, including the .patcher wrap that
    upscale.prepare_upscaled keys residency on)."""
    import comfy.model_management
    import comfy.model_patcher
    import comfy.utils
    import folder_paths
    from comfy_extras.nodes_upscale_model import ImageModelDescriptor, ModelLoader

    model_path = folder_paths.get_full_path_or_raise("upscale_models", UPSCALE_MODEL_NAME)
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
    out = ModelLoader().load_from_state_dict(sd).eval()
    if not isinstance(out, ImageModelDescriptor):
        raise RuntimeError("Upscale model must be a single-image model.")
    out.patcher = comfy.model_patcher.CoreModelPatcher(
        out.model, load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device())
    return out


def stage_upscale(base, force):
    """The node's own whole-image upscale stage, cached: 4x model pass, then one
    lanczos to exactly 3x. Every arm refines this same float canvas."""
    from context_anchored_tile_refine import upscale

    target_w, target_h = upscale.scale_target(int(base.shape[2]), int(base.shape[1]), REFINE.upscale_by)
    key = {"base_sha": _file_sha(BASE_PNG), "model": UPSCALE_MODEL_NAME,
           "upscale_by": REFINE.upscale_by, "stage": "prepare_upscaled"}
    cached = ab_models.cache_path(CACHE_DIR, "krea2-00676", "upscale", target_w, target_h, key)

    if cached.is_file() and not force:
        canvas = torch.load(cached, map_location="cpu")
        print("[upscale] cache hit  {}".format(cached.name))
    else:
        print("[upscale] {}x{} -> 4x {} -> lanczos {}x{}".format(
            base.shape[2], base.shape[1], UPSCALE_MODEL_NAME, target_w, target_h))
        upscale_model = load_upscale_model()
        try:
            with ab_models.VramProbe() as probe, torch.inference_mode():
                canvas = upscale.prepare_upscaled(base, upscale_model, REFINE.upscale_by)
        finally:
            del upscale_model
            ab_models.free_gpu()
        canvas = canvas.detach().float().cpu()
        cached.parent.mkdir(parents=True, exist_ok=True)
        torch.save(canvas, cached)
        print("[upscale] done  {}  -> {}".format(probe, cached.name))
    ab_models.require_image_shape(canvas, target_h, target_w, "upscaled canvas")

    # Informational cross-check against the app-era refine-input snapshot: same base,
    # same chain, so anything beyond quantization noise means the chain diverged.
    if UPSCALE_SNAPSHOT.is_file():
        snap = np.asarray(Image.open(UPSCALE_SNAPSHOT).convert("RGB")).astype(np.float32)
        ours = np.asarray(Image.fromarray(ab_models.to_uint8(canvas))).astype(np.float32)
        if snap.shape == ours.shape:
            diff = np.abs(snap - ours)
            print("[upscale] vs {}: meanabs {:.3f}/255  max {:.0f}/255".format(
                UPSCALE_SNAPSHOT.name, diff.mean(), diff.max()))
        else:
            print("[upscale] snapshot size mismatch {} vs {} — check skipped".format(snap.shape, ours.shape))
    return canvas


def solve_tiles(canvas):
    """The refine's own layout, computed up front so captions and arm-4 slices are
    built from the exact tiles the sampler will use."""
    from context_anchored_tile_refine import grid, sampling

    pixels = canvas[..., :3]
    padded, _ = sampling.pad_image_to_multiple(pixels)
    sx = grid.solve_axis(padded.shape[2], REFINE.max_tile_width, REFINE.context_anchor, REFINE.context_overlap)
    sy = grid.solve_axis(padded.shape[1], REFINE.max_tile_height, REFINE.context_anchor, REFINE.context_overlap)
    layout = grid.build_layout(padded.shape[2], padded.shape[1], sx, sy, REFINE.context_anchor, REFINE.context_overlap)
    print("[layout]  canvas {}x{}  grid {}x{}  ({} tiles)".format(
        padded.shape[2], padded.shape[1], sx.n, sy.n, len(layout.tiles)))
    return padded, layout.tiles


def _rects(tiles):
    return [[t.crop_rect.x0, t.crop_rect.y0, t.crop_rect.x1, t.crop_rect.y1] for t in tiles]


def resample_for_vl(tile_pixels):
    """AB27's caption input prep (EditPlus's recipe): area-resample a COPY of the
    tile's crop to 384^2 total pixels. Conditioning-side only."""
    import math

    import comfy.utils

    samples = tile_pixels.movedim(-1, 1)
    scale_by = math.sqrt(384 * 384 / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    resampled = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    return resampled.movedim(1, -1)[:, :, :, :3]


def generate_caption(clip, vl_input):
    """exp/hallucination-ab generate_text (c749e8b) with the instruction pinned:
    greedy decode, then the settled fallback chain for stop-token-degenerate crops."""
    tokens = clip.tokenize(CAPTION_INSTRUCTION, images=[vl_input], thinking=False)
    ids = clip.generate(tokens, do_sample=False, max_length=512, repetition_penalty=1.05)
    text = clip.decode(ids).strip()
    if not text:
        print("[caption] empty answer (raw: {!r}); trying fallbacks".format(
            clip.decode(ids, skip_special_tokens=False)), flush=True)
        ids = clip.generate(tokens, do_sample=True, max_length=512, temperature=0.7,
                            top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, seed=42)
        text = clip.decode(ids).strip()
    if not text:
        retokens = clip.tokenize("Write one short sentence describing this image.",
                                 images=[vl_input], thinking=False)
        ids = clip.generate(retokens, do_sample=False, max_length=512, repetition_penalty=1.05)
        text = clip.decode(ids).strip()
    if not text:
        raise RuntimeError("caption generation: empty answer after fallbacks (raw: {!r})".format(
            clip.decode(ids, skip_special_tokens=False)))
    return text


def load_or_generate_captions(clip, padded, tiles, force):
    """One caption per tile from the FROZEN upscaled canvas, cached to JSON so arms 3
    and 4 (and any rerun) share the identical text."""
    key = {"base_sha": _file_sha(BASE_PNG), "rects": _rects(tiles),
           "instruction": CAPTION_INSTRUCTION, "clip": CLIP_NAME}
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    cached = CACHE_DIR / "krea2-00676_captions_{}.json".format(digest)

    if cached.is_file() and not force:
        captions = json.loads(cached.read_text(encoding="utf-8"))["captions"]
        print("[caption] cache hit  {}".format(cached.name))
    else:
        captions = []
        for tile_idx, tile in enumerate(tiles):
            crop = tile.crop_rect
            vl_input = resample_for_vl(padded[:, crop.y0:crop.y1, crop.x0:crop.x1, :])
            with torch.inference_mode():
                caption = generate_caption(clip, vl_input)
            captions.append(caption)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(dict(key, captions=captions), indent=1), encoding="utf-8")
        print("[caption] generated and cached  {}".format(cached.name))
    for tile_idx, caption in enumerate(captions):
        print("[caption] tile {}: {}".format(tile_idx, caption))
    return captions


def build_caption_conds(clip, captions):
    """Arm 3 (AB27): each caption re-encoded text-only — Krea 2's own conditioning
    template, exactly what CLIPTextEncode would produce for that string."""
    from context_anchored_tile_refine import vl

    return [vl._convert(clip.encode_from_tokens_scheduled(clip.tokenize(c))) for c in captions]


def build_slice_caption_conds(clip, padded, tiles, captions):
    """Arm 4 (AB32): the caption tokens ride INSIDE the whole-canvas vision encode
    (one encode per distinct caption), and each tile keeps its slice rows plus the
    caption + template tail. Reuses the production vl helpers so the slice math is
    the shipped code's; the seq fail-fast mirrors vl._encode_canvas."""
    from context_anchored_tile_refine import vl

    copy, enc_h, enc_w = vl.resample_for_global(padded)
    grid_h, grid_w = enc_h // vl.MERGED_CELL, enc_w // vl.MERGED_CELL
    n_rows = grid_h * grid_w
    canvas_h, canvas_w = int(padded.shape[1]), int(padded.shape[2])

    cache = {}
    conds = []
    for tile, caption in zip(tiles, captions):
        if caption not in cache:
            tokens = clip.tokenize(vl.VISION_BLOCK + caption, images=[copy], llama_template=vl.KREA2_TEMPLATE)
            ids = [t[0] for t in tokens[next(iter(tokens))][0]]
            pad_pos = next(i for i, v in enumerate(ids) if isinstance(v, dict))
            tail_len = len(ids) - (pad_pos + 2)
            expected_seq = 1 + n_rows + 1 + tail_len
            encoded = clip.encode_from_tokens_scheduled(tokens)
            seq = encoded[0][0].shape[1]
            if seq != expected_seq:
                raise RuntimeError(
                    "slice+caption encode has {} rows, expected {} (grid {}x{} + caption/tail {})".format(
                        seq, expected_seq, grid_h, grid_w, tail_len))
            cache[caption] = (encoded, expected_seq)
        encoded, expected_seq = cache[caption]

        indices = vl.slice_indices(tile.crop_rect, canvas_h, canvas_w, enc_h, enc_w, expected_seq)
        sliced = []
        for entry in encoded:
            tensor, extras = entry[0], dict(entry[1])
            extras.pop("attention_mask", None)
            index = torch.tensor(indices, device=tensor.device)
            sliced.append([tensor.index_select(1, index), extras])
        conds.append(vl._convert(sliced))
    return conds


@contextlib.contextmanager
def override_tile_positives(prebuilt, tiles):
    """Swap vl.build_global_slices for a hook returning the prebuilt per-tile conds.
    sampling.py resolves `vl.build_global_slices` at call time, so this reaches the
    production swap seam without touching production code. The layout is re-asserted
    against the pre-pass tiles — a divergence would silently misassign positives."""
    from context_anchored_tile_refine import vl

    expected = _rects(tiles)

    def hook(clip, source, hook_tiles, offset_x=0, offset_y=0):
        if _rects(hook_tiles) != expected:
            raise RuntimeError("tile layout diverged between pre-pass and refine")
        return prebuilt

    original = vl.build_global_slices
    vl.build_global_slices = hook
    try:
        yield
    finally:
        vl.build_global_slices = original


def stage_conditioning(padded, tiles, force_captions):
    """Everything the text encoder produces, before the DiT ever loads: the three
    text encodes, the captions, and the prebuilt arm-3/arm-4 per-tile positives.
    Arm 2's vision encode deliberately stays INSIDE refine_image (production path)."""
    from context_anchored_tile_refine import upscale

    print("[clip]    loading {} ({})".format(CLIP_NAME, CLIP_TYPE))
    with ab_models.VramProbe() as probe:
        clip = ab_models.load_clip(CLIP_NAME, CLIP_TYPE)
        with torch.inference_mode():
            pos_full = ab_models.encode_prompt(clip, POSITIVE_PROMPT)
            negative = ab_models.encode_prompt(clip, NEGATIVE_PROMPT)
            empty = upscale.encode_empty(clip)
        captions = load_or_generate_captions(clip, padded, tiles, force_captions)
        with torch.inference_mode():
            arm3 = build_caption_conds(clip, captions)
            arm4 = build_slice_caption_conds(clip, padded, tiles, captions)
    print("[clip]    conditioning built  {}".format(probe))
    return clip, pos_full, negative, empty, captions, arm3, arm4


def build_settings(arm, captions):
    payload = dataclasses.asdict(REFINE)
    payload.update({
        "run_label": arm,
        "arm": ARMS[arm],
        "base_png": str(BASE_PNG),
        "base_sha": _file_sha(BASE_PNG),
        "reference_output": REFERENCE_OUTPUT.name,
        "unet": UNET_NAME, "clip": CLIP_NAME, "clip_type": CLIP_TYPE, "vae": VAE_NAME,
        "upscale_model": UPSCALE_MODEL_NAME,
        "guider": "CFGGuider",
        "positive_prompt": POSITIVE_PROMPT if arm == "1-text-prompt" else "",
        "negative_prompt": NEGATIVE_PROMPT,
        "caption_instruction": CAPTION_INSTRUCTION if arm in ("3-tile-captions", "4-vl-plus-captions") else "",
        "tile_captions": captions if arm in ("3-tile-captions", "4-vl-plus-captions") else [],
        "harness": "tests-AB/run_ab_krea2.py",
        "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return payload


def render(arm, canvas, tiles, clip, model, vae, pos_full, negative, empty, arm3, arm4, captions):
    """One arm end to end, exactly as ContextAnchoredTileUpscaleVL wires it (minus the
    upscale, which already ran): fresh guider/sigmas/sampler/noise, then refine_image."""
    import comfy.samplers

    from context_anchored_tile_refine import sampling, upscale

    positive = pos_full if arm == "1-text-prompt" else empty
    guider = upscale.build_guider(model, positive, negative, REFINE.cfg)
    sigmas = upscale.build_sigmas(model, REFINE.scheduler, REFINE.steps, REFINE.denoise)
    sampler = comfy.samplers.sampler_object(REFINE.sampler)
    noise = upscale.Noise_RandomNoise(REFINE.seed)

    vl_clip = None if arm == "1-text-prompt" else clip
    hook = contextlib.nullcontext()
    if arm == "3-tile-captions":
        hook = override_tile_positives(arm3, tiles)
    elif arm == "4-vl-plus-captions":
        hook = override_tile_positives(arm4, tiles)

    with hook, ab_models.VramProbe() as probe, torch.inference_mode():
        result = sampling.refine_image(
            canvas, guider, sampler, sigmas, vae, noise,
            max_tile_width=REFINE.max_tile_width,
            max_tile_height=REFINE.max_tile_height,
            context_anchor=REFINE.context_anchor,
            context_overlap=REFINE.context_overlap,
            vl_clip=vl_clip,
        )

    ab_models.require_image_shape(result, canvas.shape[1], canvas.shape[2], "refine " + arm)
    destination = output_path(arm)
    written = ab_models.save_png(destination, result.cpu(), build_settings(arm, captions))
    print("[render]  {:<20} {}  -> {} {}x{}".format(arm, probe, destination.name, written[0], written[1]))


# ------------------------------------------------------------------ reporting

def fidelity_check():
    """Arm 2 reproduces the app's own render up to cross-process nondeterminism and
    the base PNG's 8-bit round trip; a LARGE diff means the pipeline diverged."""
    produced = output_path("2-vl-slices")
    if not (produced.is_file() and REFERENCE_OUTPUT.is_file()):
        print("[check]   2-vl-slices vs {}: skipped (file missing)".format(REFERENCE_OUTPUT.name))
        return
    left = np.asarray(Image.open(REFERENCE_OUTPUT).convert("RGB")).astype(np.int16)
    right = np.asarray(Image.open(produced).convert("RGB")).astype(np.int16)
    if left.shape != right.shape:
        print("[check]   shape mismatch {} vs {}".format(left.shape, right.shape))
        return
    diff = np.abs(left - right)
    print("[check]   2-vl-slices vs {}: meanabs {:.3f}/255  p99 {:.0f}  max {}/255".format(
        REFERENCE_OUTPUT.name, diff.mean(), np.percentile(diff, 99), int(diff.max())))


def contact_sheet(arms):
    """2x2 labeled sheet of whichever arms exist, downscaled 768x1024 per panel.
    Labels live on bars ABOVE each panel — image pixels are never drawn over."""
    bar, pad = 48, 8
    panel_w, panel_h = 768, 1024
    sheet = Image.new("RGB", (panel_w * 2 + pad * 3, (panel_h + bar) * 2 + pad * 3), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
    for index, arm in enumerate(arms):
        source = output_path(arm)
        if not source.is_file():
            continue
        col, row = index % 2, index // 2
        x = pad + col * (panel_w + pad)
        y = pad + row * (panel_h + bar + pad)
        draw.text((x + 6, y + 8), "{}  ({})".format(arm, ARMS[arm]), fill=(235, 235, 235), font=font)
        with Image.open(source) as handle:
            sheet.paste(handle.convert("RGB").resize((panel_w, panel_h), Image.LANCZOS), (x, y + bar))
    destination = OUTPUT_DIR / "AB_{}__contact-sheet.jpg".format(RUN_TAG)
    sheet.save(destination, quality=92)
    print("[sheet]   {}".format(destination))


# ------------------------------------------------------------------ main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", action="append", default=[], metavar="ARM",
                        help="render only these arms (repeatable)")
    parser.add_argument("--force", action="store_true", help="re-render existing outputs")
    parser.add_argument("--force-upscale", action="store_true", help="rebuild the cached upscale")
    parser.add_argument("--force-captions", action="store_true", help="regenerate the cached captions")
    parser.add_argument("--list", action="store_true", help="show the arms and exit")
    args = parser.parse_args(argv)

    unknown = set(args.only) - set(ARMS)
    if unknown:
        raise SystemExit("unknown arm(s): {}".format(", ".join(sorted(unknown))))
    selected = [arm for arm in ARMS if not args.only or arm in args.only]

    if args.list:
        for arm in ARMS:
            marker = "*" if arm in selected else " "
            print("{} {:<20} {}  -> {}".format(marker, arm, ARMS[arm], output_path(arm).name))
        return 0

    root, note = ab_env.bootstrap()
    print("[env]     ComfyUI {} at {}  ({})".format(ab_env.version(root), root, note))
    print("[env]     torch {}  cuda {}  device {}".format(
        torch.__version__, torch.version.cuda,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))

    base = load_base()
    canvas = stage_upscale(base, args.force_upscale)
    padded, tiles = solve_tiles(canvas)
    clip, pos_full, negative, empty, captions, arm3, arm4 = stage_conditioning(
        padded, tiles, args.force_captions)

    print("[unet]    loading {}".format(UNET_NAME))
    with ab_models.VramProbe() as probe:
        model = ab_models.load_unet(UNET_NAME)
        vae = ab_models.load_vae(VAE_NAME)
    print("[unet]    loaded  {}".format(probe))

    for arm in selected:
        destination = output_path(arm)
        if destination.is_file() and not args.force:
            print("[render]  {:<20} exists, skipped (--force to redo)".format(arm))
            continue
        ab_models.clear_cache()
        render(arm, canvas, tiles, clip, model, vae, pos_full, negative, empty, arm3, arm4, captions)

    fidelity_check()
    contact_sheet(list(ARMS))

    bad = []
    for arm in selected:
        destination = output_path(arm)
        if not destination.is_file():
            bad.append("{} MISSING".format(destination.name))
            continue
        width, height = ab_models.png_size(destination)
        verdict = "OK" if (width, height) == (canvas.shape[2], canvas.shape[1]) else "WRONG SIZE"
        if verdict != "OK":
            bad.append("{} is {}x{}".format(destination.name, width, height))
        print("[done]    {:<44} {:>4}x{:<4} {}".format(destination.name, width, height, verdict))
    if bad:
        raise SystemExit("[done]    FAILED: " + "; ".join(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
