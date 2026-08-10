"""F0b stress test: full-size base, 2x upscale, 2x2 tiled refine at the VRAM ceiling.

Owner-specified configuration: base 1344x768x124f (just below the usual 1376x768),
lanczos 2x to 2688x1536, then FOUR tiles with 1344x768 cores. Each tile crop is
1408x832 — a DiT sequence ~8% past what F0 actually proved (1408x768), and ~26%
past it with --vl. Deliberately unproven territory; that is the stress.

This is the first tiled H3 run. It uses CATR's seam design in miniature:

    crop   = core + 32px overlap band + 32px anchor ring (margins on inner sides only)
    anchor = frozen via the binary denoise mask, pixels taken from the LIVE canvas,
             so tiles 2-4 see their already-refined neighbors (context anchoring)
    band   = diffused from the RAW upscaled source by both adjacent tiles (never from
             refined pixels — prime directive 2), cross-dissolved by a 64px linear
             ramp on sides bordering an already-refined neighbor
    core   = pasted hard
    noise  = ONE draw at canvas latent size, sliced per tile (the production
             convention — the shared bands see correlated noise, not redraws)

Deviations from production, deliberate (this is a stress test, not the node):
linear ramp instead of the plateau/falloff curve, no min-error seam placement,
no VL slicing — every tile gets the same t2va text conditioning. Seams and drift
seen here are an UPPER bound on what the real node will show.

Outputs (only what the owner asked for): H3S_base.mp4, H3S_refined.mp4, plus
matched full-res PNG pairs (upscaled vs refined at frames 0/62/123) for still
comparison. Videos are written to a temp name and renamed, so opening a file
mid-run can never collide with a write.

    <venv-python> tests-AB/spike_h3_stress.py                # text-cond arm (baseline)
    <venv-python> tests-AB/spike_h3_stress.py --vl           # VL-sliced arm
    <venv-python> tests-AB/spike_h3_stress.py --denoise 0.25

--vl replaces every tile's text positive with its slice of ONE video-reference
encode of the whole upscaled canvas (the port of vl.py's global-slice method):
the canvas is sampled at 2 fps into the tokenizer's <Video 1> presentation
(per-2-frame blocks: "<T.T seconds>" + vision grid, timestamps kept), encoded
once through the 32B CLIP with an EMPTY prompt, and each tile keeps the grid
cells its crop covers — with `minimax_token_tags` sliced in lockstep so the
DiT's modality runs stay aligned. 2688x1536 is under the encoder's 12.8MP
per-block cap and /32 on both axes, so the grid maps 1:1 to 32px canvas cells
with no resampling.
"""
import argparse
import faulthandler
import json
import os

faulthandler.enable()  # native-crash stack into the log (two runs died with SIGSEGV)

import ab_models  # noqa: E402
import numpy as np  # noqa: E402
import spike_h3 as poc  # noqa: E402  (module import runs the ComfyUI bootstrap)
import torch  # noqa: E402

# ------------------------------------------------------------------ CONFIG

OUTPUT_DIR = poc.OUTPUT_DIR
CACHE_DIR = poc.CACHE_DIR

BASE_W, BASE_H = 1344, 768
UP_W, UP_H = 2688, 1536
GRID = 2                                       # 2x2 tiles, cores tile the canvas exactly
CORE_W, CORE_H = UP_W // GRID, UP_H // GRID    # 1344x768 per core
ANCHOR, OVERLAP = 32, 64 - 32                  # px; margin = ANCHOR + OVERLAP = 64, /32
MARGIN = ANCHOR + OVERLAP
FPS = 24

STRESS_KEY = dict(poc.BASE_KEY, width=BASE_W, height=BASE_H)


# ------------------------------------------------------------------ helpers

def save_video_atomic(label, frames, settings):
    """crf-10 mp4 via temp-name + os.replace, plus first/middle/last PNGs."""
    import av

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_frame_pngs(label, frames, settings)

    final = OUTPUT_DIR / f"H3S_{label}.mp4"
    tmp = OUTPUT_DIR / f"H3S_{label}.tmp"
    container = av.open(str(tmp), mode="w", format="mp4")
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
    os.replace(tmp, final)
    print(f"  wrote {final}", flush=True)


def save_frame_pngs(label, frames, settings):
    for index in (0, frames.shape[0] // 2, frames.shape[0] - 1):
        ab_models.save_png(
            OUTPUT_DIR / f"H3S_{label}_f{index:03d}.png",
            frames[index:index + 1], settings)


def tile_rects():
    """Raster-order 2x2 layout. Margins exist on inner sides only; every crop is
    1408x832 and /32 on both axes by construction."""
    rects = []
    for ty in range(GRID):
        for tx in range(GRID):
            core = (tx * CORE_W, ty * CORE_H, (tx + 1) * CORE_W, (ty + 1) * CORE_H)
            crop = (core[0] - (MARGIN if tx > 0 else 0),
                    core[1] - (MARGIN if ty > 0 else 0),
                    core[2] + (MARGIN if tx < GRID - 1 else 0),
                    core[3] + (MARGIN if ty < GRID - 1 else 0))
            rects.append({
                "core": core, "crop": crop,
                "margin_sides": {"l": tx > 0, "t": ty > 0,
                                 "r": tx < GRID - 1, "b": ty < GRID - 1},
                "refined_sides": {"l": tx > 0, "t": ty > 0},   # raster order
            })
    return rects


def assemble_crop(raw, live, rect):
    """Tile input pixels: raw everywhere (core + overlap bands — directive 2:
    the shared bands diffuse from the frozen raw source), except the anchor ring,
    which takes the LIVE canvas so the tile sees refined neighbors."""
    x0, y0, x1, y1 = rect["crop"]
    crop = raw[:, y0:y1, x0:x1, :].clone()
    m = rect["margin_sides"]
    if m["l"]:
        crop[:, :, :ANCHOR, :] = live[:, y0:y1, x0:x0 + ANCHOR, :]
    if m["r"]:
        crop[:, :, -ANCHOR:, :] = live[:, y0:y1, x1 - ANCHOR:x1, :]
    if m["t"]:
        crop[:, :ANCHOR, :, :] = live[:, y0:y0 + ANCHOR, x0:x1, :]
    if m["b"]:
        crop[:, -ANCHOR:, :, :] = live[:, y1 - ANCHOR:y1, x0:x1, :]
    return crop.contiguous()


def anchor_mask(latent, rect):
    """Binary video denoise mask: 1 everywhere, 0 on the anchor ring (2 latent
    cells of 16px per 32px of anchor) on sides that carry a margin."""
    cells = ANCHOR // 16
    mask = torch.ones_like(latent, device="cpu")
    m = rect["margin_sides"]
    if m["l"]:
        mask[..., :, :cells] = 0.0
    if m["r"]:
        mask[..., :, -cells:] = 0.0
    if m["t"]:
        mask[..., :cells, :] = 0.0
    if m["b"]:
        mask[..., -cells:, :] = 0.0
    return mask


def paste_alpha(rect):
    """[h, w] blend weights for the paste region (crop minus anchor rings).
    1 in the core and on bands toward unrefined territory; a linear 0->1 ramp over
    OVERLAP*2 = 64px on sides bordering an already-refined neighbor — exactly the
    mutual coverage of the two tiles' shared diffused strips."""
    x0, y0, x1, y1 = rect["crop"]
    m, r = rect["margin_sides"], rect["refined_sides"]
    px0 = x0 + (ANCHOR if m["l"] else 0)
    py0 = y0 + (ANCHOR if m["t"] else 0)
    px1 = x1 - (ANCHOR if m["r"] else 0)
    py1 = y1 - (ANCHOR if m["b"] else 0)
    h, w = py1 - py0, px1 - px0
    ramp = OVERLAP * 2
    alpha = torch.ones(h, w)
    if r["l"]:
        line = (torch.arange(ramp, dtype=torch.float32) + 0.5) / ramp
        alpha[:, :ramp] = torch.minimum(alpha[:, :ramp], line.view(1, -1))
    if r["t"]:
        line = (torch.arange(ramp, dtype=torch.float32) + 0.5) / ramp
        alpha[:ramp, :] = torch.minimum(alpha[:ramp, :], line.view(-1, 1))
    return (px0, py0, px1, py1), alpha


def sample_2fps(frames):
    """The ref node's video presentation sampling: every FPS//2-th frame, sequential
    half-second timestamps, repeat-padded to the 2-frame temporal patch."""
    idx = list(range(0, frames.shape[0], FPS // 2))
    picks = frames[idx]
    stamps = [i / 2.0 for i in range(len(idx))]
    if picks.shape[0] % 2 == 1:
        picks = torch.cat([picks, picks[-1:]], dim=0)
        stamps.append(stamps[-1])
    return picks.contiguous(), stamps


def encode_global_vl(picks, stamps, force):
    """ONE video-reference encode of the upscaled canvas's 2fps picks, cached.

    Returns (cond [1,S,5120] cpu, tags [S] cpu, layout). layout is a list of
    ("text", row) for every literal token row and ("grid", start_row) for every
    2-frame block's 4032-cell raster grid — derived by walking the token stream
    (dict entries expand to one row per merged cell) and asserted against the
    encoder's output length, vl.py's fail-fast pattern."""
    cache = ab_models.cache_path(CACHE_DIR, "h3stress", "vlcond", UP_W, UP_H, STRESS_KEY)
    if cache.is_file() and not force:
        payload = torch.load(cache, weights_only=True)
        print(f"VL global encode from cache: {cache.name}", flush=True)
        return payload["cond"], payload["tags"], payload["layout"]

    import comfy.model_management

    # A 24k-token forward through the 32B encoder needs ~7.5 GiB of activations; with
    # the 14.6 GiB model loaded whole that leaves ~0.6 GiB of margin (reviewed, coin-
    # flip). Raising the reserve forces comfy to stream layers instead, trading ~a
    # minute of encode time for guaranteed headroom. Restored in finally.
    saved_reserve = comfy.model_management.EXTRA_RESERVED_VRAM
    comfy.model_management.EXTRA_RESERVED_VRAM = 8 * 1024 ** 3
    try:
        print("VL global encode: loading 32B encoder...", flush=True)
        clip = ab_models.load_clip(poc.CLIP_NAME, poc.CLIP_TYPE)
        items = [{"type": "video", "data": picks, "timestamps": stamps}]
        rows_per_block = (UP_H // 32) * (UP_W // 32)
        with ab_models.VramProbe() as probe, torch.inference_mode():
            tokens = clip.tokenize("", minimax_ref_items=items)
            layout = []
            pos = 0
            for entry in tokens["qwen3vl_32b"][0]:
                if isinstance(entry[0], dict):
                    layout.append(("grid", pos))
                    pos += rows_per_block
                else:
                    layout.append(("text", pos))
                    pos += 1
            encoded = clip.encode_from_tokens_scheduled(tokens)
    finally:
        comfy.model_management.EXTRA_RESERVED_VRAM = saved_reserve
    if len(encoded) != 1:
        raise RuntimeError(f"scheduled encode returned {len(encoded)} sections")
    cond, extras = encoded[0][0], encoded[0][1]
    blocks = sum(1 for kind, _ in layout if kind == "grid")
    print(f"VL global encode: {probe} ({pos} rows, {blocks} blocks)", flush=True)
    if cond.shape[1] != pos:
        raise RuntimeError(
            f"VL encode has {cond.shape[1]} rows, expected {pos} — token layout drifted")
    tags = extras.get("minimax_token_tags")
    if tags is None or int(tags.shape[0]) != pos:
        raise RuntimeError("minimax_token_tags missing or misaligned")
    del clip
    ab_models.free_gpu()

    cond, tags = cond.cpu(), tags.cpu()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cond": cond, "tags": tags, "layout": layout}, cache)
    return cond, tags, layout


def vl_slice(cond, tags, layout, rect):
    """Tile positive: every literal token row + the grid cells the crop covers,
    in original row order, tags sliced with the same indices. Crops are /32 so
    the cell ranges are exact; overlapping crops share their boundary cells."""
    x0, y0, x1, y1 = rect["crop"]
    gw = UP_W // 32
    cx0, cx1 = x0 // 32, -(-x1 // 32)
    cy0, cy1 = y0 // 32, -(-y1 // 32)
    indices = []
    for kind, start in layout:
        if kind == "text":
            indices.append(start)
        else:
            for r in range(cy0, cy1):
                base = start + r * gw
                indices.extend(range(base + cx0, base + cx1))
    idx = torch.tensor(indices)
    return [[cond.index_select(1, idx), {"minimax_token_tags": tags.index_select(0, idx)}]]


def canvas_noise(seed, video_shape, audio_shape):
    """ONE RandomNoise draw at canvas latent size, to be sliced per tile — the
    production convention ('slice noise rather than redrawing it'), so the tiles'
    shared bands see correlated noise instead of independent redraws."""
    import comfy.nested_tensor
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

    canvas_video = torch.zeros(
        [video_shape[0], video_shape[1], video_shape[2], UP_H // 16, UP_W // 16])
    dummy = {"samples": comfy.nested_tensor.NestedTensor(
        (canvas_video, torch.zeros(audio_shape)))}
    return Noise_RandomNoise(seed).generate_noise(dummy)


class SlicedNoise:
    """NOISE object handing a tile its slice of the single canvas draw."""

    def __init__(self, seed, nested_noise, rect):
        self.seed = seed
        x0, y0, x1, y1 = rect["crop"]
        video, audio = nested_noise.unbind()
        self._video = video[..., y0 // 16:y1 // 16, x0 // 16:x1 // 16].contiguous()
        self._audio = audio

    def generate_noise(self, input_latent):
        import comfy.nested_tensor

        expected = tuple(input_latent["samples"].unbind()[0].shape)
        if tuple(self._video.shape) != expected:
            raise RuntimeError(f"noise slice {tuple(self._video.shape)} != tile latent {expected}")
        return comfy.nested_tensor.NestedTensor(
            (self._video.clone(), self._audio.clone()))


def refine_tile(model, positive, video_latent, audio_latent, mask, denoise, noise,
                steps=None):
    import comfy.nested_tensor

    # Default couples step count to the base's 20-step grid density; production
    # exposes steps and denoise independently (an override starves at low denoise:
    # 4 steps at d0.22 visibly under-integrates — F0d finding).
    steps = steps if steps else max(1, round(poc.STEPS * denoise))
    guider = poc.build_basic_guider(model, positive)
    sigmas = ab_models.build_sigmas(model, poc.SCHEDULER, steps, denoise)
    sampler = ab_models.build_sampler(poc.SAMPLER)
    latent = {
        "samples": comfy.nested_tensor.NestedTensor(
            (video_latent, audio_latent.clone())),
        "noise_mask": comfy.nested_tensor.NestedTensor(
            (mask, torch.zeros_like(audio_latent, device="cpu"))),
    }
    with ab_models.VramProbe() as probe, torch.inference_mode():
        out = ab_models.sample_custom_advanced(noise, guider, sampler, sigmas, latent)
    print(f"  tile sample d={denoise} ({steps} steps): {probe}", flush=True)
    del guider
    video, _ = out["samples"].unbind()
    return video


def render_base_full(positive, force):
    """1344x768x124f t2va base, cached (same pattern as poc.render_base)."""
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent

    cache = ab_models.cache_path(CACHE_DIR, "h3stress", "base", BASE_W, BASE_H, STRESS_KEY)
    if cache.is_file() and not force:
        pair = torch.load(cache, weights_only=True)
        print(f"base latents from cache: {cache.name}", flush=True)
        return pair["video"], pair["audio"]

    # Cache miss on an arm that skipped the text cond: load it now, still in the early
    # bracket (before the VAE), so it never coexists with the full canvases.
    if positive is None:
        positive = poc.get_positive(force)

    latent, frame_count = _empty_av_latent(BASE_W, BASE_H, poc.LENGTH)
    if frame_count != poc.LENGTH:
        raise RuntimeError("LENGTH off the 17k+5 grid")
    model = ab_models.load_unet(poc.UNET_NAME)
    guider = poc.build_basic_guider(model, positive)
    sigmas = ab_models.build_sigmas(model, poc.SCHEDULER, poc.STEPS, 1.0)
    sampler = ab_models.build_sampler(poc.SAMPLER)
    noise = ab_models.build_noise(poc.BASE_SEED)
    with ab_models.VramProbe() as probe, torch.inference_mode():
        out = ab_models.sample_custom_advanced(noise, guider, sampler, sigmas, latent)
    print(f"base sample ({poc.STEPS} steps): {probe}", flush=True)
    del guider, model
    ab_models.free_gpu()

    video, audio = out["samples"].unbind()
    video, audio = video.cpu(), audio.cpu()
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"video": video, "audio": audio}, cache)
    return video, audio


# ------------------------------------------------------------------ main

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--denoise", type=float, default=0.32)
    parser.add_argument("--refine-steps", type=int, default=None,
                        help="steps INSIDE the denoise window (default: 20*denoise, "
                             "the base grid's density)")
    parser.add_argument("--vl", action="store_true",
                        help="slice one global vision encode per tile instead of text")
    parser.add_argument("--encode-only", action="store_true",
                        help="with --vl: cache the global encode, then exit (run this "
                             "foreground — background runs die at the 32B CLIP load)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    import comfy.utils

    print(f"ComfyUI root: {poc.ROOT} ({poc.ROOT_NOTE}) version {poc.ab_env.version(poc.ROOT)}", flush=True)
    # BasicScheduler semantics land the true start point off the nominal denoise
    # (int() truncation in total_steps); record the effective value for the A/B notes.
    steps = args.refine_steps if args.refine_steps else max(1, round(poc.STEPS * args.denoise))
    effective_denoise = round(steps / int(steps / args.denoise), 4)
    settings = dict(STRESS_KEY, denoise=args.denoise, denoise_effective=effective_denoise,
                    refine_steps=steps,
                    refine_seed=poc.REFINE_SEED, up_w=UP_W, up_h=UP_H, grid=GRID,
                    anchor=ANCHOR, overlap=OVERLAP,
                    comfyui_version=poc.ab_env.version(poc.ROOT),
                    conditioning="vl-slice" if args.vl else "text")
    # The base render and the lanczos upscale are not refine runs: they must not carry
    # this run's denoise/seed/grid or its arm's conditioning label.
    base_settings = dict(STRESS_KEY, denoise=1.0,
                         comfyui_version=poc.ab_env.version(poc.ROOT),
                         conditioning="text")
    upscale_settings = dict(STRESS_KEY, up_w=UP_W, up_h=UP_H,
                            comfyui_version=poc.ab_env.version(poc.ROOT),
                            conditioning="n/a")

    # --vl never consumes the text cond (vl_slice replaces it per tile), and a cached
    # base latent never touches it either — don't load the 14.6 GB encoder for nothing.
    positive = None if args.vl else poc.get_positive(args.force)
    base_video, base_audio = render_base_full(positive, args.force)

    vae = ab_models.load_vae(poc.VAE_NAME)
    # fp16 canvases: 32 GB system RAM must hold BOTH full-res canvases plus the DiT's
    # ~20 GB CPU-side staging for lowvram streaming — fp32 canvases put the commit over
    # the top (both prior segfaults sat at that bracket). Lossless here: core's lanczos
    # quantizes through 8-bit PIL internally and every output path is 8-bit.
    base_frames = poc.decode_video(vae, base_video).to(torch.float16)
    if not args.encode_only:
        save_video_atomic("base", base_frames, dict(base_settings, run_label="base"))

    vl_pack = None
    if args.vl:
        # Encode BEFORE the full canvases exist: lanczos is per-frame, so upscaling
        # just the twelve 2fps picks gives the identical encoder input, and the 32B
        # encoder load never coexists with the ~6 GB canvas buffers — the second run
        # died exactly at that RAM bracket (32 GB ceiling).
        picks, stamps = sample_2fps(base_frames)
        if args.encode_only:
            del base_frames         # this arm returns below and never uses it again
        with torch.inference_mode():
            picks = comfy.utils.common_upscale(
                picks.movedim(-1, 1), UP_W, UP_H, "lanczos", "disabled"
            ).movedim(1, -1).to(torch.float16).contiguous()
        print(f"VL picks upscaled: {tuple(picks.shape)}", flush=True)
        # Evict the VAE before the 32B encoder loads; every other model swap in this
        # harness brackets itself the same way instead of trusting comfy's estimate.
        ab_models.free_gpu()
        vl_pack = encode_global_vl(picks, stamps, args.force)
        del picks
        ab_models.free_gpu()

    if args.encode_only:
        print("encode cached; exiting (--encode-only)", flush=True)
        return

    with torch.inference_mode():
        # frame-chunked: core's lanczos materializes an fp32 list + stack of the whole
        # batch (~15 GB at this size, reviewed); 8-frame chunks cap that at ~1 GB, and
        # writing into a preallocated canvas avoids a second full-size allocation.
        raw = torch.empty([base_frames.shape[0], UP_H, UP_W, 3], dtype=torch.float16)
        for i0 in range(0, base_frames.shape[0], 8):
            i1 = min(i0 + 8, base_frames.shape[0])
            raw[i0:i1] = comfy.utils.common_upscale(
                base_frames[i0:i1].movedim(-1, 1), UP_W, UP_H, "lanczos", "disabled"
            ).movedim(1, -1)
    del base_frames
    print("canvas ready", flush=True)
    save_frame_pngs("upscaled", raw, dict(upscale_settings, run_label="upscaled"))
    live = raw.clone()

    noise_pack = canvas_noise(poc.REFINE_SEED, tuple(base_video.shape),
                              tuple(base_audio.shape))

    model = ab_models.load_unet(poc.UNET_NAME)
    for index, rect in enumerate(tile_rects()):
        print("tile {}/4 crop={} core={}".format(index + 1, rect["crop"], rect["core"]),
              flush=True)
        tile_positive = vl_slice(*vl_pack, rect) if vl_pack else positive
        crop = assemble_crop(raw, live, rect)
        z = poc.encode_video(vae, crop)
        del crop
        expected = (1, 24, base_video.shape[2],
                    (rect["crop"][3] - rect["crop"][1]) // 16,
                    (rect["crop"][2] - rect["crop"][0]) // 16)
        if tuple(z.shape) != expected:
            raise RuntimeError(f"tile latent {tuple(z.shape)} != expected {expected}")
        refined = refine_tile(model, tile_positive, z, base_audio,
                              anchor_mask(z, rect), args.denoise,
                              SlicedNoise(poc.REFINE_SEED, noise_pack, rect),
                              steps=args.refine_steps)
        del z
        tile_frames = poc.decode_video(vae, refined).to(torch.float16)
        del refined
        ab_models.clear_cache()

        (px0, py0, px1, py1), alpha = paste_alpha(rect)
        x0, y0 = rect["crop"][0], rect["crop"][1]
        region = tile_frames[:, py0 - y0:py1 - y0, px0 - x0:px1 - x0, :]
        blend = alpha.view(1, alpha.shape[0], alpha.shape[1], 1)
        # blend in fp32, store fp16 — one rounding at the end, below the 8-bit step.
        # Frame-chunked: whole-region fp32 transients are ~4.6 GB (reviewed); 16-frame
        # chunks cap them at ~0.6 GB. Per-frame math is independent, so bit-identical.
        for f0 in range(0, live.shape[0], 16):
            f1 = min(f0 + 16, live.shape[0])
            live[f0:f1, py0:py1, px0:px1, :] = (
                region[f0:f1].float() * blend
                + live[f0:f1, py0:py1, px0:px1, :].float() * (1.0 - blend)
            ).to(torch.float16)
        del tile_frames, region

    del model
    ab_models.free_gpu()
    del raw                     # last use was assemble_crop inside the tile loop
    label = "refined_{}d{:02.0f}".format("vl_" if args.vl else "", args.denoise * 100)
    save_video_atomic(label, live, dict(settings, run_label=label))

    print(f"done. compare H3S_upscaled_fNNN.png vs H3S_refined_*_fNNN.png in {OUTPUT_DIR}", flush=True)
    print(json.dumps(settings, indent=1, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
