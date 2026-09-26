"""Wall time of every part of the tags pass, per stage, over a ladder of prompt lengths.

The tags pass (context_anchored_tile_refine/tags.py) runs, per picture, the style caption and
the prompt fragment sort, and per tile propose, verify and locate (six strips). This harness
runs the shipped pass unchanged through tags.generate_tag_trace and times it from outside:
each stage function is wrapped, and inside a stage the vision tower, the text forward passes
(prefill, decode and packed), the model load, the repeat check and the resample are timed
apart. Every timed call is bracketed by torch.cuda.synchronize, so GPU time lands on the call
that queued it. Caches are cleared before every prompt, so every condition pays in full.

    python tests-AB/ab_caption_timing.py --tiles 0,9,14 --prompts none,short,medium,long
"""

import argparse
import json
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import ab_env

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "tests-AB" / "cache" / "caption_timing"
SCENE_PATH = REPO_ROOT / "samples" / "cyberpunk-city-8k.webp"
CLIP_NAME = "qwen3-vl-4b-heretic_int8.safetensors"

# The owner's 8K test workflow (VL 8k upscale - Test.json, Tile Test: Layout).
MAX_TILE_WIDTH, MAX_TILE_HEIGHT, CONTEXT_ANCHOR, CONTEXT_OVERLAP = 2048, 1728, 32, 256

# A ladder of positive prompts for the cyberpunk sample, written the way users write them.
PROMPTS = {
    "none": "",
    "short": "Cyberpunk cityscape at night",
    "medium": (
        "A sprawling cyberpunk city at night seen from a high rooftop, dense skyscrapers with "
        "thousands of glowing windows, a tall spire lit in pink and blue neon at the center, "
        "storm clouds swirling overhead, a river with a lit bridge on the left, neon signs in "
        "red and purple, cinematic lighting, highly detailed, 8k"
    ),
    "long": (
        "masterpiece, best quality, ultra detailed, 8k, photorealistic, cinematic, "
        "A breathtaking panoramic view of a sprawling cyberpunk megacity at night, seen from a "
        "high rooftop vantage point. Dense clusters of towering skyscrapers stretch to the "
        "horizon, their facades covered in thousands of glowing windows in warm amber and cold "
        "white. At the center rises a colossal art deco spire, its crown lit in vivid pink and "
        "electric blue neon, with a thin antenna piercing the clouds. To the right stands a "
        "second futuristic tower with a glowing purple spine and holographic billboards. "
        "Enormous storm clouds churn across the sky, lit from below by the city's violet and "
        "magenta glow, with faint laser beams and flying vehicles streaking between the towers. "
        "On the left a wide dark river winds through the city, crossed by a long bridge "
        "strung with golden lights that reflect on the water. Vertical neon signs in red, cyan "
        "and purple hang from the buildings, some with japanese and chinese characters, and a "
        "large sign reading CYY glows on a building at the right edge. In the foreground the "
        "rooftops of older apartment blocks are packed with water tanks, air conditioning "
        "units, antennas and satellite dishes, and narrow streets below glow with traffic and "
        "street lamps. Atmosphere: moody, dystopian, rain-soaked, hazy, volumetric fog, light "
        "bloom, lens flare, deep shadows, high contrast, teal and magenta color palette, "
        "wide angle lens, 24mm, long exposure, sharp focus, intricate details, trending on "
        "artstation, octane render, unreal engine 5, blade runner style, ghost in the shell "
        "vibes, award winning photography"
    ),
}


class Clock:
    """Accumulates synchronized wall time per (stage, part), with a stage stack."""

    def __init__(self, torch):
        self.torch = torch
        self.stages = ["outside"]
        self.ms = defaultdict(float)
        self.calls = defaultdict(int)
        self.forwards = []

    def sync(self):
        self.torch.cuda.synchronize()

    @contextmanager
    def part(self, name):
        self.sync()
        started = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            key = (self.stages[-1], name)
            self.ms[key] += (time.perf_counter() - started) * 1000.0
            self.calls[key] += 1

    @contextmanager
    def stage(self, name):
        self.stages.append(name)
        try:
            with self.part("total"):
                yield
        finally:
            self.stages.pop()


def wrap(owner, attribute, around):
    """Replace owner.attribute with around(original, *args, **kwargs), returning an undo."""
    original = getattr(owner, attribute)
    shadowed = attribute in vars(owner)

    def wrapper(*args, **kwargs):
        return around(original, *args, **kwargs)

    setattr(owner, attribute, wrapper)

    def undo():
        if shadowed:
            setattr(owner, attribute, original)
        else:
            delattr(owner, attribute)
    return undo


def instrument(clock, clip, tags, captions, lc_tags, backend_cls):
    """Wrap every timed call. Returns the undo callables."""
    model = clip.cond_stage_model
    transformer = getattr(model, model.clip).transformer
    undo = []

    def staged(name):
        def around(original, *args, **kwargs):
            before_ms = clock.ms[(name, "total")]
            before_decodes = clock.calls[(name, "decode")]
            with clock.stage(name):
                result = original(*args, **kwargs)
            decodes = clock.calls[(name, "decode")] - before_decodes
            print(f"    {name}: {clock.ms[(name, 'total')] - before_ms:.0f} ms"
                  f"{f', {decodes} decode steps' if decodes else ''}", flush=True)
            return result
        return around

    def parted(name):
        def around(original, *args, **kwargs):
            with clock.part(name):
                return original(*args, **kwargs)
        return around

    def forward(original, *args, **kwargs):
        embeds = kwargs.get("embeds")
        length = int(embeds.shape[1]) if embeds is not None else 0
        past = kwargs.get("past_key_values")
        kind = "decode" if length == 1 else ("prefill" if past is not None else "packed")
        with clock.part(kind):
            result = original(*args, **kwargs)
        if kind != "decode":
            clock.forwards.append((clock.stages[-1], kind, length))
        return result

    undo.append(wrap(tags, "style_line", staged("style caption")))
    undo.append(wrap(tags, "fragment_style_p", staged("fragment sort")))
    undo.append(wrap(tags, "propose", staged("propose")))
    undo.append(wrap(tags, "_verify_scores", staged("verify")))
    undo.append(wrap(tags, "strip_scores", staged("locate strips")))
    undo.append(wrap(tags, "_tag_cache_key", parted("cache key")))
    undo.append(wrap(captions, "resample_for_vl", parted("resample")))
    undo.append(wrap(lc_tags, "complete_tags", parted("repeat check")))
    undo.append(wrap(clip, "load_model", parted("load model")))
    undo.append(wrap(transformer.visual, "forward", parted("vision tower")))
    undo.append(wrap(transformer.model, "forward", forward))
    undo.append(wrap(backend_cls, "encode", parted("tokenize")))
    return undo


def run_condition(clip, canvas, tiles, prompt_key, with_style):
    import logit_classifier.tags as lc_tags
    import torch
    from logit_classifier.backends.comfy_clip import ComfyClipBackend

    from context_anchored_tile_refine import captions, tags

    captions.clear_caption_cache()
    tags.clear_tag_cache()
    preset = captions.with_prompt(captions.resolve_method(captions.default_vlm_method()),
                                  PROMPTS[prompt_key])
    if not with_style:
        preset = replace(preset, style_instruction="")
    clock = Clock(torch)
    undo = instrument(clock, clip, tags, captions, lc_tags, ComfyClipBackend)
    try:
        with clock.stage("whole pass"):
            run = tags.generate_tag_trace(clip, canvas, tiles, preset)
    finally:
        for step in reversed(undo):
            step()
    return preset, clock, run


def summarize(prompt_key, preset, clock, run, tiles):
    from logit_classifier.tags import split_prompt

    traces = [row[0] for row in run.tiles]
    stage_names = ("style caption", "fragment sort", "propose", "verify", "locate strips")
    per_stage = {}
    for stage in stage_names:
        parts = {part: round(ms, 1) for (s, part), ms in clock.ms.items() if s == stage}
        calls = {part: n for (s, part), n in clock.calls.items() if s == stage}
        per_stage[stage] = {"ms": parts, "calls": calls,
                            "forward_lengths": [n for s, kind, n in clock.forwards if s == stage]}
    return {
        "prompt": prompt_key,
        "prompt_words": len(PROMPTS[prompt_key].split()),
        "fragments": len(split_prompt(preset.prompt)),
        "subjects": len(run.prompt.subjects) if run.prompt else 0,
        "styles": len(run.prompt.styles) if run.prompt else 0,
        "tiles": [{"index": index, "proposed": len(t.proposed), "candidates": len(t.candidates),
                   "over_cap": sum(1 for _, why in t.dropped if why == "over the cap"),
                   "from_prompt": sum(1 for o in t.origins if o == "prompt"),
                   "verified": len(t.verified), "kept": len(t.kept), "unplaced": len(t.unplaced),
                   "text": t.text, "reply": t.reply}
                  for index, t in zip(tiles, traces, strict=True)],
        "whole_ms": round(clock.ms[("whole pass", "total")], 1),
        "outside_parts": {part: round(ms, 1) for (s, part), ms in clock.ms.items() if s == "whole pass"},
        "stages": per_stage,
        "style_text": run.style_texts[0] if run.style_texts else "",
    }


def print_summary(record, n_tiles):
    print(f"\n=== prompt {record['prompt']}: {record['prompt_words']} words, "
          f"{record['fragments']} fragments ({record['subjects']} subject, {record['styles']} style), "
          f"whole pass {record['whole_ms'] / 1000:.1f} s for {n_tiles} tiles")
    for stage, data in record["stages"].items():
        total = data["ms"].get("total", 0.0)
        if not total:
            continue
        per_call = data["calls"].get("total", 1)
        parts = ", ".join(f"{part} {ms / 1000:.2f}s x{data['calls'][part]}"
                          for part, ms in sorted(data["ms"].items(), key=lambda kv: -kv[1]) if part != "total")
        lengths = data["forward_lengths"]
        span = f", forward tokens {min(lengths)}-{max(lengths)}" if lengths else ""
        print(f"  {stage:14s} {total / 1000:6.2f} s total, {total / per_call / 1000:5.2f} s per call "
              f"[{parts}]{span}")
    for tile in record["tiles"]:
        print(f"  tile {tile['index']:2d}: proposed {tile['proposed']}, candidates {tile['candidates']} "
              f"({tile['from_prompt']} from prompt, {tile['over_cap']} over cap), verified "
              f"{tile['verified']}, kept {tile['kept']}, unplaced {tile['unplaced']}")
        print(f"     text: {tile['text'][:300]}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tiles", default="0,9,14", help="csv of tile indices in the 8K layout")
    parser.add_argument("--prompts", default="none,short,medium,long", help=f"csv of {sorted(PROMPTS)}")
    parser.add_argument("--style-every", action="store_true",
                        help="run the style caption for every prompt, not only the first (it reads no prompt)")
    parser.add_argument("--no-style", action="store_true", help="skip the style caption for every prompt")
    parser.add_argument("--label", default="", help="suffix for the output file")
    return parser.parse_args()


def main():
    args = parse_args()
    ab_env.bootstrap()
    import ab_models
    import numpy as np
    import torch
    from PIL import Image

    from context_anchored_tile_refine import grid

    with Image.open(SCENE_PATH) as handle:
        canvas = torch.from_numpy(np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0)[None,]
    height, width = int(canvas.shape[1]), int(canvas.shape[2])
    sx = grid.solve_axis(width, MAX_TILE_WIDTH, CONTEXT_ANCHOR, CONTEXT_OVERLAP, axis="width")
    sy = grid.solve_axis(height, MAX_TILE_HEIGHT, CONTEXT_ANCHOR, CONTEXT_OVERLAP, axis="height")
    layout = grid.build_layout(width, height, sx, sy, CONTEXT_ANCHOR, CONTEXT_OVERLAP)
    wanted = [int(t) for t in args.tiles.split(",") if t.strip()]
    tiles = [layout.tiles[i] for i in wanted]
    print(f"{width}x{height}, {len(layout.tiles)} tiles in the layout, timing tiles {wanted}")
    for index, tile in zip(wanted, tiles, strict=True):
        r = tile.crop_rect
        print(f"  tile {index}: r{tile.row}c{tile.col} crop {r.x1 - r.x0}x{r.y1 - r.y0}")

    started = time.perf_counter()
    clip = ab_models.load_clip(CLIP_NAME, "krea2")
    print(f"clip loaded in {time.perf_counter() - started:.1f} s", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    for index, prompt_key in enumerate(p.strip() for p in args.prompts.split(",")):
        preset, clock, run = run_condition(clip, canvas, tiles, prompt_key, not args.no_style and (index == 0 or args.style_every))
        record = summarize(prompt_key, preset, clock, run, wanted)
        record["first_condition"] = index == 0
        records.append(record)
        print_summary(record, len(tiles))
        out = OUT_DIR / f"timing{args.label}.json"
        out.write_text(json.dumps({"tiles": wanted, "layout_tiles": len(layout.tiles),
                                   "records": records}, indent=1))
    print(f"\nwritten {out}")
    propose = [r["stages"]["propose"]["ms"].get("total", 0) / len(tiles) for r in records]
    print("propose s per tile by prompt:", [round(p / 1000, 2) for p in propose],
          "median", round(statistics.median(propose) / 1000, 2))


if __name__ == "__main__":
    main()
