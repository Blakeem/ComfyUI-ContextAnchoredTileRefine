"""The tile tags benchmark: time and accuracy of tile tag arms over a fixed test set.

Five scenes at the owner's tile widgets (2048x1728, anchor 32, overlap 256), three tiles each
(a cropped subject, a busy tile, a sparse tile), three prompts each (none, short, long). An arm
turns (scene, prompt) into one tag text per tile. Accuracy comes from an independent judge (a
vision model reading the tile crop), cached per (scene, tile, item) so every arm is graded
against the same verdicts.

    python tests-AB/ab_tags_bench.py layouts                 # grid overlays, to pick tiles
    python tests-AB/ab_tags_bench.py run --arm baseline      # GPU, one arm, every scene
    python tests-AB/ab_tags_bench.py tasks                   # judge task files for unjudged items
    python tests-AB/ab_tags_bench.py merge                   # judge verdicts into judgments.json
    python tests-AB/ab_tags_bench.py report [--arms a,b]     # the table
"""

import argparse
import json
import math
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

import ab_env

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "tests-AB" / "cache" / "tags_bench"
RUNS_DIR = BENCH_DIR / "runs"
JUDGE_DIR = BENCH_DIR / "judge"
TILE_DIR = BENCH_DIR / "tiles"
JUDGMENTS = BENCH_DIR / "judgments.json"
CLIP_NAME = "qwen3-vl-4b-heretic_int8.safetensors"
MAX_TILE_WIDTH, MAX_TILE_HEIGHT, CONTEXT_ANCHOR, CONTEXT_OVERLAP = 2048, 1728, 32, 256
JUDGE_LONG_SIDE = 1536

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_caption_timing as timing  # noqa: E402

CYBER_LONG = timing.PROMPTS["long"]

HANGAR_LONG = (
    "masterpiece, best quality, ultra detailed 8k concept art, hyperrealistic, cinematic. "
    "A vast orbital shipyard hangar seen from an elevated walkway. In the center a battle-scarred "
    "grey capital warship with faded red hull stripes and rows of orange engine lights sits in "
    "dry dock, its armored hull plates scorched and dented. On top of the ship a turret sensor "
    "array with a glowing blue lens scans the bay. Yellow overhead gantry cranes cross the "
    "ceiling, and one crane hook holds a hanging blue humanoid robot at the upper right. Rows "
    "of silver humanoid robots stand at attention on the right side of the floor, while "
    "engineers in yellow hi-vis vests and white helmets work beneath the hull on a yellow "
    "scaffold platform, welding sparks flying. A fire burns in a round pit on the left, next to "
    "an orange glowing furnace. A blue utility truck with the number 36 is parked at the bottom "
    "left, white cargo containers marked 115B sit at the bottom center, and a yellow mobile "
    "crane vehicle works at the bottom right. Multi-level catwalks with yellow railings line "
    "both walls, crowded with personnel in black uniforms. Thick pipes, cables and ducts cover "
    "the ceiling, bright skylight panels pour pale light into the hall, and a glowing white "
    "opening at the far end leads into space. Floating in the upper left, an astronaut in an "
    "orange spacesuit drifts in zero gravity, and through a huge window we see the blue Earth "
    "with a squadron of small fighter ships flying past. Atmosphere: industrial, busy, hazy, "
    "volumetric light shafts, steam, teal and orange color grading, high contrast, wide angle "
    "lens, sharp focus, intricate mechanical details, octane render, unreal engine 5, trending "
    "on artstation, in the style of syd mead and star wars concept art"
)

DRAGON_LONG = (
    "epic fantasy illustration, photorealistic, cinematic, 8k, highly detailed, award winning. "
    "A massive dragon with grey armored scales, a crest of red spikes along its neck and head, "
    "and huge tattered leathery wings glowing orange against the sun swoops over a medieval "
    "harbor town at sunrise, its jaws open to show rows of sharp teeth, its clawed feet "
    "reaching down and a long barbed tail curling behind it. In the right foreground a huge "
    "weathered bronze bell hangs from a dark wooden beam in a bell tower, its surface patched "
    "with green verdigris, and a knight's steel gauntlet grips the thick bell rope, ready to "
    "ring the alarm. Below, an army of soldiers in dark armor with spears marches through "
    "narrow cobblestone streets between half-timbered houses with steep slate roofs, stone "
    "chimneys and small glowing windows. Tall-masted galleons with furled sails and red flags "
    "crowd the calm bay, a stone castle tower rises on a hill across the water, and green "
    "cliffs frame the coast on the left. Flocks of seagulls circle in the golden sky. On the "
    "town wall a wizard in blue robes casts a ball of fire at the dragon, a second smaller "
    "dragon circles above the castle, and a ship in the harbor is burning with thick black "
    "smoke. Dramatic backlight, lens flare, god rays, warm golden hour glow, teal shadows, "
    "volumetric haze, epic scale, 35mm film, depth of field, sharp focus, artstation, "
    "greg rutkowski, game of thrones"
)

MARKET_LONG = (
    "Ground-level view down a crowded Renaissance market square in Florence, 16:9. "
    "Plane 1: a wooden merchant cart in the immediate foreground, its edge cropped by the frame, "
    "loaded with copper pots, wheels of cheese, and hanging dried herbs, every rope fiber and "
    "dent visible, a bundle of wheat stalks tied with twine lying across the cheeses, and a large "
    "spoked wooden wheel. Plane 2: a lane of canvas-canopied stalls receding into the square, "
    "cloth merchants unrolling bolts of dyed fabric in orange, blue and cream stripes over a "
    "wooden table, a falconer with a grey falcon perched on his gloved fist, monks in brown "
    "hooded robes walking side by side, ladies in brocade gowns of gold and red silk, chickens "
    "underfoot on the cobblestones, wicker baskets on the ground, strings of garlic and dried "
    "peppers hanging from a stall, banners strung between timber-framed buildings. Plane 3: "
    "Brunelleschi-style cathedral dome rising above the rooftops at the far end with a tall "
    "bell tower beside it, hazy in golden afternoon light, ochre and cream palazzo facades "
    "with green shutters lining both sides. A juggler in a harlequin costume performs for a "
    "group of children at the center, a white horse pulls a carriage past the fountain, and a "
    "town crier on a wooden platform reads a scroll. Photorealistic oil-painting realism, sharp "
    "detail, warm color palette, soft shadows, cinematic composition, masterpiece, best quality, "
    "8k, rich textures, in the style of a Renaissance genre painting and National Geographic "
    "photography, natural light, depth of field"
)

FACE_LONG = (
    "breathtaking, striking cinematic dark fantasy, low angle perspective, dream girl fantasy "
    "character, strangely attractive face, striking otherworldly beauty, pale porcelain skin, "
    "piercing eyes, full dark lips, long hair in loose waves falling over her shoulders, a "
    "delicate silver circlet on her brow, wearing a black velvet gown with a high lace collar "
    "and a silver pendant, dense enchanted woodland filled with towering twisted oaks and thick "
    "undergrowth, forest floor covered in blood red flowers, ancient crumbling stone ruins "
    "half-swallowed by ivy, a broken stone archway behind her, night sky dominated by a massive "
    "blood-red full moon with faint nebula streaks visible through the forest canopy, a raven "
    "perched on a dead branch, glowing fireflies drifting between the trees, a black wolf "
    "watching from the shadows, candles flickering on a stone altar, light mist, dramatic "
    "chiaroscuro lighting, deep crimson, obsidian black, rich violet, gold accents, ethereal, "
    "ominous, mesmerizing, intensely atmospheric, gothic, romantic, painterly, highly detailed "
    "skin texture, sharp focus on the eyes, shallow depth of field, 85mm portrait lens, "
    "volumetric moonlight, rim light, subtle film grain, masterpiece, best quality, trending on "
    "artstation, by charlie bowater and tom bagshaw"
)


@dataclass(frozen=True)
class Scene:
    key: str
    source: str
    tiles: tuple
    prompts: dict = field(default_factory=dict)


SCENES = {s.key: s for s in (
    Scene("cyber8k", "samples/cyberpunk-city-8k.webp", (0, 9, 14),
          {"none": "", "short": "Cyberpunk cityscape at night", "long": CYBER_LONG}),
    Scene("hangar8k", "samples/orbital-shipyard-hangar-8k.webp", (3, 10, 21),
          {"none": "", "short": "A spaceship being built inside a huge hangar", "long": HANGAR_LONG}),
    Scene("dragon4k", "samples/dragon-4x-vl-upscale.png", (0, 2, 4),
          {"none": "", "short": "A dragon attacks a medieval harbor town", "long": DRAGON_LONG}),
    Scene("market4k", "samples/renaissance-market-4x-vl-upscale.png", (1, 3, 5),
          {"none": "", "short": "Renaissance market square in Florence", "long": MARKET_LONG}),
    Scene("face3k", "tests-AB/cache/krea2-00676_upscale_2304x3072_*.pt", (1, 3, 4),
          {"none": "", "short": "Dark fantasy portrait of a woman in a forest at night", "long": FACE_LONG}),
)}
PROMPT_KEYS = ("none", "short", "long")


def load_canvas(scene):
    import numpy as np
    import torch
    from PIL import Image

    if scene.source.endswith(".pt"):
        matches = sorted(REPO_ROOT.glob(scene.source))
        if not matches:
            raise SystemExit(f"no canvas matches {scene.source}")
        return torch.load(matches[0], map_location="cpu").float()[..., :3].contiguous()
    with Image.open(REPO_ROOT / scene.source) as handle:
        pixels = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels)[None,]


def solve_layout(canvas):
    from context_anchored_tile_refine import grid

    height, width = int(canvas.shape[1]), int(canvas.shape[2])
    sx = grid.solve_axis(width, MAX_TILE_WIDTH, CONTEXT_ANCHOR, CONTEXT_OVERLAP, axis="width")
    sy = grid.solve_axis(height, MAX_TILE_HEIGHT, CONTEXT_ANCHOR, CONTEXT_OVERLAP, axis="height")
    return grid.build_layout(width, height, sx, sy, CONTEXT_ANCHOR, CONTEXT_OVERLAP)


def to_pil(image, long_side):
    import numpy as np
    import torch
    from PIL import Image

    height, width = int(image.shape[1]), int(image.shape[2])
    scale = min(1.0, long_side / max(height, width))
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    if scale < 1.0:
        image = torch.nn.functional.interpolate(image.movedim(-1, 1), size=size, mode="area").movedim(1, -1)
    pixels = (image[0].clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(pixels)


def cmd_layouts(_args):
    from PIL import ImageDraw

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    for scene in SCENES.values():
        canvas = load_canvas(scene)
        layout = solve_layout(canvas)
        picture = to_pil(canvas, 1600)
        scale = picture.width / int(canvas.shape[2])
        draw = ImageDraw.Draw(picture)
        for index, tile in enumerate(layout.tiles):
            r = tile.crop_rect
            box = [r.x0 * scale, r.y0 * scale, r.x1 * scale - 1, r.y1 * scale - 1]
            draw.rectangle(box, outline=(255, 0, 255), width=2)
            c = tile.core
            draw.text((c.x0 * scale + 6, c.y0 * scale + 6), str(index), fill=(255, 255, 0))
        out = BENCH_DIR / f"layout-{scene.key}.png"
        picture.save(out)
        print(f"{scene.key}: {canvas.shape[2]}x{canvas.shape[1]}, {len(layout.tiles)} tiles -> {out}")


def save_judge_tiles(scene, canvas, layout):
    from PIL import ImageDraw

    TILE_DIR.mkdir(parents=True, exist_ok=True)
    for index in scene.tiles:
        plain = TILE_DIR / f"{scene.key}-t{index:02d}.jpg"
        gridded = TILE_DIR / f"{scene.key}-t{index:02d}-grid.jpg"
        if plain.exists() and gridded.exists():
            continue
        r = layout.tiles[index].crop_rect
        picture = to_pil(canvas[:, r.y0:r.y1, r.x0:r.x1, :], JUDGE_LONG_SIDE)
        picture.save(plain, quality=92)
        draw = ImageDraw.Draw(picture)
        for i in (1, 2):
            x = picture.width * i // 3
            y = picture.height * i // 3
            draw.line([(x, 0), (x, picture.height)], fill=(255, 0, 255), width=2)
            draw.line([(0, y), (picture.width, y)], fill=(255, 0, 255), width=2)
        picture.save(gridded, quality=92)


# ---------------------------------------------------------------- arms
#
# An arm is fn(ctx, scene, prompt, canvas, layout) -> {"picture_ms", "tiles": {index: {...}}},
# where each tile carries "items" as [[item, term], ...] and "ms".


@dataclass
class Context:
    clip: object
    torch: object


def shipped_preset(prompt):
    from context_anchored_tile_refine import captions

    preset = captions.with_prompt(captions.resolve_method(captions.default_vlm_method()), prompt)
    return replace(preset, style_instruction="")


def time_tiles(ctx, run_fn):
    """Runs run_fn() with trace_tile timed per call. Returns (result, per-call ms, total ms)."""
    from context_anchored_tile_refine import tags

    original = tags.trace_tile
    per_tile = []

    def timed(*args, **kwargs):
        ctx.torch.cuda.synchronize()
        started = time.perf_counter()
        result = original(*args, **kwargs)
        ctx.torch.cuda.synchronize()
        per_tile.append((time.perf_counter() - started) * 1000.0)
        return result

    tags.trace_tile = timed
    try:
        ctx.torch.cuda.synchronize()
        started = time.perf_counter()
        result = run_fn()
        ctx.torch.cuda.synchronize()
        total = (time.perf_counter() - started) * 1000.0
    finally:
        tags.trace_tile = original
    return result, per_tile, total


def tile_record(trace, ms):
    """One tile's result, with every candidate's verify score and every kept item's strips, so a
    stricter threshold can be judged offline from the same record."""
    placed = [(item, term) for item, term in zip(trace.kept, trace.terms, strict=True)
              if item not in trace.unplaced]
    return {"items": [list(pair) for pair in placed], "ms": round(ms, 1), "text": trace.text,
            "reply": trace.reply,
            "candidates": [[c, o, s] for c, o, s in zip(trace.candidates, trace.origins, trace.scores, strict=True)],
            "kept": list(trace.kept), "strips": [list(p) for p in trace.strips], "terms": list(trace.terms)}


def trace_record(run, tiles, per_tile, total):
    record = {"picture_ms": round(total - sum(per_tile), 1), "tiles": {}}
    for index, row_traces, ms in zip(tiles, run.tiles, per_tile, strict=True):
        record["tiles"][str(index)] = tile_record(row_traces[0], ms)
    return record


def arm_baseline(ctx, scene, prompt, canvas, layout):
    from context_anchored_tile_refine import tags

    tiles = [layout.tiles[i] for i in scene.tiles]
    preset = shipped_preset(prompt)
    run, per_tile, total = time_tiles(ctx, lambda: tags.generate_tag_trace(ctx.clip, canvas, tiles, preset))
    return trace_record(run, scene.tiles, per_tile, total)


# The prompt terms pass: ONE text-only generate per picture lists the visible things the prompt
# names, then every tile verifies them, and each tile's own propose never reads the prompt.
TERMS_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
TERMS_INSTRUCTION = (
    "This is a prompt for an image:\n{prompt}\n\n"
    "List every visible thing the prompt names: objects, people, animals, clothing, materials, "
    "places and parts of the scene. Use short lowercase noun phrases of one to four words, "
    "separated by commas. Keep each thing's own descriptive words, such as its color. Leave out "
    "the image's style, medium, quality, camera, lighting mood and any position words. "
    "List each thing once. Output only the list."
)
TERMS_MAX_TOKENS = 256
# Two invented terms in a row end the list: past the prompt's own things the model pads with
# generic words ("materials", "details") until the budget runs out.
UNGROUNDED_STREAK = 2
_STOP_WORDS = frozenset({"a", "an", "the", "of", "with", "and", "in", "on", "at", "to", "for", "by", "from"})


def _word_stems(text):
    import re

    stems = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        for suffix in ("ies", "es", "s"):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                word = word[:-len(suffix)] + ("y" if suffix == "ies" else "")
                break
        stems.add(word)
    return stems


def grounded(term, prompt_stems):
    """Whether every content word of `term` is a word of the prompt, plural or not."""
    words = _word_stems(term) - _STOP_WORDS
    return bool(words) and words <= prompt_stems


def stop_when(clip, should_stop):
    """tags.stop_on_repeat's patch with a caller's predicate over the complete tags so far."""
    from contextlib import contextmanager

    from logit_classifier.tags import complete_tags

    from context_anchored_tile_refine import tags

    @contextmanager
    def scope():
        transformer, stop_id = tags._sampler(clip)
        original = transformer.sample_token
        history = []

        def watch(*args, **kwargs):
            token = original(*args, **kwargs)
            history.append(int(token.reshape(-1)[0]))
            if should_stop(complete_tags(clip.decode(history))):
                return token.new_full(token.shape, stop_id)
            return token

        transformer.sample_token = watch
        try:
            yield
        finally:
            del transformer.sample_token
    return scope()


def extract_prompt_terms(clip, prompt):
    """(terms, reply, tokens written): the grounded noun phrases the prompt names."""
    from logit_classifier.tags import drop_unfinished_tag, parse_candidates, repeated_block

    from context_anchored_tile_refine import captions

    prompt_stems = _word_stems(prompt)

    def should_stop(tags_so_far):
        streak = 0
        for tag in reversed(tags_so_far):
            if grounded(tag, prompt_stems):
                break
            streak += 1
        return streak >= UNGROUNDED_STREAK or bool(repeated_block(tags_so_far))

    tokens = clip.tokenize(TERMS_TEMPLATE.format(instruction=TERMS_INSTRUCTION.format(prompt=prompt)))
    with stop_when(clip, should_stop):
        ids = captions.clip_generate(clip, tokens, do_sample=False, max_length=TERMS_MAX_TOKENS)
    reply = clip.decode(ids)
    text = drop_unfinished_tag(reply) if len(ids) >= TERMS_MAX_TOKENS else reply
    terms = [t for t in parse_candidates(text) if grounded(t, prompt_stems)]
    return terms, reply, len(ids)


def arm_terms(ctx, scene, prompt, canvas, layout, propose_with_prompt=False):
    from context_anchored_tile_refine import captions, tags

    preset = shipped_preset(prompt)
    tile_preset = preset if propose_with_prompt else replace(preset, prompt="")
    classifier = tags.build_classifier(ctx.clip)
    ctx.torch.cuda.synchronize()
    started = time.perf_counter()
    terms, reply, written, style_p, subjects = [], "", 0, (), ()
    if prompt.strip():
        terms, reply, written = extract_prompt_terms(ctx.clip, prompt)
        style_p = tags.fragment_style_p(classifier, tuple(terms))
        subjects, _styles = tags.sort_fragments(tuple(terms), style_p)
    ctx.torch.cuda.synchronize()
    record = {"picture_ms": round((time.perf_counter() - started) * 1000.0, 1), "tiles": {},
              "prompt_terms": terms, "subject_terms": list(subjects), "terms_reply": reply,
              "terms_tokens": written}
    for index in scene.tiles:
        crop = layout.tiles[index].crop_rect
        row = canvas[:, crop.y0:crop.y1, crop.x0:crop.x1, :]
        ctx.torch.cuda.synchronize()
        started = time.perf_counter()
        picture = captions.resample_for_vl(row, tags.VL_MAX_PIXELS)
        trace = tags.trace_tile(ctx.clip, classifier, row, picture, tile_preset, subjects)
        ctx.torch.cuda.synchronize()
        record["tiles"][str(index)] = tile_record(trace, (time.perf_counter() - started) * 1000.0)
    return record


# ---------------------------------------------------------------- speed options
#
# graphs      measured as arm base+graphs, now production (captions.clip_generate), so every
#             arm run after it decodes with graphs on.
# loadskip    clip.load_model skipped while the CLIP is the resident head of the loaded list.
# vision      one vision encode per picture per tile, shared by propose and verify, computed
#             inside logit_classifier's determinism window.


def _resident(clip):
    import comfy.model_management as mm

    patcher = clip.patcher
    head = mm.current_loaded_models
    return (bool(head) and head[0].model is patcher and patcher.model.device == patcher.load_device
            and patcher.model.current_weight_patches_uuid == patcher.patches_uuid
            and patcher.model.model_loaded_weight_memory > 0)


@contextmanager
def load_skip(clip):
    original = clip.load_model

    def load_model(*args, **kwargs):
        if _resident(clip):
            return clip.patcher
        return original(*args, **kwargs)

    clip.load_model = load_model
    try:
        yield
    finally:
        del clip.load_model


@contextmanager
def vision_share(clip):
    import torch
    from logit_classifier.backends._torch_window import _determinism

    from context_anchored_tile_refine import tags

    model = clip.cond_stage_model
    transformer = getattr(model, model.clip).transformer
    original = transformer.preprocess_embed
    original_trace = tags.trace_tile
    cache = {}

    def cached(embed, device):
        data = embed.get("data")
        if embed.get("type") != "image" or not torch.is_tensor(data):
            return original(embed, device=device)
        key = (id(data), tuple(data.shape), data.dtype, str(device))
        if key not in cache:
            with _determinism():
                cache[key] = (data, original(embed, device=device))
        return cache[key][1]

    def trace_tile(*args, **kwargs):
        try:
            return original_trace(*args, **kwargs)
        finally:
            cache.clear()

    transformer.preprocess_embed = cached
    tags.trace_tile = trace_tile
    try:
        yield
    finally:
        del transformer.preprocess_embed
        tags.trace_tile = original_trace


def with_options(arm, options):
    def run(ctx, *args):
        from contextlib import ExitStack

        with ExitStack() as stack:
            if "loadskip" in options:
                stack.enter_context(load_skip(ctx.clip))
            if "vision" in options:
                stack.enter_context(vision_share(ctx.clip))
            return arm(ctx, *args)
    return run


ARMS = {
    "baseline": arm_baseline,
    "terms": arm_terms,
    "terms-prompted": lambda *a: arm_terms(*a, propose_with_prompt=True),
    "base+graphs": arm_baseline,
    "base+loadskip": with_options(arm_baseline, {"loadskip"}),
    "base+vision": with_options(arm_baseline, {"vision"}),
    "base+fast": with_options(arm_baseline, {"loadskip", "vision"}),
    "terms+fast": with_options(arm_terms, {"loadskip", "vision"}),
}


def cmd_run(args):
    ab_env.bootstrap()
    import ab_models
    import torch

    from context_anchored_tile_refine import captions, tags

    arm = ARMS[args.arm]
    scenes = [SCENES[k] for k in args.scenes.split(",")] if args.scenes else list(SCENES.values())
    prompts = args.prompts.split(",") if args.prompts else list(PROMPT_KEYS)
    out = RUNS_DIR / f"{args.arm}.json"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    record = json.loads(out.read_text()) if out.exists() and not args.fresh else {}
    ctx = Context(clip=ab_models.load_clip(CLIP_NAME, "krea2"), torch=torch)

    for scene in scenes:
        canvas = load_canvas(scene)
        layout = solve_layout(canvas)
        save_judge_tiles(scene, canvas, layout)
        for prompt_key in prompts:
            captions.clear_caption_cache()
            tags.clear_tag_cache()
            with torch.inference_mode():
                result = arm(ctx, scene, scene.prompts[prompt_key], canvas, layout)
            result["grid_tiles"] = len(layout.tiles)
            record.setdefault(scene.key, {})[prompt_key] = result
            out.write_text(json.dumps(record, indent=1))
            ms = [t["ms"] for t in result["tiles"].values()]
            print(f"{args.arm} {scene.key} {prompt_key}: picture {result['picture_ms'] / 1000:.1f} s, "
                  f"tile {statistics.mean(ms) / 1000:.1f} s, items "
                  f"{[len(t['items']) for t in result['tiles'].values()]}", flush=True)
    print(f"written {out}")


# ---------------------------------------------------------------- judging

JUDGE_RUBRIC = """You judge tags a vision model wrote for ONE image tile. Look at the plain image,
and at the gridded copy, whose magenta lines cut it into thirds (3 rows x 3 columns).

For every item in "items", answer one of:
  yes     the named thing is clearly visible in this image (a part of it counts, cropped counts)
  no      the named thing is not in this image, or the name is wrong for what is there
  vague   not a visible thing: a mood, style, quality, idea, lighting word, lone color, a
          whole-scene description such as "cityscape at night", or too generic to point at
  unsure  too small or ambiguous to decide
Judge the words literally. "red neon" needs red neon light, "glowing windows" needs windows
that glow. A long phrase is "yes" only when every thing it names is visible here.

For every [item, term] in "positions", judge only the position term (assume the item is there):
  right   a clear part of the item lies in the named band. Words name bands of the 3x3 grid:
          top / center / bottom are rows, left / center / right are columns. "top-left" needs
          both. A single word (e.g. "left", "bottom") names that band only. A lone "center"
          is right when the item is in the middle row OR the middle column.
  wrong   the item is not in the named band, or it is spread evenly over the whole image so a
          band name misleads
Write JSON only: {"items": {item: verdict}, "positions": {"item|term": verdict}}."""


def load_judgments():
    return json.loads(JUDGMENTS.read_text()) if JUDGMENTS.exists() else {}


def run_records(arms=None):
    records = {}
    for path in sorted(RUNS_DIR.glob("*.json")):
        if arms and path.stem not in arms:
            continue
        records[path.stem] = json.loads(path.read_text())
    return records


def cmd_tasks(_args):
    judgments = load_judgments()
    wanted = {}
    for record in run_records().values():
        for scene_key, by_prompt in record.items():
            for result in by_prompt.values():
                for index, tile in result["tiles"].items():
                    key = f"{scene_key}|{int(index):02d}"
                    done = judgments.get(key, {"items": {}, "positions": {}})
                    task = wanted.setdefault(key, {"items": set(), "positions": set()})
                    for item, term in tile["items"]:
                        if item not in done["items"]:
                            task["items"].add(item)
                        if term and f"{item}|{term}" not in done["positions"]:
                            task["positions"].add((item, term))
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    for stale in JUDGE_DIR.glob("*.task.json"):
        stale.unlink()
    count = 0
    for key, task in sorted(wanted.items()):
        if not task["items"] and not task["positions"]:
            continue
        scene_key, index = key.split("|")
        name = f"{scene_key}-t{index}"
        body = {"tile": key, "image": str(TILE_DIR / f"{name}.jpg"), "grid_image": str(TILE_DIR / f"{name}-grid.jpg"),
                "items": sorted(task["items"]), "positions": sorted(list(p) for p in task["positions"]),
                "verdict_file": str(JUDGE_DIR / f"{name}.verdict.json")}
        (JUDGE_DIR / f"{name}.task.json").write_text(json.dumps(body, indent=1))
        count += 1
        print(f"{name}: {len(task['items'])} items, {len(task['positions'])} positions")
    (JUDGE_DIR / "RUBRIC.txt").write_text(JUDGE_RUBRIC)
    print(f"{count} task files in {JUDGE_DIR}")


def cmd_merge(_args):
    judgments = load_judgments()
    for path in sorted(JUDGE_DIR.glob("*.verdict.json")):
        task = json.loads(path.with_name(path.name.replace(".verdict.", ".task.")).read_text())
        verdict = json.loads(path.read_text())
        entry = judgments.setdefault(task["tile"], {"items": {}, "positions": {}})
        missing = [i for i in task["items"] if i not in verdict.get("items", {})]
        missing += [f"{i}|{t}" for i, t in task["positions"] if f"{i}|{t}" not in verdict.get("positions", {})]
        if missing:
            print(f"{path.name}: {len(missing)} verdicts missing, e.g. {missing[:3]}")
        entry["items"].update(verdict.get("items", {}))
        entry["positions"].update(verdict.get("positions", {}))
        path.rename(path.with_suffix(".merged"))
    JUDGMENTS.write_text(json.dumps(judgments, indent=1, sort_keys=True))
    print(f"judgments for {len(judgments)} tiles in {JUDGMENTS}")


# ---------------------------------------------------------------- report

def score(results, judgments):
    counts = {"tiles": 0, "items": 0, "yes": 0, "no": 0, "vague": 0, "unsure": 0, "unjudged": 0,
              "pos_right": 0, "pos_wrong": 0, "termed": 0, "ms": [], "picture_ms": [], "projected_s": []}
    for scene_key, result in results:
        tile_ms = []
        for index, tile in result["tiles"].items():
            entry = judgments.get(f"{scene_key}|{int(index):02d}", {"items": {}, "positions": {}})
            counts["tiles"] += 1
            tile_ms.append(tile["ms"])
            for item, term in tile["items"]:
                counts["items"] += 1
                verdict = entry["items"].get(item)
                counts[verdict if verdict in ("yes", "no", "vague", "unsure") else "unjudged"] += 1
                if term and verdict == "yes":
                    counts["termed"] += 1
                    position = entry["positions"].get(f"{item}|{term}")
                    if position in ("right", "wrong"):
                        counts[f"pos_{position}"] += 1
        counts["ms"].extend(tile_ms)
        counts["picture_ms"].append(result["picture_ms"])
        counts["projected_s"].append((result["picture_ms"] + result["grid_tiles"] * statistics.mean(tile_ms)) / 1000)
    return counts


def format_row(label, c):
    judged = c["yes"] + c["no"] + c["vague"]
    precision = c["yes"] / judged if judged else math.nan
    wrong = c["no"] / judged if judged else math.nan
    positions = c["pos_right"] + c["pos_wrong"]
    position_right = c["pos_right"] / positions if positions else math.nan
    return (f"{label:<22}{c['tiles']:>5}{c['items'] / max(c['tiles'], 1):>7.1f}{c['yes'] / max(c['tiles'], 1):>7.1f}"
            f"{precision:>7.3f}{wrong:>7.3f}{c['vague'] / max(judged, 1):>7.3f}{position_right:>7.3f}"
            f"{positions:>6}{statistics.mean(c['ms']) / 1000:>8.2f}{statistics.mean(c['picture_ms']) / 1000:>8.2f}"
            f"{statistics.mean(c['projected_s']):>9.1f}{c['unjudged']:>6}")


ITEM_KINDS = BENCH_DIR / "item_kinds.json"


def cmd_classify_items(_args):
    """p(style) of every item any arm emitted, from tags.fragment_style_p, into item_kinds.json."""
    ab_env.bootstrap()
    import ab_models
    import torch

    from context_anchored_tile_refine import tags

    kinds = json.loads(ITEM_KINDS.read_text()) if ITEM_KINDS.exists() else {}
    items = sorted({item for record in run_records().values() for by_prompt in record.values()
                    for result in by_prompt.values() for tile in result["tiles"].values()
                    for item, _term in tile["items"]} - set(kinds))
    clip = ab_models.load_clip(CLIP_NAME, "krea2")
    classifier = tags.build_classifier(clip)
    with torch.inference_mode():
        for start in range(0, len(items), 64):
            chunk = tuple(items[start:start + 64])
            kinds.update(zip(chunk, tags.fragment_style_p(classifier, chunk), strict=True))
    ITEM_KINDS.write_text(json.dumps(kinds, indent=1, sort_keys=True))
    print(f"{len(items)} new items classified, {len(kinds)} in {ITEM_KINDS}")


def filtered(record, spec, kinds):
    """The record with each tile's items cut by one offline filter: vT keeps items whose
    verify score is at least T, kT keeps items whose p(style) is below T."""
    kind, threshold = spec[0], float(spec[1:])
    out = {}
    for scene_key, by_prompt in record.items():
        for prompt_key, result in by_prompt.items():
            tiles = {}
            for index, tile in result["tiles"].items():
                scores = {c: s for c, _o, s in tile.get("candidates", [])}
                if kind == "v":
                    keep = [p for p in tile["items"] if (scores.get(p[0]) or 0.0) >= threshold]
                else:
                    keep = [p for p in tile["items"] if kinds.get(p[0], 0.0) < threshold]
                tiles[index] = {**tile, "items": keep}
            out.setdefault(scene_key, {})[prompt_key] = {**result, "tiles": tiles}
    return out


def cmd_report(args):
    judgments = load_judgments()
    arms = args.arms.split(",") if args.arms else None
    records = run_records(arms)
    kinds = json.loads(ITEM_KINDS.read_text()) if ITEM_KINDS.exists() else {}
    for spec in (s for s in args.filters.split(",") if s):
        for name in list(records):
            if "+" not in name or name.split("+")[-1][0] not in "vk":
                records[f"{name}+{spec}"] = filtered(records[name], spec, kinds)
    header = (f"{'arm / prompt':<22}{'tiles':>5}{'items':>7}{'yes':>7}{'prec':>7}{'wrong':>7}{'vague':>7}"
              f"{'pos ok':>7}{'pos n':>6}{'s/tile':>8}{'pic s':>8}{'grid s':>9}{'unjdg':>6}")
    print("items and yes are per tile. prec = yes / (yes + no + vague). pos ok = right terms of "
          "judged terms on yes items.\ngrid s = picture + the scene's full tile count x mean tile time, "
          "averaged over scenes.")
    print(header)
    for name, record in records.items():
        for prompt_key in (*PROMPT_KEYS, "all"):
            results = [(scene_key, by_prompt[p]) for scene_key, by_prompt in record.items()
                       for p in by_prompt if prompt_key in ("all", p)]
            if results:
                print(format_row(f"{name} / {prompt_key}", score(results, judgments)))
        print()


def cmd_compare(args):
    """Whether two arms wrote the same reply and the same items per tile."""
    first, second = (json.loads((RUNS_DIR / f"{name}.json").read_text()) for name in (args.a, args.b))
    same_reply = same_items = total = 0
    for scene_key, by_prompt in first.items():
        for prompt_key, result in by_prompt.items():
            other = second.get(scene_key, {}).get(prompt_key)
            if other is None:
                continue
            for index, tile in result["tiles"].items():
                twin = other["tiles"][index]
                total += 1
                same_reply += tile["reply"] == twin["reply"]
                same_items += tile["items"] == twin["items"]
                if tile["items"] != twin["items"]:
                    only_a = [i for i in tile["items"] if i not in twin["items"]]
                    only_b = [i for i in twin["items"] if i not in tile["items"]]
                    print(f"{scene_key} {prompt_key} t{index}: only {args.a} {only_a}, only {args.b} {only_b}")
    print(f"{total} tiles: same reply {same_reply}, same items {same_items}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("layouts")
    run = sub.add_parser("run")
    run.add_argument("--arm", required=True, choices=sorted(ARMS))
    run.add_argument("--scenes", default="")
    run.add_argument("--prompts", default="")
    run.add_argument("--fresh", action="store_true", help="drop the arm's earlier record")
    sub.add_parser("tasks")
    sub.add_parser("merge")
    compare = sub.add_parser("compare")
    compare.add_argument("--a", required=True)
    compare.add_argument("--b", required=True)
    sub.add_parser("classify-items")
    report = sub.add_parser("report")
    report.add_argument("--arms", default="")
    report.add_argument("--filters", default="", help="csv of offline filters, vT (verify) or kT (p(style))")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "layouts":
        ab_env.bootstrap()
    {"layouts": cmd_layouts, "run": cmd_run, "tasks": cmd_tasks, "merge": cmd_merge,
     "report": cmd_report, "compare": cmd_compare, "classify-items": cmd_classify_items}[args.command](args)


if __name__ == "__main__":
    main()
