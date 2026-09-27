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
    python tests-AB/ab_tags_bench.py export                  # every tile as a PNG, and manifest.json
"""

import argparse
import hashlib
import io
import itertools
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import ab_env

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = REPO_ROOT / "tests-AB" / "cache" / "tags_bench"
RUNS_DIR = BENCH_DIR / "runs"
JUDGE_DIR = BENCH_DIR / "judge"
TILE_DIR = BENCH_DIR / "tiles"
EXPORT_DIR = BENCH_DIR / "export"
JUDGMENTS = BENCH_DIR / "judgments.json"
CLIP_NAME = "qwen3vl_4b_fp8_scaled.safetensors"
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

# The hold-out set: tiles and prompts no arm was tuned on. "tags" is a comma tag prompt in the
# style of SD 1.5 and SDXL workflows, "xlong" is 450 to 650 words. Both name absent things.
CITY2_TAGS = (
    "masterpiece, best quality, ultra detailed, 8k, cyberpunk, city, night, skyscrapers, neon lights, "
    "empire state building, pink spire, radio antenna, river, city lights, smoke, fog, dark sky, stars, "
    "flying cars, rain, puddles, holograms, cinematic lighting, purple, blue, wide shot, from above, "
    "no humans, highly detailed, sharp focus"
)
CITY2_XLONG = (
    "A breathtaking cinematic aerial photograph of a vast cyberpunk metropolis at night, taken from the "
    "top of a skyscraper high above the streets, ultra detailed, 8k, award winning, photorealistic. The "
    "city stretches all the way to the horizon, where a wide dark river bends to the left and a long "
    "bridge lined with orange lights crosses it. Along the far shore thousands of tiny lights glitter "
    "like a sea of embers. In the center of the frame rises an art deco tower modeled on the Empire "
    "State Building, its stepped crown lit in soft pink and its needle spire glowing ice blue against "
    "the black sky. Far behind it, a very thin television tower with a red tip stands alone on the "
    "left. On the right side of the image a sharply angled glass skyscraper with a slanted, faceted top "
    "glows from within in violet and magenta, and above it a tall lattice radio mast is lit hot pink "
    "from base to tip. Behind that mast, columns of white steam rise from rooftop vents and drift to "
    "the right. A second red and white radio tower stands at the far right edge. On the far right a "
    "tall dark building carries a huge glowing billboard with red and white letters and a white apple "
    "logo. In the foreground, dense blocks of office towers and apartment buildings fill the lower "
    "half of the frame, each covered in grids of lit windows in warm yellow, cold white and teal, "
    "with a few red aircraft warning lights blinking on their roofs. A narrow avenue cuts through the "
    "buildings on the right, glowing orange with streetlights and the tail lights of traffic. Pink and "
    "red neon signs hang from several towers, one reading LAMOR in pink letters near the center. A "
    "small green glowing pyramid roof sits on a tower left of center. Snow flurries drift through "
    "the air. The sky is a deep navy with thin wisps of cloud and a few faint stars, and a soft purple "
    "haze hangs over the skyline. Flying cars with bright headlights weave between the towers, a giant "
    "hologram of a koi fish swims above the river, searchlight beams sweep the clouds, and a blimp "
    "with a glowing screen floats over the bridge. Rain slicked rooftops reflect the neon, and a "
    "crowd of people with umbrellas fills a plaza at the bottom of the image. Mood: lonely, electric, "
    "melancholic, dreamlike, blade runner, cyberpunk 2077, ghost in the shell. Style: long exposure "
    "night photography, shot on a Sony A7R IV, 24mm wide angle lens, f/8, deep depth of field, sharp "
    "focus from foreground to horizon, cinematic color grading with deep blues, violet and magenta "
    "highlights, high dynamic range, crisp details, subtle film grain, volumetric haze, light bloom, "
    "lens flare, trending on artstation, featured on 500px, masterpiece, best quality, highly detailed, "
    "intricate, octane render, unreal engine 5, ray tracing, hdr, 8k uhd"
)
STYLE_TAIL = (
    " Rendering: masterpiece, best quality, ultra detailed, 8k uhd, hdr, sharp focus, intricate "
    "details, physically based rendering, ray tracing, global illumination, subsurface scattering, "
    "volumetric lighting, soft shadows, cinematic color grading, film grain, bokeh, depth of field, "
    "award winning, trending on artstation, featured on behance, unreal engine 5, octane render, "
    "photorealistic, hyperrealistic, studio quality, professional, vivid colors, high contrast, "
    "dramatic composition, rule of thirds, golden ratio, masterful lighting, epic, stunning, gorgeous."
)
HANGAR_EXTRA = (
    " More detail: the deck is painted with yellow hazard stripes and numbered bay markings, puddles "
    "of coolant reflect the lights, and loose cables snake across the floor. Hover drones with "
    "spotlights inspect the hull, a stack of spare engine nozzles sits beside a tool cart, and a "
    "foreman with a tablet points at the ship. Behind the glass of a control tower on the left, "
    "officers watch holographic displays. Warning lights spin orange, a banner with a golden eagle "
    "emblem hangs from the ceiling, and a row of escape pods waits along the right wall."
)
DRAGON_EXTRA = (
    " More detail: the soldiers carry round shields painted with a white tree, archers on the "
    "rooftops draw their bows, and a cart of hay has overturned in the street. Laundry hangs between "
    "the houses, a black cat watches from a window sill, and church bells ring in a steeple. On the "
    "water a small rowing boat carries fishermen, and a lighthouse stands at the end of the pier. A "
    "tattered banner with a red lion flies from the castle, and smoke rises from a blacksmith's forge."
)
MARKET_EXTRA = (
    " More detail: a spice seller weighs saffron on brass scales, pyramids of oranges and lemons fill "
    "wooden crates, and clay amphorae of olive oil stand in a row. A stray dog sniffs at a basket of "
    "bread, a nobleman in a feathered hat haggles with a merchant, and pigeons peck between the "
    "cobblestones. Garlands of flowers hang from a balcony, a stone well with a bucket sits by the "
    "fountain, and a painter at an easel sketches the cathedral."
)
FACE_EXTRA = (
    " More detail: her eyes are a vivid emerald green, a single tear runs down her cheek, and a thin "
    "silver chain with a crescent moon pendant rests on her collarbone. Pale moths circle her head, a "
    "crown of thorns and black roses sits in her hair, and rain drips from the leaves above. Behind "
    "her an owl watches from a hollow tree, a stone angel statue with broken wings stands among the "
    "ruins, and a stream of glowing blue water winds between the flowers."
)
HOLDOUT = {s.key: s for s in (
    Scene("city2", "samples/cyberpunk-city2-8k.webp", (1, 10, 19),
          {"none": "", "tags": CITY2_TAGS, "xlong": CITY2_XLONG}),
    Scene("hangar8k-h", "samples/orbital-shipyard-hangar-8k.webp", (5, 18),
          {"none": "", "tags": "masterpiece, best quality, sci-fi, spaceship, hangar, robots, workers, cranes, "
           "sparks, fire, trucks, containers, catwalks, pipes, astronaut, space station, laser, highly detailed",
           "xlong": HANGAR_LONG + HANGAR_EXTRA + STYLE_TAIL}),
    Scene("dragon4k-h", "samples/dragon-4x-vl-upscale.png", (3,),
          {"none": "", "tags": "epic fantasy, dragon, medieval town, harbor, sailing ships, soldiers, army, "
           "castle, bell tower, sunrise, seagulls, wizard, fire, smoke, cinematic, 8k, masterpiece",
           "xlong": DRAGON_LONG + DRAGON_EXTRA + STYLE_TAIL}),
    Scene("market4k-h", "samples/renaissance-market-4x-vl-upscale.png", (2,),
          {"none": "", "tags": "renaissance, florence, market, stalls, merchants, monks, cheese, copper pots, "
           "garlic, herbs, pumpkins, tapestry, cathedral, horse, juggler, crowd, oil painting, masterpiece",
           "xlong": MARKET_LONG + MARKET_EXTRA + STYLE_TAIL}),
    Scene("face3k-h", "tests-AB/cache/krea2-00676_upscale_2304x3072_*.pt", (5,),
          {"none": "", "tags": "1girl, solo, long hair, black hair, red flowers, forest, night, moon, ruins, "
           "dark fantasy, black dress, looking at viewer, wolf, candles, masterpiece, best quality",
           "xlong": FACE_LONG + FACE_EXTRA + STYLE_TAIL}),
)}
HOLDOUT_PROMPT_KEYS = ("none", "tags", "xlong")
SETS = {"main": (SCENES, PROMPT_KEYS), "holdout": (HOLDOUT, HOLDOUT_PROMPT_KEYS)}


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


# The shipped pass under the labels of its three recorded runs, which were logged at different
# commits. A rerun of any label runs today's code into that label's record.
ARMS = {
    "baseline": arm_baseline,
    "base+graphs": arm_baseline,
    "prod": arm_baseline,
}


def cmd_run(args):
    ab_env.bootstrap()
    import ab_models
    import torch

    from context_anchored_tile_refine import captions, tags

    arm = ARMS[args.arm]
    registry, prompt_keys = SETS[args.set]
    scenes = [registry[k] for k in args.scenes.split(",")] if args.scenes else list(registry.values())
    prompts = args.prompts.split(",") if args.prompts else list(prompt_keys)
    out = RUNS_DIR / f"{args.arm}.json"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    record = json.loads(out.read_text()) if out.exists() and not args.fresh else {}
    ctx = Context(clip=ab_models.load_clip(args.clip, "krea2"), torch=torch)

    for scene in scenes:
        canvas = load_canvas(scene)
        layout = solve_layout(canvas)
        save_judge_tiles(scene, canvas, layout)
        for prompt_key in prompts:
            captions.clear_caption_cache()
            tags.clear_tag_cache()
            with torch.inference_mode():
                result = arm(ctx, scene, scene.prompts[prompt_key], canvas, layout)
            # Each pass is one node execution in the app, which resets DynamicVRAM's buffers after it.
            ab_env.end_node_execution()
            result["grid_tiles"] = len(layout.tiles)
            result["clip"] = args.clip
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
        path.replace(path.with_name(f"{path.stem}.{len(judgments)}.merged"))
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
    tiles = max(c["tiles"], 1)
    return (f"{label:<28}{c['tiles']:>5}{c['items'] / tiles:>7.1f}{c['yes'] / tiles:>7.1f}"
            f"{c['no'] / tiles:>6.2f}{c['vague'] / tiles:>6.2f}"
            f"{precision:>7.3f}{wrong:>7.3f}{c['vague'] / max(judged, 1):>7.3f}{position_right:>7.3f}"
            f"{positions:>6}{statistics.mean(c['ms']) / 1000:>8.2f}{statistics.mean(c['picture_ms']) / 1000:>8.2f}"
            f"{statistics.mean(c['projected_s']):>9.1f}{c['unjudged']:>6}")


def filtered(record, spec):
    """`spec` joined by "&" applies each filter in turn."""
    for part in spec.split("&"):
        record = filtered_once(record, part)
    return record


def filtered_once(record, spec):
    """The record with each tile's items cut by one offline filter:
    vT  keeps items whose verify score is at least T
    pT  keeps model tags, and prompt-only terms whose verify score is at least T
    nN  keeps prompt-origin items and the model's first N proposed tags (greedy decoding makes a
        propose stopped after N tags write exactly these)"""
    kind, threshold = spec[0], float(spec[1:])
    out = {}
    if kind not in ("v", "p", "n"):
        raise SystemExit(f"unknown filter {spec!r}. The filters are vT, pT and nN.")
    for scene_key, by_prompt in record.items():
        for prompt_key, result in by_prompt.items():
            tiles = {}
            for index, tile in result["tiles"].items():
                candidates = tile.get("candidates", [])
                scores = {c: s for c, _o, s in candidates}
                model_order = [c for c, o, _s in candidates if o != "prompt"]
                if kind == "v":
                    keep = [p for p in tile["items"] if (scores.get(p[0]) or 0.0) >= threshold]
                elif kind == "p":
                    origin = {c: o for c, o, _s in candidates}
                    keep = [p for p in tile["items"]
                            if origin.get(p[0]) != "prompt" or (scores.get(p[0]) or 0.0) >= threshold]
                else:
                    first = set(model_order[:int(threshold)])
                    prompt_items = {c for c, o, _s in candidates if o == "prompt"}
                    keep = [p for p in tile["items"] if p[0] in first or p[0] in prompt_items]
                tiles[index] = {**tile, "items": keep}
            out.setdefault(scene_key, {})[prompt_key] = {**result, "tiles": tiles}
    return out


def cmd_report(args):
    judgments = load_judgments()
    arms = args.arms.split(",") if args.arms else None
    records = run_records(arms)
    for spec in (s for s in args.filters.split(",") if s):
        for name in list(records):
            if "@" not in name:
                records[f"{name}@{spec}"] = filtered(records[name], spec)
    header = (f"{'arm / prompt':<28}{'tiles':>5}{'items':>7}{'yes':>7}{'no':>6}{'vg':>6}{'prec':>7}{'wrong':>7}{'vague':>7}"
              f"{'pos ok':>7}{'pos n':>6}{'s/tile':>8}{'pic s':>8}{'grid s':>9}{'unjdg':>6}")
    print("items, yes, no and vg are per tile. prec = yes / (yes + no + vague). pos ok = right terms of "
          "judged terms on yes items.\ngrid s = picture + the scene's full tile count x mean tile time, "
          "averaged over scenes.")
    print(header)
    registry, prompt_keys = SETS[args.set]
    for name, record in records.items():
        for prompt_key in (*prompt_keys, "all"):
            results = [(scene_key, by_prompt[p]) for scene_key, by_prompt in record.items()
                       if scene_key in registry for p in by_prompt if prompt_key in ("all", p)]
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


# ---------------------------------------------------------------- export

def export_png_name(scene_key, index):
    return f"{scene_key}-t{index:02d}.png"


def png_bytes(image):
    """One IMAGE row as a lossless PNG at its own size, 8 bits per channel."""
    buffer = io.BytesIO()
    to_pil(image, math.inf).save(buffer, format="PNG")
    return buffer.getvalue()


def export_manifest(sets, digests):
    """One entry per tile and prompt of every set, so a reader keys a run by `id` and
    `prompt_name`. `digests` maps each exported PNG's name to its sha256."""
    entries = []
    for set_name, (registry, prompt_keys) in sets.items():
        for scene in registry.values():
            for index, prompt_name in itertools.product(scene.tiles, prompt_keys):
                file = export_png_name(scene.key, index)
                entries.append({"id": f"tiles/{scene.key}-t{index:02d}", "file": file,
                                "prompt": scene.prompts[prompt_name], "prompt_name": prompt_name,
                                "scene": scene.key, "tile": index, "set": set_name, "sha256": digests[file]})
    return entries


def cmd_export(_args):
    """Every tile crop of both sets once, and manifest.json beside them for the Logit Tagger's
    cross-pack runner."""
    digests = {}

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    for registry, _prompt_keys in SETS.values():
        for scene in registry.values():
            canvas = load_canvas(scene)
            layout = solve_layout(canvas)
            for index in scene.tiles:
                name = export_png_name(scene.key, index)
                r = layout.tiles[index].crop_rect
                # The face3k canvas is a float tensor, so its 8-bit PNGs round its pixels and a crop
                # read back from them does not reproduce the recorded face3k runs bit for bit.
                data = png_bytes(canvas[:, r.y0:r.y1, r.x0:r.x1, :])
                (EXPORT_DIR / name).write_bytes(data)
                digests[name] = hashlib.sha256(data).hexdigest()
    manifest = export_manifest(SETS, digests)
    (EXPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(digests)} tiles and {len(manifest)} manifest entries in {EXPORT_DIR}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("layouts")
    run = sub.add_parser("run")
    run.add_argument("--arm", required=True, choices=sorted(ARMS))
    run.add_argument("--set", choices=sorted(SETS), default="main")
    run.add_argument("--scenes", default="")
    run.add_argument("--prompts", default="")
    run.add_argument("--fresh", action="store_true", help="drop the arm's earlier record")
    run.add_argument("--clip", default=CLIP_NAME, help="the text encoder file in models/text_encoders")
    sub.add_parser("tasks")
    sub.add_parser("merge")
    compare = sub.add_parser("compare")
    compare.add_argument("--a", required=True)
    compare.add_argument("--b", required=True)
    report = sub.add_parser("report")
    report.add_argument("--arms", default="")
    report.add_argument("--set", choices=sorted(SETS), default="main")
    report.add_argument("--filters", default="", help="csv of offline filters: vT, pT, nN (see filtered_once)")
    sub.add_parser("export")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command in ("layouts", "export"):
        ab_env.bootstrap()
    {"layouts": cmd_layouts, "run": cmd_run, "tasks": cmd_tasks, "merge": cmd_merge,
     "report": cmd_report, "compare": cmd_compare, "export": cmd_export}[args.command](args)


if __name__ == "__main__":
    main()
