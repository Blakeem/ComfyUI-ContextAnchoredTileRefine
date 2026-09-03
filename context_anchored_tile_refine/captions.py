"""Per-tile VLM captions: the VL nodes' two caption conditioning surfaces.

The `vlm_method` select routes every tile's positive through one of three surfaces, and
this module owns the two that involve the VL model's text generator:

    vision tokens               vl.build_global_slices — each tile's rows out of an encode of
                                its own crop and its row slice of one encode of the entire
                                image. Positionally exact, demand-free, and it invents
                                nothing. This module only hands it the [vision] table.
    captions                    build_caption_conds — the VL model writes a description of
                                each tile's own crop and that text IS the tile's whole
                                positive. Creative: it can repair a messy background or a
                                hallucination in the source by steering the tile toward
                                something coherent, at the cost of inventing detail the
                                source lacks.
    vision tokens and captions  build_slice_caption_conds — both halves, concatenated: the
                                tile's vision rows exactly as `vision tokens` builds them,
                                followed by that tile's caption encoded TEXT-ONLY.

Cost: both caption surfaces pay one clip.generate per tile per picture, then one cheap TEXT
encode per caption. `vision tokens and captions` adds the same vision encodes `vision tokens`
pays, one of the entire image per picture and one small one per tile. When the run's preset
carries a global_style_instruction, both caption surfaces also pay ONE whole-image style
clip.generate per picture, prepended to every tile caption before it is encoded.

The instructions live in the settings file in the node's folder (load_settings below), so
the owner and node users can edit them without touching code. Each preset there is one
vlm_method option per caption surface (`resolve_method`), and the file's [vision] table
holds the row counts and picture sizes every surface samples at.

Everything here is lifted from tests-AB/run_ab_matrix.py, which produced the renders the
owner judged on 2026-08-13; nothing is newly invented. Module scope is torch-only; comfy
is imported lazily inside functions (the same contract as vl.py / sampling.py, pinned by
a subprocess test).
"""
import functools
import math
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from . import vl

# The three conditioning SURFACES — what a vlm_method option builds, with any preset label
# stripped. "vision tokens" is served by vl.build_global_slices and reads only the settings
# file's [vision] table, never a preset. This module's slice+caption surface takes its vision
# half from the same vl.build_vision_rows the vision-only surface uses, so the two can never
# drift apart. The two caption surfaces each gain one labeled option per preset
# (`vlm_methods` below).
VLM_METHOD_VISION = "vision tokens"
VLM_METHOD_VISION_CAPTIONS = "vision tokens and captions"
VLM_METHOD_CAPTIONS = "captions"
CAPTION_SURFACES = (VLM_METHOD_VISION_CAPTIONS, VLM_METHOD_CAPTIONS)
VLM_SURFACES = (VLM_METHOD_VISION, *CAPTION_SURFACES)

# --- the SETTLED instruction pair (2026-08-13), from the seven-round search in
# tests-AB/vlm_prompt_lab.py and the owner's own ComfyUI trials of the finalists. One
# instruction per surface. tests-AB/run_ab_matrix.py carries a SUPERSEDED pre-settlement
# pair under confusingly similar names (POSITION_INSTRUCTION / RICH_INSTRUCTION); the
# SETTLED_ names are kept here so the two can never be confused again.
#
# The WHOLE pair is retired from the live surfaces since 2026-08-21: what the surfaces ask
# now comes from the settings file (load_settings below). Every constant stays defined,
# character-frozen, because tests-AB's judged arms pin themselves to these strings and
# their renders are on disk.
#
#   SETTLED_POSITION  RETIRED FROM THE SURFACE 2026-08-16 (owner decision, the text-cat
#                     campaign): it rode WITH the VL slices while the caption was encoded
#                     inside the canvas stream.
#   SETTLED_RICH      what every caption-carrying surface asked until 2026-08-21. Style,
#                     palette and lighting lead, because on the captions-ONLY surface
#                     nothing else carries appearance, and on the slice+caption surface
#                     the vision rows carry position already.
#
# Findings baked into the wording, none of which may be "tidied" out:
#   - The position prompt carries TWO independent bounds and either one alone terminates:
#     the `up to eight` ceiling and GROUP_CLAUSE. With NEITHER, 7 of 9 round-1 cells ran to
#     the token cap — one object mined for its parts (hair/eyes/lips/neck...) or one
#     repeated per instance (stall x30). What the clause does ON TOP of the ceiling is
#     suppress quantity words: removing it took numeric mentions from 1 to 5 across 9 tiles.
#   - `short phrase` holds items to 6-11 words; asking for a `sentence` licenses 32-40.
#   - the five REGIONS in the rich prompt are a stop condition set by the QUESTION, not by
#     how busy the picture is, which is why that one alone never runs away.
# Spelling is deliberate: the owner A/B'd US against EU spelling in ComfyUI and EU won —
# "centre" recovered details ("the fox") that "center" dropped. Do not Americanise.
GROUP_CLAUSE = "Name whole objects and count repeated objects as one entry."

SETTLED_POSITION_INSTRUCTION = (
    "List up to eight main things in this image, one per line, each a short phrase naming "
    f"the thing, its position in the frame, and how much of it shows. {GROUP_CLAUSE}")
SETTLED_POSITION_MAX_TOKENS = 512

SETTLED_RICH_INSTRUCTION = (
    "Describe this image, one short line per part. Start with the overall style, palette "
    "and lighting. Then say what fills the left, the centre, the right, the top and the "
    "bottom, giving each part its own description with what is there, its colour and what "
    "its surface is made of.")
SETTLED_RICH_MAX_TOKENS = 768

# The rich prompt WITH the grouping clause, what every caption surface shipped until the
# prompts moved into the settings file, and what the (standard) preset there now carries.
# The owner's explicit decision, taken against the contrary lab measurement. On the record
# both ways: the owner judged 1-face/17_CaptionOnly+Group_Lead_s42_v3
# "better across the board" against 14_CaptionOnly_Lead_s42_v3 (ungrouped, which drew a
# phantom second moon), and ruled that region repeats are acceptable on THIS surface —
# "Repeating may be fine, it often does that only for predominate stuff and that just
# increases weight when it's caption only. This was only an issue with VL method combined."
# Against that: round 7 of the lab scored the grouped wording 4/9 vs 7/9 on lab tiles
# (uniform crops repeat), and the grouped wording was rendered on the `face` scene ONLY.
RICH_GROUPED_INSTRUCTION = f"{SETTLED_RICH_INSTRUCTION} {GROUP_CLAUSE}"

# --- the LIVE prompts: the settings file in the node's folder ----------------------------
# What each caption surface asks the VLM lives in a TOML file so that it can be edited
# without touching code. `settings.user.toml` is the user's own copy and wins whenever it
# exists; `settings.toml` ships with the node and is what an update replaces.
# TWO READ CADENCES, deliberately. The PRESET LIST is read once per ComfyUI session
# (`vlm_methods`), because it becomes a combo the frontend caches at startup. A preset's own
# wording and numbers are re-read on every run (`resolve_method`), so tuning a prompt needs
# no restart. Both caption surfaces ask the SAME tile question of a given preset, as they
# have since 2026-08-16.
SETTINGS_DIR = Path(__file__).resolve().parent.parent
SETTINGS_NAME = "settings.toml"
USER_SETTINGS_NAME = "settings.user.toml"

# Every max_tokens budget must cover the reasoning turn as well (captions are always
# generated with thinking=True), which is why the shipped values are 768 rather than a
# visible-answer length. The ceiling is a typo guard: one caption is the run's slowest
# per-tile step, so a stray extra digit would multiply the whole run's wall time.
MAX_CAPTION_TOKENS = 4096

# Caption input budget (total pixels, aspect preserved) — AB27's prep, what resample_for_vl
# falls back to, and the size every judged tests-AB arm was captioned at (ab_env.caption_preset
# pins it). Conditioning-side only: what the VLM reads is a COPY of the tile's crop, never the
# sampled tile itself (prime directive 1: a sampled tile is never resized, resampled or
# otherwise degraded).
VL_INPUT_BUDGET = 384 * 384
VL_INPUT_BUDGET_MEGAPIXELS = VL_INPUT_BUDGET / 1_000_000

# The caption input size the shipped file reads at, 768x1024 px. The owner's three-scene A/B
# (TESTS.md test 3) found the VLM's sample size decides whether a caption invents or drops
# content, and this is the size where it does neither.
SHIPPED_CAPTION_MEGAPIXELS = 768 * 1024 / 1_000_000

# Floor for a non-zero caption budget. Below roughly 5e-7 MP the budget rounds to no pixels
# at all and the resample builds a 0 x 0 image, which reaches torch as an opaque error
# instead of a named one. 0.01 MP is 100 x 100 px, already past anything a caption can read.
VL_INPUT_MIN_MEGAPIXELS = 0.01

_VISION_KEYS = {
    "canvas_tokens": int,
    "crop_tokens": int,
    "caption_megapixels": float,
}

_PRESET_KEYS = {
    "tile_caption_instruction": str,
    "tile_caption_max_tokens": int,
    "global_style_instruction": str,
    "global_style_max_tokens": int,
}

# Per-preset keys this version no longer reads: the caption picture size moved to the
# [vision] table on 2026-09-02, one size for both caption surfaces. Named so a user's own
# copy from before then fails with the fix, not with "unknown key".
_REMOVED_PRESET_KEYS = ("tile_caption_megapixels", "global_style_megapixels")


@dataclass(frozen=True)
class VisionSettings:
    """The settings file's [vision] table: how every tile's conditioning samples the image.
    `canvas_tokens` and `crop_tokens` are the vision rows a tile takes from the entire
    image's encode and from its own crop's (vl.build_vision_rows; 0 turns a source off), and
    `caption_megapixels` is the picture both caption surfaces write from."""

    canvas_tokens: int
    crop_tokens: int
    caption_megapixels: float


@dataclass(frozen=True)
class Settings:
    """The validated settings file: its [vision] table and its presets in file order."""

    vision: VisionSettings
    presets: dict


@dataclass(frozen=True)
class Preset:
    """One vlm_method option, resolved: the conditioning surface it builds, the [vision]
    table every surface samples by, and, on the two caption surfaces, everything its settings
    block asks for. `label` is "" for the vision-only surface, which reads no preset, and
    `style_instruction` is "" when this preset asks for no whole-image style caption."""

    surface: str
    label: str
    vision: VisionSettings
    tile_instruction: str = ""
    tile_max_tokens: int = 0
    style_instruction: str = ""
    style_max_tokens: int = 0


def settings_path():
    """The settings file in force: the user's own copy when it exists, else the shipped one.
    Nothing in the package ever writes settings.user.toml, which is what makes it the edit
    surface that survives a node update."""
    user_path = SETTINGS_DIR / USER_SETTINGS_NAME
    return user_path if user_path.is_file() else SETTINGS_DIR / SETTINGS_NAME


def _read_toml(path):
    # Every way a hand-edited file can fail to parse, each named so the console line says
    # which file and what is wrong with it.
    try:
        with open(path, "rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): {path.name} is missing at {path}. The file "
            "ships with the node. Restore it from the repository.") from error
    except tomllib.TOMLDecodeError as error:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): {path} is not valid TOML ({error}).") from error
    except UnicodeDecodeError as error:
        # An editor saving in a legacy codepage (a curly quote in cp1252) raises this
        # instead of TOMLDecodeError.
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): {path} is not UTF-8 ({error}). Save the file "
            "as UTF-8.") from error
    except OSError as error:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): {path} could not be read ({error}).") from error


def _check_preset(path, label, block):
    # One [presets.<label>] block, checked key by key. A defect here is a hard error rather
    # than a fallback: a preset that silently loses a key would caption every tile with a
    # question its author never wrote.
    if not isinstance(block, dict):
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): preset {label!r} in {path} must be a "
            f"[presets.{label}] table, got {type(block).__name__}.")
    missing = sorted(set(_PRESET_KEYS) - set(block))
    if missing:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): preset {label!r} in {path} is missing {missing}.")
    removed = sorted(set(block) & set(_REMOVED_PRESET_KEYS))
    if removed:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): preset {label!r} in {path} carries {removed}, "
            "which this version no longer reads. The caption picture size is now "
            "caption_megapixels in the [vision] table. Copy settings.toml to settings.user.toml "
            "again and move your own values over.")
    unknown = sorted(set(block) - set(_PRESET_KEYS))
    if unknown:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): preset {label!r} in {path} carries unknown keys "
            f"{unknown}. A misspelled key would otherwise change nothing, silently.")
    for key, expected in _PRESET_KEYS.items():
        # A TOML int is a legal float value, so the float keys accept both; bool is an int
        # subclass and is never either.
        allowed = (int, float) if expected is float else expected
        if not isinstance(block[key], allowed) or isinstance(block[key], bool):
            raise RuntimeError(
                f"Context-Anchored Tile Refine (VL): preset {label!r} key {key} in {path} must be "
                f"of type {expected.__name__}, got {type(block[key]).__name__}.")
    if not block["tile_caption_instruction"].strip():
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): preset {label!r} in {path} has an empty "
            "tile_caption_instruction. The caption vlm_methods need a question to ask about "
            "each tile.")
    for key in ("tile_caption_max_tokens", "global_style_max_tokens"):
        if not 1 <= block[key] <= MAX_CAPTION_TOKENS:
            raise RuntimeError(
                f"Context-Anchored Tile Refine (VL): preset {label!r} key {key} in {path} must be "
                f"between 1 and {MAX_CAPTION_TOKENS}, got {block[key]}.")


def _check_vision(path, block):
    # The [vision] table, checked key by key like a preset. A table that silently lost a key
    # would sample every tile at a size nobody chose.
    if not isinstance(block, dict):
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): [vision] in {path} must be a table, got "
            f"{type(block).__name__}.")
    missing = sorted(set(_VISION_KEYS) - set(block))
    if missing:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): the [vision] table in {path} is missing {missing}.")
    unknown = sorted(set(block) - set(_VISION_KEYS))
    if unknown:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): the [vision] table in {path} carries unknown keys "
            f"{unknown}. A misspelled key would otherwise change nothing, silently.")
    for key, expected in _VISION_KEYS.items():
        allowed = (int, float) if expected is float else expected
        if not isinstance(block[key], allowed) or isinstance(block[key], bool):
            raise RuntimeError(
                f"Context-Anchored Tile Refine (VL): [vision] key {key} in {path} must be of type "
                f"{expected.__name__}, got {type(block[key]).__name__}.")
    for key in ("canvas_tokens", "crop_tokens"):
        if not 0 <= block[key] <= vl.MAX_VISION_TOKENS:
            raise RuntimeError(
                f"Context-Anchored Tile Refine (VL): [vision] key {key} in {path} must be between 0 "
                f"and {vl.MAX_VISION_TOKENS} (0 turns that source off), got {block[key]}.")
    if block["canvas_tokens"] == 0 and block["crop_tokens"] == 0:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): [vision] in {path} sets canvas_tokens and "
            "crop_tokens both to 0. A tile needs vision rows from at least one of the two.")
    value = block["caption_megapixels"]
    if value != 0 and not VL_INPUT_MIN_MEGAPIXELS <= value <= vl.PICTURE_CAP_MEGAPIXELS:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): [vision] key caption_megapixels in {path} must be "
            f"0, which reads the picture's own size, or between {VL_INPUT_MIN_MEGAPIXELS} and "
            f"{vl.PICTURE_CAP_MEGAPIXELS}. Got {value}.")


def load_settings(path=None):
    """The validated settings file: its [vision] table and its presets in file order.

    Every defect is a hard error here, which is reached twice: once at startup when the
    vlm_method selector is built, and once per run before any encode spends GPU time.
    `path` exists for tests.
    """
    settings_path_ = settings_path() if path is None else path
    data = _read_toml(settings_path_)

    unknown = sorted(set(data) - {"vision", "presets"})
    if unknown:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): {settings_path_} carries unknown top-level keys "
            f"{unknown}. The file holds one [vision] table and [presets.<label>] blocks.")
    if "vision" not in data:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): {settings_path_} has no [vision] table. This "
            "version needs one. Copy settings.toml to settings.user.toml again and move your "
            "own values over.")
    _check_vision(settings_path_, data["vision"])
    vision = VisionSettings(
        canvas_tokens=int(data["vision"]["canvas_tokens"]),
        crop_tokens=int(data["vision"]["crop_tokens"]),
        caption_megapixels=float(data["vision"]["caption_megapixels"]))
    presets = data.get("presets")
    if not isinstance(presets, dict) or not presets:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): {settings_path_} defines no presets. It needs at "
            "least one [presets.<label>] block, whose label names the vlm_method options it adds.")
    for label, block in presets.items():
        if not label.strip() or "(" in label or ")" in label:
            raise RuntimeError(
                f"Context-Anchored Tile Refine (VL): preset label {label!r} in {settings_path_} is "
                "not usable. A label carries the vlm_method option's own parentheses, so it must "
                "be non-blank and hold neither '(' nor ')'.")
        _check_preset(settings_path_, label, block)
    return Settings(vision=vision, presets=presets)


def build_vlm_methods(presets):
    """The vlm_method selector's options: the vision-only surface, then both caption surfaces
    of every preset. Grouped by preset and in file order, so a preset's two options sit
    together and the list is ordered by whoever wrote the settings file.

    The FIRST preset is the DEFAULT and its two options carry NO label, so they read
    "vision tokens and captions" and "captions". Those are the exact strings a workflow saved
    before the presets existed holds, so such a workflow keeps its own value and its own
    result with nothing to re-pick. Labelling them instead would have made every one of them
    a value the selector no longer offers.
    """
    options = [VLM_METHOD_VISION]
    for index, label in enumerate(presets):
        suffix = "" if index == 0 else f" ({label})"
        options.extend(f"{surface}{suffix}" for surface in CAPTION_SURFACES)
    return options


@functools.lru_cache(maxsize=1)
def vlm_methods():
    """The selector's options, built ONCE per ComfyUI session.

    The frontend caches a node's definition at startup, so a list that changed between calls
    would offer values the backend then rejects, or hide values a saved workflow carries.
    A new or renamed preset therefore needs a restart, while a preset's own wording does not.
    """
    return tuple(build_vlm_methods(load_settings().presets))


def default_vlm_method():
    # The default preset's slice+caption option, which carries no label (build_vlm_methods),
    # so this is the plain "vision tokens and captions" the widget defaulted to before the
    # presets existed. The vision-only surface leads the list and is not it: the two halves
    # together are what the campaign settled on.
    return vlm_methods()[1]


def method_surface(vlm_method):
    """Which conditioning surface a vlm_method option builds, with any preset label stripped.

    Pure string work, so the branches that only need the surface (the progress plan, the
    engine's dispatch) never read the settings file — which is what keeps a broken file from
    failing a "vision tokens" run that asks it nothing.
    """
    for surface in VLM_SURFACES:
        if vlm_method == surface:
            return surface
        if vlm_method.startswith(f"{surface} (") and vlm_method.endswith(")"):
            return surface
    raise ValueError(
        f"vlm_method {vlm_method!r} names no conditioning surface. Expected one of "
        f"{list(VLM_SURFACES)}, each optionally followed by a preset label in parentheses.")


def method_label(vlm_method):
    """The preset label a vlm_method option carries. "" for the vision-only surface, and for
    the two unlabeled caption options, which are the DEFAULT preset's (`build_vlm_methods`)
    and are what a workflow saved before the presets existed holds."""
    surface = method_surface(vlm_method)
    if vlm_method == surface:
        return ""
    return vlm_method[len(surface) + 2:-1]


def resolve_method(vlm_method):
    """One vlm_method option resolved to the `Preset` the engine runs on.

    The settings file is read HERE, once per run, so an edit applies with no ComfyUI restart.
    "vision tokens" takes only the [vision] table. A caption option takes its preset as well;
    an unlabeled one takes the first preset, which is the default one the selector offers
    unlabeled. Its label still resolves when a workflow spells it out, so the two forms of the
    default preset are one block and a workflow saved under either keeps running.
    """
    surface = method_surface(vlm_method)
    settings = load_settings()
    if surface == VLM_METHOD_VISION:
        return Preset(surface=surface, label="", vision=settings.vision)
    presets = settings.presets
    label = method_label(vlm_method) or next(iter(presets))
    if label not in presets:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): vlm_method {vlm_method!r} asks for preset "
            f"{label!r}, which {settings_path()} does not define. It offers {sorted(presets)}. "
            "Restart ComfyUI after adding or renaming a preset.")
    block = presets[label]
    style = block["global_style_instruction"]
    return Preset(
        surface=surface,
        label=label,
        vision=settings.vision,
        tile_instruction=block["tile_caption_instruction"],
        tile_max_tokens=block["tile_caption_max_tokens"],
        # Whitespace-only is "off" too, so a user clearing the line by hand cannot leave a
        # blank style caption riding on top of every tile.
        style_instruction=style if style.strip() else "",
        style_max_tokens=block["global_style_max_tokens"],
    )


def caption_budget_pixels(megapixels, source):
    # The caption_megapixels semantics, in one place: 0 (or less) is the source's own area, so
    # the VL model reads every pixel the crop has, capped at vl.PICTURE_CAP_PIXELS. Above 0
    # the value is the budget itself, already range-checked by _check_vision.
    if megapixels <= 0:
        return min(int(source.shape[1]) * int(source.shape[2]), vl.PICTURE_CAP_PIXELS)
    return round(megapixels * 1_000_000)


# The alternation is core's own (comfy_extras/nodes_textgen.py:261) and is load-bearing:
# without the `|$` an unclosed open makes the sub a no-op, and a tile whose reasoning turn
# exhausts max_tokens returns that reasoning AS its caption. Non-empty, so generate_caption's
# fallback chain never fires and the model's own deliberation reaches the DiT as the tile's
# whole positive.
_THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL)

_META_LINE = ("wait,", "wait ", "here's a revised", "here is a revised", "revised version",
              "let me", "i need to", "actually,")


def resample_for_vl(tile_pixels, budget=None):
    # AB27's caption input prep: area-resample a COPY of the tile's crop to `budget` total
    # pixels, VL_INPUT_BUDGET by default. Unlike vl.resample_picture there is no
    # /MERGED_CELL snap, because nothing slices this encode by row — the tokenizer's own
    # rounding is free to apply.
    import comfy.utils

    samples = tile_pixels.movedim(-1, 1)
    pixels = VL_INPUT_BUDGET if budget is None else budget
    scale_by = math.sqrt(pixels / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    resampled = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    return resampled.movedim(1, -1)[:, :, :, :3]


def strip_thinking(text):
    """Cut Qwen3's reasoning turn off the front of an answer.

    Mirrors comfy_extras/nodes_textgen.py TextGenerateLTX2Prompt, which is the only place
    core does this. The plain TextGenerate node returns the reasoning to the user. Here it is
    mandatory, because the caption is encoded as text, so an unstripped <think> block would
    reach the DiT as several hundred tokens of the model talking to itself.

    An answer that is nothing BUT an unclosed reasoning turn comes back "", which is what
    makes generate_caption's fallback chain fire on it."""
    if "<think>" not in text:
        return text.strip()
    body = _THINK_BLOCK.sub("", text)
    if "</think>" in body:                  # truncated/unclosed: keep what follows the last
        body = body.rsplit("</think>", 1)[-1]
    return re.sub(r"</?think>", "", body).strip()


def clean_caption(text):
    """Remove the VLM's own formatting artifacts so the DiT never reads them as content.

    Three observed failures, all of which reach the conditioning verbatim because the
    caption is encoded as text:
      heading      "**Answer:**", "**Image as compact list:**" — a label, not a description
      meta         "Wait, I need to rephrase to fit under 50 words." — the model narrating
      repetition   a market crop looped six items THREE times inside one 84-word caption,
                   which is exactly the duplication the instruction exists to avoid
    Dedup is by normalized line, so 'chicken, left' and 'chicken, right' both survive; only
    an exact repeat is dropped. Content is never rewritten, only whole artifact lines cut."""
    kept, seen, dropped = [], set(), False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower().lstrip("*_#-• ")
        if low.startswith(_META_LINE) or (
                not kept and line.startswith("**") and line.endswith(":**")):
            dropped = True                              # heading or meta narration
            continue
        key = " ".join(low.rstrip(".").split())
        if key in seen:
            dropped = True                              # the repetition loop
            continue
        seen.add(key)
        kept.append(line)
    # Nothing to remove => hand back the ORIGINAL string, byte for byte. Rejoining would
    # otherwise drop the markdown line-break spaces and re-tokenize a caption that was
    # already fine.
    return text if not dropped else "\n".join(kept).strip()


def _tokenize_images(clip, text, image, **kwargs):
    # vl._encode_one's two tokenizer guards, worded for this surface, plus the tail length
    # the slice+caption layout is derived from: the rows AFTER vision_end, i.e. the caption
    # text and the template tail. Returns (tokens, tail_len). Nothing is encoded here.
    try:
        tokens = clip.tokenize(text, images=[image], **kwargs)
    except TypeError as error:
        raise RuntimeError(
            "Context-Anchored Tile Refine (VL): this CLIP's tokenizer does not accept images. "
            "The caption vlm_methods need a vision-language text encoder (Krea 2 family). "
            f"({error})") from error
    ids = [t[0] for t in tokens[next(iter(tokens))][0]]
    pad_pos = next((i for i, v in enumerate(ids) if isinstance(v, dict)), None)
    if pad_pos is None:
        raise RuntimeError(
            "Context-Anchored Tile Refine (VL): the tokenizer produced no image tokens. The "
            "caption vlm_methods need a vision-language text encoder (Krea 2 family).")
    return tokens, len(ids) - (pad_pos + 2)


def generate_caption(clip, vl_input, instruction, max_length, thinking=True):
    """One greedy caption of `vl_input`, then the settled fallback chain for a crop whose
    stop token fires immediately. `max_length` is per-instruction (the preset's max_tokens on
    the live surfaces) and has to cover the reasoning turn as well as the answer, which is
    why the budgets sit far above the visible answer length."""
    if not hasattr(clip, "generate") or not hasattr(clip, "decode"):
        raise RuntimeError(
            "Context-Anchored Tile Refine (VL): this CLIP cannot generate text. The caption "
            "vlm_methods need a vision-language text encoder with a text-generation head "
            "(Krea 2 family). Use vlm_method 'vision tokens' with any other CLIP.")

    tokens, _tail = _tokenize_images(clip, instruction, vl_input, thinking=thinking)
    ids = clip.generate(tokens, do_sample=False, max_length=max_length, repetition_penalty=1.05)
    text = strip_thinking(clip.decode(ids))
    if not text:
        ids = clip.generate(tokens, do_sample=True, max_length=max_length, temperature=0.7,
                            top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, seed=42)
        text = strip_thinking(clip.decode(ids))
    if not text:
        retokens, _tail = _tokenize_images(clip, "Write one short sentence describing this image.",
                                           vl_input, thinking=False)
        ids = clip.generate(retokens, do_sample=False, max_length=max_length, repetition_penalty=1.05)
        text = strip_thinking(clip.decode(ids))
    if not text:
        raise RuntimeError(
            "Context-Anchored Tile Refine (VL): caption generation returned an empty answer "
            "after every fallback.")
    return text


def generate_tile_captions(clip, source, tiles, preset, batch_size=1, batch_index=0,
                           progress=None, style_source=None):
    """One caption per tile per batch row, read off the FROZEN raw canvas.

    Returns captions[tile_index][batch_row]. Batch rows are captioned INDEPENDENTLY: core's
    tokenizer attaches images[0] alone (comfy/text_encoders/qwen_vl.py process_qwen2vl_images),
    so a whole [B,H,W,3] crop would describe every row with row 0's picture. Through the node
    `source` always holds exactly ONE picture — sampling.refine_image's picture loop is outside
    this pre-pass — so the row axis is length 1 there and batch_size/batch_index carry the
    picture's place in the run, which is all the ProgressBar below needs to span it.

    `preset` is the run's resolved settings block (`resolve_method`), which carries the tile
    question, both generation budgets and the [vision] table's caption picture size, read for
    the tile caption and the style caption alike. A non-empty
    `preset.style_instruction` adds ONE whole-image style caption per batch row, generated
    FIRST from `style_source` (default `source`, and the region path passes the full image so
    that a masked refine's style stays global) and prepended to every tile caption of that
    row. This way all tiles follow one style description. It is counted as the segment's
    first caption(s). An empty one leaves this function byte-identical to the style-free path.

    The pre-pass this drives is no longer "one encode" — it is one clip.generate per tile per
    row at up to the preset's max_tokens, which on a 16-tile grid runs for minutes
    before the first tile samples. Hence the per-tile interrupt check and the ProgressBar:
    without them the run is uncancellable and the UI shows nothing until the tile loop starts.

    `progress` is the VL run's ledger (progress.py) when a node created one. With it the
    standalone bar is NOT constructed — a second bar is exactly the display reset the ledger
    exists to remove — and each finished caption is reported to the ledger's open caption
    segment with the SAME (done, total) counters the bar carries, so its per-caption chunk
    snaps to its boundary while core's per-token bar (routed by the ledger's shim) fills it
    in between. Without a ledger nothing here changes.
    """
    import comfy.model_management
    import comfy.utils

    batch = int(source.shape[0])
    style_on = bool(preset.style_instruction)
    per_picture = (len(tiles) + (1 if style_on else 0)) * batch
    total = per_picture * batch_size
    pbar = None if progress is not None else comfy.utils.ProgressBar(total)
    done = per_picture * batch_index
    style_texts = []
    captions = []

    if style_on:
        style_canvas = source if style_source is None else style_source
        if int(style_canvas.shape[0]) != batch:
            raise RuntimeError(
                f"Context-Anchored Tile Refine (VL): {batch} batch row(s) to caption but the "
                f"style canvas has {int(style_canvas.shape[0])}. Every row needs its own style "
                "caption or a row would carry another row's style.")
        comfy.model_management.throw_exception_if_processing_interrupted()
        for b in range(batch):
            row = style_canvas[b:b + 1]
            vl_input = resample_for_vl(row, caption_budget_pixels(preset.vision.caption_megapixels, row))
            text = generate_caption(clip, vl_input, preset.style_instruction,
                                    preset.style_max_tokens, thinking=True)
            style_texts.append(clean_caption(text))
            done += 1
            if pbar is None:
                progress.caption_done(done, total)
            else:
                pbar.update_absolute(done, total)

    for tile in tiles:
        comfy.model_management.throw_exception_if_processing_interrupted()
        crop = tile.crop_rect
        row_captions = []
        for b in range(batch):
            row = source[b:b + 1, crop.y0:crop.y1, crop.x0:crop.x1, :]
            vl_input = resample_for_vl(row, caption_budget_pixels(preset.vision.caption_megapixels, row))
            text = generate_caption(clip, vl_input, preset.tile_instruction,
                                    preset.tile_max_tokens, thinking=True)
            caption = clean_caption(text)
            if style_on:
                caption = f"{style_texts[b]}\n{caption}"
            row_captions.append(caption)
            done += 1
            if pbar is None:
                progress.caption_done(done, total)
            else:
                pbar.update_absolute(done, total)
        captions.append(row_captions)
    return captions


def build_caption_conds(clip, captions):
    """Captions WITHOUT slices: each caption re-encoded text-only, exactly what
    CLIPTextEncode would produce for that string. `captions` keeps the [tile][batch row]
    shape generate_tile_captions returns, with exactly ONE row: sampling.refine_image
    refines one picture at a time, so there is never a second row to concatenate."""
    tile_positives = []
    for tile_captions in captions:
        encoded = clip.encode_from_tokens_scheduled(clip.tokenize(tile_captions[0]))
        tile_positives.append(vl._convert(encoded))
    return tile_positives


def _caption_tail_len(clip, caption, probe_image):
    # How many rows this caption occupies after vision_end (caption text + template tail),
    # read off the TOKEN stream rather than off any encoder output — which is what makes it an
    # independent expectation for _encode_caption_text_only to be checked against. Nothing is
    # encoded here; the image only makes the stream well-formed for the tokenizer.
    _tokens, tail_len = _tokenize_images(clip, vl.VISION_BLOCK + caption, probe_image,
                                         llama_template=vl.KREA2_TEMPLATE)
    return tail_len


def _encode_caption_text_only(clip, caption, expected_rows):
    # The caption encoded exactly as CLIPTextEncode would encode it. Krea 2's template strip
    # removes a PREFIX only, so what survives is [caption rows][template tail] and nothing
    # else (measured, docs/vl-conditioning-encode-cost.md section 10). That is asserted
    # against the in-stream tail length so a template change fails fast instead of silently
    # concatenating a differently-shaped stream onto every tile's vision rows.
    encoded = clip.encode_from_tokens_scheduled(clip.tokenize(caption))
    seq = int(encoded[0][0].shape[1])
    if seq != expected_rows:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): the text-only caption encode has {seq} rows, "
            f"expected {expected_rows} (caption + template tail). The text encoder's template "
            "or strip layout does not match the Krea 2 contract this surface concatenates by.")
    return encoded


def build_slice_caption_conds(clip, encode_source, tiles, captions, vision, offset_x=0, offset_y=0):
    """VL rows AND captions: each tile's positive is its vision rows exactly as the `vision
    tokens` surface builds them (vl.build_vision_rows: its own crop's cells, then its slice
    of the entire image), followed by that tile's own caption encoded TEXT-ONLY, concatenated
    on the row axis.

    Settled 2026-08-16 by the owner's A/B (tests-AB/run_ab_split.py arm 2 against the previous
    arm 1, then the sync-tiles campaign). Until then the caption rode INSIDE a whole-canvas
    vision encode, which cost one whole-canvas encode PER TILE and let far-canvas content leak
    into every tile's caption rows through attention — the phantom-moon failure. Encoding the
    caption alone removes both. The vision half is unaffected by the change: attention is
    causal and the caption sat after the grid rows, so those rows were never reading it
    (docs/vl-conditioning-encode-cost.md sections 6-7, measured bit-identical at matched
    stream length).

    `encode_source` is the image the OFFSET tile rects index, taken separately from the canvas
    the captions describe — mirroring vl.build_vision_rows: on the whole-image path both are
    the padded canvas at offset 0, while on the mask path the captions describe each region
    tile's own crop and the canvas rows come from the FULL image with the bbox origin as the
    offset, so a masked refine stays globally informed."""
    batch = int(encode_source.shape[0])
    tile_positives = []

    if any(len(tile_captions) != batch for tile_captions in captions):
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): {batch} image(s) in the batch but a tile was "
            "captioned a different number of times. Every batch row must carry its own caption "
            "or a row would be conditioned on another row's picture.")

    # The vision half is vl's own, so this surface and `vision tokens` can never build
    # differently shaped rows. Core's tokenizer attaches images[0] alone, so the source is
    # narrowed to one picture here; refine_image's picture loop is what makes that the whole
    # batch. The tail is left off: the one template tail the stream may carry arrives with
    # the caption rows concatenated after the vision rows.
    tile_rows, probe_image = vl.build_vision_rows(clip, encode_source[:1], tiles, vision,
                                                  offset_x, offset_y, with_tail=False)

    for tile_captions, rows in zip(captions, tile_rows, strict=True):
        # Exactly ONE row, so nothing is concatenated across rows: the count guard above ties
        # len(tile_captions) to the encode source's batch, and refine_image's picture loop
        # makes that batch 1.
        caption = tile_captions[0]
        # The probe is never encoded and _tokenize_images counts the rows after a single image
        # token, so any resampled copy gives the same tail length.
        tail_len = _caption_tail_len(clip, caption, probe_image)
        caption_entries = _encode_caption_text_only(clip, caption, tail_len)
        tile_positives.append(vl._convert(vl.cat_rows(rows, caption_entries, "caption encode")))
    return tile_positives
