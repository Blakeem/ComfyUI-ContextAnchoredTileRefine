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

The two caption surfaces build each tile's text by the preset's tile_text kind: the caption
kind (the VL model describes the crop, generate_caption_set here) or the tags kind (tags.py
writes verified tags, and this module holds its schema).

Cost: on the caption kind, both caption surfaces pay one clip.generate per tile per picture,
then one cheap TEXT encode per caption. The tags kind's per-tile requests are tags.py's.
`vision tokens and captions` adds the same vision encodes `vision tokens` pays, one of the
entire image per picture and one small one per tile. When the run's preset
carries a global_style_instruction, both caption surfaces also pay ONE whole-image style
clip.generate per picture, prepended to every tile caption before it is encoded.

The instructions live in the settings file in the node's folder (load_settings below), so
the owner and node users can edit them without touching code. Each preset there is one
vlm_method option per caption surface (`resolve_method`), and the file's [vision] table
holds the row counts and picture sizes every surface samples at. comfy_toolkit (the library
gate every VL run passes) and ClipBoundCache are shared with tags.py.

Module scope is torch-only. comfy and logit_classifier are imported lazily inside functions
(the same contract as vl.py / sampling.py, pinned by a subprocess test).
"""
import functools
import hashlib
import math
import re
import tomllib
import weakref
from collections import OrderedDict
from dataclasses import dataclass, replace
from pathlib import Path

from . import vl

# The three conditioning SURFACES — what a vlm_method option builds, with any preset label
# stripped. "vision tokens" is served by vl.build_global_slices and reads only the settings
# file's [vision] table, never a preset. This module's slice+caption surface takes its vision
# half from the same vl.build_vision_rows the vision-only surface uses, so the two can never
# drift apart. The first preset adds the two bare caption options and every other preset adds
# two labeled ones (`build_vlm_methods`).
VLM_METHOD_VISION = "vision tokens"
VLM_METHOD_VISION_CAPTIONS = "vision tokens and captions"
VLM_METHOD_CAPTIONS = "captions"
CAPTION_SURFACES = (VLM_METHOD_VISION_CAPTIONS, VLM_METHOD_CAPTIONS)
VLM_SURFACES = (VLM_METHOD_VISION, *CAPTION_SURFACES)

# --- the SETTLED instruction pair, from the seven-round search in tests-AB/vlm_prompt_lab.py.
# The pair is retired from the live surfaces, since the settings file asks now (load_settings
# below). Every constant stays defined and character-frozen, because tests-AB's judged arms
# pin themselves to these strings. tests-AB/run_ab_matrix.py carries a superseded pair under
# similar names (POSITION_INSTRUCTION / RICH_INSTRUCTION).
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

# The rich prompt with the grouping clause, which the (standard) preset in
# settings.user.example.toml carries and the pin test ties to it.
RICH_GROUPED_INSTRUCTION = f"{SETTLED_RICH_INSTRUCTION} {GROUP_CLAUSE}"

# --- the LIVE prompts: the settings file in the node's folder ----------------------------
# What each caption surface asks the VLM lives in a TOML file so that it can be edited
# without touching code. `settings.user.toml` is the user's own copy and wins whenever it
# exists; `settings.toml` ships with the node and is what an update replaces.
# TWO READ CADENCES, deliberately. The PRESET LIST is read once per ComfyUI session
# (`vlm_methods`), because it becomes a combo the frontend caches at startup. A preset's own
# wording and numbers are re-read on every run (`resolve_method`), so tuning a prompt needs
# no restart. Both caption surfaces ask the SAME tile question of a given preset. An
# instruction may carry PROMPT_PLACEHOLDER, which the nodes fill from their prompt input
# (`with_prompt`) before the preset reaches the caption pass.
SETTINGS_DIR = Path(__file__).resolve().parent.parent
SETTINGS_NAME = "settings.toml"
USER_SETTINGS_NAME = "settings.user.toml"
# A sample the package ships and never reads on its own. It carries the presets that no
# longer ship in settings.toml, so a saved option naming one of them has a place to point to.
EXAMPLE_SETTINGS_NAME = "settings.user.example.toml"

# Every max_tokens budget must cover the reasoning turn as well (captions are always
# generated with thinking=True), which is why the shipped values are 768 rather than a
# visible-answer length. The ceiling is a typo guard: one caption is the run's slowest
# per-tile step, so a stray extra digit would multiply the whole run's wall time.
MAX_CAPTION_TOKENS = 4096

# Where an instruction takes the node's prompt input (`with_prompt`). The prompt reaches the
# VL model's QUESTION only, never the DiT: what the DiT reads is still the caption written
# about the crop, so the A/B finding that text conditioning re-admits phantom objects is
# untouched. A literal replace rather than str.format, since a user's prompt can carry braces.
PROMPT_PLACEHOLDER = "{PROMPT}"

# Where a tags preset's tile_tags_verification_statement takes each candidate tag.
TAG_PLACEHOLDER = "{TAG}"

# A preset's `tile_text` kind: what each tile's text is written from. A block without the key
# is the caption kind, so every settings.user.toml written before the key existed loads as it
# did.
TILE_TEXT_CAPTION = "caption"
TILE_TEXT_TAGS = "tags"
TILE_TEXT_KINDS = (TILE_TEXT_CAPTION, TILE_TEXT_TAGS)

# Restates the library's STRICT_THRESHOLDS, as logit_classifier is imported only in functions. Against the former 0.999
# and 0.9999 prompt-only pair, an offline cut of fp8 tiles rose from precision 0.958 to 0.977 (tags-bench-log.md 13).
SHIPPED_TAGS_VERIFICATION_THRESHOLD = 0.99998
# On the owner's 8K storm sky tile every tag under 0.9 on every strip named a bay or buildings the tile lacks.
SHIPPED_TAGS_POSITION_THRESHOLD = 0.9
# STRICT_THRESHOLDS as well. The owner ranks fewer wrong tags above more correct tags, and the correct tags kept per
# tile fell from 12.32 at the former 0.999 and 0.9999 prompt-only pair to 7.33.
SHIPPED_PROMPT_TAGS_VERIFICATION_THRESHOLD = 0.99998

# Caption input budget (total pixels, aspect preserved) — AB27's prep, what resample_for_vl
# falls back to, and the size every judged tests-AB arm was captioned at (ab_env.caption_preset
# pins it). Conditioning-side only: what the VLM reads is a COPY of the tile's crop, never the
# sampled tile itself (prime directive 1: a sampled tile is never resized, resampled or
# otherwise degraded).
VL_INPUT_BUDGET = 384 * 384
VL_INPUT_BUDGET_MEGAPIXELS = VL_INPUT_BUDGET / 1_000_000

# The caption input size the shipped file reads at, 768x1024 px. The owner's three-scene A/B
# found the VLM's sample size decides whether a caption invents or drops content, and this is
# the size where it does neither.
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

# One key table per tile_text kind. `tile_text` itself is checked first and sits outside both.
_PRESET_KEYS = {
    TILE_TEXT_CAPTION: {
        "tile_caption_instruction": str,
        "tile_caption_max_tokens": int,
        "global_style_instruction": str,
        "global_style_max_tokens": int,
    },
    TILE_TEXT_TAGS: {
        "tile_tags_instruction": str,
        "prompt_tags_instruction": str,
        "tile_tags_verification_statement": str,
        "tile_tags_verification_threshold": float,
        "prompt_tags_verification_threshold": float,
        "tile_tags_position_threshold": float,
        "global_style_instruction": str,
        "global_style_max_tokens": int,
    },
}

# The placeholder each tags key must hold, since without it the prompt or the tag never
# reaches the question.
_TAGS_PLACEHOLDERS = {"prompt_tags_instruction": PROMPT_PLACEHOLDER,
                      "tile_tags_verification_statement": TAG_PLACEHOLDER}

# The instruction keys with_prompt writes the prompt into, on either kind. A tags preset's
# prompt_tags_instruction stays a template, since the tags pass asks it only for a non-empty prompt.
_PROMPT_FILLED_KEYS = ("tile_caption_instruction", "global_style_instruction")

# The tags keys that never take the prompt. The tags pass fills {PROMPT} in
# prompt_tags_instruction only, so anywhere else it reaches the VL model literally.
_TAGS_PROMPT_FREE_KEYS = ("tile_tags_instruction", "tile_tags_verification_statement")

# The tagging stages take a verification threshold above 0 only, while a strip reads any score.
_TAGS_VERIFICATION_THRESHOLD_KEYS = ("tile_tags_verification_threshold", "prompt_tags_verification_threshold")

# Per-preset keys this version no longer reads, since the caption picture size lives in the
# [vision] table. Named so a user's older copy fails with the fix, not with "unknown key".
_REMOVED_PRESET_KEYS = ("tile_caption_megapixels", "global_style_megapixels")


@dataclass(frozen=True)
class VisionSettings:
    """The settings file's [vision] table: how every tile's conditioning samples the image.
    `canvas_tokens` and `crop_tokens` are the vision rows a tile takes from the entire
    image's encode and from its own crop's (vl.build_vision_rows; 0 turns a source off), and
    `caption_megapixels` is the picture the caption kind's tile captions and every style
    caption are written from. A tags preset reads it for the style caption only."""

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
    `global_style_instruction` is "" when this preset asks for no whole-image style caption.

    `tile_text` is the caption or the tags kind. A tags preset carries the tags fields in place
    of the tile question and budget, and `prompt` is the node's prompt input, which `with_prompt`
    stores on a tags preset because its prompt_tags_instruction is asked only when the prompt
    is not empty. An empty tile_tags_verification_statement, reachable only on a preset built
    in code, keeps every candidate tag without verifying or locating it. The thresholds are the
    scores that statement must reach on the entire tile to keep a tag the model listed, to keep
    a prompt tag it did not list, and on a strip of the tile to place a tag there."""

    surface: str
    label: str
    vision: VisionSettings
    tile_caption_instruction: str = ""
    tile_caption_max_tokens: int = 0
    global_style_instruction: str = ""
    global_style_max_tokens: int = 0
    tile_text: str = TILE_TEXT_CAPTION
    tile_tags_instruction: str = ""
    prompt_tags_instruction: str = ""
    tile_tags_verification_statement: str = ""
    tile_tags_verification_threshold: float = SHIPPED_TAGS_VERIFICATION_THRESHOLD
    prompt_tags_verification_threshold: float = SHIPPED_PROMPT_TAGS_VERIFICATION_THRESHOLD
    tile_tags_position_threshold: float = SHIPPED_TAGS_POSITION_THRESHOLD
    prompt: str = ""


def settings_path():
    """The settings file in force: the user's own copy when it exists, else the shipped one.
    Nothing in the package ever writes settings.user.toml, which is what makes it the edit
    surface that survives a node update."""
    user_path = SETTINGS_DIR / USER_SETTINGS_NAME
    return user_path if user_path.is_file() else SETTINGS_DIR / SETTINGS_NAME


def settings_fingerprint():
    """The file in force, as a value that changes whenever an edit to it would change a run.
    The name is carried beside the digest so that a user copy appearing or disappearing
    changes the fingerprint even when its bytes match the shipped file. A missing file is
    reported rather than raised, so the queue proceeds to the run-time read whose error
    message names the problem."""
    path = settings_path()
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return f"missing:{path.name}"
    return f"{path.name}:{digest}"


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


# The value rules below serve the settings file and the Tile Test nodes' widgets alike, so a
# value passes or fails one rule wherever it is set. The caller prefixes where it came from.

def caption_megapixels_error(value):
    """The caption_megapixels rule, a message or None. A widget range cannot express it."""
    if value == 0 or VL_INPUT_MIN_MEGAPIXELS <= value <= vl.PICTURE_CAP_MEGAPIXELS:
        return None
    return (
        "caption_megapixels must be 0, which reads the picture's own size, or between "
        f"{VL_INPUT_MIN_MEGAPIXELS} and {vl.PICTURE_CAP_MEGAPIXELS}. Got {value}.")


def vision_error(canvas_tokens, crop_tokens, caption_megapixels):
    """The [vision] table's rule over its three values, a message or None."""
    for key, value in (("canvas_tokens", canvas_tokens), ("crop_tokens", crop_tokens)):
        if not 0 <= value <= vl.MAX_VISION_TOKENS:
            return (f"{key} must be between 0 and {vl.MAX_VISION_TOKENS}, and 0 turns that source "
                    f"off. Got {value}.")
    if canvas_tokens == 0 and crop_tokens == 0:
        return ("canvas_tokens and crop_tokens are both 0. A tile needs vision rows from at least "
                "one of the two.")
    return caption_megapixels_error(caption_megapixels)


def score_error(key, value, above_zero):
    """A score's rule, a message or None. `above_zero` is the verification thresholds' floor."""
    floor_ok = value > 0 if above_zero else value >= 0
    if floor_ok and value <= 1:
        return None
    span = "above 0 and at most 1" if above_zero else "between 0 and 1"
    return f"{key} is a score and must be {span}. Got {value}."


def budget_error(key, value):
    """A caption budget's rule, a message or None."""
    if 1 <= value <= MAX_CAPTION_TOKENS:
        return None
    return (f"{key} is a caption budget and must be between 1 and {MAX_CAPTION_TOKENS} tokens. "
            f"Got {value}.")


def prompt_free_error(key, text):
    """The rule for a tags text that never takes the prompt (`_TAGS_PROMPT_FREE_KEYS`), a
    message or None."""
    if PROMPT_PLACEHOLDER not in text:
        return None
    return (f"{key} holds {PROMPT_PLACEHOLDER}, which only prompt_tags_instruction takes. Move "
            f"{PROMPT_PLACEHOLDER} to prompt_tags_instruction.")


def _reject(where, error):
    if error is not None:
        raise RuntimeError(f"Context-Anchored Tile Refine (VL): {where}: {error}")


def _check_types(where, block, keys):
    # A TOML int is a legal float value, so the float keys accept both. A bool is an int
    # subclass and is never either.
    for key, expected in keys.items():
        allowed = (int, float) if expected is float else expected
        value = block[key]
        if not isinstance(value, allowed) or isinstance(value, bool):
            _reject(where, f"{key} must be of type {expected.__name__}, got {type(value).__name__}.")


def _check_preset(path, label, block):
    # One [presets.<label>] block, checked key by key. A defect here is a hard error rather
    # than a fallback: a preset that silently loses a key would caption every tile with a
    # question its author never wrote.
    where = f"preset {label!r} in {path}"
    if not isinstance(block, dict):
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): preset {label!r} in {path} must be a "
            f"[presets.{label}] table, got {type(block).__name__}.")
    kind = block.get("tile_text", TILE_TEXT_CAPTION)
    _check_types(where, {"tile_text": kind}, {"tile_text": str})
    if kind not in TILE_TEXT_KINDS:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): preset {label!r} in {path} sets tile_text to "
            f"{kind!r}. Set it to one of {list(TILE_TEXT_KINDS)}, or leave it out for "
            f"{TILE_TEXT_CAPTION!r}.")
    keys = _PRESET_KEYS[kind]
    given = set(block) - {"tile_text"}
    missing = sorted(set(keys) - given)
    if missing:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): preset {label!r} in {path} is missing {missing}.")
    removed = sorted(given & set(_REMOVED_PRESET_KEYS))
    if removed:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): preset {label!r} in {path} carries {removed}, "
            "which this version no longer reads. The caption picture size is now "
            "caption_megapixels in the [vision] table. Copy settings.toml to settings.user.toml "
            "again and move your own values over.")
    unknown = sorted(given - set(keys))
    if unknown:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): preset {label!r} in {path} carries unknown keys "
            f"{unknown} for tile_text {kind!r}. A misspelled key would otherwise change nothing, "
            "silently.")
    _check_types(where, block, keys)
    if kind == TILE_TEXT_TAGS:
        if not block["tile_tags_instruction"].strip():
            raise RuntimeError(
                f"Context-Anchored Tile Refine (VL): preset {label!r} in {path} has an empty "
                "tile_tags_instruction. The tags kind needs a question that asks for the things "
                "in each tile.")
        for key, placeholder in _TAGS_PLACEHOLDERS.items():
            if placeholder not in block[key]:
                raise RuntimeError(
                    f"Context-Anchored Tile Refine (VL): preset {label!r} key {key} in {path} "
                    f"does not hold {placeholder}. Write {placeholder} where the text goes.")
        for key in _TAGS_PROMPT_FREE_KEYS:
            _reject(where, prompt_free_error(key, block[key]))
        for key in _TAGS_VERIFICATION_THRESHOLD_KEYS:
            _reject(where, score_error(key, block[key], above_zero=True))
        _reject(where, score_error("tile_tags_position_threshold", block["tile_tags_position_threshold"],
                                   above_zero=False))
        budget_keys = ("global_style_max_tokens",)
    else:
        if not block["tile_caption_instruction"].strip():
            raise RuntimeError(
                f"Context-Anchored Tile Refine (VL): preset {label!r} in {path} has an empty "
                "tile_caption_instruction. The caption vlm_methods need a question to ask about "
                "each tile.")
        budget_keys = ("tile_caption_max_tokens", "global_style_max_tokens")
    for key in budget_keys:
        _reject(where, budget_error(key, block[key]))


def _check_vision(path, block):
    # The [vision] table, checked key by key like a preset. A table that silently lost a key
    # would sample every tile at a size nobody chose.
    where = f"the [vision] table in {path}"
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
    _check_types(where, block, _VISION_KEYS)
    _reject(where, vision_error(block["canvas_tokens"], block["crop_tokens"], block["caption_megapixels"]))


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
    together and the list is ordered by whoever wrote the settings file. The FIRST preset is
    the DEFAULT, and its two options carry no label, so the shipped selector is the three bare
    surfaces. Every other preset's options read "<surface> (<label>)".

    The first preset's labeled form is what a workflow saved while every preset was labeled
    holds. The selector does not offer it, but `resolve_method` still resolves it by its label
    and the VL nodes' VALIDATE_INPUTS bypasses core's combo-list check.
    """
    options = [VLM_METHOD_VISION]
    for index, label in enumerate(presets):
        if index == 0:
            options.extend(CAPTION_SURFACES)
        else:
            options.extend(f"{surface} ({label})" for surface in CAPTION_SURFACES)
    return options


@functools.lru_cache(maxsize=1)
def vlm_methods():
    """The selector's options, built ONCE per ComfyUI session.

    The frontend caches a node's definition at startup, so a list that changed between calls
    would offer values the backend then rejects, or hide values a saved workflow carries.
    A new or renamed preset therefore needs a restart, while a preset's own wording does not.
    """
    return tuple(build_vlm_methods(load_settings().presets))


@functools.lru_cache(maxsize=1)
def preset_labels():
    """Every preset label in file order, caption and tags kind alike, read ONCE per ComfyUI
    session.

    Same cadence and same reason as `vlm_methods`: this becomes a combo the frontend caches
    at startup. An uncached re-read would let the preset a selector offers differ from the
    preset the vlm_method list was built from.
    """
    return tuple(load_settings().presets)


def default_vlm_method():
    # The first preset's slice+caption option, "vision tokens and captions" (build_vlm_methods).
    # The vision-only surface leads the list and is not it: the two halves together are what
    # the campaign settled on.
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
    """The preset label a vlm_method option carries. "" for the vision-only surface and for
    the two bare caption options, which the selector offers for the first preset and
    `resolve_method` routes to it."""
    surface = method_surface(vlm_method)
    if vlm_method == surface:
        return ""
    return vlm_method[len(surface) + 2:-1]


def resolve_method(vlm_method):
    """One vlm_method option resolved to the `Preset` the engine runs on.

    The settings file is read HERE, once per run, so an edit applies with no ComfyUI restart.
    "vision tokens" takes only the [vision] table. A caption option takes its preset as well,
    and a bare one (no label, what the selector offers for the first preset) takes the first
    preset, so the default preset resolves under the bare string and under its labeled form
    alike.
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
            f"{EXAMPLE_SETTINGS_NAME} in the node's folder carries the presets that no longer "
            f"ship. Copy that block into {USER_SETTINGS_NAME} and restart ComfyUI, or pick "
            "another vlm_method option.")
    block = presets[label]
    return build_preset(surface, label, settings.vision, block.get("tile_text", TILE_TEXT_CAPTION), block)


def build_preset(surface, label, vision, tile_text, values):
    """The `Preset` of one settings block, from `values` keyed by settings key. Only the keys of
    the `tile_text` kind are read. The settings file and the Tile Test: Captions node's sockets
    both build through here, so a new key lands in one constructor."""
    # A TOML int passes a float key's check, so each float key is cast to the type its field holds.
    fields = {key: float(values[key]) if expected is float else values[key]
              for key, expected in _PRESET_KEYS[tile_text].items()}
    # Whitespace-only is "off" too, so a user clearing the line by hand cannot leave a blank
    # style caption riding on top of every tile.
    if not fields["global_style_instruction"].strip():
        fields["global_style_instruction"] = ""
    return Preset(surface=surface, label=label, vision=vision, tile_text=tile_text, **fields)


def _fill_prompt(instruction, prompt, label, key):
    if PROMPT_PLACEHOLDER not in instruction:
        return instruction
    if prompt is None or not prompt.strip():
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): preset {label!r} asks for {PROMPT_PLACEHOLDER} "
            f"in its {key} and the node's prompt input is not connected or is empty. Connect "
            f"the positive prompt's text to prompt, or remove {PROMPT_PLACEHOLDER} from the "
            "instruction.")
    return instruction.replace(PROMPT_PLACEHOLDER, prompt.strip())


def with_prompt(preset, prompt):
    """`preset` with the node's prompt input written into every {PROMPT} its
    tile_caption_instruction and global_style_instruction carry. `prompt` is None when the
    optional socket is unconnected. The text is stripped, so a multiline primitive's trailing
    newline never lands inside the instruction's quotes. A
    preset without the placeholder is handed back unchanged, so the prompt is a no-op on it
    and on the vision-only surface. A placeholder met by no prompt is a hard error here,
    before any GPU time, rather than a question that quotes an empty prompt at every tile.

    A tags preset stores the stripped prompt, "" for none, and keeps
    `prompt_tags_instruction` as the template, since it is asked only when the prompt is not
    empty. Its style question still follows the rule above, so it never asks with the literal
    placeholder."""
    filled = {key: _fill_prompt(getattr(preset, key), prompt, preset.label, key)
              for key in _PROMPT_FILLED_KEYS}
    if preset.tile_text == TILE_TEXT_TAGS:
        filled["prompt"] = "" if prompt is None else prompt.strip()
    return replace(preset, **filled)


def style_row_count(preset, rows):
    """How many style rows a picture of `rows` batch rows writes, known before any generate.
    One per row when the preset asks a style question, for both kinds, since the style line
    is the style caption alone. The caption pass, the tags pass and the progress ledger all
    size from this one rule, so their counts cannot disagree."""
    if preset.global_style_instruction:
        return rows
    return 0


def _check_prompt_filled(preset):
    # A direct caller that skipped with_prompt would otherwise ask the VL model a question
    # holding the literal placeholder, at every tile, with nothing to say so.
    for key in _PROMPT_FILLED_KEYS:
        if PROMPT_PLACEHOLDER in getattr(preset, key):
            raise RuntimeError(
                f"Context-Anchored Tile Refine (VL): preset {preset.label!r} still carries "
                f"{PROMPT_PLACEHOLDER} in its {key}. Hand the preset through captions.with_prompt "
                "before captioning.")
    for key in _TAGS_PROMPT_FREE_KEYS:
        _reject(f"preset {preset.label!r}", prompt_free_error(key, getattr(preset, key)))


def caption_budget_pixels(megapixels, source):
    # The caption_megapixels semantics, in one place: 0 (or less) is the source's own area, so
    # the VL model reads every pixel the crop has, capped at vl.PICTURE_CAP_PIXELS. Above 0
    # the value is the budget itself, already range-checked by _check_vision.
    if megapixels <= 0:
        return min(int(source.shape[1]) * int(source.shape[2]), vl.PICTURE_CAP_PIXELS)
    return round(megapixels * 1_000_000)


# The `|$` is load-bearing: without it an unclosed <think> makes the sub a no-op, so a
# reasoning turn cut short by max_tokens comes back as the caption, non-empty, and
# generate_caption's fallback chain never fires.
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

    Here it is mandatory, because the caption is encoded as text, so an unstripped <think>
    block would reach the DiT as several hundred tokens of the model talking to itself.

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


# A seed re-roll on the production upscale node re-executes the entire node, captions
# included, which costs minutes on a 24 tile grid. Every caption here is greedy
# (do_sample=False, so every token is the argmax) and depends on nothing but the picture, the
# question, the budget and the CLIP, so the stored text is what a second pass would write.
# Hashing a 0.79 megapixel float32 picture takes a few milliseconds, far below one VLM token.
CAPTION_CACHE_ENTRIES = 512


class ClipBoundCache:
    """A least recently used store of values bound to the CLIP that wrote them, capped at
    `entries`. A stored value is never None, so `get` returns None for a miss."""

    def __init__(self, entries):
        self.entries = entries
        self._store = OrderedDict()

    def get(self, key, clip):
        # The entry holds a weak reference, so a CLIP the user has unloaded is never kept alive
        # by the cache and its stale value is never served to a different model.
        entry = self._store.get(key)
        if entry is None or entry[0]() is not clip:
            return None
        self._store.move_to_end(key)
        return entry[1]

    def put(self, key, clip, value):
        self._store[key] = (weakref.ref(clip), value)
        self._store.move_to_end(key)
        while len(self._store) > self.entries:
            self._store.popitem(last=False)

    def clear(self):
        self._store.clear()


_CAPTION_CACHE = ClipBoundCache(CAPTION_CACHE_ENTRIES)


def clear_caption_cache():
    """Empty the caption cache. This is the one public way to reset it."""
    _CAPTION_CACHE.clear()


def picture_digest(picture):
    """A cache key part for a picture tensor: the sha256 of its pixels, its dtype and its shape."""
    # float32 because bfloat16 has no numpy dtype and .numpy() raises on it, while
    # resample_for_vl keeps whatever dtype the IMAGE arrived with. The dtype and shape ride
    # alongside so two pictures that share a byte pattern in different layouts stay apart.
    pixels = picture.contiguous().cpu().float().numpy().tobytes()
    return (hashlib.sha256(pixels).hexdigest(), str(picture.dtype), tuple(picture.shape))


# 0.5.0 grounds a prompt tag "sky" in a prompt reading "skies". The tags pass keeps a prompt tag only when every word
# grounds, so an older library drops correct prompt tags.
LIBRARY_VERSION = "0.5.0"


def _library_error(reason):
    return RuntimeError(
        f"Context-Anchored Tile Refine (VL): every VL run needs logit-classifier {LIBRARY_VERSION} or newer, "
        f"and {reason}. Install or upgrade it in the ComfyUI Python environment with: "
        f'pip install -U "logit-classifier>={LIBRARY_VERSION}"')


def comfy_toolkit():
    """logit_classifier.toolkit.comfyui, or a RuntimeError naming the pip command that installs
    LIBRARY_VERSION when the library is missing, older, or reports no version this gate can read."""
    installed = None
    installed_version = None

    try:
        import logit_classifier
        import logit_classifier.toolkit.comfyui as toolkit
    except ImportError as error:
        raise _library_error(f"its toolkit cannot be imported ({error})") from error
    # Core requires transformers, which requires packaging, so pip's own PEP 440 ordering is present wherever this
    # node runs.
    from packaging.version import InvalidVersion, Version

    # The module attribute, never importlib.metadata, since an editable install can keep a stale version in its
    # dist metadata.
    installed = getattr(logit_classifier, "__version__", None)
    if installed is None:
        raise _library_error("the installed library reports no version")
    try:
        installed_version = Version(installed)
    except (InvalidVersion, TypeError) as error:
        raise _library_error(
            f"the installed library reports version {installed!r}, which is not a version number") from error
    if installed_version < Version(LIBRARY_VERSION):
        raise _library_error(f"version {installed} is installed")
    return toolkit


def generate_caption(clip, vl_input, instruction, max_length, thinking=True, scope=()):
    """One greedy caption of `vl_input`, then the settled fallback chain for a crop whose
    stop token fires immediately. `max_length` is per-instruction (the preset's max_tokens on
    the live surfaces) and has to cover the reasoning turn as well as the answer, which is
    why the budgets sit far above the visible answer length.

    The answer is cached in process against the picture, the question, the budget, the
    reasoning flag, `scope` and the CLIP, so a re-run of the node writes it once.
    `scope` is a hashable tuple naming this request's place in a run. Two byte-equal crops
    within one run are captioned separately because their scopes differ. A direct caller with
    nothing to name may leave it empty."""
    if not hasattr(clip, "generate") or not hasattr(clip, "decode"):
        raise RuntimeError(
            "Context-Anchored Tile Refine (VL): this CLIP cannot generate text. The caption "
            "vlm_methods need a vision-language text encoder with a text-generation head "
            "(Krea 2 family). Use vlm_method 'vision tokens' with any other CLIP.")
    clip_generate = comfy_toolkit().clip_generate

    key = (picture_digest(vl_input), instruction, max_length, thinking, scope)
    stored = _CAPTION_CACHE.get(key, clip)
    if stored is not None:
        return stored

    tokens, _tail = vl.tokenize_with_image(clip, instruction, vl_input, thinking=thinking)
    ids = clip_generate(clip, tokens, do_sample=False, max_length=max_length, repetition_penalty=1.05)
    text = strip_thinking(clip.decode(ids))
    if not clean_caption(text):
        ids = clip_generate(clip, tokens, do_sample=True, max_length=max_length, temperature=0.7,
                            top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, seed=42)
        text = strip_thinking(clip.decode(ids))
    if not clean_caption(text):
        retokens, _tail = vl.tokenize_with_image(clip, "Write one short sentence describing this image.",
                                                 vl_input, thinking=False)
        ids = clip_generate(clip, retokens, do_sample=False, max_length=max_length, repetition_penalty=1.05)
        text = strip_thinking(clip.decode(ids))
    if not clean_caption(text):
        raise RuntimeError(
            "Context-Anchored Tile Refine (VL): caption generation returned an empty answer "
            "after every fallback.")
    _CAPTION_CACHE.put(key, clip, text)
    return text


def style_caption(clip, row, preset, scope):
    """One picture row's cleaned style caption, read at the [vision] table's caption size."""
    vl_input = resample_for_vl(row, caption_budget_pixels(preset.vision.caption_megapixels, row))
    return clean_caption(generate_caption(clip, vl_input, preset.global_style_instruction,
                                          preset.global_style_max_tokens, thinking=True, scope=scope))


def generate_caption_set(clip, source, tiles, preset, batch_size=1, batch_index=0,
                         progress=None, style_source=None):
    """The style caption and one caption per tile per batch row, read off the FROZEN raw
    canvas and returned APART, as (style_texts, captions).

    `style_texts` is one whole-image style caption per batch row, or empty when the preset asks
    for none. `captions[tile_index][batch_row]` is each tile's OWN caption, without the style.
    `join_style_captions` is the form the DiT reads, and `generate_tile_captions` is the two
    together. The Tile Test: Captions node reads this function so its listing can show the
    style once, labelled, instead of as the unlabelled first line of every tile.
    Batch rows are captioned INDEPENDENTLY: core's
    tokenizer attaches images[0] alone (comfy/text_encoders/qwen_vl.py process_qwen2vl_images),
    so a whole [B,H,W,3] crop would describe every row with row 0's picture. Through the node
    `source` always holds exactly ONE picture — sampling.refine_image's picture loop is outside
    this pre-pass — so the row axis is length 1 there and batch_size/batch_index carry the
    picture's place in the run, which is all the ProgressBar below needs to span it.

    `preset` is the run's resolved settings block (`resolve_method`), which carries the tile
    question, both generation budgets and the [vision] table's caption picture size, read for
    the tile caption and the style caption alike. A non-empty
    `preset.global_style_instruction` adds ONE whole-image style caption per batch row, generated
    FIRST from `style_source` (default `source`, and the region path passes the full image so
    that a masked refine's style stays global). It is counted as the segment's first
    caption(s). An empty one leaves this function byte-identical to the style-free path.

    The pre-pass is one clip.generate per tile per row at up to the preset's max_tokens, which
    on a 16-tile grid runs for minutes before the first tile samples. Hence the per-tile
    interrupt check and the ProgressBar: without them the run is uncancellable and the UI
    shows nothing until the tile loop starts.

    `progress` is the VL run's ledger (progress.py) when a node created one. With it the
    standalone bar is NOT constructed — a second bar is exactly the display reset the ledger
    exists to remove — and each finished caption is reported to the ledger's open caption
    segment with the SAME (done, total) counters the bar carries, so its per-caption chunk
    snaps to its boundary while core's per-token bar (routed by the ledger's shim) fills it
    in between. Without a ledger nothing here changes.
    """
    import comfy.model_management
    import comfy.utils

    if preset.tile_text == TILE_TEXT_TAGS:
        # A tags preset carries no tile question and a 0 budget, so the caption pass would ask
        # every tile nothing.
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): preset {preset.label!r} is the tags kind "
            f"(tile_text = {TILE_TEXT_TAGS!r}), which runs through the tags module and never "
            "through the caption pass. Pick a preset without tile_text, or one set to "
            f"{TILE_TEXT_CAPTION!r}.")
    _check_prompt_filled(preset)
    batch = int(source.shape[0])
    style_rows = style_row_count(preset, batch)
    style_on = style_rows > 0
    per_picture = len(tiles) * batch + style_rows
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
            style_texts.append(style_caption(clip, style_canvas[b:b + 1], preset, ("style", b, batch_index)))
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
            text = generate_caption(clip, vl_input, preset.tile_caption_instruction,
                                    preset.tile_caption_max_tokens, thinking=True,
                                    scope=("tile", crop.x0, crop.y0, crop.x1, crop.y1, b,
                                           batch_index))
            row_captions.append(clean_caption(text))
            done += 1
            if pbar is None:
                progress.caption_done(done, total)
            else:
                pbar.update_absolute(done, total)
        captions.append(row_captions)
    return style_texts, captions


def join_style_captions(style_texts, captions):
    """The form the DiT reads: every tile caption of a batch row with that row's style caption
    on top, so all tiles follow one style description. Empty `style_texts` hands `captions`
    back untouched. Keeps the [tile][batch row] shape."""
    if not style_texts:
        return captions
    return [[f"{style_texts[b]}\n{caption}" for b, caption in enumerate(rows)]
            for rows in captions]


def generate_tile_captions(clip, source, tiles, preset, batch_size=1, batch_index=0,
                           progress=None, style_source=None):
    """`generate_caption_set` joined by `join_style_captions`: captions[tile_index][batch_row]
    in the form the engine's pre-pass encodes."""
    style_texts, captions = generate_caption_set(
        clip, source, tiles, preset, batch_size=batch_size, batch_index=batch_index,
        progress=progress, style_source=style_source)
    return join_style_captions(style_texts, captions)


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
    _tokens, tail_len = vl.tokenize_with_image(clip, vl.VISION_BLOCK + caption, probe_image,
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


def build_slice_caption_conds(clip, encode_source, tiles, captions, vision, offset_x=0, offset_y=0,
                              budget_tiles=None):
    """VL rows AND captions: each tile's positive is its vision rows exactly as the `vision
    tokens` surface builds them (vl.build_vision_rows: its own crop's cells, then its slice
    of the entire image), followed by that tile's own caption encoded TEXT-ONLY, concatenated
    on the row axis.

    The caption is encoded alone, never inside a whole-canvas vision encode, which would leak
    far-canvas content into every tile's caption rows through attention (the phantom-moon
    failure) and cost one whole-canvas encode per tile. The vision half is unaffected because
    attention is causal (docs/vl-conditioning-encode-cost.md sections 6-7).

    `encode_source` is the image the OFFSET tile rects index, taken separately from the canvas
    the captions describe — mirroring vl.build_vision_rows: on the whole-image path both are
    the padded canvas at offset 0, while on the mask path the captions describe each region
    tile's own crop and the canvas rows come from the FULL image with the bbox origin as the
    offset, so a masked refine stays globally informed.

    `budget_tiles` is vl.build_vision_rows' own override, passed straight through, so this
    surface and `vision tokens` can never sample the canvas at different sizes."""
    batch = int(encode_source.shape[0])
    tile_positives = []

    if any(len(tile_captions) != batch for tile_captions in captions):
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): {batch} image(s) in the batch but a tile was "
            "captioned a different number of times. Every batch row must carry its own caption "
            "or a row would be conditioned on another row's picture.")

    # The vision half is vl's own, so this surface and `vision tokens` can never build
    # differently shaped rows, and vl.build_vision_rows rejects a batch. The tail is left off:
    # the one template tail the stream may carry arrives with the caption rows concatenated
    # after the vision rows.
    tile_rows, probe_image = vl.build_vision_rows(clip, encode_source, tiles, vision,
                                                  offset_x, offset_y, with_tail=False,
                                                  budget_tiles=budget_tiles)

    for tile_captions, rows in zip(captions, tile_rows, strict=True):
        # Exactly ONE row, so nothing is concatenated across rows: the count guard above ties
        # len(tile_captions) to the encode source's batch, and refine_image's picture loop
        # makes that batch 1.
        caption = tile_captions[0]
        # The probe is never encoded and vl.tokenize_with_image counts the rows after a single image
        # token, so any resampled copy gives the same tail length.
        tail_len = _caption_tail_len(clip, caption, probe_image)
        caption_entries = _encode_caption_text_only(clip, caption, tail_len)
        tile_positives.append(vl._convert(vl.cat_rows(rows, caption_entries, "caption encode")))
    return tile_positives
