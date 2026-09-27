"""Per-tile tag text: the tile text a tags preset writes in place of a caption.

Two entry points run one pass. `generate_tag_trace` returns every stage's result as a
`TagRun`: the style line per picture row, the prompt tags as a `PromptTrace` and a
`TileTrace` per tile row, so a test node can show what each stage did. `generate_tag_set` is
the engine's, and reads only the texts from that run, as `(style_texts, tile_texts)` in the
shape `captions.generate_caption_set` returns, so the encode stage reads either one. The stages:

    picture pass   the style caption (captions.style_caption), which is the style line
                   alone, then the prompt tags (logit_classifier.toolkit's prompt_tags): one
                   text-only generate lists the physical things the prompt names, and the
                   thing check keeps the ones that are things. They join every tile's
                   candidates. The prompt reaches no other question, so a long prompt never
                   lengthens a tile's reply.
    per tile row   the toolkit's tag_picture on a 1 MP copy of the crop: propose (stopped
                   after MAX_PROPOSED_TAGS tags), merge (the model's tags first, then the prompt
                   tags, category nouns and repeats dropped), thing check, verify (the model's
                   tags at the verification threshold and a prompt tag the model did not list
                   at the prompt tags threshold) and clean (drop_subsets). Then this module's
                   locate (the verification statement on three horizontal and three vertical
                   strips of the full-resolution crop, a tag present on no strip dropped) and
                   render ("<item> <term>"). An empty verification statement skips verify and
                   locate and keeps every candidate.

`tag_settings` hands the toolkit this pack's values. Every constant below was measured in the
Logit Tagger's harnesses or in tests-AB, and the 2026-09-25 changes in tests-AB/ab_tags_bench.py
(log: tests-AB/tags-bench-log.md). Module scope is torch and stdlib only. comfy and
logit_classifier are imported inside functions, so a missing library fails with its pip command
(a subprocess test pins the comfy half).
"""
from dataclasses import dataclass

from . import captions
from .grid import Rect

# Core only resizes a VL image above 12.8 MP. The Logit Tagger's tests-AB/ab_tagger.py fit a
# 42 candidate verify pass beside a 1 MP picture, and tests-AB/ab_tile_tags.py ran at 1 MP.
VL_MAX_PIXELS = 1024 * 1024

# In the Logit Tagger's tests-AB/ab_propose_budget.py both images that reached 128 were
# repeating tags, and 192 added no tag on nine images.
PROPOSE_MAX_TOKENS = 128

# The wrong and vague tags of a greedy list sit at its tail. Stopping after 25 tags raised the
# judged precision of the no-prompt lists from 0.865 to 0.915 (tests-AB/tags-bench-log.md).
MAX_PROPOSED_TAGS = 25

# The model's 25 and a long prompt's 30 to 40 tags, in one packed verify pass beside a 1 MP
# picture. Model tags come first, so prompt tags never push a model tag out.
MAX_MERGED_TAGS = 64

# tests-AB/ab_tile_tags.py on market: the propose instruction's own nouns came back as tags
# and passed verify.
CATEGORY_NOUNS = frozenset({"objects", "people", "animals", "clothing", "materials", "setting"})

# tests-AB/ab_tile_position.py: strips at 0.25 MP placed items as precisely as 0.5 MP.
STRIP_MEGAPIXELS = 0.25

ROW_WORDS = ("top", "center", "bottom")
COLUMN_WORDS = ("left", "center", "right")

# A 500 word prompt listed 28 to 45 tags in 70 to 256 tokens (tests-AB/tags-bench-log.md).
PROMPT_TAGS_MAX_TOKENS = 256

# Past the prompt's own things the model pads the list with words the prompt lacks
# ("materials", "details"), so two such tags in a row end it.
UNGROUNDED_STREAK = 2

# The thing check: one text-only choice per tag string. It drops whole-scene and setting words
# ("cityscape", "urban environment"), lighting, color and quality words, which no threshold on
# the verify score separates from real things. This wording counts landscape features, groups
# and light sources as things, which an earlier one dropped ("red moon", "army of soldiers").
THING_QUESTION = 'What does the image tag "{TAG}" name'
THING_CRITERIA = {
    "thing": "something that could be pointed at in a picture: an object, a person, a group of people "
             "or animals, an animal, a plant, a body part, a garment, a material, a building, a "
             "landscape feature such as hills, cliffs, sky, the moon or stars, a light source such as "
             "lamps, signs or lit windows, or a substance such as water, smoke, fire, clouds or rain",
    "other": "nothing to point at: a kind of place or a whole scene, a time of day, a color alone, a "
             "shape or form, lighting in general, a camera or render effect, a style, a mood, a "
             "quality or an idea",
}
# Mid scores mix real things with non-things ("chinese characters" 0.45, "huge" 0.43), so only a
# confident "other" drops a tag.
THING_THRESHOLD = 0.9


def check_tags_ready(clip):
    """Raises before any generate when a tags preset cannot run: a logit_classifier missing or
    older than captions.LIBRARY_VERSION, or a CLIP that is not a Qwen3-VL text encoder with its
    vision tower. The library is never judged by importlib.metadata, since an editable install
    can report a stale version in its dist metadata."""
    build_classifier(clip)


def build_classifier(clip):
    """One Classifier over the workflow's CLIP, with no prior drift and no calibration file. A
    wrong or vision-less CLIP raises the toolkit's error, which names this node and the fix."""
    return captions.comfy_toolkit().comfy_classifier(clip, node="Context-Anchored Tile Refine (VL)")


def tag_settings(preset):
    """The toolkit's TagSettings for a tags preset: its wordings and thresholds, and this
    module's budgets, caps and thing check."""
    from logit_classifier.toolkit.comfyui import TagSettings

    return TagSettings(
        propose_instruction=preset.tile_tags_instruction,
        # The Captions node and a Preset built in code leave it "", and TagSettings refuses a
        # wording without {PROMPT}. An empty one is never asked.
        prompt_tags_instruction=preset.prompt_tags_instruction or captions.PROMPT_PLACEHOLDER,
        verify_statement=preset.tile_tags_verification_statement,
        verify_threshold=preset.tile_tags_verification_threshold,
        prompt_only_threshold=preset.prompt_tags_verification_threshold,
        propose_cap=MAX_PROPOSED_TAGS,
        merge_cap=MAX_MERGED_TAGS,
        category_nouns=CATEGORY_NOUNS,
        propose_max_tokens=PROPOSE_MAX_TOKENS,
        prompt_tags_max_tokens=PROMPT_TAGS_MAX_TOKENS,
        ungrounded_streak=UNGROUNDED_STREAK,
        thing_question=THING_QUESTION,
        thing_criteria=tuple(THING_CRITERIA.items()),
        thing_threshold=THING_THRESHOLD,
        thing_check="all",
        # The Logit Tagger measured the language gate, the echo fallback, score order and the
        # head-noun rule, and this pack's bench has not, so the pass keeps the stages it judged.
        subsets_rule="words",
        order="proposed",
        language_question=None,
        echo_fallback_instruction=None,
    )


def _statements(items, statement):
    return [statement.replace(captions.TAG_PLACEHOLDER, item) for item in items]


def is_thing(p_other):
    return p_other < THING_THRESHOLD


def prompt_tags_question(preset):
    """prompt_tags_instruction with the prompt in place of its placeholder."""
    return preset.prompt_tags_instruction.replace(captions.PROMPT_PLACEHOLDER, preset.prompt)


def strip_rects(height, width):
    """The top, center and bottom thirds at full width, then the left, center and right thirds
    at full height, in crop pixels."""
    rows = [height * i // 3 for i in range(4)]
    columns = [width * i // 3 for i in range(4)]
    return (*(Rect(0, rows[i], width, rows[i + 1]) for i in range(3)),
            *(Rect(columns[i], 0, columns[i + 1], height) for i in range(3)))


def axis_word(probabilities, words, threshold):
    """The word of the one strip on this axis that holds the item, or None when no strip or
    more than one does.

    Two adjacent strips named the center or the side by a coin flip, and a term covering part
    of a spread tag points the diffusion model at the wrong place. On the grounded items of
    tests-AB/ab_tile_position.py the one strip rule was right 0.980 of the time against 0.960
    for the p-weighted mean of the holding strips, and it names about half as many."""
    present = [index for index, p in enumerate(probabilities) if p >= threshold]
    if len(present) != 1:
        return None
    return words[present[0]]


def position_term(row, column):
    """"<row>-<column>", or one word when an axis names nothing or the column is the center."""
    if row is None or column is None:
        return row or column or ""
    if column == "center":
        return row
    return f"{row}-{column}"


def strip_scores(classifier, crop, items, statement):
    """Per item the verify p on each strip of `crop` in `strip_rects` order, from one packed
    verify request per strip."""
    budget = round(STRIP_MEGAPIXELS * 1_000_000)
    statements = _statements(items, statement)
    per_strip = []
    for rect in strip_rects(int(crop.shape[1]), int(crop.shape[2])):
        strip = captions.resample_for_vl(crop[:, rect.y0:rect.y1, rect.x0:rect.x1, :], budget)
        per_strip.append(classifier.nouls(statements, image=strip))
    return tuple(tuple(p[index] for p in per_strip) for index in range(len(items)))


def strip_term(strips, threshold):
    """The position term of one item's six strip probabilities."""
    return position_term(axis_word(strips[:3], ROW_WORDS, threshold),
                         axis_word(strips[3:], COLUMN_WORDS, threshold))


def on_no_strip(strips, threshold):
    """Whether no strip holds the item. The entire tile's score can pass a tag that no part of
    the tile shows, and on the owner's storm sky tile every such tag named something the tile
    lacks."""
    return max(strips) < threshold


def render_tags(items, terms):
    """"<item> <term>" per item, or the item alone without a term, joined by ", "."""
    return ", ".join(f"{item} {term}" if term else item for item, term in zip(items, terms, strict=True))


@dataclass(frozen=True)
class PromptTrace:
    """The prompt tags of one picture. `reply` is the VL model's list as written, cut back to its
    last whole tag when it fills the budget, `listed` its tags whose every word is a word of the
    prompt, `p_other` each listed tag's thing check score, and `tags` the listed tags the thing
    check keeps, which join every tile's candidates."""

    reply: str
    listed: tuple
    p_other: tuple
    tags: tuple


@dataclass(frozen=True)
class TileTrace:
    """One tile row's stages. `origins` names where each candidate came from and `dropped`
    what the merge and the thing check left out and why. `scores` holds None per candidate
    when verify is off. `strips` holds per kept item its six strip probabilities in
    `strip_rects` order, and `unplaced` the kept items no strip holds, which the text leaves
    out. With locate off every `strips` entry is (), every term "" and `unplaced` empty."""

    reply: str
    proposed: tuple
    candidates: tuple
    origins: tuple
    dropped: tuple
    scores: tuple
    verified: tuple
    kept: tuple
    strips: tuple
    terms: tuple
    unplaced: tuple
    text: str


def trace_tile(clip, classifier, crop, picture, preset, prompt_tags, known_things, locate=True):
    """Every stage of one tile row: the toolkit's tag_picture on `picture` (propose, merge with
    `prompt_tags`, the thing check with answers shared through `known_things`, verify and
    clean), then locate on the full-resolution `crop` unless `locate` is off. An empty
    verification statement skips verify and locate, so every candidate is kept."""
    from logit_classifier.toolkit.comfyui import tag_picture

    statement = preset.tile_tags_verification_statement
    position_threshold = preset.tile_tags_position_threshold
    trace = tag_picture(clip, classifier, picture, prompt_tags=prompt_tags, settings=tag_settings(preset),
                        known=known_things)
    kept = trace.kept
    strips = ((),) * len(kept)
    terms = ("",) * len(kept)
    unplaced = ()

    if statement and locate and kept:
        strips = strip_scores(classifier, crop, kept, statement)
        terms = tuple(strip_term(p, position_threshold) for p in strips)
        unplaced = tuple(item for item, p in zip(kept, strips, strict=True)
                         if on_no_strip(p, position_threshold))
    placed = [(item, term) for item, term in zip(kept, terms, strict=True) if item not in unplaced]
    return TileTrace(reply=trace.reply, proposed=trace.proposed, candidates=trace.candidates,
                     origins=trace.origins, dropped=trace.dropped, scores=trace.scores,
                     verified=trace.verified, kept=kept, strips=strips, terms=terms, unplaced=unplaced,
                     text=render_tags([item for item, _ in placed], [term for _, term in placed]))


@dataclass(frozen=True)
class TagRun:
    """Every stage of one tags pass. `style_texts` is the style line per batch row, "" for
    none, `prompt` is the prompt tags or None when the preset carries no prompt, and
    `tiles[tile_index][batch_row]` is a TileTrace."""

    style_texts: tuple
    prompt: PromptTrace | None
    tiles: tuple


# The tags pass is greedy and its requests deterministic, so a seed re-roll that re-executes
# the node would otherwise pay every propose and verify again, as captions.generate_caption's
# cache explains for captions.
TAG_CACHE_ENTRIES = 512
_TAG_CACHE = captions.ClipBoundCache(TAG_CACHE_ENTRIES)


def clear_tag_cache():
    """Empty the tag cache. This is the one public way to reset it."""
    _TAG_CACHE.clear()


def _tag_cache_key(picture, preset, scope):
    # The settings object holds every value a toolkit stage reads, so a changed value never
    # serves text written under the old one. A picture of None keys a text-only request.
    digest = () if picture is None else captions.picture_digest(picture)
    return (digest, tag_settings(preset), VL_MAX_PIXELS, STRIP_MEGAPIXELS, ROW_WORDS, COLUMN_WORDS,
            preset.tile_tags_position_threshold, preset.style_instruction, preset.prompt, scope)


def _cached(key, clip, compute):
    stored = _TAG_CACHE.get(key, clip)
    if stored is not None:
        return stored
    value = compute()
    _TAG_CACHE.put(key, clip, value)
    return value


def _prompt_trace(clip, classifier, preset, known_things):
    from logit_classifier.toolkit.comfyui import prompt_tags

    listing = prompt_tags(clip, classifier, preset.prompt, tag_settings(preset), known_things)
    return PromptTrace(reply=listing.reply, listed=listing.listed, p_other=listing.p_other,
                       tags=listing.tags)


def generate_tag_trace(clip, source, tiles, preset, batch_size=1, batch_index=0, progress=None,
                       style_source=None, locate=True):
    """Every stage of the tags pass, as a TagRun.

    Its `style_texts` hold one style caption per batch row when `captions.style_row_count`
    counts style rows, else "" per row. `style_source` (default `source`) is what the style
    caption reads, and `progress` is the VL run's ledger or None. `locate` off skips the strip
    requests, so every tile text carries no position term."""
    import comfy.model_management
    import comfy.utils

    if preset.kind != captions.TILE_TEXT_TAGS:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): preset {preset.label!r} is the {preset.kind!r} "
            "kind and the tags pass runs only a preset with tile_text = "
            f"{captions.TILE_TEXT_TAGS!r}. Hand a caption preset to captions.generate_caption_set.")
    captions._check_prompt_filled(preset)
    # Built before any generate, so a CLIP or a library the pass cannot run fails here.
    classifier = build_classifier(clip)

    batch = int(source.shape[0])
    style_rows = captions.style_row_count(preset, batch)
    per_picture = len(tiles) * batch + style_rows
    total = per_picture * batch_size
    pbar = None if progress is not None else comfy.utils.ProgressBar(total)
    done = per_picture * batch_index
    known_things = {}
    prompt_trace = None
    style_texts = [""] * batch
    tile_traces = []

    def advance():
        nonlocal done
        done += 1
        if pbar is None:
            progress.caption_done(done, total)
        else:
            pbar.update_absolute(done, total)

    with captions.comfy_toolkit().skip_resident_loads(clip):
        if preset.prompt and preset.prompt_tags_instruction:
            prompt_trace = _cached(_tag_cache_key(None, preset, ("prompt tags",)), clip,
                                   lambda: _prompt_trace(clip, classifier, preset, known_things))
        prompt_tags = () if prompt_trace is None else prompt_trace.tags

        if style_rows:
            style_canvas = source if style_source is None else style_source
            if int(style_canvas.shape[0]) != batch:
                raise RuntimeError(
                    f"Context-Anchored Tile Refine (VL): {batch} batch row(s) to tag but the style "
                    f"canvas has {int(style_canvas.shape[0])}. Every row needs its own style line or "
                    "a row would carry another row's style.")
            comfy.model_management.throw_exception_if_processing_interrupted()
            for b in range(batch):
                style_texts[b] = captions.style_caption(clip, style_canvas[b:b + 1], preset,
                                                        ("style", b, batch_index))
                advance()

        for tile in tiles:
            comfy.model_management.throw_exception_if_processing_interrupted()
            crop = tile.crop_rect
            row_traces = []
            for b in range(batch):
                row = source[b:b + 1, crop.y0:crop.y1, crop.x0:crop.x1, :]
                picture = captions.resample_for_vl(row, VL_MAX_PIXELS)
                scope = ("tile", crop.x0, crop.y0, crop.x1, crop.y1, b, batch_index, locate)
                row_traces.append(_cached(_tag_cache_key(picture, preset, scope), clip,
                                          lambda row=row, picture=picture: trace_tile(
                                              clip, classifier, row, picture, preset, prompt_tags,
                                              known_things, locate)))
                advance()
            tile_traces.append(tuple(row_traces))
    return TagRun(style_texts=tuple(style_texts), prompt=prompt_trace, tiles=tuple(tile_traces))


def generate_tag_set(clip, source, tiles, preset, batch_size=1, batch_index=0, progress=None,
                     style_source=None):
    """The style line and one tag text per tile per batch row, as (style_texts, tile_texts).

    The same contract as `captions.generate_caption_set`: `tile_texts[tile_index][batch_row]`
    is each tile's own text without the style, `style_texts` is one line per batch row or
    empty, and the arguments are `generate_tag_trace`'s with locate on. A style row is counted
    when `captions.style_row_count` says so, whether or not a style line came out, so the
    ledger and this pass agree."""
    run = generate_tag_trace(clip, source, tiles, preset, batch_size, batch_index, progress,
                             style_source, locate=True)
    return tag_texts(run)


def tag_texts(run):
    """A TagRun's texts as (style_texts, tile_texts), in `generate_tag_set`'s shape. A run whose
    style rows wrote no text returns empty `style_texts`, so no tile carries a blank style line."""
    style_texts = list(run.style_texts)
    tile_texts = [[trace.text for trace in row_traces] for row_traces in run.tiles]

    if not any(style_texts):
        style_texts = []
    return style_texts, tile_texts
