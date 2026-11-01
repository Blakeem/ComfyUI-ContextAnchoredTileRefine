"""Per-tile tag text: the tile text a tags preset writes in place of a caption.

Two entry points run one pass. `generate_tag_trace` returns every stage's result as a
`TagRun`: the style line per picture row, the prompt tags as the toolkit's `PromptTagTrace` and
a `TileTrace` per tile row, so a test node can show what each stage did. `generate_tag_set` is
the engine's, and reads only the texts from that run, as `(style_texts, tile_texts)` in the
shape `captions.generate_caption_set` returns, so the encode stage reads either one. The stages:

    picture pass   the prompt tags (logit_classifier.toolkit's tag_prompt): the language
                   question, a translation when the prompt is not English, one text-only
                   generate listing the physical things its English text names, and the thing
                   check, which keeps the ones that are things. They join every tile's
                   candidates. Then the style caption (captions.style_caption), which is the
                   style line alone. The prompt reaches no tile question, so a long prompt
                   never lengthens a tile's reply.
    per tile row   the toolkit's tag_image on a 1 MP copy of the crop: propose (stopped
                   after the settings' propose_cap tags, asked once more with the fallback
                   question when the reply names only category nouns), merge (the model's tags
                   first, then the prompt tags, category nouns and repeats dropped), thing
                   check, verify (the model's tags at the verification threshold and a prompt
                   tag the model did not list at the prompt tags threshold) and clean
                   (drop_subsets). Then the toolkit's locate_tags (the verification statement
                   on its strip_boxes, three horizontal and three vertical strips of the
                   full-resolution crop, a tag present on no strip dropped) and render_placed
                   ("<item> <term>"). An empty verification statement skips verify and locate
                   and keeps every candidate.

`tag_settings` hands the toolkit this pack's wordings and thresholds, and the toolkit's
defaults hold every other value, measured in the Logit Tagger's harnesses and in tests-AB
(tests-AB/ab_tags_bench.py, log: tests-AB/tags-bench-log.md). Module scope is torch and stdlib
only. comfy and logit_classifier are imported inside functions, so a missing library fails
with its pip command (a subprocess test pins the comfy half).
"""
from dataclasses import dataclass

from . import captions

# Core only resizes a VL image above 12.8 MP. The Logit Tagger's tests-AB/ab_tagger.py fit a
# 42 candidate verify pass beside a 1 MP picture, and tests-AB/ab_tile_tags.py ran at 1 MP.
VL_MAX_PIXELS = 1024 * 1024


def check_tags_ready(clip):
    """Raises before any generate when a tags preset cannot run: a logit_classifier missing or
    older than captions.LIBRARY_VERSION (the version gate in captions.comfy_toolkit), or a CLIP
    that is not a Qwen3-VL text encoder with its vision tower."""
    build_classifier(clip)


def build_classifier(clip):
    """One Classifier over the workflow's CLIP, with no prior drift and no calibration file. A
    wrong or vision-less CLIP raises the toolkit's error, which names this node and the fix."""
    return captions.comfy_toolkit().comfy_classifier(clip, node="Context-Anchored Tile Refine (VL)")


def tag_settings(preset):
    """The toolkit's TagSettings for a tags preset: its wordings, its thresholds and candidate
    order. Every other value is the toolkit's default, so a library upgrade that moves one fails
    tests/test_tags.py rather than changing tile texts unseen."""
    from logit_classifier.toolkit.comfyui import TagSettings
    from logit_classifier.toolkit.tags import PresenceThresholds

    return TagSettings(
        propose_instruction=preset.tile_tags_instruction,
        # The Captions node and a Preset built in code leave it "", and TagSettings refuses a
        # wording without {PROMPT}. An empty one is never asked.
        prompt_tags_instruction=preset.prompt_tags_instruction or captions.PROMPT_PLACEHOLDER,
        verify_statement=preset.tile_tags_verification_statement,
        verify_thresholds=PresenceThresholds(proposed=preset.tile_tags_verification_threshold,
                                             declared=preset.prompt_tags_verification_threshold),
        # This pack's tags bench judged tile texts in candidate order. The toolkit's score order is the
        # Logit Tagger's measurement, and the bench has not run it.
        order="candidate",
    )


def prompt_tags_question(preset, text):
    """prompt_tags_instruction with `text`, the prompt's English text, in place of its placeholder."""
    return preset.prompt_tags_instruction.replace(captions.PROMPT_PLACEHOLDER, text)


@dataclass(frozen=True)
class TileTrace:
    """One tile row's stages. `echo_reply` is the first propose reply when it named only
    category nouns and the fallback question's reply replaced it, else None. `origins` names
    where each candidate came from and `dropped` what the merge and the thing check left out and
    why. `scores` holds None per candidate when verify is off. `strips` holds per kept item its
    six strip probabilities in the toolkit's `strip_boxes` order, and `unplaced` the kept items no
    strip holds, which the text leaves out. With locate off every `strips` entry is (), every
    term "" and `unplaced` empty."""

    reply: str
    echo_reply: str | None
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


def trace_tile(clip, classifier, crop, image, preset, prompt_tags, thing_cache, locate=True):
    """Every stage of one tile row: the toolkit's tag_image on `image` (propose, merge with
    `prompt_tags`, the thing check with answers shared through `thing_cache`, verify and
    clean), then the toolkit's locate_tags on the full-resolution `crop` unless `locate` is
    off. An empty verification statement skips verify and locate, so every candidate is kept."""
    toolkit = captions.comfy_toolkit()
    statement = preset.tile_tags_verification_statement
    trace = toolkit.tag_image(clip, classifier, image, prompt_tags=prompt_tags, settings=tag_settings(preset),
                              thing_cache=thing_cache)
    kept = trace.kept
    strips = ((),) * len(kept)
    terms = ("",) * len(kept)
    unplaced = ()
    placed = tuple((item, "") for item in kept)

    # presence_statement raises on an empty statement, so an unverified tile is never located.
    if statement and locate and kept:
        places = toolkit.locate_tags(classifier, crop, kept, statement=statement,
                                     threshold=preset.tile_tags_position_threshold)
        strips, terms, unplaced, placed = places.strips, places.terms, places.unplaced, places.placed
    return TileTrace(reply=trace.reply, echo_reply=trace.echo_reply, proposed=trace.proposed,
                     candidates=trace.candidates,
                     origins=trace.origins, dropped=trace.dropped, scores=trace.scores,
                     verified=trace.verified, kept=kept, strips=strips, terms=terms, unplaced=unplaced,
                     text=toolkit.render_placed(placed))


@dataclass(frozen=True)
class TagRun:
    """Every stage of one tags pass. `style_texts` is the style line per batch row, "" for
    none, `prompt` is the prompt tags or None when the prompt is blank or
    `prompt_tags_instruction` is empty, and `tiles[tile_index][batch_row]` is a TileTrace.
    `prompt` is the toolkit's PromptTagTrace, which module scope cannot import."""

    style_texts: tuple
    prompt: object
    tiles: tuple


# The tags pass is greedy and its requests deterministic, so a seed re-roll that re-executes
# the node would otherwise pay every propose and verify again, as captions.generate_caption's
# cache explains for captions.
TAG_CACHE_ENTRIES = 512
_TAG_CACHE = captions.ClipBoundCache(TAG_CACHE_ENTRIES)


def clear_tag_cache():
    """Empty the tag cache. This is the one public way to reset it."""
    _TAG_CACHE.clear()


def _tag_cache_key(image, preset, scope):
    # The settings object holds every value a toolkit stage reads, so a changed value never
    # serves text written under the old one. An image of None keys a text-only request.
    digest = () if image is None else captions.picture_digest(image)
    return (digest, tag_settings(preset), VL_MAX_PIXELS, preset.tile_tags_position_threshold,
            preset.global_style_instruction, preset.prompt, scope)


def _cached(key, clip, compute):
    stored = _TAG_CACHE.get(key, clip)
    if stored is not None:
        return stored
    value = compute()
    _TAG_CACHE.put(key, clip, value)
    return value


def _prompt_trace(clip, classifier, preset, thing_cache):
    return captions.comfy_toolkit().tag_prompt(clip, classifier, preset.prompt, tag_settings(preset), thing_cache)


def generate_tag_trace(clip, source, tiles, preset, batch_size=1, batch_index=0, progress=None,
                       style_source=None, locate=True):
    """Every stage of the tags pass, as a TagRun.

    Its `style_texts` hold one style caption per batch row when `captions.style_row_count`
    counts style rows, else "" per row. `style_source` (default `source`) is what the style
    caption reads, and `progress` is the VL run's ledger or None. `locate` off skips the strip
    requests, so every tile text carries no position term."""
    import comfy.model_management
    import comfy.utils

    if preset.tile_text != captions.TILE_TEXT_TAGS:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): preset {preset.label!r} is the {preset.tile_text!r} "
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
    thing_cache = {}
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
                                   lambda: _prompt_trace(clip, classifier, preset, thing_cache))
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
                image = captions.resample_for_vl(row, VL_MAX_PIXELS)
                scope = ("tile", crop.x0, crop.y0, crop.x1, crop.y1, b, batch_index, locate)
                row_traces.append(_cached(_tag_cache_key(image, preset, scope), clip,
                                          lambda row=row, image=image: trace_tile(
                                              clip, classifier, row, image, preset, prompt_tags,
                                              thing_cache, locate)))
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
