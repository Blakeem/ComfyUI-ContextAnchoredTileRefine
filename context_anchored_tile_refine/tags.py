"""Per-tile tag text: the tile text a tags preset writes in place of a caption.

Two entry points run one pass. `generate_tag_trace` returns every stage's result as a
`TagRun`: the style line per picture row, the prompt fragment sort as a `PromptTrace` and a
`TileTrace` per tile row, so a test node can show what each stage did. `generate_tag_set` is
the engine's, and reads only the texts from that run, as `(style_texts, tile_texts)` in the
shape `captions.generate_caption_set` returns, so the encode stage reads either one. The stages:

    picture pass   the style caption (captions.generate_caption), which is the style line
                   alone, then one text-only subject or style choice per prompt fragment.
                   Style fragments are left out of the tile candidates and dropped.
    per tile row   propose (clip.generate over the crop), merge (subject fragments first,
                   normalized, category nouns and repeats dropped), verify (one noul per
                   candidate on the crop, kept at the preset's verification threshold), clean
                   (logit_classifier.tags.drop_subsets), locate (the verification statement on
                   three horizontal and three vertical strips of the full-resolution crop, a
                   tag present on no strip dropped), render ("<item> <term>"). An empty
                   verification statement skips verify and locate and keeps every candidate.

The pipeline is the Logit Tagger's (ComfyUI-LogitTagger/logit_tagger/tagging.py) run per
tile, and every constant below was measured there or in tests-AB/ab_tile_tags.py. Module
scope is torch and stdlib only. comfy and logit_classifier are imported inside functions, so
a missing library fails with its pip command (a subprocess test pins the comfy half).
"""
import hashlib
import importlib
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass

from . import captions
from .grid import Rect

# Core only resizes a VL image above 12.8 MP. The Logit Tagger's tests-AB/ab_tagger.py fit a
# 42 candidate verify pass beside a 1 MP picture, and tests-AB/ab_tile_tags.py ran at 1 MP.
VL_MAX_PIXELS = 1024 * 1024

# In the Logit Tagger's tests-AB/ab_propose_budget.py both images that reached 128 were
# repeating tags, and 192 added no tag on nine images.
PROPOSE_MAX_TOKENS = 128

# tests-AB/ab_prompt_fragments.py: style fragments scored 0.98 and up, mixed ones such as
# "the moon" 0.67 to 0.69, so 0.9 keeps the mixed ones as subjects.
STYLE_THRESHOLD = 0.9

# tests-AB/ab_tile_tags.py on market: the propose instruction's own nouns came back as tags
# and passed verify.
CATEGORY_NOUNS = frozenset({"objects", "people", "animals", "clothing", "materials", "setting"})

# tests-AB/ab_tile_position.py: strips at 0.25 MP placed items as precisely as 0.5 MP.
STRIP_MEGAPIXELS = 0.25

ROW_WORDS = ("top", "center", "bottom")
COLUMN_WORDS = ("left", "center", "right")

# Core's empty think block (thinking=False) made Qwen3-VL reply empty or refuse in the Logit
# Tagger's probe, so the template is written here. Text starting with <|im_start|> skips core's.
PROPOSE_TEMPLATE = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{instruction}"
    "<|im_end|>\n<|im_start|>assistant\n"
)

# The subject or style choice, worded as tests-AB/ab_prompt_fragments.py measured it.
FRAGMENT_QUESTION = 'What does the image prompt phrase "{fragment}" describe'
FRAGMENT_CRITERIA = {
    "subject": "a thing, person, animal, place or part of the scene that could be pointed at in the picture",
    "style": "the picture's medium, art style, quality, lighting, colour palette, camera, framing or mood",
}

# What the tags pass reads from the library. 0.2.0 lacks drop_unfinished_tag and keeps no
# initialism whole, and an older release lacks the tags module and ComfyClipBackend.
LIBRARY_VERSION = "0.2.1"
_LIBRARY_NAMES = {
    "logit_classifier": ("Classifier", "Config", "ChoiceQuestion", "NoulQuestion", "SystemOneRequest"),
    "logit_classifier.tags": ("split_prompt", "parse_candidates", "normalize_item", "drop_subsets",
                              "complete_tags", "repeated_block", "drop_unfinished_tag",
                              "MAX_CANDIDATES"),
    "logit_classifier.backends.comfy_clip": ("ComfyClipBackend",),
}
_LIBRARY_FIX = f'pip install -U "logit-classifier>={LIBRARY_VERSION}"'


def check_tags_ready(clip):
    """Raises before any generate when a tags preset cannot run: a CLIP without a text
    generator, or a logit_classifier missing or older than LIBRARY_VERSION. The library is judged by
    what it provides and never by importlib.metadata, since an editable install can report a
    stale version in its dist metadata."""
    if not callable(getattr(clip, "generate", None)) or not callable(getattr(clip, "decode", None)):
        raise RuntimeError(
            "Context-Anchored Tile Refine (VL): this CLIP cannot generate text, and a tags preset "
            "asks the VL model to list each tile's contents. Load a vision-language text encoder "
            "with a text-generation head (Krea 2 family), or use vlm_method 'vision tokens'.")
    for module_name, names in _LIBRARY_NAMES.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise RuntimeError(
                f"Context-Anchored Tile Refine (VL): a tags preset needs logit-classifier {LIBRARY_VERSION} "
                f"or newer and {module_name} cannot be imported ({error}). Install or upgrade it in "
                f"the ComfyUI Python environment with: {_LIBRARY_FIX}") from error
        missing = [name for name in names if not hasattr(module, name)]
        if missing:
            raise RuntimeError(
                f"Context-Anchored Tile Refine (VL): a tags preset needs logit-classifier {LIBRARY_VERSION} "
                f"or newer and {module_name} lacks {', '.join(missing)}. Upgrade it in the ComfyUI "
                f"Python environment with: {_LIBRARY_FIX}")


def build_classifier(clip):
    """One Classifier over the workflow's CLIP, with no prior drift and no calibration file."""
    from logit_classifier import Classifier, Config
    from logit_classifier.backends.comfy_clip import ComfyClipBackend

    backend = ComfyClipBackend(clip)
    if not backend.sees_images:
        raise RuntimeError(
            "Context-Anchored Tile Refine (VL): this CLIP was loaded without its vision tower, so "
            "a tags preset cannot read the tiles. Load the Qwen3-VL text encoder with the Load "
            "CLIP type set to krea2.")
    return Classifier(Config(use_prior_debias=False, calibration_path=None), backend)


def _nouls(classifier, picture, statements):
    # One packed request, so every statement shares one encode of the picture.
    from logit_classifier import NoulQuestion, SystemOneRequest

    if not statements:
        return []
    questions = {f"q{index}": NoulQuestion(instructions=statement)
                 for index, statement in enumerate(statements)}
    response, _diagnostics = classifier.classify(SystemOneRequest(state="", questions=questions),
                                                 image=picture)
    return [response.answers[qid].noul for qid in questions]


def _statements(items, statement):
    return [statement.replace(captions.TAG_PLACEHOLDER, item) for item in items]


def _verify_scores(classifier, picture, items, statement):
    return tuple(_nouls(classifier, picture, _statements(items, statement)))


def _passed(items, scores, threshold):
    return tuple(item for item, p in zip(items, scores, strict=True) if p >= threshold)


def fragment_style_p(classifier, fragments):
    """p(style) per fragment, from one text-only subject or style choice per fragment."""
    from logit_classifier import ChoiceQuestion, SystemOneRequest

    if not fragments:
        return ()
    questions = {f"f{index}": ChoiceQuestion(instructions=FRAGMENT_QUESTION.format(fragment=fragment),
                                             criteria=FRAGMENT_CRITERIA)
                 for index, fragment in enumerate(fragments)}
    response, _diagnostics = classifier.classify(SystemOneRequest(state="", questions=questions))
    return tuple(response.answers[qid].probabilities["style"] for qid in questions)


def sort_fragments(fragments, style_p):
    """(subject fragments, style fragments): a fragment is style at p(style) >= STYLE_THRESHOLD."""
    pairs = tuple(zip(fragments, style_p, strict=True))
    subjects = tuple(fragment for fragment, p in pairs if p < STYLE_THRESHOLD)
    styles = tuple(fragment for fragment, p in pairs if p >= STYLE_THRESHOLD)
    return subjects, styles


def _sampler(clip):
    # Core's generating transformer and its first stop token, or None for a CLIP without them.
    model = getattr(clip, "cond_stage_model", None)
    encoder_name = getattr(model, "clip", None)
    encoder = getattr(model, encoder_name, None) if isinstance(encoder_name, str) else None
    transformer = getattr(encoder, "transformer", None)
    config = getattr(getattr(transformer, "model", None), "config", None)
    stop_tokens = getattr(config, "stop_tokens", None)

    if not callable(getattr(transformer, "sample_token", None)) or not stop_tokens:
        return None
    return transformer, stop_tokens[0]


@contextmanager
def stop_on_repeat(clip):
    """Ends core's greedy decode on a stop token once the reply repeats a block of tags.

    Core's generate loop takes no stop condition, so the token sample_token returns is
    replaced. Core copies that token into the sequence and breaks on it
    (comfy/text_encoders/llama.py, the generate loop)."""
    from logit_classifier.tags import complete_tags, repeated_block

    sampler = _sampler(clip)
    history = []

    if sampler is None:
        yield
        return
    transformer, stop_id = sampler
    original = transformer.sample_token
    shadowed = "sample_token" in vars(transformer)
    previous = vars(transformer).get("sample_token")

    def watch(*args, **kwargs):
        token = original(*args, **kwargs)
        history.append(int(token.reshape(-1)[0]))
        if repeated_block(complete_tags(clip.decode(history))):
            return token.new_full(token.shape, stop_id)
        return token

    transformer.sample_token = watch
    try:
        yield
    finally:
        if shadowed:
            transformer.sample_token = previous
        else:
            del transformer.sample_token


def tile_tags_question(preset):
    """The tile tags question without the chat template, the filled
    tile_tags_with_prompt_instruction first when the preset carries a prompt."""
    if not preset.prompt:
        return preset.tile_tags_instruction
    prompt_text = preset.tile_tags_with_prompt_instruction.replace(captions.PROMPT_PLACEHOLDER, preset.prompt)
    return prompt_text + preset.tile_tags_instruction


def propose_text(preset):
    """The propose request as the tokenizer reads it."""
    return PROPOSE_TEMPLATE.format(instruction=tile_tags_question(preset))


def propose(clip, picture, preset):
    """The VL model's greedy list of what `picture` holds, as comma separated text."""
    from logit_classifier.tags import drop_unfinished_tag

    tokens, _tail = captions._tokenize_images(clip, propose_text(preset), picture)
    with captions.cuda_graphs_disabled(), stop_on_repeat(clip):
        ids = clip.generate(tokens, do_sample=False, max_length=PROPOSE_MAX_TOKENS)
    text = clip.decode(ids)
    # Core stops early only on a stop token, so a decode that fills the budget ends mid tag.
    if len(ids) >= PROPOSE_MAX_TOKENS:
        return drop_unfinished_tag(text)
    return text


def merge_trace(fragments, proposed):
    """(candidates, origins, dropped): the subject fragments, then the proposed tags,
    normalized, without the propose instruction's category nouns or a repeat, capped at the
    library's MAX_CANDIDATES. An origin is "prompt", "model", or "both" for a subject fragment the
    model also proposed, and `dropped` holds each left-out name with its reason."""
    from logit_classifier.tags import MAX_CANDIDATES, normalize_item

    index_of = {}
    origins = []
    dropped = []
    for origin, items in (("prompt", fragments), ("model", proposed)):
        for item in items:
            name = normalize_item(item)
            index = index_of.get(name)
            if not name:
                continue
            if name in CATEGORY_NOUNS:
                dropped.append((name, "category noun"))
            elif index is not None and origin == "model" and origins[index] != "model":
                origins[index] = "both"
            elif index is not None:
                dropped.append((name, "repeat"))
            elif len(index_of) == MAX_CANDIDATES:
                dropped.append((name, "over the cap"))
            else:
                index_of[name] = len(origins)
                origins.append(origin)
    return tuple(index_of), tuple(origins), tuple(dropped)


def merge_candidates(fragments, proposed):
    """The candidates `merge_trace` keeps, in merge order."""
    return list(merge_trace(fragments, proposed)[0])


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
        per_strip.append(_nouls(classifier, strip, statements))
    return tuple(tuple(p[index] for p in per_strip) for index in range(len(items)))


def strip_term(strips, threshold):
    """The position term of one item's six strip probabilities."""
    return position_term(axis_word(strips[:3], ROW_WORDS, threshold),
                         axis_word(strips[3:], COLUMN_WORDS, threshold))


def on_no_strip(strips, threshold):
    """Whether no strip holds the item. The whole tile's score can pass a tag that no part of
    the tile shows, and on the owner's storm sky tile every such tag named something the tile
    lacks."""
    return max(strips) < threshold


def render_tags(items, terms):
    """"<item> <term>" per item, or the item alone without a term, joined by ", "."""
    return ", ".join(f"{item} {term}" if term else item for item, term in zip(items, terms, strict=True))


@dataclass(frozen=True)
class PromptTrace:
    """The prompt fragment sort of one picture. `fragments` and `style_p` are the prompt's
    parts and their p(style). `subjects` join every tile's candidates first and `styles` are
    dropped."""

    fragments: tuple
    style_p: tuple
    subjects: tuple
    styles: tuple


@dataclass(frozen=True)
class TileTrace:
    """One tile row's stages. `origins` names where each candidate came from and `dropped`
    what the merge left out and why. `scores` holds None per candidate when verify is off.
    `strips` holds per kept item its six strip probabilities in `strip_rects` order, and
    `unplaced` the kept items no strip holds, which the text leaves out. With locate off every
    `strips` entry is (), every term "" and `unplaced` empty."""

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


def trace_tile(clip, classifier, crop, picture, preset, subjects, locate=True):
    """Every stage of one tile row: propose on `picture`, merge, verify, clean, then locate on
    the full-resolution `crop` unless `locate` is off. An empty verification statement skips
    verify and locate, so every candidate is kept."""
    from logit_classifier.tags import MAX_CANDIDATES, drop_subsets, parse_candidates

    statement = preset.tile_tags_verification_statement
    position_threshold = preset.tile_tags_position_threshold
    reply = propose(clip, picture, preset)
    proposed = tuple(parse_candidates(reply, max_candidates=MAX_CANDIDATES))
    candidates, origins, dropped = merge_trace(subjects, proposed)
    scores = (None,) * len(candidates)
    verified = candidates

    if statement:
        scores = _verify_scores(classifier, picture, candidates, statement)
        verified = _passed(candidates, scores, preset.tile_tags_verification_threshold)
    kept = tuple(drop_subsets(list(verified)))
    strips = ((),) * len(kept)
    terms = ("",) * len(kept)
    unplaced = ()

    if statement and locate and kept:
        strips = strip_scores(classifier, crop, kept, statement)
        terms = tuple(strip_term(p, position_threshold) for p in strips)
        unplaced = tuple(item for item, p in zip(kept, strips, strict=True)
                         if on_no_strip(p, position_threshold))
    placed = [(item, term) for item, term in zip(kept, terms, strict=True) if item not in unplaced]
    return TileTrace(reply=reply, proposed=proposed, candidates=candidates, origins=origins,
                     dropped=dropped, scores=scores, verified=verified, kept=kept, strips=strips,
                     terms=terms, unplaced=unplaced,
                     text=render_tags([item for item, _ in placed], [term for _, term in placed]))


@dataclass(frozen=True)
class TagRun:
    """Every stage of one tags pass. `style_texts` is the style line per batch row, "" for
    none, `prompt` is the fragment sort or None when the preset carries no prompt, and
    `tiles[tile_index][batch_row]` is a TileTrace."""

    style_texts: tuple
    prompt: PromptTrace | None
    tiles: tuple


def style_line(clip, row, preset, scope):
    """One picture row's style line: its cleaned style caption, read through
    captions.generate_caption and never the tag cache."""
    vl_input = captions.resample_for_vl(
        row, captions.caption_budget_pixels(preset.vision.caption_megapixels, row))
    return captions.clean_caption(captions.generate_caption(
        clip, vl_input, preset.style_instruction, preset.style_max_tokens, thinking=True, scope=scope))


# The tags pass is greedy and its requests deterministic, so a seed re-roll that re-executes
# the node would otherwise pay every propose and verify again, as captions.generate_caption's
# cache explains for captions.
TAG_CACHE_ENTRIES = 512
_TAG_CACHE = OrderedDict()


def clear_tag_cache():
    """Empty the tag cache. This is the one public way to reset it."""
    _TAG_CACHE.clear()


def _tuning():
    # Read per call, so a changed constant never serves text written under the old one.
    return (VL_MAX_PIXELS, PROPOSE_MAX_TOKENS, STYLE_THRESHOLD,
            tuple(sorted(CATEGORY_NOUNS)), PROPOSE_TEMPLATE, FRAGMENT_QUESTION,
            tuple(FRAGMENT_CRITERIA.items()), STRIP_MEGAPIXELS)


def _tag_cache_key(picture, preset, scope):
    # float32 because bfloat16 has no numpy dtype. A picture of None keys a text-only request.
    digest = ()
    if picture is not None:
        pixels = picture.contiguous().cpu().float().numpy().tobytes()
        digest = (hashlib.sha256(pixels).hexdigest(), str(picture.dtype), tuple(picture.shape))
    wording = (preset.tile_tags_instruction, preset.tile_tags_with_prompt_instruction,
               preset.tile_tags_verification_statement, preset.tile_tags_verification_threshold,
               preset.tile_tags_position_threshold, preset.style_instruction)
    return (digest, wording, preset.prompt, _tuning(), scope)


def _cached(key, clip, compute):
    # The weak reference keeps an unloaded CLIP from living on in the cache and keeps its text
    # from reaching a different model.
    entry = _TAG_CACHE.get(key)
    if entry is not None and entry[0]() is clip:
        _TAG_CACHE.move_to_end(key)
        return entry[1]
    value = compute()
    _TAG_CACHE[key] = (weakref.ref(clip), value)
    _TAG_CACHE.move_to_end(key)
    while len(_TAG_CACHE) > TAG_CACHE_ENTRIES:
        _TAG_CACHE.popitem(last=False)
    return value


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
    check_tags_ready(clip)
    # Imported after the guard, so a missing library fails with its pip command.
    from logit_classifier.tags import split_prompt

    batch = int(source.shape[0])
    style_rows = captions.style_row_count(preset, batch)
    per_picture = len(tiles) * batch + style_rows
    total = per_picture * batch_size
    pbar = None if progress is not None else comfy.utils.ProgressBar(total)
    done = per_picture * batch_index
    classifier = build_classifier(clip)
    fragments = tuple(split_prompt(preset.prompt))
    style_p = ()
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

    if fragments:
        style_p = _cached(_tag_cache_key(None, preset, ("fragments",)), clip,
                          lambda: fragment_style_p(classifier, fragments))
    subjects, styles = sort_fragments(fragments, style_p)
    if preset.prompt:
        prompt_trace = PromptTrace(fragments=fragments, style_p=tuple(style_p), subjects=subjects,
                                   styles=styles)

    if style_rows:
        style_canvas = source if style_source is None else style_source
        if int(style_canvas.shape[0]) != batch:
            raise RuntimeError(
                f"Context-Anchored Tile Refine (VL): {batch} batch row(s) to tag but the style "
                f"canvas has {int(style_canvas.shape[0])}. Every row needs its own style line or "
                "a row would carry another row's style.")
        comfy.model_management.throw_exception_if_processing_interrupted()
        for b in range(batch):
            style_texts[b] = style_line(clip, style_canvas[b:b + 1], preset, ("style", b, batch_index))
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
                                          clip, classifier, row, picture, preset, subjects, locate)))
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
    style_texts = list(run.style_texts)
    tile_texts = [[trace.text for trace in row_traces] for row_traces in run.tiles]

    if not any(style_texts):
        style_texts = []
    return style_texts, tile_texts
