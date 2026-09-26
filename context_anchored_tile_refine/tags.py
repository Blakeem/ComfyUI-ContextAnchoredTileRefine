"""Per-tile tag text: the tile text a tags preset writes in place of a caption.

Two entry points run one pass. `generate_tag_trace` returns every stage's result as a
`TagRun`: the style line per picture row, the prompt tags as a `PromptTrace` and a
`TileTrace` per tile row, so a test node can show what each stage did. `generate_tag_set` is
the engine's, and reads only the texts from that run, as `(style_texts, tile_texts)` in the
shape `captions.generate_caption_set` returns, so the encode stage reads either one. The stages:

    picture pass   the style caption (captions.style_caption), which is the style line
                   alone, then the prompt tags: one text-only generate lists the physical
                   things the prompt names, and the thing check keeps the ones that are
                   things. They join every tile's candidates. The prompt reaches no other
                   question, so a long prompt never lengthens a tile's reply.
    per tile row   propose (clip.generate over the crop, stopped after MAX_PROPOSED_TAGS
                   tags), merge (the model's tags first, then the prompt tags, normalized,
                   category nouns and repeats dropped), thing check, verify (one noul per
                   candidate on the crop, the model's tags at the verification threshold and a
                   prompt tag the model did not list at the prompt tags threshold), clean
                   (logit_classifier.tags.drop_subsets), locate (the verification statement on
                   three horizontal and three vertical strips of the full-resolution crop, a
                   tag present on no strip dropped), render ("<item> <term>"). An empty
                   verification statement skips verify and locate and keeps every candidate.

Every constant below was measured in the Logit Tagger's harnesses or in tests-AB, and the
2026-09-25 changes in tests-AB/ab_tags_bench.py (log: tests-AB/tags-bench-log.md). Module
scope is torch and stdlib only. comfy and logit_classifier are imported inside functions, so
a missing library fails with its pip command (a subprocess test pins the comfy half).
"""
import functools
import importlib
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass

from . import captions
from .grid import Rect

logger = logging.getLogger(__name__)

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

# Core's empty think block (thinking=False) made Qwen3-VL reply empty or refuse in the Logit
# Tagger's probe, so the template is written here. Text starting with <|im_start|> skips core's.
PROPOSE_TEMPLATE = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{instruction}"
    "<|im_end|>\n<|im_start|>assistant\n"
)
PROMPT_TAGS_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

# A 500 word prompt listed 28 to 45 tags in 70 to 256 tokens (tests-AB/tags-bench-log.md).
PROMPT_TAGS_MAX_TOKENS = 256

# Past the prompt's own things the model pads the list with words the prompt lacks
# ("materials", "details"), so two such tags in a row end it.
UNGROUNDED_STREAK = 2
_STOP_WORDS = frozenset({"a", "an", "the", "of", "with", "and", "in", "on", "at", "to", "for", "by", "from"})

# The thing check: one text-only choice per tag string. It drops whole-scene and setting words
# ("cityscape", "urban environment"), lighting, color and quality words, which no threshold on
# the verify score separates from real things. This wording counts landscape features, groups
# and light sources as things, which an earlier one dropped ("red moon", "army of soldiers").
THING_QUESTION = 'What does the image tag "{tag}" name'
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

# What the tags pass reads from the library. 0.2.0 lacks drop_unfinished_tag and keeps no
# initialism whole, and an older release lacks the tags module and ComfyClipBackend.
LIBRARY_VERSION = "0.2.1"
_LIBRARY_NAMES = {
    "logit_classifier": ("Classifier", "Config", "ChoiceQuestion", "NoulQuestion", "SystemOneRequest"),
    "logit_classifier.tags": ("parse_candidates", "normalize_item", "drop_subsets", "complete_tags",
                              "repeated_block", "drop_unfinished_tag"),
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


def _transformer(clip):
    # Core's generating transformer, or None for a CLIP without one.
    model = getattr(clip, "cond_stage_model", None)
    encoder_name = getattr(model, "clip", None)
    encoder = getattr(model, encoder_name, None) if isinstance(encoder_name, str) else None
    return getattr(encoder, "transformer", None)


def _resident(clip):
    # Whether the CLIP heads core's loaded list with its weights in place, the state its own
    # load_model call leaves it in. Another model's load always inserts at the head.
    try:
        import comfy.model_management as model_management

        patcher = clip.patcher
        head = model_management.current_loaded_models
        return (bool(head) and head[0].model is patcher
                and patcher.model.device == patcher.load_device
                and patcher.model.current_weight_patches_uuid == patcher.patches_uuid
                and patcher.model.model_loaded_weight_memory > 0)
    except (ImportError, AttributeError):
        return False


@contextmanager
def skip_resident_loads(clip):
    """clip.load_model returns at once while the CLIP is resident.

    Core's load_models_gpu has no fast path for a loaded model and costs about 0.1 s, and
    logit_classifier loads once per request, seven requests per tile. The skip belongs in
    logit_classifier's ComfyClipBackend, which should load once per pass."""
    original = clip.load_model
    shadowed = "load_model" in vars(clip)

    def load_model(*args, **kwargs):
        if _resident(clip):
            return clip.patcher
        return original(*args, **kwargs)

    clip.load_model = load_model
    try:
        yield
    finally:
        if shadowed:
            clip.load_model = original
        else:
            del clip.load_model


@functools.cache
def _warn_unshared_vision_encode():
    # A library rename would otherwise cost a tower pass per tile with nothing in the log.
    logger.warning(
        "Context-Anchored Tile Refine (VL): logit_classifier.backends._torch_window has no "
        "_determinism. The tags pass encodes each tile picture twice, once for propose and once "
        "for verify, which costs one extra vision tower pass per tile.")


@contextmanager
def shared_vision_encode(clip):
    """One vision tower pass per picture tensor for the duration, so the propose generate and
    the verify request, which read the same picture, encode it once.

    The encode runs inside logit_classifier's determinism window, where the verify request
    always ran it, so the verify scores keep their values. That window is a private helper of
    the library, and a library without it skips the sharing with one warning per session."""
    import torch

    transformer = _transformer(clip)
    original = getattr(transformer, "preprocess_embed", None)
    try:
        from logit_classifier.backends._torch_window import _determinism
    except ImportError:
        _determinism = None
        _warn_unshared_vision_encode()
    cache = {}

    if not callable(original) or _determinism is None:
        yield
        return

    def preprocess_embed(embed, device):
        data = embed.get("data")
        if embed.get("type") != "image" or not torch.is_tensor(data):
            return original(embed, device=device)
        # The tensor itself is held beside its result, so its id cannot be reused while cached.
        key = (id(data), tuple(data.shape), data.dtype, str(device))
        if key not in cache:
            with _determinism():
                cache[key] = (data, original(embed, device=device))
        return cache[key][1]

    transformer.preprocess_embed = preprocess_embed
    try:
        yield
    finally:
        del transformer.preprocess_embed


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


def _passed(items, origins, scores, preset):
    """The items whose score reaches their origin's threshold: a prompt tag the model did not
    list itself needs the prompt tags threshold, since the verify statement also passes a near
    name for what is there ("wooden carriage" for a cart)."""
    passed = []
    for item, origin, p in zip(items, origins, scores, strict=True):
        threshold = (preset.prompt_tags_verification_threshold if origin == "prompt"
                     else preset.tile_tags_verification_threshold)
        if p >= threshold:
            passed.append(item)
    return tuple(passed)


def thing_scores(classifier, tags, known):
    """p("other") of the thing check per tag, text-only, each distinct tag asked once per
    `known` dict, which carries the answers across a run's tiles."""
    from logit_classifier import ChoiceQuestion, SystemOneRequest

    unknown = tuple(dict.fromkeys(tag for tag in tags if tag not in known))
    if unknown:
        questions = {f"t{index}": ChoiceQuestion(instructions=THING_QUESTION.format(tag=tag),
                                                 criteria=THING_CRITERIA)
                     for index, tag in enumerate(unknown)}
        response, _diagnostics = classifier.classify(SystemOneRequest(state="", questions=questions))
        known.update((tag, response.answers[qid].probabilities["other"])
                     for tag, qid in zip(unknown, questions, strict=True))
    return tuple(known[tag] for tag in tags)


def is_thing(p_other):
    return p_other < THING_THRESHOLD


@contextmanager
def stop_tag_list(clip, should_stop):
    """Ends core's greedy decode on a stop token once `should_stop(complete tags so far)`.

    Core's generate loop takes no stop condition, so the token sample_token returns is
    replaced. Core copies that token into the sequence and breaks on it
    (comfy/text_encoders/llama.py, the generate loop)."""
    from logit_classifier.tags import complete_tags

    transformer = _transformer(clip)
    config = getattr(getattr(transformer, "model", None), "config", None)
    stop_tokens = getattr(config, "stop_tokens", None)
    history = []

    if not callable(getattr(transformer, "sample_token", None)) or not stop_tokens:
        yield
        return
    stop_id = stop_tokens[0]
    original = transformer.sample_token
    shadowed = "sample_token" in vars(transformer)
    previous = vars(transformer).get("sample_token")

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
        if shadowed:
            transformer.sample_token = previous
        else:
            del transformer.sample_token


def _full_or_repeating(tags):
    from logit_classifier.tags import repeated_block

    return len(tags) >= MAX_PROPOSED_TAGS or bool(repeated_block(tags))


def propose_text(preset):
    """The propose request as the tokenizer reads it."""
    return PROPOSE_TEMPLATE.format(instruction=preset.tile_tags_instruction)


def propose(clip, picture, preset):
    """The VL model's greedy list of what `picture` holds, as comma separated text."""
    from logit_classifier.tags import drop_unfinished_tag

    tokens, _tail = captions._tokenize_images(clip, propose_text(preset), picture)
    with stop_tag_list(clip, _full_or_repeating):
        ids = captions.clip_generate(clip, tokens, do_sample=False, max_length=PROPOSE_MAX_TOKENS)
    text = clip.decode(ids)
    # Core stops early only on a stop token, so a decode that fills the budget ends mid tag.
    if len(ids) >= PROPOSE_MAX_TOKENS:
        return drop_unfinished_tag(text)
    return text


def _word_forms(text):
    # Each lowercase word mapped to itself plus every plural ending cut. One cut per word would
    # stem "trees" to "tre" and leave "tree" whole, so the two sides match through their forms.
    forms = {}
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        candidates = {word}
        for suffix, ending in (("s", ""), ("es", ""), ("ies", "y")):
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                candidates.add(word[:-len(suffix)] + ending)
        forms[word] = candidates
    return forms


def prompt_forms(prompt):
    """Every form of every prompt word, the set `grounded` matches a tag against."""
    return set().union(*_word_forms(prompt).values())


def grounded(tag, prompt_words):
    """Whether every content word of `tag` shares a form with a word of the prompt."""
    content = [forms for word, forms in _word_forms(tag).items() if word not in _STOP_WORDS]
    return bool(content) and all(forms & prompt_words for forms in content)


def prompt_tags_question(preset):
    """prompt_tags_instruction with the prompt in place of its placeholder."""
    return preset.prompt_tags_instruction.replace(captions.PROMPT_PLACEHOLDER, preset.prompt)


def prompt_tags_text(preset):
    """The prompt tags request as the tokenizer reads it."""
    return PROMPT_TAGS_TEMPLATE.format(instruction=prompt_tags_question(preset))


def list_prompt_tags(clip, preset):
    """(reply, listed): the VL model's greedy list of the things the prompt names, and its tags
    whose every word is a word of the prompt."""
    from logit_classifier.tags import drop_unfinished_tag, parse_candidates, repeated_block

    prompt_words = prompt_forms(preset.prompt)

    def ungrounded_or_repeating(tags):
        streak = 0
        for tag in reversed(tags):
            if grounded(tag, prompt_words):
                break
            streak += 1
        return streak >= UNGROUNDED_STREAK or bool(repeated_block(tags))

    with stop_tag_list(clip, ungrounded_or_repeating):
        ids = captions.clip_generate(clip, clip.tokenize(prompt_tags_text(preset)), do_sample=False,
                                     max_length=PROMPT_TAGS_MAX_TOKENS)
    reply = clip.decode(ids)
    text = drop_unfinished_tag(reply) if len(ids) >= PROMPT_TAGS_MAX_TOKENS else reply
    return reply, tuple(tag for tag in parse_candidates(text) if grounded(tag, prompt_words))


def merge_trace(proposed, prompt_tags):
    """(candidates, origins, dropped): the proposed tags, then the prompt tags, normalized,
    without the propose instruction's category nouns or a repeat, capped at MAX_MERGED_TAGS. An
    origin is "model", "prompt", or "both" for a prompt tag the model also proposed, and
    `dropped` holds each left-out name with its reason."""
    from logit_classifier.tags import normalize_item

    index_of = {}
    origins = []
    dropped = []
    for origin, items in (("model", proposed), ("prompt", prompt_tags)):
        for item in items:
            name = normalize_item(item)
            index = index_of.get(name)
            if not name:
                continue
            if name in CATEGORY_NOUNS:
                dropped.append((name, "category noun"))
            elif index is not None and origin == "prompt" and origins[index] == "model":
                origins[index] = "both"
            elif index is not None:
                dropped.append((name, "repeat"))
            elif len(index_of) == MAX_MERGED_TAGS:
                dropped.append((name, "over the cap"))
            else:
                index_of[name] = len(origins)
                origins.append(origin)
    return tuple(index_of), tuple(origins), tuple(dropped)


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
    """Whether no strip holds the item. The entire tile's score can pass a tag that no part of
    the tile shows, and on the owner's storm sky tile every such tag named something the tile
    lacks."""
    return max(strips) < threshold


def render_tags(items, terms):
    """"<item> <term>" per item, or the item alone without a term, joined by ", "."""
    return ", ".join(f"{item} {term}" if term else item for item, term in zip(items, terms, strict=True))


@dataclass(frozen=True)
class PromptTrace:
    """The prompt tags of one picture. `reply` is the VL model's list as written, `listed` its
    tags whose every word is a word of the prompt, `p_other` each listed tag's thing check
    score, and `tags` the listed tags the thing check keeps, which join every tile's
    candidates."""

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
    """Every stage of one tile row: propose on `picture`, merge with `prompt_tags`, the thing
    check (answers shared through `known_things`), verify, clean, then locate on the
    full-resolution `crop` unless `locate` is off. An empty verification statement skips verify
    and locate, so every candidate is kept."""
    from logit_classifier.tags import drop_subsets, parse_candidates

    statement = preset.tile_tags_verification_statement
    position_threshold = preset.tile_tags_position_threshold

    with shared_vision_encode(clip):
        reply = propose(clip, picture, preset)
        proposed = tuple(parse_candidates(reply, max_candidates=MAX_PROPOSED_TAGS))
        merged, merged_origins, dropped = merge_trace(proposed, prompt_tags)
        p_other = thing_scores(classifier, merged, known_things)
        candidates = tuple(c for c, p in zip(merged, p_other, strict=True) if is_thing(p))
        origins = tuple(o for o, p in zip(merged_origins, p_other, strict=True) if is_thing(p))
        dropped += tuple((c, "not a thing") for c, p in zip(merged, p_other, strict=True) if not is_thing(p))
        scores = (None,) * len(candidates)
        verified = candidates
        if statement:
            scores = _verify_scores(classifier, picture, candidates, statement)
            verified = _passed(candidates, origins, scores, preset)
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


def _tuning():
    # Read per call, so a changed constant never serves text written under the old one.
    return (VL_MAX_PIXELS, PROPOSE_MAX_TOKENS, MAX_PROPOSED_TAGS, MAX_MERGED_TAGS,
            tuple(sorted(CATEGORY_NOUNS)), PROPOSE_TEMPLATE, PROMPT_TAGS_TEMPLATE,
            PROMPT_TAGS_MAX_TOKENS, UNGROUNDED_STREAK, THING_QUESTION,
            tuple(THING_CRITERIA.items()), THING_THRESHOLD, STRIP_MEGAPIXELS)


def _tag_cache_key(picture, preset, scope):
    # A picture of None keys a text-only request.
    digest = () if picture is None else captions.picture_digest(picture)
    wording = (preset.tile_tags_instruction, preset.prompt_tags_instruction,
               preset.tile_tags_verification_statement, preset.tile_tags_verification_threshold,
               preset.prompt_tags_verification_threshold, preset.tile_tags_position_threshold,
               preset.style_instruction)
    return (digest, wording, preset.prompt, _tuning(), scope)


def _cached(key, clip, compute):
    stored = _TAG_CACHE.get(key, clip)
    if stored is not None:
        return stored
    value = compute()
    _TAG_CACHE.put(key, clip, value)
    return value


def _prompt_trace(clip, classifier, preset, known_things):
    reply, listed = list_prompt_tags(clip, preset)
    p_other = thing_scores(classifier, listed, known_things)
    return PromptTrace(reply=reply, listed=listed, p_other=p_other,
                       tags=tuple(tag for tag, p in zip(listed, p_other, strict=True) if is_thing(p)))


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

    batch = int(source.shape[0])
    style_rows = captions.style_row_count(preset, batch)
    per_picture = len(tiles) * batch + style_rows
    total = per_picture * batch_size
    pbar = None if progress is not None else comfy.utils.ProgressBar(total)
    done = per_picture * batch_index
    classifier = build_classifier(clip)
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

    with skip_resident_loads(clip):
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
