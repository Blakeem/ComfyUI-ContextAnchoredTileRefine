"""tags.py: the tile text a tags preset writes, from the prompt tags, propose, merge, the thing
check, verify, clean and render.

A duck-typed CLIP stands in for the VL text encoder and a scripted classifier, handed out by the
toolkit's comfy_classifier (see `open_toolkit`), answers every noul and thing check.
logit_classifier itself is the real library (its toolkit stages, tags helpers and question
types), so no comfy install and no model are needed.
"""
import contextlib
import dataclasses
import importlib.metadata
import re
import sys
import types

import pytest
import torch
from test_vl import Tile

from context_anchored_tile_refine import captions, tags
from context_anchored_tile_refine.grid import Rect

TILE_A = Rect(0, 0, 16, 16)
TILE_B = Rect(16, 0, 32, 16)
PROMPT_TAGS = "List the things this prompt names: {PROMPT}"
PROPOSE = "Tag this image."
VERIFY = "This image visibly contains {TAG}"
# Unequal, unlike the shipped pair, so one score can keep a model tag and drop a prompt-only one.
TILE_THRESHOLD = 0.999
PROMPT_ONLY_THRESHOLD = 0.9999


def a_tags_preset(prompt="", style="", style_tokens=128, **thresholds):
    thresholds = {"tile_tags_verification_threshold": TILE_THRESHOLD,
                  "prompt_tags_verification_threshold": PROMPT_ONLY_THRESHOLD, **thresholds}
    return captions.Preset(
        surface=captions.VLM_METHOD_CAPTIONS, label="tags",
        vision=captions.VisionSettings(canvas_tokens=1, crop_tokens=0,
                                       caption_megapixels=captions.VL_INPUT_BUDGET_MEGAPIXELS),
        style_instruction=style, style_max_tokens=style_tokens, kind=captions.TILE_TEXT_TAGS,
        tile_tags_instruction=PROPOSE, prompt_tags_instruction=PROMPT_TAGS,
        tile_tags_verification_statement=VERIFY, prompt=prompt, **thresholds)


def statement(item):
    return VERIFY.replace("{TAG}", item)


def chat_text(user, image=False):
    """The chat text of one user turn, byte for byte what the pass sent before the toolkit
    wrote it."""
    slot = "<|vision_start|><|image_pad|><|vision_end|>" if image else ""
    return f"<|im_start|>user\n{slot}{user}<|im_end|>\n<|im_start|>assistant\n"


PROPOSE_TEXT = chat_text(PROPOSE, image=True)


def open_toolkit(monkeypatch, fake):
    """Let a duck-typed CLIP through the toolkit, whose comfy_classifier and
    shared_vision_encode read core's model parts: the first hands out `fake` and the second
    passes through."""
    import logit_classifier.toolkit.comfyui as toolkit
    import logit_classifier.toolkit.comfyui.tagger as tagger

    monkeypatch.setattr(toolkit, "comfy_classifier", lambda clip, **kwargs: fake)
    monkeypatch.setattr(tagger, "shared_vision_encode", lambda clip: contextlib.nullcontext())


class FakeTagClip:
    """Duck-typed VL clip. A chat-templated request with a picture is a propose and answers
    with `proposal`, or `fallback_proposal` when set and asked the fallback question. One
    without a picture is the translate request when it holds the translate instruction and
    answers with `translation`, else the prompt tags request and answers with
    `prompt_tags_reply`. Anything else is the style caption and answers with `style_answer`.
    generate returns `proposal_tokens` ids for a propose, which is what the budget check reads."""

    def __init__(self, proposal="red apple, wooden table", style_answer="<think>hm</think>Oil painting.",
                 proposal_tokens=5, image_tokens=True, rejects_images=False, prompt_tags_reply="",
                 translation="", fallback_proposal=None):
        self.proposal = proposal
        self.fallback_proposal = fallback_proposal
        self.style_answer = style_answer
        self.prompt_tags_reply = prompt_tags_reply
        self.translation = translation
        self.proposal_tokens = proposal_tokens
        self.image_tokens = image_tokens
        self.rejects_images = rejects_images
        self.tokenize_calls = []
        self.generate_calls = []
        self._answers = {}

    def load_model(self, *args, **kwargs):
        return None

    def tokenize(self, text, images=None, **kwargs):
        if self.rejects_images and images is not None:
            raise TypeError("tokenize() got an unexpected keyword argument 'images'")
        image = None if images is None else images[0]
        self.tokenize_calls.append({"text": text, "image": image, **kwargs})
        stream = [(10, 1.0)] * 3
        if image is not None and self.image_tokens:
            stream += [(151652, 1.0), ({"type": "image"}, 1.0), (151653, 1.0)]
        stream += [(20, 1.0)] * 4
        return {"qwen3vl_4b": [stream], "_probe": (text, image)}

    def generate(self, tokens, **kwargs):
        text, image = tokens["_probe"]
        kind = "style"
        if text.startswith("<|im_start|>"):
            kind = "propose" if image is not None else "prompt tags"
        if kind == "prompt tags" and translate_instruction_start() in text:
            kind = "translate"
        answer = {"propose": self.proposal, "prompt tags": self.prompt_tags_reply, "translate": self.translation,
                  "style": self.style_answer}[kind]
        if kind == "propose" and self.fallback_proposal is not None and text == fallback_text():
            answer = self.fallback_proposal
        self.generate_calls.append({"text": text, "image": image, "kind": kind,
                                    "propose": kind == "propose", **kwargs})
        handle = len(self.generate_calls)
        self._answers[handle] = answer
        return [handle] * (self.proposal_tokens if kind == "propose" else 3)

    def decode(self, token_ids, skip_special_tokens=True):
        return self._answers[token_ids[0]]


def toolkit_defaults():
    from logit_classifier.toolkit.comfyui import TagSettings

    return TagSettings()


def language_question():
    return toolkit_defaults().language_question


def translate_text(prompt):
    return chat_text(toolkit_defaults().translate_instruction.replace("{PROMPT}", prompt))


def translate_instruction_start():
    return toolkit_defaults().translate_instruction.split("{PROMPT}")[0]


def fallback_text():
    return chat_text(toolkit_defaults().echo_fallback_instruction, image=True)


class FakeClassifier:
    """Scripted classifier: `noul(statement)` answers every noul and `other(tag)` the p("other")
    of every thing check, 0.0 (a thing) by default. The language question gets `english`, so a
    prompt reads as English unless a test says otherwise. `noul_at(statement, picture)`, when
    set, answers the other nouls in place of `noul`, so a test can script each strip. Every
    request is recorded with its statements and its picture. `nouls` is the library's own, so
    the statements are packed into requests as a real classifier packs them."""

    def __init__(self, noul=None, other=None, english=1.0):
        # 0.9995 passes a_tags_preset's verify and position thresholds and misses its prompt-only one.
        self.noul = noul if noul is not None else (lambda text: 0.9995)
        self.other = other if other is not None else (lambda tag: 0.0)
        self.english = english
        self.noul_at = None
        self.requests = []

    def nouls(self, statements, *, image=None, state=""):
        from logit_classifier import Classifier

        return Classifier.nouls(self, statements, image=image, state=state)

    def classify(self, request, image=None):
        from logit_classifier import ChoiceAnswer, NoulAnswer

        answers = {}
        texts = []
        kinds = set()
        for qid, question in request.questions.items():
            texts.append(question.instructions)
            kinds.add(question.type)
            if question.type == "choice":
                tag = re.search(r'"(.*)"', question.instructions).group(1)
                p = self.other(tag)
                answers[qid] = ChoiceAnswer(choice="other" if p >= 0.5 else "thing", confidence=max(p, 1.0 - p),
                                            probabilities={"thing": 1.0 - p, "other": p})
            elif question.instructions == language_question():
                answers[qid] = NoulAnswer(noul=self.english)
            elif self.noul_at is not None:
                answers[qid] = NoulAnswer(noul=self.noul_at(question.instructions, image))
            else:
                answers[qid] = NoulAnswer(noul=self.noul(question.instructions))
        self.requests.append({"kind": kinds.pop() if len(kinds) == 1 else kinds, "texts": texts,
                              "image": image, "state": request.state})
        return types.SimpleNamespace(answers=answers), None


@pytest.fixture
def classifier(comfy_stubs, monkeypatch):
    fake = FakeClassifier()
    open_toolkit(monkeypatch, fake)
    return fake


def run(clip, preset, tiles=(TILE_A,), source=None, **kwargs):
    source = torch.rand(1, 16, 32, 3) if source is None else source
    return tags.generate_tag_set(clip, source, [Tile(rect) for rect in tiles], preset, **kwargs)


def propose_calls(clip):
    return [call for call in clip.generate_calls if call["propose"]]


def calls_of(clip, kind):
    return [call for call in clip.generate_calls if call["kind"] == kind]


def nouls(classifier):
    return [request for request in classifier.requests if request["kind"] == "noul"]


def thing_checks(classifier):
    return [request for request in classifier.requests if request["kind"] == "choice"]


def is_strip(request):
    # The resample stub returns the size it was asked for, so a strip is known by its area.
    budget = tags.STRIP_MEGAPIXELS * 1_000_000
    height, width = request["image"].shape[1], request["image"].shape[2]
    return abs(height * width - budget) / budget < 0.01


def is_language(request):
    return request["texts"] == [language_question()]


def language_requests(classifier):
    return [request for request in nouls(classifier) if is_language(request)]


def strip_requests(classifier):
    return [request for request in nouls(classifier) if not is_language(request) and is_strip(request)]


def verify_requests(classifier):
    return [request for request in nouls(classifier) if not is_language(request) and not is_strip(request)]


# --- the result shape and the render stage ---------------------------------------------

def test_every_tile_row_gets_its_kept_items_joined_and_no_style_without_a_style_source(classifier):
    style_texts, tile_texts = run(FakeTagClip(), a_tags_preset(), tiles=(TILE_A, TILE_B))

    assert style_texts == []
    assert tile_texts == [["red apple, wooden table"], ["red apple, wooden table"]]


def test_batch_rows_are_tagged_one_picture_each(classifier):
    clip = FakeTagClip()

    _style, tile_texts = run(clip, a_tags_preset(), source=torch.rand(2, 16, 32, 3))

    assert tile_texts == [["red apple, wooden table", "red apple, wooden table"]]
    assert [int(call["image"].shape[0]) for call in propose_calls(clip)] == [1, 1]


def test_a_tile_with_nothing_kept_gets_an_empty_text(classifier):
    classifier.noul = lambda text: 0.1

    _style, tile_texts = run(FakeTagClip(), a_tags_preset())

    assert tile_texts == [[""]]


def test_a_tile_with_nothing_proposed_gets_an_empty_text_and_no_verify_request(classifier):
    _style, tile_texts = run(FakeTagClip(proposal="objects, people"), a_tags_preset())

    assert tile_texts == [[""]]
    assert classifier.requests == []


# --- the propose stage -----------------------------------------------------------------

def test_propose_decodes_greedily_at_the_budget_on_a_1mp_copy_of_the_crop(classifier, comfy_stubs):
    clip = FakeTagClip()

    run(clip, a_tags_preset())

    call = propose_calls(clip)[0]
    assert call["do_sample"] is False
    assert call["max_length"] == tags.PROPOSE_MAX_TOKENS == 128
    assert "thinking" not in clip.tokenize_calls[0]
    _shape, width, height, method, _crop = comfy_stubs["common_upscale_calls"][0]
    assert method == "area"
    assert abs(width * height - tags.VL_MAX_PIXELS) / tags.VL_MAX_PIXELS < 0.01
    # The verify stage reads the same picture the propose stage read.
    assert nouls(classifier)[0]["image"] is call["image"]


def test_propose_drops_the_decode_graphs_after_its_generate(classifier, monkeypatch):
    cleanups = []
    prefetch = types.ModuleType("comfy.model_prefetch")
    prefetch.cleanup_prefetch_queues = lambda: cleanups.append(len(clip.generate_calls))
    monkeypatch.setitem(sys.modules, "comfy.model_prefetch", prefetch)
    clip = FakeTagClip()
    run(clip, a_tags_preset())

    assert any(call["propose"] for call in clip.generate_calls)
    assert cleanups == list(range(1, len(clip.generate_calls) + 1))


def test_the_prompt_is_asked_once_per_picture_text_only_and_never_in_the_propose_question(classifier):
    without = FakeTagClip()
    with_prompt = FakeTagClip(prompt_tags_reply="cat, mat")

    run(without, a_tags_preset(), tiles=(TILE_A, TILE_B))
    run(with_prompt, a_tags_preset(prompt="a cat {on} a mat"), tiles=(TILE_A, TILE_B))

    assert [call["text"] for call in propose_calls(without)] == [PROPOSE_TEXT] * 2
    assert [call["text"] for call in propose_calls(with_prompt)] == [PROPOSE_TEXT] * 2
    assert calls_of(without, "prompt tags") == []
    (asked,) = calls_of(with_prompt, "prompt tags")
    assert asked["text"] == chat_text("List the things this prompt names: a cat {on} a mat")
    assert asked["image"] is None
    assert asked["do_sample"] is False
    assert asked["max_length"] == tags.PROMPT_TAGS_MAX_TOKENS == 256
    assert "thinking" not in with_prompt.tokenize_calls[0]
    assert language_requests(classifier) == [
        {"kind": "noul", "texts": [language_question()], "image": None, "state": "a cat {on} a mat"}]
    assert calls_of(with_prompt, "translate") == []


def test_a_prompt_the_language_question_scores_below_the_threshold_is_listed_from_its_translation(classifier):
    classifier.english = 0.1
    translation = "a moon above a lantern"
    clip = FakeTagClip(proposal="red apple", translation=translation, prompt_tags_reply="moon, lantern")
    preset = a_tags_preset(prompt="une lune au-dessus d'une lanterne")

    tag_run = run_trace(clip, preset)

    (translate,) = calls_of(clip, "translate")
    assert translate["text"] == translate_text(preset.prompt)
    assert translate["image"] is None
    assert translate["max_length"] == toolkit_defaults().translate_max_tokens
    (asked,) = calls_of(clip, "prompt tags")
    assert asked["text"] == chat_text(tags.prompt_tags_question(preset, translation))
    # Both tags are grounded in the translation's words, which the prompt lacks.
    assert tag_run.prompt == tags.PromptTrace(text=translation, reply="moon, lantern", listed=("moon", "lantern"),
                                              p_other=(0.0, 0.0), tags=("moon", "lantern"))


def test_the_pass_restores_the_clips_own_load_model(classifier):
    clip = FakeTagClip()

    run(clip, a_tags_preset(prompt="an apple"))

    assert "load_model" not in vars(clip)


def test_a_decode_that_fills_the_budget_drops_its_cut_tail(classifier):
    clip = FakeTagClip(proposal="red apple, wooden table, gre", proposal_tokens=tags.PROPOSE_MAX_TOKENS)

    _style, tile_texts = run(clip, a_tags_preset())

    assert nouls(classifier)[0]["texts"] == [statement("red apple"), statement("wooden table")]
    assert tile_texts == [["red apple, wooden table"]]


STOP_ID = 99


class ScriptedTransformer:
    """Core's generating transformer as the stop reads it: stop tokens on its config, and a
    sample_token that returns the next scripted id."""

    def __init__(self, ids):
        self.model = types.SimpleNamespace(config=types.SimpleNamespace(stop_tokens=[STOP_ID]))
        self.feed = iter(ids)

    def sample_token(self, *args, **kwargs):
        return torch.tensor([next(self.feed)])


class GeneratingClip:
    """A clip whose generate runs core's loop shape over a ScriptedTransformer: one
    sample_token per step, the stop token copied in and the loop broken on it. Token id n
    decodes to `words[n]` and the stop token to nothing."""

    def __init__(self, words, ids):
        self.words = words
        self.transformer = ScriptedTransformer(ids)
        self.cond_stage_model = types.SimpleNamespace(
            clip="qwen", qwen=types.SimpleNamespace(transformer=self.transformer))
        self.tokenize_calls = []
        self.generate_calls = []

    def tokenize(self, text, **kwargs):
        self.tokenize_calls.append(text)
        return {"qwen3vl_4b": [[(10, 1.0)]]}

    def generate(self, tokens, **kwargs):
        self.generate_calls.append(kwargs)
        ids = []
        while len(ids) < kwargs["max_length"]:
            ids.append(int(self.transformer.sample_token()))
            if ids[-1] == STOP_ID:
                break
        return ids

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.words.get(i, "") for i in ids)


@pytest.mark.parametrize("clip_options", [{"rejects_images": True}, {"image_tokens": False}])
def test_a_tokenizer_that_reads_no_image_fails_at_the_first_propose_naming_the_fix(classifier, clip_options):
    from logit_classifier import VisionUnsupportedError

    clip = FakeTagClip(**clip_options)

    with pytest.raises(VisionUnsupportedError, match=r"tokenizer does not read images.*Qwen3-VL text encoder"):
        run(clip, a_tags_preset())
    assert clip.generate_calls == []


# --- the merge, verify and clean stages -------------------------------------------------

def test_the_merge_drops_category_nouns_articles_and_repeats(classifier):
    clip = FakeTagClip(proposal="objects, people, a red apple, Red Apple, setting, the table, materials")

    run(clip, a_tags_preset())

    assert nouls(classifier)[0]["texts"] == [statement("red apple"), statement("table")]


def test_the_verify_keeps_items_at_the_threshold_and_asks_one_packed_request_per_tile_row(classifier):
    classifier.noul = lambda text: {statement("red apple"): 0.999, statement("wooden table"): 0.9989}[text]

    _style, tile_texts = run(FakeTagClip(), a_tags_preset(), tiles=(TILE_A, TILE_B))

    assert a_tags_preset().tile_tags_verification_threshold == 0.999
    assert tile_texts == [["red apple"], ["red apple"]]
    assert len(verify_requests(classifier)) == 2


def test_the_verify_keeps_items_at_the_presets_own_threshold(classifier):
    classifier.noul = lambda text: {statement("red apple"): 0.5, statement("wooden table"): 0.49}[text]

    _style, tile_texts = run(FakeTagClip(), a_tags_preset(tile_tags_verification_threshold=0.5,
                                                          tile_tags_position_threshold=0.5))

    assert tile_texts == [["red apple"]]


def test_the_clean_stage_drops_a_kept_subset_after_verify(classifier):
    clip = FakeTagClip(proposal="herbs, hanging dried herbs, jar")

    _style, tile_texts = run(clip, a_tags_preset())

    assert tile_texts == [["hanging dried herbs, jar"]]


def test_a_subset_survives_when_its_superset_fails_verify(classifier):
    classifier.noul = lambda text: 0.1 if text == statement("hanging dried herbs") else 0.9995
    clip = FakeTagClip(proposal="herbs, hanging dried herbs, jar")

    _style, tile_texts = run(clip, a_tags_preset())

    assert tile_texts == [["herbs, jar"]]


def test_the_proposed_tags_are_capped_at_25_and_the_prompt_tags_follow_them(classifier):
    proposal = ", ".join(f"thing {n}" for n in range(60))
    clip = FakeTagClip(proposal=proposal, prompt_tags_reply="lantern, moon")

    run(clip, a_tags_preset(prompt="a lantern, the moon"))

    texts = verify_requests(classifier)[-1]["texts"]
    assert texts == [statement(f"thing {n}") for n in range(tags.MAX_PROPOSED_TAGS)] + [
        statement("lantern"), statement("moon")]


def test_an_initialism_in_the_prompt_tags_reaches_the_candidates_whole(classifier):
    clip = FakeTagClip(proposal="", prompt_tags_reply="flag of the U.S.A., Washington D.C.")

    run(clip, a_tags_preset(prompt="a flag of the U.S.A., Washington D.C. at night"))

    assert verify_requests(classifier)[-1]["texts"] == [
        statement("flag of the u.s.a."), statement("washington d.c.")]


def test_a_prompt_tag_the_model_did_not_list_needs_the_prompt_threshold(classifier):
    scores = {"red apple": 0.999, "lantern": 0.9995, "moon": 0.9999, "boat": 0.9998}
    classifier.noul = lambda text: scores[text.removeprefix(statement(""))]
    clip = FakeTagClip(proposal="red apple, lantern", prompt_tags_reply="lantern, moon, boat")
    prompt = "a lantern, the moon and a boat"

    held = run_trace(clip, a_tags_preset(prompt=prompt), locate=False).tiles[0][0]
    lowered = run_trace(clip, a_tags_preset(prompt=prompt, prompt_tags_verification_threshold=0.5),
                        locate=False).tiles[0][0]

    assert held.origins == ("model", "both", "prompt", "prompt")
    assert a_tags_preset().prompt_tags_verification_threshold == 0.9999
    # "lantern" is listed by both, so it is held to the tile threshold, not the prompt one.
    assert held.verified == ("red apple", "lantern", "moon")
    assert lowered.verified == ("red apple", "lantern", "moon", "boat")


# --- the locate and render stages -------------------------------------------------------

CROP_SIZE = 16
THIRDS = [CROP_SIZE * i // 3 for i in range(4)]
SPANS = [(THIRDS[i], THIRDS[i + 1]) for i in range(3)]


def coordinate_source():
    # Channel 0 holds each pixel's y and channel 1 its x, so a strip names itself.
    shape = (1, CROP_SIZE, 2 * CROP_SIZE, 1)
    ys = torch.arange(CROP_SIZE).view(1, CROP_SIZE, 1, 1).expand(shape)
    xs = torch.arange(2 * CROP_SIZE).view(1, 1, 2 * CROP_SIZE, 1).expand(shape)
    return torch.cat([ys, xs, torch.zeros(shape)], dim=-1).float()


def strip_of(picture):
    """("whole", None) for the verify picture, else ("row" or "column", third index)."""
    ys = (int(picture[0, 0, 0, 0]), int(picture[0, -1, 0, 0]) + 1)
    xs = (int(picture[0, 0, 0, 1]), int(picture[0, 0, -1, 1]) + 1)
    if ys == xs == (0, CROP_SIZE):
        return "whole", None
    if xs == (0, CROP_SIZE):
        return "row", SPANS.index(ys)
    return "column", SPANS.index(xs)


def scripted_strips(presence, whole=None):
    """noul_at answering each item's ((top, center, bottom), (left, center, right)) presence,
    and `whole[item]` (default 0.9995) on the verify picture."""
    def noul_at(text, picture):
        item = text.removeprefix(statement(""))
        kind, index = strip_of(picture)
        if kind == "whole":
            return (whole or {}).get(item, 0.9995)
        return presence[item][0 if kind == "row" else 1][index]
    return noul_at


def test_the_strips_are_the_crop_thirds_with_floor_boundaries_on_an_odd_sized_crop():
    assert tags.strip_rects(17, 25) == (
        Rect(0, 0, 25, 5), Rect(0, 5, 25, 11), Rect(0, 11, 25, 17),
        Rect(0, 0, 8, 17), Rect(8, 0, 16, 17), Rect(16, 0, 25, 17))


@pytest.mark.parametrize(("probabilities", "word"), [
    ((0.1, 0.2, 0.89), None),
    ((0.9, 0.9, 0.9), None),
    ((0.9, 0.1, 0.9), None),
    # Two adjacent strips name nothing: the storm tile's masts stood in the center and right
    # columns and the weighted mean named the center.
    ((0.9, 1.0, 0.1), None),
    ((0.1, 1.0, 0.9), None),
    ((0.9, 0.89, 0.1), "top"),
    ((0.1, 0.9, 0.1), "center"),
    ((0.1, 0.1, 0.95), "bottom"),
])
def test_an_axis_names_the_one_strip_that_holds_the_item(probabilities, word):
    assert tags.axis_word(probabilities, tags.ROW_WORDS, 0.9) == word


def test_an_axis_reads_presence_at_the_threshold_it_is_given():
    assert tags.axis_word((0.6, 0.1, 0.1), tags.ROW_WORDS, 0.5) == "top"
    assert tags.axis_word((0.6, 0.1, 0.1), tags.ROW_WORDS, 0.7) is None


@pytest.mark.parametrize(("row", "column", "term"), [
    ("top", "left", "top-left"),
    ("top", "center", "top"),
    ("top", "right", "top-right"),
    ("center", "left", "center-left"),
    ("center", "center", "center"),
    ("center", "right", "center-right"),
    ("bottom", "left", "bottom-left"),
    ("bottom", "center", "bottom"),
    ("bottom", "right", "bottom-right"),
    ("top", None, "top"),
    ("center", None, "center"),
    ("bottom", None, "bottom"),
    (None, "left", "left"),
    (None, "center", "center"),
    (None, "right", "right"),
    (None, None, ""),
])
def test_every_term_shape(row, column, term):
    assert tags.position_term(row, column) == term


def test_the_render_stage_writes_each_kept_item_with_its_term(classifier, monkeypatch):
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels, budget=None: pixels)
    classifier.noul_at = scripted_strips({
        "wall": ((0.9, 0.9, 0.9), (0.9, 0.9, 0.9)),
        "sky": ((0.9, 0.1, 0.1), (0.9, 0.9, 0.9)),
        "boats": ((0.1, 0.1, 0.9), (0.9, 0.2, 0.9)),
        "lamp": ((0.9, 0.1, 0.1), (0.1, 0.1, 0.9)),
        "door": ((0.1, 0.9, 0.1), (0.9, 0.1, 0.1)),
        "tall towers": ((1.0, 1.0, 1.0), (0.0, 1.0, 1.0)),
    })
    clip = FakeTagClip(proposal="wall, sky, boats, lamp, door, tall towers")

    _style, tile_texts = run(clip, a_tags_preset(), source=coordinate_source())

    assert tile_texts == [["wall, sky top, boats bottom, lamp top-right, door center-left, tall towers"]]


# The storm sky tile's scores: the whole tile passed both bays, and no strip held either.
STORM_STRIPS = {
    "storm clouds": ((1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
    "distant bay": ((0.0, 0.0, 0.27), (0.05, 0.02, 0.07)),
    "bay in the distance": ((0.0, 0.0, 0.73), (0.1, 0.6, 0.77)),
}


def test_a_kept_tag_no_strip_holds_is_left_out_of_the_text(classifier, monkeypatch):
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels, budget=None: pixels)
    classifier.noul_at = scripted_strips(STORM_STRIPS)
    clip = FakeTagClip(proposal="storm clouds, distant bay, bay in the distance")

    trace = run_trace(clip, a_tags_preset(), source=coordinate_source()).tiles[0][0]

    assert trace.kept == ("storm clouds", "distant bay", "bay in the distance")
    assert trace.unplaced == ("distant bay", "bay in the distance")
    assert trace.text == "storm clouds"


def test_a_lower_position_threshold_holds_and_places_the_tag(classifier, monkeypatch):
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels, budget=None: pixels)
    classifier.noul_at = scripted_strips(STORM_STRIPS)
    clip = FakeTagClip(proposal="storm clouds, distant bay, bay in the distance")

    trace = run_trace(clip, a_tags_preset(tile_tags_position_threshold=0.7),
                      source=coordinate_source()).tiles[0][0]

    assert trace.unplaced == ("distant bay",)
    assert trace.text == "storm clouds, bay in the distance bottom-right"


def test_locate_off_drops_no_tag_for_its_strips(classifier, monkeypatch):
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels, budget=None: pixels)
    classifier.noul_at = scripted_strips(STORM_STRIPS)
    clip = FakeTagClip(proposal="storm clouds, distant bay")

    trace = run_trace(clip, a_tags_preset(), source=coordinate_source(), locate=False).tiles[0][0]

    assert trace.unplaced == ()
    assert trace.text == "storm clouds, distant bay"


def test_six_strip_requests_per_tile_row_pack_every_kept_item_at_the_strip_size(classifier, comfy_stubs):
    classifier.noul = lambda text: 0.1 if text == statement("wooden table") else 0.9995

    run(FakeTagClip(), a_tags_preset(), tiles=(TILE_A, TILE_B), source=torch.rand(2, 16, 32, 3))

    strips = strip_requests(classifier)
    assert len(strips) == 2 * 2 * 6
    assert all(request["texts"] == [statement("red apple")] for request in strips)
    strip_calls = comfy_stubs["common_upscale_calls"][1:7]
    assert [shape for shape, *_rest in strip_calls] == [
        (1, 3, 5, 16), (1, 3, 5, 16), (1, 3, 6, 16), (1, 3, 16, 5), (1, 3, 16, 5), (1, 3, 16, 6)]
    for _shape, width, height, method, _crop in strip_calls:
        assert method == "area"
        assert abs(width * height - 250_000) / 250_000 < 0.01


def test_a_tile_with_nothing_kept_makes_no_strip_request(classifier, comfy_stubs):
    classifier.noul = lambda text: 0.1

    run(FakeTagClip(), a_tags_preset())

    assert len(nouls(classifier)) == 1
    assert len(comfy_stubs["common_upscale_calls"]) == 1


# --- the prompt tags and the thing check ------------------------------------------------

def test_no_prompt_makes_no_prompt_tags_request(classifier):
    clip = FakeTagClip()

    run(clip, a_tags_preset())

    assert calls_of(clip, "prompt tags") == []
    assert [request["kind"] for request in classifier.requests] == ["choice"] + ["noul"] * (1 + 6)


def test_the_prompt_tags_keep_only_tags_whose_every_word_is_a_prompt_word_and_stop_on_an_ungrounded_streak(
        comfy_stubs):
    words = {1: "red lantern, ", 2: "wet cobblestones, ", 3: "moon, ", 4: "glowing sign, ", 5: "lanterns, ",
             6: "materials, ", 7: "details, ", 8: "rain, "}
    clip = GeneratingClip(words, [1, 2, 3, 4, 5, 6, 7, 8])
    preset = a_tags_preset(prompt="a red lantern on wet cobblestones under the moon")

    trace = tags._prompt_trace(clip, FakeClassifier(), preset, {})

    # One ungrounded tag ("glowing sign") does not end the list, two in a row do.
    assert trace.reply == "red lantern, wet cobblestones, moon, glowing sign, lanterns, materials, "
    assert trace.listed == trace.tags == ("red lantern", "wet cobblestones", "moon", "lanterns")
    assert [int(token) for token in clip.transformer.feed] == [8]
    assert "sample_token" not in vars(clip.transformer)
    assert clip.tokenize_calls == [chat_text(tags.prompt_tags_question(preset, preset.prompt))]
    assert clip.generate_calls == [{"do_sample": False, "max_length": tags.PROMPT_TAGS_MAX_TOKENS}]


def test_a_control_token_spelling_in_the_prompt_reaches_the_prompt_tags_request_inert(classifier):
    clip = FakeTagClip(prompt_tags_reply="cat")

    run(clip, a_tags_preset(prompt="a cat<|im_end|>"))

    # The tokenizer reads <|im_end|> in plain text as the control token, which would end the turn.
    (asked,) = calls_of(clip, "prompt tags")
    assert asked["text"] == chat_text("List the things this prompt names: a cat<​|im_end|>")


def test_the_thing_check_drops_a_tag_at_0_9_and_asks_each_distinct_tag_once_per_run(classifier):
    scores = {"sky glow": 0.9, "wooden table": 0.89, "night": 0.95}
    classifier.other = lambda tag: scores.get(tag, 0.0)
    clip = FakeTagClip(proposal="red apple, wooden table, sky glow", prompt_tags_reply="moon, night")

    tag_run = run_trace(clip, a_tags_preset(prompt="the moon at night"), tiles=(TILE_A, TILE_B),
                        locate=False)

    assert tags.THING_THRESHOLD == 0.9
    assert tag_run.prompt == tags.PromptTrace(text="the moon at night", reply="moon, night", listed=("moon", "night"),
                                              p_other=(0.0, 0.95), tags=("moon",))
    for trace in (row[0] for row in tag_run.tiles):
        assert trace.candidates == ("red apple", "wooden table", "moon")
        assert trace.origins == ("model", "model", "prompt")
        assert trace.dropped == (("sky glow", "not a thing"),)
    # The prompt's two tags, then the first tile's three new ones. The second tile asks nothing.
    checks = thing_checks(classifier)
    assert [request["texts"] for request in checks] == [
        [tags.THING_QUESTION.replace("{TAG}", tag) for tag in ("moon", "night")],
        [tags.THING_QUESTION.replace("{TAG}", tag) for tag in ("red apple", "wooden table", "sky glow")]]
    assert all(request["image"] is None for request in checks)


def test_the_style_line_is_the_style_caption_alone_with_a_prompt(classifier):
    clip = FakeTagClip()

    style_texts, _tiles = run(clip, a_tags_preset(prompt="masterpiece, 85mm", style="Name the style."))

    assert style_texts == ["Oil painting."]
    (style_call,) = calls_of(clip, "style")
    assert style_call["max_length"] == 128
    assert next(call for call in clip.tokenize_calls if call["text"] == "Name the style.")["thinking"] is True
    # No request reads the whole style picture: the language question reads the prompt alone,
    # then one verify on the tile and its six strips.
    assert [request["image"] for request in language_requests(classifier)] == [None]
    assert len(verify_requests(classifier)) == 1
    assert len(nouls(classifier)) == 1 + 1 + 6


def test_the_style_caption_alone_is_the_style_line_without_a_prompt(classifier):
    style_texts, _tiles = run(FakeTagClip(), a_tags_preset(style="Name the style."))

    assert style_texts == ["Oil painting."]


def test_the_style_stages_read_the_style_source(classifier, monkeypatch):
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels, budget=None: pixels)
    clip = FakeTagClip()
    style_source = torch.rand(1, 40, 48, 3)

    run(clip, a_tags_preset(prompt="masterpiece", style="Name the style."), style_source=style_source)

    assert calls_of(clip, "style")[0]["image"].shape == (1, 40, 48, 3)
    assert all(request["image"] is None or request["image"].shape != (1, 40, 48, 3)
               for request in nouls(classifier))


def test_a_prompt_without_a_style_instruction_writes_no_style_line(classifier):
    clip = FakeTagClip()

    style_texts, _tiles = run(clip, a_tags_preset(prompt="masterpiece"))

    assert style_texts == []
    assert calls_of(clip, "style") == []


# --- the cache --------------------------------------------------------------------------

def test_a_second_identical_call_runs_no_generate_and_no_classifier_request(classifier):
    clip = FakeTagClip(prompt_tags_reply="moon")
    preset = a_tags_preset(prompt="masterpiece, the moon", style="Name the style.")
    source = torch.rand(1, 16, 32, 3)

    first = run(clip, preset, tiles=(TILE_A, TILE_B), source=source)
    generates, requests = len(clip.generate_calls), len(classifier.requests)
    second = run(clip, preset, tiles=(TILE_A, TILE_B), source=source)

    assert second == first
    assert (len(clip.generate_calls), len(classifier.requests)) == (generates, requests)


def test_clear_tag_cache_and_a_changed_wording_both_run_the_tile_again(classifier):
    clip = FakeTagClip()
    source = torch.rand(1, 16, 32, 3)

    run(clip, a_tags_preset(), source=source)
    tags.clear_tag_cache()
    run(clip, a_tags_preset(), source=source)
    run(clip, a_tags_preset(prompt="a cat"), source=source)

    assert len(propose_calls(clip)) == 3


def test_a_changed_strip_size_tag_setting_or_threshold_runs_the_tile_again(classifier, monkeypatch):
    clip = FakeTagClip()
    source = torch.rand(1, 16, 32, 3)

    run(clip, a_tags_preset(), source=source)
    monkeypatch.setattr(tags, "STRIP_MEGAPIXELS", 0.5)
    run(clip, a_tags_preset(), source=source)
    # A setting the toolkit's stages read, which reaches the key through tag_settings.
    monkeypatch.setattr(tags, "MAX_PROPOSED_TAGS", 24)
    run(clip, a_tags_preset(), source=source)
    run(clip, a_tags_preset(tile_tags_verification_threshold=0.6), source=source)
    run(clip, a_tags_preset(tile_tags_position_threshold=0.6), source=source)
    run(clip, a_tags_preset(prompt_tags_verification_threshold=0.6), source=source)

    assert len(propose_calls(clip)) == 6


def test_the_cache_never_serves_one_clips_text_to_another(classifier):
    source = torch.rand(1, 16, 32, 3)
    other = FakeTagClip(proposal="pear")

    run(FakeTagClip(), a_tags_preset(), source=source)
    _style, tile_texts = run(other, a_tags_preset(), source=source)

    assert tile_texts == [["pear"]]


# --- the trace --------------------------------------------------------------------------

def run_trace(clip, preset, tiles=(TILE_A,), source=None, **kwargs):
    source = torch.rand(1, 16, 32, 3) if source is None else source
    return tags.generate_tag_trace(clip, source, [Tile(rect) for rect in tiles], preset, **kwargs)


TRACE_PROPOSAL = "moon, objects, a red apple, Red Apple, herbs, hanging dried herbs, wooden table, night"
TRACE_PRESENCE = {
    "moon": ((0.9, 0.1, 0.1), (0.1, 0.1, 0.9)),
    "red apple": ((0.1, 0.1, 0.9), (0.9, 0.2, 0.9)),
    "hanging dried herbs": ((0.9, 0.9, 0.9), (0.9, 0.9, 0.9)),
}


def test_the_tile_trace_records_every_stage(classifier, monkeypatch):
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels, budget=None: pixels)
    classifier.noul_at = scripted_strips(TRACE_PRESENCE, whole={"wooden table": 0.2})
    classifier.other = lambda tag: 0.95 if tag == "night" else 0.0
    clip = FakeTagClip(proposal=TRACE_PROPOSAL, prompt_tags_reply="moon")

    tag_run = run_trace(clip, a_tags_preset(prompt="the moon"), source=coordinate_source())

    trace = tag_run.tiles[0][0]
    assert trace.reply == TRACE_PROPOSAL
    assert trace.echo_reply is None
    assert trace.proposed == ("moon", "objects", "a red apple", "red apple", "herbs", "hanging dried herbs",
                              "wooden table", "night")
    assert trace.candidates == ("moon", "red apple", "herbs", "hanging dried herbs", "wooden table")
    assert trace.origins == ("both", "model", "model", "model", "model")
    assert trace.dropped == (("objects", "category noun"), ("red apple", "repeat"), ("night", "not a thing"))
    assert trace.scores == (0.9995, 0.9995, 0.9995, 0.9995, 0.2)
    assert trace.verified == ("moon", "red apple", "herbs", "hanging dried herbs")
    assert trace.kept == ("moon", "red apple", "hanging dried herbs")
    assert trace.strips == tuple(rows + columns for rows, columns in
                                 (TRACE_PRESENCE[item] for item in trace.kept))
    assert trace.terms == ("top-right", "bottom", "")
    assert trace.text == "moon top-right, red apple bottom, hanging dried herbs"
    assert tag_run.style_texts == ("",)
    assert tag_run.prompt == tags.PromptTrace(text="the moon", reply="moon", listed=("moon",), p_other=(0.0,),
                                              tags=("moon",))


def test_a_reply_of_only_category_nouns_is_replaced_by_the_fallback_reply(classifier):
    clip = FakeTagClip(proposal="objects, people", fallback_proposal="red apple, jar")

    tag_run = run_trace(clip, a_tags_preset())

    assert [call["text"] for call in propose_calls(clip)] == [PROPOSE_TEXT, fallback_text()]
    trace = tag_run.tiles[0][0]
    assert (trace.echo_reply, trace.reply, trace.proposed) == ("objects, people", "red apple, jar",
                                                               ("red apple", "jar"))
    assert trace.text == "red apple, jar"


def test_locate_off_makes_no_strip_request_and_writes_no_term(classifier):
    tag_run = run_trace(FakeTagClip(), a_tags_preset(), tiles=(TILE_A, TILE_B),
                        source=torch.rand(2, 16, 32, 3), locate=False)

    assert tag_run.style_texts == ("", "")
    assert tag_run.prompt is None
    assert strip_requests(classifier) == []
    assert [len(row_traces) for row_traces in tag_run.tiles] == [2, 2]
    for trace in (trace for row_traces in tag_run.tiles for trace in row_traces):
        assert trace.strips == ((), ())
        assert trace.terms == ("", "")
        assert trace.text == "red apple, wooden table"


def test_the_run_records_the_style_caption_and_the_prompt_tags(classifier):
    classifier.other = lambda tag: 0.95 if tag == "city" else 0.0
    prompt = "masterpiece, 85mm, the moon over a city"
    reply = "moon, city, skyline"

    with_caption = run_trace(FakeTagClip(prompt_tags_reply=reply),
                             a_tags_preset(prompt=prompt, style="Name the style."))
    without_caption = run_trace(FakeTagClip(prompt_tags_reply=reply), a_tags_preset(prompt=prompt))

    prompt_tags = tags.PromptTrace(text=prompt, reply=reply, listed=("moon", "city"), p_other=(0.0, 0.95),
                                   tags=("moon",))
    assert with_caption.style_texts == ("Oil painting.",)
    assert with_caption.prompt == prompt_tags
    assert without_caption.style_texts == ("",)
    assert without_caption.prompt == prompt_tags
    # The kept prompt tag joins the candidates after the model's tags and the dropped one never does.
    assert with_caption.tiles[0][0].candidates == ("red apple", "wooden table", "moon")


def test_the_set_and_the_trace_share_one_cache_keyed_on_locate(classifier):
    clip = FakeTagClip(prompt_tags_reply="moon")
    preset = a_tags_preset(prompt="masterpiece, the moon", style="Name the style.")
    source = torch.rand(1, 16, 32, 3)

    style_texts, tile_texts = run(clip, preset, tiles=(TILE_A, TILE_B), source=source)
    generates, requests = len(clip.generate_calls), len(classifier.requests)
    tag_run = run_trace(clip, preset, tiles=(TILE_A, TILE_B), source=source)

    assert (len(clip.generate_calls), len(classifier.requests)) == (generates, requests)
    assert list(tag_run.style_texts) == style_texts
    assert [[trace.text for trace in row] for row in tag_run.tiles] == tile_texts

    run_trace(clip, preset, tiles=(TILE_A, TILE_B), source=source, locate=False)

    assert len(propose_calls(clip)) == 4


def test_a_changed_caption_size_or_style_budget_writes_a_new_style_caption(classifier):
    clip = FakeTagClip()
    source = torch.rand(1, 16, 32, 3)
    preset = a_tags_preset(style="Name the style.")
    resized = dataclasses.replace(preset, vision=dataclasses.replace(preset.vision, caption_megapixels=0.5))

    run(clip, preset, source=source)
    run(clip, resized, source=source)
    run(clip, a_tags_preset(style="Name the style.", style_tokens=256), source=source)

    style_calls = [call for call in clip.generate_calls if not call["propose"]]
    assert [call["max_length"] for call in style_calls] == [128, 128, 256]
    assert style_calls[0]["image"].shape != style_calls[1]["image"].shape
    # The tile traces stay cached, since neither setting reaches the tags stages.
    assert len(propose_calls(clip)) == 1


# --- the empty verification statement --------------------------------------------------

def test_an_empty_verification_statement_keeps_every_candidate_unverified(classifier):
    preset = dataclasses.replace(a_tags_preset(), tile_tags_verification_statement="")
    clip = FakeTagClip(proposal="herbs, hanging dried herbs, jar")

    tag_run = run_trace(clip, preset)

    trace = tag_run.tiles[0][0]
    assert nouls(classifier) == []
    assert trace.candidates == ("herbs", "hanging dried herbs", "jar")
    assert trace.scores == (None, None, None)
    assert trace.verified == trace.candidates
    # drop_subsets still runs on the unverified candidates.
    assert trace.kept == ("hanging dried herbs", "jar")
    assert trace.strips == ((), ())
    assert trace.terms == ("", "")
    assert trace.text == "hanging dried herbs, jar"


# --- progress and interrupts ------------------------------------------------------------

def test_the_standalone_bar_counts_a_style_row_and_every_tile_row(classifier, comfy_stubs):
    run(FakeTagClip(), a_tags_preset(style="Name the style."), tiles=(TILE_A, TILE_B))

    pbar = comfy_stubs["progress_bars"][-1]
    assert pbar.total == 3
    assert [update[0] for update in pbar.updates] == [1, 2, 3]
    assert comfy_stubs["interrupt_calls"] == 3


@pytest.mark.parametrize("prompt", ["", "the moon, masterpiece"])
def test_without_a_style_row_the_bar_counts_tiles_and_checks_one_interrupt_per_tile(classifier, comfy_stubs,
                                                                                    prompt):
    run(FakeTagClip(), a_tags_preset(prompt=prompt), tiles=(TILE_A, TILE_B))

    assert comfy_stubs["progress_bars"][-1].total == 2
    assert comfy_stubs["interrupt_calls"] == 2


def test_a_ledger_replaces_the_bar_and_counts_run_wide_across_the_picture_loop(classifier, comfy_stubs):
    reported = []

    class Recorder:
        def caption_done(self, index, count):
            reported.append((index, count))

    before = len(comfy_stubs["progress_bars"])
    run(FakeTagClip(), a_tags_preset(style="Name the style."), tiles=(TILE_A, TILE_B),
        batch_size=2, batch_index=1, progress=Recorder())

    assert reported == [(4, 6), (5, 6), (6, 6)]
    assert len(comfy_stubs["progress_bars"]) == before


# --- the fail-fast guards ---------------------------------------------------------------

def test_a_caption_preset_is_refused(classifier):
    preset = captions.Preset(surface=captions.VLM_METHOD_CAPTIONS, label="standard",
                             vision=a_tags_preset().vision, tile_instruction="describe",
                             tile_max_tokens=256)

    with pytest.raises(RuntimeError, match=r"'standard'.*tile_text = 'tags'"):
        run(FakeTagClip(), preset)


def test_an_unfilled_prompt_placeholder_is_refused_before_any_generate(classifier):
    clip = FakeTagClip()

    with pytest.raises(RuntimeError, match=re.escape("captions.with_prompt")):
        run(clip, a_tags_preset(style="Style of {PROMPT}"))
    assert clip.generate_calls == []


@pytest.mark.parametrize("key", ["tile_tags_instruction", "tile_tags_verification_statement"])
def test_a_prompt_placeholder_outside_the_prompt_tags_instruction_is_refused_before_any_generate(classifier, key):
    clip = FakeTagClip()
    preset = dataclasses.replace(a_tags_preset(prompt="a cat"), **{key: "{TAG} from {PROMPT}"})

    with pytest.raises(RuntimeError, match=rf"'tags' carries \{{PROMPT\}} in its {key}, which only "
                                           r"prompt_tags_instruction takes"):
        run(clip, preset)
    assert clip.generate_calls == []


def test_the_guard_refuses_a_clip_that_is_not_a_qwen3_vl_encoder_naming_the_node():
    from logit_classifier import UnsupportedModelError

    with pytest.raises(UnsupportedModelError,
                       match=r"^Context-Anchored Tile Refine \(VL\): this CLIP is not a Qwen3-VL text encoder"):
        tags.check_tags_ready(types.SimpleNamespace(tokenize=lambda text: None))


def test_the_guard_refuses_a_clip_loaded_without_its_vision_tower_naming_the_node(monkeypatch):
    import logit_classifier.toolkit.comfyui.classifier as toolkit_classifier
    from logit_classifier import VisionUnsupportedError

    monkeypatch.setattr(toolkit_classifier, "ComfyClipBackend",
                        lambda clip, **kwargs: types.SimpleNamespace(sees_images=False))

    with pytest.raises(VisionUnsupportedError,
                       match=r"^Context-Anchored Tile Refine \(VL\): this CLIP was loaded without its vision tower"):
        tags.check_tags_ready(FakeTagClip())


# The whole library missing, and a library older than 0.3.0, which has no toolkit.
@pytest.mark.parametrize("module", ["logit_classifier", "logit_classifier.toolkit.comfyui"])
def test_the_guard_names_the_pip_command_when_the_library_is_missing_or_too_old(monkeypatch, module):
    monkeypatch.setitem(sys.modules, module, None)

    with pytest.raises(RuntimeError, match=re.escape('pip install -U "logit-classifier>=0.4.0"')):
        tags.check_tags_ready(FakeTagClip())


def test_the_guard_names_the_pip_command_for_a_0_3_library_without_presence_statement(monkeypatch):
    import logit_classifier.toolkit.tags as library_tags

    monkeypatch.delattr(library_tags, "presence_statement")

    with pytest.raises(RuntimeError, match=re.escape('pip install -U "logit-classifier>=0.4.0"')):
        tags.check_tags_ready(FakeTagClip())


def test_the_strip_statements_are_the_librarys_presence_statements():
    from logit_classifier.toolkit.tags import presence_statement

    statements = tags._statements(["  red apple  ", "moon"], VERIFY)

    assert statements == ["This image visibly contains red apple", "This image visibly contains moon"]
    assert statements == [presence_statement(item, wording=VERIFY) for item in ["  red apple  ", "moon"]]


def test_the_shipped_thresholds_are_the_librarys_strict_thresholds():
    import logit_classifier.toolkit.tags as library_tags

    preset = captions.resolve_method(captions.VLM_METHOD_CAPTIONS)

    strict = tuple(library_tags.STRICT_THRESHOLDS)
    in_file = (preset.tile_tags_verification_threshold, preset.prompt_tags_verification_threshold)
    in_code = (captions.SHIPPED_TAGS_VERIFICATION_THRESHOLD, captions.SHIPPED_PROMPT_TAGS_VERIFICATION_THRESHOLD)

    assert in_file == strict
    assert in_code == strict


def test_the_guard_passes_while_the_dist_metadata_reports_0_1_0(monkeypatch):
    open_toolkit(monkeypatch, FakeClassifier())
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1.0")

    tags.check_tags_ready(FakeTagClip())


def test_a_missing_library_fails_before_any_generate(classifier, monkeypatch):
    monkeypatch.setitem(sys.modules, "logit_classifier.toolkit.comfyui", None)
    clip = FakeTagClip()

    with pytest.raises(RuntimeError, match="logit-classifier"):
        run(clip, a_tags_preset(style="Name the style."))
    assert clip.generate_calls == []


def test_every_tuning_value_is_the_measured_one():
    assert tags.VL_MAX_PIXELS == 1024 * 1024
    assert tags.MAX_PROPOSED_TAGS == 25
    assert tags.MAX_MERGED_TAGS == 64
    assert tags.PROMPT_TAGS_MAX_TOKENS == 256
    assert tags.UNGROUNDED_STREAK == 2
    assert tags.THING_THRESHOLD == 0.9
    assert tags.STRIP_MEGAPIXELS == 0.25
    assert sorted(tags.CATEGORY_NOUNS) == ["animals", "clothing", "materials", "objects", "people", "setting"]


def test_the_shipped_tags_preset_hands_the_toolkit_every_value_the_pass_ran_with():
    from logit_classifier.toolkit.comfyui import TagSettings

    preset = captions.resolve_method(captions.VLM_METHOD_CAPTIONS)
    defaults = TagSettings()

    settings = tags.tag_settings(preset)

    assert preset.label == "tags"
    assert settings.propose_instruction == preset.tile_tags_instruction
    assert settings.prompt_tags_instruction == preset.prompt_tags_instruction
    assert settings.verify_statement == "This image visibly contains {TAG}"
    assert (settings.verify_threshold, settings.prompt_only_threshold) == (0.99998, 0.99998)
    assert (settings.propose_cap, settings.merge_cap) == (25, 64)
    assert settings.category_nouns == tags.CATEGORY_NOUNS
    assert (settings.propose_max_tokens, settings.prompt_tags_max_tokens) == (128, 256)
    assert settings.ungrounded_streak == 2
    assert settings.thing_question == 'What does the image tag "{TAG}" name'
    assert settings.thing_criteria == tuple(tags.THING_CRITERIA.items())
    assert settings.thing_threshold == 0.9
    assert (settings.thing_check, settings.subsets_rule, settings.order) == ("all", "words", "proposed")
    # The language gate and the echo fallback are the toolkit's, as the Logit Tagger measured them.
    assert (settings.language_question, settings.english_threshold, settings.translate_instruction,
            settings.translate_max_tokens) == (defaults.language_question, defaults.english_threshold,
                                               defaults.translate_instruction, defaults.translate_max_tokens)
    assert settings.echo_fallback_instruction == defaults.echo_fallback_instruction
    # A Preset built in code, or by the Captions node without the socket, carries "".
    unlisted = dataclasses.replace(preset, prompt_tags_instruction="")
    assert tags.tag_settings(unlisted).prompt_tags_instruction == "{PROMPT}"
