"""tags.py: the tile text a tags preset writes, from propose, merge, verify, clean and render.

A duck-typed CLIP stands in for the VL text encoder and a scripted classifier, patched in at
`tags.build_classifier`, answers every noul and choice. logit_classifier itself is the real
library (its tags helpers and question types), so no comfy install and no model are needed.
"""
import dataclasses
import importlib.metadata
import re
import sys
import types

import pytest
import torch
from logit_classifier.tags import MAX_CANDIDATES
from test_vl import Tile

from context_anchored_tile_refine import captions, tags
from context_anchored_tile_refine.grid import Rect

TILE_A = Rect(0, 0, 16, 16)
TILE_B = Rect(16, 0, 32, 16)
ANCHOR = "This image was made from the prompt: {PROMPT}\n"
PROPOSE = "Tag this image."
VERIFY = "This image visibly contains {TAG}"


def a_tags_preset(prompt="", style="", style_tokens=128, **thresholds):
    return captions.Preset(
        surface=captions.VLM_METHOD_CAPTIONS, label="tags",
        vision=captions.VisionSettings(canvas_tokens=1, crop_tokens=0,
                                       caption_megapixels=captions.VL_INPUT_BUDGET_MEGAPIXELS),
        style_instruction=style, style_max_tokens=style_tokens, kind=captions.TILE_TEXT_TAGS,
        tile_tags_instruction=PROPOSE, tile_tags_with_prompt_instruction=ANCHOR,
        tile_tags_verification_statement=VERIFY, prompt=prompt, **thresholds)


def statement(item):
    return VERIFY.replace("{TAG}", item)


class FakeTagClip:
    """Duck-typed VL clip. A request rendered from PROPOSE_TEMPLATE is a propose and answers
    with `proposal`, anything else is the style caption and answers with `style_answer`.
    generate returns `proposal_tokens` ids, which is what the budget check reads."""

    def __init__(self, proposal="red apple, wooden table", style_answer="<think>hm</think>Oil painting.",
                 proposal_tokens=5, image_tokens=True, rejects_images=False):
        self.proposal = proposal
        self.style_answer = style_answer
        self.proposal_tokens = proposal_tokens
        self.image_tokens = image_tokens
        self.rejects_images = rejects_images
        self.tokenize_calls = []
        self.generate_calls = []
        self._answers = {}

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
        import comfy.model_management

        text, image = tokens["_probe"]
        is_propose = text.startswith("<|im_start|>")
        self.generate_calls.append({"text": text, "image": image, "propose": is_propose,
                                    "cuda_graphs_off": comfy.model_management.args.disable_cuda_graphs,
                                    **kwargs})
        handle = len(self.generate_calls)
        self._answers[handle] = self.proposal if is_propose else self.style_answer
        return [handle] * (self.proposal_tokens if is_propose else 3)

    def decode(self, token_ids, skip_special_tokens=True):
        return self._answers[token_ids[0]]


class FakeClassifier:
    """Scripted classifier: `noul(statement)` answers every noul and `style(fragment)` every
    subject or style choice. `noul_at(statement, picture)`, when set, answers the nouls in
    place of `noul`, so a test can script each strip. Every request is recorded with its
    statements and its picture."""

    def __init__(self, noul=None, style=None):
        self.noul = noul if noul is not None else (lambda text: 0.95)
        self.style = style if style is not None else (lambda fragment: 0.0)
        self.noul_at = None
        self.requests = []

    def classify(self, request, image=None):
        answers = {}
        texts = []
        kinds = set()
        for qid, question in request.questions.items():
            texts.append(question.instructions)
            kinds.add(question.type)
            if question.type == "choice":
                fragment = re.search(r'"(.*)"', question.instructions).group(1)
                p = self.style(fragment)
                answers[qid] = types.SimpleNamespace(probabilities={"subject": 1.0 - p, "style": p})
            elif self.noul_at is not None:
                answers[qid] = types.SimpleNamespace(noul=self.noul_at(question.instructions, image))
            else:
                answers[qid] = types.SimpleNamespace(noul=self.noul(question.instructions))
        self.requests.append({"kind": kinds.pop() if len(kinds) == 1 else kinds, "texts": texts,
                              "image": image, "state": request.state})
        return types.SimpleNamespace(answers=answers), None


@pytest.fixture
def classifier(comfy_stubs, monkeypatch):
    fake = FakeClassifier()
    monkeypatch.setattr(tags, "build_classifier", lambda clip: fake)
    return fake


def run(clip, preset, tiles=(TILE_A,), source=None, **kwargs):
    source = torch.rand(1, 16, 32, 3) if source is None else source
    return tags.generate_tag_set(clip, source, [Tile(rect) for rect in tiles], preset, **kwargs)


def propose_calls(clip):
    return [call for call in clip.generate_calls if call["propose"]]


def nouls(classifier):
    return [request for request in classifier.requests if request["kind"] == "noul"]


def is_strip(request):
    # The resample stub returns the size it was asked for, so a strip is known by its area.
    budget = tags.STRIP_MEGAPIXELS * 1_000_000
    height, width = request["image"].shape[1], request["image"].shape[2]
    return abs(height * width - budget) / budget < 0.01


def strip_requests(classifier):
    return [request for request in nouls(classifier) if is_strip(request)]


def verify_requests(classifier):
    return [request for request in nouls(classifier) if not is_strip(request)]


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


def test_propose_turns_cuda_graphs_off_and_restores_the_flag(classifier):
    import comfy.model_management

    clip = FakeTagClip()
    run(clip, a_tags_preset())

    assert propose_calls(clip)[0]["cuda_graphs_off"] is True
    assert comfy.model_management.args.disable_cuda_graphs is False


def test_the_with_prompt_instruction_is_prepended_only_when_the_preset_carries_a_prompt(classifier):
    without = FakeTagClip()
    with_prompt = FakeTagClip()

    run(without, a_tags_preset())
    run(with_prompt, a_tags_preset(prompt="a cat {on} a mat"))

    assert without.tokenize_calls[0]["text"] == tags.PROPOSE_TEMPLATE.format(instruction=PROPOSE)
    assert with_prompt.tokenize_calls[-1]["text"] == tags.PROPOSE_TEMPLATE.format(
        instruction="This image was made from the prompt: a cat {on} a mat\n" + PROPOSE)


def test_a_decode_that_fills_the_budget_drops_its_cut_tail(classifier):
    clip = FakeTagClip(proposal="red apple, wooden table, gre", proposal_tokens=tags.PROPOSE_MAX_TOKENS)

    _style, tile_texts = run(clip, a_tags_preset())

    assert nouls(classifier)[0]["texts"] == [statement("red apple"), statement("wooden table")]
    assert tile_texts == [["red apple, wooden table"]]


def test_the_repeat_stop_ends_the_decode_on_a_repeated_block_and_restores_sample_token():
    words = {1: "apple, ", 2: "pear, "}

    class Transformer:
        def __init__(self):
            self.model = types.SimpleNamespace(config=types.SimpleNamespace(stop_tokens=[99]))

        def sample_token(self, *args, **kwargs):
            return torch.tensor([next(feed)])

    transformer = Transformer()
    clip = types.SimpleNamespace(
        cond_stage_model=types.SimpleNamespace(clip="qwen", qwen=types.SimpleNamespace(transformer=transformer)),
        decode=lambda ids: "".join(words[i] for i in ids))
    feed = iter([1, 2, 1, 2])

    with tags.stop_on_repeat(clip):
        sampled = [int(transformer.sample_token()) for _ in range(4)]

    assert sampled == [1, 2, 1, 99]
    assert "sample_token" not in vars(transformer)


def test_a_tokenizer_that_rejects_images_fails_at_the_first_propose_naming_the_fix(classifier):
    clip = FakeTagClip(rejects_images=True)

    with pytest.raises(RuntimeError, match="vision-language text encoder"):
        run(clip, a_tags_preset())
    assert clip.generate_calls == []


def test_a_tokenizer_with_no_image_token_fails_at_the_first_propose_naming_the_fix(classifier):
    clip = FakeTagClip(image_tokens=False)

    with pytest.raises(RuntimeError, match=r"no image tokens.*vision-language text encoder"):
        run(clip, a_tags_preset())
    assert clip.generate_calls == []


# --- the merge, verify and clean stages -------------------------------------------------

def test_the_merge_drops_category_nouns_articles_and_repeats(classifier):
    clip = FakeTagClip(proposal="objects, people, a red apple, Red Apple, setting, the table, materials")

    run(clip, a_tags_preset())

    assert nouls(classifier)[0]["texts"] == [statement("red apple"), statement("table")]


def test_the_verify_keeps_items_at_the_threshold_and_asks_one_packed_request_per_tile_row(classifier):
    classifier.noul = lambda text: {statement("red apple"): 0.9, statement("wooden table"): 0.89}[text]

    _style, tile_texts = run(FakeTagClip(), a_tags_preset(), tiles=(TILE_A, TILE_B))

    assert a_tags_preset().tile_tags_verification_threshold == 0.9
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
    classifier.noul = lambda text: 0.1 if text == statement("hanging dried herbs") else 0.9
    clip = FakeTagClip(proposal="herbs, hanging dried herbs, jar")

    _style, tile_texts = run(clip, a_tags_preset())

    assert tile_texts == [["herbs, jar"]]


def test_the_candidates_are_capped_at_48_with_the_subject_fragments_first(classifier):
    proposal = ", ".join(f"thing {n}" for n in range(60))
    clip = FakeTagClip(proposal=proposal)

    run(clip, a_tags_preset(prompt="a lantern, the moon"))

    texts = verify_requests(classifier)[-1]["texts"]
    assert MAX_CANDIDATES == 48
    assert len(texts) == 48
    assert texts[:3] == [statement("lantern"), statement("moon"), statement("thing 0")]


def test_an_initialism_in_the_prompt_reaches_the_candidates_whole(classifier):
    run(FakeTagClip(proposal=""), a_tags_preset(prompt="a flag of the U.S.A., Washington D.C. at night"))

    assert verify_requests(classifier)[-1]["texts"] == [
        statement("flag of the u.s.a."), statement("washington d.c. at night")]


def test_two_spellings_of_an_initialism_merge_into_one_candidate():
    from logit_classifier.tags import parse_candidates, split_prompt

    candidates, origins, _dropped = tags.merge_trace(split_prompt("flag of the U.S.A, a cowboy"),
                                                     parse_candidates("flag of the U.S.A., sky"))

    assert candidates == ("flag of the u.s.a.", "cowboy", "sky")
    assert origins == ("both", "prompt", "model")


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
    and `whole[item]` (default 0.95) on the verify picture."""
    def noul_at(text, picture):
        item = text.removeprefix(statement(""))
        kind, index = strip_of(picture)
        if kind == "whole":
            return (whole or {}).get(item, 0.95)
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
    classifier.noul = lambda text: 0.1 if text == statement("wooden table") else 0.9

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


# --- the prompt fragments and the style line --------------------------------------------

def test_no_prompt_makes_no_fragment_request(classifier):
    run(FakeTagClip(), a_tags_preset())

    assert [request["kind"] for request in classifier.requests] == ["noul"] * (1 + 6)


def test_fragments_are_routed_to_style_at_0_9_in_one_text_only_request(classifier):
    scores = {"masterpiece": 0.98, "the moon": 0.69, "85mm": 0.9, "wet street": 0.2}
    classifier.style = lambda fragment: scores[fragment]
    clip = FakeTagClip(proposal="")

    style_texts, tile_texts = run(clip, a_tags_preset(prompt="masterpiece, the moon, 85mm, wet street"))

    choices = [request for request in classifier.requests if request["kind"] == "choice"]
    assert len(choices) == 1
    assert choices[0]["image"] is None
    assert choices[0]["texts"][1] == 'What does the image prompt phrase "the moon" describe'
    # The style fragments are dropped, so the tile verify is the only verify request.
    (tile_verify,) = verify_requests(classifier)
    assert tile_verify["texts"] == [statement("moon"), statement("wet street")]
    assert style_texts == []
    assert tile_texts == [["moon, wet street"]]


def test_the_style_line_is_the_style_caption_alone_with_style_fragments_in_the_prompt(classifier):
    classifier.style = lambda fragment: 0.99
    clip = FakeTagClip()

    style_texts, _tiles = run(clip, a_tags_preset(prompt="masterpiece, 85mm", style="Name the style."))

    assert style_texts == ["Oil painting."]
    style_call = next(call for call in clip.generate_calls if not call["propose"])
    assert style_call["max_length"] == 128
    assert clip.tokenize_calls[0]["thinking"] is True
    # No request reads the whole style picture: one verify on the tile, then its six strips.
    assert len(verify_requests(classifier)) == 1
    assert len(nouls(classifier)) == 1 + 6


def test_the_style_caption_alone_is_the_style_line_without_a_prompt(classifier):
    style_texts, _tiles = run(FakeTagClip(), a_tags_preset(style="Name the style."))

    assert style_texts == ["Oil painting."]


def test_the_style_stages_read_the_style_source(classifier, monkeypatch):
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels, budget=None: pixels)
    classifier.style = lambda fragment: 0.99
    clip = FakeTagClip()
    style_source = torch.rand(1, 40, 48, 3)

    run(clip, a_tags_preset(prompt="masterpiece", style="Name the style."), style_source=style_source)

    assert clip.generate_calls[0]["image"].shape == (1, 40, 48, 3)
    assert all(request["image"].shape != (1, 40, 48, 3) for request in nouls(classifier))


def test_a_prompt_without_a_style_instruction_writes_no_style_line(classifier):
    classifier.style = lambda fragment: 0.99
    clip = FakeTagClip()

    style_texts, _tiles = run(clip, a_tags_preset(prompt="masterpiece"))

    assert style_texts == []
    assert [call for call in clip.generate_calls if not call["propose"]] == []


# --- the cache --------------------------------------------------------------------------

def test_a_second_identical_call_runs_no_generate_and_no_classifier_request(classifier):
    classifier.style = lambda fragment: 0.99 if fragment == "masterpiece" else 0.1
    clip = FakeTagClip()
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


def test_a_changed_strip_size_or_threshold_runs_the_tile_again(classifier, monkeypatch):
    clip = FakeTagClip()
    source = torch.rand(1, 16, 32, 3)

    run(clip, a_tags_preset(), source=source)
    monkeypatch.setattr(tags, "STRIP_MEGAPIXELS", 0.5)
    run(clip, a_tags_preset(), source=source)
    run(clip, a_tags_preset(tile_tags_verification_threshold=0.6), source=source)
    run(clip, a_tags_preset(tile_tags_position_threshold=0.6), source=source)

    assert len(propose_calls(clip)) == 4


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


TRACE_PROPOSAL = "moon, objects, a red apple, Red Apple, herbs, hanging dried herbs, wooden table"
TRACE_PRESENCE = {
    "moon": ((0.9, 0.1, 0.1), (0.1, 0.1, 0.9)),
    "red apple": ((0.1, 0.1, 0.9), (0.9, 0.2, 0.9)),
    "hanging dried herbs": ((0.9, 0.9, 0.9), (0.9, 0.9, 0.9)),
}


def test_the_tile_trace_records_every_stage(classifier, monkeypatch):
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels, budget=None: pixels)
    classifier.noul_at = scripted_strips(TRACE_PRESENCE, whole={"wooden table": 0.2})
    clip = FakeTagClip(proposal=TRACE_PROPOSAL)

    tag_run = run_trace(clip, a_tags_preset(prompt="the moon"), source=coordinate_source())

    trace = tag_run.tiles[0][0]
    assert trace.reply == TRACE_PROPOSAL
    assert trace.proposed == ("moon", "objects", "a red apple", "red apple", "herbs", "hanging dried herbs",
                              "wooden table")
    assert trace.candidates == ("moon", "red apple", "herbs", "hanging dried herbs", "wooden table")
    assert trace.origins == ("both", "model", "model", "model", "model")
    assert trace.dropped == (("objects", "category noun"), ("red apple", "repeat"))
    assert trace.scores == (0.95, 0.95, 0.95, 0.95, 0.2)
    assert trace.verified == ("moon", "red apple", "herbs", "hanging dried herbs")
    assert trace.kept == ("moon", "red apple", "hanging dried herbs")
    assert trace.strips == tuple(rows + columns for rows, columns in
                                 (TRACE_PRESENCE[item] for item in trace.kept))
    assert trace.terms == ("top-right", "bottom", "")
    assert trace.text == "moon top-right, red apple bottom, hanging dried herbs"
    assert tag_run.style_texts == ("",)
    assert tag_run.prompt == tags.PromptTrace(fragments=("the moon",), style_p=(0.0,),
                                              subjects=("the moon",), styles=())


def test_the_merge_records_every_item_over_the_cap():
    proposed = [f"thing {n}" for n in range(MAX_CANDIDATES + 2)]

    candidates, origins, dropped = tags.merge_trace(["a lantern"], proposed)

    assert len(candidates) == MAX_CANDIDATES
    assert origins == ("prompt",) + ("model",) * (MAX_CANDIDATES - 1)
    assert dropped == tuple((f"thing {n}", "over the cap") for n in range(MAX_CANDIDATES - 1,
                                                                         MAX_CANDIDATES + 2))
    assert tags.merge_candidates(["a lantern"], proposed) == list(candidates)


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


def test_the_run_records_the_style_caption_and_the_fragment_sort(classifier):
    scores = {"masterpiece": 0.99, "85mm": 0.95, "the moon": 0.4}
    classifier.style = lambda fragment: scores[fragment]
    prompt = "masterpiece, 85mm, the moon"

    with_caption = run_trace(FakeTagClip(), a_tags_preset(prompt=prompt, style="Name the style."))
    without_caption = run_trace(FakeTagClip(), a_tags_preset(prompt=prompt))

    sort = tags.PromptTrace(fragments=("masterpiece", "85mm", "the moon"), style_p=(0.99, 0.95, 0.4),
                            subjects=("the moon",), styles=("masterpiece", "85mm"))
    assert with_caption.style_texts == ("Oil painting.",)
    assert with_caption.prompt == sort
    assert without_caption.style_texts == ("",)
    assert without_caption.prompt == sort
    # The subject fragment joins the candidates first and no style fragment joins them.
    assert with_caption.tiles[0][0].candidates == ("moon", "red apple", "wooden table")


def test_the_set_and_the_trace_share_one_cache_keyed_on_locate(classifier):
    classifier.style = lambda fragment: 0.99 if fragment == "masterpiece" else 0.1
    clip = FakeTagClip()
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


def test_the_tile_tags_question_is_the_request_without_the_template():
    no_prompt_text = dataclasses.replace(a_tags_preset(prompt="a cat"), tile_tags_with_prompt_instruction="")

    assert tags.tile_tags_question(a_tags_preset()) == PROPOSE
    assert tags.tile_tags_question(a_tags_preset(prompt="a cat")) == (
        "This image was made from the prompt: a cat\n" + PROPOSE)
    assert tags.tile_tags_question(no_prompt_text) == PROPOSE
    assert tags.propose_text(a_tags_preset(prompt="a cat")) == tags.PROPOSE_TEMPLATE.format(
        instruction=tags.tile_tags_question(a_tags_preset(prompt="a cat")))


# --- the empty verification statement --------------------------------------------------

def test_an_empty_verification_statement_keeps_every_candidate_unverified(classifier):
    preset = dataclasses.replace(a_tags_preset(), tile_tags_verification_statement="")
    clip = FakeTagClip(proposal="herbs, hanging dried herbs, jar")

    tag_run = run_trace(clip, preset)

    trace = tag_run.tiles[0][0]
    assert classifier.requests == []
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
def test_a_prompt_placeholder_outside_the_with_prompt_instruction_is_refused_before_any_generate(classifier, key):
    clip = FakeTagClip()
    preset = dataclasses.replace(a_tags_preset(prompt="a cat"), **{key: "{TAG} from {PROMPT}"})

    with pytest.raises(RuntimeError, match=rf"'tags' carries \{{PROMPT\}} in its {key}, which only "
                                           r"tile_tags_with_prompt_instruction takes"):
        run(clip, preset)
    assert clip.generate_calls == []


def test_the_guard_refuses_a_clip_that_cannot_generate():
    with pytest.raises(RuntimeError, match=r"cannot generate text.*'vision tokens'"):
        tags.check_tags_ready(types.SimpleNamespace(tokenize=lambda text: None))


def test_the_guard_names_the_pip_command_when_the_library_is_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "logit_classifier.tags", None)

    with pytest.raises(RuntimeError, match=re.escape('pip install -U "logit-classifier>=0.2.1"')):
        tags.check_tags_ready(FakeTagClip())


def test_the_guard_names_the_pip_command_when_the_library_is_too_old(monkeypatch):
    import logit_classifier.tags

    # 0.2.0 is the release without drop_unfinished_tag.
    monkeypatch.delattr(logit_classifier.tags, "drop_unfinished_tag")

    with pytest.raises(RuntimeError, match=r"lacks drop_unfinished_tag.*logit-classifier>=0\.2\.1"):
        tags.check_tags_ready(FakeTagClip())


def test_the_guard_passes_while_the_dist_metadata_reports_0_1_0(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "0.1.0")

    tags.check_tags_ready(FakeTagClip())


def test_a_missing_library_fails_before_any_generate(classifier, monkeypatch):
    monkeypatch.setitem(sys.modules, "logit_classifier.tags", None)
    clip = FakeTagClip()

    with pytest.raises(RuntimeError, match="logit-classifier"):
        run(clip, a_tags_preset(style="Name the style."))
    assert clip.generate_calls == []


def test_every_tuning_value_is_the_measured_one():
    assert tags.VL_MAX_PIXELS == 1024 * 1024
    assert tags.STYLE_THRESHOLD == 0.9
    assert tags.STRIP_MEGAPIXELS == 0.25
    assert sorted(tags.CATEGORY_NOUNS) == ["animals", "clothing", "materials", "objects", "people", "setting"]
