"""captions.py: the two caption conditioning surfaces of the VL nodes.

What is pinned here: the settled instruction/token pair each vlm_method asks for, the
reasoning-turn strip, that the VLM reads a 384-budget COPY and never the sampled tile,
per-batch-row captioning, the mask-path framing (captions describe the region crop, the
vision encode reads the full image at the bbox offset), the cancellable pre-pass, and every
fail-fast guard. A duck-typed clip stands in for the VL text encoder; no comfy install and
no model are needed.
"""
import pytest
import torch
from test_vl import VLGuider

from context_anchored_tile_refine import captions, sampling, vl
from context_anchored_tile_refine.grid import Rect

SIGMAS = torch.linspace(1.0, 0.0, 5)  # 4 steps

# Fixture geometry, mirroring test_vl.py: canvas 192x128 px, encode 96x64 px -> a 3x2
# merged-cell grid, n_rows = 6.
CANVAS_H, CANVAS_W = 128, 192
ENC_H, ENC_W = 64, 96
N_ROWS = 6
TAIL = 4


class Tile:
    def __init__(self, rect):
        self.crop_rect = rect


class FakeCaptionClip:
    """Duck-typed VL clip with a text generator.

    The token stream mirrors the Krea 2 layout captions._tokenize_images parses (prefix,
    vision_start, ONE dict image token, vision_end, tail), and the tail grows with the text
    so two different captions really do produce two different sequence lengths. Every
    tokenize/generate/encode is recorded, and the encode is deterministic: feature value ==
    sequence position.
    """

    def __init__(self, n_rows=N_ROWS, tail_len=TAIL, answer=None, seq_override=None):
        self.n_rows = n_rows
        self.tail_len = tail_len
        self.answer = answer if answer is not None else (lambda image, instruction: "<think>weighing it up</think>a plain caption")
        self.seq_override = seq_override
        self.tokenize_calls = []
        self.generate_calls = []
        self.encoded = []
        self._answers = {}

    def tokenize(self, text, images=None, llama_template=None, thinking=None):
        image = None if images is None else images[0]
        self.tokenize_calls.append({"text": text, "image": image, "llama_template": llama_template, "thinking": thinking})
        body = text[len(vl.VISION_BLOCK):] if text.startswith(vl.VISION_BLOCK) else text
        tail_rows = self.tail_len + len(body.split())
        stream = [(10, 1.0)] * 5
        if image is not None:
            stream += [(151652, 1.0), ({"type": "image"}, 1.0), (151653, 1.0)]
        stream += [(20, 1.0)] * tail_rows
        # "_probe" rides behind the stream key so the production `next(iter(tokens))` still
        # picks the stream; it is what lets generate/encode see what they were handed.
        return {"qwen3vl_4b": [stream], "_probe": (text, image, tail_rows)}

    def generate(self, tokens, **kwargs):
        text, image, _tail = tokens["_probe"]
        self.generate_calls.append({"text": text, "image": image, **kwargs})
        handle = len(self.generate_calls)
        self._answers[handle] = self.answer(image, text)
        return handle

    def decode(self, token_ids, skip_special_tokens=True):
        return self._answers[token_ids]

    def encode_from_tokens_scheduled(self, tokens):
        text, image, tail_rows = tokens["_probe"]
        self.encoded.append(text)
        if self.seq_override is not None:
            seq = self.seq_override
        elif image is not None:
            seq = 1 + self.n_rows + 1 + tail_rows
        else:
            seq = tail_rows                       # a text-only encode is only its own rows
        tensor = torch.arange(seq, dtype=torch.float32).reshape(1, seq, 1).expand(1, seq, 8).clone()
        return [[tensor, {"pooled_output": torch.zeros(1, 4), "attention_mask": torch.ones(1, seq)}]]


@pytest.fixture
def stubbed_slices(monkeypatch):
    # Encode geometry pinned to the fixture grid; _convert identity so the slice tensors
    # stay inspectable without comfy.
    monkeypatch.setattr(vl, "resample_for_global", lambda source: (source, ENC_H, ENC_W))
    monkeypatch.setattr(vl, "_convert", lambda cond_list: cond_list)


# --- the settled instructions --------------------------------------------------------

def test_settled_instructions_are_the_ab_settled_strings():
    # A/B-settled wording, EU spelling included: the owner compared US against EU in ComfyUI
    # and US dropped detail. Any reflow, reword or Americanisation is a silent quality change,
    # so the exact strings are pinned here rather than merely described.
    assert captions.GROUP_CLAUSE == "Name whole objects and count repeated objects as one entry."
    assert captions.SETTLED_POSITION_INSTRUCTION == (
        "List up to eight main things in this image, one per line, each a short phrase naming "
        "the thing, its position in the frame, and how much of it shows. "
        "Name whole objects and count repeated objects as one entry.")
    assert captions.SETTLED_RICH_INSTRUCTION == (
        "Describe this image, one short line per part. Start with the overall style, palette "
        "and lighting. Then say what fills the left, the centre, the right, the top and the "
        "bottom, giving each part its own description with what is there, its colour and what "
        "its surface is made of.")
    assert captions.RICH_GROUPED_INSTRUCTION == (
        captions.SETTLED_RICH_INSTRUCTION + " " + captions.GROUP_CLAUSE)
    assert "centre" in captions.SETTLED_RICH_INSTRUCTION
    assert "colour" in captions.SETTLED_RICH_INSTRUCTION
    assert captions.SETTLED_POSITION_MAX_TOKENS == 512
    assert captions.SETTLED_RICH_MAX_TOKENS == 768


def test_each_caption_method_asks_its_settled_question():
    # The pairing is the decision, not a detail: the position wording complements the vision
    # rows, and the GROUPED rich wording is what the captions-only surface ships (the owner's
    # explicit call on 17_CaptionOnly+Group_Lead_s42_v3, taken against the lab score).
    assert captions.CAPTION_INSTRUCTIONS[captions.VLM_METHOD_VISION_CAPTIONS] == (
        captions.SETTLED_POSITION_INSTRUCTION, 512)
    assert captions.CAPTION_INSTRUCTIONS[captions.VLM_METHOD_CAPTIONS] == (
        captions.RICH_GROUPED_INSTRUCTION, 768)
    # "vision tokens" never reaches the VLM at all, so it has no instruction.
    assert captions.VLM_METHOD_VISION not in captions.CAPTION_INSTRUCTIONS


# --- strip_thinking / clean_caption ---------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("plain answer", "plain answer"),
    ("  padded  ", "padded"),
    ("<think>reasoning here</think>the answer", "the answer"),
    ("<think>a</think>mid<think>b</think>end", "midend"),
    # A stray close with no open (the model reopened mid-answer): keep what follows the LAST
    # close, which is core's own rule in TextGenerateLTX2Prompt.
    ("<think>one</think>middle</think>the answer", "the answer"),
    # Unbalanced open with no close at all: the tags go, the text stays — there is no
    # reasoning boundary to cut on, and dropping the whole answer would be worse.
    ("<think>never closed", "never closed"),
])
def test_strip_thinking(raw, expected):
    # Mandatory, not cosmetic: the caption is encoded as text, so an unstripped block reaches
    # the DiT as hundreds of tokens of the model talking to itself.
    assert captions.strip_thinking(raw) == expected


def test_clean_caption_returns_the_original_string_when_nothing_is_removed():
    text = "**bold** opener\nsecond line\nthird line"
    assert captions.clean_caption(text) is text


@pytest.mark.parametrize("raw,expected", [
    ("**Answer:**\na tree, left", "a tree, left"),
    ("a tree, left\nWait, I need to rephrase.", "a tree, left"),
    ("a tree, left\na tree, left.\na cart, right", "a tree, left\na cart, right"),
])
def test_clean_caption_cuts_headings_meta_and_exact_repeats(raw, expected):
    assert captions.clean_caption(raw) == expected


# --- resample_for_vl ------------------------------------------------------------------

def test_resample_for_vl_uses_a_384_budget_copy_not_the_tile(comfy_stubs):
    # Prime directive 1: a sampled tile is never resampled. What the VLM reads is a separate
    # area-resampled COPY at 384x384 total pixels, aspect preserved.
    tile_pixels = torch.rand(1, 512, 1024, 3)

    vl_input = captions.resample_for_vl(tile_pixels)

    shape, width, height, method, crop = comfy_stubs["common_upscale_calls"][-1]
    assert shape == (1, 3, 512, 1024)          # channels-first, the crop itself
    assert (width, height) == (543, 272)       # sqrt(384*384 / (1024*512)) preserved 2:1
    assert abs(width * height - captions.VL_INPUT_BUDGET) / captions.VL_INPUT_BUDGET < 0.01
    assert (method, crop) == ("area", "disabled")
    assert vl_input.shape == (1, 272, 543, 3)
    assert vl_input is not tile_pixels
    assert vl_input.shape != tile_pixels.shape


# --- generate_caption -----------------------------------------------------------------

def test_generate_caption_asks_with_thinking_and_strips_the_reasoning():
    clip = FakeCaptionClip(answer=lambda image, instruction: "<think>hmm</think>a fox, centre")

    text = captions.generate_caption(clip, torch.zeros(1, 8, 8, 3),
                                     captions.SETTLED_POSITION_INSTRUCTION, 512)

    assert text == "a fox, centre"
    call = clip.tokenize_calls[0]
    assert call["text"] == captions.SETTLED_POSITION_INSTRUCTION
    assert call["thinking"] is True
    assert clip.generate_calls == [{
        "text": captions.SETTLED_POSITION_INSTRUCTION,
        "image": call["image"],
        "do_sample": False,
        "max_length": 512,
        "repetition_penalty": 1.05,
    }]


def test_generate_caption_falls_back_through_sampling_then_a_simpler_question():
    # A stop-token-degenerate crop answers empty; the settled chain retries with sampling on,
    # then re-asks a plainer question, before giving up.
    answers = ["", "", "<think>x</think>a wall of tools"]

    def answer(image, instruction):
        return answers[min(len(answers) - 1, len(clip.generate_calls) - 1)]

    clip = FakeCaptionClip(answer=answer)
    text = captions.generate_caption(clip, torch.zeros(1, 8, 8, 3), "describe", 256)

    assert text == "a wall of tools"
    assert [c["do_sample"] for c in clip.generate_calls] == [False, True, False]
    assert clip.generate_calls[1]["seed"] == 42
    assert clip.tokenize_calls[-1]["text"] == "Write one short sentence describing this image."
    assert clip.tokenize_calls[-1]["thinking"] is False


def test_generate_caption_raises_when_every_fallback_is_empty():
    clip = FakeCaptionClip(answer=lambda image, instruction: "")
    with pytest.raises(RuntimeError, match="empty answer after every fallback"):
        captions.generate_caption(clip, torch.zeros(1, 8, 8, 3), "describe", 256)


def test_generate_caption_rejects_a_clip_that_cannot_generate():
    class NoGeneratorClip:
        def tokenize(self, text, images=None, **kwargs):
            return {"l": [[(1, 1.0)]]}

    with pytest.raises(RuntimeError, match="cannot generate text"):
        captions.generate_caption(NoGeneratorClip(), torch.zeros(1, 8, 8, 3), "describe", 256)


def test_generate_caption_rejects_a_tokenizer_that_refuses_images():
    class StrictSignatureClip(FakeCaptionClip):
        def tokenize(self, text):
            return {"l": [[(1, 1.0)]]}

    with pytest.raises(RuntimeError, match="does not accept images"):
        captions.generate_caption(StrictSignatureClip(), torch.zeros(1, 8, 8, 3), "describe", 256)


def test_generate_caption_rejects_a_clip_without_image_tokens():
    class NoImageTokenClip(FakeCaptionClip):
        def tokenize(self, text, images=None, **kwargs):
            return {"qwen3vl_4b": [[(10, 1.0)] * 8], "_probe": (text, None, 4)}

    with pytest.raises(RuntimeError, match="no image tokens"):
        captions.generate_caption(NoImageTokenClip(), torch.zeros(1, 8, 8, 3), "describe", 256)


# --- generate_tile_captions -----------------------------------------------------------

def test_each_tile_is_captioned_from_its_own_crop(comfy_stubs, monkeypatch):
    # Identity resample so the crop the VLM is handed stays readable; the 384 budget is
    # covered by its own test above.
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels: pixels)
    source = torch.rand(1, CANVAS_H, CANVAS_W, 3)
    tiles = [Tile(Rect(0, 0, 96, CANVAS_H)), Tile(Rect(96, 0, CANVAS_W, CANVAS_H))]
    clip = FakeCaptionClip(answer=lambda image, instruction: f"tile at {float(image[0, 0, 0, 0]):.6f}")

    out = captions.generate_tile_captions(clip, source, tiles, "describe", 256)

    assert out == [[f"tile at {float(source[0, 0, 0, 0]):.6f}"],
                   [f"tile at {float(source[0, 0, 96, 0]):.6f}"]]
    handed = [call["image"] for call in clip.tokenize_calls]
    assert torch.equal(handed[0], source[:, :, 0:96, :])
    assert torch.equal(handed[1], source[:, :, 96:CANVAS_W, :])


def test_a_batch_is_captioned_per_row_from_its_own_pixels(comfy_stubs, monkeypatch):
    # Core's tokenizer attaches images[0] alone, so a whole [B,H,W,3] crop would describe
    # every row with row 0's picture. Row b must be read from row b.
    monkeypatch.setattr(captions, "resample_for_vl", lambda pixels: pixels)
    source = torch.zeros(2, 16, 16, 3)
    source[1] = 1.0
    tiles = [Tile(Rect(0, 0, 16, 16))]
    clip = FakeCaptionClip(answer=lambda image, instruction: f"row {int(image[0, 0, 0, 0])}")

    out = captions.generate_tile_captions(clip, source, tiles, "describe", 256)

    assert out == [["row 0", "row 1"]]
    assert [tuple(call["image"].shape) for call in clip.tokenize_calls] == [(1, 16, 16, 3)] * 2


def test_the_caption_pre_pass_is_cancellable_and_reports_progress(comfy_stubs):
    # One clip.generate per tile per row at up to 768 tokens: without an interrupt check and
    # a ProgressBar the node is uncancellable and silent until the first tile samples.
    source = torch.rand(2, 16, 16, 3)
    tiles = [Tile(Rect(0, 0, 16, 16)), Tile(Rect(0, 0, 16, 16)), Tile(Rect(0, 0, 16, 16))]

    captions.generate_tile_captions(FakeCaptionClip(), source, tiles, "describe", 256)

    assert comfy_stubs["interrupt_calls"] == len(tiles)
    pbar = comfy_stubs["progress_bars"][-1]
    assert pbar.total == len(tiles) * 2
    assert [update[0] for update in pbar.updates] == [1, 2, 3, 4, 5, 6]


def test_generated_captions_are_cleaned_before_they_are_returned(comfy_stubs):
    # clean_caption runs on the way out (caption_clean=True on every settled render), so the
    # DiT never reads the VLM's own headings or its repetition loop as content.
    clip = FakeCaptionClip(answer=lambda image, instruction: "**Answer:**\na fox, centre")

    out = captions.generate_tile_captions(clip, torch.rand(1, 16, 16, 3), [Tile(Rect(0, 0, 16, 16))],
                                          "describe", 256)

    assert out == [["a fox, centre"]]


# --- build_caption_conds --------------------------------------------------------------

def test_caption_conds_encode_each_tiles_caption_as_plain_text(stubbed_slices):
    clip = FakeCaptionClip()

    conds = captions.build_caption_conds(clip, [["a fox"], ["a cart, right"]])

    assert clip.encoded == ["a fox", "a cart, right"]
    assert all(call["image"] is None for call in clip.tokenize_calls)
    # Text-only: no vision grid rows at all, so the two encodes differ in length.
    assert [cond[0][0].shape[1] for cond in conds] == [TAIL + 2, TAIL + 3]


def test_caption_conds_take_the_single_row_of_each_tile(stubbed_slices):
    # [tile][row] with exactly ONE row: refine_image refines one picture at a time, so a
    # tile's positive is one plain text encode with nothing concatenated across rows.
    clip = FakeCaptionClip()

    conds = captions.build_caption_conds(clip, [["a fox here"]])

    tensor, extras = conds[0][0]
    assert tensor.shape == (1, TAIL + 3, 8)
    assert extras["pooled_output"].shape == (1, 4)


# --- build_slice_caption_conds --------------------------------------------------------

def test_slice_caption_conds_keep_each_tiles_rows_plus_its_caption(stubbed_slices):
    clip = FakeCaptionClip()
    tiles = [Tile(Rect(0, 0, 96, CANVAS_H)), Tile(Rect(96, 0, CANVAS_W, CANVAS_H))]
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)

    conds = captions.build_slice_caption_conds(clip, source, tiles, [["a fox"], ["a cart"]])

    # The caption rides INSIDE the vision encode, so one whole-canvas encode per caption.
    assert clip.encoded == [vl.VISION_BLOCK + "a fox", vl.VISION_BLOCK + "a cart"]
    assert all(call["llama_template"] == vl.KREA2_TEMPLATE for call in clip.tokenize_calls)
    expected_seq = 1 + N_ROWS + 1 + TAIL + 2
    for cond, tile in zip(conds, tiles, strict=True):
        indices = vl.slice_indices(tile.crop_rect, CANVAS_H, CANVAS_W, ENC_H, ENC_W, expected_seq)
        tensor, extras = cond[0]
        assert tensor[0, :, 0].tolist() == indices
        # The tail carries the caption, so both tiles keep every caption row.
        assert indices[-(TAIL + 2):] == list(range(1 + N_ROWS + 1, expected_seq))
        assert "attention_mask" not in extras


def test_slice_caption_conds_reuse_one_encode_per_distinct_caption(stubbed_slices):
    clip = FakeCaptionClip()
    tiles = [Tile(Rect(0, 0, 96, CANVAS_H)), Tile(Rect(96, 0, CANVAS_W, CANVAS_H))]

    captions.build_slice_caption_conds(clip, torch.zeros(1, CANVAS_H, CANVAS_W, 3), tiles,
                                       [["a fox"], ["a fox"]])

    assert clip.encoded == [vl.VISION_BLOCK + "a fox"]


def test_slice_caption_conds_offset_region_tiles_into_the_full_canvas_frame(stubbed_slices):
    # Mask path: the tiles index the bbox crop while the encode reads the FULL image, so the
    # rects need the bbox origin added — the same framing as vl.build_global_slices.
    clip = FakeCaptionClip()
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)

    shifted = captions.build_slice_caption_conds(clip, source, [Tile(Rect(0, 0, 96, 64))],
                                                 [["a fox"]], offset_x=96, offset_y=64)
    direct = captions.build_slice_caption_conds(clip, source, [Tile(Rect(96, 64, CANVAS_W, CANVAS_H))],
                                                [["a fox"]])

    assert shifted[0][0][0][0, :, 0].tolist() == direct[0][0][0][0, :, 0].tolist()


def test_slice_caption_conds_encode_one_picture_at_a_time(stubbed_slices):
    # Core's tokenizer attaches images[0] alone, so the encode is always handed ONE picture.
    # Under refine_image's picture loop that is the whole batch, and nothing is concatenated.
    clip = FakeCaptionClip()
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)

    conds = captions.build_slice_caption_conds(clip, source, [Tile(Rect(0, 0, CANVAS_W, CANVAS_H))],
                                               [["a fox"]])

    assert [tuple(call["image"].shape) for call in clip.tokenize_calls] == [(1, CANVAS_H, CANVAS_W, 3)]
    tensor, extras = conds[0][0]
    assert tensor.shape[0] == 1
    assert extras["pooled_output"].shape == (1, 4)


def test_slice_caption_conds_reject_a_caption_count_that_is_not_the_batch(stubbed_slices):
    # A row without its own caption would be conditioned on another row's picture.
    clip = FakeCaptionClip()
    source = torch.zeros(2, CANVAS_H, CANVAS_W, 3)

    with pytest.raises(RuntimeError, match="captioned a different number of times"):
        captions.build_slice_caption_conds(clip, source, [Tile(Rect(0, 0, CANVAS_W, CANVAS_H))],
                                           [["a fox"]])


def test_slice_caption_conds_reject_an_encoder_whose_layout_disagrees(stubbed_slices):
    clip = FakeCaptionClip(seq_override=1 + N_ROWS + 1 + TAIL + 99)

    with pytest.raises(RuntimeError, match="slice\\+caption encode has"):
        captions.build_slice_caption_conds(clip, torch.zeros(1, CANVAS_H, CANVAS_W, 3),
                                           [Tile(Rect(0, 0, CANVAS_W, CANVAS_H))], [["a fox"]])


# --- through the pipeline: the three-way branch in _refine_tiles ----------------------
# 80x80 image at cap 56 / overlap 16 solves to a 2x2 grid, so tiles really do select
# different rows of the 256x256 (8x8 merged cell) encode geometry pinned below.
PIPE_ENC = 256
PIPE_ROWS = (PIPE_ENC // vl.MERGED_CELL) ** 2


@pytest.fixture
def pipeline_clip(monkeypatch):
    monkeypatch.setattr(vl, "resample_for_global", lambda source: (source, PIPE_ENC, PIPE_ENC))
    return FakeCaptionClip(n_rows=PIPE_ROWS)


def _run(image, guider, clip, vlm_method, mask=None, ctx=0):
    return sampling.refine_image(
        image, guider, object(), SIGMAS, *_engine(), max_tile_width=56, max_tile_height=56,
        context_anchor=ctx, context_overlap=16, mask=mask, vl_clip=clip, vlm_method=vlm_method,
    )


def _engine():
    from test_tiling import GridNoise, GridVAE
    return GridVAE(), GridNoise()


def test_vision_tokens_is_the_default_and_still_routes_through_build_global_slices(comfy_stubs, pipeline_clip, monkeypatch):
    # The byte-identical guarantee is structural: the default arm calls the unchanged
    # vl.build_global_slices and never touches the VLM at all.
    image = torch.rand(1, 80, 80, 3)
    seen = []
    real_build = vl.build_global_slices
    monkeypatch.setattr(vl, "build_global_slices", lambda *a, **k: (seen.append(k), real_build(*a, **k))[1])

    default = sampling.refine_image(image, VLGuider(), object(), SIGMAS, *_engine(),
                                    max_tile_width=56, max_tile_height=56, context_anchor=0,
                                    context_overlap=16, vl_clip=pipeline_clip)
    explicit = _run(image, VLGuider(), pipeline_clip, "vision tokens")

    assert torch.equal(default, explicit)
    assert len(seen) == 2 and all(call == {"offset_x": 0, "offset_y": 0} for call in seen)
    assert pipeline_clip.generate_calls == []


@pytest.mark.parametrize("method,instruction,max_length", [
    ("vision tokens and captions", captions.SETTLED_POSITION_INSTRUCTION, 512),
    ("captions", captions.RICH_GROUPED_INSTRUCTION, 768),
])
def test_each_caption_method_reaches_the_vlm_with_its_settled_question(comfy_stubs, pipeline_clip, method, instruction, max_length):
    image = torch.rand(1, 80, 80, 3)

    _run(image, VLGuider(), pipeline_clip, method)

    assert len(pipeline_clip.generate_calls) == 4          # one per tile of the 2x2 grid
    assert {call["text"] for call in pipeline_clip.generate_calls} == {instruction}
    assert {call["max_length"] for call in pipeline_clip.generate_calls} == {max_length}
    assert all(call["thinking"] is True for call in pipeline_clip.tokenize_calls[:4])


def test_captions_only_gives_each_tile_a_text_only_positive(comfy_stubs, pipeline_clip):
    # The caption IS the tile's whole positive: no vision block, no grid rows.
    image = torch.rand(1, 80, 80, 3)
    guider = VLGuider()
    pristine = guider.original_conds

    _run(image, guider, pipeline_clip, "captions")

    assert len(guider.seen_conds) == 4
    assert pipeline_clip.encoded == ["a plain caption"] * 4     # one text encode per tile
    for seen in guider.seen_conds:
        assert seen["positive"][0]["cross_attn"].shape[1] == TAIL + 3
    assert guider.original_conds is pristine


def test_vision_and_captions_slices_the_encode_that_carries_the_caption(comfy_stubs, pipeline_clip):
    image = torch.rand(1, 80, 80, 3)
    guider = VLGuider()

    _run(image, guider, pipeline_clip, "vision tokens and captions")

    assert pipeline_clip.encoded == [vl.VISION_BLOCK + "a plain caption"]   # cached per caption
    expected_seq = 1 + PIPE_ROWS + 1 + TAIL + 3
    from test_tiling import _layout
    layout = _layout(80, 80, 56, 56, overlap=16)
    for tile, seen in zip(layout.tiles, guider.seen_conds, strict=True):
        indices = vl.slice_indices(tile.crop_rect, 80, 80, PIPE_ENC, PIPE_ENC, expected_seq)
        assert seen["positive"][0]["cross_attn"][0, :, 0].tolist() == indices


def test_the_mask_path_captions_the_region_crop_and_encodes_the_full_image(comfy_stubs, pipeline_clip):
    # Two different canvases, deliberately: the caption describes the tile the sampler
    # actually runs (the bbox crop), while the vision encode still reads the whole image at
    # the bbox origin so the region stays globally informed.
    image = torch.rand(1, 80, 80, 3)
    mask = torch.zeros(1, 80, 80)
    mask[:, 16:64, 16:64] = 1.0

    _run(image, VLGuider(), pipeline_clip, "vision tokens and captions", mask=mask, ctx=8)

    y0, y1, x0, x1 = sampling._expand_snap_clamp(sampling._mask_bbox(mask >= 0.5), 8, 80, 80)
    assert (y0, y1, x0, x1) == (8, 72, 8, 72)
    caption_inputs = [call["image"] for call in pipeline_clip.tokenize_calls if call["llama_template"] is None]
    encode_inputs = [call["image"] for call in pipeline_clip.tokenize_calls if call["llama_template"] is not None]
    # comfy_stubs' common_upscale returns the resampled COPY, so only the shape is readable —
    # which is the point: what the VLM reads is never the sampled tile.
    assert caption_inputs and all(tuple(x.shape[1:3]) != (y1 - y0, x1 - x0) for x in caption_inputs)
    assert encode_inputs and all(torch.equal(x, image) for x in encode_inputs)


@pytest.mark.parametrize("method", ["captions", "vision tokens and captions"])
def test_a_two_picture_batch_with_different_length_captions_completes(comfy_stubs, pipeline_clip, method):
    # The case that used to raise: every picture is captioned from its OWN pixels, two
    # captions rarely tokenize to the same length, and one shared cond could not concatenate
    # them. Refining one picture at a time removes the concatenation entirely.
    image = torch.rand(2, 80, 80, 3)
    pipeline_clip.answer = lambda img, instruction: (
        "a fox" if len(pipeline_clip.generate_calls) <= 4 else "a cart in a very busy market")

    out = _run(image, VLGuider(), pipeline_clip, method)

    assert out.shape == (2, 80, 80, 3)
    assert len(pipeline_clip.generate_calls) == 8            # 4 tiles x 2 pictures
    # Both captions reach the encoder; the slice surface carries them inside the vision block.
    prefix = "" if method == "captions" else vl.VISION_BLOCK
    assert set(pipeline_clip.encoded) == {prefix + "a fox", prefix + "a cart in a very busy market"}


def test_an_unknown_vlm_method_is_rejected_by_name(comfy_stubs, pipeline_clip):
    with pytest.raises(ValueError, match="vlm_method must be one of"):
        _run(torch.rand(1, 80, 80, 3), VLGuider(), pipeline_clip, "vision")
