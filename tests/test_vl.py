"""vl.py: vision conditioning. The slice layout the whole feature rests on is
[0]=vision_start, [1..N]=grid rows (raster), [N+1]=vision_end, [N+2..]=template tail;
these tests pin the rect->cell mapping, the shared boundary cells, the per-tile crop encode
beside the canvas slice, the canvas budget math, the fail-fast guards, and the per-tile
tensor selection (fake duck-typed clip; _convert stubbed to identity so no comfy is needed)."""
import logging

import pytest
import torch
from test_conds import FakeControl
from test_tiling import GridNoise, GridVAE

from conftest import BaseModel
from context_anchored_tile_refine import captions, grid, sampling, vl
from context_anchored_tile_refine.grid import Rect

SIGMAS = torch.linspace(1.0, 0.0, 5)  # 4 steps


# --- slice_indices: pure index math -------------------------------------------------
# Fixture geometry: canvas 192x128 px, encode 96x64 px -> merged-cell grid 3x2
# (grid_w=3, grid_h=2), n_rows=6, tail_len=4, expected_seq = 1 + 6 + 1 + 4 = 12.
CANVAS_H, CANVAS_W = 128, 192
ENC_H, ENC_W = 64, 96
N_ROWS = 6
EXPECTED_SEQ = 12
TAIL = [8, 9, 10, 11]


class Tile:
    """grid.Tile cut to what the conditioning pre-pass reads: the crop rect it slices and
    encodes by."""

    def __init__(self, rect):
        self.crop_rect = rect


def strip_tiles(*rects):
    return [Tile(rect) for rect in rects]


def layout_tiles(cols, rows, w=CANVAS_W, h=CANVAS_H, ctx=8, overlap=8):
    # REAL tiles at an exact grid size: build_layout reads only n and base off each axis
    # solution, so handing it the counts directly is the production assembler on a chosen
    # grid rather than a solver search for one.
    r = ctx + overlap
    sx = grid.AxisSolution(n=cols, base=w // cols, last=w // cols, overhead=0, r=r)
    sy = grid.AxisSolution(n=rows, base=h // rows, last=h // rows, overhead=0, r=r)
    return grid.build_layout(w, h, sx, sy, ctx, overlap).tiles


def vision(canvas=1, crop=0):
    # The [vision] table a test hands the pre-pass. Under the stubbed resample only on/off
    # matters, so the counts are 1 or 0.
    return captions.VisionSettings(canvas_tokens=canvas, crop_tokens=crop, caption_megapixels=0.786432)


def test_full_canvas_tile_selects_every_row():
    indices = vl.slice_indices(Rect(0, 0, CANVAS_W, CANVAS_H), CANVAS_H, CANVAS_W, ENC_H, ENC_W, EXPECTED_SEQ)
    assert indices == list(range(EXPECTED_SEQ))


def test_left_and_right_tiles_share_the_boundary_cell_column():
    left = vl.slice_indices(Rect(0, 0, 96, CANVAS_H), CANVAS_H, CANVAS_W, ENC_H, ENC_W, EXPECTED_SEQ)
    right = vl.slice_indices(Rect(96, 0, CANVAS_W, CANVAS_H), CANVAS_H, CANVAS_W, ENC_H, ENC_W, EXPECTED_SEQ)
    # 96 px is 1.5 cells into the 3-wide grid: floor/ceil intersection keeps the
    # partly-covered middle column in BOTH tiles (the row-space overlap band).
    assert left == [0, 1, 2, 4, 5, 7, *TAIL]
    assert right == [0, 2, 3, 5, 6, 7, *TAIL]
    shared_rows = set(left) & set(right) - {0, 7} - set(TAIL)
    assert shared_rows == {2, 5}
    # Together the tiles cover every grid row.
    assert set(left) | set(right) == set(range(EXPECTED_SEQ))


def test_rows_are_in_raster_order_and_delimiters_bracket_them():
    indices = vl.slice_indices(Rect(96, 64, CANVAS_W, CANVAS_H), CANVAS_H, CANVAS_W, ENC_H, ENC_W, EXPECTED_SEQ)
    # Bottom-right quadrant: grid row 1, columns 1..2 -> sequence rows 1+3+1=5, 1+3+2=6.
    assert indices == [0, 5, 6, 7, *TAIL]
    rows = indices[1:indices.index(1 + N_ROWS)]
    assert rows == sorted(rows)


def test_offset_places_region_tiles_in_the_full_canvas_frame():
    # Mask path: a bbox-crop tile rect plus the bbox origin must land on exactly the
    # cells the equivalent full-canvas rect selects.
    shifted = vl.slice_indices(Rect(0, 0, 96, 64), CANVAS_H, CANVAS_W, ENC_H, ENC_W, EXPECTED_SEQ, offset_x=96, offset_y=64)
    direct = vl.slice_indices(Rect(96, 64, CANVAS_W, CANVAS_H), CANVAS_H, CANVAS_W, ENC_H, ENC_W, EXPECTED_SEQ)
    assert shifted == direct == [0, 5, 6, 7, *TAIL]


def test_offset_overreach_from_padding_clamps_to_the_grid():
    # The region crop is padded to /8 before tiling, so a tile rect can overreach the
    # full canvas by up to 7px; the cell range must clamp instead of indexing past it.
    indices = vl.slice_indices(Rect(0, 0, 96 + 7, 64 + 7), CANVAS_H, CANVAS_W, ENC_H, ENC_W, EXPECTED_SEQ, offset_x=96, offset_y=64)
    assert indices == [0, 5, 6, 7, *TAIL]


def test_a_tail_free_expected_seq_leaves_the_template_tail_out():
    # expected_seq = n_rows + 2 empties the trailing range: what every block but the last
    # of a tile's positive is sliced with, so the one tail arrives last.
    indices = vl.slice_indices(Rect(0, 0, 96, CANVAS_H), CANVAS_H, CANVAS_W, ENC_H, ENC_W, N_ROWS + 2)
    assert indices == [0, 1, 2, 4, 5, 7]


# --- canvas_budget_pixels: pure budget math ------------------------------------------

def test_a_lone_tile_covering_the_canvas_is_sampled_at_its_tokens():
    tiles = strip_tiles(Rect(0, 0, CANVAS_W, CANVAS_H))
    assert vl.canvas_budget_pixels(tiles, CANVAS_H, CANVAS_W, 165) == 165 * vl.PIXELS_PER_TOKEN


def test_the_canvas_budget_grows_with_the_tile_count():
    # Four quarter tiles: each holds a quarter of the picture, so the picture is sampled
    # four times larger to hand each its tokens.
    tiles = strip_tiles(Rect(0, 0, 96, 64), Rect(96, 0, 192, 64), Rect(0, 64, 96, 128), Rect(96, 64, 192, 128))
    assert vl.canvas_budget_pixels(tiles, CANVAS_H, CANVAS_W, 165) == 4 * 165 * vl.PIXELS_PER_TOKEN


def test_the_canvas_budget_is_sized_off_the_mean_crop_area():
    # 3/4 and 1/4 of the canvas average to a half, so the sample is twice the tokens.
    tiles = strip_tiles(Rect(0, 0, 144, CANVAS_H), Rect(144, 0, CANVAS_W, CANVAS_H))
    assert vl.canvas_budget_pixels(tiles, CANVAS_H, CANVAS_W, 100) == 2 * 100 * vl.PIXELS_PER_TOKEN


def test_the_canvas_budget_is_capped_at_the_picture_cap():
    # An 8x8 tile is 1/384 of the canvas, so the uncapped sample would be 65 MP.
    tiles = strip_tiles(Rect(0, 0, 8, 8))
    assert vl.canvas_budget_pixels(tiles, CANVAS_H, CANVAS_W, 165) == vl.PICTURE_CAP_PIXELS
    assert vl.MAX_VISION_TOKENS == vl.PICTURE_CAP_PIXELS // vl.PIXELS_PER_TOKEN == 1953


def test_the_canvas_budget_needs_a_tile():
    with pytest.raises(ValueError, match="at least one tile"):
        vl.canvas_budget_pixels([], CANVAS_H, CANVAS_W, 165)


# --- fake clip: build_global_slices end to end (comfy-free) -------------------------

class FakeVLClip:
    """Duck-typed VL clip. Token stream mirrors the Krea 2 layout _encode_one
    parses: template prefix, vision_start, ONE dict image token, vision_end, tail.
    The encode is deterministic: feature value == sequence position."""

    def __init__(self, tail_len=4, seq_override=None):
        self.tail_len = tail_len
        self.seq_override = seq_override

    def tokenize(self, text, images=None, llama_template=None):
        assert text == vl.VISION_BLOCK
        assert llama_template == vl.KREA2_TEMPLATE
        assert len(images) == 1
        stream = [(10, 1.0)] * 5
        stream += [(151652, 1.0), ({"type": "image"}, 1.0), (151653, 1.0)]
        stream += [(20, 1.0)] * self.tail_len
        return {"qwen3vl_4b": [stream]}

    def encode_from_tokens_scheduled(self, tokens):
        seq = self.seq_override if self.seq_override is not None else EXPECTED_SEQ
        tensor = torch.arange(seq, dtype=torch.float32).reshape(1, seq, 1).expand(1, seq, 8).clone()
        return [[tensor, {"pooled_output": None, "attention_mask": torch.ones(1, seq)}]]


class RecordingVLClip(FakeVLClip):
    """FakeVLClip that records every picture it is handed and tags each encode with its call
    index (+100 per call), so a build has to show which encode every row came from, in
    order. `tag_pooled` gives each encode a real pooled_output as well, for the batch cat
    test alone: Krea 2 returns None there, which is what lets two blocks concatenate."""

    def __init__(self, tail_len=4, seq_override=None, tag_pooled=False):
        super().__init__(tail_len=tail_len, seq_override=seq_override)
        self.tag_pooled = tag_pooled
        self.canvases = []

    def tokenize(self, text, images=None, llama_template=None):
        self.canvases.append(images[0])
        return super().tokenize(text, images=images, llama_template=llama_template)

    def encode_from_tokens_scheduled(self, tokens):
        call = len(self.canvases) - 1
        encoded = super().encode_from_tokens_scheduled(tokens)
        encoded[0][0] += 100.0 * call
        if self.tag_pooled:
            encoded[0][1]["pooled_output"] = torch.full((1, 4), float(call))
        return encoded


@pytest.fixture
def stubbed_vl(comfy_stubs, monkeypatch):
    # Encode geometry pinned to the fixture grid whatever the budget; _convert identity so
    # the slice tensors stay inspectable. comfy_stubs serves encode_picture's interrupt check.
    monkeypatch.setattr(vl, "resample_picture", lambda source, budget: (source, ENC_H, ENC_W))
    monkeypatch.setattr(vl, "_convert", lambda cond_list: cond_list)
    return vl


def test_canvas_rows_alone_select_each_tiles_slice(stubbed_vl):
    # crop_tokens 0: the positive is the tile's slice of ONE canvas encode and nothing else,
    # which is the whole-canvas method of 1.6.1.
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)

    tiles = strip_tiles(Rect(0, 0, 96, CANVAS_H), Rect(96, 0, CANVAS_W, CANVAS_H))
    clip = RecordingVLClip()
    positives = stubbed_vl.build_global_slices(clip, source, tiles, vision(canvas=1, crop=0))
    assert len(positives) == 2
    assert [tuple(canvas.shape) for canvas in clip.canvases] == [(1, CANVAS_H, CANVAS_W, 3)]
    expected = [[0, 1, 2, 4, 5, 7, *TAIL], [0, 2, 3, 5, 6, 7, *TAIL]]
    for positive, indices in zip(positives, expected, strict=True):
        tensor, extras = positive[0]
        assert tensor.shape == (1, len(indices), 8)
        assert tensor[0, :, 0].tolist() == indices
        # The full-canvas attention mask must not survive onto a slice.
        assert "attention_mask" not in extras
        assert "pooled_output" in extras


def test_crop_rows_alone_are_every_cell_of_the_tiles_own_crop(stubbed_vl):
    # canvas_tokens 0: one encode per tile of exactly its crop's pixels, and the positive is
    # every cell of it plus the tail, out of its OWN encode (the +100 tag per call).
    source = torch.arange(CANVAS_H * CANVAS_W * 3, dtype=torch.float32).reshape(1, CANVAS_H, CANVAS_W, 3)

    tiles = strip_tiles(Rect(0, 0, 96, CANVAS_H), Rect(96, 0, CANVAS_W, CANVAS_H))
    clip = RecordingVLClip()
    positives = stubbed_vl.build_global_slices(clip, source, tiles, vision(canvas=0, crop=1))

    assert len(clip.canvases) == 2
    assert torch.equal(clip.canvases[0], source[:, :, 0:96, :])
    assert torch.equal(clip.canvases[1], source[:, :, 96:CANVAS_W, :])
    for call, positive in enumerate(positives):
        tensor, extras = positive[0]
        assert tensor[0, :, 0].tolist() == [100.0 * call + row for row in range(EXPECTED_SEQ)]
        assert "attention_mask" not in extras


def test_a_tile_concatenates_its_crop_rows_its_canvas_slice_and_one_tail(stubbed_vl):
    # Both sources on: the canvas is encoded first (call 0), then each tile's crop (calls 1
    # and 2). A tile's rows are [vision_start][its crop's cells][vision_end] out of its crop
    # encode, then [vision_start][its slice][vision_end] out of the canvas encode, then the
    # template tail ONCE, from the canvas encode. The extras are the first block's.
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)
    tiles = strip_tiles(Rect(0, 0, 96, CANVAS_H), Rect(96, 0, CANVAS_W, CANVAS_H))
    clip = RecordingVLClip()

    positives = stubbed_vl.build_global_slices(clip, source, tiles, vision(canvas=1, crop=1))

    assert [tuple(canvas.shape) for canvas in clip.canvases] == [
        (1, CANVAS_H, CANVAS_W, 3), (1, CANVAS_H, 96, 3), (1, CANVAS_H, 96, 3)]
    canvas_slices = [[0, 1, 2, 4, 5, 7], [0, 2, 3, 5, 6, 7]]
    for index, positive in enumerate(positives):
        tensor, extras = positive[0]
        crop_block = [100.0 * (index + 1) + row for row in range(N_ROWS + 2)]
        assert tensor[0, :, 0].tolist() == [*crop_block, *canvas_slices[index], *TAIL]
        assert "attention_mask" not in extras
        assert "pooled_output" in extras


def test_each_encode_is_handed_its_own_budget(comfy_stubs, monkeypatch):
    # The wiring between the [vision] table's two counts and the two sample sizes, pinned
    # end to end: the canvas at canvas_budget_pixels for its tokens, every crop at exactly
    # crop_tokens x PIXELS_PER_TOKEN.
    budgets = []
    monkeypatch.setattr(vl, "resample_picture",
                        lambda source, budget: (budgets.append(budget), (source, ENC_H, ENC_W))[1])
    monkeypatch.setattr(vl, "_convert", lambda cond_list: cond_list)
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)
    tiles = strip_tiles(Rect(0, 0, 96, CANVAS_H), Rect(96, 0, CANVAS_W, CANVAS_H))

    vl.build_global_slices(RecordingVLClip(), source, tiles, vision(canvas=165, crop=100))

    canvas_budget = vl.canvas_budget_pixels(tiles, CANVAS_H, CANVAS_W, 165)
    assert canvas_budget == 2 * 165 * vl.PIXELS_PER_TOKEN        # two half-canvas tiles
    assert budgets == [canvas_budget, 100 * vl.PIXELS_PER_TOKEN, 100 * vl.PIXELS_PER_TOKEN]


def test_the_caption_probe_is_the_smallest_copy_of_the_run(stubbed_vl):
    # The caption surface tokenizes the probe once per tile, so it is a crop copy whenever
    # the crop rows are on, and the canvas copy only when they are off.
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)
    tiles = strip_tiles(Rect(0, 0, 96, CANVAS_H), Rect(96, 0, CANVAS_W, CANVAS_H))

    _rows, probe = stubbed_vl.build_vision_rows(RecordingVLClip(), source, tiles, vision(canvas=1, crop=1))
    assert tuple(probe.shape) == (1, CANVAS_H, 96, 3)
    _rows, probe = stubbed_vl.build_vision_rows(RecordingVLClip(), source, tiles, vision(canvas=1, crop=0))
    assert tuple(probe.shape) == (1, CANVAS_H, CANVAS_W, 3)


def test_both_sources_off_is_rejected(stubbed_vl):
    with pytest.raises(ValueError, match="both 0"):
        stubbed_vl.build_global_slices(RecordingVLClip(), torch.zeros(1, CANVAS_H, CANVAS_W, 3),
                                       strip_tiles(Rect(0, 0, CANVAS_W, CANVAS_H)), vision(canvas=0, crop=0))


def test_a_region_tiles_crop_is_cut_from_the_full_image_at_the_bbox_offset(stubbed_vl):
    # Mask path: the tile indexes the bbox crop while the source is the FULL image, so its
    # crop picture is the offset rect of that image and its canvas slice the offset cells.
    source = torch.arange(CANVAS_H * CANVAS_W * 3, dtype=torch.float32).reshape(1, CANVAS_H, CANVAS_W, 3)
    tile = Tile(Rect(0, 0, 32, 32))
    clip = RecordingVLClip()

    positives = stubbed_vl.build_global_slices(clip, source, [tile], vision(canvas=1, crop=1),
                                               offset_x=96, offset_y=48)

    assert torch.equal(clip.canvases[0], source)
    assert torch.equal(clip.canvases[1], source[:, 48:80, 96:128, :])
    # 96..128 x 48..80 covers grid column 1 over both rows: cells (0,1) and (1,1).
    crop_block = [100.0 + row for row in range(N_ROWS + 2)]
    assert positives[0][0][0][0, :, 0].tolist() == [*crop_block, 0, 2, 5, 7, *TAIL]


def test_a_region_crop_that_overruns_the_image_is_clamped(stubbed_vl):
    # A region canvas padded to /8 lets a tile reach up to 7px past the image.
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)
    tile = Tile(Rect(0, 0, 16 + 7, 16 + 7))
    clip = RecordingVLClip()

    stubbed_vl.build_global_slices(clip, source, [tile], vision(canvas=0, crop=1),
                                   offset_x=CANVAS_W - 16, offset_y=CANVAS_H - 16)

    assert tuple(clip.canvases[0].shape) == (1, 16, 16, 3)


def test_batched_canvas_is_encoded_one_row_at_a_time(stubbed_vl):
    # Core's tokenizer attaches images[0] alone (qwen_vl.process_qwen2vl_images), so handing
    # over the whole [B,H,W,3] canvas would condition EVERY image on row 0's picture. Each
    # row is encoded on its own and the results are concatenated on the batch axis, in order.
    source = torch.zeros(2, CANVAS_H, CANVAS_W, 3)

    clip = RecordingVLClip(tag_pooled=True)
    positives = stubbed_vl.build_global_slices(clip, source, strip_tiles(Rect(0, 0, CANVAS_W, CANVAS_H)),
                                               vision(canvas=1, crop=0))

    assert len(clip.canvases) == 2
    assert [tuple(canvas.shape) for canvas in clip.canvases] == [(1, CANVAS_H, CANVAS_W, 3)] * 2
    tensor, extras = positives[0][0]
    assert tensor.shape == (2, EXPECTED_SEQ, 8)
    assert tensor[0, :, 0].tolist() == list(range(EXPECTED_SEQ))
    assert tensor[1, :, 0].tolist() == [100.0 + i for i in range(EXPECTED_SEQ)]
    # pooled_output is per-image, so it rides the same cat; the canvas mask is still dropped.
    assert extras["pooled_output"].shape == (2, 4)
    assert extras["pooled_output"][:, 0].tolist() == [0.0, 1.0]
    assert "attention_mask" not in extras


def test_single_row_canvas_takes_one_unconcatenated_encode(stubbed_vl):
    # B=1 must stay on the single encode with no cat at all -- the byte-for-byte path.
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)

    clip = RecordingVLClip()
    positives = stubbed_vl.build_global_slices(clip, source, strip_tiles(Rect(0, 0, CANVAS_W, CANVAS_H)),
                                               vision(canvas=1, crop=0))

    assert len(clip.canvases) == 1
    tensor, _ = positives[0][0]
    assert tensor.shape == (1, EXPECTED_SEQ, 8)
    assert tensor[0, :, 0].tolist() == list(range(EXPECTED_SEQ))


def test_build_global_slices_rejects_wrong_encoder_seq(stubbed_vl):
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)

    with pytest.raises(RuntimeError, match=f"expected {EXPECTED_SEQ}"):
        stubbed_vl.build_global_slices(FakeVLClip(seq_override=EXPECTED_SEQ + 3), source,
                                       strip_tiles(Rect(0, 0, CANVAS_W, CANVAS_H)), vision())


def test_build_global_slices_rejects_clip_whose_tokenizer_signature_refuses_images(stubbed_vl):
    # Core CLIPs swallow images= via **kwargs (comfy/sd.py CLIP.tokenize) and fall through to
    # the no-image-tokens guard; this pins the TypeError guard for third-party wrappers with a
    # strict tokenize signature.
    class StrictSignatureClip:
        def tokenize(self, text):
            return {"l": [[(1, 1.0)]]}

    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)
    with pytest.raises(RuntimeError, match="does not accept images"):
        stubbed_vl.build_global_slices(StrictSignatureClip(), source,
                                       strip_tiles(Rect(0, 0, CANVAS_W, CANVAS_H)), vision())


def test_build_global_slices_rejects_clip_without_image_tokens(stubbed_vl):
    class NoImageTokenClip(FakeVLClip):
        def tokenize(self, text, images=None, llama_template=None):
            return {"qwen3vl_4b": [[(10, 1.0)] * 8]}

    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)
    with pytest.raises(RuntimeError, match="no image tokens"):
        stubbed_vl.build_global_slices(NoImageTokenClip(), source,
                                       strip_tiles(Rect(0, 0, CANVAS_W, CANVAS_H)), vision())


# --- the pre-pass is cancellable per encode ------------------------------------------

@pytest.mark.parametrize(("settings", "checks"), [
    (vision(canvas=1, crop=1), 4),
    (vision(canvas=1, crop=0), 1),
    (vision(canvas=0, crop=1), 3),
])
def test_the_vision_pre_pass_checks_for_a_cancel_before_every_encode(stubbed_vl, comfy_stubs, settings, checks):
    # One tower pass per canvas and per tile crop, so a check per pass is what makes the
    # pre-pass cancellable at all; a source at 0 tokens runs no pass and costs no check.
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)
    before = comfy_stubs["interrupt_calls"]

    stubbed_vl.build_global_slices(RecordingVLClip(), source, layout_tiles(3, 1), settings)

    assert comfy_stubs["interrupt_calls"] - before == checks


def test_a_cancel_stops_the_vision_pre_pass_at_the_encode_it_arrives_on(stubbed_vl, monkeypatch):
    # The check has to come BEFORE the tower pass it guards, or a cancel still pays for the
    # encode it arrived on.
    import comfy.model_management

    checks = []

    def cancel_on_the_second():
        checks.append(1)
        if len(checks) == 2:
            raise RuntimeError("Processing interrupted")

    monkeypatch.setattr(comfy.model_management, "throw_exception_if_processing_interrupted",
                        cancel_on_the_second)
    clip = RecordingVLClip()

    with pytest.raises(RuntimeError, match="Processing interrupted"):
        stubbed_vl.build_global_slices(clip, torch.zeros(1, CANVAS_H, CANVAS_W, 3),
                                       layout_tiles(3, 1), vision(canvas=1, crop=1))

    assert len(clip.canvases) == 1


# --- resample_picture: real comfy resample ------------------------------------------

@pytest.mark.comfy
def test_resample_picture_snaps_to_merged_cells(comfy_env):
    # The old whole-canvas shape: 2304x3072 at 768x1024 px -> exactly 768x1024 (scale
    # 1/3), grid 24x32.
    source = torch.rand(1, 3072, 2304, 3)
    copy, enc_h, enc_w = vl.resample_picture(source, 768 * 1024)
    assert (enc_h, enc_w) == (1024, 768)
    assert enc_h % vl.MERGED_CELL == 0 and enc_w % vl.MERGED_CELL == 0
    assert copy.shape == (1, 1024, 768, 3)
    assert enc_h * enc_w == 768 * 1024


@pytest.mark.comfy
def test_a_crop_budget_of_one_hundred_tokens_is_about_one_hundred_cells(comfy_env):
    # The [vision] contract: crop_tokens x PIXELS_PER_TOKEN, snapped to whole cells. The
    # owner's 8K tile (1944x1440) lands on a 12x9 grid.
    source = torch.rand(1, 1440, 1944, 3)
    _copy, enc_h, enc_w = vl.resample_picture(source, 100 * vl.PIXELS_PER_TOKEN)
    assert (enc_h // vl.MERGED_CELL) * (enc_w // vl.MERGED_CELL) == 108


# --- through the pipeline: the VL dispatch into the sync engine -----------------------
# From 1.6.0 a vl_clip sends refine_image to the sync engine (sync.py), mask or no mask, so
# what is pinned HERE is the dispatch's own preamble — the last thing that runs before the
# engine does — driven through a REAL end-to-end run. The three tests that pinned the retired
# raster VL branch moved WITH that branch:
#   the per-tile positive swap (a distinct slice per tile, the negative untouched, the
#     caller's conds pristine) -> test_sync.py's
#     test_each_lane_guider_carries_its_own_tile_positive, asserted of the LANE guiders the
#     engine builds, because the caller's own guider is never swapped at all;
#   the pristine-conds restore after a mid-run raise -> test_sync.py's
#     test_the_callers_guider_is_untouched_after_a_lane_raises;
#   the mask path's full-image encode at the bbox offset -> test_sync.py's
#     test_the_mask_path_encodes_the_full_image_at_the_bbox_offset.
# Its own encode geometry: 256x256 -> an 8x8 merged-cell grid over the 80px test canvas, so
# neighboring tiles really do select different rows (the 3x2 grid above is coarser than the
# tiles and every tile would take every row).
PIPE_ENC = 256
PIPE_ROWS = (PIPE_ENC // vl.MERGED_CELL) ** 2
PIPE_SEQ = 1 + PIPE_ROWS + 1 + 4


class SyncPatcher:
    """comfy's ModelPatcher cut to the three members the sync engine reads."""

    def __init__(self, model):
        self.model = model
        self.load_device = torch.device("cpu")

    def get_model_object(self, name):
        return getattr(self.model, name)


class SyncModelK:
    """comfy's KSamplerX0Inpaint (samplers.py:634-643) cut to its blend contract: the released
    cells take the model's prediction and the frozen ring is restored from the clean
    `latent_image`. Both tensors are held as attributes, exactly as core's are, because they
    are what the engine's live-canvas surgery rewrites between steps."""

    def __init__(self, latent, noise, denoise_mask):
        self.latent_image = latent
        self.noise = noise
        self.denoise_mask = denoise_mask

    def __call__(self, x, sigma, **kwargs):
        return x * self.denoise_mask + self.latent_image * (1.0 - self.denoise_mask)


class VLGuider:
    """comfy's CFGGuider cut to sample()'s contract, recording what every tile saw.

    The sync engine samples each tile on its own shallow COPY of this object, and a shallow
    copy shares `seen_conds`, so the caller's list collects every LANE's conds in lane order.
    """

    def __init__(self):
        self.model_patcher = SyncPatcher(BaseModel())
        self.original_conds = {
            "positive": [{"cross_attn": torch.zeros(1, 1, 8)}],
            "negative": [{"cross_attn": torch.zeros(1, 1, 8)}],
        }
        self.seen_conds = []

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None,
               disable_pbar=False, seed=None):
        self.seen_conds.append(self.original_conds)
        model = self.model_patcher.model
        latent = model.process_latent_in(latent_image)
        x = model.model_sampling.noise_scaling(sigmas[0], noise, latent)
        out = sampler.sampler_function(
            SyncModelK(latent, noise, denoise_mask), x, sigmas, extra_args={},
            callback=callback, disable=disable_pbar, **sampler.extra_options)
        return model.process_latent_out(out)


def sync_sampler():
    # A stand-in k-diffusion function with euler's eval CADENCE and nothing else: one model
    # call per sigma step. Named so the stepper's `sample_` strip resolves it in
    # EVALS_PER_STEP — the dispatch rejects a bare object() now, before any encode.
    import comfy.samplers

    def fn(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        for step in range(int(sigmas.shape[-1]) - 1):
            x = model(x, sigmas[step], **(extra_args or {})).clone()
        return x

    fn.__name__ = "sample_euler"
    return comfy.samplers.KSAMPLER(fn, {}, {})


@pytest.fixture
def pipeline_clip(monkeypatch):
    # The clip these tests drive, plus the encode geometry pinned to match its sequence
    # length. _convert is deliberately NOT stubbed here: these tests run through
    # comfy.sampler_helpers.convert_cond (comfy_stubs') exactly like production does.
    monkeypatch.setattr(vl, "resample_picture", lambda source, budget: (source, PIPE_ENC, PIPE_ENC))
    return FakeVLClip(seq_override=PIPE_SEQ)


def _run_vl(image, guider, clip, cap=56, ctx=0, overlap=16, mask=None):
    return sampling.refine_image(
        image, guider, sync_sampler(), SIGMAS, GridVAE(), GridNoise(),
        max_tile_width=cap, max_tile_height=cap, context_anchor=ctx,
        context_overlap=overlap, mask=mask, vl_clip=clip,
    )


def test_controlnet_is_ignored_on_the_vl_path(comfy_stubs, pipeline_clip, caplog):
    # Each tile's positive is a fresh vision slice that carries no control chain, so a
    # cropped hint would reach the negative branch alone (asymmetric CFG) and every
    # positive control copy would be built and thrown away. It is dropped, and said once.
    # Skipping the CROP is not enough: core reads `control` off the cond it is left on and
    # rescales the FULL-image hint onto the tile, so the KEY itself must go from both.
    image = torch.rand(1, 80, 80, 3)
    control = FakeControl(torch.rand(1, 3, 80, 80))
    negative_cond = {"cross_attn": torch.zeros(1, 1, 8), "control": control}
    guider = VLGuider()
    guider.original_conds = {
        "positive": [{"cross_attn": torch.zeros(1, 1, 8), "control": control}],
        "negative": [negative_cond],
    }
    pristine = guider.original_conds

    with caplog.at_level(logging.WARNING):
        _run_vl(image, guider, pipeline_clip)

    assert control.copies == 0                     # no per-tile control copy is built at all
    for seen in guider.seen_conds:
        assert "control" not in seen["positive"][0]
        assert "control" not in seen["negative"][0]
        assert torch.equal(seen["negative"][0]["cross_attn"], negative_cond["cross_attn"])
    # The caller's own cond dicts come out exactly as they went in.
    assert guider.original_conds is pristine
    assert negative_cond["control"] is control
    assert "ControlNet" in caplog.text


def test_the_vl_dispatch_rejects_a_sampler_the_engine_cannot_time(comfy_stubs, pipeline_clip):
    # Fail fast AT the dispatch, before any VAE or VL encode: the stepper's own rejection
    # would otherwise arrive one whole-canvas encode and one canvas of tile encodes later.
    vae, noise = GridVAE(), GridNoise()

    with pytest.raises(ValueError, match="only supports standard"):
        sampling.refine_image(
            torch.rand(1, 80, 80, 3), VLGuider(), object(), SIGMAS, vae, noise,
            max_tile_width=56, max_tile_height=56, context_anchor=0, context_overlap=16,
            vl_clip=pipeline_clip)

    assert vae.encode_calls == [] and noise.calls == []


def test_the_vl_dispatch_names_the_sampler_the_widget_offered(comfy_stubs, pipeline_clip):
    # The all-in-one node builds its SAMPLER from a name widget, and core's sampler_object
    # wraps several names in a private function (dpm_fast -> dpm_fast_function), so the
    # rejection has to resolve off the NAME threaded through — not off the object built from
    # it. The object here is a perfectly supported one, so only the name can raise.
    with pytest.raises(ValueError, match="'dpm_fast' is not supported"):
        sampling.refine_image(
            torch.rand(1, 80, 80, 3), VLGuider(), sync_sampler(), SIGMAS, GridVAE(), GridNoise(),
            max_tile_width=56, max_tile_height=56, context_anchor=0, context_overlap=16,
            vl_clip=pipeline_clip, sampler_name="dpm_fast")


def test_the_vl_dispatch_rejects_a_schedule_the_lanes_cannot_share(comfy_stubs, pipeline_clip):
    # Every lane steps ONE shared schedule, so a non-monotone one (or one that never reaches
    # sigma 0) is named here rather than mistiming the barrier later.
    vae, noise = GridVAE(), GridNoise()

    with pytest.raises(ValueError, match="strictly decreasing and end at 0"):
        sampling.refine_image(
            torch.rand(1, 80, 80, 3), VLGuider(), sync_sampler(),
            torch.tensor([1.0, 0.5, 0.7, 0.0]), vae, noise, max_tile_width=56,
            max_tile_height=56, context_anchor=0, context_overlap=16, vl_clip=pipeline_clip)

    assert vae.encode_calls == [] and noise.calls == []
