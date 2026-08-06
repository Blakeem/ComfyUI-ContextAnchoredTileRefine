"""vl.py: global-slice conditioning. The slice layout the whole feature rests on is
[0]=vision_start, [1..N]=grid rows (raster), [N+1]=vision_end, [N+2..]=template tail;
these tests pin the rect->cell mapping, the shared boundary cells, the fail-fast
guards, and the per-tile tensor selection (fake duck-typed clip; _convert stubbed to
identity so no comfy is needed)."""
import logging

import pytest
import torch

from context_anchored_tile_refine import sampling, vl
from context_anchored_tile_refine.grid import Rect
from test_conds import FakeControl
from test_tiling import GridGuider, GridNoise, GridVAE, _layout

SIGMAS = torch.linspace(1.0, 0.0, 5)  # 4 steps


# --- slice_indices: pure index math -------------------------------------------------
# Fixture geometry: canvas 192x128 px, encode 96x64 px -> merged-cell grid 3x2
# (grid_w=3, grid_h=2), n_rows=6, tail_len=4, expected_seq = 1 + 6 + 1 + 4 = 12.
CANVAS_H, CANVAS_W = 128, 192
ENC_H, ENC_W = 64, 96
N_ROWS = 6
EXPECTED_SEQ = 12
TAIL = [8, 9, 10, 11]


def test_full_canvas_tile_selects_every_row():
    indices = vl.slice_indices(Rect(0, 0, CANVAS_W, CANVAS_H), CANVAS_H, CANVAS_W, ENC_H, ENC_W, EXPECTED_SEQ)
    assert indices == list(range(EXPECTED_SEQ))


def test_left_and_right_tiles_share_the_boundary_cell_column():
    left = vl.slice_indices(Rect(0, 0, 96, CANVAS_H), CANVAS_H, CANVAS_W, ENC_H, ENC_W, EXPECTED_SEQ)
    right = vl.slice_indices(Rect(96, 0, CANVAS_W, CANVAS_H), CANVAS_H, CANVAS_W, ENC_H, ENC_W, EXPECTED_SEQ)
    # 96 px is 1.5 cells into the 3-wide grid: floor/ceil intersection keeps the
    # partly-covered middle column in BOTH tiles (the row-space overlap band).
    assert left == [0, 1, 2, 4, 5, 7] + TAIL
    assert right == [0, 2, 3, 5, 6, 7] + TAIL
    shared_rows = set(left) & set(right) - {0, 7} - set(TAIL)
    assert shared_rows == {2, 5}
    # Together the tiles cover every grid row.
    assert set(left) | set(right) == set(range(EXPECTED_SEQ))


def test_rows_are_in_raster_order_and_delimiters_bracket_them():
    indices = vl.slice_indices(Rect(96, 64, CANVAS_W, CANVAS_H), CANVAS_H, CANVAS_W, ENC_H, ENC_W, EXPECTED_SEQ)
    # Bottom-right quadrant: grid row 1, columns 1..2 -> sequence rows 1+3+1=5, 1+3+2=6.
    assert indices == [0, 5, 6, 7] + TAIL
    rows = indices[1:indices.index(1 + N_ROWS)]
    assert rows == sorted(rows)


def test_offset_places_region_tiles_in_the_full_canvas_frame():
    # Mask path: a bbox-crop tile rect plus the bbox origin must land on exactly the
    # cells the equivalent full-canvas rect selects.
    shifted = vl.slice_indices(Rect(0, 0, 96, 64), CANVAS_H, CANVAS_W, ENC_H, ENC_W, EXPECTED_SEQ, offset_x=96, offset_y=64)
    direct = vl.slice_indices(Rect(96, 64, CANVAS_W, CANVAS_H), CANVAS_H, CANVAS_W, ENC_H, ENC_W, EXPECTED_SEQ)
    assert shifted == direct == [0, 5, 6, 7] + TAIL


def test_offset_overreach_from_padding_clamps_to_the_grid():
    # The region crop is padded to /8 before tiling, so a tile rect can overreach the
    # full canvas by up to 7px; the cell range must clamp instead of indexing past it.
    indices = vl.slice_indices(Rect(0, 0, 96 + 7, 64 + 7), CANVAS_H, CANVAS_W, ENC_H, ENC_W, EXPECTED_SEQ, offset_x=96, offset_y=64)
    assert indices == [0, 5, 6, 7] + TAIL


# --- fake clip: build_global_slices end to end (comfy-free) -------------------------

class FakeVLClip:
    """Duck-typed VL clip. Token stream mirrors the Krea 2 layout _encode_canvas
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
    """FakeVLClip that records every canvas it is handed and tags each encode with its call
    index (+100 per row), so a batched build has to show one encode per row, in row order."""

    def __init__(self, tail_len=4):
        super().__init__(tail_len=tail_len)
        self.canvases = []

    def tokenize(self, text, images=None, llama_template=None):
        self.canvases.append(images[0])
        return super().tokenize(text, images=images, llama_template=llama_template)

    def encode_from_tokens_scheduled(self, tokens):
        row = len(self.canvases) - 1
        encoded = super().encode_from_tokens_scheduled(tokens)
        encoded[0][0] += 100.0 * row
        encoded[0][1]["pooled_output"] = torch.full((1, 4), float(row))
        return encoded


@pytest.fixture
def stubbed_vl(monkeypatch):
    # Encode geometry pinned to the fixture grid; _convert identity so the slice
    # tensors stay inspectable without comfy.
    monkeypatch.setattr(vl, "resample_for_global", lambda source: (source, ENC_H, ENC_W))
    monkeypatch.setattr(vl, "_convert", lambda cond_list: cond_list)
    return vl


def test_build_global_slices_selects_each_tiles_rows(stubbed_vl):
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)

    class Tile:
        def __init__(self, rect):
            self.crop_rect = rect

    tiles = [Tile(Rect(0, 0, 96, CANVAS_H)), Tile(Rect(96, 0, CANVAS_W, CANVAS_H))]
    positives = stubbed_vl.build_global_slices(FakeVLClip(), source, tiles)
    assert len(positives) == 2
    expected = [[0, 1, 2, 4, 5, 7] + TAIL, [0, 2, 3, 5, 6, 7] + TAIL]
    for positive, indices in zip(positives, expected):
        tensor, extras = positive[0]
        assert tensor.shape == (1, len(indices), 8)
        assert tensor[0, :, 0].tolist() == indices
        # The full-canvas attention mask must not survive onto a slice.
        assert "attention_mask" not in extras
        assert "pooled_output" in extras


def test_batched_canvas_is_encoded_one_row_at_a_time(stubbed_vl):
    # Core's tokenizer attaches images[0] alone (qwen_vl.process_qwen2vl_images), so handing
    # over the whole [B,H,W,3] canvas would condition EVERY image on row 0's picture. Each
    # row is encoded on its own and the results are concatenated on the batch axis, in order.
    source = torch.zeros(2, CANVAS_H, CANVAS_W, 3)

    class Tile:
        def __init__(self, rect):
            self.crop_rect = rect

    clip = RecordingVLClip()
    positives = stubbed_vl.build_global_slices(clip, source, [Tile(Rect(0, 0, CANVAS_W, CANVAS_H))])

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

    class Tile:
        def __init__(self, rect):
            self.crop_rect = rect

    clip = RecordingVLClip()
    positives = stubbed_vl.build_global_slices(clip, source, [Tile(Rect(0, 0, CANVAS_W, CANVAS_H))])

    assert len(clip.canvases) == 1
    tensor, _ = positives[0][0]
    assert tensor.shape == (1, EXPECTED_SEQ, 8)
    assert tensor[0, :, 0].tolist() == list(range(EXPECTED_SEQ))


def test_build_global_slices_rejects_wrong_encoder_seq(stubbed_vl):
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)

    class Tile:
        def __init__(self, rect):
            self.crop_rect = rect

    with pytest.raises(RuntimeError, match="expected {}".format(EXPECTED_SEQ)):
        stubbed_vl.build_global_slices(FakeVLClip(seq_override=EXPECTED_SEQ + 3), source, [Tile(Rect(0, 0, CANVAS_W, CANVAS_H))])


def test_build_global_slices_rejects_clip_whose_tokenizer_signature_refuses_images(stubbed_vl):
    # Core CLIPs swallow images= via **kwargs (comfy/sd.py CLIP.tokenize) and fall through to
    # the no-image-tokens guard; this pins the TypeError guard for third-party wrappers with a
    # strict tokenize signature.
    class StrictSignatureClip:
        def tokenize(self, text):
            return {"l": [[(1, 1.0)]]}

    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)
    with pytest.raises(RuntimeError, match="does not accept images"):
        stubbed_vl.build_global_slices(StrictSignatureClip(), source, [])


def test_build_global_slices_rejects_clip_without_image_tokens(stubbed_vl):
    class NoImageTokenClip(FakeVLClip):
        def tokenize(self, text, images=None, llama_template=None):
            return {"qwen3vl_4b": [[(10, 1.0)] * 8]}

    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)
    with pytest.raises(RuntimeError, match="no image tokens"):
        stubbed_vl.build_global_slices(NoImageTokenClip(), source, [])


# --- resample_for_global: real comfy resample ---------------------------------------

@pytest.mark.comfy
def test_resample_for_global_snaps_to_merged_cells(comfy_env):
    # The production shape: 2304x3072 -> exactly 768x1024 (scale 1/3), grid 24x32.
    source = torch.rand(1, 3072, 2304, 3)
    copy, enc_h, enc_w = vl.resample_for_global(source)
    assert (enc_h, enc_w) == (1024, 768)
    assert enc_h % vl.MERGED_CELL == 0 and enc_w % vl.MERGED_CELL == 0
    assert copy.shape == (1, 1024, 768, 3)
    assert enc_h * enc_w == vl.GLOBAL_SLICE_BUDGET


# --- through the pipeline: the per-tile positive swap in _refine_tiles ----------------
# Mirrors the ControlNet swap tests in test_conds.py -- same fakes, same restore checks --
# over the block sampling.py writes to the caller's guider.
# Its own encode geometry: 256x256 -> an 8x8 merged-cell grid over the 80px test canvas, so
# neighboring tiles really do select different rows (the 3x2 grid above is coarser than the
# tiles and every tile would take every row).
PIPE_ENC = 256
PIPE_ROWS = (PIPE_ENC // vl.MERGED_CELL) ** 2
PIPE_SEQ = 1 + PIPE_ROWS + 1 + 4


class VLGuider(GridGuider):
    # GridGuider carrying the CFGGuider conds convention, recording what each tile saw.
    def __init__(self):
        super().__init__()
        self.original_conds = {
            "positive": [{"cross_attn": torch.zeros(1, 1, 8)}],
            "negative": [{"cross_attn": torch.zeros(1, 1, 8)}],
        }
        self.seen_conds = []

    def sample(self, noise, latent_image, sampler, sigmas, **kwargs):
        self.seen_conds.append(self.original_conds)
        return super().sample(noise, latent_image, sampler, sigmas, **kwargs)


class VLRaisesGuider(VLGuider):
    # cf. test_conds.ControlRaisesGuider: blow up on the second tile, once a swap is live.
    def __init__(self):
        super().__init__()
        self.conds_at_raise = None

    def sample(self, noise, latent_image, sampler, sigmas, **kwargs):
        if len(self.calls) == 1:
            self.conds_at_raise = self.original_conds
            raise RuntimeError("sampler exploded")
        return super().sample(noise, latent_image, sampler, sigmas, **kwargs)


@pytest.fixture
def pipeline_clip(monkeypatch):
    # The clip these tests drive, plus the encode geometry pinned to match its sequence
    # length. _convert is deliberately NOT stubbed here: these tests run through
    # comfy.sampler_helpers.convert_cond (comfy_stubs') exactly like production does.
    monkeypatch.setattr(vl, "resample_for_global", lambda source: (source, PIPE_ENC, PIPE_ENC))
    return FakeVLClip(seq_override=PIPE_SEQ)


def _run_vl(image, guider, clip, cap=56, ctx=0, overlap=16, mask=None):
    return sampling.refine_image(
        image, guider, object(), SIGMAS, GridVAE(), GridNoise(),
        max_tile_width=cap, max_tile_height=cap, context_anchor=ctx,
        context_overlap=overlap, mask=mask, vl_clip=clip,
    )


def test_each_tile_samples_its_own_vision_slice(comfy_stubs, pipeline_clip):
    image = torch.rand(1, 80, 80, 3)     # /8-aligned: 2x2 grid at cap 56, overlap 16
    guider = VLGuider()
    pristine = guider.original_conds
    negative = pristine["negative"]

    out = _run_vl(image, guider, pipeline_clip)

    layout = _layout(80, 80, 56, 56, overlap=16)
    assert len(guider.seen_conds) == len(layout.tiles) == 4
    for tile, seen in zip(layout.tiles, guider.seen_conds):
        expected = vl.slice_indices(tile.crop_rect, 80, 80, PIPE_ENC, PIPE_ENC, PIPE_SEQ)
        assert seen["positive"][0]["cross_attn"][0, :, 0].tolist() == expected
        assert seen["negative"] is negative        # only the positive is swapped
    rows = [tuple(seen["positive"][0]["cross_attn"][0, :, 0].tolist()) for seen in guider.seen_conds]
    assert len(set(rows)) == len(rows)             # a distinct slice per tile, in tile order
    # Restored to the SAME object the caller handed in.
    assert guider.original_conds is pristine
    assert out.shape == image.shape


def test_vl_pristine_conds_restored_after_a_mid_run_raise(comfy_stubs, pipeline_clip):
    image = torch.rand(1, 80, 80, 3)
    guider = VLRaisesGuider()
    pristine = guider.original_conds

    with pytest.raises(RuntimeError, match="sampler exploded"):
        _run_vl(image, guider, pipeline_clip)

    assert guider.conds_at_raise is not None and guider.conds_at_raise is not pristine
    assert guider.original_conds is pristine


def test_mask_path_encodes_the_full_image_at_the_bbox_offset(comfy_stubs, pipeline_clip, monkeypatch):
    # The region's tiles index the bbox crop, but the encode must read the WHOLE image with
    # the bbox origin as the offset, or a masked refine only ever sees its own crop.
    image = torch.rand(1, 80, 80, 3)
    mask = torch.zeros(1, 80, 80)
    mask[:, 16:64, 16:64] = 1.0
    seen = {}
    real_build = vl.build_global_slices

    def recording_build(clip, source, tiles, offset_x=0, offset_y=0):
        seen["source"] = source
        seen["offsets"] = (offset_x, offset_y)
        return real_build(clip, source, tiles, offset_x, offset_y)

    monkeypatch.setattr(vl, "build_global_slices", recording_build)
    _run_vl(image, VLGuider(), pipeline_clip, ctx=8, mask=mask)

    y0, y1, x0, x1 = sampling._expand_snap_clamp(sampling._mask_bbox(mask >= 0.5), 8, 80, 80)
    assert (y0, x0) == (8, 8)
    assert seen["source"] is image                 # the FULL image, not the bbox crop
    assert seen["offsets"] == (x0, y0)


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
