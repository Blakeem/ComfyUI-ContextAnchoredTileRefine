"""vl.py: global-slice conditioning. The slice layout the whole feature rests on is
[0]=vision_start, [1..N]=grid rows (raster), [N+1]=vision_end, [N+2..]=template tail;
these tests pin the rect->cell mapping, the shared boundary cells, the fail-fast
guards, and the per-tile tensor selection (fake duck-typed clip; _convert stubbed to
identity so no comfy is needed)."""
import pytest
import torch

from context_anchored_tile_refine import vl
from context_anchored_tile_refine.grid import Rect


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


def test_build_global_slices_rejects_wrong_encoder_seq(stubbed_vl):
    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)

    class Tile:
        def __init__(self, rect):
            self.crop_rect = rect

    with pytest.raises(RuntimeError, match="expected {}".format(EXPECTED_SEQ)):
        stubbed_vl.build_global_slices(FakeVLClip(seq_override=EXPECTED_SEQ + 3), source, [Tile(Rect(0, 0, CANVAS_W, CANVAS_H))])


def test_build_global_slices_rejects_text_only_clip(stubbed_vl):
    class TextOnlyClip:
        def tokenize(self, text):
            return {"l": [[(1, 1.0)]]}

    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)
    with pytest.raises(RuntimeError, match="does not accept images"):
        stubbed_vl.build_global_slices(TextOnlyClip(), source, [])


def test_build_global_slices_rejects_clip_without_image_tokens(stubbed_vl):
    class NoImageTokenClip(FakeVLClip):
        def tokenize(self, text, images=None, llama_template=None):
            return {"qwen3vl_4b": [[(10, 1.0)] * 8]}

    source = torch.zeros(1, CANVAS_H, CANVAS_W, 3)
    with pytest.raises(RuntimeError, match="no image tokens"):
        stubbed_vl.build_global_slices(NoImageTokenClip(), source, [])


# --- resample_for_global: real comfy resample ---------------------------------------

@pytest.mark.comfy
def test_resample_for_global_snaps_to_merged_cells():
    # The production shape: 2304x3072 -> exactly 768x1024 (scale 1/3), grid 24x32.
    source = torch.rand(1, 3072, 2304, 3)
    copy, enc_h, enc_w = vl.resample_for_global(source)
    assert (enc_h, enc_w) == (1024, 768)
    assert enc_h % vl.MERGED_CELL == 0 and enc_w % vl.MERGED_CELL == 0
    assert copy.shape == (1, 1024, 768, 3)
    assert enc_h * enc_w == vl.GLOBAL_SLICE_BUDGET
