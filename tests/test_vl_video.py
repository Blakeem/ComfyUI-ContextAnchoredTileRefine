"""vl_video.py: the MiniMax H3 global video-reference encode and its per-tile row slices.

The layout every slice rests on is the tokenizer's ref-video presentation — literal token
rows interleaved with one vision block per 2 picked frames, each block expanding to one
row per 32px merged cell in raster order. These tests pin the 2 fps pick/stamp math, the
token-stream walk, the rect->cell mapping (shared boundary cells included), the tags
lockstep, the VRAM reserve bracket, and every fail-fast guard — with a fake duck-typed
clip, so no ComfyUI install and no GPU are needed.
"""
import sys

import pytest
import torch

from context_anchored_tile_refine import vl_video
from context_anchored_tile_refine.grid import Rect

# Fixture geometry: a 96x64 px canvas encoded 1:1 -> a 3x2 merged-cell grid (6 rows per
# vision block). The fake stream carries PREFIX literal rows ("<Video 1>: "), then per
# block a "<T.T seconds>" row, vision_start, the block, vision_end.
ENC_H, ENC_W = 64, 96
GRID_H, GRID_W = 2, 3
ROWS_PER_BLOCK = GRID_H * GRID_W
PREFIX, BLOCKS = 1, 2
TOTAL_ROWS = PREFIX + BLOCKS * (3 + ROWS_PER_BLOCK)      # 1 + 2 * 9 = 19


class FakeH3Clip:
    """Duck-typed MiniMax H3 CLIP.

    The token stream mirrors comfy/text_encoders/minimax.py tokenize_with_weights: text
    ids for the "<Video 1>: " label, then per 2-frame block a "<T.T seconds>" id plus
    vision_start / the block dict / vision_end. The encode is deterministic — feature
    value == sequence position, and the tags carry the row index as well (the real ones
    are 0/1) so a lockstep slice is decidable.
    """

    def __init__(self, rows_per_block=ROWS_PER_BLOCK, blocks=BLOCKS, prefix=PREFIX,
                 sections=1, seq_delta=0, tags="ok"):
        self.rows_per_block = rows_per_block
        self.blocks = blocks
        self.prefix = prefix
        self.sections = sections
        self.seq_delta = seq_delta
        self.tags = tags
        self.tokenize_calls = []
        self.reserve_at_encode = None

    def total_rows(self):
        return self.prefix + self.blocks * (3 + self.rows_per_block)

    def tokenize(self, text, minimax_ref_items=None):
        self.tokenize_calls.append((text, minimax_ref_items))
        stream = [(10, 1.0)] * self.prefix
        for _ in range(self.blocks):
            stream.append((11, 1.0))
            stream.append((151652, 1.0))
            stream.append(({"type": "image", "minimax_video_block": True}, 1.0))
            stream.append((151653, 1.0))
        return {"qwen3vl_32b": [stream]}

    def encode_from_tokens_scheduled(self, tokens):
        self.reserve_at_encode = sys.modules["comfy.model_management"].EXTRA_RESERVED_VRAM
        seq = self.total_rows() + self.seq_delta
        cond = torch.arange(seq, dtype=torch.float32).reshape(1, seq, 1).expand(1, seq, 4).clone()
        extras = {"pooled_output": None, "attention_mask": torch.ones(1, seq)}
        if self.tags == "ok":
            extras["minimax_token_tags"] = torch.arange(seq, dtype=torch.long)
        elif self.tags == "short":
            extras["minimax_token_tags"] = torch.arange(seq - 1, dtype=torch.long)
        return [[cond, extras]] * self.sections


@pytest.fixture
def h3_stubs(comfy_stubs, monkeypatch):
    # comfy.model_management.EXTRA_RESERVED_VRAM is a real module-level int encode_global
    # brackets; the shared stub carries only the functions sampling.py touches.
    monkeypatch.setattr(sys.modules["comfy.model_management"], "EXTRA_RESERVED_VRAM", 0,
                        raising=False)
    return comfy_stubs


def _picks(count=2, height=ENC_H, width=ENC_W):
    return torch.zeros(count, height, width, 3)


# --- 2 fps conditioning frames ------------------------------------------------------

def test_conditioning_frames_take_every_twelfth_frame_with_half_second_stamps():
    frames = torch.arange(124, dtype=torch.float32).reshape(124, 1, 1, 1).expand(124, 4, 4, 3)
    picks, stamps = vl_video.sample_conditioning_frames(frames)

    assert picks.shape == (11, 4, 4, 3)
    assert picks[:, 0, 0, 0].tolist() == list(range(0, 124, 12))
    assert stamps == [i / 2.0 for i in range(11)]


def test_odd_pick_count_is_left_unpadded_for_the_tokenizer(h3_stubs):
    # Core ships odd pick counts (124 frames -> 11 picks) and the tokenizer repeat-pads to
    # its 2-frame patch itself, producing 6 blocks. This module must not pre-pad, and its
    # layout must follow the block count the STREAM declares, not ceil(picks/2) of its own.
    frames = torch.zeros(124, ENC_H, ENC_W, 3)
    picks, stamps = vl_video.sample_conditioning_frames(frames)
    assert picks.shape[0] % 2 == 1

    clip = FakeH3Clip(blocks=6)
    pack = vl_video.encode_global(clip, picks, stamps)

    assert clip.tokenize_calls[0][1][0]["data"].shape[0] == 11
    assert len(clip.tokenize_calls[0][1][0]["timestamps"]) == 11
    assert sum(1 for kind, _ in pack.layout if kind == "grid") == 6


# --- token-stream walk --------------------------------------------------------------

def test_token_layout_walks_text_rows_between_grid_blocks():
    tokens = FakeH3Clip().tokenize("", minimax_ref_items=[])
    layout, total = vl_video.token_layout(tokens, ROWS_PER_BLOCK)

    # PREFIX=1: row 0 is the label; block 0 = seconds 1, vision_start 2, grid 3..8,
    # vision_end 9; block 1 = 10, 11, grid 12..17, vision_end 18.
    assert layout == [
        ("text", 0),
        ("text", 1), ("text", 2), ("grid", 3), ("text", 9),
        ("text", 10), ("text", 11), ("grid", 12), ("text", 18),
    ]
    assert total == TOTAL_ROWS


# --- encode_global ------------------------------------------------------------------

def test_encode_global_sends_one_empty_prompt_video_item_and_returns_the_pack(h3_stubs):
    picks = _picks()
    clip = FakeH3Clip()

    pack = vl_video.encode_global(clip, picks, [0.0, 0.5])

    text, items = clip.tokenize_calls[0]
    assert text == ""                                   # any text re-admits phantoms
    assert len(items) == 1 and items[0]["type"] == "video"
    assert items[0]["data"] is picks                    # canvas-resolution identity
    assert items[0]["timestamps"] == [0.0, 0.5]
    assert (pack.enc_h, pack.enc_w) == (ENC_H, ENC_W)
    assert pack.cond.shape == (1, TOTAL_ROWS, 4)
    assert pack.tags.shape == (TOTAL_ROWS,)
    assert h3_stubs["common_upscale_calls"] == []       # nothing resampled under the cap


def test_encode_raises_the_vram_reserve_and_restores_it(h3_stubs):
    reserve = sys.modules["comfy.model_management"]
    clip = FakeH3Clip()

    vl_video.encode_global(clip, _picks(), [0.0, 0.5])

    assert clip.reserve_at_encode == vl_video.ENCODE_RESERVED_VRAM
    assert reserve.EXTRA_RESERVED_VRAM == 0


def test_vram_reserve_is_restored_when_the_encode_raises(h3_stubs):
    class ExplodingClip(FakeH3Clip):
        def encode_from_tokens_scheduled(self, tokens):
            super().encode_from_tokens_scheduled(tokens)
            raise RuntimeError("encoder exploded")

    clip = ExplodingClip()
    with pytest.raises(RuntimeError, match="encoder exploded"):
        vl_video.encode_global(clip, _picks(), [0.0, 0.5])

    assert clip.reserve_at_encode == vl_video.ENCODE_RESERVED_VRAM
    assert sys.modules["comfy.model_management"].EXTRA_RESERVED_VRAM == 0


def test_encode_global_rejects_a_tokenizer_that_refuses_ref_items(h3_stubs):
    class StrictSignatureClip:
        def tokenize(self, text):
            return {"l": [[(1, 1.0)]]}

    with pytest.raises(RuntimeError, match="MiniMax H3 CLIP"):
        vl_video.encode_global(StrictSignatureClip(), _picks(), [0.0, 0.5])


def test_encode_global_rejects_a_stream_without_video_tokens(h3_stubs):
    # Core tokenizers swallow unknown kwargs, so a non-H3 CLIP gets as far as a text-only
    # stream instead of raising TypeError.
    class TextOnlyClip(FakeH3Clip):
        def tokenize(self, text, minimax_ref_items=None):
            return {"l": [[(10, 1.0)] * 8]}

    with pytest.raises(RuntimeError, match="no video tokens"):
        vl_video.encode_global(TextOnlyClip(), _picks(), [0.0, 0.5])


def test_encode_global_rejects_a_multi_section_schedule(h3_stubs):
    with pytest.raises(RuntimeError, match="returned 2 sections"):
        vl_video.encode_global(FakeH3Clip(sections=2), _picks(), [0.0, 0.5])


def test_encode_global_rejects_a_row_count_the_layout_does_not_explain(h3_stubs):
    with pytest.raises(RuntimeError, match="has 22 rows, expected {}".format(TOTAL_ROWS)):
        vl_video.encode_global(FakeH3Clip(seq_delta=3), _picks(), [0.0, 0.5])


def test_encode_global_rejects_missing_token_tags(h3_stubs):
    with pytest.raises(RuntimeError, match="minimax_token_tags is missing or misaligned"):
        vl_video.encode_global(FakeH3Clip(tags="none"), _picks(), [0.0, 0.5])


def test_encode_global_rejects_misaligned_token_tags(h3_stubs):
    with pytest.raises(RuntimeError, match="18 vs {} rows".format(TOTAL_ROWS)):
        vl_video.encode_global(FakeH3Clip(tags="short"), _picks(), [0.0, 0.5])


# --- over-cap resample --------------------------------------------------------------

def test_over_cap_picks_are_area_resampled_once_to_an_exact_cell_grid(h3_stubs, monkeypatch):
    # Cap lowered so the path is reachable on a test-sized tensor. 256x192 over a 128x128
    # cap: scale = sqrt(1/3), floor-snapped to /32 -> 128x96 (12288 px, inside the cap).
    monkeypatch.setattr(vl_video, "MAX_ENCODE_PIXELS", 128 * 128)
    picks = _picks(count=2, height=256, width=192)
    clip = FakeH3Clip(rows_per_block=(128 // 32) * (96 // 32), blocks=1)

    pack = vl_video.encode_global(clip, picks, [0.0, 0.5])

    assert h3_stubs["common_upscale_calls"] == [((2, 3, 256, 192), 96, 128, "area", "disabled")]
    assert (pack.enc_h, pack.enc_w) == (128, 96)
    assert pack.enc_h * pack.enc_w <= vl_video.MAX_ENCODE_PIXELS
    sent = clip.tokenize_calls[0][1][0]["data"]
    assert sent.shape == (2, 128, 96, 3) and sent is not picks


def test_unaligned_canvas_reports_the_grid_the_tokenizer_will_snap_to(h3_stubs):
    # 100px is not /32: core rounds it to 96 for the tower (its own bilinear resize), so
    # the reported grid — what the tile rects are mapped through — must be the SNAPPED one.
    clip = FakeH3Clip(rows_per_block=(96 // 32) * (96 // 32), blocks=1)

    pack = vl_video.encode_global(clip, _picks(height=100, width=96), [0.0, 0.5])

    assert (pack.enc_h, pack.enc_w) == (96, 96)
    assert h3_stubs["common_upscale_calls"] == []


# --- slice_indices: pure index math -------------------------------------------------
# Rows of the fixture layout: 0 label | 1, 2 block-0 text | 3..8 block-0 cells |
# 9 vision_end | 10, 11 block-1 text | 12..17 block-1 cells | 18 vision_end.
TEXT_ROWS = [0, 1, 2, 9, 10, 11, 18]
LAYOUT = [("text", 0), ("text", 1), ("text", 2), ("grid", 3), ("text", 9),
          ("text", 10), ("text", 11), ("grid", 12), ("text", 18)]

LEFT = [0, 1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 15, 16, 18]
RIGHT = [0, 1, 2, 4, 5, 7, 8, 9, 10, 11, 13, 14, 16, 17, 18]


def test_full_canvas_tile_selects_every_row():
    indices = vl_video.slice_indices(LAYOUT, Rect(0, 0, ENC_W, ENC_H), ENC_H, ENC_W, ENC_H, ENC_W)
    assert indices == list(range(TOTAL_ROWS))


def test_overlapping_tiles_share_the_boundary_cell_column():
    # Crops overlap by one 32px cell column (the anchor + overlap ring in row space), so
    # cell column 1 must land in BOTH tiles, in every block.
    left = vl_video.slice_indices(LAYOUT, Rect(0, 0, 64, ENC_H), ENC_H, ENC_W, ENC_H, ENC_W)
    right = vl_video.slice_indices(LAYOUT, Rect(32, 0, ENC_W, ENC_H), ENC_H, ENC_W, ENC_H, ENC_W)

    assert left == LEFT
    assert right == RIGHT
    assert sorted(set(left) & set(right) - set(TEXT_ROWS)) == [4, 7, 13, 16]
    assert set(left) | set(right) == set(range(TOTAL_ROWS))
    assert left == sorted(left) and right == sorted(right)


def test_every_literal_token_row_is_kept_by_every_tile():
    # The labels and the vision_start/vision_end delimiters flanking each block are not
    # positional — dropping any of them would change the presentation the model reads.
    tiny = vl_video.slice_indices(LAYOUT, Rect(0, 0, 32, 32), ENC_H, ENC_W, ENC_H, ENC_W)
    assert [row for row in tiny if row in TEXT_ROWS] == TEXT_ROWS
    assert tiny == [0, 1, 2, 3, 9, 10, 11, 12, 18]


def test_enc_to_canvas_ratio_maps_a_larger_canvas_onto_the_same_cells():
    # The over-cap case: a 192x128 canvas encoded at 96x64 maps a rect to the cells its
    # half-scale twin selects.
    scaled = vl_video.slice_indices(LAYOUT, Rect(0, 0, 128, 128), 128, 192, ENC_H, ENC_W)
    assert scaled == LEFT


def test_cell_range_clamps_to_the_grid():
    # A rect reaching past the canvas (padding slack) must clamp instead of indexing rows
    # that belong to the next block.
    indices = vl_video.slice_indices(LAYOUT, Rect(32, 32, ENC_W + 32, ENC_H + 32),
                                     ENC_H, ENC_W, ENC_H, ENC_W)
    assert indices == [0, 1, 2, 7, 8, 9, 10, 11, 16, 17, 18]


# --- slice_rows: the tile positive --------------------------------------------------

def test_slice_rows_selects_the_same_rows_from_cond_and_tags(h3_stubs):
    pack = vl_video.encode_global(FakeH3Clip(), _picks(), [0.0, 0.5])

    positive = vl_video.slice_rows(pack, Rect(0, 0, 64, ENC_H), ENC_H, ENC_W)

    assert len(positive) == 1
    entry = positive[0]
    assert entry["cross_attn"].shape == (1, len(LEFT), 4)
    assert entry["cross_attn"][0, :, 0].tolist() == LEFT
    # Tags in lockstep: the DiT's adaLN modality run is keyed on them.
    assert entry["minimax_token_tags"].tolist() == LEFT
    # A whole-clip attention mask is wrong for a slice; the rest of the extras survive.
    assert "attention_mask" not in entry
    assert "pooled_output" in entry


def test_slice_rows_leaves_the_pack_intact_for_the_next_tile(h3_stubs):
    pack = vl_video.encode_global(FakeH3Clip(), _picks(), [0.0, 0.5])

    first = vl_video.slice_rows(pack, Rect(0, 0, 64, ENC_H), ENC_H, ENC_W)
    second = vl_video.slice_rows(pack, Rect(32, 0, ENC_W, ENC_H), ENC_H, ENC_W)

    assert first[0]["cross_attn"][0, :, 0].tolist() == LEFT
    assert second[0]["cross_attn"][0, :, 0].tolist() == RIGHT
    assert pack.cond.shape == (1, TOTAL_ROWS, 4)
    assert "attention_mask" in pack.extras
    assert pack.extras["minimax_token_tags"].tolist() == list(range(TOTAL_ROWS))
