import math

import pytest

from context_anchored_tile_refine import grid

# --- The 12 simulator SELF_TESTS (docs/tile-simulator.html:494-656), pinned verbatim.
# Signature: solve_axis(L, cap, ctx, overlap); build_layout(W, H, sx, sy, ctx, overlap).


def test_t1_r128_two_tiles():
    # r = ctx + overlap = 128. n=2: base round8_up(1536)=1536, +r 128 = 1664 <= 2048.
    s = grid.solve_axis(3072, 2048, 64, 64)
    assert (s.n, s.base, s.last, s.overhead, s.r) == (2, 1536, 1536, 128, 128)


def test_t2_single_tile_ignores_r():
    # n=1 has zero overhead, so a large r is irrelevant when the canvas fits one tile.
    s = grid.solve_axis(2048, 2048, 448, 0)
    assert (s.n, s.base, s.last, s.overhead, s.r) == (1, 2048, 2048, 0, 448)


def test_t3_remainder_last_tile():
    # n=2: base round8_up(1540)=1544, +128 = 1672 <= 2048; last = 3080 - 1544 = 1536.
    s = grid.solve_axis(3080, 2048, 64, 64)
    assert (s.n, s.base, s.last, s.overhead) == (2, 1544, 1536, 128)


def test_t4_overhead_at_n2_is_r_not_2r():
    # ctx=0 overlap=32 -> r=32. n=2: 1032 + 32 = 1064 <= 1088 (2r=64 would still fit,
    # but n=2 pays only r).
    s = grid.solve_axis(2064, 1088, 0, 32)
    assert (s.n, s.base, s.last, s.overhead) == (2, 1032, 1032, 32)


def test_t5_five_tiles_pay_2r():
    # ctx=0 overlap=64 -> r=64. n=4: 1024 + 2r(128) = 1152 > 1024. n=5: base
    # round8_up(820)=824, +128 = 952 <= 1024; last = 4096 - 4*824 = 800.
    s = grid.solve_axis(4096, 1024, 0, 64)
    assert (s.n, s.base, s.last, s.overhead, s.r) == (5, 824, 800, 128, 64)


def test_t8_round8_up():
    assert grid.round8_up(1541) == 1544
    assert grid.round8_up(1536) == 1536


def test_t9_layout_interior_symmetric_crop_vs_directional_paste():
    sx = grid.solve_axis(2048, 1024, 64, 64)
    sy = grid.solve_axis(2048, 1024, 64, 64)
    lay = grid.build_layout(2048, 2048, sx, sy, 64, 64)

    assert (sx.n, sy.n) == (3, 3)
    (interior,) = [t for t in lay.tiles if t.cls == "interior"]  # (col=1, row=1)
    # core (688,688)..(1376,1376) = 688x688.
    assert (interior.core.x0, interior.core.y0, interior.core.x1, interior.core.y1) == (688, 688, 1376, 1376)
    # crop_rect: +r(128) on every side -> (560,560)..(1504,1504) = 944x944 (symmetric).
    assert (interior.crop_rect.x0, interior.crop_rect.y0) == (560, 560)
    assert (interior.sampled_w, interior.sampled_h) == (944, 944)
    # overlap_inner_rect: +overlap(64) on every side -> (624,624)..(1440,1440) = 816x816.
    assert (interior.overlap_inner_rect.x0, interior.overlap_inner_rect.y0) == (624, 624)
    assert (interior.overlap_inner_rect.x1, interior.overlap_inner_rect.y1) == (1440, 1440)
    # paste_rect: +overlap(64) on TOP/LEFT only -> (624,624)..(1376,1376) = 752x752.
    assert (interior.paste_rect.x0, interior.paste_rect.y0, interior.paste_rect.x1, interior.paste_rect.y1) == (624, 624, 1376, 1376)
    assert (interior.kept_top, interior.kept_left) == (True, True)


def test_t10_layout_top_left_tile_paste_equals_core():
    sx = grid.solve_axis(2048, 1024, 64, 64)
    sy = grid.solve_axis(2048, 1024, 64, 64)
    lay = grid.build_layout(2048, 2048, sx, sy, 64, 64)

    c00 = lay.tiles[0]  # row 0, col 0 — no already-processed neighbor.
    assert (c00.kept_top, c00.kept_left) == (False, False)
    # pasteRect == core.
    assert (c00.paste_rect.x0, c00.paste_rect.y0, c00.paste_rect.x1, c00.paste_rect.y1) == (
        c00.core.x0, c00.core.y0, c00.core.x1, c00.core.y1,
    )
    # crop expands right+bottom only -> (0,0)..(816,816) = 816x816.
    assert (c00.crop_rect.x0, c00.crop_rect.y0) == (0, 0)
    assert (c00.sampled_w, c00.sampled_h) == (816, 816)


def test_t11_layout_first_row_keeps_left_overlap_only():
    sx = grid.solve_axis(2048, 1024, 64, 64)
    sy = grid.solve_axis(2048, 1024, 64, 64)
    lay = grid.build_layout(2048, 2048, sx, sy, 64, 64)

    t = next(x for x in lay.tiles if x.row == 0 and x.col == 1)
    assert (t.kept_top, t.kept_left) == (False, True)
    assert (t.core.x0, t.core.y0) == (688, 0)
    # paste: left kept (x0=624), top not (y0=0) -> 752x688 at (624, 0).
    assert (t.paste_rect.x0, t.paste_rect.y0) == (624, 0)
    assert (t.paste_rect.x1 - t.paste_rect.x0, t.paste_rect.y1 - t.paste_rect.y0) == (752, 688)


def test_t12_layout_totals():
    sx = grid.solve_axis(2048, 1024, 64, 64)
    sy = grid.solve_axis(2048, 1024, 64, 64)
    lay = grid.build_layout(2048, 2048, sx, sy, 64, 64)

    interior = [t for t in lay.tiles if t.cls == "interior"]
    assert (sx.n, sy.n) == (3, 3)
    assert len(lay.tiles) == 9
    assert len(interior) == 1
    assert (interior[0].sampled_w, interior[0].sampled_h) == (944, 944)
    assert (lay.tiles[0].sampled_w, lay.tiles[0].sampled_h) == (816, 816)
    assert sx.last == 672
    assert sx.base * (sx.n - 1) + sx.last == 2048
    assert lay.total_sampled_px == 6553600
    assert all(t.sampled_w <= 1024 and t.sampled_h <= 1024 for t in lay.tiles)


# --- overlap=0 bit-identity: overlap_inner_rect == core, paste_rect == core, and
# crop_rect == core + ctx (so the ctx-only geometry is unchanged from the prior model) ---


def test_overlap0_reduces_to_context_only_geometry():
    sx = grid.solve_axis(2048, 1024, 64, 0)
    sy = grid.solve_axis(2048, 1024, 64, 0)
    lay = grid.build_layout(2048, 2048, sx, sy, 64, 0)

    assert lay.overlap == 0 and lay.r == 64
    (interior,) = [t for t in lay.tiles if t.cls == "interior"]
    core = interior.core
    # overlap_inner_rect and paste_rect collapse onto the core.
    for rect in (interior.overlap_inner_rect, interior.paste_rect):
        assert (rect.x0, rect.y0, rect.x1, rect.y1) == (core.x0, core.y0, core.x1, core.y1)
    # crop_rect = core + ctx(64) on every neighbor side.
    assert (interior.crop_rect.x0, interior.crop_rect.y0) == (core.x0 - 64, core.y0 - 64)
    assert (interior.crop_rect.x1, interior.crop_rect.y1) == (core.x1 + 64, core.y1 + 64)


def test_single_tile_paste_and_overlap_collapse_to_core():
    # 1x1 grid: no neighbors, so every derived rect equals the core regardless of r.
    sx = grid.solve_axis(2048, 2048, 64, 64)
    sy = grid.solve_axis(2048, 2048, 64, 64)
    lay = grid.build_layout(2048, 2048, sx, sy, 64, 64)

    (t,) = lay.tiles
    core = t.core
    for rect in (t.crop_rect, t.overlap_inner_rect, t.paste_rect):
        assert (rect.x0, rect.y0, rect.x1, rect.y1) == (core.x0, core.y0, core.x1, core.y1)
    assert (t.kept_top, t.kept_left) == (False, False)


# --- Solver failure semantics (only "exhausted" remains; the fade floor is gone) ---


def test_grid_config_error_message_names_the_inputs():
    with pytest.raises(grid.GridConfigError, match=r"caps too small for overlap \+ context") as excinfo:
        grid.solve_axis(64, 4, 8, 8)
    message = str(excinfo.value)
    for token in ("L=64", "cap=4", "ctx=8", "overlap=8"):
        assert token in message
    # No axis given (the bare solver call): the internal-symbol message stands alone.
    assert "max_tile_" not in message


def test_grid_config_error_axis_names_the_widgets():
    # With `axis`, the same message gains one sentence in the caller's widget names — the
    # only part of it a node user can act on. sampling.py passes width/height.
    with pytest.raises(grid.GridConfigError) as excinfo:
        grid.solve_axis(64, 4, 8, 8, axis="width")
    message = str(excinfo.value)
    assert "caps too small for overlap + context" in message
    for token in ("L=64", "cap=4", "ctx=8", "overlap=8"):
        assert token in message
    assert "max_tile_width 4 cannot hold context_anchor 8 + context_overlap 8" in message
    assert "Raise max_tile_width or lower context_overlap" in message


def test_solver_exhausted_guard():
    # Unreachable through the node (widget min cap is 256) but the guard must exist.
    with pytest.raises(grid.GridConfigError) as excinfo:
        grid.solve_axis(64, 4, 0, 0)
    exc = excinfo.value
    assert isinstance(exc, ValueError)
    assert exc.reason == "exhausted"
    assert (exc.fail_n, exc.fail_base, exc.r) == (9, 8, 0)


# --- ctx=0/overlap=0 property sweep (the exact configuration the hard-seam path runs) ---


@pytest.mark.parametrize("cap", [256, 512, 1024, 2048])
def test_ctx0_overlap0_sweep(cap):
    for L in range(8, 4097, 8):
        s = grid.solve_axis(L, cap, 0, 0)
        assert s.n == 1 or grid.round8_up(math.ceil(L / (s.n - 1))) > cap  # n is minimal
        assert s.base % 8 == 0
        assert s.base * (s.n - 1) + s.last == L
        assert 0 < s.last <= s.base


# --- Layout / rect helpers ---


def test_layout_tiles_are_raster_ordered():
    sx = grid.solve_axis(80, 32, 0, 0)
    sy = grid.solve_axis(88, 32, 0, 0)
    lay = grid.build_layout(80, 88, sx, sy, 0, 0)

    assert (sx.n, sy.n) == (3, 3)
    for i, tile in enumerate(lay.tiles):
        assert (tile.row, tile.col) == (i // sx.n, i % sx.n)


def test_expand_rect_expands_neighbor_sides_only():
    core = grid.Rect(x0=32, y0=0, x1=64, y1=32)
    nb = grid.Neighbors(left=True, right=True, top=False, bottom=True)
    rect = grid.expand_rect(core, 8, nb, 96, 96)
    assert (rect.x0, rect.y0, rect.x1, rect.y1) == (24, 0, 72, 40)
    assert rect.clamped is False


def test_expand_rect_clamps_to_canvas_and_flags_it():
    core = grid.Rect(x0=0, y0=64, x1=32, y1=96)
    nb = grid.Neighbors(left=True, right=True, top=True, bottom=False)
    rect = grid.expand_rect(core, 16, nb, 96, 96)
    assert (rect.x0, rect.y0, rect.x1, rect.y1) == (0, 48, 48, 96)
    assert rect.clamped is True


def test_axis_class_pins():
    assert grid.axis_class(0, 1) == "single"
    assert grid.axis_class(0, 3) == "end"
    assert grid.axis_class(2, 3) == "end"
    assert grid.axis_class(1, 3) == "mid"


# --- multiple=32: the MiniMax H3 granularity (VAE 16 x DiT patch 2). Same ladder,
# same first-fit n, bases snapped to 32 instead of 8. ---


def test_multiple32_single_tile_no_overhead():
    # n=1: base round_up_multiple(1408,32)=1408, overhead 0 <= cap 1408.
    s = grid.solve_axis(1408, 1408, 32, 32, multiple=32)
    assert (s.n, s.base, s.last, s.overhead, s.r) == (1, 1408, 1408, 0, 64)


def test_multiple32_two_tiles_pay_r():
    # The spike's 2688x1536 canvas at cap 1408, ctx=overlap=32 -> r=64.
    # x: n=2 base round_up_multiple(1344,32)=1344, +r 64 = 1408 <= 1408.
    # y: n=2 base round_up_multiple(768,32)=768, +r 64 = 832 <= 1408.
    sx = grid.solve_axis(2688, 1408, 32, 32, multiple=32)
    sy = grid.solve_axis(1536, 1408, 32, 32, multiple=32)
    assert (sx.n, sx.base, sx.last, sx.overhead, sx.r) == (2, 1344, 1344, 64, 64)
    assert (sy.n, sy.base, sy.last, sy.overhead) == (2, 768, 768, 64)


def test_multiple32_snaps_base_up_past_the_8_grid():
    # Same inputs as t3 (which lands on 1544 at /8): n=2 base
    # round_up_multiple(1540,32)=1568, +128 = 1696 <= 2048; last = 3080 - 1568 = 1512.
    s = grid.solve_axis(3080, 2048, 64, 64, multiple=32)
    assert (s.n, s.base, s.last, s.overhead) == (2, 1568, 1512, 128)
    assert grid.solve_axis(3080, 2048, 64, 64).base == 1544  # /8 default unchanged


def test_multiple32_five_tiles_pay_2r():
    # Same inputs as t5. r=64. n=4: 1024 + 2r(128) = 1152 > 1024. n=5: base
    # round_up_multiple(820,32)=832, +128 = 960 <= 1024; last = 4096 - 4*832 = 768.
    s = grid.solve_axis(4096, 1024, 0, 64, multiple=32)
    assert (s.n, s.base, s.last, s.overhead, s.r) == (5, 832, 768, 128, 64)


def test_multiple32_exhausted_reports_the_multiple_as_fail_base():
    # max_n = ceil(64/32) + 1 = 3; the smallest base the ladder can offer is 32, not 8.
    with pytest.raises(grid.GridConfigError) as excinfo:
        grid.solve_axis(64, 4, 0, 0, multiple=32)
    exc = excinfo.value
    assert exc.reason == "exhausted"
    assert (exc.fail_n, exc.fail_base, exc.r) == (3, 32, 0)


@pytest.mark.parametrize("cap", [512, 1024, 1408, 2048])
def test_multiple32_sweep(cap):
    # ctx=0/overlap=0 so overhead is 0 at every n and minimality is a base-only test.
    for L in range(32, 4097, 32):
        s = grid.solve_axis(L, cap, 0, 0, multiple=32)
        assert s.n == 1 or grid.round_up_multiple(math.ceil(L / (s.n - 1)), 32) > cap
        assert s.base % 32 == 0
        assert s.base * (s.n - 1) + s.last == L
        assert 0 < s.last <= s.base


def test_multiple_default_matches_explicit_8():
    for L in range(8, 4097, 8):
        assert grid.solve_axis(L, 1024, 64, 64) == grid.solve_axis(L, 1024, 64, 64, multiple=8)


def test_round_up_multiple():
    assert grid.round_up_multiple(1541, 8) == 1544
    assert grid.round_up_multiple(1536, 32) == 1536
    assert grid.round_up_multiple(1537, 32) == 1568
    # round8_up delegates, so the public /8 name keeps its exact meaning.
    assert grid.round8_up(1541) == grid.round_up_multiple(1541, 8)


@pytest.mark.parametrize(
    "cx,cy,expected",
    [
        ("single", "single", "single"),
        ("mid", "mid", "interior"),
        ("end", "end", "corner"),
        ("end", "mid", "edge"),
        ("mid", "end", "edge"),
        ("single", "mid", "strip middle"),
        ("mid", "single", "strip middle"),
        ("single", "end", "strip end"),
        ("end", "single", "strip end"),
    ],
)
def test_tile_class_label_pins(cx, cy, expected):
    assert grid.tile_class_label(cx, cy) == expected


# --- paste_rect vs the previous tile's core (overlap > base is reachable: no fade floor) ---


def test_paste_rect_clamps_at_previous_core_when_overlap_exceeds_base():
    # Widget-legal and off-nominal: max_tile 768, context_anchor 32, context_overlap 256 gives
    # r=288, so on a 4096 axis the cap admits only base<=192 -> n=22, base 192 < overlap 256.
    # Unclamped, tile col=2's paste would start at 384-256=128, inside col=0's core (0..192)
    # that col=1 already feathered — the wide two-refinement blend the node forbids.
    sx = grid.solve_axis(4096, 768, 32, 256, axis="width")
    sy = grid.solve_axis(4096, 768, 32, 256, axis="height")
    assert (sx.n, sx.base) == (22, 192)
    assert sx.base < 256
    layout = grid.build_layout(4096, 4096, sx, sy, 32, 256)

    for tile in layout.tiles:
        # The previous tile's core start on each axis; the tile's own core start when there is
        # no previous tile (paste_rect never expands on a side without a neighbor).
        prev_x0 = (tile.col - 1) * sx.base if tile.col > 0 else tile.core.x0
        prev_y0 = (tile.row - 1) * sy.base if tile.row > 0 else tile.core.y0
        assert tile.paste_rect.x0 >= prev_x0
        assert tile.paste_rect.y0 >= prev_y0
        # Never expands right/bottom, on any configuration.
        assert (tile.paste_rect.x1, tile.paste_rect.y1) == (tile.core.x1, tile.core.y1)

    t22 = next(t for t in layout.tiles if (t.row, t.col) == (2, 2))
    assert (t22.paste_rect.x0, t22.paste_rect.y0) == (192, 192)
    assert (t22.paste_rect.x1, t22.paste_rect.y1) == (576, 576)


def test_paste_rect_unchanged_when_base_exceeds_overlap():
    # The clamp is min(overlap, base), so every configuration with base >= overlap (every
    # default, and every other test in this file) keeps the full directional expansion.
    sx = grid.solve_axis(2048, 1024, 64, 64)
    sy = grid.solve_axis(2048, 1024, 64, 64)
    layout = grid.build_layout(2048, 2048, sx, sy, 64, 64)

    for tile in layout.tiles:
        want_x0 = max(0, tile.core.x0 - (64 if tile.nb.left else 0))
        want_y0 = max(0, tile.core.y0 - (64 if tile.nb.top else 0))
        assert (tile.paste_rect.x0, tile.paste_rect.y0) == (want_x0, want_y0)


# --- SubLayout: the layout of a rectangular block of a parent layout's tiles ---


def _parent_4x5():
    # 4 columns of base 1024 by 5 rows of base 824, ctx 32 + overlap 32 = r 64.
    sx = grid.solve_axis(4096, 1152, 32, 32)
    sy = grid.solve_axis(4096, 1024, 32, 32)
    assert (sx.n, sy.n) == (4, 5)
    return grid.build_layout(4096, 4096, sx, sy, 32, 32)


def _rect_of(rect):
    # A plain Rect out of any rect-shaped value, so an ExpandedRect compares to a block or region.
    return grid.Rect(rect.x0, rect.y0, rect.x1, rect.y1)


def _shifted_crops(sub):
    return [
        grid.Rect(t.crop_rect.x0 + sub.block.x0, t.crop_rect.y0 + sub.block.y0,
                  t.crop_rect.x1 + sub.block.x0, t.crop_rect.y1 + sub.block.y0)
        for t in sub.layout.tiles
    ]


def test_neighborhood_clamps_to_the_grid():
    lay = _parent_4x5()

    assert grid.neighborhood(lay, 2 * 4 + 1) == (0, 2, 1, 3)  # interior centre (col 1, row 2)
    assert grid.neighborhood(lay, 2 * 4 + 3) == (2, 3, 1, 3)  # edge centre (last column)
    assert grid.neighborhood(lay, 0) == (0, 1, 0, 1)  # corner centre (col 0, row 0)
    assert grid.neighborhood(lay, 4 * 4 + 3) == (2, 3, 3, 4)  # the opposite corner


def test_neighborhood_rejects_an_index_outside_the_grid():
    lay = _parent_4x5()

    with pytest.raises(ValueError, match="20 tiles"):
        grid.neighborhood(lay, 20)
    with pytest.raises(ValueError, match="-1"):
        grid.neighborhood(lay, -1)


def test_sub_layout_of_a_single_tile_parent_is_the_parent():
    sx = grid.solve_axis(1024, 2048, 32, 32)
    sy = grid.solve_axis(1024, 2048, 32, 32)
    lay = grid.build_layout(1024, 1024, sx, sy, 32, 32)

    assert grid.neighborhood(lay, 0) == (0, 0, 0, 0)
    sub = grid.sub_layout(lay, 0, 0, 0, 0)
    assert sub.block == grid.Rect(0, 0, 1024, 1024)
    assert sub.region == grid.Rect(0, 0, 1024, 1024)
    assert sub.layout == lay
    assert (sub.first_col, sub.first_row) == (0, 0)


def test_sub_layout_of_the_full_range_is_the_parent_layout():
    lay = _parent_4x5()

    sub = grid.sub_layout(lay, 0, 3, 0, 4)
    assert sub.block == grid.Rect(0, 0, 4096, 4096)
    assert sub.region == grid.Rect(0, 0, 4096, 4096)
    assert (sub.layout.sol_x, sub.layout.sol_y) == (lay.sol_x, lay.sol_y)
    assert sub.layout == lay
    assert _shifted_crops(sub) == [_rect_of(t.crop_rect) for t in lay.tiles]
    assert sub.parent_tiles is lay.tiles


def test_sub_layout_of_a_two_column_two_row_range():
    lay = _parent_4x5()

    # Columns 1..2 and rows 1..2: a tile lies beyond every side, so the block starts at the first
    # in-range core and ends at the last in-range crop.
    sub = grid.sub_layout(lay, 1, 2, 1, 2)
    assert sub.block == grid.Rect(1024, 824, 3136, 2536)
    assert sub.region == grid.Rect(1024 + 32, 824 + 32, 3136 - 32, 2536 - 32)
    assert (sub.first_col, sub.first_row) == (1, 1)
    assert sub.layout.sol_x == grid.AxisSolution(n=2, base=1024, last=1088, overhead=64, r=64)
    assert sub.layout.sol_y == grid.AxisSolution(n=2, base=824, last=888, overhead=64, r=64)

    # Every block crop is the parent's, except the dropped ring on the two beyond sides.
    parents = [lay.tiles[row * 4 + col] for row in (1, 2) for col in (1, 2)]
    want = [
        grid.Rect(p.crop_rect.x0 + (64 if p.col == 1 else 0), p.crop_rect.y0 + (64 if p.row == 1 else 0),
                  p.crop_rect.x1, p.crop_rect.y1)
        for p in parents
    ]
    assert _shifted_crops(sub) == want


def test_sub_layout_of_one_interior_tile_spans_its_crop():
    lay = _parent_4x5()
    tile = lay.tiles[2 * 4 + 1]

    sub = grid.sub_layout(lay, 1, 1, 2, 2)
    assert sub.block == _rect_of(tile.crop_rect)
    assert sub.region == _rect_of(tile.overlap_inner_rect)
    assert len(sub.layout.tiles) == 1
    lone = sub.layout.tiles[0]
    span = grid.Rect(0, 0, sub.block.x1 - sub.block.x0, sub.block.y1 - sub.block.y0)
    assert _rect_of(lone.core) == span
    assert _rect_of(lone.crop_rect) == span


def test_sub_layout_rejects_a_ring_that_reaches_past_the_bordering_tile():
    # Widget-legal and off-nominal: max_tile 1024, context_anchor 256, context_overlap 128 gives
    # r 384 over base 256, so a tile's ring ends beyond its bordering tile's core.
    sx = grid.solve_axis(5000, 1024, 256, 128)
    sy = grid.solve_axis(5000, 1024, 256, 128)
    assert (sx.n, sx.base, sx.r) == (20, 256, 384)
    lay = grid.build_layout(5000, 5000, sx, sy, 256, 128)

    with pytest.raises(ValueError, match="context_anchor 256 plus context_overlap 128"):
        grid.sub_layout(lay, *grid.neighborhood(lay, 2 * 20 + 2))

    # A single tile range is its own crop extent, so it never depends on the ring's reach.
    tile = lay.tiles[2 * 20 + 2]
    sub = grid.sub_layout(lay, 2, 2, 2, 2)
    assert sub.block == _rect_of(tile.crop_rect)
    assert sub.region == _rect_of(tile.overlap_inner_rect)


def test_sub_layout_with_base_equal_to_r_passes_the_self_check():
    sx = grid.solve_axis(1024, 192, 32, 32)
    sy = grid.solve_axis(1024, 192, 32, 32)
    assert (sx.n, sx.base, sx.r) == (16, 64, 64)
    lay = grid.build_layout(1024, 1024, sx, sy, 32, 32)

    # sub_layout's self-check raises RuntimeError when a block crop is not the parent's.
    for index in range(len(lay.tiles)):
        grid.sub_layout(lay, *grid.neighborhood(lay, index))

    sub = grid.sub_layout(lay, 1, 3, 1, 3)
    assert sub.block == grid.Rect(64, 64, 320, 320)
    assert sub.region == grid.Rect(96, 96, 288, 288)


def test_sub_layout_rejects_an_invalid_range():
    lay = _parent_4x5()

    for bounds in [(2, 1, 0, 0), (0, 0, 3, 2), (-1, 1, 0, 0), (0, 4, 0, 0), (0, 0, -1, 0), (0, 0, 0, 5)]:
        with pytest.raises(ValueError, match="4 by 5 tile grid"):
            grid.sub_layout(lay, *bounds)


# --- Equivalence with the harness rule that produced the judged renders ---
# block_rect, block_layout and block_mask, inlined from tests-AB/run_ab_tile_phantom.py. That rule
# produced TESTS.md test 10's judged renders, so it is kept here as the reference after its removal
# from the harness.


def _reference_block_rect(layout, tile):
    near = [t for t in layout.tiles if abs(t.col - tile.col) <= 1 and abs(t.row - tile.row) <= 1]
    x0 = min(t.crop_rect.x0 for t in near)
    y0 = min(t.crop_rect.y0 for t in near)
    if x0 > 0:
        x0 = min(t.core.x0 for t in near)
    if y0 > 0:
        y0 = min(t.core.y0 for t in near)
    return grid.Rect(x0, y0, max(t.crop_rect.x1 for t in near), max(t.crop_rect.y1 for t in near))


def _reference_block_layout(layout, tile, block):
    cols = [t.col for t in layout.tiles if abs(t.col - tile.col) <= 1]
    rows = [t.row for t in layout.tiles if abs(t.row - tile.row) <= 1]
    n_x, n_y = len(set(cols)), len(set(rows))
    r = layout.ctx + layout.overlap
    width, height = block.x1 - block.x0, block.y1 - block.y0
    sx = grid.AxisSolution(n=n_x, base=layout.sol_x.base, last=width - (n_x - 1) * layout.sol_x.base,
                           overhead=0 if n_x == 1 else r if n_x == 2 else 2 * r, r=r)
    sy = grid.AxisSolution(n=n_y, base=layout.sol_y.base, last=height - (n_y - 1) * layout.sol_y.base,
                           overhead=0 if n_y == 1 else r if n_y == 2 else 2 * r, r=r)
    return grid.build_layout(width, height, sx, sy, layout.ctx, layout.overlap), sx, sy


def _reference_block_region(layout, block):
    # The ones rect of block_mask, which is what the engine's mask path denoises.
    a = layout.ctx
    x0 = block.x0 + (a if block.x0 > 0 else 0)
    y0 = block.y0 + (a if block.y0 > 0 else 0)
    x1 = block.x1 - (a if block.x1 < layout.w else 0)
    y1 = block.y1 - (a if block.y1 < layout.h else 0)
    return grid.Rect(x0, y0, x1, y1)


@pytest.mark.parametrize(
    "wl,wcap,hl,hcap",
    [
        (15360, 1536, 8640, 2048),  # the owner's 8K pass 2
        (4096, 1536, 2304, 2048),   # the phantom harness canvas
        (904, 256, 904, 256),       # a last column no wider than r
    ],
)
def test_sub_layout_matches_the_harness_block_rule(wl, wcap, hl, hcap):
    sx = grid.solve_axis(wl, wcap, 32, 32)
    sy = grid.solve_axis(hl, hcap, 32, 32)
    assert sx.base >= sx.r and sy.base >= sy.r
    lay = grid.build_layout(wl, hl, sx, sy, 32, 32)

    for index, tile in enumerate(lay.tiles):
        block = _reference_block_rect(lay, tile)
        reference, ref_sx, ref_sy = _reference_block_layout(lay, tile, block)
        sub = grid.sub_layout(lay, *grid.neighborhood(lay, index))

        assert sub.block == block
        assert (sub.layout.sol_x, sub.layout.sol_y) == (ref_sx, ref_sy)
        assert sub.layout == reference
        assert sub.region == _reference_block_region(lay, block)
