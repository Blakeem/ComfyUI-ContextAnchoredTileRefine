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


def test_t6_round8_ties_to_even_up():
    assert grid.round8(1500) == 1504  # 187.5 -> 188 (even)


def test_t7_round8_ties_to_even_down():
    assert grid.round8(1012) == 1008  # 126.5 -> 126 (even)


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
