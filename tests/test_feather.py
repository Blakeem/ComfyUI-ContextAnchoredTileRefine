import torch

from context_anchored_tile_refine import grid, sampling

# Hand-computable interior tile: an 8x8 core, overlap 4, kept on top AND left. The
# paste_rect is the core + a 4-px band on top and left -> (4,4)..(16,16) = 12x12.
CORE = grid.Rect(x0=8, y0=8, x1=16, y1=16)
OVERLAP = 4
PASTE_TL = grid.expand_rect(CORE, OVERLAP, grid.Neighbors(left=True, right=False, top=True, bottom=False), 64, 64)
# Cell-centered ramp over a 4-wide band: (k+0.5)/4 -> ~0 at the outer edge, 1-0.5/4 =
# 0.875 immediately adjacent to the core (the seam). All values are dyadic -> exact f32.
RAMP = torch.tensor([0.125, 0.375, 0.625, 0.875])


def test_paste_geometry_sanity():
    assert (PASTE_TL.x0, PASTE_TL.y0, PASTE_TL.x1, PASTE_TL.y1) == (4, 4, 16, 16)


def test_alpha_shape_and_dtype():
    a = sampling.feather_alpha(PASTE_TL, CORE, OVERLAP, kept_top=True, kept_left=True)
    assert a.shape == (12, 12)
    assert a.dtype == torch.float32


def test_alpha_is_one_throughout_the_core():
    a = sampling.feather_alpha(PASTE_TL, CORE, OVERLAP, kept_top=True, kept_left=True)
    # top_band = left_band = 4, so the core occupies rows/cols 4.. -> alpha == 1 there.
    assert torch.equal(a[4:, 4:], torch.ones(8, 8))


def test_top_overlap_ramp_monotonic_and_seam_is_this_tile():
    a = sampling.feather_alpha(PASTE_TL, CORE, OVERLAP, kept_top=True, kept_left=True)
    # In a core column (col >= 4, ax == 1) the band is the pure vertical ramp.
    col = a[:, 6]
    assert torch.equal(col[:4], RAMP)          # ~0 outer edge -> 0.875 adjacent to core
    assert torch.equal(col[4:], torch.ones(8))  # core rows are 100% this tile
    assert torch.all(col[1:] >= col[:-1])       # monotonic non-decreasing toward the seam
    assert col[3].item() == 0.875               # seam-adjacent band pixel is ~1
    assert col[0].item() == 0.125               # outer edge is ~0 (neighbor dominates)


def test_left_overlap_ramp():
    a = sampling.feather_alpha(PASTE_TL, CORE, OVERLAP, kept_top=True, kept_left=True)
    row = a[6, :]  # core row (ay == 1) -> pure horizontal ramp
    assert torch.equal(row[:4], RAMP)
    assert torch.equal(row[4:], torch.ones(8))


def test_corner_is_product_of_the_two_ramps():
    a = sampling.feather_alpha(PASTE_TL, CORE, OVERLAP, kept_top=True, kept_left=True)
    corner = a[:4, :4]
    expected = RAMP[:, None] * RAMP[None, :]
    assert torch.equal(corner, expected)
    assert a[0, 0].item() == 0.125 * 0.125
    assert a[3, 3].item() == 0.875 * 0.875


def test_kept_top_false_makes_vertical_axis_all_ones():
    # Left band only (kept_left, not kept_top): every row identical, ramp on x alone.
    paste = grid.expand_rect(CORE, OVERLAP, grid.Neighbors(left=True, right=False, top=False, bottom=False), 64, 64)
    a = sampling.feather_alpha(paste, CORE, OVERLAP, kept_top=False, kept_left=True)
    assert a.shape == (8, 12)
    assert torch.equal(a[0], a[7])              # no vertical variation
    assert torch.equal(a[3, :4], RAMP)
    assert torch.equal(a[:, 4:], torch.ones(8, 8))


def test_kept_left_false_makes_horizontal_axis_all_ones():
    # Top band only (kept_top, not kept_left): every column identical, ramp on y alone.
    paste = grid.expand_rect(CORE, OVERLAP, grid.Neighbors(left=False, right=False, top=True, bottom=False), 64, 64)
    a = sampling.feather_alpha(paste, CORE, OVERLAP, kept_top=True, kept_left=False)
    assert a.shape == (12, 8)
    assert torch.equal(a[:, 0], a[:, 7])        # no horizontal variation
    assert torch.equal(a[:4, 3], RAMP)
    assert torch.equal(a[4:, :], torch.ones(8, 8))


def test_kept_flags_gate_the_ramp_independently_of_geometry():
    # kept_top False suppresses the vertical ramp even when the paste_rect has a top band.
    a = sampling.feather_alpha(PASTE_TL, CORE, OVERLAP, kept_top=False, kept_left=True)
    assert torch.equal(a[:, 6], torch.ones(12))  # core column: vertical is flat
    assert torch.equal(a[6, :4], RAMP)           # horizontal still ramps


def test_no_kept_sides_is_all_ones_hard_paste():
    # Top-left tile: paste_rect == core, no kept side -> alpha all ones (hard core paste).
    a = sampling.feather_alpha(CORE, CORE, OVERLAP, kept_top=False, kept_left=False)
    assert a.shape == (8, 8)
    assert torch.equal(a, torch.ones(8, 8))


def test_overlap_zero_is_all_ones_even_with_kept_flags():
    # context_overlap=0: paste_rect == core (no expansion); alpha is all ones regardless
    # of the kept flags, so the composite reduces to the fe6f6a7 hard core paste.
    a = sampling.feather_alpha(CORE, CORE, 0, kept_top=True, kept_left=True)
    assert torch.equal(a, torch.ones(8, 8))


def test_clamped_kept_bands_normalize_by_actual_width():
    # A tile near the top-left canvas edge: core starts at (4, 4) with overlap 8, so BOTH
    # kept bands clamp to 4 px (not 8). The ramp MUST normalize by the actual band width so
    # it still reaches ~1 immediately adjacent to the core (0.875); normalizing by the
    # nominal overlap would give 3.5/8 = 0.4375 -> a visible seam. Guards both the top and
    # left normalization (the unclamped tests above can't, since there band == overlap).
    core = grid.Rect(x0=4, y0=4, x1=16, y1=16)
    overlap = 8
    paste = grid.expand_rect(core, overlap, grid.Neighbors(left=True, right=False, top=True, bottom=False), 64, 64)
    assert (paste.x0, paste.y0, paste.x1, paste.y1) == (0, 0, 16, 16)
    assert (core.y0 - paste.y0, core.x0 - paste.x0) == (4, 4)   # both bands clamped 8 -> 4
    a = sampling.feather_alpha(paste, core, overlap, kept_top=True, kept_left=True)
    ramp4 = torch.tensor([0.125, 0.375, 0.625, 0.875])           # (k+0.5)/4, NOT (k+0.5)/8
    col = a[:, 10]  # core column -> pure vertical (top) ramp
    assert torch.equal(col[:4], ramp4)
    assert torch.equal(col[4:], torch.ones(12))
    assert col[3].item() == 0.875                                # seam ~1 (0.4375 if /overlap)
    row = a[10, :]  # core row -> pure horizontal (left) ramp
    assert torch.equal(row[:4], ramp4)
    assert torch.equal(row[4:], torch.ones(12))
    assert row[3].item() == 0.875
