import math

import torch

from context_anchored_tile_refine import grid, sampling

# Hand-computable interior tile: an 8x8 core, overlap 8, kept on top AND left. The
# paste_rect is the core + an 8-px band on top and left -> (0,0)..(16,16) = 16x16.
# The band is a multiple of 8 on purpose: context_overlap always is, and the baked 10%
# plateau only reaches a cell center once band >= 5, so a narrower band would test a
# degenerate curve the node can never produce.
CORE = grid.Rect(x0=8, y0=8, x1=16, y1=16)
OVERLAP = 8
PASTE_TL = grid.expand_rect(CORE, OVERLAP, grid.Neighbors(left=True, right=False, top=True, bottom=False), 64, 64)


def _ref_ramp(band, plateau=0.10, k=2.0):
    # Independent pure-Python mirror of sampling._feather_ramp (outer edge -> seam), used to
    # pin the curve: a solid plateau over the seam-most `plateau` fraction, then a (1-w)^k
    # fall-off. Cross-checks the torch implementation without sharing its code, and is written
    # in the (1-w)^k form rather than sampling.py's algebraically-equal u/(1-plateau) form so
    # a refactor of one cannot silently drag the other with it. The defaults are the baked-in
    # curve, restated as literals so changing the constants fails here loudly.
    vals = []
    for i in range(band):
        u = (i + 0.5) / band          # 0..1, outer -> seam
        d = 1.0 - u                   # seam-distance
        w = min(1.0, max(0.0, (d - plateau) / (1.0 - plateau)))
        vals.append((1.0 - w) ** k)
    return torch.tensor(vals, dtype=torch.float32)


# Baked-curve ramp over the OVERLAP=8 band, outer -> seam. Only the seam-most pixel falls
# inside a 10%-of-8 plateau, so it is exactly 1.0; the rest fall off.
RAMP = _ref_ramp(OVERLAP)
LINEAR = (torch.arange(OVERLAP, dtype=torch.float32) + 0.5) / OVERLAP  # plateau 0, k 1


def test_baked_curve_matches_the_ab_settled_values():
    # The A/B outcome, pinned: these were node widgets during tuning and are now the curve.
    # Restated as literals (not read from the module) so a change has to be deliberate.
    assert sampling.FEATHER_PLATEAU == 0.10
    assert sampling.FEATHER_FALLOFF == 2.0
    # ...and they really are what feather_alpha uses when the pipeline omits them.
    explicit = sampling.feather_alpha(PASTE_TL, CORE, OVERLAP, kept_top=True, kept_left=True, plateau=0.10, k=2.0)
    assert torch.equal(sampling.feather_alpha(PASTE_TL, CORE, OVERLAP, kept_top=True, kept_left=True), explicit)


def test_paste_geometry_sanity():
    assert (PASTE_TL.x0, PASTE_TL.y0, PASTE_TL.x1, PASTE_TL.y1) == (0, 0, 16, 16)
    assert (CORE.y0 - PASTE_TL.y0, CORE.x0 - PASTE_TL.x0) == (8, 8)   # full bands, unclamped


def test_alpha_shape_and_dtype():
    a = sampling.feather_alpha(PASTE_TL, CORE, OVERLAP, kept_top=True, kept_left=True)
    assert a.shape == (16, 16)
    assert a.dtype == torch.float32


def test_alpha_is_one_throughout_the_core():
    a = sampling.feather_alpha(PASTE_TL, CORE, OVERLAP, kept_top=True, kept_left=True)
    # top_band = left_band = 8, so the core occupies rows/cols 8.. -> alpha == 1 there.
    assert torch.equal(a[8:, 8:], torch.ones(8, 8))


def test_top_overlap_ramp_monotonic_and_seam_is_this_tile():
    a = sampling.feather_alpha(PASTE_TL, CORE, OVERLAP, kept_top=True, kept_left=True)
    # In a core column (col >= 8, ax == 1) the band is the pure vertical ramp.
    col = a[:, 12]
    assert torch.allclose(col[:8], RAMP, atol=1e-6)  # curve fall-off -> 1.0 at the seam
    assert torch.equal(col[8:], torch.ones(8))       # core rows are 100% this tile
    assert torch.all(col[1:] >= col[:-1])            # monotonic non-decreasing toward the seam
    assert col[7].item() == 1.0                      # seam-adjacent band pixel is solid (plateau)
    assert col[0].item() < 0.05                      # outer edge is ~0 (neighbor dominates)


def test_left_overlap_ramp():
    a = sampling.feather_alpha(PASTE_TL, CORE, OVERLAP, kept_top=True, kept_left=True)
    row = a[12, :]  # core row (ay == 1) -> pure horizontal ramp
    assert torch.allclose(row[:8], RAMP, atol=1e-6)
    assert torch.equal(row[8:], torch.ones(8))


def test_corner_is_product_of_the_two_ramps():
    a = sampling.feather_alpha(PASTE_TL, CORE, OVERLAP, kept_top=True, kept_left=True)
    corner = a[:8, :8]
    expected = RAMP[:, None] * RAMP[None, :]
    assert torch.allclose(corner, expected, atol=1e-6)
    assert a[7, 7].item() == 1.0                      # seam corner is solid (both axes plateau)
    assert math.isclose(a[0, 0].item(), RAMP[0].item() ** 2, rel_tol=1e-5, abs_tol=1e-8)


def test_kept_top_false_makes_vertical_axis_all_ones():
    # Left band only (kept_left, not kept_top): every row identical, ramp on x alone.
    paste = grid.expand_rect(CORE, OVERLAP, grid.Neighbors(left=True, right=False, top=False, bottom=False), 64, 64)
    a = sampling.feather_alpha(paste, CORE, OVERLAP, kept_top=False, kept_left=True)
    assert a.shape == (8, 16)
    assert torch.equal(a[0], a[7])              # no vertical variation
    assert torch.allclose(a[3, :8], RAMP, atol=1e-6)
    assert torch.equal(a[:, 8:], torch.ones(8, 8))


def test_kept_left_false_makes_horizontal_axis_all_ones():
    # Top band only (kept_top, not kept_left): every column identical, ramp on y alone.
    paste = grid.expand_rect(CORE, OVERLAP, grid.Neighbors(left=False, right=False, top=True, bottom=False), 64, 64)
    a = sampling.feather_alpha(paste, CORE, OVERLAP, kept_top=True, kept_left=False)
    assert a.shape == (16, 8)
    assert torch.equal(a[:, 0], a[:, 7])        # no horizontal variation
    assert torch.allclose(a[:8, 3], RAMP, atol=1e-6)
    assert torch.equal(a[8:, :], torch.ones(8, 8))


def test_kept_flags_gate_the_ramp_independently_of_geometry():
    # kept_top False suppresses the vertical ramp even when the paste_rect has a top band.
    a = sampling.feather_alpha(PASTE_TL, CORE, OVERLAP, kept_top=False, kept_left=True)
    assert torch.equal(a[:, 12], torch.ones(16))      # core column: vertical is flat
    assert torch.allclose(a[12, :8], RAMP, atol=1e-6)  # horizontal still ramps


def test_no_kept_sides_is_all_ones_hard_paste():
    # Top-left tile: paste_rect == core, no kept side -> alpha all ones (hard core paste).
    a = sampling.feather_alpha(CORE, CORE, OVERLAP, kept_top=False, kept_left=False)
    assert a.shape == (8, 8)
    assert torch.equal(a, torch.ones(8, 8))


def test_overlap_zero_is_all_ones_even_with_kept_flags():
    # context_overlap=0: paste_rect == core (no expansion); alpha is all ones regardless
    # of the kept flags or the curve params, so the composite reduces to the hard core paste.
    a = sampling.feather_alpha(CORE, CORE, 0, kept_top=True, kept_left=True, plateau=0.5, k=4.0)
    assert torch.equal(a, torch.ones(8, 8))


def test_plateau_zero_falloff_one_reproduces_the_linear_ramp():
    # The pre-curve linear ramp is exactly the plateau=0, k=1 special case, bit-exact at ANY
    # band width (guards that the curve is a strict generalization of the old behavior). Cover
    # a power-of-two band (8) AND a non-power-of-two band (24) — overlap is always a multiple
    # of 8, and the /8 bands are exactly where a naive 1-(1-u) formulation loses a ULP.
    a8 = sampling.feather_alpha(PASTE_TL, CORE, OVERLAP, kept_top=True, kept_left=True, plateau=0.0, k=1.0)
    assert torch.equal(a8[:8, 12], LINEAR)   # core-column vertical ramp == linear
    assert torch.equal(a8[12, :8], LINEAR)   # core-row horizontal ramp == linear

    core24 = grid.Rect(x0=24, y0=0, x1=48, y1=32)
    paste24 = grid.expand_rect(core24, 24, grid.Neighbors(left=True, right=False, top=False, bottom=False), 96, 48)
    assert core24.x0 - paste24.x0 == 24     # a full 24-wide left band (non-power-of-two)
    a24 = sampling.feather_alpha(paste24, core24, 24, kept_top=False, kept_left=True, plateau=0.0, k=1.0)
    linear24 = (torch.arange(24, dtype=torch.float32) + 0.5) / 24
    assert torch.equal(a24[0, :24], linear24)


def test_plateau_and_falloff_shape_the_ramp():
    # The two knobs still move the curve the way the baked comment claims, even though they
    # are no longer exposed: a wider plateau pins more seam-adjacent pixels to exactly 1.0;
    # a higher fall-off k lowers the mid-band weight. This is what justifies the chosen pair.
    band = 16
    core = grid.Rect(x0=16, y0=0, x1=48, y1=32)
    paste = grid.expand_rect(core, band, grid.Neighbors(left=True, right=False, top=False, bottom=False), 128, 128)

    def left_ramp(plateau, k):
        a = sampling.feather_alpha(paste, core, band, kept_top=False, kept_left=True, plateau=plateau, k=k)
        return a[0, :band]  # kept_top False -> ay all ones, so any row is the pure x ramp

    def solid(r):
        return int((r == 1.0).sum())

    assert solid(left_ramp(0.5, 2.0)) > solid(left_ramp(0.10, 2.0)) > solid(left_ramp(0.0, 2.0))
    mid = band // 2  # a fall-off-region pixel for the baked plateau
    assert left_ramp(0.10, 4.0)[mid] < left_ramp(0.10, 2.0)[mid]
    # The baked 10% still pins the seam-adjacent pixel — the plateau's whole job — so a wider
    # one only spends band on solid pixels instead of on the fall-off that hides the DC step.
    assert left_ramp(0.10, 2.0)[band - 1].item() == 1.0


def test_clamped_kept_bands_normalize_by_actual_width():
    # A tile near the top-left canvas edge: core starts at (4, 4) with overlap 8, so BOTH
    # kept bands clamp to 4 px (not 8). The curve MUST normalize by the actual band width so
    # the fall-off spans the real band instead of being cut off mid-curve.
    core = grid.Rect(x0=4, y0=4, x1=16, y1=16)
    overlap = 8
    paste = grid.expand_rect(core, overlap, grid.Neighbors(left=True, right=False, top=True, bottom=False), 64, 64)
    assert (paste.x0, paste.y0, paste.x1, paste.y1) == (0, 0, 16, 16)
    assert (core.y0 - paste.y0, core.x0 - paste.x0) == (4, 4)   # both bands clamped 8 -> 4
    a = sampling.feather_alpha(paste, core, overlap, kept_top=True, kept_left=True)
    ramp4 = _ref_ramp(4)                                         # normalized by 4, NOT 8
    col = a[:, 10]  # core column -> pure vertical (top) ramp
    assert torch.allclose(col[:4], ramp4, atol=1e-6)
    assert torch.equal(col[4:], torch.ones(12))
    # Discriminating check: had it normalized by the nominal 8, the seam-adjacent pixel would
    # sit a quarter of the way up the curve instead of near-solid.
    assert col[3].item() > 0.9
    assert col[3].item() > _ref_ramp(8)[3].item()
    row = a[10, :]  # core row -> pure horizontal (left) ramp
    assert torch.allclose(row[:4], ramp4, atol=1e-6)
    assert torch.equal(row[4:], torch.ones(12))
    assert row[3].item() > 0.9
