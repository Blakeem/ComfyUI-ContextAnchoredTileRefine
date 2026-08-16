from itertools import pairwise

import torch

from context_anchored_tile_refine import grid, sampling

# A REAL layout tile that keeps both seams, with a deliberately NON-SQUARE paste_rect
# (96x128) so a top/left mix-up cannot pass a shape check by accident.
_SX = grid.solve_axis(160, 96, 0, 16)
_SY = grid.solve_axis(224, 128, 0, 16)
_LAYOUT = grid.build_layout(160, 224, _SX, _SY, 0, 16)
TILE = _LAYOUT.tiles[3]

# Interior-style tile with generous bands so a displacement has room to move: a 32x32 core
# with a 16px kept band on top and left -> paste_rect (16,16)..(64,64) = 48x48.
CORE = grid.Rect(x0=32, y0=32, x1=64, y1=64)
BAND = 16
PASTE = grid.expand_rect(CORE, BAND, grid.Neighbors(left=True, right=False, top=True, bottom=False), 128, 128)
TOP_BAND = CORE.y0 - PASTE.y0
LEFT_BAND = CORE.x0 - PASTE.x0


def _alpha(disp_top=None, disp_left=None, plateau=sampling.FEATHER_PLATEAU, k=sampling.FEATHER_FALLOFF):
    # Defaults track the baked curve so `_alpha()` is exactly what the pipeline produces;
    # the hostile-parameter test below overrides them explicitly.
    return sampling.feather_alpha(PASTE, CORE, BAND, kept_top=True, kept_left=True, plateau=plateau, k=k, disp_top=disp_top, disp_left=disp_left)


def test_paste_geometry_sanity():
    assert (PASTE.x0, PASTE.y0, PASTE.x1, PASTE.y1) == (16, 16, 64, 64)
    assert (TOP_BAND, LEFT_BAND) == (16, 16)


def test_zero_displacement_reproduces_the_straight_ramp_bit_for_bit():
    # disp=None takes a 1-D code path; a zero displacement takes the 2-D one. They must agree
    # EXACTLY, which is what keeps every hand-computed feather/tiling expectation valid and
    # makes a zero displacement identical to no displacement. Comparing _alpha() against
    # feather_alpha's own defaults would just be the same call twice and assert nothing.
    zeros_top = torch.zeros(PASTE.x1 - PASTE.x0)
    zeros_left = torch.zeros(PASTE.y1 - PASTE.y0)
    assert torch.equal(_alpha(zeros_top, zeros_left), _alpha())
    # Each side independently, so a bug in one factor cannot be masked by the other.
    assert torch.equal(_alpha(disp_top=zeros_top), _alpha())
    assert torch.equal(_alpha(disp_left=zeros_left), _alpha())


# --- The core-safety invariant, which every seam mode must respect ------------------------
# Inside the core, `region` is RAW un-refined canvas (no tile has pasted there yet), so any
# alpha < 1 would blend raw pixels into finished content. The ramp is only ever written into
# the band slice, so this must hold for ANY displacement, however extreme.


def test_core_stays_exactly_one_under_extreme_displacement():
    for amp in (-4.0, -1.0, 1.0, 4.0):
        disp_top = torch.full((PASTE.x1 - PASTE.x0,), amp)
        disp_left = torch.full((PASTE.y1 - PASTE.y0,), amp)
        a = _alpha(disp_top, disp_left)
        assert torch.equal(a[TOP_BAND:, LEFT_BAND:], torch.ones(32, 32)), amp


def test_alpha_stays_in_range_and_finite_for_hostile_params():
    # Fractional k over a displaced (possibly negative) coordinate must never yield NaN: the
    # curve clamps the base to [0,1] before the power, so the base is never negative.
    for plateau, k in ((0.0, 1.0), (0.9, 6.0), (0.25, 2.5), (0.5, 1.3)):
        disp = torch.linspace(-3.0, 3.0, PASTE.x1 - PASTE.x0)
        a = _alpha(disp, disp.clone(), plateau=plateau, k=k)
        assert torch.isfinite(a).all(), (plateau, k)
        assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0, (plateau, k)


def test_uniform_displacement_shifts_the_transition_toward_the_seam():
    # A positive displacement moves the ramp toward the seam, so more of the band is handed
    # to this tile; a negative one hands more back to the neighbor. Compared in a core column
    # (ax == 1 there) so the top band is the pure vertical ramp.
    w = PASTE.x1 - PASTE.x0
    straight = _alpha()[:TOP_BAND, 40]
    pushed = _alpha(disp_top=torch.full((w,), 0.3))[:TOP_BAND, 40]
    pulled = _alpha(disp_top=torch.full((w,), -0.3))[:TOP_BAND, 40]
    assert float(pushed.sum()) > float(straight.sum()) > float(pulled.sum())


def test_displacement_makes_the_boundary_non_straight():
    # The point of the feature: with a varying displacement the iso-alpha contour is no
    # longer a straight line, i.e. the row where alpha first crosses 0.5 differs by column.
    w = PASTE.x1 - PASTE.x0
    disp = 0.4 * torch.sin(torch.linspace(0.0, 6.0, w))
    a = _alpha(disp_top=disp)
    crossings = (a[:TOP_BAND, LEFT_BAND:] >= 0.5).float().argmax(dim=0)
    assert int(crossings.max() - crossings.min()) >= 2       # the boundary genuinely meanders
    straight = (_alpha()[:TOP_BAND, LEFT_BAND:] >= 0.5).float().argmax(dim=0)
    assert int(straight.max() - straight.min()) == 0          # ...unlike the straight ramp


def test_separability_preserved_corner_is_product_of_the_two_seams():
    # Displacing a side turns its factor 2-D, but alpha must stay the product of a vertical
    # and a horizontal factor, so the corner is still the two seams' curves multiplied. Read
    # each factor off a CORE column/row, where the other factor is exactly 1 (uniform
    # displacement, so the top factor is the same in every column and vice versa).
    w, h = PASTE.x1 - PASTE.x0, PASTE.y1 - PASTE.y0
    a = _alpha(torch.full((w,), 0.2), torch.full((h,), -0.2))
    vertical = a[:, LEFT_BAND + 4]     # core column -> the top-seam factor alone
    horizontal = a[TOP_BAND + 4, :]    # core row    -> the left-seam factor alone
    assert torch.allclose(a, vertical[:, None] * horizontal[None, :], atol=1e-6)


# --- Minimum-error path ------------------------------------------------------------------


def test_min_error_path_follows_a_zero_cost_valley():
    # A synthetic error surface with one obvious low-cost route: the DP must find it rather
    # than cutting straight across.
    band, length = 8, 40
    err = torch.ones(band, length)
    valley = (torch.sin(torch.linspace(0.0, 3.0, length)) * 3 + 4).round().long().clamp(0, band - 1)
    err[valley, torch.arange(length)] = 0.0

    path = sampling._min_error_path(err)

    assert path.shape == (length,)
    assert torch.equal(path, valley)                      # routed exactly along the valley
    assert int((path[1:] - path[:-1]).abs().max()) <= 1    # connected: at most one step/column


def test_min_error_path_stays_in_band_and_is_connected_on_random_surfaces():
    for seed in (0, 1, 2):
        generator = torch.Generator().manual_seed(seed)
        err = torch.rand(6, 50, generator=generator)
        path = sampling._min_error_path(err)
        assert int(path.min()) >= 0 and int(path.max()) <= 5
        assert int((path[1:] - path[:-1]).abs().max()) <= 1


def test_min_error_path_beats_every_straight_cut():
    # The whole justification for the DP: its total cost must be <= the best straight line.
    generator = torch.Generator().manual_seed(11)
    err = torch.rand(8, 60, generator=generator)
    path = sampling._min_error_path(err)
    routed = float(err[path, torch.arange(60)].sum())
    best_straight = float(err.sum(dim=1).min())
    assert routed <= best_straight


def _crossing_for_valley(row, band, length=24, plateau=None, k=None):
    # Where the composited feather actually hands over, for a straight zero-cost cut at `row`.
    plateau = sampling.FEATHER_PLATEAU if plateau is None else plateau
    k = sampling.FEATHER_FALLOFF if k is None else k
    err = torch.ones(band, length)
    err[row, :] = 0.0
    disp = sampling._path_displacement(err, band, plateau, k)
    a = sampling._displaced_ramp(band, length, plateau, k, disp)
    crossing = (a >= 0.5).float().argmax(dim=0)
    assert int(crossing.min()) == int(crossing.max())   # a uniform cut -> a uniform crossing
    return int(crossing[0])


# --- seam_displacements: the wiring the composite actually runs through -------------------
# Everything above tests the pieces. These test that the RIGHT error surface reaches the
# RIGHT seam with the RIGHT orientation and sign — the part a containment-only test cannot
# see, and where a swapped, negated, reversed or core-sourced surface would go unnoticed
# because the output still "looks like" a feather.


PASTE_H = TILE.paste_rect.y1 - TILE.paste_rect.y0
PASTE_W = TILE.paste_rect.x1 - TILE.paste_rect.x0
TOP_BAND_T = TILE.core.y0 - TILE.paste_rect.y0
LEFT_BAND_T = TILE.core.x0 - TILE.paste_rect.x0


def _staircase(length, band):
    # A zero-cost cut that rises 1 -> band-2 along the seam, one row at a time (the DP only
    # allows |step| <= 1). Asymmetric on purpose: reversing it is a DIFFERENT answer, which
    # is what catches a path flipped along the seam.
    return torch.arange(length) * (band - 3) // (length - 1) + 1


def _planted_tile_inputs(seam):
    # sub/region over TILE.paste_rect whose squared difference is 0 exactly on one planted
    # staircase cut and 1 everywhere else, so the DP's answer is known in advance. Only ONE
    # seam is planted per call: a cut planted for the top seam spans every column and a cut
    # for the left seam spans every row, so planting both at once would hand each DP a free
    # shortcut through the other's zero line and neither answer would be predictable.
    diff = torch.ones(1, PASTE_H, PASTE_W, 3)
    if seam == "top":
        valley = _staircase(PASTE_W, TOP_BAND_T)
        diff[0, valley, torch.arange(PASTE_W), :] = 0.0    # one zero row per column
    else:
        valley = _staircase(PASTE_H, LEFT_BAND_T)
        diff[0, torch.arange(PASTE_H), valley, :] = 0.0    # one zero column per row
    return diff, torch.zeros(1, PASTE_H, PASTE_W, 3), valley


def _disp(seam):
    sub, region, valley = _planted_tile_inputs(seam)
    disp_top, disp_left = sampling.seam_displacements(TILE, sub, region)
    return disp_top, disp_left, valley


def _u50():
    return (0.5 ** (1.0 / sampling.FEATHER_FALLOFF)) * (1.0 - sampling.FEATHER_PLATEAU)


def test_min_error_routes_each_seam_from_its_own_band():
    assert PASTE_W != PASTE_H                     # a top/left swap cannot pass a shape check

    disp_top, _, top_valley = _disp("top")
    _, disp_left, left_valley = _disp("left")

    # Oriented correctly: the top seam runs along x, the left seam along y.
    assert disp_top.shape == (PASTE_W,)
    assert disp_left.shape == (PASTE_H,)

    # Each seam found ITS OWN planted cut, with the sign _displaced_ramp expects. Stated
    # against the KNOWN cut rather than by re-deriving the path, so a swapped, negated,
    # reversed or core-sourced error surface all fail here.
    assert torch.allclose(disp_top, _u50() - (top_valley.float() + 0.5) / TOP_BAND_T, atol=1e-6)
    assert torch.allclose(disp_left, _u50() - (left_valley.float() + 0.5) / LEFT_BAND_T, atol=1e-6)

    # The staircases really are asymmetric, so "reversed along the seam" is a distinct answer.
    assert not torch.allclose(disp_top, disp_top.flip(0), atol=1e-6)
    assert not torch.allclose(disp_left, disp_left.flip(0), atol=1e-6)


def test_min_error_reads_the_band_not_the_core():
    # The error surface must come from the overlap band only. Inside the core `region` is RAW
    # un-refined canvas, so its disagreement with `sub` is meaningless here — a cut routed by
    # it would be steered by noise. Planting a cut in the band and CONTRADICTORY garbage in
    # the core must leave the routing untouched.
    sub, region, valley = _planted_tile_inputs("top")
    sub_polluted = sub.clone()
    sub_polluted[:, TOP_BAND_T:, :, :] = torch.rand(1, PASTE_H - TOP_BAND_T, PASTE_W, 3) * 10.0

    clean, _, _ = _disp("top")
    polluted, _ = sampling.seam_displacements(TILE, sub_polluted, region)

    assert torch.equal(clean, polluted)
    assert torch.allclose(clean, _u50() - (valley.float() + 0.5) / TOP_BAND_T, atol=1e-6)


def test_min_error_alpha_handover_follows_the_planted_cut():
    # End to end: the alpha the compositor uses must hand over along the planted staircase,
    # not along a straight line. (The crossing tracks the cut monotonically rather than
    # landing on it — see test_path_displacement_biases_the_feather_toward_the_cut.)
    disp_top, disp_left, _ = _disp("top")

    alpha = sampling.feather_alpha(TILE.paste_rect, TILE.core, 16, TILE.kept_top, TILE.kept_left,
                                   disp_top=disp_top, disp_left=disp_left)
    # Read over CORE columns only, where the horizontal factor is exactly 1 and the top band
    # is therefore the pure vertical ramp; the left band's own feather would otherwise pull
    # the crossing down in the first LEFT_BAND_T columns.
    crossing = (alpha[:TOP_BAND_T, LEFT_BAND_T:] >= 0.5).float().argmax(dim=0).tolist()

    assert max(crossing) - min(crossing) >= 2                  # genuinely not a straight line
    assert crossing[0] < crossing[-1]                          # rises with the staircase
    assert all(b >= a for a, b in pairwise(crossing))           # tracks it over the whole seam


def test_path_displacement_biases_the_feather_toward_the_cut():
    # The honest contract, checked at the BAKED curve over EVERY row of the band. It is a
    # monotone BIAS, not an exact landing: _displaced_ramp tapers the shift by sin(pi*u), so
    # the realized crossing is pulled toward the cut but compressed away from the band ends
    # (see _path_displacement). An earlier version of this test asserted an exact landing at
    # one hand-picked row — the only window where that happens to hold.
    band = 32
    crossings = [_crossing_for_valley(row, band) for row in range(band)]

    # Monotone: a cut nearer the seam always moves the handover nearer the seam.
    assert all(b >= a for a, b in pairwise(crossings)), crossings
    # Genuinely responsive — it tracks the cut rather than sitting still.
    assert crossings[-1] - crossings[0] >= band // 2, crossings
    assert len(set(crossings)) >= band // 3, crossings
    # ...and always strictly inside the band, which is what the taper buys: the handover is
    # never shoved onto the core edge or the outer edge, where it would re-create a step.
    assert crossings[0] > 0 and crossings[-1] < band - 1, crossings


def test_path_displacement_attenuation_is_documented_accurately():
    # Pins the specific numbers _path_displacement's comment cites, so the comment cannot
    # quietly go stale (it is the thing that stops someone "fixing" the taper by accident).
    band = 32
    crossings = [_crossing_for_valley(row, band) for row in range(band)]
    assert (crossings[0], crossings[-1]) == (7, 26)
