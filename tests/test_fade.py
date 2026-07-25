import torch

from context_anchored_tile_refine import grid, sampling

# Hand-computable geometry: 64x64 crop, core in the bottom-right quadrant, so the
# top/left sides sit outside the core (0 in the binary mask).
CROP = grid.Rect(x0=0, y0=0, x1=64, y1=64)
CORE = grid.Rect(x0=32, y0=32, x1=64, y1=64)


# --- tile_gradient is now ALWAYS the binary core indicator (the fractional 1->0 ramp
# was removed with the pixel-space blend and the directional-feather rewrite). ---


def test_gradient_binary_core_indicator_latent_scale():
    m = sampling.tile_gradient(CROP, CORE, scale=8)

    assert m.shape == (8, 8)
    assert m.dtype == torch.float32
    # 1 for every cell whose center lies inside the core (cells 4..7 on each axis), 0 else.
    expected = torch.zeros(8, 8)
    expected[4:, 4:] = 1.0
    assert torch.equal(m, expected)
    assert ((m == 0.0) | (m == 1.0)).all()


def test_gradient_binary_core_indicator_pixel_scale():
    a = sampling.tile_gradient(CROP, CORE, scale=1)

    assert a.shape == (64, 64)
    expected = torch.zeros(64, 64)
    expected[32:, 32:] = 1.0
    assert torch.equal(a, expected)


def test_gradient_all_ones_when_crop_equals_core():
    rect = grid.Rect(x0=16, y0=24, x1=56, y1=64)

    for scale in (1, 8):
        m = sampling.tile_gradient(rect, rect, scale=scale)
        assert torch.equal(m, torch.ones(40 // scale, 40 // scale))


def test_gradient_one_sided_binary_indicator():
    # Left neighbor only: every row identical, hard step from 0 to 1 at the core edge.
    crop = grid.Rect(x0=0, y0=0, x1=56, y1=40)
    core = grid.Rect(x0=16, y0=0, x1=56, y1=40)

    m = sampling.tile_gradient(crop, core, scale=8)

    assert m.shape == (5, 7)
    expected_row = torch.tensor([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert torch.equal(m, expected_row.expand(5, 7))
