import torch

from context_anchored_tile_refine import grid, sampling
from test_tiling import GridGuider, GridNoise, GridVAE, SIGMAS

# A real 2x2 layout with 16px overlap bands, so every rect below comes from build_layout
# rather than being hand-constructed: 160 with cap 96 and r=16 solves to n=2, base 80.
_SX = grid.solve_axis(160, 96, 0, 16)
_SY = grid.solve_axis(160, 96, 0, 16)
LAYOUT = grid.build_layout(160, 160, _SX, _SY, 0, 16)
FIRST = LAYOUT.tiles[0]      # (0,0) no kept side  -> sets the gauge
LEFT_ONLY = LAYOUT.tiles[1]  # (0,1) kept_left     -> one constraint, simple arithmetic
BOTH = LAYOUT.tiles[3]       # (1,1) kept top+left -> two constraints to combine


def _paste_shape(tile, channels=3):
    paste = tile.paste_rect
    return 1, paste.y1 - paste.y0, paste.x1 - paste.x0, channels


def _pair(tile, diff, channels=3):
    """(sub, region) whose difference over the paste rect is exactly `diff`."""
    region = torch.zeros(_paste_shape(tile, channels))
    return region + diff, region


def _band_counts(tile, overlap):
    """(top, left) sample counts under the DISJOINT split: the left band starts below the
    top band, so the band x band corner belongs to the top band only."""
    paste = tile.paste_rect
    return overlap * (paste.x1 - paste.x0), (paste.y1 - paste.y0 - overlap) * overlap


def test_layout_fixture_is_the_geometry_the_tests_assume():
    assert (_SX.n, _SY.n) == (2, 2)
    assert (BOTH.core.x0, BOTH.core.y0) == (80, 80)
    assert (BOTH.paste_rect.x0, BOTH.paste_rect.y0) == (64, 64)
    assert (BOTH.kept_top, BOTH.kept_left) == (True, True)
    assert (LEFT_ONLY.kept_top, LEFT_ONLY.kept_left) == (False, True)
    assert (FIRST.kept_top, FIRST.kept_left) == (False, False)


# --- seam_dc_offset -----------------------------------------------------------------------

def test_first_tile_has_no_neighbour_so_it_sets_the_gauge():
    # None, not zero: the caller must be able to tell "no constraint" from "measured zero".
    sub, region = _pair(FIRST, 0.004)
    assert sampling.seam_dc_offset(FIRST, sub, region) is None


def test_uniform_disagreement_is_recovered_exactly():
    sub, region = _pair(LEFT_ONLY, 0.004)
    offset = sampling.seam_dc_offset(LEFT_ONLY, sub, region)
    assert torch.allclose(offset, torch.full((3,), 0.004), atol=1e-7)


def test_subtracting_the_offset_lands_the_band_on_the_neighbour():
    # The property the whole mechanism exists for: after correction the shared band agrees.
    sub, region = _pair(LEFT_ONLY, 0.004)
    offset = sampling.seam_dc_offset(LEFT_ONLY, sub, region)
    band = LEFT_ONLY.core.x0 - LEFT_ONLY.paste_rect.x0
    corrected = sub - offset
    assert torch.allclose(corrected[:, :, :band, :].mean(dim=(0, 1, 2)), region[:, :, :band, :].mean(dim=(0, 1, 2)), atol=1e-7)


def test_channels_are_measured_independently():
    # The offset is partly chroma, so a single luminance scalar would leave a hue step.
    diff = torch.tensor([0.004, -0.002, 0.009])
    sub, region = _pair(LEFT_ONLY, diff)
    offset = sampling.seam_dc_offset(LEFT_ONLY, sub, region)
    assert torch.allclose(offset, diff, atol=1e-7)


def test_only_the_band_is_measured_not_the_core():
    # Poison the core with a huge value. It must not move the offset, because the core is
    # where `region` is still RAW canvas -- differencing there would be meaningless.
    band = LEFT_ONLY.core.x0 - LEFT_ONLY.paste_rect.x0
    sub, region = _pair(LEFT_ONLY, 0.004)
    sub[:, :, band:, :] += 5.0
    offset = sampling.seam_dc_offset(LEFT_ONLY, sub, region)
    assert torch.allclose(offset, torch.full((3,), 0.004), atol=1e-7)


def test_median_resists_a_high_contrast_edge_crossing_the_band():
    # A silhouette crossing the seam makes the two refinements genuinely disagree by a lot
    # over a few pixels. That is texture disagreement, not a DC offset, and it must not drag
    # the correction. The median stays exactly on the uncontaminated band value; the band's
    # own MEAN is asserted alongside it to show what that robustness is worth here.
    band = LEFT_ONLY.core.x0 - LEFT_ONLY.paste_rect.x0
    sub, region = _pair(LEFT_ONLY, 0.002)
    sub[:, :20, :band, :] += 0.048          # 20 of 80 rows inside the band
    offset = sampling.seam_dc_offset(LEFT_ONLY, sub, region)
    contaminated_mean = float((sub[:, :, :band, :] - region[:, :, :band, :]).mean())
    assert contaminated_mean > 0.002 + 0.008
    assert torch.allclose(offset, torch.full((3,), 0.002), atol=1e-7)


def test_the_larger_band_wins_and_the_corner_is_counted_once():
    # Under a median, pooling disjoint bands means the LARGER band's value wins outright.
    # Here the top band is 16x96 = 1536 samples and the left band only (96-16)x16 = 1280,
    # so the answer is the top band's `a`.
    #
    # That doubles as the disjointness assertion, and it does not depend on the reduction:
    # if the band x band corner were counted twice (a full 96x16 left band) the two counts
    # would tie at 1536, and torch.median takes the LOWER of the two middle values, so the
    # answer would flip to `b`.
    a, b = 0.006, -0.003
    top, left = _band_counts(BOTH, 16)
    assert (top, left) == (1536, 1280)
    sub, region = _pair(BOTH, 0.0)
    sub[:, :16, :, :] = a
    sub[:, 16:, :16, :] = b
    offset = sampling.seam_dc_offset(BOTH, sub, region)
    assert torch.allclose(offset, torch.full((3,), a), rtol=0, atol=1e-7)


# A tile whose LEFT band outnumbers its top band, which BOTH cannot produce: paste 56x80
# there, so top = 16x56 = 896 and left = (80-16)x16 = 1024. Without it "the larger band
# wins" is indistinguishable from "the top band always wins".
_TALL = grid.build_layout(80, 240, grid.solve_axis(80, 64, 0, 16), grid.solve_axis(240, 96, 0, 16), 0, 16)
LEFT_HEAVY = _TALL.tiles[3]   # row 1, col 1: core 40 wide x 64 tall


def test_the_winning_band_is_the_bigger_one_not_the_top_one():
    assert (LEFT_HEAVY.kept_top, LEFT_HEAVY.kept_left) == (True, True)
    top, left = _band_counts(LEFT_HEAVY, 16)
    assert (top, left) == (896, 1024)      # the opposite ordering to BOTH
    a, b = 0.006, -0.003
    sub, region = _pair(LEFT_HEAVY, 0.0)
    sub[:, :16, :, :] = a
    sub[:, 16:, :16, :] = b
    offset = sampling.seam_dc_offset(LEFT_HEAVY, sub, region)
    assert torch.allclose(offset, torch.full((3,), b), rtol=0, atol=1e-7)


def test_zero_overlap_has_no_band_to_measure_and_does_not_raise():
    # context_overlap=0 makes paste_rect == core, so a tile has kept sides but no shared
    # band. That is a silent no-op (None), never an error.
    zero = grid.build_layout(160, 160, grid.solve_axis(160, 96, 0, 0), grid.solve_axis(160, 96, 0, 0), 0, 0)
    tile = zero.tiles[3]
    assert (tile.kept_top, tile.kept_left) == (True, True)
    assert tile.paste_rect.x0 == tile.core.x0 and tile.paste_rect.y0 == tile.core.y0
    sub, region = _pair(tile, 0.004)
    assert sampling.seam_dc_offset(tile, sub, region) is None


def test_clamp_bounds_a_pathological_estimate():
    sub, region = _pair(LEFT_ONLY, 1.0)
    offset = sampling.seam_dc_offset(LEFT_ONLY, sub, region)
    assert torch.allclose(offset, torch.full((3,), sampling.DC_MATCH_CLAMP), atol=1e-7)
    sub, region = _pair(LEFT_ONLY, -1.0)
    offset = sampling.seam_dc_offset(LEFT_ONLY, sub, region)
    assert torch.allclose(offset, torch.full((3,), -sampling.DC_MATCH_CLAMP), atol=1e-7)


# --- the region gate on the mask path ------------------------------------------------------

def test_region_gate_keeps_only_the_in_mask_band_samples():
    # Band pixels outside the region are frozen (denoise 0) and discarded at composite time,
    # so they carry VAE crop-framing disagreement over pixels the output never uses. Poison
    # them heavily: the gated measurement must return the in-mask value untouched.
    band = LEFT_ONLY.core.x0 - LEFT_ONLY.paste_rect.x0
    sub, region = _pair(LEFT_ONLY, 0.002)
    region_mask = torch.zeros(_paste_shape(LEFT_ONLY)[:3])
    region_mask[:, :24, :] = 1.0             # rows 0..24 of an 80-row paste are in-region
    sub[:, 24:, :band, :] += 0.5             # out-of-region band pixels, wildly off
    offset = sampling.seam_dc_offset(LEFT_ONLY, sub, region, region_mask)
    assert torch.allclose(offset, torch.full((3,), 0.002), atol=1e-7)
    # Ungated, the same inputs are dragged all the way to the clamp -- so the gate is what
    # is doing the work above, not the median.
    ungated = sampling.seam_dc_offset(LEFT_ONLY, sub, region)
    assert torch.allclose(ungated, torch.full((3,), sampling.DC_MATCH_CLAMP), atol=1e-7)


def test_too_few_in_mask_samples_take_no_correction():
    # A median over a handful of pixels can sit on an outlier and would then shift a WHOLE
    # tile, so below the floor the tile is left alone. None, not zero: it is "no measurement",
    # and the DC_MATCH_CLAMP rail is not a substitute for having enough samples.
    band = LEFT_ONLY.core.x0 - LEFT_ONLY.paste_rect.x0
    rows = (sampling.DC_MATCH_MIN_SAMPLES // band) - 1
    sub, region = _pair(LEFT_ONLY, 0.002)
    region_mask = torch.zeros(_paste_shape(LEFT_ONLY)[:3])
    region_mask[:, :rows, :] = 1.0
    assert rows * band < sampling.DC_MATCH_MIN_SAMPLES
    assert sampling.seam_dc_offset(LEFT_ONLY, sub, region, region_mask) is None
    # One more row clears the floor, so the floor is what refused it -- not the geometry.
    region_mask[:, :rows + 1, :] = 1.0
    assert (rows + 1) * band >= sampling.DC_MATCH_MIN_SAMPLES
    assert sampling.seam_dc_offset(LEFT_ONLY, sub, region, region_mask) is not None


# --- end to end ----------------------------------------------------------------------------

# 80 with cap 64 and r=16 solves to a 2x2 grid of 40px cores -- the smallest real grid that
# still has an 8px overlap band and an 8px anchor ring, i.e. the production shape in
# miniature. GridVAE strides by 8, so every crop extent here stays a multiple of 8.
E2E = dict(overlap=8, ctx=8, cap=64)


def _refine(image, guider=None, mask=None, overlap=E2E["overlap"], ctx=E2E["ctx"], cap=E2E["cap"]):
    return sampling.refine_image(image, GridGuider() if guider is None else guider, object(), SIGMAS,
                                 GridVAE(), GridNoise(), max_tile_width=cap, max_tile_height=cap,
                                 context_anchor=ctx, context_overlap=overlap, mask=mask)


def _disable_dc(monkeypatch):
    """The uncorrected control arm. The correction is unconditional and the public API no
    longer offers a way off it, so "the same run without it" has to be stubbed at the call
    site. Every use below renders the corrected arm FIRST, then patches."""
    monkeypatch.setattr(sampling, "seam_dc_offset", lambda *args, **kwargs: None)


class DriftGuider(GridGuider):
    """GridGuider plus a per-tile constant added to the latent, i.e. exactly the defect
    under test: every tile lands at a slightly different DC. decode is a nearest-neighbour
    upsample, so a latent constant becomes the same pixel constant across the whole tile."""

    def __init__(self, biases):
        super().__init__()
        self.biases = biases

    def sample(self, *args, **kwargs):
        samples = super().sample(*args, **kwargs)
        return samples + self.biases[len(self.calls) - 1]


# Small enough that no correction reaches DC_MATCH_CLAMP.
BIASES = (0.002, -0.003, 0.005, -0.001)

# Float32 headroom for the FAKES, not for the correction. GridNoise is an arange over the
# whole canvas, so a latent runs into the hundreds and a band difference of ~0.005 is computed
# by cancelling two O(100) numbers: each pixel's difference is quantized to ~ulp(300) ~ 3e-5.
# A mean averages those quanta away, but the median lands on one of them, so the recovered
# offset carries that much error. Measured worst case over 60 seeds: 6.1e-5, against a signal
# 80x larger. Every "the correction fired" assertion below stays well outside this.
FAKE_FP32_SLACK = 1e-4


def test_band_match_is_a_no_op_when_the_tiles_already_agree(comfy_stubs, monkeypatch):
    # The fakes are position-equivariant: both tiles encode the shared band from the same
    # frozen source and get the same spatially-anchored noise slice, so their refinements of
    # it are identical and the measured disagreement is exactly zero. The correction must
    # then do NOTHING -- it may only ever remove a difference that is really there. Exact
    # equality is honest under the median, which returns 0.0 on the nose here.
    image = torch.rand(1, 80, 80, 3)
    corrected = _refine(image)
    _disable_dc(monkeypatch)
    assert torch.equal(corrected, _refine(image))


def test_band_match_removes_an_injected_per_tile_drift(comfy_stubs):
    # Inject a known per-tile DC offset and require the correction to take it back out.
    # Tile 0 is the gauge and keeps its own bias, so every later tile is pulled onto
    # BIASES[0] and the corrected canvas is the undrifted one shifted by that constant.
    image = torch.rand(1, 80, 80, 3)
    clean = _refine(image)
    corrected = _refine(image, guider=DriftGuider(BIASES))
    assert torch.allclose(corrected, clean + BIASES[0], atol=FAKE_FP32_SLACK)


def test_uncorrected_drift_really_does_leave_a_seam(comfy_stubs, monkeypatch):
    # Guards the test above from passing vacuously: without the correction the same
    # injected drift must NOT collapse onto a single constant.
    image = torch.rand(1, 80, 80, 3)
    clean = _refine(image)
    _disable_dc(monkeypatch)
    drifted = _refine(image, guider=DriftGuider(BIASES))
    assert not torch.allclose(drifted, clean + BIASES[0], atol=FAKE_FP32_SLACK)


def test_gauge_tile_is_uncorrected_and_the_correction_stays_in_its_own_tile(comfy_stubs, monkeypatch):
    # Must run under DRIFT: with the plain fakes every offset is zero, so this would assert
    # a subset of an equality that already holds everywhere and could not fail.
    # Tile (0,0) has no processed neighbour, so it takes no correction. The part of its core
    # that no later tile feathers into must therefore be bit-identical between the corrected
    # and uncorrected drifted runs — and must DIFFER from the undrifted run, which is what
    # proves the region really does carry tile 0's bias.
    image = torch.rand(1, 80, 80, 3)
    axis = grid.solve_axis(80, E2E["cap"], E2E["ctx"], E2E["overlap"])
    layout = grid.build_layout(80, 80, axis, axis, E2E["ctx"], E2E["overlap"])
    first = layout.tiles[0]
    safe_x = first.core.x1 - layout.overlap
    safe_y = first.core.y1 - layout.overlap
    assert safe_x > 0 and safe_y > 0

    clean = _refine(image)
    corrected = _refine(image, guider=DriftGuider(BIASES))
    _disable_dc(monkeypatch)
    drifted = _refine(image, guider=DriftGuider(BIASES))
    window = (slice(None), slice(None, safe_y), slice(None, safe_x), slice(None))
    assert torch.equal(corrected[window], drifted[window])
    assert not torch.allclose(corrected[window], clean[window], rtol=0, atol=FAKE_FP32_SLACK)


def test_routing_sees_the_corrected_tile_not_the_raw_decode(comfy_stubs, monkeypatch):
    # The DC correction is applied BEFORE seam_displacements so the min-error surface
    # describes texture rather than a constant. Nothing downstream can observe that ordering
    # (a constant surface routes identically to a zero one), so pin it at the call itself.
    seen = []
    real = sampling.seam_displacements

    def spy(tile, sub, region):
        band = tile.core.x0 - tile.paste_rect.x0
        if tile.kept_left and band > 0:
            seen.append((sub[:, :, :band, :] - region[:, :, :band, :]).mean().item())
        return real(tile, sub, region)

    monkeypatch.setattr(sampling, "seam_displacements", spy)
    _refine(torch.rand(1, 80, 80, 3), guider=DriftGuider(BIASES))
    assert seen, "no kept-left tile reached seam_displacements"
    # Corrected: the routing surface sees a band that already agrees.
    assert max(abs(v) for v in seen) < FAKE_FP32_SLACK, seen

    seen.clear()
    _disable_dc(monkeypatch)
    _refine(torch.rand(1, 80, 80, 3), guider=DriftGuider(BIASES))
    # Uncorrected, same drift: it does not, by more than an order of magnitude. Without this
    # the assertion above could pass simply because the fakes never disagree.
    assert max(abs(v) for v in seen) > 1e-3, seen


def test_zero_overlap_runs_as_a_silent_no_op(comfy_stubs, monkeypatch):
    # No shared band anywhere in the grid, so nothing is measurable. The run must complete
    # and land byte-for-byte on the uncorrected arm rather than raising. Under DRIFT so the
    # tiles genuinely disagree: any correction that did fire would show up as a difference.
    image = torch.rand(1, 80, 80, 3)
    corrected = _refine(image, guider=DriftGuider(BIASES), overlap=0)
    _disable_dc(monkeypatch)
    assert torch.equal(corrected, _refine(image, guider=DriftGuider(BIASES), overlap=0))


# --- the mask path -------------------------------------------------------------------------

# 112x112 with a mask over [16,96) expands by ctx=8 and snaps to a 96x96 crop at (8,8),
# which tiles 2x2 at cap 64 -- so the crop has real interior seams AND a real region gate,
# and the crop itself does not touch the image border.
MASK_IMAGE = 112
MASK_BOX = (16, 96)


def _mask_inputs():
    image = torch.rand(1, MASK_IMAGE, MASK_IMAGE, 3)
    mask = torch.zeros(1, MASK_IMAGE, MASK_IMAGE)
    mask[:, MASK_BOX[0]:MASK_BOX[1], MASK_BOX[0]:MASK_BOX[1]] = 1.0
    return image, mask


def test_mask_path_corrects_tile_seams_inside_the_region(comfy_stubs, monkeypatch):
    # The correction applies at TILE seams on the mask path too, measured only over band
    # pixels inside the region. DriftGuider is required: with the plain fakes the median is
    # exactly 0.0 everywhere and this would pass vacuously.
    image, mask = _mask_inputs()
    corrected = _refine(image, guider=DriftGuider(BIASES), mask=mask)
    _disable_dc(monkeypatch)
    uncorrected = _refine(image, guider=DriftGuider(BIASES), mask=mask)
    assert corrected.shape == (1, MASK_IMAGE, MASK_IMAGE, 3)
    assert not torch.equal(corrected, uncorrected)


def test_mask_path_hands_the_region_gate_down_to_the_measurement(comfy_stubs, monkeypatch):
    # Pins the WIRING, which nothing else reaches: _refine_tiles could pass region_mask=None
    # and every other mask test here would still pass, because they assert on the OUTPUT and
    # this geometry's gated and ungated measurements agree closely enough. Capture the
    # argument itself instead.
    image, mask = _mask_inputs()
    seen = []
    real = sampling.seam_dc_offset

    def spy(tile, sub, region, region_mask=None):
        seen.append((tile, None if region_mask is None else region_mask.clone()))
        return real(tile, sub, region, region_mask)

    monkeypatch.setattr(sampling, "seam_dc_offset", spy)
    _refine(image, guider=DriftGuider(BIASES), mask=mask)

    assert seen, "seam_dc_offset was never called on the mask path"
    gated = [(tile, region_mask) for tile, region_mask in seen if region_mask is not None]
    assert gated, "the mask path handed seam_dc_offset region_mask=None -- the gate is not wired"
    for tile, region_mask in gated:
        paste = tile.paste_rect
        assert region_mask.shape == (1, paste.y1 - paste.y0, paste.x1 - paste.x0)
    # An all-ones gate would be indistinguishable from no gate, so at least one tile's mask
    # must actually exclude something -- i.e. it is the real region, sliced, not a stand-in.
    assert any((region_mask > 0).any() and (region_mask == 0).any() for _, region_mask in gated)


def test_mask_path_correction_leaves_the_background_byte_identical(comfy_stubs):
    # The background is never shaded by the DC match: outside the crop, and wherever the 1px
    # anti-alias is 0 inside it, the output is still the original pixels bit for bit.
    image, mask = _mask_inputs()
    out = _refine(image, guider=DriftGuider(BIASES), mask=mask)

    y0, y1, x0, x1 = sampling._expand_snap_clamp(sampling._mask_bbox(mask >= 0.5), E2E["ctx"], MASK_IMAGE, MASK_IMAGE)
    assert (y0, y1, x0, x1) == (8, 104, 8, 104)
    outside = torch.ones(1, MASK_IMAGE, MASK_IMAGE, dtype=torch.bool)
    outside[:, y0:y1, x0:x1] = False
    assert torch.equal(out[outside], image[outside])
    aa = sampling._aa_alpha((mask >= 0.5)[:, y0:y1, x0:x1].to(image.dtype))
    zero = aa == 0
    assert zero.any()
    assert torch.equal(out[:, y0:y1, x0:x1, :][zero], image[:, y0:y1, x0:x1, :][zero])
