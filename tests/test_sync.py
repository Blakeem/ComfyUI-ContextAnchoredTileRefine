"""sync.py: the sync engine's component layer and its no-mask run loop.

Geometry the COMPONENT tests share: an 80x80 /8-aligned canvas at cap 56, context_anchor 8,
context_overlap 8 -> a 2x2 grid, base 40, ring r = 16. At /8 that is a 10x10 latent canvas;
the left tile's window is cells [0:7]x[0:7] and the right tile's [0:7]x[3:10], so the two
share columns 3..6 — the band every "both lanes see one value" assertion is made over.

Geometry the RUN tests share (see the run-loop section for the full arithmetic): the same
80x80 canvas at cap 72x80, context_overlap 16 -> a 2x1 grid, so the whole run is TWO lanes
whose 8-cell windows meet in one 2-cell feathered band, and every canvas cell is a
hand-computable blend of two per-lane constants. The REGION runs reuse that grid: a mask band
over pixel rows 24..55 crops to rows 16..63 at context_anchor 8, and a 48x80 crop solves to
the same two columns over six latent rows.

CPU only, no model loads: the VAE, noise, guider and CLIP are position-preserving fakes, so
a value landing at the wrong cell is a test failure rather than a rounding difference. Every
test that reaches comfy (the CONST precondition, the conditioning pre-pass, the whole run
loop) runs under the `comfy_stubs` fixture and says so in its signature; the pure-math cases
take no fixture at all. Nothing here needs the real `comfy_env` install.
"""
import copy
from types import SimpleNamespace

import pytest
import torch
from test_stepper import run_bounded
from test_tiling import GridNoise, GridVAE, _layout
from test_vl import FakeVLClip

from conftest import BaseModel
from context_anchored_tile_refine import captions, conds, grid, sampling, sync, vl

# Strictly decreasing, ends at 0, starts below 1 — every unconditional precondition satisfied
# and the live-canvas ring algebra defined.
SIGMAS = torch.linspace(0.8, 0.0, 5)
# denoise 1.0 under sgm_uniform/normal: sigmas[0] is exactly 1.0. A currently-working widget
# value, so "source image" must accept it.
FULL_SIGMAS = torch.linspace(1.0, 0.0, 5)

CANVAS = 80
CAP = 56
CTX = 8
OVERLAP = 8

# The encode geometry the fake CLIP is pinned to: 256x256 -> an 8x8 merged-cell grid over the
# 80px canvas, so neighbouring tiles really do select different rows.
ENC = 256
ENC_ROWS = (ENC // vl.MERGED_CELL) ** 2
ENC_SEQ = 1 + ENC_ROWS + 1 + 4


# ---- fakes -----------------------------------------------------------------


class ModelPatcher:
    """comfy's ModelPatcher cut to the three members the engine reads."""

    def __init__(self, model):
        self.model = model
        self.load_device = torch.device("cpu")

    def get_model_object(self, name):
        assert name == "model_sampling"
        return self.model.model_sampling


class SyncGuider:
    """comfy's CFGGuider cut to what `_prepare_run` touches: the patcher and the
    original_conds convention the per-lane positive swap keys on."""

    def __init__(self, model=None, negative=None):
        self.model_patcher = ModelPatcher(BaseModel() if model is None else model)
        self.original_conds = {
            "positive": [{"cross_attn": torch.zeros(1, 1, 8)}],
            "negative": [{"cross_attn": torch.ones(1, 1, 8)}] if negative is None else negative,
        }


class LostCondsGuider(SyncGuider):
    """A third-party guider pack whose copy does not carry original_conds."""

    def __copy__(self):
        return SimpleNamespace(model_patcher=self.model_patcher)


class OverridingModel(BaseModel):
    """WAN21/WAN22/HunyuanVideo/LTXAV: already holds the frozen region clean, so the ring
    construction the engine is derived for does not apply."""

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        return latent_image


def image_canvas(size=CANVAS):
    # Distinct value per pixel, so a mis-placed encode window changes values.
    return torch.arange(size * size * 3, dtype=torch.float32).reshape(1, size, size, 3)


def square_region(lo=33, hi=63, size=CANVAS):
    # A deliberately NON-/8 region: pixels 33..62 are covered by latent cells 4..7, so the
    # max-pool cover has to over-cover rather than drop the mask's edge.
    region = torch.zeros(1, size, size)
    region[:, lo:hi, lo:hi] = 1.0
    return region


def band_mask(rows=(24, 56), size=CANVAS):
    # A full-width band, so the region crop keeps the RUN geometry's two columns.
    mask = torch.zeros(1, size, size)
    mask[:, rows[0]:rows[1], :] = 1.0
    return mask


@pytest.fixture
def vl_clip(monkeypatch):
    # The encode geometry pinned to the fake CLIP's sequence length. _convert is deliberately
    # NOT stubbed: the positives run through comfy.sampler_helpers.convert_cond (comfy_stubs')
    # exactly as production does, so a lane's positive is the flat cond dict core consumes.
    monkeypatch.setattr(vl, "resample_for_global", lambda source: (source, ENC, ENC))
    return FakeVLClip(seq_override=ENC_SEQ)


def prepare(clip, guider=None, image=None, vae=None, noise=None, sigmas=SIGMAS,
            vlm_method=captions.VLM_METHOD_VISION, anchor_source=sync.ANCHOR_SOURCE_IMAGE,
            batch_size=1, batch_index=0, ctx=CTX, overlap=OVERLAP, region_pixel=None,
            vl_context=None):
    guider = SyncGuider() if guider is None else guider
    return sync._prepare_run(
        image_canvas() if image is None else image, guider, sigmas,
        GridVAE() if vae is None else vae, GridNoise() if noise is None else noise,
        CAP, CAP, ctx, overlap, clip, vlm_method=vlm_method, anchor_source=anchor_source,
        batch_size=batch_size, batch_index=batch_index, region_pixel=region_pixel,
        vl_context=vl_context)


def expected_layout(ctx=CTX, overlap=OVERLAP):
    return _layout(CANVAS, CANVAS, CAP, CAP, ctx=ctx, overlap=overlap)


# ---- the module contract ---------------------------------------------------


def test_module_exposes_the_component_layer_and_the_engine_entry():
    # Smoke import: nothing imports sync.py yet (the wiring feature does), so this is what
    # keeps the module loadable in the default gate.
    for name in ("encode_canvas_latent", "check_preconditions", "build_canvas_noise",
                 "build_tile_positives", "build_lane_guiders", "_prepare_run",
                 "latent_plateau", "build_blends", "blend_lanes", "make_sync_progress",
                 "present_live_ring", "decode_composite", "_refine_canvas", "refine_sync"):
        assert callable(getattr(sync, name))
    assert sync.ANCHOR_SOURCES == (sync.ANCHOR_SOURCE_IMAGE, sync.ANCHOR_SOURCE_CANVAS)


# ---- the shared denoise-mask helper ----------------------------------------


def test_normalize_denoise_mask_is_the_canonical_form_for_a_4d_latent():
    mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)

    normalized = sampling._normalize_denoise_mask(mask, torch.zeros(1, 4, 2, 2), torch.device("cpu"))

    assert normalized.shape == (1, 1, 2, 2)
    assert normalized.dtype == torch.float32
    assert torch.equal(normalized[0, 0], mask.float())


def test_normalize_denoise_mask_matches_a_5d_video_latent_rank():
    # reshape_mask folds any sub-5-D mask headed for a 5-D latent into (1,1,-1,h,w), pooling
    # the batch into the temporal axis — so the rank has to match before core ever sees it.
    mask = torch.ones(2, 2)

    normalized = sampling._normalize_denoise_mask(mask, torch.zeros(1, 4, 1, 2, 2), torch.device("cpu"))

    assert normalized.shape == (1, 1, 1, 2, 2)


def test_lane_masks_are_binary_and_already_normalized(comfy_stubs, vl_clip):
    # The stepper calls guider.sample itself, so sample_latent (which normalizes on the raster
    # path) is not on this path at all: the lanes must arrive canonical.
    run = prepare(vl_clip)
    layout = expected_layout()

    for tile, lane in zip(layout.tiles, run.lanes, strict=True):
        expected = sampling.tile_gradient(tile.crop_rect, tile.overlap_inner_rect, scale=8)
        assert lane.denoise_mask.shape == (1, 1, *expected.shape)
        assert lane.denoise_mask.dtype == torch.float32
        assert torch.equal(lane.denoise_mask[0, 0], expected)
        assert set(lane.denoise_mask.unique().tolist()) <= {0.0, 1.0}
    # Hand-computed: tile 0's crop is 56x56 (7x7 cells) and its diffused region 48x48 (6x6),
    # so exactly the last row and column of the window are the frozen anchor ring.
    first = run.lanes[0].denoise_mask[0, 0]
    assert first.shape == (7, 7)
    assert torch.equal(first[:6, :6], torch.ones(6, 6))
    assert float(first[6, :].sum()) == 0.0 and float(first[:, 6].sum()) == 0.0


def test_without_a_region_the_ring_gate_is_the_whole_frozen_anchor(comfy_stubs, vl_clip):
    # The two gates partition every lane's window: what it diffuses, and what it freezes.
    run = prepare(vl_clip)

    for lane, gate in zip(run.lanes, run.ring_gates, strict=True):
        assert torch.equal(gate, 1.0 - lane.denoise_mask)


# ---- the region gate -------------------------------------------------------


def test_the_region_gate_is_the_max_pool_cover_of_the_mask(comfy_stubs, vl_clip):
    # The raster path's rule verbatim (sampling._refine_tiles): cover-downsample by 8 with
    # max_pool2d, NOT avg+threshold, so every pixel with region == 1 lands in a DIFFUSED cell.
    # Hand-computed: latent cell k spans pixels 8k..8k+7, so the region's pixels 33..62 cover
    # cells 4..7 — the gate OVER-covers a non-/8 mask instead of dropping its edge.
    run = prepare(vl_clip, region_pixel=square_region())
    layout = expected_layout()

    expected_region = torch.zeros(1, 10, 10)
    expected_region[:, 4:8, 4:8] = 1.0
    for tile, lane in zip(layout.tiles, run.lanes, strict=True):
        rect = sync._latent_rect(tile.crop_rect)
        gradient = sampling.tile_gradient(tile.crop_rect, tile.overlap_inner_rect, scale=8)
        expected = gradient * expected_region[:, rect.y0:rect.y1, rect.x0:rect.x1]
        assert torch.equal(lane.denoise_mask[0], expected)
        assert set(lane.denoise_mask.unique().tolist()) <= {0.0, 1.0}
    # Hand-computed pin on tile 0: it diffuses cells 0..5 on both axes and the region covers
    # 4..7, so exactly the 2x2 block {4,5} x {4,5} is released. Everything else in its window
    # is frozen — the background outside the region as well as its own context_anchor ring.
    released = torch.nonzero(run.lanes[0].denoise_mask[0, 0])
    assert released.tolist() == [[4, 4], [4, 5], [5, 4], [5, 5]]


def test_the_ring_gate_is_the_region_the_lane_leaves_to_a_neighbour(comfy_stubs, vl_clip):
    # What the live-canvas rewrite targets: mask-0 cells INSIDE the region, i.e. the ones some
    # NEIGHBOUR diffuses. Tile 0's window keeps the L where one axis is its ring cell 6 and the
    # region is 1. Mask-0 cells OUTSIDE the region are the frozen background, which NO lane
    # ever diffuses — they must stay 0 here, or the surgery would present a trajectory that
    # nobody is refining.
    run = prepare(vl_clip, region_pixel=square_region())

    ring = torch.nonzero(run.ring_gates[0][0, 0])
    assert ring.tolist() == [[4, 6], [5, 6], [6, 4], [6, 5], [6, 6]]
    for lane, gate in zip(run.lanes, run.ring_gates, strict=True):
        assert gate.shape == lane.denoise_mask.shape
        assert gate.dtype == torch.float32
        assert set(gate.unique().tolist()) <= {0.0, 1.0}
        # Disjoint by construction: a cell is either diffused here or frozen here, never both.
        assert float((gate * lane.denoise_mask).sum()) == 0.0


# ---- preconditions ---------------------------------------------------------


def _check(sigmas=SIGMAS, model=None, anchor_source=sync.ANCHOR_SOURCE_IMAGE):
    guider = SyncGuider(model=model)
    return sync.check_preconditions(guider.model_patcher, sigmas, anchor_source)


def test_precondition_rejects_a_non_monotone_schedule(comfy_stubs):
    with pytest.raises(RuntimeError, match="not strictly decreasing"):
        _check(sigmas=torch.tensor([0.8, 0.4, 0.4, 0.0]))


def test_precondition_rejects_a_schedule_that_does_not_end_at_zero(comfy_stubs):
    with pytest.raises(RuntimeError, match="does not end at sigma 0"):
        _check(sigmas=torch.tensor([0.8, 0.5, 0.2]))


def test_precondition_rejects_a_non_const_model_sampling(comfy_stubs):
    # The ring algebra IS CONST's noise_scaling inverted; on an EPS/V_PREDICTION model the
    # same expression mis-scales silently, so this is a hard error.
    class EPS:
        pass

    with pytest.raises(RuntimeError, match="CONST"):
        _check(model=BaseModel(model_sampling=EPS()))


def test_precondition_rejects_a_model_that_overrides_scale_latent_inpaint(comfy_stubs):
    with pytest.raises(RuntimeError, match="overrides scale_latent_inpaint"):
        _check(model=OverridingModel())


def test_live_canvas_rejects_a_schedule_starting_at_sigma_one(comfy_stubs):
    # x / (1 - sigma) is undefined at sigma 1.0 — the message has to name the algebra, not
    # the harness's per-step resume identity, which this port does not have.
    with pytest.raises(RuntimeError, match=r"x / \(1 - sigma\)"):
        _check(sigmas=FULL_SIGMAS, anchor_source=sync.ANCHOR_SOURCE_CANVAS)


def test_source_image_accepts_a_schedule_starting_at_sigma_one(comfy_stubs):
    # denoise 1.0 is a working widget value under sgm_uniform/normal, and comfy's samplers
    # handle sigmas[0] == 1.0 themselves (k_diffusion/sampling.py:168-176).
    assert float(FULL_SIGMAS[0]) == 1.0
    assert _check(sigmas=FULL_SIGMAS) is None


def test_preconditions_run_before_any_encode_or_conditioning(comfy_stubs, vl_clip):
    # Fail fast at ENGINE ENTRY: no VAE work, no VLM work, no GPU time.
    vae, noise = GridVAE(), GridNoise()

    with pytest.raises(RuntimeError, match="not strictly decreasing"):
        prepare(vl_clip, vae=vae, noise=noise, sigmas=torch.tensor([0.8, 0.9, 0.0]))

    assert vae.encode_calls == []
    assert noise.calls == []


def test_unknown_anchor_source_is_rejected(comfy_stubs, vl_clip):
    with pytest.raises(ValueError, match="anchor_source must be one of"):
        prepare(vl_clip, anchor_source="whatever")


def test_a_batched_image_is_rejected(comfy_stubs, vl_clip):
    with pytest.raises(ValueError, match="ONE picture at a time"):
        prepare(vl_clip, image=torch.zeros(2, CANVAS, CANVAS, 3))


# ---- C_0: the assembled canvas latent --------------------------------------


def test_canvas_latent_is_assembled_from_window_encodes_at_canvas_positions(comfy_stubs, vl_clip):
    image = image_canvas()
    vae = GridVAE()

    run = prepare(vl_clip, image=image, vae=vae)

    # One encode per tile — never a whole-canvas VAE call — and the butted cores reproduce the
    # position-preserving stride-8 encode of the entire canvas exactly.
    assert len(vae.encode_calls) == 4
    assert [tuple(call.shape[1:3]) for call in vae.encode_calls] == [(56, 56), (56, 56), (56, 56), (56, 56)]
    assert torch.equal(run.raw_latent, image[:, ::8, ::8, :].movedim(-1, 1))


def test_canvas_coverage_hole_is_rejected():
    # A geometry change that leaves a cell owned by no core would silently ship raw zeros
    # inside every lane's ring.
    def tile(core):
        return SimpleNamespace(core=core, crop_rect=core)

    tiles = [tile(grid.Rect(0, 0, 40, 80)), tile(grid.Rect(48, 0, 80, 80))]

    with pytest.raises(RuntimeError, match="coverage hole"):
        sync.encode_canvas_latent(GridVAE(), torch.zeros(1, 80, 80, 3), tiles)


def test_window_encode_shape_mismatch_is_rejected():
    class SixteenthVAE(GridVAE):
        def encode(self, pixels):
            return pixels[:, ::16, ::16, :].movedim(-1, 1)

    tiles = [SimpleNamespace(core=grid.Rect(0, 0, 80, 80), crop_rect=grid.Rect(0, 0, 80, 80))]

    with pytest.raises(RuntimeError, match="not the /8 the grid math models"):
        sync.encode_canvas_latent(SixteenthVAE(), torch.zeros(1, 80, 80, 3), tiles)


def test_latent_rect_rejects_a_sheared_rect():
    assert sync._latent_rect(grid.Rect(8, 16, 40, 56)) == grid.Rect(1, 2, 5, 7)
    with pytest.raises(RuntimeError, match="not /8"):
        sync._latent_rect(grid.Rect(0, 0, 12, 16))


# ---- lanes: step-0 agreement and the two spaces ----------------------------


def test_overlapping_lanes_hold_the_same_values_on_every_shared_cell(comfy_stubs, vl_clip):
    # The canvas-slice rule: a lane's latent is a slice of the ONE assembled C_0, never its own
    # window encode, so two lanes cannot disagree by a VAE framing difference at step 0.
    run = prepare(vl_clip)
    left, right = run.lanes[0], run.lanes[1]

    assert left.window == (0, 7, 0, 7)
    assert right.window == (0, 7, 3, 10)
    assert torch.equal(left.latent, run.raw_latent[..., 0:7, 0:7])
    assert torch.equal(right.latent, run.raw_latent[..., 0:7, 3:10])
    # Shared columns 3..6 of the canvas: the last 4 of the left window, the first 4 of the right.
    assert torch.equal(left.latent[..., 3:7], right.latent[..., 0:4])


def test_lanes_carry_raw_values_while_the_canvas_is_process_space(comfy_stubs, vl_clip):
    # CANVAS SPACE: comfy applies process_latent_in to latent_image itself, so lanes get RAW
    # slices while the maintained canvas is process-space. A non-identity affine (what Krea 2's
    # Wan21 format really is) is what makes a mixed-space blend detectable at all.
    model = BaseModel(scale=2.0, shift=1.0)
    run = prepare(vl_clip, guider=SyncGuider(model=model))

    assert torch.equal(run.canvas, run.raw_latent * 2.0 + 1.0)
    assert not torch.equal(run.canvas, run.raw_latent)
    assert torch.equal(model.process_latent_out(run.canvas), run.raw_latent)
    for lane in run.lanes:
        y0, y1, x0, x1 = lane.window
        assert torch.equal(lane.latent, run.raw_latent[..., y0:y1, x0:x1])


def test_identity_process_pair_still_gives_the_canvas_its_own_storage(comfy_stubs, vl_clip):
    # A latent_format whose process_in hands the tensor straight back would leave the canvas
    # ALIASING C_0 — and the surgery feature's canvas writes would then rewrite every lane's
    # frozen ring source.
    class PassThroughModel(BaseModel):
        def process_latent_in(self, latent):
            return latent

    run = prepare(vl_clip, guider=SyncGuider(model=PassThroughModel()))

    assert torch.equal(run.canvas, run.raw_latent)
    assert run.canvas.data_ptr() != run.raw_latent.data_ptr()


# ---- noise ----------------------------------------------------------------


def test_canvas_noise_is_one_draw_sliced_per_lane_window(comfy_stubs, vl_clip):
    noise = GridNoise()

    run = prepare(vl_clip, noise=noise)

    assert len(noise.calls) == 1
    dummy = noise.calls[0]["samples"]
    assert dummy.shape == (1, 3, 10, 10)
    assert dummy.dtype == torch.float32
    assert run.canvas_noise.shape == run.raw_latent.shape
    for lane in run.lanes:
        y0, y1, x0, x1 = lane.window
        assert torch.equal(lane.noise, run.canvas_noise[..., y0:y1, x0:x1])


def test_lane_noise_windows_are_clones(comfy_stubs, vl_clip):
    # comfy binds this tensor as model_k.noise, and present_live_ring zeroes each lane's ring
    # cells in place — an aliasing view would zero a NEIGHBOUR's core cells.
    run = prepare(vl_clip)
    canvas_before = run.canvas_noise.clone()
    others_before = [lane.noise.clone() for lane in run.lanes[1:]]

    run.lanes[0].noise.zero_()

    assert torch.equal(run.canvas_noise, canvas_before)
    for lane, before in zip(run.lanes[1:], others_before, strict=True):
        assert torch.equal(lane.noise, before)


def test_a_single_tile_lane_still_gets_a_copy_of_the_canvas_draw(comfy_stubs, vl_clip):
    # The case a .contiguous() window would NOT protect: a 40px canvas solves to a 1x1 grid,
    # so the lane's window IS the whole tensor and contiguous() would hand back the canvas draw
    # itself. Only a clone is unconditional.
    run = prepare(vl_clip, image=image_canvas(40))
    canvas_before = run.canvas_noise.clone()

    assert len(run.lanes) == 1
    assert run.lanes[0].window == (0, 5, 0, 5)
    assert run.lanes[0].noise.data_ptr() != run.canvas_noise.data_ptr()
    run.lanes[0].noise.zero_()
    assert torch.equal(run.canvas_noise, canvas_before)


def test_noise_is_drawn_at_the_full_batch_and_the_picture_row_selected(comfy_stubs, vl_clip):
    # The raster contract (sampling._refine_tiles): prepare_noise draws ONE randn over the whole
    # latent size, so a [1,...] dummy would hand every picture of a batch the same noise.
    noise = GridNoise()

    run = prepare(vl_clip, noise=noise, batch_size=3, batch_index=2)

    dummy = noise.calls[0]["samples"]
    assert dummy.shape == (3, 3, 10, 10)
    whole_draw = torch.arange(3 * 3 * 10 * 10, dtype=torch.float32).reshape(3, 3, 10, 10)
    assert torch.equal(run.canvas_noise, whole_draw[2:3])


def test_noise_shape_mismatch_against_the_canvas_latent_is_rejected(comfy_stubs, vl_clip):
    class WrongChannelVAE(GridVAE):
        latent_channels = 4

    with pytest.raises(RuntimeError, match="canvas noise"):
        prepare(vl_clip, vae=WrongChannelVAE())


# ---- per-lane guiders ------------------------------------------------------


def test_each_lane_guider_carries_its_own_tile_positive(comfy_stubs, vl_clip):
    guider = SyncGuider()
    pristine = guider.original_conds
    negative = pristine["negative"]

    run = prepare(vl_clip, guider=guider)

    layout = expected_layout()
    assert len(run.lanes) == len(layout.tiles) == 4
    rows = []
    for tile, lane in zip(layout.tiles, run.lanes, strict=True):
        expected = vl.slice_indices(tile.crop_rect, CANVAS, CANVAS, ENC, ENC, ENC_SEQ)
        seen = lane.guider.original_conds
        assert seen["positive"][0]["cross_attn"][0, :, 0].tolist() == expected
        assert seen["negative"] is negative          # only the positive is swapped
        rows.append(tuple(expected))
    assert len(set(rows)) == len(rows)               # a distinct slice per tile, in tile order
    # The caller's guider is never mutated: no swap, no restore, nothing to leak on a raise.
    assert guider.original_conds is pristine
    assert all(lane.guider is not guider for lane in run.lanes)
    assert len({id(lane.guider) for lane in run.lanes}) == len(run.lanes)


def test_lane_guiders_share_the_model_patcher(comfy_stubs, vl_clip):
    # Per-lane guiders isolate per-run state, NOT the model: one model, one GPU stream.
    guider = SyncGuider()

    run = prepare(vl_clip, guider=guider)

    assert all(lane.guider.model_patcher is guider.model_patcher for lane in run.lanes)


def test_control_on_the_negative_never_reaches_a_lane(comfy_stubs, vl_clip):
    # The strip_control regression (conds.py:129-137): each tile's positive is a fresh vision
    # slice carrying no control chain, so a control left on the negative would steer the uncond
    # branch alone against an uncropped hint.
    control = object()
    guider = SyncGuider(negative=[{"cross_attn": torch.ones(1, 1, 8), "control": control}])

    run = prepare(vl_clip, guider=guider)

    assert guider.original_conds["negative"][0]["control"] is control
    for lane in run.lanes:
        assert "control" not in lane.guider.original_conds["negative"][0]
        assert not conds.has_spatial_conds(lane.guider.original_conds)


def test_a_guider_without_positive_conds_is_rejected(comfy_stubs, vl_clip):
    guider = SyncGuider()
    guider.original_conds = [("positive", {})]

    with pytest.raises(ValueError, match="CFGGuider convention"):
        prepare(vl_clip, guider=guider)


def test_a_guider_copy_that_loses_original_conds_is_rejected(comfy_stubs, vl_clip):
    assert not hasattr(copy.copy(LostCondsGuider()), "original_conds")

    with pytest.raises(ValueError, match="did not carry a dict 'original_conds'"):
        prepare(vl_clip, guider=LostCondsGuider())


# ---- one layout for both the conditioning and the tiling --------------------


def test_the_conditioning_pre_pass_slices_the_engines_own_tiles(comfy_stubs, vl_clip, monkeypatch):
    # One grid solve: the layout that slices a tile's positive IS the layout the engine tiles
    # with, by object identity — there is no second solve to drift from.
    seen = {}
    real_build = vl.build_global_slices

    def recording_build(clip, source, tiles, offset_x=0, offset_y=0):
        seen["tiles"] = tiles
        seen["source"] = source
        return real_build(clip, source, tiles, offset_x, offset_y)

    monkeypatch.setattr(vl, "build_global_slices", recording_build)
    run = prepare(vl_clip)

    assert seen["tiles"] is run.layout.tiles
    assert seen["source"] is run.padded


def test_caption_surfaces_route_through_the_same_tiles(comfy_stubs, vl_clip, monkeypatch):
    # The other two vlm_methods build their positives from the same layout.tiles object.
    seen = {}

    def recording_captions(clip, source, tiles, instruction, max_length, batch_size=1, batch_index=0):
        seen["tiles"] = tiles
        seen["instruction"] = instruction
        return [[f"tile {index}"] for index in range(len(tiles))]

    monkeypatch.setattr(captions, "generate_tile_captions", recording_captions)
    monkeypatch.setattr(captions, "build_caption_conds",
                        lambda clip, tile_captions: [[{"cross_attn": torch.zeros(1, 1, 8)}]] * len(tile_captions))
    run = prepare(vl_clip, vlm_method=captions.VLM_METHOD_CAPTIONS)

    assert seen["tiles"] is run.layout.tiles
    assert seen["instruction"] == captions.CAPTION_INSTRUCTIONS[captions.VLM_METHOD_CAPTIONS][0]


def test_unknown_vlm_method_is_rejected(comfy_stubs, vl_clip):
    with pytest.raises(ValueError, match="vlm_method must be one of"):
        prepare(vl_clip, vlm_method="interpretive dance")


def test_missing_vl_clip_is_rejected(comfy_stubs):
    with pytest.raises(ValueError, match="needs the VL CLIP"):
        prepare(None)


# ---- the run's own geometry -------------------------------------------------


def test_prepare_run_keeps_the_padded_canvas_and_the_crop_back_size(comfy_stubs, vl_clip):
    # A non-/8 input is reflect-padded for tiling and the ORIGINAL size is carried so the run
    # can be cropped back to it.
    image = torch.rand(1, 76, 78, 3)

    run = prepare(vl_clip, image=image)

    assert run.padded.shape == (1, 80, 80, 3)
    assert run.size == (76, 78)
    assert run.layout.w == 80 and run.layout.h == 80


# ---- the run loop ----------------------------------------------------------
#
# RUN GEOMETRY, worked out once. Caps 72 (width) x 80 (height) over the same 80x80 canvas
# with context_anchor 8 and context_overlap 16 (r = 24) solve to a 2x1 grid, base 40:
#
#   cells        0   1   2   3   4 | 5   6   7   8   9
#   tile 0 core  <----------------->                        cols [0:5]
#   tile 0 window<------------------------------->          cols [0:8]  (ring: col 7)
#   tile 1 core                      <----------------->    cols [5:10]
#   tile 1 window        <------------------------------>   cols [2:10] (ring: col 2)
#   tile 1 paste                 <------------------------> cols [3:10], band = cols 3,4
#
# Both windows span every row. Tile 0 pastes its bare core (no already-blended side), tile 1
# pastes its core plus the 2-cell band, so exactly ONE cell (col 3) is ever a fractional
# blend and col 4 — the seam-most band cell — is pinned to tile 1 at exactly 1.0.
RUN_CAP_W = 72
RUN_CAP_H = 80
RUN_OVERLAP = 16
RUN_STEPS = int(SIGMAS.shape[-1]) - 1
# The base of the per-lane constant the fake model writes: tile 0 -> 10.0, tile 1 -> 20.0.
LANE_VALUE = 10.0


def lane_value(lane_index, eval_index):
    # What lane `lane_index` predicts at its `eval_index`th eval. Distinct per LANE so a
    # misplaced window changes the value, and distinct per EVAL so a step's results can never
    # be mistaken for another's — which is what pins the preview to the PREVIOUS step and the
    # final consolidation to the LAST one.
    return LANE_VALUE * (lane_index + 1) + eval_index


class LaneModelK:
    """comfy's KSamplerX0Inpaint (samplers.py:634-643) cut to its blend contract.

    The frozen cells of the model INPUT are re-presented through the model's own
    `scale_latent_inpaint` — the method the once-around ring patch replaces — and the frozen
    cells of the OUTPUT are restored to the clean `latent_image`, which is why a lane's own
    window carries an UNREFINED halo for the whole run and decoding it would show. The
    refined region is filled with `lane_value(lane, eval)`, so every canvas cell below is
    hand-computable and carries the identity of the eval it came from.

    `noise` and `latent_image` are read FRESH at every eval and logged as of that eval, which
    is what makes the live-canvas surgery (which zeroes one and rebinds the other between
    steps) observable at all.
    """

    def __init__(self, model, latent, noise, denoise_mask, lane_index, log):
        self.model = model
        self.latent_image = latent
        self.noise = noise
        self.denoise_mask = denoise_mask
        self.lane_index = lane_index
        self.log = log
        self.evals = 0

    def __call__(self, x, sigma, **kwargs):
        mask = self.denoise_mask
        value = lane_value(self.lane_index, self.evals)
        self.evals += 1
        presented = self.model.scale_latent_inpaint(
            x=x, sigma=sigma, noise=self.noise, latent_image=self.latent_image)
        self.log.append({"lane": self.lane_index, "value": value, "x": x.clone(),
                         "noise": self.noise.clone(), "latent_image": self.latent_image.clone(),
                         "input": x * mask + presented * (1.0 - mask)})
        return torch.full_like(x, value) * mask + self.latent_image * (1.0 - mask)


class RunGuider(SyncGuider):
    """comfy's CFGGuider cut to sample()'s CONTRACT, so a synthetic run exercises the real
    spaces: process_latent_in on the latent (samplers.py:1224), CONST noise_scaling for the
    initial x (:992), KSamplerX0Inpaint's per-eval blend (:634-643), and process_latent_out
    on the way out (:1238). Every lane is a shallow copy of this object (build_lane_guiders),
    so `log` is shared across the fleet deliberately — it is the whole run's eval record."""

    def __init__(self, model=None, negative=None):
        super().__init__(model=model, negative=negative)
        self.log = []
        self.lane_index = 0

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None,
               disable_pbar=False, seed=None):
        model = self.model_patcher.model
        latent = model.process_latent_in(latent_image)
        x = model.model_sampling.noise_scaling(sigmas[0], noise, latent)
        model_k = LaneModelK(model, latent, noise, denoise_mask, self.lane_index, self.log)
        out = sampler.sampler_function(
            model_k, x, sigmas, extra_args={"denoise_mask": denoise_mask, "seed": seed},
            callback=callback, disable=disable_pbar, **sampler.extra_options)
        return model.process_latent_out(out)


class RecordingVAE(GridVAE):
    """GridVAE that keeps every latent it was handed to decode."""

    def __init__(self):
        super().__init__()
        self.decoded = []

    def decode(self, samples):
        self.decoded.append(samples.clone())
        return super().decode(samples)


class RecordingPreviewer:
    """latent_preview's previewer cut to the one call the progress hook makes."""

    def __init__(self):
        self.previews = []

    def decode_latent_to_preview_image(self, image_format, x0):
        self.previews.append((image_format, x0.clone()))
        return f"preview-{len(self.previews)}"


def euler_sampler(name="sample_euler"):
    # A stand-in k-diffusion function with euler's eval CADENCE and nothing else: one model
    # call per sigma step, and a step update that walks the trajectory all the way onto the
    # prediction. Named so the stepper's `sample_` strip resolves it in EVALS_PER_STEP.
    # The .clone() is the structural half of every real sampler's update (`x = x + d * dt`
    # allocates): `x` must be the sampler's OWN tensor, never the model-output object the
    # stepper stashes as `denoised`, or the hook's in-place scatter into x would rewrite the
    # prediction a preview reads.
    import comfy.samplers

    def fn(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        s_in = x.new_ones([x.shape[0]])
        for step in range(int(sigmas.shape[-1]) - 1):
            x = model(x, sigmas[step] * s_in, **(extra_args or {})).clone()
        return x

    fn.__name__ = name
    return comfy.samplers.KSAMPLER(fn, {}, {})


@pytest.fixture
def lane_stamp(monkeypatch):
    # The engine builds its lane guiders itself (shallow copies of the caller's), so the
    # harness stamps each copy with its tile index on the way out — that is what gives every
    # lane a distinct constant to write. The real builder still runs.
    real_build = sync.build_lane_guiders

    def stamped(guider, tile_positives):
        built = real_build(guider, tile_positives)
        for index, lane_guider in enumerate(built):
            lane_guider.lane_index = index
        return built

    monkeypatch.setattr(sync, "build_lane_guiders", stamped)


def run_sync(clip, guider=None, image=None, vae=None, noise=None, sigmas=SIGMAS, sampler=None,
             anchor_source=sync.ANCHOR_SOURCE_IMAGE, batch_size=1, batch_index=0,
             overlap=RUN_OVERLAP, mask=None):
    # run_bounded: the engine runs the REAL stepper, so a deadlock must fail the test instead
    # of wedging the suite.
    return run_bounded(lambda: sync.refine_sync(
        image_canvas() if image is None else image, RunGuider() if guider is None else guider,
        euler_sampler() if sampler is None else sampler, sigmas,
        GridVAE() if vae is None else vae, GridNoise() if noise is None else noise,
        RUN_CAP_W, RUN_CAP_H, CTX, overlap, clip, mask=mask, anchor_source=anchor_source,
        batch_size=batch_size, batch_index=batch_index))


def run_layout(overlap=RUN_OVERLAP):
    return _layout(CANVAS, CANVAS, RUN_CAP_W, RUN_CAP_H, ctx=CTX, overlap=overlap)


def band_alpha():
    # The shipped curve spelled out at the run geometry's 2-cell band, independent of
    # sampling.py: u = (i + 0.5)/band, base = clamp(u / (1 - plateau)), alpha = base**2, at
    # the LATENT plateau max(0.10, 0.5/2) = 0.25. -> [1/9, 1.0], so the seam-most cell is
    # exactly tile 1's own value.
    u = torch.tensor([0.25, 0.75], dtype=torch.float32)
    return u.div(0.75).clamp(0.0, 1.0).pow(2.0)


def expected_canvas(eval_index=RUN_STEPS - 1):
    # The consolidated PROCESS-space canvas holding the results of eval `eval_index` (the
    # LAST one by default, which is what the post-stepper consolidation must land): tile 0
    # writes its value over its core (cells 0..4), then tile 1 writes its own over cells 3..9
    # through the 2-cell feather. Every cell is one of three hand-computable values, and no
    # frozen-ring value is anywhere in it.
    alpha = band_alpha()
    left, right = lane_value(0, eval_index), lane_value(1, eval_index)
    canvas = torch.full((1, 3, CANVAS // 8, CANVAS // 8), right)
    canvas[..., 0:3] = left
    canvas[..., 3] = alpha[0] * right + (1.0 - alpha[0]) * left
    return canvas


# ---- the latent feather ----------------------------------------------------


def test_latent_plateau_pins_the_seam_most_cell_at_exactly_one():
    # The whole reason the latent band does not reuse the pixel plateau: at 10% of a 4-cell
    # band the seam-most cell would sit at 0.945 and leak the neighbour across the seam line
    # at EVERY step. Exact float equality, not a tolerance — the clamp lands on 1.0 or it
    # does not.
    for band_cells in (1, 2, 4, 16):
        ramp = sampling._feather_ramp(band_cells, sync.latent_plateau(band_cells), sampling.FEATHER_FALLOFF)
        assert float(ramp[-1]) == 1.0
        assert float(ramp[0]) < 1.0 or band_cells == 1
    # context_overlap 0 is a legal widget value and leaves no band at all: the plateau falls
    # back to the shipped constant instead of dividing by zero.
    assert sync.latent_plateau(0) == sampling.FEATHER_PLATEAU
    assert sync.latent_plateau(4) == 0.125
    assert sync.latent_plateau(16) == sampling.FEATHER_PLATEAU


def test_the_pixel_plateau_would_not_pin_a_narrow_latent_band():
    # The negative control for the test above: the SHIPPED pixel plateau is what fails here,
    # so the override is load-bearing rather than decorative.
    ramp = sampling._feather_ramp(4, sampling.FEATHER_PLATEAU, sampling.FEATHER_FALLOFF)

    assert float(ramp[-1]) < 1.0


def test_the_write_back_order_never_blends_against_a_stale_value():
    # Boolean-canvas simulation of ONE consolidation pass over a 2x2 grid: walk the blends in
    # raster order and mark every cell each tile writes. Every cell a tile cross-dissolves
    # against (alpha < 1) must ALREADY be marked, or that tile would be mixing this step's
    # value into the PREVIOUS step's; and by the end every cell must be marked, or a cell
    # would still hold what it held before the pass.
    layout = _layout(CANVAS, CANVAS, RUN_CAP_W, RUN_CAP_W, ctx=CTX, overlap=RUN_OVERLAP)
    cells = CANVAS // 8
    blends = sync.build_blends(layout, torch.zeros(1, 3, cells, cells))
    written = torch.zeros(cells, cells, dtype=torch.bool)

    assert len(blends) == 4
    feathered_cells = 0
    for paste, alpha in blends:
        region = written[paste.y0:paste.y1, paste.x0:paste.x1]
        feathered = alpha[0, 0] < 1.0
        feathered_cells += int(feathered.sum())
        assert bool(region[feathered].all())
        region[:] = True

    assert feathered_cells > 0          # the simulation is not vacuous
    assert bool(written.all())          # full coverage: no cell survives a pass unwritten


# ---- the run: canvas, spaces, decode ---------------------------------------


def test_refine_sync_runs_end_to_end_through_the_real_stepper(comfy_stubs, vl_clip, lane_stamp):
    guider = RunGuider()
    vae = RecordingVAE()

    out = run_sync(vl_clip, guider=guider, vae=vae)

    assert out.shape == (1, CANVAS, CANVAS, 3)
    assert torch.isfinite(out).all()
    # Two lanes, each stepping the whole shared schedule, then one decode per tile.
    assert len(guider.log) == 2 * RUN_STEPS
    assert [entry["lane"] for entry in guider.log].count(0) == RUN_STEPS
    assert vae.decode_calls == 2


def test_the_output_keeps_the_full_input_size(comfy_stubs, vl_clip, lane_stamp):
    # The crop-back: a non-/8 input is reflect-padded for tiling and returned at its own size.
    out = run_sync(vl_clip, image=torch.rand(1, 76, 78, 3))

    assert out.shape == (1, 76, 78, 3)


def test_a_single_tile_run_is_one_lane_with_an_all_ones_feather(comfy_stubs, vl_clip, lane_stamp):
    # The degenerate geometry every widget combination can reach: a canvas that solves to a
    # 1x1 grid has no kept side at all, so the consolidation is a hard whole-canvas write and
    # the decode is one window. It must not need a band to work.
    vae = RecordingVAE()

    out = run_sync(vl_clip, image=image_canvas(40), vae=vae, guider=RunGuider())

    assert out.shape == (1, 40, 40, 3)
    assert len(vae.decoded) == 1
    assert torch.equal(vae.decoded[0], torch.full((1, 3, 5, 5), lane_value(0, RUN_STEPS - 1)))


def test_the_canvas_holds_one_value_where_the_two_lanes_meet(comfy_stubs, vl_clip, lane_stamp):
    # What the whole engine is for: after the run every cell shared by the two windows holds
    # ONE value, so both sides of the band decode one latent and there is nothing for a DC
    # match or a min-error cut to correct.
    vae = RecordingVAE()

    run_sync(vl_clip, vae=vae, guider=RunGuider())

    left, right = vae.decoded
    assert left.shape == (1, 3, 10, 8) and right.shape == (1, 3, 10, 8)
    # Canvas cells 2..7 are in BOTH windows: the left window's cells 2..7, the right's 0..5.
    assert torch.equal(left[..., 2:8], right[..., 0:6])


def test_the_decode_reads_the_canvas_never_a_lanes_own_window(comfy_stubs, vl_clip, lane_stamp):
    # Hand-computed, cell by cell. The discriminating part is cells 5..7 of tile 0's window:
    # its own lane holds its own prediction there (its unpasted right overlap) and the
    # UNREFINED latent at cell 7 (its frozen ring), while the canvas holds tile 1's core,
    # because that is who owns those cells.
    vae = RecordingVAE()

    run_sync(vl_clip, vae=vae, guider=RunGuider())

    left, right = lane_value(0, RUN_STEPS - 1), lane_value(1, RUN_STEPS - 1)
    expected = expected_canvas()
    assert torch.equal(vae.decoded[0], expected[..., 0:8])
    assert torch.equal(vae.decoded[1], expected[..., 2:10])
    assert torch.equal(vae.decoded[0][..., 5:8], torch.full((1, 3, 10, 3), right))
    # The LAST eval's values, not the previous step's: the stepper's hook is PRE-step, so the
    # final step's results are consolidated only after it returns.
    assert torch.equal(vae.decoded[0][..., 0:3], torch.full((1, 3, 10, 3), left))
    assert not torch.equal(vae.decoded[0], expected_canvas(RUN_STEPS - 2)[..., 0:8])
    # ... and the one fractional cell really is the feather, not a hard handover.
    assert left < float(vae.decoded[0][0, 0, 0, 3]) < right


def test_every_lane_starts_a_step_on_the_consolidated_canvas(comfy_stubs, vl_clip, lane_stamp):
    # The scatter-back half of the hook: a lane must enter each step holding the CONSOLIDATED
    # view of its own window, not its private one. Lane 0's step-1 trajectory therefore
    # carries tile 1's core over cells 4..7 and the feathered value at cell 3, where its own
    # step-0 output held its own prediction (cells 3..6) and the unrefined latent (cell 7).
    guider = RunGuider()

    run_sync(vl_clip, guider=guider)

    step_one = guider.log[2]                                   # lane 0's second eval
    assert step_one["lane"] == 0
    assert torch.equal(step_one["x"], expected_canvas(0)[..., 0:8])
    assert not torch.equal(step_one["x"][..., 4:8],
                           torch.full((1, 3, 10, 4), lane_value(0, 0)))


def test_the_decode_window_is_the_canvas_converted_back_to_raw_space(comfy_stubs, vl_clip, lane_stamp):
    # CANVAS SPACE round trip under a NON-identity process pair (what Krea 2's Wan21 format
    # really is): the lanes and the canvas live in process space, and every window handed to
    # the VAE is process_latent_out of it. A blend that mixed the two spaces — e.g. writing a
    # lane's RAW return straight into the canvas — lands on different numbers here.
    vae = RecordingVAE()
    model = BaseModel(scale=2.0, shift=1.0)

    run_sync(vl_clip, vae=vae, guider=RunGuider(model=model))

    raw = (expected_canvas() - 1.0) / 2.0        # process_latent_out spelled out
    assert torch.equal(vae.decoded[0], raw[..., 0:8])
    assert torch.equal(vae.decoded[1], raw[..., 2:10])
    assert not torch.equal(vae.decoded[0], expected_canvas()[..., 0:8])


# ---- progress, preview, interrupt ------------------------------------------


def test_the_preview_is_the_pasted_denoised_canvas_of_the_previous_step(comfy_stubs, vl_clip, lane_stamp):
    # The refine-round-4 gap: a preview built from the trajectory `x` shows noise. This one is
    # built from each lane's `denoised` — the model's x0 prediction — pasted whole in raster
    # order, so tile 1's window (cells 2..9) overwrites tile 0 there, INCLUDING tile 1's own
    # unrefined ring cell 2. That is what makes it distinguishable from the trajectory canvas.
    comfy_stubs["previewer"] = RecordingPreviewer()
    vae = RecordingVAE()

    run_sync(vl_clip, vae=vae, guider=RunGuider())

    previewer = comfy_stubs["previewer"]
    assert [image_format for image_format, _ in previewer.previews] == ["JPEG"] * (RUN_STEPS - 1)
    # Cells 0..1 are tile 0's prediction, cell 2 is tile 1's frozen ring (the raw canvas
    # latent, in process space), cells 3..9 are tile 1's prediction.
    latent = image_canvas()[:, ::8, ::8, :].movedim(-1, 1)
    for step_index, (_image_format, preview) in enumerate(previewer.previews, start=1):
        expected = torch.full((1, 3, 10, 10), lane_value(1, step_index - 1))
        expected[..., 0:2] = lane_value(0, step_index - 1)
        expected[..., 2] = latent[..., 2]
        assert torch.equal(preview, expected)                  # step i-1's prediction
        assert not torch.equal(preview, expected_canvas(step_index - 1))   # not the trajectory


def test_no_preview_at_step_zero_and_one_bar_over_the_whole_run(comfy_stubs, vl_clip, lane_stamp):
    # Step 0 has no x0 prediction at all (every lane's `denoised` is None until its first eval
    # returns), so the bar still advances but carries no image.
    comfy_stubs["previewer"] = RecordingPreviewer()

    run_sync(vl_clip, guider=RunGuider())

    pbar = comfy_stubs["progress_bars"][-1]
    assert pbar.total == RUN_STEPS
    assert [value for value, _total, _preview in pbar.updates] == [1, 2, 3, 4]
    assert pbar.updates[0][2] is None
    assert [update[2] for update in pbar.updates[1:]] == ["preview-1", "preview-2", "preview-3"]


def test_the_bar_spans_the_whole_picture_batch(comfy_stubs, vl_clip, lane_stamp):
    # refine_image's picture loop: every picture reports on the SAME total and starts where
    # the previous one ended, so a B>1 run reads as ONE bar instead of restarting per picture.
    run_sync(vl_clip, guider=RunGuider(), batch_size=3, batch_index=2)

    pbar = comfy_stubs["progress_bars"][-1]
    assert pbar.total == RUN_STEPS * 3
    assert [value for value, _total, _preview in pbar.updates] == [9, 10, 11, 12]


def test_the_hook_checks_for_an_interrupt_before_releasing_the_fleet(comfy_stubs, vl_clip,
                                                                    lane_stamp, monkeypatch):
    # A user cancel must stop every lane, not let N full-length sampler calls run to the end.
    # The step-0 hook fires before any lane has evaluated, so a raise there leaves an empty
    # eval log — and the exception reaches the caller unchanged.
    import comfy.model_management

    class Cancelled(Exception):
        pass

    def cancel():
        raise Cancelled("cancelled")

    monkeypatch.setattr(comfy.model_management, "throw_exception_if_processing_interrupted", cancel)
    guider = RunGuider()

    with pytest.raises(Cancelled):
        run_sync(vl_clip, guider=guider)

    assert guider.log == []


class RaisingRunGuider(RunGuider):
    """RunGuider whose SECOND lane blows up once the fleet is under way."""

    def sample(self, *args, **kwargs):
        if self.lane_index == 1:
            raise RuntimeError("sampler exploded")
        return super().sample(*args, **kwargs)


def test_the_callers_guider_is_untouched_after_a_lane_raises(comfy_stubs, vl_clip, lane_stamp):
    # The raster path swapped the caller's own original_conds per tile and restored it in a
    # `finally`; the engine never swaps it at all — each LANE gets its own guider — so there
    # is nothing to leak when a lane fails mid-run, and the failure reaches the caller
    # unchanged rather than as a barrier timeout.
    guider = RaisingRunGuider()
    pristine = guider.original_conds
    positive = pristine["positive"]

    with pytest.raises(RuntimeError, match="sampler exploded"):
        run_sync(vl_clip, guider=guider)

    assert guider.original_conds is pristine
    assert guider.original_conds["positive"] is positive


# ---- the once-around ring --------------------------------------------------


def test_the_ring_schedule_is_entered_once_for_the_whole_run(comfy_stubs, vl_clip, lane_stamp, monkeypatch):
    # ONE instance patch on the shared model, held around the entire stepper run: entering it
    # per lane would patch the same model N times, and entering it per CALL would normalize
    # against the wrong sigma_first. The swap point is sampling.anchor_ring_schedule.
    entered = []
    real_schedule = sampling.anchor_ring_schedule

    def counting(guider, sigmas, enabled=True):
        entered.append({"enabled": enabled, "sigmas": sigmas})
        return real_schedule(guider, sigmas, enabled=enabled)

    monkeypatch.setattr(sampling, "anchor_ring_schedule", counting)
    guider = RunGuider()

    run_sync(vl_clip, guider=guider)

    assert len(entered) == 1
    assert entered[0]["enabled"] is True                      # anchor_source "source image"
    assert torch.equal(entered[0]["sigmas"], SIGMAS)          # the FULL schedule, not a step
    # Restored on the way out: core caches model objects session-wide, so a patch left behind
    # would follow the model into every other node in the graph.
    assert "scale_latent_inpaint" not in vars(guider.model_patcher.model)


def test_the_ring_patch_is_live_while_the_lanes_evaluate(comfy_stubs, vl_clip, lane_stamp):
    # The counting test above proves WHERE the context is entered; this proves the model
    # really sees the LEAD ring through it. Lane 0's window cell 7 is its frozen ring, so the
    # model input there is whatever scale_latent_inpaint presented — under the patch that is
    # sigma * anchor_ring_factor(r), and under core's own default it would be plain sigma.
    guider = RunGuider()

    run_sync(vl_clip, guider=guider)

    model = guider.model_patcher.model
    noise_window = torch.arange(3 * 10 * 10, dtype=torch.float32).reshape(1, 3, 10, 10)[..., 0:8]
    latent_window = image_canvas()[:, ::8, ::8, :].movedim(-1, 1)[..., 0:8]
    factor = sampling.anchor_ring_factor(float(SIGMAS[1]) / float(SIGMAS[0]))
    lead = model.model_sampling.noise_scaling(
        (SIGMAS[1:2] * factor).reshape(1, 1, 1, 1), noise_window, latent_window)
    plain = model.model_sampling.noise_scaling(
        SIGMAS[1:2].reshape(1, 1, 1, 1), noise_window, latent_window)

    step_one = guider.log[2]                                   # lane 0's second eval
    assert step_one["lane"] == 0
    assert 0.0 < factor < 1.0                                  # the ring leads the core here
    assert torch.equal(step_one["input"][..., 7], lead[..., 7])
    assert not torch.equal(step_one["input"][..., 7], plain[..., 7])


# ---- the region (mask) path -------------------------------------------------
#
# REGION RUN GEOMETRY, worked out once. band_mask() covers pixel rows 24..55 across the whole
# width, so the bbox is rows 24..55 x cols 0..79. _expand_snap_clamp at context_anchor 8 snaps
# it out to rows 16..63 (cols clamp back to 0..79), i.e. a 48x80 crop. Width solves to the RUN
# geometry's own two columns (base 40, r 24) and height to a single row of tiles, so the run is
# again TWO lanes, over six latent rows. Inside the crop the mask occupies pixel rows 8..39,
# whose /8 cover is latent rows 1..4 — rows 0 and 5 are the frozen background no lane diffuses.
REGION_CROP = (16, 64, 0, 80)


def region_bbox(mask, ctx=CTX):
    return sampling._expand_snap_clamp(sampling._mask_bbox(mask >= 0.5), ctx, CANVAS, CANVAS)


def edge_alpha(mask, crop=REGION_CROP):
    # The shipped 1px anti-alias, laid back out over the whole canvas: 0 everywhere outside the
    # crop, and the composite weight inside it.
    y0, y1, x0, x1 = crop
    alpha = torch.zeros(1, CANVAS, CANVAS)
    alpha[:, y0:y1, x0:x1] = sampling._aa_alpha((mask >= 0.5)[:, y0:y1, x0:x1].float())
    return alpha


def test_the_region_crop_is_the_bbox_expanded_by_the_context_anchor(comfy_stubs, vl_clip):
    # The raster path's own crop math, reused rather than re-derived.
    assert region_bbox(band_mask()) == REGION_CROP


def test_the_region_run_leaves_every_pixel_the_edge_weights_at_zero_byte_identical(
        comfy_stubs, vl_clip, lane_stamp):
    # The mask contract: outside the crop, and wherever the anti-aliased edge is 0 inside it,
    # the output is the ORIGINAL pixels — never a VAE round trip of them. torch.equal, not a
    # tolerance: a decode-and-recomposite would land within a tolerance and still be wrong.
    image = image_canvas()
    mask = band_mask()

    out = run_sync(vl_clip, image=image, mask=mask, guider=RunGuider())

    rgb = image[..., :3]
    untouched = edge_alpha(mask) == 0.0
    assert out.shape == rgb.shape
    assert torch.equal(out[untouched], rgb[untouched])
    # ... and the region really did diffuse: the fake model writes a lane constant, so the
    # masked rows cannot survive as the input's arange values.
    assert not torch.equal(out[~untouched], rgb[~untouched])
    # Hand-computed: the mask spans pixel rows 24..55 and the 1px anti-alias reaches exactly
    # one row past each edge, so rows 0..22 and 57..79 are untouched and 23..56 are not.
    assert bool(untouched[:, :23].all()) and bool(untouched[:, 57:].all())
    assert not bool(untouched[:, 23:57].any())


def test_the_region_composite_is_the_shipped_anti_aliased_edge(comfy_stubs, vl_clip,
                                                               lane_stamp, monkeypatch):
    # NO feather at the mask boundary (CLAUDE.md prime directive 3): the refined crop goes back
    # through sampling._aa_alpha and nothing else, so the whole output is reproducible from the
    # crop the engine returned plus that one alpha.
    image = image_canvas()
    mask = band_mask()
    seen = {}
    real_refine = sync._refine_canvas

    def recording(*args, **kwargs):
        refined = real_refine(*args, **kwargs)
        seen["refined"] = refined
        seen["region_pixel"] = kwargs["region_pixel"]
        seen["vl_context"] = kwargs["vl_context"]
        return refined

    monkeypatch.setattr(sync, "_refine_canvas", recording)
    out = run_sync(vl_clip, image=image, mask=mask, guider=RunGuider())

    y0, y1, x0, x1 = REGION_CROP
    rgb = image[..., :3]
    alpha = sampling._aa_alpha(seen["region_pixel"])[..., None]
    expected = rgb.clone()
    expected[:, y0:y1, x0:x1, :] = alpha * seen["refined"] + (1.0 - alpha) * rgb[:, y0:y1, x0:x1, :]

    assert torch.equal(out, expected)
    assert seen["refined"].shape == (1, y1 - y0, x1 - x0, 3)
    # The gate handed to the engine is the mask's own crop, hardened to binary.
    assert torch.equal(seen["region_pixel"], (mask >= 0.5)[:, y0:y1, x0:x1].float())


def test_the_region_lanes_diffuse_only_the_masked_rows(comfy_stubs, vl_clip, lane_stamp):
    # The gate reaching the SAMPLER, not just _prepare_run: inside the crop the mask covers
    # pixel rows 8..39 == latent rows 1..4, so the canvas the decode reads carries refined lane
    # values on those rows and the UNTOUCHED raw latent on rows 0 and 5 (KSamplerX0Inpaint
    # restores every frozen cell to its latent_image, which the region path keeps raw).
    vae = RecordingVAE()

    run_sync(vl_clip, image=image_canvas(), mask=band_mask(), vae=vae, guider=RunGuider())

    y0, y1, _x0, _x1 = REGION_CROP
    latent = image_canvas()[:, y0:y1, :, :][:, ::8, ::8, :].movedim(-1, 1)
    assert len(vae.decoded) == 2
    left = vae.decoded[0]
    window = latent[..., 0:8]
    assert left.shape == (1, 3, 6, 8)
    # Columns 0..2 are tile 0's own core and 5..7 tile 1's — neither lies in the 2-cell band,
    # so each cell is exactly one lane's value with no feather arithmetic in between.
    for cols in (slice(0, 3), slice(5, 8)):
        assert torch.equal(left[:, :, 0, cols], window[:, :, 0, cols])
        assert torch.equal(left[:, :, 5, cols], window[:, :, 5, cols])
        assert not torch.equal(left[:, :, 1:5, cols], window[:, :, 1:5, cols])


def test_an_empty_mask_refines_nothing_and_never_reaches_the_engine(comfy_stubs, vl_clip):
    image = image_canvas()
    vae, noise = GridVAE(), GridNoise()

    out = run_sync(vl_clip, image=image, mask=torch.zeros(1, CANVAS, CANVAS), vae=vae, noise=noise)

    assert torch.equal(out, image[..., :3])
    assert out.data_ptr() != image.data_ptr()      # a clone, so the caller's input is safe
    assert vae.encode_calls == [] and noise.calls == []


def test_the_mask_path_encodes_the_full_image_at_the_bbox_offset(comfy_stubs, vl_clip,
                                                                 lane_stamp, monkeypatch):
    # The region's tiles index the bbox crop, but the encode must read the WHOLE image with the
    # bbox origin as the offset, or a masked refine only ever sees its own crop.
    image = image_canvas()
    mask = band_mask()
    seen = {}
    real_build = vl.build_global_slices

    def recording_build(clip, source, tiles, offset_x=0, offset_y=0):
        seen["source"] = source
        seen["offsets"] = (offset_x, offset_y)
        seen["tiles"] = tiles
        return real_build(clip, source, tiles, offset_x, offset_y)

    monkeypatch.setattr(vl, "build_global_slices", recording_build)
    run_sync(vl_clip, image=image, mask=mask, guider=RunGuider())

    y0, _y1, x0, _x1 = REGION_CROP
    assert seen["source"] is image                 # the FULL image, not the bbox crop
    assert seen["offsets"] == (x0, y0)
    # One grid solve on the region path too: the rects sliced are the crop's own tiles.
    assert [tile.crop_rect.y1 for tile in seen["tiles"]] == [48, 48]


# ---- the "live canvas" ring -------------------------------------------------
#
# Each lane's frozen ring, as a column index into its OWN window: tile 0's window is canvas
# cells 0..7 and its ring is the last of them (canvas col 7, inside tile 1's core); tile 1's
# window is cells 2..9 and its ring is the first (canvas col 2, inside tile 0's core).
LIVE_RING_COL = (7, 0)
LIVE_WINDOWS = ((0, 8), (2, 10))


class RingModelK:
    """comfy's KSamplerX0Inpaint cut to the two tensors the live-canvas surgery rewrites."""

    def __init__(self, noise, latent_image):
        self.noise = noise
        self.latent_image = latent_image


class RingLane:
    """stepper.Lane cut to what present_live_ring reads."""

    def __init__(self, model_k, x):
        self.model_k = model_k
        self.x = x


def test_zeroing_one_lanes_ring_noise_cannot_touch_a_neighbours_window():
    # The clone rule made load-bearing. Both lanes hold CLONES of overlapping slices of the one
    # canvas draw, so lane 0's in-place ring zeroing must leave lane 1's copy of the SAME canvas
    # cells intact — through a view it would zero the neighbour's core noise instead.
    canvas_noise = torch.arange(2 * 4 * 6, dtype=torch.float32).reshape(1, 2, 4, 6)
    left = RingLane(RingModelK(canvas_noise[..., 0:4].clone(), torch.zeros(1, 2, 4, 4)),
                    torch.full((1, 2, 4, 4), 3.0))
    right = RingLane(RingModelK(canvas_noise[..., 2:6].clone(), torch.zeros(1, 2, 4, 4)),
                     torch.full((1, 2, 4, 4), 5.0))
    gates = [torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4)]
    gates[0][..., 3] = 1.0                          # lane 0's ring == canvas column 3
    right_before = right.model_k.noise.clone()

    sync.present_live_ring([left, right], gates, torch.tensor(0.5), zero_noise=True)

    assert float(left.model_k.noise[..., 3].abs().sum()) == 0.0
    assert torch.equal(left.model_k.noise[..., 0:3], canvas_noise[..., 0:3])
    assert torch.equal(right.model_k.noise, right_before)
    # Canvas column 3 is lane 1's window column 1 — a CORE cell it still needs its noise for.
    assert torch.equal(right.model_k.noise[..., 1], canvas_noise[..., 3])
    # The rewrite is x / (1 - sigma), gated: 3.0 / 0.5 == 6.0 at the ring, untouched elsewhere.
    assert torch.equal(left.model_k.latent_image[..., 3], torch.full((1, 2, 4), 6.0))
    assert float(left.model_k.latent_image[..., 0:3].abs().sum()) == 0.0
    assert float(right.model_k.latent_image.abs().sum()) == 0.0     # an empty gate rewrites nothing


def test_the_live_ring_refuses_a_sigma_that_makes_its_algebra_undefined():
    lane = RingLane(RingModelK(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 2)),
                    torch.zeros(1, 1, 2, 2))

    with pytest.raises(RuntimeError, match=r"x / \(1 - sigma\)"):
        sync.present_live_ring([lane], [torch.ones(1, 1, 2, 2)], torch.tensor(1.0))


def test_live_canvas_presents_the_neighbours_live_trajectory_at_every_step(comfy_stubs, vl_clip,
                                                                           lane_stamp):
    # THE ALGEBRA, measured at the model's input. With the ring cells of `noise` zeroed and
    # `latent_image` rewritten to x / (1 - sigma), CONST's noise_scaling returns exactly x —
    # and x is the consolidated canvas the hook just scattered, i.e. the NEIGHBOUR's live
    # trajectory. Asserted PER STEP (this sampler is one eval per step); a 2-eval sampler's
    # intra-step eval sees the same state rescaled by (1 - sigma_mid) / (1 - sigma_step).
    guider = RunGuider()

    run_sync(vl_clip, guider=guider, anchor_source=sync.ANCHOR_SOURCE_CANVAS)

    assert len(guider.log) == 2 * RUN_STEPS
    for step in range(RUN_STEPS):
        for lane_index in (0, 1):
            entry = guider.log[2 * step + lane_index]
            assert entry["lane"] == lane_index
            col = LIVE_RING_COL[lane_index]
            presented, live = entry["input"][..., col], entry["x"][..., col]
            if step == 0:
                # Step 0's canvas is the raw sigma0 * noise + (1 - sigma0) * latent
                # construction, whose arbitrary values do not all survive the division and the
                # re-multiplication exactly. The residue is ONE float32 ulp and it belongs to
                # the round trip, not to the engine — the steps below pin the algebra exactly
                # on values that do round-trip.
                assert bool((presented - live).abs().le(live.abs() * torch.finfo(torch.float32).eps).all())
            else:
                assert torch.equal(presented, live)
    # ... and that value really is the NEIGHBOUR's, hand-computed: at step 1 the canvas holds
    # step 0's results, so lane 0's ring (canvas col 7, inside tile 1's core) carries tile 1's
    # constant and lane 1's ring (canvas col 2, inside tile 0's core) carries tile 0's.
    assert torch.equal(guider.log[2]["input"][..., 7], torch.full((1, 3, 10), lane_value(1, 0)))
    assert torch.equal(guider.log[3]["input"][..., 0], torch.full((1, 3, 10), lane_value(0, 0)))
    # The discriminator against "source image": that mode presents the unrefined input latent.
    latent = image_canvas()[:, ::8, ::8, :].movedim(-1, 1)
    assert not torch.equal(guider.log[2]["input"][..., 7], latent[..., 7])


def test_live_canvas_zeroes_the_ring_noise_once_in_the_step_zero_hook(comfy_stubs, vl_clip,
                                                                      lane_stamp):
    # TIMING. The zeroing runs inside the STEP-0 hook: after comfy bound model_k.noise
    # (samplers.py:991) and consumed it for the initial noise_scaling (:993), before the first
    # eval. So the lane's step-0 trajectory still carries the noise its tile would have had —
    # the raster contract — while every eval from the first one onward sees a zeroed ring.
    guider = RunGuider()

    run_sync(vl_clip, guider=guider, anchor_source=sync.ANCHOR_SOURCE_CANVAS)

    canvas_noise = torch.arange(3 * 10 * 10, dtype=torch.float32).reshape(1, 3, 10, 10)
    latent = image_canvas()[:, ::8, ::8, :].movedim(-1, 1)
    # The ring column of the step-0 trajectory: still sigma0 * noise + (1 - sigma0) * latent,
    # which a zeroing done before the noise_scaling would have flattened.
    initial = SIGMAS[0] * canvas_noise + (1.0 - SIGMAS[0]) * latent
    assert torch.equal(guider.log[0]["x"][..., 7], initial[..., 7])
    # Zeroed at the ring and NOWHERE else, at every eval — including the cells this lane shares
    # with its neighbour, which is the aliasing check at run scale.
    for step in range(RUN_STEPS):
        for lane_index, (x0, x1) in enumerate(LIVE_WINDOWS):
            expected = canvas_noise[..., x0:x1].clone()
            expected[..., LIVE_RING_COL[lane_index]] = 0.0
            assert torch.equal(guider.log[2 * step + lane_index]["noise"], expected)


def test_live_canvas_never_enters_the_ring_schedule(comfy_stubs, vl_clip, lane_stamp, monkeypatch):
    # The lead schedule bridges a frozen REFINED ring to a noisy core; in live-canvas mode
    # nothing frozen-refined is left to lead, and the model must keep core's own
    # scale_latent_inpaint — which is exactly the algebra present_live_ring inverts. (This
    # replaces test_anchor_ring.py's "adjacent leaves the model unpatched while sampling" pin.)
    entered = []
    real_schedule = sampling.anchor_ring_schedule

    def counting(guider, sigmas, enabled=True):
        entered.append(enabled)
        return real_schedule(guider, sigmas, enabled=enabled)

    monkeypatch.setattr(sampling, "anchor_ring_schedule", counting)
    guider = RunGuider()

    run_sync(vl_clip, guider=guider, anchor_source=sync.ANCHOR_SOURCE_CANVAS)

    assert entered == [False]
    assert "scale_latent_inpaint" not in vars(guider.model_patcher.model)


def test_live_canvas_leaves_the_region_paths_frozen_background_on_the_source(
        comfy_stubs, vl_clip, lane_stamp):
    # The region path's mask-0 BACKGROUND is diffused by nobody, so there is no live trajectory
    # to present there: its ring gate is 0 and its latent_image keeps the raw source in either
    # mode. Only the cells a NEIGHBOUR diffuses are rewritten.
    guider = RunGuider()

    run_sync(vl_clip, image=image_canvas(), mask=band_mask(), guider=guider,
             anchor_source=sync.ANCHOR_SOURCE_CANVAS)

    y0, y1, _x0, _x1 = REGION_CROP
    latent = image_canvas()[:, y0:y1, :, :][:, ::8, ::8, :].movedim(-1, 1)
    for lane_index, (x0, x1) in enumerate(LIVE_WINDOWS):
        entry = guider.log[2 + lane_index]                    # step 1, this lane
        window = latent[..., x0:x1]
        ring_col = LIVE_RING_COL[lane_index]
        # Latent rows 0 and 5 are the frozen background: untouched, whole rows.
        assert torch.equal(entry["latent_image"][:, :, 0], window[:, :, 0])
        assert torch.equal(entry["latent_image"][:, :, 5], window[:, :, 5])
        # ... while the ring cells INSIDE the region (rows 1..4) did move onto the live canvas.
        assert not torch.equal(entry["latent_image"][:, :, 1:5, ring_col],
                               window[:, :, 1:5, ring_col])


# ---- modes and intake ------------------------------------------------------


def test_an_unsupported_sampler_passes_the_steppers_error_through(comfy_stubs, vl_clip, lane_stamp):
    sampler = euler_sampler(name="sample_euler_ancestral")

    with pytest.raises(ValueError) as excinfo:
        run_sync(vl_clip, sampler=sampler)

    message = str(excinfo.value)
    assert "euler_ancestral" in message
    for supported in ("euler", "dpmpp_2m", "dpm_fast"):
        assert supported in message
