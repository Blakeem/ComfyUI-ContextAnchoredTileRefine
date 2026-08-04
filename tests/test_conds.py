import pytest
import torch

from context_anchored_tile_refine import conds, grid, sampling
from test_tiling import GridGuider, GridNoise, GridVAE, _layout

SIGMAS = torch.linspace(1.0, 0.0, 5)  # 4 steps


class FakeControl:
    # Duck-typed stand-in for comfy's ControlBase: exactly the attributes conds.py touches,
    # plus a copy() that mirrors ControlBase.copy_to (comfy/controlnet.py:168-183) — including
    # that it does NOT carry previous_controlnet, which is what forces the explicit re-link.
    def __init__(self, hint=None, extra_concat=None, name="cn"):
        self.name = name
        self.cond_hint_original = hint
        self.extra_concat_orig = list(extra_concat) if extra_concat else []
        self.previous_controlnet = None
        self.strength = 1.0
        self.copies = 0

    def copy(self):
        self.copies += 1
        clone = FakeControl(self.cond_hint_original, self.extra_concat_orig, self.name)
        clone.strength = self.strength
        return clone

    def set_previous_controlnet(self, controlnet):
        self.previous_controlnet = controlnet
        return self


def _conds_with(control):
    # The shape core builds: a LIST OF FLAT DICTS per key (convert_cond), with the SAME control
    # object shared between positive and negative (nodes.py:948-956).
    return {
        "positive": [{"cross_attn": torch.zeros(1, 1, 4), "control": control}],
        "negative": [{"cross_attn": torch.zeros(1, 1, 4), "control": control}],
    }


# ---- guard -----------------------------------------------------------------


def test_has_spatial_conds_only_fires_on_control():
    assert conds.has_spatial_conds(None) is False
    assert conds.has_spatial_conds({}) is False
    assert conds.has_spatial_conds({"positive": [{"cross_attn": torch.zeros(1)}]}) is False
    # The descoped USD-style keys must NOT arm the transform — they pass through untouched.
    assert conds.has_spatial_conds({"positive": [{"gligen": object(), "area": (1, 2, 3, 4)}]}) is False
    assert conds.has_spatial_conds({"positive": [{"mask": torch.ones(1, 8, 8), "reference_latents": [torch.zeros(1, 4, 8, 8)]}]}) is False
    assert conds.has_spatial_conds(_conds_with(FakeControl(torch.zeros(1, 3, 8, 8)))) is True


# ---- prepare_hint_canvas ---------------------------------------------------


def test_mismatched_hint_size_is_a_hard_error_naming_both_sizes():
    control = FakeControl(torch.zeros(1, 3, 64, 48))

    with pytest.raises(ValueError) as excinfo:
        conds.prepare_hint_canvas(_conds_with(control), (80, 40))

    message = str(excinfo.value)
    assert "64x48" in message   # the hint
    assert "80x40" in message   # the image being refined


def test_prepare_hint_canvas_keys_by_tensor_identity_and_passes_bbox_through():
    hint = torch.rand(1, 3, 48, 40)
    canvas = conds.prepare_hint_canvas(_conds_with(FakeControl(hint)), (48, 40))

    # One entry even though positive and negative both carry it (one shared tensor).
    assert list(canvas) == [id(hint)]
    assert canvas[id(hint)] is hint


def test_prepare_hint_canvas_applies_the_mask_path_bbox_slice():
    hint = torch.rand(1, 3, 48, 40)

    canvas = conds.prepare_hint_canvas(_conds_with(FakeControl(hint)), (48, 40), (8, 40, 16, 32))

    assert torch.equal(canvas[id(hint)], hint[:, :, 8:40, 16:32])
    assert canvas[id(hint)].shape == (1, 3, 32, 16)


# ---- pad_hint_canvas -------------------------------------------------------


def test_pad_hint_canvas_matches_the_pixel_pad_and_keeps_the_channel_count():
    # C != B and H != W, so a transposed pad (or a channels-last pad) changes the shape.
    hint = torch.rand(2, 5, 50, 34)

    padded = conds.pad_hint_canvas({id(hint): hint}, (50, 34), (56, 40))[id(hint)]

    assert padded.shape == (2, 5, 56, 40)
    assert torch.equal(padded[:, :, :50, :34], hint)
    # Identical to routing the same tensor through the pixel padder channels-last and back.
    expected, _ = sampling.pad_image_to_multiple(hint.movedim(1, -1))
    assert torch.equal(padded, expected.movedim(-1, 1))


def test_pad_hint_canvas_on_an_aligned_canvas_is_the_same_map():
    hint = torch.rand(1, 3, 48, 40)
    canvas = {id(hint): hint}

    assert conds.pad_hint_canvas(canvas, (48, 40), (48, 40)) is canvas


# ---- crop_tile_conds -------------------------------------------------------


def test_crop_tile_conds_slices_the_hint_at_crop_rect():
    hint = torch.rand(1, 3, 64, 64)
    original = _conds_with(FakeControl(hint))
    canvas = conds.prepare_hint_canvas(original, (64, 64))

    cropped = conds.crop_tile_conds(original, canvas, grid.Rect(x0=16, y0=8, x1=48, y1=56))

    for key in ("positive", "negative"):
        tile_hint = cropped[key][0]["control"].cond_hint_original
        assert tile_hint.shape == (1, 3, 48, 32)   # (y1-y0, x1-x0) on dims (-2,-1)
        assert torch.equal(tile_hint, hint[:, :, 8:56, 16:48])


def test_positive_and_negative_share_one_control_copy():
    control = FakeControl(torch.rand(1, 3, 32, 32))
    original = _conds_with(control)
    canvas = conds.prepare_hint_canvas(original, (32, 32))

    cropped = conds.crop_tile_conds(original, canvas, grid.Rect(x0=0, y0=0, x1=16, y1=16))

    assert cropped["positive"][0]["control"] is cropped["negative"][0]["control"]
    assert cropped["positive"][0]["control"] is not control
    assert control.copies == 1   # memoized by id(original), not copied once per cond dict


def test_control_chain_is_copied_and_relinked():
    # copy_to drops previous_controlnet, so without the explicit re-link the second net of a
    # two-ControlNet chain would silently vanish from every tile.
    hint_top = torch.rand(1, 3, 32, 32)
    hint_prev = torch.rand(1, 1, 32, 32)
    previous = FakeControl(hint_prev, name="depth")
    top = FakeControl(hint_top, name="canny")
    top.set_previous_controlnet(previous)
    top.strength = 0.75
    original = _conds_with(top)
    canvas = conds.prepare_hint_canvas(original, (32, 32))

    cropped = conds.crop_tile_conds(original, canvas, grid.Rect(x0=8, y0=16, x1=24, y1=32))

    copied_top = cropped["positive"][0]["control"]
    copied_prev = copied_top.previous_controlnet
    assert copied_top is not top and copied_top.name == "canny" and copied_top.strength == 0.75
    assert copied_prev is not None and copied_prev is not previous
    assert copied_prev.name == "depth" and copied_prev.previous_controlnet is None
    assert torch.equal(copied_top.cond_hint_original, hint_top[:, :, 16:32, 8:24])
    assert torch.equal(copied_prev.cond_hint_original, hint_prev[:, :, 16:32, 8:24])
    # The memo reaches down the chain too: one copy per original link, shared pos/neg.
    assert cropped["negative"][0]["control"].previous_controlnet is copied_prev
    assert top.copies == 1 and previous.copies == 1


def test_placeholder_extra_concat_passes_through_at_a_non_origin_tile():
    # Core auto-inserts this [1,1,1,1] for a concat_mask controlnet (controlnet.py:112-113);
    # slicing it at a non-origin crop_rect would yield a 0-dim tensor and crash common_upscale.
    hint = torch.rand(1, 3, 32, 32)
    placeholder = torch.tensor([[[[1.0]]]])
    original = _conds_with(FakeControl(hint, extra_concat=[placeholder]))
    canvas = conds.prepare_hint_canvas(original, (32, 32))
    assert id(placeholder) not in canvas

    cropped = conds.crop_tile_conds(original, canvas, grid.Rect(x0=16, y0=16, x1=32, y1=32))

    passed = cropped["positive"][0]["control"].extra_concat_orig
    assert len(passed) == 1
    assert passed[0] is placeholder
    assert passed[0].shape == (1, 1, 1, 1)


def test_canvas_sized_extra_concat_is_cropped_with_the_hint():
    hint = torch.rand(1, 3, 32, 32)
    inpaint_mask = torch.rand(1, 1, 32, 32)
    placeholder = torch.tensor([[[[1.0]]]])
    original = _conds_with(FakeControl(hint, extra_concat=[inpaint_mask, placeholder]))
    canvas = conds.prepare_hint_canvas(original, (32, 32))

    cropped = conds.crop_tile_conds(original, canvas, grid.Rect(x0=8, y0=0, x1=32, y1=24))

    passed = cropped["positive"][0]["control"].extra_concat_orig
    assert torch.equal(passed[0], inpaint_mask[:, :, 0:24, 8:32])
    assert passed[1] is placeholder


def test_originals_are_never_mutated():
    hint = torch.rand(1, 3, 32, 32)
    extra = torch.rand(1, 1, 32, 32)
    control = FakeControl(hint, extra_concat=[extra])
    original = _conds_with(control)
    hint_before, extra_before = hint.clone(), extra.clone()
    positive_dict = original["positive"][0]
    canvas = conds.prepare_hint_canvas(original, (32, 32))

    cropped = conds.crop_tile_conds(original, canvas, grid.Rect(x0=8, y0=8, x1=24, y1=24))

    assert control.cond_hint_original is hint
    assert len(control.extra_concat_orig) == 1 and control.extra_concat_orig[0] is extra
    assert control.previous_controlnet is None
    assert torch.equal(hint, hint_before) and torch.equal(extra, extra_before)
    assert original["positive"][0] is positive_dict
    assert positive_dict["control"] is control
    assert cropped["positive"][0] is not positive_dict
    assert cropped["positive"][0]["cross_attn"] is positive_dict["cross_attn"]


def test_cond_without_control_passes_through_by_reference():
    control = FakeControl(torch.rand(1, 3, 16, 16))
    original = {"positive": [{"control": control}], "negative": [{"cross_attn": torch.zeros(1)}]}
    canvas = conds.prepare_hint_canvas(original, (16, 16))

    cropped = conds.crop_tile_conds(original, canvas, grid.Rect(x0=0, y0=0, x1=8, y1=8))

    assert set(cropped) == {"positive", "negative"}
    assert cropped["negative"][0] is original["negative"][0]


# ---- wiring through the pipeline -------------------------------------------


class ControlGuider(GridGuider):
    # GridGuider carrying core-shaped original_conds, recording what each tile actually saw.
    def __init__(self, control):
        super().__init__()
        self.original_conds = {
            "positive": [{"cross_attn": torch.zeros(1, 1, 4), "control": control}],
            "negative": [{"cross_attn": torch.zeros(1, 1, 4), "control": control}],
        }
        self.seen_conds = []

    def sample(self, noise, latent_image, sampler, sigmas, **kwargs):
        self.seen_conds.append(self.original_conds)
        return super().sample(noise, latent_image, sampler, sigmas, **kwargs)


class ControlRaisesGuider(ControlGuider):
    # cf. SecondCallRaisesGuider: blow up on the second tile, once a swap is definitely live.
    def __init__(self, control):
        super().__init__(control)
        self.conds_at_raise = None

    def sample(self, noise, latent_image, sampler, sigmas, **kwargs):
        if len(self.calls) == 1:
            self.conds_at_raise = self.original_conds
            raise RuntimeError("sampler exploded")
        return super().sample(noise, latent_image, sampler, sigmas, **kwargs)


def _run_control(image, guider, cap=56, ctx=0, overlap=16, mask=None):
    return sampling.refine_image(
        image, guider, object(), SIGMAS, GridVAE(), GridNoise(),
        max_tile_width=cap, max_tile_height=cap, context_anchor=ctx,
        context_overlap=overlap, mask=mask,
    )


def test_each_tile_sees_its_own_hint_crop(comfy_stubs):
    image = torch.rand(1, 80, 80, 3)     # /8-aligned: the canvas needs no pad
    hint = torch.rand(1, 3, 80, 80)
    control = FakeControl(hint)
    guider = ControlGuider(control)
    pristine = guider.original_conds

    out = _run_control(image, guider)

    layout = _layout(80, 80, 56, 56, overlap=16)
    assert len(guider.seen_conds) == len(layout.tiles) == 4
    for tile, seen in zip(layout.tiles, guider.seen_conds):
        crop = tile.crop_rect
        for key in ("positive", "negative"):
            assert torch.equal(seen[key][0]["control"].cond_hint_original,
                               hint[:, :, crop.y0:crop.y1, crop.x0:crop.x1])
        assert seen["positive"][0]["control"] is seen["negative"][0]["control"]
        assert seen["positive"][0]["control"] is not control
    # A FRESH copy per tile: get_control only invalidates its hint cache on a shape change,
    # so a reused object would serve tile k's hint to a same-shaped tile k+1.
    per_tile = [seen["positive"][0]["control"] for seen in guider.seen_conds]
    assert len({id(c) for c in per_tile}) == len(per_tile)
    # Restored to the SAME object, with the user's control and hint untouched.
    assert guider.original_conds is pristine
    assert guider.original_conds["positive"][0]["control"] is control
    assert control.cond_hint_original is hint
    assert out.shape == image.shape


def test_hint_rides_the_same_reflect_pad_as_the_canvas(comfy_stubs):
    image = torch.rand(1, 84, 76, 3)     # pads to 88x80 -> 3x3 grid at caps 32
    hint = torch.rand(1, 3, 84, 76)
    guider = ControlGuider(FakeControl(hint))

    _run_control(image, guider, cap=32, ctx=0, overlap=0)

    # Independent route to the padded hint: through the pixel padder, channels-last and back.
    padded_pixels, _ = sampling.pad_image_to_multiple(hint.movedim(1, -1))
    padded_hint = padded_pixels.movedim(-1, 1)
    assert padded_hint.shape == (1, 3, 88, 80)
    layout = _layout(80, 88, 32, 32)
    assert len(guider.seen_conds) == len(layout.tiles) == 9
    for tile, seen in zip(layout.tiles, guider.seen_conds):
        crop = tile.crop_rect
        assert torch.equal(seen["positive"][0]["control"].cond_hint_original,
                           padded_hint[:, :, crop.y0:crop.y1, crop.x0:crop.x1])


def test_mask_path_hint_takes_the_same_bbox_slice_as_the_image(comfy_stubs):
    image = torch.rand(1, 80, 80, 3)
    hint = torch.rand(1, 3, 80, 80)
    mask = torch.zeros(1, 80, 80)
    mask[:, 16:64, 16:64] = 1.0
    guider = ControlGuider(FakeControl(hint))

    _run_control(image, guider, cap=56, ctx=8, overlap=16, mask=mask)

    y0, y1, x0, x1 = sampling._expand_snap_clamp(sampling._mask_bbox(mask >= 0.5), 8, 80, 80)
    assert (y0, y1, x0, x1) == (8, 72, 8, 72)
    sub_hint = hint[:, :, y0:y1, x0:x1]          # the same crop sub_image gets
    layout = _layout(64, 64, 56, 56, ctx=8, overlap=16)
    assert len(guider.seen_conds) == len(layout.tiles) == 4
    for tile, seen in zip(layout.tiles, guider.seen_conds):
        crop = tile.crop_rect
        assert torch.equal(seen["positive"][0]["control"].cond_hint_original,
                           sub_hint[:, :, crop.y0:crop.y1, crop.x0:crop.x1])


def test_hint_that_is_not_the_input_size_raises_before_any_sampling(comfy_stubs):
    image = torch.rand(1, 80, 80, 3)
    guider = ControlGuider(FakeControl(torch.rand(1, 3, 40, 40)))
    vae, noise = GridVAE(), GridNoise()

    with pytest.raises(ValueError, match="40x40"):
        sampling.refine_image(image, guider, object(), SIGMAS, vae, noise, max_tile_width=56, max_tile_height=56, context_anchor=0, context_overlap=16)

    assert vae.encode_calls == [] and noise.calls == [] and guider.calls == []


def test_pristine_conds_restored_after_a_mid_run_raise(comfy_stubs):
    image = torch.rand(1, 80, 80, 3)
    hint = torch.rand(1, 3, 80, 80)
    control = FakeControl(hint)
    guider = ControlRaisesGuider(control)
    pristine = guider.original_conds

    with pytest.raises(RuntimeError, match="sampler exploded"):
        _run_control(image, guider)

    assert guider.conds_at_raise is not None and guider.conds_at_raise is not pristine
    assert guider.original_conds is pristine
    assert guider.original_conds["positive"][0]["control"] is control
    assert control.cond_hint_original is hint


def test_conds_without_control_leave_the_pipeline_byte_identical(comfy_stubs):
    image = torch.rand(1, 80, 80, 3)

    plain = sampling.refine_image(image, GridGuider(), object(), SIGMAS, GridVAE(), GridNoise(), max_tile_width=56, max_tile_height=56, context_anchor=0, context_overlap=16)

    guider = GridGuider()
    guider.original_conds = {"positive": [{"cross_attn": torch.zeros(1, 1, 4)}], "negative": [{}]}
    pristine = guider.original_conds
    with_conds = _run_control(image, guider)

    assert torch.equal(with_conds, plain)
    assert guider.original_conds is pristine
