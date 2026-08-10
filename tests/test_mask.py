import torch
from test_tiling import GridGuider, GridNoise, GridVAE, _layout

from context_anchored_tile_refine import sampling

SIGMAS = torch.linspace(1.0, 0.0, 5)  # 4 steps


def _run_mask(image, mask, cap_w=64, cap_h=64, ctx=8, overlap=0, guider=None):
    guider = guider if guider is not None else GridGuider()
    vae, noise = GridVAE(), GridNoise()
    out = sampling.refine_image(image, guider, object(), SIGMAS, vae, noise, max_tile_width=cap_w, max_tile_height=cap_h, context_anchor=ctx, context_overlap=overlap, mask=mask)
    return out, guider, vae, noise


def _refine_tiles_direct(sub_image, sub_mask, cap_w, cap_h, ctx, overlap):
    # Fresh position-aware fakes reproduce refined_sub bit-for-bit (all deterministic), so
    # the AA composite can be asserted against the SAME expression production uses.
    guider, vae, noise = GridGuider(), GridVAE(), GridNoise()
    return sampling._refine_tiles(sub_image, guider, object(), SIGMAS, vae, noise, cap_w, cap_h, ctx, overlap, region_pixel=sub_mask)


# ---- bbox / crop geometry -------------------------------------------------


def test_mask_bbox_union_over_batch_exclusive():
    mask = torch.zeros(2, 40, 48, dtype=torch.bool)
    mask[0, 8:16, 8:16] = True
    mask[1, 20:24, 30:40] = True

    # Union over the batch; y1/x1 are EXCLUSIVE (max index + 1).
    assert sampling._mask_bbox(mask) == (8, 24, 8, 40)


def test_mask_bbox_empty_is_none():
    assert sampling._mask_bbox(torch.zeros(1, 16, 16, dtype=torch.bool)) is None


def test_expand_snap_clamp_snaps_out():
    # anchor 0: x0/y0 snap DOWN to /8, x1/y1 snap UP to /8.
    assert sampling._expand_snap_clamp((10, 21, 10, 21), 0, 64, 64) == (8, 24, 8, 24)


def test_expand_snap_clamp_expands_by_anchor():
    assert sampling._expand_snap_clamp((16, 32, 16, 32), 8, 100, 100) == (8, 40, 8, 40)


def test_expand_snap_clamp_1px_bbox_is_at_least_8():
    y0, y1, x0, x1 = sampling._expand_snap_clamp((100, 101, 103, 104), 0, 200, 200)
    assert (y0, y1, x0, x1) == (96, 104, 96, 104)
    assert y1 - y0 >= 8 and x1 - x0 >= 8


def test_expand_snap_clamp_border_clamp_non_multiple():
    # A mask touching a non-/8 image border: y1 snaps up to 336 then clamps to H=322
    # (non-/8) — the resulting crop triggers _refine_tiles' internal pad.
    assert sampling._expand_snap_clamp((300, 322, 4, 12), 8, 322, 330) == (288, 322, 0, 24)


def test_expand_snap_clamp_anchor0_non_multiple_border_floors_at_8():
    # anchor=0 leaves no snap-down slack: y1/x1 snap up to 24 then clamp to 17, which
    # would leave a 1px crop. _refine_tiles' reflect pad needs pad < dim, so the crop is
    # re-expanded back to the 8px floor.
    assert sampling._expand_snap_clamp((16, 17, 16, 17), 0, 17, 17) == (9, 17, 9, 17)


def test_masked_refine_anchor0_corner_mask_on_non_multiple_image(comfy_stubs):
    # Regression: the 1px crop above used to reach pad_image_to_multiple and raise
    # "Padding size should be less than the corresponding input dimension".
    image = torch.rand(1, 17, 17, 3)
    mask = torch.zeros(1, 17, 17)
    mask[:, 16, 16] = 1.0

    out, _guider, _vae, _noise = _run_mask(image, mask, ctx=0, overlap=0)

    assert out.shape == image.shape
    # Everything outside the 8px-floored crop (9:17, 9:17) stays byte-identical.
    assert torch.equal(out[:, :9, :, :], image[:, :9, :, :])
    assert torch.equal(out[:, :, :9, :], image[:, :, :9, :])


# ---- latent region gate (tile_gradient ∩ region_latent) -------------------


def test_region_gate_is_grad_times_region_latent(comfy_stubs):
    # ctx=8 extends the crop to the full /8 image so background cells (region 0) sit inside
    # the crop; overlap=16 forces a 2x2 grid so tile_gradient is a real ring indicator.
    image = torch.rand(1, 80, 80, 3)
    mask = torch.zeros(1, 80, 80)
    mask[:, 16:64, 16:64] = 1.0

    _out, guider, _vae, _noise = _run_mask(image, mask, cap_w=56, cap_h=56, ctx=8, overlap=16)

    # crop = bbox (16,64) expand 8 -> (8,72), full 64x64 sub-canvas at origin 8.
    y0, y1, x0, x1 = sampling._expand_snap_clamp(sampling._mask_bbox(mask >= 0.5), 8, 80, 80)
    assert (y0, y1, x0, x1) == (8, 72, 8, 72)
    sub_mask = (mask >= 0.5)[:, y0:y1, x0:x1].to(image.dtype)
    region_latent = (torch.nn.functional.max_pool2d(sub_mask[:, None], 8) > 0).float()[:, 0]

    layout = _layout(64, 64, 56, 56, ctx=8, overlap=16)
    assert len(guider.calls) == 4
    for tile, call in zip(layout.tiles, guider.calls, strict=True):
        crop = tile.crop_rect
        grad = sampling.tile_gradient(crop, tile.overlap_inner_rect, scale=8)
        reg = region_latent[:, crop.y0 // 8:crop.y1 // 8, crop.x0 // 8:crop.x1 // 8]
        dm = call["denoise_mask"]
        assert dm.shape == (1, 1, crop.y1 // 8 - crop.y0 // 8, crop.x1 // 8 - crop.x0 // 8)
        assert torch.equal(dm[:, 0], grad * reg)
        assert ((dm == 0.0) | (dm == 1.0)).all()   # strictly binary


def test_n1_masked_tile_denoise_mask_is_region_latent(comfy_stubs):
    # Sub-tile mask: a 1x1 grid still passes a NON-None denoise_mask (== region_latent),
    # anchoring the in-crop frozen background so the subject conditions on it.
    image = torch.rand(1, 40, 40, 3)
    mask = torch.zeros(1, 40, 40)
    mask[:, 16:24, 16:24] = 1.0

    _out, guider, _vae, _noise = _run_mask(image, mask, cap_w=64, cap_h=64, ctx=8, overlap=0)

    assert len(guider.calls) == 1
    dm = guider.calls[0]["denoise_mask"]
    assert dm is not None
    sub_mask = (mask >= 0.5)[:, 8:32, 8:32].to(image.dtype)
    region_latent = (torch.nn.functional.max_pool2d(sub_mask[:, None], 8) > 0).float()[:, 0]
    assert torch.equal(dm[:, 0], region_latent)
    expected = torch.zeros(1, 3, 3)
    expected[:, 1, 1] = 1.0
    assert torch.equal(dm[:, 0], expected)


def test_masked_run_covers_mid_cell_subject_edge(comfy_stubs):
    # Cover downsample (max_pool2d, NOT avg+threshold): a subject block straddling cell
    # boundaries so cell (1,1) is only 25% masked — every masked pixel must still land in a
    # RELEASED (==1) latent cell, else a subject-edge pixel would be frozen yet pasted.
    image = torch.rand(1, 32, 32, 3)
    mask = torch.zeros(1, 32, 32)
    mask[:, 12:20, 12:20] = 1.0

    _out, guider, _vae, _noise = _run_mask(image, mask, cap_w=64, cap_h=64, ctx=8, overlap=0)

    assert len(guider.calls) == 1
    dm = guider.calls[0]["denoise_mask"]   # crop == full 32x32 -> region_latent [1,4,4]
    assert dm.shape == (1, 1, 4, 4)
    for r in range(12, 20):
        for c in range(12, 20):
            assert dm[0, 0, r // 8, c // 8] == 1.0
    expected = torch.zeros(1, 4, 4)
    expected[:, 1:3, 1:3] = 1.0
    assert torch.equal(dm[:, 0], expected)


# ---- AA composite (bitwise) -----------------------------------------------


def test_aa_alpha_step_edge_values():
    sub_mask = torch.zeros(1, 1, 8)
    sub_mask[:, :, 4:] = 1.0

    aa = sampling._aa_alpha(sub_mask)

    assert aa.shape == (1, 1, 8)
    # Separable [0.25,0.5,0.25] blur of a step edge: ...,0,0.25,0.75,1,...
    assert torch.equal(aa[0, 0], torch.tensor([0.0, 0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 1.0]))


def test_aa_alpha_b2_hand_assembled():
    sub_mask = torch.zeros(2, 1, 8)
    sub_mask[0, :, 4:] = 1.0
    sub_mask[1, :, :4] = 1.0

    aa = sampling._aa_alpha(sub_mask)

    assert torch.equal(aa[0, 0], torch.tensor([0.0, 0.0, 0.0, 0.25, 0.75, 1.0, 1.0, 1.0]))
    assert torch.equal(aa[1, 0], torch.tensor([1.0, 1.0, 1.0, 0.75, 0.25, 0.0, 0.0, 0.0]))


def test_aa_alpha_all_ones_stays_ones():
    # Replicate padding keeps an all-ones mask all-ones (no dark border on a full mask).
    assert torch.equal(sampling._aa_alpha(torch.ones(2, 6, 6)), torch.ones(2, 6, 6))


def test_aa_composite_matches_hand_assembled_bitwise(comfy_stubs):
    image = torch.rand(2, 48, 48, 3)
    mask = torch.zeros(2, 48, 48)
    mask[0, 8:32, 8:32] = 1.0
    mask[1, 16:40, 16:40] = 1.0

    out, _guider, _vae, _noise = _run_mask(image, mask, cap_w=64, cap_h=64, ctx=8, overlap=0)

    mask_bin = mask >= 0.5
    y0, y1, x0, x1 = sampling._expand_snap_clamp(sampling._mask_bbox(mask_bin), 8, 48, 48)
    sub_image = image[:, y0:y1, x0:x1, :]
    sub_mask = mask_bin[:, y0:y1, x0:x1].to(image.dtype)
    refined_sub = _refine_tiles_direct(sub_image, sub_mask, 64, 64, 8, 0)
    aa = sampling._aa_alpha(sub_mask)[..., None]
    expected_crop = aa * refined_sub + (1.0 - aa) * image[:, y0:y1, x0:x1, :]

    assert torch.equal(out[:, y0:y1, x0:x1, :], expected_crop)


# ---- batch, edge cases, locality ------------------------------------------


def test_batch_per_row_masks(comfy_stubs):
    image = torch.rand(2, 48, 48, 3)
    mask = torch.zeros(2, 48, 48)
    mask[0, 8:24, 8:24] = 1.0
    mask[1, 24:40, 24:40] = 1.0

    _out, guider, _vae, _noise = _run_mask(image, mask, cap_w=64, cap_h=64, ctx=8, overlap=0)

    # One shared union crop for both rows (expands+snaps to the full image here).
    assert sampling._mask_bbox(mask >= 0.5) == (8, 40, 8, 40)
    assert sampling._expand_snap_clamp((8, 40, 8, 40), 8, 48, 48) == (0, 48, 0, 48)
    # Per-row region gate differs.
    assert len(guider.calls) == 1
    dm = guider.calls[0]["denoise_mask"]
    assert dm.shape[0] == 2
    assert not torch.equal(dm[0], dm[1])


def test_empty_mask_returns_clone_no_side_effects():
    image = torch.rand(1, 48, 48, 3)
    mask = torch.zeros(1, 48, 48)
    guider, vae, noise = GridGuider(), GridVAE(), GridNoise()

    out = sampling.refine_image(image, guider, object(), SIGMAS, vae, noise, max_tile_width=64, max_tile_height=64, context_anchor=8, context_overlap=0, mask=mask)

    assert torch.equal(out, image)
    assert out is not image
    assert vae.encode_calls == [] and vae.decode_calls == 0
    assert guider.calls == [] and noise.calls == []


def test_whole_image_mask_matches_no_mask_refine(comfy_stubs):
    image = torch.rand(1, 48, 48, 3)

    out_masked, *_ = _run_mask(image, torch.ones(1, 48, 48), cap_w=40, cap_h=40, ctx=8, overlap=0)
    out_plain = sampling.refine_image(image, GridGuider(), object(), SIGMAS, GridVAE(), GridNoise(), max_tile_width=40, max_tile_height=40, context_anchor=8, context_overlap=0, mask=None)

    # A full mask ≈ the whole-image refine, and replicate-padded AA leaves no dark border.
    assert torch.equal(out_masked, out_plain)
    assert torch.equal(sampling._aa_alpha(torch.ones(1, 48, 48)), torch.ones(1, 48, 48))


def test_soft_mask_hardened_at_half(comfy_stubs):
    image = torch.rand(1, 48, 48, 3)
    soft = torch.zeros(1, 48, 48)
    soft[:, 8:32, 8:32] = 0.7    # >= 0.5 -> in
    soft[:, 32:40, 8:32] = 0.3   # <  0.5 -> out
    hard = (soft >= 0.5).float()

    out_soft, *_ = _run_mask(image, soft, cap_w=64, cap_h=64, ctx=8, overlap=0)
    out_hard, *_ = _run_mask(image, hard, cap_w=64, cap_h=64, ctx=8, overlap=0)

    assert torch.equal(out_soft, out_hard)


def test_context_anchor_zero_crops_tight(comfy_stubs):
    image = torch.rand(1, 48, 48, 3)
    mask = torch.zeros(1, 48, 48)
    mask[:, 16:32, 16:32] = 1.0

    out, _guider, _vae, _noise = _run_mask(image, mask, cap_w=64, cap_h=64, ctx=0, overlap=0)

    # anchor 0 -> crop is exactly the /8-snapped bbox, no frozen halo.
    assert sampling._expand_snap_clamp(sampling._mask_bbox(mask >= 0.5), 0, 48, 48) == (16, 32, 16, 32)
    full = torch.ones(1, 48, 48, dtype=torch.bool)
    full[:, 16:32, 16:32] = False
    assert torch.equal(out[full], image[full])
    assert out.shape == (1, 48, 48, 3)


def test_border_touching_mask_internal_pad(comfy_stubs):
    # Non-/8 image; mask touches the bottom-right border -> clamped non-/8 crop -> the
    # internal pad fires (region downsampled in-helper). Background stays untouched.
    image = torch.rand(1, 50, 50, 3)
    mask = torch.zeros(1, 50, 50)
    mask[:, 30:50, 30:50] = 1.0

    out, _guider, _vae, _noise = _run_mask(image, mask, cap_w=64, cap_h=64, ctx=8, overlap=0)

    y0, y1, x0, x1 = sampling._expand_snap_clamp(sampling._mask_bbox(mask >= 0.5), 8, 50, 50)
    assert (y0, y1, x0, x1) == (16, 50, 16, 50)   # 34x34 crop, non-/8 -> internal pad
    assert out.shape == (1, 50, 50, 3)
    full = torch.ones(1, 50, 50, dtype=torch.bool)
    full[:, 16:50, 16:50] = False
    assert torch.equal(out[full], image[full])


def test_locality_outside_crop_and_where_aa_zero(comfy_stubs):
    image = torch.rand(1, 64, 64, 3)
    mask = torch.zeros(1, 64, 64)
    mask[:, 24:40, 24:40] = 1.0
    before_img, before_mask = image.clone(), mask.clone()

    out, _guider, _vae, _noise = _run_mask(image, mask, cap_w=64, cap_h=64, ctx=8, overlap=0)

    y0, y1, x0, x1 = sampling._expand_snap_clamp(sampling._mask_bbox(mask >= 0.5), 8, 64, 64)
    # Outside the crop: byte-identical to the input.
    outside = torch.ones(1, 64, 64, dtype=torch.bool)
    outside[:, y0:y1, x0:x1] = False
    assert torch.equal(out[outside], image[outside])
    # Inside the crop, wherever aa == 0 the output is byte-identical to the input.
    aa = sampling._aa_alpha((mask >= 0.5)[:, y0:y1, x0:x1].to(image.dtype))
    zero = aa == 0
    assert zero.any()
    assert torch.equal(out[:, y0:y1, x0:x1, :][zero], image[:, y0:y1, x0:x1, :][zero])
    # Caller tensors never mutated.
    assert torch.equal(image, before_img)
    assert torch.equal(mask, before_mask)


def test_rgba_mask_input_returns_three_channels(comfy_stubs):
    # A 4-channel RGBA image + mask must not channel-mismatch on the composite: _refine_tiles
    # decodes a 3-channel refined_sub, so the mask path narrows to RGB like the no-mask path
    # (which drops alpha) rather than raising on the 3ch-vs-4ch blend.
    image = torch.rand(1, 64, 64, 4)
    mask = torch.zeros(1, 64, 64)
    mask[:, 16:48, 16:48] = 1.0

    out, _guider, _vae, _noise = _run_mask(image, mask, cap_w=64, cap_h=64, ctx=8, overlap=0)
    assert out.shape == (1, 64, 64, 3)
