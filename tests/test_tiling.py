import pytest
import torch
from test_sampling import fake_model_patcher

from context_anchored_tile_refine import grid, sampling

SIGMAS = torch.linspace(1.0, 0.0, 5)  # 4 steps


class GridVAE:
    # Position-preserving fakes: encode = stride-8 downsample, decode = 8x nearest
    # upsample, so a tile pasted or sliced at the wrong position changes values.
    latent_channels = 3

    def __init__(self):
        self.encode_calls = []
        self.decode_calls = 0

    def encode(self, pixels):
        # Snapshot: refine_image passes a view of its live canvas, which mutates
        # after this call; recordings must keep encode-time values.
        pixels = pixels.clone()
        self.encode_calls.append(pixels)
        return pixels[:, ::8, ::8, :].movedim(-1, 1)

    def decode(self, samples):
        self.decode_calls += 1
        return samples.movedim(1, -1).repeat_interleave(8, dim=1).repeat_interleave(8, dim=2)


class GridNoise:
    # arange over the whole canvas: every latent cell gets a unique value, so slice
    # position errors are detectable exactly.
    def __init__(self, seed=11):
        self.seed = seed
        self.calls = []

    def generate_noise(self, input_latent):
        self.calls.append(input_latent)
        samples = input_latent["samples"]
        return torch.arange(samples.numel(), dtype=torch.float32).reshape(samples.shape)


class GridGuider:
    def __init__(self):
        self.model_patcher = fake_model_patcher()
        self.model_options = {}
        self.calls = []

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        self.calls.append({
            "noise": noise,
            "latent_image": latent_image,
            "seed": seed,
            "denoise_mask": denoise_mask,
            "mask_fn": self.model_options.get("denoise_mask_function"),
        })
        steps = sigmas.shape[-1] - 1
        for step in range(steps):
            callback(step, latent_image, latent_image, steps)
        return latent_image + noise


class RecordingPreviewer:
    def __init__(self):
        self.calls = []

    def decode_latent_to_preview_image(self, preview_format, x0):
        self.calls.append((preview_format, x0))
        return ("JPEG", len(self.calls), 512)


def _run(image, cap_w=32, cap_h=32, ctx=0, overlap=0, guider=None):
    # Tiny caps force real grids (refine_image sits below the widget-min-256
    # validation): L=80 cap=32 -> n=3, bases 32/32/16.
    guider = guider if guider is not None else GridGuider()
    vae, noise = GridVAE(), GridNoise()
    out = sampling.refine_image(image, guider, object(), SIGMAS, vae, noise, max_tile_width=cap_w, max_tile_height=cap_h, context_anchor=ctx, context_overlap=overlap)
    return out, guider, vae, noise


def _layout(w, h, cap_w, cap_h, ctx=0, overlap=0):
    sx = grid.solve_axis(w, cap_w, ctx, overlap)
    sy = grid.solve_axis(h, cap_h, ctx, overlap)
    return grid.build_layout(w, h, sx, sy, ctx, overlap)


def test_noise_drawn_once_for_whole_canvas(comfy_stubs):
    image = torch.rand(1, 80, 80, 3)

    _out, _guider, _vae, noise = _run(image)

    assert len(noise.calls) == 1
    assert set(noise.calls[0].keys()) == {"samples"}
    dummy = noise.calls[0]["samples"]
    assert dummy.shape == (1, 3, 10, 10)
    assert dummy.dtype == torch.float32
    assert torch.equal(dummy, torch.zeros_like(dummy))


def test_each_tile_gets_its_canvas_noise_slice(comfy_stubs):
    image = torch.rand(1, 80, 80, 3)

    _out, guider, _vae, _noise = _run(image)

    layout = _layout(80, 80, 32, 32)
    canvas_noise = torch.arange(300, dtype=torch.float32).reshape(1, 3, 10, 10)
    assert len(guider.calls) == 9
    for tile, call in zip(layout.tiles, guider.calls, strict=True):
        crop = tile.crop_rect
        expected = canvas_noise[:, :, crop.y0 // 8:crop.y1 // 8, crop.x0 // 8:crop.x1 // 8]
        assert torch.equal(call["noise"], expected)
        assert call["noise"].is_contiguous()


def test_each_encode_gets_the_core_crop(comfy_stubs):
    image = torch.rand(1, 80, 80, 3)

    _out, _guider, vae, _noise = _run(image)

    layout = _layout(80, 80, 32, 32)
    assert len(vae.encode_calls) == 9
    for tile, pixels in zip(layout.tiles, vae.encode_calls, strict=True):
        core = tile.core  # crop_rect == core at r=0
        assert pixels.shape == (1, core.y1 - core.y0, core.x1 - core.x0, 3)
        assert torch.equal(pixels, image[:, core.y0:core.y1, core.x0:core.x1, :])


def test_output_equals_hand_assembled_paste(comfy_stubs):
    image = torch.rand(1, 84, 76, 3)  # pads to 88x80 -> 3x3 grid at caps 32

    out, _guider, _vae, _noise = _run(image)

    padded, _ = sampling.pad_image_to_multiple(image)
    layout = _layout(80, 88, 32, 32)
    canvas_noise = torch.arange(330, dtype=torch.float32).reshape(1, 3, 11, 10)
    expected = torch.empty(1, 88, 80, 3)
    for tile in layout.tiles:
        crop = tile.crop_rect
        latent = padded[:, crop.y0:crop.y1, crop.x0:crop.x1, :][:, ::8, ::8, :].movedim(-1, 1)
        refined = latent + canvas_noise[:, :, crop.y0 // 8:crop.y1 // 8, crop.x0 // 8:crop.x1 // 8]
        expected[:, crop.y0:crop.y1, crop.x0:crop.x1, :] = (
            refined.movedim(1, -1).repeat_interleave(8, dim=1).repeat_interleave(8, dim=2)
        )
    assert out.shape == (1, 84, 76, 3)
    assert torch.equal(out, expected[:, :84, :76, :])


def test_single_tile_matches_whole_image_sequence(comfy_stubs):
    # A cap-fitting image delegates to the feature-2 call sequence: one
    # encode/sample/decode, full-canvas noise dict semantics, whole-image ProgressBar.
    image = torch.rand(1, 32, 24, 3)

    out, guider, vae, noise = _run(image, cap_w=64, cap_h=64)

    assert len(vae.encode_calls) == 1 and vae.decode_calls == 1
    assert len(guider.calls) == 1
    assert len(noise.calls) == 1
    dummy = noise.calls[0]["samples"]
    assert dummy.shape == (1, 3, 4, 3)
    assert torch.equal(dummy, torch.zeros_like(dummy))
    assert torch.equal(vae.encode_calls[0], image)
    canvas_noise = torch.arange(36, dtype=torch.float32).reshape(1, 3, 4, 3)
    assert torch.equal(guider.calls[0]["noise"], canvas_noise)
    assert guider.calls[0]["noise"].is_contiguous()
    (pbar,) = comfy_stubs["progress_bars"]
    assert pbar.total == 4  # steps * 1 tile == prepare_callback's whole-image total
    assert out.shape == (1, 32, 24, 3)


def test_interrupt_checked_once_per_tile(comfy_stubs):
    image = torch.rand(1, 80, 80, 3)

    _run(image)

    assert comfy_stubs["interrupt_calls"] == 9


def test_progress_aggregates_across_tiles(comfy_stubs):
    image = torch.rand(1, 80, 80, 3)

    _run(image)

    (pbar,) = comfy_stubs["progress_bars"]
    assert pbar.total == 36  # 4 steps * 9 tiles
    assert pbar.updates == [
        (tile * 4 + step + 1, 36, None) for tile in range(9) for step in range(4)
    ]
    assert pbar.updates[-1][0] == 36


def test_previewer_output_forwarded_to_progress(comfy_stubs):
    comfy_stubs["previewer"] = RecordingPreviewer()
    image = torch.rand(1, 80, 80, 3)

    _out, guider, _vae, _noise = _run(image)

    previewer = comfy_stubs["previewer"]
    assert len(previewer.calls) == 36
    for i, (preview_format, x0) in enumerate(previewer.calls):
        assert preview_format == "JPEG"
        assert x0 is guider.calls[i // 4]["latent_image"]
    (pbar,) = comfy_stubs["progress_bars"]
    assert [update[2] for update in pbar.updates] == [("JPEG", i + 1, 512) for i in range(36)]


# ---- feather-active pipeline ----------------------------------------------
# Config used throughout: 80x80 image, caps 56x56, ctx=0, overlap=16 -> 2x2 grid
# (base=40, last=40, every crop 56x56, latent masks 7x7 all-ones since ctx=0).


class SecondCallRaisesGuider(GridGuider):
    def sample(self, noise, latent_image, sampler, sigmas, **kwargs):
        if len(self.calls) == 1:
            raise RuntimeError("sampler exploded")
        return super().sample(noise, latent_image, sampler, sigmas, **kwargs)


def _decode_tile(canvas, source, layout, idx):
    # The fakes' VAE round-trip for a single tile, mirroring refine_image's encode-source
    # split exactly: the DIFFUSED core+overlap band (overlap_inner_rect) is taken from the
    # FROZEN raw `source`, the frozen context_anchor ring (crop \ inner) from the live
    # `canvas`. Then encode = stride-8 downsample, sample = + canvas_noise slice, decode =
    # 8x nearest upsample. Returns the full decoded crop.
    tile = layout.tiles[idx]
    crop, inner = tile.crop_rect, tile.overlap_inner_rect
    batch, height, width = canvas.shape[0], canvas.shape[1], canvas.shape[2]
    canvas_noise = torch.arange(batch * 3 * (height // 8) * (width // 8), dtype=torch.float32).reshape(batch, 3, height // 8, width // 8)
    enc = canvas[:, crop.y0:crop.y1, crop.x0:crop.x1, :].clone()
    enc[:, inner.y0 - crop.y0:inner.y1 - crop.y0, inner.x0 - crop.x0:inner.x1 - crop.x0, :] = source[:, inner.y0:inner.y1, inner.x0:inner.x1, :]
    latent = enc[:, ::8, ::8, :].movedim(-1, 1)
    refined = latent + canvas_noise[:, :, crop.y0 // 8:crop.y1 // 8, crop.x0 // 8:crop.x1 // 8]
    return refined.movedim(1, -1).repeat_interleave(8, dim=1).repeat_interleave(8, dim=2)


def _simulate_feather_canvas(image, layout, n_tiles=None, displace=True):
    # The fakes' math with the SAME encode-source split, directional feather, paste_rect,
    # and raster order as production (all tiles here are expanded: crop != core). Each tile
    # decodes its full crop — diffused core+overlap from the frozen raw source, anchor ring
    # from the live canvas — then its paste_rect is composited with feather_alpha: 1 through
    # the core (so the seam is 100% this tile), ramping to 0 across the kept top/left overlap
    # bands. The composite expression matches sampling.refine_image exactly; feather_alpha's
    # own math is pinned independently in test_feather.py. Test images are /8-aligned, so the
    # frozen raw source is the input `image` itself (padded == image).
    #
    # The seam displacement and the per-tile DC offset come from sampling, deliberately: both
    # are reductions over the two tiles' own disagreement, so re-deriving them here would be a
    # second copy of the algorithm rather than an independent check. Their VALUES are pinned in
    # test_seam_routing.py and test_dc_match.py; what this mirror independently reproduces is
    # the VAE round trip, the raster order, the encode-source split, and the paste geometry —
    # including that the DC offset is subtracted BEFORE the routing surface is built.
    # displace=False gives the undisplaced composite, used to show what routing changed.
    source = image
    canvas = image.clone()
    tiles = layout.tiles if n_tiles is None else layout.tiles[:n_tiles]
    for i, tile in enumerate(tiles):
        crop, core, paste = tile.crop_rect, tile.core, tile.paste_rect
        decoded = _decode_tile(canvas, source, layout, i)
        sub = decoded[:, paste.y0 - crop.y0:paste.y1 - crop.y0, paste.x0 - crop.x0:paste.x1 - crop.x0, :]
        region = canvas[:, paste.y0:paste.y1, paste.x0:paste.x1, :]
        dc_offset = sampling.seam_dc_offset(tile, sub, region)
        if dc_offset is not None:
            sub = sub - dc_offset
        disp_top, disp_left = sampling.seam_displacements(tile, sub, region) if displace else (None, None)
        alpha = sampling.feather_alpha(paste, core, layout.overlap, tile.kept_top, tile.kept_left,
                                       disp_top=disp_top, disp_left=disp_left)[..., None]
        canvas[:, paste.y0:paste.y1, paste.x0:paste.x1, :] = alpha * sub + (1.0 - alpha) * region
    return canvas


def test_overlap_masks_forwarded_per_tile(comfy_stubs):
    image = torch.rand(1, 80, 80, 3)

    _out, guider, _vae, _noise = _run(image, cap_w=56, cap_h=56, overlap=16)

    layout = _layout(80, 80, 56, 56, overlap=16)
    assert len(guider.calls) == 4
    for tile, call in zip(layout.tiles, guider.calls, strict=True):
        # Handed over in prepare_mask's output form ([B,1,h,w]); values stay the binary
        # overlap_inner_rect indicator (core+overlap = 1, ring = 0), not a ramp.
        # At ctx=0 the crop == overlap_inner_rect, so there is no context ring and the mask
        # is all-ones; the ctx>0 freeze is covered by test_ctx_ring_frozen_in_binary_mask.
        assert call["denoise_mask"].shape == (1, 1, 7, 7)
        assert torch.equal(call["denoise_mask"][0, 0], sampling.tile_gradient(tile.crop_rect, tile.overlap_inner_rect, scale=8))
        assert torch.equal(call["denoise_mask"], torch.ones(1, 1, 7, 7))


def test_ctx_ring_frozen_in_binary_mask(comfy_stubs):
    # ctx>0 creates a real context_anchor ring the ctx=0 tests can't exercise. The binary
    # latent mask must be 1 over core+overlap (== tile.overlap_inner_rect) and hard 0
    # through the frozen ring, so the whole core+overlap denoises at full strength and the
    # anchor stays frozen. Config: 80x80, caps 72, ctx=16, overlap=16 ->
    # r=32, 2x2 grid (base=40), every crop 72x72 -> latent 9x9, inner 56x56 -> 7x7 ones.
    image = torch.rand(1, 80, 80, 3)

    _out, guider, _vae, _noise = _run(image, cap_w=72, cap_h=72, ctx=16, overlap=16)

    layout = _layout(80, 80, 72, 72, ctx=16, overlap=16)
    assert len(guider.calls) == 4
    for tile, call in zip(layout.tiles, guider.calls, strict=True):
        crop, oir = tile.crop_rect, tile.overlap_inner_rect
        assert call["denoise_mask"].shape == (1, 1, 9, 9)
        mask = call["denoise_mask"][0, 0]
        assert torch.equal(mask, sampling.tile_gradient(crop, oir, scale=8))
        # Strictly binary, with BOTH a frozen ring (0s) and a released core+overlap (1s).
        assert ((mask == 0.0) | (mask == 1.0)).all()
        assert (mask == 0.0).any()
        assert (mask == 1.0).any()
        # The 1-region is exactly overlap_inner_rect's /8-aligned cell block; else 0.
        expected = torch.zeros(9, 9)
        expected[(oir.y0 - crop.y0) // 8:(oir.y1 - crop.y0) // 8, (oir.x0 - crop.x0) // 8:(oir.x1 - crop.x0) // 8] = 1.0
        assert torch.equal(mask, expected)


def test_feather_run_never_writes_model_options(comfy_stubs):
    # A feather-active run passes the binary latent denoise_mask statically to guider.sample
    # and never writes guider.model_options (no denoise_mask_function injection).
    _out, guider, _vae, _noise = _run(torch.rand(1, 80, 80, 3), cap_w=56, cap_h=56, overlap=16)

    assert guider.model_options == {}


def test_feather_model_options_untouched_when_guider_raises(comfy_stubs):
    # Even when the sampler raises mid-run, model_options is never written (no try/finally
    # to clean up because nothing was ever injected).
    guider = SecondCallRaisesGuider()

    with pytest.raises(RuntimeError, match="sampler exploded"):
        _run(torch.rand(1, 80, 80, 3), cap_w=56, cap_h=56, overlap=16, guider=guider)

    assert len(guider.calls) == 1
    assert guider.model_options == {}


def test_feather_preset_mask_function_respected(comfy_stubs):
    def sentinel(sigma, denoise_mask, extra_options):
        raise AssertionError("never invoked by the fakes")

    guider = GridGuider()
    guider.model_options["denoise_mask_function"] = sentinel

    _out, guider, _vae, _noise = _run(torch.rand(1, 80, 80, 3), cap_w=56, cap_h=56, overlap=16, guider=guider)

    assert len(guider.calls) == 4
    for call in guider.calls:
        assert call["mask_fn"] is sentinel
    assert guider.model_options.get("denoise_mask_function") is sentinel


def test_feather_output_matches_hand_assembled_composite(comfy_stubs):
    # Composition is a directional feather: each tile writes its paste_rect (core + the
    # kept top/left overlap bands) blended by feather_alpha. Expected reproduces the exact
    # composite in raster order from a live canvas.
    image = torch.rand(2, 80, 80, 3)

    out, _guider, _vae, _noise = _run(image, cap_w=56, cap_h=56, overlap=16)

    layout = _layout(80, 80, 56, 56, overlap=16)
    expected = _simulate_feather_canvas(image, layout)
    assert out.shape == (2, 80, 80, 3)
    assert torch.equal(out, expected)


def test_later_tile_feathers_into_earlier_neighbor(comfy_stubs):
    # The crux: tile 1 (row 0, col 1) keeps its LEFT overlap band and feathers into tile
    # 0's already-pasted core across canvas cols [24,40). alpha follows the directional
    # feather (feather_alpha): a solid plateau at the seam falling off to ~0 at the outer
    # edge (keep the neighbor). Rows [0,24) of that band are touched only by tile 1 (tiles
    # 2/3 paste from row 24 down), so the composite there is exactly tile 1 over tile 0 —
    # differing from a hard core paste, which would leave the neighbor untouched.
    image = torch.rand(1, 80, 80, 3)

    out, _guider, _vae, _noise = _run(image, cap_w=56, cap_h=56, overlap=16)

    layout = _layout(80, 80, 56, 56, overlap=16)
    t1 = layout.tiles[1]
    assert (t1.kept_left, t1.kept_top) == (True, False)
    assert (t1.paste_rect.x0, t1.paste_rect.x1, t1.paste_rect.y0, t1.paste_rect.y1) == (24, 80, 0, 40)

    after_t0 = _simulate_feather_canvas(image, layout, n_tiles=1)  # tile 0 = hard core paste
    # ctx=0 → inner == crop, so tile 1's diffused crop encodes entirely from the RAW source
    # (the anchor ring is empty); after_t0 only supplies a ring here, so it is unused.
    t1_decoded = _decode_tile(after_t0, image, layout, 1)          # tile 1's decoded crop (56x56)
    # The exact alpha the pipeline uses. The left band is routed onto the minimum-error path,
    # so it varies row by row and cannot be written as a single column; the displacement comes
    # from the production helper (its values are pinned in test_seam_routing.py) while the
    # composite arithmetic below stays hand-written.
    paste, crop = t1.paste_rect, t1.crop_rect
    sub = t1_decoded[:, paste.y0 - crop.y0:paste.y1 - crop.y0, paste.x0 - crop.x0:paste.x1 - crop.x0, :]
    region = after_t0[:, paste.y0:paste.y1, paste.x0:paste.x1, :]
    # Second mirror of the production composite (this test does not use
    # _simulate_feather_canvas), so it carries the DC match and the same subtract-then-route
    # order. With these fakes both tiles refine the shared band identically, so the offset is
    # exactly 0.0 and `sub` is unchanged by it; the call is here for parity, not for effect.
    dc_offset = sampling.seam_dc_offset(t1, sub, region)
    if dc_offset is not None:
        sub = sub - dc_offset
    disp_top, disp_left = sampling.seam_displacements(t1, sub, region)
    alpha = sampling.feather_alpha(paste, t1.core, layout.overlap, t1.kept_top, t1.kept_left,
                                   disp_top=disp_top, disp_left=disp_left)
    alpha_band = alpha[0:24, 0:16]                  # the part tiles 2/3 never touch

    neighbor_band = after_t0[:, 0:24, 24:40, :]     # tile 0's independent refinement (what hard paste keeps)
    tile_band = sub[:, 0:24, 0:16, :]               # tile 1's INDEPENDENT refinement of the same RAW pixels
    expected_band = alpha_band[None, ..., None] * tile_band + (1.0 - alpha_band)[None, ..., None] * neighbor_band

    # Feathered band matches the hand-computed blend exactly, and is NOT the hard paste.
    assert torch.equal(out[:, 0:24, 24:40, :], expected_band)
    assert not torch.equal(out[:, 0:24, 24:40, :], neighbor_band)
    # Ramp endpoints: near-solid at the seam, ~neighbor at the outer edge. Not exactly 1.0 and
    # 0.0 any more, because routing slides the transition along the band per row.
    assert float(alpha_band[:, 15].min()) > 0.9
    assert float(alpha_band[:, 0].max()) < 0.2
    # No assertion that the routed path MEANDERS here: the fake VAE's arange noise makes the
    # error surface uniform along this seam, so a constant path is the correct answer for it.
    # Meandering is pinned against a planted staircase in test_seam_routing.py.
    # Seam side: tile 1's core (cols 40..80) is 100% tile 1 — no blend past the seam.
    assert torch.equal(out[:, 0:24, 40:80, :], sub[:, 0:24, 16:56, :])


def test_routing_only_moves_pixels_inside_a_kept_band(comfy_stubs):
    # The safety property that makes minimum-error routing admissible: bending the feather can
    # only ever re-weight pixels INSIDE a kept overlap band (paste_rect \ core). Everywhere
    # else alpha is exactly 1.0, so the output is this tile's own refinement and can never be
    # a blend of raw canvas. Checked at ctx=0, where the frozen anchor ring is empty and every
    # tile's encode comes from the raw source, so a band change cannot propagate into a later
    # tile's decode and the comparison isolates the composite. (At ctx>0 a band change
    # legitimately re-conditions later tiles, which is the anchor doing its job.)
    image = torch.rand(1, 80, 80, 3)

    routed, _, _, _ = _run(image, cap_w=56, cap_h=56, overlap=16)
    layout = _layout(80, 80, 56, 56, overlap=16)
    straight = _simulate_feather_canvas(image, layout, displace=False)

    assert torch.isfinite(routed).all()
    assert not torch.equal(routed, straight)      # the routed cut genuinely does something

    band_mask = torch.zeros(80, 80, dtype=torch.bool)
    for tile in layout.tiles:
        paste, core = tile.paste_rect, tile.core
        band_mask[paste.y0:paste.y1, paste.x0:paste.x1] = True
        band_mask[core.y0:core.y1, core.x0:core.x1] = False   # cores are never re-weighted
    assert bool(band_mask.any())

    differs = (routed != straight).any(dim=-1)[0]
    assert bool(differs.any())
    assert not bool((differs & ~band_mask).any())  # every difference lies in a kept band


def test_output_keeps_the_full_input_size(comfy_stubs):
    # Cheap end-to-end guard against a whole class of bug: a withdrawn seam experiment once
    # shadowed the local holding the canvas width, so the final crop_image_to() cropped the
    # output to 8px wide. Every unit test still passed, because they exercise the alpha
    # builders directly and never go through _refine_tiles. Shape catches it; nothing else did.
    for image, cap in ((torch.rand(1, 80, 80, 3), 56), (torch.rand(1, 96, 64, 3), 48)):
        out, _, _, _ = _run(image, cap_w=cap, cap_h=cap, overlap=16)
        assert out.shape == image.shape
        assert torch.isfinite(out).all()


def test_feather_second_tile_encodes_raw_source_at_ctx0(comfy_stubs):
    image = torch.rand(1, 80, 80, 3)

    _out, _guider, vae, _noise = _run(image, cap_w=56, cap_h=56, overlap=16)

    layout = _layout(80, 80, 56, 56, overlap=16)
    after_tile0 = _simulate_feather_canvas(image, layout, n_tiles=1)
    crop1 = layout.tiles[1].crop_rect  # x [24, 80), y [0, 56)
    # ctx=0 → no anchor ring → overlap_inner_rect == crop, so tile 1's ENTIRE encode comes
    # from the FROZEN raw source, NOT the live canvas: bit-identical to the raw crop.
    assert torch.equal(vae.encode_calls[1], image[:, crop1.y0:crop1.y1, crop1.x0:crop1.x1, :])
    # Tile 0 (top-left, hard core paste of (0,0,40,40)) overwrote tile 1's crop only in
    # canvas rows [0,40) x cols [24,40) (crop-local y [0,40), x [0,16)). The fix keeps that
    # band RAW in the encode — NOT tile 0's decoded output — so the feather later dissolves
    # two INDEPENDENT refinements of the raw band, never a serial re-diffusion.
    assert torch.equal(vae.encode_calls[1][:, :40, :16, :], image[:, 0:40, 24:40, :])
    assert not torch.equal(vae.encode_calls[1][:, :40, :16, :], after_tile0[:, 0:40, 24:40, :])


def test_feather_encode_splits_raw_inner_live_ring(comfy_stubs):
    # ctx>0 creates BOTH a real context_anchor ring and a top/left context_overlap band a
    # prior tile has already refined into the live canvas. The fix must encode the diffused
    # inner band (core+overlap == overlap_inner_rect) from the FROZEN raw source, while the
    # frozen anchor ring (crop \ inner) still comes from the LIVE canvas. Config: 80x80,
    # caps 72, ctx=16, overlap=16 -> r=32, 2x2 (base=40). Tile 3 (col1,row1) borders
    # already-processed neighbors on top and left and has a 16px top/left anchor ring.
    image = torch.rand(1, 80, 80, 3)

    _out, _guider, vae, _noise = _run(image, cap_w=72, cap_h=72, ctx=16, overlap=16)

    layout = _layout(80, 80, 72, 72, ctx=16, overlap=16)
    live_before_t3 = _simulate_feather_canvas(image, layout, n_tiles=3)
    t3 = layout.tiles[3]
    crop, inner = t3.crop_rect, t3.overlap_inner_rect
    assert (crop.x0, crop.y0, crop.x1, crop.y1) == (8, 8, 80, 80)
    assert (inner.x0, inner.y0, inner.x1, inner.y1) == (24, 24, 80, 80)
    enc = vae.encode_calls[3]
    iy0, iy1 = inner.y0 - crop.y0, inner.y1 - crop.y0   # crop-local inner rows [16, 72)
    ix0, ix1 = inner.x0 - crop.x0, inner.x1 - crop.x0   # crop-local inner cols [16, 72)

    # Diffused inner band == FROZEN raw source, and NOT the live composited canvas (whose
    # top/left overlap band carries tiles 0/1/2's decoded output). This is the fix.
    assert torch.equal(enc[:, iy0:iy1, ix0:ix1, :], image[:, inner.y0:inner.y1, inner.x0:inner.x1, :])
    assert not torch.equal(enc[:, iy0:iy1, ix0:ix1, :], live_before_t3[:, inner.y0:inner.y1, inner.x0:inner.x1, :])

    # Frozen context_anchor ring (crop \ inner: top rows [0,16) and left cols [0,16)) ==
    # the LIVE canvas — it carries the neighbors' already-refined pixels as conditioning,
    # and is therefore NOT the raw source there.
    live_crop = live_before_t3[:, crop.y0:crop.y1, crop.x0:crop.x1, :]
    raw_crop = image[:, crop.y0:crop.y1, crop.x0:crop.x1, :]
    assert torch.equal(enc[:, :iy0, :, :], live_crop[:, :iy0, :, :])   # top ring band
    assert torch.equal(enc[:, :, :ix0, :], live_crop[:, :, :ix0, :])   # left ring band
    assert not torch.equal(enc[:, :iy0, :, :], raw_crop[:, :iy0, :, :])


def test_anchor_only_overlap0_encode_is_noop(comfy_stubs):
    # The user's current config: context_anchor>0, context_overlap=0. overlap_inner_rect
    # == core, and no tile ever pastes into a LATER tile's core (paste_rect only extends
    # up/left into already-processed neighbors), so canvas[core] is still raw at encode
    # time. The source overwrite is therefore a byte-for-byte no-op: every recorded encode
    # equals the live canvas crop exactly — bit-identical to the pre-split behavior.
    # Config: 80x80, caps 56, ctx=16, overlap=0 -> r=16, 2x2 (base=40); every crop != core
    # (expanded, so the new branch runs) while inner == core.
    image = torch.rand(1, 80, 80, 3)

    _out, _guider, vae, _noise = _run(image, cap_w=56, cap_h=56, ctx=16, overlap=0)

    layout = _layout(80, 80, 56, 56, ctx=16, overlap=0)
    assert len(vae.encode_calls) == 4
    for i, tile in enumerate(layout.tiles):
        crop, core, inner = tile.crop_rect, tile.core, tile.overlap_inner_rect
        # expanded (new branch runs) but inner == core, so the overwrite is a no-op.
        assert (crop.x0, crop.y0, crop.x1, crop.y1) != (core.x0, core.y0, core.x1, core.y1)
        assert (inner.x0, inner.y0, inner.x1, inner.y1) == (core.x0, core.y0, core.x1, core.y1)
        canvas_before = _simulate_feather_canvas(image, layout, n_tiles=i)
        # Recorded encode == the LIVE canvas crop, with no raw substitution visible.
        assert torch.equal(vae.encode_calls[i], canvas_before[:, crop.y0:crop.y1, crop.x0:crop.x1, :])


def test_r0_multi_tile_never_touches_model_options(comfy_stubs):
    _out, guider, _vae, _noise = _run(torch.rand(1, 80, 80, 3))  # ctx=overlap=0 escape hatch

    assert guider.model_options == {}
    assert len(guider.calls) == 9
    for call in guider.calls:
        assert call["denoise_mask"] is None
        assert call["mask_fn"] is None


def test_grid_config_error_before_any_side_effects(comfy_stubs):
    guider, vae, noise = GridGuider(), GridVAE(), GridNoise()

    with pytest.raises(grid.GridConfigError, match="caps too small"):
        sampling.refine_image(torch.rand(1, 80, 80, 3), guider, object(), SIGMAS, vae, noise, max_tile_width=32, max_tile_height=32, context_anchor=64, context_overlap=64)

    assert vae.encode_calls == []
    assert noise.calls == []
    assert guider.calls == []


def test_feather_run_does_not_mutate_input(comfy_stubs):
    # pad_image_to_multiple returns the input itself for an aligned image; only the
    # canvas clone keeps the caller's tensor intact.
    image = torch.rand(1, 80, 80, 3)
    before = image.clone()

    _run(image, cap_w=56, cap_h=56, overlap=16)

    assert torch.equal(image, before)
