"""video.py: the tiled MiniMax H3 refine engine.

Everything runs on stubs — a duck-typed AV VAE, guider and NestedTensor — so no ComfyUI
install and no GPU are needed. The fixture canvas is 192x256 with caps 160x192 and
anchor/overlap 32, which solves to a 2x2 grid of 160x192 crops, every rect on the /32 grid
H3 requires. The stub VAE mirrors H3's layout: /16 spatially, the 17k+5 frame grid
temporally, 24 video latent channels, and core's 5-D [B,T,H,W,C] decode.
"""
import sys

import pytest
import torch

from context_anchored_tile_refine import grid, sampling, video, vl_video
from test_sampling import fake_model_patcher
from test_tiling import RecordingPreviewer

CANVAS_W, CANVAS_H = 192, 256
CAP_W, CAP_H = 160, 192
ANCHOR, OVERLAP = 32, 32
FRAMES = 5            # on the 17k+5 grid, so the fixture clip is never repeat-padded
LATENT_T = 2          # video_latent_t(5)
AUDIO_T = 8           # round(5 / 24 * 40)
SEED = 71
SAMPLER = object()
SIGMAS = torch.linspace(1.0, 0.0, 3)   # 2 steps

_SX = grid.solve_axis(CANVAS_W, CAP_W, ANCHOR, OVERLAP, multiple=32)
_SY = grid.solve_axis(CANVAS_H, CAP_H, ANCHOR, OVERLAP, multiple=32)
LAYOUT = grid.build_layout(CANVAS_W, CANVAS_H, _SX, _SY, ANCHOR, OVERLAP)
TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT = LAYOUT.tiles


# --- stub kit -----------------------------------------------------------------------

def _nested(video_tensor, audio_tensor):
    return sys.modules["comfy.nested_tensor"].NestedTensor((video_tensor, audio_tensor))


class VideoVAE:
    """Position-preserving stub in H3's latent layout.

    encode: [T,H,W,3] -> [1,24,video_latent_t(T),H/16,W/16], carrying a stride-16 spatial
    sample of the first latent_t frames in the first three latent channels, so a tile taken
    from the wrong place changes values. decode inverts that and repeats the latent frames
    back out to the clip length, returning core's 5-D [B,T,H,W,C].
    """

    latent_channels = video.VIDEO_LATENT_CHANNELS

    def __init__(self):
        self.encode_calls = []
        self.decode_calls = 0
        self.frame_count = None

    def encode(self, pixels):
        # Snapshot: refine_video hands over a buffer built from its live canvas, which keeps
        # changing after this call.
        self.encode_calls.append(pixels.clone())
        self.frame_count = int(pixels.shape[0])
        latent_t = video.video_latent_t(self.frame_count)
        cells = pixels[:latent_t, ::16, ::16, :].float()
        samples = torch.zeros(1, video.VIDEO_LATENT_CHANNELS, latent_t,
                              pixels.shape[1] // 16, pixels.shape[2] // 16)
        samples[0, :3] = cells.permute(3, 0, 1, 2)
        return samples

    def decode(self, samples):
        self.decode_calls += 1
        latent_t = samples.shape[2]
        cells = samples[0, :3].permute(1, 2, 3, 0)
        frames = cells.repeat_interleave(16, dim=1).repeat_interleave(16, dim=2)
        repeats = -(-self.frame_count // latent_t)
        return frames.repeat_interleave(repeats, dim=0)[:self.frame_count][None]


class VideoGuider:
    """Records every sample() call and returns (refined video, the audio it was handed).

    Returning the audio stream untouched is deliberate: the frozen nested mask is what
    guarantees it, so a stub that regenerated it would hide a mask regression. With `fill`
    the video stream comes back as a per-tile CONSTANT, which makes the composite decidable
    tile by tile; without it, latent + noise.
    """

    def __init__(self, fill=None, base_positive=("base-positive",)):
        self.model_patcher = fake_model_patcher()
        self.model_options = {}
        self.original_conds = {"positive": list(base_positive), "negative": ["base-negative"]}
        self.pristine_conds = self.original_conds
        self.fill = fill
        self.calls = []

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        video_latent, audio_latent = latent_image.unbind()
        video_noise, audio_noise = noise.unbind()
        self.calls.append({
            "video_latent": video_latent,
            "audio_latent": audio_latent,
            "video_noise": video_noise,
            "audio_noise": audio_noise,
            "denoise_mask": denoise_mask,
            "sampler": sampler,
            "sigmas": sigmas,
            "seed": seed,
            "disable_pbar": disable_pbar,
            "conds": self.original_conds,
            "positive": self.original_conds["positive"],
        })
        steps = sigmas.shape[-1] - 1
        for step in range(steps):
            callback(step, latent_image, latent_image, steps)
        if self.fill is None:
            refined = video_latent + video_noise
        else:
            refined = torch.full_like(video_latent, self.fill + len(self.calls))
        return _nested(refined, audio_latent)


class RaisingGuider(VideoGuider):
    def sample(self, *args, **kwargs):
        super().sample(*args, **kwargs)
        raise RuntimeError("sampler exploded")


@pytest.fixture
def video_stubs(comfy_stubs, monkeypatch):
    # arange noise per stream: every latent cell gets a unique value, so a slice-position
    # error is detectable exactly. The shared stub draws zeros, which would hide one.
    def prepare_noise(latent_image, seed, batch_inds=None):
        comfy_stubs["prepare_noise_calls"].append((latent_image, seed, batch_inds))
        streams = [torch.arange(t.numel(), dtype=torch.float32).reshape(t.shape)
                   for t in latent_image.unbind()]
        return _nested(*streams)

    monkeypatch.setattr(sys.modules["comfy.sample"], "prepare_noise", prepare_noise)
    return comfy_stubs


def _frames(count=FRAMES, height=CANVAS_H, width=CANVAS_W):
    # Values inside [0,1), so the stub guider's fill (>= 1) is always distinguishable from
    # raw canvas content.
    total = count * height * width * 3
    return (torch.arange(total, dtype=torch.float32) / total).reshape(count, height, width, 3)


def _vl_pack(height=CANVAS_H, width=CANVAS_W):
    # One text row then one vision block covering the canvas, the shape vl_video.token_layout
    # walks out of a real H3 stream. Feature value == row index, so a slice is readable.
    rows = 1 + (height // 32) * (width // 32)
    cond = torch.arange(rows, dtype=torch.float32).reshape(1, rows, 1).expand(1, rows, 4).clone()
    return vl_video.GlobalEncode(cond, torch.arange(rows), [("text", 0), ("grid", 1)],
                                 height, width, {})


def _run(frames, guider=None, vae=None, audio_latent=None, sigmas=SIGMAS, pack=None):
    guider = VideoGuider(fill=1.0) if guider is None else guider
    vae = VideoVAE() if vae is None else vae
    pack = _vl_pack() if pack is None else pack
    out = video.refine_video(frames, guider, SAMPLER, sigmas, vae, SEED, CAP_W, CAP_H,
                             ANCHOR, OVERLAP, pack, audio_latent=audio_latent)
    return out, guider, vae


# --- fixture geometry ----------------------------------------------------------------

def test_fixture_layout_is_a_2x2_grid_of_32_aligned_crops():
    assert (_SX.n, _SY.n) == (2, 2)
    for tile in LAYOUT.tiles:
        crop = tile.crop_rect
        assert (crop.x1 - crop.x0, crop.y1 - crop.y0) == (CAP_W, CAP_H)
        assert all(value % 32 == 0 for value in (crop.x0, crop.y0, crop.x1, crop.y1))
    assert (TOP_LEFT.kept_top, TOP_LEFT.kept_left) == (False, False)
    assert (BOTTOM_RIGHT.kept_top, BOTTOM_RIGHT.kept_left) == (True, True)


# --- frame grid ----------------------------------------------------------------------

def test_frame_counts_snap_to_the_17k_plus_5_grid():
    # comfy_extras/nodes_minimax_h3.py align_frame_count / video_latent_t / temporal_shape.
    assert [video.align_frame_count(n) for n in (1, 5, 6, 22, 23, 124)] == [5, 5, 22, 22, 39, 124]
    assert [video.video_latent_t(n) for n in (5, 22, 124)] == [2, 7, 37]
    assert [video.audio_latent_length(n) for n in (5, 22, 124)] == [8, 37, 207]


def test_on_grid_clip_is_not_padded():
    frames = _frames()
    padded, count = video.pad_frames_to_grid(frames)
    assert count == FRAMES
    assert padded is frames


def test_off_grid_clip_is_repeat_padded_with_its_last_frame():
    frames = _frames(count=6)
    padded, count = video.pad_frames_to_grid(frames)

    assert count == 22
    assert padded.shape == (22, CANVAS_H, CANVAS_W, 3)
    assert torch.equal(padded[:6], frames)
    for index in range(6, 22):
        assert torch.equal(padded[index], frames[5])


def test_off_grid_clip_is_sampled_on_the_grid_and_trimmed_back(video_stubs):
    frames = _frames(count=6)

    out, guider, vae = _run(frames)

    # Every tile sampled 22 frames (7 latent frames), and the output is the caller's 6.
    assert vae.encode_calls[0].shape[0] == 22
    assert guider.calls[0]["video_latent"].shape[2] == video.video_latent_t(22)
    assert guider.calls[0]["audio_latent"].shape == (1, 32, 2, 37)
    assert out.shape == (6, CANVAS_H, CANVAS_W, 3)


# --- output contract -----------------------------------------------------------------

def test_output_is_float32_at_the_input_frame_count_and_size(video_stubs):
    frames = _frames()

    out, guider, vae = _run(frames)

    assert out.shape == frames.shape
    assert out.dtype == torch.float32
    assert len(guider.calls) == 4 and vae.decode_calls == 4


def test_unaligned_canvas_is_padded_to_32_and_cropped_back(video_stubs):
    # 224x176 pads to 224x192; the output must come back at the caller's size.
    frames = _frames(height=224, width=176)

    out, guider, vae = _run(frames)

    assert vae.encode_calls[0].shape[2] % 32 == 0
    assert out.shape == (FRAMES, 224, 176, 3)


def test_a_cap_fitting_clip_runs_as_one_untiled_tile(video_stubs):
    # n=1 on both axes: no anchor ring, no neighbour, so the mask releases the whole latent
    # and the composite is a plain paste — the video path's analogue of the whole-image case.
    frames = _frames()
    guider, vae = VideoGuider(fill=1.0), VideoVAE()

    out = video.refine_video(frames, guider, SAMPLER, SIGMAS, vae, SEED, 256, 256,
                             ANCHOR, OVERLAP, _vl_pack())

    assert len(guider.calls) == 1 and vae.decode_calls == 1
    video_mask = guider.calls[0]["denoise_mask"].unbind()[0]
    assert torch.equal(video_mask, torch.ones_like(video_mask))
    assert out.shape == frames.shape and float(out.min()) == 2.0


def test_denoise_zero_returns_the_clip_untouched(video_stubs):
    frames = _frames()

    out, guider, vae = _run(frames, sigmas=torch.FloatTensor([]))

    assert torch.equal(out, frames)
    assert out is not frames
    assert vae.encode_calls == [] and guider.calls == []


# --- canvas discipline ---------------------------------------------------------------

def test_every_canvas_pixel_is_painted_and_a_hard_core_owns_it(video_stubs):
    frames = _frames()

    out, guider, vae = _run(frames)

    # Zero unpainted pixels: the cores tile the canvas exactly, and every raw value is < 1.0
    # while every tile fill is >= 2.0.
    assert float(out.min()) > 1.0
    # In raster order a later tile's overlap band feathers over its neighbour's core edge, so
    # the pixels that come out at alpha 1 are the core pixels no later paste_rect reaches.
    # Those must carry exactly their own tile's fill (shifted at most by the DC clamp).
    last_writer = torch.full((CANVAS_H, CANVAS_W), -1, dtype=torch.long)
    for index, tile in enumerate(LAYOUT.tiles):
        paste = tile.paste_rect
        last_writer[paste.y0:paste.y1, paste.x0:paste.x1] = index
    assert bool((last_writer >= 0).all())
    for index, tile in enumerate(LAYOUT.tiles):
        core = tile.core
        owned = torch.zeros(CANVAS_H, CANVAS_W, dtype=torch.bool)
        owned[core.y0:core.y1, core.x0:core.x1] = True
        owned &= last_writer == index
        assert bool(owned.any()), index
        values = out[:, owned, :]
        assert float(values.max() - values.min()) == 0.0, index
        assert abs(float(values.reshape(-1)[0]) - (2.0 + index)) <= sampling.DC_MATCH_CLAMP, index


def test_bands_encode_from_raw_while_the_anchor_ring_reads_the_live_canvas(video_stubs):
    frames = _frames()

    out, guider, vae = _run(frames)

    raw = frames.to(torch.float16)
    crop = TOP_RIGHT.crop_rect              # (32,0)-(192,192); anchor ring = cols 32..63
    enc = vae.encode_calls[1]
    # Rows the top-left tile already refined: the ring carries its output (fill >= 2), never
    # raw canvas — that is the context anchoring the whole seam design rests on.
    assert torch.all(enc[:, :TOP_LEFT.core.y1, :ANCHOR, :] > 1.0)
    # Below that tile nothing has been pasted, so the ring is still the raw canvas.
    assert torch.equal(enc[:, TOP_LEFT.core.y1:, :ANCHOR, :],
                       raw[:, TOP_LEFT.core.y1:crop.y1, crop.x0:crop.x0 + ANCHOR, :])
    # Everything from overlap_inner_rect inward is the FROZEN RAW source — including the
    # columns the top-left tile already pasted into (canvas x 64..96), which is the point:
    # the shared band is diffused twice from raw, never re-diffused from a neighbour.
    inner = TOP_RIGHT.overlap_inner_rect
    assert torch.equal(enc[:, :, inner.x0 - crop.x0:, :], raw[:, crop.y0:crop.y1, inner.x0:crop.x1, :])


def test_the_first_tile_encodes_the_raw_canvas(video_stubs):
    frames = _frames()

    out, guider, vae = _run(frames)

    crop = TOP_LEFT.crop_rect
    assert torch.equal(vae.encode_calls[0],
                       frames.to(torch.float16)[:, crop.y0:crop.y1, crop.x0:crop.x1, :])


def test_canvases_are_fp16_and_the_input_is_never_mutated(video_stubs):
    frames = _frames()
    original = frames.clone()

    out, guider, vae = _run(frames)

    assert vae.encode_calls[0].dtype == torch.float16
    assert torch.equal(frames, original)


# --- noise -----------------------------------------------------------------------------

def test_noise_is_drawn_once_for_the_whole_canvas(video_stubs):
    frames = _frames()

    _run(frames)

    assert len(video_stubs["prepare_noise_calls"]) == 1
    dummy, seed, batch_inds = video_stubs["prepare_noise_calls"][0]
    assert seed == SEED and batch_inds is None
    canvas_video, canvas_audio = dummy.unbind()
    assert canvas_video.shape == (1, 24, LATENT_T, CANVAS_H // 16, CANVAS_W // 16)
    assert canvas_audio.shape == (1, 32, 2, AUDIO_T)
    assert torch.equal(canvas_video, torch.zeros_like(canvas_video))


def test_each_tile_gets_its_canvas_noise_slice(video_stubs):
    frames = _frames()

    out, guider, vae = _run(frames)

    canvas_noise = torch.arange(24 * LATENT_T * 16 * 12, dtype=torch.float32).reshape(
        1, 24, LATENT_T, CANVAS_H // 16, CANVAS_W // 16)
    for tile, call in zip(LAYOUT.tiles, guider.calls):
        crop = tile.crop_rect
        expected = canvas_noise[..., crop.y0 // 16:crop.y1 // 16, crop.x0 // 16:crop.x1 // 16]
        assert torch.equal(call["video_noise"], expected)
        assert call["video_noise"].is_contiguous()


def test_the_audio_noise_is_the_same_draw_for_every_tile(video_stubs):
    frames = _frames()

    out, guider, vae = _run(frames)

    expected = torch.arange(32 * 2 * AUDIO_T, dtype=torch.float32).reshape(1, 32, 2, AUDIO_T)
    for call in guider.calls:
        assert torch.equal(call["audio_noise"], expected)
    # Cloned per tile: one shared buffer would let an in-place sampler write leak forward.
    assert guider.calls[0]["audio_noise"] is not guider.calls[1]["audio_noise"]


def test_a_latent_that_disagrees_with_its_noise_slice_fails_fast(video_stubs):
    class WrongLayoutVAE(VideoVAE):
        def encode(self, pixels):
            samples = super().encode(pixels)
            return samples[:, :16]      # 16 channels: not H3's video latent

    with pytest.raises(RuntimeError, match="noise slice"):
        _run(_frames(), vae=WrongLayoutVAE())


# --- denoise mask ------------------------------------------------------------------------

def test_video_mask_is_the_binary_anchor_ring_over_every_latent_frame(video_stubs):
    frames = _frames()

    out, guider, vae = _run(frames)

    for tile, call in zip(LAYOUT.tiles, guider.calls):
        video_mask, audio_mask = call["denoise_mask"].unbind()
        gradient = sampling.tile_gradient(tile.crop_rect, tile.overlap_inner_rect, scale=16)
        assert video_mask.shape == (1, 1, LATENT_T, CAP_H // 16, CAP_W // 16)
        assert set(video_mask.unique().tolist()) <= {0.0, 1.0}
        for frame in range(LATENT_T):
            assert torch.equal(video_mask[0, 0, frame], gradient)


def test_the_top_left_tile_freezes_only_its_right_and_bottom_anchor_rings(video_stubs):
    # Hand-computed: crop (0,0)-(160,192), overlap_inner_rect (0,0)-(128,160) -> the diffused
    # block is latent cells x 0..7, y 0..9; the two anchor rings are 2 cells (32px) wide.
    frames = _frames()

    out, guider, vae = _run(frames)

    video_mask = guider.calls[0]["denoise_mask"].unbind()[0][0, 0, 0]
    expected = torch.zeros(CAP_H // 16, CAP_W // 16)
    expected[:10, :8] = 1.0
    assert torch.equal(video_mask, expected)


def test_the_audio_stream_is_frozen_for_every_tile(video_stubs):
    # A stream with no mask defaults to ones (comfy/samplers.py:1304-1305), which would
    # regenerate the audio once per tile.
    frames = _frames()

    out, guider, vae = _run(frames)

    for call in guider.calls:
        audio_mask = call["denoise_mask"].unbind()[1]
        assert audio_mask.shape == (1, 32, 2, AUDIO_T)
        assert torch.equal(audio_mask, torch.zeros_like(audio_mask))


# --- audio stream ------------------------------------------------------------------------

def test_a_connected_audio_latent_rides_along_unchanged(video_stubs):
    audio = torch.full((1, 32, 2, AUDIO_T), 0.25)

    out, guider, vae = _run(_frames(), audio_latent=audio)

    for call in guider.calls:
        assert torch.equal(call["audio_latent"], audio)
        # Cloned, so a sampler writing in place cannot reach the caller's AV latent.
        assert call["audio_latent"] is not audio


def test_an_unconnected_audio_latent_falls_back_to_silence(video_stubs):
    out, guider, vae = _run(_frames())

    audio = guider.calls[0]["audio_latent"]
    assert audio.shape == (1, 32, 2, AUDIO_T)
    assert torch.equal(audio, torch.zeros_like(audio))


def test_the_silent_audio_placeholder_is_built_where_core_puts_an_av_latent(video_stubs, monkeypatch):
    # A real AV latent (comfy_extras/nodes_minimax_h3.py _empty_av_latent) and vae.encode
    # (comfy/sd.py:1027) both land on intermediate_device(), which is the GPU under
    # --gpu-only (comfy/model_management.py:1227-1231). 'meta' stands in for that off-CPU
    # device, which no test machine is required to have.
    monkeypatch.setattr(sys.modules["comfy.model_management"], "intermediate_device",
                        lambda: torch.device("meta"))

    audio = video.audio_stream(None, FRAMES)

    assert audio.device.type == "meta"
    assert audio.shape == (1, 32, 2, AUDIO_T)


def test_the_nested_latent_streams_share_the_encodes_device(video_stubs):
    # guider.sample cats the two streams (comfy/samplers.py:1281-1283 -> comfy/utils.py:1374
    # pack_latents) BEFORE outer_sample moves them onto the load device
    # (comfy/samplers.py:1254-1255), so a pair straddling two devices is a hard crash on the
    # first tile, not a slow path. The guider raises so nothing downstream touches the
    # dataless meta decode.
    class OffDeviceVAE(VideoVAE):
        def encode(self, pixels):
            return super().encode(pixels).to("meta")

    guider = RaisingGuider(fill=1.0)

    with pytest.raises(RuntimeError, match="sampler exploded"):
        _run(_frames(), guider=guider, vae=OffDeviceVAE())

    call = guider.calls[0]
    assert call["video_latent"].device.type == "meta"
    assert call["audio_latent"].device == call["video_latent"].device


def test_an_audio_latent_of_the_wrong_length_is_rejected(video_stubs):
    audio = torch.zeros(1, 32, 2, AUDIO_T + 3)

    with pytest.raises(ValueError, match="audio length 11 does not match .* 5 frames need 8"):
        _run(_frames(), audio_latent=audio)


def test_the_audio_length_is_checked_against_the_PADDED_frame_count(video_stubs):
    # 6 frames pad to 22, so the latent the H3 sampler made for 6 frames is the wrong one.
    audio = torch.zeros(1, 32, 2, video.audio_latent_length(22))

    out, guider, vae = _run(_frames(count=6), audio_latent=audio)

    assert guider.calls[0]["audio_latent"].shape[-1] == 37
    with pytest.raises(ValueError, match="22 frames need 37"):
        _run(_frames(count=6), audio_latent=torch.zeros(1, 32, 2, 8))


# --- conditioning ------------------------------------------------------------------------

def test_each_tile_samples_with_its_own_vl_slice(video_stubs):
    pack = _vl_pack()

    out, guider, vae = _run(_frames(), pack=pack)

    for tile, call in zip(LAYOUT.tiles, guider.calls):
        expected = vl_video.slice_rows(pack, tile.crop_rect, CANVAS_H, CANVAS_W)
        assert call["positive"][0]["cross_attn"].shape == expected[0]["cross_attn"].shape
        assert torch.equal(call["positive"][0]["cross_attn"], expected[0]["cross_attn"])
        assert torch.equal(call["positive"][0]["minimax_token_tags"],
                           expected[0]["minimax_token_tags"])
    # Neighbouring tiles share the boundary cells and differ everywhere else.
    left = guider.calls[0]["positive"][0]["cross_attn"]
    right = guider.calls[1]["positive"][0]["cross_attn"]
    assert left.shape != right.shape or not torch.equal(left, right)


def test_the_pristine_conds_map_is_restored_after_every_tile(video_stubs):
    guider = VideoGuider(fill=1.0)
    pristine = guider.original_conds

    out, _, _ = _run(_frames(), guider=guider)

    assert guider.original_conds is pristine
    assert guider.original_conds["positive"] == ["base-positive"]
    # The swap was a fresh map each time, never a write into the caller's.
    for call in guider.calls:
        assert call["conds"] is not pristine
        assert call["conds"]["negative"] == ["base-negative"]


def test_the_pristine_conds_map_is_restored_when_the_sampler_raises(video_stubs):
    guider = RaisingGuider(fill=1.0)
    pristine = guider.original_conds

    with pytest.raises(RuntimeError, match="sampler exploded"):
        _run(_frames(), guider=guider)

    assert guider.original_conds is pristine
    assert guider.original_conds["positive"] == ["base-positive"]


# --- sampling plumbing --------------------------------------------------------------------

def test_interrupt_is_checked_once_per_tile(video_stubs):
    _run(_frames())

    assert video_stubs["interrupt_calls"] == 4


def test_progress_aggregates_across_tiles(video_stubs):
    out, guider, vae = _run(_frames())

    (pbar,) = video_stubs["progress_bars"]
    assert pbar.total == 8      # 2 steps * 4 tiles
    assert pbar.updates == [(tile * 2 + step + 1, 8, None)
                            for tile in range(4) for step in range(2)]
    for call in guider.calls:
        assert call["seed"] == SEED
        assert call["sigmas"] is SIGMAS
        assert call["sampler"] is SAMPLER
        assert call["disable_pbar"] is False


def test_tile_preview_unwraps_a_nested_x0(video_stubs):
    # Core hands a nested-latent callback a NestedTensor (comfy/samplers.py:1289-1295) and
    # H3 has a previewer, so without the guard every tile preview would crash.
    previewer = RecordingPreviewer()
    video_stubs["previewer"] = previewer
    for_tile = sampling.make_tile_progress(fake_model_patcher(), 2, 1)
    video_latent = torch.zeros(1, 24, LATENT_T, 4, 3)
    nested_x0 = _nested(video_latent, torch.zeros(1, 32, 2, AUDIO_T))

    for_tile(0)(0, nested_x0, nested_x0, 2)

    assert previewer.calls[0][1] is video_latent


def test_an_image_x0_still_reaches_the_previewer_unchanged(video_stubs):
    # The guard must be a no-op for the image path, which is pinned byte-for-byte.
    previewer = RecordingPreviewer()
    video_stubs["previewer"] = previewer
    for_tile = sampling.make_tile_progress(fake_model_patcher(), 2, 1)
    x0 = torch.zeros(1, 4, 8, 8)

    for_tile(0)(0, x0, x0, 2)

    assert previewer.calls[0][1] is x0


# --- fp16 seam math -----------------------------------------------------------------------
# The composite reduces fp16 bands over the FRAME axis. These pin that path numerically
# before any owner A/B: a known DC offset and a known minimum-error cut must come back.

DC_BAND_FRAMES = 8


def _fp16_band_pair(tile, offset, frames=DC_BAND_FRAMES):
    """(sub, region) over the tile's paste_rect: fp16 content plus a per-channel offset.

    Content values are multiples of 1/64 in [0.25, 0.5) and the offsets are dyadic, so every
    value and every difference is EXACT in fp16 — what is being pinned is the reduction, not
    float noise.
    """
    paste = tile.paste_rect
    shape = (frames, paste.y1 - paste.y0, paste.x1 - paste.x0, 3)
    pattern = (torch.arange(int(torch.tensor(shape).prod()), dtype=torch.float32) % 16) / 64.0
    region = (0.25 + pattern.reshape(shape)).to(torch.float16)
    return (region + torch.tensor(offset, dtype=torch.float16)), region


def test_seam_dc_offset_recovers_a_known_offset_from_fp16_frames():
    offset = [2.0 ** -7, -(2.0 ** -8), 3.0 * 2.0 ** -8]
    sub, region = _fp16_band_pair(BOTTOM_RIGHT, offset)

    measured = sampling.seam_dc_offset(BOTTOM_RIGHT, sub, region)

    assert measured.dtype == torch.float16
    assert measured.tolist() == offset


def test_a_non_dyadic_fp16_offset_is_recovered_to_well_inside_one_8_bit_step():
    # The real-world case: fp16 quantizes both operands, so the estimate lands on a
    # quantization level rather than the exact offset. It must still be far finer than the
    # ~1.3/255 step the DC match exists to remove.
    offset = [0.005, -0.003, 0.011]
    sub, region = _fp16_band_pair(BOTTOM_RIGHT, offset)

    measured = sampling.seam_dc_offset(BOTTOM_RIGHT, sub, region).float()

    assert torch.allclose(measured, torch.tensor(offset), atol=0.25 / 255.0)


def test_the_dc_estimate_survives_frame_subsampling(video_stubs):
    # video.py measures the offset on every DC_FRAME_STRIDE-th frame; the offset is a global
    # statistic, so the subsample must return the same value the full pool does.
    offset = [2.0 ** -7, -(2.0 ** -8), 3.0 * 2.0 ** -8]
    sub, region = _fp16_band_pair(BOTTOM_RIGHT, offset, frames=16)

    full = sampling.seam_dc_offset(BOTTOM_RIGHT, sub, region)
    strided = sampling.seam_dc_offset(BOTTOM_RIGHT, sub[::video.DC_FRAME_STRIDE],
                                      region[::video.DC_FRAME_STRIDE])

    assert torch.equal(full, strided)


def test_the_dc_correction_is_applied_in_place_under_the_image_paths_gamut_rule():
    # video.py corrects the tile's own decode IN PLACE and in frame chunks (a corrected copy
    # is a second ~0.9 GB allocation at clip scale). The values must match
    # sampling._refine_tiles' formula exactly, across a chunk boundary.
    count = video.BLEND_FRAME_CHUNK + 3
    sub = torch.tensor([0.0, 0.5, 1.0, 1.25, -0.25]).view(1, 1, 5, 1).repeat(count, 1, 1, 3).contiguous()
    original = sub.clone()
    offset = torch.tensor([0.01, 0.01, 0.01])

    video.apply_dc_offset(sub, offset)

    corrected = original - offset
    in_gamut = (original >= 0.0) & (original <= 1.0)
    assert torch.equal(sub, torch.where(in_gamut, corrected.clamp(0.0, 1.0), corrected))
    # The rail is held for a pixel that was in gamut: 0.0 - 0.01 would leave the VAE's
    # [0,1] output contract...
    assert float(sub[0, 0, 0, 0]) == 0.0
    # ...while a pixel from an identity process_output is shifted, never flattened.
    assert float(sub[-1, 0, 3, 0]) == pytest.approx(1.24)


def _u50():
    # Where the undisplaced feather crosses alpha 0.5 (sampling._path_displacement).
    return (0.5 ** (1.0 / sampling.FEATHER_FALLOFF)) * (1.0 - sampling.FEATHER_PLATEAU)


def test_the_top_seam_routes_fp16_frames_onto_the_agreeing_row():
    tile = BOTTOM_LEFT                      # kept_top only, so the surface is unambiguous
    band = tile.core.y0 - tile.paste_rect.y0
    sub, region = _fp16_band_pair(tile, [0.0, 0.0, 0.0])
    sub = sub.clone()
    sub[:, :band] += 2.0 ** -6              # the two tiles disagree across the band...
    sub[:, 7] = region[:, 7]                # ...except on row 7, where the cut belongs

    disp_top, disp_left = sampling.seam_displacements(tile, sub, region)

    assert disp_left is None
    expected = _u50() - (7 + 0.5) / band
    assert disp_top.shape == (tile.paste_rect.x1 - tile.paste_rect.x0,)
    assert torch.allclose(disp_top, torch.full_like(disp_top, expected))


def test_the_left_seam_routes_fp16_frames_onto_the_agreeing_column():
    tile = TOP_RIGHT                        # kept_left only
    band = tile.core.x0 - tile.paste_rect.x0
    sub, region = _fp16_band_pair(tile, [0.0, 0.0, 0.0])
    sub = sub.clone()
    sub[:, :, :band] += 2.0 ** -6
    sub[:, :, 21] = region[:, :, 21]

    disp_top, disp_left = sampling.seam_displacements(tile, sub, region)

    assert disp_top is None
    expected = _u50() - (21 + 0.5) / band
    assert disp_left.shape == (tile.paste_rect.y1 - tile.paste_rect.y0,)
    assert torch.allclose(disp_left, torch.full_like(disp_left, expected))
