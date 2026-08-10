from types import SimpleNamespace

import pytest
import torch

from context_anchored_tile_refine import sampling

SIGMAS = torch.linspace(1.0, 0.0, 5)  # 4 steps


def fake_model_patcher():
    # make_tile_progress reads .load_device and .model.latent_format even when the
    # stubbed get_previewer returns None.
    return SimpleNamespace(load_device=torch.device("cpu"), model=SimpleNamespace(latent_format=object()))


class FakeNoise:
    def __init__(self, seed=7):
        self.seed = seed
        self.calls = []

    def generate_noise(self, input_latent):
        self.calls.append(input_latent)
        return torch.zeros_like(input_latent["samples"])


class FakeGuider:
    def __init__(self):
        self.model_patcher = fake_model_patcher()
        self.model_options = {}
        self.sample_calls = 0
        self.calls = []
        self.call = None

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        self.sample_calls += 1
        self.call = {
            "noise": noise,
            "latent_image": latent_image,
            "sampler": sampler,
            "sigmas": sigmas,
            "denoise_mask": denoise_mask,
            "callback": callback,
            "disable_pbar": disable_pbar,
            "seed": seed,
        }
        self.calls.append(self.call)
        steps = sigmas.shape[-1] - 1
        for step in range(steps):
            callback(step, latent_image, latent_image, steps)
        return latent_image + noise + 0.5


class FakeVAE:
    latent_channels = 4

    def __init__(self):
        self.encoded = None
        self.encode_calls = 0
        self.decode_calls = 0

    def encode(self, pixels):
        self.encode_calls += 1
        batch, height, width, channels = pixels.shape
        assert height % 8 == 0 and width % 8 == 0, "encode received non-/8 dims"
        assert channels == 3, f"encode received {channels} channels"
        self.encoded = torch.zeros(batch, 4, height // 8, width // 8)
        return self.encoded

    def decode(self, samples):
        self.decode_calls += 1
        batch, _, height, width = samples.shape
        return torch.full((batch, height * 8, width * 8, 3), 0.25)


@pytest.mark.parametrize(
    "height,width,expected",
    [
        (96, 104, (96, 104)),
        (100, 101, (104, 104)),
        (8, 8, (8, 8)),
        (9, 15, (16, 16)),
    ],
)
def test_pad_image_to_multiple_shapes(height, width, expected):
    image = torch.rand(2, height, width, 3)

    padded, (orig_h, orig_w) = sampling.pad_image_to_multiple(image)

    assert (orig_h, orig_w) == (height, width)
    assert padded.shape == (2, expected[0], expected[1], 3)


def test_pad_aligned_is_no_copy():
    image = torch.rand(1, 96, 104, 3)

    padded, _ = sampling.pad_image_to_multiple(image)

    assert padded is image


def test_pad_reflect_values():
    image = torch.arange(9 * 15, dtype=torch.float32).reshape(1, 9, 15, 1)

    padded, _ = sampling.pad_image_to_multiple(image)

    assert padded.shape == (1, 16, 16, 1)
    # Reflect excludes the edge pivot: first padded column mirrors column W-2,
    # first padded row mirrors row H-2.
    assert torch.equal(padded[0, :9, 15, 0], image[0, :, 13, 0])
    assert torch.equal(padded[0, 9, :15, 0], image[0, 7, :, 0])


def test_crop_image_to_returns_view_of_exact_size():
    image = torch.rand(2, 104, 104, 3)

    cropped = sampling.crop_image_to(image, 100, 101)

    assert cropped.shape == (2, 100, 101, 3)
    assert cropped.data_ptr() == image.data_ptr()  # view, no copy


def test_refine_image_non_multiple_roundtrip(comfy_stubs):
    guider, sampler, vae, noise = FakeGuider(), object(), FakeVAE(), FakeNoise()
    image = torch.rand(1, 100, 101, 3)

    out = sampling.refine_image(image, guider, sampler, SIGMAS, vae, noise, max_tile_width=1024, max_tile_height=1024, context_anchor=0, context_overlap=0)

    assert out.shape == (1, 100, 101, 3)
    assert vae.encoded.shape == (1, 4, 13, 13)  # encode saw the padded 104x104


@pytest.mark.parametrize("sigmas", [torch.empty(0), torch.tensor([1.0])])
def test_empty_schedule_returns_clone_without_vae_or_sampling(sigmas):
    guider, sampler, vae, noise = FakeGuider(), object(), FakeVAE(), FakeNoise()
    image = torch.rand(1, 96, 104, 3)

    out = sampling.refine_image(image, guider, sampler, sigmas, vae, noise, max_tile_width=1024, max_tile_height=1024, context_anchor=0, context_overlap=0)

    assert torch.equal(out, image)
    assert out is not image
    assert vae.encode_calls == 0 and vae.decode_calls == 0
    assert guider.sample_calls == 0 and noise.calls == []


def test_empty_schedule_narrows_an_rgba_input_to_three_channels():
    # The output channel count must not depend on a widget value: every sampled path drops
    # alpha, so the zero-step short-circuit does too (denoise 0 is an advertised mode).
    guider, sampler, vae, noise = FakeGuider(), object(), FakeVAE(), FakeNoise()
    image = torch.rand(1, 96, 104, 4)

    out = sampling.refine_image(image, guider, sampler, torch.empty(0), vae, noise, max_tile_width=1024, max_tile_height=1024, context_anchor=0, context_overlap=0)

    assert out.shape == (1, 96, 104, 3)
    assert torch.equal(out, image[..., :3])
    assert vae.encode_calls == 0 and guider.sample_calls == 0


def test_batch_dimension_preserved(comfy_stubs):
    guider, sampler, vae, noise = FakeGuider(), object(), FakeVAE(), FakeNoise()
    image = torch.rand(3, 96, 104, 3)

    out = sampling.refine_image(image, guider, sampler, SIGMAS, vae, noise, max_tile_width=1024, max_tile_height=1024, context_anchor=0, context_overlap=0)

    assert out.shape == (3, 96, 104, 3)
    assert vae.encoded.shape[0] == 3
    assert len(noise.calls) == 1  # one whole-batch generate_noise call
    assert guider.sample_calls == 1  # one whole-batch guider.sample call


def test_rgba_input_encodes_three_channels(comfy_stubs):
    guider, sampler, vae, noise = FakeGuider(), object(), FakeVAE(), FakeNoise()
    image = torch.rand(1, 96, 104, 4)

    out = sampling.refine_image(image, guider, sampler, SIGMAS, vae, noise, max_tile_width=1024, max_tile_height=1024, context_anchor=0, context_overlap=0)

    # FakeVAE.encode asserts it received exactly 3 channels; output is RGB.
    assert out.shape == (1, 96, 104, 3)


def test_sampling_call_contract(comfy_stubs):
    guider, sampler, vae, noise = FakeGuider(), object(), FakeVAE(), FakeNoise(seed=99)
    image = torch.rand(2, 96, 104, 3)

    out = sampling.refine_image(image, guider, sampler, SIGMAS, vae, noise, max_tile_width=1024, max_tile_height=1024, context_anchor=0, context_overlap=0)

    # SPEC CHANGE (grid-tiling): generate_noise now receives an all-zeros dummy sized
    # like the whole-canvas latent, not the encoded latent itself.
    assert len(noise.calls) == 1
    assert set(noise.calls[0].keys()) == {"samples"}
    dummy = noise.calls[0]["samples"]
    assert dummy.shape == vae.encoded.shape
    assert dummy.dtype == torch.float32
    assert torch.equal(dummy, torch.zeros_like(dummy))

    call = guider.call
    assert guider.sample_calls == 1  # caps 1024 on a 96x104 image: a 1x1 grid
    assert call["noise"].shape == vae.encoded.shape
    assert call["latent_image"] is vae.encoded
    assert call["sampler"] is sampler
    assert call["sigmas"] is SIGMAS
    assert call["denoise_mask"] is None
    assert call["disable_pbar"] is False
    assert call["seed"] == 99

    # The callback is the aggregate one: one ProgressBar sized steps * n_tiles.
    device, latent_format = comfy_stubs["get_previewer_args"]
    assert device is guider.model_patcher.load_device
    assert latent_format is guider.model_patcher.model.latent_format
    (pbar,) = comfy_stubs["progress_bars"]
    assert pbar.total == 4
    assert pbar.updates == [(step + 1, 4, None) for step in range(4)]
    assert out.device == torch.device("cpu")  # stubbed intermediate_device()


def test_sample_latent_default_callback_uses_prepare_callback(comfy_stubs):
    # refine_image now injects the aggregate callback, so the callback=None →
    # prepare_callback default keeps its own dedicated coverage here.
    guider, latent = FakeGuider(), torch.zeros(1, 4, 12, 13)

    out = sampling.sample_latent(guider, object(), SIGMAS, torch.zeros_like(latent), 7, latent)

    model, steps, x0_output_dict = comfy_stubs["prepare_callback_args"]
    assert model is guider.model_patcher
    assert steps == SIGMAS.numel() - 1
    assert x0_output_dict is None
    assert guider.call["callback"] is comfy_stubs["callback"]
    assert comfy_stubs["callback_calls"] == [(step, 4) for step in range(4)]
    assert out.device == torch.device("cpu")


def test_sample_latent_normalizes_denoise_mask(comfy_stubs):
    # sample_latent hands the guider the PREPARED mask form — [B,1,h,w] float32 on the
    # guider's load device — so a guider whose copied sample() predates core moving
    # prepare_mask out of outer_sample never receives a raw CPU [h,w] mask.
    guider, latent = FakeGuider(), torch.zeros(1, 4, 4, 6)
    mask2d = torch.rand(4, 6) > 0.5   # bool input pins the float32 cast as a real conversion

    sampling.sample_latent(guider, object(), SIGMAS, torch.zeros_like(latent), 7, latent,
                           denoise_mask=mask2d, callback=lambda *a: None)

    dm = guider.call["denoise_mask"]
    assert dm.shape == (1, 1, 4, 6)
    assert dm.dtype == torch.float32
    assert torch.equal(dm[0, 0], mask2d.to(torch.float32))


def test_sample_latent_mask_lands_on_guider_load_device(comfy_stubs):
    # The device is READ FROM THE GUIDER, not assumed: a meta-device patcher must
    # receive a meta mask. (The default stub device is cpu, where a hardcoded
    # .to("cpu") would pass by accident; meta pins the load_device read.)
    guider, latent = FakeGuider(), torch.zeros(1, 4, 4, 6)
    guider.model_patcher.load_device = torch.device("meta")

    sampling.sample_latent(guider, object(), SIGMAS, torch.zeros_like(latent), 7, latent,
                           denoise_mask=torch.ones(4, 6), callback=lambda *a: None)

    dm = guider.call["denoise_mask"]
    assert dm.device.type == "meta"
    assert dm.shape == (1, 1, 4, 6)


def test_sample_latent_normalizes_batched_denoise_mask(comfy_stubs):
    # [B,h,w] -> [B,1,h,w]: per-batch rows preserved bit for bit, never pooled.
    guider, latent = FakeGuider(), torch.zeros(2, 4, 4, 6)
    mask = torch.zeros(2, 4, 6)
    mask[0, :2] = 1.0
    mask[1, 2:] = 1.0

    sampling.sample_latent(guider, object(), SIGMAS, torch.zeros_like(latent), 7, latent,
                           denoise_mask=mask, callback=lambda *a: None)

    dm = guider.call["denoise_mask"]
    assert dm.shape == (2, 1, 4, 6)
    assert torch.equal(dm[:, 0], mask)


def test_sample_latent_none_denoise_mask_passthrough(comfy_stubs):
    # No mask -> None reaches the guider unchanged (no tensor is fabricated).
    guider, latent = FakeGuider(), torch.zeros(1, 4, 4, 6)

    sampling.sample_latent(guider, object(), SIGMAS, torch.zeros_like(latent), 7, latent,
                           callback=lambda *a: None)

    assert guider.call["denoise_mask"] is None


class FakeVideoVAE:
    # Wan/Qwen-Image family stub: latent_dim 3, an image batch encodes to a 5-D
    # [B,C,1,h,w] latent (comfy's VAE.encode unsqueezes a time axis of 1) and decode
    # returns the 5-D [B,T,H,W,C] pixel layout of core's VAE.decode (movedim(1,-1)
    # of [B,C,T,H,W]) — core's VAEDecode node, not VAE.decode, folds T into batch.
    latent_dim = 3
    latent_channels = 4

    def __init__(self):
        self.encoded = None

    def encode(self, pixels):
        batch, height, width, _channels = pixels.shape
        self.encoded = torch.zeros(batch, 4, 1, height // 8, width // 8)
        return self.encoded

    def decode(self, samples):
        batch, _, time, height, width = samples.shape
        return torch.full((batch, time, height * 8, width * 8, 3), 0.25)


def test_video_vae_noise_matches_5d_latent(comfy_stubs):
    # Krea 2 regression: a video-family VAE (latent_dim 3) encodes to [B,C,1,h,w]; the
    # canvas noise draw must carry the same 5-D layout, or the sampler's
    # sigma*noise + latent mix broadcasts a fake temporal axis.
    guider, sampler, vae, noise = FakeGuider(), object(), FakeVideoVAE(), FakeNoise()
    image = torch.rand(1, 96, 104, 3)

    out = sampling.refine_image(image, guider, sampler, SIGMAS, vae, noise, max_tile_width=1024, max_tile_height=1024, context_anchor=0, context_overlap=0)

    assert out.shape == (1, 96, 104, 3)
    dummy = noise.calls[0]["samples"]
    assert dummy.shape == (1, 4, 1, 12, 13)
    assert guider.call["noise"].shape == guider.call["latent_image"].shape


def test_video_vae_tiled_noise_and_mask_are_5d(comfy_stubs):
    # Multi-tile on a 5-D latent: every tile's noise slice matches its latent, and the
    # denoise mask reaches the guider in the 5-D canonical form [B,1,1,h,w].
    guider, sampler, vae, noise = FakeGuider(), object(), FakeVideoVAE(), FakeNoise()
    image = torch.rand(1, 96, 104, 3)

    out = sampling.refine_image(image, guider, sampler, SIGMAS, vae, noise, max_tile_width=96, max_tile_height=96, context_anchor=16, context_overlap=16)

    assert out.shape == (1, 96, 104, 3)
    assert guider.sample_calls > 1
    for call in guider.calls:
        assert call["latent_image"].ndim == 5
        assert call["noise"].shape == call["latent_image"].shape
        if call["denoise_mask"] is not None:
            assert call["denoise_mask"].shape == (1, 1, 1, *call["latent_image"].shape[-2:])


def test_sample_latent_normalizes_denoise_mask_for_5d_latent(comfy_stubs):
    # 5-D latent -> [B,1,1,h,w]: a 4-D mask would hit core reshape_mask's
    # (1,1,-1,h,w) fold, which pools the batch into the temporal axis at B>1.
    guider, latent = FakeGuider(), torch.zeros(2, 4, 1, 4, 6)
    mask = torch.zeros(2, 4, 6)
    mask[0, :2] = 1.0
    mask[1, 2:] = 1.0

    sampling.sample_latent(guider, object(), SIGMAS, torch.zeros_like(latent), 7, latent,
                           denoise_mask=mask, callback=lambda *a: None)

    dm = guider.call["denoise_mask"]
    assert dm.shape == (2, 1, 1, 4, 6)
    assert torch.equal(dm[:, 0, 0], mask)


class FoldingVideoVAE(FakeVideoVAE):
    # What core's VAE.encode really does for the Wan family (comfy/sd.py): a 4-D image batch
    # is reshaped to [1,C,B,H,W], so the images become FRAMES of one clip and temporal
    # compression merges them into ONE latent row. A single-row input comes back as the
    # [1,C,1,h,w] every single-image run already sees.
    def __init__(self):
        super().__init__()
        self.encode_calls = []

    def encode(self, pixels):
        self.encode_calls.append(pixels.shape[0])
        _, height, width, _ = pixels.shape
        return torch.zeros(1, 4, 1, height // 8, width // 8)


def test_encode_pixels_splits_a_batch_for_a_video_vae():
    # The helper's whole job: one encode per image, concatenated back to the [B,C,1,h,w]
    # layout the noise dummy mirrors -- instead of one folded [1,C,1,h,w] row.
    vae = FoldingVideoVAE()

    latent = sampling.encode_pixels(vae, torch.rand(3, 32, 24, 3))

    assert vae.encode_calls == [1, 1, 1]
    assert latent.shape == (3, 4, 1, 4, 3)


def test_encode_pixels_passes_a_4d_vae_through_in_one_call():
    # B=1 and every 4-D VAE must take the single unchanged vae.encode call: that is what
    # keeps the byte-for-byte pinned no-mask path byte-for-byte.
    vae = FakeVAE()

    latent = sampling.encode_pixels(vae, torch.rand(2, 32, 24, 3))

    assert vae.encode_calls == 1
    assert latent.shape == (2, 4, 4, 3)
    assert sampling.encode_pixels(FoldingVideoVAE(), torch.rand(1, 32, 24, 3)).shape == (1, 4, 1, 4, 3)


def test_batch_through_a_folding_video_vae_tiles_row_by_row(comfy_stubs):
    # Owner-reported live defect: a 2-image batch through a Wan-family VAE died on the
    # encode/noise fail-fast, because the VAE folded both images into one latent row.
    # Encoding row by row keeps every image on its own row, so the batch tiles.
    guider, sampler, vae, noise = FakeGuider(), object(), FoldingVideoVAE(), FakeNoise()
    image = torch.rand(2, 96, 104, 3)

    out = sampling.refine_image(image, guider, sampler, SIGMAS, vae, noise, max_tile_width=64, max_tile_height=64, context_anchor=0, context_overlap=16)

    assert out.shape == (2, 96, 104, 3)
    assert guider.sample_calls > 1
    for call in guider.calls:
        assert call["latent_image"].ndim == 5
        assert (call["latent_image"].shape[0], call["latent_image"].shape[2]) == (2, 1)
        assert call["noise"].shape == call["latent_image"].shape
    assert vae.encode_calls == [1] * (2 * guider.sample_calls)


def test_batch_folding_non_video_vae_fails_fast(comfy_stubs):
    # The encode/noise guard stays the net for the layouts encode_pixels does NOT handle:
    # a 4-D VAE that folds an image batch is still untileable, and must say so instead of
    # crashing downstream with a broadcast shape error.
    class FoldingImageVAE(FakeVAE):
        def encode(self, pixels):
            _, height, width, _ = pixels.shape
            return torch.zeros(1, 4, height // 8, width // 8)

    guider, sampler, vae, noise = FakeGuider(), object(), FoldingImageVAE(), FakeNoise()
    image = torch.rand(2, 96, 104, 3)

    with pytest.raises(RuntimeError, match="latent layout is not supported"):
        sampling.refine_image(image, guider, sampler, SIGMAS, vae, noise, max_tile_width=1024, max_tile_height=1024, context_anchor=0, context_overlap=0)
