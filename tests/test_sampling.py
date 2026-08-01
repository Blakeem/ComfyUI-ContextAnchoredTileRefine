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
        assert channels == 3, "encode received {} channels".format(channels)
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
