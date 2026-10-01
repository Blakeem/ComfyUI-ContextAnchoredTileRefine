"""upscale.py: the whole-image pre-tiling upscale stage, plus the sampling plumbing the
all-in-one node builds from widgets. The upscale tests pin the two things that can
silently cost quality — a same-size lanczos round trip that should have been skipped, and
a second resize after a model that already landed on the target — plus both model
residency branches (with and without `.patcher`) and the OOM tile-halving retry. The
builder tests pin the BasicScheduler / Noise_RandomNoise arithmetic against the stubbed
core functions. The upscale model is a duck-typed fake and comfy is the `comfy_stubs`
fixture: no GPU, no real model."""
from types import SimpleNamespace

import pytest
import torch

from context_anchored_tile_refine import grid, progress, upscale


def _oom():
    # The stub raise_non_oom keys off the message, as the real one keys off the type.
    return RuntimeError("Allocation on device failed: CUDA out of memory")


class FakeUpscaleModel:
    """Duck-typed UPSCALE_MODEL: callable NCHW->NCHW at `scale`, plus the residency
    surface _upscale_with_model touches. `patcher=None` leaves the attribute ABSENT,
    which is the pre-.patcher loader the version-defensive branch exists for. `errors`
    are raised on successive calls before the first real upscale."""

    def __init__(self, scale=2, patcher=None, errors=()):
        self.scale = scale
        self.errors = list(errors)
        self.devices = []
        self.calls = 0
        if patcher is not None:
            self.patcher = patcher

    def to(self, device):
        self.devices.append(device)
        return self

    def __call__(self, samples):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return torch.nn.functional.interpolate(samples, scale_factor=self.scale, mode="nearest")


class FakePatcher:
    load_device = torch.device("cpu")


def _memory_required(image, scale):
    # comfy_extras/nodes_upscale_model.py ImageUpscaleWithModel's estimate.
    return (512 * 512 * 3) * image.element_size() * max(scale, 1.0) * 384.0 + image.nelement() * image.element_size()


# --- scale_target: pure rounding ----------------------------------------------------

@pytest.mark.parametrize("width,height,upscale_by,expected", [
    (1024, 768, 2.0, (2048, 1536)),
    (1023, 767, 1.5, (1534, 1150)),      # round(1534.5) is 1534 (banker's), as in core
    (1000, 1000, 0.5, (500, 500)),
    (1920, 1080, 1.0, (1920, 1080)),
])
def test_scale_target_matches_core_rounding(width, height, upscale_by, expected):
    assert upscale.scale_target(width, height, upscale_by) == expected
    assert expected == (round(width * upscale_by), round(height * upscale_by))


def test_scale_target_floors_at_one_pixel():
    # round(4 * 0.01) is 0; core would ask for a 0-px axis, the floor keeps it at 1.
    assert upscale.scale_target(4, 4, 0.01) == (1, 1)


# --- prepare_upscaled without a model -----------------------------------------------

def test_no_model_size_change_runs_exactly_one_lanczos_resize(comfy_stubs):
    image = torch.rand(1, 64, 48, 3)
    out = upscale.prepare_upscaled(image, None, 2.0)

    assert comfy_stubs["common_upscale_calls"] == [((1, 3, 64, 48), 96, 128, "lanczos", "disabled")]
    assert out.shape == (1, 128, 96, 3)


@pytest.mark.parametrize("upscale_by", [1.0, 1.004])
def test_no_model_at_target_size_returns_the_input_untouched(comfy_stubs, upscale_by):
    # 1.004 still rounds 48x64 back onto 48x64: the skip is keyed on the TARGET, not on
    # upscale_by == 1.0. A lanczos here would be a pure uint8 round-trip loss.
    image = torch.rand(1, 64, 48, 3)
    out = upscale.prepare_upscaled(image, None, upscale_by)

    assert out is image
    assert comfy_stubs["common_upscale_calls"] == []


# --- prepare_upscaled with a model --------------------------------------------------

def test_model_output_already_at_target_skips_the_second_resize(comfy_stubs):
    # Out-of-gamut input values double as the check that the decode-side clamp survives.
    image = torch.full((1, 8, 8, 3), 2.0)
    image[0, 0, 0, :] = -1.0
    model = FakeUpscaleModel(scale=2, patcher=FakePatcher())

    out = upscale.prepare_upscaled(image, model, 2.0)

    assert len(comfy_stubs["tiled_scale_calls"]) == 1
    assert comfy_stubs["common_upscale_calls"] == []
    assert out.shape == (1, 16, 16, 3)
    assert float(out.max()) == 1.0 and float(out.min()) == 0.0


def test_model_output_off_target_is_lanczos_trimmed_once(comfy_stubs):
    image = torch.rand(1, 64, 48, 3)
    model = FakeUpscaleModel(scale=2, patcher=FakePatcher())

    out = upscale.prepare_upscaled(image, model, 1.5)

    # The model lands on 96x128; the single lanczos pass takes it to the 72x96 target.
    assert comfy_stubs["common_upscale_calls"] == [((1, 3, 128, 96), 72, 96, "lanczos", "disabled")]
    assert out.shape == (1, 96, 72, 3)


def test_model_runs_at_core_tile_geometry(comfy_stubs):
    image = torch.rand(1, 64, 48, 3)
    model = FakeUpscaleModel(scale=2, patcher=FakePatcher())

    upscale.prepare_upscaled(image, model, 2.0)

    assert comfy_stubs["tiled_scale_calls"] == [
        {"tile_x": 512, "tile_y": 512, "overlap": 32, "upscale_amount": 2},
    ]


# --- _upscale_with_model: the two residency branches --------------------------------

def test_patcher_branch_hands_the_managed_model_to_load_models_gpu(comfy_stubs):
    image = torch.rand(1, 64, 48, 3)
    patcher = FakePatcher()
    model = FakeUpscaleModel(scale=2, patcher=patcher)

    upscale._upscale_with_model(model, image)

    # force_full_load=True is core's own argument (nodes_upscale_model.py:73). Without it a
    # spandrel module past the lowvram budget is never moved to the device, and the forward
    # pass raises a device mismatch that raise_non_oom re-raises before the retry can run.
    assert comfy_stubs["load_models_gpu_calls"] == [([patcher], _memory_required(image, 2), True)]
    # load_models_gpu owns residency on this branch: no hand-rolled reserve, no .to().
    assert comfy_stubs["free_memory_calls"] == []
    assert model.devices == []


def test_patcherless_branch_reserves_moves_and_offloads(comfy_stubs):
    image = torch.rand(1, 64, 48, 3)
    model = FakeUpscaleModel(scale=2)

    upscale._upscale_with_model(model, image)

    assert comfy_stubs["free_memory_calls"] == [(_memory_required(image, 2), torch.device("cpu"))]
    assert comfy_stubs["load_models_gpu_calls"] == []
    assert model.devices == [torch.device("cpu"), "cpu"]


def test_patcherless_branch_offloads_after_a_raise(comfy_stubs):
    image = torch.rand(1, 64, 48, 3)
    model = FakeUpscaleModel(scale=2, errors=[ValueError("boom")])

    with pytest.raises(ValueError, match="boom"):
        upscale._upscale_with_model(model, image)

    # The `finally` offload is the whole point: a raise must not strand the model on device.
    assert model.devices == [torch.device("cpu"), "cpu"]


def test_non_oom_error_is_not_retried(comfy_stubs):
    image = torch.rand(1, 64, 48, 3)
    model = FakeUpscaleModel(scale=2, patcher=FakePatcher(), errors=[ValueError("boom")])

    with pytest.raises(ValueError, match="boom"):
        upscale._upscale_with_model(model, image)

    assert len(comfy_stubs["tiled_scale_calls"]) == 1


class RgbOnlyUpscaleModel(FakeUpscaleModel):
    """A spandrel model's first conv: it rejects any input that is not 3 channels."""

    def __call__(self, samples):
        if samples.shape[1] != 3:
            raise RuntimeError(f"expected input to have 3 channels, but got {samples.shape[1]}")
        return super().__call__(samples)


def test_an_rgba_image_upscales_its_rgb_through_the_model_and_its_alpha_bilinear(comfy_stubs):
    # Core's ImageUpscaleWithModel splits the alpha off, so a 4-channel IMAGE must not reach
    # the model's first conv.
    image = torch.rand(1, 64, 48, 4)
    model = RgbOnlyUpscaleModel(scale=2, patcher=FakePatcher())

    out = upscale._upscale_with_model(model, image)

    assert out.shape == (1, 128, 96, 4)
    assert comfy_stubs["common_upscale_calls"] == [((1, 1, 64, 48), 96, 128, "bilinear", "disabled")]


# --- the OOM tile-halving retry -----------------------------------------------------

def test_oom_halves_the_tile_until_it_fits(comfy_stubs):
    image = torch.rand(1, 64, 48, 3)
    model = FakeUpscaleModel(scale=2, patcher=FakePatcher(), errors=[_oom(), _oom()])

    out = upscale._upscale_with_model(model, image)

    assert [call["tile_x"] for call in comfy_stubs["tiled_scale_calls"]] == [512, 256, 128]
    assert out.shape == (1, 128, 96, 3)


def test_oom_below_the_tile_floor_raises(comfy_stubs):
    image = torch.rand(1, 64, 48, 3)
    model = FakeUpscaleModel(scale=2, patcher=FakePatcher(), errors=[_oom(), _oom(), _oom()])

    # 512 -> 256 -> 128 -> 64, which is under MODEL_TILE_MIN, so the last OOM propagates.
    with pytest.raises(RuntimeError, match="out of memory"):
        upscale._upscale_with_model(model, image)

    assert [call["tile_x"] for call in comfy_stubs["tiled_scale_calls"]] == [512, 256, 128]


# --- the ledger's upscale segment ---------------------------------------------------

def test_the_model_pass_refits_the_upscale_segment_on_every_attempt(comfy_stubs):
    # The tiled_scale step count is the segment's TRUE size and is not knowable before the
    # pass starts; a halved tile means MORE steps, so the OOM retry has to re-fit again.
    # 600x600 at tile 512/overlap 32 is a 2x2 walk (4 steps); at 256 it is 3x3 (9).
    image = torch.rand(1, 600, 600, 3)
    model = FakeUpscaleModel(scale=2, patcher=FakePatcher(), errors=[_oom()])
    ledger = progress.Ledger(((progress.UPSCALE, progress.W_UPSCALE_STEP),))

    with ledger:
        out = upscale.prepare_upscaled(image, model, 2.0, progress=ledger)

    assert out.shape == (1, 1200, 1200, 3)
    assert [name for name, _units in ledger.segments] == [progress.UPSCALE]
    assert ledger.segments[0][1] == 9 * progress.W_UPSCALE_STEP


def test_no_upscale_model_opens_no_upscale_segment(comfy_stubs):
    # Skipped entirely, not opened at zero: there is no model pass to report on.
    ledger = progress.Ledger(((progress.UPSCALE, progress.W_UPSCALE_STEP),))

    with ledger:
        upscale.prepare_upscaled(torch.rand(1, 32, 32, 3), None, 2.0, progress=ledger)

    assert ledger.segments == []


# --- build_sigmas: BasicScheduler's arithmetic --------------------------------------

class FakeModel:
    """The one thing build_sigmas touches on a MODEL: get_model_object("model_sampling")."""

    def __init__(self):
        self.model_sampling = object()
        self.requested = []

    def get_model_object(self, name):
        self.requested.append(name)
        return self.model_sampling


def test_build_sigmas_full_denoise_schedules_exactly_steps(comfy_stubs):
    model = FakeModel()

    sigmas = upscale.build_sigmas(model, "sgm_uniform", 20, 1.0)

    assert model.requested == ["model_sampling"]
    assert comfy_stubs["calculate_sigmas_calls"] == [(model.model_sampling, "sgm_uniform", 20)]
    # The stub returns 0..total_steps, so the tail slice is directly readable.
    assert torch.equal(sigmas, torch.arange(21, dtype=torch.float32))


@pytest.mark.parametrize("steps,denoise,total_steps", [
    (20, 0.5, 40),
    (20, 0.35, 57),   # int(20/0.35) truncates 57.14 -> 57, as in core
    (8, 0.75, 10),    # int(8/0.75) truncates 10.66 -> 10
])
def test_build_sigmas_partial_denoise_slices_the_tail(comfy_stubs, steps, denoise, total_steps):
    sigmas = upscale.build_sigmas(FakeModel(), "normal", steps, denoise)

    assert comfy_stubs["calculate_sigmas_calls"][0][2] == total_steps
    assert sigmas.numel() == steps + 1
    # The tail of 0..total_steps: the schedule starts partway down, which IS the denoise.
    assert torch.equal(sigmas, torch.arange(total_steps - steps, total_steps + 1, dtype=torch.float32))


@pytest.mark.parametrize("denoise", [0.0, -1.0])
def test_build_sigmas_zero_denoise_returns_empty_without_scheduling(comfy_stubs, denoise):
    # An empty tensor is what makes refine_image return the upscaled image unchanged
    # (sigmas.numel() < 2), i.e. denoise 0 == upscale only.
    sigmas = upscale.build_sigmas(FakeModel(), "normal", 20, denoise)

    assert sigmas.numel() == 0
    assert comfy_stubs["calculate_sigmas_calls"] == []


# --- Noise_RandomNoise ---------------------------------------------------------------

def test_noise_stores_the_seed():
    assert upscale.Noise_RandomNoise(1234).seed == 1234


def test_noise_delegates_to_prepare_noise(comfy_stubs):
    samples = torch.rand(2, 4, 8, 8)
    noise = upscale.Noise_RandomNoise(99)

    out = noise.generate_noise({"samples": samples})

    assert comfy_stubs["prepare_noise_calls"] == [(samples, 99, None)]
    assert out.shape == samples.shape


def test_noise_passes_batch_index_through(comfy_stubs):
    samples = torch.rand(2, 4, 8, 8)

    upscale.Noise_RandomNoise(99).generate_noise({"samples": samples, "batch_index": [1, 0]})

    assert comfy_stubs["prepare_noise_calls"] == [(samples, 99, [1, 0])]


# --- SlicedCanvasNoise ---------------------------------------------------------------

# A non-square canvas and a non-square rect inside it, so a swapped axis cannot pass: 256x320
# px is a 32x40 cell canvas, and the rect keeps cells y 4..16 by x 8..24.
CANVAS_H, CANVAS_W = 256, 320
BLOCK_RECT = grid.Rect(x0=64, y0=32, x1=192, y1=128)
BLOCK_CELLS = (4, 16, 8, 24)


class FakeVAE:
    """The two attributes sampling.build_canvas_noise reads off a VAE: the latent channel count,
    and latent_dim 3 for a video-family VAE, which encodes an image batch to a 5-D latent."""

    def __init__(self, latent_dim=2, latent_channels=4):
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels


def _positional_prepare_noise(monkeypatch):
    # The pure suite's stub draws ZEROS, and zeros against zeros cannot catch a swapped axis
    # or a missing // 8. arange gives every latent cell a value that names its position.
    import comfy.sample

    def prepare_noise(latent_image, seed, batch_inds=None):
        return torch.arange(latent_image.numel(), dtype=torch.float32).reshape(latent_image.shape)

    monkeypatch.setattr(comfy.sample, "prepare_noise", prepare_noise)


@pytest.mark.parametrize(("latent_dim", "time_dims"), [(2, ()), (3, (1,))])
def test_sliced_canvas_noise_keeps_the_full_draws_own_window(comfy_stubs, monkeypatch, latent_dim, time_dims):
    _positional_prepare_noise(monkeypatch)

    noise = upscale.SlicedCanvasNoise(FakeVAE(latent_dim=latent_dim), 7, CANVAS_H, CANVAS_W, BLOCK_RECT)

    full_shape = (1, 4, *time_dims, CANVAS_H // 8, CANVAS_W // 8)
    full = torch.arange(4 * (CANVAS_H // 8) * (CANVAS_W // 8), dtype=torch.float32).reshape(full_shape)
    y0, y1, x0, x1 = BLOCK_CELLS
    assert noise.seed == 7
    assert noise.canvas_shape == full_shape
    assert noise.cell_origin == (y0, x0)
    assert torch.equal(noise.slice, full[..., y0:y1, x0:x1])


def test_sliced_canvas_noise_hands_out_a_fresh_copy_of_the_slice(comfy_stubs, monkeypatch):
    _positional_prepare_noise(monkeypatch)
    noise = upscale.SlicedCanvasNoise(FakeVAE(), 7, CANVAS_H, CANVAS_W, BLOCK_RECT)
    latent = {"samples": torch.zeros(1, 4, 12, 16)}

    first = noise.generate_noise(latent)
    second = noise.generate_noise(latent)

    assert torch.equal(first, second)
    assert first is not second
    assert first is not noise.slice


def test_sliced_canvas_noise_rejects_a_latent_the_slice_does_not_fit(comfy_stubs, monkeypatch):
    _positional_prepare_noise(monkeypatch)
    noise = upscale.SlicedCanvasNoise(FakeVAE(), 7, CANVAS_H, CANVAS_W, BLOCK_RECT)

    with pytest.raises(RuntimeError, match="does not match the latent") as raised:
        noise.generate_noise({"samples": torch.zeros(1, 4, 16, 12)})

    assert "(1, 4, 12, 16)" in str(raised.value)
    assert "(1, 4, 16, 12)" in str(raised.value)


@pytest.mark.comfy
def test_sliced_canvas_noise_fields_read_the_full_canvas_at_the_blocks_origin(comfy_env, monkeypatch):
    # The block's SDE field: drawn at the FULL canvas shape and read at the block's origin, so
    # every injection is the one the entire canvas run would have made there.
    import comfy.k_diffusion.sampling
    import comfy.samplers

    from context_anchored_tile_refine import stepper

    _positional_prepare_noise(monkeypatch)
    sigmas = torch.tensor([0.8, 0.5, 0.25, 0.0])
    noise = upscale.SlicedCanvasNoise(FakeVAE(), 7, CANVAS_H, CANVAS_W, BLOCK_RECT)
    deterministic = comfy.samplers.KSAMPLER(comfy.k_diffusion.sampling.sample_dpmpp_2m, {}, {})
    stochastic = comfy.samplers.KSAMPLER(comfy.k_diffusion.sampling.sample_dpmpp_2m_sde, {}, {})

    assert noise.noise_fields(deterministic, sigmas) is None

    fields = noise.noise_fields(stochastic, sigmas)
    reference = stepper.build_noise_fields(stochastic, noise.canvas_shape, noise.seed, sigmas)
    y0, y1, x0, x1 = BLOCK_CELLS
    block = fields.for_window((0, y1 - y0, 0, x1 - x0))(sigmas[0], sigmas[1])

    assert torch.equal(block, reference.for_window(BLOCK_CELLS)(sigmas[0], sigmas[1]))
    assert not torch.equal(block, reference.for_window((0, y1 - y0, 0, x1 - x0))(sigmas[0], sigmas[1]))


# --- build_guider / encode_empty -----------------------------------------------------

def test_build_guider_sets_conds_and_cfg(comfy_stubs):
    model = FakeModel()
    positive, negative = object(), object()

    guider = upscale.build_guider(model, positive, negative, 3.5)

    assert guider.model_patcher is model
    assert guider.cfg == 3.5
    # refine_image's VL check requires exactly this: 'positive' present in original_conds.
    assert guider.original_conds == {"positive": positive, "negative": negative}


def test_encode_empty_encodes_the_empty_prompt():
    conditioning = object()
    clip = SimpleNamespace(
        tokenized=[],
        encoded=[],
    )
    clip.tokenize = lambda text: clip.tokenized.append(text) or ("tokens", text)
    clip.encode_from_tokens_scheduled = lambda tokens: clip.encoded.append(tokens) or conditioning

    assert upscale.encode_empty(clip) is conditioning
    assert clip.tokenized == [""]
    assert clip.encoded == [("tokens", "")]
