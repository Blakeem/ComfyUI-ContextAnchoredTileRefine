import threading

import pytest
import torch

from context_anchored_tile_refine import stepper

# Every test builds a real comfy.samplers.KSAMPLER (the layer isinstance-checks it) and the
# bit-identity cases drive the real k-diffusion functions. CPU only, no model loads.
pytestmark = pytest.mark.comfy

# 3 steps, ending at 0 so the final-step exception in EVALS_PER_STEP engages, and every
# interior sigma in (0, 1) so the CONST flow's logit-based logSNR is defined.
SIGMAS = torch.tensor([0.8, 0.5, 0.25, 0.0])
LATENT_SHAPE = (1, 4, 8, 8)

# Lane windows into that canvas, as the noise providers read them: (y0, y1, x0, x1) in
# latent cells. LEFT and RIGHT overlap on canvas columns 3 and 4.
FULL_WINDOW = (0, 8, 0, 8)
LEFT_WINDOW = (0, 8, 0, 5)
RIGHT_WINDOW = (0, 8, 3, 8)


# ---- fakes -----------------------------------------------------------------


class _ModelPatcher:
    def __init__(self, model_sampling):
        self._model_sampling = model_sampling

    def get_model_object(self, name):
        assert name == "model_sampling"
        return self._model_sampling


class _InnerModel:
    # The chain sample_seeds_2 reaches through (k_diffusion/sampling.py:1604):
    # model.inner_model.model_patcher.get_model_object('model_sampling').
    def __init__(self, model_sampling):
        self.model_patcher = _ModelPatcher(model_sampling)


class FakeModelK:
    """comfy's KSamplerX0Inpaint stand-in: a deterministic f(x, sigma) plus a full record of
    every eval. No RNG anywhere, so a reference run and a lane run are bit-identical."""

    def __init__(self, model_sampling=None):
        self.guider = None
        self.probe_guiders = []
        self.fail_at = None
        self.failure = None
        self.evals = 0
        self.inputs = []
        self.returns = []
        self.seen_positive = []
        self.seen_torn_down = []
        if model_sampling is not None:
            self.inner_model = _InnerModel(model_sampling)

    def __call__(self, x, sigma, **kwargs):
        if self.fail_at == self.evals:
            raise self.failure
        self.evals += 1
        self.inputs.append(x.clone())
        self.seen_positive.append(self.guider.original_conds["positive"] if self.guider else None)
        self.seen_torn_down.append(tuple(guider.torn_down for guider in self.probe_guiders))
        view = sigma.reshape([-1] + [1] * (x.ndim - 1))
        out = x * 0.5 - view * 0.25
        self.returns.append(out)
        return out


class LaneGuider:
    """comfy's CFGGuider cut to the ONE method the layer calls. A real guider builds the
    KSamplerX0Inpaint and hands it to sampler.sample (comfy/samplers.py:1000-1006); this
    hands the lane's rebuilt sampler function a FakeModelK instead, which is the whole
    surface the barrier sits on. `torn_down` stands in for outer_sample's finally-block
    cleanup: it must never be observable from another lane's eval."""

    def __init__(self, model_k, positive=None):
        self.model_k = model_k
        self.original_conds = {"positive": positive}
        self.calls = []
        self.torn_down = False
        # Stands in for a failure in outer_sample's POST-eval tail (inverse_noise_scaling,
        # process_latent_out, cleanup_models) -- past the layer's final exit barrier.
        self.fail_after_sample = None

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None,
               disable_pbar=False, seed=None):
        self.calls.append({
            "noise": noise, "latent_image": latent_image, "denoise_mask": denoise_mask,
            "callback": callback, "disable_pbar": disable_pbar, "seed": seed,
        })
        self.model_k.guider = self
        extra_args = {"denoise_mask": denoise_mask, "seed": seed}
        out = sampler.sampler_function(
            self.model_k, noise, sigmas, extra_args=extra_args, callback=callback,
            disable=disable_pbar, **sampler.extra_options,
        )
        self.torn_down = True
        if self.fail_after_sample is not None:
            raise self.fail_after_sample
        return out


def cadence_fn(name="sample_euler", evals=1, final_evals=1):
    """A stand-in k-diffusion function with a real sampler's eval CADENCE and nothing else.
    Returns (function, records); `records["extra_options"]` is what the solver knobs landed
    as. Named so the layer's `sample_` prefix strip resolves it against EVALS_PER_STEP."""
    records = {"extra_options": [], "returned": []}

    def fn(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        records["extra_options"].append(dict(kwargs))
        s_in = x.new_ones([x.shape[0]])
        for step in range(int(sigmas.shape[-1]) - 1):
            count = final_evals if float(sigmas[step + 1]) == 0.0 else evals
            for _ in range(count):
                x = model(x, sigmas[step] * s_in, **(extra_args or {}))
        records["returned"].append(x)
        return x

    fn.__name__ = name
    return fn, records


def sde_cadence_fn(name="sample_dpmpp_2m_sde", evals=1, final_evals=1, draws=1):
    """A stand-in with a stochastic sampler's eval AND noise-draw cadence: `draws` calls to
    the INJECTED noise_sampler per non-final step, none on the final one. `records["draws"]`
    holds one list of returned draws per lane, in lane order (the layer enters each lane's
    sampler function under the scheduler's token, which is handed out in lane order)."""
    records = {"draws": []}

    def fn(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None,
           **kwargs):
        drawn = []
        records["draws"].append(drawn)
        s_in = x.new_ones([x.shape[0]])
        for step in range(int(sigmas.shape[-1]) - 1):
            final = float(sigmas[step + 1]) == 0.0
            for _ in range(final_evals if final else evals):
                x = model(x, sigmas[step] * s_in, **(extra_args or {}))
            if not final:
                for _ in range(draws):
                    drawn.append(noise_sampler(sigmas[step], sigmas[step + 1]))
        return x

    fn.__name__ = name
    return fn, records


def recording_noise_sampler(fields, window, log):
    """One window's noise_sampler with the (sigma, sigma_next) pairs it is asked for logged —
    the reference run and the lane run must be asked for exactly the same ones."""
    inner = fields.for_window(window)

    def noise_sampler(sigma, sigma_next):
        log.append((float(sigma), float(sigma_next)))
        return inner(sigma, sigma_next)

    return noise_sampler


def flow_model_sampling():
    # A REAL CONST-flow model_sampling, exactly as model_base.model_sampling composes it for
    # ModelType.FLOW (comfy/model_base.py:119-121). seeds_2 computes logSNR through it.
    import comfy.model_sampling

    class FlowSampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
        pass

    return FlowSampling()


def make_sampler(fn, extra_options=None, inpaint_options=None):
    import comfy.samplers

    return comfy.samplers.KSAMPLER(fn, extra_options or {}, inpaint_options or {})


def make_spec(model_k, positive=None, sigmas=SIGMAS, window=None, extra_options=None, seed=42,
              denoise_mask=None):
    guider = LaneGuider(model_k, positive)
    return guider, stepper.LaneSpec(
        guider=guider, sigmas=sigmas, noise=torch.zeros(LATENT_SHAPE),
        latent=torch.zeros(LATENT_SHAPE), denoise_mask=denoise_mask, seed=seed, window=window,
        extra_options=extra_options,
    )


def run_bounded(call, timeout=60.0):
    """Drives run_lanes on a worker thread so a hang FAILS instead of wedging the suite, and
    re-raises whatever it raised."""
    box = {}

    def target():
        try:
            box["value"] = call()
        except BaseException as error:
            # BaseException, or this harness would itself swallow the very interrupt the
            # interrupt tests below assert run_lanes re-raises.
            box["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), f"run_lanes did not return within {timeout}s"
    if "error" in box:
        raise box["error"]
    return box["value"]


def live_lane_threads():
    return [thread.name for thread in threading.enumerate() if thread.name.startswith("catr-lane-")]


def noop_hook(step_index, sigma, lanes):
    return None


@pytest.fixture(autouse=True)
def _comfy_on_path(comfy_env):
    # Every test here builds a real KSAMPLER; the fixture is what puts the ComfyUI source
    # root on sys.path, so no test may run before it.
    return comfy_env


# ---- the table -------------------------------------------------------------


def test_supported_samplers_is_exactly_the_table():
    assert set(stepper.SUPPORTED_SAMPLERS) == set(stepper.EVALS_PER_STEP)
    assert stepper.SUPPORTED_SAMPLERS == (
        "euler", "dpmpp_2m", "heun", "dpm_2", "exp_heun_2_x0",
        "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu",
        "exp_heun_2_x0_sde",
    )


def test_every_stochastic_key_is_in_the_table_and_no_deterministic_one_is():
    assert set(stepper.SDE_NOISE_KIND) <= set(stepper.EVALS_PER_STEP)
    assert set(stepper.SDE_NOISE_KIND).isdisjoint(
        {"euler", "dpmpp_2m", "heun", "dpm_2", "exp_heun_2_x0"})


# ---- bit-identity against the stock samplers -------------------------------


@pytest.mark.parametrize(("name", "expected_evals"), [
    ("euler", 3),
    ("dpmpp_2m", 3),
    ("heun", 5),
    ("dpm_2", 5),
    ("exp_heun_2_x0", 5),
])
def test_one_lane_matches_the_stock_sampler_bit_for_bit(name, expected_evals):
    # The portability claim: a lane is the stock sampler function, unchanged, with the model
    # callable swapped. The eval count doubles as the EVALS_PER_STEP cross-check against the
    # installed comfy source.
    import comfy.k_diffusion.sampling

    real_fn = getattr(comfy.k_diffusion.sampling, f"sample_{name}")
    model_sampling = flow_model_sampling() if name == "exp_heun_2_x0" else None
    noise = torch.linspace(-1.0, 1.0, 4 * 8 * 8).reshape(LATENT_SHAPE)

    reference_model = FakeModelK(model_sampling)
    reference = real_fn(
        reference_model, noise.clone(), SIGMAS.clone(),
        extra_args={"denoise_mask": None, "seed": 42}, callback=None, disable=True,
    )

    lane_model = FakeModelK(model_sampling)
    guider = LaneGuider(lane_model)
    spec = stepper.LaneSpec(guider=guider, sigmas=SIGMAS, noise=noise, latent=torch.zeros(LATENT_SHAPE), seed=42)
    samples = run_bounded(lambda: stepper.run_lanes([spec], make_sampler(real_fn), noop_hook))

    assert lane_model.evals == expected_evals
    assert reference_model.evals == expected_evals
    assert torch.equal(samples[0], reference)


# ---- the stochastic five, on the shared noise field ------------------------

# (resolved name, model evals over SIGMAS, noise draws over SIGMAS). The dpmpp_2m_sde family
# is 1 eval + 1 draw per step with the final step short-circuiting before either; the
# seeds_2 alias is 2 evals + 2 draws per non-final step and 1 eval on the final one.
STOCHASTIC_CADENCE = [
    ("dpmpp_2m_sde", 3, 2),
    ("dpmpp_2m_sde_gpu", 3, 2),
    ("dpmpp_2m_sde_heun", 3, 2),
    ("dpmpp_2m_sde_heun_gpu", 3, 2),
    ("exp_heun_2_x0_sde", 5, 4),
]


@pytest.mark.parametrize(("name", "expected_evals", "expected_draws"), STOCHASTIC_CADENCE)
def test_one_lane_matches_the_stock_sde_sampler_bit_for_bit(name, expected_evals, expected_draws):
    # Same portability claim as the deterministic cases, with the shared field injected on
    # BOTH sides: the wrapper names differ from sample_dpmpp_2m_sde only in the noise sampler
    # they would build themselves, so handing both sides the same one is what isolates the
    # layer. Every fake here exposes a REAL CONST model_sampling -- the whole family reaches
    # through model.inner_model.model_patcher for it.
    import comfy.k_diffusion.sampling

    real_fn = getattr(comfy.k_diffusion.sampling, f"sample_{name}")
    model_sampling = flow_model_sampling()
    noise = torch.linspace(-1.0, 1.0, 4 * 8 * 8).reshape(LATENT_SHAPE)
    sampler = make_sampler(real_fn)
    # ONE provider per side, both built from the same (shape, seed, sigmas): a field is a pure
    # function of those three, so both sides draw bit-identical noise -- which the final
    # torch.equal proves end to end -- without either side replaying the other's draws. A
    # provider is not a tape: it holds only the draw its fleet is currently on.
    reference_fields = stepper.build_noise_fields(sampler, LATENT_SHAPE, 42, SIGMAS)
    lane_fields = stepper.build_noise_fields(sampler, LATENT_SHAPE, 42, SIGMAS)

    reference_draws = []
    reference_model = FakeModelK(model_sampling)
    reference = real_fn(
        reference_model, noise.clone(), SIGMAS.clone(),
        extra_args={"denoise_mask": None, "seed": 42}, callback=None, disable=True,
        noise_sampler=recording_noise_sampler(reference_fields, FULL_WINDOW, reference_draws),
    )

    lane_draws = []
    lane_model = FakeModelK(model_sampling)
    guider = LaneGuider(lane_model)
    spec = stepper.LaneSpec(guider=guider, sigmas=SIGMAS, noise=noise,
                            latent=torch.zeros(LATENT_SHAPE), seed=42, window=FULL_WINDOW)
    samples = run_bounded(lambda: stepper.run_lanes(
        [spec], sampler, noop_hook,
        noise_fields=lambda lane_spec: recording_noise_sampler(
            lane_fields, lane_spec.window, lane_draws),
    ))

    assert lane_model.evals == expected_evals
    assert reference_model.evals == expected_evals
    # The injected field really is what the sampler drew from, at the same sigma pairs -- so
    # the identity below is an identity of the LAYER, not of a sampler that ignored the noise.
    assert len(lane_draws) == expected_draws
    assert lane_draws == reference_draws
    assert torch.equal(samples[0], reference)


@pytest.mark.parametrize(("name", "draws_per_step"), [
    ("dpmpp_2m_sde", 1),
    ("exp_heun_2_x0_sde", stepper.SEEDS_2_DRAWS_PER_STEP),
])
def test_overlapping_windows_draw_the_same_noise_on_every_shared_cell(name, draws_per_step):
    # The property the whole provider exists for, for BOTH provider kinds: two lanes whose
    # windows overlap see one field, so no overlap band ever carries two different noises.
    import comfy.k_diffusion.sampling

    sampler = make_sampler(getattr(comfy.k_diffusion.sampling, f"sample_{name}"))
    fields = stepper.build_noise_fields(sampler, LATENT_SHAPE, 42, SIGMAS)
    left = fields.for_window(LEFT_WINDOW)
    right = fields.for_window(RIGHT_WINDOW)

    seen = []
    for step in range(2):   # the two non-final steps; the final one draws nothing
        for _ in range(draws_per_step):
            a = left(SIGMAS[step], SIGMAS[step + 1])
            b = right(SIGMAS[step], SIGMAS[step + 1])
            assert a.shape == (1, 4, 8, 5)
            assert b.shape == (1, 4, 8, 5)
            # Canvas columns 3 and 4 are shared: the left window's last two, the right's first
            # two. The second assert is what stops a constant field from passing the first.
            assert torch.equal(a[..., 3:5], b[..., 0:2])
            assert not torch.equal(a[..., 0:2], b[..., 0:2])
            seen.append(a)

    assert len(seen) == 2 * draws_per_step
    assert not torch.equal(seen[0], seen[-1])   # a fresh field per (step, draw), not one reused


def test_run_lanes_injects_each_lanes_own_window_slice_of_the_shared_field():
    # The wire-in: the provider object itself is what run_lanes calls per lane, and what the
    # lane's sampler function receives as noise_sampler.
    fn, records = sde_cadence_fn()
    sampler = make_sampler(fn)
    fields = stepper.build_noise_fields(sampler, LATENT_SHAPE, 42, SIGMAS)
    _, left = make_spec(FakeModelK(), window=LEFT_WINDOW)
    _, right = make_spec(FakeModelK(), window=RIGHT_WINDOW)

    run_bounded(lambda: stepper.run_lanes([left, right], sampler, noop_hook, noise_fields=fields))

    lane_draws = records["draws"]
    assert [len(drawn) for drawn in lane_draws] == [2, 2]
    for a, b in zip(lane_draws[0], lane_draws[1], strict=True):
        assert a.shape == (1, 4, 8, 5)
        assert b.shape == (1, 4, 8, 5)
        assert torch.equal(a[..., 3:5], b[..., 0:2])
        assert not torch.equal(a[..., 0:2], b[..., 0:2])


def test_the_seeds_2_noise_generator_is_seeded_seed_plus_one():
    import comfy.k_diffusion.sampling

    sampler = make_sampler(comfy.k_diffusion.sampling.sample_exp_heun_2_x0_sde)
    fields = stepper.build_noise_fields(sampler, LATENT_SHAPE, 42, SIGMAS)

    first = fields.for_window(FULL_WINDOW)(SIGMAS[0], SIGMAS[1])

    offset = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(43))
    assert torch.equal(first, offset)
    # The BARE seed is prepare_noise's own canvas draw -- same engine, seed, shape, dtype and
    # device -- so a generator seeded with it would re-apply the construction epsilon on the
    # first injection. comfy's default_noise_sampler takes the identical +1 on CPU.
    bare = torch.randn(LATENT_SHAPE, generator=torch.Generator().manual_seed(42))
    assert not torch.equal(first, bare)


def test_the_seeds_2_provider_holds_one_field_and_rejects_a_backwards_request():
    # The one-slot invariant: only the draw the whole fleet is on is retained, so a run cannot
    # accumulate its schedule's worth of canvas-shaped fields. A cache would happily re-serve
    # draw 0 here; the single slot cannot, and says so instead of returning the wrong field.
    import comfy.k_diffusion.sampling

    sampler = make_sampler(comfy.k_diffusion.sampling.sample_exp_heun_2_x0_sde)
    fields = stepper.build_noise_fields(sampler, LATENT_SHAPE, 42, SIGMAS)
    left = fields.for_window(LEFT_WINDOW)
    right = fields.for_window(RIGHT_WINDOW)

    first = left(SIGMAS[0], SIGMAS[1])           # left takes draw 0
    assert torch.equal(right(SIGMAS[0], SIGMAS[1])[..., 0:2], first[..., 3:5])   # right, draw 0
    left(SIGMAS[0], SIGMAS[1])                   # left takes draw 1
    left(SIGMAS[1], SIGMAS[2])                   # left takes draw 2 -- the held slot

    with pytest.raises(RuntimeError, match="lockstep"):
        right(SIGMAS[0], SIGMAS[1])              # right asks for draw 1, which is gone


def test_build_noise_fields_returns_none_for_a_deterministic_sampler():
    import comfy.k_diffusion.sampling

    sampler = make_sampler(comfy.k_diffusion.sampling.sample_dpmpp_2m)

    assert stepper.build_noise_fields(sampler, LATENT_SHAPE, 42, SIGMAS) is None


def test_a_stochastic_sampler_without_a_shared_noise_field_is_rejected():
    import comfy.k_diffusion.sampling

    _, spec = make_spec(FakeModelK(flow_model_sampling()), window=FULL_WINDOW)
    sampler = make_sampler(comfy.k_diffusion.sampling.sample_dpmpp_2m_sde)

    with pytest.raises(ValueError, match="build_noise_fields"):
        stepper.run_lanes([spec], sampler, noop_hook)


def test_a_deterministic_sampler_with_a_shared_noise_field_is_rejected():
    import comfy.k_diffusion.sampling

    _, spec = make_spec(FakeModelK(), window=FULL_WINDOW)
    fields = stepper.build_noise_fields(
        make_sampler(comfy.k_diffusion.sampling.sample_dpmpp_2m_sde), LATENT_SHAPE, 42, SIGMAS)

    with pytest.raises(ValueError, match="deterministic"):
        stepper.run_lanes([spec], make_sampler(comfy.k_diffusion.sampling.sample_euler),
                          noop_hook, noise_fields=fields)


@pytest.mark.parametrize("carrier", ["sampler", "lane"])
def test_a_caller_supplied_noise_sampler_is_never_silently_replaced(carrier):
    import comfy.k_diffusion.sampling

    def caller_noise(sigma, sigma_next):
        return torch.zeros(LATENT_SHAPE)

    real_fn = comfy.k_diffusion.sampling.sample_dpmpp_2m_sde
    options = {"noise_sampler": caller_noise}
    sampler = make_sampler(real_fn, options if carrier == "sampler" else None)
    _, spec = make_spec(FakeModelK(flow_model_sampling()), window=FULL_WINDOW,
                        extra_options=options if carrier == "lane" else None)
    fields = stepper.build_noise_fields(make_sampler(real_fn), LATENT_SHAPE, 42, SIGMAS)

    with pytest.raises(ValueError, match="noise_sampler"):
        stepper.run_lanes([spec], sampler, noop_hook, noise_fields=fields)


# ---- _LaneModel ------------------------------------------------------------


def test_lane_model_proxies_attributes_of_the_wrapped_model():
    inner = FakeModelK(flow_model_sampling())
    lane = stepper.Lane(0, "window")

    wrapper = stepper._LaneModel(inner, lane, None)

    assert wrapper.inner_model is inner.inner_model
    assert wrapper.inner_model.model_patcher.get_model_object("model_sampling") is not None
    assert lane.model_k is inner
    with pytest.raises(AttributeError):
        _ = wrapper.definitely_not_an_attribute


# ---- the hook contract -----------------------------------------------------


def test_hook_fires_once_per_step_after_every_lane_finished_the_previous_one():
    models = [FakeModelK(), FakeModelK()]
    guiders, specs = zip(*[make_spec(model, window=f"w{i}") for i, model in enumerate(models)], strict=True)
    fn, _ = cadence_fn()
    seen = []

    def hook(step_index, sigma, lanes):
        seen.append((step_index, float(sigma), tuple(model.evals for model in models),
                     tuple(lane.window for lane in lanes)))

    run_bounded(lambda: stepper.run_lanes(list(specs), make_sampler(fn), hook))

    # One entry per sigma step, in order; at step i EVERY lane has exactly i evals behind it
    # and none has begun step i.
    assert [entry[0] for entry in seen] == [0, 1, 2]
    assert [entry[1] for entry in seen] == [float(SIGMAS[step]) for step in range(3)]
    assert [entry[2] for entry in seen] == [(0, 0), (1, 1), (2, 2)]
    assert seen[0][3] == ("w0", "w1")
    assert [guider.torn_down for guider in guiders] == [True, True]


def test_hook_edit_of_a_lanes_x_reaches_that_lanes_next_model_input():
    models = [FakeModelK(), FakeModelK()]
    _, specs = zip(*[make_spec(model) for model in models], strict=True)
    fn, _ = cadence_fn()

    def hook(step_index, sigma, lanes):
        if step_index == 1:
            lanes[1].x.copy_(torch.full_like(lanes[1].x, 7.0))

    run_bounded(lambda: stepper.run_lanes(list(specs), make_sampler(fn), hook))

    assert torch.equal(models[1].inputs[1], torch.full(LATENT_SHAPE, 7.0))
    # Lane 0 was untouched, and lane 1's step-0 input was untouched.
    assert not torch.equal(models[0].inputs[1], torch.full(LATENT_SHAPE, 7.0))
    assert torch.equal(models[1].inputs[0], torch.zeros(LATENT_SHAPE))


def test_each_lanes_evals_only_ever_see_its_own_guider():
    models = [FakeModelK(), FakeModelK()]
    _, specs = zip(*[make_spec(model, positive=f"positive-{i}") for i, model in enumerate(models)], strict=True)
    fn, _ = cadence_fn(evals=2, final_evals=1, name="sample_heun")

    run_bounded(lambda: stepper.run_lanes(list(specs), make_sampler(fn), noop_hook))

    assert models[0].seen_positive == ["positive-0"] * 5
    assert models[1].seen_positive == ["positive-1"] * 5


def test_lane_denoised_is_the_previous_steps_last_eval_result():
    # heun cadence: 2 evals per step, 1 on the final step -> step starts at evals 0, 2, 4. The
    # hook must see the SECOND eval of the previous step, not the first.
    model = FakeModelK()
    _, spec = make_spec(model)
    fn, _ = cadence_fn(name="sample_heun", evals=2, final_evals=1)
    seen = []

    def hook(step_index, sigma, lanes):
        seen.append(lanes[0].denoised)

    run_bounded(lambda: stepper.run_lanes([spec], make_sampler(fn), hook))

    assert model.evals == 5
    assert seen[0] is None                     # nothing has been evaluated yet at step 0
    assert seen[1] is model.returns[1]         # last eval of step 0
    assert seen[2] is model.returns[3]         # last eval of step 1
    assert torch.equal(seen[1], model.returns[1])
    assert torch.equal(seen[2], model.returns[3])


def test_no_lane_tears_down_before_every_lane_finished_its_last_eval():
    models = [FakeModelK(), FakeModelK()]
    guiders, specs = zip(*[make_spec(model) for model in models], strict=True)
    for model in models:
        model.probe_guiders = list(guiders)
    fn, _ = cadence_fn()

    run_bounded(lambda: stepper.run_lanes(list(specs), make_sampler(fn), noop_hook))

    # Without the final exit barrier lane 0 returns from its sampler function -- and a real
    # guider's cleanup() runs -- while lane 1 still has its last eval to do.
    for model in models:
        assert model.seen_torn_down == [(False, False)] * 3
    assert [guider.torn_down for guider in guiders] == [True, True]


def test_runs_and_mutates_from_inside_inference_mode():
    # inference/grad mode is THREAD-LOCAL. The step-0 x is the caller's own noise tensor, so
    # it is an INFERENCE tensor here; a lane thread that did not enter inference_mode would
    # raise "Inplace update to inference tensor outside InferenceMode is not allowed" on the
    # hook's edit. Mutating at step 0 is what makes this test discriminating -- a later step's
    # x is produced inside the lane thread and carries that thread's mode, not the caller's.
    with torch.inference_mode():
        model = FakeModelK()
        _, spec = make_spec(model)
        fn, _ = cadence_fn()

        def hook(step_index, sigma, lanes):
            if step_index == 0:
                lanes[0].x.copy_(torch.full_like(lanes[0].x, 3.0))

        run_bounded(lambda: stepper.run_lanes([spec], make_sampler(fn), hook))

        assert torch.equal(model.inputs[0], torch.full(LATENT_SHAPE, 3.0))


# ---- guider call surface ---------------------------------------------------


def test_lane_sample_gets_the_specs_tensors_and_no_per_lane_progress_bar():
    model = FakeModelK()
    mask = torch.ones(1, 1, 8, 8)
    guider, spec = make_spec(model, seed=7, denoise_mask=mask)
    fn, _ = cadence_fn()

    run_bounded(lambda: stepper.run_lanes([spec], make_sampler(fn), noop_hook))

    call = guider.calls[0]
    assert call["disable_pbar"] is True
    assert call["callback"] is None
    assert call["seed"] == 7
    assert call["denoise_mask"] is mask
    assert call["noise"] is spec.noise
    assert call["latent_image"] is spec.latent


def test_extra_options_carry_through_and_per_lane_entries_merge_over_them():
    model = FakeModelK()
    _, spec = make_spec(model, extra_options={"s_noise": 0.9})
    fn, records = cadence_fn()

    run_bounded(lambda: stepper.run_lanes([spec], make_sampler(fn, {"eta": 0.5}), noop_hook))

    assert records["extra_options"] == [{"eta": 0.5, "s_noise": 0.9}]


# ---- intake rejections -----------------------------------------------------


def test_non_ksampler_sampler_object_is_rejected():
    model = FakeModelK()
    _, spec = make_spec(model)

    with pytest.raises(ValueError, match="KSAMPLER"):
        stepper.run_lanes([spec], object(), noop_hook)


def test_unsupported_sampler_names_the_supported_list_and_the_by_design_exclusions():
    import comfy.k_diffusion.sampling

    model = FakeModelK()
    _, spec = make_spec(model)
    sampler = make_sampler(comfy.k_diffusion.sampling.sample_euler_ancestral)

    with pytest.raises(ValueError) as excinfo:
        stepper.run_lanes([spec], sampler, noop_hook)

    message = str(excinfo.value)
    assert "euler_ancestral" in message
    for supported in stepper.SUPPORTED_SAMPLERS:
        assert supported in message
    for excluded in ("dpm_fast", "dpm_adaptive", "uni_pc"):
        assert excluded in message


def test_inpaint_options_random_is_rejected():
    import comfy.k_diffusion.sampling

    model = FakeModelK()
    _, spec = make_spec(model)
    sampler = make_sampler(comfy.k_diffusion.sampling.sample_euler, inpaint_options={"random": True})

    with pytest.raises(ValueError, match="ddim"):
        stepper.run_lanes([spec], sampler, noop_hook)


def test_two_lanes_sharing_one_guider_are_rejected():
    fn, _ = cadence_fn()
    guider = LaneGuider(FakeModelK())
    spec = stepper.LaneSpec(guider=guider, sigmas=SIGMAS, noise=torch.zeros(LATENT_SHAPE),
                            latent=torch.zeros(LATENT_SHAPE))

    with pytest.raises(ValueError, match="its OWN guider"):
        stepper.run_lanes([spec, spec], make_sampler(fn), noop_hook)


def test_lanes_with_different_schedules_are_rejected():
    fn, _ = cadence_fn()
    _, first = make_spec(FakeModelK())
    _, second = make_spec(FakeModelK(), sigmas=torch.tensor([0.9, 0.5, 0.25, 0.0]))

    with pytest.raises(ValueError, match="different sigma schedule"):
        stepper.run_lanes([first, second], make_sampler(fn), noop_hook)


# ---- the eval-count cross-check --------------------------------------------


def test_cross_check_trips_when_the_table_predicts_too_many_evals(monkeypatch):
    monkeypatch.setattr(stepper, "EVALS_PER_STEP", {"euler": (2, 2)})
    model = FakeModelK()
    _, spec = make_spec(model)
    fn, _ = cadence_fn()

    with pytest.raises(RuntimeError, match="EVALS_PER_STEP"):
        run_bounded(lambda: stepper.run_lanes([spec], make_sampler(fn), noop_hook))
    assert live_lane_threads() == []


def test_cross_check_trips_when_the_table_predicts_too_few_evals(monkeypatch):
    monkeypatch.setattr(stepper, "EVALS_PER_STEP", {"heun": (1, 1)})
    model = FakeModelK()
    _, spec = make_spec(model)
    fn, _ = cadence_fn(name="sample_heun", evals=2, final_evals=1)

    with pytest.raises(RuntimeError, match="EVALS_PER_STEP"):
        run_bounded(lambda: stepper.run_lanes([spec], make_sampler(fn), noop_hook))
    assert live_lane_threads() == []


# ---- exception safety and the abort flag -----------------------------------


class Boom(Exception):
    pass


def test_a_lane_exception_propagates_and_every_thread_is_joined():
    models = [FakeModelK(), FakeModelK()]
    _, specs = zip(*[make_spec(model) for model in models], strict=True)
    fn, _ = cadence_fn()
    models[0].fail_at = 1
    models[0].failure = Boom("lane 0 failed")

    with pytest.raises(Boom, match="lane 0 failed"):
        run_bounded(lambda: stepper.run_lanes(list(specs), make_sampler(fn), noop_hook))

    assert live_lane_threads() == []


def test_a_hook_exception_stops_the_whole_fleet():
    models = [FakeModelK(), FakeModelK(), FakeModelK()]
    _, specs = zip(*[make_spec(model) for model in models], strict=True)
    fn, _ = cadence_fn()

    def hook(step_index, sigma, lanes):
        if step_index == 1:
            raise Boom("interrupted")

    with pytest.raises(Boom, match="interrupted"):
        run_bounded(lambda: stepper.run_lanes(list(specs), make_sampler(fn), hook))

    # Without the abort flag each released lane would run out its remaining full-length
    # schedule (3 evals) before the join returned. Every lane unwinds within one further eval
    # of the step that raised -- here, zero: all three were parked at the barrier.
    for model in models:
        assert model.evals <= 2
        assert model.evals == 1
    assert live_lane_threads() == []


# ---- the user's cancel (a BaseException, not an Exception) -----------------


def interrupt_exception():
    # comfy/model_management.py:2098 -- InterruptProcessingException derives from
    # BaseException, so `except Exception` at either catch site would MISS the cancel.
    import comfy.model_management

    return comfy.model_management.InterruptProcessingException()


def test_an_interrupt_raised_in_a_model_eval_reaches_the_caller_unchanged():
    # Where the real one comes from: throw_exception_if_processing_interrupted() inside the
    # model forward (comfy/ops.py run_every_op), i.e. inside _LaneModel.__call__'s forward.
    # It must arrive at the caller AS ITSELF -- execution.py special-cases it to mark the
    # prompt interrupted -- never as the EVALS_PER_STEP RuntimeError from `finish`.
    interrupt = interrupt_exception()
    models = [FakeModelK(), FakeModelK()]
    _, specs = zip(*[make_spec(model) for model in models], strict=True)
    fn, _ = cadence_fn()
    models[0].fail_at = 1
    models[0].failure = interrupt

    with pytest.raises(type(interrupt)) as excinfo:
        run_bounded(lambda: stepper.run_lanes(list(specs), make_sampler(fn), noop_hook))

    assert excinfo.value is interrupt
    assert live_lane_threads() == []


def test_an_interrupt_raised_in_the_hook_reaches_the_caller_unchanged():
    interrupt = interrupt_exception()
    models = [FakeModelK(), FakeModelK()]
    _, specs = zip(*[make_spec(model) for model in models], strict=True)
    fn, _ = cadence_fn()

    def hook(step_index, sigma, lanes):
        if step_index == 1:
            raise interrupt

    with pytest.raises(type(interrupt)) as excinfo:
        run_bounded(lambda: stepper.run_lanes(list(specs), make_sampler(fn), hook))

    assert excinfo.value is interrupt
    # The fleet still unwinds: every lane was parked at the barrier the hook raised from.
    for model in models:
        assert model.evals == 1
    assert live_lane_threads() == []


def test_a_failure_after_the_final_exit_barrier_is_not_a_silent_none_sample():
    # A BaseException from the guider's POST-eval tail leaves eval_count == total_evals, so
    # `finish` adds no abort of its own. If the lane thread does not route it through the
    # abort flag, run_lanes returns [None] where a latent belongs.
    interrupt = interrupt_exception()
    model = FakeModelK()
    guider, spec = make_spec(model)
    guider.fail_after_sample = interrupt
    fn, _ = cadence_fn()

    with pytest.raises(type(interrupt)) as excinfo:
        run_bounded(lambda: stepper.run_lanes([spec], make_sampler(fn), noop_hook))

    assert excinfo.value is interrupt
    assert model.evals == 3
    assert live_lane_threads() == []


# ---- the per-eval progress tick --------------------------------------------


def test_on_eval_ticks_once_per_completed_model_eval():
    # The engine's fine-grained progress tick: n_lanes calls per sigma step where the hook
    # fires once. 2 lanes x (2+2+1) evals on this schedule; serial scheduler, so the tick
    # list is a clean count with no interleaving hazard.
    fn, _records = cadence_fn("sample_exp_heun_2_x0", evals=2, final_evals=1)
    _, spec_a = make_spec(FakeModelK())
    _, spec_b = make_spec(FakeModelK())
    ticks = []

    run_bounded(lambda: stepper.run_lanes(
        [spec_a, spec_b], make_sampler(fn), noop_hook, on_eval=lambda: ticks.append(object())))

    assert len(ticks) == 2 * (2 + 2 + 1)


def test_on_eval_default_is_no_callback():
    # Omitted entirely: the scheduler stores None and the eval path takes zero extra calls,
    # which is what keeps every direct caller and the bit-identity tests unchanged.
    model = FakeModelK()
    _, spec = make_spec(model)
    fn, _records = cadence_fn()

    run_bounded(lambda: stepper.run_lanes([spec], make_sampler(fn), noop_hook))

    assert model.evals == 3
