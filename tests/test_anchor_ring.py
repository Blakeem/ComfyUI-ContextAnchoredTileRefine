"""The anchor-ring schedule: the curve, who it applies to, and that it never leaks.

The ring is the frozen context_anchor halo. Core re-presents it to the model at every step
(comfy samplers.py:639) and, on core's default scale_latent_inpaint, re-noises it to the
CURRENT sigma — so it is mostly noise exactly while structure is decided. `anchor_ring_factor`
replaces that noise level; `anchor_ring_schedule` installs it for one sampler run.
"""
from types import SimpleNamespace

import pytest
import torch

from context_anchored_tile_refine import sampling

SIGMAS = torch.linspace(1.0, 0.0, 5)


class BaseModel:
    """Stands in for comfy's BaseModel: the class whose scale_latent_inpaint is the default."""

    def __init__(self):
        self.model_sampling = SimpleNamespace(noise_scaling=self._noise_scaling)
        self.scaling_calls = []

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        return self._noise_scaling(sigma, noise, latent_image)

    def _noise_scaling(self, sigma, noise, latent_image):
        self.scaling_calls.append(sigma)
        return sigma * noise + (1.0 - sigma) * latent_image


class HoldingModel(BaseModel):
    """Stands in for WAN21/WAN22/HunyuanVideo/LTXAV: already holds the frozen region clean."""

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        return latent_image


class NoInpaintModel:
    pass


def guider_for(model):
    return SimpleNamespace(model_patcher=SimpleNamespace(model=model))


# ---------------------------------------------------------------- the curve

def test_factor_is_one_at_the_first_step():
    # The ring must start MATCHED to the core: a noise-level discontinuity at step 0 lands
    # exactly where composition is decided, which is what this whole schedule exists to avoid.
    assert sampling.anchor_ring_factor(1.0) == 1.0


def test_factor_leads_linearly_above_the_release_point():
    for r in (0.9, 0.5, 0.3, sampling.ANCHOR_RING_RELEASE + 1e-9):
        assert sampling.anchor_ring_factor(r) == pytest.approx(r)


def test_factor_is_continuous_at_the_release_point():
    release = sampling.ANCHOR_RING_RELEASE
    below = sampling.anchor_ring_factor(release - 1e-9)
    assert sampling.anchor_ring_factor(release) == pytest.approx(release)
    assert below == pytest.approx(release, abs=1e-6)


def test_factor_rejoins_the_core_at_the_end():
    # Fully rejoined at r == 0 means the last steps generate their own texture rather than
    # copying the neighbour's.
    assert sampling.anchor_ring_factor(0.0) == pytest.approx(1.0)


def test_factor_dips_only_negligibly_just_past_the_release_point():
    # Blending lead1 (which keeps falling with r) against 1.0 with a smoothstep (zero slope at
    # the handover) means the factor DIPS very slightly below the lead1 line for the first
    # fraction of the release window before climbing to 1.0. It is bounded under 1% of the
    # release value (measured maximum 0.00224, i.e. 1.5% of RELEASE) and the rendered result is
    # what the owner approved, so the curve stands as is — this test pins the dip's size so a
    # future edit cannot enlarge it unnoticed.
    release = sampling.ANCHOR_RING_RELEASE
    rs = [release * i / 200.0 for i in range(201)]
    dip = max(sampling.anchor_ring_factor(release) - sampling.anchor_ring_factor(r) for r in rs)
    assert 0.0 < dip < 0.02 * release


def test_factor_rises_to_one_over_the_release_window():
    # Past the dip the ring rejoins the core steadily; sampled coarsely so the tiny dip near
    # the handover does not make this a duplicate of the test above.
    release = sampling.ANCHOR_RING_RELEASE
    values = [sampling.anchor_ring_factor(release * i / 10.0) for i in range(10, -1, -1)]
    assert values == sorted(values)


def test_factor_never_exceeds_one():
    # An SDE/ancestral sampler can evaluate above sigma_first; the ring must never be noisier
    # than the core it anchors.
    for r in (1.0001, 1.5, 10.0):
        assert sampling.anchor_ring_factor(r) == 1.0


def test_factor_clamps_below_zero():
    assert sampling.anchor_ring_factor(-0.5) == pytest.approx(1.0)


# ---------------------------------------------------------------- who it applies to

def test_applies_to_a_model_on_cores_default():
    model = BaseModel()
    with sampling.anchor_ring_schedule(guider_for(model), SIGMAS):
        assert "scale_latent_inpaint" in vars(model)


def test_skips_a_model_that_already_holds_the_anchor():
    # WAN22 & co. hold the frozen region clean at every sigma — the endpoint this schedule
    # approaches. Overriding their override would make their ring NOISIER early.
    model = HoldingModel()
    with sampling.anchor_ring_schedule(guider_for(model), SIGMAS):
        assert "scale_latent_inpaint" not in vars(model)


def test_skips_a_model_without_the_method():
    model = NoInpaintModel()
    with sampling.anchor_ring_schedule(guider_for(model), SIGMAS):
        assert "scale_latent_inpaint" not in vars(model)


@pytest.mark.parametrize("sigmas", [torch.empty(0), torch.zeros(3)])
def test_skips_a_degenerate_schedule(sigmas):
    model = BaseModel()
    with sampling.anchor_ring_schedule(guider_for(model), sigmas):
        assert "scale_latent_inpaint" not in vars(model)


def test_skips_a_guider_without_a_model():
    guider = SimpleNamespace(model_patcher=SimpleNamespace(model=None))
    with sampling.anchor_ring_schedule(guider, SIGMAS):
        pass


# ---------------------------------------------------------------- it never leaks

def test_restores_the_model_on_exit():
    model = BaseModel()
    with sampling.anchor_ring_schedule(guider_for(model), SIGMAS):
        pass
    assert "scale_latent_inpaint" not in vars(model)
    assert model.scale_latent_inpaint.__self__ is model


def test_restores_the_model_when_sampling_raises():
    # Core caches model objects for the whole session, so a leaked patch would follow the
    # model into every other node in the graph.
    model = BaseModel()
    with pytest.raises(RuntimeError), sampling.anchor_ring_schedule(guider_for(model), SIGMAS):
        raise RuntimeError("sampler exploded")
    assert "scale_latent_inpaint" not in vars(model)


def test_patches_the_instance_never_the_class():
    # A class patch would follow the cached model into every other node in the graph; a second
    # model of the same class must be completely unaffected while the first one is patched.
    original = vars(BaseModel)["scale_latent_inpaint"]
    model = BaseModel()
    other = BaseModel()
    with sampling.anchor_ring_schedule(guider_for(model), SIGMAS):
        assert vars(BaseModel)["scale_latent_inpaint"] is original
        assert "scale_latent_inpaint" not in vars(other)


# ---------------------------------------------------------------- what the model receives

def test_ring_is_presented_at_the_scheduled_sigma():
    model = BaseModel()
    noise = torch.ones(1, 4, 8, 8)
    latent = torch.zeros(1, 4, 8, 8)
    sigma = torch.tensor([0.5])
    with sampling.anchor_ring_schedule(guider_for(model), SIGMAS):
        model.scale_latent_inpaint(sigma=sigma, noise=noise, latent_image=latent)
    used = model.scaling_calls[-1]
    assert used.reshape(-1)[0].item() == pytest.approx(0.5 * sampling.anchor_ring_factor(0.5))


def test_sigma_is_reshaped_to_broadcast_over_the_latent():
    # comfy model_base.py:409 does this; sigma is per-batch-item and must broadcast over the
    # latent's trailing dims. Without it a B>1 run broadcasts the wrong way and corrupts.
    model = BaseModel()
    noise = torch.ones(3, 4, 8, 8)
    sigma = torch.tensor([0.5, 0.5, 0.5])
    with sampling.anchor_ring_schedule(guider_for(model), SIGMAS):
        model.scale_latent_inpaint(sigma=sigma, noise=noise, latent_image=torch.zeros_like(noise))
    assert tuple(model.scaling_calls[-1].shape) == (3, 1, 1, 1)


def test_five_dimensional_latent_gets_a_five_dimensional_sigma():
    model = BaseModel()
    noise = torch.ones(1, 16, 4, 8, 8)
    sigma = torch.tensor([0.5])
    with sampling.anchor_ring_schedule(guider_for(model), SIGMAS):
        model.scale_latent_inpaint(sigma=sigma, noise=noise, latent_image=torch.zeros_like(noise))
    assert tuple(model.scaling_calls[-1].shape) == (1, 1, 1, 1, 1)


def test_first_step_is_value_identical_to_cores_default():
    # factor == 1.0 at r == 1.0, so the schedule must be a no-op on the first step — the
    # property that makes "matched to the core at step 0" true rather than approximate.
    model = BaseModel()
    noise = torch.randn(1, 4, 8, 8)
    latent = torch.randn(1, 4, 8, 8)
    sigma = SIGMAS[:1].clone()
    expected = model.scale_latent_inpaint(sigma=sigma, noise=noise, latent_image=latent)
    with sampling.anchor_ring_schedule(guider_for(model), SIGMAS):
        got = model.scale_latent_inpaint(sigma=sigma, noise=noise, latent_image=latent)
    assert torch.equal(got, expected)


def test_curve_is_swappable_at_one_point():
    # The A/B harness expresses every arm as a different factor curve here, instead of
    # patching comfy — which the instance patch would silently shadow.
    model = BaseModel()
    noise = torch.ones(1, 4, 8, 8)
    sigma = torch.tensor([0.5])
    original = sampling.anchor_ring_factor
    sampling.anchor_ring_factor = lambda r: 1.0
    try:
        with sampling.anchor_ring_schedule(guider_for(model), SIGMAS):
            model.scale_latent_inpaint(sigma=sigma, noise=noise, latent_image=torch.zeros_like(noise))
    finally:
        sampling.anchor_ring_factor = original
    assert model.scaling_calls[-1].reshape(-1)[0].item() == pytest.approx(0.5)


# --------------------------------------------------------------- the per-node gate

def test_schedule_is_a_noop_when_disabled():
    # enabled=False must leave the model untouched, not merely restore it afterwards: the
    # base node's output has to stay what it was before the ring shipped.
    model = BaseModel()
    with sampling.anchor_ring_schedule(guider_for(model), SIGMAS, enabled=False):
        assert "scale_latent_inpaint" not in vars(model)
        model.scale_latent_inpaint(sigma=torch.tensor([0.5]), noise=torch.ones(1, 4, 8, 8),
                                   latent_image=torch.zeros(1, 4, 8, 8))
    assert model.scaling_calls[-1].reshape(-1)[0].item() == pytest.approx(0.5)


def test_schedule_defaults_to_enabled():
    # Every existing caller passes no flag and must keep the ring.
    model = BaseModel()
    with sampling.anchor_ring_schedule(guider_for(model), SIGMAS):
        assert "scale_latent_inpaint" in vars(model)


def test_sample_latent_forwards_the_gate(comfy_stubs, monkeypatch):
    # The flag has to reach the call site, not just exist on the context manager.
    import contextlib

    seen = []

    @contextlib.contextmanager
    def recording(guider, sigmas, enabled=True):
        seen.append(enabled)
        yield

    monkeypatch.setattr(sampling, "anchor_ring_schedule", recording)

    class Guider:
        def sample(self, *args, **kwargs):
            return torch.zeros(1, 4, 8, 8)

    latent = torch.zeros(1, 4, 8, 8)
    for flag in (True, False):
        sampling.sample_latent(Guider(), object(), SIGMAS, torch.zeros_like(latent), 7, latent,
                               callback=lambda *a: None, anchor_ring=flag)
    assert seen == [True, False]


def test_refine_tiles_gates_the_ring_on_vl_clip(comfy_stubs, monkeypatch):
    # The SHIPPING rule: ring on for the VL nodes (vl_clip present), off for the base node.
    # Asserted through refine_image so a future re-wiring of the argument cannot pass.
    from test_sampling import FakeGuider, FakeNoise, FakeVAE

    from context_anchored_tile_refine import vl

    seen = []
    original = sampling.sample_latent

    def recording(*args, **kwargs):
        seen.append(kwargs.get("anchor_ring"))
        return original(*args, **kwargs)

    monkeypatch.setattr(sampling, "sample_latent", recording)
    monkeypatch.setattr(vl, "build_global_slices",
                        lambda clip, source, tiles, offset_x=0, offset_y=0: [None] * len(tiles))

    def run(**kwargs):
        seen.clear()
        guider = FakeGuider()
        # The VL path requires the CFGGuider conds convention before it will sample at all.
        guider.original_conds = {"positive": [], "negative": []}
        sampling.refine_image(torch.zeros(1, 64, 64, 3), guider, object(), SIGMAS,
                              FakeVAE(), FakeNoise(), 64, 64, 8, 8, **kwargs)
        return seen

    assert run() and all(flag is False for flag in run()), "base node must not get the ring"
    assert all(flag is True for flag in run(vl_clip=object())), "VL node must get the ring"
