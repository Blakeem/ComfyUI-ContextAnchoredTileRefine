"""ContextAnchoredTileUpscaleVL's wiring. The node's whole job is to build the four
custom-sampling inputs from widgets and hand them, plus the UPSCALED image, to the one
existing entry point. So every builder and sampling.refine_image itself is replaced by a
recorder: what is pinned here is which value reaches which parameter, not any pixel math
(that is covered by test_upscale / test_tiling / test_vl against the real functions)."""
import pytest
import torch

from context_anchored_tile_refine import sampling, upscale
from context_anchored_tile_refine.node import ContextAnchoredTileUpscaleVL

WIDGETS = {
    "seed": 1234,
    "sampler_name": "dpmpp_2m",
    "scheduler": "sgm_uniform",
    "steps": 20,
    "cfg": 3.5,
    "denoise": 0.5,
    "upscale_by": 2.0,
    "max_tile_width": 1536,
    "max_tile_height": 2048,
    "context_anchor": 32,
    "context_overlap": 32,
    "anchor_source": "source image",
    "vlm_method": "vision tokens and captions",
}


class FakeGuider:
    def __init__(self, model, positive, negative, cfg):
        self.model = model
        self.positive = positive
        self.negative = negative
        self.cfg = cfg


def _drive(monkeypatch, image=None, upscale_model=None, negative=None, upscaled=None, **overrides):
    """Run refine() with every collaborator faked; return (recorded, result)."""
    recorded = {
        "prepare_upscaled": None,
        "encode_empty": None,
        "build_guider": None,
        "build_sigmas": None,
        "refine_image": None,
        "upscaled": torch.rand(1, 64, 64, 3) if upscaled is None else upscaled,
        "empty_cond": [("empty", {})],
        "sigmas": torch.linspace(1.0, 0.0, 5),
        "refined": torch.rand(1, 64, 64, 3),
    }

    def fake_prepare_upscaled(image, upscale_model, upscale_by, progress=None):
        recorded["prepare_upscaled"] = (image, upscale_model, upscale_by)
        recorded["upscale_progress"] = progress
        return recorded["upscaled"]

    def fake_encode_empty(clip):
        recorded["encode_empty"] = clip
        return recorded["empty_cond"]

    def fake_build_guider(model, positive, negative, cfg):
        recorded["build_guider"] = (model, positive, negative, cfg)
        return FakeGuider(model, positive, negative, cfg)

    def fake_build_sigmas(model, scheduler, steps, denoise):
        recorded["build_sigmas"] = (model, scheduler, steps, denoise)
        return recorded["sigmas"]

    def fake_refine_image(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=None, vl_clip=None, vlm_method=None, anchor_source=None, sampler_name=None, progress=None):
        recorded["refine_image"] = {
            "image": image,
            "guider": guider,
            "sampler": sampler,
            "sigmas": sigmas,
            "vae": vae,
            "noise": noise,
            "max_tile_width": max_tile_width,
            "max_tile_height": max_tile_height,
            "context_anchor": context_anchor,
            "context_overlap": context_overlap,
            "mask": mask,
            "vl_clip": vl_clip,
            "anchor_source": anchor_source,
            "vlm_method": vlm_method,
            "sampler_name": sampler_name,
            "progress": progress,
        }
        return recorded["refined"]

    monkeypatch.setattr(upscale, "prepare_upscaled", fake_prepare_upscaled)
    monkeypatch.setattr(upscale, "encode_empty", fake_encode_empty)
    monkeypatch.setattr(upscale, "build_guider", fake_build_guider)
    monkeypatch.setattr(upscale, "build_sigmas", fake_build_sigmas)
    monkeypatch.setattr(sampling, "refine_image", fake_refine_image)

    recorded["image"] = torch.rand(1, 32, 32, 3) if image is None else image
    recorded["model"] = object()
    recorded["clip"] = object()
    recorded["vae"] = object()

    widgets = dict(WIDGETS)
    widgets.update(overrides)
    result = ContextAnchoredTileUpscaleVL().refine(
        image=recorded["image"],
        model=recorded["model"],
        clip=recorded["clip"],
        vae=recorded["vae"],
        upscale_model=upscale_model,
        negative=negative,
        **widgets,
    )
    return recorded, result


def test_an_upscale_result_below_8px_is_rejected_naming_upscale_by(comfy_stubs, monkeypatch):
    # The upscale REPLACES the validated input and upscale_by goes down to 0.01, so the
    # tensor reaching refine_image can be smaller than the /8 reflect pad allows. Without
    # this guard torch raises a padding error naming neither the node nor the widget.
    with pytest.raises(ValueError, match=r"upscale_by 0\.01 takes the 400x400 input to 4x4"):
        _drive(monkeypatch, image=torch.rand(1, 400, 400, 3), upscaled=torch.rand(1, 4, 4, 3), upscale_by=0.01)


def test_returns_a_one_tuple_of_refine_images_result(comfy_stubs, monkeypatch):
    recorded, result = _drive(monkeypatch)

    assert isinstance(result, tuple) and len(result) == 1
    assert result[0] is recorded["refined"]


def test_refine_image_receives_the_upscaled_image_not_the_input(comfy_stubs, monkeypatch):
    # The whole point of the upscale stage: tiling must see the upscaled pixels.
    upscale_model = object()
    recorded, _ = _drive(monkeypatch, upscale_model=upscale_model)

    assert recorded["prepare_upscaled"] == (recorded["image"], upscale_model, 2.0)
    assert recorded["refine_image"]["image"] is recorded["upscaled"]


def test_runs_the_vl_path_with_no_mask(comfy_stubs, monkeypatch):
    recorded, _ = _drive(monkeypatch)

    assert recorded["refine_image"]["mask"] is None
    assert recorded["refine_image"]["vl_clip"] is recorded["clip"]
    # The same CLIP produces the placeholder positive and the per-tile vision slices.
    assert recorded["encode_empty"] is recorded["clip"]


def test_geometry_widgets_pass_through_unchanged(comfy_stubs, monkeypatch):
    recorded, _ = _drive(monkeypatch)
    call = recorded["refine_image"]

    assert call["vae"] is recorded["vae"]
    assert call["max_tile_width"] == 1536
    assert call["max_tile_height"] == 2048
    assert call["context_anchor"] == 32
    assert call["context_overlap"] == 32
    assert call["sigmas"] is recorded["sigmas"]


@pytest.mark.parametrize("choice", ["source image", "live canvas"])
def test_anchor_source_widget_reaches_refine_image(comfy_stubs, monkeypatch, choice):
    # The widget is inert unless it arrives as refine_image's anchor_source: that string IS
    # the sync engine's mode, so nothing maps or renames it on the way through.
    from context_anchored_tile_refine import sync

    recorded, _ = _drive(monkeypatch, anchor_source=choice)

    assert recorded["refine_image"]["anchor_source"] == choice
    assert choice in sync.ANCHOR_SOURCES


@pytest.mark.parametrize("choice", ["vision tokens", "vision tokens and captions", "captions"])
def test_vlm_method_widget_reaches_refine_image(comfy_stubs, monkeypatch, choice):
    # Same wiring rule as the anchor select: the widget is inert unless it arrives as
    # refine_image's vlm_method, which is the only name the VL pre-pass branches on.
    recorded, _ = _drive(monkeypatch, vlm_method=choice)

    assert recorded["refine_image"]["vlm_method"] == choice


def test_builders_receive_the_model_and_their_widgets(comfy_stubs, monkeypatch):
    recorded, _ = _drive(monkeypatch)

    assert recorded["build_sigmas"] == (recorded["model"], "sgm_uniform", 20, 0.5)
    assert recorded["build_guider"][0] is recorded["model"]
    assert recorded["build_guider"][3] == 3.5
    assert recorded["refine_image"]["guider"].model is recorded["model"]


def test_unconnected_negative_falls_back_to_the_empty_encode(comfy_stubs, monkeypatch):
    recorded, _ = _drive(monkeypatch, negative=None)
    _, positive, negative, _ = recorded["build_guider"]

    assert positive is recorded["empty_cond"]
    assert negative is recorded["empty_cond"]


def test_connected_negative_is_used_verbatim(comfy_stubs, monkeypatch):
    connected = [("negative", {})]
    recorded, _ = _drive(monkeypatch, negative=connected)
    _, positive, negative, _ = recorded["build_guider"]

    assert negative is connected
    # The positive stays the placeholder either way — vl.py overwrites it per tile.
    assert positive is recorded["empty_cond"]


def test_sampler_is_built_from_the_widget_name(comfy_stubs, monkeypatch):
    recorded, _ = _drive(monkeypatch, sampler_name="euler")

    assert comfy_stubs["sampler_object_calls"] == ["euler"]
    # The stub mirrors core: sampler_object returns a KSAMPLER whose sampler_function is
    # sample_<name>, which is the identity the sync stepper resolves its timing table on.
    assert recorded["refine_image"]["sampler"].sampler_function.__name__ == "sample_euler"
    # The NAME rides along too: core wraps several samplers in a private function
    # (dpm_fast -> dpm_fast_function), so a rejection resolved off the object alone would
    # name something this node's widget never offered.
    assert recorded["refine_image"]["sampler_name"] == "euler"


def test_noise_carries_the_seed(comfy_stubs, monkeypatch):
    recorded, _ = _drive(monkeypatch, seed=4242)
    noise = recorded["refine_image"]["noise"]

    assert isinstance(noise, upscale.Noise_RandomNoise)
    assert noise.seed == 4242


def test_one_ledger_is_created_here_and_handed_to_both_stages(comfy_stubs, monkeypatch):
    # LEDGER CREATION SITE: the node owns it, so the upscale stage and the refine engine
    # report into ONE bar. Without this the progress feature would be unreachable from the
    # all-in-one node, which is the node that has the most phases to cover.
    from context_anchored_tile_refine import progress

    recorded, _ = _drive(monkeypatch, upscale_model=object())
    ledger = recorded["upscale_progress"]

    assert isinstance(ledger, progress.Ledger)
    assert recorded["refine_image"]["progress"] is ledger
    assert len(comfy_stubs["progress_bars"]) == 1


def test_the_first_clip_call_gets_its_own_segment_and_the_shim_is_restored(comfy_stubs, monkeypatch):
    # encode_empty is the run's FIRST CLIP call — it pays CLIP.load_model and the move onto
    # the GPU, with nothing else covering it. The shim around the whole run must be gone by
    # the time refine() returns.
    import comfy.utils

    from context_anchored_tile_refine import progress

    real = comfy.utils.ProgressBar
    recorded, _ = _drive(monkeypatch)

    assert progress.CLIP_LOAD in [name for name, _units in recorded["upscale_progress"].segments]
    assert comfy.utils.ProgressBar is real


def test_rejects_a_non_4d_image(comfy_stubs, monkeypatch):
    with pytest.raises(ValueError, match="B,H,W,C"):
        _drive(monkeypatch, image=torch.rand(96, 104, 3))


def test_rejects_a_sub_8_image(comfy_stubs, monkeypatch):
    with pytest.raises(ValueError, match="at least 8x8"):
        _drive(monkeypatch, image=torch.rand(1, 4, 104, 3))
