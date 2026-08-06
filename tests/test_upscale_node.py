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

    def fake_prepare_upscaled(image, upscale_model, upscale_by):
        recorded["prepare_upscaled"] = (image, upscale_model, upscale_by)
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

    def fake_refine_image(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=None, vl_clip=None):
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
    assert recorded["refine_image"]["sampler"] == ("sampler_object", "euler")


def test_noise_carries_the_seed(comfy_stubs, monkeypatch):
    recorded, _ = _drive(monkeypatch, seed=4242)
    noise = recorded["refine_image"]["noise"]

    assert isinstance(noise, upscale.Noise_RandomNoise)
    assert noise.seed == 4242


def test_rejects_a_non_4d_image(comfy_stubs, monkeypatch):
    with pytest.raises(ValueError, match="B,H,W,C"):
        _drive(monkeypatch, image=torch.rand(96, 104, 3))


def test_rejects_a_sub_8_image(comfy_stubs, monkeypatch):
    with pytest.raises(ValueError, match="at least 8x8"):
        _drive(monkeypatch, image=torch.rand(1, 4, 104, 3))
