import inspect

import pytest
import torch
from test_sampling import FakeGuider, FakeNoise, FakeVAE

from context_anchored_tile_refine import sampling
from context_anchored_tile_refine.node import (
    ContextAnchoredTileRefine,
    ContextAnchoredTileRefineVL,
)


def _refine(image, sigmas=None, mask=None):
    return ContextAnchoredTileRefine().refine(
        image=image,
        guider=FakeGuider(),
        sampler=object(),
        sigmas=sigmas if sigmas is not None else torch.linspace(1.0, 0.0, 5),
        vae=FakeVAE(),
        noise=FakeNoise(),
        max_tile_width=1024,
        max_tile_height=1024,
        context_anchor=64,
        context_overlap=8,
        mask=mask,
    )


def test_no_vl_selects_on_the_base_node():
    # Both VL selects are conditioning decisions this node cannot act on: it never enters the
    # sync engine whose ring source anchor_source picks, and there is no VL CLIP here to slice
    # or to caption with. Absent from the widget list AND from refine(), so nothing can pass
    # either in by accident.
    input_types = ContextAnchoredTileRefine.INPUT_TYPES()
    all_inputs = {**input_types["required"], **input_types["optional"]}
    parameters = inspect.signature(ContextAnchoredTileRefine.refine).parameters
    for widget in ("anchor_source", "vlm_method"):
        assert widget not in all_inputs, widget
        assert widget not in parameters, widget


@pytest.mark.parametrize("choice", ["source image", "live canvas"])
@pytest.mark.parametrize("method", ["vision tokens", "vision tokens and captions", "captions"])
def test_vl_node_forwards_its_widgets(monkeypatch, choice, method):
    # The VL refine node's own refine() is driven nowhere else in the suite, so a parameter
    # renamed on one side of the ComfyUI keyword call would only fail in a real workflow.
    recorded = {}

    def fake_refine_image(image, guider, sampler, sigmas, vae, noise, max_tile_width, max_tile_height, context_anchor, context_overlap, mask=None, vl_clip=None, vlm_method=None, anchor_source=None, sampler_name=None):
        recorded.update(context_anchor=context_anchor, vl_clip=vl_clip,
                        anchor_source=anchor_source, vlm_method=vlm_method)
        return image

    monkeypatch.setattr(sampling, "refine_image", fake_refine_image)
    clip = object()

    result = ContextAnchoredTileRefineVL().refine(
        image=torch.rand(1, 96, 104, 3),
        guider=FakeGuider(),
        sampler=object(),
        sigmas=torch.linspace(1.0, 0.0, 5),
        vae=FakeVAE(),
        noise=FakeNoise(),
        max_tile_width=1024,
        max_tile_height=1024,
        context_anchor=64,
        context_overlap=8,
        anchor_source=choice,
        vlm_method=method,
        clip=clip,
        mask=None,
    )

    assert isinstance(result, tuple) and len(result) == 1
    assert recorded == {"context_anchor": 64, "vl_clip": clip, "anchor_source": choice,
                        "vlm_method": method}


def test_connected_mask_refines(comfy_stubs):
    # A connected mask now runs the region-mask path instead of raising: crop to the
    # masked region + context_anchor, refine, composite back. Output keeps image shape.
    image = torch.rand(1, 96, 104, 3)
    mask = torch.zeros(1, 96, 104)
    mask[:, 32:64, 40:72] = 1.0

    result = _refine(image, mask=mask)

    assert isinstance(result, tuple) and len(result) == 1
    assert result[0].shape == (1, 96, 104, 3)


def test_2d_mask_normalized_to_3d(comfy_stubs):
    # A [H,W] mask is normalized to [1,H,W] and runs the region path.
    image = torch.rand(1, 96, 104, 3)
    mask = torch.zeros(96, 104)
    mask[32:64, 40:72] = 1.0

    result = _refine(image, mask=mask)

    assert result[0].shape == (1, 96, 104, 3)


def test_b1_mask_broadcasts_to_image_batch(comfy_stubs):
    # A B==1 mask broadcasts to a B>1 image (no batch-mismatch error).
    image = torch.rand(2, 96, 104, 3)
    mask = torch.zeros(1, 96, 104)
    mask[:, 32:64, 40:72] = 1.0

    result = _refine(image, mask=mask)

    assert result[0].shape == (2, 96, 104, 3)


def test_rejects_mask_spatial_mismatch():
    with pytest.raises(ValueError, match="must match image size"):
        _refine(torch.rand(1, 96, 104, 3), mask=torch.ones(1, 80, 104))


def test_rejects_mask_batch_mismatch():
    with pytest.raises(ValueError, match="mask batch"):
        _refine(torch.rand(2, 96, 104, 3), mask=torch.ones(3, 96, 104))


def test_rejects_non_4d_image():
    with pytest.raises(ValueError):
        _refine(torch.rand(96, 104, 3))


@pytest.mark.parametrize("shape", [(1, 4, 104, 3), (1, 96, 4, 3)])
def test_rejects_sub_8_image(shape):
    with pytest.raises(ValueError):
        _refine(torch.rand(*shape))


def test_empty_sigmas_returns_clone_untouched():
    image = torch.rand(2, 96, 104, 3)
    before = image.clone()

    result = _refine(image, sigmas=torch.empty(0))

    assert isinstance(result, tuple) and len(result) == 1
    assert torch.equal(result[0], image)
    assert result[0] is not image
    assert torch.equal(image, before)


def test_full_fake_pipeline_returns_image_tuple(comfy_stubs):
    image = torch.rand(1, 100, 101, 3)

    result = _refine(image)

    assert isinstance(result, tuple) and len(result) == 1
    assert isinstance(result[0], torch.Tensor)
    assert result[0].shape == (1, 100, 101, 3)


def test_multi_tile_smoke(comfy_stubs):
    # refine() sits below VALIDATE_INPUTS, so tiny caps force a 3x3 grid with fakes.
    # ctx=overlap=0 is the internal escape hatch (a ring at caps 32 would correctly raise
    # GridConfigError); feather-active coverage lives in test_tiling.
    guider, vae, noise = FakeGuider(), FakeVAE(), FakeNoise()

    result = ContextAnchoredTileRefine().refine(
        image=torch.rand(1, 80, 80, 3),
        guider=guider,
        sampler=object(),
        sigmas=torch.linspace(1.0, 0.0, 5),
        vae=vae,
        noise=noise,
        max_tile_width=32,
        max_tile_height=32,
        context_anchor=0,
        context_overlap=0,
        mask=None,
    )

    assert isinstance(result, tuple) and len(result) == 1
    assert result[0].shape == (1, 80, 80, 3)
    assert vae.encode_calls == 9 and vae.decode_calls == 9
    assert guider.sample_calls == 9
    assert len(noise.calls) == 1
