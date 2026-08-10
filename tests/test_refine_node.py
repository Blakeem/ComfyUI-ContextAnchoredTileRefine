import pytest
import torch
from test_sampling import FakeGuider, FakeNoise, FakeVAE

from context_anchored_tile_refine.node import ContextAnchoredTileRefine


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
