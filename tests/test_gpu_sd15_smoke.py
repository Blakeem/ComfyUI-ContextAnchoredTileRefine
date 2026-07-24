from pathlib import Path

import pytest
import torch

from context_anchored_tile_refine.node import ContextAnchoredTileRefine

pytestmark = [pytest.mark.comfy, pytest.mark.gpu, pytest.mark.slow]

CKPT = Path(r"C:\Users\Blake\Documents\ComfyUI\models\checkpoints\v1-5-pruned-emaonly-fp16.safetensors")
STEPS = 6
DENOISE = 0.5


def _gradient_image():
    # Deliberately non-/8 on both axes so pad/crop runs end-to-end against the real VAE.
    height, width = 322, 330
    y = torch.linspace(0.0, 1.0, height).view(height, 1, 1)
    x = torch.linspace(0.0, 1.0, width).view(1, width, 1)
    channels = torch.cat([y.expand(height, width, 1), x.expand(height, width, 1), (y * x).expand(height, width, 1)], dim=-1)
    return channels.unsqueeze(0).contiguous()


def test_sd15_whole_image_refine(comfy_env):
    if not CKPT.is_file():
        pytest.skip("SD1.5 checkpoint not found: {}".format(CKPT))
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    import comfy.samplers
    import comfy.sd
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

    model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(str(CKPT))
    positive = clip.encode_from_tokens_scheduled(clip.tokenize("a sharp, detailed photograph"))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize("blurry, jpeg artifacts"))
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(4.0)
    sampler = comfy.samplers.sampler_object("euler")
    # BasicScheduler-equivalent truncated schedule for denoise < 1.0.
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), "normal", int(STEPS / DENOISE)
    ).cpu()[-(STEPS + 1):]
    image = _gradient_image()

    node = ContextAnchoredTileRefine()
    inputs = {
        "image": image,
        "guider": guider,
        "sampler": sampler,
        "sigmas": sigmas,
        "vae": vae,
        "noise": Noise_RandomNoise(42),
        "max_tile_width": 1024,
        "max_tile_height": 1024,
        "context_anchor": 64,
        "context_overlap": 64,
    }
    (out1,) = node.refine(**inputs)
    (out2,) = node.refine(**inputs)

    assert out1.shape == image.shape
    assert out1.dtype == torch.float32
    assert out1.min() >= 0.0 and out1.max() <= 1.0
    assert not torch.equal(out1.cpu(), image)
    assert torch.equal(out1, out2)  # same process/model/seed → bitwise-identical


def test_sd15_masked_refine(comfy_env):
    if not CKPT.is_file():
        pytest.skip("SD1.5 checkpoint not found: {}".format(CKPT))
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    import comfy.samplers
    import comfy.sd
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

    model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(str(CKPT))
    positive = clip.encode_from_tokens_scheduled(clip.tokenize("a sharp, detailed photograph"))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize("blurry, jpeg artifacts"))
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(4.0)
    sampler = comfy.samplers.sampler_object("euler")
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), "normal", int(STEPS / DENOISE)
    ).cpu()[-(STEPS + 1):]
    image = _gradient_image()

    # Central rectangle mask: the crop is mask + context_anchor, so the top-left corner is
    # far background — it must survive from the ORIGINAL pixels, byte-identical.
    mask = torch.zeros(1, 322, 330)
    mask[:, 128:192, 132:196] = 1.0

    node = ContextAnchoredTileRefine()
    inputs = {
        "image": image,
        "guider": guider,
        "sampler": sampler,
        "sigmas": sigmas,
        "vae": vae,
        "noise": Noise_RandomNoise(42),
        "max_tile_width": 1024,
        "max_tile_height": 1024,
        "context_anchor": 64,
        "context_overlap": 64,
        "mask": mask,
    }
    (out1,) = node.refine(**inputs)
    (out2,) = node.refine(**inputs)

    assert out1.shape == image.shape
    assert out1.dtype == torch.float32
    assert out1.min() >= 0.0 and out1.max() <= 1.0
    assert torch.equal(out1, out2)  # same process/model/seed → bitwise-identical
    out1c = out1.cpu()
    # The masked region was refined (changed from the input).
    assert not torch.equal(out1c[:, 128:192, 132:196, :], image[:, 128:192, 132:196, :])
    # Far background (outside mask + context_anchor) is byte-identical to the input — proves
    # the [B,h,w] region mask survives the real prepare_mask/reshape_mask/KSamplerX0Inpaint
    # path and the background is never re-diffused.
    assert torch.equal(out1c[:, :32, :32, :], image[:, :32, :32, :])


def test_sd15_grid_refine(comfy_env):
    if not CKPT.is_file():
        pytest.skip("SD1.5 checkpoint not found: {}".format(CKPT))
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    import comfy.samplers
    import comfy.sd
    from comfy_extras.nodes_custom_sampler import Noise_RandomNoise

    model, clip, vae, _ = comfy.sd.load_checkpoint_guess_config(str(CKPT))
    positive = clip.encode_from_tokens_scheduled(clip.tokenize("a sharp, detailed photograph"))
    negative = clip.encode_from_tokens_scheduled(clip.tokenize("blurry, jpeg artifacts"))
    guider = comfy.samplers.CFGGuider(model)
    guider.set_conds(positive, negative)
    guider.set_cfg(4.0)
    sampler = comfy.samplers.sampler_object("euler")
    # BasicScheduler-equivalent truncated schedule for denoise < 1.0.
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), "normal", int(STEPS / DENOISE)
    ).cpu()[-(STEPS + 1):]
    image = _gradient_image()

    node = ContextAnchoredTileRefine()
    # 322x330 pads to 328x336; at caps 256 with r = ctx + overlap = 64 both axes solve
    # n=2, base=168, every crop <= 232 <= 256 -> 2x2 feather-active grid (interior seams
    # feathered by the later tile). Seam judgment is visual (user A/B); quantitative
    # metrics land in gpu-validation.
    inputs = {
        "image": image,
        "guider": guider,
        "sampler": sampler,
        "sigmas": sigmas,
        "vae": vae,
        "noise": Noise_RandomNoise(42),
        "max_tile_width": 256,
        "max_tile_height": 256,
        "context_anchor": 32,
        "context_overlap": 32,
    }
    (out1,) = node.refine(**inputs)
    (out2,) = node.refine(**inputs)

    assert out1.shape == image.shape
    assert out1.dtype == torch.float32
    assert out1.min() >= 0.0 and out1.max() <= 1.0
    assert not torch.equal(out1.cpu(), image)
    assert torch.equal(out1, out2)  # same process/model/seed → bitwise-identical
