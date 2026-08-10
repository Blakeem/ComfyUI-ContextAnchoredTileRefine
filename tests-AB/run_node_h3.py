"""End-to-end validation of ContextAnchoredTileUpscaleVLVideo on the cached H3 base.

Drives the PRODUCTION node class (not the spike loop) exactly the way the app
does — model/clip/vae objects handed in, comfy's model management doing every
swap — on the cached 1344x768x124f base at the owner-settled settings
(VL slice, denoise 0.22, 8 steps, res_multistep/simple, 2x, av_latent
connected). Same seed, canvas size and grid as the spike's confirmation run, so
H3N_* frames are directly comparable to H3S_refined_vl_d22_*: identical noise
and conditioning, with the production composite (plateau/falloff feather,
min-error routing, DC match) replacing the spike's linear ramp.

    <venv-python> tests-AB/run_node_h3.py

One GPU sampling job at a time on this machine; check nvidia-smi first.
"""
import faulthandler
import sys
from pathlib import Path

faulthandler.enable()

import spike_h3 as poc  # module import runs the ComfyUI bootstrap
import spike_h3_stress as stress

import torch

import ab_models

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from context_anchored_tile_refine.node import ContextAnchoredTileUpscaleVLVideo

SETTINGS = dict(stress.STRESS_KEY, denoise=0.22, refine_steps=8,
                refine_seed=poc.REFINE_SEED, up_w=stress.UP_W, up_h=stress.UP_H,
                anchor=stress.ANCHOR, overlap=stress.OVERLAP,
                conditioning="vl-slice", run_label="node")


def main():
    import comfy.nested_tensor

    print("ComfyUI root: {} version {}".format(poc.ROOT, poc.ab_env.version(poc.ROOT)),
          flush=True)

    base_video, base_audio = stress.render_base_full(None, force=False)
    av_latent = {"samples": comfy.nested_tensor.NestedTensor(
        (base_video, base_audio))}

    vae = ab_models.load_vae(poc.VAE_NAME)
    base_frames = poc.decode_video(vae, base_video)

    model = ab_models.load_unet(poc.UNET_NAME)
    clip = ab_models.load_clip(poc.CLIP_NAME, poc.CLIP_TYPE)

    node = ContextAnchoredTileUpscaleVLVideo()
    with torch.inference_mode():
        (refined,) = node.refine(
            image=base_frames, model=model, clip=clip, vae=vae,
            seed=poc.REFINE_SEED, sampler_name=poc.SAMPLER, scheduler=poc.SCHEDULER,
            steps=8, denoise=0.22, upscale_by=2.0,
            max_tile_width=1408, max_tile_height=1408,
            context_anchor=stress.ANCHOR, context_overlap=stress.OVERLAP,
            upscale_model=None, av_latent=av_latent)

    del model, clip
    ab_models.free_gpu()
    refined = refined.to(torch.float16)

    for index in (0, refined.shape[0] // 2, refined.shape[0] - 1):
        ab_models.save_png(
            stress.OUTPUT_DIR / "H3N_refined_vl_d22_f{:03d}.png".format(index),
            refined[index:index + 1], SETTINGS)
    stress.save_video_atomic("node_refined_vl_d22", refined, SETTINGS)
    print("done. compare H3N_refined_vl_d22_f*.png vs H3S_refined_vl_d22_f*.png",
          flush=True)


if __name__ == "__main__":
    main()
