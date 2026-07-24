# comfy_extras/nodes_differential_diffusion.py — complete file

**Source:** `comfy_extras/nodes_differential_diffusion.py` (lines 1–43, entire file)
**ComfyUI version:** 0.3.45 (local source tree — `C:\Users\Blake\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI`)
**Retrieved:** 2026-07-22

This is the reference implementation for a per-pixel progressive-denoise-mask ("threshold
release") gradient — the exact mechanism this project's node mirrors for its own gradient-driven
denoise mask. Captured in full; the file is 43 lines.

```python
# code adapted from https://github.com/exx8/differential-diffusion

import torch

class DifferentialDiffusion():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "_for_testing"
    INIT = False

    def apply(self, model):
        model = model.clone()
        model.set_model_denoise_mask_function(self.forward)
        return (model,)

    def forward(self, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to)

        return (denoise_mask >= threshold).to(denoise_mask.dtype)

NODE_CLASS_MAPPINGS = {
    "DifferentialDiffusion": DifferentialDiffusion,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DifferentialDiffusion": "Differential Diffusion",
}
```

`apply()`'s `model.set_model_denoise_mask_function(...)` is the `ModelPatcher` method (in
`comfy/model_patcher.py`, not in this brief's listed surfaces) that installs the
`model_options["denoise_mask_function"]` hook consumed at the call site documented in
`comfy-samplers-cfgguider-and-pipeline.md` (`KSamplerX0Inpaint.__call__`).
