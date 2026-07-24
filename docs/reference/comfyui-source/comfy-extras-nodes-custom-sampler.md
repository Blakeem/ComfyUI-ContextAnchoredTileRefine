# comfy_extras/nodes_custom_sampler.py — Noise objects, guider wrapper nodes, SamplerCustomAdvanced

**Source:** `comfy_extras/nodes_custom_sampler.py` (see per-section line ranges below)
**ComfyUI version:** 0.3.45 (local source tree — `C:\Users\Blake\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI`)
**Retrieved:** 2026-07-22

Extracted for: the `Noise_RandomNoise` / `Noise_EmptyNoise` objects and their `generate_noise()`
contract, the `RandomNoise` node, the GUIDER wrapper nodes (`CFGGuider`, `BasicGuider`,
`DualCFGGuider` and their `Guider_*` implementation classes), and `SamplerCustomAdvanced` (the
node this project's interface mirrors: `noise` from `RandomNoise`, `guider` from a GUIDER-wrapper
node, `sampler`, `sigmas`, `latent_image`).

Only the sections relevant to those topics are captured here; scheduler nodes (`BasicScheduler`,
`KarrasScheduler`, etc.), `SamplerCustom` (the older non-GUIDER node), and the individual
`Sampler*` selector nodes (`SamplerEulerAncestral`, `SamplerDPMPP_2M_SDE`, etc.) are omitted as out
of scope for this project.

## Noise_EmptyNoise / Noise_RandomNoise (lines 597–613)

```python
class Noise_EmptyNoise:
    def __init__(self):
        self.seed = 0

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        return torch.zeros(latent_image.shape, dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")

class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)
```

## Guider_Basic / BasicGuider (lines 669–690)

```python
class Guider_Basic(comfy.samplers.CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})

class BasicGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "conditioning": ("CONDITIONING", ),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, conditioning):
        guider = Guider_Basic(model)
        guider.set_conds(conditioning)
        return (guider,)
```

## CFGGuider node (lines 692–712)

```python
class CFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, cfg):
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        return (guider,)
```

(Note: this node class is also named `CFGGuider`; the underlying class it wraps is qualified as
`comfy.samplers.CFGGuider` — see `comfy-samplers-cfgguider-and-pipeline.md`.)

## Guider_DualCFG / DualCFGGuider (lines 714–766)

```python
class Guider_DualCFG(comfy.samplers.CFGGuider):
    def set_cfg(self, cfg1, cfg2, nested=False):
        self.cfg1 = cfg1
        self.cfg2 = cfg2
        self.nested = nested

    def set_conds(self, positive, middle, negative):
        middle = node_helpers.conditioning_set_values(middle, {"prompt_type": "negative"})
        self.inner_set_conds({"positive": positive, "middle": middle, "negative": negative})

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        negative_cond = self.conds.get("negative", None)
        middle_cond = self.conds.get("middle", None)
        positive_cond = self.conds.get("positive", None)

        if self.nested:
            out = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond, middle_cond, positive_cond], x, timestep, model_options)
            pred_text = comfy.samplers.cfg_function(self.inner_model, out[2], out[1], self.cfg1, x, timestep, model_options=model_options, cond=positive_cond, uncond=middle_cond)
            return out[0] + self.cfg2 * (pred_text - out[0])
        else:
            if model_options.get("disable_cfg1_optimization", False) == False:
                if math.isclose(self.cfg2, 1.0):
                    negative_cond = None
                    if math.isclose(self.cfg1, 1.0):
                        middle_cond = None

            out = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond, middle_cond, positive_cond], x, timestep, model_options)
            return comfy.samplers.cfg_function(self.inner_model, out[1], out[0], self.cfg2, x, timestep, model_options=model_options, cond=middle_cond, uncond=negative_cond) + (out[2] - out[1]) * self.cfg1

class DualCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "cond1": ("CONDITIONING", ),
                    "cond2": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg_conds": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "cfg_cond2_negative": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "style": (["regular", "nested"],),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, cond1, cond2, negative, cfg_conds, cfg_cond2_negative, style):
        guider = Guider_DualCFG(model)
        guider.set_conds(cond1, cond2, negative)
        guider.set_cfg(cfg_conds, cfg_cond2_negative, nested=(style == "nested"))
        return (guider,)
```

## DisableNoise / RandomNoise (lines 768–798)

```python
class DisableNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
                     }
                }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = "sampling/custom_sampling/noise"

    def get_noise(self):
        return (Noise_EmptyNoise(),)

class RandomNoise(DisableNoise):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
            }
        }

    def get_noise(self, noise_seed):
        return (Noise_RandomNoise(noise_seed),)
```

## SamplerCustomAdvanced (lines 801–845)

```python
class SamplerCustomAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"noise": ("NOISE", ),
                    "guider": ("GUIDER", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"

    def sample(self, noise, guider, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return (out, out_denoised)
```

See also `comfy-sample.md` (`fix_empty_latent_channels`), `nodes-vae-encode-decode-mask.md`
(`SetLatentNoiseMask`, source of `latent["noise_mask"]`), and
`comfy-samplers-cfgguider-and-pipeline.md` (`CFGGuider.sample` signature).

## Node registration (NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS) — relevant entries (lines 893–933)

```python
NODE_CLASS_MAPPINGS = {
    "SamplerCustom": SamplerCustom,
    "BasicScheduler": BasicScheduler,
    "KarrasScheduler": KarrasScheduler,
    "ExponentialScheduler": ExponentialScheduler,
    "PolyexponentialScheduler": PolyexponentialScheduler,
    "LaplaceScheduler": LaplaceScheduler,
    "VPScheduler": VPScheduler,
    "BetaSamplingScheduler": BetaSamplingScheduler,
    "SDTurboScheduler": SDTurboScheduler,
    "KSamplerSelect": KSamplerSelect,
    "SamplerEulerAncestral": SamplerEulerAncestral,
    "SamplerEulerAncestralCFGPP": SamplerEulerAncestralCFGPP,
    "SamplerLMS": SamplerLMS,
    "SamplerDPMPP_3M_SDE": SamplerDPMPP_3M_SDE,
    "SamplerDPMPP_2M_SDE": SamplerDPMPP_2M_SDE,
    "SamplerDPMPP_SDE": SamplerDPMPP_SDE,
    "SamplerDPMPP_2S_Ancestral": SamplerDPMPP_2S_Ancestral,
    "SamplerDPMAdaptative": SamplerDPMAdaptative,
    "SamplerER_SDE": SamplerER_SDE,
    "SamplerSASolver": SamplerSASolver,
    "SplitSigmas": SplitSigmas,
    "SplitSigmasDenoise": SplitSigmasDenoise,
    "FlipSigmas": FlipSigmas,
    "SetFirstSigma": SetFirstSigma,
    "ExtendIntermediateSigmas": ExtendIntermediateSigmas,
    "SamplingPercentToSigma": SamplingPercentToSigma,

    "CFGGuider": CFGGuider,
    "DualCFGGuider": DualCFGGuider,
    "BasicGuider": BasicGuider,
    "RandomNoise": RandomNoise,
    "DisableNoise": DisableNoise,
    "AddNoise": AddNoise,
    "SamplerCustomAdvanced": SamplerCustomAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerEulerAncestralCFGPP": "SamplerEulerAncestralCFG++",
}
```
