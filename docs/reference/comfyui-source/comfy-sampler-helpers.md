# comfy/sampler_helpers.py — prepare_mask, prepare_sampling, cleanup_models, convert_cond

**Source**: `comfy/sampler_helpers.py` (local ComfyUI Desktop install,
`C:\Users\Blake\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI`)
**Version**: ComfyUI 0.3.45
**Retrieved**: 2026-07-22

The helper bodies referenced from `CFGGuider.outer_sample` / `inner_set_conds`
(see `comfy-samplers-cfgguider-and-pipeline.md`). Captured verbatim; line ranges cited per block.

## prepare_mask — lines 16–17

The full `noise_mask` → `denoise_mask` production step: reshape to the latent's shape, move to device.

```python
def prepare_mask(noise_mask, shape, device):
    return comfy.utils.reshape_mask(noise_mask, shape).to(device)
```

(`comfy.utils.reshape_mask` is captured in `comfy-utils-progressbar-upscale-mask.md`.)

## convert_cond — lines 57–67

Converts node-level `[[cross_attn, dict], ...]` cond lists into the internal dict form used by the
sampling pipeline (`CFGGuider.inner_set_conds` calls this).

```python
def convert_cond(cond):
    out = []
    for c in cond:
        temp = c[1].copy()
        model_conds = temp.get("model_conds", {})
        if c[0] is not None:
            temp["cross_attn"] = c[0]
        temp["model_conds"] = model_conds
        temp["uuid"] = uuid.uuid4()
        out.append(temp)
    return out
```

## prepare_sampling / _prepare_sampling — lines 125–141

Called at the top of `CFGGuider.outer_sample`: gathers additional models (controlnets/gligen/hooks),
estimates memory, and loads everything onto GPU via `comfy.model_management.load_models_gpu`.

```python
def prepare_sampling(model: ModelPatcher, noise_shape, conds, model_options=None):
    executor = comfy.patcher_extension.WrapperExecutor.new_executor(
        _prepare_sampling,
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING, model_options, is_model_options=True)
    )
    return executor.execute(model, noise_shape, conds, model_options=model_options)

def _prepare_sampling(model: ModelPatcher, noise_shape, conds, model_options=None):
    real_model: BaseModel = None
    models, inference_memory = get_additional_models(conds, model.model_dtype())
    models += get_additional_models_from_model_options(model_options)
    models += model.get_nested_additional_models()  # TODO: does this require inference_memory update?
    memory_required, minimum_memory_required = estimate_memory(model, noise_shape, conds)
    comfy.model_management.load_models_gpu([model] + models, memory_required=memory_required + inference_memory, minimum_memory_required=minimum_memory_required + inference_memory)
    real_model = model.model

    return real_model, conds, models
```

## cleanup_models / cleanup_additional_models — lines 103–107, 143–150

Called at the end of `CFGGuider.outer_sample` (finally-block): cleans up per-sample additional
models (controlnets etc.). Note it does NOT unload the diffusion model itself — residency is
managed by `comfy.model_management` (see `comfy-model-management-device-offload.md`), which is why
calling `guider.sample(...)` in a per-tile loop is safe and does not thrash full model reloads.

```python
def cleanup_additional_models(models):
    """cleanup additional models that were loaded"""
    for m in models:
        if hasattr(m, 'cleanup'):
            m.cleanup()
```

```python
def cleanup_models(conds, models):
    cleanup_additional_models(models)

    control_cleanup = []
    for k in conds:
        control_cleanup += get_models_from_cond(conds[k], "control")

    cleanup_additional_models(set(control_cleanup))
```

---

*Mechanism summary (gatherer-authored, not source text):* the chain the brief's topic 3 names is:
`SamplerCustomAdvanced.sample` reads `latent["noise_mask"]` → passes it as
`guider.sample(..., denoise_mask=noise_mask, ...)` → `CFGGuider.outer_sample` calls
`prepare_mask(denoise_mask, noise.shape, device)` (reshape to `[B,1,H,W]` latent shape + `.to(device)`)
→ `inner_sample` hands it to the sampler as `denoise_mask`, where `KSamplerX0Inpaint.__call__`
applies it each step — first routing it through `model_options["denoise_mask_function"]` when set
(the Differential Diffusion hook point).
