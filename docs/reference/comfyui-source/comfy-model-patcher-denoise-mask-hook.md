# ModelPatcher: the `denoise_mask_function` hook and `model_options` storage/cloning

- **Source**: `comfy/model_patcher.py`
- **Repo**: `C:\Users\Blake\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI` (ComfyUI Desktop install tree)
- **Version**: ComfyUI 0.3.45
- **Retrieved**: 2026-07-22

This is the exact hook-install mechanism `DifferentialDiffusion.apply` uses (via
`model.set_model_denoise_mask_function(...)`, see
`comfy_extras/nodes_differential_diffusion.py`) and the exact storage/clone semantics of
`model_options` that a custom node mirroring this pattern must respect.

## Module-level `model_options` helper functions

`comfy/model_patcher.py` lines 54-96 — free functions that mutate/return a `model_options`
dict. `create_model_options_clone` (used for cheap per-sample-call copies of `model_options`,
e.g. in `CFGGuider`) is the sibling of `set_model_denoise_mask_function` below and is the
function named in scope for this capture.

```python
def set_model_options_patch_replace(model_options, patch, name, block_name, number, transformer_index=None):
    to = model_options["transformer_options"].copy()

    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if name not in to["patches_replace"]:
        to["patches_replace"][name] = {}
    else:
        to["patches_replace"][name] = to["patches_replace"][name].copy()

    if transformer_index is not None:
        block = (block_name, number, transformer_index)
    else:
        block = (block_name, number)
    to["patches_replace"][name][block] = patch
    model_options["transformer_options"] = to
    return model_options

def set_model_options_post_cfg_function(model_options, post_cfg_function, disable_cfg1_optimization=False):
    model_options["sampler_post_cfg_function"] = model_options.get("sampler_post_cfg_function", []) + [post_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options

def set_model_options_pre_cfg_function(model_options, pre_cfg_function, disable_cfg1_optimization=False):
    model_options["sampler_pre_cfg_function"] = model_options.get("sampler_pre_cfg_function", []) + [pre_cfg_function]
    if disable_cfg1_optimization:
        model_options["disable_cfg1_optimization"] = True
    return model_options

def create_model_options_clone(orig_model_options: dict):
    return comfy.patcher_extension.copy_nested_dicts(orig_model_options)

def create_hook_patches_clone(orig_hook_patches):
    new_hook_patches = {}
    for hook_ref in orig_hook_patches:
        new_hook_patches[hook_ref] = {}
        for k in orig_hook_patches[hook_ref]:
            new_hook_patches[hook_ref][k] = orig_hook_patches[hook_ref][k][:]
    return new_hook_patches
```

`create_model_options_clone` delegates entirely to `comfy.patcher_extension.copy_nested_dicts`
(imported at the top of this file as `import comfy.patcher_extension`) — see the companion file
`comfy-patcher-extension-copy-nested-dicts.md` in this folder for that function's exact body.

## `ModelPatcher.__init__` — where `model_options` (and `denoise_mask_function`'s home) is created

`comfy/model_patcher.py` lines 204-256:

```python
class ModelPatcher:
    def __init__(self, model, load_device, offload_device, size=0, weight_inplace_update=False):
        self.size = size
        self.model = model
        if not hasattr(self.model, 'device'):
            logging.debug("Model doesn't have a device attribute.")
            self.model.device = offload_device
        elif self.model.device is None:
            self.model.device = offload_device

        self.patches = {}
        self.backup = {}
        self.object_patches = {}
        self.object_patches_backup = {}
        self.weight_wrapper_patches = {}
        self.model_options = {"transformer_options":{}}
        self.model_size()
        self.load_device = load_device
        self.offload_device = offload_device
        self.weight_inplace_update = weight_inplace_update
        self.force_cast_weights = False
        self.patches_uuid = uuid.uuid4()
        self.parent = None

        self.attachments: dict[str] = {}
        self.additional_models: dict[str, list[ModelPatcher]] = {}
        self.callbacks: dict[str, dict[str, list[Callable]]] = CallbacksMP.init_callbacks()
        self.wrappers: dict[str, dict[str, list[Callable]]] = WrappersMP.init_wrappers()

        self.is_injected = False
        self.skip_injection = False
        self.injections: dict[str, list[PatcherInjection]] = {}

        self.hook_patches: dict[comfy.hooks._HookRef] = {}
        self.hook_patches_backup: dict[comfy.hooks._HookRef] = None
        self.hook_backup: dict[str, tuple[torch.Tensor, torch.device]] = {}
        self.cached_hook_patches: dict[comfy.hooks.HookGroup, dict[str, torch.Tensor]] = {}
        self.current_hooks: Optional[comfy.hooks.HookGroup] = None
        self.forced_hooks: Optional[comfy.hooks.HookGroup] = None  # NOTE: only used for CLIP at this time
        self.is_clip = False
        self.hook_mode = comfy.hooks.EnumHookMode.MaxSpeed

        if not hasattr(self.model, 'model_loaded_weight_memory'):
            self.model.model_loaded_weight_memory = 0

        if not hasattr(self.model, 'lowvram_patch_counter'):
            self.model.lowvram_patch_counter = 0

        if not hasattr(self.model, 'model_lowvram'):
            self.model.model_lowvram = False

        if not hasattr(self.model, 'current_weight_patches_uuid'):
            self.model.current_weight_patches_uuid = None
```

`model_options` starts as a plain dict with only `transformer_options` pre-seeded — every other
key (`denoise_mask_function`, `sampler_cfg_function`, `sampler_post_cfg_function`, etc.) is added
later, only if some node calls the corresponding `set_model_*` method.

## `ModelPatcher.clone()` — the full-patcher clone that deep-copies `model_options`

`comfy/model_patcher.py` lines 270-326. This is the clone path used when a node calls
`model.clone()` (e.g. before installing per-workflow hooks so the original patcher/model_options
is left untouched). Note line `n.model_options = copy.deepcopy(self.model_options)` — a full
`copy.deepcopy`, distinct from the shallower `create_model_options_clone` /
`copy_nested_dicts` used for the cheaper, per-sampling-call copies made inside the sampling
pipeline (`comfy/samplers.py`, out of scope for this file's capture).

```python
    def clone(self):
        n = self.__class__(self.model, self.load_device, self.offload_device, self.size, weight_inplace_update=self.weight_inplace_update)
        n.patches = {}
        for k in self.patches:
            n.patches[k] = self.patches[k][:]
        n.patches_uuid = self.patches_uuid

        n.object_patches = self.object_patches.copy()
        n.weight_wrapper_patches = self.weight_wrapper_patches.copy()
        n.model_options = copy.deepcopy(self.model_options)
        n.backup = self.backup
        n.object_patches_backup = self.object_patches_backup
        n.parent = self

        n.force_cast_weights = self.force_cast_weights

        # attachments
        n.attachments = {}
        for k in self.attachments:
            if hasattr(self.attachments[k], "on_model_patcher_clone"):
                n.attachments[k] = self.attachments[k].on_model_patcher_clone()
            else:
                n.attachments[k] = self.attachments[k]
        # additional models
        for k, c in self.additional_models.items():
            n.additional_models[k] = [x.clone() for x in c]
        # callbacks
        for k, c in self.callbacks.items():
            n.callbacks[k] = {}
            for k1, c1 in c.items():
                n.callbacks[k][k1] = c1.copy()
        # sample wrappers
        for k, w in self.wrappers.items():
            n.wrappers[k] = {}
            for k1, w1 in w.items():
                n.wrappers[k][k1] = w1.copy()
        # injection
        n.is_injected = self.is_injected
        n.skip_injection = self.skip_injection
        for k, i in self.injections.items():
            n.injections[k] = i.copy()
        # hooks
        n.hook_patches = create_hook_patches_clone(self.hook_patches)
        n.hook_patches_backup = create_hook_patches_clone(self.hook_patches_backup) if self.hook_patches_backup else self.hook_patches_backup
        for group in self.cached_hook_patches:
            n.cached_hook_patches[group] = {}
            for k in self.cached_hook_patches[group]:
                n.cached_hook_patches[group][k] = self.cached_hook_patches[group][k]
        n.hook_backup = self.hook_backup
        n.current_hooks = self.current_hooks.clone() if self.current_hooks else self.current_hooks
        n.forced_hooks = self.forced_hooks.clone() if self.forced_hooks else self.forced_hooks
        n.is_clip = self.is_clip
        n.hook_mode = self.hook_mode

        for callback in self.get_all_callbacks(CallbacksMP.ON_CLONE):
            callback(self, n)
        return n
```

## `set_model_denoise_mask_function` and its sibling hook-setter methods

`comfy/model_patcher.py` lines 368-398 — this is the exact family of one-line
`self.model_options[...] = ...` setter methods on `ModelPatcher`. `set_model_denoise_mask_function`
(the method `DifferentialDiffusion.apply` calls) follows the simplest variant of the pattern: a
plain overwrite of a single `model_options` key, no list-accumulation, no `transformer_options`
nesting.

```python
    def set_model_sampler_cfg_function(self, sampler_cfg_function, disable_cfg1_optimization=False):
        if len(inspect.signature(sampler_cfg_function).parameters) == 3:
            self.model_options["sampler_cfg_function"] = lambda args: sampler_cfg_function(args["cond"], args["uncond"], args["cond_scale"]) #Old way
        else:
            self.model_options["sampler_cfg_function"] = sampler_cfg_function
        if disable_cfg1_optimization:
            self.model_options["disable_cfg1_optimization"] = True

    def set_model_sampler_post_cfg_function(self, post_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_post_cfg_function(self.model_options, post_cfg_function, disable_cfg1_optimization)

    def set_model_sampler_pre_cfg_function(self, pre_cfg_function, disable_cfg1_optimization=False):
        self.model_options = set_model_options_pre_cfg_function(self.model_options, pre_cfg_function, disable_cfg1_optimization)

    def set_model_sampler_calc_cond_batch_function(self, sampler_calc_cond_batch_function):
        self.model_options["sampler_calc_cond_batch_function"] = sampler_calc_cond_batch_function

    def set_model_unet_function_wrapper(self, unet_wrapper_function: UnetWrapperFunction):
        self.model_options["model_function_wrapper"] = unet_wrapper_function

    def set_model_denoise_mask_function(self, denoise_mask_function):
        self.model_options["denoise_mask_function"] = denoise_mask_function

    def set_model_patch(self, patch, name):
        to = self.model_options["transformer_options"]
        if "patches" not in to:
            to["patches"] = {}
        to["patches"][name] = to["patches"].get(name, []) + [patch]

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        self.model_options = set_model_options_patch_replace(self.model_options, patch, name, block_name, number, transformer_index=transformer_index)
```

Note there is no `get_model_options()` accessor anywhere in this file — `model_options` is a
plain public attribute (`self.model_options`) read and passed by reference wherever a
`ModelPatcher` is used for sampling.
