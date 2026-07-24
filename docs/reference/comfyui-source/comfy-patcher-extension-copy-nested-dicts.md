# `comfy.patcher_extension.copy_nested_dicts` — the implementation behind `create_model_options_clone`

- **Source**: `comfy/patcher_extension.py` (lines 135-157)
- **Repo**: `C:\Users\Blake\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI` (ComfyUI Desktop install tree)
- **Version**: ComfyUI 0.3.45
- **Retrieved**: 2026-07-22

`comfy/model_patcher.py`'s `create_model_options_clone(orig_model_options)` is a one-line
delegation to this function:

```python
def create_model_options_clone(orig_model_options: dict):
    return comfy.patcher_extension.copy_nested_dicts(orig_model_options)
```

Its exact body (and the sibling `merge_nested_dicts`, used elsewhere to combine two
`model_options`/`transformer_options` dicts) is:

```python
def copy_nested_dicts(input_dict: dict):
    new_dict = input_dict.copy()
    for key, value in input_dict.items():
        if isinstance(value, dict):
            new_dict[key] = copy_nested_dicts(value)
        elif isinstance(value, list):
            new_dict[key] = value.copy()
    return new_dict

def merge_nested_dicts(dict1: dict, dict2: dict, copy_dict1=True):
    if copy_dict1:
        merged_dict = copy_nested_dicts(dict1)
    else:
        merged_dict = dict1
    for key, value in dict2.items():
        if isinstance(value, dict):
            curr_value = merged_dict.setdefault(key, {})
            merged_dict[key] = merge_nested_dicts(value, curr_value)
        elif isinstance(value, list):
            merged_dict.setdefault(key, []).extend(value)
        else:
            merged_dict[key] = value
    return merged_dict
```

Semantics that matter for anything installing/reading a `model_options["denoise_mask_function"]`
hook: `copy_nested_dicts` is a **shallow-per-level** recursive copy — it copies the top dict, and
recurses into nested `dict`/`list` values, but any other value (including a function/callable
stored directly under a key, e.g. `denoise_mask_function` itself, or a `torch.Tensor`) is copied
by reference, not deep-copied. This is cheaper than `copy.deepcopy` (used by
`ModelPatcher.clone()`, see the companion file `comfy-model-patcher-denoise-mask-hook.md`) and
is what lets the sampling pipeline clone `model_options` once per `guider.sample(...)` /
`outer_sample` call without deep-copying tensors or the installed hook callables on every call.
