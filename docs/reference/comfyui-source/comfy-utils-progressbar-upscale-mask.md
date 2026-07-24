# comfy/utils.py — ProgressBar, common_upscale, tiled-scale helpers, reshape_mask

**Source:** `comfy/utils.py` (see per-section line ranges below)
**ComfyUI version:** 0.3.45 (local source tree — `C:\Users\Blake\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI`)
**Retrieved:** 2026-07-22

Extracted for: `ProgressBar` (long-running node progress reporting), `common_upscale` (reference
upscale idiom), `get_tiled_scale_steps`/`tiled_scale` (used by `VAE.decode_tiled_`/`encode_tiled_`
— see `comfy-sd-vae.md`), and `reshape_mask` (used internally by
`comfy.sampler_helpers.prepare_mask`, called from `CFGGuider.outer_sample` — see
`comfy-samplers-cfgguider-and-pipeline.md`).

## common_upscale (lines 841–873)

```python
def common_upscale(samples, width, height, upscale_method, crop):
        orig_shape = tuple(samples.shape)
        if len(orig_shape) > 4:
            samples = samples.reshape(samples.shape[0], samples.shape[1], -1, samples.shape[-2], samples.shape[-1])
            samples = samples.movedim(2, 1)
            samples = samples.reshape(-1, orig_shape[1], orig_shape[-2], orig_shape[-1])
        if crop == "center":
            old_width = samples.shape[-1]
            old_height = samples.shape[-2]
            old_aspect = old_width / old_height
            new_aspect = width / height
            x = 0
            y = 0
            if old_aspect > new_aspect:
                x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
            elif old_aspect < new_aspect:
                y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
            s = samples.narrow(-2, y, old_height - y * 2).narrow(-1, x, old_width - x * 2)
        else:
            s = samples

        if upscale_method == "bislerp":
            out = bislerp(s, width, height)
        elif upscale_method == "lanczos":
            out = lanczos(s, width, height)
        else:
            out = torch.nn.functional.interpolate(s, size=(height, width), mode=upscale_method)

        if len(orig_shape) == 4:
            return out

        out = out.reshape((orig_shape[0], -1, orig_shape[1]) + (height, width))
        return out.movedim(2, 1).reshape(orig_shape[:-2] + (height, width))
```

## get_tiled_scale_steps / tiled_scale (lines 875–878, 991–992)

```python
def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    rows = 1 if height <= tile_y else math.ceil((height - overlap) / (tile_y - overlap))
    cols = 1 if width <= tile_x else math.ceil((width - overlap) / (tile_x - overlap))
    return rows * cols
```

```python
def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap = 8, upscale_amount = 4, out_channels = 3, output_device="cpu", pbar = None):
    return tiled_scale_multidim(samples, function, (tile_y, tile_x), overlap=overlap, upscale_amount=upscale_amount, out_channels=out_channels, output_device=output_device, pbar=pbar)
```

(`tiled_scale_multidim`, the general N-dimensional implementation `tiled_scale` delegates to, is
defined at lines 881–990 — omitted here as it is long and not directly named by the brief; see
`comfy/sd.py`'s tiled encode/decode methods in `comfy-sd-vae.md` for its call sites and parameters
in context.)

## ProgressBar (lines 994–1022)

```python
PROGRESS_BAR_ENABLED = True
def set_progress_bar_enabled(enabled):
    global PROGRESS_BAR_ENABLED
    PROGRESS_BAR_ENABLED = enabled

PROGRESS_BAR_HOOK = None
def set_progress_bar_global_hook(function):
    global PROGRESS_BAR_HOOK
    PROGRESS_BAR_HOOK = function

class ProgressBar:
    def __init__(self, total, node_id=None):
        global PROGRESS_BAR_HOOK
        self.total = total
        self.current = 0
        self.hook = PROGRESS_BAR_HOOK
        self.node_id = node_id

    def update_absolute(self, value, total=None, preview=None):
        if total is not None:
            self.total = total
        if value > self.total:
            value = self.total
        self.current = value
        if self.hook is not None:
            self.hook(self.current, self.total, preview, node_id=self.node_id)

    def update(self, value):
        self.update_absolute(self.current + value)
```

## reshape_mask (lines 1024–1043)

```python
def reshape_mask(input_mask, output_shape):
    dims = len(output_shape) - 2

    if dims == 1:
        scale_mode = "linear"

    if dims == 2:
        input_mask = input_mask.reshape((-1, 1, input_mask.shape[-2], input_mask.shape[-1]))
        scale_mode = "bilinear"

    if dims == 3:
        if len(input_mask.shape) < 5:
            input_mask = input_mask.reshape((1, 1, -1, input_mask.shape[-2], input_mask.shape[-1]))
        scale_mode = "trilinear"

    mask = torch.nn.functional.interpolate(input_mask, size=output_shape[2:], mode=scale_mode)
    if mask.shape[1] < output_shape[1]:
        mask = mask.repeat((1, output_shape[1]) + (1,) * dims)[:,:output_shape[1]]
    mask = repeat_to_batch_size(mask, output_shape[0])
    return mask
```
