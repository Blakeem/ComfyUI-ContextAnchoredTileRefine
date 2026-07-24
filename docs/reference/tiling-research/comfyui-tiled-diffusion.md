Source: https://github.com/shiimizu/ComfyUI-TiledDiffusion (README.md; tiled_diffusion.py), branch `main`
Repo: shiimizu/ComfyUI-TiledDiffusion — "Tiled Diffusion & VAE for ComfyUI" (ports the MultiDiffusion /
Mixture-of-Diffusers tiling+blend methods, and the Tiled VAE algorithm, into a ComfyUI node)
Retrieved: 2026-07-22

Note: captures the node's documented options and the seam-blend mechanism as it is actually
wired into ComfyUI (a `model_options` patch invoked from the KSampler's `apply_model`), not the
Tiled VAE half of the repo (out of scope for this doc set — this brief is about tile-to-tile seam
blending, not VAE memory tiling).

## README

This extension enables **large image drawing & upscaling with limited VRAM** via the following techniques:

- Reproduced SOTA Tiled Diffusion methods
    - [MultiDiffusion](https://github.com/omerbt/MultiDiffusion) (arXiv:2302.08113)
    - [Mixture of Diffusers](https://github.com/albarji/mixture-of-diffusers) (arXiv:2302.02412)
- pkuliyi2015 & Kahsolt's Tiled VAE algorithm

> Sizes/dimensions are in pixels and then converted to latent-space sizes.

### Tiled Diffusion

> * Set `tile_overlap` to 0 and `denoise` to 1 to see the tile seams and then adjust the options to your needs.
> * Increase `tile_batch_size` to increase speed (if your machine can handle it).
> * Use the [colorfix node](https://github.com/gameltb/Comfyui-StableSR) if your colors look off.

#### Options

| Name              | Description                                                  |
|-------------------|--------------------------------------------------------------|
| `method`          | Tiling strategy.                                              |
| `tile_width`      | Tile's width                                                 |
| `tile_height`     | Tile's height                                                |
| `tile_overlap`    | Tile's overlap                                               |
| `tile_batch_size` | The number of tiles to process in a batch                    |

#### SpotDiffusion

[Paper](https://arxiv.org/abs/2407.15507)

A tiling algorithm that attempts to eliminate seams by randomly shifting the denoise window per timestep. It is mainly used for fast inferences by setting `tile_overlap` to 0; otherwise, it's better to stick with the other tiling strategies as they produce better outputs.

This additional feature is experimental, in testing, and subject to change.

### License

The implementation of MultiDiffusion, Mixture of Diffusers, and Tiled VAE code is currently under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License since it was borrowed from the SD-WebUI extension https://github.com/pkuliyi2015/multidiffusion-upscaler-for-automatic1111/. Anything else GPLv3.

## `tiled_diffusion.py` — tile splitting with overlap (lines 68-85)

```python
def split_bboxes(w:int, h:int, tile_w:int, tile_h:int, overlap:int=16, init_weight:Union[Tensor, float]=1.0) -> Tuple[List[BBox], Tensor]:
    cols = ceildiv((w - overlap) , (tile_w - overlap))
    rows = ceildiv((h - overlap) , (tile_h - overlap))
    dx = (w - tile_w) / (cols - 1) if cols > 1 else 0
    dy = (h - tile_h) / (rows - 1) if rows > 1 else 0

    bbox_list: List[BBox] = []
    weight = torch.zeros((1, 1, h, w), device=devices.device, dtype=torch.float32)
    for row in range(rows):
        y = min(int(row * dy), h - tile_h)
        for col in range(cols):
            x = min(int(col * dx), w - tile_w)

            bbox = BBox(x, y, tile_w, tile_h)
            bbox_list.append(bbox)
            weight[bbox.slicer] += init_weight

    return bbox_list, weight
```

Tiles are spaced evenly (`dx`/`dy`) so that the requested `overlap` is met on average across the
whole row/column count, rather than fixed to a constant stride; `weight` accumulates how many
tiles cover each pixel (its role is the "contributors" / normalizer buffer used later).

## `tiled_diffusion.py` — Gaussian tile weights (lines 448-462, explicitly ported from Mixture of Diffusers)

```python
import numpy as np
from numpy import pi, exp, sqrt
def gaussian_weights(tile_w:int, tile_h:int) -> Tensor:
    '''
    Copy from the original implementation of Mixture of Diffusers
    https://github.com/albarji/mixture-of-diffusers/blob/master/mixdiff/tiling.py
    This generates gaussian weights to smooth the noise of each tile.
    This is critical for this method to work.
    '''
    f = lambda x, midpoint, var=0.01: exp(-(x-midpoint)*(x-midpoint) / (tile_w*tile_w) / (2*var)) / sqrt(2*pi*var)
    x_probs = [f(x, (tile_w - 1) / 2) for x in range(tile_w)]   # -1 because index goes from 0 to latent_width - 1
    y_probs = [f(y,  tile_h      / 2) for y in range(tile_h)]

    w = np.outer(y_probs, x_probs)
    return torch.from_numpy(w).to(devices.device, dtype=torch.float32)
```

## `tiled_diffusion.py` — `MultiDiffusion` method's per-step blend (lines 466-542)

This is the ComfyUI integration point: `MultiDiffusion.__call__` is installed as a
`model_function` wrapper (invoked from inside the sampler's `apply_model`, once per tile-batch,
per denoise step), accumulating every tile's UNet output into a shared buffer and normalizing by
tile coverage count — the same mechanism as the original repo's `count`/`value` accumulation,
adapted to ComfyUI's `model_options` patching hook instead of a standalone pipeline loop:

```python
class MultiDiffusion(AbstractDiffusion):
    
    @torch.inference_mode()
    def __call__(self, model_function: BaseModel.apply_model, args: dict):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]
        cond_or_uncond: List = args["cond_or_uncond"]

        N, C, H, W = x_in.shape

        # comfyui can feed in a latent that's a different size cause of SetArea, so we'll refresh in that case.
        self.refresh = False
        if self.weights is None or self.h != H or self.w != W:
            self.h, self.w = H, W
            self.refresh = True
            self.init_grid_bbox(self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)
            # init everything done, perform sanity check & pre-computations
            self.init_done()
        self.h, self.w = H, W
        # clear buffer canvas
        self.reset_buffer(x_in)

        # Background sampling (grid bbox)
        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if processing_interrupted(): 
                    return x_in

                # batching & compute tiles
                x_tile = torch.cat([x_in[bbox.slicer] for bbox in bboxes], dim=0)   # [TB, C, TH, TW]
                t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])
                c_tile = {}
                for k, v in c_in.items():
                    if isinstance(v, torch.Tensor):
                        if len(v.shape) == len(x_tile.shape):
                            bboxes_ = bboxes
                            if v.shape[-2:] != x_in.shape[-2:]:
                                cf = x_in.shape[-1] * self.compression // v.shape[-1] # compression factor
                                bboxes_ = self.get_grid_bbox(
                                    self.width // cf,
                                    self.height // cf,
                                    self.overlap // cf,
                                    self.tile_batch_size,
                                    v.shape[-1],
                                    v.shape[-2],
                                    x_in.device,
                                    self.get_tile_weights,
                                )
                            v = torch.cat([v[bbox_.slicer] for bbox_ in bboxes_[batch_id]])
                        if v.shape[0] != x_tile.shape[0]:
                            v = repeat_to_batch_size(v, x_tile.shape[0])
                    c_tile[k] = v

                if 'control' in c_in:
                    self.process_controlnet(x_tile, c_in, cond_or_uncond, bboxes, N, batch_id)
                    c_tile['control'] = c_in['control'].get_control_orig(x_tile, t_tile, c_tile, len(cond_or_uncond), c_in['transformer_options'])

                x_tile_out = model_function(x_tile, t_tile, **c_tile)

                for i, bbox in enumerate(bboxes):
                    self.x_buffer[bbox.slicer] += x_tile_out[i*N:(i+1)*N, :, :, :]
                del x_tile_out, x_tile, t_tile, c_tile

        # Averaging background buffer
        x_out = torch.where(self.weights > 1, self.x_buffer / self.weights, self.x_buffer)

        return x_out
```

## `tiled_diffusion.py` — `MixtureOfDiffusers` method's per-step blend (lines 706-820)

Same tiling/batching structure, but each tile's contribution is pre-multiplied by its Gaussian
weight before accumulating, and weights are rescaled once (`rescale_factor = 1 / self.weights`)
for numerical stability since raw Gaussian values can be extremely small far from the tile center:

```python
class MixtureOfDiffusers(AbstractDiffusion):
    """
        Mixture-of-Diffusers Implementation
        https://github.com/albarji/mixture-of-diffusers
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # weights for custom bboxes
        self.custom_weights: List[Tensor] = []
        self.get_weight = gaussian_weights

    def init_done(self):
        super().init_done()
        # The original gaussian weights can be extremely small, so we rescale them for numerical stability
        self.rescale_factor = 1 / self.weights
        # Meanwhile, we rescale the custom weights in advance to save time of slicing
        for bbox_id, bbox in enumerate(self.custom_bboxes):
            if bbox.blend_mode == BlendMode.BACKGROUND:
                self.custom_weights[bbox_id] *= self.rescale_factor[bbox.slicer]

    @grid_bbox
    def get_tile_weights(self) -> Tensor:
        # weights for grid bboxes
        self.tile_weights = self.get_weight(self.tile_w, self.tile_h)
        return self.tile_weights

    @torch.inference_mode()
    def __call__(self, model_function: BaseModel.apply_model, args: dict):
        x_in: Tensor = args["input"]
        t_in: Tensor = args["timestep"]
        c_in: dict = args["c"]
        cond_or_uncond: List= args["cond_or_uncond"]

        N, C, H, W = x_in.shape

        self.refresh = False
        if self.weights is None or self.h != H or self.w != W:
            self.h, self.w = H, W
            self.refresh = True
            self.init_grid_bbox(self.tile_width, self.tile_height, self.tile_overlap, self.tile_batch_size)
            self.init_done()
        self.h, self.w = H, W
        self.reset_buffer(x_in)

        if self.draw_background:
            for batch_id, bboxes in enumerate(self.batched_bboxes):
                if processing_interrupted(): 
                    return x_in
                x_tile_list = []
                for bbox in bboxes:
                    x_tile_list.append(x_in[bbox.slicer])

                x_tile = torch.cat(x_tile_list, dim=0)
                t_tile = repeat_to_batch_size(t_in, x_tile.shape[0])
                c_tile = {}
                for k, v in c_in.items():
                    if isinstance(v, torch.Tensor):
                        if len(v.shape) == len(x_tile.shape):
                            bboxes_ = bboxes
                            if v.shape[-2:] != x_in.shape[-2:]:
                                cf = x_in.shape[-1] * self.compression // v.shape[-1]
                                bboxes_ = self.get_grid_bbox(
                                    (tile_w := self.width // cf),
                                    (tile_h := self.height // cf),
                                    self.overlap // cf,
                                    self.tile_batch_size,
                                    v.shape[-1],
                                    v.shape[-2],
                                    x_in.device,
                                    lambda: self.get_weight(tile_w, tile_h),
                                )
                            v = torch.cat([v[bbox_.slicer] for bbox_ in bboxes_[batch_id]])
                        if v.shape[0] != x_tile.shape[0]:
                            v = repeat_to_batch_size(v, x_tile.shape[0])
                    c_tile[k] = v

                if 'control' in c_in:
                    self.process_controlnet(x_tile, c_in, cond_or_uncond, bboxes, N, batch_id)
                    c_tile['control'] = c_in['control'].get_control_orig(x_tile, t_tile, c_tile, len(cond_or_uncond), c_in['transformer_options'])

                x_tile_out = model_function(x_tile, t_tile, **c_tile)

                for i, bbox in enumerate(bboxes):
                    # These weights can be calcluated in advance, but will cost a lot of vram 
                    # when you have many tiles. So we calculate it here.
                    w = self.tile_weights * self.rescale_factor[bbox.slicer]
                    self.x_buffer[bbox.slicer] += x_tile_out[i*N:(i+1)*N, :, :, :] * w
                del x_tile_out, x_tile, t_tile, c_tile
        x_out = self.x_buffer

        return x_out
```

## `tiled_diffusion.py` — the `TiledDiffusion` ComfyUI node (lines 822-871)

```python
MAX_RESOLUTION=8192
class TiledDiffusion():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL", ),
                                "method": (["MultiDiffusion", "Mixture of Diffusers", "SpotDiffusion"], {"default": "Mixture of Diffusers"}),
                                "tile_width": ("INT", {"default": 96*opt_f, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                                "tile_height": ("INT", {"default": 96*opt_f, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                                "tile_overlap": ("INT", {"default": 8*opt_f, "min": 0, "max": 256*opt_f, "step": 4*opt_f}),
                                "tile_batch_size": ("INT", {"default": 4, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                            }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "_for_testing"
    instances = WeakSet()

    @classmethod
    def IS_CHANGED(s, *args, **kwargs):
        for o in s.instances:
            o.impl.reset()
        return ""
    
    def __init__(self) -> None:
        self.__class__.instances.add(self)

    def apply(self, model: ModelPatcher, method, tile_width, tile_height, tile_overlap, tile_batch_size):
        if method == "Mixture of Diffusers":
            self.impl = MixtureOfDiffusers()
        elif method == "MultiDiffusion":
            self.impl = MultiDiffusion()
        else:
            self.impl = SpotDiffusion()

        compression = 4 if "CASCADE" in str(model.model.model_type) else 8
        self.impl.tile_width = tile_width // compression
        self.impl.tile_height = tile_height // compression
        self.impl.tile_overlap = tile_overlap // compression
        self.impl.tile_batch_size = tile_batch_size
        
        self.impl.compression = compression
        self.impl.width = tile_width
        self.impl.height  = tile_height
        self.impl.overlap = tile_overlap
```

The node itself is a `MODEL`→`MODEL` patch: it selects an `AbstractDiffusion` subclass
(`MultiDiffusion`, `MixtureOfDiffusers`, or `SpotDiffusion`) as `self.impl` and (elsewhere in the
same `apply` method, not reproduced here) registers `self.impl.__call__` against the model's
`model_options` so it runs in place of the normal single-pass UNet call — i.e. the tiling +
seam-blend logic above replaces `apply_model` for every step of whatever sampler the patched
model is later run through, rather than being its own sampling loop.
