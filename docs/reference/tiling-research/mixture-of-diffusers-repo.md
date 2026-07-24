Source: https://github.com/albarji/mixture-of-diffusers (README.md; mixdiff/tiling.py; mixdiff/canvas.py), branch `master`
Repo: albarji/mixture-of-diffusers (reference implementation for arXiv:2302.02412, "Mixture of Diffusers for scene composition and high resolution image generation")
Retrieved: 2026-07-22

Note: captures the seam-blending mechanism only (tile/region overlap geometry, the per-pixel
Gaussian/quartic weighting used to blend overlapping tiles, and the weighted-average accumulation
in the denoising loop). Full parameter reference tables from the README are kept since they
document the tiling/overlap and mask-weighting knobs directly; usage prerequisites and gallery
images are omitted as out of scope.

## README

This repository holds various scripts and tools implementing a method for integrating a mixture of different diffusion processes collaborating to generate a single image. Each diffuser focuses on a particular region on the image, taking into account boundary effects to promote a smooth blending.

The mixture of diffusion processes is done in a way that harmonizes the generation process, preventing "seam" effects in the generated image.

Using several diffusion processes in parallel has also practical advantages when generating very large images, as the GPU memory requirements are similar to that of generating an image of the size of a single tile.

### Usage

This repository provides two new pipelines, `StableDiffusionTilingPipeline` and `StableDiffusionCanvasPipeline`, that extend the standard Stable Diffusion pipeline from [Diffusers](https://github.com/huggingface/diffusers). They feature new options that allow defining the mixture of diffusers, which are distributed as a number of "diffusion regions" over the image to be generated. `StableDiffusionTilingPipeline` is simpler to use and arranges the diffusion regions as a grid over the canvas, while `StableDiffusionCanvasPipeline` allows a more flexible placement and also features image2image capabilities.

#### StableDiffusionTilingPipeline

```python
from diffusers import LMSDiscreteScheduler
from mixdiff import StableDiffusionTilingPipeline

# Creater scheduler and model (similar to StableDiffusionPipeline)
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
pipeline = StableDiffusionTilingPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, use_auth_token=True).to("cuda:0")

# Mixture of Diffusers generation
image = pipeline(
    prompt=[[
        "A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
        "A dirt road in the countryside crossing pastures, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece",
        "An old and rusty giant robot lying on a dirt road, by jakub rozalski, dark sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"
    ]],
    tile_height=640,
    tile_width=640,
    tile_row_overlap=0,
    tile_col_overlap=256,
    guidance_scale=8,
    seed=7178915308,
    num_inference_steps=50,
)["sample"][0]
```

The prompts must be provided as a list of lists, where each list represents a row of diffusion regions. The geometry of the canvas is inferred from these lists, e.g. in the example above we are creating a grid of 1x3 diffusion regions (1 row and 3 columns). The rest of parameters provide information on the size of these regions, and how much they overlap with their neighbors.

The full list of arguments to a `StableDiffusionTilingPipeline` is:

> `prompt`: either a single string (no tiling) or a list of lists with all the prompts to use (one list for each row of tiles). This will also define the tiling structure.
>
> `num_inference_steps`: number of diffusions steps.
>
> `guidance_scale`: classifier-free guidance.
>
> `seed`: general random seed to initialize latents.
>
> `tile_height`: height in pixels of each grid tile.
>
> `tile_width`: width in pixels of each grid tile.
>
> `tile_row_overlap`: number of overlap pixels between tiles in consecutive rows.
>
> `tile_col_overlap`: number of overlap pixels between tiles in consecutive columns.
>
> `guidance_scale_tiles`: specific weights for classifier-free guidance in each tile. If `None`, the value provided in `guidance_scale` will be used.
>
> `seed_tiles`: specific seeds for the initialization latents in each tile. These will override the latents generated for the whole canvas using the standard `seed` parameter.
>
> `seed_tiles_mode`: either `"full"` `"exclusive"`. If `"full"`, all the latents affected by the tile be overriden. If `"exclusive"`, only the latents that are affected exclusively by this tile (and no other tiles) will be overrriden.
>
> `seed_reroll_regions`: a list of tuples in the form (start row, end row, start column, end column, seed) defining regions in pixel space for which the latents will be overriden using the given seed. Takes priority over `seed_tiles`.
>
> `cpu_vae`: the decoder from latent space to pixel space can require too mucho GPU RAM for large images. If you find out of memory errors at the end of the generation process, try setting this parameter to `True` to run the decoder in CPU. Slower, but should run without memory issues.

#### StableDiffusionCanvasPipeline

The `StableDiffusionCanvasPipeline` works by defining a list of `Text2ImageRegion` objects that detail the region of influence of each diffuser.

```python
from diffusers import LMSDiscreteScheduler
from mixdiff import StableDiffusionCanvasPipeline, Text2ImageRegion

# Creater scheduler and model (similar to StableDiffusionPipeline)
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
pipeline = StableDiffusionCanvasPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=scheduler, use_auth_token=True).to("cuda:0")

# Mixture of Diffusers generation
image = pipeline(
    canvas_height=640,
    canvas_width=1408,
    regions=[
        Text2ImageRegion(0, 640, 0, 640, guidance_scale=8,
            prompt=f"A charming house in the countryside, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"),
        Text2ImageRegion(0, 640, 384, 1024, guidance_scale=8,
            prompt=f"A dirt road in the countryside crossing pastures, by jakub rozalski, sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"),
        Text2ImageRegion(0, 640, 768, 1408, guidance_scale=8,
            prompt=f"An old and rusty giant robot lying on a dirt road, by jakub rozalski, dark sunset lighting, elegant, highly detailed, smooth, sharp focus, artstation, stunning masterpiece"),
    ],
    num_inference_steps=50,
    seed=7178915308,
)["sample"][0]
```

The full list of arguments to a `StableDiffusionCanvasPipeline` is:

> `canvas_height`: height in pixels of the image to generate. Must be a multiple of 8.
>
> `canvas_width`: width in pixels of the image to generate. Must be a multiple of 8.
>
> `regions`: list of `Text2Image` or `Image2Image` diffusion regions (see below).
>
> `num_inference_steps`: number of diffusions steps.
>
> `seed`: general random seed to initialize latents.
>
> `reroll_regions`: list of `RerollRegion` regions in which to reroll latents (see below). Useful if you like the overall aspect of the generated image, but want to regenerate a specific region using a different random seed.
>
> `cpu_vae`: whether to perform encoder-decoder operations in CPU, even if the diffusion process runs in GPU. Use `cpu_vae=True` if you run out of GPU memory at the end of the generation process for large canvas dimensions, or if you create large `Image2Image` regions.
>
> `decode_steps`: if `True` the result will include not only the final image, but also all the intermediate steps in the generation. Note: this will greatly increase running times.

All regions are configured with the following parameters:

> `row_init`: starting row in pixel space (included). Must be a multiple of 8.
>
> `row_end`: end row in pixel space (not included). Must be a multiple of 8.
>
> `col_init`: starting column in pixel space (included). Must be a multiple of 8.
>
> `col_end`: end column in pixel space (not included). Must be a multiple of 8.
>
> `region_seed`: seed for random operations in this region
>
> `noise_eps`: deviation of a zero-mean gaussian noise to be applied over the latents in this region. Useful for slightly "rerolling" latents

Additionally, `Text2Image` regions use the following arguments:

> `prompt`: text prompt guiding the diffuser in this region
>
> `guidance_scale`: guidance scale of the diffuser in this region. If None, randomize.
>
> `mask_type`: kind of weight mask applied to this region, must be one of `["constant", gaussian", quartic"]`.
>
> `mask_weight`: global weights multiplier of the mask.

`Image2Image` regions are configured with the basic region parameters plus ther following:

> `reference_image`: image to use as guidance. Must be loaded as a PIL image and pre-processed using the `preprocess_image` function (see example above). It will be automatically rescaled to the shape of the region.
>
> `strength`: strength of the image guidance, must lie in the range `[0.0, 1.0]` (from no guidance to absolute priority of the original image).

## `mixdiff/tiling.py` — tile geometry and weighted accumulation (lines 170-229)

```python
        # Mask for tile weights strenght
        tile_weights = self._gaussian_weights(tile_width, tile_height, batch_size)

        # Diffusion timesteps
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # Diffuse each tile
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    px_row_init, px_row_end, px_col_init, px_col_end = _tile2latent_indices(row, col, tile_width, tile_height, tile_row_overlap, tile_col_overlap)
                    tile_latents = latents[:, :, px_row_init:px_row_end, px_col_init:px_col_end]
                    # expand the latents if we are doing classifier free guidance
                    latent_model_input = torch.cat([tile_latents] * 2) if do_classifier_free_guidance else tile_latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    # predict the noise residual
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings[row][col])["sample"]
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        guidance = guidance_scale if guidance_scale_tiles is None or guidance_scale_tiles[row][col] is None else guidance_scale_tiles[row][col]
                        noise_pred_tile = noise_pred_uncond + guidance * (noise_pred_text - noise_pred_uncond)
                        noise_preds_row.append(noise_pred_tile)
                noise_preds.append(noise_preds_row)
            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(latents.shape, device=self.device)
            contributors = torch.zeros(latents.shape, device=self.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    px_row_init, px_row_end, px_col_init, px_col_end = _tile2latent_indices(row, col, tile_width, tile_height, tile_row_overlap, tile_col_overlap)
                    noise_pred[:, :, px_row_init:px_row_end, px_col_init:px_col_end] += noise_preds[row][col] * tile_weights
                    contributors[:, :, px_row_init:px_row_end, px_col_init:px_col_end] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width // 8
        latent_height = tile_height // 8

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]
        
        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))
```

Tile-to-pixel/latent index mapping (`mixdiff/tiling.py`, module-level functions):

```python
def _tile2pixel_indices(tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap):
    """Given a tile row and column numbers returns the range of pixels affected by that tiles in the overall image
    
    Returns a tuple with:
        - Starting coordinates of rows in pixel space
        - Ending coordinates of rows in pixel space
        - Starting coordinates of columns in pixel space
        - Ending coordinates of columns in pixel space
    """
    px_row_init = 0 if tile_row == 0 else tile_row * (tile_height - tile_row_overlap)
    px_row_end = px_row_init + tile_height
    px_col_init = 0 if tile_col == 0 else tile_col * (tile_width - tile_col_overlap)
    px_col_end = px_col_init + tile_width
    return px_row_init, px_row_end, px_col_init, px_col_end


def _pixel2latent_indices(px_row_init, px_row_end, px_col_init, px_col_end):
    """Translates coordinates in pixel space to coordinates in latent space"""
    return px_row_init // 8, px_row_end // 8, px_col_init // 8, px_col_end // 8


def _tile2latent_indices(tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap):
    """Given a tile row and column numbers returns the range of latents affected by that tiles in the overall image
    
    Returns a tuple with:
        - Starting coordinates of rows in latent space
        - Ending coordinates of rows in latent space
        - Starting coordinates of columns in latent space
        - Ending coordinates of columns in latent space
    """
    px_row_init, px_row_end, px_col_init, px_col_end = _tile2pixel_indices(tile_row, tile_col, tile_width, tile_height, tile_row_overlap, tile_col_overlap)
    return _pixel2latent_indices(px_row_init, px_row_end, px_col_init, px_col_end)
```

## `mixdiff/canvas.py` — per-region mask weight builders (lines 166-215)

```python
@dataclass
class MaskWeightsBuilder:
    """Auxiliary class to compute a tensor of weights for a given diffusion region"""
    latent_space_dim: int  # Size of the U-net latent space
    nbatch: int = 1  # Batch size in the U-net

    def compute_mask_weights(self, region: DiffusionRegion) -> torch.tensor:
        """Computes a tensor of weights for a given diffusion region"""
        MASK_BUILDERS = {
            MaskModes.CONSTANT.value: self._constant_weights,
            MaskModes.GAUSSIAN.value: self._gaussian_weights,
            MaskModes.QUARTIC.value: self._quartic_weights,
        }
        return MASK_BUILDERS[region.mask_type](region)

    def _constant_weights(self, region: DiffusionRegion) -> torch.tensor:
        """Computes a tensor of constant for a given diffusion region"""
        latent_width = region.latent_col_end - region.latent_col_init
        latent_height = region.latent_row_end - region.latent_row_init
        return torch.ones(self.nbatch, self.latent_space_dim, latent_height, latent_width) * region.mask_weight

    def _gaussian_weights(self, region: DiffusionRegion) -> torch.tensor:
        """Generates a gaussian mask of weights for tile contributions"""
        latent_width = region.latent_col_end - region.latent_col_init
        latent_height = region.latent_row_end - region.latent_row_init

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = (latent_height -1) / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]
        
        weights = np.outer(y_probs, x_probs) * region.mask_weight
        return torch.tile(torch.tensor(weights), (self.nbatch, self.latent_space_dim, 1, 1))

    def _quartic_weights(self, region: DiffusionRegion) -> torch.tensor:
        """Generates a quartic mask of weights for tile contributions
        
        The quartic kernel has bounded support over the diffusion region, and a smooth decay to the region limits.
        """
        quartic_constant = 15. / 16.        

        support = (np.array(range(region.latent_col_init, region.latent_col_end)) - region.latent_col_init) / (region.latent_col_end - region.latent_col_init - 1) * 1.99 - (1.99 / 2.)
        x_probs = quartic_constant * np.square(1 - np.square(support))
        support = (np.array(range(region.latent_row_init, region.latent_row_end)) - region.latent_row_init) / (region.latent_row_end - region.latent_row_init - 1) * 1.99 - (1.99 / 2.)
        y_probs = quartic_constant * np.square(1 - np.square(support))

        weights = np.outer(y_probs, x_probs) * region.mask_weight
        return torch.tile(torch.tensor(weights), (self.nbatch, self.latent_space_dim, 1, 1))
```

`MaskModes` (from `mixdiff/canvas.py`, near top of file): `CONSTANT`, `GAUSSIAN` (default), `QUARTIC` — the quartic kernel comment cites https://en.wikipedia.org/wiki/Kernel_(statistics) as its reference.

`StableDiffusionCanvasPipeline`'s denoising loop accumulates the same way as the tiling pipeline (`mixdiff/canvas.py`, around lines 342-371): a `mask_weights` list is built once per region via `compute_mask_weights`, then every step does
```python
contributors = torch.zeros(latents.shape, device=self.device)
for region, noise_pred_region, mask_weights_region in zip(text2image_regions, noise_preds_regions, mask_weights):
    noise_pred[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end] += noise_pred_region * mask_weights_region
    contributors[:, :, region.latent_row_init:region.latent_row_end, region.latent_col_init:region.latent_col_end] += mask_weights_region

noise_pred /= contributors
```
— the same weighted-accumulate-then-normalize pattern as `tiling.py`, generalized to arbitrarily placed/weighted regions instead of a fixed grid.
