# Context-Anchored Tile Refine

ComfyUI nodes for tiled refining and upscaling. An already-upscaled image is refined tile by tile with no visible seams, or only inside a masked region with the rest left untouched. Seams are hidden by conditioning, not by blending tricks. For models with a vision-language text encoder (Krea 2), the VL nodes replace the prompt entirely with vision conditioning, which removes the classic tiled-upscale failure of prompt objects reappearing in every tile.

**Jump to:**

- [Installation](#installation)
- [The nodes](#the-nodes)
  - [Tile Refine](#tile-refine)
  - [Tile Refine (VL)](#tile-refine-vl)
  - [Tile Upscale (VL)](#tile-upscale-vl)
- [Shared behavior](#shared-behavior)
  - [How seams are hidden](#how-seams-are-hidden)
  - [Masked refine](#masked-refine)
  - [Guider and ControlNet](#guider-and-controlnet)
  - [Model support](#model-support)
- [The VL method: RoI token slicing](#the-vl-method-roi-token-slicing)
- [Example workflows](#example-workflows)
- [License](#license)

## Installation

Search for "Context-Anchored Tile Refine" in ComfyUI Manager, or clone into your `custom_nodes` folder:

```
git clone https://github.com/blakeem/ComfyUI-ContextAnchoredTileRefine
```

## The nodes

Three nodes share one tiling engine. Pick by model and workflow shape:

| Node | Use when |
|---|---|
| Tile Refine | Any model that samples through a GUIDER. You wire the sampling nodes yourself. |
| Tile Refine (VL) | A vision-language model (Krea 2). Same wiring, no positive prompt needed. |
| Tile Upscale (VL) | A vision-language model, everything in one node: upscale, then refine. |

These four inputs appear on every node and are the ones worth explaining. Everything else (models, sampler wiring, seed, steps) behaves exactly as it does in the standard ComfyUI sampling nodes.

| Input | Type | What it does |
|---|---|---|
| `max_tile_width` / `max_tile_height` | INT | Largest pixel size the model sees per tile, including the context rings. Set to the largest size your model handles well. |
| `context_anchor` | INT | Width of the frozen border each tile sees but never changes. Holds already-refined neighbors, and the area around a mask, steady. |
| `context_overlap` | INT | Width of the band shared with an already-processed neighbor. 0 gives hard seams. |

Tuning notes from real use:

- Denoise 0.5 works well for smaller upscales (2x to 3x, around 4 tiles). For a 4x upscale to around 4k, 0.42 gives better detail with less chance of artifacts. It depends on the scene.
- More tiles need more context: raise `context_anchor` (32 to 128 tested well) and `context_overlap` (32 to 256 tested well). Both default to 32, which is invisible on most scenes; large smooth gradients such as an open sky want the higher end of `context_overlap`. Detailed scenes need less, not more.

Preview any layout with the [tile simulator](https://blakeem.github.io/ComfyUI-ContextAnchoredTileRefine/tile-simulator.html).

### Tile Refine

![Context-Anchored Tile Refine](refine-node.png)

The base node, for any model that samples through a GUIDER. Upscale the image however you like first, then feed it in; wire `guider`, `sampler`, `sigmas`, `vae`, and `noise` as you would for SamplerCustomAdvanced. The optional `mask` refines only that region and leaves everything outside it untouched. Only the tiling geometry is tunable; how the seam itself is hidden is baked in.

### Tile Refine (VL)

![Context-Anchored Tile Refine (VL)](vl-refine-node.png)

The vision-conditioned variant for models whose text encoder is a vision-language model (Krea 2). Inputs are the base node's plus `clip`, and it needs no positive prompt at all: each tile's positive conditioning is a slice of one whole-image vision encode (see [the VL method](#the-vl-method-roi-token-slicing)). The guider's positive prompt is ignored; its negative still applies. Non-VL encoders (SD/SDXL CLIP, T5, plain Qwen3) are rejected with a clear error.

The `mask` works here too and keeps the global view: the whole image is still encoded once, and the masked region's tiles slice their true place in that encode, so the region is refined aware of everything around it. That fits refining one subject with its own sampler settings, and upscale-inpainting: inpaint at low resolution, composite into the full-size image, then mask-refine the pasted region so it matches the surrounding resolution and grain.

### Tile Upscale (VL)

![Context-Anchored Tile Upscale (VL)](vl-upscale-node.png)

The whole flow in one node: image in, refined image out. It upscales the entire image first, through the optional `upscale_model` if connected, then a single lanczos pass to exactly `input size x upscale_by` (lanczos alone when no model; a resize that would not change the size is skipped). It then runs the same VL tile refine as Tile Refine (VL).

Noise, sampler, schedule, and CFG guidance are built inside the node from widgets (`seed`, `sampler_name`, `scheduler`, `steps`, `denoise`, `cfg`), so no custom-sampling nodes are needed. There is no positive prompt. The optional `negative` input is the one text channel that applies; left unconnected it behaves as an empty prompt. `denoise 0.0` skips diffusion and returns the pure upscale. No mask input: for a region pass, use Tile Refine (VL).

## Shared behavior

Everything below applies to all three nodes.

### How seams are hidden

Tiles are processed in raster order, so each tile's top and left neighbors are already refined by the time it is sampled. `context_anchor` gives every tile a frozen border of that finished content to condition against, so tiles continue each other instead of drifting apart.

`context_overlap` is the band a tile shares with an already-processed neighbor. Both tiles diffuse that band from the same raw pixels, each anchored to the content around it, so the two results land close together. The node then cross-dissolves them. This happens only where a tile meets an earlier tile, never at the image border.

The feather holds the pixels at the seam fully on the new tile, then falls off to nothing at the outer edge: the seam-most 10 percent of the band stays solid, then a squared ramp carries it to zero. Squared matters because it lands on zero with zero slope; a linear ramp meets the neighbor's untouched pixels at an angle, and the eye reads that kink as a line.

Where the handover sits comes from the minimum error boundary cut in Efros and Freeman, *Image Quilting for Texture Synthesis and Transfer* (SIGGRAPH 2001): dynamic programming finds the path through the overlap where the two results already agree, and the feather's midpoint bends along it, so the handover follows image content rather than running dead straight.

Tiles diffused separately also land at slightly different brightness and color levels. Panorama stitchers fix this with gain compensation (Brown and Lowe, *Automatic Panoramic Image Stitching*, IJCV 2007); here the correction is additive, per channel, and sequential. The shared band is the only place both tiles refined the same raw pixels, so the difference there is pure disagreement with no content in it. The node subtracts the per-channel median of that difference, putting each tile on its neighbor's level. This runs only at tile seams, never at a mask edge.

### Masked refine

With a `mask`, the node crops to the masked region plus a `context_anchor` border, refines only that region against the frozen surrounding pixels, and composites it back with a 1px anti-aliased edge. The rest of the image is left byte for byte untouched. Feed an inverted mask on a second pass to refine, for example, the background and the character separately with different settings.

### Guider and ControlNet

The `guider` input takes NAG (Normalized Attention Guidance) or any other guider. ControlNet is supported on the base Tile Refine node: it re-crops the control hint to each tile, so depth, canny, or pose guidance lands on the right pixels tile by tile. Build the hint at the same size as the image you feed the node. Conditioning without a per-tile meaning (GLIGEN, area masks, reference latents) passes through unchanged.

The VL nodes ignore ControlNet entirely (a warning is logged if one is wired): every tile's positive is replaced by its vision slice, so there is nothing for a control hint to attach to. Use the base node for control.

Image batches work on all three nodes, including with video-family VAEs (Krea 2): each image in the batch is encoded and conditioned on its own picture.

### Model support

Any model that samples through a GUIDER works with Tile Refine, including models whose VAE uses video-style 5-D latents, such as Krea 2 with the Qwen image VAE. Tiles are encoded, sampled, and decoded in the VAE's native latent layout. The VL nodes additionally need a vision-language text encoder and are verified with Krea 2.

## The VL method: RoI token slicing

Tiled refining has a classic failure: the prompt describes the whole image, but each tile only holds part of it, so strongly prompt-adherent models re-create prompt objects inside tiles that should not contain them. A moon in the sky reappears in every tile. Lowering denoise hides it but gives up refinement.

The VL nodes fix the mismatch in conditioning space instead. The technique is **shared visual self-conditioning**: tiled refinement via RoI-sliced vision tokens from a single global encode. We have not found this exact combination published elsewhere; the pieces, individually, have established names:

**Stage 1: one global vision encode.** The whole image is area-resampled to the encoder's budget and run once through the vision tower of the model's own text encoder (Krea 2's Qwen3-VL). The output is a grid of vision tokens: one embedding per 32-pixel cell, each carrying what that cell holds and where it sits in the image.

**Stage 2: the tokens are the conditioning.** Those tokens are used directly as the positive conditioning, with no text at all. This is visual self-conditioning: the image being refined is its own prompt. It is also zero-shot, because a VLM-conditioned DiT was already trained to read vision tokens in its conditioning stream. Same idea as IP-Adapter's image prompting, but through the model's native multimodal interface instead of a bolted-on encoder and learned projection.

**Stage 3: RoI token slicing.** Each tile keeps the delimiter tokens plus only the grid cells its crop covers; partly covered boundary cells go to both neighbors, the token-space analogue of the pixel overlap band. This is RoI-based token selection: RoIAlign from the detection literature, applied to a conditioning token grid instead of a detector feature map.

Why this beats a prompt: vision tokens carry no demands, only what each cell actually holds. And because every tile slices the same global encode, tiles agree on the story: tone, palette, gaze, and structures that cross seams stay coherent even at high denoise. Systems like MultiDiffusion or Ultimate SD Upscale hide seams in pixel or latent space while every tile shares one global text prompt; the VL nodes give each tile conditioning that is true for that tile.

## Example workflows

- [Krea 2 workflow](Krea%202.json)
- [Chroma + Z-Image hybrid workflow](Chroma%20+%20z-image%20Hybrid%20workflow.json)

## License

[GPLv3](LICENSE)
