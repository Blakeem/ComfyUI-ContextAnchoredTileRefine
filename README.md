# Context-Anchored Tile Refine

ComfyUI nodes for tiled upscaling, and for refining a masked area of a large image without processing the rest. The base node refines an already-upscaled image tile by tile with no visible seams. Upscale the image however you like first, then feed it in. A second node, [Context-Anchored Tile Refine (VL)](#the-vl-node), replaces the prompt entirely with vision conditioning for models with a vision-language text encoder (Krea 2).

## Node inputs

![Context-Anchored Tile Refine Node](node.png)

| Input | Type | What it does |
|---|---|---|
| `image` | IMAGE | The image to refine (already upscaled or composited). |
| `guider` | GUIDER | Denoises each tile. |
| `sampler` | SAMPLER | The sampler used for each tile. |
| `sigmas` | SIGMAS | The schedule. Sets denoise strength. |
| `vae` | VAE | Encodes and decodes each tile. |
| `noise` | NOISE | Noise source (e.g. RandomNoise), as in SamplerCustomAdvanced. |
| `max_tile_width` / `max_tile_height` | INT | Largest pixel size the model sees per tile, including the context rings. Set to the largest your model handles well. |
| `context_anchor` | INT | Width of the frozen border each tile sees but never changes. Holds already-refined neighbors, and the area around a mask, steady. Applies on every tile edge. |
| `context_overlap` | INT | Width of the band shared with an already-processed neighbor. 0 gives hard seams. Applies only where a tile borders an earlier tile. |
| `mask` | MASK (optional) | Refine only this region. Everything outside it is left untouched. |

Only the tiling geometry is tunable. How the seam itself is hidden is baked in.

Both context inputs default to 32, which is invisible on most scenes. Raise `context_overlap` toward 128 for large smooth gradients such as an open sky, where there is no texture for the seam to hide in. Detailed scenes need less, not more.

## How seams are hidden

Tiles are processed in raster order, so each tile's top and left neighbors are already refined by the time it is sampled. `context_anchor` gives every tile a frozen border of that finished content to condition against, so tiles continue each other instead of drifting apart.

`context_overlap` is the band a tile shares with an already-processed neighbor. Both tiles diffuse that band from the same raw pixels, each anchored to the content around it, so the two results land close together. The node then cross-dissolves them. This happens only where a tile meets an earlier tile, never at the image border.

### The blend

The feather holds the pixels at the seam fully on the new tile, then falls off to nothing at the outer edge. The seam-most 10 percent of the band stays solid, then a squared ramp carries it to zero. Squared matters because it lands on zero with zero slope. A linear ramp meets the neighbor's untouched pixels at an angle, and the eye reads that kink as a line.

Where the handover sits comes from the **minimum error boundary cut** in Efros and Freeman, *Image Quilting for Texture Synthesis and Transfer*, SIGGRAPH 2001. Take the squared difference between the two overlapping patches, then use dynamic programming to find the connected path through it with the lowest total error. That path threads through the pixels where the two results already agree, so handing over there shows nothing.

Image quilting cuts along the path and pastes. Here the path pulls the feather's midpoint toward it instead, so the handover bends around image content rather than running dead straight. A straight boundary is easy to spot even when it is faint. One that follows detail is not.

Tiles diffused separately also land at slightly different brightness and color levels, up to about 2/255 on the images we measured. That step spans the whole tile, so no choice of where to hand over can move it. Panorama stitchers fix the same problem with a **gain compensation** stage, as in Brown and Lowe, *Automatic Panoramic Image Stitching using Invariant Features*, IJCV 2007. Theirs is one multiplicative gain per image, solved over all overlaps at once. Here it is additive because the offset does not scale with brightness, per channel because it is partly color, and sequential because a tile can only match neighbors that are already refined.

The shared band is the only place both tiles have refined the same raw pixels, so the difference there is pure disagreement with no content in it. The node subtracts the per-channel median of that difference from the tile, putting it on its neighbor's level; the first tile sets the level for the rest. The offset comes off before the error surface above is built, which is why the median matters. A mean is dragged by the few pixels where the tiles genuinely disagree, leaving a pedestal that makes every path cost about the same. The median removes the typical disagreement, leaving a surface that describes texture for the cut to follow. This runs only at tile seams, never at a mask edge.

Open the [tile simulator](https://blakeem.github.io/ComfyUI-ContextAnchoredTileRefine/tile-simulator.html) to preview the tile layout for any image size and settings.

You can see how I use it in my [personal ComfyUI workflow](Chroma%20+%20z-image%20Hybrid%20workflow.json), or with Krea 2 in the [Krea 2 workflow](Krea%202.json).

## Masked refine

With a `mask`, the node crops to the masked region plus a `context_anchor` border, refines only that region against the frozen surrounding pixels, and composites it back with a 1px anti-aliased edge. The rest of the image is left byte for byte untouched. Feed an inverted mask on a second pass to refine, for example, the background and the character separately with different settings.

## Guider and ControlNet

The `guider` input lets you use NAG (Normalized Attention Guidance), or any other guider. ControlNet is supported: the node re-crops the control hint to each tile, so depth, canny, or pose guidance lands on the right pixels tile by tile. Build the hint at the same size as the image you feed the node. Conditioning without a per-tile meaning (GLIGEN, area masks, reference latents) passes through unchanged.

## The VL node

Tiled refining has a classic failure: the prompt describes the whole image, but each tile only holds part of it, so strongly prompt-adherent models re-create prompt objects inside tiles that shouldn't contain them — a moon in the sky reappears in every tile. Lowering denoise hides it but gives up refinement.

**Context-Anchored Tile Refine (VL)** solves this for models whose text encoder is a vision-language model (Krea 2). It takes the workflow's CLIP as a required input and needs **no positive prompt at all**: the whole image is encoded once through the encoder's vision path, and each tile's positive conditioning becomes the slice of that encode covering the tile — a positionally exact description of what the tile actually holds, informed by the whole image. Tiles neither re-instantiate prompt objects nor drift apart in tone, gaze, or palette across seams, even at high denoise. The guider's positive prompt is ignored; the negative still applies.

Inputs are the base node's minus `mask`, plus `clip`. Wire the same CLIP the workflow loads for the model. Non-VL encoders (SD/SDXL CLIP, T5, plain Qwen3) are rejected with a clear error — use the base node for those models.

### How it works: RoI token slicing

The technique is **shared visual self-conditioning** — tiled refinement via RoI-sliced vision tokens from a single global encode. We have not found this exact combination published elsewhere; the pieces, individually, have established names:

**Stage 1 — one global vision encode.** The whole image is area-resampled to the encoder's budget and run once through the vision tower of the model's own text encoder (Krea 2's Qwen3-VL). The output is a grid of *vision tokens*: one embedding per 32-pixel cell, each carrying what that cell holds and where it sits in the image.

**Stage 2 — the tokens are the conditioning.** Those tokens are used directly as the positive conditioning, with no text at all. This is *visual self-conditioning*: the image being refined is its own prompt. It is also *zero-shot* — no adapter is trained, because a VLM-conditioned DiT was already trained to read vision tokens in its conditioning stream. This is the same idea as IP-Adapter's image prompting, but through the model's native multimodal interface instead of a bolted-on encoder and learned projection.

**Stage 3 — RoI token slicing.** Each tile keeps the delimiter tokens plus only the grid cells its crop covers; partly-covered boundary cells go to both neighbors, the token-space analogue of the pixel overlap band. This is *RoI-based token selection* — RoIAlign from the detection literature, applied to a conditioning token grid instead of a detector feature map, cell-quantized instead of interpolated.

Why this beats a prompt: a text prompt describes the whole image, so every tile inherits demands for objects it doesn't contain, and strongly prompt-adherent models re-create them. Vision tokens carry no demands — only what each cell actually holds. And because every tile slices the *same* global encode, tiles agree on the story: tone, palette, gaze, and structures that cross seams stay coherent even at high denoise. Tiled-upscale systems like MultiDiffusion or Ultimate SD Upscale hide seams in pixel or latent space while every tile shares one global text prompt; this node instead fixes the mismatch in conditioning space, giving each tile conditioning that is true for that tile.

## Model support

Any model that samples through a GUIDER works with the base node, including models whose VAE uses video-style 5-D latents — Krea 2 with the Qwen image VAE, for example. Tiles are encoded, sampled, and decoded in the VAE's native latent layout. The VL node additionally needs a vision-language text encoder and is verified with Krea 2.

## License

[GPLv3](LICENSE)
