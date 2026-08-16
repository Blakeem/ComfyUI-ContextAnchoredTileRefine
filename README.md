# Context-Anchored Tile Refine

ComfyUI nodes for tiled refining and upscaling. An already-upscaled image is refined tile by tile with no visible seams, or only inside a masked region with the rest left untouched. On the VL nodes the prompt is replaced entirely by vision conditioning, which removes the classic tiled-upscale failure of prompt objects reappearing in every tile.

Sample results with Krea 2 and the Tile Upscale (VL) node, upscaled in two passes: 4x at denoise 0.5 (6 tiles), then 2x at denoise 0.35 (30 tiles). These are the first two images produced with this method, unedited and not cherry-picked. Click any image for the full-size WebP.

<table>
<tr>
<td align="center"><a href="samples/cyberpunk-city.webp"><img src="samples/cyberpunk-city.webp" alt="Cyberpunk city, original" width="100%"></a><br><sub>Original, 1024x576</sub></td>
<td align="center"><a href="samples/cyberpunk-city-4k.webp"><img src="samples/cyberpunk-city-4k-preview.jpg" alt="Cyberpunk city, 4x Tile Upscale (VL)" width="100%"></a><br><sub>4x, denoise 0.5, 6 tiles, 4096x2304</sub></td>
<td align="center"><a href="samples/cyberpunk-city-8k.webp"><img src="samples/cyberpunk-city-8k-preview.jpg" alt="Cyberpunk city, 8K Tile Upscale (VL)" width="100%"></a><br><sub>then 2x, denoise 0.35, 30 tiles, 8192x4608</sub></td>
</tr>
<tr>
<td colspan="3"><sub><b>Prompt:</b> Cyberpunk cityscape at night</sub></td>
</tr>
<tr>
<td align="center"><a href="samples/orbital-shipyard-hangar.webp"><img src="samples/orbital-shipyard-hangar.webp" alt="Orbital shipyard hangar, original" width="100%"></a><br><sub>Original, 1024x576</sub></td>
<td align="center"><a href="samples/orbital-shipyard-hangar-4k.webp"><img src="samples/orbital-shipyard-hangar-4k-preview.jpg" alt="Orbital shipyard hangar, 4x Tile Upscale (VL)" width="100%"></a><br><sub>4x, denoise 0.5, 6 tiles, 4096x2304</sub></td>
<td align="center"><a href="samples/orbital-shipyard-hangar-8k.webp"><img src="samples/orbital-shipyard-hangar-8k-preview.jpg" alt="Orbital shipyard hangar, 8K Tile Upscale (VL)" width="100%"></a><br><sub>then 2x, denoise 0.35, 30 tiles, 8192x4608</sub></td>
</tr>
<tr>
<td colspan="3"><sub><b>Prompt:</b> Interior of a kilometers-long orbital shipyard hangar, a massive capital starship under construction surrounded by scaffold gantries, crane arms, welding sparks, and swarms of worker mechs, cargo trams and crew walkways at every level, the hangar ceiling dense with lights, pipes, and docking cranes, hull plating covered in panel lines and markings, everything in sharp focus</sub></td>
</tr>
</table>

## Which node, which model

| Node | Use when |
|---|---|
| [Tile Refine](#tile-refine) | Any model that samples through a GUIDER. You upscale first and wire the sampling nodes yourself. |
| [Tile Refine (VL)](#tile-refine-vl) | Krea 2. The same wiring plus `clip`, and no positive prompt is needed. |
| [Tile Upscale (VL)](#tile-upscale-vl) | Krea 2, everything in one node: upscale, then refine. |

The VL nodes were built for and tested with Krea 2. Other models, such as Krea 2 Turbo or other Qwen3-VL based models, are untested.

## The inputs that matter

Everything else (models, sampler wiring, seed, steps) behaves exactly as in the standard ComfyUI sampling nodes. Each input below links to its feature section for the deeper mechanics.

| Input | Nodes | What it does and why you would change it |
|---|---|---|
| [`max_tile_width` / `max_tile_height`](#dynamic-tile-layout) | all | Largest pixel size the model sees per tile, context rings included. Set to the largest size your model handles well. |
| [`context_anchor`](#context-rings) | all | Width of the context ring each tile conditions against. It is what holds neighboring tiles, and the area around a mask, together. |
| [`context_overlap`](#context-rings) | all | Width of the band neighboring tiles share. 0 gives hard seams. Smooth gradients want more, detailed scenes less. |
| [`anchor_source`](#anchor-source) | VL | What the context ring shows: the unmodified input for maximum fidelity, or the in-progress result so flawed content can be repaired. |
| [`vlm_method`](#conditioning-on-the-vl-nodes) | VL | What the model is told about each tile. Adding captions repairs flawed content during upscale, because naming a thing removes the ambiguity that let it stay mush. |
| [`mask`](#masked-refine) | Refine, Refine (VL) | Refines only the masked region and leaves everything outside it untouched. |
| `clip` | VL | The vision-language encoder that builds all tile conditioning. Tested only with Qwen3-VL as used by Krea 2. |

Preview any layout with the [tile simulator](https://blakeem.github.io/ComfyUI-ContextAnchoredTileRefine/tile-simulator.html).

**Jump to:** [Installation](#installation) | [The nodes](#the-nodes) | [Features](#features) | [Example workflows](#example-workflows)

## Installation

Search for "Context-Anchored Tile Refine" in ComfyUI Manager, or clone into your `custom_nodes` folder:

```
git clone https://github.com/blakeem/ComfyUI-ContextAnchoredTileRefine
```

## The nodes

### Tile Refine

![Context-Anchored Tile Refine](refine-node.png)

The base node, for any model that samples through a GUIDER, including models whose VAE uses video-style 5-D latents such as Krea 2. Upscale the image however you like first, then feed it in; wire `guider`, `sampler`, `sigmas`, `vae`, and `noise` as you would for SamplerCustomAdvanced. The optional `mask` refines only that region. Only the tiling geometry is tunable; how the seam itself is hidden is baked in.

### Tile Refine (VL)

![Context-Anchored Tile Refine (VL)](vl-refine-node.png)

The vision-conditioned variant, built for Krea 2. Inputs are the base node's plus `clip` and the two VL selects, and it needs no positive prompt: each tile's positive conditioning is built from the image itself (see [Conditioning on the VL nodes](#conditioning-on-the-vl-nodes)). The guider's positive prompt is ignored; its negative still applies. Non-VL encoders (SD/SDXL CLIP, T5, plain Qwen3) are rejected with a clear error. The `mask` works here too and keeps the global view (see [Masked refine](#masked-refine)).

### Tile Upscale (VL)

![Context-Anchored Tile Upscale (VL)](vl-upscale-node.png)

The whole flow in one node: image in, refined image out. It upscales the entire image first, through the optional `upscale_model` if connected, then a single lanczos pass to exactly `input size x upscale_by`. It then runs the same VL tile refine as Tile Refine (VL).

Noise, sampler, schedule, and CFG guidance are built inside the node from widgets, so no custom-sampling nodes are needed. The optional `negative` input is the one text channel that applies; left unconnected it behaves as an empty prompt. `denoise 0.0` skips diffusion and returns the pure upscale. No mask input: for a region pass, use Tile Refine (VL).

## Features

### Tiling geometry

#### Dynamic tile layout

*Nodes: [Tile Refine](#tile-refine) | [Tile Refine (VL)](#tile-refine-vl) | [Tile Upscale (VL)](#tile-upscale-vl)*

The grid is solved from the image size and `max_tile_width` / `max_tile_height`, the largest size the model is allowed to see per tile, and every crop lands on 8 pixel boundaries. Tiles are extracted, sampled, and pasted back at their native pixel size, never resized or resampled, so no tile ever loses quality to interpolation.

#### Context rings

*Nodes: [Tile Refine](#tile-refine) | [Tile Refine (VL)](#tile-refine-vl) | [Tile Upscale (VL)](#tile-upscale-vl)*

Each tile is sampled oversized by two rings that are cropped away afterwards: `context_overlap`, the band it shares with neighbors, and `context_anchor`, pure context beyond that. The anchor ring is what the tile conditions against; on the base node it always shows frozen finished content, while on the VL nodes [`anchor_source`](#anchor-source) picks whether it shows the unmodified input or the in-progress result, updated every step.

### Seams on the base node

The base node refines tiles one after another in raster order, so each tile's top and left neighbors are finished before it is sampled. Seams are hidden by conditioning first, then a narrow handover.

#### Anchor conditioning

*Nodes: [Tile Refine](#tile-refine)*

The anchor ring shows each tile its already-refined neighbors, frozen, so the tile continues them instead of drifting apart from them. This is the main seam mechanism; the blending below is only the thin handover.

#### Directional feather

*Nodes: [Tile Refine](#tile-refine)*

Both tiles diffuse the shared overlap band from the same raw pixels, and the two results are cross-dissolved: the seam-most 10 percent of the band stays fully on the new tile, then a squared ramp carries it to zero. Squared matters because it lands on zero with zero slope; a linear ramp meets the neighbor's pixels at an angle the eye reads as a line.

#### Minimum error boundary cut

*Nodes: [Tile Refine](#tile-refine)*

Dynamic programming finds the path through the overlap band where the two refinements already agree, from Efros and Freeman, *Image Quilting for Texture Synthesis and Transfer* (SIGGRAPH 2001). Unlike the paper's hard cut, the feather's midpoint bends along that path, so the handover follows image content instead of running dead straight.

#### Brightness and color match

*Nodes: [Tile Refine](#tile-refine)*

Tiles diffused separately land at slightly different brightness and color levels. The fix is a variant of gain compensation from Brown and Lowe, *Automatic Panoramic Image Stitching* (IJCV 2007): additive, per channel, and sequential. The shared band is the only place both tiles refined the same raw pixels, so the per-channel median of the difference there is pure disagreement, and subtracting it puts each tile on its neighbor's level. Runs only at tile seams, never at a mask edge.

### Seams on the VL nodes

The VL nodes do not refine one tile after another. They run a different engine with nothing to correct afterwards.

#### Synchronized latent tiling

*Nodes: [Tile Refine (VL)](#tile-refine-vl) | [Tile Upscale (VL)](#tile-upscale-vl)*

The whole image becomes one shared canvas latent, and every tile is stepped together, one sigma at a time; between steps the tiles are consolidated back into that canvas, with the overlap bands cross-dissolved by the same directional feather in latent space, and each tile is handed the consolidated view back before the next step. Both sides of every band decode one latent, so this engine needs no boundary cut and no color match. The per-step fusion of overlapping windows is the MultiDiffusion family (Bar-Tal et al., ICML 2023); here each tile runs a stock sampler full length, held at a barrier every step, so multistep sampler state is preserved and no sampler needs rewriting.

#### Anchor source

*Nodes: [Tile Refine (VL)](#tile-refine-vl) | [Tile Upscale (VL)](#tile-upscale-vl)*

`source image` (default) shows the ring the unmodified input, presented slightly ahead of the tile in the schedule so the tile follows it. Maximum fidelity: placement, style, and objects stay locked to the input, including its flaws. `live canvas` rewrites the ring every step to the neighbors' in-progress result, so the refine can reinterpret or repair damaged content; expect more invention and slightly brighter output.

#### Supported samplers

*Nodes: [Tile Refine (VL)](#tile-refine-vl) | [Tile Upscale (VL)](#tile-upscale-vl)*

Stepping every tile against one schedule requires timing each sampler's model evaluations, so the VL path supports `euler`, `heun`, `dpm_2`, `dpmpp_2m`, `dpmpp_2m_sde` (all variants), `exp_heun_2_x0`, and `exp_heun_2_x0_sde`. Any other choice is rejected with a clear error before sampling starts; `dpm_fast`, `dpm_adaptive`, and `uni_pc` own their own schedule and cannot be supported. The base node accepts every sampler.

#### Whole-image preview

*Nodes: [Tile Refine (VL)](#tile-refine-vl) | [Tile Upscale (VL)](#tile-upscale-vl)*

The sampling preview shows the entire image sharpening once per step. The raster engine can only preview the one tile it is currently sampling; the synchronized engine holds every tile in one shared canvas latent, so the whole picture exists at every step and can be shown. A live status line under the node's progress bar names the phase the run is in — upscaling, captioning, encoding, sampling with its percent, decoding — and clears when the run ends.

### Conditioning on the VL nodes

The classic tiled-upscale failure: a prompt describes the whole image, each tile holds only part of it, and a prompt-adherent model re-creates prompt objects inside tiles that should not contain them. The VL nodes replace the prompt with conditioning that is true for each tile, built by the same Qwen3-VL encoder wired to `clip`. Two distinct operations are involved: encoding the whole image through the vision tower (shared by all tiles), and writing a caption from one tile's crop (local to that tile). `vlm_method` picks which fills each tile's positive.

#### Vision tokens

*Nodes: [Tile Refine (VL)](#tile-refine-vl) | [Tile Upscale (VL)](#tile-upscale-vl)*

The whole image is encoded once through the vision tower, producing a grid of position-aware tokens, and each tile's positive is its own slice of that grid, with boundary cells shared by both neighbors. No text is involved: the image being refined is its own prompt. Tokens carry what each cell actually holds rather than demands, and every tile reads the same encode, so nothing phantom is introduced and tiles agree on tone, palette, and structures that cross seams. The slicing is region-of-interest token selection, the conditioning-space analogue of RoIAlign (He et al., *Mask R-CNN*, ICCV 2017). Fastest method: one encode serves every tile.

#### Captions

*Nodes: [Tile Refine (VL)](#tile-refine-vl) | [Tile Upscale (VL)](#tile-upscale-vl)*

The same VL model writes a short description of each tile from that tile's crop alone, used as that tile's prompt. Naming content removes ambiguity, so ambiguous artifacts or mushy areas in the source are repaired toward the named thing. Cost scales with tile count; no whole-image encode is built.

#### Vision tokens and captions

*Nodes: [Tile Refine (VL)](#tile-refine-vl) | [Tile Upscale (VL)](#tile-upscale-vl)*

The default: the tile's vision slice followed by its own caption in a single positive. The shared encode keeps the tile faithful to the picture and coherent with its neighbors; the caption names what is there and adds detail. Costs one shared encode plus one caption per tile.

### Regions, batches, and control

#### Masked refine

*Nodes: [Tile Refine](#tile-refine) | [Tile Refine (VL)](#tile-refine-vl)*

With a `mask`, the node crops to the masked region plus a `context_anchor` border, refines only that region against the frozen surrounding pixels, and composites it back with a 1px anti-aliased edge; the rest of the image is left byte for byte untouched. On the VL node the whole image is still encoded once and the region's tiles slice their true place in that encode, so the region is refined aware of everything around it. Feed an inverted mask on a second pass to refine, for example, background and character separately with different settings.

#### Batches

*Nodes: [Tile Refine](#tile-refine) | [Tile Refine (VL)](#tile-refine-vl) | [Tile Upscale (VL)](#tile-upscale-vl)*

Each picture in a batch is refined on its own, so peak VRAM does not scale with batch size and every picture gets its own conditioning, its own seam placement and color match on the base node, and its own synchronized run on the VL nodes.

#### Guider and ControlNet

*Nodes: [Tile Refine](#tile-refine) | [Tile Refine (VL)](#tile-refine-vl) | [Tile Upscale (VL)](#tile-upscale-vl)*

The `guider` input takes any guider, including NAG for models without negative prompt support. ControlNet works on the base node: the control hint is re-cropped to each tile, so depth, canny, or pose guidance lands on the right pixels; build the hint at the same size as the input image. The VL nodes ignore ControlNet (a warning is logged), because every tile's positive is replaced by its vision conditioning and there is nothing for a hint to attach to. GLIGEN, area masks, and reference latents pass through unchanged.

## Example workflows

- [Krea 2 upscale workflow](Krea%202.json)
- [Krea 2 8K upscale workflow](Krea%202%208k%20upscale.json) (the two-pass 4x then 2x chain the sample images above were made with)
- [Krea 2 refine workflow](Krea%202%20(refine).json)
- [Chroma + Z-Image hybrid workflow](Chroma%20+%20z-image%20Hybrid%20workflow.json)

## License

[GPLv3](LICENSE)
