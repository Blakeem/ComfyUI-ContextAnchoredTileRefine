# Context-Anchored Tile Refine

A single ComfyUI node for tiled upscaling, and for refining a masked area of a large image without processing the rest. It refines an already-upscaled image tile by tile with no visible seams. Upscale the image however you like first, then feed it in.

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

Tiles diffused separately also land at slightly different brightness and color levels, around 1/255. That step covers a whole tile, so no choice of where to hand over can move it. Stitching pipelines fix this with an **exposure compensation** stage between finding the seam and blending it, as in Brown and Lowe, *Automatic Panoramic Image Stitching using Invariant Features*, IJCV 2007. Theirs corrects a multiplicative gain. Here it is additive and per channel, because the measured offset does not scale with brightness and is partly color.

The shared band is the only place two tiles have refined the same raw pixels, so the difference between them there carries no content, only the disagreement. The node takes the median of that difference per channel and subtracts it before compositing, putting the tile on its neighbor's level. The first tile has no processed neighbor, so it sets the level. Median rather than mean matters because the offset comes off before the error surface above is built. A mean is dragged by the few pixels where the two results genuinely disagree, leaving a constant pedestal that makes every path cost about the same. The median removes the typical disagreement instead, so what is left describes texture and the cut has something to follow. Under a mask, only band pixels inside the masked region count, since the rest were never diffused.

Open the [tile simulator](https://blakeem.github.io/ComfyUI-ContextAnchoredTileRefine/tile-simulator.html) to preview the tile layout for any image size and settings.

You can see how I use it in my [personal ComfyUI workflow](Chroma%20+%20z-image%20Hybrid%20workflow.json).

## Masked refine

With a `mask`, the node crops to the masked region plus a `context_anchor` border, refines only that region against the frozen surrounding pixels, and composites it back with a 1px anti-aliased edge. The rest of the image is left byte for byte untouched. Feed an inverted mask on a second pass to refine, for example, the background and the character separately with different settings.

## Guider and ControlNet

The `guider` input lets you use Perp-Neg Guider, or any other guider. The tradeoff is that the guider's conditioning covers the whole image and is not re-cropped per tile, so ControlNet is not supported.

## License

[GPLv3](LICENSE)
