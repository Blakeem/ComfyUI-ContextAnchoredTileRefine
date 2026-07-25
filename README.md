# Context-Anchored Tile Refine

A single ComfyUI node for tiled upscaling and for refining masked areas of large images without processing the whole image. It refines an already-upscaled image tile by tile with no visible seams. Upscale the image however you like first, then feed it in.

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
| `context_overlap` | INT | Width of the blend band shared with an already-processed neighbor. 0 gives hard seams. Applies only where a tile borders an earlier tile. |
| `mask` | MASK (optional) | Refine only this region. Everything outside it is left untouched. |

There is nothing else to tune. How the seam itself is hidden is baked in, because testing settled it once and the answer did not depend on the image. What does depend on the image is tiling geometry, which is why those four inputs remain.

`context_anchor` and `context_overlap` default to 32, which is invisible on most scenes. Raise `context_overlap` toward 128 for large smooth gradients such as an open sky, where there is no texture for the seam to hide in. Detailed scenes need less, not more.

## How seams are hidden

Tiles are processed in raster order, so each tile's top and left neighbors are already refined when it is sampled. `context_anchor` gives every tile a frozen border of that finished content to condition against, so tiles continue each other instead of drifting.

`context_overlap` keeps the raw input of the shared band and diffuses it twice, once from each side, each reaching out to its `context_anchor`, then blends the two results together. Both start from the same raw pixels and each saw the other side as context, so they agree and the seam disappears. This happens only where a tile borders a tile that was already processed, never at the image border.

### The blend

Two things happen in that band: a feather decides how the handover fades, and a routed path decides where it sits.

The feather holds the pixels next to the seam fully on the new tile, then falls off to nothing at the outer edge. It reaches zero with zero slope, so there is no junction against the neighbor's untouched pixels. A plain linear ramp does leave one: two independent refinements of the same content differ slightly in brightness, and a linear ramp turns that difference into a bounded gradient that the eye reads as a line. Concretely the curve holds the seam-most 10 percent of the band solid, then falls off as a squared ramp.

The path comes from the **minimum error boundary cut** in Efros and Freeman, *Image Quilting for Texture Synthesis and Transfer*, SIGGRAPH 2001. Their problem is different from ours, but the sub-problem is identical: two overlapping patches, and a need to pick the cut between them that is least visible. Their answer is to build the squared difference between the two patches across the overlap, then run dynamic programming to find the connected path through it with the lowest total error. That path threads through the pixels where the two already agree, so switching there shows nothing.

This node builds the same error surface between a tile's refinement and its already-refined neighbor, runs the same DP, and then uses the result differently: the path positions the feather rather than acting as a hard cut. Image quilting cuts along the path and pastes; here the feather's midpoint is drawn toward it, so the handover follows the image content instead of running dead straight. A straight boundary is easy to spot even when it is faint. One that bends around detail is not.

Both parts are fixed rather than exposed. Four alternatives were built and compared on a night sky with a radiating moon and a close-up face: a straight transition, a meander along seeded noise, a ramp landing exactly on the routed path, and a hard cut with no blending. Every one measured the same, because the brightness difference between two tiles spans the whole tile and no choice of handover position inside a 32 pixel band can move it. The hard cut was the only one that differed visibly, and it lost, tearing at a high-contrast silhouette edge.

Open [`docs/tile-simulator.html`](https://blakeem.github.io/ComfyUI-ContextAnchoredTileRefine/tile-simulator.html) in a browser to preview the tile layout for any image size and settings.

You can see how I use it in my [`personal ComfyUI workflow`](Chroma%20+%20z-image%20Hybrid%20workflow.json).

## Masked refine

With a `mask`, the node crops to the masked region plus a `context_anchor` border, refines only that region against the frozen surrounding pixels, and blends it back with a 1px anti-aliased edge. The rest of the image is left byte-for-byte untouched. Feed an inverted mask on a second pass to refine, for example, the background and the character separately with different settings.

## Guider and ControlNet

The `guider` input lets you use Perp-Neg Guider (or any guider). The tradeoff: the guider's conditioning covers the whole image and is not re-cropped per tile, so ControlNet is not supported.

## License

[GPLv3](LICENSE)
