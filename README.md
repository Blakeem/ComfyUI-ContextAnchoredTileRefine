# Context-Anchored Tile Refine

A single ComfyUI node for tiled upscaling and for refining masked areas of large images without processing the whole image. It refines an already-upscaled image tile by tile with no visible seams. Upscale the image however you like first, then feed it in.

## Node inputs

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

## How seams are hidden

Tiles are processed in raster order, so each tile's top and left neighbors are already refined when it is sampled. `context_anchor` gives every tile a frozen border of that finished content to condition against, so tiles continue each other instead of drifting.

`context_overlap` keeps the raw input of the shared band and diffuses it twice, once from each side, each reaching out to its `context_anchor`, then blends the two results together. Both start from the same raw pixels and each saw the other side as context, so they agree and the seam disappears. This happens only where a tile borders a tile that was already processed, never at the image border.

Open [`docs/tile-simulator.html`](https://blakeem.github.io/ComfyUI-ContextAnchoredTileRefine/tile-simulator.html) in a browser to preview the tile layout for any image size and settings.

## Masked refine

With a `mask`, the node crops to the masked region plus a `context_anchor` border, refines only that region against the frozen surrounding pixels, and blends it back with a 1px anti-aliased edge. The rest of the image is left byte-for-byte untouched. Feed an inverted mask on a second pass to refine, for example, the background and the character separately with different settings.

## Guider and ControlNet

The `guider` input lets you use Perp-Neg Guider (or any guider). The tradeoff: the guider's conditioning covers the whole image and is not re-cropped per tile, so ControlNet is not supported.

## License

[GPLv3](LICENSE)
