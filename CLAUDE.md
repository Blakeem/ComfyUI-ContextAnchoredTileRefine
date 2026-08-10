# Context-Anchored Tile Refine, project guide

ComfyUI custom node package, one tiling engine behind four nodes. The image nodes refine an
already-upscaled IMAGE by dynamic tiling (or only a masked region, leaving the rest untouched;
upscaling happens outside the node). The video node (`ContextAnchoredTileUpscaleVLVideo`,
MiniMax H3) upscales a whole clip in-node and then tile-refines it the same way.
Target: ComfyUI 0.3.45+ (video node needs an H3-capable tree), V1 node schema,
Python 3.12, torch 2.9.

## Prime directives (highest priority, override convenience)

1. **Quality first, efficiency second.** Output image quality is the top priority and is never
   traded for speed, memory, or simpler code. Optimize only *after* quality is guaranteed, and
   never in a way that risks a visible quality regression. **Never resize, resample, or apply any
   lossy operation to a tile.** Tiles are extracted, processed, and pasted back at their native
   pixel size (multiples of 8 by construction).

2. **Never re-diffuse finished pixels that survive into the output.** Running diffusion again over
   already-refined content compounds grit with the samplers this node targets, visibly, even once
   and even untiled. This is the root constraint behind every seam decision below.

3. **Seams are hidden by conditioning, not by blending.**
   - *Between tiles:* each tile is sampled oversized (core + `context_overlap` + a frozen
     `context_anchor` halo). The anchor halo encodes from the live canvas, so the tile sees its
     already-refined neighbors and is drawn to continue them. On sides bordering an
     already-processed neighbor (top/left in raster order), the `context_overlap` band is diffused
     from the FROZEN RAW source by both tiles independently, and the two results are cross-dissolved
     by a thin directional feather. Prohibited: a wide blend of two independent refinements
     (ghosting), and a double hard-paste of a shared strip (compounding artifacts).
   - *At a mask boundary:* no feather at all. The masked region is diffused against the frozen
     background as context, then composited back with a 1px anti-alias only. An inward feather
     would under-process a ring around the subject; an outward feather would re-diffuse finished
     background (directive 2).

## Architecture (respect these invariants)

- `context_anchored_tile_refine/node.py`: the V1 nodes (`INPUT_TYPES` / `VALIDATE_INPUTS` /
  `refine`). `ContextAnchoredTileRefine` normalizes and validates the optional MASK;
  `ContextAnchoredTileRefineVL` subclasses it (required CLIP, no prompt input) and
  routes through `refine_image(vl_clip=...)`; `ContextAnchoredTileUpscaleVL` is the
  all-in-one variant (widgets replace the NOISE/SAMPLER/SIGMAS/GUIDER inputs, optional
  UPSCALE_MODEL + negative, no mask) — it runs `upscale.prepare_upscaled` on the whole
  image, builds the sampling objects via `upscale.py`, and calls the same
  `refine_image(vl_clip=...)`; `ContextAnchoredTileUpscaleVLVideo` is the MiniMax H3
  all-in-one (IMAGE frames of ONE clip in, optional UPSCALE_MODEL + av_latent, no cfg /
  negative / prompt — H3 is CFG-free) — it runs the RAM-safe stage order (2 fps picks
  upscaled and VL-encoded BEFORE the full-clip canvases exist; the guider placeholder IS
  the global encode, no second CLIP load) and calls `video.refine_video`. Comfy-free at
  module scope (the combo lists come from a lazy `import comfy.samplers` inside
  `INPUT_TYPES`).
- `context_anchored_tile_refine/grid.py`: pure grid math (tile layout: `core`,
  `overlap_inner_rect`, `crop_rect`, `paste_rect`; `solve_axis`, `build_layout`).
  `solve_axis(multiple=)` sets the pixel granularity crops land on: 8 default (image,
  bit-identical), 32 for H3 (VAE 16x x DiT patch 2). **Stdlib only**, no torch, no comfy.
- `context_anchored_tile_refine/sampling.py`: the pipeline. `refine_image` is the entry point.
  Without a mask it delegates to `_refine_tiles` (pad to /8, solve the grid, per-tile
  encode/sample/decode from a live canvas, directional-feather composite, crop back). With a mask
  it crops to the mask bbox plus `context_anchor`, gates every tile's denoise mask to the region,
  and composites back through a 1px anti-aliased edge, leaving everything outside byte-identical.
  **torch-only at module scope**; `comfy` / `latent_preview` are imported lazily inside functions
  (a subprocess test pins this). `make_tile_progress` guards a nested x0 before previewing
  (core hands nested-latent callbacks a NestedTensor; the guard previews stream 0, a no-op
  for image latents).
- `context_anchored_tile_refine/conds.py`: per-tile ControlNet support. `refine_image` validates
  every control hint against the full input size (hard error on mismatch) and bbox-slices it on
  the mask path; `_refine_tiles` pads the hints like the canvas and, per tile, swaps
  `guider.original_conds` for fresh control-chain copies carrying the tile's `crop_rect` slice
  (exact-size crop = core's hint rescale is an identity), restoring the pristine map in
  `try/finally`. Control objects are duck-typed (`copy()` / attributes) — **torch-only at module
  scope, comfy never imported at all** (same subprocess test pins it). Without a `control` cond
  the guard keeps the pipeline byte-identical. `gligen`/`area`/`mask`/`reference_latents` pass
  through untouched by design (unresolved mask-path coordinate semantics; cropping
  `reference_latents` would regress Kontext-style workflows).
- `context_anchored_tile_refine/vl.py`: the VL node's conditioning. The whole padded canvas is
  area-resampled ONCE to `GLOBAL_SLICE_BUDGET` (/32-snapped so the merged-patch grid is exact)
  and encoded through the CLIP's vision path with NO text (Krea 2's own template, explicit —
  the default image template would survive the strip and shift the layout). Each tile's
  positive becomes its row slice of that encode: `[0]=vision_start, [1..N]=grid rows (raster),
  [N+1]=vision_end, tail]`, cells intersecting the tile's `crop_rect` (boundary cells shared by
  both neighbors — the row-space overlap band). A/B-settled (AB26-AB36): vision rows are
  positionally exact and demand-free, one shared encode keeps the cross-tile story coherent,
  and ANY text (user prompt, generated style, captions) re-admits phantom objects in proportion
  to its volume — hence no prompt input at all. Fail-fast guards: non-VL CLIP (tokenizer
  rejects images / no image token), encoder seq-length vs token-derived layout. The per-tile
  positive swap composes with the ControlNet swap in the same pristine-map try/finally; the
  guider must keep `positive` in `original_conds` (CFGGuider convention). On the mask path
  the FULL image is encoded and each region tile's rect is offset by the bbox origin
  (`slice_indices` offsets), so a masked refine stays globally informed. **torch-only at
  module scope**, comfy lazy (subprocess test pins it).
- `context_anchored_tile_refine/upscale.py`: the all-in-one nodes' internals. Whole-image
  upscale stage (`prepare_upscaled`: optional model pass mirroring core ImageUpscaleWithModel
  — version-defensive around `.patcher`, OOM tile-halving — then at most ONE lanczos to the
  exact `upscale_by` target; a same-size resize is skipped because core's lanczos is an 8-bit
  PIL round trip, so it would be a quality loss, not a no-op; `prepare_upscaled_video` runs
  the same stage in 8-frame chunks into a preallocated fp16 canvas — a whole-clip lanczos
  would materialize ~15 GB fp32 at once) plus in-process builders
  mirroring the core custom-sampling nodes (`Noise_RandomNoise`, `build_sigmas` ==
  BasicScheduler incl. denoise<=0 -> empty sigmas -> refine returns the upscale untouched,
  `build_guider` == core CFGGuider — required, its `original_conds` convention is what the VL
  positive swap keys on — `build_basic_guider` == core Guider_Basic for the CFG-free H3 path,
  same `original_conds` convention — and `encode_empty` for the placeholder positive /
  default negative). **torch-only at module scope**, comfy lazy (subprocess test pins it).
- `context_anchored_tile_refine/vl_video.py`: the video node's conditioning, the H3 analogue
  of `vl.py`. Core's ref-video presentation mirrored VERBATIM (every 12th frame, sequential
  half-second timestamps, empty prompt); ONE `clip.tokenize("", minimax_ref_items=[...])`
  encode bracketed by an 8 GiB `EXTRA_RESERVED_VRAM` raise (restored in try/finally — the
  24k-token forward otherwise runs ~0.6 GiB of margin on a 24 GB card); the row layout is
  derived by WALKING the token stream (per-block strides vary), never assumed, and asserted
  against the encoder output; per tile `slice_rows` keeps every text row plus the 32px grid
  cells the crop intersects with `minimax_token_tags` index-selected in lockstep (the DiT's
  AdaLN modality runs key on them). Picks are pre-resampled only above the tokenizer's
  per-frame `max_pixels=12845056` cap so its own bilinear resize never fires. **torch-only
  at module scope**, comfy lazy (subprocess test pins it).
- `context_anchored_tile_refine/video.py`: the video pipeline, `refine_video` entry point.
  Frames repeat-padded onto H3's 17k+5 grid (trimmed back at the end), reflect-padded /32;
  fp16 raw/live canvases with the image path's discipline (bands + core encode from RAW,
  anchor ring from LIVE); nested AV latents per tile with the audio stream ALWAYS present
  and frozen (`(video mask, zeros_like(audio))` — a missing stream mask defaults to ones
  and would REGENERATE the audio); audio-length fail-fast against core's `temporal_shape`
  formula; ONE nested canvas noise draw sliced per tile at /16; per-tile positive swap in
  the same pristine-map try/finally; the production composite (plateau/falloff feather,
  min-error routing, DC match — reductions run over the frame axis, one offset and one cut
  per clip, temporally stable) blended fp32 in 16-frame chunks. **torch-only at module
  scope**, comfy lazy (subprocess test pins it).
- The denoise mask handed to the sampler is always **binary**. ComfyUI re-applies it every step,
  so a fractional cell is only ever partially denoised and leaves an under-refined halo at low
  step counts. `sample_latent` hands it over pre-normalized to the canonical float32 form on
  the guider's load device — [B,1,h,w] for a 4-D latent, [B,1,1,h,w] for a 5-D video-family
  latent (the fixed points of core's `prepare_mask`) — a value no-op for core guiders, and it
  shields guider packs whose copied `sample()` lacks core's mask prep from ever seeing a raw
  CPU mask. Noise is drawn from a dummy mirroring `vae.encode`'s latent layout (`latent_dim` 3
  → 5-D), and `_refine_tiles` fails fast if a tile's encoded latent and noise slice disagree.
- Curated ComfyUI API references: `docs/reference/INDEX.md`. Tile-layout playground:
  `docs/tile-simulator.html`, a self-testing mirror of `grid.py`.

## Tests / gates

Venv python: `C:\Users\Blake\Documents\ComfyUI\.venv\Scripts\python.exe` (do not `pip install`).
- Default gate, must be green with **0 skips**: `<venv> -m pytest tests -m "not gpu"`
- GPU tests, real SD1.5 sampling: `<venv> -m pytest tests -m gpu -v`
- Markers: `comfy`, `gpu`, `slow`. GPU tests load
  `models\checkpoints\v1-5-pruned-emaonly-fp16.safetensors`.
- The no-mask path is pinned byte-for-byte by hand-computed value tests. Treat them as the
  regression net for any change to `_refine_tiles`.
- **One GPU sampling job at a time** on this machine (24 GB 3090 Ti): a single 3x-canvas
  refine peaks ~19-20 GiB reserved, so a second concurrent sampler (another harness process,
  or the ComfyUI app with models resident) spills into the Windows sysmem fallback — an
  order of magnitude slower, and it can end a run as a SILENT native crash with no Python
  traceback. Check `nvidia-smi` is idle before launching. At 3x+ also render one config per
  process (`tests-AB\... --only <label>`): a long multi-config process dies the same silent
  way even alone (allocator-state accumulation; seen on Z-Image 2026-07 and Krea 2 2026-08-09).
- Final judgement on seams is always the owner's visual A/B in ComfyUI, not a metric.

## Release

`pyproject.toml` `version` drives the Comfy Registry. Pushing a change to `pyproject.toml` on
`main` triggers `.github/workflows/publish_action.yml`, which needs the `REGISTRY_ACCESS_TOKEN`
repo secret. `.comfyignore` keeps `docs/`, `tests/`, `conftest.py`, and `.github/` out of the
published archive.

## Conventions

- Match ComfyUI core naming and comment style; comment only non-obvious constraints.
- Keep `grid.py` stdlib-only, and `node.py` / `sampling.py` / `conds.py` / `vl.py` /
  `vl_video.py` / `video.py` module scopes comfy-free (lazy imports; `conds.py` never imports
  comfy anywhere — control objects are duck-typed, and `vl.py` duck-types the CLIP the same
  way).
- Prefer views over copies and slice noise rather than redrawing it, but only after the prime
  directives above are satisfied.
