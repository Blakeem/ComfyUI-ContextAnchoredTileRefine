# Context-Anchored Tile Refine, project guide

ComfyUI custom node package, three production nodes plus a four node testing chain
(`testing.py`) over ONE tile geometry and TWO engines: the base
node's raster path (`sampling._refine_tiles`) and the VL nodes' synchronized path (`sync.py`
over `stepper.py`), which since 1.6.0 is the only VL path. They refine an
already-upscaled IMAGE by dynamic tiling (or only a masked region, leaving the rest
untouched; upscaling happens outside the node), except the all-in-one variant which
upscales in-node first. Target: ComfyUI 0.3.45+, V1 node schema, Python 3.12, torch 2.9.

A MiniMax H3 VIDEO node (`ContextAnchoredTileUpscaleVLVideo`, `video.py`, `vl_video.py`)
was built and then REMOVED on 2026-08-13: the spatial tile method is seamless on static
shots and unusable under camera motion, where the seam is a fixed line in a moving field
that no anchor/overlap setting removes. What was learned is kept, not the code —
`docs/h3-video-chunking-findings.md` holds the full result set and the temporal-chunking
approach that replaces it, and `tests-AB/` keeps every H3 harness. Do not re-add it
without that doc's temporal design.

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
   - *Between tiles, BASE node (the raster path, `sampling._refine_tiles`):* each tile is sampled
     oversized (core + `context_overlap` + a frozen
     `context_anchor` halo). The anchor halo encodes from the live canvas, so the tile sees its
     already-refined neighbors and is drawn to continue them. On sides bordering an
     already-processed neighbor (top/left in raster order), the `context_overlap` band is diffused
     from the FROZEN RAW source by both tiles independently, and the two results are cross-dissolved
     by a thin directional feather. Prohibited: a wide blend of two independent refinements
     (ghosting), and a double hard-paste of a shared strip (compounding artifacts).
   - *Between tiles, VL nodes (the sync path, `sync.py`):* the seam is PREVENTED, not hidden —
     there is no earlier tile to hide it from. Every tile is a lane of ONE run over ONE shared
     canvas latent, all stepped together per sigma, and the per-step consolidation feathers the
     bands at LATENT scale and scatters the result back, so no two lanes ever disagree on a shared
     cell at a step start. Both sides of every band therefore decode ONE latent, which is why this
     path runs NO min-error cut and NO DC match: neither has anything left to correct. The raster
     rules above still govern the base node, unchanged.
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
  `refine_image(vl_clip=..., sampler_name=...)` — the widget NAME rides along beside the built
  SAMPLER so a sampler the sync engine rejects is named as the user picked it (core's
  `sampler_object` wraps several names in a private function). Both VL nodes carry two selects,
  defined ONCE each as `_anchor_source()` / `_vlm_method()` so their option lists and tooltips
  cannot drift apart, and both APPENDED after `context_overlap` (see the ANCHOR RING invariant
  for why never mid-list), and an OPTIONAL `prompt` STRING SOCKET (`_prompt()`, `forceInput`,
  shared with the Captions test node, after `mask` on the refine node and after `negative` on
  the upscale node; a socket never enters widgets_values, so its place in the list is free of
  the positional rule) (2026-09-16). Each VL node resolves the preset ITSELF,
  `captions.with_prompt(captions.resolve_method(vlm_method), prompt)`, and hands it down as
  `refine_image(preset=...)`, so an unconnected prompt against a preset that asks for `{PROMPT}`
  fails before any VAE or VL encode; the engine's own resolve in `sync._prepare_run` is now the
  direct-caller path only. `anchor_source` takes its option strings from `sync.ANCHOR_SOURCES`
  and `vlm_method` from `captions.vlm_methods()`, so what the widget offers and what the engine
  branches on cannot diverge. Comfy-free at module scope (the combo lists come from a lazy
  `import comfy.samplers` inside `INPUT_TYPES`, and the option strings from lazy package
  imports). `check_geometry` (2026-09-03) is the /8 and range rule for the four tile widgets in
  ONE place, called by the base node's `VALIDATE_INPUTS` and the Layout test node's. Both VL
  nodes carry `IS_CHANGED(s, **kwargs)` returning `captions.settings_fingerprint()` (see the
  captions.py bullet).
- `context_anchored_tile_refine/grid.py`: pure grid math (tile layout: `core`,
  `overlap_inner_rect`, `crop_rect`, `paste_rect`; `solve_axis`, `build_layout`).
  `solve_axis(multiple=)` sets the pixel granularity crops land on: 8 is the default and
  the only value production uses. The parameter and its `multiple=32` tests stay — the
  granularity is a property of the model family (32 = VAE 16x x DiT patch 2), so any
  coarser-grid model reuses this solver rather than forking it.
  **Stdlib only**, no torch, no comfy.
  `SubLayout` / `neighborhood` / `sub_layout` (2026-09-03) are the block math the Render test
  node and the tile phantom harness run ONE block of a grid with: `sub_layout(layout, col0,
  col1, row0, row1)` returns `block` (what a run samples), `region` (what it denoises), a
  block-local `Layout` built from FORCED axis solutions (the parent base, `last` = the span
  minus the other bases), `first_col` / `first_row` and `parent_tiles`. Per axis: n == 1 keeps
  the tile's own crop as the block and its overlap_inner span as the region. n >= 2 starts the
  block at the first in-range CORE when a tile lies beyond that side (`col0 > 0`, read from
  the RANGE and never from a clamped rect, which is the harness bug this replaced: a crop
  clamped to 0 at column 1 is not evidence that no tile lies before it) and ends at the last
  crop, the last tile absorbing that ring as core. The region is the block inset by ctx on a
  side that does not touch the canvas edge (the RECT predicate, the harness reference's own,
  which differs from the range when the final column is no wider than r). RING REACH: with
  n >= 2 and a side beyond, `base < r` raises a named ValueError, because the block's tiles
  clamp their rings at the block edge while the parent's reach past the bordering tile, so no
  origin reproduces the crops. `_check_block_crops` then asserts every block crop equals the
  parent's, or sits exactly r inside it on a dropped-ring side. The harness's retired
  `block_rect` / `block_layout` / `block_mask` are inlined as the reference in
  `tests/test_grid.py`, so the judged geometry is pinned without a GPU.
- `context_anchored_tile_refine/sampling.py`: the pipeline. `refine_image` is the entry point
  and the ONE place all three nodes meet, which is why every cross-cutting behaviour belongs
  here rather than in a node. **A batched IMAGE is refined ONE PICTURE AT A TIME**: `B > 1`
  recurses per row and cats the results, so a tile latent is always `[1,C,h,w]` and peak VRAM
  never scales with batch size. That is a correctness fix, not just a memory one — the seam
  DC offset (`seam_dc_offset`) and the min-error cut (`seam_displacements`) both REDUCE over
  the batch axis, so before this every picture in a batch shared one offset and one cut
  measured across unrelated content. The canvas noise dummy is still drawn at the FULL batch
  and row `batch_index` selected from it, so each picture keeps the noise it always had;
  ControlNet hints take the matching row (`conds.slice_hint_row`). `B == 1` is byte-identical
  throughout and is what the hand-computed value tests pin.
  **THE DISPATCH: a `vl_clip` hands the WHOLE refine to the sync engine** (`sync.refine_sync`,
  lazily imported — sync.py imports this module at ITS module scope, so the pair is acyclic only
  while that stays inside the function), mask or no mask: refine_sync owns the bbox crop, the
  region gate and the anti-aliased composite itself. `_check_sync_intake` is the fail-fast that
  runs FIRST, before any VAE or VL encode (both cost minutes of GPU time to reach the same
  rejection deep inside the engine): the sampler must be in `stepper.EVALS_PER_STEP` (checked by
  the caller's `sampler_name` first, so the message names the widget's string) and the schedule
  must be strictly decreasing and end at 0. Everything below the dispatch is the BASE node's
  raster path; since 1.6.0 the raster path has NO VL branches at all — they were deleted, not
  flagged off (the fallback for non-VL models is the base node itself).
  Without a mask the raster path delegates to `_refine_tiles` (pad to /8, solve the grid, per-tile
  encode/sample/decode from a live canvas, directional-feather composite, crop back). With a mask
  it crops to the mask bbox plus `context_anchor`, gates every tile's denoise mask to the region,
  and composites back through a 1px anti-aliased edge, leaving everything outside byte-identical.
  **torch-only at module scope**; `comfy` / `latent_preview` are imported lazily inside functions
  (a subprocess test pins this). `make_tile_progress` guards a nested x0 before previewing
  (core hands nested-latent callbacks a NestedTensor; the guard previews stream 0, a no-op
  for image latents). `anchor_ring_schedule` is the ring's context manager, entered by the SYNC
  engine and disabled on the raster path — see the ANCHOR RING invariant below.
  `preset` / `tile_captions` / `layout` / `noise_fields` (2026-09-03) are the sync engine's
  OVERRIDES, forwarded to `refine_sync`: a caller running one block of a larger grid hands the
  block the settings block, the captions, the tile rects and the SDE field that grid's own run
  used. Any of them with `vl_clip=None` raises (the raster path reads none of them), and all
  but `preset` raise on a batch above 1, since they describe one picture and the picture loop
  forwards `preset` only.
- **The frozen region is presented on a SCHEDULE, not re-noised** (`anchor_ring_factor` /
  `anchor_ring_schedule` in `sampling.py`). Since 1.6.0 the schedule is the SYNC path's, gated by
  the **`anchor_source` widget** on the two VL nodes: `"source image"` (the default) leaves every
  lane's ring on the unmodified input and presents it on this schedule, `"live canvas"` rewrites
  the ring's CONTENT per step to the neighbour's live trajectory (`sync.present_live_ring`) and
  does NOT enter the schedule at all — the curve exists to bridge a frozen REFINED ring to a
  noisy core, and in live-canvas mode nothing frozen-refined is left to lead. Exactly one of the
  two is ever active. `sync._refine_canvas` enters the context manager ONCE around the whole lane
  set (one instance patch on the shared model; the manager is NOT re-entrant, and per-call
  normalization equals run normalization because the sigmas handed over ARE the full run), and
  the raster path passes `anchor_ring=False` ALWAYS — the base node never schedules its ring.
  The widget is APPENDED after `context_overlap`, never
  inserted mid-list: the ComfyUI frontend restores `widgets_values` positionally and
  `migrateWidgetsValues` no-ops when the length changes, so a mid-list insert silently shifts
  every saved workflow's tuned values. Originally settled by
  the owner's visual A/B 2026-08-13
  (`tests-AB/run_ab_matrix.py`, scene `portrait`, baseline vs Lead x {d0.35, d0.50} x
  {seed 42, 1234}): with VL slices the schedule wins across scenes, but on the plain-
  conditioning node it was consistently worse — a distorted ear, duller colour and texture at
  the seam. The mechanism fits: the ring resolves ahead of the core so the core follows it,
  and only the VL positive tells a tile what its neighbourhood contains; without that the tile
  is pulled toward a ring it cannot interpret. Core rebuilds the model's
  input every step as `x*mask + scale_latent_inpaint(...)*(1-mask)` (`comfy/samplers.py:639`),
  and its default re-noises the frozen region to the CURRENT sigma — so the `context_anchor`
  ring reaches the model as mostly noise exactly while structure is decided. Instead the ring
  is presented at `sigma * anchor_ring_factor(sigma/sigma_first)`: matched to the core at step
  0, LEADING it (resolving first, so the core follows) down to `ANCHOR_RING_RELEASE`, then
  smoothstep-rejoining so the last steps generate their own texture. Owner-A/B settled at
  production scale against core's default and five other curves; it removes freckle
  amplification and a white-spotting artifact. Applies to the mask path's frozen background
  too. Models that already hold the frozen region clean (WAN21/WAN22/HunyuanVideo/LTXAV
  override `scale_latent_inpaint`) are detected via `__mro__` and SKIPPED on the raster path —
  their override is the endpoint this curve approaches — and REJECTED outright on the sync path
  (`sync.check_preconditions`), whose ring construction is derived from core's default and
  mis-scales silently against an override. The patch goes on the model INSTANCE and is restored in
  `finally`: core caches model objects session-wide, so a class patch would follow the model
  into every other node. An A/B harness overrides `sampling.anchor_ring_factor` — the single
  swap point — never comfy, which the instance patch would shadow.
- `context_anchored_tile_refine/stepper.py`: the sampler-portability layer — N LANES, ONE shared
  sigma schedule, STOCK samplers. Each lane is one tile running an ORDINARY full-length
  `guider.sample()` on its own cooperative thread; a barrier inside the model callable holds every
  lane at each sigma step, the last arriver runs the caller's surgery hook with the whole fleet
  parked, then all release. Because every lane runs the stock k-diffusion function end to end on
  its own stack, multistep state (dpmpp_2m's `old_denoised`, a Brownian stream's position) needs
  no unrolling and no resume identity — the sigma-slicing traps in
  `docs/sync-tiling-research-and-port-plan.md` are structurally absent because nothing is sliced.
  **The only per-sampler knowledge left is `EVALS_PER_STEP`**: how many model evals a sampler
  makes per sigma step (and on a step whose NEXT sigma is 0), which is what times the barrier.
  `SUPPORTED_SAMPLERS` is that table's key set — everything else is rejected BY NAME, and a lane
  that runs long or short raises rather than silently mistiming the surgery. A
  `threading.Condition` token keeps the lanes cooperatively SERIAL (one comfy call at a time, one
  GPU stream): the threads buy independent stacks, never parallelism. Two invariants that are
  correctness, not tidiness — every lane needs its OWN guider (CFGGuider stores per-run state on
  itself), and after its last eval a lane waits at the **FINAL EXIT BARRIER** so no guider
  teardown (`cleanup()` → `current_patcher = None`, dereferenced on EVERY eval) can precede
  another lane's final eval. A stochastic sampler draws from ONE canvas-wide field sliced per
  window (`build_noise_fields`), never a per-lane one — two independent fields meeting in a band
  IS the seam this engine exists to remove. A lane failure, a hook failure or a user cancel sets
  one abort flag and the FIRST exception reaches the caller unchanged; both catch sites take
  BaseException because comfy's `InterruptProcessingException` is one. **torch + stdlib at module
  scope**, comfy lazy (a subprocess test pins it). `offset_noise_fields(fields, dy, dx)`
  (2026-09-03) wraps a provider so a window given in BLOCK cells reads the FULL canvas field at
  the window plus the block's origin, and None passes through.
- `context_anchored_tile_refine/sync.py`: the VL path's engine — the run's components, the run
  loop (`_refine_canvas`) and the region path (`refine_sync`, which owns the bbox crop and the
  1px anti-aliased composite back). Stages: solve the grid ONCE and run the VL conditioning
  pre-pass over THAT layout, so a tile's positive can never be sliced for a rect it does not
  sample; per-tile `vae.encode` of the RAW crop windows with the butted CORES assembled into ONE
  canvas latent C_0 and coverage asserted (never a whole-canvas VAE call — its ~21 GiB spike is
  why the engine tiles at all); ONE canvas-shaped noise draw sliced per lane, the identical
  contract as the raster path so a picture keeps the noise it would have had; one lane per tile,
  each with its OWN guider carrying that tile's positive; ONE stepper run whose per-step hook
  consolidates the lanes into the maintained canvas (directional feather at LATENT scale, raster
  order) and scatters every window back, so no two lanes disagree on a shared cell at a step
  start; then per-tile decode of the CANVAS windows — never a lane's own `x`, whose ring cells
  carry the unrefined source — composited by the stock pixel feather.
- **The two VAE windows are sized to what each stage KEEPS, not to the sampled extent**
  (`VAE_ENCODE_MARGIN` / `VAE_DECODE_MARGIN` and `encode_window` / `decode_window` in
  `sync.py`). Both calls used the whole `crop_rect` until 2026-08-22, which is
  `r = context_anchor + context_overlap` wider than either keeps, and every discarded cell was
  already computed by the neighbour whose core covers it. At the owner's 8K config that was 53%
  of every encode and 32% of every decode, 15.5s and ~2.4 GiB per pass. The LANES are untouched:
  a lane's latent is still the full `crop_rect` slice of C_0, so every tile still sees its whole
  ring. `None` restores the old window and is what the A/B harness sweeps. The two shipped
  values are NOT interchangeable and are pinned by a test:
  **DECODE 0** (the kept rect exactly) is provably free, because every `paste_rect` border lands
  where the pixel feather weights that tile ZERO — top and left are where its own alpha starts
  at 0, right and bottom are a neighbour's core start where the neighbour's alpha is 1 — which
  is why the measured seam delta was exactly 0.000/255 on all 5 scenes.
  **ENCODE 32**, not 0, because the encode's kept rect is the CORE and cores butt into C_0 with
  weight 1 on both sides, with no feather protecting that boundary. At margin 0 the window edge
  coincides with it and `tests-AB/probe_c0_core_seam.py` measures the decoded C_0 boundary
  damped to 0.807 of its neighbourhood against 1.184 shipped; at 32 it is 1.184, and 32 is the
  smallest tested margin that is. A bigger margin is NOT safer in general: the Wan 2.1 VAE runs
  a full spatial attention block at the /8 bottleneck in both `Encoder3d.middle` and
  `Decoder3d.middle` whatever its empty `attn_scales` suggests, so every kept cell reads the
  whole window and the error has no analytic bound. Settled by owner A/B 2026-08-22
  (`tests-AB/run_ab_vae_window.py`, 5 scenes x 5 decode margins sharing ONE sampled canvas, then
  3 encode margins as full renders); `tests-AB/probe_vae_window_vram.py` holds the timings.
  **CANVAS SPACE, the rule every block follows**: a lane's `latent_image` is handed over in RAW
  space (comfy applies `process_latent_in` itself inside `guider.sample`) and is always a SLICE
  of the one C_0, so two overlapping lanes hold equal values on every shared cell at step 0;
  the canvas this engine MAINTAINS lives in PROCESS space, because the lanes' live `.x` tensors
  are already there, and is cast to **float32** because `vae.encode` writes at
  `intermediate_dtype()` (fp16 under `--fp16-intermediates`) while every lane's `x` is float32,
  so an fp16 canvas would round the consolidated trajectory once per step and scatter the
  rounded values back into every lane. The two spaces are never mixed — a window headed for
  `vae.decode` is converted back with `process_latent_out`. `check_preconditions` runs before any GPU time
  (strictly decreasing sigmas ending at 0, a CONST flow model, core's default
  `scale_latent_inpaint`, and — live-canvas only — `sigmas[0] < 1`, where that mode's `x / (1 -
  sigma)` algebra is defined). The lead ring is entered ONCE around the whole lane set (see the
  ANCHOR RING invariant). **torch + stdlib at module scope**; comfy is lazy and so is stepper.py
  (a subprocess test pins the comfy half).
  OVERRIDES (2026-09-03, the Render test node's and the harness's route into the engine):
  `_prepare_run` takes `preset` (must be a `captions.Preset` whose `surface` equals
  `method_surface(vlm_method)`, since the ledger is sized from the string and the pre-pass
  branches on the preset), `tile_captions` (`_check_given_captions`: never on the vision
  surface, never with a ledger, one entry per tile with one caption per row, and then the VLM
  pass is skipped) and `layout` (`_override_layout`: a `grid.Layout` or a `grid.SubLayout`,
  which must equal the padded canvas size AND the run's ctx and overlap, because `refine_sync`
  still cuts the region crop with `context_anchor` and every ring gate is built from the
  rects. A SubLayout's `parent_tiles` become `budget_tiles`, handed to the vision builders so a
  block samples the canvas at the full grid's density). `_refine_canvas` takes `noise_fields`
  (callable) in place of its own `build_noise_fields(sampler, canvas.shape, ...)`, since a field
  drawn at a block's shape and origin gives every lane injections the full run never made.
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
- `context_anchored_tile_refine/vl.py`: the VL node's conditioning. Each tile's positive is
  two pure vision encodes through the CLIP's vision path with NO text (Krea 2's own template,
  explicit, since the default image template would survive the strip and shift the layout),
  concatenated on the row axis by `build_vision_rows`: the CROP rows, every cell of the
  tile's own crop resampled to `crop_tokens` x 1024 px and encoded ALONE (one encode per
  tile), then the CANVAS rows, the tile's row slice of ONE encode of the entire image
  (`slice_indices`: `[0]=vision_start, [1..N]=grid rows (raster), [N+1]=vision_end, tail]`,
  cells intersecting the tile's `crop_rect`, boundary cells shared by both neighbors, the
  row-space overlap band), with the template tail once after the last block. Both counts are
  the settings file's `[vision]` table (`captions.VisionSettings`, read per run through the
  `Preset` every surface carries). 0 turns a source off and both at 0 is rejected at load.
  The canvas is sampled so a tile's share of it holds about `canvas_tokens` cells
  (`canvas_budget_pixels`: tokens x 1024 x canvas area / MEAN crop area, capped at
  `PICTURE_CAP_MEGAPIXELS` = 2 MP, past which the owner measured the VLM breaking down), so
  the sample grows with the tile count and a tile gets the same rows at every image size.
  Rows still vary by tile, since edge tiles are smaller and boundary cells are shared.
  Settled by the owner's block A/B of 2026-09-02 (TESTS.md test 10,
  `tests-AB/run_ab_tile_phantom.py`): a cell carries the picture it was encoded in, so the
  canvas slice of a flat sky tile grows a copy of the image's salient object (a tower in the
  clouds), larger the more canvas rows it gets (70 rows at 0.79 MP grew two towers, 165 rows
  at 2 MP a skyline), and the crop rows of that tile hold only sky and cancel the demand.
  About 100 crop rows do that without redrawing a content tile, where 200 swirl the tile's
  own subject and 768 gouge it, so the shipped 165 canvas and 100 crop tokens are the judged
  point. A 3x3 NEIGHBORHOOD WINDOW encode (2026-09-01 to 2026-09-02) sat between the two and
  failed like the canvas slice, so it was removed, not flagged off. A/B-settled (AB26-AB36):
  vision rows are positionally exact and demand-free, and ANY text (user prompt, generated
  style, captions) re-admits phantom objects in proportion to its volume, so there is no
  prompt input for the DiT at all (the VL nodes' `prompt` widget fills the caption QUESTION
  only, see captions.py). The rows are a bag. Krea 2's DiT gives every conditioning row RoPE
  position 0 (measured bit-exact under a shuffle), so the block order carries nothing.
  Fail-fast guards: non-VL CLIP (tokenizer rejects images / no image token), encoder
  seq-length vs token-derived layout, and a real `pooled_output` or stray extras on a
  concatenated block (`cat_rows`. Krea 2 returns None on every encode). One interrupt check
  per tower pass. The rows are built ONCE per run by the sync engine's pre-pass and handed to
  `sync.build_lane_guiders`, which gives each lane its OWN guider copy carrying that tile's
  positive. The caller's guider is never swapped and there is nothing to restore. It must
  keep `positive` in `original_conds` (CFGGuider convention). On the mask path the encode
  source is the FULL image at the bbox origin (`slice_indices` offsets, `crop_picture`
  clamps), so a region's canvas rows are the entire image's and a masked refine sees the
  image around the mask. **torch-only at module scope**, comfy lazy (subprocess test pins
  it). `budget_tiles` (2026-09-03) sizes the canvas sample off ANOTHER layout's tiles through
  `block_budget_pixels`: the full grid's budget at the full grid's own area (read back off its
  tiles), rescaled by the two source areas, so the 2 MP cap binds a block as it binds the full
  run. Asking `canvas_budget_pixels` for the block's own area instead would sit under the cap
  and give an interior tile 224 canvas rows against the full run's 195 at the 8192x4608,
  24 tile config.
- `context_anchored_tile_refine/captions.py`: the `vlm_method` surfaces that are not pure
  vision rows. Per-tile VLM captions generated from the tile's own crop by the SAME CLIP that
  encodes the vision rows: `clip.tokenize(instruction, images=[...], thinking=True)` ->
  `clip.generate(do_sample=False, repetition_penalty=1.05)` -> `strip_thinking` (mandatory —
  core's plain `TextGenerate` does NOT strip, and an unstripped `<think>` block reaches the
  DiT as hundreds of tokens of the model talking to itself) -> `clean_caption`. `_THINK_BLOCK`
  carries core's own `(?:</think>|$)` alternation and it is load-bearing: without the `|$` a
  tile whose reasoning turn exhausts `max_tokens` returns that REASONING as its caption,
  non-empty, so `generate_caption`'s fallback chain never fires and the model's deliberation
  becomes the tile's whole positive. **The instructions live in `settings.toml` at the repo root since 2026-08-21, as NAMED
  PRESETS since 2026-08-22, beside a `[vision]` table since 2026-09-02** (`load_settings`
  returns `Settings(vision, presets)`, `resolve_method` / `vlm_methods`), deployed with the
  node. The `[vision]` table (`canvas_tokens`, `crop_tokens`, `caption_megapixels`) is
  validated like a preset, rides on every `Preset`, and is what vl.py samples by. `settings.user.toml` beside it is the USER'S own copy and wins whenever it exists,
  which is what makes an edit survive a node update (`.gitignore`d, never written by the
  package). TWO READ CADENCES, deliberately: the PRESET LIST is read once per session by
  `vlm_methods` (an `lru_cache`) because it becomes a combo the frontend caches at startup, so
  a new or renamed preset needs a restart, while a preset's own wording is re-read per run by
  `resolve_method` so tuning a prompt does not. Each `[presets.<label>]` block adds ONE option
  per caption surface, grouped by preset and in file order, every one read `"<surface>
  (<label>)"`, the FIRST preset (the DEFAULT) included since 2026-09-16 (before that its two
  options carried no label, which hid which preset the default was). The bare strings
  `"captions"` / `"vision tokens and captions"` are what a pre-preset workflow saved: the
  selector no longer offers them, `method_surface` accepts them, `resolve_method` routes them
  to the first preset, and the VL nodes' `VALIDATE_INPUTS` names `vlm_method` so core's
  combo-list check never rejects them. "vision tokens" reads the `[vision]` table and no preset (pinned end to end).
  THE PROMPT INPUT (2026-09-16): an instruction may carry `PROMPT_PLACEHOLDER` (`{PROMPT}`),
  which `with_prompt(preset, prompt)` fills by literal replace (never str.format, a prompt can
  hold braces) into BOTH instructions, stripping the widget's trailing newline; a placeholder
  met by a blank prompt raises there, naming the preset, the key and the input, and
  `generate_caption_set` runs `_check_prompt_filled` first so a direct caller that skipped
  `with_prompt` never captions with the literal placeholder. The prompt reaches the VL model's
  question only, never the DiT. The FIRST shipped preset is `prompted`, the owner's wording
  under test (quotes the prompt, holds the caption to the crop, no style caption), so the
  default options now REQUIRE the prompt socket connected; `standard` is second, whose wording
  and budgets are the pre-settings-file constants character for character
  (`RICH_GROUPED_INSTRUCTION`, 768 tokens). The caption
  picture size is the `[vision]` table's `caption_megapixels`, ONE size for the tile caption
  and the style caption since 2026-09-02 (a user's own copy still carrying the two per-preset
  `*_megapixels` keys fails with a message naming the move), shipped at
  `SHIPPED_CAPTION_MEGAPIXELS` (768x1024 px, settled by the owner's three-scene A/B on
  2026-09-01, TESTS.md test 3, over the 384^2 px it first shipped with). A broken file is a hard error
  before any clip.generate, and the all-in-one node resolves it FIRST, before its
  upscale-model pass and text-encoder load. The engine resolves ONCE per picture in
  `sync._prepare_run` when no caller hands a preset down, and both VL nodes now do (node.py),
  so a production run reads the file once per node execution and the engine's own resolve is
  the direct-caller path; either way the ledger's caption count and the pre-pass's own can
  never come from two different reads.
  BOTH caption surfaces ask the one `tile_caption_instruction`. A non-empty
  `global_style_instruction` adds ONE whole-image style caption per
  picture, generated FIRST from the vision-encode source (the FULL image on the mask path)
  and prepended to every tile caption, so all tiles follow one style description; "" turns
  it off, and the ledger's caption segment counts it (`preset_picture(style_rows=)` must
  match `build_tile_positives`' open). `generate_caption_set` writes the two APART, as
  `(style_texts, captions)`, `join_style_captions` puts the style on top of every tile
  caption, and `generate_tile_captions` (the engine's call) is the two together. The Captions
  test node reads the set so its listing shows the style once, labelled, and the Render test
  node joins it per lane set, so the joined form exists only where the engine reads it. The three shipped presets' wording is pinned
  character-for-character by `test_settings_toml_ships_the_owner_tested_wording` (the owner's
  testing found small wording changes lose consistency), so a deliberate prompt change updates
  pin and file together. The retired settled pair (`RICH_GROUPED_INSTRUCTION`,
  `SETTLED_POSITION_INSTRUCTION`) stays defined, character-frozen EU spelling included,
  because tests-AB's judged arms pin themselves to it (`ab_env.caption_preset` is how a
  harness asks its own pinned question). The caption input is an area-resampled COPY sized by `caption_megapixels`
  (`VL_INPUT_BUDGET` keeps the 384^2 px the judged harness arms were captioned at) and never
  the sampled tile (prime directive 1). `0` reads the crop's own size, capped at
  `vl.PICTURE_CAP_MEGAPIXELS`. `build_caption_conds` encodes the
  caption as plain text; `build_slice_caption_conds` concatenates, on the ROW axis, the tile's vision rows (the
  same `vl.build_vision_rows` the vision-only surface uses, tail left off) and that tile's
  caption encoded TEXT-ONLY, so the vision cost is exactly the vision-only surface's. It used to put the caption INSIDE the canvas encode, at one canvas
  encode PER TILE; the owner's A/B retired that (far-canvas content leaked into every tile's
  caption rows — the phantom moon), and the vision rows are provably unchanged by the switch
  because attention is causal (`docs/vl-conditioning-encode-cost.md` sections 6-7 and its
  2026-08-16 addendum). **torch-only at module scope**, comfy lazy (subprocess test
  pins it). Search history: `tests-AB/vlm_prompt_lab.py`, 7 rounds.
  THREE ADDITIONS 2026-09-03. `settings_fingerprint()` (the file name plus the sha256 of the
  file in force, `missing:<name>` rather than a raise) is what both VL nodes' `IS_CHANGED`
  return: ComfyUI re-executes a node only when an input, a widget or IS_CHANGED changed, so
  before it a settings edit under a fixed seed was served from cache and never ran.
  `preset_labels()` is the `lru_cache` sibling of `vlm_methods` the Captions test node builds
  its combo from, same once-per-session cadence for the same reason. THE CAPTION CACHE:
  `generate_caption` stores its final text in a bounded `OrderedDict` (`CAPTION_CACHE_ENTRIES`
  512) keyed by the sha256 of a float32 view of the resampled picture (bfloat16 has no numpy
  dtype), its dtype and shape, the instruction, the budget, the reasoning flag and a `scope`
  tuple naming the request's place in the run (tile crop rect, row and picture index, or
  "style"), with a weakref to the CLIP that must still be the same object on a hit. Captions
  are greedy (`do_sample=False`, so core builds no generator), so the stored text is what a
  second pass writes. The scope keeps it a CROSS-execution cache, which is what a seed re-roll
  on the production upscale node needs (minutes on a 24 tile grid) and what keeps the pure
  suite's zero-picture call counts true. `clear_caption_cache()` is the reset, and an autouse
  conftest fixture calls it before every test.
- `context_anchored_tile_refine/upscale.py`: the all-in-one nodes' internals. Whole-image
  upscale stage (`prepare_upscaled`: optional model pass mirroring core ImageUpscaleWithModel
  — version-defensive around `.patcher`, OOM tile-halving — then at most ONE lanczos to the
  exact `upscale_by` target; a same-size resize is skipped because core's lanczos is an 8-bit
  PIL round trip, so it would be a quality loss, not a no-op) plus in-process builders
  mirroring the core custom-sampling nodes (`Noise_RandomNoise`, `build_sigmas` ==
  BasicScheduler incl. denoise<=0 -> empty sigmas -> refine returns the upscale untouched,
  `build_guider` == core CFGGuider — required, its `original_conds` convention is what the VL
  positive swap keys on — and `encode_empty` for the placeholder positive / default
  negative). **torch-only at module scope**, comfy lazy (subprocess test pins it).
  `SlicedCanvasNoise(vae, seed, canvas_h, canvas_w, rect)` (2026-09-03) is the NOISE a block
  run needs: the full canvas draw from `sync.build_canvas_noise` sliced to `rect` (a clone per
  call, since the live-canvas ring zeroes the handed noise in place), plus `noise_fields(sampler,
  sigmas)`, the full canvas SDE field read at the block's cell origin through
  `stepper.offset_noise_fields`. A one tile block therefore draws a canvas-sized CPU field per
  SDE step (about 132 MB at the owner's 8K config), accepted for draws identical to the full
  run's.
- `context_anchored_tile_refine/progress.py`: the VL path's progress ledger — ONE ProgressBar
  per node execution, divided into budget segments whose UNIT is one DiT eval of one tile.
  Only the sampling segment is exact (`stepper.plan_evals` x n_tiles, sized at stepper intake
  and advanced by the step→eval-index map, because a naive per-step increment overshoots a
  2-eval sampler's final step); every other phase is a named module-level constant
  (`W_UPSCALE_STEP` / `W_CLIP_LOAD` / `K_CAPTION` / `W_ENCODE` / `W_ENCODE_CROP` /
  `W_ENCODE_CAPTION_TEXT` / `W_ENCODE_TILE` / `W_DECODE_TILE`, with `vision_encode_units`
  the one place the vision segment is sized from the `[vision]` table), i.e. calibration knobs in ONE place. **The ledger is
  created by NODES and nowhere else**: node.py's two VL nodes through `build_ledger`, and the
  Captions test node through `build_caption_ledger` (one CAPTIONS segment, one chunk per
  caption, since without the shim core's per-token bar reset the display at every caption and
  stopped where the stop token fired); `sampling.refine_image`,
  `sync.refine_sync`, `sync.build_tile_positives`, `captions.generate_tile_captions` and
  `upscale.prepare_upscaled` only ACCEPT one as `progress=None` and build nothing when it is
  None — so the base node, every direct caller and the `tests-AB` harnesses keep their old
  bars byte-for-byte, and three pinned tests assert exactly that. `refine_image`'s B>1
  recursion forwards it, so a batch shares one bar. The two invariants, and the only two: the
  emitted VALUE never decreases and the TOTAL never drops below it — segments re-fit whenever
  a true size arrives (the upscale model's step count, per-picture grids). SEGMENT ORDER
  follows the CODE per `vlm_method` (captions are written BEFORE the conditioning is built).
  **The shim** is a scoped patch of `comfy.utils.ProgressBar` held around the run: bars core
  constructs inside it (llama.py's per-token bar, sd.py's VAE tiled fallbacks, upscale.py's
  tiled_scale bar) route into the ledger's current segment instead of resetting the display,
  and a NEW inner bar resumes the segment's fill rather than restarting it (the caption retry
  chain and the OOM tile-halving retry both depend on that). A comfy MODULE-global patch is
  normally against the rules here; it is the deliberate exception, because those bars are
  built inside core functions with no instance to patch and no pbar parameter to pass — the
  ledger's own bar is created from the class captured BEFORE the patch, and the patch is
  restored in `finally` (`with ledger:`). Its scope is the `comfy.utils.ProgressBar`
  ATTRIBUTE only; the known escape is sd.py:360's module-level binding under CLIP hook
  scheduling. **Stdlib only at module scope — no torch either**; comfy is lazy (subprocess
  test pins all three).
- `context_anchored_tile_refine/testing.py`: the TILE TESTING CHAIN (2026-09-03), four nodes
  that are not production nodes and say so in each docstring, `Tile Test: Layout` /
  `Upscale` / `Captions` / `Render` in `image/upscaling/tile testing`. Built so the owner can
  tune tiles, prompts and token counts without upscaling again: ComfyUI re-executes a node only
  when an input, a widget or IS_CHANGED changed, so the chain puts the seed on the Render node
  ONLY and the upscale and the captions come from cache across a re-roll. LAYOUT (`TestLayout`,
  socket `CATR_LAYOUT`) is solved by the Layout node from the image, `upscale_by` and the four
  geometry widgets, on the target size padded to /8 with `sync._prepare_run`'s own two solves,
  and carries those widgets so no node below has a copy of its own. Its overlay is a PIL
  drawing over an area-resampled preview capped at `OVERLAY_MEGAPIXELS` (a preview, never
  sampled) with the crop, overlap and core rects and a `"{index} r{row}c{col}"` label per tile.
  The Upscale node is `upscale.prepare_upscaled` at the layout's multiplier and is OPTIONAL (an
  already upscaled image at `upscale_by` 1.0 skips it). The Captions node runs
  `captions.generate_caption_set` over the PADDED canvas from a `preset`: a settings file
  preset IGNORES the three instruction widgets (so a trial wording stays in them), and the
  `CUSTOM_PRESET` option (`"custom instructions"`, appended after the file's labels, and a
  file preset of that name is refused at INPUT_TYPES) builds a `captions.Preset` from
  `tile_instruction` (required non-empty), `style_instruction` (empty = no style caption) and
  `max_tokens` (required above 0, both budgets), and `prompt` fills `{PROMPT}` in either
  through the same `captions.with_prompt`. `style_caption` off and `caption_megapixels`
  apply to both (0 reads the crop's own size). `tiles` (csv, same parser as the Render node's,
  `_parse_tile_numbers(text, count, node_name)`) captions the named tiles, plus their
  bordering tiles from `grid.neighborhood` when `with_neighbors` is on, since a block run at
  the Render node conditions every lane; empty captions every tile. It returns `TestCaptions`
  (socket `CATR_CAPTIONS`: one entry per tile in layout order with None for an uncaptioned
  tile, each the tile's OWN caption, the `style` caption apart (None when the run asked for
  none), the named `tiles`, the target size, the grid shape and every crop rect, so the Render
  node rejects captions written for another grid; its `__str__` is the readable listing,
  the style once under a `style caption` label above the tiles, the same for a file preset
  and the custom option, because core's Preview as Text falls back to `str()`), that listing
  as `text`, and the
  named tiles as a csv `tiles` STRING for the Render node's `tiles` input. The run is ONE
  progress bar through `progress.build_caption_ledger` (hidden `unique_id` for the status
  line). Its `IS_CHANGED` is the settings fingerprint and its `VALIDATE_INPUTS` re-checks the
  entire caption_megapixels rule, since naming a widget there disables core's own range check.
  The Render node rejects a caption set with a None on any lane it runs (`_full_captions` for
  the full canvas, `_block_captions` per block, both BEFORE any model call, naming the tiles
  and the with_neighbors fix). The Render node builds its sampling objects as the all-in-one
  node does, passes `progress=None`,
  and either runs the full canvas (empty `tiles`, with `layout=` handed in so the solve cannot
  drift) or, per csv tile number, ONE REGION RUN over `grid.sub_layout`'s block: the mask is
  ones on `sub.region` through `node._normalize_mask` (image device), `_check_region_crop`
  asserts the engine's bbox plus anchor crop IS the block, the captions are the parent's at
  the block's parent indices with the style joined on top by `_lane_captions` (the ONE place
  the test chain joins them), the noise is `upscale.SlicedCanvasNoise` at the padded canvas
  plus its `noise_fields`, and the outputs are the tile's parent crop and the block cut from
  the result, as IMAGE lists. Every sub layout is solved BEFORE the first model call so the
  ring reach error is cheap. `with_neighbors` off renders the tile alone (block = its crop,
  region = its overlap_inner) and the anchor ring stays unrefined source. Every node takes ONE
  picture. torch + stdlib at module scope, PIL inside the drawing function, comfy inside
  methods (subprocess pin). The tile phantom harness runs the same `sub_layout` and
  `SlicedCanvasNoise` but NOT `noise_fields`, so its judged arms stay reproducible.
- The denoise mask handed to the sampler is always **binary**. ComfyUI re-applies it every step,
  so a fractional cell is only ever partially denoised and leaves an under-refined halo at low
  step counts. Both paths hand it over pre-normalized through the ONE shared helper
  `sampling._normalize_denoise_mask` — `sample_latent` on the raster path, `sync._prepare_run`
  per lane on the sync path (the stepper calls `guider.sample` directly, so `sample_latent` is
  not on that path at all) — to the canonical float32 form on
  the guider's load device — [B,1,h,w] for a 4-D latent, [B,1,1,h,w] for a 5-D video-family
  latent (the fixed points of core's `prepare_mask`) — a value no-op for core guiders, and it
  shields guider packs whose copied `sample()` lacks core's mask prep from ever seeing a raw
  CPU mask. Noise is drawn from a dummy mirroring `vae.encode`'s latent layout (`latent_dim` 3
  → 5-D), and both engines fail fast if a tile's encoded latent and noise slice disagree.
- Curated ComfyUI API references: `docs/reference/INDEX.md`. Tile-layout playground:
  `docs/tile-simulator.html`, a self-testing mirror of `grid.py`.

## Tests / gates

Venv python: `C:\Users\Blake\Documents\ComfyUI\.venv\Scripts\python.exe` (do not `pip install`).
- Default gate, must be green with **0 skips**: `<venv> -m pytest tests -m "not gpu"`
- **Lint gate, required before any commit**: `uvx ruff@0.16.2 check .` must report zero
  findings. The version is pinned (ruff 0.16 changed the default rule set) and the config
  lives in `pyproject.toml` `[tool.ruff]`: ComfyUI core's own selection (E/W/F/T/N805/
  S102/S307 — S102/S307/E702 are what `comfy node publish` scans at registry time) plus
  I/UP/B/C4/SIM/RUF. The `N` family and `PLC0415` stay OFF deliberately: `INPUT_TYPES(s)`
  is the core node contract and the lazy comfy imports are architecture, not accidents.
  No formatter (`ruff format` is not adopted; core does not use it). CI runs the same
  check in `.github/workflows/lint.yml`.
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
  `sync.py` / `stepper.py` module scopes comfy-free (lazy imports; `conds.py` never imports
  comfy anywhere — control objects are duck-typed, and `vl.py` duck-types the CLIP the same
  way).
- Prefer views over copies and slice noise rather than redrawing it, but only after the prime
  directives above are satisfied.
