# Context-Anchored Tile Refine, project guide

ComfyUI custom node package, three nodes over ONE tile geometry and TWO engines: the base
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
  for why never mid-list). `anchor_source` takes its option strings from `sync.ANCHOR_SOURCES`
  and `vlm_method` from `captions.VLM_METHODS`, so what the widget offers and what the engine
  branches on cannot diverge. Comfy-free at module scope (the combo lists come from a lazy
  `import comfy.samplers` inside `INPUT_TYPES`, and the option strings from lazy package
  imports).
- `context_anchored_tile_refine/grid.py`: pure grid math (tile layout: `core`,
  `overlap_inner_rect`, `crop_rect`, `paste_rect`; `solve_axis`, `build_layout`).
  `solve_axis(multiple=)` sets the pixel granularity crops land on: 8 is the default and
  the only value production uses. The parameter and its `multiple=32` tests stay — the
  granularity is a property of the model family (32 = VAE 16x x DiT patch 2), so any
  coarser-grid model reuses this solver rather than forking it.
  **Stdlib only**, no torch, no comfy.
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
  scope**, comfy lazy (a subprocess test pins it).
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
  **CANVAS SPACE, the rule every block follows**: a lane's `latent_image` is handed over in RAW
  space (comfy applies `process_latent_in` itself inside `guider.sample`) and is always a SLICE
  of the one C_0, so two overlapping lanes hold equal values on every shared cell at step 0;
  the canvas this engine MAINTAINS lives in PROCESS space, because the lanes' live `.x` tensors
  are already there. The two spaces are never mixed — a window headed for `vae.decode` is
  converted back with `process_latent_out`. `check_preconditions` runs before any GPU time
  (strictly decreasing sigmas ending at 0, a CONST flow model, core's default
  `scale_latent_inpaint`, and — live-canvas only — `sigmas[0] < 1`, where that mode's `x / (1 -
  sigma)` algebra is defined). The lead ring is entered ONCE around the whole lane set (see the
  ANCHOR RING invariant). **torch + stdlib at module scope**; comfy is lazy and so is stepper.py
  (a subprocess test pins the comfy half).
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
  rejects images / no image token), encoder seq-length vs token-derived layout. The slices are
  built ONCE per run by the sync engine's pre-pass and handed to `sync.build_lane_guiders`, which
  gives each lane its OWN guider copy carrying that tile's positive — the caller's guider is
  never swapped and there is nothing to restore; it must keep `positive` in `original_conds`
  (CFGGuider convention). On the mask path
  the FULL image is encoded and each region tile's rect is offset by the bbox origin
  (`slice_indices` offsets), so a masked refine stays globally informed. **torch-only at
  module scope**, comfy lazy (subprocess test pins it).
- `context_anchored_tile_refine/captions.py`: the `vlm_method` surfaces that are not pure
  vision rows. Per-tile VLM captions generated from the tile's own crop by the SAME CLIP that
  encodes the vision rows: `clip.tokenize(instruction, images=[...], thinking=True)` ->
  `clip.generate(do_sample=False, repetition_penalty=1.05)` -> `strip_thinking` (mandatory —
  core's plain `TextGenerate` does NOT strip, and an unstripped `<think>` block reaches the
  DiT as hundreds of tokens of the model talking to itself) -> `clean_caption`. The settled
  instructions are A/B-decided and **character-for-character frozen, EU spelling included**
  (US spelling measurably dropped detail in the owner's A/B). BOTH caption surfaces ask
  `RICH_GROUPED_INSTRUCTION` (768) since 2026-08-16; `SETTLED_POSITION_INSTRUCTION` (512) is
  RETIRED from the table but stays defined, because tests-AB's judged "pos" arms pin
  themselves to it. The caption input is an area-resampled COPY at 384^2
  total px — never the sampled tile (prime directive 1). `build_caption_conds` encodes the
  caption as plain text; `build_slice_caption_conds` concatenates, on the ROW axis, the
  tile's slice of ONE shared pure-vision canvas encode and that tile's caption encoded
  TEXT-ONLY — so the canvas encode cost is one for the whole image, exactly as on the
  vision-only surface. It used to put the caption INSIDE the canvas encode, at one canvas
  encode PER TILE; the owner's A/B retired that (far-canvas content leaked into every tile's
  caption rows — the phantom moon), and the vision rows are provably unchanged by the switch
  because attention is causal (`docs/vl-conditioning-encode-cost.md` sections 6-7 and its
  2026-08-16 addendum). **torch-only at module scope**, comfy lazy (subprocess test
  pins it). Search history: `tests-AB/vlm_prompt_lab.py`, 7 rounds.
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
