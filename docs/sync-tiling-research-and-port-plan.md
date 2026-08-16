# Sync tiling: research findings and the production port plan

2026-08-15. Two web+source research sweeps (opus-dev agents), run while the sync-tiles
A/B campaign (tests-AB/run_ab_sync.py arms 20-22, run_ab_sync_market.py arm 23) was
in flight. This file is the keep-on-hand record for the production port that starts
AFTER A/B testing settles. Memory files `stepwise-sampler-layer-research` and
`sync-generality-research` hold the compressed versions; this is the working copy.

## 1. What the sync method is (as proven in the harness)

One shared canvas latent; NO tile advances to sigma[i+1] until every tile has stepped
at sigma[i]. Per tile per step: an ordinary masked sampler call over the tile window
(binary denoise mask = core+overlap, anchor ring = frozen context), then a directional
feather write-back in latent space (seam-most cell pinned 1.0). Ring content = frozen
RAW canvas re-noised, presented on the shipped lead curve (run-global sigma_first).
No DC match, no min-error cut — both sides of a band decode ONE latent (measured band
residue 0.11-0.24/255 = VAE window framing, every run). Conditioning: sliced vision
rows + text-only captions (arm 2's surface).

Judged results: arm 22 = new best on the portrait ("beats 9-cap-time-switch50 in
every way"); arm 23 = market stress at 6 tiles / dpmpp_2m / anchor 128 — "looks
better than it ever has", first-ever consistent floor texture, tone -0.45/255
(the +3.94 portrait tone drift is scene-dependent, not method-constant).

Open defect (one arm weird at a fixed seam line): **CLOSED**. It was arm 23 — the
market scene at `context_overlap` 32 under a 4x upscale. Arm 24 changed ONLY
`context_overlap`, to 128, keeping the same source-image lead ring and the same
dpmpp_2m, and the owner judged it 2026-08-14: "Looks perfect to me! I don't see a
single issue." So the root cause was OVERLAP SIZE at a high upscale factor — not the
lead ring and not dpmpp_2m, the two candidates arm 24 held fixed while the defect
disappeared. Those identical defaults then carried the four-scene sweep (arms 25-28
plus the -cap/-vlcap surface runs) and the 2026-08-16 shipping verdict, and
`anchor_source="source image"` ships as the DEFAULT.

## 2. Research sweep A: is there a per-step sampler layer? (verdict: NO — and a better design)

### The stack, and where systems cut it

```
Stage 2  guider.sample(...)          CFGGuider
Stage 5    KSAMPLER.sample           noise_scaling -> ... -> inverse_noise_scaling
Stage 6      sampler_function        <-- multistep STATE is loop locals, born here
Stage 7        for i in steps:
Stage 8          denoised = model(x, sigma)      <-- 1..k evals per step
Stage 9          x = f(x, denoised, history)
```

- FUSION (cut inside Stage 8: model-eval hook): A1111 MultiDiffusion, Forge,
  ComfyUI-TiledDiffusion, comfy core's `context_windows` (1-D only). One sampler
  frame, one state — the state problem never exists. CANNOT express: per-tile
  anchor rings, per-tile solver identity, mid-schedule handoffs.
- SLICING (re-enter Stage 2/5 per segment): TiledKSampler, restart_sampling,
  Impact RegionalSampler, Extra-Samplers mixture, WanVideoWrapper — ALL silently
  degrade multistep samplers to first order at every boundary.
- STEPPER (Stage 6 inverted into a stateful object): diffusers' scheduler.step
  design. In comfy land: only RES4LYF's `state_info` protocol (its own rk_beta
  solver family ONLY; discards the incoming latent on resume). Nothing general
  exists. Comfy core has NO injection point inside Stage 7, by design.

### The three sigma-slicing traps (and our current engine's status)

1. CONST models re-apply `noise_scaling`/`inverse_noise_scaling` at every Stage-5
   entry/exit — naive slicing shrinks/divides the latent by mismatched (1-sigma)
   factors, compounding. OUR ENGINE IS THE ONLY CORRECT IMPLEMENTATION FOUND
   ANYWHERE: the canonical `process_latent_out(x/(1-sigma))` storage form (the
   arm-15 truncate/resume identity) makes each per-step call bit-exact.
2. Brownian-tree SDE samplers build their tree over the SLICE's sigma range — a
   different noise path per step even with a fixed seed. Fix: build ONE tree over
   the global range and inject via `extra_options["noise_sampler"]` (comfy
   forwards it; Impact-Pack discovered this independently). Our seeds_2-family
   sampler uses default_noise_sampler, so our StepNoise injection already covers it.
3. `transformer_options["sample_sigmas"]` becomes the 2-length slice: comfy
   hooks.py LoRA keyframe percentages read max(sample_sigmas) (silently wrong),
   HunyuanVideo indexes the schedule (hard break). Benign for our Krea 2 runs
   without hook-keyframed LoRAs — a DOCUMENTED LIMITATION of the per-step engine.

### Sampler scope (comfy's 41 KSAMPLER_NAMES)

- 6 stateless per step: euler, euler_cfg_pp, heun, heunpp2, dpm_2, exp_heun_2_x0.
- 12 RNG-stream only (free once a globally-built noise_sampler is injected): all
  ancestral + SDE, incl. exp_heun_2_x0_sde, dpmpp_sde, ddpm, lcm, seeds_2/3.
- 21 need hand-unrolled step functions with explicit state: dpmpp_2m (+cfg_pp),
  dpmpp_2m_sde family, dpmpp_3m_sde family, lms, ipndm(_v), deis,
  gradient_estimation(+cfg_pp), res_multistep family, er_sde, sa_solver(+pece).
- 2 impossible (own their schedule): dpm_fast, dpm_adaptive. uni_pc: internal
  history, treat as impossible.

We hand-unrolled exactly one (dpmpp_2m, selftest bit-identical) and injected noise
for one (exp_heun_2_x0_sde). That path scales at 2-4 weeks + permanent upstream
tracking for the rest. Do not take it. Instead:

### THE PORT ARCHITECTURE: barrier threads (decided 2026-08-15)

Run each tile's ORDINARY full-length `guider.sample()` on its own thread. A barrier
inside the model callable holds every tile at each step until all arrive; the canvas
surgery (band feather blend, ring refresh) happens in place on the tensors held at
the barrier; then all tiles release. Properties:

- 39/41 samplers with ZERO per-sampler unrolling — each tile's solver state lives
  untouched on its own stack. Remaining per-sampler knowledge: evals-per-step only
  (to identify the step boundary among model calls).
- The anchor ring and denoise mask SURVIVE (each tile is a normal masked sampler
  call — the thing FUSION cannot keep).
- All three slicing traps vanish: no re-entry, honest sample_sigmas, native
  Brownian range. Bit-identical to a stock run when the surgery no-ops.
- ONE prepare_sampling per tile instead of per step — removes the per-call
  overhead that costs minutes per render today.
- Thread precedent in core: the multigpu thread pool. Estimated ~1 week.
- Open design points for the port: which model call is a step boundary per
  sampler (evals-per-step table); in-place surgery contract (both neighbours'
  window tensors hold copies of shared bands — blend must write both); sequential
  GPU execution under the barrier (one tile evals at a time, or batched into one
  model call later); mask path and B>1 composition.

Side-project verdict: the layer is genuinely unclaimed (zero arXiv hits for
"synchronized tiled diffusion" / "per-tile sampler"; SyncTweedies — the taxonomy
paper — assumes sigma_t=0 throughout) but commercially marginal: the ecosystem
avoided the problem by cutting at FUSION. Build it as OUR node's infrastructure.

## 3. Research sweep B: does sync generalize across models? (verdict: generalize the GROUNDING, not the loop)

The phantom-object cause in tiled diffusion is the ungrounded shared global prompt —
stated independently by FIVE sources (AccDiffusion with isolating ablations;
C-Upscale; Tiled Prompts — CFG amplifies irrelevant global text into hallucinations;
SuperGen; pkuliyi2015's wiki 2 years earlier). Matches our AB37-40 result exactly.
Grounding channels already exist on the target models and are consumable by the
EXISTING general node through conds.py's per-tile ControlNet route, no new code:

| Model      | Grounding channel                          | Effort      |
|------------|--------------------------------------------|-------------|
| SD1.5      | controlnet-tile (the classic)              | config-only |
| SDXL       | xinsir Union tile                          | config-only |
| Flux 1 dev | jasperai Upscaler CN (weak at low strength)| config-only |
| Z-Image    | Fun Tile CN — FIRST-PARTY, Apache-2.0      | config-only; recommended first A/B |
| Krea 2     | our VL vision-row slicing (no tile CN)     | done        |

DECISION: keep the general (non-VL) refiner node as the grounded fallback for
non-Krea models; ship sync as the Krea 2-class method. The synchronized branch of
the ecosystem is dead precisely because it re-ports per architecture. Porting the
CONDITIONING method, if ever: Qwen-Image-Edit, HiDream-O1-Image (MIT, ungated,
pixel-space). Flow-port cost for the sync loop on v-pred models: "adapt the
one-step prediction" (FrescoDiffusion, arXiv 2603.17555) — one point, nothing else.

## 4. Paper positioning (for the RoI vision-token-slicing paper)

- Nearest published neighbour: C-Upscale (arXiv 2505.16976, IJCV 2025) — MUST READ
  before writing. Synchronized overlapping tiles merged per step, per-region LLaVA
  prompts, t = 0.45T, SD3 flow DiT, real-photo eval. Its seam treatment is
  "overlap 512", no feather/cut/DC/ring — our seam machinery is the differentiator.
- TWO claims graded ABSENT from the literature: (1) per-tile vision-token slicing
  of one shared whole-canvas VL encode (core claim survives; only prior art is the
  AdvancedRefluxControl custom node, not a paper); (2) per-step shared-noise-VALUE
  alignment across overlapping SDE tiles (PoreDiT, arXiv 2604.10171, publishes the
  1/N variance-contraction argument that motivates it; everything published is
  offset randomization or shared INITIAL noise only).
- Incumbent contrast: Ultimate SD Upscale re-diffuses finished pixels for its seam
  fix (seam_fix_denoise defaults 1.0 in the comfy port; its own wiki: "just another
  redraw passes"); open issues #69/#78/#103 describe the artifacts our DC match /
  feather / prime-directive-2 design exists to prevent.
- Seam-jitter prior art (for the boundary-shifting experiment): SpotDiffusion
  (per-timestep shifted offsets) — SLICING-shaped, safe there only with stateless
  steps; our sync version keeps every pixel at one sigma.

## 5. Owner's roadmap notes (2026-08-15)

- dpmpp_2m is the focus sampler (~2x faster than exp_heun_2_x0_sde; far faster to
  test). The barrier port makes sampler choice free anyway.
- Future target: larger upscales at denoise 0.35 with larger context_overlap once
  d0.50 is solved — 8K output as the goal ("hasn't happened with any upscaler of
  this kind"). Sync scales linearly: tiles stay native-sized, the canvas latent is
  tiny, so 8K = more tiles x same per-tile cost (plus the barrier port's batched
  evals to claw back overhead).
