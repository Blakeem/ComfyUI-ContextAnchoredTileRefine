# MiniMax H3 video refine: chunking and handoff findings

Record of a full investigation into refining a clip longer than one attention block, run
2026-08-11/12 against ComfyUI 0.31.0 then 0.32.0, on `minimax_h3_ref2va_pruned_int8_convrot`
at 2688x1536, denoise 0.22, 8 steps, `res_multistep`/`simple`, seed 42.

Harness: `tests-AB/run_ab_h3seam.py` (one arm per process). Analysis:
`tests-AB/build_quiet_join.py`, `chunk_position.py`, `handoff_profile.py`,
`temporal_alignment.py`, `grain_notch.py`, `compare_frame.py`.

---

## 1. The problem

A 158-frame 2688x1536 clip is 189,504 packed rows and does not fit in 24 GB. It has to be
split. There are exactly two axes to split on, and the artefact differs by axis:

```
SPATIAL   cut the FRAME into tiles      -> a fixed LINE, present on every frame
TEMPORAL  cut the TIMELINE into chunks  -> a MOMENT, present on one frame
NO SPLIT  fits ~56 frames (68,544 rows, 14.61 GiB) -> no artefact, but too short
```

Cost of each, for the full clip:

| decomposition | total rows | seams |
|---|---|---|
| spatial 2x2 | 215,072 | 4 seams x 158 frames |
| temporal 3 x 56f | **205,632** | 2 seams x 1 frame |

Temporal is cheaper AND has fewer seam-frames. That reframing is what the whole
investigation turned on.

---

## 2. Spatial tiling (the shipped VL Video node)

The seam measures **0.2-1.5/255** after DC match, is **seed-dependent** (seed moves it ~3x,
`context_anchor` moves it ~0.05), and is invisible in a static scene but salient under camera
motion — a fixed line in a moving field.

| method | result |
|---|---|
| `context_overlap` 32 -> 64 -> 128 | no reliable improvement |
| `context_anchor` 32 -> 128 | won on seed 42, did NOT replicate on seed 1234 |
| grain-notch theory (feather averaging two grain fields) | **disconfirmed** — band reads 1.035, not the predicted 0.707; `base` and `ov0` identical on the horizontal axis though `ov0` has no feather at all |
| keyframe cond block per tile (`kfprobe`) | steers output and binds positionally, but does not fix a seam |

**Owner verdict:** works well on static content and close-ups; unusable on pans and action.
Metrics on this material bottom out around 2/255 — four independent detectors missed even a
deliberately unfeathered seam (`ov0`).

---

## 3. Temporal chunking: the conditioning family (all failed)

Every arm below hands chunk N+1 a *description* of chunk N. Boundary cost is the delivered
frame-to-frame delta at source 93->94 against a same-chunk control of 9.490/255.

| method | boundary | detail vs no-boundary control (2.403) | verdict |
|---|---|---|---|
| naive splice | 1.40x | 2.309 (-3.9%) | visible glitch |
| `minimax_keyframes` (prev chunk's last frame) | 1.06x | 2.026 (**-15.7%**) | fixes motion, **costs 12-16% detail** |
| VL reference extended over prev chunk's refined frames | 1.07x | 2.029 (-15.6%) | **null**, colour ~23% worse |
| 5-frame reference video | 1.04x | 2.095 (-12.8%) | null |
| 22-frame reference video + `[video continuation]` prompt | 0.98x | 2.022 (-15.8%) | null |
| whole clip @ half res as global context (+17% rows) | 1.02x | 2.155 (-10.3%) | **null**, +31% time |
| denoise 0.16 (both chunks) | 1.02x | — | colour shift down ~60% |

**Seven reference/conditioning variants, no benefit.** H3's `minimax_refs` pathway appears to
be trained as a subject/style reference, not as temporal context — regardless of length,
resolution, timing, or whether it carries past or future.

### Why they could not work

Reference blocks carry their **own area-normalised grid** (`_frame_grid`, model.py:77-81) and
are placed at `cursor` coordinates preceding the target. Velocity is a relationship between
positions *in the target's own coordinate frame*; a block on a different sampling density and
an unrelated time origin cannot express it. Owner's observation: *"totally unaware of how fast
background elements were moving."*

---

## 4. Temporal chunking: the frozen-head family (this works)

Put the previous chunk's **actual pixels inside the target segment** instead of describing
them. Chunk N+1 starts K frames early, those frames are overwritten with chunk N's refined
output, and the denoise mask is zeroed over the latent frames they occupy. Delivery starts
after them, so nothing frozen reaches the output and nothing is re-diffused.

K must land on H3's grid — `latent_t(n) = ((n-5)//17)*5+2` — so 5 -> 2 latent, 22 -> 7,
39 -> 12.

| method | detail | DC shift R/G/B | motion (owner) |
|---|---|---|---|
| 5 frozen, anchor re-noised (`frznk`) | **2.396 (-0.3%)** | +1.23/+0.63/-0.04 | **jumps** |
| 5 frozen + keyframe (`frzn`) | 2.096 (-12.8%) | +1.28/+0.78/+0.13 | good |
| 5 frozen, hold anchor + clean label (`frznc`) | 2.175 (-9.5%) | +2.37/+2.20/+1.21 | as good as keyframe |
| **22 frozen (`frz22`)** | **2.183 (-9.2%)** | +1.92/+1.91/+1.00 | **best** |
| 39 frozen (`frz39`) | 2.125 (-11.6%) | +1.75/+1.59/+0.91 | quality drop at the cut |
| 22 frozen, anchor released below sigma 0.55 (`frz22r`) | 2.189 (-8.9%) | +3.12/+2.92/+2.02 | big jump at the cut |

Every arm costs 1345-1400 s and 14.76 GiB — **freezing more is free per chunk**; the cost is
that fewer frames are delivered, so a 158-frame clip needs 3 / 5 / 9 chunks at K = 5 / 22 / 39.

### Two supporting mechanisms

**Continued noise.** Two chunks at the same seed and canvas draw *identical* noise, indexed
chunk-locally, so the field restarts at every join. Drawing long and slicing at a global
offset continues it instead. Measured on the 5-frame arm: DC blue **+1.37 -> +0.13**, green
+2.11 -> +0.78.

**Hold anchor (`scale_latent_inpaint`).** `comfy/samplers.py:639` re-noises masked rows to the
current sigma every step, and `MiniMaxH3` does **not** override `scale_latent_inpaint` — so a
frozen head arrives as ~77% noise at step 0 (sigma 0.774) and its motion information emerges
too late to steer. WAN21 (model_base.py:1919), WAN22 (:2058), HunyuanVideo (:1391) and LTXAV
(:1237) all override it to `return latent_image`. Borrowing that is what makes the frozen head
usable from step 1.

**Clean timestep label.** Once the rows really are clean, the per-segment label (`seg_t`,
model.py:547) is wrong for them. Splitting the video segment's `mod_segments` entry gives the
frozen prefix `t = 0.999` at **zero added rows**. Only expressible because video rows are
ordered (t, h, w), making the frozen head a contiguous prefix — a spatial anchor ring is not
contiguous, which is why this lever was unavailable for the tile seam.

---

## 5. Recommended configuration

```
chunk N+1 overlaps chunk N by 22 frames
  those 22 frames = chunk N's refined output, written into the input
  denoise mask 0 over latent frames 0..6
  scale_latent_inpaint -> return latent_image        (anchor clean at every sigma)
  mod_segments split -> latent frames 0..6 at t=0.999 (zero added rows)
  canvas noise drawn long, sliced at a global offset
  NO keyframe, NO reference block, NO prompt, NO VL prepend
  deliver from frame 22

1384 s / 14.76 GiB per chunk;  5 chunks for a 158-frame clip
```

Residual: **-9.2% high-frequency detail** against an unsplit render.

---

## 6. Things that turned out to be true, and cost time to learn

- **The boundary-cost metric is unreliable for ranking.** It called the naive boundary
  invisible (owner saw it), ranked a 22-frame reference above the frozen head (owner saw the
  reverse), and scores the best arm worst. It is sound for LEVEL (DC, brightness) and useless
  for MOTION and SHARPNESS.
- **Frame alignment is 1:1** in every arm (`temporal_alignment.py`, delta +0 throughout) — the
  perceived speed-up is not an indexing bug.
- **VAE round trip costs 1.849/255**, so any pixel handoff is lossy before the model runs.
- **ComfyUI 0.31.0 AND 0.32.0 cannot combine keyframes and refs** — the refs branch overwrites
  `cond_video_latents` instead of extending it (model_base.py:2168-2172) while `PackedLayout`
  counts both, giving a shape mismatch at model.py:580. Needs a node-side patch.
- **Reference videos need >= 5 frames** and are silently trimmed until `n % 17 == 5`
  (nodes_minimax_h3.py:249-253). Asking for 20 gets you 5, with no error.
- **Text is not the phantom-object risk it was recorded as.** The failure mode is *ungrounded
  or duplicated* demand across tiles; a whole-canvas chunk is one tile, so neither applies.
- Scene cuts in the source are at frames **37 and 106**; a chunk boundary near one measures
  ~3x worse, so test windows must be cut-free.

---

## 7. Not tested / open

**Detail recovery.** The -9.2% is the only remaining deficit and three attempts failed to
escape it (more frozen frames, global context, sigma-scheduled release). Evidence says the
clean anchor buys motion by steering the trajectory *early*, and that same steering is what
costs the detail — the two are one mechanism, not two. Untried: a partial//fractional anchor
weight rather than a binary mask, or a lower `denoise` paired with the frozen head.

**Full-clip end-to-end run.** Everything here is ONE boundary in isolation. Drift across
multiple handoffs over 5 chunks is unmeasured and is the most likely production surprise.

**Both-ends freeze (two-pass).** Render once, then re-render each chunk with head frozen from
chunk N-1 and tail from chunk N+1, restoring the future context bidirectional attention wants.
Owner assessment: not worth it, because a source cut inside the tail region means the frozen
tail shows nothing relevant to the frames before it.

**Productionizing.** `hold_anchor`, the clean-label patch and the noise offset are harness
monkeypatches. `video.py` would need a real overlap/freeze/noise-offset path.
`scale_latent_inpaint` is arguably a ComfyUI bug report — H3 is the only recent video model
without the anchor-holding override.

**4K.** Whole-canvas at 3840x2160 is ~138,720 rows for 56 frames (~29 GiB) and cannot fit, so
spatial tiling returns — and the spatial seam is still unsolved. The untested spatial lever is
per-row timesteps for the anchor ring, which needs the ring expressed as row ranges; it is not
contiguous, so it needs a different approach than the temporal head used.

**Model choice.** Owner's read: a different model may suit moving content better. The tile
method here is seamless on static and close-up shots and unusable on pans and action, and
that split is a property of the content, not of the settings.
