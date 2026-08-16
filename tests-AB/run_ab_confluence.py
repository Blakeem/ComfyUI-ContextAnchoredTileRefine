"""Tile confluence: two-pass draft-then-detail refine, owner's design 2026-08-14.

PRE-PORT HARNESS, PINNED TO COMMIT 0dfd1e4. Not runnable against the current package: it
passes `anchor_type=` (:944/:1016/:1061/:1147/:1231), the kwarg the 1.6.0 sync port removed
along with the raster VL branch. Kept verbatim as the frozen record of this campaign
(tests-AB is a record, not a maintained suite).

The seams are treated as tiles first ("tiles within tiles"): a structural DRAFT of the
whole canvas is built at the first CUT_FRACTION of the steps, with every seam resolved by
a dedicated strip diffusion and feathered at draft softness, then the stock production
pipeline re-noises the draft at the cut sigma and runs the remaining steps as normal.
No hard seam ever survives into pass 2. Cores receive CUT + rest steps; the seam band is
DRAFTED TWICE by design — once inside a butted tile, then re-drafted from sigma[0] by its
strip — before the shared tail, so it carries one extra draft and one extra VAE round
trip. That is the price of a seam resolved by ONE diffusion both neighbors inherit.

STATUS 2026-08-15: PAUSED, superseded for testing by the synchronized single-pass
method (tests-AB/run_ab_sync.py, arm 20). Every remaining confluence defect traces to
the one structural act the sync method deletes — the COMMIT at the cut, where a region
meets neighbour content from a DIFFERENT denoising stage.
  Open defects at pause (arm 17, owner-judged + probed):
  * white speckles: 225 px (probe: luma > 200 & local-median delta > 55, bottom-left
    interior) vs 18 for a full-schedule single-pass arm-1 run. Generator 1, dominant
    (~150): pass 2 reacting to the committed posterior-MEAN draft, which survives as
    the 256px LEADING rings around every pass-2 core (discriminant: arm 13 = 101 vs
    arm 14 = 1, only the draft type differed). Generator 2 (~78): stage 1's arm-1
    whole-image-caption surface x the window length (0 at 7 steps, 78 at 14).
  * colour drift (owner probe: saturation ~+5, lightness ~-5, spatially varying, skin
    too red): the same commit family — chroma frozen at the cut sigma plus the bright
    mean-draft rings steering every pass-2 tile. Tone: final +3.62/255 vs +0.67 for
    single-pass arms.
  * strip INVENTIONS are closed (the invention law's three leak sites, arm 17 sealed
    the last), but the strips' caption-free vision rows still moved eye colour and
    floated petals — band captions were the queued fix.
  Queued arms if the sync method fails, in order:
  18-confluence-raw-rings   pass 2 runs over the RAW canvas; the draft exits the
                            pipeline entirely (rings = raw or refined neighbours, the
                            stock single-pass regime). Kills generator 1 and is the
                            direct colour suspect. ~14 min, draft cache reused.
  19-stage1-arm2-surface    stage 1 swaps the arm-1 surface for arm 2's text-only
                            captions (its seam role is obsolete — the strips + the
                            latent blend hold the seam now). Kills generator 2.
  then: band captions for the strips; shared canvas-shaped SDE noise for the strips
  (ramp exactness); cut 0.375-0.40 (seam depth vs residue leak).

    12-tile-confluence    the original render (2026-08-14): the strip and pass-2 vision
                          encodes read the DRAFT. Zero seams, but native-res forensics
                          traced invented content (butterflies/blossoms) to exactly that:
                          the strip's vision rows described the tile-0 sparkle smear
                          beside its band and its near-full-noise re-draft crystallized
                          butterflies; pass 2's draft-read vision rows then demanded and
                          multiplied them. The caption rows were raw-grounded in BOTH
                          arms (text-only captions are image-blind, generated from RAW),
                          so the VISION rows are isolated as the culprit. Kept for
                          lineage; the default sweep never re-renders it.
    13-confluence-raw-vl  the owner's fix (2026-08-15): "we never want to anchor to
                          something that isn't complete." EVERY vision encode reads the
                          RAW canvas — the strips' and pass 2's conditioning describe the
                          true scene while the sampled pixels still come from the draft,
                          which is diffused together in pass 2. Stage 1 already read RAW.
                          JUDGED: invention gone, zero seams — but tonally OFF (bright /
                          bokeh / soft skin / blocky moon). Measured (Rec.601 luma vs
                          the raw float canvas at 33.02): single-pass arms +0.6, the
                          stage-1 canvas +2.53 AT the collapse, final +5.08. The
                          one-jump collapse commits the model's posterior MEAN at sigma
                          0.65 — an average, not an image — and pass 2 elaborates the
                          drifted average.
    14-confluence-sample-draft
                          arm 13 with ONE change (owner: adjust one thing at a time):
                          the collapse becomes a SAMPLE-COLLAPSE — 3 intermediate rungs
                          picked from the schedule's own tail, then 0, instead of the
                          single jump — so the draft handed to pass 2 is a fast COMPLETE
                          image rather than the mean. The cut, grounding, strips, blends
                          and pass 2 are byte-identical to 13. The run prints the ladder.
                          JUDGED (2026-08-15): tone UNCHANGED (stage-1 +2.77 vs 13's
                          +2.53, final +4.71 vs +5.08) — the mean-vs-sample hypothesis
                          is falsified for tone — and the coarse 5-11x-spaced rungs made
                          the horizontal strip paint watermark-style LETTERS across its
                          band (Krea 2's typography prior resolving integrator smears;
                          the MIT interpolation paper reports the same spurious-text
                          failure on blended latents). Skin texture DID improve.
    15-confluence-latent-handoff
                          arm 13 with ONE structural change (owner's original instinct,
                          2026-08-15): what crosses the cut is the TRUE noisy trajectory
                          state x at the cut sigma, held in latent space, instead of the
                          collapsed x0 estimate. Mechanism: a flow state is
                          x = (1-sigma)*signal + sigma*noise, and the signal component
                          at the cut is the FINAL image the trajectory is heading to
                          (sharp, dark-toned) — the collapse swapped in the sigma-0.65
                          posterior estimate (bright, defocused) and pass 2 faithfully
                          preserved it: arm 13's pass 2 already re-noised with the SAME
                          seed-42 canvas noise field, so the epsilon-conflict half of
                          the theory was falsified and the collapse itself is the
                          remaining cause. Cores are captured at the cut (an observer on
                          sample_latent's callback — the collapse step fires it with x
                          still the pre-collapse state, zero extra model calls), strips
                          likewise, the seam ramp is blended by per-token SLERP in
                          model latent space (Aligning Latent Geometry, arXiv
                          2605.15193: latent tokens live on thin norm shells and the
                          decoder reads direction, not radius — a lerp dips off-shell
                          into inputs the model rarely saw), and pass 2's tiles resume
                          from the blend with zero added noise (comfy's truncate/resume
                          identity: inverse_noise_scaling divides by (1-sigma), CONST
                          noise_scaling with zero noise multiplies it back — exact).
                          The PIXEL draft is still rendered byte-identically to arm 13:
                          it is the live-canvas ring reference and the forensic view,
                          no longer what pass 2 consumes.
                          JUDGED (2026-08-15): pass-2 tone amplification ELIMINATED
                          (+0.09 added vs 13's +1.80; final +3.37) and brightness is
                          acceptable ("mostly her face"), but a SEAM RETURNED in the
                          final at the fog: the draft is seamless, and pass 2's tile
                          split re-opened it — the 25% draft was too shallow for the
                          fog to survive the split.
    16-confluence-latent-cut50
                          arm 15 with ONE change (owner: "it just needs more time to
                          cook"): CUT_FRACTION 0.25 -> 0.50. Pass 1 drafts to HALF the
                          schedule before the latent handoff, so the strip-resolved
                          seam state is committed twice as deep and pass 2 has half as
                          many steps to re-split it. Handoff, grounding, strips,
                          blends, conditioning all byte-identical to 15.
                          JUDGED (2026-08-15): seams BETTER — the cut depth works —
                          but the horizontal strip invented a SPIDERWEB in her hair
                          (proven at (1660,1550): absent in stage-1 AND in arm 15's
                          draft, fully formed in 16's draft, dissolved by pass 2 into
                          filament arcs), and colour is a bit saturated. Third
                          instance of one law: the strip invents whenever any of its
                          inputs carries structured residue of the draft process —
                          arm 12 leaked via CONDITIONING (vision rows read the smear
                          -> butterflies), arm 14 via the INTEGRATOR (big-h smears ->
                          letters), 16 via the PIXEL SUBSTRATE: the strip re-noises
                          the tile draft to sigma 0.76, and at cut 0.50 that draft
                          holds committed bright hair strands whose 0.24x residue the
                          strip crystallizes into a scene-plausible web. Harmless at
                          cut 0.25 only because the sigma-0.65 draft had nothing
                          sharp to leak. Saturation = the same commit-freeze family
                          (more chroma frozen at 0.51; pass-2 tone add +0.09 -> +0.64).
    17-confluence-strips-raw
                          arm 16 with ONE change (the owner's rule completed at the
                          last remaining site): the strips DRAFT FROM THE RAW CANVAS
                          instead of the tile draft. The strip becomes a first-class
                          drafting trajectory exactly like the tiles — raw source,
                          same sigma ladder, mask = its band, frozen ring = raw
                          context — so NO input carries committed draft detail to
                          crystallize, at any cut depth. The tile draft and the strip
                          draft reconcile where they always did: the confluence ramp
                          (pixels for the ring reference, slerp for the latent
                          handoff). The strips_vision_raw patch becomes a redundant
                          identity (encode source already raw) and stays for its
                          shape assert.

PASS 1 (structure, sigmas[0:mid] plus the collapse — one jump, or arm 14's
        3-rung ladder)                                       RAW -> DRAFT
  stage 1  butted tiles: refine_image at context_overlap=0 — cores butt exactly on the
           seam lines, the context_anchor ring is frozen context reaching into the
           neighbors — under the SHIPPED whole-image caption surface (arm 1: the caption
           rides inside the whole-canvas vision encode). Hard butt seams, draft-level
           only. The stage-1-only canvas is saved as *-stage1.png so invention stays
           attributable (tile output vs strip output) on sight.
  stage 2  seam strips: one refine_image MASK call per internal seam line, on the
           ARM_STRIP_SOURCE canvas (the tile draft for arms 12-16; RAW since 17),
           mask = within STRIP_HALF_BAND px of the line. The mask path crops to
           band+anchor (768x3072 vertical, 2304x768 horizontal); the tile caps are
           overridden to the CANVAS dims so each strip samples as ONE tile — the default
           caps would split it into segments whose junction lands exactly ON the crossing
           seam line (review finding 2), and a single strip tile is SMALLER than the
           largest pass-2 tile anyway. Conditioning is PURE VISION ROWS
           (VLM_METHOD_VISION, owner's call: a caption would narrate tile-edge content).
           The frozen Lead-ring context beside the band comes from the same
           ARM_STRIP_SOURCE canvas: the tile drafts for arms 12-16, raw since 17.
  stage 3  confluence blend, harness-side: DC-match the strip to the tile drafts over the
           blend ramp (band_median's pattern), then smoothstep-feather the strip into the
           draft — weight 1 within STRIP_BLEND_PLATEAU of the line, 0 at STRIP_HALF_BAND.

PASS 2 (detail, sigmas[mid:])                                DRAFT -> OUTPUT
  the stock production pipeline, untouched: the tail schedule's first sigma re-noises the
  draft at the cut (flow noise_scaling: sigma*noise + (1-sigma)*draft, shared canvas
  noise), normal overlap-32 geometry with DC + min-error cut + directional feather, and
  arm 2's conditioning (caption encoded TEXT-ONLY, cat'd after sliced vision rows) —
  round 1's content winner. The min-error cut runs HERE only (owner's spec: "we don't
  use the lowest-error path until the very end").

The cut handoff, per ARM_HANDOFF:
  "pixel" (arms 12-14): partially denoised latents are never blended or stored. Each
  pass-1 sigma tensor ends with an appended 0.0: exp_heun_2_x0_sde (sample_seeds_2,
  comfy/k_diffusion/sampling.py) special-cases sigma_next == 0 to `x = denoised` — one
  model call, no SDE noise, no integrator body — so the stock decode-and-paste path
  composites clean structural drafts, in pixels, with no capture machinery. The appended
  zero is load-bearing twice: guider.sample also runs inverse_noise_scaling(sigmas[-1]),
  which for a CONST model divides by (1 - sigma) — a truncated schedule ending at the
  cut sigma would brighten the draft 2-3x. JUDGED COST (arms 13/14): committing the x0
  estimate freezes the sigma-0.65 exposure/defocus; the finals land +4.7 to +5.1 luma
  over raw vs +0.6 for single-pass arms.
  "latent" (arm 15): the SAME schedules run (the collapse still produces the pixel
  draft), but the trajectory state x at the cut is ALSO captured from the collapse
  step's callback — sample_seeds_2 fires callback({'x': x, ...}) with x still the
  pre-collapse state at the cut sigma, in MODEL latent space, before
  inverse_noise_scaling and process_latent_out — blended across the seam ramps by
  per-token slerp, and resumed by pass 2 with zero added noise. Noise-field coherence:
  stage-1 tiles slice ONE seed-42 canvas draw, so the butted cores share an epsilon
  field (the MIT interpolation paper's shared-noise requirement); the STRIPS draw at
  their sub-canvas shape, so their field differs positionally from the tiles' — that
  decorrelation, plus the SDE sampler's per-step draws, lives only inside the 4-cell
  blend ramp, which is why the ramp is slerped (norm-preserving), not lerped.

Evidence this configuration leans on (memory: caption-encode-context-ab):
  arm 9  arm-1 rows early + arm-2 late, everywhere: NO moon — the short arm-1 window in
         pass 1 is moon-safe — but seam WORSE: the time split alone lacks a seam
         mechanism. The strips are that mechanism.
  arm 8/10  the masked dual positive holds the seam at 1.33x/1.13x — the cost this
         design undercuts by giving each stage ONE positive (2 forwards/step everywhere;
         the only extra work is the strips, ~+15%).
  arm 12  zero seams at 1.16x, both strip DC medians exactly 0 — the confluence
         machinery works; the invention was a conditioning-grounding fault, fixed in 13.

Scene: the 00676 woman portrait, 768x1024 base -> 3x -> 2304x3072, 2x2 grid, seed 42,
denoise 0.50, 28 steps (mid = 7 for arms 12-15: pass 1 = 7 steps + collapse, pass 2 =
21 steps; mid = 14 for arm 16: 14 + collapse / 14), cfg 3.5,
exp_heun_2_x0_sde / sgm_uniform, anchor 256, Lead ring, canonical byte-identical captions.

    <venv-python> tests-AB/run_ab_confluence.py                     # pending arms (17)
    <venv-python> tests-AB/run_ab_confluence.py --only 17-confluence-strips-raw

Outputs (output\\AB-Test-Images\\), per arm:
    AB_00676-d050__{arm}.png            the render (judge this)
    AB_00676-d050__{arm}-draft.png      the pass-1 DRAFT canvas (inspectable)
    AB_00676-d050__{arm}-stage1.png     stage 1 only, before the strips (forensics)
"""
import argparse
import contextlib
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_env
import ab_models
import run_ab_krea2 as krea2_run  # sets COMFYUI_ROOT at import, before bootstrap() runs
import run_ab_split as split_run
from run_ab_krea2 import (
    BASE_PNG,
    CLIP_NAME,
    CLIP_TYPE,
    NEGATIVE_PROMPT,
    UNET_NAME,
    UPSCALE_MODEL_NAME,
    VAE_NAME,
)

OUTPUT_DIR = split_run.OUTPUT_DIR
CACHE_DIR = split_run.CACHE_DIR
RUN_TAG = split_run.RUN_TAG

REFINE = split_run.REFINE          # the d0.50 settings every split arm shares
_file_sha = krea2_run._file_sha

ARMS = {
    "12-tile-confluence": "confluence with DRAFT-grounded vision encodes (round-5 "
                          "render; strip invented butterflies — kept for lineage)",
    "13-confluence-raw-vl": "confluence with ALL vision encodes reading the RAW canvas "
                            "(owner's rule: never anchor to something incomplete; "
                            "judged: invention gone, tone drifted — mean-collapse)",
    "14-confluence-sample-draft": "arm 13 with the collapse replaced by 3 real "
                                  "completion steps (sample draft, not the posterior "
                                  "mean) — the ONLY change vs 13 (judged: tone "
                                  "unchanged, ladder painted letters — dead branch)",
    "15-confluence-latent-handoff": "arm 13 with the cut crossed in LATENT space: the "
                                    "true noisy trajectory state at the cut sigma is "
                                    "captured, slerp-blended across the seam ramps, and "
                                    "resumed by pass 2 with zero added noise — the ONLY "
                                    "change vs 13 (judged: pass-2 tone amplification "
                                    "gone, but pass 2 re-split the fog seam the draft "
                                    "had resolved)",
    "16-confluence-latent-cut50": "arm 15 with CUT_FRACTION 0.25 -> 0.50 — the draft "
                                  "cooks to half the schedule before the latent "
                                  "handoff, so pass 2 has half the steps to re-split "
                                  "the seam — the ONLY change vs 15 (judged: seams "
                                  "better, but the strip crystallized a spiderweb "
                                  "from the sharper draft's re-noised residue)",
    "17-confluence-strips-raw": "arm 16 with the strips drafting from the RAW canvas "
                                "instead of the tile draft — no strip input carries "
                                "committed draft detail to crystallize — the ONLY "
                                "change vs 16",
}

# arm -> what the strip and pass-2 vision encodes READ. "draft" = the canvas being
# refined (arm 12's fault); "raw" = the untouched upscaled canvas. Stage 1 reads RAW in
# both (its input canvas IS raw). This selects the ENCODE source only; what the strips
# SAMPLE from (their pixel substrate + frozen ring) is ARM_STRIP_SOURCE — draft for
# arms 12-16, raw since 17 (the substrate leak arm 16 exposed).
# One string per arm is deliberate while both switch sites move together (arm 13 changes
# strips AND pass 2 jointly; the forensics justified it: strip = origin, pass 2 =
# amplifier). An isolation arm (strip-raw/pass2-draft or the inverse) needs this value
# to become a (strip, pass2) pair read at the two sites.
ARM_GROUNDING = {
    "12-tile-confluence": "draft",
    "13-confluence-raw-vl": "raw",
    "14-confluence-sample-draft": "raw",
    "15-confluence-latent-handoff": "raw",
    "16-confluence-latent-cut50": "raw",
    "17-confluence-strips-raw": "raw",
}

# arm -> number of COMPLETION steps in the pass-1 collapse. 0 = the single jump to
# sigma 0, which returns the model's x0 estimate — the posterior MEAN at the cut sigma.
# Arm 13's judged failure (bright/daytime tone, generic bokeh, soft skin, blocky moon)
# is the signature of committing that mean: measured (Rec.601 luma, review round 3)
# +2.53 at the stage-1 collapse, +3.28 after the strips, +5.08 by the final — while
# normal single-pass arms sit at +0.6 over the raw canvas. N > 0 appends N real steps
# taken from the schedule's own tail (index-linspace, so the ladder follows the model's
# shift) before the final 0, so the draft is a fast complete SAMPLE instead of an
# average. Costs 2N extra model calls per region (the sampler is 2nd order: every
# non-terminal step is two forwards; only the sigma-0 step is one).
ARM_COMPLETION = {
    "12-tile-confluence": 0,
    "13-confluence-raw-vl": 0,
    "14-confluence-sample-draft": 3,
    "15-confluence-latent-handoff": 0,
    "16-confluence-latent-cut50": 0,
    "17-confluence-strips-raw": 0,
}

# arm -> what representation of pass 1's result crosses the cut to pass 2.
# "pixel": the collapsed x0 draft is decoded, blended in pixels, re-encoded and
# RE-NOISED at the cut sigma — pass 2 treats the committed draft as the signal to
# preserve (arms 13/14's judged tonal freeze: the sigma-0.65 posterior's bright
# exposure and generic defocus survive into the final).
# "latent": the true noisy trajectory state at the cut sigma is captured per region
# (LatentCapture), slerp-blended across the seam ramps in model latent space, and
# substituted into pass 2's tiles with ZERO added noise — pass 2 continues the same
# trajectory, whose late steps still hold the darkening/sharpening the schedule had
# ahead of it. The pixel draft is still rendered (ring reference + forensics).
ARM_HANDOFF = {
    "12-tile-confluence": "pixel",
    "13-confluence-raw-vl": "pixel",
    "14-confluence-sample-draft": "pixel",
    "15-confluence-latent-handoff": "latent",
    "16-confluence-latent-cut50": "latent",
    "17-confluence-strips-raw": "latent",
}

# arm -> what the STRIPS sample from (their img2img substrate AND frozen ring context).
# "draft" (12-16): the strip re-noises the stage-1 tile draft band to sigma[0] and
# re-drafts it — at cut 0.50 the draft's committed bright hair strands survive the
# re-noising as 0.24x filament residue that the strip crystallized into a spiderweb
# (owner-judged 16; third instance of the invention law, this time via the substrate).
# "raw" (17): the strip drafts the RAW band like the tiles draft their raw cores — a
# first-class independent drafting trajectory; nothing committed anywhere in its
# inputs. The tile and strip drafts reconcile in the confluence ramp (pixels + slerp).
ARM_STRIP_SOURCE = {
    "12-tile-confluence": "draft",
    "13-confluence-raw-vl": "draft",
    "14-confluence-sample-draft": "draft",
    "15-confluence-latent-handoff": "draft",
    "16-confluence-latent-cut50": "draft",
    "17-confluence-strips-raw": "raw",
}

# Arms already rendered AND judged by the owner: the PNG on disk is the lineage
# reference, so the default sweep never re-renders them (a bare --force would otherwise
# run two full confluence renders in one process and overwrite the judged artifact).
# Re-rendering one deliberately requires naming it: --only <arm> --force.
LINEAGE_ARMS = {"12-tile-confluence", "13-confluence-raw-vl", "14-confluence-sample-draft",
                "15-confluence-latent-handoff", "16-confluence-latent-cut50"}

# Where the schedule splits: pass 1 covers the first cut fraction of the steps (0.25 =
# arm 10's judged-good structure window), pass 2 the rest. The cut index is computed
# from the run's own sigma tensor, so the two passes share the exact boundary sigma.
# Per arm since 16 (owner, 2026-08-15, after judging 15): the fog seam was resolved in
# the draft but pass 2's tile split re-opened it — at 0.25 the seam state is committed
# too shallow to survive; 0.50 gives it "more time to cook" and pass 2 half the steps
# to undo it.
CUT_FRACTION = 0.25
ARM_CUT_FRACTION = {
    "12-tile-confluence": CUT_FRACTION,
    "13-confluence-raw-vl": CUT_FRACTION,
    "14-confluence-sample-draft": CUT_FRACTION,
    "15-confluence-latent-handoff": CUT_FRACTION,
    "16-confluence-latent-cut50": 0.50,
    "17-confluence-strips-raw": 0.50,
}

# Strip geometry, PIXELS, symmetric around each internal seam line. The strip DIFFUSES
# everything within STRIP_HALF_BAND; the confluence blend keeps it at full weight within
# STRIP_BLEND_PLATEAU and smoothsteps to 0 at STRIP_HALF_BAND, so the outer
# (HALF_BAND - PLATEAU) ring is the cross-dissolve into the tile drafts. The mask path
# adds context_anchor of frozen draft context around the band on its own.
# The ramp is 32px — the engine's own band width (the owner's "normal blend") — and sits
# at the strip's OUTER edge, where the strip is most tightly held to continue the frozen
# draft context beside it (review finding 3: a wider ramp cross-dissolves two independent
# draft renderings where they agree least).
STRIP_HALF_BAND = 128
STRIP_BLEND_PLATEAU = 96


def output_path(arm, suffix=""):
    return OUTPUT_DIR / f"AB_{RUN_TAG}__{arm}{suffix}.png"


def mean_luma(t):
    """Mean Rec.601 luma of an IMAGE tensor, x255 — the unit the arm-13 tonal diagnosis
    was measured in. Mean-RGB diverges ~0.4-0.5 from luma on this crimson-heavy scene
    (review round 3, D1), so the readout must be luma to be comparable to the
    baselines: raw float canvas 33.02; normal single-pass arms +0.6 above raw;
    arm-13 stage-1 +2.53, draft +3.28, final +5.08."""
    w = torch.tensor((0.299, 0.587, 0.114), dtype=t.dtype, device=t.device)
    return float((t[..., :3] * w).sum(-1).mean()) * 255.0


# ------------------------------------------------------------------ schedule split

def split_sigmas(sigmas, completion_steps, cut_fraction):
    # Both knobs are REQUIRED (review): a defaulted cut_fraction would let a future
    # caller silently draft a 0.50 arm to 25%.
    """(head, tail, mid): head = the first `mid` steps plus the pass-1 COLLAPSE, tail =
    the remaining schedule starting AT the cut sigma — exactly the level pass 2
    re-noises the draft to. Both passes meet at sigmas[mid] with no gap and no overlap.

    completion_steps = 0: the collapse is one appended 0.0 — a final step to sigma 0
    returns the sampler's clean x0 estimate (one extra model call), which is the
    posterior MEAN at the cut sigma (arms 12/13; judged tonally drifted).
    completion_steps = N > 0: the collapse is N real steps whose sigmas are picked from
    the REMAINING schedule by index-linspace (so the ladder follows the model's own
    shift), then 0 — the draft becomes a fast complete SAMPLE (arm 14)."""
    steps_n = int(sigmas.shape[-1]) - 1
    if steps_n < 2:
        raise RuntimeError(f"confluence needs >= 2 steps; got {steps_n}")
    if completion_steps < 0:
        raise RuntimeError(f"completion_steps must be >= 0, got {completion_steps}")
    mid = min(steps_n - 1, max(1, round(steps_n * cut_fraction)))
    if completion_steps > 0:
        if completion_steps > steps_n - mid - 1:
            raise RuntimeError(
                f"completion_steps {completion_steps} does not fit between the cut "
                f"(step {mid}) and the schedule end (step {steps_n})")
        # linspace over the remaining INDICES, endpoints mid..steps_n; drop the first
        # (== the cut itself). The last index is steps_n, whose sigma is 0.0 by
        # schedule construction, so the head still ends at exactly 0. torch.round is
        # half-to-EVEN (17.5 -> 18 here), so the positions are not a pure nearest
        # mapping on ties; monotonicity and distinctness hold for any tie under the
        # completion_steps guard above (review D8).
        positions = torch.linspace(mid, steps_n, completion_steps + 2)[1:].round().long()
        collapse = sigmas[positions]
        if float(collapse[-1]) != 0.0:
            raise RuntimeError("schedule does not end at sigma 0 — collapse ladder invalid")
    else:
        collapse = sigmas.new_zeros(1)
    head = torch.cat([sigmas[:mid + 1], collapse])
    if not torch.all(head[1:] < head[:-1]):
        raise RuntimeError(f"head sigmas are not strictly decreasing: {[float(v) for v in head]}")
    tail = sigmas[mid:]
    return head, tail, mid


# ------------------------------------------------------------------ seam geometry

def internal_seam_lines(tiles, canvas_w, canvas_h):
    """The core boundaries no tile owns alone: x of every vertical seam line, y of every
    horizontal one, canvas borders excluded. For the 2x2 grid this is one of each."""
    xs, ys = set(), set()
    for tile in tiles:
        for x in (tile.core.x0, tile.core.x1):
            if 0 < x < canvas_w:
                xs.add(x)
        for y in (tile.core.y0, tile.core.y1):
            if 0 < y < canvas_h:
                ys.add(y)
    return sorted(xs), sorted(ys)


def _seam_distance(canvas_h, canvas_w, line, axis):
    """[H,W] float32 distance of each pixel CENTER from the seam line (a boundary
    coordinate, so +0.5 keeps the field symmetric across it)."""
    if axis == "x":
        d = (torch.arange(canvas_w, dtype=torch.float32) + 0.5 - line).abs()
        return d[None, :].expand(canvas_h, canvas_w)
    d = (torch.arange(canvas_h, dtype=torch.float32) + 0.5 - line).abs()
    return d[:, None].expand(canvas_h, canvas_w)


def strip_mask(canvas_h, canvas_w, line, axis):
    """[1,H,W] binary float mask of the strip's DIFFUSED region for refine_image."""
    dist = _seam_distance(canvas_h, canvas_w, line, axis)
    return (dist < STRIP_HALF_BAND).to(torch.float32)[None]


def blend_weight(canvas_h, canvas_w, line, axis):
    """[1,H,W,1] confluence weight: 1 within STRIP_BLEND_PLATEAU of the line,
    smoothstep down to exactly 0 at STRIP_HALF_BAND (the strip's diffusion edge and its
    1px AA composite line land where the weight is already 0)."""
    dist = _seam_distance(canvas_h, canvas_w, line, axis)
    u = ((dist - STRIP_BLEND_PLATEAU) / (STRIP_HALF_BAND - STRIP_BLEND_PLATEAU)).clamp(0.0, 1.0)
    return (1.0 - u * u * (3.0 - 2.0 * u))[None, :, :, None]


def confluence_blend(draft, strip_out, line, axis):
    """Stage 3 for one strip: DC-match the strip's draft to the tile drafts over the
    blend ramp (sampling.seam_dc_offset's pattern: per-channel median where both sides
    rendered the same content, clamped, gamut-guarded), then feather it in. Outside the
    weight's support the draft is returned byte-identical.
    Known limit (review finding 11): stage 1 ran at overlap 0, so its butted tiles carry
    their full per-tile DC offsets, and this SINGLE pooled median centres the strip
    BETWEEN the two sides rather than matching either — the DC step is smoothed over the
    two ramps, not removed. Pass 2's own always-on DC match handles the residual.
    Readout caveat (review D11, verified on arm 12's cached draft): dark scenes put a
    9-31% atom of EXACTLY-0.0 pixels (black railed in both drafts) into the ramp;
    torch.median returns an actual element, so the atom straddles the median rank and
    the printed dc deadbands to +0.000000 even though the blend demonstrably removed the
    butted DC step. dc=0 here means "estimator deadbanded", not "blend inactive"."""
    from context_anchored_tile_refine import sampling

    weight = blend_weight(draft.shape[1], draft.shape[2], line, axis).to(draft.device, draft.dtype)
    ramp = (weight[0, :, :, 0] > 0.0) & (weight[0, :, :, 0] < 1.0)

    adjusted = strip_out
    diff = (draft[0] - strip_out[0])[ramp]              # [N,C], the measuring band
    if diff.shape[0] >= sampling.DC_MATCH_MIN_SAMPLES:
        offset = diff.median(dim=0).values.clamp(-sampling.DC_MATCH_CLAMP, sampling.DC_MATCH_CLAMP)
        corrected = strip_out + offset
        in_gamut = (strip_out >= 0.0) & (strip_out <= 1.0)
        adjusted = torch.where(in_gamut, corrected.clamp(0.0, 1.0), corrected)
        print(f"[blend]   {axis}={line}: dc=[{','.join(f'{v:+.6f}' for v in offset.tolist())}]"
              f"  ramp {diff.shape[0]} px")
    else:
        print(f"[blend]   {axis}={line}: dc skipped ({diff.shape[0]} ramp px < "
              f"{sampling.DC_MATCH_MIN_SAMPLES})")
    return draft * (1.0 - weight) + adjusted * weight


# ------------------------------------------------------------------ pass-1 patches

@contextlib.contextmanager
def no_min_error_cut():
    """Owner's spec: no lowest-error path until the very end. Straighten every pass-1
    feather by patching the single swap point _refine_tiles resolves at call time —
    (None, None) is seam_displacements' own no-neighbor return, so feather_alpha takes
    its straight-band shape. With overlap-0 tiles and single-tile strips nothing in
    pass 1 routes anyway; this pins the spec against any future geometry change."""
    from context_anchored_tile_refine import sampling

    def straight(tile, sub, region):
        return None, None

    saved = sampling.seam_displacements
    sampling.seam_displacements = straight
    try:
        yield
    finally:
        sampling.seam_displacements = saved


@contextlib.contextmanager
def inject_captions_by_core(tile_captions, expected_cores):
    """Stage 1's caption injection. The butted layout's CROPS differ from the canonical
    overlap-32 crops (no 32px extension), so split_run.inject_captions' rect assert
    cannot be reused — but the CORES are identical (solve_axis picks the same grid), so
    the canonical byte-identical captions are keyed by core instead. The text was
    generated from crops 32px wider; same content, and identical text everywhere is the
    property that matters."""
    from context_anchored_tile_refine import captions

    def hook(clip, source, hook_tiles, instruction, max_length, batch_size=1, batch_index=0):
        cores = [[t.core.x0, t.core.y0, t.core.x1, t.core.y1] for t in hook_tiles]
        if cores != expected_cores:
            raise RuntimeError("stage-1 tile cores diverged from the canonical layout")
        return tile_captions

    original = captions.generate_tile_captions
    captions.generate_tile_captions = hook
    try:
        yield
    finally:
        captions.generate_tile_captions = original


@contextlib.contextmanager
def strips_vision_raw(raw_canvas):
    """Owner's rule (2026-08-15): never anchor to something that isn't complete. The
    strip calls run ON the draft, so vl.build_global_slices would vision-encode
    partially-diffused pixels — the round-5 forensics traced the invented butterflies to
    exactly that (the strip's rows described the tile-0 sparkle smear beside its band).
    This swaps the ENCODE SOURCE only; what the strip samples and freezes is
    ARM_STRIP_SOURCE's pick (draft for arms 12-16, raw since 17)."""
    from context_anchored_tile_refine import vl

    original = vl.build_global_slices

    def hook(clip, source, tiles, offset_x=0, offset_y=0):
        if tuple(source.shape) != tuple(raw_canvas.shape):
            raise RuntimeError(
                f"raw grounding: strip encode source {tuple(source.shape)} diverged "
                f"from the raw canvas {tuple(raw_canvas.shape)}")
        return original(clip, raw_canvas, tiles, offset_x=offset_x, offset_y=offset_y)

    vl.build_global_slices = hook
    try:
        yield
    finally:
        vl.build_global_slices = original


@contextlib.contextmanager
def pass2_builder(grounding, raw_padded):
    """Pass 2's conditioning surface: arm 2's text-only caption builder, with the vision
    encode reading `raw_padded` when grounding == "raw" (same rule as strips_vision_raw:
    pass 2 refines the draft, but its vision rows must describe the COMPLETE image, not
    the draft's committed ambiguity — arm 12's amplifier loop). grounding == "draft"
    reproduces arm 12's surface exactly (split_run.caption_encode_mode)."""
    from context_anchored_tile_refine import captions

    if grounding == "draft":
        with split_run.caption_encode_mode("text-only"):
            yield
        return

    inner = split_run.make_split_builder("text-only")

    def builder(clip, encode_source, tiles, tile_captions, offset_x=0, offset_y=0):
        if tuple(encode_source.shape) != tuple(raw_padded.shape):
            raise RuntimeError(
                f"raw grounding: pass-2 encode source {tuple(encode_source.shape)} "
                f"diverged from the raw canvas {tuple(raw_padded.shape)}")
        return inner(clip, raw_padded, tiles, tile_captions, offset_x=offset_x, offset_y=offset_y)

    original = captions.build_slice_caption_conds
    captions.build_slice_caption_conds = builder
    print("[render]  pass-2 builder: text-only captions, vision rows grounded in RAW")
    try:
        yield
    finally:
        captions.build_slice_caption_conds = original


# ------------------------------------------------------------------ latent handoff (arm 15)
#
# The cut is crossed by the TRUE trajectory state, not the collapsed x0 estimate.
# Coordinate contract, verified against the installed comfy source (0.32.0):
#   - sample_seeds_2 fires callback({'x': x, 'i': i, ...}) at the TOP of each step,
#     with x the state AT sigmas[i] (pre-update). The final head step is the collapse
#     (sigmas[i] == cut sigma -> 0), so its callback hands over the pre-collapse state
#     at exactly the cut sigma — zero extra model calls, zero behavioral change.
#   - That x is in MODEL latent space: CFGGuider.inner_sample runs process_latent_in
#     on the way in and process_latent_out + inverse_noise_scaling only on the RETURN.
#     Krea 2's latent_format is Wan21 — per-channel affine (z - mean) / std, exactly
#     invertible — so captures are blended in model space and converted ONCE at
#     injection: latent_region = process_latent_out(x / (1 - cut_sigma)). Pass 2 then
#     applies process_latent_in and CONST.noise_scaling(cut_sigma, noise=0, z) =
#     (1 - cut_sigma) * z, reproducing x bit-exactly (the truncate/resume identity).
#   - offset_first_sigma_for_snr only rewrites sigmas[0] >= 1; every cut sigma is
#     below 1 (sigma[0] is 0.7596 — denoise 0.5 under Krea 2's shift 1.15; cut 0.25 ->
#     sigma 0.6548, resume scale 2.897; cut 0.50 -> sigma 0.5132, resume scale 2.054),
#     so pass 2's first model evaluation happens at exactly the cut sigma on the blend.
# KNOWN CONFOUND (review finding 2): pass 2's frozen anchor rings still come from the
# PIXEL draft — 256px of the bright sigma-0.65 posterior as LEADING conditioning around
# every core. A "tone unchanged" verdict is therefore ambiguous (handoff failed, or the
# ring pulled the resumed trajectory back); the paired follow-up is an adjacent-ring or
# ring-free arm before concluding against the handoff. A "tone fixed" verdict is clean.
# Blend geometry (Aligning Latent Geometry, arXiv 2605.15193): latent tokens
# concentrate on thin norm shells and the decoder reads DIRECTION far more than
# radius; a lerp of two weakly-correlated tokens dips toward R/sqrt(2) — off-shell,
# into inputs the model rarely saw (the letter/butterfly generator condition). The
# ramp is therefore slerped per token. Where tile and strip agree (cosine ~1, most of
# the ramp) slerp == lerp; where they disagree — the seam disagreement zone plus the
# SDE-decorrelated per-step noise — slerp stays on-shell.


class LatentCapture:
    """Observer patch on sampling.sample_latent: wraps each call's progress callback and
    records the trajectory state x at the FINAL step — for a pass-1 head schedule that
    is the collapse step, whose callback x is the pre-collapse state at the cut sigma —
    plus the call's binary denoise mask (the diffused region, latent resolution) and the
    cut sigma itself. Tensors go to CPU fp32; the wrapped call is otherwise untouched
    (the original callback still runs, so previews and the progress bar are unchanged)."""

    def __init__(self):
        self.records = []

    @contextlib.contextmanager
    def active(self):
        from context_anchored_tile_refine import sampling

        original = sampling.sample_latent
        records = self.records

        def wrapper(guider, sampler, sigmas, noise_tensor, seed, latent_samples,
                    denoise_mask=None, callback=None, anchor_ring=True):
            steps = int(sigmas.shape[-1]) - 1
            record = {
                "x_cut": None,
                "cut_sigma": float(sigmas[-2]),
                "mask": None if denoise_mask is None
                        else denoise_mask.detach().to("cpu", torch.float32) > 0.5,
            }

            def capture(step, x0, x, total_steps):
                if step == steps - 1:
                    record["x_cut"] = x.detach().to("cpu", torch.float32).clone()
                if callback is not None:
                    callback(step, x0, x, total_steps)

            out = original(guider, sampler, sigmas, noise_tensor, seed, latent_samples,
                           denoise_mask=denoise_mask, callback=capture, anchor_ring=anchor_ring)
            if record["x_cut"] is None:
                raise RuntimeError("latent capture: the final sampler step never fired its callback")
            records.append(record)
            return out

        sampling.sample_latent = wrapper
        try:
            yield self
        finally:
            sampling.sample_latent = original


def mask_2d(mask):
    """The capture's denoise mask as a [h,w] bool grid. Every pass-1 mask is a single
    picture's binary latent gate ([h,w], [1,h,w] or [B=1,...]); reshape fails fast on
    anything with more elements than one grid."""
    return mask.reshape(mask.shape[-2], mask.shape[-1])


def stage1_latent_layout(canvas_w, canvas_h):
    """The stage-1 butted grid, re-solved EXACTLY as _refine_tiles does (same three grid
    calls, overlap 0). The canvas is /8 by construction (asserted in main), so the pad
    step is a no-op and these rects are the rects the captures were sampled at."""
    from context_anchored_tile_refine import grid

    sx = grid.solve_axis(canvas_w, REFINE.max_tile_width, REFINE.context_anchor, 0, axis="width")
    sy = grid.solve_axis(canvas_h, REFINE.max_tile_height, REFINE.context_anchor, 0, axis="height")
    return grid.build_layout(canvas_w, canvas_h, sx, sy, REFINE.context_anchor, 0)


def build_latent_canvas(records, layout, canvas_h, canvas_w):
    """Assemble the stage-1 captures into one canvas-sized model-space latent: each
    tile's CORE cells (its denoise-mask region — the butted cores tile the canvas
    exactly) pasted at its canvas position. Raster order is the tile loop's own order.
    Returns (latent_canvas, cut_sigma)."""
    if len(records) != len(layout.tiles):
        raise RuntimeError(f"stage-1 captured {len(records)} calls, layout has {len(layout.tiles)} tiles")
    first = records[0]["x_cut"]
    h8, w8 = canvas_h // 8, canvas_w // 8
    canvas = torch.zeros((*first.shape[:-2], h8, w8), dtype=torch.float32)
    covered = torch.zeros((h8, w8), dtype=torch.bool)
    cut_sigma = records[0]["cut_sigma"]
    for record, tile in zip(records, layout.tiles, strict=True):
        crop, core = tile.crop_rect, tile.core
        x = record["x_cut"]
        if record["cut_sigma"] != cut_sigma:
            raise RuntimeError("stage-1 captures disagree on the cut sigma")
        if x.shape[-2:] != ((crop.y1 - crop.y0) // 8, (crop.x1 - crop.x0) // 8):
            raise RuntimeError(
                f"capture shape {tuple(x.shape[-2:])} != crop {crop} of the re-solved stage-1 layout")
        cy0, cy1 = (core.y0 - crop.y0) // 8, (core.y1 - crop.y0) // 8
        cx0, cx1 = (core.x0 - crop.x0) // 8, (core.x1 - crop.x0) // 8
        m = mask_2d(record["mask"])
        if not bool(m[cy0:cy1, cx0:cx1].all()) or int(m.sum()) != (cy1 - cy0) * (cx1 - cx0):
            raise RuntimeError("stage-1 denoise mask is not the core indicator — capture misaligned")
        canvas[..., core.y0 // 8:core.y1 // 8, core.x0 // 8:core.x1 // 8] = x[..., cy0:cy1, cx0:cx1]
        covered[core.y0 // 8:core.y1 // 8, core.x0 // 8:core.x1 // 8] = True
    if not bool(covered.all()):
        raise RuntimeError("stage-1 cores do not tile the latent canvas — coverage hole")
    return canvas, cut_sigma


def latent_blend_weight(canvas_h, canvas_w, line, axis):
    """[h8,w8] confluence weight at latent CELL CENTERS ((i+0.5)*8 px) — the same
    smoothstep as blend_weight (1 within STRIP_BLEND_PLATEAU, 0 at STRIP_HALF_BAND),
    evaluated on the /8 grid the latent blend runs on."""
    if axis == "x":
        d = (torch.arange(canvas_w // 8, dtype=torch.float32) * 8.0 + 4.0 - line).abs()
        dist = d[None, :].expand(canvas_h // 8, canvas_w // 8)
    else:
        d = (torch.arange(canvas_h // 8, dtype=torch.float32) * 8.0 + 4.0 - line).abs()
        dist = d[:, None].expand(canvas_h // 8, canvas_w // 8)
    u = ((dist - STRIP_BLEND_PLATEAU) / (STRIP_HALF_BAND - STRIP_BLEND_PLATEAU)).clamp(0.0, 1.0)
    return 1.0 - u * u * (3.0 - 2.0 * u)


def slerp_tokens(a, b, weight):
    """Per-token slerp from a toward b, weight [h,w] in [0,1]. A token is the channel
    vector at one latent cell (dim 1 of [1,C,(1,)h,w]). Degenerate tokens (near-parallel
    or near-zero) fall back to lerp, where the two are equal anyway."""
    w = weight.reshape((1, 1) + (1,) * (a.ndim - 4) + weight.shape).to(a.dtype)
    dot = (a * b).sum(dim=1, keepdim=True)
    norm_a = a.pow(2).sum(dim=1, keepdim=True).sqrt()
    norm_b = b.pow(2).sum(dim=1, keepdim=True).sqrt()
    raw_cos = dot / (norm_a * norm_b).clamp(min=1e-12)
    cos = raw_cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    omega = torch.acos(cos)
    sin_omega = torch.sin(omega)
    lerp = (1.0 - w) * a + w * b
    slerp = (torch.sin((1.0 - w) * omega) * a + torch.sin(w * omega) * b) / sin_omega
    # Degeneracy is tested on the UNclamped cosine (review finding 5: a post-clamp
    # sin_omega test can never fire) — near-(anti)parallel tokens and near-zero norms
    # take the lerp, which the slerp equals there anyway.
    degenerate = (raw_cos.abs() > 1.0 - 1e-6) | (norm_a < 1e-6) | (norm_b < 1e-6)
    return torch.where(degenerate, lerp, slerp)


def blend_strip_latent(latent_canvas, record, mask, line, axis, canvas_h, canvas_w, cut_sigma):
    """Stage 3 in latent space: slerp the strip's captured state into the canvas over
    the confluence ramp. The strip's sub-canvas rect is re-derived with the SAME mask
    bbox helpers refine_image uses, so there is one implementation of that geometry.
    In-place on latent_canvas. No DC step, eyes open (review finding 4): the pixel
    confluence's dc=0 prints are a DEADBANDED estimator (see confluence_blend), not
    evidence of zero offset, and pass 2's DC match canNOT reach this location — the
    ramps sit at 96-128px from the seam, INSIDE the tile cores, while seam_dc_offset
    measures and corrects only the 32px overlap band. The expected residual is the
    VAE crop-framing term (sub-1/255) smoothed over the 32px ramp; if a step shows in
    the -draft forensics, the fix is a per-channel median over the ramp cells here."""
    from context_anchored_tile_refine import sampling

    if record["cut_sigma"] != cut_sigma:
        raise RuntimeError("strip capture disagrees with stage 1 on the cut sigma")
    bbox = sampling._mask_bbox(mask >= 0.5)
    y0, y1, x0, x1 = sampling._expand_snap_clamp(bbox, REFINE.context_anchor, canvas_h, canvas_w)
    x_strip = record["x_cut"]
    if x_strip.shape[-2:] != ((y1 - y0) // 8, (x1 - x0) // 8):
        raise RuntimeError(
            f"strip capture shape {tuple(x_strip.shape[-2:])} != re-derived region "
            f"({(y1 - y0) // 8}, {(x1 - x0) // 8})")
    band = mask_2d(record["mask"])
    weight = latent_blend_weight(canvas_h, canvas_w, line, axis)[y0 // 8:y1 // 8, x0 // 8:x1 // 8]
    # Every ramp cell must be a cell the strip actually diffused: the last in-band cell
    # center sits at |dist| <= 124 < STRIP_HALF_BAND, and the latent band over-covers
    # by max-pool, so support outside the band means the geometry regressed.
    if bool(((weight > 0.0) & ~band).any()):
        raise RuntimeError("confluence ramp extends outside the strip's diffused band")
    sub = latent_canvas[..., y0 // 8:y1 // 8, x0 // 8:x1 // 8]
    latent_canvas[..., y0 // 8:y1 // 8, x0 // 8:x1 // 8] = slerp_tokens(sub, x_strip, weight)


@contextlib.contextmanager
def inject_latent_handoff(latent_canvas, tiles, cut_sigma, model):
    """Pass-2 patch on sampling.sample_latent: each tile RESUMES the pass-1 trajectory
    instead of re-noising the draft. The tile's diffused region (denoise mask == 1,
    core + overlap) of the encoded latent is replaced by the blended model-space state
    converted back to canonical latent space — process_latent_out(x / (1 - cut_sigma)) —
    and the tile's initial noise is zeroed, so comfy's own prep (process_latent_in, then
    CONST.noise_scaling(cut_sigma, 0, z) = (1 - cut_sigma) * z) reproduces x bit-exactly.
    The frozen anchor ring keeps its live-canvas encode as the clean reference, exactly
    as stock — the ring and the substituted region are complementary by the mask."""
    from context_anchored_tile_refine import sampling

    if not 0.0 < cut_sigma < 1.0:
        raise RuntimeError(f"cut sigma {cut_sigma} outside (0,1) — the resume scale 1/(1-sigma) is invalid")
    original = sampling.sample_latent
    base_model = model.model  # comfy BaseModel: process_latent_out inverts process_latent_in
    expected = [(tile.crop_rect, tile.overlap_inner_rect) for tile in tiles]
    state = {"calls": 0}

    def wrapper(guider, sampler, sigmas, noise_tensor, seed, latent_samples,
                denoise_mask=None, callback=None, anchor_ring=True):
        index = state["calls"]
        state["calls"] += 1
        if index >= len(expected):
            raise RuntimeError(f"pass 2 made {index + 1} tile calls, layout has {len(expected)}")
        if abs(float(sigmas[0]) - cut_sigma) > 1e-6:
            raise RuntimeError(
                f"pass-2 tile starts at sigma {float(sigmas[0])}, capture was cut at {cut_sigma}")
        if denoise_mask is None:
            raise RuntimeError("pass-2 tile has no denoise mask — cannot locate the region to resume")
        crop, inner = expected[index]
        x_slice = latent_canvas[..., crop.y0 // 8:crop.y1 // 8, crop.x0 // 8:crop.x1 // 8]
        if x_slice.shape != latent_samples.shape:
            raise RuntimeError(
                f"pass-2 tile latent {tuple(latent_samples.shape)} != captured slice "
                f"{tuple(x_slice.shape)} at crop {crop}")
        region = mask_2d(denoise_mask.detach() > 0.5)
        # Elementwise, not count-only (review finding 7): a same-area mask at the wrong
        # position must fail, so compare against the indicator the pipeline itself builds.
        expected_region = sampling.tile_gradient(crop, inner, scale=8) > 0.5
        if not torch.equal(region.cpu(), expected_region):
            raise RuntimeError(
                f"pass-2 tile {index}: denoise mask is not the core+overlap indicator of "
                f"crop {crop} — layout diverged from the canonical tiles")
        resumed = base_model.process_latent_out(x_slice / (1.0 - cut_sigma))
        resumed = resumed.to(latent_samples.device, latent_samples.dtype)
        gate = region.reshape((1, 1) + (1,) * (latent_samples.ndim - 4) + region.shape)
        gate = gate.to(latent_samples.device)
        new_latent = torch.where(gate, resumed, latent_samples)
        # Zero the noise ONLY inside the resumed region. The tensor is consumed twice:
        # once by the initial noise_scaling (where the gate's zero makes the resume
        # identity exact), and once PER STEP as model_k.noise inside KSamplerX0Inpaint's
        # scale_latent_inpaint, which re-noises the frozen anchor RING from it (review
        # finding 1: a whole-tensor zero silently hands every pass-2 tile a noise-free
        # 0.345-scaled ring — off-manifold conditioning on the exact mechanism under
        # test). Outside the gate the arm-13 canvas-noise slice is kept bit-for-bit.
        new_noise = torch.where(gate, torch.zeros_like(noise_tensor), noise_tensor)
        return original(guider, sampler, sigmas, new_noise, seed, new_latent,
                        denoise_mask=denoise_mask, callback=callback, anchor_ring=anchor_ring)

    sampling.sample_latent = wrapper
    try:
        yield
        if state["calls"] != len(expected):
            raise RuntimeError(
                f"pass 2 resumed {state['calls']} tiles, layout has {len(expected)}")
    finally:
        sampling.sample_latent = original


# ------------------------------------------------------------------ pass 1: the draft

def render_draft(arm, canvas, tiles, clip, model, vae, negative, empty, tile_captions,
                 head, mid, tail, timings):
    """RAW -> DRAFT. Returns (draft, latent_canvas, cut_sigma): the blended pixel draft
    (same dims as `canvas`) plus, for ARM_HANDOFF "latent", the blended model-space
    trajectory state at the cut sigma (None, None otherwise).
    `canvas` is the raw upscaled image — stage 1's input, the vision-encode source of
    the strip calls (grounding "raw"), AND the strips' pixel substrate + frozen ring
    (ARM_STRIP_SOURCE "raw")."""
    import comfy.samplers

    from context_anchored_tile_refine import captions, sampling, upscale

    grounding = ARM_GROUNDING[arm]
    handoff = ARM_HANDOFF[arm]
    sampler = comfy.samplers.sampler_object(REFINE.sampler)
    canvas_h, canvas_w = int(canvas.shape[1]), int(canvas.shape[2])
    expected_cores = [[t.core.x0, t.core.y0, t.core.x1, t.core.y1] for t in tiles]
    latent_canvas = None
    cut_sigma = None

    # ---- stage 1: butted tiles, shipped whole-image caption surface (arm 1)
    started = time.perf_counter()
    guider = upscale.build_guider(model, empty, negative, REFINE.cfg)
    noise = upscale.Noise_RandomNoise(REFINE.seed)
    stage1_capture = LatentCapture() if handoff == "latent" else None
    stage1_observer = stage1_capture.active() if stage1_capture else contextlib.nullcontext()
    # Allocator reset before EVERY sampler call in this file: one process running four
    # 3x-scale refines is structurally the "long multi-config process" the machine notes
    # say dies as a silent native crash (allocator-state accumulation). Cheap; models
    # stay loaded.
    ab_models.clear_cache()
    with inject_captions_by_core(tile_captions, expected_cores), no_min_error_cut(), \
            stage1_observer, ab_models.VramProbe() as probe, torch.inference_mode():
        draft = sampling.refine_image(
            canvas, guider, sampler, head, vae, noise,
            max_tile_width=REFINE.max_tile_width,
            max_tile_height=REFINE.max_tile_height,
            context_anchor=REFINE.context_anchor,
            context_overlap=0,
            vl_clip=clip,
            anchor_type="leading",
            vlm_method=captions.VLM_METHOD_VISION_CAPTIONS,
        )
    timings["stage1-butted-tiles"] = time.perf_counter() - started
    print(f"[draft]   stage 1 (butted tiles) {timings['stage1-butted-tiles']:.1f}s  {probe}")
    if stage1_capture is not None:
        # Captures are made under inference_mode, so the assembly math runs there too.
        with torch.inference_mode():
            latent_canvas, cut_sigma = build_latent_canvas(
                stage1_capture.records, stage1_latent_layout(canvas_w, canvas_h),
                canvas_h, canvas_w)
        print(f"[latent]  {len(stage1_capture.records)} cores captured at sigma "
              f"{cut_sigma:.4f} -> canvas {tuple(latent_canvas.shape)}")
    # The earliest readout of the collapse mechanism, uncontaminated by strips/pass 2
    # (review D3/D4): arm-13 measured +2.53 here; +0.70 of that is the arm-1 caption
    # surface itself (a full-schedule arm-1 run sits at +1.37), so a WORKING
    # sample-collapse should land this near +1.4, not near 0.
    print(f"[tone]    stage-1 luma delta vs raw: {mean_luma(draft) - mean_luma(canvas):+.2f}/255"
          "  (arm-13: +2.53; target ~+1.4)")
    # Forensic save: tile output BEFORE any strip touches the canvas, so invented
    # content is attributable (tile vs strip) on sight instead of by band arithmetic.
    stage1_png = output_path(arm, "-stage1")
    ab_models.save_png(stage1_png, draft.cpu(),
                       build_settings(arm, tile_captions, mid, head, tail, stage="pass1-stage1"))
    print(f"[draft]   -> {stage1_png.name}")

    # ---- stages 2+3: one strip per internal seam line, vertical first, each blended
    # into the draft before the next runs. On strip_source "draft" (arms 12-16) the
    # horizontal strip re-drafts the crossing anchored by the vertical band's draft
    # above and below it; on "raw" (17+) the strips are independent of the tiles AND of
    # each other, and the crossing reconciles only in the two blends (pixel + latent) —
    # where the v plateau meets the h ramp, two independent raw drafts cross-dissolve
    # over 32px (x[1056,1248) x y[1408,1440) and y[1632,1664)); inspect those
    # rectangles in -draft.png when judging.
    x_lines, y_lines = internal_seam_lines(tiles, canvas_w, canvas_h)
    if not x_lines and not y_lines:
        raise RuntimeError("no internal seam lines — a 1x1 grid has no confluence to test")
    print(f"[draft]   seam lines: x={x_lines} y={y_lines}")
    # [..., :3] so the assert always compares 3-channel against the 3-channel draft —
    # the same narrowing refine_image applies (review D6: one raw source, one meaning).
    strip_source = ARM_STRIP_SOURCE[arm]
    strip_grounding = strips_vision_raw(canvas[..., :3]) if grounding == "raw" else contextlib.nullcontext()
    with strip_grounding:
        for axis, line in [("x", x) for x in x_lines] + [("y", y) for y in y_lines]:
            started = time.perf_counter()
            guider = upscale.build_guider(model, empty, negative, REFINE.cfg)
            noise = upscale.Noise_RandomNoise(REFINE.seed)
            mask = strip_mask(canvas_h, canvas_w, line, axis)
            strip_capture = LatentCapture() if handoff == "latent" else None
            strip_observer = strip_capture.active() if strip_capture else contextlib.nullcontext()
            # The strip's img2img substrate AND frozen ring context (see ARM_STRIP_SOURCE):
            # "draft" re-noises the tile draft band (arm-16's spiderweb generator at deep
            # cuts); "raw" drafts the raw band as an independent trajectory, like the
            # tiles. On the "raw" path the strips_vision_raw hook above becomes a
            # redundant identity — the encode source is already the raw canvas — and its
            # shape assert still runs.
            strip_input = canvas[..., :3] if strip_source == "raw" else draft
            ab_models.clear_cache()
            with no_min_error_cut(), strip_observer, ab_models.VramProbe() as probe, \
                    torch.inference_mode():
                strip_out = sampling.refine_image(
                    strip_input, guider, sampler, head, vae, noise,
                    # Canvas-sized caps: each strip samples as ONE tile. The default caps
                    # would segment it, and the segment junction lands exactly ON the
                    # crossing seam line (review finding 2); a single strip tile
                    # (768x3072 / 2304x768 + anchor) is smaller than a pass-2 tile.
                    max_tile_width=canvas_w,
                    max_tile_height=canvas_h,
                    context_anchor=REFINE.context_anchor,
                    context_overlap=REFINE.context_overlap,
                    mask=mask,
                    vl_clip=clip,
                    anchor_type="leading",
                    vlm_method=captions.VLM_METHOD_VISION,   # pure vision rows, no caption
                )
            draft = confluence_blend(draft, strip_out, line, axis)
            if strip_capture is not None:
                if len(strip_capture.records) != 1:
                    raise RuntimeError(
                        f"strip {axis}={line} made {len(strip_capture.records)} sampler "
                        "calls — the canvas caps no longer keep it a single tile")
                with torch.inference_mode():
                    blend_strip_latent(latent_canvas, strip_capture.records[0], mask,
                                       line, axis, canvas_h, canvas_w, cut_sigma)
                print(f"[latent]  strip {axis}={line} slerp-blended into the latent canvas")
            timings[f"strip-{axis}{line}"] = time.perf_counter() - started
            print(f"[draft]   strip {axis}={line} {timings[f'strip-{axis}{line}']:.1f}s  {probe}")
    return draft, latent_canvas, cut_sigma


def load_or_render_draft(arm, canvas, tiles, clip, model, vae, negative, empty,
                         tile_captions, head, mid, tail, timings, force):
    """The draft is deterministic given (base, settings, geometry, captions), so cache the
    TENSOR: a pass-2 crash or a later pass-2-only variant reruns in seconds instead of
    re-rendering ~7 minutes of pass 1. The PNG next to the render is for the owner's eyes;
    the pipeline always continues from the float tensor (no 8-bit round trip).
    Returns (draft, latent_canvas, cut_sigma) — latent fields None for pixel-handoff arms.
    A latent arm's payload is a dict {draft, latent, cut_sigma}; its digest differs from
    the pixel arms' via the "handoff" key (absent for them, so their digests are stable)."""
    handoff = ARM_HANDOFF[arm]
    key = {
        "base_sha": _file_sha(BASE_PNG),
        "settings": dataclasses.asdict(REFINE),
        "cut_fraction": ARM_CUT_FRACTION[arm],
        "head_sigmas": [float(v) for v in head],
        "strip_half_band": STRIP_HALF_BAND,
        "strip_blend_plateau": STRIP_BLEND_PLATEAU,
        "captions": tile_captions,
        "clip": CLIP_NAME, "unet": UNET_NAME, "vae": VAE_NAME,
        "stage": "confluence-draft",
        # Literals render_draft hardcodes, plus a recipe tag to bump on ANY change to
        # render_draft / confluence_blend / blend_weight — without these a stale cached
        # draft would silently mismatch the settings block of the render it feeds.
        "stage1_overlap": 0,
        "stage1_vlm": "vision tokens and captions",
        "strip_vlm": "vision tokens",
        "strip_caps": "canvas",
        "anchor_type": "leading",
        "seam_mode": "feather-only",
        "strip_order": "x-then-y",
        "vision_grounding": ARM_GROUNDING[arm],
        "recipe": "confluence-v2",
    }
    if handoff == "latent":
        key["handoff"] = "latent"
    if ARM_STRIP_SOURCE[arm] != "draft":
        # Conditional like "handoff": absent for arms 12-16, so their digests are stable.
        key["strip_source"] = ARM_STRIP_SOURCE[arm]
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    cached = CACHE_DIR / f"krea2-00676_confluence_draft_{digest}.pt"

    latent_canvas = None
    cut_sigma = None
    if cached.is_file() and not force:
        payload = torch.load(cached, map_location="cpu")
        if handoff == "latent":
            draft = payload["draft"]
            latent_canvas = payload["latent"]
            cut_sigma = payload["cut_sigma"]
        else:
            draft = payload
        timings["draft-cache-hit"] = 0.0
        print(f"[draft]   cache hit  {cached.name}")
        print("[draft]   -stage1.png not regenerated (pass 1 did not run this process)")
    else:
        draft, latent_canvas, cut_sigma = render_draft(
            arm, canvas, tiles, clip, model, vae, negative, empty,
            tile_captions, head, mid, tail, timings)
        draft = draft.detach().float().cpu()
        cached.parent.mkdir(parents=True, exist_ok=True)
        if handoff == "latent":
            torch.save({"draft": draft, "latent": latent_canvas, "cut_sigma": cut_sigma}, cached)
        else:
            torch.save(draft, cached)
        print(f"[draft]   cached -> {cached.name}")
    ab_models.require_image_shape(draft, canvas.shape[1], canvas.shape[2], "confluence draft")
    if handoff == "latent":
        if latent_canvas.shape[-2:] != (canvas.shape[1] // 8, canvas.shape[2] // 8):
            raise RuntimeError(
                f"latent canvas {tuple(latent_canvas.shape)} does not match the /8 canvas "
                f"({canvas.shape[1] // 8}, {canvas.shape[2] // 8})")
        if abs(cut_sigma - float(tail[0])) > 1e-6:
            raise RuntimeError(
                f"cached latent was cut at sigma {cut_sigma}, this run's tail starts at "
                f"{float(tail[0])} — CUT_FRACTION or the schedule changed under the cache")
    return draft, latent_canvas, cut_sigma


# ------------------------------------------------------------------ settings / render

def build_settings(arm, tile_captions, mid, head, tail, stage="final"):
    from context_anchored_tile_refine import captions

    instruction, max_length = captions.CAPTION_INSTRUCTIONS[captions.VLM_METHOD_VISION_CAPTIONS]
    payload = dataclasses.asdict(REFINE)
    payload.update({
        "run_label": arm if stage == "final" else f"{arm}-{stage}",
        "stage": stage,
        "arm": ARMS[arm],
        "vision_grounding": ARM_GROUNDING[arm],
        "cut_fraction": ARM_CUT_FRACTION[arm],
        "cut_step": mid,
        "cut_sigma": float(tail[0]),
        "collapse_completion_steps": ARM_COMPLETION[arm],
        "cut_handoff": ARM_HANDOFF[arm] + (
            "" if ARM_HANDOFF[arm] == "pixel" else
            " (trajectory state captured at the cut, slerp-blended, resumed with zero noise; "
            "the pixel draft is ring reference + forensics only)"),
        "pass1_sigmas": [float(v) for v in head],
        "pass1_stage1": "butted tiles (overlap 0), arm-1 whole-image caption surface",
        "pass1_strips": "mask-path strips, VLM_METHOD_VISION (pure vision rows)",
        "pass1_strip_source": ARM_STRIP_SOURCE[arm] + (
            " (re-noises the tile draft band)" if ARM_STRIP_SOURCE[arm] == "draft" else
            " (drafts the raw band as an independent trajectory)"),
        "pass1_seam_mode": "feather only (seam_displacements patched to straight)",
        "strip_half_band": STRIP_HALF_BAND,
        "strip_blend_plateau": STRIP_BLEND_PLATEAU,
        "pass2": "stock pipeline over the draft, arm-2 conditioning (text-only captions)",
        "base_png": str(BASE_PNG),
        "base_sha": _file_sha(BASE_PNG),
        "unet": UNET_NAME, "clip": CLIP_NAME, "clip_type": CLIP_TYPE, "vae": VAE_NAME,
        "upscale_model": UPSCALE_MODEL_NAME,
        "guider": "CFGGuider",
        "anchor_type": "leading",
        # The shipped curve, but anchor_ring_schedule normalizes r per CALL: pass 1's r
        # bottoms out at sigma_cut/sigma_0 (well above the 0.15 release), so pass 1 runs
        # pure lead1; pass 2's release lands at 15% of the TAIL schedule.
        "ring_curve": "lead1r15 per call (pass1 = pure lead1, release never reached; "
                      "pass2 releases at 15% of the tail)",
        "vlm_method": captions.VLM_METHOD_VISION_CAPTIONS,
        "negative_prompt": NEGATIVE_PROMPT,
        "caption_instruction": instruction,
        "caption_max_length": max_length,
        "tile_captions": [row[0] for row in tile_captions],
        "harness": "tests-AB/run_ab_confluence.py",
        "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    return payload


def render(arm, canvas, padded, tiles, clip, model, vae, negative, empty, tile_captions,
           force_draft):
    """One confluence arm end to end: pass 1 (cached draft) then pass 2, both saved."""
    import comfy.samplers

    from context_anchored_tile_refine import captions, sampling, upscale

    grounding = ARM_GROUNDING[arm]
    if grounding not in ("draft", "raw"):
        # Both switch sites key off this value; an unknown one would silently produce a
        # hybrid grounding (review D5).
        raise RuntimeError(f"unknown vision grounding {grounding!r} for {arm}")
    if ARM_STRIP_SOURCE[arm] not in ("draft", "raw"):
        raise RuntimeError(f"unknown strip source {ARM_STRIP_SOURCE[arm]!r} for {arm}")
    completion = ARM_COMPLETION[arm]
    sigmas = upscale.build_sigmas(model, REFINE.scheduler, REFINE.steps, REFINE.denoise)
    head, tail, mid = split_sigmas(sigmas, completion, ARM_CUT_FRACTION[arm])
    collapse_ladder = ", ".join(f"{float(v):.4f}" for v in head[mid + 1:])
    print(f"[render]  {arm}: schedule split at step {mid}/{int(sigmas.shape[-1]) - 1}: "
          f"pass 1 = {int(head.shape[-1]) - 1} steps (incl. collapse), "
          f"pass 2 = {int(tail.shape[-1]) - 1} steps, cut sigma {float(tail[0]):.4f}, "
          f"vision grounding '{grounding}'")
    print(f"[render]  collapse ladder ({completion} completion steps): "
          f"{float(head[mid]):.4f} -> {collapse_ladder}")

    timings = {}
    total_started = time.perf_counter()

    # ---- PASS 1: the draft (cached), saved as a PNG for the owner's inspection
    draft, latent_canvas, cut_sigma = load_or_render_draft(
        arm, canvas, tiles, clip, model, vae, negative, empty,
        tile_captions, head, mid, tail, timings, force_draft)
    # Tonal-drift instrumentation (the arm-13 failure mode as a number), Rec.601 luma —
    # see mean_luma for the measured baselines. Stage 1 prints its own delta inside
    # render_draft, ~2 minutes in: the earliest go/no-go on the collapse mechanism.
    # A latent arm's PIXEL draft is expected to sit at arm 13's bright +2.5/+3.3 — it is
    # the x0 forensic view and ring reference, not what pass 2 consumes; the hypothesis
    # lives in the FINAL tone.
    raw_luma = mean_luma(canvas)
    print(f"[tone]    draft luma delta vs raw: {mean_luma(draft) - raw_luma:+.2f}/255"
          "  (arm-13 draft: +3.28)")
    draft_png = output_path(arm, "-draft")
    ab_models.save_png(draft_png, draft.cpu(),
                       build_settings(arm, tile_captions, mid, head, tail, stage="pass1-draft"))
    print(f"[draft]   -> {draft_png.name}")

    # ---- PASS 2: the stock pipeline over the draft, arm-2 conditioning. On the latent
    # handoff, each tile's diffused region resumes the captured trajectory state instead
    # of re-noising the draft (inject_latent_handoff); the draft pixels still feed the
    # per-tile encodes, so the frozen anchor rings keep their live-canvas reference.
    started = time.perf_counter()
    guider = upscale.build_guider(model, empty, negative, REFINE.cfg)
    noise = upscale.Noise_RandomNoise(REFINE.seed)
    sampler = comfy.samplers.sampler_object(REFINE.sampler)
    handoff_resume = (inject_latent_handoff(latent_canvas, tiles, cut_sigma, model)
                      if latent_canvas is not None else contextlib.nullcontext())
    ab_models.clear_cache()   # allocator reset before every sampler call (see render_draft)
    with split_run.inject_captions(tile_captions, tiles), \
            pass2_builder(grounding, padded), handoff_resume, \
            ab_models.VramProbe() as probe, torch.inference_mode():
        result = sampling.refine_image(
            draft, guider, sampler, tail, vae, noise,
            max_tile_width=REFINE.max_tile_width,
            max_tile_height=REFINE.max_tile_height,
            context_anchor=REFINE.context_anchor,
            context_overlap=REFINE.context_overlap,
            vl_clip=clip,
            anchor_type="leading",
            vlm_method=captions.VLM_METHOD_VISION_CAPTIONS,
        )
    timings["pass2-detail"] = time.perf_counter() - started
    print(f"[render]  pass 2 (stock, arm-2 conds) {timings['pass2-detail']:.1f}s  {probe}")

    ab_models.require_image_shape(result, canvas.shape[1], canvas.shape[2], "confluence refine")
    print(f"[tone]    final luma delta vs raw: {mean_luma(result) - raw_luma:+.2f}/255"
          "  (arm-13 final: +5.08; single-pass arms: +0.6 — the latent handoff's success band)")
    destination = output_path(arm)
    written = ab_models.save_png(destination, result.cpu(),
                                 build_settings(arm, tile_captions, mid, head, tail))
    timings["total"] = time.perf_counter() - total_started
    print(f"[render]  {arm} -> {destination.name} {written[0]}x{written[1]}")
    print("[timing]  " + "  ".join(f"{name}={seconds:.1f}s" for name, seconds in timings.items()))


# ------------------------------------------------------------------ main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", action="append", default=[], metavar="ARM",
                        help="render only these arms (repeatable; ONE per process at 3x)")
    parser.add_argument("--force", action="store_true", help="re-render existing outputs")
    parser.add_argument("--force-draft", action="store_true", help="rebuild the cached pass-1 draft")
    parser.add_argument("--force-captions", action="store_true", help="regenerate the cached captions")
    parser.add_argument("--list", action="store_true", help="show the arms and exit")
    args = parser.parse_args(argv)

    unknown = set(args.only) - set(ARMS)
    if unknown:
        raise SystemExit("unknown arm(s): {}".format(", ".join(sorted(unknown))))
    selected = [arm for arm in ARMS if not args.only or arm in args.only]

    if args.list:
        for arm in ARMS:
            marker = "*" if arm in selected else " "
            print(f"{marker} {arm:<28} {ARMS[arm]}  -> {output_path(arm).name}")
        return 0

    root, note = ab_env.bootstrap()
    print(f"[env]     ComfyUI {ab_env.version(root)} at {root}  ({note})")
    print("[env]     torch {}  cuda {}  device {}".format(
        torch.__version__, torch.version.cuda,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))

    base = krea2_run.load_base()
    canvas = krea2_run.stage_upscale(base, force=False)
    padded, tiles = krea2_run.solve_tiles(canvas)
    if tuple(padded.shape[1:3]) != tuple(canvas.shape[1:3]):
        raise SystemExit("canvas is not /8 — seam lines were solved in padded coordinates "
                         "and would not match the working canvas")

    print(f"[clip]    loading {CLIP_NAME} ({CLIP_TYPE})")
    with ab_models.VramProbe() as probe:
        clip = ab_models.load_clip(CLIP_NAME, CLIP_TYPE)
        with torch.inference_mode():
            from context_anchored_tile_refine import upscale
            negative = ab_models.encode_prompt(clip, NEGATIVE_PROMPT)
            empty = upscale.encode_empty(clip)
        tile_captions = split_run.load_or_generate_captions(clip, padded, tiles, args.force_captions)
    print(f"[clip]    conditioning prepared  {probe}")

    print(f"[unet]    loading {UNET_NAME}")
    with ab_models.VramProbe() as probe:
        model = ab_models.load_unet(UNET_NAME)
        vae = ab_models.load_vae(VAE_NAME)
    print(f"[unet]    loaded  {probe}")

    for arm in selected:
        destination = output_path(arm)
        if arm in LINEAGE_ARMS and arm not in args.only:
            print(f"[render]  {arm:<28} lineage arm (judged), skipped — name it with "
                  "--only to re-render deliberately")
            continue
        if destination.is_file() and not args.force:
            print(f"[render]  {arm:<28} exists, skipped (--force to redo)")
            continue
        render(arm, canvas, padded, tiles, clip, model, vae, negative, empty,
               tile_captions, args.force_draft)

    bad = []
    for arm in selected:
        destination = output_path(arm)
        if not destination.is_file():
            bad.append(f"{destination.name} MISSING")
            continue
        width, height = ab_models.png_size(destination)
        verdict = "OK" if (width, height) == (canvas.shape[2], canvas.shape[1]) else "WRONG SIZE"
        if verdict != "OK":
            bad.append(f"{destination.name} is {width}x{height}")
        print(f"[done]    {destination.name:<48} {width:>4}x{height:<4} {verdict}")
    if bad:
        raise SystemExit("[done]    FAILED: " + "; ".join(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(
        "run_ab_confluence.py: PRE-PORT CAMPAIGN, pinned to commit 0dfd1e4 (see the module"
        " docstring). refine_image no longer accepts anchor_type= — the raster VL branch these"
        " arms drove was removed in 1.6.0. Check out 0dfd1e4 to re-run. Module helpers stay"
        " importable (the active sync harnesses use them)."
    )
