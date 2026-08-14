"""Conditioning matrix for the Krea 2 nodes: VL slicing x per-tile captions x anchor Lead.

The shipping decision this feeds: WHICH knobs belong on ContextAnchoredTileUpscaleVL, and
whether Lead is worth adding to the base (non-VL) node. Three independent binary knobs =>
2^3 = 8 configurations, run identically on three scene types (round 1, 24 renders), plus
targeted follow-ups (round 2, `_v2` suffix) that vary seed, denoise, or VL row strength.

    VL slicing  per-tile row slices of ONE whole-canvas vision encode (vl.py, production).
                Holds the original style, adds high-frequency detail, and each tile knows
                what it contains and what the canvas contains.
    Captions    per-tile VLM caption of that tile's crop, encoded as plain text. Names what
                is in each tile (helps the join) but invents detail the source lacks.
    Lead        the shipped anchor-ring schedule (sampling.anchor_ring_factor): the frozen
                context_anchor ring is presented at sigma * factor(r) so it resolves AHEAD
                of the core and the core is drawn toward it. Stronger adherence to the
                source, less invented detail. OFF here means comfy's default — the ring
                re-noised to the current sigma, which is what shipped before 2026-08-12.

Geometry is fixed and deliberately exaggerated so one seam is easy to judge: 512x288 base
-> 3x -> 1536x864 (16:9), max_tile 1152 => a 2x1 grid with ONE vertical seam at x=768,
context_anchor 256 (the widest ring worth recommending), context_overlap 32 (settled).
A cap of 1024 would solve to FOUR columns at anchor 256 (768 + 288 > 1024), so the cap is
part of the experiment, not a detail.

    <venv-python> tests-AB/run_ab_matrix.py --list            # the full plan, renders nothing
    <venv-python> tests-AB/run_ab_matrix.py                   # everything not already on disk
    <venv-python> tests-AB/run_ab_matrix.py --scene portrait
    <venv-python> tests-AB/run_ab_matrix.py --only 2_Lead --force

Outputs: output\\AB-Final-Matrix\\{n}-{scene}\\{cell}.png — one folder per scene, the cell
named in the file, numbered so a viewer steps through in truth-table order. Every PNG
carries its full settings (and its tile captions) in a `catr_ab` tEXt chunk.
"""
import argparse
import contextlib
import dataclasses
import hashlib
import itertools
import json
import os
import re
import sys
import time
from pathlib import Path

import torch

# ------------------------------------------------------------------ CONFIG

OUTPUT_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI\output\AB-Final-Matrix")
CACHE_DIR = Path(__file__).resolve().parent / "cache"

KREA2_ROOT = Path(r"C:\Users\Blake\ComfyUI-Installs\ComfyUI\ComfyUI")

UNET_NAME = "krea2Center_v408bitFp8Int8.safetensors"
CLIP_NAME = "qwen3-vl-4b-heretic_int8.safetensors"
CLIP_TYPE = "krea2"
VAE_NAME = "krea2RealVae_v10.safetensors"
UPSCALE_MODEL_NAME = "4xFaceUpDAT.safetensors"

# --- base generation (run_ab_anchor.py's chain, verbatim, so the city scene reuses its cache)
GEN_WIDTH, GEN_HEIGHT = 512, 288
GEN_SEED = 16928
GEN_STEPS = 52
GEN_SAMPLER = "dpmpp_2m"
GEN_SCHEDULER = "sgm_uniform"

# --- refine stage (the owner's node 364 widgets). seed and denoise are per-cell defaults.
REFINE_SEED = 42
REFINE_SAMPLER = "exp_heun_2_x0_sde"
REFINE_SCHEDULER = "sgm_uniform"
REFINE_STEPS = 28
DENOISE = 0.50
CFG = 3.5
UPSCALE_BY = 3.0
MAX_TILE = 1152
CONTEXT_ANCHOR = 256
CONTEXT_OVERLAP = 32
EXPECTED_GRID = (2, 1)          # (columns, rows) — asserted, never assumed

# AB27's caption instruction: Krea 2's own conditioning-template register. Qwen3-VL reads
# it as an essay assignment and returns 370-410 words per tile, which is ~500 rows against
# a tile's 546 vision rows — the caption is HALF the sequence, not a garnish. Kept as the
# default because it is what the captions-only cells were settled on.
CAPTION_INSTRUCTION = ("Describe the image by detailing the color, shape, size, texture, "
                       "quantity, text, spatial relationships of the objects and background.")
CAPTION_MAX_TOKENS = 512

# The COMPLEMENT instruction, for cells that also carry VL slices. The vision rows already
# encode appearance, and they encode it positionally — so an appearance caption is
# redundant mass competing with rows that say the same thing more precisely. What the
# vision rows do NOT carry is which object a region belongs to and how much of that object
# the tile actually contains. This asks for that and nothing else: no colour, texture,
# material, lighting, style or mood, all of which are already in the slices.
POSITION_INSTRUCTION = (
    "State only where each thing is in this image and how much of it is in frame. One "
    "short clause per thing, naming it in one or two plain words and giving its position "
    "and whether an edge cuts it off, for example 'lower half of a tree trunk at the left "
    "edge' or 'woman's face, centre'. Do not describe colour, texture, material, lighting, "
    "style, mood, or picture quality. No introduction and no summary. Under 50 words.")
# Deliberately NOT hardened against the heading/meta/repeat failures seen on 2026-08-13:
# this exact wording produced the caption behind the owner's approved face render, and
# clean_caption removes all three artifacts as a post-process without touching a caption
# that has none (an identity on the face text). Change the wording only if the loop recurs
# somewhere dedup cannot fix.
POSITION_MAX_TOKENS = 160

# The CAPTIONS-ONLY instruction. Same positional skeleton as POSITION_INSTRUCTION, because
# that is what made the caption stop fighting the vision rows — but with style, palette and
# lighting added ONCE at the front, since with no VL slices nothing else carries appearance.
# "Saying nothing twice" is the anti-duplication clause; the focus/sharpness ban stays,
# because "soft focus" and "slightly blurred" are demands to destroy detail in a refine pass
# and the descriptive instruction produced them on every scene.
RICH_INSTRUCTION = (
    "Describe this image as a compact list, saying nothing twice. First one short clause for "
    "the overall style, palette and lighting. Then one short clause per thing: its position "
    "in the frame, whether an edge cuts it off, what it is, and its material or colour. Do "
    "not comment on focus, sharpness, blur, or picture quality. No introduction and no "
    "summary. Under 90 words.")
RICH_MAX_TOKENS = 220

# --- the SETTLED pair (2026-08-13), from the six-round search in tests-AB/vlm_prompt_lab.py
# and the owner's own ComfyUI trials of the finalists. One instruction per surface:
#
#   SETTLED_POSITION  rides WITH the VL slices. The vision rows carry appearance
#                     positionally; what they lack is which object a region belongs to and
#                     how much of it the tile holds, which is exactly what this asks for.
#   SETTLED_RICH      the captions-ONLY surface. Style, palette and lighting lead, because
#                     with no vision rows nothing else carries appearance.
#
# Findings baked into the wording, none of which may be "tidied" out:
#   - The position prompt carries TWO independent bounds and either one alone terminates:
#     the `up to eight` ceiling and `Name whole objects and count repeated objects as one
#     entry.` With NEITHER, 7 of 9 round-1 cells ran to the token cap — one object mined
#     for its parts (hair/eyes/lips/neck...) or one repeated per instance (stall x30).
#     What the clause does ON TOP of the ceiling is suppress quantity words: removing it
#     took numeric mentions from 1 to 5 across 9 tiles (round 7) without costing the stop.
#   - `short phrase` holds items to 6-11 words; asking for a `sentence` licenses 32-40.
#   - the five REGIONS in the rich prompt are a stop condition set by the QUESTION, not by
#     how busy the picture is, which is why that one alone never runs away.
#   - The rich prompt must NOT gain the clause, tempting as it looks after a bad caption:
#     round 7 measured it at 7/9 -> 4/9, because it brings back the uniform-crop repeat
#     that `giving each part its own description` exists to prevent (four tiles regressed).
#     Round 7 also found the VLM's counting is ACCURATE — 5 of 6 quantity words across 36
#     generations were true ("two chickens", "multiple stalls"); the single false one was
#     "Two massive, glowing red moons" on the face canvas, where tile 1's crop really does
#     hold a second bright object (a blue-white orb in red nebula) that got the moon's
#     label. A second base draw of the same prompt (scene face2, gen_seed 7391) captioned
#     it cleanly, so that miss is a property of one picture, not of this wording.
# Spelling is deliberate: the owner A/B'd US against EU spelling in ComfyUI and EU won —
# "centre" recovered details ("the fox") that "center" dropped. Do not Americanise.
GROUP_CLAUSE = "Name whole objects and count repeated objects as one entry."

SETTLED_POSITION_INSTRUCTION = (
    "List up to eight main things in this image, one per line, each a short phrase naming "
    f"the thing, its position in the frame, and how much of it shows. {GROUP_CLAUSE}")
SETTLED_POSITION_MAX_TOKENS = 512

SETTLED_RICH_INSTRUCTION = (
    "Describe this image, one short line per part. Start with the overall style, palette "
    "and lighting. Then say what fills the left, the centre, the right, the top and the "
    "bottom, giving each part its own description with what is there, its colour and what "
    "its surface is made of.")
SETTLED_RICH_MAX_TOKENS = 768

# The rich prompt WITH the grouping clause. SHIPPED as the captions-only surface of the
# nodes' vlm_method select (context_anchored_tile_refine/captions.py RICH_GROUPED_INSTRUCTION)
# — the owner's explicit decision on 2026-08-13, taken against the lab score. Cell 17
# (1-face/17_CaptionOnly+Group_Lead_s42_v3) was judged "better across the board" against cell
# 14 (ungrouped, which drew a phantom second moon), and region repeats were ruled acceptable
# on this surface: "Repeating may be fine, it often does that only for predominate stuff and
# that just increases weight when it's caption only. This was only an issue with VL method
# combined." The contrary evidence stands on the record: round 7 measured this wording at
# 7/9 -> 4/9 across the lab tiles, because it brings back the uniform-crop repeat that
# "giving each part its own description" exists to prevent, and it was rendered on the face
# scene ONLY.
RICH_GROUPED_INSTRUCTION = f"{SETTLED_RICH_INSTRUCTION} {GROUP_CLAUSE}"


@dataclasses.dataclass(frozen=True)
class Scene:
    key: str            # cache label + folder name
    title: str          # what this scene is testing
    positive: str
    negative: str
    # The BASE generation's seed — a different picture from the same prompt, which is the
    # only knob that can re-roll a caption. A cell's own `seed` is the REFINE draw and
    # enters after the canvas and its captions are already fixed and cached, so two cells
    # of one scene always read identical caption text however their seeds differ.
    # Defaults to GEN_SEED, so every scene written before this field keeps its cache key.
    gen_seed: int = GEN_SEED


SCENES = [
    Scene(
        key="face",
        title="close-up character",
        # The 00676 dark-fantasy prompt (run_ab_krea2.py node 344), verbatim.
        positive=(
            "breathtaking, striking cinematic dark fantasy, low angle perspective, dream girl "
            "fantasy character, strangely attractive face, striking otherworldly beauty, long "
            "hair in loose waves, dense enchanted woodland filled with towering twisted oaks "
            "and thick undergrowth, forest floor covered in blood red flowers, ancient "
            "crumbling stone ruins half-swallowed by ivy, night sky dominated by a massive "
            "blood-red full moon with faint nebula streaks visible through the forest canopy, "
            "light mist, dramatic chiaroscuro lighting, deep crimson, obsidian black, rich "
            "violet, gold accents, ethereal, ominous, mesmerizing, intensely atmospheric."
        ),
        negative=("undead, zombie, jpeg compression, scribbles, AI generated, "
                  "plastic toy doll, shredded cloth, freckles"),
    ),
    Scene(
        key="cybercity",
        title="distant city",
        # run_ab_anchor.py's pair, verbatim ("hredded" [sic]) so the cached base is reused.
        positive="Cyberpunk cityscape at night",
        negative="jpeg compression, scribbles, AI generated, hredded cloth",
    ),
    Scene(
        key="market",
        title="busy mid-distance crowd",
        positive=(
            "Ground-level view down a crowded Renaissance market square in Florence, 16:9. "
            "Plane 1: a wooden merchant cart in the immediate foreground, its edge cropped by "
            "the frame, loaded with copper pots, wheels of cheese, and hanging dried herbs, "
            "every rope fiber and dent visible. Plane 2: a lane of canvas-canopied stalls "
            "receding into the square cloth merchants unrolling bolts of dyed fabric, a "
            "falconer, monks, ladies in brocade gowns, chickens underfoot, banners strung "
            "between timber-framed buildings. Plane 3: Brunelleschi-style cathedral dome "
            "rising above the rooftops at the far end, hazy in golden afternoon light. "
            "Photorealistic oil-painting realism, sharp detail."
        ),
        negative="jpeg compression, scribbles, AI generated",
    ),
    Scene(
        key="portrait",
        title="close-up photo, detailed background",
        # The base node's stress case: a real face straddling the x=768 seam (the seam runs
        # down the middle of it, so any left/right mismatch is on the most legible subject
        # there is), against a background that is DETAILED rather than bokeh — a shallow
        # depth of field would give the tiler nothing to disagree about on the other side.
        positive=(
            "Close-up documentary photograph of a weathered older man's face centered in the "
            "frame, deep depth of field, sharp throughout: every pore, grey stubble, sun "
            "crease and iris fiber visible, catchlight in both eyes. Behind him a cluttered "
            "workshop wall in full focus, pegboard hung with hand tools, coils of copper "
            "wire, glass jars of screws, a curling paper wall calendar, sawdust drifting in "
            "hard afternoon light from a dusty window. Editorial photography, 85mm, natural "
            "color, fine film grain."
        ),
        negative=("illustration, painting, drawing, 3d render, cgi, jpeg compression, "
                  "scribbles, AI generated, plastic skin, waxy, blurry background, bokeh"),
    ),
]

# A SECOND DRAW of the face prompt, appended so the folder numbers of the four original
# scenes do not move. Same prompt, same negative, same everything except the base seed —
# which is the only knob that can put different pixels in front of the VLM and therefore
# the only way to ask whether a caption defect is a property of the WORDING or of one
# picture. Added 2026-08-13 for the "two moons" caption on face tile 1: the refine seed
# cannot answer that question because captions are cached per canvas, not per draw.
SCENES.append(dataclasses.replace(SCENES[0], key="face2", gen_seed=7391,
                                  title="close-up character, second base draw"))

SCENES_BY_KEY = {s.key: s for s in SCENES}


@dataclasses.dataclass(frozen=True)
class Cell:
    """One render. Everything that can vary between renders lives here; anything not
    listed is held identical across the whole campaign by construction."""
    scene: str          # Scene.key
    name: str           # file stem — the label the owner reads, no extension
    vl: bool
    captions: bool
    lead: bool
    note: str
    seed: int = REFINE_SEED
    denoise: float = DENOISE
    # TBG-ETUR's row-strength trick (run_ab_anchor.py vl_scale_hook): multiply the VL
    # encode's SEQUENCE tensor before slicing, extras untouched. 1.0 = production.
    vl_scale: float = 1.0
    # Which rows the scale reaches. The slice+caption encode is one tensor holding BOTH
    # the vision grid and the caption text, so the historical whole-tensor scale and a
    # vision-rows-only scale are different experiments once captions are on.
    #   "vision"  grid rows only; vision_start/vision_end/caption/template tail stay 1.0
    #   "all"     the whole encode, i.e. the historical hook applied literally
    vl_scope: str = "vision"
    # What the VLM is asked for. Per-cell because the right question differs by cell: a
    # captions-ONLY cell wants the full description (nothing else supplies appearance),
    # while a VL+captions cell wants only what the vision rows lack.
    caption_instruction: str = CAPTION_INSTRUCTION
    caption_max_tokens: int = CAPTION_MAX_TOKENS
    # Strip the VLM's own formatting artifacts before the text is encoded (clean_caption).
    # OFF by default on purpose: the descriptive-caption cells were judged on their exact
    # raw text, and turning this on for them would silently change what those files mean.
    caption_clean: bool = False
    # Qwen3's reasoning turn. OFF for every cell rendered before 2026-08-13, which is why
    # it defaults off: those captions are cached and their renders judged. With it off the
    # tokenizer appends an EMPTY <think></think> block, the Qwen3 convention that suppresses
    # reasoning (qwen3vl.py:185) — and the lab measured a meta preamble ("Here are the
    # distinct things in the image:") on every tile as a result. With it on the model
    # reasons first and answers clean; generate_caption cuts the reasoning back off.
    caption_thinking: bool = False

    @property
    def caption_key(self):
        # `thinking` joins the key only when set, so every caption cached before the flag
        # existed keeps its digest and its render stays reproducible from disk.
        base = (self.caption_instruction, self.caption_max_tokens, self.caption_clean)
        return (*base, True) if self.caption_thinking else base

    @property
    def conditioning(self):
        if self.vl and self.captions:
            return "vl+captions"
        if self.vl:
            return "vl"
        if self.captions:
            return "captions"
        return "text"


def _combo_label(vl, captions, lead):
    # The owner's own names for the knobs; the all-off cell is the base node's mode.
    parts = [name for name, on in (("VL", vl), ("Captions", captions), ("Lead", lead)) if on]
    return "+".join(parts) if parts else "baseline"


def _factorial(scene_key):
    """The full 2^3, VL as the high bit so the two base-node cells sort first and the two
    richest VL cells sort last. itertools.product IS the enumeration — nothing is
    hand-listed, so a configuration cannot be forgotten."""
    cells = []
    for index, (vl, captions, lead) in enumerate(itertools.product((False, True), repeat=3), start=1):
        surface = "VL node" if (vl or captions) else (
            "base node AS IT SHIPS TODAY" if not lead else "base node + Lead")
        knobs = ", ".join(f"{k}={'on' if v else 'off'}" for k, v in
                          (("VL", vl), ("captions", captions), ("lead", lead)))
        cells.append(Cell(scene=scene_key, name=f"{index}_{_combo_label(vl, captions, lead)}",
                          vl=vl, captions=captions, lead=lead, note=f"{surface}: {knobs}"))
    return cells


# --- round 2, market: the bottom-left artifact in cells 7 and 8 ------------------------
# Two questions, two axes. Same seed answers "is it deterministic, or did that one run go
# wrong" — a byte-identical re-render means the artifact is a real property of this
# configuration, not a transient. A different seed answers "is it this configuration or
# this draw" — the seed is the strongest lever on this pipeline's per-tile behaviour
# (h3-seam-residual-dark-sky), so a clean seed-1234 pair points at the draw.
_MARKET_V2 = [
    Cell(scene="market", name=f"{index}_{label}{suffix}", vl=True, captions=True, lead=lead,
         seed=seed, note=note)
    for seed, suffix, note in (
        (42, "_v2", "re-run at the SAME seed — is the bottom-left artifact deterministic?"),
        (1234, "_s1234_v2", "different refine seed — artifact from the config or the draw?"))
    for index, label, lead in ((7, "VL+Captions", False), (8, "VL+Captions+Lead", True))
]

# --- round 2, face: TBG's VL row scaling on top of the richest cell --------------------
# vl_scale 0.5 alone melted a d0.35 render (vl-strength-scaling-destructive) but is what
# the other developer runs WITH captions — the caption may be carrying the semantic load
# that the halved vision rows drop. Both scopes are rendered because "lower the VL by 50%"
# is ambiguous once the caption rides inside the same tensor.
_FACE_V2 = [
    Cell(scene="face", name="9_VLx0.5-vision+Captions+Lead_v2", vl=True, captions=True,
         lead=True, vl_scale=0.5, vl_scope="vision",
         note="cell 8 with the vision GRID ROWS at half strength; caption rows left at 1.0"),
    Cell(scene="face", name="10_VLx0.5-whole+Captions+Lead_v2", vl=True, captions=True,
         lead=True, vl_scale=0.5, vl_scope="all",
         note="cell 8 with the WHOLE encode at half strength — the historical hook, literally"),
]

# --- round 2, portrait: is Lead worth adding to the base (non-VL) node? ----------------
# Only the two base-node cells, but across BOTH denoise levels and two refine seeds: the
# owner's objection to Lead on the baseline ("blends things weird at the seam") was seen at
# 0.5, which is far hotter than this node is ever run at. 0.35 is the real operating point.
# The refine seed varies, NOT the gen seed, so all eight share one photo.
_PORTRAIT = [
    Cell(scene="portrait", name=f"d{int(denoise * 100):03d}_s{seed}_{index}_{label}_v2",
         vl=False, captions=False, lead=lead, seed=seed, denoise=denoise,
         note=f"base node, lead {'on' if lead else 'off'}, denoise {denoise}, seed {seed}")
    for denoise in (0.35, 0.50)
    for seed in (42, 1234)
    for index, label, lead in ((1, "baseline", False), (2, "Lead", True))
]

# --- round 3: the two caption instructions, settled shapes -----------------------------
# Cell 11 (face) showed a 34-word positional caption sitting HALF as far from the VL-only
# render as the 389-word descriptive one, with the edge artifact gone. Round 3 asks the two
# remaining questions:
#   11 on market/cybercity — does the fix hold where the artifact was WORST (market's
#      bottom-left band reproduced on both seeds), and on a scene with no subject at all?
#   12 everywhere it is useful — the captions-ONLY surface, whose caption must also carry
#      style because no vision rows do. Pairs against cell 4 (same config, long caption).
def _positional_vl(scene_key, seed=REFINE_SEED):
    # A second seed is a REPLICATION arm, not a variant: the market artifact was only
    # settled as a property of the configuration once it reproduced on seed 1234, so the
    # fix has to clear the same bar before it counts as a fix.
    suffix = "" if seed == REFINE_SEED else f"_s{seed}"
    return Cell(scene=scene_key, name=f"11_VL+CaptionsPositional+Lead{suffix}_v2", vl=True,
                captions=True, lead=True, seed=seed,
                caption_instruction=POSITION_INSTRUCTION,
                caption_max_tokens=POSITION_MAX_TOKENS, caption_clean=True,
                note="cell 8 with a POSITION-ONLY caption — does complementary info stop the fight?"
                     + ("" if seed == REFINE_SEED else f" (replication, seed {seed})"))


def _rich_captions(scene_key):
    return Cell(scene=scene_key, name="12_CaptionsPosStyle+Lead_v2", vl=False, captions=True,
                lead=True, caption_instruction=RICH_INSTRUCTION,
                caption_max_tokens=RICH_MAX_TOKENS, caption_clean=True,
                note="captions-ONLY with the concise positional+style caption; pairs with cell 4")


# --- round 4 (v3): the settled pair, one render per surface per scene ------------------
# The final sign-off pair. Same seed (42), same denoise (0.50), same Lead-on configuration
# as the v2 cells 11 and 12, so v3 vs v2 is a clean read on the INSTRUCTION alone. The
# method is in the filename because that is what is being judged:
#
#   13_VL+Caption_Lead_s42_v3     VL slices + SETTLED_POSITION  — the VL node's surface
#   14_CaptionOnly_Lead_s42_v3    captions only + SETTLED_RICH  — the captions-only surface
#   15_VL+Caption_NoLead_s42_v3   the same two with anchor_ring_factor forced to 1.0, so
#   16_CaptionOnly_NoLead_s42_v3  Lead is re-confirmed against the SETTLED captions rather
#                                 than inherited from the round-1 factorial's wording
#
# All four carry caption_thinking=True: the settled wordings were searched and chosen with
# the reasoning turn on, and the owner's ComfyUI trials that picked them ran with it on too.
# Lead is in the FILENAME because with four cells per scene the tEXt chunk is no longer the
# fastest way to tell which is which.
#
# SEED IS A REPLICATION ARM, NOT A VARIANT — and it cannot re-roll a caption. Captions are
# generated from the FROZEN upscaled canvas and cached per (scene, instruction), so every
# seed of a scene reads the identical text. A second seed therefore answers "does the
# sampler act on this demand every time", never "does the VLM say it again".
def _v3_pair(scene_key, seed=REFINE_SEED):
    return [
        Cell(scene=scene_key, name=f"{index}_VL+Caption_{tag}_s{seed}_v3", vl=True, captions=True,
             lead=lead, seed=seed,
             caption_instruction=SETTLED_POSITION_INSTRUCTION,
             caption_max_tokens=SETTLED_POSITION_MAX_TOKENS, caption_clean=True,
             caption_thinking=True,
             note=f"SETTLED position caption alongside the VL slices, lead {'on' if lead else 'off'}"
                  + ("; pairs with 11_..._v2" if lead else ""))
        for index, tag, lead in ((13, "Lead", True), (15, "NoLead", False))
    ] + [
        Cell(scene=scene_key, name=f"{index}_CaptionOnly_{tag}_s{seed}_v3", vl=False, captions=True,
             lead=lead, seed=seed,
             caption_instruction=SETTLED_RICH_INSTRUCTION,
             caption_max_tokens=SETTLED_RICH_MAX_TOKENS, caption_clean=True,
             caption_thinking=True,
             note=f"SETTLED rich caption, no VL rows, lead {'on' if lead else 'off'}"
                  + ("; pairs with 12_..._v2" if lead else ""))
        for index, tag, lead in ((14, "Lead", True), (16, "NoLead", False))
    ]


PLAN = tuple(
    _factorial("face") + _FACE_V2
    + [_positional_vl("face"), _positional_vl("face", 1234), _rich_captions("face")]
    + _v3_pair("face") + _v3_pair("face", 1234)
    # The moon A/B: identical canvas, identical seed, identical Lead-on configuration as
    # 14_CaptionOnly_Lead_s42_v3 — the ONLY difference is the grouping clause on the rich
    # instruction. 14 renders two moons (the caption asks for two and the crop has a second
    # bright object to attach one to); this one's caption asks for one.
    + [Cell(scene="face", name="17_CaptionOnly+Group_Lead_s42_v3", vl=False, captions=True,
            lead=True, caption_instruction=RICH_GROUPED_INSTRUCTION,
            caption_max_tokens=SETTLED_RICH_MAX_TOKENS, caption_clean=True,
            caption_thinking=True,
            note="cell 14 with the grouping clause added to the rich caption — does it "
                 "remove the second moon, and what does it cost elsewhere in the frame?")]
    + _factorial("cybercity") + [_positional_vl("cybercity")] + _v3_pair("cybercity")
    + _factorial("market") + _MARKET_V2
    + [_positional_vl("market"), _positional_vl("market", 1234), _rich_captions("market")]
    + _v3_pair("market")
    + _PORTRAIT + _v3_pair("portrait")
    + _v3_pair("face2")
)

# ------------------------------------------------------------------ bootstrap

sys.path.insert(0, str(Path(__file__).resolve().parent))
if "COMFYUI_ROOT" not in os.environ:
    if not (KREA2_ROOT / "comfy" / "text_encoders" / "krea2.py").is_file():
        raise SystemExit(f"No Krea 2 support at {KREA2_ROOT} — set COMFYUI_ROOT")
    os.environ["COMFYUI_ROOT"] = str(KREA2_ROOT)
import ab_env  # noqa: E402  (must precede every comfy import)
import ab_models  # noqa: E402


def scene_dir(scene_key):
    index = [s.key for s in SCENES].index(scene_key) + 1
    return OUTPUT_DIR / f"{index}-{scene_key}"


def output_path(cell):
    return scene_dir(cell.scene) / f"{cell.name}.png"


def _rects(tiles):
    return [[t.crop_rect.x0, t.crop_rect.y0, t.crop_rect.x1, t.crop_rect.y1] for t in tiles]


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def validate_plan():
    """Fail before any GPU time on a plan that cannot be rendered as written."""
    seen = set()
    for cell in PLAN:
        key = (cell.scene, cell.name)
        if key in seen:
            raise SystemExit(f"duplicate cell {cell.scene}/{cell.name}")
        seen.add(key)
        if cell.scene not in SCENES_BY_KEY:
            raise SystemExit(f"{cell.name}: unknown scene {cell.scene}")
        if cell.vl_scope not in ("vision", "all"):
            raise SystemExit(f"{cell.name}: vl_scope must be 'vision' or 'all'")
        if cell.vl_scale != 1.0 and not (cell.vl and cell.captions):
            # Scaling is implemented inside the slice+caption encode, which is the only
            # place the vision grid rows are separable from everything else.
            raise SystemExit(f"{cell.name}: vl_scale is only wired for VL+captions cells")


# ------------------------------------------------------------------ pixel stages

def gen_key(scene):
    # run_ab_anchor.py's key, field for field, so the cybercity entry already on disk hits.
    return {"prompt": scene.positive, "negative": scene.negative, "seed": scene.gen_seed,
            "sampler": GEN_SAMPLER, "scheduler": GEN_SCHEDULER, "steps": GEN_STEPS,
            "cfg": CFG, "w": GEN_WIDTH, "h": GEN_HEIGHT, "unet": UNET_NAME}


def stage_base(scene, model, clip, vae, force):
    """The scene's own 512x288 base generation, cached."""
    from context_anchored_tile_refine import upscale

    cached = ab_models.cache_path(CACHE_DIR, scene.key, "base", GEN_WIDTH, GEN_HEIGHT, gen_key(scene))
    if cached.is_file() and not force:
        print(f"[base]    {scene.key:<10} cache hit  {cached.name}")
        return torch.load(cached, map_location="cpu")

    with ab_models.VramProbe() as probe, torch.inference_mode():
        positive = ab_models.encode_prompt(clip, scene.positive)
        negative = ab_models.encode_prompt(clip, scene.negative)
        guider = upscale.build_guider(model, positive, negative, CFG)
        sigmas = ab_models.build_sigmas(model, GEN_SCHEDULER, GEN_STEPS, 1.0)
        sampler = ab_models.build_sampler(GEN_SAMPLER)
        noise = ab_models.build_noise(scene.gen_seed)
        latent = ab_models.empty_sd3_latent(GEN_WIDTH, GEN_HEIGHT, 1)
        out = ab_models.sample_custom_advanced(noise, guider, sampler, sigmas, latent)
        base = ab_models.vae_decode(vae, out).detach().float().cpu()
    ab_models.require_image_shape(base, GEN_HEIGHT, GEN_WIDTH, f"base gen {scene.key}")
    cached.parent.mkdir(parents=True, exist_ok=True)
    torch.save(base, cached)
    print(f"[base]    {scene.key:<10} generated  {probe}  -> {cached.name}")
    return base


def _load_upscale_model():
    """UpscaleModelLoader body incl. the .patcher wrap prepare_upscaled keys residency on
    (the V3 io node's NodeOutput return is not indexable, so the body is reproduced)."""
    import comfy.model_management
    import comfy.model_patcher
    import comfy.utils
    import folder_paths
    from comfy_extras.nodes_upscale_model import ImageModelDescriptor, ModelLoader

    model_path = folder_paths.get_full_path_or_raise("upscale_models", UPSCALE_MODEL_NAME)
    sd = comfy.utils.load_torch_file(model_path, safe_load=True)
    if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
        sd = comfy.utils.state_dict_prefix_replace(sd, {"module.": ""})
    out = ModelLoader().load_from_state_dict(sd).eval()
    if not isinstance(out, ImageModelDescriptor):
        raise RuntimeError("Upscale model must be a single-image model.")
    out.patcher = comfy.model_patcher.CoreModelPatcher(
        out.model, load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device())
    return out


def stage_upscale(scene, base, force):
    """The node's own whole-image upscale stage, cached. Every cell refines this canvas."""
    from context_anchored_tile_refine import upscale

    target_w, target_h = upscale.scale_target(GEN_WIDTH, GEN_HEIGHT, UPSCALE_BY)
    key = {"gen_seed": scene.gen_seed, "model": UPSCALE_MODEL_NAME, "upscale_by": UPSCALE_BY,
           "stage": "prepare_upscaled", "unet": UNET_NAME}
    if scene.key != "cybercity":
        # cybercity's cached entry predates this harness and its key has no prompt field;
        # every other scene gets the prompt folded in so two scenes cannot collide.
        key["gen"] = _digest(gen_key(scene))
    cached = ab_models.cache_path(CACHE_DIR, scene.key, "upscale", target_w, target_h, key)

    if cached.is_file() and not force:
        canvas = torch.load(cached, map_location="cpu")
        print(f"[upscale] {scene.key:<10} cache hit  {cached.name}")
    else:
        upscale_model = _load_upscale_model()
        try:
            with ab_models.VramProbe() as probe, torch.inference_mode():
                canvas = upscale.prepare_upscaled(base, upscale_model, UPSCALE_BY)
        finally:
            del upscale_model
            ab_models.free_gpu()
        canvas = canvas.detach().float().cpu()
        cached.parent.mkdir(parents=True, exist_ok=True)
        torch.save(canvas, cached)
        print(f"[upscale] {scene.key:<10} done  {probe}  -> {cached.name}")
    ab_models.require_image_shape(canvas, target_h, target_w, f"upscaled canvas {scene.key}")
    return canvas


def solve_tiles(scene, canvas):
    """The refine's own layout, computed up front so captions and the slice+caption encode
    are built from the exact tiles the sampler will use. The grid is ASSERTED: a layout
    that is not 2x1 is a different experiment and must fail before an hour of GPU time."""
    from context_anchored_tile_refine import grid, sampling

    padded, _ = sampling.pad_image_to_multiple(canvas[..., :3])
    sx = grid.solve_axis(padded.shape[2], MAX_TILE, CONTEXT_ANCHOR, CONTEXT_OVERLAP)
    sy = grid.solve_axis(padded.shape[1], MAX_TILE, CONTEXT_ANCHOR, CONTEXT_OVERLAP)
    layout = grid.build_layout(padded.shape[2], padded.shape[1], sx, sy,
                               CONTEXT_ANCHOR, CONTEXT_OVERLAP)
    if (sx.n, sy.n) != EXPECTED_GRID:
        raise SystemExit(
            f"[layout] {scene.key}: solved {sx.n}x{sy.n}, expected {EXPECTED_GRID[0]}x{EXPECTED_GRID[1]} "
            f"(canvas {padded.shape[2]}x{padded.shape[1]}, cap {MAX_TILE}, anchor {CONTEXT_ANCHOR}, "
            f"overlap {CONTEXT_OVERLAP})")
    print(f"[layout]  {scene.key:<10} canvas {padded.shape[2]}x{padded.shape[1]}  grid {sx.n}x{sy.n}  "
          f"seam x={layout.tiles[1].core.x0}  tile crop {layout.tiles[0].sampled_w}x{layout.tiles[0].sampled_h}")
    return padded, layout.tiles


# ------------------------------------------------------------------ conditioning

def resample_for_vl(tile_pixels):
    """AB27's caption input prep: area-resample a COPY of the tile's crop to 384^2 total
    pixels. Conditioning-side only — the sampled tile is never resampled."""
    import math

    import comfy.utils

    samples = tile_pixels.movedim(-1, 1)
    scale_by = math.sqrt(384 * 384 / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    resampled = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    return resampled.movedim(1, -1)[:, :, :, :3]


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_thinking(text):
    """Cut Qwen3's reasoning turn off the front of an answer.

    Mirrors comfy_extras/nodes_textgen.py TextGenerateLTX2Prompt, which is the only place
    core does this — the plain TextGenerate node returns the reasoning to the user. Here it
    is mandatory: the caption is encoded as plain text, so an unstripped <think> block would
    reach the DiT as several hundred tokens of the model talking to itself."""
    if "<think>" not in text:
        return text.strip()
    body = _THINK_BLOCK.sub("", text)
    if "</think>" in body:                  # truncated/unclosed: keep what follows the last
        body = body.rsplit("</think>", 1)[-1]
    return re.sub(r"</?think>", "", body).strip()


def generate_caption(clip, vl_input, instruction, max_length, thinking=False):
    """run_ab_krea2.py's generate_caption: greedy decode, then the settled fallback chain
    for stop-token-degenerate crops. `max_length` is per-instruction — the default 512 is
    where the descriptive captions truncate MID-SENTENCE, so an instruction that asks for
    a short answer gets a cap it will not reach rather than one it will be cut by. With
    `thinking` the cap also has to cover the reasoning turn, which is why the settled
    instructions carry 512 / 768 rather than the old 160 / 220."""
    tokens = clip.tokenize(instruction, images=[vl_input], thinking=thinking)
    ids = clip.generate(tokens, do_sample=False, max_length=max_length, repetition_penalty=1.05)
    text = strip_thinking(clip.decode(ids))
    if not text:
        ids = clip.generate(tokens, do_sample=True, max_length=max_length, temperature=0.7,
                            top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, seed=42)
        text = strip_thinking(clip.decode(ids))
    if not text:
        retokens = clip.tokenize("Write one short sentence describing this image.",
                                 images=[vl_input], thinking=False)
        ids = clip.generate(retokens, do_sample=False, max_length=max_length, repetition_penalty=1.05)
        text = strip_thinking(clip.decode(ids))
    if not text:
        raise RuntimeError("caption generation: empty answer after fallbacks")
    return text


_META_LINE = ("wait,", "wait ", "here's a revised", "here is a revised", "revised version",
              "let me", "i need to", "actually,")


def clean_caption(text):
    """Remove the VLM's own formatting artifacts so the DiT never reads them as content.

    Three observed failures, all on 2026-08-13 round 3, all of which reach the conditioning
    verbatim because the caption is encoded as plain text:
      heading      "**Answer:**", "**Image as compact list:**" — a label, not a description
      meta         "Wait, I need to rephrase to fit under 50 words." — the model narrating
      repetition   the market crop looped six items THREE times inside one 84-word caption,
                   which is exactly the duplication this instruction exists to avoid
    Dedup is by normalized line, so 'chicken, left' and 'chicken, right' both survive; only
    an exact repeat is dropped. Content is never rewritten, only whole artifact lines cut."""
    kept, seen, dropped = [], set(), False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower().lstrip("*_#-• ")
        if low.startswith(_META_LINE) or (
                not kept and line.startswith("**") and line.endswith(":**")):
            dropped = True                              # heading or meta narration
            continue
        key = " ".join(low.rstrip(".").split())
        if key in seen:
            dropped = True                              # the repetition loop
            continue
        seen.add(key)
        kept.append(line)
    # Nothing to remove => hand back the ORIGINAL string, byte for byte. Rejoining would
    # otherwise drop the markdown line-break spaces and re-tokenize a caption that was
    # already fine — silently moving a render the owner has already judged.
    return text if not dropped else "\n".join(kept).strip()


def load_or_generate_captions(scene, clip, padded, tiles, instruction, max_tokens, clean,
                              thinking, force):
    """One caption per tile from the FROZEN upscaled canvas, cached to JSON so every cell
    asking the SAME question of the VLM shares identical text (that is what makes the
    factorial clean). Seed-independent by construction: the canvas it describes is the same
    for every cell of a scene. The instruction is part of the cache key, so two instructions
    coexist rather than overwriting each other."""
    key = {"gen": _digest(gen_key(scene)), "rects": _rects(tiles), "instruction": instruction,
           "max_tokens": max_tokens, "clean": clean, "clip": CLIP_NAME}
    if thinking:
        # Added only when set — see Cell.caption_key. Every caption cached before the flag
        # existed keeps its digest, so no already-judged render can silently move.
        key["thinking"] = True
    cached = CACHE_DIR / f"{scene.key}_captions_{_digest(key)}.json"

    if cached.is_file() and not force:
        captions = json.loads(cached.read_text(encoding="utf-8"))["captions"]
        print(f"[caption] {scene.key:<10} cache hit  {cached.name}")
    else:
        captions = []
        for tile in tiles:
            crop = tile.crop_rect
            vl_input = resample_for_vl(padded[:, crop.y0:crop.y1, crop.x0:crop.x1, :])
            with torch.inference_mode():
                text = generate_caption(clip, vl_input, instruction, max_tokens, thinking)
            captions.append(clean_caption(text) if clean else text)
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(dict(key, captions=captions), indent=1), encoding="utf-8")
        print(f"[caption] {scene.key:<10} generated and cached  {cached.name}")
    for tile_idx, caption in enumerate(captions):
        print(f"[caption] {scene.key:<10} tile {tile_idx} ({len(caption.split())} words): {caption}")
    return captions


def build_caption_conds(clip, captions):
    """Captions WITHOUT slices: each caption re-encoded text-only, exactly what
    CLIPTextEncode would produce for that string."""
    from context_anchored_tile_refine import vl

    return [vl._convert(clip.encode_from_tokens_scheduled(clip.tokenize(c))) for c in captions]


def _scale_rows(tensor, scale, scope, n_rows):
    """TBG-ETUR's row-strength trick, applied to ONE encode's sequence tensor.

    Row layout of the slice+caption encode, pinned by vl.slice_indices:
        [0] vision_start | [1 .. n_rows] vision grid, raster | [n_rows+1] vision_end |
        [n_rows+2 ..] caption text + template tail
    "all" multiplies the whole tensor, which is the historical hook applied literally.
    "vision" multiplies only the grid rows, leaving the caption at full strength — the
    reading that matches "lower the VL so the conditioning does not carry too much mass".
    Neither touches extras (pooled_output etc.), as the historical hook did not."""
    if scale == 1.0:
        return tensor
    if scope == "all":
        return tensor * scale
    scaled = tensor.clone()
    scaled[:, 1:1 + n_rows] = scaled[:, 1:1 + n_rows] * scale
    return scaled


def build_slice_caption_conds(clip, padded, tiles, captions, scale=1.0, scope="vision"):
    """VL slices AND captions: the caption tokens ride INSIDE the whole-canvas vision
    encode (one encode per distinct caption), and each tile keeps its slice rows plus the
    caption + template tail. Reuses the production vl helpers so the slice math is the
    shipped code's; the seq fail-fast mirrors vl._encode_canvas. The row scale is applied
    BEFORE slicing, which is where the historical vl_scale_hook applied it."""
    from context_anchored_tile_refine import vl

    copy, enc_h, enc_w = vl.resample_for_global(padded)
    grid_h, grid_w = enc_h // vl.MERGED_CELL, enc_w // vl.MERGED_CELL
    n_rows = grid_h * grid_w
    canvas_h, canvas_w = int(padded.shape[1]), int(padded.shape[2])

    cache = {}
    conds = []
    for tile, caption in zip(tiles, captions, strict=True):
        if caption not in cache:
            tokens = clip.tokenize(vl.VISION_BLOCK + caption, images=[copy],
                                   llama_template=vl.KREA2_TEMPLATE)
            ids = [t[0] for t in tokens[next(iter(tokens))][0]]
            pad_pos = next(i for i, v in enumerate(ids) if isinstance(v, dict))
            tail_len = len(ids) - (pad_pos + 2)
            expected_seq = 1 + n_rows + 1 + tail_len
            encoded = clip.encode_from_tokens_scheduled(tokens)
            seq = encoded[0][0].shape[1]
            if seq != expected_seq:
                raise RuntimeError(
                    f"slice+caption encode has {seq} rows, expected {expected_seq} "
                    f"(grid {grid_h}x{grid_w} + caption/tail {tail_len})")
            encoded = [[_scale_rows(entry[0], scale, scope, n_rows), entry[1]] for entry in encoded]
            cache[caption] = (encoded, expected_seq)
        encoded, expected_seq = cache[caption]

        indices = vl.slice_indices(tile.crop_rect, canvas_h, canvas_w, enc_h, enc_w, expected_seq)
        sliced = []
        for entry in encoded:
            tensor, extras = entry[0], dict(entry[1])
            extras.pop("attention_mask", None)
            index = torch.tensor(indices, device=tensor.device)
            sliced.append([tensor.index_select(1, index), extras])
        conds.append(vl._convert(sliced))
    return conds


@dataclasses.dataclass
class SceneState:
    """Everything a scene's cells need before the DiT runs, built once per scene and only
    as far as the SELECTED cells actually require (a portrait run never calls the VLM)."""
    scene: Scene
    canvas: torch.Tensor
    padded: torch.Tensor
    tiles: tuple
    pos_text: object
    negative: object
    empty: object
    # All three are keyed by what the VLM was asked, so two instructions coexist in one run.
    captions: dict = dataclasses.field(default_factory=dict)      # caption_key -> [str]
    caption_conds: dict = dataclasses.field(default_factory=dict)  # caption_key -> conds
    slice_conds: dict = dataclasses.field(default_factory=dict)   # (scale, scope, key) -> conds


def build_state(scene, canvas, clip, cells, force_captions):
    from context_anchored_tile_refine import upscale

    padded, tiles = solve_tiles(scene, canvas)
    with torch.inference_mode():
        state = SceneState(
            scene=scene, canvas=canvas, padded=padded, tiles=tiles,
            pos_text=ab_models.encode_prompt(clip, scene.positive),
            negative=ab_models.encode_prompt(clip, scene.negative),
            empty=upscale.encode_empty(clip))

    for key in {c.caption_key for c in cells if c.captions}:
        state.captions[key] = load_or_generate_captions(
            scene, clip, padded, tiles, key[0], key[1], key[2], len(key) > 3, force_captions)
    with torch.inference_mode():
        for key in {c.caption_key for c in cells if c.conditioning == "captions"}:
            state.caption_conds[key] = build_caption_conds(clip, state.captions[key])
        for scale, scope, key in {(c.vl_scale, c.vl_scope, c.caption_key) for c in cells
                                  if c.conditioning == "vl+captions"}:
            state.slice_conds[(scale, scope, key)] = build_slice_caption_conds(
                clip, padded, tiles, state.captions[key], scale, scope)
            if scale != 1.0:
                print(f"[cond]    {scene.key:<10} slice+caption encode at VL x{scale:g} ({scope} rows)")
    return state


# ------------------------------------------------------------------ hooks

@contextlib.contextmanager
def override_tile_positives(prebuilt, tiles):
    """Swap vl.build_global_slices for a hook returning the prebuilt per-tile conds.
    sampling.py resolves `vl.build_global_slices` at call time, so this reaches the
    production swap seam without touching production code. The layout is re-asserted
    against the pre-pass tiles — a divergence would silently misassign positives."""
    from context_anchored_tile_refine import vl

    expected = _rects(tiles)

    def hook(clip, source, hook_tiles, offset_x=0, offset_y=0):
        if _rects(hook_tiles) != expected:
            raise RuntimeError("tile layout diverged between pre-pass and refine")
        return prebuilt

    original = vl.build_global_slices
    vl.build_global_slices = hook
    try:
        yield
    finally:
        vl.build_global_slices = original


@contextlib.contextmanager
def lead_off():
    """Force comfy's DEFAULT anchor-ring behaviour for one render.

    The lead now SHIPS (sampling.anchor_ring_factor, applied by anchor_ring_schedule via an
    instance patch), so the off cell is the override, not the on cell. factor == 1.0 means
    the frozen ring is re-noised to the current sigma every step — comfy/samplers.py:639 ->
    CONST.noise_scaling, since Krea2 (model_base.py:2456) has no scale_latent_inpaint
    override. That is exactly the pre-2026-08-12 behaviour.

    Overriding the FACTOR, never comfy: anchor_ring_schedule patches the model instance, and
    an instance attribute shadows a class attribute, so a class-level patch would be silently
    ignored. anchor_ring_factor is resolved through the module at call time — one swap point.
    """
    from context_anchored_tile_refine import sampling

    saved = sampling.anchor_ring_factor
    sampling.anchor_ring_factor = lambda r: 1.0
    try:
        yield
    finally:
        sampling.anchor_ring_factor = saved


# ------------------------------------------------------------------ render

def build_settings(cell, state, sigma0):
    scene = state.scene
    return {
        "run_label": f"{scene.key}/{cell.name}",
        "image_label": scene.key,
        "combination": cell.name,
        "vl_slicing": cell.vl, "captions": cell.captions, "lead": cell.lead,
        "vl_scale": cell.vl_scale, "vl_scale_scope": cell.vl_scope if cell.vl_scale != 1.0 else "",
        "conditioning": cell.conditioning,
        "note": cell.note,
        "scene_title": scene.title,
        "positive_prompt": scene.positive if cell.conditioning == "text" else "",
        "negative_prompt": scene.negative,
        "gen_prompt": scene.positive,
        "caption_instruction": cell.caption_instruction if cell.captions else "",
        "caption_max_tokens": cell.caption_max_tokens if cell.captions else 0,
        "caption_cleaned": cell.caption_clean if cell.captions else False,
        "tile_captions": state.captions.get(cell.caption_key, []) if cell.captions else [],
        "seed": cell.seed, "sampler": REFINE_SAMPLER, "scheduler": REFINE_SCHEDULER,
        "steps": REFINE_STEPS, "denoise": cell.denoise, "cfg": CFG, "sigma_start": sigma0,
        "upscale_by": UPSCALE_BY,
        "max_tile_width": MAX_TILE, "max_tile_height": MAX_TILE,
        "context_anchor": CONTEXT_ANCHOR, "context_overlap": CONTEXT_OVERLAP,
        "gen": {"seed": scene.gen_seed, "steps": GEN_STEPS, "sampler": GEN_SAMPLER,
                "scheduler": GEN_SCHEDULER, "cfg": CFG,
                "width": GEN_WIDTH, "height": GEN_HEIGHT},
        "unet": UNET_NAME, "clip": CLIP_NAME, "clip_type": CLIP_TYPE, "vae": VAE_NAME,
        "upscale_model": UPSCALE_MODEL_NAME, "guider": "CFGGuider",
        "harness": "tests-AB/run_ab_matrix.py",
        "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def render(cell, state, model, vae, clip):
    """One cell end to end, exactly as ContextAnchoredTileUpscaleVL wires it (minus the
    upscale, which already ran): fresh guider/sigmas/sampler/noise, then refine_image."""
    import comfy.samplers

    from context_anchored_tile_refine import sampling, upscale

    # With ANY per-tile conditioning the guider's positive is a placeholder the pre-pass
    # overwrites; only the all-text cell keeps the scene prompt as the real positive.
    per_tile = cell.conditioning != "text"
    positive = state.empty if per_tile else state.pos_text
    guider = upscale.build_guider(model, positive, state.negative, CFG)
    sigmas = upscale.build_sigmas(model, REFINE_SCHEDULER, REFINE_STEPS, cell.denoise)
    sampler = comfy.samplers.sampler_object(REFINE_SAMPLER)
    noise = upscale.Noise_RandomNoise(cell.seed)
    sigma0 = float(sigmas[0])

    if cell.conditioning == "vl+captions":
        hook = override_tile_positives(
            state.slice_conds[(cell.vl_scale, cell.vl_scope, cell.caption_key)], state.tiles)
    elif cell.conditioning == "captions":
        hook = override_tile_positives(state.caption_conds[cell.caption_key], state.tiles)
    else:
        hook = contextlib.nullcontext()          # VL-only takes the production pre-pass
    ring = contextlib.nullcontext() if cell.lead else lead_off()

    with hook, ring, ab_models.VramProbe() as probe, torch.inference_mode():
        result = sampling.refine_image(
            state.canvas, guider, sampler, sigmas, vae, noise,
            max_tile_width=MAX_TILE, max_tile_height=MAX_TILE,
            context_anchor=CONTEXT_ANCHOR, context_overlap=CONTEXT_OVERLAP,
            vl_clip=clip if per_tile else None,
        )

    ab_models.require_image_shape(result, state.canvas.shape[1], state.canvas.shape[2],
                                  f"refine {cell.scene}/{cell.name}")
    destination = output_path(cell)
    written = ab_models.save_png(destination, result.cpu(), build_settings(cell, state, sigma0))
    print(f"[render]  {cell.scene:<10} {cell.name:<34} d{cell.denoise:.2f} s{cell.seed:<5} "
          f"{probe}  -> {destination.parent.name}/{destination.name} {written[0]}x{written[1]}")


# ------------------------------------------------------------------ plan / index

def print_plan(selected):
    print("Knobs (three independent binary options => 2^3 = 8 base configurations):")
    print("  VL slicing  per-tile row slices of one whole-canvas vision encode")
    print("  Captions    per-tile VLM caption of that tile's crop, as plain text")
    print("  Lead        shipped anchor-ring schedule; off = comfy's re-noised ring")
    print()
    chosen = set(selected)
    for scene in SCENES:
        cells = [c for c in PLAN if c.scene == scene.key]
        if not cells:
            continue
        print(f"{scene_dir(scene.key).name}  ({scene.title})")
        print(f"   {'file':<36} {'VL':<4} {'Cap':<4} {'Lead':<5} {'d':<5} {'seed':<5} what it is")
        for cell in cells:
            mark = "*" if cell in chosen else " "
            on = ("x", "-")
            vl = f"x{cell.vl_scale:g}" if cell.vl and cell.vl_scale != 1.0 else on[not cell.vl]
            print(f"  {mark}{cell.name:<36} {vl:<4} {on[not cell.captions]:<4} "
                  f"{on[not cell.lead]:<5} {cell.denoise:<5.2f} {cell.seed:<5} {cell.note}")
        print()
    print(f"selected: {len(selected)} of {len(PLAN)} renders")


def write_index():
    """One index at the folder root. The scene subfolders stay images-only so a viewer
    steps straight through a set."""
    lines = [
        "# Conditioning matrix — Krea 2 VL node and base node",
        "",
        f"{len(PLAN)} renders across {len(SCENES)} scenes. Held identical everywhere: 512x288 base",
        f"(seed {GEN_SEED}, {GEN_SAMPLER} / {GEN_SCHEDULER} {GEN_STEPS} steps, cfg {CFG}) -> "
        f"{UPSCALE_MODEL_NAME} + lanczos to",
        f"1536x864 -> refine with {REFINE_SAMPLER} / {REFINE_SCHEDULER}, {REFINE_STEPS} steps, "
        f"cfg {CFG}, max_tile {MAX_TILE},",
        f"context_anchor {CONTEXT_ANCHOR}, context_overlap {CONTEXT_OVERLAP}. That solves to "
        f"{EXPECTED_GRID[0]}x{EXPECTED_GRID[1]} tiles — ONE vertical seam at x=768.",
        "Refine seed and denoise vary only where the file name says so.",
        "",
        "## The knobs",
        "",
        "| knob | what it does |",
        "|---|---|",
        "| VL slicing | per-tile row slices of ONE whole-canvas vision encode. Holds the original "
        "style, adds high-frequency detail, and every tile knows both its own content and the "
        "canvas's. |",
        "| Captions | per-tile VLM caption of that tile's crop, encoded as plain text. Names what "
        "is in each tile; also invents detail the source does not contain. |",
        "| Lead | the shipped anchor-ring schedule: the frozen context_anchor ring is presented "
        "ahead of the core so the core is drawn toward it. Stronger source adherence, less "
        "invented detail. OFF = comfy's default re-noised ring (what shipped before 2026-08-12). |",
        "",
        "`_v2` marks a round-2 follow-up. `VLx0.5` marks TBG-ETUR's row-strength trick applied to "
        "the VL encode's sequence tensor before slicing (`-vision` = grid rows only, caption left "
        "at full strength; `-whole` = the entire encode).",
        "",
    ]
    for scene in SCENES:
        cells = [c for c in PLAN if c.scene == scene.key]
        if not cells:
            continue
        prompt = scene.positive if len(scene.positive) < 200 else scene.positive[:197] + "..."
        lines += [f"## `{scene_dir(scene.key).name}` — {scene.title}", "",
                  f"> {prompt}", "",
                  "| file | VL | Cap | Lead | denoise | seed | what it is |",
                  "|---|---|---|---|---|---|---|"]
        for cell in cells:
            vl = f"x{cell.vl_scale:g}" if cell.vl and cell.vl_scale != 1.0 else ("x" if cell.vl else "")
            lines.append(f"| `{cell.name}.png` | {vl} | {'x' if cell.captions else ''} | "
                         f"{'x' if cell.lead else ''} | {cell.denoise:.2f} | {cell.seed} | "
                         f"{cell.note} |")
        lines.append("")
    lines += [
        "Each PNG carries its full settings and its tile captions in a `catr_ab` tEXt chunk.",
        "Regenerate any cell with `tests-AB/run_ab_matrix.py --only <file stem> --force`.",
        "",
    ]
    destination = OUTPUT_DIR / "_README.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(f"[index]   {destination}")


# ------------------------------------------------------------------ main

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scene", action="append", default=[], metavar="KEY",
                        help="render only these scenes (repeatable)")
    parser.add_argument("--only", action="append", default=[], metavar="NAME",
                        help="render only these cells, by file stem (repeatable)")
    parser.add_argument("--force", action="store_true", help="re-render existing outputs")
    parser.add_argument("--force-base", action="store_true", help="rebuild base + upscale caches")
    parser.add_argument("--force-captions", action="store_true", help="regenerate cached captions")
    parser.add_argument("--list", action="store_true", help="print the full plan and exit")
    args = parser.parse_args(argv)

    validate_plan()
    unknown_scenes = set(args.scene) - set(SCENES_BY_KEY)
    if unknown_scenes:
        raise SystemExit("unknown scene(s): {}".format(", ".join(sorted(unknown_scenes))))
    unknown_cells = set(args.only) - {c.name for c in PLAN}
    if unknown_cells:
        raise SystemExit("unknown cell(s): {}".format(", ".join(sorted(unknown_cells))))
    selected = [c for c in PLAN
                if (not args.scene or c.scene in args.scene)
                and (not args.only or c.name in args.only)]

    if args.list:
        print_plan(selected)
        return 0
    if not selected:
        raise SystemExit("nothing selected")

    root, note = ab_env.bootstrap()
    print(f"[env]     ComfyUI {ab_env.version(root)} at {root}  ({note})")
    print("[env]     torch {}  cuda {}  device {}".format(
        torch.__version__, torch.version.cuda,
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
    print_plan(selected)

    todo = [c for c in selected if args.force or not output_path(c).is_file()]
    for cell in selected:
        if cell not in todo:
            print(f"[render]  {cell.scene:<10} {cell.name:<34} exists, skipped (--force to redo)")
    scenes = [s for s in SCENES if any(c.scene == s.key for c in todo)]

    if not todo:
        write_index()
        print("[done]    nothing to render")
        return 0

    print(f"[clip]    loading {CLIP_NAME} ({CLIP_TYPE})")
    clip = ab_models.load_clip(CLIP_NAME, CLIP_TYPE)
    print(f"[unet]    loading {UNET_NAME}")
    model = ab_models.load_unet(UNET_NAME)
    vae = ab_models.load_vae(VAE_NAME)

    # Pixels first for every scene (base + upscale), then all conditioning, then all
    # renders: the upscale stage's free_gpu() evicts the UNET, so doing it once up front
    # costs one re-upload instead of one per scene.
    canvases = {s.key: stage_upscale(s, stage_base(s, model, clip, vae, args.force_base),
                                     args.force_base) for s in scenes}
    states = {s.key: build_state(s, canvases[s.key], clip,
                                 [c for c in todo if c.scene == s.key], args.force_captions)
              for s in scenes}

    for cell in todo:
        ab_models.clear_cache()
        render(cell, states[cell.scene], model, vae, clip)

    write_index()

    bad = []
    for cell in selected:
        destination = output_path(cell)
        if not destination.is_file():
            bad.append(f"{destination.name} MISSING")
            continue
        width, height = ab_models.png_size(destination)
        canvas = canvases.get(cell.scene)
        if canvas is not None and (width, height) != (canvas.shape[2], canvas.shape[1]):
            bad.append(f"{destination.name} is {width}x{height}")
    print(f"[done]    {len(selected) - len(bad)}/{len(selected)} selected renders present")
    if bad:
        raise SystemExit("[done]    FAILED: " + "; ".join(bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
