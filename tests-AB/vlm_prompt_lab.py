"""VLM prompt lab: search for the two caption instructions, no ComfyUI server, no sampling.

WHAT THIS DECIDES. Two instructions ship, one per surface:

    POSITION   used together WITH the VL slices. The vision rows already carry appearance
               positionally, so this must add only what they lack: which object a region
               belongs to and how much of that object the tile contains. Every extra word
               is mass competing with the rows.
    RICH       used on the captions-ONLY surface. Nothing else carries appearance there,
               so style, palette and lighting go in, once, at the front.

Both must be free of duplicates and free of text that is not description (headings, meta
narration, "Here is..."), because the caption is encoded as plain text and the DiT reads
every token of it as demand.

WHY A LAB AND NOT A RENDER. A render costs ~7 min of GPU and answers "did the picture get
better". Prompt shape is answerable in seconds from the text alone: an instruction the 4B
model cannot hold produces duplicates, markdown, or a repetition loop on the FIRST tile.
This tool runs the same instruction over tiles cut from unrelated canvases and scores the
text, so only instructions that already hold their shape ever reach a render.

    <venv-python> tests-AB/vlm_prompt_lab.py --round 1              # search, 2 tiles
    <venv-python> tests-AB/vlm_prompt_lab.py --round 1 --show P4    # full text of one cell
    <venv-python> tests-AB/vlm_prompt_lab.py --round 3 --tiles all  # validation sweep
    <venv-python> tests-AB/vlm_prompt_lab.py --list

Every (instruction, tile) answer is cached to JSON under cache/promptlab/, keyed by the
instruction text — so re-running a round to add ONE variant costs one generation, and the
reports stay reproducible without the GPU.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch

# ------------------------------------------------------------------ CONFIG

CACHE_DIR = Path(__file__).resolve().parent / "cache"
LAB_DIR = CACHE_DIR / "promptlab"

KREA2_ROOT = Path(r"C:\Users\Blake\ComfyUI-Installs\ComfyUI\ComfyUI")
CLIP_NAME = "qwen3-vl-4b-heretic_int8.safetensors"
CLIP_TYPE = "krea2"

# Generation is greedy (do_sample=False) so a variant's answer is a property of the
# instruction and the pixels, not of a draw. repetition_penalty mirrors the harness's
# settled 1.05 (core's TextGenerate uses 1.0 when sampling is off — a difference worth
# knowing about, and the reason `--reppen` exists).
REPETITION_PENALTY = 1.05


# ------------------------------------------------------------------ TILES
#
# Crops are taken from the cached UPSCALED canvases the matrix harness already rendered,
# so the lab sees exactly the pixel statistics a production tile sees (3x upscaled, not a
# base render). The 1536x864 canvases use the production layout at max_tile 1152 / anchor
# 256 / overlap 32 => two 1056x864 crops at x0=0 and x0=480. The 2304x3072 canvases are
# sliced arbitrarily; their only job is content variety.

@dataclasses.dataclass(frozen=True)
class Tile:
    key: str
    canvas: str                     # cache filename prefix
    rect: tuple                     # (x0, y0, x1, y1)
    what: str                       # what the crop contains, for reading the report


TILES = [
    Tile("face-L", "face_upscale_1536x864", (0, 0, 1056, 864),
         "dark-fantasy woman, face fills the left, forest behind"),
    Tile("face-R", "face_upscale_1536x864", (480, 0, 1536, 864),
         "the SAME canvas as face-L, tile 1's crop — holds the red moon at its left edge AND "
         "a small blue-white orb in red nebula mid-frame; the crop whose rich caption said "
         "'Two massive, glowing red moons' on 2026-08-13"),
    Tile("market-R", "market_upscale_1536x864", (480, 0, 1536, 864),
         "renaissance market, many small objects and figures"),
    Tile("city-L", "cybercity_upscale_1536x864", (0, 0, 1056, 864),
         "cyberpunk street, signage and reflections"),
    Tile("portrait-R", "portrait_upscale_1536x864", (480, 0, 1536, 864),
         "close-up photo of a man, detailed background"),
    Tile("skin-macro", "skin-macro_upscale_2304x3072", (576, 960, 1728, 2112),
         "macro skin texture — almost no nameable objects"),
    Tile("night-sky", "nightsky_upscale_2304x3072", (0, 0, 1152, 1152),
         "dark sky, very few objects — the empty-scene case"),
    Tile("bokeh", "bokeh-night_upscale_2304x3072", (1152, 1920, 2304, 3072),
         "out-of-focus night lights — the ambiguous-content case"),
    Tile("black-rim", "black-rim_upscale_2304x3072", (576, 0, 1728, 1152),
         "dark rim-lit subject — near-black field with one lit edge"),
]

TILE_SETS = {
    "search": ("face-L", "market-R"),
    "core": ("face-L", "market-R", "city-L", "portrait-R"),
    "all": tuple(t.key for t in TILES),
}

# Quantity words attached to objects. The owner's hypothesis on 2026-08-13: a wording that
# makes the model enumerate what is in a region may be what produces a COUNT, and a count
# it cannot support is how "a red moon plus a small blue orb" became "two red moons". This
# turns that from one anecdote into a number over every tile.
_NUMERIC_WORDS = (
    " two ", " three ", " four ", " five ", " six ", " several ", " multiple ", " numerous ",
    " pair of ", " both ", " a couple ", " dozens ", " many ",
)


# ------------------------------------------------------------------ VARIANTS

@dataclasses.dataclass(frozen=True)
class Variant:
    key: str
    kind: str                       # "position" | "rich"
    parse: str                      # how to cut the answer into items — see split_items
    instruction: str
    note: str = ""
    max_tokens: int = 1024          # thinking tokens count against this
    thinking: bool = True

    @property
    def digest(self):
        payload = json.dumps([self.instruction, self.parse, self.max_tokens,
                              self.thinking, CLIP_NAME, REPETITION_PENALTY])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


# --- round 1: which OUTPUT FORMAT can a 4B model hold at all? ---------------------------
# The owner's finding is the seed: a $variable template reads as code and produces the
# right shape, but ran away into a 50-item loop on trees; brackets did not read as code at
# all. So round 1 varies the format and holds the content request nearly constant. Every
# instruction says what IS wanted (no "do not"), gives no word limit, and carries no
# example — an example was echoed verbatim as content on 2026-08-13.

_POS_CONTENT = "name the thing, where it is in the frame, and how much of it is visible"

ROUND_1 = [
    Variant("P1-template", "position", "commas",
            "List each distinct thing in this image in the following format: "
            "$thing $visibility $position, $thing $visibility $position. "
            "Name each thing once.",
            note="owner's template, minus the '$thing next to $thing' clause that looped"),
    Variant("P2-json", "position", "json",
            "List the distinct things in this image as a JSON array of strings. "
            f"In each string, {_POS_CONTENT}.",
            note="code register — the strongest structure signal available"),
    Variant("P3-numbered", "position", "numbered",
            f"List the distinct things in this image. Number each line. On each line, {_POS_CONTENT}.",
            note="the format an instruct model is most drilled on"),
    Variant("P4-lines", "position", "lines",
            f"List the distinct things in this image, one per line. On each line, {_POS_CONTENT}.",
            note="line-per-item, joined with commas afterwards in code"),
    Variant("P5-pipes", "position", "pipes",
            "List the distinct things in this image, separated by the | character. "
            f"For each one, {_POS_CONTENT}.",
            note="a separator the model never writes inside a phrase"),
    Variant("P6-csv", "position", "csv",
            "List the distinct things in this image as CSV with the columns "
            "thing, position, visibility. One row per thing.",
            note="code register with the three facts as named columns"),
    Variant("P7-commas", "position", "commas",
            f"Name each distinct thing in this image. For each one, {_POS_CONTENT}. "
            "Separate each one with a comma.",
            note="the plain ask that mixed commas everywhere — kept as the control"),
    Variant("P8-inventory", "position", "lines",
            "Inventory the things visible in this image, one per line: what it is, where "
            "it sits in the frame, and whether the frame cuts it off.",
            note="different register — an inventory, not a description"),
    Variant("P9-shortphrase", "position", "lines",
            "List the distinct things in this image, one per line, each a short phrase "
            "naming the thing, its position in the frame, and how much of it shows.",
            note="caps the ITEM instead of the answer — a short phrase has no room for a comma"),
    Variant("P10-annot", "position", "lines",
            "Write region annotations for this image, one per line, each naming an object, "
            "its location in the frame, and whether it is fully or partly visible.",
            note="annotation register — closest to how a VLM's grounding data is written"),
    Variant("P4-nothink", "position", "lines",
            f"List the distinct things in this image, one per line. On each line, {_POS_CONTENT}.",
            note="P4 with thinking OFF — the harness's current setting, as the control",
            thinking=False, max_tokens=320),

    Variant("R1-numbered", "rich", "numbered",
            "Describe this image. First line: the overall style, palette and lighting. "
            "Then one numbered line per distinct thing, giving its position in the frame, "
            "what it is, and its colour and material.",
            note="rich = the captions-only surface; style leads, then the same item shape"),
    Variant("R2-json", "rich", "json",
            "Describe this image as a JSON array of strings. The first string is the "
            "overall style, palette and lighting. Each string after it describes one "
            "distinct thing: its position in the frame, what it is, its colour and material.",
            note="code register for the rich surface"),
    Variant("R3-lines", "rich", "lines",
            "Describe this image as a list, one line each. Start with the overall style, "
            "palette and lighting. Then one line per distinct thing: its position in the "
            "frame, what it is, and its colour and material.",
            note="line-per-item rich"),
    Variant("R4-prose", "rich", "prose",
            "Describe this image in one paragraph. Cover the overall style, palette and "
            "lighting first, then each distinct thing once, with its position in the "
            "frame, its colour and its material.",
            note="prose — what the shipped descriptive instruction produces, made unique"),
    Variant("R5-pipes", "rich", "pipes",
            "Describe this image as entries separated by the | character. The first entry "
            "is the overall style, palette and lighting. Each entry after it covers one "
            "distinct thing: its position in the frame, what it is, its colour and material.",
            note="pipe-delimited rich"),
]


# --- round 2: the failure round 1 found is TERMINATION, not format --------------------
# Round 1 result: 7 of the first 9 cells ran to the 1024-token cap without ever emitting a
# stop token. The format was fine in every one of them — the model simply never stopped
# enumerating. Two distinct runaways, needing two distinct bounds:
#
#   PART decomposition   face-L: "Woman's hair / eyes / lips / neck / shoulder / ear /
#                        cheek / nose / brow / eyelashes..." — one object mined for parts.
#   INSTANCE repetition  market-R: "market stall, market stall, market stall..." x30 —
#                        many identical objects each getting their own entry.
#
# So round 2 holds the format at line-per-item (the only one that terminated cleanly) and
# varies the BOUND. Every bound is phrased as a thing to do, never as a thing to avoid,
# and none of them limits the ANSWER length — the 2026-08-13 finding was that an
# answer-level word cap makes the model narrate its own correction into the caption.
# Q8/S3 are the deliberate "something totally different" arm: naming REGIONS instead of
# objects has a natural stop built into the question, so termination stops being a
# property of how busy the picture is.

_Q_CONTENT = "name the thing, where it is in the frame, and how much of it is visible"
_WHOLE = "Name whole objects and count repeated objects as one entry."

ROUND_2 = [
    Variant("Q1-group", "position", "lines",
            f"List the main things in this image, one per line. On each line, {_Q_CONTENT}. "
            f"{_WHOLE}",
            note="grouping clause alone — aimed straight at both runaways", max_tokens=512),
    Variant("Q2-count8", "position", "lines",
            f"List the eight most prominent things in this image, one per line. On each line, {_Q_CONTENT}.",
            note="a hard count: bounds the ANSWER without bounding its words", max_tokens=512),
    Variant("Q3-short", "position", "lines",
            "List the main things in this image, one per line, each a short phrase naming "
            f"the thing, its position in the frame, and how much of it shows. {_WHOLE}",
            note="grouping + a short-phrase item: no room inside an item for a comma",
            max_tokens=512),
    Variant("Q4-fields", "position", "fields",
            "For each main thing in this image write one line: thing | position in the "
            f"frame | how much is visible. {_WHOLE}",
            note="named fields on one line — code register with a per-line stop",
            max_tokens=512),
    Variant("Q5-json8", "position", "json",
            "List the eight most prominent things in this image as a JSON array of "
            f"strings. In each string, {_Q_CONTENT}.",
            note="count bound inside the code register", max_tokens=512),
    Variant("Q6-notice", "position", "lines",
            f"List what a viewer notices first in this image, one per line. On each line, {_Q_CONTENT}.",
            note="semantic bound instead of a count — does 'first' terminate?", max_tokens=512),
    Variant("Q7-largest", "position", "lines",
            f"List the largest things in this image, one per line. On each line, {_Q_CONTENT}. {_WHOLE}",
            note="size bound — the parts of a face are small, so this should kill P-decomposition",
            max_tokens=512),
    Variant("Q8-regions", "position", "lines",
            "Say what fills each part of this image, one line per part: the left, the "
            "centre, the right, the top and the bottom. On each line name what is there "
            "and how much of it is visible.",
            note="TOTALLY DIFFERENT: five regions is a stop condition the picture cannot "
                 "overrun, and region-to-content is exactly what the vision rows lack",
            max_tokens=512),

    # Round 1 produced the two shortest items of the whole round from these two registers:
    # P6-csv wrote 3-word fields (21 words for 13 items) and P9's short-phrase wording read
    # best as English. Both ran away on market-R, which is the one thing round 2 adds.
    Variant("Q9-csv", "position", "csv",
            "List the main things in this image as rows of thing,position,visibility. "
            f"One row per thing. {_WHOLE}",
            note="round 1's tightest items (3 words each) with the bound it lacked",
            max_tokens=512),
    Variant("Q10-shortgroup", "position", "lines",
            "List the main things in this image, one per line, each a short phrase naming "
            f"the thing and its position in the frame. {_WHOLE}",
            note="P9's wording, bounded, and with the visibility field dropped — three "
                 "facts per line is what pushed items past a comma",
            max_tokens=512),

    Variant("S1-group", "rich", "lines",
            "Describe this image as a list, one line each. Start with the overall style, "
            "palette and lighting. Then one line per main thing: its position in the "
            f"frame, what it is, and its colour and material. {_WHOLE}",
            note="R3 plus the grouping clause", max_tokens=768),
    Variant("S2-count8", "rich", "lines",
            "Describe this image as a list, one line each. Start with the overall style, "
            "palette and lighting. Then the eight most prominent things, one line each, "
            "with its position in the frame, what it is, and its colour and material.",
            note="R3 plus a hard count", max_tokens=768),
    Variant("S3-regions", "rich", "lines",
            "Describe this image, one line per part. Start with the overall style, palette "
            "and lighting. Then say what fills the left, the centre, the right, the top "
            "and the bottom, naming what is there, its colour and its material.",
            note="the region framing on the rich surface", max_tokens=768),
]


# --- round 3: the text-to-image-prompt register, and the accepted-comma fallback -------
# The owner's question: is the model trained to recognise "a prompt for an image
# generator", and does that wording buy anything? It should, mechanically — SD-style
# prompt data IS natively a comma-delimited list of short phrases, which is the exact
# shape being asked for, so the register may hand over comma discipline for free. The
# known cost is the other half of that training set: prompt data also carries demand
# words ("masterpiece, highly detailed, 8k"), and AB37-40 settled that ungrounded demand
# is what grows phantom objects. `demand` in the score table counts them.
#
# U4/U5 are the fallback the owner named: stop fighting for comma-only splitting, accept
# one SENTENCE per object, and let the period carry the split. Round 2's Q1 already
# produces this shape — U4 asks for it explicitly to see whether naming it helps or hurts.

_DEMAND_WORDS = (
    "masterpiece", "highly detailed", "8k", "4k", "ultra", "best quality", "high quality",
    "award", "trending", "artstation", "photorealistic", "hyperrealistic", "intricate",
    "stunning", "beautiful", "sharp focus", "bokeh", "depth of field", "soft focus",
    "blurred", "blurry", "shallow depth",
)

ROUND_3 = [
    Variant("U1-prompt", "position", "commas",
            "Write a text to image prompt for this image as short comma separated "
            f"phrases. Each phrase names one thing and where it is in the frame. {_WHOLE}",
            note="the owner's question: does the prompt register hand over comma discipline?",
            max_tokens=512),
    Variant("U2-tags", "position", "commas",
            "Describe this image as image generation tags, comma separated. Each tag "
            f"names one thing and its position in the frame. {_WHOLE}",
            note="tag register — the terser half of the same training data", max_tokens=512),
    Variant("U7-part", "position", "lines",
            "For each main thing in this image write one short line naming the part of it "
            f"that is in the frame and where that part sits. {_WHOLE}",
            note="folds visibility INTO the thing's name, so a line carries two facts and "
                 "needs one comma instead of two — round 2's Q10 shape with the visibility "
                 "field the owner asked for put back",
            max_tokens=512),
    Variant("U4-sentence", "position", "lines",
            "Write one sentence for each main thing in this image. Each sentence names "
            f"the thing, where it is in the frame, and how much of it is visible. {_WHOLE}",
            note="the accepted-comma fallback: the PERIOD carries the split, so the model "
                 "never has to hold comma discipline at all",
            max_tokens=512),

    Variant("U3-prompt", "rich", "commas",
            "Write a text to image prompt for this image as short comma separated "
            "phrases. Start with the style, palette and lighting. Then one phrase per "
            f"thing giving what it is, where it is in the frame, its colour and material. {_WHOLE}",
            note="prompt register on the captions-only surface, where a prompt is what is "
                 "actually wanted", max_tokens=768),
    Variant("U5-sentence", "rich", "lines",
            "Write one sentence for the overall style, palette and lighting of this image, "
            "then one sentence for each main thing giving where it is in the frame, what "
            f"it is, its colour and its material. {_WHOLE}",
            note="the accepted-comma fallback on the rich surface", max_tokens=768),
    Variant("U6-material", "rich", "lines",
            "Describe this image as a list, one line each. Start with the overall style, "
            "palette and lighting. Then one line per main thing: where it is in the frame, "
            f"what it is, and what its surface is made of. {_WHOLE}",
            note="'what its surface is made of' as the POSITIVE replacement for banning "
                 "focus/blur talk — round 1's rich cells all volunteered 'soft focus' and "
                 "'shallow depth of field', which are demands to destroy detail in a refine",
            max_tokens=768),
]


# --- round 4: validation of the survivors across unrelated tiles ----------------------
# Rounds 1-3 searched on two tiles. Round 4 asks the only remaining question: does the
# winning wording hold its shape on content it has never seen — a macro skin crop with no
# nameable objects, an almost-empty night sky, out-of-focus bokeh, a video frame.
#
# The three carried forward are byte-identical to their round-2 originals (same
# instruction string => same cache key), so their face-L / market-R cells are re-read from
# disk and only the six new tiles cost GPU time.
#
# Round 3 settled the owner's prompt-register question in the negative and it is not
# carried forward: asking for "a text to image prompt" or "image generation tags" fires
# the tag-spam half of that training data — U2-tags repeated one 9-tag cycle until the
# cap (115 duplicate items, position coverage 0.00) and U3-prompt leaked "material:" /
# "texture:" schema words as content. Naming a SENTENCE (U4) also backfired: it licenses
# 32-40 word sentences, where round 2's "short phrase" wording holds items to 6-11 words.

ROUND_4 = [
    # verbatim Q3-short — thing, position, visibility: three facts, two commas
    Variant("W1-short", "position", "lines",
            "List the main things in this image, one per line, each a short phrase naming "
            f"the thing, its position in the frame, and how much of it shows. {_WHOLE}",
            note="round 2's Q3: the shape that keeps the visibility fact", max_tokens=512),
    # verbatim Q10-shortgroup — thing, position: two facts, one comma
    Variant("W2-shortgroup", "position", "lines",
            "List the main things in this image, one per line, each a short phrase naming "
            f"the thing and its position in the frame. {_WHOLE}",
            note="round 2's Q10: the tightest wording, visibility dropped", max_tokens=512),
    # verbatim S3-regions
    Variant("W3-regions", "rich", "lines",
            "Describe this image, one line per part. Start with the overall style, palette "
            "and lighting. Then say what fills the left, the centre, the right, the top "
            "and the bottom, naming what is there, its colour and its material.",
            note="round 2's S3: five regions is a stop the picture cannot overrun",
            max_tokens=768),
    Variant("W4-regions-short", "rich", "lines",
            "Describe this image, one short line per part. Start with the overall style, "
            "palette and lighting. Then say what fills the left, the centre, the right, "
            "the top and the bottom, naming what is there, its colour and what its "
            "surface is made of.",
            note="W3 with round 2's 'short' lever applied, and 'what its surface is made "
                 "of' in place of 'its material' — the positive way to crowd out the "
                 "'soft focus' / 'shallow depth of field' talk every round-1 rich cell "
                 "volunteered, which is a demand to destroy the detail a refine adds",
            max_tokens=768),
]


# --- round 5: the three defects the 8-tile validation found ----------------------------
#   1. W1 (thing + position + visibility) RUNS AWAY on a second busy scene: city-L gave
#      334 words and 22 duplicate items. W2 (thing + position) held there at 49 words, so
#      the third fact per line is what costs the stop token.
#   2. W2 loses POSITION on that same tile (coverage 0.29): "Central skyscraper with blue
#      vertical lighting" names a thing and no place. Dropping the visibility fact made
#      room for a longer NAME instead of a position.
#   3. The region framing repeats itself on a uniform crop: skin-macro got five lines whose
#      bodies are identical ("smooth, matte beige material, likely leather" x5).
#
# X1/X3 add a count bound as an upper limit rather than a target, so a featureless crop is
# still allowed to answer with one line. X2 carries the region framing over to the position
# surface: it is the only structure in the whole search whose stop condition is set by the
# QUESTION rather than by the picture, and it is inherently positional, which is the one
# thing the vision rows cannot supply.

ROUND_5 = [
    Variant("X1-short8", "position", "lines",
            "List up to eight main things in this image, one per line, each a short phrase "
            "naming the thing, its position in the frame, and how much of it shows. "
            f"{_WHOLE}",
            note="W1 with a ceiling, not a target — the runaway fix that still lets a "
                 "featureless crop answer in one line", max_tokens=512),
    Variant("X2-regions", "position", "lines",
            "Say what is in each part of this image, one short line per part: the left, "
            "the centre, the right, the top and the bottom. On each line name what is "
            "there and how much of it is in the frame.",
            note="the region framing on the POSITION surface — a stop condition set by the "
                 "question, and positional by construction", max_tokens=512),
    Variant("X3-group8", "position", "lines",
            "List up to eight main things in this image, one per line, each a short phrase "
            f"naming the thing and where it sits in the frame. {_WHOLE}",
            note="W2 with a ceiling and 'where it sits' in place of 'its position', to see "
                 "whether the position fact survives on the tile that dropped it",
            max_tokens=512),

    Variant("X4-regions-own", "rich", "lines",
            "Describe this image, one short line per part. Start with the overall style, "
            "palette and lighting. Then say what fills the left, the centre, the right, "
            "the top and the bottom, giving each part its own description with what is "
            "there, its colour and what its surface is made of.",
            note="W4 plus 'giving each part its own description' — the positive form of "
                 "the anti-repeat clause, aimed at the uniform-crop failure",
            max_tokens=768),
]


# --- round 6: is the count doing the work, or is "where it sits"? -----------------------
# Round 5's X3 fixed BOTH round-4 defects at once, so it is not yet known which half did
# it. Two changes went in together: a count ceiling, and "where it sits in the frame" in
# place of "its position in the frame". The distinction matters because the ceiling has a
# cost of its own — on the featureless skin crop both X1 and X3 read "up to eight" as a
# QUOTA and padded to exactly eight entries, the last of which was "Minimalist composition
# with no distinct objects": meta-commentary about the picture, not content in it.
#
# Y1 is X3 with the count removed. If it holds position coverage on city-L (the tile that
# collapsed to 0.29 under the old wording) then "where it sits" was the fix, the ceiling is
# unnecessary, and a sparse crop gets to answer with as few lines as it has things.
# Y2 puts the visibility fact back the cheap way — inside the thing's NAME rather than as a
# third comma-separated field, which is what made round 4's W1 run away.

ROUND_6 = [
    Variant("Y1-nocount", "position", "lines",
            "List the main things in this image, one per line, each a short phrase naming "
            f"the thing and where it sits in the frame. {_WHOLE}",
            note="X3 minus the count — isolates which half of round 5's fix mattered",
            max_tokens=512),
    Variant("Y2-partname", "position", "lines",
            "List the main things in this image, one per line, each a short phrase naming "
            "the part of the thing that is in the frame and where it sits. "
            f"{_WHOLE}",
            note="the visibility fact folded into the NAME, so a line still has one comma",
            max_tokens=512),
]


# --- round 7: does the wording induce a COUNT? -----------------------------------------
# The face render's rich caption said "Top: Two massive, glowing red moons - one large and
# full, the other smaller and partially obscured". The crop really does hold two bright
# objects: the red moon at its left edge, and a small blue-white orb in red nebula
# mid-frame. So the defect is an IDENTIFICATION folded into a COUNT — the orb got the
# moon's label, and the plural is what reached the DiT as a demand.
#
# Two candidate causes, and they point at opposite fixes:
#   OWNER'S    the wording leads the witness. `say what fills the left, the centre, the
#              right, the top and the bottom` makes every region a slot that must be
#              filled, so a slot holding two bright things gets enumerated and the second
#              one needs a label. Predicts counts track the REGION framing.
#   MINE       the rich prompt lacks `Name whole objects and count repeated objects as one
#              entry.`, which the position prompt has and which would have collapsed the
#              pair. Predicts counts track the CLAUSE. The owner's objection to this is
#              itself well-founded: it would put the word "count" into a prompt that has
#              none, and priming a model to count is a way to get counts.
#
# The four arms separate them. AA/AB are the shipped pair as-is (cached, free); AC adds the
# clause to the rich prompt; AD REMOVES it from the position prompt — which also re-tests
# whether the position prompt still terminates without the clause that originally stopped
# its runaways, so a "just delete it" outcome cannot slip through unmeasured.
_SETTLED_POSITION = (
    "List up to eight main things in this image, one per line, each a short phrase naming "
    "the thing, its position in the frame, and how much of it shows.")
_SETTLED_RICH = (
    "Describe this image, one short line per part. Start with the overall style, palette "
    "and lighting. Then say what fills the left, the centre, the right, the top and the "
    "bottom, giving each part its own description with what is there, its colour and what "
    "its surface is made of.")

ROUND_7 = [
    Variant("AA-pos-shipped", "position", "lines", f"{_SETTLED_POSITION} {_WHOLE}",
            note="the shipped position prompt, verbatim (== X1-short8, reads from cache)",
            max_tokens=512),
    Variant("AD-pos-noclause", "position", "lines", _SETTLED_POSITION,
            note="the shipped position prompt with the clause REMOVED — the owner's "
                 "hypothesis, and a re-test of whether it still terminates without it",
            max_tokens=512),
    Variant("AB-rich-shipped", "rich", "lines", _SETTLED_RICH,
            note="the shipped rich prompt, verbatim (== X4-regions-own, reads from cache)",
            max_tokens=768),
    Variant("AC-rich-clause", "rich", "lines", f"{_SETTLED_RICH} {_WHOLE}",
            note="the shipped rich prompt with the clause ADDED", max_tokens=768),
]

ROUNDS = {1: ROUND_1, 2: ROUND_2, 3: ROUND_3, 4: ROUND_4, 5: ROUND_5, 6: ROUND_6,
          7: ROUND_7}


# ------------------------------------------------------------------ PARSING

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_JSON_ARRAY = re.compile(r"\[.*\]", re.DOTALL)


def strip_thinking(raw):
    """Return (answer, closed). `closed` is False when the reasoning block never ended —
    the answer was cut off inside it, which is a max_tokens failure, not a wording one."""
    if "<think>" not in raw:
        return raw.strip(), True
    closed = "</think>" in raw
    body = _THINK_BLOCK.sub("", raw)
    if "</think>" in body:                      # unclosed leader: keep what follows the last one
        body = body.rsplit("</think>", 1)[-1]
    body = re.sub(r"</?think>", "", body)
    return body.strip(), closed


def split_items(answer, parse):
    """Cut an answer into the items the instruction asked for. Never rewrites an item."""
    text = _FENCE.sub("", answer.strip())

    if parse == "json":
        match = _JSON_ARRAY.search(text)
        if match is None:
            return []
        try:
            loaded = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        return [str(item).strip() for item in loaded if str(item).strip()]

    if parse == "csv":
        rows = []
        for line in text.splitlines():
            fields = [f.strip() for f in line.split(",")]
            if len(fields) < 2 or line.lower().startswith("thing,"):
                continue                        # header row or prose line
            rows.append(" ".join(f for f in fields if f))
        return rows

    if parse == "fields":
        # one item per LINE, its fields pipe-separated: fold the pipes to spaces so the
        # item is a phrase with no punctuation of its own.
        return [item for item in (" ".join(f.strip() for f in _BULLET.sub("", line).split("|") if f.strip())
                                  for line in text.splitlines()) if item]

    if parse == "pipes":
        raw_items = text.replace("\n", " ").split("|")
    elif parse == "commas":
        raw_items = text.replace("\n", " ").split(",")
    elif parse == "prose":
        raw_items = re.split(r"(?<=[.;])\s+", text)
    else:                                       # "lines" and "numbered"
        raw_items = text.splitlines()

    return [item for item in (_BULLET.sub("", i).strip().strip(".") for i in raw_items) if item]


# ------------------------------------------------------------------ SCORING

_POSITION_WORDS = (
    "left", "right", "center", "centre", "top", "bottom", "upper", "lower", "middle",
    "foreground", "background", "corner", "edge", "behind", "front", "beside", "above",
    "below", "next to", "across", "side", "mid-", "midground",
)
_VISIBILITY_WORDS = (
    "full", "whole", "entire", "partial", "partly", "part of", "half", "cut", "crop",
    "visible", "shows", "showing", "off-screen", "off screen", "out of frame", "edge",
)
_STYLE_WORDS = (
    "style", "palette", "lighting", "lit", "tone", "photo", "painting", "render",
    "cinematic", "colour", "color", "warm", "cool", "muted", "saturated", "contrast",
)
# An answer that opens with any of these is talking to the reader, not describing pixels.
_META_OPENERS = (
    "here", "sure", "okay", "ok,", "certainly", "of course", "let me", "i ", "i'll",
    "wait", "based on", "answer", "output", "result", "response", "note", "this is a",
    "looking at", "in this image", "the image shows", "the image is a", "below",
)
_MARKUP = ("**", "##", "```", "__", "|---")
# Field NAMES written out as content. Round 2's Q9-csv bought its zero internal commas by
# emitting "Woman, position: foreground right, visibility: fully visible" — the schema
# leaked into the caption, which is exactly the non-description text that must not reach
# the DiT. Region words ("Left:", "Top:") are content and deliberately absent here.
# Deliberately NOT here: "style:", "palette:", "lighting:", "left:", "top:" — those are the
# region framing's own line prefixes, structure the reader wants, not schema echo.
_LABEL_LEAK = ("position:", "visibility:", "thing:", "object:", "colour:", "color:",
               "material:", "texture:", "location:", "name:")
# Leading structure to strip before asking "did this line say something new". The region
# framing's failure mode on a uniform crop is five lines with different prefixes and one
# identical body ("The left is filled with smooth matte beige material..." x5), which the
# plain line-level dedup cannot see.
_PREFIX = re.compile(r"^(?:the\s+)?(?:left|right|centre|center|top|bottom|foreground|"
                     r"background|middle|style|palette|lighting)\b[^a-zA-Z]*"
                     r"(?:is\s+filled\s+with\s+|is\s+|shows\s+|features\s+)?", re.IGNORECASE)


def _normalize(item):
    return " ".join(item.lower().strip().rstrip(".,;:").split())


def _hits(item, words):
    low = item.lower()
    return any(word in low for word in words)


def analyze(answer, items, kind, closed):
    """Everything measurable about one answer. No verdict here — see `verdict`."""
    norm = [_normalize(i) for i in items]
    counts = Counter(norm)
    words = len(answer.split())

    return {
        "words": words,
        "items": len(items),
        "dup": len(norm) - len(set(norm)),
        "maxrep": max(counts.values()) if counts else 0,
        "longest": max((len(i.split()) for i in items), default=0),
        "median": sorted(len(i.split()) for i in items)[len(items) // 2] if items else 0,
        # The owner's actual requirement: a comma may separate two statements and may not
        # appear inside one. Measured AFTER the parse, so a format whose fields are folded
        # to spaces (csv, fields) scores 0 by construction and a line-per-item format is
        # scored on the commas the model itself wrote.
        "inner_comma": sum("," in i for i in items),
        "markup": any(mark in answer for mark in _MARKUP),
        "meta": bool(norm) and norm[0].startswith(_META_OPENERS),
        "pos_cov": (sum(_hits(i, _POSITION_WORDS) for i in items) / len(items)) if items else 0.0,
        "vis_cov": (sum(_hits(i, _VISIBILITY_WORDS) for i in items) / len(items)) if items else 0.0,
        "style": _hits(answer, _STYLE_WORDS),
        # Quality/demand vocabulary. Two separate harms, both settled: an ungrounded
        # demand grows phantom objects (AB37-40), and "soft focus"/"blurred" is an
        # instruction to destroy the detail the refine exists to add.
        "demand": sum(answer.lower().count(word) for word in _DEMAND_WORDS),
        "label": sum(answer.lower().count(word) for word in _LABEL_LEAK),
        "body_dup": len(norm) - len({_PREFIX.sub("", i) for i in norm}),
        "numeric": sum(f" {answer.lower()} ".count(word) for word in _NUMERIC_WORDS),
        "truncated": not closed,
        "kind": kind,
    }


# Pass bars. POSITION is the tighter surface: it rides alongside 546 vision rows, so a long
# item is mass that competes with them. RICH must carry appearance, so it is allowed the
# words but not the repeats.
def verdict(stats):
    fails = []
    if stats["truncated"]:
        fails.append("truncated")
    if stats["markup"]:
        fails.append("markup")
    if stats["meta"]:
        fails.append("meta")
    if stats["dup"]:
        fails.append(f"dup{stats['dup']}")
    if stats["body_dup"] > stats["dup"]:
        fails.append(f"same{stats['body_dup']}")
    # Only a PARSE failure is a defect here. A macro skin crop or a black field honestly
    # contains one nameable thing, and both leading candidates answered it in one correct
    # line — scoring that as "too few items" would reject the right answer.
    if not stats["items"]:
        fails.append("no-items")
    if stats["demand"]:
        fails.append(f"demand{stats['demand']}")
    if stats["label"]:
        fails.append(f"label{stats['label']}")

    if stats["kind"] == "position":
        if stats["items"] > 25:
            fails.append("runaway")
        # `inner_comma` is REPORTED, not failed. The owner's decision on 2026-08-13, after
        # seeing that the only zero-comma register (csv) buys it with leaked field labels:
        # accept one statement per object with the PERIOD carrying the split, and let
        # commas sit inside a statement where they are ordinary English. The column stays
        # because it is still the cheapest read on whether items are phrases or sentences.
        if stats["longest"] > 9:
            fails.append(f"long{stats['longest']}")
        if stats["pos_cov"] < 0.8:
            fails.append(f"pos{stats['pos_cov']:.2f}")
        if stats["words"] > 90:
            fails.append(f"fat{stats['words']}")
    else:
        if not stats["style"]:
            fails.append("no-style")
        # Upper bar set from the two known points: the shipped descriptive instruction
        # produced 370-410 words (half a tile's sequence, the thing that made captions
        # fight the vision rows), and the settled positional instruction sat near 90. A
        # rich caption must carry appearance, so it is allowed the middle of that range.
        if not 40 <= stats["words"] <= 220:
            fails.append(f"len{stats['words']}")
        if stats["pos_cov"] < 0.5:
            fails.append(f"pos{stats['pos_cov']:.2f}")

    return "PASS" if not fails else " ".join(fails)


def canonical(items):
    """The form the caption would actually be encoded as: unique items, commas ONLY between
    them. This is the join the node would do — the model never has to hold comma discipline."""
    seen, kept = set(), []
    for item in items:
        key = _normalize(item)
        if key and key not in seen:
            seen.add(key)
            kept.append(item.strip().rstrip(".,;"))
    return ", ".join(kept)


# ------------------------------------------------------------------ GENERATION

def load_canvas(prefix):
    """An IMAGE canvas [B,H,W,3] from the harness cache. The same directory also holds
    latents and conditioning payloads (h3*_vlimg is a {cond, tags, layout} dict, h3*_base
    a {video, audio} latent pair), so the shape is checked here rather than failing later
    with an attribute error inside the crop."""
    matches = sorted(CACHE_DIR.glob(f"{prefix}_*.pt"))
    if not matches:
        raise SystemExit(f"No cached canvas matching {prefix}_*.pt in {CACHE_DIR}")
    canvas = torch.load(matches[0], map_location="cpu", weights_only=True)
    if not isinstance(canvas, torch.Tensor) or canvas.ndim != 4 or canvas.shape[-1] != 3:
        kind = type(canvas).__name__ if not isinstance(canvas, torch.Tensor) else tuple(canvas.shape)
        raise SystemExit(f"{matches[0].name} is not an image canvas [B,H,W,3] but {kind}")
    return canvas


def resample_for_vl(tile_pixels):
    """The caption input prep the node uses: area-resample a COPY of the crop to 384^2
    total pixels. Conditioning-side only; no sampled pixel is ever resampled."""
    import comfy.utils

    samples = tile_pixels.movedim(-1, 1)
    scale_by = math.sqrt(384 * 384 / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    resampled = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    return resampled.movedim(1, -1)[:, :, :, :3]


def tile_input(tile):
    canvas = load_canvas(tile.canvas)
    x0, y0, x1, y1 = tile.rect
    if y1 > canvas.shape[1] or x1 > canvas.shape[2]:
        raise SystemExit(f"{tile.key}: rect {tile.rect} does not fit canvas {tuple(canvas.shape)}")
    return resample_for_vl(canvas[:, y0:y1, x0:x1, :].float())


def generate(clip, vl_input, variant):
    """One greedy decode. Returns (raw_text, seconds)."""
    started = time.perf_counter()
    tokens = clip.tokenize(variant.instruction, images=[vl_input], thinking=variant.thinking)
    with torch.inference_mode():
        ids = clip.generate(tokens, do_sample=False, max_length=variant.max_tokens,
                            repetition_penalty=REPETITION_PENALTY)
        raw = clip.decode(ids)
    return raw, time.perf_counter() - started


def answer_for(clip_box, variant, tile, inputs, force):
    """Cached generation. `clip_box` is a one-slot list so the CLIP loads only if a cache
    miss actually needs it — a report over cached answers costs no VRAM at all."""
    cached = LAB_DIR / f"{tile.key}_{variant.key}_{variant.digest}.json"
    if not force:
        # The digest is the real key — the variant's own name is only a label. Carrying a
        # settled instruction into a later round under a new name must not re-spend GPU
        # time, so any file for this tile with this digest is a hit.
        hits = [cached] if cached.is_file() else sorted(LAB_DIR.glob(f"{tile.key}_*_{variant.digest}.json"))
        if hits:
            return json.loads(hits[0].read_text(encoding="utf-8"))

    if clip_box[0] is None:
        import ab_models
        print(f"[clip]  loading {CLIP_NAME}")
        clip_box[0] = ab_models.load_clip(CLIP_NAME, CLIP_TYPE)
    if tile.key not in inputs:
        inputs[tile.key] = tile_input(tile)

    raw, seconds = generate(clip_box[0], inputs[tile.key], variant)
    answer, closed = strip_thinking(raw)
    record = {"variant": variant.key, "tile": tile.key, "instruction": variant.instruction,
              "thinking": variant.thinking, "seconds": round(seconds, 1), "raw": raw,
              "answer": answer, "closed": closed}
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


# ------------------------------------------------------------------ REPORT

HEADER = (f"{'variant':<15} {'tile':<11} {'s':>5} {'wd':>4} {'it':>3} {'dup':>3} "
          f"{'rep':>3} {'sam':>3} {'ic':>3} {'lng':>3} {'dmd':>3} {'lbl':>3} {'num':>3} "
          f"{'pos':>4} {'vis':>4}  verdict")


def report(rows, out_path):
    """Print the table; write every full answer to a text file for reading."""
    print(f"\n{HEADER}\n{'-' * len(HEADER)}")
    lines = []
    for variant, tile, record, stats, call in rows:
        print(f"{variant.key:<15} {tile.key:<11} {record['seconds']:>5.0f} {stats['words']:>4} "
              f"{stats['items']:>3} {stats['dup']:>3} {stats['maxrep']:>3} "
              f"{stats['body_dup']:>3} "
              f"{stats['inner_comma']:>3} {stats['longest']:>3} {stats['demand']:>3} "
              f"{stats['label']:>3} {stats['numeric']:>3} "
              f"{stats['pos_cov']:>4.2f} {stats['vis_cov']:>4.2f}  {call}")
        lines.append(
            f"{'=' * 100}\n{variant.key}  [{variant.kind}/{variant.parse}]  tile={tile.key}"
            f"  thinking={variant.thinking}  {record['seconds']:.0f}s  -> {call}\n"
            f"note: {variant.note}\n"
            f"INSTRUCTION: {variant.instruction}\n{'-' * 100}\n{record['answer']}\n"
            f"{'-' * 100}\nCANONICAL: {canonical(split_items(record['answer'], variant.parse))}\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[report] full answers -> {out_path}")


def summarize(rows):
    """Per variant across its tiles: how many tiles it passed. Consistency is the point —
    a variant that passes one tile and loops on another is not a candidate."""
    by_variant = {}
    for variant, _tile, _record, _stats, call in rows:
        hit = by_variant.setdefault(variant.key, {"kind": variant.kind, "pass": 0, "n": 0,
                                                  "fails": Counter()})
        hit["n"] += 1
        if call == "PASS":
            hit["pass"] += 1
        else:
            hit["fails"].update(call.split())

    for kind in ("position", "rich"):
        print(f"\n--- {kind} ---")
        ranked = sorted(((k, v) for k, v in by_variant.items() if v["kind"] == kind),
                        key=lambda kv: -kv[1]["pass"])
        for key, hit in ranked:
            fails = " ".join(f"{name}x{n}" for name, n in hit["fails"].most_common(4))
            print(f"  {key:<15} {hit['pass']}/{hit['n']} pass   {fails}")


# ------------------------------------------------------------------ MAIN

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--round", type=int, default=max(ROUNDS), help="which variant set")
    parser.add_argument("--tiles", default="search", help=f"{'|'.join(TILE_SETS)} or a comma list")
    parser.add_argument("--only", default=None, help="comma list of variant keys")
    parser.add_argument("--kind", default=None, choices=("position", "rich"))
    parser.add_argument("--show", default=None, help="print one variant's full answers and exit")
    parser.add_argument("--list", action="store_true", help="print the plan, generate nothing")
    parser.add_argument("--force", action="store_true", help="regenerate, ignoring the cache")
    args = parser.parse_args()

    if args.round not in ROUNDS:
        raise SystemExit(f"No round {args.round}; have {sorted(ROUNDS)}")
    variants = ROUNDS[args.round]
    if args.kind:
        variants = [v for v in variants if v.kind == args.kind]
    if args.only:
        wanted = {k.strip() for k in args.only.split(",")}
        variants = [v for v in variants if v.key in wanted]
    if args.show:
        variants = [v for v in variants if v.key == args.show]
    if not variants:
        raise SystemExit("No variants selected")

    keys = TILE_SETS.get(args.tiles) or tuple(k.strip() for k in args.tiles.split(","))
    by_key = {t.key: t for t in TILES}
    missing = [k for k in keys if k not in by_key]
    if missing:
        raise SystemExit(f"Unknown tiles: {missing}. Have {sorted(by_key)}")
    tiles = [by_key[k] for k in keys]

    if args.list:
        print(f"round {args.round}: {len(variants)} variants x {len(tiles)} tiles "
              f"= {len(variants) * len(tiles)} generations")
        for variant in variants:
            print(f"  {variant.key:<15} {variant.kind:<9} {variant.parse:<9} "
                  f"think={variant.thinking!s:<5} {variant.note}")
        for tile in tiles:
            print(f"  tile {tile.key:<11} {tile.canvas} {tile.rect}  {tile.what}")
        return

    root, note = ab_env.bootstrap()
    print(f"[env]    ComfyUI {ab_env.version(root)} at {root}  ({note})")
    print(f"[plan]   round {args.round}: {len(variants)} variants x {len(tiles)} tiles")

    clip_box, inputs, rows = [None], {}, []
    for variant in variants:
        for tile in tiles:
            record = answer_for(clip_box, variant, tile, inputs, args.force)
            answer = record["answer"]
            items = split_items(answer, variant.parse)
            stats = analyze(answer, items, variant.kind, record["closed"])
            rows.append((variant, tile, record, stats, verdict(stats)))

    if args.show:
        for variant, tile, record, _stats, call in rows:
            print(f"\n{'=' * 90}\n{variant.key} / {tile.key}  ({call})\n{'-' * 90}")
            print(record["raw"] if variant.thinking else record["answer"])
        return

    report(rows, LAB_DIR / f"round{args.round}_{args.tiles}.txt")
    summarize(rows)


# ------------------------------------------------------------------ bootstrap

sys.path.insert(0, str(Path(__file__).resolve().parent))
if "COMFYUI_ROOT" not in os.environ:
    if not (KREA2_ROOT / "comfy" / "text_encoders" / "krea2.py").is_file():
        raise SystemExit(f"No Krea 2 support at {KREA2_ROOT} — set COMFYUI_ROOT")
    os.environ["COMFYUI_ROOT"] = str(KREA2_ROOT)
import ab_env  # noqa: E402  (must precede every comfy import)

if __name__ == "__main__":
    main()
