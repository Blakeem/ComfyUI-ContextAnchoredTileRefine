"""Per-tile VLM captions: the VL nodes' two caption conditioning surfaces.

The `vlm_method` select routes every tile's positive through one of three surfaces, and
this module owns the two that involve the VL model's text generator:

    vision tokens               vl.build_global_slices — ONE whole-canvas vision encode,
                                row-sliced per tile. Positionally exact, demand-free, and
                                it invents nothing. The default, and untouched by this
                                module.
    captions                    build_caption_conds — the VL model writes a description of
                                each tile's own crop and that text IS the tile's whole
                                positive. Creative: it can repair a messy background or a
                                hallucination in the source by steering the tile toward
                                something coherent, at the cost of inventing detail the
                                source lacks.
    vision tokens and captions  build_slice_caption_conds — the caption rides INSIDE the
                                whole-canvas vision encode, so each tile keeps its vision
                                rows AND the caption rows.

Cost, which the two caption surfaces do NOT share: both pay one clip.generate per tile per
picture, but `captions` then pays one cheap TEXT encode per caption while
`vision tokens and captions` pays one whole-canvas VISION encode per caption (the encode
carries the caption, so it cannot be shared across tiles the way vl.py's single encode is).

Everything here is lifted from tests-AB/run_ab_matrix.py, which produced the renders the
owner judged on 2026-08-13; nothing is newly invented. Module scope is torch-only; comfy
is imported lazily inside functions (the same contract as vl.py / sampling.py, pinned by
a subprocess test).
"""
import math
import re

import torch

from . import vl

# The three conditioning surfaces, in widget order. "vision tokens" is the default and is
# served by vl.build_global_slices with nothing from this module in the path, so today's
# output stays byte-identical structurally rather than by promise.
VLM_METHOD_VISION = "vision tokens"
VLM_METHOD_VISION_CAPTIONS = "vision tokens and captions"
VLM_METHOD_CAPTIONS = "captions"
VLM_METHODS = [VLM_METHOD_VISION, VLM_METHOD_VISION_CAPTIONS, VLM_METHOD_CAPTIONS]

# --- the SETTLED instruction pair (2026-08-13), from the seven-round search in
# tests-AB/vlm_prompt_lab.py and the owner's own ComfyUI trials of the finalists. One
# instruction per surface. tests-AB/run_ab_matrix.py carries a SUPERSEDED pre-settlement
# pair under confusingly similar names (POSITION_INSTRUCTION / RICH_INSTRUCTION); the
# SETTLED_ names are kept here so the two can never be confused again.
#
#   SETTLED_POSITION  rides WITH the VL slices. The vision rows carry appearance
#                     positionally; what they lack is which object a region belongs to and
#                     how much of it the tile holds, which is exactly what this asks for.
#   SETTLED_RICH      the captions-ONLY surface. Style, palette and lighting lead, because
#                     with no vision rows nothing else carries appearance.
#
# Findings baked into the wording, none of which may be "tidied" out:
#   - The position prompt carries TWO independent bounds and either one alone terminates:
#     the `up to eight` ceiling and GROUP_CLAUSE. With NEITHER, 7 of 9 round-1 cells ran to
#     the token cap — one object mined for its parts (hair/eyes/lips/neck...) or one
#     repeated per instance (stall x30). What the clause does ON TOP of the ceiling is
#     suppress quantity words: removing it took numeric mentions from 1 to 5 across 9 tiles.
#   - `short phrase` holds items to 6-11 words; asking for a `sentence` licenses 32-40.
#   - the five REGIONS in the rich prompt are a stop condition set by the QUESTION, not by
#     how busy the picture is, which is why that one alone never runs away.
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

# The rich prompt WITH the grouping clause — what the `captions` surface SHIPS. The owner's
# explicit decision, taken against the contrary lab measurement, so it is not a mistake to
# correct. On the record both ways: the owner judged 1-face/17_CaptionOnly+Group_Lead_s42_v3
# "better across the board" against 14_CaptionOnly_Lead_s42_v3 (ungrouped, which drew a
# phantom second moon), and ruled that region repeats are acceptable on THIS surface —
# "Repeating may be fine, it often does that only for predominate stuff and that just
# increases weight when it's caption only. This was only an issue with VL method combined."
# Against that: round 7 of the lab scored the grouped wording 4/9 vs 7/9 on lab tiles
# (uniform crops repeat), and the grouped wording was rendered on the `face` scene ONLY.
RICH_GROUPED_INSTRUCTION = f"{SETTLED_RICH_INSTRUCTION} {GROUP_CLAUSE}"

# What each surface asks the VLM, and how long an answer it budgets for. The cap must cover
# the reasoning turn as well (captions are always generated with thinking=True), which is
# why these are 512/768 rather than the pre-settlement 160/220.
CAPTION_INSTRUCTIONS = {
    VLM_METHOD_VISION_CAPTIONS: (SETTLED_POSITION_INSTRUCTION, SETTLED_POSITION_MAX_TOKENS),
    VLM_METHOD_CAPTIONS: (RICH_GROUPED_INSTRUCTION, SETTLED_RICH_MAX_TOKENS),
}

# Caption input budget (total pixels, aspect preserved) — AB27's prep. Conditioning-side
# only: what the VLM reads is a COPY of the tile's crop, never the sampled tile itself
# (prime directive 1: a sampled tile is never resized, resampled or otherwise degraded).
VL_INPUT_BUDGET = 384 * 384

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

_META_LINE = ("wait,", "wait ", "here's a revised", "here is a revised", "revised version",
              "let me", "i need to", "actually,")


def resample_for_vl(tile_pixels):
    # AB27's caption input prep: area-resample a COPY of the tile's crop to VL_INPUT_BUDGET
    # total pixels. Unlike vl.resample_for_global there is no /MERGED_CELL snap, because
    # nothing slices this encode by row — the tokenizer's own rounding is free to apply.
    import comfy.utils

    samples = tile_pixels.movedim(-1, 1)
    scale_by = math.sqrt(VL_INPUT_BUDGET / (samples.shape[3] * samples.shape[2]))
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    resampled = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
    return resampled.movedim(1, -1)[:, :, :, :3]


def strip_thinking(text):
    """Cut Qwen3's reasoning turn off the front of an answer.

    Mirrors comfy_extras/nodes_textgen.py TextGenerateLTX2Prompt, which is the only place
    core does this — the plain TextGenerate node returns the reasoning to the user. Here it
    is mandatory: the caption is encoded as text, so an unstripped <think> block would reach
    the DiT as several hundred tokens of the model talking to itself."""
    if "<think>" not in text:
        return text.strip()
    body = _THINK_BLOCK.sub("", text)
    if "</think>" in body:                  # truncated/unclosed: keep what follows the last
        body = body.rsplit("</think>", 1)[-1]
    return re.sub(r"</?think>", "", body).strip()


def clean_caption(text):
    """Remove the VLM's own formatting artifacts so the DiT never reads them as content.

    Three observed failures, all of which reach the conditioning verbatim because the
    caption is encoded as text:
      heading      "**Answer:**", "**Image as compact list:**" — a label, not a description
      meta         "Wait, I need to rephrase to fit under 50 words." — the model narrating
      repetition   a market crop looped six items THREE times inside one 84-word caption,
                   which is exactly the duplication the instruction exists to avoid
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
    # already fine.
    return text if not dropped else "\n".join(kept).strip()


def _tokenize_images(clip, text, image, **kwargs):
    # vl._encode_canvas' two tokenizer guards, worded for this surface, plus the tail length
    # the slice+caption layout is derived from: the rows AFTER vision_end, i.e. the caption
    # text and the template tail. Returns (tokens, tail_len).
    try:
        tokens = clip.tokenize(text, images=[image], **kwargs)
    except TypeError as error:
        raise RuntimeError(
            "Context-Anchored Tile Refine (VL): this CLIP's tokenizer does not accept images. "
            "The caption vlm_methods need a vision-language text encoder (Krea 2 family). "
            f"({error})") from error
    ids = [t[0] for t in tokens[next(iter(tokens))][0]]
    pad_pos = next((i for i, v in enumerate(ids) if isinstance(v, dict)), None)
    if pad_pos is None:
        raise RuntimeError(
            "Context-Anchored Tile Refine (VL): the tokenizer produced no image tokens. The "
            "caption vlm_methods need a vision-language text encoder (Krea 2 family).")
    return tokens, len(ids) - (pad_pos + 2)


def generate_caption(clip, vl_input, instruction, max_length, thinking=True):
    """One greedy caption of `vl_input`, then the settled fallback chain for a crop whose
    stop token fires immediately. `max_length` is per-instruction and has to cover the
    reasoning turn as well as the answer, which is why the settled instructions carry
    512 / 768 rather than the pre-settlement 160 / 220."""
    if not hasattr(clip, "generate") or not hasattr(clip, "decode"):
        raise RuntimeError(
            "Context-Anchored Tile Refine (VL): this CLIP cannot generate text. The caption "
            "vlm_methods need a vision-language text encoder with a text-generation head "
            "(Krea 2 family); use vlm_method 'vision tokens' with any other CLIP.")

    tokens, _tail = _tokenize_images(clip, instruction, vl_input, thinking=thinking)
    ids = clip.generate(tokens, do_sample=False, max_length=max_length, repetition_penalty=1.05)
    text = strip_thinking(clip.decode(ids))
    if not text:
        ids = clip.generate(tokens, do_sample=True, max_length=max_length, temperature=0.7,
                            top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05, seed=42)
        text = strip_thinking(clip.decode(ids))
    if not text:
        retokens, _tail = _tokenize_images(clip, "Write one short sentence describing this image.",
                                           vl_input, thinking=False)
        ids = clip.generate(retokens, do_sample=False, max_length=max_length, repetition_penalty=1.05)
        text = strip_thinking(clip.decode(ids))
    if not text:
        raise RuntimeError(
            "Context-Anchored Tile Refine (VL): caption generation returned an empty answer "
            "after every fallback.")
    return text


def generate_tile_captions(clip, source, tiles, instruction, max_length, batch_size=1, batch_index=0):
    """One caption per tile per batch row, read off the FROZEN raw canvas.

    Returns captions[tile_index][batch_row]. Batch rows are captioned INDEPENDENTLY: core's
    tokenizer attaches images[0] alone (comfy/text_encoders/qwen_vl.py process_qwen2vl_images),
    so a whole [B,H,W,3] crop would describe every row with row 0's picture. Through the node
    `source` always holds exactly ONE picture — sampling.refine_image's picture loop is outside
    this pre-pass — so the row axis is length 1 there and batch_size/batch_index carry the
    picture's place in the run, which is all the ProgressBar below needs to span it.

    The pre-pass this drives is no longer "one encode" — it is one clip.generate per tile per
    row at up to SETTLED_RICH_MAX_TOKENS tokens, which on a 16-tile grid runs for minutes
    before the first tile samples. Hence the per-tile interrupt check and the ProgressBar:
    without them the run is uncancellable and the UI shows nothing until the tile loop starts.
    """
    import comfy.model_management
    import comfy.utils

    batch = int(source.shape[0])
    per_picture = len(tiles) * batch
    total = per_picture * batch_size
    pbar = comfy.utils.ProgressBar(total)
    done = per_picture * batch_index
    captions = []

    for tile in tiles:
        comfy.model_management.throw_exception_if_processing_interrupted()
        crop = tile.crop_rect
        row_captions = []
        for b in range(batch):
            vl_input = resample_for_vl(source[b:b + 1, crop.y0:crop.y1, crop.x0:crop.x1, :])
            text = generate_caption(clip, vl_input, instruction, max_length, thinking=True)
            row_captions.append(clean_caption(text))
            done += 1
            pbar.update_absolute(done, total)
        captions.append(row_captions)
    return captions


def _slice_rows(encoded, indices):
    # vl.build_global_slices' per-tile selection, verbatim: keep this tile's rows and drop
    # the full-canvas attention mask (its absence means "attend to everything", which is
    # exact for the rows kept).
    sliced = []
    for entry in encoded:
        tensor, extras = entry[0], dict(entry[1])
        extras.pop("attention_mask", None)
        index = torch.tensor(indices, device=tensor.device)
        sliced.append([tensor.index_select(1, index), extras])
    return sliced


def build_caption_conds(clip, captions):
    """Captions WITHOUT slices: each caption re-encoded text-only, exactly what
    CLIPTextEncode would produce for that string. `captions` keeps the [tile][batch row]
    shape generate_tile_captions returns, with exactly ONE row: sampling.refine_image
    refines one picture at a time, so there is never a second row to concatenate."""
    tile_positives = []
    for tile_captions in captions:
        encoded = clip.encode_from_tokens_scheduled(clip.tokenize(tile_captions[0]))
        tile_positives.append(vl._convert(encoded))
    return tile_positives


def _encode_slice_caption(clip, canvas_copy, caption, n_rows):
    # ONE whole-canvas vision encode that also carries `caption`. The expected stripped
    # length is derived from the token stream and asserted against the encoder's output, so
    # a core layout change fails fast instead of silently scrambling every slice — the same
    # fail-fast vl._encode_canvas runs, extended by the caption's own rows.
    tokens, tail_len = _tokenize_images(clip, vl.VISION_BLOCK + caption, canvas_copy,
                                        llama_template=vl.KREA2_TEMPLATE)
    expected_seq = 1 + n_rows + 1 + tail_len
    encoded = clip.encode_from_tokens_scheduled(tokens)
    seq = encoded[0][0].shape[1]
    if seq != expected_seq:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): the slice+caption encode has {seq} rows, expected "
            f"{expected_seq} ({n_rows} vision grid rows + caption/tail {tail_len}). The text "
            "encoder's template or strip layout does not match the Krea 2 contract this node "
            "slices by.")
    return encoded, expected_seq


def build_slice_caption_conds(clip, encode_source, tiles, captions, offset_x=0, offset_y=0):
    """VL slices AND captions: the caption tokens ride INSIDE the whole-canvas vision encode
    and each tile keeps its grid rows plus the caption and template tail.

    `encode_source` is the canvas the OFFSET tile rects index and is taken separately from
    the canvas the captions describe — mirroring vl.build_global_slices: on the whole-image
    path both are the padded canvas at offset 0, while on the mask path the captions describe
    each region tile's own crop and this encodes the FULL image with the bbox origin as the
    offset, so a masked refine stays globally informed. One encode per DISTINCT (batch row,
    caption) pair: the caption is inside the encode, so unlike vl.py's single vision encode it
    cannot be shared across tiles that were captioned differently."""
    canvas_copy, enc_h, enc_w = vl.resample_for_global(encode_source)
    grid_h, grid_w = enc_h // vl.MERGED_CELL, enc_w // vl.MERGED_CELL
    n_rows = grid_h * grid_w
    canvas_h, canvas_w = int(encode_source.shape[1]), int(encode_source.shape[2])
    batch = int(canvas_copy.shape[0])
    cache = {}
    tile_positives = []

    if any(len(tile_captions) != batch for tile_captions in captions):
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): {batch} image(s) in the batch but a tile was "
            "captioned a different number of times. Every batch row must carry its own caption "
            "or a row would be conditioned on another row's picture.")

    for tile, tile_captions in zip(tiles, captions, strict=True):
        per_row = []
        for b, caption in enumerate(tile_captions):
            if (b, caption) not in cache:
                cache[(b, caption)] = _encode_slice_caption(clip, canvas_copy[b:b + 1], caption, n_rows)
            per_row.append(cache[(b, caption)])
        # Exactly ONE row, so nothing is concatenated across rows: the count guard above ties
        # len(tile_captions) to the encode canvas's batch, and refine_image's picture loop
        # makes that batch 1.
        encoded, expected_seq = per_row[0]
        indices = vl.slice_indices(tile.crop_rect, canvas_h, canvas_w, enc_h, enc_w, expected_seq, offset_x, offset_y)
        tile_positives.append(vl._convert(_slice_rows(encoded, indices)))
    return tile_positives
