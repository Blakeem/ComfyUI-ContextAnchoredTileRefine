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
    vision tokens and captions  build_slice_caption_conds — both halves, concatenated: the
                                tile's row slice of ONE shared pure-vision canvas encode,
                                followed by that tile's caption encoded TEXT-ONLY.

Cost: both caption surfaces pay one clip.generate per tile per picture, then one cheap TEXT
encode per caption. `vision tokens and captions` adds exactly ONE whole-canvas vision encode
for the run — the same single encode `vision tokens` pays, shared by every tile.

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
#   SETTLED_POSITION  RETIRED FROM THE SURFACE 2026-08-16 (owner decision, the text-cat
#                     campaign): it rode WITH the VL slices while the caption was encoded
#                     inside the canvas stream. Both VL-carrying methods now ask the RICH
#                     question. Kept defined, character-frozen, because tests-AB's judged
#                     "pos" arms pin themselves to it and their renders are on disk.
#   SETTLED_RICH      what every caption-carrying surface asks. Style, palette and lighting
#                     lead, because on the captions-ONLY surface nothing else carries
#                     appearance, and on the slice+caption surface the vision rows carry
#                     position already.
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
#
# Both caption surfaces ask the SAME question since 2026-08-16: with the caption encoded
# text-only it no longer reads the canvas while it is encoded, so the position wording had
# nothing left to complement — the full RICH description is what the owner judged better on
# the text-cat surface. SETTLED_POSITION_* stays defined above; nothing here selects it.
CAPTION_INSTRUCTIONS = {
    VLM_METHOD_VISION_CAPTIONS: (RICH_GROUPED_INSTRUCTION, SETTLED_RICH_MAX_TOKENS),
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
    # text and the template tail. Returns (tokens, tail_len). Nothing is encoded here.
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


def generate_tile_captions(clip, source, tiles, instruction, max_length, batch_size=1, batch_index=0,
                           progress=None):
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

    `progress` is the VL run's ledger (progress.py) when a node created one. With it the
    standalone bar is NOT constructed — a second bar is exactly the display reset the ledger
    exists to remove — and each finished caption is reported to the ledger's open caption
    segment with the SAME (done, total) counters the bar carries, so its per-caption chunk
    snaps to its boundary while core's per-token bar (routed by the ledger's shim) fills it
    in between. Without a ledger nothing here changes.
    """
    import comfy.model_management
    import comfy.utils

    batch = int(source.shape[0])
    per_picture = len(tiles) * batch
    total = per_picture * batch_size
    pbar = None if progress is not None else comfy.utils.ProgressBar(total)
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
            if pbar is None:
                progress.caption_done(done, total)
            else:
                pbar.update_absolute(done, total)
        captions.append(row_captions)
    return captions


def _entry_extras(entry):
    # A slice's extras: the encode's own, minus the full-canvas attention mask (its absence
    # means "attend to everything", which is exact for the rows kept).
    extras = dict(entry[1])
    extras.pop("attention_mask", None)
    return extras


def _slice_rows(encoded, indices):
    # vl.build_global_slices' per-tile selection, verbatim.
    sliced = []
    for entry in encoded:
        index = torch.tensor(indices, device=entry[0].device)
        sliced.append([entry[0].index_select(1, index), _entry_extras(entry)])
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


def _slice_vision_rows(encoded, crop, canvas_h, canvas_w, enc_h, enc_w, n_rows, offset_x, offset_y):
    # [vision_start][this tile's grid cells][vision_end] out of the ONE pure-vision canvas
    # encode — the shipped slice MINUS its template tail. Passing expected_seq = n_rows + 2
    # makes vl.slice_indices' trailing range empty; the one template tail the stream may carry
    # arrives with the caption rows that are concatenated after these.
    indices = vl.slice_indices(crop, canvas_h, canvas_w, enc_h, enc_w, n_rows + 2, offset_x, offset_y)
    return _slice_rows(encoded, indices)


def _caption_tail_len(clip, caption, probe_image):
    # How many rows this caption occupies after vision_end (caption text + template tail),
    # read off the TOKEN stream rather than off any encoder output — which is what makes it an
    # independent expectation for _encode_caption_text_only to be checked against. Nothing is
    # encoded here; the image only makes the stream well-formed for the tokenizer.
    _tokens, tail_len = _tokenize_images(clip, vl.VISION_BLOCK + caption, probe_image,
                                         llama_template=vl.KREA2_TEMPLATE)
    return tail_len


def _encode_caption_text_only(clip, caption, expected_rows):
    # The caption encoded exactly as CLIPTextEncode would encode it. Krea 2's template strip
    # removes a PREFIX only, so what survives is [caption rows][template tail] and nothing
    # else (measured, docs/vl-conditioning-encode-cost.md section 10). That is asserted
    # against the in-stream tail length so a template change fails fast instead of silently
    # concatenating a differently-shaped stream onto every tile's vision rows.
    encoded = clip.encode_from_tokens_scheduled(clip.tokenize(caption))
    seq = int(encoded[0][0].shape[1])
    if seq != expected_rows:
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): the text-only caption encode has {seq} rows, "
            f"expected {expected_rows} (caption + template tail). The text encoder's template "
            "or strip layout does not match the Krea 2 contract this surface concatenates by.")
    return encoded


def _cat_rows(vision_entries, caption_entries):
    # One tile's positive: its sliced vision rows, then its caption rows, on the ROW axis.
    # Extras are the VISION encode's — the caption encode's are dropped — so anything the
    # caption encode carries that the vision encode does not would vanish silently. Both are
    # hard errors instead: a stray extra key, and a real pooled_output (Krea 2 produces None
    # on both encodes, measured; a CLIP that does not is outside what this surface settled on).
    merged = []
    for vision, caption in zip(vision_entries, caption_entries, strict=True):
        vision_extras, caption_extras = _entry_extras(vision), _entry_extras(caption)
        stray = sorted(set(caption_extras) - set(vision_extras))
        if stray:
            raise RuntimeError(
                "Context-Anchored Tile Refine (VL): the caption encode carries conditioning "
                f"extras the vision encode lacks ({stray}); concatenating the rows would drop "
                "them silently.")
        if caption_extras.get("pooled_output") is not None:
            raise RuntimeError(
                "Context-Anchored Tile Refine (VL): the caption encode has a real "
                "pooled_output; the concatenated positive keeps the vision encode's, so this "
                "one would be dropped silently.")
        rows = torch.cat([vision[0], caption[0].to(vision[0].device, vision[0].dtype)], dim=1)
        merged.append([rows, vision_extras])
    return merged


def build_slice_caption_conds(clip, encode_source, tiles, captions, offset_x=0, offset_y=0):
    """VL slices AND captions: each tile's positive is its row slice of ONE shared pure-vision
    canvas encode, followed by that tile's own caption encoded TEXT-ONLY, concatenated on the
    row axis.

    Settled 2026-08-16 by the owner's A/B (tests-AB/run_ab_split.py arm 2 against the previous
    arm 1, then the sync-tiles campaign). Until then the caption rode INSIDE a whole-canvas
    vision encode, which cost one whole-canvas encode PER TILE and let far-canvas content leak
    into every tile's caption rows through attention — the phantom-moon failure. Encoding the
    caption alone removes both. The vision half is unaffected by the change: attention is
    causal and the caption sat after the grid rows, so those rows were never reading it
    (docs/vl-conditioning-encode-cost.md sections 6-7, measured bit-identical at matched
    stream length).

    `encode_source` is the canvas the OFFSET tile rects index and is taken separately from
    the canvas the captions describe — mirroring vl.build_global_slices: on the whole-image
    path both are the padded canvas at offset 0, while on the mask path the captions describe
    each region tile's own crop and this encodes the FULL image with the bbox origin as the
    offset, so a masked refine stays globally informed."""
    canvas_copy, enc_h, enc_w = vl.resample_for_global(encode_source)
    grid_h, grid_w = enc_h // vl.MERGED_CELL, enc_w // vl.MERGED_CELL
    n_rows = grid_h * grid_w
    canvas_h, canvas_w = int(encode_source.shape[1]), int(encode_source.shape[2])
    batch = int(canvas_copy.shape[0])
    tile_positives = []

    if any(len(tile_captions) != batch for tile_captions in captions):
        raise RuntimeError(
            f"Context-Anchored Tile Refine (VL): {batch} image(s) in the batch but a tile was "
            "captioned a different number of times. Every batch row must carry its own caption "
            "or a row would be conditioned on another row's picture.")

    # ONE pure-vision encode for the whole picture, shared by every tile — literally the encode
    # `vision tokens` pays, and the reason this surface no longer scales its encode cost with
    # tile count. Core's tokenizer attaches images[0] alone, so the canvas is narrowed to one
    # picture here; refine_image's picture loop is what makes that the whole batch.
    encoded, _expected_seq = vl._encode_canvas(clip, canvas_copy[:1], grid_h, grid_w)

    for tile, tile_captions in zip(tiles, captions, strict=True):
        # Exactly ONE row, so nothing is concatenated across rows: the count guard above ties
        # len(tile_captions) to the encode canvas's batch, and refine_image's picture loop
        # makes that batch 1.
        caption = tile_captions[0]
        tail_len = _caption_tail_len(clip, caption, canvas_copy[:1])
        vision_entries = _slice_vision_rows(encoded, tile.crop_rect, canvas_h, canvas_w,
                                            enc_h, enc_w, n_rows, offset_x, offset_y)
        caption_entries = _encode_caption_text_only(clip, caption, tail_len)
        tile_positives.append(vl._convert(_cat_rows(vision_entries, caption_entries)))
    return tile_positives
