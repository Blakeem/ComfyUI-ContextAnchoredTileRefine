# VL conditioning: what each surface encodes, and what the caption surface costs

Record of the 2026-08-14 investigation into why `vision tokens and captions` re-encodes the
whole image once per tile, whether that cost buys anything, and what a cheaper design would
change. Written to be the reference for the timing/quality tests that have NOT been run yet.

Sections 1-7 are settled from source; section 8 is an estimate explicitly marked as such;
sections 9-11 are the test plan that replaces it. **Update 2026-08-14:** section 10's cheap
bit-identity check has now been RUN on the installed model (`tests-AB/probe_split_encode.py`,
results in section 10) and it validates the causality claim exactly; a first per-encode
timing came with it. Section 9's in-pipeline timing matrix and section 10's render A/B remain
unrun.

---

## 1. Vocabulary: there are exactly two calls

Most of the confusion in this area comes from the word "encode" being used for both. They are
different operations with different outputs.

| call | input | output | runs the vision tower? |
|---|---|---|---|
| `clip.generate(...)` | one tile's crop, resampled to `VL_INPUT_BUDGET` | **a string** | yes, over the crop |
| `clip.encode_from_tokens_scheduled(...)` | one token stream | **a tensor `[1, seq, dim]`** — one vector per token row | only if an image is in the stream |

`encode_from_tokens_scheduled` is the only thing in the system that produces vectors. It takes
**tokens**, never vectors — a slice can never be fed back into it. Any combining of two
conditioning tensors happens on the output side, with `torch.cat` on the row axis.

Two image budgets are involved, and they differ by 5.3x:

```
caption input   captions.py:104   VL_INPUT_BUDGET      = 384 x 384   =  147,456 px
encode input    vl.py:44          GLOBAL_SLICE_BUDGET  = 768 x 1024  =  786,432 px
```

At `GLOBAL_SLICE_BUDGET` with `MERGED_CELL = 32` (`vl.py:35`) the canvas encodes to roughly
768 grid cells — so the encoded stream is ~770-800 rows, of which the tile keeps only its own
cells plus the delimiters and tail.

---

## 2. The row table

Everything below is a statement about this structure. One `encode_from_tokens_scheduled` call
produces one of these.

```
row:      0              1 … N               N+1            N+2 … end
      <vis_start>   image grid cells     <vis_end>     caption + template tail
                    (raster order)                     (caption present only in
                                                        vision tokens and captions)
```

`vl.slice_indices` (`vl.py:62-78`) returns:

```python
[0, *rows, 1 + n_rows, *range(1 + n_rows + 1, expected_seq)]
```

Read as: keep `<vis_start>`, keep only the grid cells this tile's `crop_rect` intersects, keep
`<vis_end>`, then keep **every row after it, unconditionally**. That final `range` is where the
caption rows live — the tile's positive is vision rows AND caption rows, not vision rows alone.

---

## 3. The three surfaces

Selected by the `vlm_method` widget; branched at `sampling.py:798-806`.

```
vision tokens                ONE encode for the whole run
  stream   [tmpl] <vis_start> cell0…cellN <vis_end> [tmpl-close]
  tile k   row 0 + its cells + N+1 + tmpl-close rows
  note     every tile's stream is identical, so one encode is sliced N ways
  code     vl.build_global_slices (vl.py:145)

captions                     ONE encode per tile, NO image
  stream   caption_k
  tile k   all of it, no slicing
  note     no vision tower in the encode; the tower still runs inside generate
  code     captions.build_caption_conds (captions.py:276)

vision tokens and captions   ONE encode per tile, WITH image
  stream   [tmpl] <vis_start> cell0…cellN <vis_end> caption_k [tmpl-close]
  tile k   row 0 + its cells + N+1 + caption_k rows + tmpl-close rows
  note     a different caption is a different stream, so the encode cannot be shared
  code     captions.build_slice_caption_conds (captions.py:307)
```

In `vision tokens and captions` there is **no separate global encode**.
`vl.build_global_slices` is not called at all. The whole-image encode and the caption encode
are the same call (`captions._encode_slice_caption`, `captions.py:288-304`), which is the
entire reason the image goes through once per tile.

Encodes are cached per `(batch row, caption)` (`captions.py:336`), so two tiles that happen to
produce identical caption text share one encode. In practice captions differ per tile.

---

## 4. Where the slice happens

On the output tensor, row axis. Same operation in both slicing surfaces:

```
vl.py:166         tensor.index_select(1, index)     vision tokens
captions.py:272   tensor.index_select(1, index)     vision tokens and captions  (_slice_rows)
```

`captions.py:342` builds the indices via `vl.slice_indices`, `captions.py:343` applies them.
Nothing is re-encoded in order to slice.

---

## 5. Where the caption comes from

`captions.generate_tile_captions` (`captions.py:224`), per tile:

```python
vl_input = resample_for_vl(source[b:b + 1, crop.y0:crop.y1, crop.x0:crop.x1, :])
text = generate_caption(clip, vl_input, instruction, max_length, thinking=True)
```

The VLM sees **that tile's crop only**. The whole image is never in view while captioning.
The input is an area-resampled COPY at `VL_INPUT_BUDGET` — the sampled tile itself is never
resampled (prime directive 1).

Instructions and budgets (`captions.py:96-99`), A/B-settled 2026-08-13, character-frozen:

| surface | instruction | max_length |
|---|---|---|
| vision tokens and captions | `SETTLED_POSITION_INSTRUCTION` | 512 |
| captions | `RICH_GROUPED_INSTRUCTION` | 768 |

`thinking=True` always; `strip_thinking` is mandatory before the text is ever encoded.

---

## 6. Attention direction: settled, causal

The decisive question was whether a caption token sitting AFTER `<|vision_end|>` can influence
the hidden states OF the image rows. Under causal attention it cannot; under bidirectional it
can, and each tile's vision rows would be tinted by that tile's caption.

**Verdict: causal. There is no flag to disable it.**

`comfy/text_encoders/llama.py:745-755`, `Llama2_.forward`, pinned at ComfyUI commit
`bd34f338ac505ea79e43968753968a464060e609` (v0.32.0+5), verified byte-identical to the
installed copy:

```python
mask = None
if attention_mask is not None:
    mask = 1.0 - attention_mask.to(x.dtype).reshape(...)
    mask = mask.masked_fill(mask.to(torch.bool), torch.finfo(x.dtype).min / 4)

if seq_len > 1:
    causal_mask = torch.empty(past_len + seq_len, past_len + seq_len, dtype=x.dtype,
                              device=x.device).fill_(torch.finfo(x.dtype).min / 4).triu_(1)
    if mask is not None:
        mask += causal_mask
    else:
        mask = causal_mask
```

The `attention_mask` argument is a **padding** mask only. Causality is the separate,
unconditional `.triu_(1)` term.

Route for Krea 2: `Krea2Qwen3VLClipModel` -> `Qwen3VLClipModel` -> `Qwen3VL` ->
`self.model = Llama2_` (`comfy/text_encoders/qwen3vl.py:57`), reached from
`encode_from_tokens` at `comfy/sd1_clip.py:279`.

Corroborations:

- Replicating the mask gives an image-row -> caption-column weight of exactly `-16376.0`,
  softmax `0.0`. Not merely structurally causal; the measured coupling is zero.
- ComfyUI **does** implement bidirectional image attention elsewhere — `gemma4.py:373-384`,
  a `vision_bidirectional` flag. No counterpart exists in `llama.py`, so the Qwen path's
  causality is a deliberate choice, not an unimplemented one.
- `clip.generate` uses the same mask code (`llama.py:919`, `attention_mask=None` plus a KV
  cache). Prefill builds the identical causal mask; decode steps run at `seq_len == 1` and
  skip the `if`. Both paths are causal — the conditioning tensor is produced under the same
  mask the captions are generated under.

**Evidence grade: DIRECT** (pinned source, byte-verified against the installed file).

### What is NOT in the papers

Neither the Qwen3-VL nor the Qwen2.5-VL technical report contains the words `causal`,
`bidirectional`, or `autoregressive`. The released modeling code is the only authoritative
source that exists. The paper-level warrant for the LLM half is DERIVED across three papers:
Qwen3-VL "built upon Qwen3 backbones" -> Qwen3 "similar to Qwen2.5" -> Qwen2.5 "we maintain
the Transformer-based decoder architecture".

### Vision-tower facts, for the record

- Qwen3-VL's vision tower is **not** the windowed Qwen2.5-VL ViT. It was swapped for
  **SigLIP-2** (Qwen3-VL Technical Report §2). Krea 2 is the 4B, so its tower is
  **SigLIP2-Large (300M)**.
- Attention there is full/bidirectional over image patches. This is almost certainly the
  source of any "bi-directional (full) self-attention" claim found in secondary sources — it
  describes the tower, and says nothing about whether the caption reaches the image rows.
- The tower is block-diagonal by `cu_seqlens` at **one frame**, not one clip — no attention
  across frames. A no-op for a single-image canvas; it is why the removed H3 video node had
  no cross-frame vision coupling to lean on.

Sources:

- Qwen2.5-VL Technical Report, window attention §2.1.1 — https://arxiv.org/html/2502.13923v1
- Qwen3-VL Technical Report, SigLIP-2 vision encoder §2 — https://arxiv.org/abs/2511.21631
- Qwen3 Technical Report — https://arxiv.org/html/2505.09388v1
- Qwen2.5 Technical Report, "Transformer-based decoder architecture" — https://arxiv.org/html/2412.15115v2
- SigLIP 2 — https://arxiv.org/abs/2502.14786
- ComfyUI `llama.py` at the pinned commit —
  https://github.com/comfyanonymous/ComfyUI/blob/bd34f338ac505ea79e43968753968a464060e609/comfy/text_encoders/llama.py#L745-L755

**Falsification line, unchecked:** the vision-tower non-causal claim would be wrong if Qwen's
own inference stack (the `QwenLM/Qwen3-VL` repo, or vLLM/SGLang kernels) applied a causal mask
the transformers port does not. Only the transformers port was read. This does not touch the
LLM-half conclusion, which rests on ComfyUI's own code — the code that actually runs here.

---

## 7. What causality implies

Place it back on the row table from section 2:

```
rows 1…N      the grid cells the tile slices     sit BEFORE the caption
rows N+2…     the caption                        sit AFTER the image
```

A row attends only to rows at or before its own index. Therefore:

| half of the tile's positive | today vs a split design | status |
|---|---|---|
| **vision rows** (1…N) | **identical up to kernel numerics** | MEASURED 2026-08-14, section 10: bit-identical at matched stream length; across different lengths a max 2.4e-3 GEMM-tiling residue (vs mean row magnitude 1.2) |
| **caption rows** (N+2…) | **different values** | needs an A/B — measured max 4.0e+1 apart, but only a render judges better/worse |

The N whole-image encodes performed today produce the **same grid rows N times**. That part of
the cost buys nothing. The caption rows are the only thing that differs, because today they are
computed with the image sitting ahead of them in the stream and can read it; encoded alone they
cannot.

---

## 8. The three designs, and an ESTIMATE of the saving

```
                     whole-image encodes   vision rows   caption rows      build cost
today                       N              baseline      baseline          shipped
plain split                 1              identical     DIFFERENT         small, public API only
KV-cache split              1 + N suffix   identical     identical         reaches into comfy internals
```

**plain split** = encode the image once, encode each caption text-only, slice both, `torch.cat`
on dim 1. All public API. Wrinkle: `clip.tokenize(caption)` wraps the text in the full
`KREA2_TEMPLATE`, so the caption tensor carries its own head and tail rows — those must be
sliced away so the concatenated stream has one template head and one tail, not two of each.
Same index math as `slice_indices`.

**KV-cache split** = prefill the image prefix once, keep its keys/values, run each caption
suffix against the cache. Causality is exactly what makes this possible, and it is what comfy's
own generate path already does (`llama.py:919`, `past_key_values` in and out). Output would be
bit-identical to today. See section 11 for why this is not simply "implement it ourselves".

### Fermi estimate — NOT measured, replace with section 9

Krea 2's text encoder is Qwen3-VL-4B, ~8 GB at bf16. Machine: 3090 Ti, 1008 GB/s.

```
per tile, generate:  tower over 147k px            ~0.1 s
                     + autoregressive decode       300-512 tokens at ~25-60 tok/s
                                                   = 5-20 s          <-- dominant
per tile, encode:    tower over 786k px            ~0.05 s
                     + one prefill over ~800 rows  ~0.15 s
                                                   = 0.2-0.5 s
```

4-tile grid: today ~32-82 s of generate against ~0.8-2 s of encode. The plain split removes 3
encodes, i.e. **~0.6-1.5 s out of ~33-84 s — on the order of 2%.**

**So the owner's intuition is very likely correct: the caption generation dominates, and the
encode we would remove is small.**

**Dominant uncertainty is NOT compute — it is memory management.** The FLOP arithmetic above
cannot see model offload/reload. The `krea2-wan-vae-headless-testing` and `h3-f0b-tiled-stress`
notes record a VL encode at **183 s / 21 GiB under aimdo** — three orders of magnitude above
the FLOP estimate. If comfy offloads the VL CLIP between calls, per-encode cost is set by
transfer, not by arithmetic, and N encodes could be a large fraction of the run. That single
factor is the reason to measure rather than estimate, and it is the only factor that could flip
the conclusion.

---

## 9. Timing test protocol

Goal: replace section 8's estimate with numbers, and decide whether the saving is one a user
would notice.

Preconditions (from CLAUDE.md, non-negotiable):

- `nvidia-smi` idle before launching. **One GPU sampling job at a time** on this machine.
- At 3x+ canvases, one config per process.

Instrument these three points with wall-clock timers:

| point | file:symbol | what it measures |
|---|---|---|
| A | `captions.generate_tile_captions`, inner loop | one `clip.generate` per tile |
| B | `captions._encode_slice_caption` | one whole-image encode per tile |
| C | `sampling.refine_image` entry/exit | total refine |

Report per run: `N` tiles, sum(A), sum(B), C, and **sum(B)/C as a percentage** — that ratio is
the entire decision.

Matrix, same image and same seed throughout:

1. `vision tokens` — baseline, 1 encode, 0 generates.
2. `captions` — N generates, N cheap text encodes, 0 image encodes.
3. `vision tokens and captions` at a **2x2** grid — N = 4.
4. `vision tokens and captions` at a **4x4** grid — N = 16.

Runs 3 and 4 together separate the per-tile cost from the fixed cost: if sum(B) scales linearly
with N and is a meaningful share of C at N = 16, the optimisation is worth building. If sum(B)
stays under ~5% of C even at N = 16, it is not.

Also record, per run, whether the VL CLIP stayed resident between calls (watch VRAM in
`nvidia-smi`, or log `comfy.model_management` load events). This is the factor section 8 flags
as dominant, and it is invisible in the timing totals alone.

---

## 10. Quality test protocol

Only needed if section 9 says the saving is worth having. It tests the **caption rows**, the
one half that actually changes.

The vision half needs no render test — section 7 proves it bit-identical. Prove that in code
instead, cheaply, on the real installed model (a `gpu`-marked test):

```
encode the SAME canvas twice with two DIFFERENT captions
slice the vision rows (indices 0 … N+1) from each
assert torch.equal(rows_a, rows_b)
```

That single assertion validates the whole causality argument against the model that actually
runs here, rather than against ComfyUI's source. If it fails, section 6 is wrong and everything
downstream of it is void.

### RESULT 2026-08-14 — run, causality CONFIRMED (`tests-AB/probe_split_encode.py`)

Ran on the installed `qwen3-vl-4b-heretic_int8.safetensors` (krea2), ComfyUI 0.32.0
(`C:\Users\Blake\ComfyUI-Installs\ComfyUI\ComfyUI`), canvas
`tests-AB/inputs/krea2-00676-base-768x1024.png` (grid 24x32, n_rows=768). Fixed caption
strings, no `clip.generate`. Six encodes, five checks:

```
determinism      same stream twice                 vision rows bit-identical (floor = 0)
same length      caption A vs caption B, both      vision rows BIT-IDENTICAL   <- the verdict
                 padded to the same token count
cross length     caption A (809 rows) vs B (797)   max|diff| 2.4e-3 on rows of mean |x| 1.2
                 vs no caption (775)               — GEMM shape changes with seq length, so
                                                   bf16 reduction order changes; numerics,
                                                   not information (the same-length check
                                                   is what proves it is not information)
caption rows     image-aware (today) vs text-only  max|diff| 4.0e+1 — the real change; part
                 (plain split's operand)           of it is RoPE position (row ~770 today vs
                                                   ~0 in a split), not image-awareness alone
pooled_output    absent on this CLIP (None)        nothing extra to reconcile in a split
```

Refinement the probe forced on section 7's "bit-identical" claim: exact bit-identity holds
only at matched GEMM shape. The plain split slices its vision rows from the 775-row pure
encode while today's come from an 809-row caption-carrying encode, so split output would
differ from today's by that 2.4e-3 kernel residue even before the caption rows change. That
residue is ~16,000x smaller than the caption-row change and is noise, not signal.

Timing datum (same run, CLIP alone resident, peak 0.63 GiB reserved): whole-image encode
**~0.7 s** once loaded (first call 1.8 s with weight upload), text-only caption encode 0.2 s.
The `183 s / 21 GiB` figure in the memory notes was H3's 32B encoder, not this 4B. So the
plain split saves roughly `(N-1) x 0.7 s` per picture — ~2 s on a 2x2 grid, ~10 s on 4x4 —
against a `generate` pre-pass of 5-20 s per tile. Section 8's ~2% estimate stands, and the
offload-dominates scenario it feared did not appear at this model size. Section 9's
in-pipeline matrix would confirm sum(B)/C with the DiT resident, but the ceiling it can
find is already bounded low.

For the caption half, render the same scene both ways at a fixed seed:

- Arm 1: `vision tokens and captions` as shipped (caption rows image-aware).
- Arm 2: plain split (caption rows image-blind).

Use the scenes that already discriminated on this surface — `4-portrait` (where captions turned
an unreadable tool wall into readable tools) and `3-market`. Same seed, denoise 0.5, overlap 32,
anchor 256, matching the v3 matrix in `output/AB-Final-Matrix`. Final judgement is the owner's
visual A/B, not a metric.

---

## 11. "Could we just implement it ourselves?"

The question was whether writing our own KV-cache path would remove the dependency on comfy
internals. It does not — it converts a loud failure into a silent one.

```
Option B  call comfy's Llama2_.forward directly with past_key_values
          couples to: the attribute chain, the forward signature, the embeds protocol
          when comfy changes: AttributeError / TypeError at runtime      <- LOUD

Option A  vendor a copy of that forward into this repo
          couples to: the same things, plus the weight layout and RoPE it assumes
          when comfy changes: our copy keeps running against a model it
          no longer matches, producing subtly wrong conditioning         <- SILENT

Option C  plain split, public API only (torch.cat on dim 1)
          couples to: nothing beyond encode_from_tokens_scheduled
          cost: caption rows change; needs the section 10 A/B
```

Option A is strictly worse than B, and it is worse for the specific reason this project cares
about: it fails silently. It also means owning far more than one function — the cache path
needs the model class, the embeds/`embeds_info` protocol, and the weight layout, all of which
live in comfy and all of which move.

**The only design that genuinely decouples is Option C**, and its price is the behaviour change
in section 7's second row. There is no version of "implement it ourselves" that gives
bit-identical output AND independence from comfy internals — bit-identity requires the image
prefix's keys and values, and the public API returns final hidden states only.

Both surfaces (`vl.py`, `captions.py`) currently duck-type the CLIP and touch nothing below
`clip.tokenize` / `clip.generate` / `clip.encode_from_tokens_scheduled`. Option B is the first
thing that would break that contract.

---

## 12. Open questions

1. ~~Unmeasured: everything in section 8.~~ 2026-08-14: per-encode cost measured standalone
   (~0.7 s resident, section 10's result block); only the in-pipeline sum(B)/C ratio with the
   DiT resident remains unmeasured, and its ceiling is already bounded low.
2. ~~Unmeasured: whether image-blind caption rows render differently enough to matter.~~
   RENDERED 2026-08-14 (`tests-AB/run_ab_split.py`, 00676 portrait, d0.5, seed 42, four
   arms: whole-image / no-image / tile-crop / grey-control caption encodes). Owner verdict:
   **no-image (plain split) wins on content** — best background detail, the ONLY arm with no
   phantom moon, seam faint and best-blended; **whole-image (shipped) wins on seams** — the
   only arm with no visible seam anywhere, but less background detail and a phantom moon.
   tile-crop: obvious seam, simplest background, moon. grey-control: worst on every axis —
   so arm 1's seam advantage is CONTENT, not RoPE position (the position-matched grey encode
   lost to the position-mismatched text-only one). Readings, one scene, INFERRED: whole-image
   grounding smuggles far-canvas content (the moon) into every tile's caption rows through
   attention — the same leak that helps seams; and since tile-crop's grounding already
   included the 256px anchor halo yet still seamed, the seam benefit looks like SHARED
   context across all tiles' caption rows (story coherence), not local neighbourhood
   coverage. Next question: a combination that keeps no-image's content with whole-image's
   seams (row-aligned lerp of the two caption tensors, or cat both) — unrendered.
3. **Unverified:** the vision-tower falsification line in section 6 (Qwen's own inference stack
   was not read). Does not affect the LLM-half conclusion.
4. **Untested end to end:** the masked VL refine. No A/B render has ever used a mask, on any
   surface. Unrelated to this investigation but it is the other unvalidated path.
