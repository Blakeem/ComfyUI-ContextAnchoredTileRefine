"""Vision-row bit-identity probe: the cheap pre-check from docs/vl-conditioning-encode-cost.md.

PRE-PORT HARNESS, PINNED TO COMMIT 0dfd1e4. Not runnable against the current package: this
file calls `captions._encode_slice_caption`, which was DELETED on 2026-08-16 when the
text-cat surface it validated was promoted into `captions.build_slice_caption_conds`. Kept
verbatim as the frozen record of that measurement (roadmap decision — tests-AB is a record,
not a maintained suite).

Section 10 of that doc: before any timing matrix or render A/B, verify on the REAL installed
model that the vision rows (indices 0 .. N+1) of a caption-carrying encode are bit-identical
to the same rows of (a) an encode carrying a DIFFERENT caption and (b) the pure-vision encode
`vl._encode_canvas` produces. If they are, the "plain split" design's vision half is provably
free and only the caption rows need a render A/B. If they are not, section 6's causality
claim is wrong and the whole split design is void.

Five encoder calls, no clip.generate, no sampling:

    E1  _encode_slice_caption(canvas, CAPTION_A)     today's stream, caption A
    E2  _encode_slice_caption(canvas, CAPTION_A)     the SAME call again — determinism floor
    E3  _encode_slice_caption(canvas, CAPTION_B)     today's stream, caption B
    E4  vl._encode_canvas(canvas)                    pure vision — the plain split's ONE encode
    E5  encode_from_tokens_scheduled(tokenize(A))    text-only caption A — the plain split's
                                                     cat operand (build_caption_conds' call)
    E6  _encode_slice_caption(canvas, CAPTION_B      caption B padded until its token count
                              padded to A's length)  MATCHES caption A's

Checks, in dependency order:

    C0  vision rows of E1 vs E2      the floor: if the same input is not reproducible,
                                     bit-identity is unmeasurable and every later check
                                     is read against this floor instead of zero
    C1  vision rows of E1 vs E3      causality across captions, DIFFERENT stream lengths
    C1b vision rows of E1 vs E6      causality across captions, SAME stream length — the
                                     discriminator: a causal-but-nonzero C1 can come purely
                                     from the GEMM shape changing with sequence length
                                     (different kernel tiling => different bf16 reduction
                                     order). Identical shapes remove that; any C1b residue
                                     is real information flow, not numerics.
    C2  vision rows of E1 vs E4      the actual substitution the plain split makes
    C3  caption rows of E1 vs E5     informational: how far the image-blind caption rows
                                     sit from today's image-aware ones (expected NONZERO —
                                     this is the half section 10's render A/B exists for;
                                     part of it is RoPE position, not image-awareness)

Per-call wall time is printed as a first datum for the section 9 timing question (call 1
includes the weight upload; later calls show the resident per-encode cost).

Run alone on an idle GPU (CLAUDE.md: one GPU job at a time):
    <venv python> tests-AB/probe_split_encode.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_env

ab_env.bootstrap()

import ab_models  # noqa: E402

from context_anchored_tile_refine import captions, vl  # noqa: E402

CLIP_NAME = "qwen3-vl-4b-heretic_int8.safetensors"
CLIP_TYPE = "krea2"
BASE_PNG = Path(__file__).resolve().parent / "inputs" / "krea2-00676-base-768x1024.png"

# Fixed strings, not clip.generate output: the check is about where caption tokens SIT in
# the stream, not what they say. Shaped like SETTLED_POSITION_INSTRUCTION answers, clearly
# different from each other in content and length.
CAPTION_A = ("a stone bridge, centre of the frame, fully visible\n"
             "a river, bottom half, flowing left to right\n"
             "a willow tree, upper left, half in frame")
CAPTION_B = ("a red lantern, top right corner, fully visible\n"
             "a wooden boat, lower centre, most of it showing")


def load_base():
    """The saved Krea 2 base as LoadImage would hand it over: RGB / 255, [1,H,W,3]."""
    with Image.open(BASE_PNG) as handle:
        arr = np.asarray(handle.convert("RGB")).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr)[None]
    ab_models.require_image_shape(tensor, 1024, 768, "base png")
    return tensor


def timed(label, fn):
    started = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - started
    print(f"  {label:<44s} {elapsed:7.2f}s")
    return result


def main_tensor(encoded):
    """The cond tensor of the first (and for this CLIP, only) scheduled entry."""
    return encoded[0][0]


def tail_rows(clip, canvas, caption):
    """How many rows this caption's stream carries after vision_end (caption + tail)."""
    _, tail_len = captions._tokenize_images(clip, vl.VISION_BLOCK + caption, canvas,
                                            llama_template=vl.KREA2_TEMPLATE)
    return tail_len


def pad_to_length(clip, canvas, caption, target):
    """Append filler tokens until the caption's stream is exactly `target` rows long, so
    its encode runs at the SAME GEMM shape as the reference caption's."""
    padded = caption
    for _ in range(4 * target):
        length = tail_rows(clip, canvas, padded)
        if length == target:
            return padded
        if length > target:
            raise RuntimeError(f"overshot: {length} > {target} rows for {padded!r}")
        padded += " very"
    raise RuntimeError(f"could not reach {target} rows from {caption!r}")


def compare(label, left, right, floor=0.0):
    """Bit-equality plus the max abs diff, read against the determinism floor."""
    if left.shape != right.shape:
        print(f"  {label:<28s} SHAPE MISMATCH {tuple(left.shape)} vs {tuple(right.shape)}")
        return None
    equal = torch.equal(left, right)
    diff = (left.float() - right.float()).abs().max().item()
    at_floor = "" if equal else ("  (== floor)" if diff <= floor else "  (ABOVE floor)")
    print(f"  {label:<28s} equal={equal}  max|diff|={diff:.3e}{at_floor}")
    return diff


def main():
    base = load_base()
    canvas, enc_h, enc_w = vl.resample_for_global(base)
    grid_h, grid_w = enc_h // vl.MERGED_CELL, enc_w // vl.MERGED_CELL
    n_rows = grid_h * grid_w
    print(f"canvas {enc_w}x{enc_h}, grid {grid_w}x{grid_h}, n_rows={n_rows}")

    clip = ab_models.load_clip(CLIP_NAME, CLIP_TYPE)

    print("\nencodes (call 1 includes the weight upload):")
    with ab_models.VramProbe() as probe, torch.inference_mode():
        e1, seq_a = timed("E1 caption A (today's stream)",
                          lambda: captions._encode_slice_caption(clip, canvas, CAPTION_A, n_rows))
        e2, _ = timed("E2 caption A again (determinism floor)",
                      lambda: captions._encode_slice_caption(clip, canvas, CAPTION_A, n_rows))
        e3, seq_b = timed("E3 caption B (today's stream)",
                          lambda: captions._encode_slice_caption(clip, canvas, CAPTION_B, n_rows))
        e4, seq_pure = timed("E4 pure vision (plain split's one encode)",
                             lambda: vl._encode_canvas(clip, canvas, grid_h, grid_w))
        e5 = timed("E5 caption A text-only (split's operand)",
                   lambda: clip.encode_from_tokens_scheduled(clip.tokenize(CAPTION_A)))
        caption_b_padded = pad_to_length(clip, canvas, CAPTION_B,
                                         tail_rows(clip, canvas, CAPTION_A))
        e6, seq_b2 = timed("E6 caption B padded to A's length",
                           lambda: captions._encode_slice_caption(clip, canvas, caption_b_padded, n_rows))
    print(f"  {probe}")
    print(f"  seq: A={seq_a}  B={seq_b}  pure={seq_pure}  text-only={main_tensor(e5).shape[1]}  B-padded={seq_b2}")

    vis = slice(0, n_rows + 2)                     # rows 0 .. N+1: vision_start, cells, vision_end
    t1, t2, t3, t4, t5, t6 = (main_tensor(e) for e in (e1, e2, e3, e4, e5, e6))
    vision_scale = t1[:, vis].float().abs()
    print(f"  vision-row magnitude: mean|x|={vision_scale.mean().item():.3f}  max|x|={vision_scale.max().item():.3f}")

    print("\nC0 determinism floor — E1 vs E2, vision rows:")
    floor = compare("same input twice", t1[:, vis], t2[:, vis]) or 0.0

    print("\nC1 causality, different lengths — E1 vs E3, vision rows:")
    d1 = compare("caption A vs caption B", t1[:, vis], t3[:, vis], floor)

    print("\nC1b causality, SAME length — E1 vs E6, vision rows (the discriminator):")
    d1b = compare("caption A vs padded B", t1[:, vis], t6[:, vis], floor)

    print("\nC2 the substitution — E1 vs E4, vision rows:")
    d2 = compare("caption A vs no caption", t1[:, vis], t4[:, vis], floor)

    print("\nC3 caption rows, image-aware vs image-blind — E1 tail vs E5 (expected NONZERO):")
    compare("E1[N+2:] vs text-only", t1[:, n_rows + 2:], t5, floor)

    print("\nsanity — caption rows of E1 vs E3 must NOT match (captions differ):")
    if seq_a == seq_b:
        compare("E1[N+2:] vs E3[N+2:]", t1[:, n_rows + 2:], t3[:, n_rows + 2:], floor)
    else:
        print(f"  lengths differ ({seq_a - n_rows - 2} vs {seq_b - n_rows - 2} rows) — trivially different")

    pooled_a = e1[0][1].get("pooled_output")
    pooled_pure = e4[0][1].get("pooled_output")
    if isinstance(pooled_a, torch.Tensor) and isinstance(pooled_pure, torch.Tensor):
        print("\npooled_output — E1 vs E4:")
        compare("caption A vs no caption", pooled_a, pooled_pure, floor)
    else:
        print(f"\npooled_output: A={type(pooled_a).__name__}, pure={type(pooled_pure).__name__} — nothing to compare")

    print("\nverdict:")
    if d1 is None or d1b is None or d2 is None:
        print("  a shape mismatch blocked a check — no verdict.")
    elif d1 <= floor and d2 <= floor:
        print("  vision rows are caption-independent bit-for-bit. The plain split's vision")
        print("  half is validated; only the caption rows (C3) change — section 10's render")
        print("  A/B is what measures those.")
    elif d1b <= floor:
        print("  at MATCHED stream length the vision rows are identical down to the floor, so")
        print("  no caption information reaches them — causality holds on the installed model.")
        print("  The nonzero C1/C2 residue is sequence-length kernel numerics (GEMM tiling),")
        print("  not information flow. The plain split's vision half differs from today's by")
        print("  that numeric noise only; the caption rows (C3) remain the real change.")
    else:
        print("  vision rows DEPEND on the caption even at matched stream length — section 6's")
        print("  causality claim fails on the installed model. The plain-split design is void.")


if __name__ == "__main__":
    raise SystemExit(
        "probe_split_encode.py: PRE-PORT PROBE, pinned to commit 0dfd1e4 (see the module"
        " docstring). captions._encode_slice_caption was deleted in 1.6.0 when the text-cat"
        " surface was promoted into build_slice_caption_conds. Check out 0dfd1e4 to re-run."
    )
