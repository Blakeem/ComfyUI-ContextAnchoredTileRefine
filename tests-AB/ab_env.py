"""This machine's ComfyUI root, model folder and memory flags for the library's harness bootstrap, comfy_env.

`bootstrap()` MUST run before the first `comfy`, `comfy_extras` or `folder_paths`
import, because it decides which ComfyUI source tree those names resolve to and
what `folder_paths.base_path` is.

Root-resolution order and the `comfy/utils.py` vs top-level `utils/` shadowing
hazard are lifted from tests/conftest.py's `comfy_env` fixture. Only <root> ever
goes on sys.path — never <root>/comfy.

Deviation from tests/conftest.py, deliberate: the candidate list is ordered by
CAPABILITY, not by install location. The z-image checkpoints this harness renders
(ungloryhailZImage / qwen_3_4b as lumina2) are only supported by newer ComfyUI —
the 0.3.45 desktop tree under AppData has no z-image support at all
(no comfy/text_encoders/z_image.py, nothing matching z_image in comfy/), so
loading the UNET there fails outright. `Z_IMAGE_MARKER` probes for that support and
roots that have it win. Set COMFYUI_ROOT to override the whole search.
"""
import dataclasses
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# --base-directory: where models/, output/, custom_nodes/ live. Matches the desktop
# app's own launch config (%APPDATA%/ComfyUI/config.json -> "basePath"), so
# folder_paths resolves exactly the files the workflow used.
MODEL_BASE_DIR = Path(r"C:\Users\Blake\Documents\ComfyUI")

# --reserve-vram override, in GB. None = leave ComfyUI's default (0.7 GB on a 16 GB+
# Windows card), which is what the desktop app runs with and therefore what produced
# the reference images. LEAVE THIS AT None unless you have re-read the following.
#
# At 3x (2304x3072) the grid is 2x2 and each tile encodes/decodes a 1216x1600 crop.
# VAE.decode estimates that at ~8.5 GB and asks load_models_gpu for exactly that; with
# the 12.3 GB z-image UNET resident on a 24 GB card the estimate still "fits", so the
# UNET is not evicted and the decode runs with ~10 GB. On an RTX 3090 Ti that is enough
# only just: some decodes overrun and fall back to VAE.decode_tiled_, and across a
# multi-run process one eventually died with cuDNN CUDNN_STATUS_EXECUTION_FAILED
# (which comfy's raise_non_oom does NOT treat as an OOM, so the tiled rescue never ran).
#
# Raising the reserve to force the UNET out before each VAE call looks like the fix and
# is NOT usable here: it pushes the ENCODE into VAE.encode_tiled_, which is broken in
# ComfyUI 0.19.5 — it does `samples += comfy.utils.tiled_scale(...)` on the output of
# an @torch.inference_mode() function, so it raises "Inplace update to inference tensor
# outside InferenceMode is not allowed" every time. (decode_tiled_ adds with `a + b + c`
# and so survives, which is why only the decode fallback ever works.)
#
# What actually works is process isolation: render one run per process (run_ab.py
# --only <label>), so no run inherits another's allocator state. free_gpu() between
# runs in-process helps but is not sufficient on its own at 3x.
RESERVE_VRAM_GB = None

# Memory-path flags the desktop app launches with on this 32 GB-RAM machine. They are
# I/O-path only (no math changes, A/B-safe) and load-bearing for MiniMax H3: its ~20 GB
# streamed DiT over-commits physical RAM when staging is pinned — two harness runs died
# with native SIGSEGV (exit 139) at exactly the DiT-load bracket before these were added.
# --disable-pinned-memory keeps weight staging pageable; --fast-disk streams from NVMe.
MEMORY_ARGS = ("--disable-pinned-memory", "--fast-disk")

# comfy_env ships in the library's repository and in no install. Appended, so its neighbours
# (the library's own ab_env among them) never shadow a module of this folder.
LIBRARY_HARNESS_DIR = Path(r"E:\logit-classifier\tests-AB")

if not (LIBRARY_HARNESS_DIR / "comfy_env.py").is_file():
    raise SystemExit(f"{LIBRARY_HARNESS_DIR} holds no comfy_env.py. Point LIBRARY_HARNESS_DIR at the "
                     "logit-classifier repository's tests-AB folder.")
if str(LIBRARY_HARNESS_DIR) not in sys.path:
    sys.path.append(str(LIBRARY_HARNESS_DIR))

import comfy_env  # noqa: E402

# Probed inside <root>/comfy to tell a z-image-capable tree from an older one.
Z_IMAGE_MARKER = Path("comfy") / "text_encoders" / "z_image.py"

# Searched in order; the first z-image-capable hit wins, else the first hit at all.
#
# ORDER MATTERS. The A/B only means anything if it runs on the SAME ComfyUI the reference
# images came from. ComfyUI-Installs is the current production source install (0.32.0, the
# tree tests/conftest.py also resolves first); verified 2026-08-14 after every previous
# candidate went dead — the old Comfy Desktop resources\ComfyUI layout no longer exists
# ("Comfy Desktop" ships no source tree) and E:\ComfyUI was removed. The stale desktop
# paths stay as last resorts only.
CANDIDATE_ROOTS = (
    Path(r"C:\Users\Blake\ComfyUI-Installs\ComfyUI\ComfyUI"),                   # current install, 0.32.0
    Path(r"C:\Users\Blake\AppData\Local\Programs\ComfyUI\resources\ComfyUI"),   # gone as of 2026-08-14
    Path(r"C:\Users\Blake\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI"),
    REPO_ROOT.parent.parent,
)


def _candidates():
    """Yield (path, has_comfy, supports_z_image) for every candidate root."""
    seen = set()
    for candidate in CANDIDATE_ROOTS:
        root = candidate.resolve()
        if root in seen:
            continue
        seen.add(root)
        yield root, (root / "comfy").is_dir(), (root / Z_IMAGE_MARKER).is_file()


def resolve_root():
    """Return (root, note): the ComfyUI source root to import, plus why it won."""
    env_root = os.environ.get("COMFYUI_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        if not (root / "comfy").is_dir():
            raise SystemExit(f"COMFYUI_ROOT={env_root} has no 'comfy' directory")
        return root, "COMFYUI_ROOT env var"

    checked = list(_candidates())
    for root, has_comfy, z_image in checked:
        if has_comfy and z_image:
            return root, "first z-image-capable candidate"
    for root, has_comfy, _ in checked:
        if has_comfy:
            return root, "WARNING: no z-image-capable root found; falling back (model load will likely fail)"
    raise SystemExit(
        "No ComfyUI root found. Checked:\n"
        + "\n".join(f"  - {r} (comfy={c}, z_image={z})" for r, c, z in checked)
    )


def bootstrap():
    """Put this repo on sys.path, then import comfy from the chosen root with this machine's
    model folder and flags, as comfy_env.bootstrap does for every harness.
    Returns (root, note). Safe to call twice."""
    root, note = resolve_root()
    extra_args = list(MEMORY_ARGS)

    if RESERVE_VRAM_GB is not None:
        extra_args += ["--reserve-vram", str(RESERVE_VRAM_GB)]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    comfy_env.bootstrap(root, MODEL_BASE_DIR, extra_args)
    return root, note


def end_node_execution():
    """The cleanup core's executor runs after every node, for a harness that runs several node executions."""
    comfy_env.end_node_execution()


def caption_preset(instruction, max_tokens, surface=None):
    """A caption preset carrying ONE pinned instruction, for a harness that asks its own
    question rather than the settings file's.

    captions.generate_tile_captions takes a resolved settings block since 2026-08-22, and
    every judged arm here predates the presets: no whole-image style caption, and the
    384-square input budget every render on disk was captioned at. Both are what keeps a
    re-run byte-identical to the arm it is labelled as. `surface` names which conditioning
    the arm builds — nothing reads it today, so it is stated rather than assumed.
    """
    from context_anchored_tile_refine import captions

    # The vision half is the run's own [vision] table; only the caption picture is pinned.
    vision = captions.load_settings().vision
    return captions.Preset(
        surface=captions.VLM_METHOD_VISION_CAPTIONS if surface is None else surface,
        label="pinned",
        vision=dataclasses.replace(vision, caption_megapixels=captions.VL_INPUT_BUDGET_MEGAPIXELS),
        tile_instruction=instruction, tile_max_tokens=max_tokens,
        style_instruction="", style_max_tokens=max_tokens)


def version(root):
    """ComfyUI's reported version string, or '?' if the file is missing."""
    marker = Path(root) / "comfyui_version.py"
    if not marker.is_file():
        return "?"
    scope = {}
    exec(marker.read_text(encoding="utf-8"), scope)  # noqa: S102 - tiny generated file
    return scope.get("__version__", "?")
