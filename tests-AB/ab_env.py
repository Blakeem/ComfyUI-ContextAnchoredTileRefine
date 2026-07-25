"""ComfyUI bootstrap for the standalone A/B harness (no ComfyUI server).

`bootstrap()` MUST run before the first `comfy` / `comfy_extras` / `folder_paths`
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
import importlib.util
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

# Probed inside <root>/comfy to tell a z-image-capable tree from an older one.
Z_IMAGE_MARKER = Path("comfy") / "text_encoders" / "z_image.py"

# Searched in order; the first z-image-capable hit wins, else the first hit at all.
#
# ORDER MATTERS. The A/B only means anything if it runs on the SAME ComfyUI the reference
# images came from. Comfy Desktop (0.22.3, "Comfy Desktop.exe") is the production install —
# it is the one %APPDATA%\ComfyUI\config.json points at basePath C:\Users\Blake\Documents\
# ComfyUI. E:\ComfyUI is an older 0.19.5 desktop install that is ALSO z-image-capable, so
# listing it first silently won and the harness rendered against the wrong version.
CANDIDATE_ROOTS = (
    Path(r"C:\Users\Blake\AppData\Local\Programs\ComfyUI\resources\ComfyUI"),   # Comfy Desktop 0.22.3
    Path(r"E:\ComfyUI\resources\ComfyUI"),                                      # older 0.19.5
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
            raise SystemExit("COMFYUI_ROOT={} has no 'comfy' directory".format(env_root))
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
        + "\n".join("  - {} (comfy={}, z_image={})".format(r, c, z) for r, c, z in checked)
    )


def bootstrap():
    """Put the ComfyUI root + this repo on sys.path and point folder_paths at the
    real model directory. Returns (root, note). Safe to call twice."""
    root, note = resolve_root()
    for path in (str(REPO_ROOT), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    # A regular 'comfy' package elsewhere (comfy_cli ships one) beats the source
    # tree regardless of sys.path order — verify the intended source resolves.
    spec = importlib.util.find_spec("comfy.samplers")
    origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
    if origin is None or origin.parent != (root / "comfy").resolve():
        raise SystemExit(
            "A different 'comfy' package shadows the ComfyUI source at {} "
            "(comfy.samplers resolved to: {}).".format(root, origin)
        )

    # folder_paths reads comfy.cli_args.args at import; cli_args only parses argv
    # once comfy.options.enable_args_parsing() has been called (main.py does this).
    # Enable it and hand it a synthetic argv so --base-directory is honoured and our
    # own flags are never seen by ComfyUI's parser.
    import comfy.options

    comfy.options.enable_args_parsing()
    saved_argv = sys.argv
    sys.argv = [saved_argv[0], "--base-directory", str(MODEL_BASE_DIR)]
    if RESERVE_VRAM_GB is not None:
        sys.argv += ["--reserve-vram", str(RESERVE_VRAM_GB)]
    try:
        import folder_paths  # noqa: F401  (import side effect is the point)
    finally:
        sys.argv = saved_argv
    return root, note


def version(root):
    """ComfyUI's reported version string, or '?' if the file is missing."""
    marker = Path(root) / "comfyui_version.py"
    if not marker.is_file():
        return "?"
    scope = {}
    exec(marker.read_text(encoding="utf-8"), scope)  # noqa: S102 - tiny generated file
    return scope.get("__version__", "?")
