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

# Memory-path flags the desktop app launches with on this 32 GB-RAM machine. They are
# I/O-path only (no math changes, A/B-safe) and load-bearing for MiniMax H3: its ~20 GB
# streamed DiT over-commits physical RAM when staging is pinned — two harness runs died
# with native SIGSEGV (exit 139) at exactly the DiT-load bracket before these were added.
# --disable-pinned-memory keeps weight staging pageable; --fast-disk streams from NVMe.
MEMORY_ARGS = ("--disable-pinned-memory", "--fast-disk")

# (enabled, why) of the one-per-process DynamicVRAM bring-up; see _enable_dynamic_vram.
_DYNAMIC_VRAM = None

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


def _enable_dynamic_vram():
    """Mirror main.py's DynamicVRAM (comfy_aimdo) enablement. Returns (enabled, why).

    SOURCE OF TRUTH: <root>/main.py — its module-level `enables_dynamic_vram()` block
    (control.init) plus the `init_devices` block that follows the comfy.model_management
    import. Re-read BOTH when the pinned root moves; nothing else in the tree ever
    assigns `comfy.memory_management.aimdo_enabled`, and that flag gates the streaming
    weight loader (comfy/utils.py), the free-memory budget (model_patcher.py), three
    load branches (model_management.py), every --fast-disk staging path (ops.py,
    model_prefetch.py) and VAE offload (sd.py). Without it the harness runs a different
    memory path than the app it is supposed to reproduce."""
    from comfy.cli_args import args, enables_dynamic_vram
    import comfy_aimdo.control

    # main.py module level: control.init() before any torch device init.
    if enables_dynamic_vram():
        simple_vram_headroom = None if args.reserve_vram is None else int(args.reserve_vram * 1024 ** 3)
        try:
            comfy_aimdo.control.init(simple_vram_headroom=simple_vram_headroom,
                                     nvml_pressure=not args.disable_nvml_pressure)
        except TypeError:
            # comfy-aimdo 0.4.10 protocol.
            try:
                comfy_aimdo.control.init(simple_vram_headroom=simple_vram_headroom)
            except TypeError:
                # comfy-aimdo 0.4.9 protocol.
                comfy_aimdo.control.init()

    # main.py after its `import comfy.model_management`: device init, then the two globals.
    import comfy.memory_management
    import comfy.model_management
    import comfy.model_patcher

    if not (args.enable_dynamic_vram or (enables_dynamic_vram()
                                         and comfy.model_management.is_nvidia()
                                         and not comfy.model_management.is_wsl())):
        return False, "not enabled for this device/flags (main.py's own condition)"
    if (not args.enable_dynamic_vram) and (comfy.model_management.torch_version_numeric < (2, 8)):
        return False, "torch {} < 2.8".format(comfy.model_management.torch_version_numeric)

    try:
        aimdo_initialized = comfy_aimdo.control.init_devices(
            (d.index, int(args.vram_headroom * 1024 ** 3))
            for d in comfy.model_management.get_all_torch_devices())
    except TypeError:
        # comfy-aimdo 0.4.9 protocol.
        aimdo_initialized = comfy_aimdo.control.init_devices(
            d.index for d in comfy.model_management.get_all_torch_devices())
    if not aimdo_initialized:
        return False, "comfy_aimdo.control.init_devices() reported no working install"

    comfy_aimdo.control.set_log_info()   # main.py's default console log level branch
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True
    return True, "comfy_aimdo devices initialized"


def bootstrap():
    """Put the ComfyUI root + this repo on sys.path, point folder_paths at the real
    model directory, and bring up DynamicVRAM the way main.py does.
    Returns (root, note). Safe to call twice."""
    # main.py sets this at module level, before native/torch init; the top of bootstrap()
    # is the same point relative to native init here.
    if os.name == "nt":
        os.environ["MIMALLOC_PURGE_DELAY"] = "0"

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
    sys.argv = [saved_argv[0], "--base-directory", str(MODEL_BASE_DIR), *MEMORY_ARGS]
    if RESERVE_VRAM_GB is not None:
        sys.argv += ["--reserve-vram", str(RESERVE_VRAM_GB)]
    try:
        import folder_paths  # noqa: F401  (import side effect is the point)
    finally:
        sys.argv = saved_argv

    # cli_args parses argv exactly once, at its own import, and says nothing when it
    # already ran: anything that imported comfy.cli_args / folder_paths before this
    # leaves base_directory None and every MEMORY_ARGS flag False, silently. Read the
    # parsed state back instead of trusting the side effect.
    from comfy.cli_args import args as parsed_args

    if parsed_args.base_directory != str(MODEL_BASE_DIR):
        raise SystemExit(
            "comfy.cli_args was parsed before bootstrap(): --base-directory is {!r}, "
            "expected {!r}. Nothing may import comfy/folder_paths first.".format(
                parsed_args.base_directory, str(MODEL_BASE_DIR)))
    for flag in MEMORY_ARGS:
        attribute = flag.lstrip("-").replace("-", "_")
        if not getattr(parsed_args, attribute, False):
            raise SystemExit(
                "comfy.cli_args was parsed before bootstrap(): {} never took effect "
                "(args.{} is not set). Nothing may import comfy/folder_paths first.".format(
                    flag, attribute))

    # Attempted once per process — main.py runs these stages once, and bootstrap()
    # must stay safe to call twice.
    global _DYNAMIC_VRAM
    if _DYNAMIC_VRAM is None:
        try:
            _DYNAMIC_VRAM = _enable_dynamic_vram()
        except Exception as exc:   # comfy_aimdo missing, or any init raising
            _DYNAMIC_VRAM = (False, "{}: {}".format(type(exc).__name__, exc))
    dynamic_vram, why = _DYNAMIC_VRAM
    if dynamic_vram:
        print("DynamicVRAM (comfy_aimdo) enabled, as main.py does: {}".format(why))
    else:
        print("WARNING: DynamicVRAM (comfy_aimdo) NOT enabled — legacy ModelPatcher, "
              "estimate-based VRAM accounting, --fast-disk inert: {}".format(why))
    return root, note


def version(root):
    """ComfyUI's reported version string, or '?' if the file is missing."""
    marker = Path(root) / "comfyui_version.py"
    if not marker.is_file():
        return "?"
    scope = {}
    exec(marker.read_text(encoding="utf-8"), scope)  # noqa: S102 - tiny generated file
    return scope.get("__version__", "?")
