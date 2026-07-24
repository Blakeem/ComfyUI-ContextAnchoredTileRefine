# Root-resolution order (env var first), error-message style, and the comfy/utils.py
# vs utils/ shadowing hazard are credited to ComfyUI_UltimateSDUpscaleGuider/test/conftest.py.
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).parent.parent.resolve()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DESKTOP_COMFYUI_ROOT = Path(
    r"C:\Users\Blake\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI"
)


def _resolve_comfyui_root():
    """Return (root, trace): the ComfyUI source root, or None with the paths checked."""
    trace = []

    env_root = os.environ.get("COMFYUI_ROOT")
    if env_root:
        path = Path(env_root).resolve()
        if (path / "comfy").is_dir():
            return path, trace
        raise ValueError(
            "COMFYUI_ROOT={} does not contain a 'comfy' directory".format(env_root)
        )
    trace.append("COMFYUI_ROOT env var: not set")

    standard_root = REPO_ROOT.parent.parent.resolve()
    if (standard_root / "comfy").is_dir():
        return standard_root, trace
    trace.append("standard layout {}: no 'comfy' directory".format(standard_root))

    if (DESKTOP_COMFYUI_ROOT / "comfy").is_dir():
        return DESKTOP_COMFYUI_ROOT, trace
    trace.append("Desktop install {}: no 'comfy' directory".format(DESKTOP_COMFYUI_ROOT))

    return None, trace


@pytest.fixture(scope="session")
def comfy_env():
    """Resolve a real ComfyUI install and put its root on sys.path.

    Only <root> is inserted — never <root>/comfy, whose utils.py would shadow
    ComfyUI's top-level utils/ package.
    """
    root, trace = _resolve_comfyui_root()
    if root is None:
        pytest.skip(
            "ComfyUI root not found. Checked:\n"
            + "\n".join("  - {}".format(step) for step in trace)
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # A regular 'comfy' package elsewhere (e.g. comfy_cli's comfy 0.0.1 in
    # site-packages) beats the source tree's PEP 420 namespace portion regardless
    # of sys.path order — verify the intended source is what actually resolves.
    spec = importlib.util.find_spec("comfy.samplers")
    origin = Path(spec.origin).resolve() if spec is not None and spec.origin else None
    if origin is None or origin.parent != (root / "comfy").resolve():
        pytest.skip(
            "A different 'comfy' package shadows the ComfyUI source at {} "
            "(comfy.samplers resolved to: {}). Set COMFYUI_ROOT or remove the "
            "conflicting package.".format(root, origin)
        )
    return root


@pytest.fixture()
def comfy_stubs(monkeypatch):
    """Stub comfy/latent_preview in sys.modules so sampling.py's lazy imports resolve
    without a real ComfyUI install. Returns a dict recording stub activity."""
    recorded = {
        "prepare_callback_args": None,
        "callback": None,
        "callback_calls": [],
        "progress_bars": [],
        "get_previewer_args": None,
        "previewer": None,  # tests may set a stub before calling refine_image
        "interrupt_calls": 0,
    }

    comfy_module = types.ModuleType("comfy")
    utils_module = types.ModuleType("comfy.utils")
    utils_module.PROGRESS_BAR_ENABLED = True

    class RecordingProgressBar:
        def __init__(self, total):
            self.total = total
            self.updates = []
            recorded["progress_bars"].append(self)

        def update_absolute(self, value, total=None, preview=None):
            self.updates.append((value, total, preview))

    utils_module.ProgressBar = RecordingProgressBar

    model_management_module = types.ModuleType("comfy.model_management")
    model_management_module.intermediate_device = lambda: torch.device("cpu")

    def throw_exception_if_processing_interrupted():
        recorded["interrupt_calls"] += 1

    model_management_module.throw_exception_if_processing_interrupted = (
        throw_exception_if_processing_interrupted
    )
    # 'import comfy.utils' attribute-resolves through the hand-inserted parent module.
    comfy_module.utils = utils_module
    comfy_module.model_management = model_management_module

    latent_preview_module = types.ModuleType("latent_preview")

    def get_previewer(device, latent_format):
        recorded["get_previewer_args"] = (device, latent_format)
        return recorded["previewer"]

    latent_preview_module.get_previewer = get_previewer

    def prepare_callback(model, steps, x0_output_dict=None):
        recorded["prepare_callback_args"] = (model, steps, x0_output_dict)

        def callback(step, x0, x, total_steps):
            recorded["callback_calls"].append((step, total_steps))

        recorded["callback"] = callback
        return callback

    latent_preview_module.prepare_callback = prepare_callback

    monkeypatch.setitem(sys.modules, "comfy", comfy_module)
    monkeypatch.setitem(sys.modules, "comfy.utils", utils_module)
    monkeypatch.setitem(sys.modules, "comfy.model_management", model_management_module)
    monkeypatch.setitem(sys.modules, "latent_preview", latent_preview_module)
    return recorded
