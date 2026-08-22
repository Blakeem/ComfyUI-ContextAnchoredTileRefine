# Root-resolution order (env var first), error-message style, and the comfy/utils.py
# vs utils/ shadowing hazard are credited to ComfyUI_UltimateSDUpscaleGuider/test/conftest.py.
import importlib.util
import math
import os
import sys
import types
import uuid
from pathlib import Path
from typing import ClassVar

import pytest
import torch

REPO_ROOT = Path(__file__).parent.parent.resolve()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Comfy Desktop's bundled source tree, tried after the standard custom_nodes layout.
# Ordered newest-install-first: ComfyUI-Installs holds the current source install (the
# only H3-capable tree on this machine); the @comfyorgcomfyui-electron path is a
# pre-rename desktop build (0.3.45, no z-image) that may not exist any more, so it is
# only a last resort.
DESKTOP_COMFYUI_ROOTS = (
    Path(r"C:\Users\Blake\ComfyUI-Installs\ComfyUI\ComfyUI"),
    Path(r"C:\Users\Blake\AppData\Local\Programs\ComfyUI\resources\ComfyUI"),
    Path(r"C:\Users\Blake\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI"),
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
            f"COMFYUI_ROOT={env_root} does not contain a 'comfy' directory"
        )
    trace.append("COMFYUI_ROOT env var: not set")

    standard_root = REPO_ROOT.parent.parent.resolve()
    if (standard_root / "comfy").is_dir():
        return standard_root, trace
    trace.append(f"standard layout {standard_root}: no 'comfy' directory")

    for desktop_root in DESKTOP_COMFYUI_ROOTS:
        if (desktop_root / "comfy").is_dir():
            return desktop_root, trace
        trace.append(f"Desktop install {desktop_root}: no 'comfy' directory")

    return None, trace


@pytest.fixture(autouse=True)
def shipped_settings_file(monkeypatch):
    """Every test reads the SHIPPED settings.toml, never a developer's own copy.

    settings.user.toml is the documented edit surface (README) and wins over the shipped
    file whenever it exists, so a developer who follows that instruction would otherwise
    turn the whole gate red on prompts they changed deliberately. Renaming the constant is
    what neutralizes it: `captions.settings_path` still runs, and a test that needs the
    override mechanism itself restores the real name at its own tmp_path.
    The method list is cached for a session by design, so it is cleared around every test.
    """
    from context_anchored_tile_refine import captions

    monkeypatch.setattr(captions, "USER_SETTINGS_NAME", "settings.user.toml.absent-in-tests")
    captions.vlm_methods.cache_clear()
    yield
    captions.vlm_methods.cache_clear()


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
            + "\n".join(f"  - {step}" for step in trace)
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
            f"A different 'comfy' package shadows the ComfyUI source at {root} "
            f"(comfy.samplers resolved to: {origin}). Set COMFYUI_ROOT or remove the "
            "conflicting package."
        )
    return root


class BaseModel:
    """comfy's BaseModel stand-in, shared by every test that drives a guider's model.

    The class NAME is load-bearing: sampling._model_re_noises_anchor walks __mro__ for the
    class that DEFINES scale_latent_inpaint and answers True only for "BaseModel" — core's
    default, which the anchor-ring schedule and the sync engine's precondition are both
    derived for.

    process_latent_in/out are the CANVAS SPACE pair (roadmap decision): comfy applies
    process_latent_in to latent_image inside guider.sample, so the engine's maintained canvas
    lives in process-space while VAE encode/decode are raw-space. The default scale/shift make
    the pair the IDENTITY; a test passes a non-identity affine (what Krea 2's Wan21 format
    really is) to prove the two spaces are never mixed.
    """

    def __init__(self, scale=1.0, shift=0.0, model_sampling=None):
        # Lazy, so the class resolves comfy.model_sampling.CONST through whichever module is
        # live — the comfy_stubs one in the pure suite, the real one under comfy_env.
        import comfy.model_sampling

        self.latent_format = object()
        self.model_sampling = comfy.model_sampling.CONST() if model_sampling is None else model_sampling
        self.scale = scale
        self.shift = shift
        self.process_in_calls = []
        self.process_out_calls = []

    def scale_latent_inpaint(self, sigma, noise, latent_image, **kwargs):
        return self.model_sampling.noise_scaling(sigma, noise, latent_image)

    def process_latent_in(self, latent):
        self.process_in_calls.append(latent)
        return latent * self.scale + self.shift

    def process_latent_out(self, latent):
        self.process_out_calls.append(latent)
        return (latent - self.shift) / self.scale


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
        "common_upscale_calls": [],
        "tiled_scale_calls": [],
        "load_models_gpu_calls": [],
        "free_memory_calls": [],
        "raise_non_oom_calls": [],
        "calculate_sigmas_calls": [],
        "sampler_object_calls": [],
        "prepare_noise_calls": [],
        "convert_cond_calls": [],
        "guiders": [],
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

    def common_upscale(samples, width, height, upscale_method, crop):
        # Recording fake: shape/method/crop are what the callers are pinned on, and the
        # zeros return is deliberately NOT the input, so "did it resize?" is decidable.
        recorded["common_upscale_calls"].append(
            (tuple(samples.shape), width, height, upscale_method, crop)
        )
        return torch.zeros(
            (samples.shape[0], samples.shape[1], height, width),
            dtype=samples.dtype,
            device=samples.device,
        )

    utils_module.common_upscale = common_upscale

    def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
        # comfy.utils.get_tiled_scale_steps verbatim, so a halved tile really does change
        # the ProgressBar total the caller derives from it.
        rows = 1 if height <= tile_y else math.ceil((height - overlap) / (tile_y - overlap))
        cols = 1 if width <= tile_x else math.ceil((width - overlap) / (tile_x - overlap))
        return rows * cols

    utils_module.get_tiled_scale_steps = get_tiled_scale_steps

    def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap=8, upscale_amount=4, output_device="cpu", pbar=None, **kwargs):
        # Records the tile geometry (the value the OOM retry halves) and runs `function`
        # over the whole tensor in one go — no tiling — so a fake model's output IS the
        # upscale and a fake model raising drives the caller's OOM path.
        recorded["tiled_scale_calls"].append(
            {"tile_x": tile_x, "tile_y": tile_y, "overlap": overlap, "upscale_amount": upscale_amount}
        )
        return function(samples).to(output_device)

    utils_module.tiled_scale = tiled_scale

    model_management_module = types.ModuleType("comfy.model_management")
    model_management_module.intermediate_device = lambda: torch.device("cpu")
    model_management_module.intermediate_dtype = lambda: torch.float32
    model_management_module.get_torch_device = lambda: torch.device("cpu")

    def free_memory(memory_required, device):
        recorded["free_memory_calls"].append((memory_required, device))

    model_management_module.free_memory = free_memory

    def load_models_gpu(models, memory_required=0, force_full_load=False):
        recorded["load_models_gpu_calls"].append((list(models), memory_required, force_full_load))

    model_management_module.load_models_gpu = load_models_gpu

    def raise_non_oom(error):
        # comfy.model_management.raise_non_oom's contract: anything that is not an
        # out-of-memory condition propagates instead of triggering a retry. The real one
        # tests exception types the stub environment has no way to raise, so the stub keys
        # off the message instead.
        recorded["raise_non_oom_calls"].append(error)
        if "out of memory" not in str(error).lower():
            raise error

    model_management_module.raise_non_oom = raise_non_oom

    def throw_exception_if_processing_interrupted():
        recorded["interrupt_calls"] += 1

    model_management_module.throw_exception_if_processing_interrupted = (
        throw_exception_if_processing_interrupted
    )
    samplers_module = types.ModuleType("comfy.samplers")

    class StubKSampler:
        # Only the two combo lists node.py's INPUT_TYPES reads. Short on purpose, but they
        # must contain every default the node declares or the widget would be invalid.
        SAMPLERS: ClassVar[list[str]] = ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde", "ddim", "res_multistep"]
        SCHEDULERS: ClassVar[list[str]] = ["normal", "karras", "simple", "sgm_uniform", "beta"]

    samplers_module.KSampler = StubKSampler

    class StubKSAMPLER:
        # comfy.samplers.KSAMPLER's construction surface: the sync stepper isinstance-checks
        # this class, keys its EVALS_PER_STEP table on sampler_function.__name__, and carries
        # extra_options/inpaint_options onto every lane's rebuilt sampler.
        def __init__(self, sampler_function, extra_options=None, inpaint_options=None):
            self.sampler_function = sampler_function
            self.extra_options = {} if extra_options is None else extra_options
            self.inpaint_options = {} if inpaint_options is None else inpaint_options

    samplers_module.KSAMPLER = StubKSAMPLER

    def calculate_sigmas(model_sampling, scheduler_name, steps):
        # Returns 0..steps so the caller's tail slice is readable as plain integers.
        recorded["calculate_sigmas_calls"].append((model_sampling, scheduler_name, steps))
        return torch.arange(steps + 1, dtype=torch.float32)

    samplers_module.calculate_sigmas = calculate_sigmas

    def sampler_object(name):
        # Core returns a KSAMPLER whose sampler_function is sample_<name> (comfy/samplers.py
        # ksampler); the stub mirrors that, because the sync stepper resolves the sampler by
        # that function's __name__ and rejects anything that is not a KSAMPLER.
        recorded["sampler_object_calls"].append(name)

        def sampler_function(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
            raise AssertionError("the stub sampler function is never run")

        sampler_function.__name__ = f"sample_{name}"
        return StubKSAMPLER(sampler_function)

    samplers_module.sampler_object = sampler_object

    class StubCFGGuider:
        """comfy.samplers.CFGGuider's construction surface only: the conds land in
        `original_conds` under 'positive'/'negative', which is the convention
        sampling.refine_image's VL check and per-tile positive swap both rely on."""

        def __init__(self, model_patcher):
            self.model_patcher = model_patcher
            self.model_options = {}
            self.original_conds = {}
            self.cfg = 1.0
            recorded["guiders"].append(self)

        def set_conds(self, positive, negative):
            self.inner_set_conds({"positive": positive, "negative": negative})

        def inner_set_conds(self, conds):
            # comfy/samplers.py:1201-1205, the seam upscale.build_basic_guider's Guider_Basic
            # subclass reaches set_conds through. The real one runs convert_cond on the way
            # in; the stub stores what it was handed so callers can assert identity.
            self.original_conds.update(conds)

        def set_cfg(self, cfg):
            self.cfg = cfg

    samplers_module.CFGGuider = StubCFGGuider

    model_sampling_module = types.ModuleType("comfy.model_sampling")

    class StubCONST:
        """comfy.model_sampling.CONST cut to what the engine touches: the TYPE the sync
        preconditions isinstance-check (the flow family), and the noise_scaling /
        inverse_noise_scaling pair the frozen-ring algebra is derived from."""

        def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
            return sigma * noise + (1.0 - sigma) * latent_image

        def inverse_noise_scaling(self, sigma, latent):
            return latent / (1.0 - sigma)

    model_sampling_module.CONST = StubCONST

    sampler_helpers_module = types.ModuleType("comfy.sampler_helpers")

    def convert_cond(cond):
        # comfy.sampler_helpers.convert_cond's body, minus nothing that matters here: the
        # extras dict is COPIED, cross_attn comes from element 0, model_conds is ensured and
        # every cond gets a fresh uuid. vl._convert calls this to turn each tile's sliced
        # [tensor, extras] pair into the flat-dict cond core's samplers consume.
        recorded["convert_cond_calls"].append(cond)
        out = []
        for c in cond:
            temp = c[1].copy()
            model_conds = temp.get("model_conds", {})
            if c[0] is not None:
                temp["cross_attn"] = c[0]
            temp["model_conds"] = model_conds
            temp["uuid"] = uuid.uuid4()
            out.append(temp)
        return out

    sampler_helpers_module.convert_cond = convert_cond

    nested_tensor_module = types.ModuleType("comfy.nested_tensor")

    class StubNestedTensor:
        """comfy/nested_tensor.py NestedTensor cut to the surface the video path touches:
        the stream list, the `is_nested` flag core's sample() and previewer branch on, and
        unbind()."""

        def __init__(self, tensors):
            self.tensors = list(tensors)
            self.is_nested = True

        def unbind(self):
            return self.tensors

    nested_tensor_module.NestedTensor = StubNestedTensor

    sample_module = types.ModuleType("comfy.sample")

    def prepare_noise(latent_image, seed, batch_inds=None):
        recorded["prepare_noise_calls"].append((latent_image, seed, batch_inds))
        if getattr(latent_image, "is_nested", False):
            # comfy/sample.py:29-34 draws one noise tensor per stream of a nested latent.
            return StubNestedTensor([torch.zeros_like(t) for t in latent_image.unbind()])
        return torch.zeros_like(latent_image)

    sample_module.prepare_noise = prepare_noise

    # 'import comfy.utils' attribute-resolves through the hand-inserted parent module.
    comfy_module.utils = utils_module
    comfy_module.model_management = model_management_module
    comfy_module.samplers = samplers_module
    comfy_module.sample = sample_module
    comfy_module.sampler_helpers = sampler_helpers_module
    comfy_module.model_sampling = model_sampling_module
    comfy_module.nested_tensor = nested_tensor_module

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
    monkeypatch.setitem(sys.modules, "comfy.samplers", samplers_module)
    monkeypatch.setitem(sys.modules, "comfy.sample", sample_module)
    monkeypatch.setitem(sys.modules, "comfy.sampler_helpers", sampler_helpers_module)
    monkeypatch.setitem(sys.modules, "comfy.model_sampling", model_sampling_module)
    monkeypatch.setitem(sys.modules, "comfy.nested_tensor", nested_tensor_module)
    monkeypatch.setitem(sys.modules, "latent_preview", latent_preview_module)
    return recorded
