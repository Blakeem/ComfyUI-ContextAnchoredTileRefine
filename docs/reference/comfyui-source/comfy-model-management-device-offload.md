# comfy/model_management.py — device/offload basics (load_models_gpu, get_free_memory, unet/vae offload devices, module load-state)

**Source:** `comfy/model_management.py` (multiple extracts — line ranges cited per section below)
**ComfyUI version:** 0.3.45 (local source tree — `C:\Users\Blake\AppData\Local\Programs\@comfyorgcomfyui-electron\resources\ComfyUI`)
**Retrieved:** 2026-07-22

Extracted for: the device/offload/model-residency mechanics relevant to a custom node that calls
`guider.sample(...)` repeatedly in a loop (once per tile) — what `load_models_gpu` does before
sampling, how free VRAM is queried, where the UNet/VAE load vs. offload devices come from, and the
module-level state (`current_loaded_models`, `LoadedModel`, `vram_state`) that tracks model
residency between calls. Interrupt handling
(`InterruptProcessingException` / `throw_exception_if_processing_interrupted`) is captured
separately in `comfy-model-management-interrupt.md` and is not repeated here.
Hardware-backend detection (directml/xpu/npu/mlu availability, `should_use_fp16`/`should_use_bf16`,
xformers/attention flags, fp8 dtype selection) is out of scope for this capture and omitted.

## VRAMState / CPUState enums + initial module state (comfy/model_management.py, lines 29–47)

```python
class VRAMState(Enum):
    DISABLED = 0    #No vram present: no need to move models to vram
    NO_VRAM = 1     #Very low vram: enable all the options to save vram
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5      #No dedicated vram: memory shared between CPU and GPU but models still need to be moved between both.

class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2

# Determine VRAM State
vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU

total_vram = 0
```

`vram_state` is the module-global that `unet_offload_device`, `unet_inital_load_device`,
`load_models_gpu`, etc. all read to decide load/offload behavior. It gets its final value from the
CLI-args resolution block below (lines 341–365) after hardware-backend detection (not reproduced
here — see lines 75–133 of the file for directml/xpu/npu/mlu/mps detection, which sets `cpu_state`).

## is_nvidia() + MIN_WEIGHT_MEMORY_RATIO (comfy/model_management.py, lines 263–279)

```python
def is_nvidia():
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.cuda:
            return True
    return False

def is_amd():
    global cpu_state
    if cpu_state == CPUState.GPU:
        if torch.version.hip:
            return True
    return False

MIN_WEIGHT_MEMORY_RATIO = 0.4
if is_nvidia():
    MIN_WEIGHT_MEMORY_RATIO = 0.0
```

`MIN_WEIGHT_MEMORY_RATIO` feeds directly into the low-vram weight budget calculation inside
`load_models_gpu` (see below).

## VRAM state resolution from CLI args (comfy/model_management.py, lines 341–365)

```python
if args.lowvram:
    set_vram_to = VRAMState.LOW_VRAM
    lowvram_available = True
elif args.novram:
    set_vram_to = VRAMState.NO_VRAM
elif args.highvram or args.gpu_only:
    vram_state = VRAMState.HIGH_VRAM

FORCE_FP32 = False
if args.force_fp32:
    logging.info("Forcing FP32, if this improves things please report it.")
    FORCE_FP32 = True

if lowvram_available:
    if set_vram_to in (VRAMState.LOW_VRAM, VRAMState.NO_VRAM):
        vram_state = set_vram_to


if cpu_state != CPUState.GPU:
    vram_state = VRAMState.DISABLED

if cpu_state == CPUState.MPS:
    vram_state = VRAMState.SHARED

logging.info(f"Set vram state to: {vram_state.name}")
```

## DISABLE_SMART_MEMORY (comfy/model_management.py, lines 367–370)

```python
DISABLE_SMART_MEMORY = args.disable_smart_memory

if DISABLE_SMART_MEMORY:
    logging.info("Disabling smart memory management")
```

`DISABLE_SMART_MEMORY` gates a fast-path in `free_memory` (skip the "is there already enough free
memory" check and always compute `memory_to_free`) and disables the free-vram short-circuit in
`unet_inital_load_device`.

## get_torch_device() (comfy/model_management.py, lines 154–172)

```python
def get_torch_device():
    global directml_enabled
    global cpu_state
    if directml_enabled:
        global directml_device
        return directml_device
    if cpu_state == CPUState.MPS:
        return torch.device("mps")
    if cpu_state == CPUState.CPU:
        return torch.device("cpu")
    else:
        if is_intel_xpu():
            return torch.device("xpu", torch.xpu.current_device())
        elif is_ascend_npu():
            return torch.device("npu", torch.npu.current_device())
        elif is_mlu():
            return torch.device("mlu", torch.mlu.current_device())
        else:
            return torch.device(torch.cuda.current_device())
```

This is the "current compute device" primitive that `unet_offload_device`, `unet_inital_load_device`,
`vae_device`, `vae_offload_device`, `intermediate_device` (when `args.gpu_only`), and
`get_free_memory`/`get_total_memory` (default `dev=None`) all fall back to.

## current_loaded_models — module load-state list (comfy/model_management.py, line 397)

```python
current_loaded_models = []
```

The single module-global list of `LoadedModel` wrappers tracking every model currently resident
(loaded or partially loaded) on any device. `load_models_gpu`, `free_memory`, `loaded_models`,
`cleanup_models_gc`, and `cleanup_models` all read/mutate this list directly.

## LoadedModel class (comfy/model_management.py, lines 407–493)

```python
class LoadedModel:
    def __init__(self, model):
        self._set_model(model)
        self.device = model.load_device
        self.real_model = None
        self.currently_used = True
        self.model_finalizer = None
        self._patcher_finalizer = None

    def _set_model(self, model):
        self._model = weakref.ref(model)
        if model.parent is not None:
            self._parent_model = weakref.ref(model.parent)
            self._patcher_finalizer = weakref.finalize(model, self._switch_parent)

    def _switch_parent(self):
        model = self._parent_model()
        if model is not None:
            self._set_model(model)

    @property
    def model(self):
        return self._model()

    def model_memory(self):
        return self.model.model_size()

    def model_loaded_memory(self):
        return self.model.loaded_size()

    def model_offloaded_memory(self):
        return self.model.model_size() - self.model.loaded_size()

    def model_memory_required(self, device):
        if device == self.model.current_loaded_device():
            return self.model_offloaded_memory()
        else:
            return self.model_memory()

    def model_load(self, lowvram_model_memory=0, force_patch_weights=False):
        self.model.model_patches_to(self.device)
        self.model.model_patches_to(self.model.model_dtype())

        # if self.model.loaded_size() > 0:
        use_more_vram = lowvram_model_memory
        if use_more_vram == 0:
            use_more_vram = 1e32
        self.model_use_more_vram(use_more_vram, force_patch_weights=force_patch_weights)
        real_model = self.model.model

        if is_intel_xpu() and not args.disable_ipex_optimize and 'ipex' in globals() and real_model is not None:
            with torch.no_grad():
                real_model = ipex.optimize(real_model.eval(), inplace=True, graph_mode=True, concat_linear=True)

        self.real_model = weakref.ref(real_model)
        self.model_finalizer = weakref.finalize(real_model, cleanup_models)
        return real_model

    def should_reload_model(self, force_patch_weights=False):
        if force_patch_weights and self.model.lowvram_patch_counter() > 0:
            return True
        return False

    def model_unload(self, memory_to_free=None, unpatch_weights=True):
        if memory_to_free is not None:
            if memory_to_free < self.model.loaded_size():
                freed = self.model.partially_unload(self.model.offload_device, memory_to_free)
                if freed >= memory_to_free:
                    return False
        self.model.detach(unpatch_weights)
        self.model_finalizer.detach()
        self.model_finalizer = None
        self.real_model = None
        return True

    def model_use_more_vram(self, extra_memory, force_patch_weights=False):
        return self.model.partially_load(self.device, extra_memory, force_patch_weights=force_patch_weights)

    def __eq__(self, other):
        return self.model is other.model

    def __del__(self):
        if self._patcher_finalizer is not None:
            self._patcher_finalizer.detach()

    def is_dead(self):
        return self.real_model() is not None and self.model is None
```

`LoadedModel` wraps a `ModelPatcher` (`self.model`, held via `weakref.ref`) and records which
`device` it's meant to live on. `model_size()`, `loaded_size()`, `current_loaded_device()`,
`model_patches_to()`, `model_dtype()`, `lowvram_patch_counter()`, `detach()`, `partially_unload()`,
`partially_load()`, and `is_clone()` are all `ModelPatcher` methods defined in
`comfy/model_patcher.py` (not part of this capture) — `LoadedModel` only orchestrates calls to them
and tracks residency (`self.device`, `self.currently_used`, `self.real_model`).

## Reserved-memory / minimum-inference-memory helpers (comfy/model_management.py, lines 510–524)

```python
WINDOWS = any(platform.win32_ver())

EXTRA_RESERVED_VRAM = 400 * 1024 * 1024
if WINDOWS:
    EXTRA_RESERVED_VRAM = 600 * 1024 * 1024 #Windows is higher because of the shared vram issue

if args.reserve_vram is not None:
    EXTRA_RESERVED_VRAM = args.reserve_vram * 1024 * 1024 * 1024
    logging.debug("Reserving {}MB vram for other applications.".format(EXTRA_RESERVED_VRAM / (1024 * 1024)))

def extra_reserved_memory():
    return EXTRA_RESERVED_VRAM

def minimum_inference_memory():
    return (1024 * 1024 * 1024) * 0.8 + extra_reserved_memory()
```

## free_memory() (comfy/model_management.py, lines 526–561)

```python
def free_memory(memory_required, device, keep_loaded=[]):
    cleanup_models_gc()
    unloaded_model = []
    can_unload = []
    unloaded_models = []

    for i in range(len(current_loaded_models) -1, -1, -1):
        shift_model = current_loaded_models[i]
        if shift_model.device == device:
            if shift_model not in keep_loaded and not shift_model.is_dead():
                can_unload.append((-shift_model.model_offloaded_memory(), sys.getrefcount(shift_model.model), shift_model.model_memory(), i))
                shift_model.currently_used = False

    for x in sorted(can_unload):
        i = x[-1]
        memory_to_free = None
        if not DISABLE_SMART_MEMORY:
            free_mem = get_free_memory(device)
            if free_mem > memory_required:
                break
            memory_to_free = memory_required - free_mem
        logging.debug(f"Unloading {current_loaded_models[i].model.model.__class__.__name__}")
        if current_loaded_models[i].model_unload(memory_to_free):
            unloaded_model.append(i)

    for i in sorted(unloaded_model, reverse=True):
        unloaded_models.append(current_loaded_models.pop(i))

    if len(unloaded_model) > 0:
        soft_empty_cache()
    else:
        if vram_state != VRAMState.HIGH_VRAM:
            mem_free_total, mem_free_torch = get_free_memory(device, torch_free_too=True)
            if mem_free_torch > mem_free_total * 0.25:
                soft_empty_cache()
    return unloaded_models
```

## load_models_gpu() / load_model_gpu() / loaded_models() (comfy/model_management.py, lines 563–650)

```python
def load_models_gpu(models, memory_required=0, force_patch_weights=False, minimum_memory_required=None, force_full_load=False):
    cleanup_models_gc()
    global vram_state

    inference_memory = minimum_inference_memory()
    extra_mem = max(inference_memory, memory_required + extra_reserved_memory())
    if minimum_memory_required is None:
        minimum_memory_required = extra_mem
    else:
        minimum_memory_required = max(inference_memory, minimum_memory_required + extra_reserved_memory())

    models = set(models)

    models_to_load = []

    for x in models:
        loaded_model = LoadedModel(x)
        try:
            loaded_model_index = current_loaded_models.index(loaded_model)
        except:
            loaded_model_index = None

        if loaded_model_index is not None:
            loaded = current_loaded_models[loaded_model_index]
            loaded.currently_used = True
            models_to_load.append(loaded)
        else:
            if hasattr(x, "model"):
                logging.info(f"Requested to load {x.model.__class__.__name__}")
            models_to_load.append(loaded_model)

    for loaded_model in models_to_load:
        to_unload = []
        for i in range(len(current_loaded_models)):
            if loaded_model.model.is_clone(current_loaded_models[i].model):
                to_unload = [i] + to_unload
        for i in to_unload:
            current_loaded_models.pop(i).model.detach(unpatch_all=False)

    total_memory_required = {}
    for loaded_model in models_to_load:
        total_memory_required[loaded_model.device] = total_memory_required.get(loaded_model.device, 0) + loaded_model.model_memory_required(loaded_model.device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_memory(total_memory_required[device] * 1.1 + extra_mem, device)

    for device in total_memory_required:
        if device != torch.device("cpu"):
            free_mem = get_free_memory(device)
            if free_mem < minimum_memory_required:
                models_l = free_memory(minimum_memory_required, device)
                logging.info("{} models unloaded.".format(len(models_l)))

    for loaded_model in models_to_load:
        model = loaded_model.model
        torch_dev = model.load_device
        if is_device_cpu(torch_dev):
            vram_set_state = VRAMState.DISABLED
        else:
            vram_set_state = vram_state
        lowvram_model_memory = 0
        if lowvram_available and (vram_set_state == VRAMState.LOW_VRAM or vram_set_state == VRAMState.NORMAL_VRAM) and not force_full_load:
            loaded_memory = loaded_model.model_loaded_memory()
            current_free_mem = get_free_memory(torch_dev) + loaded_memory

            lowvram_model_memory = max(128 * 1024 * 1024, (current_free_mem - minimum_memory_required), min(current_free_mem * MIN_WEIGHT_MEMORY_RATIO, current_free_mem - minimum_inference_memory()))
            lowvram_model_memory = max(0.1, lowvram_model_memory - loaded_memory)

        if vram_set_state == VRAMState.NO_VRAM:
            lowvram_model_memory = 0.1

        loaded_model.model_load(lowvram_model_memory, force_patch_weights=force_patch_weights)
        current_loaded_models.insert(0, loaded_model)
    return

def load_model_gpu(model):
    return load_models_gpu([model])

def loaded_models(only_currently_used=False):
    output = []
    for m in current_loaded_models:
        if only_currently_used:
            if not m.currently_used:
                continue

        output.append(m.model)
    return output
```

Every call to a guider/sampler in ComfyUI core (`CFGGuider.sample`, `KSampler`, etc.) calls
`load_models_gpu(...)` with the models it needs (unet, controlnets, etc. — not the VAE, which is
loaded separately by `VAEEncode`/`VAEDecode` via their own call into this same function) before
sampling starts. Calling `guider.sample(...)` repeatedly (e.g. once per tile) re-enters this same
function each time: if the model is already the top-priority entry in `current_loaded_models` with
`loaded_model_index is not None`, the function mostly short-circuits (marks `currently_used = True`,
skips a fresh `LoadedModel(x)` load) — it does not reload/re-patch the model from scratch on every
call as long as nothing else evicted it in between. `models = set(models)` also means passing the
same model object across repeated calls is safe/idempotent for the purposes of this function.

## cleanup_models_gc() / cleanup_models() (comfy/model_management.py, lines 653–681)

```python
def cleanup_models_gc():
    do_gc = False
    for i in range(len(current_loaded_models)):
        cur = current_loaded_models[i]
        if cur.is_dead():
            logging.info("Potential memory leak detected with model {}, doing a full garbage collect, for maximum performance avoid circular references in the model code.".format(cur.real_model().__class__.__name__))
            do_gc = True
            break

    if do_gc:
        gc.collect()
        soft_empty_cache()

        for i in range(len(current_loaded_models)):
            cur = current_loaded_models[i]
            if cur.is_dead():
                logging.warning("WARNING, memory leak with model {}. Please make sure it is not being referenced from somewhere.".format(cur.real_model().__class__.__name__))

def cleanup_models():
    to_delete = []
    for i in range(len(current_loaded_models)):
        if current_loaded_models[i].real_model() is None:
            to_delete = [i] + to_delete

    for i in to_delete:
        x = current_loaded_models.pop(i)
        del x
```

`cleanup_models_gc()` runs at the top of both `free_memory()` and `load_models_gpu()` — it is the
routine "is anything a dangling weakref" sweep that happens on every sampling call, not a manual
step the node needs to invoke. `cleanup_models()` is wired in as the `weakref.finalize` callback on
the real model object (see `LoadedModel.model_load` above), so it runs automatically when a real
model is garbage-collected.

## dtype_size() (comfy/model_management.py, lines 683–694)

```python
def dtype_size(dtype):
    dtype_size = 4
    if dtype == torch.float16 or dtype == torch.bfloat16:
        dtype_size = 2
    elif dtype == torch.float32:
        dtype_size = 4
    else:
        try:
            dtype_size = dtype.itemsize
        except: #Old pytorch doesn't have .itemsize
            pass
    return dtype_size
```

Used by `unet_inital_load_device` (below) to estimate a model's in-memory footprint from its
parameter count and dtype.

## unet_offload_device() / unet_inital_load_device() (comfy/model_management.py, lines 696–718)

```python
def unet_offload_device():
    if vram_state == VRAMState.HIGH_VRAM:
        return get_torch_device()
    else:
        return torch.device("cpu")

def unet_inital_load_device(parameters, dtype):
    torch_dev = get_torch_device()
    if vram_state == VRAMState.HIGH_VRAM or vram_state == VRAMState.SHARED:
        return torch_dev

    cpu_dev = torch.device("cpu")
    if DISABLE_SMART_MEMORY or vram_state == VRAMState.NO_VRAM:
        return cpu_dev

    model_size = dtype_size(dtype) * parameters

    mem_dev = get_free_memory(torch_dev)
    mem_cpu = get_free_memory(cpu_dev)
    if mem_dev > mem_cpu and model_size < mem_dev:
        return torch_dev
    else:
        return cpu_dev
```

`unet_offload_device()` is where a UNet gets sent back to when it's not actively being used
(`torch.device("cpu")` unless the user is running in `HIGH_VRAM` mode, in which case models simply
stay on the compute device). `unet_inital_load_device` is only consulted once, at model-patcher
construction time, to decide the *first* load location for a freshly-loaded checkpoint — it is not
re-consulted on every `guider.sample()` call.

## intermediate_device() / vae_device() / vae_offload_device() (comfy/model_management.py, lines 849–864)

```python
def intermediate_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")

def vae_device():
    if args.cpu_vae:
        return torch.device("cpu")
    return get_torch_device()

def vae_offload_device():
    if args.gpu_only:
        return get_torch_device()
    else:
        return torch.device("cpu")
```

`intermediate_device()` is the device ComfyUI core moves latents/images to for anything that isn't
meant to stay pinned to the compute device between pipeline stages (e.g. the tensor a node hands
back as its output) — `torch.device("cpu")` unless `--gpu-only` is passed. `vae_device()` is where
VAE encode/decode actually runs (the compute device, unless `--cpu-vae`); `vae_offload_device()` is
where the VAE model sits when it isn't actively encoding/decoding (again CPU unless `--gpu-only`).
A node that calls `vae.encode()`/`vae.decode()` and then `guider.sample()` repeatedly per tile should
expect the VAE and UNet to independently cycle between their respective offload device and the
compute device across calls, governed by `load_models_gpu`/`free_memory` — not by anything the node
itself needs to manage.

## get_free_memory() (comfy/model_management.py, lines 1079–1123)

```python
def get_free_memory(dev=None, torch_free_too=False):
    global directml_enabled
    if dev is None:
        dev = get_torch_device()

    if hasattr(dev, 'type') and (dev.type == 'cpu' or dev.type == 'mps'):
        mem_free_total = psutil.virtual_memory().available
        mem_free_torch = mem_free_total
    else:
        if directml_enabled:
            mem_free_total = 1024 * 1024 * 1024 #TODO
            mem_free_torch = mem_free_total
        elif is_intel_xpu():
            stats = torch.xpu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_torch = mem_reserved - mem_active
            mem_free_xpu = torch.xpu.get_device_properties(dev).total_memory - mem_reserved
            mem_free_total = mem_free_xpu + mem_free_torch
        elif is_ascend_npu():
            stats = torch.npu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_npu, _ = torch.npu.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_npu + mem_free_torch
        elif is_mlu():
            stats = torch.mlu.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_mlu, _ = torch.mlu.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_mlu + mem_free_torch
        else:
            stats = torch.cuda.memory_stats(dev)
            mem_active = stats['active_bytes.all.current']
            mem_reserved = stats['reserved_bytes.all.current']
            mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
            mem_free_torch = mem_reserved - mem_active
            mem_free_total = mem_free_cuda + mem_free_torch

    if torch_free_too:
        return (mem_free_total, mem_free_torch)
    else:
        return mem_free_total
```

`get_free_memory(device)` (no args → the current compute device) is the primitive `free_memory()`
and `load_models_gpu()` both call to decide whether anything needs to be evicted before a sampling
call proceeds; it is safe for a custom node to call directly (e.g. to log/decide tile batching) —
it has no side effects beyond querying the allocator/OS.

## is_device_type() / is_device_cpu() (comfy/model_management.py, lines 1133–1140)

```python
def is_device_type(device, type):
    if hasattr(device, 'type'):
        if (device.type == type):
            return True
    return False

def is_device_cpu(device):
    return is_device_type(device, 'cpu')
```

(`is_device_mps` / `is_device_cuda` follow the same one-line pattern immediately after this in the
file; they aren't called by any of the functions captured above, so they're omitted here.)

## soft_empty_cache() (comfy/model_management.py, lines 1300–1312)

```python
def soft_empty_cache(force=False):
    global cpu_state
    if cpu_state == CPUState.MPS:
        torch.mps.empty_cache()
    elif is_intel_xpu():
        torch.xpu.empty_cache()
    elif is_ascend_npu():
        torch.npu.empty_cache()
    elif is_mlu():
        torch.mlu.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
```

Called by `free_memory()` whenever it actually unloaded something (or, on non-`HIGH_VRAM` states,
whenever torch's reserved-but-inactive memory exceeds 25% of total free memory) — this is the
`torch.cuda.empty_cache()`-equivalent cache release a node would otherwise be tempted to call
manually between tiles; core already does it as part of `free_memory`/`load_models_gpu`.

## unload_all_models() (comfy/model_management.py, lines 1314–1315)

```python
def unload_all_models():
    free_memory(1e30, get_torch_device())
```

A blunt utility (requests an effectively-infinite amount of free memory, forcing every model off the
compute device) — available if a node wants to explicitly force a full unload between phases rather
than rely on the LRU-ish eviction in `free_memory`/`load_models_gpu`.
