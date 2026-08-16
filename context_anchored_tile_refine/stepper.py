"""Barrier-thread sampler stepping: N tiles, one shared sigma schedule, stock samplers.

Each LANE is one tile running an ORDINARY full-length `guider.sample()` on its own
cooperative thread. A barrier inside the model callable holds every lane at each sigma
step until all have arrived; a caller-supplied surgery hook then edits the lanes' live
tensors in place; then all lanes release. Because every lane runs the stock k-diffusion
function end to end on its own stack, multistep solver state (dpmpp_2m's `old_denoised`,
a Brownian stream's position) needs no unrolling and no resume identity — the
sigma-slicing traps in docs/sync-tiling-research-and-port-plan.md are structurally
absent because nothing is sliced. The only per-sampler knowledge left is EVALS_PER_STEP:
which model evals BEGIN a sigma step.

    stage 1   `run_lanes` resolves the SAMPLER once (name, option carry, rejections)
              and precomputes the eval -> step map from the shared schedule.
    stage 2   one thread per lane, started together; a `threading.Condition` scheduler
              lets exactly ONE lane run at any moment (round-robin inside an eval index,
              handed on when a lane blocks at a barrier or finishes). No concurrent comfy
              calls, no reliance on comfy thread-safety, one GPU stream.
    stage 3   at each barrier that BEGINS a sigma step the last arriving lane calls
              `hook(step_index, sigma, lanes)`, then releases the fleet.
    stage 4   after its LAST eval a lane waits at the FINAL EXIT BARRIER, so no lane's
              guider teardown (`cleanup()` -> `current_patcher = None`) can precede
              another lane's final eval.

Exceptions: a lane failure, or a hook failure, sets a layer-wide abort flag. Every lane
tests that flag at both wait points, so the fleet unwinds instead of each released lane
running out its remaining full-length schedule. The FIRST exception reaches the caller
UNCHANGED; every thread is joined. Both catch sites take BaseException, not Exception,
because comfy's InterruptProcessingException derives from BaseException
(comfy/model_management.py:2098) — a user cancel is raised inside a lane thread's model
forward, the interrupt flag is CONSUMED by the first checker, so a swallowed one is lost
for good and ComfyUI would report a failure instead of an interruption.

KNOWN RESIDUAL (carried from tests-AB/run_ab_sync.py:355-399): under sync surgery a
MULTISTEP sampler's per-lane history is its own PRE-blend prediction, while the `x` it
corrects was feather-blended with the neighbour's write in the overlap bands — so a
stale term enters those bands each step. Second order in inter-tile disagreement and
band-confined; single-step samplers carry none of it. If a band artifact appears under
a multistep sampler only, suspect this before geometry.

STOCHASTIC samplers draw from a SHARED CANVAS FIELD, never from a private per-lane one:
`build_noise_fields` returns one canvas-shaped provider and `run_lanes` injects each lane's
window slice of it as the sampler's `noise_sampler`, ADDED over the merged extra_options so
the node SAMPLER's own eta/s_noise/solver_type still ride along. Two lanes whose windows
overlap therefore hold torch.equal noise on every shared cell; a per-lane noise sampler
would put an independent field on each side of every band, which is the seam this engine
exists to remove.

torch + stdlib at module scope; `comfy` is imported lazily inside functions (a subprocess
test pins it).
"""
import functools
import threading

import torch

# Model evals per sigma step, keyed by the resolved sampler name, as
# (evals, evals_on_a_step_whose_NEXT_sigma_is_0). Verified line by line against the
# installed ComfyUI 0.33.0 comfy/k_diffusion/sampling.py — a comfy-side change to any of
# these silently mistimes the surgery, which is what the completion cross-check in
# `run_lanes` exists to catch.
#   euler          :190  one `model(...)` per loop iteration, no branch.
#   dpmpp_2m       :796  one `model(...)` per loop iteration; the sigmas[i+1] == 0 case
#                        changes the update, not the eval count.
#   heun           :269  two evals, falling back to a single Euler eval at sigma_next == 0.
#   dpm_2          :304  two evals, same Euler fallback at sigma_next == 0.
#   exp_heun_2_x0  :1655 an alias that forwards to sample_seeds_2 (:1592) with eta=0,
#                        s_noise=0, r=1 — two evals, and `if sigmas[i+1] == 0: x =
#                        denoised; continue` makes the last step a single eval.
#   dpmpp_2m_sde   :822  one `model(...)` per loop iteration (:845); the sigmas[i+1] == 0
#                        case short-circuits to `x = denoised` (:848), which changes the
#                        update, not the eval count.
#   dpmpp_2m_sde_gpu      :965
#   dpmpp_2m_sde_heun     :877
#   dpmpp_2m_sde_heun_gpu :955
#                        thin wrappers over sample_dpmpp_2m_sde whose ONLY delta is the
#                        device of the default BrownianTreeNoiseSampler they build when
#                        `noise_sampler is None` — which never happens here, the layer
#                        always injects one, so they are exact passthroughs. They MUST be
#                        supported: core's own SamplerDPMPP_2M_SDE emits `dpmpp_2m_sde_gpu`
#                        by DEFAULT (its noise_device combo lists 'gpu' first,
#                        comfy_extras/nodes_custom_sampler.py:431-442).
#   exp_heun_2_x0_sde
#                  :1661 the same sample_seeds_2 alias with eta/s_noise live and r=1 — two
#                        evals, and the same `x = denoised; continue` (:1618-1620) makes the
#                        last step a single eval.
EVALS_PER_STEP = {
    "euler": (1, 1),
    "dpmpp_2m": (1, 1),
    "heun": (2, 1),
    "dpm_2": (2, 1),
    "exp_heun_2_x0": (2, 1),
    "dpmpp_2m_sde": (1, 1),
    "dpmpp_2m_sde_gpu": (1, 1),
    "dpmpp_2m_sde_heun": (1, 1),
    "dpmpp_2m_sde_heun_gpu": (1, 1),
    "exp_heun_2_x0_sde": (2, 1),
}

SUPPORTED_SAMPLERS = tuple(EVALS_PER_STEP)

# Named in the rejection message so the failure is self-explaining: these own their own
# schedule (dpm_fast/dpm_adaptive derive sigmas internally from a step count) or run a
# solver whose model calls do not map onto the caller's sigma steps at all (uni_pc), so no
# evals-per-step row can time them.
UNSUPPORTED_BY_DESIGN = ("dpm_fast", "dpm_adaptive", "uni_pc")

# Which shared noise field each STOCHASTIC sampler needs, by resolved name. Deterministic
# samplers are ABSENT: they take no `noise_sampler` at all, so `build_noise_fields` answers
# None for them and `run_lanes` injects nothing.
#   brownian   the dpmpp_2m_sde family: ONE `noise_sampler(sigmas[i], sigmas[i + 1])` per
#              non-final step (:869). A BrownianTree draw is a deterministic function of that
#              (sigma, sigma_next) pair, so a single canvas-wide tree hands every lane the
#              same field with no coordination at all.
#   generator  exp_heun_2_x0_sde: sample_seeds_2 draws TWICE per non-final step (:1634,
#              :1650) and at its r == 1 the two draws' sigma pairs COINCIDE (sigma_s_1 ==
#              sigmas[i + 1]), so a (sigma, sigma_next) key cannot tell them apart — the
#              field is keyed by (step, draw index) off one run-long generator instead. At
#              r == 1 the second draw's integrator weight is exactly zero (segment_factor =
#              (r - 1) * h * eta = 0), but the sampler still REQUESTS it, so it is still
#              served: skipping it would shift every later draw by one.
SDE_NOISE_KIND = {
    "dpmpp_2m_sde": "brownian",
    "dpmpp_2m_sde_gpu": "brownian",
    "dpmpp_2m_sde_heun": "brownian",
    "dpmpp_2m_sde_heun_gpu": "brownian",
    "exp_heun_2_x0_sde": "generator",
}

# sample_seeds_2's noise draws per non-final sigma step (:1634 and :1650, both under
# `if inject_noise`) — what maps a lane's Nth request onto the run's Nth (step, draw) field.
SEEDS_2_DRAWS_PER_STEP = 2


class LaneSpec:
    """One tile's sampling inputs — everything `guider.sample` needs, plus `window`.

    `guider` must be a DISTINCT object per lane: comfy's CFGGuider stores per-run state on
    itself (`inner_model`, `conds`, `loaded_models`) and tears it down in `outer_sample`'s
    finally, so two lanes sharing one guider would overwrite each other mid-run.
    `denoise_mask` arrives ALREADY normalized by the caller (sampling.sample_latent's
    canonical form). `extra_options` are per-lane solver knobs merged OVER the node
    SAMPLER's own; the shared `noise_sampler` is ADDED over that merge, so neither dict may
    carry one of its own (`_reject_carried_noise_sampler`).
    `window` is opaque to the barrier machinery — the hook gets it back untouched, and its
    only other reader is the noise-field provider (`_NoiseFields`), which is a caller-
    replaceable callable precisely so the layer itself never has to know the schema.
    """

    __slots__ = ("denoise_mask", "extra_options", "guider", "latent", "noise", "seed", "sigmas", "window")

    def __init__(self, guider, sigmas, noise, latent, denoise_mask=None, seed=0, window=None,
                 extra_options=None):
        self.guider = guider
        self.sigmas = sigmas
        self.noise = noise
        self.latent = latent
        self.denoise_mask = denoise_mask
        self.seed = seed
        self.window = window
        self.extra_options = extra_options


class Lane:
    """The hook's view of one running lane.

    `x` is the LIVE trajectory tensor the lane is about to feed the model — mutate it in
    place (`x.copy_(...)`, slice assignment) and the next eval sees the edit; rebinding it
    does nothing, the sampler holds its own reference. `model_k` is the real
    KSamplerX0Inpaint, so the hook can rewrite its `latent_image` / `noise` cells.
    `denoised` is the last eval RESULT — the model's x0 prediction in process-space, which
    is what a live preview decodes; the trajectory `x` is noisy at the current sigma and
    previews as noise. It is None until the lane's first eval returns, so at step 0 the
    hook sees None and from step i it sees each lane's last denoised of step i-1.
    """

    __slots__ = ("denoised", "eval_count", "index", "model_k", "result", "window", "x")

    def __init__(self, index, window):
        self.index = index
        self.window = window
        self.x = None
        self.model_k = None
        self.denoised = None
        self.result = None
        self.eval_count = 0


class _Aborted(Exception):
    """Unwinds a lane after ANOTHER lane (or the hook) failed. Never escapes the layer —
    `run_lanes` re-raises the first real exception instead."""


def _resolve_sampler_name(sampler):
    # Sampler intake. The layer needs the k-diffusion function's identity to time the
    # barrier, and only a stock KSAMPLER exposes it in a form the table can key on.
    import comfy.samplers

    if not isinstance(sampler, comfy.samplers.KSAMPLER):
        raise ValueError(
            "ContextAnchoredTileRefine: the synchronized stepper only supports standard "
            f"KSAMPLER samplers, got {type(sampler).__name__}. Custom sampler objects expose "
            "no k-diffusion function to time the per-step barrier against."
        )
    if sampler.inpaint_options.get("random", False):
        # comfy/samplers.py:988 redraws the inpaint noise from seed+1 inside KSAMPLER.sample;
        # that private draw collides with the shared per-lane noise field the sync engine
        # feeds every lane. It is reachable only through sampler_object("ddim") (:1393).
        raise ValueError(
            "ContextAnchoredTileRefine: the synchronized stepper does not support the 'ddim' "
            "sampler (inpaint_options random noise): its seed+1 noise redraw collides with the "
            "shared noise field every lane samples against."
        )

    raw_name = getattr(sampler.sampler_function, "__name__", "")
    name = raw_name[len("sample_"):] if raw_name.startswith("sample_") else raw_name
    if name not in EVALS_PER_STEP:
        raise ValueError(
            f"ContextAnchoredTileRefine: sampler '{name or raw_name}' is not supported by the "
            f"synchronized stepper. Supported: {', '.join(EVALS_PER_STEP)}. "
            f"{', '.join(UNSUPPORTED_BY_DESIGN)} are unsupported BY DESIGN — they own their own "
            "schedule or internal history, so no evals-per-step entry can time them."
        )
    return name


class _NoiseFields:
    """Base of the shared-noise providers: `provider(spec)` -> that lane's `noise_sampler`.

    `spec.window` is read HERE and nowhere else in the layer, which is what keeps the window
    opaque to the barrier machinery: a caller whose window is not a latent-cell rect passes
    its own callable of this shape instead. `window` is `(y0, y1, x0, x1)` in LATENT CELLS —
    the rectangle of the canvas the lane occupies, i.e. the same rect the engine slices its
    canvas latent with — and the lane's draw is the canvas field sliced `[..., y0:y1, x0:x1]`.
    """

    def __call__(self, spec):
        return self.for_window(spec.window)

    def for_window(self, window):
        raise NotImplementedError


class _BrownianNoiseFields(_NoiseFields):
    """The dpmpp_2m_sde family's field: ONE canvas-shaped BrownianTreeNoiseSampler over the
    run's whole sigma range, sliced per lane window.

    `cpu=True` is sample_dpmpp_2m_sde's own default (:833). The `_gpu` wrappers differ only
    in the device they would build a default sampler on, which never happens here.
    """

    def __init__(self, shape, seed, sigmas):
        import comfy.k_diffusion.sampling

        positive = sigmas[sigmas > 0]
        if positive.numel() == 0:
            raise ValueError(
                "ContextAnchoredTileRefine: a stochastic sampler's Brownian noise tree needs a "
                "positive sigma to span, and this schedule has none."
            )
        self._tree = comfy.k_diffusion.sampling.BrownianTreeNoiseSampler(
            torch.zeros(tuple(shape)), positive.min(), sigmas.max(), seed=seed, cpu=True,
        )
        self._key = None
        self._canvas_field = None

    def _canvas(self, sigma, sigma_next):
        # A one-entry cache, and an OPTIMIZATION only: the tree is deterministic in
        # (sigma, sigma_next), so every lane would compute the same field anyway. One entry
        # suffices because the barrier holds the whole fleet inside the same sigma step.
        key = (float(sigma), float(sigma_next))
        if key != self._key:
            self._canvas_field = self._tree(sigma, sigma_next)
            self._key = key
        return self._canvas_field

    def for_window(self, window):
        y0, y1, x0, x1 = window

        def noise_sampler(sigma, sigma_next):
            # The tree already returns the draw on sigma's device and dtype.
            return self._canvas(sigma, sigma_next)[..., y0:y1, x0:x1]

        return noise_sampler


class _GeneratorNoiseFields(_NoiseFields):
    """The seeds_2 family's field: one canvas-shaped CPU draw per (step, draw index), sliced
    per lane window.

    Seeded `seed + 1` — comfy's own CPU offset convention (default_noise_sampler, :78-88).
    Seeding with the BARE refine seed makes this generator's first draw bit-identical to
    prepare_noise's canvas draw (same engine, seed, shape, dtype, device), so the first SDE
    injection would re-apply the construction epsilon: a coherent over-noise arriving on
    every lane at once, which reads as sync-tiling fidelity loss instead of as the collision
    it is.
    """

    def __init__(self, shape, seed, sigmas, draws_per_step):
        self._generator = torch.Generator().manual_seed(int(seed) + 1)
        self._shape = tuple(shape)
        self._draws_per_step = draws_per_step
        # The final step draws nothing (sigma_next == 0 short-circuits before the first
        # request), so only the steps whose NEXT sigma is non-zero contribute fields.
        self._noisy_steps = sum(
            1 for step in range(max(int(sigmas.shape[-1]) - 1, 0))
            if float(sigmas[step + 1]) != 0.0
        )
        self._index = -1        # ordinal of the ONE field currently held
        self._field = None

    def _canvas(self, index):
        # Drawn lazily but STRICTLY IN ORDER, so which lane asks first cannot change which
        # field a given (step, draw) gets. Lanes run cooperatively serial, so no lock.
        # ONE slot, never a per-index cache: a canvas field is latent-sized, so retaining the
        # run's would cost 2 * noisy_steps * that at once (gigabytes at production canvas x
        # step count). One suffices because a BARRIER separates draw k from draw k+1 —
        # sample_seeds_2 makes a model call between them (:1636) and between the step's last
        # draw and the next step's first (:1614), and every model call is a barrier — so every
        # lane consumes k before any lane can ask for k+1. A backwards request would mean that
        # invariant broke, so it raises rather than silently serving the wrong field.
        if index < self._index:
            raise RuntimeError(
                f"ContextAnchoredTileRefine: a lane asked for SDE noise draw {index + 1} after the "
                f"fleet had moved on to draw {self._index + 1} — the lanes are no longer stepping "
                "the shared schedule in lockstep."
            )
        while self._index < index:
            self._field = torch.randn(self._shape, generator=self._generator)
            self._index += 1
        return self._field

    def for_window(self, window):
        y0, y1, x0, x1 = window
        total = self._noisy_steps * self._draws_per_step
        state = {"index": 0}

        def noise_sampler(sigma, sigma_next):
            # (step, draw index) flattened to one ordinal: every lane runs the SAME sampler
            # over the SAME schedule, so a lane's Nth request is the run's Nth (step, draw).
            index = state["index"]
            state["index"] = index + 1
            if index >= total:
                raise RuntimeError(
                    f"ContextAnchoredTileRefine: a lane requested SDE noise draw {index + 1} but "
                    f"this schedule plans {total} ({self._noisy_steps} noisy sigma steps x "
                    f"{self._draws_per_step} draws) — the draws-per-step contract no longer "
                    "matches this ComfyUI's sampler."
                )
            return self._canvas(index)[..., y0:y1, x0:x1].to(device=sigma.device)

        return noise_sampler


def build_noise_fields(sampler, shape, seed, sigmas):
    """The shared canvas-wide SDE noise for `sampler`, or None when it is deterministic.

    `shape` is the CANVAS latent shape; every lane draws that ONE field sliced to its own
    window, so overlapping lanes hold torch.equal noise on every shared cell. Hand the
    result straight to `run_lanes(..., noise_fields=)`. None is a real answer, not a
    failure: a deterministic sampler accepts no `noise_sampler` argument at all.
    """
    kind = SDE_NOISE_KIND.get(_resolve_sampler_name(sampler))
    if kind is None:
        return None
    if kind == "generator":
        return _GeneratorNoiseFields(shape, seed, sigmas, SEEDS_2_DRAWS_PER_STEP)
    return _BrownianNoiseFields(shape, seed, sigmas)


def _validate_noise_fields(sampler_name, noise_fields):
    # Both directions are errors, because both are silent quality bugs otherwise: a missing
    # field leaves every lane drawing its own private noise (two different fields meeting in
    # every overlap band), and a field handed to a deterministic sampler reaches it as an
    # unexpected keyword argument deep inside k-diffusion.
    stochastic = sampler_name in SDE_NOISE_KIND
    if stochastic and noise_fields is None:
        raise ValueError(
            f"ContextAnchoredTileRefine: sampler '{sampler_name}' is stochastic, so it needs the "
            "shared canvas noise field — build it with stepper.build_noise_fields(...) and pass "
            "it as noise_fields. Without it every lane draws its own independent noise and each "
            "overlap band carries two different fields."
        )
    if not stochastic and noise_fields is not None:
        raise ValueError(
            f"ContextAnchoredTileRefine: sampler '{sampler_name}' is deterministic and takes no "
            "noise_sampler, so noise_fields must be None — which is what build_noise_fields "
            "returns for it."
        )


def _reject_carried_noise_sampler(sampler, lane_specs):
    # The layer ADDS noise_sampler over the merged options, so anything already there would be
    # silently replaced. Never do that to a caller's knob: say so instead.
    sources = [("the SAMPLER's", sampler.extra_options)]
    sources += [(f"lane {position}'s", spec.extra_options) for position, spec in enumerate(lane_specs)]
    for owner, options in sources:
        if options and "noise_sampler" in options:
            raise ValueError(
                f"ContextAnchoredTileRefine: {owner} extra_options already carries a "
                "'noise_sampler'. The synchronized stepper injects the shared canvas-wide field "
                "every lane draws from, so it would silently replace that one — and a per-lane "
                "noise sampler is exactly the seam this engine exists to remove."
            )


def _plan_evals(name, sigmas):
    # Map the shared schedule onto model evals ONCE: the total each lane must perform, and
    # {eval index that BEGINS a step: step index}. The final-step exception keys on the NEXT
    # sigma being 0 — the exact predicate the sampler bodies branch on — not on the step's
    # ordinal, so a schedule that does not land on 0 is timed correctly too.
    per_step, final_per_step = EVALS_PER_STEP[name]
    steps = int(sigmas.shape[-1]) - 1
    hook_step_at = {}
    total = 0
    for step in range(max(steps, 0)):
        hook_step_at[total] = step
        total += final_per_step if float(sigmas[step + 1]) == 0.0 else per_step
    return total, hook_step_at


class _LaneModel:
    """Wraps one lane's KSamplerX0Inpaint. The barrier site, the denoised stash, and a
    transparent attribute proxy.

    The proxy is load-bearing, not politeness: samplers reach THROUGH the model object
    (`model.inner_model.model_patcher.get_model_object('model_sampling')` at
    k_diffusion/sampling.py:836 for the dpmpp_2m_sde family and :1604 for the seeds_2
    family that exp_heun_2_x0 aliases), so a call-only wrapper would break them.
    """

    def __init__(self, model_k, lane, scheduler):
        self._inner = model_k
        self._lane = lane
        self._scheduler = scheduler
        lane.model_k = model_k

    def __getattr__(self, name):
        # Only fires for attributes the wrapper does not own. object.__getattribute__ never
        # re-enters __getattr__, so a missing _inner raises AttributeError instead of
        # recursing.
        return getattr(object.__getattribute__(self, "_inner"), name)

    def __call__(self, x, sigma, **kwargs):
        lane = self._lane
        scheduler = self._scheduler
        index = lane.eval_count

        if index >= scheduler.total_evals:
            raise RuntimeError(
                f"ContextAnchoredTileRefine: sampler '{scheduler.sampler_name}' asked for model "
                f"eval {index + 1} but EVALS_PER_STEP predicts {scheduler.total_evals} for this "
                "schedule — the evals-per-step table no longer matches this ComfyUI's sampler."
            )

        lane.x = x
        scheduler.arrive(lane, index)
        result = self._inner(x, sigma, **kwargs)
        lane.denoised = result
        lane.eval_count = index + 1
        if lane.eval_count == scheduler.total_evals:
            # FINAL EXIT BARRIER: the result is computed, but the lane holds here until every
            # other lane has computed its own last eval, so no guider teardown can race one.
            scheduler.arrive(lane, scheduler.total_evals)
        return result


class _Scheduler:
    """Cooperative serial scheduler: one token, one barrier, one abort flag.

    `_turn` is the token — the index of the only lane allowed to run. A lane hands it on
    (`_turn = own index + 1`) when it blocks at a barrier or finishes, so exactly one lane
    executes at a time and the run order inside an eval index is lane order. `_arrived[i]`
    is the barrier index lane i last reached; when every entry equals the same index that
    barrier's last arriver opens it (running the hook first when the index begins a step)
    and resets the token to lane 0.
    """

    def __init__(self, lanes, sigmas, sampler_name, total_evals, hook_step_at, hook):
        self.total_evals = total_evals
        self.sampler_name = sampler_name
        self._cv = threading.Condition()
        self._lanes = lanes
        self._sigmas = sigmas
        self._hook_step_at = hook_step_at
        self._hook = hook
        self._phase = -1                    # highest barrier index already opened
        self._turn = 0                      # lane index holding the token inside _phase
        self._arrived = [-1] * len(lanes)   # barrier index each lane last reached
        self._abort = None

    @property
    def abort(self):
        return self._abort

    def _pass_token(self, from_index):
        # Caller holds the lock.
        self._turn = from_index + 1
        self._cv.notify_all()

    def _raise_if_abort(self):
        # Caller holds the lock.
        if self._abort is not None:
            raise _Aborted

    def fail(self, error):
        with self._cv:
            if self._abort is None:
                self._abort = error
            self._cv.notify_all()

    def start(self, lane):
        # A lane's pre-eval segment (prepare_sampling, process_conds, pre_run) is real comfy
        # work and takes the token like any other segment.
        with self._cv:
            self._cv.wait_for(lambda: self._abort is not None or self._turn == lane.index)
            self._raise_if_abort()

    def arrive(self, lane, index):
        with self._cv:
            self._raise_if_abort()
            self._arrived[lane.index] = index
            self._pass_token(lane.index)
            coordinator = all(reached == index for reached in self._arrived)

        if coordinator:
            self._open(index)

        with self._cv:
            self._cv.wait_for(
                lambda: self._abort is not None or (self._phase >= index and self._turn == lane.index)
            )
            self._raise_if_abort()

    def _open(self, index):
        # Runs on the last arriving lane, with every lane parked at `index` and the phase not
        # yet advanced — so the hook has the whole fleet stopped and to itself, and runs
        # inside that thread's inference_mode.
        step = self._hook_step_at.get(index)
        if step is not None:
            try:
                self._hook(step, self._sigmas[step], self._lanes)
            except BaseException as error:  # BaseException: see _run_lane, the interrupt is one
                self.fail(error)
        with self._cv:
            self._phase = index
            self._turn = 0
            self._cv.notify_all()

    def finish(self, lane):
        with self._cv:
            if self._abort is None and lane.eval_count != self.total_evals:
                # A lane that stopped short would otherwise strand the rest at a barrier that can
                # never fill; a lane that ran long is caught in _LaneModel.__call__.
                self._abort = RuntimeError(
                    f"ContextAnchoredTileRefine: sampler '{self.sampler_name}' performed "
                    f"{lane.eval_count} model evals but EVALS_PER_STEP predicts "
                    f"{self.total_evals} for this schedule — the evals-per-step table no longer "
                    "matches this ComfyUI's sampler."
                )
            self._pass_token(lane.index)


def _wrap_sampler_function(real_fn, lane, scheduler):
    # KSAMPLER.sample calls this as sampler_function(model_k, noise, sigmas, extra_args=,
    # callback=, disable=, **extra_options) (comfy/samplers.py:1006). Everything is forwarded
    # untouched except model_k, which is swapped for the lane's barrier wrapper.
    @functools.wraps(real_fn)
    def wrapped(model_k, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        return real_fn(
            _LaneModel(model_k, lane, scheduler), x, sigmas,
            extra_args=extra_args, callback=callback, disable=disable, **kwargs,
        )

    return wrapped


def _build_lane_sampler(sampler, lane, scheduler, extra_options, noise_sampler):
    import comfy.samplers

    # The user's solver knobs (SamplerDPMPP_2M_SDE's eta/s_noise/solver_type,
    # comfy_extras/nodes_custom_sampler.py:442) and inpaint options are CARRIED, never
    # replaced: per-lane entries are merged OVER them.
    merged = dict(sampler.extra_options)
    if extra_options:
        merged.update(extra_options)
    if noise_sampler is not None:
        # This lane's window slice of the ONE shared canvas field. Added, never replacing:
        # _reject_carried_noise_sampler has already ruled out a caller-supplied entry.
        merged["noise_sampler"] = noise_sampler
    return comfy.samplers.KSAMPLER(
        _wrap_sampler_function(sampler.sampler_function, lane, scheduler),
        extra_options=merged,
        inpaint_options=dict(sampler.inpaint_options),
    )


def _validate_specs(lane_specs):
    if not lane_specs:
        raise ValueError("ContextAnchoredTileRefine: the synchronized stepper needs at least one lane.")

    guider_ids = {id(spec.guider) for spec in lane_specs}
    if len(guider_ids) != len(lane_specs):
        raise ValueError(
            "ContextAnchoredTileRefine: every lane needs its OWN guider. comfy's CFGGuider "
            "stores per-run state on itself and tears it down when its sample() returns, so "
            "two lanes sharing one guider corrupt each other."
        )

    first = lane_specs[0].sigmas
    for position, spec in enumerate(lane_specs[1:], start=1):
        if not torch.equal(spec.sigmas, first):
            raise ValueError(
                f"ContextAnchoredTileRefine: lane {position} was handed a different sigma "
                "schedule. Every lane must step the SAME schedule — that is the whole point of "
                "the barrier."
            )


def _run_lane(scheduler, lane, spec, lane_sampler):
    # inference/grad mode is THREAD-LOCAL: ComfyUI wraps node execution in inference_mode
    # (execution.py:751) but a fresh Thread starts outside it, and prepare_sampling /
    # process_latent_in / pre_run would then run grad-enabled. First statement, every lane.
    with torch.inference_mode():
        try:
            scheduler.start(lane)
            lane.result = spec.guider.sample(
                spec.noise, spec.latent, lane_sampler, spec.sigmas,
                denoise_mask=spec.denoise_mask, callback=None, disable_pbar=True, seed=spec.seed,
            )
        except _Aborted:
            pass
        except BaseException as error:
            # BaseException, not Exception: comfy's InterruptProcessingException derives from
            # BaseException (comfy/model_management.py:2098) and is raised from INSIDE the model
            # forward (comfy/ops.py run_every_op), i.e. inside a lane thread. `except Exception`
            # would let it kill the thread unhandled, leaving _abort unset — the user's cancel
            # lost (the flag is consumed by the first checker, so nothing re-raises it) and
            # `finish` installing a bogus EVALS_PER_STEP error in its place. Routing it through
            # the abort flag makes run_lanes re-raise the ORIGINAL object, so ComfyUI marks the
            # prompt interrupted. Same catch covers a failure raised by guider.sample AFTER the
            # final exit barrier, which would otherwise return a None sample.
            scheduler.fail(error)
        finally:
            scheduler.finish(lane)


def run_lanes(lane_specs, sampler, hook, noise_fields=None):
    """Step every lane through one shared sigma schedule, calling `hook` between steps.

    `sampler` is the node's SAMPLER object, resolved once and rebuilt per lane around the
    barrier. `hook(step_index, sigma, lanes)` fires exactly once per sigma step, on the
    thread of that step's last arriving lane, with every lane parked and its `x` / `denoised`
    readable. It is PRE-step by construction: the caller owns whatever consolidation the
    final step's results need after this returns. `noise_fields` is the shared canvas noise
    provider from `build_noise_fields` — REQUIRED for a stochastic sampler, and rejected for
    a deterministic one; each lane gets `noise_fields(spec)` as its `noise_sampler`. Returns
    the lanes' samples in lane order.
    """
    _validate_specs(lane_specs)
    sampler_name = _resolve_sampler_name(sampler)
    _validate_noise_fields(sampler_name, noise_fields)
    _reject_carried_noise_sampler(sampler, lane_specs)
    sigmas = lane_specs[0].sigmas
    total_evals, hook_step_at = _plan_evals(sampler_name, sigmas)

    lanes = [Lane(index, spec.window) for index, spec in enumerate(lane_specs)]
    scheduler = _Scheduler(lanes, sigmas, sampler_name, total_evals, hook_step_at, hook)
    lane_samplers = [
        _build_lane_sampler(
            sampler, lane, scheduler, spec.extra_options,
            None if noise_fields is None else noise_fields(spec),
        )
        for lane, spec in zip(lanes, lane_specs, strict=True)
    ]
    threads = [
        threading.Thread(
            target=_run_lane,
            args=(scheduler, lane, spec, lane_sampler),
            name=f"catr-lane-{lane.index}",
            daemon=True,
        )
        for lane, spec, lane_sampler in zip(lanes, lane_specs, lane_samplers, strict=True)
    ]

    started = []
    try:
        for thread in threads:
            thread.start()
            started.append(thread)
    except BaseException as error:
        # A lane that never starts would strand the rest at a barrier it can never reach, so the
        # join below would hang. BaseException for the same reason as in _run_lane.
        scheduler.fail(error)
    finally:
        for thread in started:
            thread.join()

    if scheduler.abort is not None:
        raise scheduler.abort
    return [lane.result for lane in lanes]
