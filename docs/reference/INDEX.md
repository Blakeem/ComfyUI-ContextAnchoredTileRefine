# Reference doc set — Context-Anchored Tile Refine (ComfyUI 0.3.45)

Verbatim reference for building the `comfyui-contextanchoredtilerefine` custom node. Every file
carries a source header (source, version, retrieval date). Content is copied verbatim from
docs.comfy.org, the local ComfyUI 0.3.45 source tree, the SciPy 1.15 manual, and the cited
research repos/papers — read the source, not a rewrite. Grouped by the brief's priority topics.

Target versions: ComfyUI **0.3.45** (ComfyUI Desktop) · Python 3.12.9 · torch 2.9.1+cu130 · scipy 1.15.1.

## comfy-docs/ — official custom-node development (docs.comfy.org, current/main) — brief topic 1

- `comfy-docs/walkthrough.md` — Getting Started tutorial: `comfy node scaffold`, minimal node class, `INPUT_TYPES`/`RETURN_TYPES`/`FUNCTION`/`CATEGORY`, `NODE_CLASS_MAPPINGS`. Read first for the end-to-end shape of a node. (Frontend/JS "Tweak the UI" section removed as out of scope.)
- `comfy-docs/node-properties.md` — "Properties": the authoritative per-attribute reference — `INPUT_TYPES` (required/optional/hidden), `RETURN_TYPES`/`RETURN_NAMES`, `CATEGORY`, `FUNCTION`, `OUTPUT_NODE`, `IS_CHANGED`, `SEARCH_ALIASES`, and `VALIDATE_INPUTS` (incl. `input_types`). Read for node lifecycle + validation semantics.
- `comfy-docs/code-interface.md` — "Code Interface": second official reference on the same attributes from the LoadCheckpoint node, plus `WEB_DIRECTORY`, `NODE_CLASS_MAPPINGS`/`NODE_DISPLAY_NAME_MAPPINGS`. Read alongside node-properties (overlaps it — see Coverage notes).
- `comfy-docs/hidden-and-flexible-inputs.md` — "Hidden and Flexible inputs": `hidden` inputs (`PROMPT`/`EXTRA_PNGINFO`/`UNIQUE_ID`/`DYNPROMPT`), custom datatypes, `forceInput`, wildcard `*` inputs, and dynamically-created (`ContainsAnyDict`) inputs. Read for the input-type machinery beyond required/optional.
- `comfy-docs/datatypes.md` — the datatype reference: `INT`/`FLOAT`/`STRING`/`BOOLEAN` widget config (default/min/max/step), `IMAGE`/`LATENT`/`MASK`/`AUDIO` tensor shapes, custom-sampling `NOISE`/`SAMPLER`/`SIGMAS`/`GUIDER` contracts, and the full "extra options" key table (`forceInput`, `lazy`, `rawLink`, …). Read for every input/output type this node touches.
- `comfy-docs/tensors.md` — `torch.Tensor` primer: rank/shape, squeeze/unsqueeze/reshape, `None`/`:`/`...` slicing, elementwise ops, tensor truthiness (`is not None`). Read for the tensor idioms used throughout.
- `comfy-docs/images-and-masks.md` — IMAGE `[B,H,W,C]`, MASK `[B,H,W]`, LATENT `dict{samples:[B,C,H,W]}`; channel-last vs channel-first; LoadImage mask behavior. Read for shape conventions at node boundaries.
- `comfy-docs/lifecycle.md` — how Comfy discovers custom nodes: `__init__.py`, `NODE_CLASS_MAPPINGS`/`NODE_DISPLAY_NAME_MAPPINGS`, `WEB_DIRECTORY`, `__all__`. Read for package structure.
- `comfy-docs/lazy-evaluation.md` — `lazy: True` inputs, `check_lazy_status`, and `ExecutionBlocker`. Read if the node should skip evaluating unused inputs.
- `comfy-docs/annotated-examples.md` — copy-paste snippets: load/save image, invert mask, mask→image shape, mask-as-transparency, and `Noise_MixedNoise` (mixing two NOISE sources). Read for concrete idioms.
- `comfy-docs/registry-overview.md` — what the Comfy Registry is, semantic versioning, deprecation. Read before publishing.
- `comfy-docs/registry-publishing.md` — publisher account, API key, `comfy node init`, `.comfyignore`, CLI vs GitHub Action publish. Read to publish.
- `comfy-docs/registry-specifications.md` — full `pyproject.toml` spec: `[project]` (name/version/license/requires-python/classifiers) and `[tool.comfy]` (PublisherId/DisplayName/Icon/Banner/`requires-comfyui`/`includes`). Read when writing pyproject.toml.
- `comfy-docs/registry-standards.md` — publishing security standards (no eval/exec, no subprocess pip install, no obfuscation) + community/fork rules. Read for compliance.
- `comfy-docs/v3-migration-overview.md` — **excerpt** flagging the V1-vs-V3 node-schema split. **Read this to understand why the rest of the set uses V1** (`INPUT_TYPES`/`RETURN_TYPES`/`NODE_CLASS_MAPPINGS`) and how current docs.comfy.org increasingly targets V3 (`io.ComfyNode`/`define_schema`/`comfy_entrypoint`). See Coverage notes.

## comfyui-source/ — ComfyUI 0.3.45 local source tree — brief topics 2–6

- `comfyui-source/comfy-extras-nodes-custom-sampler.md` — (topic 2) `Noise_EmptyNoise`/`Noise_RandomNoise.generate_noise`, `RandomNoise`/`DisableNoise` nodes, the GUIDER wrapper nodes (`CFGGuider`, `BasicGuider`, `DualCFGGuider` + their `Guider_*` classes), and **`SamplerCustomAdvanced.sample`** — the exact node this project's interface mirrors (noise/guider/sampler/sigmas/latent flow, `fix_empty_latent_channels`, `output` vs `denoised_output`). Read first for the sampling entry point.
- `comfyui-source/comfy-samplers-cfgguider-and-pipeline.md` — (topic 3) `CFGGuider` (`set_conds`/`set_cfg`/`sample`/`inner_sample`/`outer_sample`), `cfg_function`/`sampling_function`, **`KSamplerX0Inpaint.__call__`** (the exact `denoise_mask_function` hook call site with signature + `extra_options={model, sigmas}`), `Sampler`/`KSAMPLER`/`ksampler()`/`sampler_object()`, and how `denoise_mask` is prepared and threaded to the model. Core of the masking integration.
- `comfyui-source/comfy-sample.md` — (topic 2) full `comfy/sample.py`: `prepare_noise`, `fix_empty_latent_channels`, `sample_custom` (→ `comfy.samplers.sample` → `CFGGuider`). Read for how noise/latent are prepped pre-sampling.
- `comfyui-source/comfy-extras-nodes-differential-diffusion.md` — (topic 4) the complete `DifferentialDiffusion` node — the per-pixel threshold-release `denoise_mask_function` (`forward`) and its sigma→timestep threshold math. **The exact mechanism this project mirrors.** Read for the gradient/mask math.
- `comfyui-source/comfy-model-patcher-denoise-mask-hook.md` — (topic 4/3) `comfy/model_patcher.py`: **`set_model_denoise_mask_function`** (the one-line `self.model_options["denoise_mask_function"] = fn` setter that `DifferentialDiffusion.apply` installs) alongside its sibling `set_model_*` hook-setters, plus `ModelPatcher.__init__` (`model_options = {"transformer_options":{}}`), `ModelPatcher.clone()` (`copy.deepcopy(model_options)`), and the module-level `create_model_options_clone`. Read for exactly how a node installs the hook and how `model_options` is stored/cloned.
- `comfyui-source/comfy-patcher-extension-copy-nested-dicts.md` — (topic 4/3) `comfy/patcher_extension.py` `copy_nested_dicts` (+ `merge_nested_dicts`) — the shallow-per-level recursive copy behind `create_model_options_clone`, i.e. how the sampling pipeline clones `model_options` per `guider.sample(...)` call without deep-copying the installed hook callable/tensors. Read with the model-patcher file.
- `comfyui-source/comfy-sd-vae.md` — (topic 5) `comfy/sd.py` `VAE` class: `encode`/`decode` shapes + dtype/device, `vae_encode_crop_pixels` (8-px `downscale_ratio` alignment), OOM→tiled fallback, `vae_dtype`. Read for VAE round-trips per tile.
- `comfyui-source/nodes-vae-encode-decode-mask.md` — (topic 5) `VAEEncode`/`VAEDecode`(+Tiled), `VAEEncodeForInpaint` (downscale align + `grow_mask_by`), and **`SetLatentNoiseMask`** (populates `samples["noise_mask"]` consumed by `guider.sample(denoise_mask=...)`). Read for the node-level VAE + noise-mask wiring.
- `comfyui-source/comfy-extras-nodes-mask.md` — (topic 8) core mask idioms: `composite` (bilinear mask composite), `MaskComposite`, `FeatherMask`, **`GrowMask`** (scipy grey erosion/dilation), `ThresholdMask`. Read for reusable feather/grow/blend patterns.
- `comfyui-source/comfy-utils-progressbar-upscale-mask.md` — (topic 6) `ProgressBar`, `common_upscale` (reference upscale idiom), `get_tiled_scale_steps`/`tiled_scale`, `reshape_mask` (used by `prepare_mask`). Read for progress reporting + mask reshape.
- `comfyui-source/comfy-model-management-interrupt.md` — (topic 6) `InterruptProcessingException`, `throw_exception_if_processing_interrupted`, `before_node_execution`. Read to make the tile loop interruptible.
- `comfyui-source/comfy-sampler-helpers.md` — (topic 3) `comfy/sampler_helpers.py` bodies: **`prepare_mask`** (the one-line `reshape_mask(...).to(device)` step that turns `noise_mask` into the sampler's `denoise_mask`), `convert_cond`, `prepare_sampling`/`_prepare_sampling` (additional-model gathering + `load_models_gpu`), `cleanup_models`. Closes the noise_mask→denoise_mask chain end to end.
- `comfyui-source/comfy-model-management-device-offload.md` — (topic 6) `comfy/model_management.py` device/offload basics: `load_models_gpu`/`free_memory`/`get_free_memory`, `current_loaded_models`/`LoadedModel`, `VRAMState`, `intermediate_device`/`vae_device`/`vae_offload_device`/`unet_offload_device`, `soft_empty_cache`/`unload_all_models`. Read for what happens to model residency when calling `guider.sample(...)`/`vae.encode(...)` repeatedly per tile (the loop is safe/idempotent; core manages eviction — the node need not).

## scipy-docs/ — SciPy 1.15.1 manual (docs.scipy.org) — brief topic 8

- `scipy-docs/distance-transform-edt.md` — `scipy.ndimage.distance_transform_edt` reference (signature, params, returns, Notes, worked examples). Read for the EDT-based region-mask fade (distance-to-background → feather gradient).

## tiling-research/ — seam-handling & gradient background (web, papers + repos) — brief topics 7–8

- `tiling-research/differential-diffusion-paper-method.md` — arXiv:2306.00950 Method section: Algorithm 1, the `mask = μ_s ≥ t/k` threshold-release, change-map down-sampling, gradual injection + future hinting. Read for the theory behind topic 4's node.
- `tiling-research/differential-diffusion-repo-implementation.md` — exx8/differential-diffusion `diff_pipe.py`: pipeline signature + the concrete `thresholds`/`masks = map > thresholds` convex-mix loop (runnable form of Algorithm 1). Read for the reference implementation.
- `tiling-research/multidiffusion-paper-method.md` — arXiv:2302.08113 Method + Panorama/Region apps: the least-squares FTD fusion objective and its closed-form weighted-average solution (Eq. 5), bootstrapping. Read for the seam-averaging theory.
- `tiling-research/multidiffusion-repo-panorama-implementation.md` — omerbt/MultiDiffusion `panorama.py` (full): sliding overlapping windows + `value/count` per-pixel averaging (Eq. 5, uniform weights). Read for the simplest concrete seam-blend.
- `tiling-research/mixture-of-diffusers-repo.md` — albarji/mixture-of-diffusers: tile geometry, Gaussian/quartic per-pixel weight kernels, weighted-accumulate-then-normalize blend. Read for weighted (non-uniform) seam blending.
- `tiling-research/ultimate-sd-upscale-seam-fix.md` — Coyote-A/ultimate-upscale: redraw modes (linear/chess) + seam-fix passes (half-tile / band-pass gradient masks, mask-blur). Read for post-hoc seam-repair strategies.
- `tiling-research/comfyui-tiled-diffusion.md` — shiimizu/ComfyUI-TiledDiffusion: how MultiDiffusion/Mixture-of-Diffusers blending is wired into ComfyUI as a `model_options`/`apply_model` patch (`split_bboxes`, gaussian weights, per-step buffer averaging), plus the node's options. Read for the ComfyUI-native integration pattern.

## Coverage notes

### Cross-source inconsistencies

1. **V1 vs V3 node schema (version drift — the key decision).** The entire `comfy-docs/` set and the 0.3.45 repo source use the classic **V1** schema (`INPUT_TYPES`/`RETURN_TYPES`/`FUNCTION`/`CATEGORY`/`NODE_CLASS_MAPPINGS`). Current docs.comfy.org guidance is increasingly written against the newer **V3** schema (`io.ComfyNode`/`define_schema`/`comfy_entrypoint`), captured only as a flag in `comfy-docs/v3-migration-overview.md`. The 0.3.45 repo (`comfyui-source/comfy-extras-nodes-custom-sampler.md`, `...-differential-diffusion.md`) is authoritative for the target version and uses V1 — so **build against V1**; treat V3 pages on docs.comfy.org as forward-looking, not applicable to 0.3.45.

2. **MASK shape disagreement inside the official docs.** `comfy-docs/datatypes.md` (MASK section) states shape `[H,W] or [B,C,H,W]`; `comfy-docs/images-and-masks.md` states MASK is `[B,H,W]` (channel dim implicit). The 0.3.45 code path normalizes to `[B,1,H,W]` internally (`SetLatentNoiseMask` and `reshape_mask` both `reshape((-1, 1, H, W))` — see `comfyui-source/nodes-vae-encode-decode-mask.md`, `comfyui-source/comfy-utils-progressbar-upscale-mask.md`). Treat node-boundary MASK as `[B,H,W]` and expect an internal reshape to `[B,1,H,W]` for the noise/denoise mask.

3. **`requires-comfyui` version scheme.** `comfy-docs/registry-specifications.md` illustrates `requires-comfyui` with `>=1.0.0`-style values, but the ComfyUI core version in scope is `0.3.45` (the `1.x` line is the separate `comfyui-frontend-package`). When pinning compatibility for this node, use the `0.3.x` core scheme (e.g. `requires-comfyui = ">=0.3.45"`), not the doc's illustrative `1.0.0`.

### Referenced-but-not-captured helpers (behavior available, bodies not in set)

- **RESOLVED (post-run):** `comfy.sampler_helpers.prepare_mask` / `prepare_sampling` / `cleanup_models` / `convert_cond` bodies are now captured verbatim in `comfyui-source/comfy-sampler-helpers.md`.
- **RESOLVED since round 1:** `ModelPatcher.set_model_denoise_mask_function` and `create_model_options_clone` / `copy_nested_dicts` are now captured verbatim (`comfyui-source/comfy-model-patcher-denoise-mask-hook.md`, `comfyui-source/comfy-patcher-extension-copy-nested-dicts.md`).

### Fidelity

All 35 files carry a source header (source path/URL, version, retrieval date) and reproduce source verbatim (HTML→Markdown for docs; code copied unchanged). The `tiling-research/` files and the new `comfyui-source/comfy-model-*` / `comfy-patcher-extension-*` files append clearly-delineated gatherer-authored connective paragraphs ("Key mechanism"/"Mechanism summary"/semantics notes) after the verbatim quoted code — connective explanation of the quoted code, not represented as source text. `comfy-sd-vae.md`, `v3-migration-overview.md`, and `comfy-model-management-device-offload.md` mark internal omissions inline (`# ... [omitted, lines …]` / excerpt banners / per-section line ranges). No file reads as a paraphrase in place of a verbatim copy. Note: the round-2 gap-fill files use a `# Title` + `**Source**`/`**Repo**`/`**Version**`/`**Retrieved**` header style (scipy uses plain `Source:`/`Version:`/`Retrieved:` lines) rather than the blockquote style of `comfy-docs/`; all carry the three required fields.

### Open gaps

Round-1 gaps are all **resolved** by the gap-fill round: (1) scipy 1.15 `distance_transform_edt` → `scipy-docs/distance-transform-edt.md`; (2) `ModelPatcher.set_model_denoise_mask_function` + `model_options` storage/clone → `comfyui-source/comfy-model-patcher-denoise-mask-hook.md` + `comfyui-source/comfy-patcher-extension-copy-nested-dicts.md`; (3) `comfy/model_management.py` device/offload basics → `comfyui-source/comfy-model-management-device-offload.md`.

The round-2 open gap (`comfy/sampler_helpers.py` bodies) was closed post-run: captured verbatim from the local 0.3.45 tree into `comfyui-source/comfy-sampler-helpers.md`. **No gaps remain open.**
