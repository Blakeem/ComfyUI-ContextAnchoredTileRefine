# Release note, 1.6.0 — the synchronized VL engine

Text for the `pyproject.toml` version-bump commit body. Nothing here is pushed by this
change: pushing a `pyproject.toml` change to `main` is what publishes to the Comfy
Registry, so the owner pushes after the GPU verification pass.

## What changed for the user

1. **The VL nodes now run the synchronized-tiles engine.** Tile Refine (VL) and Tile
   Upscale (VL) no longer refine one tile after another. The whole image becomes one
   shared canvas latent and every tile is a lane of a single run, all stepped together
   one sigma at a time, consolidated back into that one canvas between steps. Because
   both sides of every overlap band decode one latent, the minimum-error cut and the DC
   match are gone from this path — there is nothing left for them to correct. The live
   preview is the whole image once per step instead of one tile at a time. The base
   Tile Refine node is unchanged: it is still the raster engine, cut and DC match
   included, and it is still the path for every non-VL model.

2. **`vision tokens and captions` keeps its name and gets better and cheaper.** The
   surface now concatenates the tile's slice of ONE shared whole-image vision encode
   with that tile's own caption encoded as text, and the caption is the FULL rich one.
   Previously the caption was placed inside a whole-image encode built per tile; now
   the whole-image encode is built once for the entire run, exactly as on
   `vision tokens`. Owner-judged across all four sweep scenes: richer detail, no
   artifacts. It is now the DEFAULT `vlm_method` on both VL nodes (previously
   `vision tokens`); saved workflows keep their stored value, only newly added
   nodes pick up the new default.

3. **New `anchor_source` widget on both VL nodes.** It picks what the frozen context
   ring around each tile shows the model. `source image` (the default, and what the
   whole A/B campaign ran on) shows the unmodified input, presented on the settled lead
   schedule: maximum fidelity, so placement, style, and objects stay locked to the
   input, including its flaws. `live canvas` shows the in-progress result itself, so
   the refine may reinterpret or repair damaged content — expect more invention and
   slightly brighter output. It replaces the never-released `context_anchor_type`.

4. **Supported samplers on the VL path are enumerated.** Stepping every tile against
   one schedule needs a sampler whose model evaluations the engine can time:
   `euler`, `dpmpp_2m`, `heun`, `dpm_2`, `exp_heun_2_x0`, `dpmpp_2m_sde`,
   `dpmpp_2m_sde_gpu`, `dpmpp_2m_sde_heun`, `dpmpp_2m_sde_heun_gpu`, and
   `exp_heun_2_x0_sde`. The sampler list on the node is NOT narrowed; any other choice
   — `dpm_fast`, `dpm_adaptive`, and `uni_pc` among them — is rejected with a clear
   error before any encoding or sampling starts.

## Compatibility

Saved workflows reopen unchanged: both selects are appended after `context_overlap`, so
the frontend's positional `widgets_values` restore leaves them at their defaults.
