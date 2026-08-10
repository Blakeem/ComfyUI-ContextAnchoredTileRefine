# Saved A/B inputs

- `krea2-00676-base-768x1024.png` — the 768x1024 Krea 2 base generation that fed
  `ContextAnchoredTileUpscaleVL` (node 364) in `output\ComfyUI-2x_00676_.png`.
  Rescued 2026-08-09 from the volatile temp previews
  (`temp\ComfyUI_temp_kcxgh_00005_.png`, pixel-identical to `behrx_00002` /
  `kcxgh_00002..4`); the workflow itself never saved the base. Identity verified
  geometrically (moon-disk centroid matches the refined output; base #1 and the
  other session images do not) — a fresh server run of the gen chain produces a
  DIFFERENT composition, so this file is the only reproducible source.
- `input\AB-refine-input.png` (ComfyUI input dir, already durable) — the same base
  after the 3x upscale stage (2304x3072, upscale-only, no diffusion): the
  refine-input snapshot and the artifact-free comparison base.

Used by `run_ab_krea2.py`.
