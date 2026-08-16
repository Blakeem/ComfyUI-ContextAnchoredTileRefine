# Image-context channels: what a model can be told about a tile, and what it costs

Working reference for deciding **which models are worth testing a new per-tile context method on**,
and what each method's hard limits are.

The question this answers: *besides the pixels in the tile itself, through what channel can a
denoiser be told what the surrounding image looks like — and does that channel need extra model
files?*

**Verified against:** ComfyUI **0.30.0** at `C:\Users\Blake\ComfyUI-Installs\ComfyUI\ComfyUI`,
on 2026-08-07. Every file:line below refers to that tree.
Note: the ComfyUI **Desktop** install at `C:\Users\Blake\AppData\Local\Programs\ComfyUI\resources\ComfyUI`
is 0.22.3 and has **no Krea 2 support at all** — do not read core facts from it.

Re-verification commands are in the appendix. Claims are tagged `DIRECT` (read from source or
measured), `DERIVED` (follows from source + a stated premise), or `INFERRED` (my reasoning, no
external check).

---

## 1. Terminology: one channel, two names

Not two mechanisms:

```
ReferenceLatent node ──sets conditioning key "reference_latents"──►
    model_base.<Class>.extra_conds  ──renames to model kwarg "ref_latents"──►
        comfy/ldm/<arch>/model.py  forward(..., ref_latents=...)
```

`reference_latents` is the name on the **conditioning**. `ref_latents` is the name inside the
**model**. Same tensors. `DIRECT` (`comfy_extras/nodes_edit_model.py:25`, `comfy/model_base.py:2458`).

---

## 2. The structure: where each channel enters a modern DiT

Using Krea 2 / Lumina (Z-Image) as the worked example. One attention sequence, four segments,
each carrying its own RoPE id triple `(axis0, row, col)`:

```
segment       built at                        RoPE (axis0, row, col)      exists when
────────────────────────────────────────────────────────────────────────────────────────────
[cap]  text   lumina embed_cap :664           (arange+1+offset, 0, 0)     always
[sig]  SigLIP lumina embed_all :686-689       (cap_len+2,  ~8r,  ~8c)     siglip_embedder != None
[ref]  ref img lumina :732-742 / krea2 :315   own index plane, origin     ref_latents non-empty
[x]    noised  lumina pos_ids_x :431-434      (cap_len+1,  i,    j)       always
────────────────────────────────────────────────────────────────────────────────────────────
                     concatenated → self-attention over the whole sequence
```

For Krea 2 specifically (`comfy/ldm/krea2/model.py:353`):
`combined = cat(context, img)`, where `context` = text **and** VLM vision rows, `img` = noised
tokens **and** ref tokens.

The two facts that matter most for tiling:

- `[cap]` gets `txtpos = zeros` — **all three axes zero** (`krea2/model.py:344`). Anything routed
  through the positive conditioning, including our VL node's vision rows, carries **no spatial
  coordinates**. `DIRECT`
- `[ref]` and `[sig]` get **real 2-D coordinates** on the same axes as the image tokens. `DIRECT`

---

## 3. Table A — channel inventory by what it costs

| Channel | Conditioning key | Extra model files needed | Spatially positioned? | Available on |
|---|---|---|---|---|
| **Reference latents** | `reference_latents` | **none** — the VAE is already in the graph | yes, real RoPE ids | flux, krea2, qwen_image, lumina/Z-Image, omnigen2, boogu, joyimage, mage_flow |
| **Channel concat (inpaint)** | `concat_latent_image` / `concat_keys` | an **inpaint-trained checkpoint** | exact, per-pixel, shape-enforced | SD1.5-inpaint, SDXL-inpaint, Flux Fill, and video families |
| **SigLIP grid** | `unclip_conditioning` → `siglip_feats` | CLIP-vision model **+ Omni weights** | yes, coarse (×8 stride) | Z-Image **Omni** only |
| **unCLIP pooled** | `unclip_conditioning` | CLIP-vision model + unclip checkpoint | **no** — pooled vector, no grid | SD2.1-unclip, SDXL revision |
| **IP-Adapter** | n/a — model patch | IP-Adapter weights + CLIP vision + custom node | tokens, no RoPE binding to the canvas | SD1.5, SDXL, Flux |
| **VLM vision rows** (this repo) | positive `CONDITIONING` | none — VLM text encoder already in the graph | **no** — `txtpos = zeros` | Krea 2 |

**The headline:** `reference_latents` is the only channel that needs *nothing beyond what a normal
workflow already loads*. Everything else costs at least one extra file.

**For SD1.5 / SDXL / Chroma there is no zero-extra-model context channel** other than swapping to
an inpaint checkpoint (which changes the model, not just the wiring).

---

## 4. Table B — which models actually consume `reference_latents`

Two independent halves must both be present: the **emitter** (`model_base.<Class>.extra_conds`
builds the kwarg) and the **consumer** (`comfy/ldm/<arch>/model.py` `forward` accepts it). An
emitter without a consumer is a **silent no-op**. `DERIVED`

| Model class | Emitter | Consumer | Verdict |
|---|---|---|---|
| `Flux` (Kontext) | :1043 | `flux/model.py:360` | **works, trained for it** |
| `LongCatImage`, `Flux2` | inherits Flux | `flux/model.py` | works |
| `QwenImage` (Image Edit) | :2377 | `qwen_image/model.py:459` | **works, trained for it** |
| `Krea2` | :2458 | `krea2/model.py:311` | **works only if `reference_latents_method` is co-set** — see §5.1 |
| `Lumina2` / `ZImage` / `ZImagePixelSpace` | :1527 | `lumina/model.py:730` | path runs on **any** Z-Image checkpoint; only Omni is trained for it |
| `Omnigen2`, `Boogu` | :2347 | `omnigen/omnigen2.py:414`, `boogu/model.py:255` | present in core 0.30, untested here |
| `MageFlow` | :2406 | `mage_flow/model.py:94` | present in core 0.30, untested here |
| `JoyImage` | :2421 | `joyimage/model.py:370` | present in core 0.30, untested here |
| `HiDreamO1` | :2245 | passed as `ref_images=` | different internal name |
| `WAN21` family | :1665 etc. | singular `reference_latent` | **video family**, different path |
| `TripoSplat` | :2192 | `triposplat/model.py:279` | 3D, out of scope |
| **`Chroma`, `ChromaRadiance`** | **inherits Flux's emitter** | **none — `chroma/model.py:277` has no `ref_latents` parameter** | **SILENT NO-OP.** Key is built, passed, dropped into `**kwargs`. Also inflates the memory estimate for nothing. |
| **SD1.5, SDXL, SD3** | none | none | **no path exists** — a UNet has no token sequence to append to |

---

## 5. Method summaries

### 5.1 Reference latents

**Mechanism.** A VAE-encoded latent is patchified and **concatenated onto the image token stream**,
each ref getting its own RoPE ids from `process_img(ref, index=i)`. The model reaches it by
attention. `DIRECT` (`krea2/model.py:310-325`, `flux/model.py:360-388`)

**Nodes.**
- `ReferenceLatent` (core, `comfy_extras/nodes_edit_model.py`) — takes LATENT, sets the key,
  chainable for multiple refs.
- `FluxKontextMultiReferenceLatentMethod`, display name **"Edit Model Reference Method"**
  (core, `comfy_extras/nodes_flux.py:155`) — sets `reference_latents_method`. Options:
  `offset`, `index`, `uxo/uno`, `index_timestep_zero`. Marked experimental.
- `ReferenceLatentPlus` (third-party) — see §5.2.

**Positional behavior — the limit that matters for tiling.**
Krea 2 calls `self.process_img(ref, index=index)` with **no `h_offset`, no `w_offset`**
(`krea2/model.py:325`). Every reference starts at coordinate `(0,0)` of its own index plane.
**A ref crop cannot be registered at its true canvas position on Krea 2.** `DIRECT`
Flux's `offset` and `uxo` methods *do* accept offsets (`flux/model.py:369-386`), so this limit is
Krea 2's, not the mechanism's. `DIRECT`

Consequence: the trick that makes our VL node work — *one global encode, each tile slices its own
rows* — has **no ref-latent analogue on Krea 2**. The choices are:
- tile-crop ref → positionally meaningless, loses global context
- whole downscaled canvas as ref → keeps global context, no positional binding to the tile

**Cost.** Refs concat at full weight into the image token stream. A tile-size self-ref roughly
doubles image tokens → ~4× image–image attention, ~2× refine time. **No strength or weight knob
exists in core.** `DIRECT`

**Tested in this project** (branch `exp/hallucination-ab`, Krea 2, 2304×3072, 2×2 tiles):
- `b_selfref` (method `index`), AB05: **total phantom suppression** — including the base
  generation's own artifacts — but Kontext-like re-composition (Δ17.8/255): the ref licenses each
  tile to re-compose prompt-faithfully. Seams held. ~2× time.
- `offset` == `index` == `uxo`: byte-identical diff profile (AB07).
- `index_timestep_zero`: **broken on base Krea 2** — severe speckle. Its batch-concat `t=0`
  embedding needs edit-trained weights.
- `b_selfref_early` (refs for the first 8/20 steps, split-sigma continuation), AB10-AB12 at CFG 5 /
  anchor 128: **total suppression, zero artifacts, clean seams, ~+50% time**, but style drifts
  painterly/cinematic (largest Δ of that round, 23/255; CFG-5 confound).

**Key lesson: "untrained" ≠ "won't work."** Krea 2 base is *not* an edit model
(`default_ref_method=None`) yet produced the strongest suppression result of the whole run. What
untrained costs is **predictability**, and one of the four methods is outright broken.

**Untested lead.** Whole-image **downscaled** canvas as a single global ref (rung-0d of the decide
run). Three stock nodes, zero code, spatially invariant — so `conds.py` already passes it through
to every tile untouched. Cheaper than a tile-size self-ref (fewer added tokens) and it is the only
ref shape that keeps global context without pretending to positional accuracy Krea 2 cannot honor.

### 5.2 ReferenceLatentPlus (third-party)

[shootthesound/comfyui-ReferenceLatentPlus](https://github.com/shootthesound/comfyui-ReferenceLatentPlus).
**Conditioning-keys only — no model patch, no forward wrapper.** `DIRECT` (read from its `nodes.py`)

| Advertised feature | Actual implementation | Assessment |
|---|---|---|
| per-image strength | `latent = latent * img_strength` | scales the **VAE latent's values**. Not an attention weight — it moves the ref off the VAE manifold. Treat any result with suspicion. |
| per-image timestep gating | partitions the schedule, sets `start_percent` / `end_percent` per region, appends refs only to active regions | real and sound — standard core conditioning mechanism |
| MediaPipe auto-masks, 1–4 images, VAE input | convenience wiring | no new mechanism |

**Its timestep gating is the same lever as our `b_selfref_early` experiment**, reached via
conditioning timestep ranges instead of split sigmas. It offers nothing mechanically new.

**Krea 2 is not in its supported list**, and Krea 2 is precisely the model that needs
`reference_latents_method` co-set or the entire ref block no-ops. Its method key is set
"for Flux models specifically". Pair it with the core `Edit Model Reference Method` node, or it
will silently do nothing on Krea 2. `DERIVED`

Its supported list: Flux1, Flux2/Klein, LongCatImage, Lumina2 / Z-Image, Wan family, Hunyuan,
Qwen-Image.

### 5.3 SigLIP grid (Z-Image Omni)

**Mechanism.** A CLIP-vision output's `last_hidden_state` is reshaped into a **spatial grid**
`(h/16, w/16)` and embedded as its own sequence segment. `DIRECT`
(`model_base.py:1515-1523`, `lumina/model.py:682-703`)

**This is the closest non-VLM analogue to what this repo does on Krea 2** — a 2-D grid of vision
features, sliceable by tile rect the way `vl.py` slices Qwen3-VL rows — but produced by a separate
SigLIP tower via `CLIPVisionEncode`, not by a VLM text encoder.

**It is strictly better positioned than our Krea 2 vision rows.** `DIRECT`:
- `axis0 = cap_feats_len + 2` — a constant plane id, one step off the image tokens' `+ 1`
- `axis1 ≈ 8r`, `axis2 ≈ 8c` — from `linspace(0, h*8-1, steps=h).floor()`, the **same two spatial
  axes the image tokens use**

Compare: our Krea 2 vision rows sit at `(0,0,0)` for every row.

**Two hard gates.**

```
Gate A — the weights          gated on CHECKPOINT CONTENT
  model_detection.py:584      needs key `siglip_embedder.0.weight`
  absent ⇒ siglip_feat_dim unset ⇒ siglip_embedder = None (lumina/model.py:610)
  ⇒ embed_all:676 short-circuits, segment never built

Gate B — the trigger          gated on DATA
  lumina/model.py:788         `if omni and len(embeds[1]) > 0`
  and omni itself = len(ref_latents) > 0  (:730)
```

Gate B means **SigLIP feats can never be used alone** — each one rides *with* a reference latent,
and the noised image itself gets `main_siglip = None` (`lumina/model.py:722`). `DIRECT`

**Status on the local checkpoints** — I read the safetensors headers on 2026-08-07:

| file | total keys | `siglip_*` keys | pixel-space (`dec_net.*`) |
|---|---|---|---|
| `ungloryhailZImage_bf16.safetensors` | 453 | **0** | 0 → latent-space `ZImage` |
| `zImagePro_v11.safetensors` | 454 | **0** | 0 → latent-space `ZImage` |

Neither can reach the SigLIP channel. The plumbing exists on every Z-Image (`Lumina2.extra_conds`
reads `clip_vision_output` off `unclip_conditioning`); the **weights that consume it do not**.
This is a per-checkpoint fact, not a Turbo-vs-base fact — only the **Omni** edit model ships
`siglip_embedder.*`.

**Unverified.** Whether the SigLIP extent `0..8h-1` coincides with the image extent
`0..H_tokens-1` depends on SigLIP input size vs generation size — I did not measure it. The
distinct `axis0` plane suggests the model reads it as a *separate image*, not a cell-to-pixel
overlay. `INFERRED`

### 5.4 Channel concat / inpaint models

**Mechanism.** A VAE-encoded image and mask are concatenated onto the latent's **channel** axis
before the first conv/patch layer. Enabled by `BaseModel.set_inpaint()`
(`model_base.py:396`, `concat_keys = ("mask", "masked_image")`), which
`supported_models_base.py:87` calls when the checkpoint carries the extra input channels. `DIRECT`

**This is the one built-in, spatially-exact context channel available to SD1.5 and SDXL.**
Registration is per-pixel and shape-enforced — strictly better positioning than any token-based
channel. The cost is that it requires an inpaint-trained checkpoint, which is a model swap rather
than a wiring change.

Also present on: `Flux` (Fill, `model_base.py:982`), `HunyuanVideoI2V`, `CosmosVideo`,
`CosmosPredict2`, `Kandinsky5`, `CogVideoX`, `WAN21`, `SD_X4Upscaler`.

**Relevance to tiling.** The concat image is naturally the tile's own crop, so this is a per-tile
channel by construction. It carries no information about *neighbors* unless the concat crop is
deliberately drawn wider than the tile — untested, and it would need the tile's own denoise mask
to keep directive 2 (never re-diffuse finished pixels). `INFERRED`

### 5.5 unCLIP pooled

CLIP-vision `image_embeds` (a **pooled vector**, no grid) added to the model's ADM/`y` conditioning
(`model_base.py:439-478`). **Not sliceable** — there are no rows to take a subset of. Useful for
global style, useless for per-tile positional context. Needs a CLIP-vision model plus an
unclip-trained checkpoint. `DIRECT`

### 5.6 IP-Adapter (third-party)

Injects gridded CLIP-ViT tokens through **extra cross-attention layers** patched into the UNet.
Works on SD1.5, SDXL, and Flux — the only channel with sliceable spatial vision tokens on the SD
family. `INFERRED` — not verified against source in this repo; no IP-Adapter pack is installed
locally.

**Fails the "no extra models" bar**: needs an IP-Adapter weights file, a CLIP-vision model, and a
custom node pack. Its tokens have no RoPE binding to canvas coordinates, so slicing them by tile
rect is not obviously meaningful the way the SigLIP grid's coordinates are. Worth a look only if
SD1.5/SDXL users turn up for the upscaler.

---

## 6. Not the same as ControlNet

ControlNet is a **different injection point**, and the analogy misleads:

```
ControlNet (SD1.5 / SDXL)
  hint ──► parallel encoder copy ──► per-block feature maps
                                          │
       UNet decoder block output h  ──►  h = h + control[block]
```

| | ControlNet | Reference latents / SigLIP |
|---|---|---|
| Operation | **addition** into block outputs | **concatenation** into the attention sequence |
| How the model reaches it | unavoidable, every block | only via attention, if it attends |
| Registration | exact, per-pixel, shape-enforced | coarse RoPE coordinates, separate plane |
| Strength knob | `strength` multiplier on the residual | **none** |

The only thing they share is being "a second image entering the denoiser."

This repo already handles a **third** route: `conds.py` swaps the per-tile control-chain hint slice
inside `guider.original_conds` — the *conditioning* route, distinct from both of the above.
See `CLAUDE.md` and the `controlnet-two-routes` note.

---

## 7. Practical triage: is a model worth testing?

Run in order. Stop at the first `no`.

1. **Does its class emit `reference_latents`?** → Table B, or the appendix grep.
2. **Does its `comfy/ldm/<arch>/model.py` `forward` accept `ref_latents`?** If not, it is a silent
   no-op (the Chroma trap).
3. **Does it need a method key?** Krea 2 does. Check `default_ref_method` in the arch's model.py
   and whether `model_detection.py` sets it for that checkpoint.
4. **Does the checkpoint carry the weights the channel needs?** Read the safetensors header —
   appendix script. This is what disqualified both local Z-Image files.
5. **Is it trained for the channel?** If not, it may still work (Krea 2 AB05) but expect
   re-composition and at least one broken method variant.

---

## 8. Open leads

| Lead | Cost | Status |
|---|---|---|
| Whole-image downscaled global ref latent on Krea 2 | 3 stock nodes, zero code | **untried** (rung-0d) |
| `b_selfref_early` at CFG 3.5, ref_steps 2–4 | existing branch | partially explored; drift may be a CFG-5 confound |
| SigLIP row-slicing on Z-Image **Omni** | needs the Omni checkpoint + new code | blocked on acquiring weights |
| Wider-than-tile concat crop on an inpaint checkpoint | new code + denoise-mask care | untried, SD-family only |
| Survey: has anyone trained a ref-latent/IP-Adapter-style adapter for architectures with no channel? | web research | not run |

A civitai popularity sweep is **not** the way to answer the mechanism question — civitai models are
overwhelmingly fine-tunes and LoRAs of the architectures already in Table B, and a fine-tune cannot
add a token path the architecture lacks. `DERIVED` The authoritative list is local and is already
enumerated above.

---

## Appendix — re-verification commands

Set `CORE=C:/Users/Blake/ComfyUI-Installs/ComfyUI/ComfyUI`, then:

```bash
# Which model classes EMIT reference_latents
grep -n 'kwargs.get("reference_latents"' $CORE/comfy/model_base.py

# Which DiTs CONSUME it (the authoritative list)
grep -rln "ref_latents" $CORE/comfy/ldm/ | grep -v pycache

# Whether a given arch's forward actually takes it
grep -n "def _forward\|def forward" $CORE/comfy/ldm/<arch>/model.py

# Whether an arch needs a method key
grep -rn "default_ref_method" $CORE/comfy/ldm/<arch>/model.py $CORE/comfy/model_detection.py

# Which classes have a channel-concat path
grep -n "def concat_cond\|self.concat_keys" $CORE/comfy/model_base.py
```

Checkpoint header inspection — no full load, reads only the JSON header:

```python
import json, struct
f = "path/to/model.safetensors"
with open(f, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n))
keys = [k for k in hdr if k != "__metadata__"]
print("total:", len(keys))
print("siglip:", [k for k in keys if "siglip" in k.lower()])     # Z-Image Omni gate
print("dec_net:", len([k for k in keys if ".dec_net." in k or k.startswith("dec_net.")]))  # pixel-space variant
```
