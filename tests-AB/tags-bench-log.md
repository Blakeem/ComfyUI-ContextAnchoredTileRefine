# Tile tags speed and accuracy log

Goal: a faster tags pass that scales to long prompts with no loss of accuracy against the
baseline. Accuracy first, speed second. Fewer tags are fine when they are right.

## Test set

`tests-AB/ab_tags_bench.py`. Five scenes at the owner's tile widgets (2048x1728, anchor 32,
overlap 256). Three tiles per scene: a cropped subject, a busy tile and a sparse tile. Three
prompts per scene: none, short (4 to 9 words) and long (190 to 330 words). The long prompts
name things the image holds, style words, and things the image does not hold.

| Scene | Size | Grid | Tiles | Absent things the long prompt names |
|---|---|---|---|---|
| cyber8k | 8192x4608 | 24 | 0 sky, 9 spire, 14 street | flying vehicles, laser beams |
| hangar8k | 8192x4608 | 24 | 3 ceiling, 10 turret and robot, 21 workers | astronaut, Earth, fighter squadron |
| dragon4k | 4096x2304 | 6 | 0 wing and sun, 2 bell, 4 army | wizard, second dragon, burning ship |
| market4k | 4096x2304 | 6 | 1 dome, 3 fabric and bird, 5 cart | juggler, horse and carriage, town crier |
| face3k | 2304x3072 | 6 | 1 moon, 3 face, 4 flowers and hair | raven, wolf, candles, circlet |

Accuracy judge: a vision model reads each tile crop and a copy with thirds lines, and grades
every emitted item yes, no, vague or unsure, and every position term right or wrong
(`judge/RUBRIC.txt`). Verdicts are cached per (scene, tile, item), so every arm is graded
against the same verdicts. Timing: synchronized wall time per tile, CUDA graphs off unless an
arm says otherwise, RTX 3090 Ti, Qwen3-VL 4B int8.

Metrics: `prec` = yes / (yes + no + vague). `wrong` = no / judged, the phantom risk.
`pos ok` = right terms among judged terms on yes items. `grid s` = picture stage plus the
scene's full tile count times the mean tile time.

## Log

### 1. Baseline (the committed tags pass, 508a13d)

| Prompt | Correct per tile | Precision | Wrong | Vague | Position right | Seconds per tile | 8K grid seconds |
|---|---|---|---|---|---|---|---|
| none | 14.1 | 0.865 | 0.024 | 0.110 | 73/73 | 9.6 | 230 |
| short | 14.7 | 0.804 | 0.040 | 0.156 | 75/75 | 11.4 | 275 |
| long | 7.8 | 0.626 | 0.123 | 0.251 | 59/59 | 15.5 | 373 |

8K grid seconds = 24 tiles x seconds per tile, the owner's 8K layout.

- A long prompt costs 60% more time, halves the correct tags per tile and multiplies the wrong
  rate by 5. The wrong and vague tags are raw prompt fragments: whole-scene phrases ("dragon
  attacks a medieval harbor town"), phrases that fix a place the tile contradicts ("crane hook
  holds a blue humanoid robot at the upper right") and things absent from the tile.
- Positions are right on every judged term of a correct tag. The six strips place well. Every
  wrong position the judges found was on a whole-scene phrase, which is vague anyway.
- Time: the propose generate is 60 to 75% of the tile, at 100 ms per written token with CUDA
  graphs off. The prompt adds reading cost only through the reply, which grows from 57 to
  121 tokens and stops at the 128 cap.

### 2. Determinism

A second baseline run matched the first on all 45 tiles (same replies, same items). Tile time
differed by 1 to 2%. Any change in items between two arms is therefore the arm's doing.

### 3. CUDA graph decode (arm base+graphs): ADOPT

Root cause of core issue #16441 in this core: each decoder layer keeps the CUDA graph it
captured during a generate (`module._comfy_graph`, comfy/model_prefetch.py), bound to that
generate's KV cache and decode buffers. Core frees them when the generate returns but keeps the
graph, and the next generate replays it against freed memory, which trips the device side
assert. Core's own `comfy.model_prefetch.cleanup_prefetch_queues()` drops the graphs. Core runs
it between nodes, never between two generates inside one node, which is why one generate per
node works. The open upstream PR #16476 resets cache objects that this core builds fresh on
every call, so it does not cover this.

Fix: graphs stay on, and cleanup_prefetch_queues runs after every generate.

- Same reply and same items on all 45 tiles, 45 image generates in one process, no assert.
- Seconds per tile 12.1 -> 4.8 over all prompts. Decode 100 -> 16.5 ms per token.

### 4. Load skip and one shared vision encode (arm base+fast): ADOPT

- Load skip: `clip.load_model` returns at once while the CLIP heads core's loaded list with its
  weights in place. Core's `load_models_gpu` has no fast path for a loaded model and costs about
  0.1 s, and a tile made 8 calls (1 propose, 1 verify, 6 strips).
- Shared vision encode: the propose generate and the verify request read the same 1 MP
  picture, so its vision tower output is computed once per tile, inside logit_classifier's
  determinism window.
- Same reply and same items on all 45 tiles, with graphs on and with the prompt terms arm.
- Seconds per tile 4.73 -> 3.46 (graphs on). From the start: 12.1 -> 3.46.

### 5. Prompt terms (arm terms)

One text-only generate per picture lists the visible things the prompt names, with a stop when
two listed terms in a row hold a word the prompt lacks. The terms go to every tile as
candidates, and each tile's propose reads no prompt.

| Prompt | Correct per tile | Wrong per tile | Vague per tile | Precision |
|---|---|---|---|---|
| short, baseline | 14.7 | 0.73 | 2.87 | 0.804 |
| short, terms | 14.5 | 0.40 | 2.87 | 0.816 |
| long, baseline | 7.8 | 1.53 | 3.13 | 0.626 |
| long, terms | 15.3 | 0.60 | 1.27 | 0.891 |

With no prompt the arm is the baseline. The picture stage costs 1 to 4 s once per picture with
graphs on.

### 6. Offline filters on the recorded runs

- Verify threshold 0.99 or 0.999: removes few wrong tags and many correct ones. Rejected.
- Style sort (p(style) >= 0.9) on every tag: small gain. Superseded by the concreteness choice.
- Concreteness choice, a text-only "is this one physical thing" question per tag string:
  the first wording also dropped "red moon", "green cliffs", "army of soldiers" and "lights".
  The second wording counts landscape features, groups and light sources as things.
- First N proposed tags: wrong and vague tags sit at the tail of the greedy reply. N = 25
  keeps most correct tags.
- Agreement rule (keep a prompt term only when its head noun is among the tile's own tags):
  removes all 9 wrong prompt terms and 77 correct ones. Rejected.

### 7. v2 arm (terms + concreteness wording 2 at 0.9 + propose stopped at 25 tags)

| Prompt | Correct per tile | Wrong per tile | Vague per tile | Precision | Seconds per tile |
|---|---|---|---|---|---|
| none | 12.3 | 0.07 | 0.20 | 0.979 | 3.17 |
| short | 12.5 | 0.07 | 1.27 | 0.904 | 3.00 |
| long | 13.5 | 0.60 | 0.27 | 0.940 | 3.04 |
| all | 12.8 | 0.24 | 0.58 | 0.940 | 3.07 |

Baseline all: 12.2 correct, 0.89 wrong, 2.60 vague, 0.778 precision, 12.1 s per tile.

- The short prompt's vague tags were prompt terms that are not things ("huge", "being built",
  "attack", "florence"). No concreteness threshold separates them from real things with
  mid scores ("chinese characters" 0.45, "banners" 0.69).
- On the long prompt about 40 prompt terms filled the 48 candidate cap before the model's own
  tags, which pushed correct model tags out.

### 8. v3 arm (prompt terms after the model's tags, prompt terms at the tile threshold)

Changes from v2: extraction wording 2 (lists physical things only, which drops "huge",
"attack" and style words, and runs 40 to 50% faster), merge puts the model's 25 tags first and
the prompt terms after them, cap 64.

| Set / prompt | Correct per tile | Wrong per tile | Vague per tile | Precision |
|---|---|---|---|---|
| main, long | 17.4 | 1.13 | 0.20 | 0.929 |
| main, all | 14.3 | 0.42 | 0.31 | 0.951 |
| hold-out, xlong | 15.1 | 1.38 | 1.00 | 0.864 |
| hold-out, all | 12.8 | 0.83 | 1.00 | 0.874 |

- The model's own tags survive the long prompt now. The wrong tags are prompt-only terms that
  pass the verify statement as a near name for what is there ("wooden carriage" for a cart,
  "raven" for a dark bird shape), with whole-tile scores from 0.9 to 0.9999.

### 9. Hold-out set

Never used for tuning. Scene city2 (a second 8K cyberpunk city, tiles 1, 10, 19) plus one or
two unseen tiles of each main scene: 8 tiles. Prompts: none, tags (a comma list of the long
prompt's things) and xlong (the long prompt plus more absent things plus a style tail, 400 to
520 words).

### 10. Production (prod arm): prompt-only terms need 0.9999

v3 with one change: a prompt term the tile's own list lacks must score 0.9999 on the verify
statement. A term both lists name keeps the tile threshold 0.9. Ported to tags.py and
settings.toml as `prompt_tags_verification_threshold`. The prod arm runs the shipped code.

Main set:

| Prompt | Correct per tile | Wrong per tile | Vague per tile | Precision | Seconds per tile | Picture seconds |
|---|---|---|---|---|---|---|
| none | 12.5 | 0.07 | 0.20 | 0.979 | 3.16 | 0.10 |
| short | 13.0 | 0.07 | 0.40 | 0.965 | 2.95 | 0.57 |
| long | 15.6 | 0.27 | 0.20 | 0.971 | 3.09 | 2.01 |
| all | 13.7 | 0.13 | 0.27 | 0.972 | 3.07 | 0.89 |

Hold-out set:

| Prompt | Correct per tile | Wrong per tile | Vague per tile | Precision |
|---|---|---|---|---|
| none, base+fast | 10.6 | 0.38 | 2.12 | 0.810 |
| none, prod | 10.5 | 0.25 | 0.75 | 0.913 |
| tags, base+fast | 12.1 | 0.88 | 4.25 | 0.703 |
| tags, prod | 11.8 | 0.25 | 0.88 | 0.913 |
| xlong, base+fast | 5.6 | 1.50 | 2.88 | 0.562 |
| xlong, prod | 12.8 | 0.38 | 0.88 | 0.911 |
| all, base+fast | 9.5 | 0.92 | 3.08 | 0.703 |
| all, prod | 11.7 | 0.29 | 0.83 | 0.912 |

base+fast holds the baseline's items, so its rows are the baseline's accuracy.

- Against the baseline, on both sets: fewer wrong tags and fewer vague tags at every prompt
  length, and more correct tags on every long prompt. Positions stay right on every judged term.
- With no prompt or a short prompt, the main set keeps 1.6 to 1.7 fewer correct tags per tile
  (14.1 -> 12.5, 14.7 -> 13.0). The 25-tag stop and the thing check cause it. The hold-out set
  keeps 0.1 to 0.3 fewer. This is the trade the goal allows: fewer tags, and the ones kept are
  right.
- Time: 12.1 -> 3.07 s per tile. Tile time no longer grows with the prompt. The prompt costs
  one picture stage of 1 to 3 s, once per picture. An 8K grid of 24 tiles: 167 -> 42 s.
- Remaining ideas, not tried: one packed pass for the six strips (about 0.3 s per tile) and a
  batched decode over several tiles.

### 11. Thing check packing (`probe_thing_pack.py`): keep packing

A review flagged that the thing check packs each tile's new tags into one request, so a tag's
p(other) depends on the tags beside it and on which tiles were cache hits. The probe replays every
recorded prod run's packs against one tag per request.

| Measure | Value |
|---|---|
| Tags | 1225 |
| Median p(other) change | 0.001 |
| 99th percentile change | 0.12 |
| Largest change | 0.21 |
| Verdicts that flip at 0.9 | 7 (0.6%) |
| Thing check time, packed | 43 s |
| Thing check time, one tag per request | 258 s (210 ms per tag) |

- Every flip is a tag packed scoring keeps and single scoring drops. The judged ones are correct
  ("red flags" yes, "tapestry" yes, "green eyes" unsure).
- Single scoring costs 6 times the thing check time and drops correct tags, so the packed check
  stays.
