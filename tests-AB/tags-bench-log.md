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
