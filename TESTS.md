# VL node test plan

Ideas from the 2026-09-01 audit that are worth an A/B, in the order to run them. Each entry has what changes, why it should help, how to test it, and its status. Update the status as tests run.

## Baseline

The reference is the `Krea 2 8k upscale.json` workflow with caps 2048x1728, anchor 32, overlap 256, dpmpp_2m_sde and cfg 3.5.

| | pass 1 | pass 2 |
|---|---|---|
| upscale | 4x to 4096x3072 | 2x to 8192x6144 |
| denoise, steps | 0.5, 28 | 0.35, 20 |
| anchor source | live canvas | source image |
| vlm_method | captions | vision tokens and captions |
| tiles | 9 | 36 |
| interior crop | 1944x1600 | 1944x1600 |
| vision rows per tile, mean (min) | 168 (143) | 59 (42) |
| canvas px per vision cell | 128 | 256 |
| sampled px over canvas px | 1.76x | 1.99x |
| DiT forwards | 504 | 1440 |
| caption share of wall time | about 17% | about 23% |

Both arms of a test use the same seed and the same base PNG, and differ in one thing. The GPU runs one job at a time and one config per process. The final judgement is by eye in ComfyUI.

## Tests

### 1. Vision encode budget scaled to the canvas

Change: the canvas encode budget follows the canvas size at one cell per 128 px, `max(768x1024, W x H / 16)`, capped at 2 MP.
Why: pass 2 hands each tile a third of the rows pass 1 does, at twice the coarseness. At 4096x3072 the rule lands on today's budget, so pass 1 is unchanged. At 8K the cap binds and a tile gets about 127 rows at 161 px per cell.
Cost: one encode of about 1950 rows, and about 70 more text tokens per DiT eval.
Test: pass 2 with the shipped budget against the scaled budget, same seed. The owner measured the VLM breaking down past 2 MP in VLMAnchoredRemix, so the cap stays at 2 MP.
Code: `vl.GLOBAL_SLICE_BUDGET` is replaced by a function of the canvas, and `tests/test_vl.py:232` pins it.
Status: budget sweep run on the portrait 2026-09-01 (`tests-AB/run_ab_vlm_budget.py`, 3072x1728, one seam, 1824x1728 crops). Vision rows per tile were 462 at 0.79 MP, 575 at 1 MP, 1188 at 2 MP, 1763 at 3 MP and 2350 at 4 MP. Owner: every budget above 0.79 MP hallucinates more and ruins the image. Krea 2's own encoder (`krea-2/encoder.py`) trains the DiT's text conditioning at a 512 row cap, so the lever is rows per tile with a ceiling near 512, not megapixels. The 8K pass sits at 59 rows and the 4K pass at 168, both far under it. Market and face then ran at 0.15, 0.79, 2 and 4 MP (90, 462, 1188 and 2350 rows per tile). Owner verdict across the three scenes: 0.79 MP is the clear winner. The larger budgets add things not in the source (clouds, extra people, sparks, muck on the face) and drop things that are (the ivy on the tree), and they add no detail the 0.79 MP arm lacks. 0.15 MP only gets worse. Decision: the picture the VLM reads is always sampled at 0.79 MP, whether it is the entire canvas or a tile's neighborhood window, and the tile keeps every row of its slice. The hallucinations belong to the sample size, not to the number of rows injected, and the two-tile case already injects 462 rows at 0.79 MP without harm. Krea 2's 512-row training length is noted, not enforced. Built as test 2's neighborhood encode.
Status: built and staged, not committed. The caption input moved to 0.79 MP in both shipped presets in the same change.

### 2. Neighborhood vision encode

Change: each tile is conditioned from an encode of its own crop and the crops of its bordering tiles, the 3x3 block, resampled to the budget and sliced to the tile's crop. When that block covers the entire canvas, the canvas encode is shared as today, so a 2x2 grid runs as it does now.
Why: the rows describe the tile's own neighborhood at a density an encode of the entire canvas cannot reach, and the neighbors keep the tile aware of what continues across its borders.

| encode | px per cell | rows per tile at 8K |
|---|---|---|
| canvas, 0.79 MP (shipped) | 256 | 59 |
| canvas, 2 MP | 161 | 127 |
| neighborhood, 0.79 MP | 127 | 195 |
| neighborhood, 2 MP | 80 | 480 |

Cost: one encode per tile, about 0.7 s each with the text encoder resident, so about 25 s at 36 tiles.
Risk: AB26 found that independent encodes per tile moved the seam into the story, such as eye color and gaze. Neighboring windows share two thirds of their area and the sync engine consolidates every step, so the risk is smaller than in AB26 but untested. A tile at the canvas edge has a smaller window, so its cells are finer than an interior tile's.
Test: pass 2, three arms at the same seed. Shipped canvas encode, neighborhood at 0.79 MP, neighborhood at 2 MP. Check story agreement across seams first, then detail.
Code: a window per tile in `vl.build_global_slices` and in `captions.build_slice_caption_conds`, with one encode per distinct window.
Status: built and staged, not committed, without a prior A/B by the owner's decision. Each tile's window is sampled at 0.79 MP, test 1's rule, and the tile keeps every row of its slice. On a side with no neighbor tile the window grows by one core into the image, which only matters on the mask path, where a one-tile region still sees its surroundings. Windows that coincide are encoded once, so a grid of up to 2 x 2 tiles is byte-identical to before. Captions keep reading the tile's own crop.
Owner's 8K result 2026-09-02 (cyberpunk city, 8192x4608, 6x4 grid, pass 2 with vision tokens only from the clean 4K `ComfyUI-2x_00785_.png`, output `00787`): detail is better, and a leaning copy of the Empire State crown appears in the clouds of the top row. The vision tokens and captions arm (`00786`) shows no tower but a second crescent moon beside the real one. Both are duplicated salient objects, the failure class of test 5's phantom moons. Probes the same day: the slice math is correct for every tile of both 8K layouts, a top row tile's rows read as clouds under a logit lens with either encode, and the rows per tile under the window equal the 4K canvas regime, so the picture the VLM reads changed and not the row count. The window for that tile is 54% by 56% of the canvas with the tower crown at its edge. Test 10 holds the fix candidates.

### 3. Caption input size

Change: `tile_caption_megapixels` in settings.user.toml. Today's 0.147456 is a 384x384 thumbnail. 0 reads the crop at its own size, capped at 2 MP.
Why: a 1944x1600 crop is 3000 vision tokens at its own size and 144 tokens at 384x384. The VLM names what it can see, so a 5x thumbnail gives coarse captions. Qwen3-VL's position table is native at 768x768 (0.59 MP) and interpolated past it.
Cost: prefill only, about 450 tokens at 0.59 MP, against a decode of about 500 tokens.
Test: captions of the same tiles at 0.147, 0.59, 1.0 and 2.0 MP through `vlm-caption-probe.json`, then one render pair at the best size. The 2 MP limit was measured for the vision encode and may differ for captions.
Status: rendered on the portrait 2026-09-01 at 0.15, 1, 2, 3 and 4 MP (`AB_vlmbudget-portrait__cap-*.png` with the captions beside each as `-captions.txt`). Caption time for two tiles grew from 38 s to 50 s, so the input size is cheap. A larger input reads finer detail and the render follows every word of it. The calendar photo's ground was "a wooden surface" at 0.15 MP, "a roof" at 1 MP and "a paved surface" from 2 MP up, and each render drew that ground. The 3 MP caption invented "two pairs of scissors" and the render turned a saw into scissors. The 3 and 4 MP captions added "textured with dirt and age" and those renders put dirt on the cheeks. Owner: the captions make the man far older, which is a prompt problem, and 0.15 MP looks best. A small input holds the VLM to the main things. Market and face then ran at 0.15, 0.79, 2 and 4 MP (`AB_vlmbudget-market__cap-*.png`, `AB_vlmbudget-face__cap-*.png`). On both scenes every size named the main content correctly and invented no object. The larger inputs add material words and grow the text, 293 to 368 words on the market and 298 to 486 on the face. The 4 MP market caption added "the deep blue of the sky" over a hazy white sky. Eye colour on the face scene, tile 0 then tile 1, was unnamed and blue at 0.15 MP, green and green at 0.79 MP, unnamed and light at 2 MP, green and green at 4 MP, against hazel eyes in the source. The attribute naming wobbles at every size, which is test 4's problem, not a size effect. Owner decision after the three scenes: the caption input is sampled at 0.79 MP like the vision encode, so the shipped presets move to 0.786432. In build with test 2.

### 4. Global caption as context for tile captions

Change: the entire image caption is given to the VLM inside each tile instruction as context, and only the tile answer is encoded. Today the artwork preset prepends the style caption to the tile caption text instead.
Why: C-Upscale (Qian et al., IJCV 2025) captions each region with the global description as context. AB27 found the same eyes captioned "greenish-gray" in one tile and "light blue" in the next, which the DiT rendered as heterochromia. Context lets the VLM use one name and one set of attributes across tiles.
Risk: the 4B VLM may describe global content that is not in the crop.
Test: manual, no build. In `vlm-caption-probe.json`, crop one tile out of an image and feed the tile instruction with the entire image description as context. Repeat across several images and prompt wordings. The test passes when the answer names only what is in the crop and uses the global names. Build it only once a wording is consistent.
Status: not started.

### 5. CFG-free tail

Change: the uncond branch is skipped once the noise level is below the guidance floor. The hook sets each lane guider's cfg to 1.0 at those steps, since comfy reads it at every eval and skips the uncond forward at 1.0.
Why: Kynkäänniemi et al. (NeurIPS 2024) find guidance unnecessary at low noise and use σ in (0.28, 5.42] for SDXL. On Krea 2 that floor is t below 0.19.

| pass | steps below the floor | forwards saved |
|---|---|---|
| pass 1 | 4 of 28 | 7% |
| pass 2 | 4 of 20 | 10% |

Test: pass 2 with and without the tail at the same seed, with the wall time measured and the last steps' texture checked by eye.
Status: not started.

### 6. Denoised consolidation after each eval

Change: a barrier after each lane's model eval, where the returned denoised windows are feathered into one canvas prediction and each lane gets its window back before the sampler uses it. The pre-step hook stays for the ring scatter.
Why: dpmpp_2m and dpmpp_2m_sde keep the previous prediction per lane, so shared cells carry two histories. With one shared prediction, both lanes compute the same update on every diffused shared cell.
Risk: it may lower the fine detail seen at 8K.
Test: in isolation. A two lane unit test asserting equal denoised on shared cells, then pass 2 with and without it at the same seed on a scene with many bands.
Code: `stepper._LaneModel.__call__` and a second hook in `sync.py`.
Status: not started.

### 7. Global reference latent

Change: a 1024x768 downsample of the canvas is VAE encoded once and attached to every lane's positive as a reference latent with method "index". Krea 2's DiT reads reference tokens at index 1.
Why: the reference gives each tile the entire picture through the image stream, which is how Krea 2 was trained to read a reference.
Cost: 768 tokens per eval, about 6%.
Risk: AB05 found a tile sized self reference suppressed phantoms but rearranged content. `index_timestep_zero` is broken on base Krea 2.
Test: pass 2 with and without the reference, same seed.
Code: `sync.build_lane_guiders` attaches `reference_latents` and `reference_latents_method` to each lane's positive.
Status: not started.

### 8. Batched caption decoding

Change: captions for several tiles decode in one batched generate loop.
Why: captions are about 23% of pass 2. comfy's generate loop is single stream and decode is bandwidth bound, so 6 to 8 streams cost about one stream per token.
Route: a PR to ComfyUI adding batch support to `BaseGenerate.generate` with a stop and a repetition history per row, or an own loop over comfy internals, which `docs/vl-conditioning-encode-cost.md` section 11 rejects as silent failure coupling.
Test: time the caption phase per tile today, then batched. Expected 4x to 6x on the caption phase, about 15% off pass 2.
Status: not started.

### 9. Low frequency pull toward the source

Change: after the consolidation, the shared prediction's low frequency component is pulled toward the source latent with a weight that decays to zero over the first half of the schedule. One blur pass on the canvas, never per tile.
Why: ResMaster (Shi et al., AAAI 2025) and C-Upscale hold low frequency structure to a low resolution reference. The dark scene tone drift at denoise 0.5 (face +6.9, cybercity +9.1 per 255) is low frequency.
Test: first reproduce the drift with the current captions on the face and cybercity scenes. If it is gone, skip this test. Otherwise run arms at weight 0.05, 0.1 and 0.2.
Status: not started.

### 10. Global context for the window encode

Change: each window encode carries two images in one sequence, the entire canvas sample first and the tile's window second, and the tile's rows are sliced from the window block. The row count the DiT reads is unchanged.
Why: test 2's phantoms are copies of the window's salient object. The VLM reads the window as a complete picture, and the window rows carry that picture's summary. With the canvas in front of the window the language model reads the window as a part of the canvas. Verified 2026-09-02 with the text encoder alone. comfy accepts the two images, the stream is [vision_start, 777 canvas cells, vision_end, vision_start, 756 window cells, vision_end, tail], the canvas block is identical to a plain canvas encode since attention is causal, and the window rows move with depth under the canvas context, with a cosine of 0.996 to the context free rows at layer 2 and 0.80 at layer 35.
Cost: one more 0.79 MP image through the vision tower and 780 more rows through the language model per window, so the vision pre pass about doubles, about 17 s more at 24 windows.
Variant: the two scale slice adds the tile's slice of the canvas block, about 70 rows at 8K, in front of its window slice. It is free once the two image encode exists and is the counterweight test for the rows that describe the tile at the coarse scale.
Rejected: arithmetic between global and local rows. Every row the DiT reads must be a real language model state, and rows scaled to 0.5 already melted the render. All canvas rows in every tile is the duplicated demand of test 5's phantom moons.
Test: 8K pass 2 with vision tokens only from `ComfyUI-2x_00785_.png`, one arm per process. Shipped window (`00787`, done), global context, two scale slice. Optional fourth arm, the canvas encode at 2.5 MP, which matches the window's 122 px cells and separates the row count from the picture.
Code: `vl._encode_canvas` takes the canvas copy and the window copy, `vl.encode_windows` passes the canvas copy, `vl.slice_indices` takes the window block's row offset, and the ledger's encode weight doubles.
Status: reproduced on one tile 2026-09-02. `tests-AB/run_ab_tile_phantom.py` refines tile r0c1 of the 8K canvas alone through the sync engine, with the node's own window slice, the owner's pass 2 settings and model, the full run's noise slice at seed 42, and the 32 px anchor band frozen through the mask path. Two minutes per arm. With the shipped window slice the pure cloud crop became an entire city with the Empire State tower at denoise 0.35 (`AB_tilephantom-r0c1__window-d0.35-s42.png` beside `__source.png`), a stronger form of the full run's single tower, and seed 1234 gave a different city. Tile r0c3, whose crop holds the tower top and the moon, stayed faithful with the same slice.
Single lane arms on r0c1, all at seed 42. The canvas slice (70 rows), the global context encode and the two scale slice each gave a city. The window slice at cfg 1 gave a city, so CFG and the negative are not the driver. The empty positive kept the clouds. The text prompt "dark storm clouds in a night sky" kept the clouds and added lightning. So the vision rows of a tile with no anchoring content carry the picture's identity, a city, and the DiT builds it where the local signal is weak. In the full run r0c1 is the only pure cloud tile in the top row, its neighbors are each anchored by real content and hold its overlap bands to clouds every step, and one tower grows in the free center. The old canvas slice does the same in a single lane, so the window did not create this failure, it made the rows for that tile name the tower at its edge and pushed a damped case over the line.
Where the demand enters. An isolated encode of the tile's own crop kept the clouds, at 768 rows and at 204 rows (0.21 MP, the window's density), so row density is not the cause. The crop encoded after its window as language model context also kept the clouds. So the tower's cells carry what the vision tower saw with them, a picture that contains the tower, and language model context does not add it. The window slice at denoise 0.25 still built a city across half the tile. On the content tile r0c3 the isolated crop encode drifted, the clouds beside the burning spire became smoke plumes, and the empty positive drifted further, the Empire State top became a different art deco spire with searchlights. So the vision rows keep a content tile's identity, and the picture they are cut from decides an empty tile's fate.
Block mode (`--block`, the tile and its five neighbors as one six lane sync run with the full run's rects, 12 minutes per arm) reproduces the full run. The window slice grows a leaning copy of the Empire State crown in r0c1's clouds beside a tilted building fragment, as in `00787`. The old whole canvas slice (56 to 79 rows per tile) grows the same two phantoms, larger, so the window did not introduce this failure and the claim that it never happened before was not tested on this scene at 8K. The crop-context arm is clean, no towers, the megastructure and the tower keep their identity, the clouds stay clouds, and a few small debris flecks are the only invented detail (`AB_tilephantom-block-r0c1__{source,window,canvas,crop-context}-d0.35-s42.png`).
Two more block arms at the owner's request, both caption free. The crop alone (every tile's cells from its own crop, 768 rows) keeps r0c1's clouds clean and drifts the content tiles, a tornado swirl beside the Empire State tower and a gouge in the megastructure. The crop cells followed by the tile's whole canvas slice (crop-global, 847 rows) keeps the scale the canvas slice carries, brings a burning smoke mass into r0c1's sky, and drifts the content tiles the same way. So the window and canvas cells keep content faithful and grow towers in an empty tile, and crop cells fix the empty tile and unsettle content tiles. The crop is sampled at a 1.9x downsample at 0.79 MP, the window at 3.8x and the canvas at 6.9x, and test 1 found the VLM invents detail as the sampling gets finer, so the next block pair samples the crop at 0.21 MP, the window's density.
Density matched block arms (the crop sampled at 0.21 MP, 204 cells, the window's 122 px per cell). Crop alone: the megastructure intact again, so the gouge came from the crop's fine sampling, the smoke swirl beside the tower stays and black debris floats near the spire. Crop-global: the owner's best so far, the debris fans out from the tower as if it came from it. Crop-context: the megastructure broken again and the swirl larger, so language model context brings drift back even at the coarser sampling. Mixed, window cells on the five content tiles and crop-context cells at 0.79 MP on r0c1: content tiles faithful, no swirl and no gouge, and r0c1 carries the crop-context smoke mass. Caveat on the 0.21 MP crop-global and mixed runs: `--budget` applied to every encode of the run, so their canvas slice was sampled at 0.21 MP too, 24 cells instead of 70. The harness now samples the crop alone at `--crop-budget` and both are being re-run with the canvas at 0.79 MP.
Row order settled. A permutation of a tile's rows reproduces the unpermuted run bit for bit (a repeat with identical rows and the shuffled run differ by 0.000/255), so the DiT reads the rows as a bag, measured. The first run of each mode differs from every later one by 6/255 with visibly different details because it sampled the DAT stage's float pixels before the crop cache existed, and every later run loads the 8 bit cache. A rounding of at most half a level per pixel at the input is enough to redraw textures and floor heights, so only composition level differences count as evidence between arms.
Corrected pair with the canvas at 0.79 MP: the same pictures as the 24 cell versions, so the canvas sampling was not the lever. The all tiles crop-global at 0.79 MP crop with neighbors brings back the broken megastructure and a burning smoke mass in the sky, so the crop stays at 0.21 MP. Owner's read: `AB_tilephantom-block-r0c1__mixed-d0.35-s42-crop-global-0.21mp.png` looks perfect, and the 0.79 MP crop images share the giant billboard man, the sign of local rows too strong. Owner rejected a per tile emptiness rule (too many assumptions) after two candidates were measured: row uniformity fails, since facades are as uniform as clouds, and crop against window agreement at layer 2 separates the cyberpunk sky tile (0.572 against 0.629 next) but flags asphalt and flower tiles on the elf city at the same threshold, tiles that were fine under window cells. Owner's direction: one method on every tile, and the canvas sampled by tile count or image size so the global slice balances the local crop. First balance test, mixed with r0c1 on 204 crop cells plus 165 cells of a 2 MP canvas (378 rows, 55 to 45): no tower, no fire in the sky, the same picture as the favorite. All tiles on the crop at 0.21 MP plus a 2 MP canvas slice, no window and no selection (`AB_tilephantom-block-r0c1__crop-global-d0.35-s42-crop0.21mp-canvas2mp.png`, 336 to 414 rows per tile, the crop share 50 to 60 percent): the sky tile stays free of towers with a small ember cluster in one cloud and a debris block at its bottom edge, the megastructure keeps its shape, and the tower tile grows a smoke swirl with a debris field around the spire. The same swirl sits in the two earlier all tiles crop-global runs with the crop at 0.21 MP and the canvas at 24 and 70 cells, so the canvas share is not the lever on a content tile. Every run where the tower tile reads its own crop cells swirls, and only window cells on that tile keep the plume straight, which is what the mixed favorite did. Context helps a content tile and hurts an empty one. Next single method arms: the crop at 0.1 MP with the canvas at 2 MP, since the 0.79 MP crop drifted more than the 0.21 MP crop, and the canvas at 2 MP alone. The crop at 0.1 MP with the canvas at 2 MP (`AB_tilephantom-block-r0c1__crop-global-d0.35-s42-crop0.1mp-canvas2mp.png`, 90 to 100 crop cells and 132 to 195 canvas cells, 240 to 294 rows per tile): no tower in the sky, no swirl or debris field on the tower tile, and the megastructure intact. Against the source block the tower's plume follows the source's thin wisp up and to the left, where the favorite grew a thick plume, and the megastructure keeps the source's balcony light and smoke puff, where the favorite invented graffiti and a balcony fire. Both arms turn the source's abstract billboard art into people. This is the first single method arm with a clean sky and a faithful tower tile. The canvas at 2 MP alone (`AB_tilephantom-block-r0c1__canvas-d0.35-s42-2mp.png`, 132 to 195 rows per tile, the shipped 1.6.1 method with the budget raised): the sky tile grows a cluster of burning skyscrapers in the clouds, the largest phantom of the set, while the tower tile and the megastructure stay faithful. More canvas rows make the phantom larger, 70 rows at 0.79 MP gave two towers and 165 rows at 2 MP give a skyline, so the canvas slice alone is out at any budget. Standing single method result: the crop at 0.1 MP plus the 2 MP canvas slice. Built 2026-09-02: the node's vision tokens are the tile's crop encode plus its slice of one canvas encode, sized by the `[vision]` table in settings.toml (canvas_tokens 165, crop_tokens 100, caption_megapixels 0.786432, each count derived into a sample size, the canvas one capped at 2 MP). The neighborhood window was removed. The harness `tests-AB/run_ab_tile_phantom.py` now drives the node's own builder, with `--canvas-tokens` and `--crop-tokens`. Awaiting the owner's node tests.
Superseded: the two image encode, the tile's neighborhood window as language model context and then the tile's own crop, was the crop-context arm above. It kept the sky clean and drifted the tower tile, and the build took the crop plus canvas recipe instead.
