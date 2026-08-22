"""progress.py: the ledger's PURE units — segments, re-fit, the clamp rule, the shim, the
caption mapping, the sampling-budget arithmetic and the status line.

Every RUN-LEVEL case (a whole synthetic VL run, two pictures, the per-vlm_method segment
order, finalization) lives in tests/test_sync.py instead, where the engine harness that
drives refine_sync already exists. Nothing here touches an engine: the status-line cases
below drive the ledger with the SAME call sequence sync.py makes, which those engine cases
pin separately.

CPU only, no model loads. `comfy_stubs` supplies comfy.utils.ProgressBar, whose recording
stand-in is what "one real bar" is counted against: a bar the shim routed is a plain object
of the shim's own class and never lands in that list.
"""
import sys
import types

import pytest
import torch

from context_anchored_tile_refine import captions, progress, stepper

# 4 steps ending at 0, so the final-step exception in EVALS_PER_STEP engages.
SIGMAS = torch.linspace(0.8, 0.0, 5)

# The hidden UNIQUE_ID a node is handed; ComfyUI's is a string.
NODE_ID = "7"


def scaled(units):
    # progress.py carries fractional budgets and comfy's bar carries integers; this is the
    # one conversion, spelled out here so every expectation below is in budget units.
    return round(units * progress.EMIT_SCALE)


def named_sampler(name):
    # A KSAMPLER whose only job is to carry a resolvable name into stepper.plan_evals.
    import comfy.samplers

    def sampler_function(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
        raise AssertionError("the timing-only sampler is never run")

    sampler_function.__name__ = f"sample_{name}"
    return comfy.samplers.KSAMPLER(sampler_function, {}, {})


def inner_bar(total):
    # A bar constructed the way core constructs its own — through the comfy.utils ATTRIBUTE,
    # which is exactly what the shim replaces.
    import comfy.utils

    return comfy.utils.ProgressBar(total)


# ---- the plan --------------------------------------------------------------


def test_the_plan_follows_the_code_order_per_vlm_method():
    # PHASE ORDER: sync.build_tile_positives writes the captions FIRST and builds the
    # conditioning after, so the plan must never put an encode before the captions that feed
    # it. "captions" pays no canvas encode at all — only its per-tile text encodes.
    vision = progress.build_plan(captions.VLM_METHOD_VISION, 20)
    combined = progress.build_plan(captions.VLM_METHOD_VISION_CAPTIONS, 20)
    caption_only = progress.build_plan(captions.VLM_METHOD_CAPTIONS, 20)

    assert [name for name, _units in vision] == [
        progress.VISION_ENCODE, progress.CANVAS_ENCODE, progress.SAMPLING, progress.DECODE]
    assert [name for name, _units in combined] == [
        progress.CAPTIONS, progress.VISION_ENCODE, progress.CANVAS_ENCODE, progress.SAMPLING,
        progress.DECODE]
    assert [name for name, _units in caption_only] == [
        progress.CAPTIONS, progress.CAPTION_ENCODE, progress.CANVAS_ENCODE, progress.SAMPLING,
        progress.DECODE]
    assert progress.VISION_ENCODE not in [name for name, _units in caption_only]


def test_the_upscale_nodes_two_extra_phases_lead_the_plan():
    # The model pass is skipped entirely without an upscale_model; the CLIP load is not,
    # because the upscale node's first CLIP call pays it either way.
    with_model = progress.build_plan(captions.VLM_METHOD_VISION, 20, upscale_model=True, clip_load=True)
    without = progress.build_plan(captions.VLM_METHOD_VISION, 20, upscale_model=False, clip_load=True)

    assert [name for name, _units in with_model][:2] == [progress.UPSCALE, progress.CLIP_LOAD]
    assert [name for name, _units in without][:1] == [progress.CLIP_LOAD]
    assert progress.UPSCALE not in [name for name, _units in without]


def test_every_picture_of_a_batch_gets_its_own_per_picture_block():
    # B>1 refines one picture at a time, so the tile-scaled phases repeat; the two node-wide
    # phases do not.
    plan = progress.build_plan(captions.VLM_METHOD_VISION, 20, batch=2, clip_load=True)
    names = [name for name, _units in plan]

    assert names.count(progress.CLIP_LOAD) == 1
    assert names.count(progress.SAMPLING) == 2
    assert names.count(progress.CANVAS_ENCODE) == 2


# ---- segments, re-fit and the clamp rule ------------------------------------


def test_the_ledger_builds_exactly_one_real_bar_over_the_whole_plan(comfy_stubs):
    progress.Ledger((("a", 2.0), ("b", 3.0)))

    assert len(comfy_stubs["progress_bars"]) == 1
    assert comfy_stubs["progress_bars"][0].total == scaled(5.0)


def test_a_segment_fills_inside_its_own_boundary_and_closing_snaps_to_it(comfy_stubs):
    ledger = progress.Ledger((("a", 2.0), ("b", 3.0)))

    ledger.open("a")
    ledger.advance(1.0)
    assert ledger.value == scaled(1.0)
    # Past the segment's own end is clamped: a phase can never spend a later phase's budget.
    ledger.advance(99.0)
    assert ledger.value == scaled(2.0)
    ledger.open("b")
    assert (ledger.value, ledger.total) == (scaled(2.0), scaled(5.0))


def test_updates_cannot_move_the_value_with_no_segment_open(comfy_stubs):
    # The routing rule is DEFINED, not incidental: with nothing open there is no segment to
    # map an update into, so the value holds at the last closed boundary.
    ledger = progress.Ledger((("a", 2.0), ("b", 3.0)))
    ledger.open("a")
    ledger.advance(2.0)
    ledger.close()

    ledger.advance(99.0)
    ledger.route(1, 2)

    assert ledger.value == scaled(2.0)


def test_resize_refits_the_open_segment_but_never_below_what_is_filled(comfy_stubs):
    # The upscale segment's true step count arrives mid-segment, and again on every OOM
    # retry. Shrinking it below the fill would put the total under the emitted value.
    ledger = progress.Ledger((("a", 2.0),))
    ledger.open("a")
    ledger.advance(1.5)

    ledger.resize(8.0)
    assert ledger.total == scaled(8.0)
    ledger.resize(0.5)
    assert ledger.total == scaled(1.5) and ledger.value == scaled(1.5)


def test_open_walks_past_a_plan_entry_the_run_skipped(comfy_stubs):
    # An empty mask returns before the canvas encode, so entries can go unopened; the cursor
    # walks to the named one and drops what it passed rather than stalling.
    ledger = progress.Ledger((("a", 1.0), ("b", 2.0), ("c", 4.0)))

    ledger.open("c")

    assert ledger.total == scaled(4.0)
    assert [name for name, _units in ledger.segments] == ["c"]


def test_an_unplanned_segment_is_inserted_rather_than_dropped(comfy_stubs):
    ledger = progress.Ledger((("a", 1.0),))

    ledger.open("surprise", 3.0)

    assert ledger.total == scaled(4.0)
    assert [name for name, _units in ledger.segments] == ["surprise"]


def test_the_value_never_decreases_and_the_total_never_drops_below_it(comfy_stubs):
    # The ONLY two invariants the re-fit machinery has to keep, driven through the upscale
    # node's own order: a ledger built before the grid is known, then re-fit repeatedly.
    ledger = progress.Ledger((("a", 1.0), ("b", 1.0), ("c", 1.0)))
    ledger.open("a")
    ledger.advance(1.0)
    ledger.open("b", 40.0)
    ledger.advance(30.0)
    ledger.resize(31.0)
    ledger.open("c", 0.5)
    ledger.finish()

    bar = comfy_stubs["progress_bars"][0]
    values = [value for value, _total, _preview in bar.updates]
    assert values == sorted(values)
    assert all(total >= value for value, total, _preview in bar.updates)


def test_finish_lands_the_value_exactly_on_the_total(comfy_stubs):
    # denoise <= 0 returns before the picture loop, so most of the plan is never opened —
    # without this the bar would freeze mid-run on a legitimate setting.
    ledger = progress.Ledger((("a", 1.0), ("b", 2.0), ("c", 4.0)))
    ledger.open("a")

    ledger.finish()

    assert ledger.value == ledger.total == scaled(7.0)


# ---- the caption mapping ----------------------------------------------------


def test_caption_fill_maps_mid_fill_then_holds_at_the_boundary():
    # Greedy decode stops at roughly 50-75% of max_length, so the token bar is mapped against
    # CAPTION_FILL_RATIO of it, not against the whole cap.
    assert progress.caption_fill(65, 200) == pytest.approx(0.5)
    assert progress.caption_fill(130, 200) == pytest.approx(1.0)
    assert progress.caption_fill(190, 200) == 1.0        # HOLDS, never past the chunk
    assert progress.caption_fill(0, 200) == 0.0
    assert progress.caption_fill(5, 0) == 1.0            # a degenerate cap is "done"


def test_a_caption_chunk_snaps_to_its_boundary_on_completion(comfy_stubs):
    ledger = progress.Ledger(((progress.CAPTIONS, 24.0),))
    ledger.open(progress.CAPTIONS, 24.0, chunks=2)

    ledger.caption_done(1, 2)

    assert ledger.value == scaled(12.0)
    assert ledger.chunks == (1, 2)


def test_caption_chunks_are_re_based_onto_this_pictures_segment(comfy_stubs):
    # generate_tile_captions counts RUN-WIDE (picture 2 of 2 reports 3/4, 4/4), while the
    # segment is this picture's alone — so the first completion in a segment is its chunk 1.
    ledger = progress.Ledger(((progress.CAPTIONS, 24.0),))
    ledger.open(progress.CAPTIONS, 24.0, chunks=2)

    ledger.caption_done(3, 4)
    assert ledger.value == scaled(12.0)
    ledger.caption_done(4, 4)
    assert ledger.value == scaled(24.0)
    assert ledger.chunks == (4, 4)


# ---- the shim ---------------------------------------------------------------


def test_the_shim_is_scoped_to_the_run_and_restored_after_a_raise(comfy_stubs):
    import comfy.utils

    real = comfy.utils.ProgressBar
    ledger = progress.Ledger((("a", 1.0),))

    with pytest.raises(RuntimeError, match="mid-run"), ledger:
        assert comfy.utils.ProgressBar is not real
        raise RuntimeError("mid-run")

    assert comfy.utils.ProgressBar is real
    # The ledger's OWN bar was built from the class captured BEFORE the patch, so it is the
    # genuine one — and it is the only one this package constructed.
    assert len(comfy_stubs["progress_bars"]) == 1


@pytest.mark.parametrize("segment", [progress.CANVAS_ENCODE, progress.SAMPLING])
def test_an_inner_bar_maps_into_the_current_segment_and_cannot_exceed_it(comfy_stubs, segment):
    # The VAE tiled fallback (sd.py) constructs a bar mid-run — during the canvas encode it
    # covers, and equally during sampling if a decode is forced there. Either way it fills
    # whatever segment is OPEN, without restarting the display or spending the next one's
    # budget.
    ledger = progress.Ledger(((segment, 4.0), ("next", 10.0)))
    with ledger:
        ledger.open(segment, 4.0)
        bar = inner_bar(8)
        bar.update_absolute(4)
        assert ledger.value == scaled(2.0)
        bar.update_absolute(99)

    assert ledger.value == scaled(4.0)
    # Still exactly one real bar: the inner one is the shim's.
    assert len(comfy_stubs["progress_bars"]) == 1


def test_an_inner_token_bar_sized_to_max_length_drives_the_caption_mapping(comfy_stubs):
    # Core's llama.py token bar is sized to max_length and abandoned at the stop token. Under
    # the shim it becomes the open caption chunk's sub-progress.
    ledger = progress.Ledger(((progress.CAPTIONS, 24.0),))
    with ledger:
        ledger.open(progress.CAPTIONS, 24.0, chunks=2)
        bar = inner_bar(200)

        bar.update_absolute(65)
        assert ledger.value == scaled(6.0)          # half of chunk 0
        bar.update_absolute(190)
        assert ledger.value == scaled(12.0)         # holds at the chunk's end


def test_a_retry_bar_resumes_the_caption_chunk_instead_of_restarting_it(comfy_stubs):
    # generate_caption can call clip.generate up to THREE times per caption (the empty-answer
    # fallback chain), each with its own token bar. A restart would snap the chunk back.
    ledger = progress.Ledger(((progress.CAPTIONS, 24.0),))
    with ledger:
        ledger.open(progress.CAPTIONS, 24.0, chunks=2)
        first = inner_bar(200)
        first.update_absolute(65)
        before = ledger.value

        retry = inner_bar(200)
        retry.update_absolute(0)
        retry.update_absolute(13)
        assert ledger.value == before

        ledger.caption_done(1, 2)

    assert ledger.value == scaled(12.0)             # the chunk's exact boundary


def test_the_oom_retrys_bigger_inner_bar_never_moves_the_value_back(comfy_stubs):
    # upscale.py halves the tile on an OOM and builds a FRESH bar with a much larger total
    # while the segment is already part-filled.
    ledger = progress.Ledger(((progress.UPSCALE, 10.0),))
    with ledger:
        ledger.open(progress.UPSCALE, 10.0)
        first = inner_bar(4)
        for _step in range(4):
            first.update(1)
        assert ledger.value == scaled(10.0)

        second = inner_bar(16)
        values = []
        for _step in range(16):
            second.update(1)
            values.append(ledger.value)

    assert values == sorted(values)
    assert values[0] == scaled(10.0)


# ---- the sampling budget ----------------------------------------------------


def test_plan_evals_is_the_public_wrapper_over_the_private_core(comfy_stubs):
    # ONE eval-math source: run_lanes enforces this exact total at every lane's eval, so a
    # second implementation of the final-step exception could disagree with the engine.
    sampler = named_sampler("exp_heun_2_x0")

    assert stepper.plan_evals(sampler, SIGMAS) == stepper._plan_evals("exp_heun_2_x0", SIGMAS)


def test_the_eval_total_carries_the_final_step_exception(comfy_stubs):
    # exp_heun_2_x0 runs two evals per step and ONE on the step whose next sigma is 0.
    total, hook_step_at = stepper.plan_evals(named_sampler("exp_heun_2_x0"), SIGMAS)

    assert total == 2 + 2 + 2 + 1
    assert hook_step_at == {0: 0, 2: 1, 4: 2, 6: 3}


# ---- the status line --------------------------------------------------------


@pytest.fixture()
def prompt_server(monkeypatch):
    """A recording stand-in for ComfyUI's PromptServer, installed as the `server` module.

    Installed rather than merely faked: the real server.py pulls in aiohttp and a whole web
    app, and is importable in this session once any comfy_env test has put the ComfyUI root
    on sys.path — so the fixture is what keeps the real module out of the process as much as
    it is what makes the emit reachable. Returns the list of (text, node_id) sent.
    """
    sent = []

    class RecordingPromptServer:
        def send_progress_text(self, text, node_id, sid=None):
            sent.append((text, node_id))

    module = types.ModuleType("server")
    module.PromptServer = RecordingPromptServer
    RecordingPromptServer.instance = RecordingPromptServer()
    monkeypatch.setitem(sys.modules, "server", module)
    return sent


def texts(sent):
    return [text for text, _node_id in sent]


def drive_combined_picture(ledger, n_tiles, caption_counters, targets):
    """The exact ledger calls the engine makes for ONE picture in "vision tokens and captions".

    Mirrors sync.build_tile_positives (captions FIRST, then the conditioning build),
    encode_canvas_latent's per-tile advance, the stepper's per-step hook and
    decode_composite — the same order tests/test_sync.py pins against the real engine.
    `caption_counters` are generate_tile_captions' RUN-WIDE (i, n) pairs.
    """
    ledger.open(progress.CAPTIONS, n_tiles * progress.K_CAPTION, chunks=n_tiles)
    for index, count in caption_counters:
        ledger.caption_done(index, count)
    ledger.open(progress.VISION_ENCODE, progress.W_ENCODE + n_tiles * progress.W_ENCODE_CAPTION_TEXT)
    ledger.open(progress.CANVAS_ENCODE, n_tiles * progress.W_ENCODE_TILE)
    for index in range(n_tiles):
        ledger.advance((index + 1) * progress.W_ENCODE_TILE)
    ledger.close()
    ledger.open(progress.SAMPLING, targets[-1])
    for target in targets:
        ledger.advance(target)
    ledger.open(progress.DECODE, n_tiles * progress.W_DECODE_TILE)
    for index in range(n_tiles):
        ledger.advance((index + 1) * progress.W_DECODE_TILE)


def test_a_vl_run_names_every_phase_it_passes_through(comfy_stubs, prompt_server):
    # 2 tiles, 4 steps, one eval per step: the sampling percent is the segment's own fill, so
    # the four steps read 25/50/75/100 and the last one lands exactly on the boundary. The
    # phases arrive in the code's order — captions BEFORE the conditioning build — and the
    # run ends by CLEARING the line.
    ledger = progress.build_ledger(captions.VLM_METHOD_VISION_CAPTIONS, 4, unique_id=NODE_ID)

    drive_combined_picture(ledger, 2, ((1, 2), (2, 2)), (2, 4, 6, 8))
    ledger.finish()

    assert texts(prompt_server) == [
        "captioning 1/2",
        "captioning 2/2",
        "encoding image",
        "encoding tiles",
        "sampling 0%",
        "sampling 25%",
        "sampling 50%",
        "sampling 75%",
        "sampling 100%",
        "decoding",
        "",
    ]
    # "sampling 0%" appears AT the segment's open: the diffusion model loads onto the GPU
    # between the open and the first eval, and the previous phase's line must not stand
    # across that wait. Every line is addressed to the node that owns the run, never
    # broadcast.
    assert {node_id for _text, node_id in prompt_server} == {NODE_ID}


def test_the_upscale_nodes_run_leads_with_its_two_extra_phases(comfy_stubs, prompt_server):
    # The upscale model pass and the cold text-encoder load are minutes each with nothing
    # else covering them, which is the whole reason they are named segments.
    ledger = progress.build_ledger(captions.VLM_METHOD_VISION_CAPTIONS, 4, upscale_model=True,
                                   clip_load=True, unique_id=NODE_ID)

    ledger.open(progress.UPSCALE)
    ledger.resize(4 * progress.W_UPSCALE_STEP)
    for step in range(4):
        ledger.advance((step + 1) * progress.W_UPSCALE_STEP)
    ledger.open(progress.CLIP_LOAD)
    drive_combined_picture(ledger, 2, ((1, 2), (2, 2)), (2, 4, 6, 8))
    ledger.finish()

    assert texts(prompt_server) == [
        "upscaling",
        "loading text encoder",
        "captioning 1/2",
        "captioning 2/2",
        "encoding image",
        "encoding tiles",
        "sampling 0%",
        "sampling 25%",
        "sampling 50%",
        "sampling 75%",
        "sampling 100%",
        "decoding",
        "",
    ]


def test_a_batch_says_which_picture_every_line_belongs_to(comfy_stubs, prompt_server):
    # B>1 refines one picture at a time under ONE ledger, so without the prefix the phases
    # would simply repeat with no way to tell how far through the batch the run is.
    ledger = progress.build_ledger(captions.VLM_METHOD_VISION_CAPTIONS, 4, batch=2, unique_id=NODE_ID)

    drive_combined_picture(ledger, 1, ((1, 2),), (1, 2, 3, 4))
    drive_combined_picture(ledger, 1, ((2, 2),), (1, 2, 3, 4))
    ledger.finish()

    lines = texts(prompt_server)
    # Picture 1's pre-completion line only knows its LOCAL count (1 tile); the completion
    # refines the denominator to the run-wide one. Picture 2's pre-line anticipates the
    # next run-wide index, so the count never appears to restart at the boundary.
    assert lines[0] == "image 1/2: captioning 1/1"
    assert lines[lines.index("image 1/2: decoding") + 1] == "image 2/2: captioning 2/2"
    assert lines[-1] == ""
    assert [line for line in lines if "captioning" in line] == [
        "image 1/2: captioning 1/1", "image 1/2: captioning 1/2",
        "image 2/2: captioning 2/2"]


def test_the_line_is_written_once_per_caption_not_once_per_token(comfy_stubs, prompt_server):
    # Core's token bar fires ~200 times per caption through the shim. Those move the BAR, and
    # must not move the line: the text cadence is per caption, per sigma step and per phase.
    ledger = progress.build_ledger(captions.VLM_METHOD_CAPTIONS, 4, unique_id=NODE_ID)
    with ledger:
        ledger.open(progress.CAPTIONS, 2 * progress.K_CAPTION, chunks=2)
        for counters in ((1, 2), (2, 2)):
            bar = inner_bar(200)
            for token in range(1, 200):
                bar.update_absolute(token)
            ledger.caption_done(*counters)

    assert texts(prompt_server) == ["captioning 1/2", "captioning 2/2", ""]


def test_an_exception_mid_run_still_clears_the_line(comfy_stubs, prompt_server):
    # An OOM mid-sampling is the reachable case; without this the node would keep "sampling
    # 50%" under it for the rest of the session.
    ledger = progress.build_ledger(captions.VLM_METHOD_VISION, 4, unique_id=NODE_ID)

    with pytest.raises(RuntimeError, match="out of memory"), ledger:
        ledger.open(progress.SAMPLING, 8.0)
        ledger.advance(4.0)
        raise RuntimeError("out of memory")

    assert texts(prompt_server) == ["sampling 0%", "sampling 50%", ""]


def test_a_run_with_no_node_id_emits_nothing(comfy_stubs, prompt_server):
    # Every direct caller builds its ledger without a node id (the tests-AB harnesses, any
    # script driving refine_image), and the base node owns no ledger at all.
    ledger = progress.build_ledger(captions.VLM_METHOD_VISION_CAPTIONS, 4)

    drive_combined_picture(ledger, 2, ((1, 2), (2, 2)), (2, 4, 6, 8))
    ledger.finish()

    assert prompt_server == []


def test_no_server_module_is_silence_rather_than_an_error(monkeypatch):
    # None in sys.modules is exactly what makes `import server` raise ImportError, which is
    # the state of any run outside a live ComfyUI — this suite and every tests-AB harness.
    monkeypatch.setitem(sys.modules, "server", None)

    assert progress.send_status(NODE_ID, "sampling 50%") is None


def test_a_server_with_no_instance_is_silence_rather_than_an_error(monkeypatch):
    # Core sets PromptServer.instance inside __init__, so on a headless run the attribute is
    # ABSENT, not None — a plain PromptServer.instance would raise AttributeError there.
    module = types.ModuleType("server")

    class PromptServer:
        def send_progress_text(self, text, node_id, sid=None):
            raise AssertionError("no server instance exists, so nothing may be sent")

    module.PromptServer = PromptServer
    monkeypatch.setitem(sys.modules, "server", module)

    assert progress.send_status(NODE_ID, "sampling 50%") is None
    PromptServer.instance = None
    assert progress.send_status(NODE_ID, "sampling 50%") is None


# ---- preset: true sizes at the grid solve ----------------------------------


def test_preset_sizes_a_pending_entry_and_open_never_moves_the_total_again(comfy_stubs):
    # The fraction-collapse fix: the big segment's true size lands BEFORE anything fills,
    # and the later open() at the same size is a no-op on the total.
    ledger = progress.Ledger((("a", 1.0), ("b", 4.0)))
    ledger.preset("b", 40.0)
    settled = ledger.total
    assert settled == scaled(41.0)

    ledger.open("a")
    ledger.advance(1.0)
    ledger.open("b", 40.0)
    ledger.advance(40.0)
    ledger.close()
    assert ledger.total == settled and ledger.value == settled


def test_preset_skips_the_open_segment(comfy_stubs):
    # resize() owns the OPEN segment; preset only ever touches pending entries, so a
    # same-named later entry (B>1 plans repeat names per picture) is the one that moves.
    ledger = progress.Ledger((("a", 2.0), ("a", 3.0)))
    ledger.open("a")
    ledger.preset("a", 30.0)

    assert ledger.total == scaled(2.0 + 30.0)


def test_preset_seeds_every_pending_match_not_just_the_first(comfy_stubs):
    # The review's B>1 finding: seeding only picture 1's block left every later picture at
    # provisional sizes, re-creating the fraction collapse at each picture boundary. One
    # preset now seeds all pending same-named entries; a later picture's own preset then
    # moves the total only by its grid delta.
    ledger = progress.Ledger((("a", 1.0), ("b", 4.0), ("a", 1.0), ("b", 4.0)))
    ledger.preset("b", 40.0)

    assert ledger.total == scaled(1.0 + 40.0 + 1.0 + 40.0)


def test_preset_picture_settles_the_total_before_any_fill(comfy_stubs):
    # The grid-solve call: every per-tile budget at its true multiplier, in one burst,
    # while the value is still zero — from here to the end the total is a constant.
    ledger = progress.Ledger(progress.build_plan("vision tokens and captions", 4))
    ledger.preset_picture("vision tokens and captions", 30, 1, 4)
    settled = ledger.total

    assert ledger.value == 0
    assert settled == scaled(
        30 * progress.K_CAPTION
        + progress.W_ENCODE + 30 * progress.W_ENCODE_CAPTION_TEXT
        + 30 * progress.W_ENCODE_TILE
        + 4 * 30
        + 30 * progress.W_DECODE_TILE
    )

    ledger.open(progress.CAPTIONS, 30 * progress.K_CAPTION, chunks=30)
    ledger.open(progress.VISION_ENCODE, progress.W_ENCODE + 30 * progress.W_ENCODE_CAPTION_TEXT)
    ledger.open(progress.CANVAS_ENCODE, 30 * progress.W_ENCODE_TILE)
    ledger.open(progress.SAMPLING, 4 * 30)
    ledger.open(progress.DECODE, 30 * progress.W_DECODE_TILE)
    ledger.finish()
    assert ledger.total == settled and ledger.value == settled


def test_preset_picture_counts_the_style_caption_like_the_engines_open(comfy_stubs):
    # style_rows is the whole-image style caption count (one per row when the run's preset sets
    # a style prompt). The preset and the engine's open must carry the identical caption
    # count, or open would move the total the preset exists to settle.
    ledger = progress.Ledger(progress.build_plan("captions", 4))
    ledger.preset_picture("captions", 30, 1, 4, style_rows=1)
    settled = ledger.total

    ledger.open(progress.CAPTIONS, 31 * progress.K_CAPTION, chunks=31)
    ledger.open(progress.CAPTION_ENCODE, 30 * progress.W_ENCODE_CAPTION_TEXT)
    ledger.open(progress.CANVAS_ENCODE, 30 * progress.W_ENCODE_TILE)
    ledger.open(progress.SAMPLING, 4 * 30)
    ledger.open(progress.DECODE, 30 * progress.W_DECODE_TILE)
    ledger.finish()
    assert ledger.total == settled and ledger.value == settled
