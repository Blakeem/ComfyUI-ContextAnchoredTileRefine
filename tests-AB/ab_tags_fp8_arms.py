"""The fp8 tile arms of tags-bench-log.md section 12, each patching one stage of the shipped tags pass.

It ran at 5b2ea54 and patches names that 7d46f34 moved into the library toolkit, so check out 5b2ea54 to run it.
usage: ab_tags_fp8_arms.py sizes
       ab_tags_fp8_arms.py run <arm> <set>
Each arm writes cache/tags_bench/runs/<arm>.json.
"""
import functools
import sys
from contextlib import contextmanager
from pathlib import Path

CATR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CATR / "tests-AB"))
sys.path.insert(0, str(CATR))

import ab_tags_bench as bench  # noqa: E402

bench.CLIP_NAME = "qwen3vl_4b_fp8_scaled.safetensors"


@contextmanager
def patched(pairs):
    saved = [(owner, name, getattr(owner, name)) for owner, name, _ in pairs]
    for owner, name, value in pairs:
        setattr(owner, name, value)
    try:
        yield
    finally:
        for owner, name, value in saved:
            setattr(owner, name, value)


def head_noun_subsets():
    import logit_classifier.tags as compat
    from logit_classifier.toolkit.tags import drop_subsets

    return [(compat, "drop_subsets", functools.partial(drop_subsets, rule="head-noun"))]


def caps_40_80():
    from context_anchored_tile_refine import tags

    return [(tags, "MAX_PROPOSED_TAGS", 40), (tags, "MAX_MERGED_TAGS", 80)]


def thing_check_prompt_only():
    # The prompt tags already passed the thing check in _prompt_trace, so the tile-level check is skipped.
    from context_anchored_tile_refine import tags

    original_trace = tags.trace_tile
    original_scores = tags.thing_scores
    inside = []

    def trace_tile(*args, **kwargs):
        inside.append(True)
        try:
            return original_trace(*args, **kwargs)
        finally:
            inside.pop()

    def thing_scores(classifier, items, known):
        if inside:
            return (0.0,) * len(items)
        return original_scores(classifier, items, known)

    return [(tags, "trace_tile", trace_tile), (tags, "thing_scores", thing_scores)]


def tile_half_mp():
    from context_anchored_tile_refine import tags

    return [(tags, "VL_MAX_PIXELS", 512 * 1024)]


def tile_fit_no_upscale():
    from context_anchored_tile_refine import captions, tags

    original = captions.resample_for_vl

    def resample_for_vl(tile_pixels, budget=None):
        if budget == tags.VL_MAX_PIXELS and tile_pixels.shape[1] * tile_pixels.shape[2] <= budget:
            return tile_pixels[..., :3]
        return original(tile_pixels, budget)

    return [(captions, "resample_for_vl", resample_for_vl)]


def strips_half_mp():
    from context_anchored_tile_refine import tags

    return [(tags, "STRIP_MEGAPIXELS", 0.5)]


def presence_wording():
    from logit_classifier.toolkit.tags import presence_question

    from context_anchored_tile_refine import tags

    return [(tags, "_statements", lambda items, _statement: [presence_question(item) for item in items])]


PATCHES = {
    "fp8-hn": [head_noun_subsets],
    "fp8-cap40": [caps_40_80],
    "fp8-thingp": [thing_check_prompt_only],
    "fp8-lt": [head_noun_subsets, caps_40_80, thing_check_prompt_only],
    "fp8-mp05": [tile_half_mp],
    "fp8-fit": [tile_fit_no_upscale],
    "fp8-strip05": [strips_half_mp],
    "fp8-q": [presence_wording],
}


def arm_for(builders):
    def arm(ctx, scene, prompt, canvas, layout):
        pairs = [pair for build in builders for pair in build()]
        with patched(pairs):
            return bench.arm_baseline(ctx, scene, prompt, canvas, layout)

    return arm


def sizes():
    for set_name, (registry, _keys) in bench.SETS.items():
        for scene in registry.values():
            canvas = bench.load_canvas(scene)
            layout = bench.solve_layout(canvas)
            crops = [layout.tiles[i].crop_rect for i in scene.tiles]
            dims = [(c.x1 - c.x0, c.y1 - c.y0) for c in crops]
            print(set_name, scene.key, tuple(canvas.shape[1:3]), [f"{w}x{h}={w * h / 2**20:.2f}MP" for w, h in dims])


if __name__ == "__main__":
    for name, builders in PATCHES.items():
        bench.ARMS[name] = arm_for(builders)
    bench.ARMS["prod-fp8"] = bench.arm_baseline
    if sys.argv[1] == "sizes":
        bench.ab_env.bootstrap()
        sizes()
    else:
        sys.argv = ["ab_tags_bench.py", "run", "--arm", sys.argv[2], "--set", sys.argv[3]]
        bench.main()
