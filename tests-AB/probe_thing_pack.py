"""Probe: does packing move the thing check's p(thing) enough to flip at THING_THRESHOLD, and what does
an unpacked check cost? Replays each recorded run's candidate order (prompt tags first, then each
tile's new tags as one pack) against one tag per request."""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tests-AB"))
sys.path.insert(0, str(REPO))
import ab_env  # noqa: E402

ab_env.bootstrap()
import ab_models  # noqa: E402
import torch  # noqa: E402

from context_anchored_tile_refine import captions, tags  # noqa: E402

RUNS = REPO / "tests-AB" / "cache" / "tags_bench" / "runs"
CLIP_NAME = "qwen3vl_4b_fp8_scaled.safetensors"


def run_packs(record):
    prompt_tags = []
    tiles = []
    for tile in record["tiles"].values():
        candidates = [c[0] for c in tile.get("candidates", [])]
        prompt_tags += [c[0] for c in tile.get("candidates", []) if c[1] == "prompt"]
        tiles.append(candidates)
    return list(dict.fromkeys(prompt_tags)), tiles


def main(argv=None):
    from logit_classifier import Classifier, Config
    from logit_classifier.backends.comfy_clip import ComfyClipBackend
    from logit_classifier.toolkit.comfyui import thing_scores

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clip", default=CLIP_NAME, help="the text encoder file in models/text_encoders")
    args = parser.parse_args(argv)

    clip = ab_models.load_clip(args.clip, "krea2")
    settings = tags.tag_settings(captions.resolve_method(captions.default_vlm_method()))
    packed = tags.build_classifier(clip)
    unpacked = Classifier(Config(use_prior_debias=False, calibration_path=None),
                          ComfyClipBackend(clip, batch_branches=False))
    deltas, flips, near = [], [], 0
    t_packed = t_unpacked = 0.0
    n_tags = 0
    for name in ("prod.json",):
        runs = json.loads((RUNS / name).read_text(encoding="utf-8"))
        for scene, prompts in runs.items():
            for prompt, record in prompts.items():
                prompt_tags, tiles = run_packs(record)
                thing_cache = {}
                torch.cuda.synchronize()
                start = time.perf_counter()
                if prompt_tags:
                    thing_scores(packed, prompt_tags, settings, thing_cache)
                for candidates in tiles:
                    thing_scores(packed, candidates, settings, thing_cache)
                torch.cuda.synchronize()
                t_packed += time.perf_counter() - start
                alone = {}
                start = time.perf_counter()
                for tag in thing_cache:
                    thing_scores(unpacked, [tag], settings, alone)
                torch.cuda.synchronize()
                t_unpacked += time.perf_counter() - start
                n_tags += len(thing_cache)
                for tag, p in thing_cache.items():
                    q = alone[tag]
                    deltas.append(abs(p - q))
                    if (p >= tags.THING_THRESHOLD) != (q >= tags.THING_THRESHOLD):
                        flips.append((scene, prompt, tag, p, q))
                    if abs(q - tags.THING_THRESHOLD) < 0.01:
                        near += 1
                print(f"{scene}/{prompt}: {len(thing_cache)} tags, max delta so far {max(deltas):.2e}", flush=True)
    deltas.sort()
    print(json.dumps({
        "tags": n_tags, "max_delta": deltas[-1], "median_delta": deltas[len(deltas) // 2],
        "p99_delta": deltas[int(len(deltas) * 0.99)], "flips": flips, "within_0.01_of_threshold": near,
        "packed_s": round(t_packed, 2), "unpacked_s": round(t_unpacked, 2),
        "unpacked_ms_per_tag": round(1000 * t_unpacked / n_tags, 1)}, indent=1))


if __name__ == "__main__":
    main()
