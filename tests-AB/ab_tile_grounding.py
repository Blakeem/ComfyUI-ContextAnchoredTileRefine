"""A trained-grounding reference for tile location: Qwen3-VL's own bbox JSON per kept item.

It imports the Logit Tagger's first tagger (its logit_tagger.tagging at e41818d), which that pack has
since replaced, so it no longer runs.

Reads an ab_tile_tags.py record, asks the CLIP to box every kept item of each tile (a
generate, far too slow for production, fine as a judge), and adds two cheap location arms:
L9P, the nine cell choice relettered four times (which cancels a letter position bias),
and L9D, the nine cells with a spatial description on each option. Everything lands in
grounding.json beside the record, and ab_tile_tags_report.py --grounding scores it.

    python tests-AB/ab_tile_grounding.py tests-AB/cache/tile_tags/market-1mp --tiles 0,1,2
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import ab_env

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_tile_tags as tags_run

GROUND_TEMPLATE = (
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Locate every instance of each "
    "of these in the image: {items}. Report each one's bbox coordinates in JSON format as a list "
    "of objects with the keys bbox_2d and label.<|im_end|>\n<|im_start|>assistant\n"
)
TOKENS_PER_ITEM = 48
DESCRIPTIONS = {
    "top left": "the upper left corner", "top": "the upper middle", "top right": "the upper right corner",
    "center left": "the middle of the left side", "center": "the middle of the image",
    "center right": "the middle of the right side", "bottom left": "the lower left corner",
    "bottom": "the lower middle", "bottom right": "the lower right corner",
}
_BOX = re.compile(r'"bbox_2d"\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\]\s*,\s*"label"\s*:\s*"([^"]*)"')


def ground(clip, picture, items):
    from logit_tagger import tagging

    rendered = GROUND_TEMPLATE.format(items="; ".join(items))
    tokens = clip.tokenize(rendered, images=[picture])
    started = time.perf_counter()
    with tagging.cuda_graphs_disabled():
        ids = clip.generate(tokens, do_sample=False, max_length=TOKENS_PER_ITEM * len(items) + 64)
    text = clip.decode(ids)
    boxes = [{"box": [float(v) for v in m.groups()[:4]], "label": m.group(5)} for m in _BOX.finditer(text)]
    return text, boxes, (time.perf_counter() - started) * 1000.0


def choice_arms(classifier_permuted, classifier, picture, items):
    from logit_classifier import ChoiceQuestion

    result = {item: {} for item in items}
    plain = {f"i{n}": ChoiceQuestion(instructions=f"Where in this image is {item}",
                                     criteria=dict.fromkeys(tags_run.CELLS)) for n, item in enumerate(items)}
    described = {f"i{n}": ChoiceQuestion(instructions=f"Where in this image is {item}",
                                         criteria=dict(DESCRIPTIONS)) for n, item in enumerate(items)}
    permuted_answers, _ = tags_run.ask(classifier_permuted, picture, plain)
    described_answers, _ = tags_run.ask(classifier, picture, described)
    for n, item in enumerate(items):
        result[item]["L9P"] = {"p": permuted_answers[f"i{n}"].probabilities}
        result[item]["L9D"] = {"p": described_answers[f"i{n}"].probabilities}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record_dir", type=Path)
    parser.add_argument("--tiles", default="", help="csv of tile indices, empty for every tile with locations")
    args = parser.parse_args()

    ab_env.bootstrap()
    sys.path.insert(0, str(tags_run.TAGGER_ROOT))
    import torch
    from logit_classifier import Classifier, Config
    from logit_classifier.backends.comfy_clip import ComfyClipBackend
    from logit_tagger import tagging

    from context_anchored_tile_refine import grid

    record = json.loads((args.record_dir / "record.json").read_text())
    canvas, _prompt = tags_run.load_canvas(record["scene"])
    width, height = record["size"]
    sx = grid.solve_axis(width, tags_run.MAX_TILE_WIDTH, tags_run.CONTEXT_ANCHOR, tags_run.CONTEXT_OVERLAP, axis="width")
    sy = grid.solve_axis(height, tags_run.MAX_TILE_HEIGHT, tags_run.CONTEXT_ANCHOR, tags_run.CONTEXT_OVERLAP, axis="height")
    layout = grid.build_layout(width, height, sx, sy, tags_run.CONTEXT_ANCHOR, tags_run.CONTEXT_OVERLAP)
    wanted = {int(t) for t in args.tiles.split(",") if t.strip()}

    clip = tags_run.ab_env_load_clip()
    classifier = tagging.build_classifier(clip)
    permuted = Classifier(Config(use_prior_debias=False, calibration_path=None, permutations=4),
                          ComfyClipBackend(clip))
    out = {"tiles": []}
    with torch.inference_mode():
        for entry in record["tiles"]:
            if "locations" not in entry or (wanted and entry["index"] not in wanted):
                continue
            rect = layout.tiles[entry["index"]].crop_rect
            picture = tags_run.resize_to(canvas[:, rect.y0:rect.y1, rect.x0:rect.x1, :], record["megapixels"])
            items = list(entry["locations"])
            text, boxes, ms = ground(clip, picture, items)
            arms = choice_arms(permuted, classifier, picture, items)
            print(f"tile {entry['index']}: {len(items)} items, {len(boxes)} boxes, ground {ms / 1000:.1f} s", flush=True)
            out["tiles"].append({"index": entry["index"], "picture": [int(picture.shape[2]), int(picture.shape[1])],
                                 "text": text, "boxes": boxes, "ground_ms": ms, "arms": arms})
            (args.record_dir / "grounding.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
