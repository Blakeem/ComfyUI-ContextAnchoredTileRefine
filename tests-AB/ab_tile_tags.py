"""Tile tagging and tile location on the owner's 8K tile layout, over the real Qwen3-VL CLIP.

It imports the Logit Tagger's first tagger (its logit_tagger.tagging at e41818d), which that pack has
since replaced, so it no longer runs.

Per tile it runs the Logit Tagger's own propose stage on the tile's crop, splits the scene
prompt into fragments, and noul-verifies both against the crop. For every kept item it then
asks four location methods and one reference:

  L9    one choice over the nine cells ("top left" .. "bottom right")
  L10   L9 plus an "entire image" option
  AX    two choices, top/center/bottom and left/center/right
  N9    one noul per cell
  REF   the noul presence statement on each of the crop's 3x3 sub-crops (the reference)

A synthetic test pastes the tile, shrunk to one cell, onto grey at each of the nine cells,
so the true cell is known exactly. Everything is written to JSON, and
ab_tile_tags_report.py scores it without a GPU.

    python tests-AB/ab_tile_tags.py --scene market
    python tests-AB/ab_tile_tags.py --scene cyber8k --no-prompt --skip-location
"""

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import ab_env

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "tests-AB" / "cache"
OUT_DIR = REPO_ROOT / "tests-AB" / "cache" / "tile_tags"
TAGGER_ROOT = REPO_ROOT.parent / "ComfyUI-LogitTagger"
CLIP_NAME = "qwen3-vl-4b-heretic_int8.safetensors"

# The owner's 8K workflow widgets (Krea 2 8k upscale.json).
MAX_TILE_WIDTH, MAX_TILE_HEIGHT, CONTEXT_ANCHOR, CONTEXT_OVERLAP = 2048, 1728, 32, 256

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_ab_matrix as matrix_run  # noqa: E402  (scene prompts only, no comfy import)

SCENES = {
    "market": ("market1024_upscale_4096x2304_*.pt", matrix_run.SCENES_BY_KEY["market"].positive),
    "portrait": ("portrait1024_upscale_3072x1728_*.pt", matrix_run.SCENES_BY_KEY["portrait"].positive),
    "face": ("krea2-00676_upscale_2304x3072_*.pt", matrix_run.SCENES_BY_KEY["face"].positive),
    "cyber8k": (str(REPO_ROOT / "samples" / "cyberpunk-city-8k.webp"), "Cyberpunk cityscape at night"),
}

ROWS = ("top", "center", "bottom")
COLS = ("left", "center", "right")
ENTIRE = "entire image"

# The splitter these records were made with, kept frozen so a rerun reproduces them. The
# library's split_prompt keeps an initialism such as u.s.a. whole.
# A period or colon between two digits is part of a number ("2.5", "16:9") and not a divider.
_PROMPT_DIVIDERS = re.compile(r"[,;!?\n\r()\[\]{}|/\"<>]|(?<!\d)[.:]|[.:](?!\d)")


def cell_name(row, col):
    if row == "center" and col == "center":
        return "center"
    if col == "center":
        return row
    return f"{row} {col}"


CELLS = [cell_name(r, c) for r in ROWS for c in COLS]


def split_prompt(prompt):
    fragments = []
    seen = set()
    for part in _PROMPT_DIVIDERS.split(prompt):
        text = " ".join(part.strip(" \t'`*-_").lower().split())
        if not re.search(r"[a-z]", text) or text in seen:
            continue
        seen.add(text)
        fragments.append(text)
    return fragments


def load_canvas(scene):
    import numpy as np
    import torch
    from PIL import Image

    pattern, prompt = SCENES[scene]
    if pattern.endswith(".pt"):
        matches = sorted(CACHE_DIR.glob(pattern))
        if not matches:
            raise SystemExit(f"no cached canvas matches {pattern} in {CACHE_DIR}")
        return torch.load(matches[0], map_location="cpu").float(), prompt
    with Image.open(pattern) as handle:
        pixels = np.asarray(handle.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(pixels)[None,], prompt


def resize_to(image, megapixels):
    import torch

    height, width = image.shape[1], image.shape[2]
    budget = megapixels * 1_000_000
    if height * width <= budget:
        return image[..., :3].contiguous()
    scale = math.sqrt(budget / (height * width))
    size = (max(1, math.floor(height * scale)), max(1, math.floor(width * scale)))
    return torch.nn.functional.interpolate(image[..., :3].movedim(-1, 1), size=size,
                                           mode="area").movedim(1, -1).contiguous()


def save_preview(path, image):
    import numpy as np
    from PIL import Image

    pixels = (image[0].clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(pixels).save(path)


def ask(classifier, image, questions):
    from logit_classifier import SystemOneRequest

    if not questions:
        return {}, 0.0
    started = time.perf_counter()
    response, _ = classifier.classify(SystemOneRequest(state="", questions=questions), image=image)
    return response.answers, (time.perf_counter() - started) * 1000.0


def verify_items(classifier, image, items, statement):
    from logit_classifier import NoulQuestion

    questions = {f"i{n}": NoulQuestion(instructions=statement.format(tag=item)) for n, item in enumerate(items)}
    answers, ms = ask(classifier, image, questions)
    return [answers[f"i{n}"].noul for n in range(len(items))], ms


def locate(classifier, image, items):
    """Every location method's raw probabilities for each item, plus each method's time."""
    from logit_classifier import ChoiceQuestion, NoulQuestion

    result = {item: {} for item in items}
    times = {}
    methods = {
        "L9": {f"i{n}": ChoiceQuestion(instructions=f"Where in this image is {item}",
                                       criteria=dict.fromkeys(CELLS)) for n, item in enumerate(items)},
        "L10": {f"i{n}": ChoiceQuestion(instructions=f"Where in this image is {item}",
                                        criteria=dict.fromkeys([*CELLS, ENTIRE])) for n, item in enumerate(items)},
        "AX": {q: question for n, item in enumerate(items) for q, question in (
            (f"i{n}v", ChoiceQuestion(instructions=f"Which part of this image, from top to bottom, holds {item}",
                                      criteria=dict.fromkeys(ROWS))),
            (f"i{n}h", ChoiceQuestion(instructions=f"Which part of this image, from left to right, holds {item}",
                                      criteria=dict.fromkeys(COLS))))},
        "N9": {f"i{n}c{c}": NoulQuestion(instructions=f"This image shows {item} in its {cell} region")
               for n, item in enumerate(items) for c, cell in enumerate(CELLS)},
    }
    for method, questions in methods.items():
        answers, times[method] = ask(classifier, image, questions)
        for n, item in enumerate(items):
            if method in ("L9", "L10"):
                answer = answers[f"i{n}"]
                result[item][method] = {"p": answer.probabilities, "abstain": answer.abstain}
            elif method == "AX":
                result[item][method] = {"v": answers[f"i{n}v"].probabilities, "h": answers[f"i{n}h"].probabilities}
            else:
                result[item][method] = {cell: answers[f"i{n}c{c}"].noul for c, cell in enumerate(CELLS)}
    return result, times


def reference(classifier, crop, items, statement, megapixels):
    """Presence of each item in each of the crop's 3x3 sub-crops, cut from the full-resolution crop."""
    height, width = crop.shape[1], crop.shape[2]
    result = {item: {} for item in items}
    total_ms = 0.0
    for r, row in enumerate(ROWS):
        for c, col in enumerate(COLS):
            y0, y1 = height * r // 3, height * (r + 1) // 3
            x0, x1 = width * c // 3, width * (c + 1) // 3
            cell_image = resize_to(crop[:, y0:y1, x0:x1, :], megapixels)
            probabilities, ms = verify_items(classifier, cell_image, items, statement)
            total_ms += ms
            for item, p in zip(items, probabilities, strict=True):
                result[item][cell_name(row, col)] = p
    return result, total_ms


def synthetic(classifier, picture, items, statement, threshold):
    """The picture shrunk to one cell on grey, at each cell, so the true cell is known."""
    import torch

    height, width = picture.shape[1], picture.shape[2]
    small = torch.nn.functional.interpolate(picture.movedim(-1, 1), size=(height // 3, width // 3),
                                            mode="area").movedim(1, -1)
    trials = []
    for r, row in enumerate(ROWS):
        for c, col in enumerate(COLS):
            board = torch.full_like(picture, 0.5)
            y0, x0 = (height // 3) * r, (width // 3) * c
            board[:, y0:y0 + small.shape[1], x0:x0 + small.shape[2], :] = small
            probabilities, _ = verify_items(classifier, board, items, statement)
            seen = [item for item, p in zip(items, probabilities, strict=True) if p >= threshold]
            located, _ = locate(classifier, board, seen)
            trials.append({"cell": cell_name(row, col), "seen": seen, "locations": located})
    return trials


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", choices=sorted(SCENES), required=True)
    parser.add_argument("--megapixels", type=float, default=1.0, help="tile picture size the VLM reads")
    parser.add_argument("--no-prompt", action="store_true", help="run without the scene prompt")
    parser.add_argument("--skip-location", action="store_true", help="propose and verify only")
    parser.add_argument("--synthetic-tiles", type=int, default=2, help="tiles run through the synthetic test")
    parser.add_argument("--tiles", default="", help="csv of tile indices, empty for all")
    parser.add_argument("--ref-megapixels", type=float, default=0.5, help="size of each REF sub-crop")
    parser.add_argument("--label", default="", help="suffix for the output file")
    return parser.parse_args()


def main():
    args = parse_args()
    ab_env.bootstrap()
    sys.path.insert(0, str(TAGGER_ROOT))
    import torch
    from logit_tagger import tagging

    from context_anchored_tile_refine import grid

    canvas, prompt = load_canvas(args.scene)
    prompt = "" if args.no_prompt else prompt
    height, width = canvas.shape[1], canvas.shape[2]
    sx = grid.solve_axis(width, MAX_TILE_WIDTH, CONTEXT_ANCHOR, CONTEXT_OVERLAP, axis="width")
    sy = grid.solve_axis(height, MAX_TILE_HEIGHT, CONTEXT_ANCHOR, CONTEXT_OVERLAP, axis="height")
    layout = grid.build_layout(width, height, sx, sy, CONTEXT_ANCHOR, CONTEXT_OVERLAP)
    wanted = {int(t) for t in args.tiles.split(",") if t.strip()} or set(range(len(layout.tiles)))
    fragments = split_prompt(prompt)
    tag = f"{args.scene}-{args.megapixels:g}mp{'-noprompt' if args.no_prompt else ''}{args.label}"
    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.scene}: {width}x{height}, {len(layout.tiles)} tiles, {len(fragments)} prompt fragments")
    print("fragments:", fragments, flush=True)

    clip = ab_env_load_clip()
    classifier = tagging.build_classifier(clip)
    record = {"scene": args.scene, "prompt": prompt, "ref_megapixels": args.ref_megapixels,
              "fragments": fragments, "megapixels": args.megapixels,
              "clip": CLIP_NAME, "size": [width, height], "tiles": []}
    threshold = tagging.NOUL_THRESHOLD
    statement = tagging.VERIFY_STATEMENT

    with torch.inference_mode():
        for index, tile in enumerate(layout.tiles):
            if index not in wanted:
                continue
            rect = tile.crop_rect
            crop = canvas[:, rect.y0:rect.y1, rect.x0:rect.x1, :]
            picture = resize_to(crop, args.megapixels)
            save_preview(out_dir / f"tile{index:02d}.png", resize_to(crop, 0.35))

            started = time.perf_counter()
            proposal = tagging.propose(clip, picture, prompt, tagging.PROPOSE_INSTRUCTION)
            propose_ms = (time.perf_counter() - started) * 1000.0
            tags = tagging.parse_candidates(proposal)
            items = tags + [f for f in fragments if f not in tags]
            probabilities, verify_ms = verify_items(classifier, picture, items, statement)
            kept = [item for item, p in zip(items, probabilities, strict=True) if p >= threshold]
            entry = {
                "index": index, "row": tile.row, "col": tile.col,
                "crop": [rect.x0, rect.y0, rect.x1, rect.y1],
                "picture": [int(picture.shape[2]), int(picture.shape[1])],
                "proposal": proposal, "tags": tags,
                "verify": [{"item": item, "p": p, "source": "tag" if item in tags else "prompt"}
                           for item, p in zip(items, probabilities, strict=True)],
                "kept": kept, "propose_ms": propose_ms, "verify_ms": verify_ms,
            }
            print(f"tile {index} r{tile.row}c{tile.col}: {len(tags)} tags, {len(items)} items, "
                  f"{len(kept)} kept, propose {propose_ms:.0f} ms, verify {verify_ms:.0f} ms", flush=True)

            if not args.skip_location and kept:
                entry["locations"], entry["locate_ms"] = locate(classifier, picture, kept)
                entry["reference"], entry["reference_ms"] = reference(classifier, crop, kept, statement,
                                                                         args.ref_megapixels)
                print(f"   locate {entry['locate_ms']}, reference {entry['reference_ms']:.0f} ms", flush=True)
                if len([t for t in record["tiles"] if "synthetic" in t]) < args.synthetic_tiles:
                    entry["synthetic"] = synthetic(classifier, picture, kept, statement, threshold)
            record["tiles"].append(entry)
            (out_dir / "record.json").write_text(json.dumps(record, indent=1))
    print(f"written {out_dir / 'record.json'}")


def ab_env_load_clip():
    import ab_models

    return ab_models.load_clip(CLIP_NAME, "krea2")


if __name__ == "__main__":
    main()
