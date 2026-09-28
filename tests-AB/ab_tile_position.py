"""Every position method for a tile's kept items, timed, on an ab_tile_tags.py record.

It imports the Logit Tagger's first tagger (its logit_tagger.tagging at e41818d), which that pack has
since replaced, so it no longer runs.

Language methods read the whole tile once (1 MP):
  L5F   one choice "Where is {item} in the frame": top, bottom, left, right, center (relettered 4x)
  L5I   the same with "image" in place of "frame"
  L5R   L5F with one lettering, to show the letter-position bias
  N5F   five nouls anchored on the item: "{item} at the top of the frame", ...
  AXF   two frame choices, top/center/bottom and left/center/right (relettered 4x)
Split methods ask the verify statement on pieces of the full-resolution crop:
  G9    3x3 cells, each cut and resized to 0.25 MP
  G6    three horizontal and three vertical strips, 0.25 MP each
  G4    2x2 quadrants, 0.25 MP each
  G6M   G6 by masking: the whole tile at 1 MP with everything outside the strip set to grey

ab_tile_position_report.py scores the result against the grounding judge.

    python tests-AB/ab_tile_position.py tests-AB/cache/tile_tags/market-1mp [--tiles 0,1]
"""

import argparse
import json
import sys
from pathlib import Path

import ab_env

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_tile_tags as tags_run

FIVE = ("top", "bottom", "left", "right", "center")
ROWS = ("top", "center", "bottom")
COLS = ("left", "center", "right")
SPLIT_MEGAPIXELS = 0.25
ANCHORED = {"top": "{item} at the top of the frame", "bottom": "{item} at the bottom of the frame",
            "left": "{item} to the left of the frame", "right": "{item} to the right of the frame",
            "center": "{item} in the center of the frame"}


def choice_arm(classifier, picture, items, instruction, options):
    from logit_classifier import ChoiceQuestion

    questions = {f"i{n}": ChoiceQuestion(instructions=instruction.format(item=item), criteria=dict.fromkeys(options))
                 for n, item in enumerate(items)}
    answers, ms = tags_run.ask(classifier, picture, questions)
    return {item: answers[f"i{n}"].probabilities for n, item in enumerate(items)}, ms


def anchored_arm(classifier, picture, items):
    from logit_classifier import NoulQuestion

    questions = {f"i{n}{key}": NoulQuestion(instructions=text.format(item=item))
                 for n, item in enumerate(items) for key, text in ANCHORED.items()}
    answers, ms = tags_run.ask(classifier, picture, questions)
    return {item: {key: answers[f"i{n}{key}"].noul for key in ANCHORED} for n, item in enumerate(items)}, ms


def pieces(kind, height, width):
    """(name, y0, y1, x0, x1) for each piece of a split, in crop pixels."""
    third_h = [height * i // 3 for i in range(4)]
    third_w = [width * i // 3 for i in range(4)]
    if kind == "G9":
        return [(f"{ROWS[r]}|{COLS[c]}", third_h[r], third_h[r + 1], third_w[c], third_w[c + 1])
                for r in range(3) for c in range(3)]
    if kind == "G6":
        return ([(f"row|{ROWS[r]}", third_h[r], third_h[r + 1], 0, width) for r in range(3)]
                + [(f"col|{COLS[c]}", 0, height, third_w[c], third_w[c + 1]) for c in range(3)])
    if kind == "G4":
        half_h, half_w = height // 2, width // 2
        return [(f"{v}|{h}", y0, y1, x0, x1) for v, y0, y1 in (("top", 0, half_h), ("bottom", half_h, height))
                for h, x0, x1 in (("left", 0, half_w), ("right", half_w, width))]
    raise ValueError(kind)


def split_arm(classifier, crop, picture, items, kind, statement):
    import torch

    masked = kind.endswith("M")
    base = kind[:-1] if masked else kind
    result = {item: {} for item in items}
    total_ms = 0.0
    source = picture if masked else crop
    height, width = source.shape[1], source.shape[2]
    for name, y0, y1, x0, x1 in pieces(base, height, width):
        if masked:
            view = torch.full_like(picture, 0.5)
            view[:, y0:y1, x0:x1, :] = picture[:, y0:y1, x0:x1, :]
        else:
            view = tags_run.resize_to(crop[:, y0:y1, x0:x1, :], SPLIT_MEGAPIXELS)
        probabilities, ms = tags_run.verify_items(classifier, view, items, statement)
        total_ms += ms
        for item, p in zip(items, probabilities, strict=True):
            result[item][name] = p
    return result, total_ms


def gate_arm_set(plain, permuted, picture, items):
    """The spread gate arms: a sixth "whole frame" option, two fill nouls, and a nine cell frame choice."""
    from logit_classifier import NoulQuestion

    arms, times = {}, {}
    six = (*FIVE, "the whole frame")
    arms["L6F"], times["L6F"] = choice_arm(permuted, picture, items, "Where is {item} in the frame", six)
    arms["L6R"], times["L6R"] = choice_arm(plain, picture, items, "Where is {item} in the frame", six)
    cells = [f"{r} {c}" if (r, c) != ("center", "center") else "center" for r in ROWS for c in COLS]
    arms["L9F"], times["L9F"] = choice_arm(permuted, picture, items, "Where is {item} in the frame", cells)
    for key, text in (("SPN", "{item} fills most of the frame"), ("SPW", "{item} is spread across the whole frame")):
        questions = {f"i{n}": NoulQuestion(instructions=text.format(item=item)) for n, item in enumerate(items)}
        answers, times[key] = tags_run.ask(plain, picture, questions)
        arms[key] = {item: answers[f"i{n}"].noul for n, item in enumerate(items)}
    return arms, times


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("record_dir", type=Path)
    parser.add_argument("--tiles", default="", help="csv of tile indices, empty for every tile")
    parser.add_argument("--gates", action="store_true", help="run only the spread gate arms and merge them in")
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
    plain = tagging.build_classifier(clip)
    permuted = Classifier(Config(use_prior_debias=False, calibration_path=None, permutations=4),
                          ComfyClipBackend(clip))
    out_path = args.record_dir / "position.json"
    out = json.loads(out_path.read_text()) if out_path.is_file() else {"tiles": {}}
    with torch.inference_mode():
        for entry in record["tiles"]:
            if wanted and entry["index"] not in wanted:
                continue
            rect = layout.tiles[entry["index"]].crop_rect
            crop = canvas[:, rect.y0:rect.y1, rect.x0:rect.x1, :]
            picture = tags_run.resize_to(crop, record["megapixels"])
            items = entry["kept"]
            if args.gates:
                gate_arms, gate_times = gate_arm_set(plain, permuted, picture, items)
                stored = out["tiles"][str(entry["index"])]
                stored["arms"].update(gate_arms)
                stored["ms"].update(gate_times)
                print(f"tile {entry['index']}: gates " + ", ".join(f"{k} {v / 1000:.2f}s" for k, v in gate_times.items()),
                      flush=True)
                out_path.write_text(json.dumps(out, indent=1))
                continue
            arms, times = {}, {}
            arms["L5F"], times["L5F"] = choice_arm(permuted, picture, items, "Where is {item} in the frame", FIVE)
            arms["L5I"], times["L5I"] = choice_arm(permuted, picture, items, "Where is {item} in the image", FIVE)
            arms["L5R"], times["L5R"] = choice_arm(plain, picture, items, "Where is {item} in the frame", FIVE)
            arms["N5F"], times["N5F"] = anchored_arm(plain, picture, items)
            vertical, t_v = choice_arm(permuted, picture, items, "Is {item} at the top, center or bottom of the frame", ROWS)
            horizontal, t_h = choice_arm(permuted, picture, items, "Is {item} at the left, center or right of the frame", COLS)
            arms["AXF"] = {item: {"v": vertical[item], "h": horizontal[item]} for item in items}
            times["AXF"] = t_v + t_h
            for kind in ("G9", "G6", "G4", "G6M"):
                arms[kind], times[kind] = split_arm(plain, crop, picture, items, kind,
                                                        tagging.VERIFY_STATEMENT)
            out["tiles"][str(entry["index"])] = {"items": len(items), "arms": arms, "ms": times}
            print(f"tile {entry['index']}: {len(items)} items, " +
                  ", ".join(f"{k} {v / 1000:.2f}s" for k, v in times.items()), flush=True)
            out_path.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
