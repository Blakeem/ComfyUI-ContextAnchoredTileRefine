"""Judge every candidate tag of a tile record, including the ones verify dropped, with two blind judges.

The bench judges only the tags an arm kept, so a candidate below the recorded verify threshold was never judged, and a
wording that keeps it cannot be scored. This writes each unjudged candidate to two judge sets that see the tags in
different orders, mixed with up to CONTROLS stored verdicts per tile. merge stores a verdict both judges gave in
judgments.json and reports how often the judges agreed with each other and with the stored verdicts.

    python tests-AB/ab_tags_candidates.py tasks [--arm prod-fp8]   # judge-candidates/<set>/<tile>.task.json
    python tests-AB/ab_tags_candidates.py merge                      # agreed verdicts into judgments.json
"""

import argparse
import json
import random
from pathlib import Path

from ab_tags_bench import JUDGMENTS, RUNS_DIR, TILE_DIR

CANDIDATE_DIR = Path(__file__).resolve().parent / "cache" / "tags_bench" / "judge-candidates"
# Beside the judge folders and never in a task file, so a judge cannot tell a control from a candidate.
CONTROLS_FILE = CANDIDATE_DIR / "controls.json"
JUDGE_SETS = ("A", "B")
CONTROLS = 4
SEED = 7
VERDICTS = ("yes", "no", "vague", "unsure")

RUBRIC = """You judge tags for ONE image tile. Open the image file yourself with your Read tool and look at it.
The tags mix things the tile holds with things it does not, so judge only what is visible.

For every tag in "items", answer one of:
  yes     the named thing is clearly visible in this image (a part of it counts, cropped counts)
  no      the named thing is not in this image, or the name is wrong for what is there
  vague   not a visible thing: a mood, style, quality, idea, lighting word, lone color, a
          whole-scene description such as "cityscape at night", or too generic to point at
  unsure  too small or ambiguous to decide
Judge the words literally. "red neon" needs red neon light, "glowing windows" needs windows
that glow. A long phrase is "yes" only when every thing it names is visible here.
Write JSON only to the verdict file: {"items": {tag: verdict}}, with every tag you were given."""


def load_judgments():
    return json.loads(JUDGMENTS.read_text(encoding="utf-8")) if JUDGMENTS.exists() else {}


def record_candidates(arm):
    """Every (tile key, candidate) the arm's record scored, over all its prompts."""
    record = json.loads((RUNS_DIR / f"{arm}.json").read_text(encoding="utf-8"))
    candidates = {}

    for scene_key, by_prompt in record.items():
        for result in by_prompt.values():
            for index, tile in result["tiles"].items():
                key = f"{scene_key}|{int(index):02d}"
                candidates.setdefault(key, set()).update(candidate for candidate, _origin, _score in tile["candidates"])
    return candidates


def cmd_tasks(args):
    judgments = load_judgments()
    candidates = record_candidates(args.arm)
    controls_by_tile = {}
    total = 0

    for judge_set in JUDGE_SETS:
        folder = CANDIDATE_DIR / judge_set
        folder.mkdir(parents=True, exist_ok=True)
        # A verdict from an earlier round would otherwise answer this round's tasks.
        for stale in [*folder.glob("*.task.json"), *folder.glob("*.verdict.json")]:
            stale.unlink()
    for key, tags in sorted(candidates.items()):
        stored = judgments.get(key, {"items": {}})["items"]
        unjudged = sorted(tag for tag in tags if tag not in stored)
        if not unjudged:
            continue
        judged = sorted(tag for tag in tags if stored.get(tag) in VERDICTS)
        controls = sorted(random.Random(f"{SEED}-{key}").sample(judged, min(CONTROLS, len(judged))))
        controls_by_tile[key] = controls
        scene_key, index = key.split("|")
        name = f"{scene_key}-t{index}"
        for judge_set in JUDGE_SETS:
            items = [*unjudged, *controls]
            random.Random(f"{SEED}-{judge_set}-{key}").shuffle(items)
            folder = CANDIDATE_DIR / judge_set
            body = {"tile": key, "image": str(TILE_DIR / f"{name}.jpg"), "items": items,
                    "verdict_file": str(folder / f"{name}.verdict.json")}
            (folder / f"{name}.task.json").write_text(json.dumps(body, indent=1), encoding="utf-8")
        total += len(unjudged)
        print(f"{name}: {len(unjudged)} unjudged, {len(controls)} controls")
    CONTROLS_FILE.write_text(json.dumps(controls_by_tile, indent=1), encoding="utf-8")
    (CANDIDATE_DIR / "RUBRIC.txt").write_text(RUBRIC, encoding="utf-8")
    print(f"{total} unjudged candidates, task files in {CANDIDATE_DIR}")


def read_verdicts(judge_set):
    verdicts = {}

    for task_path in sorted((CANDIDATE_DIR / judge_set).glob("*.task.json")):
        task = json.loads(task_path.read_text(encoding="utf-8"))
        verdict_path = Path(task["verdict_file"])
        if not verdict_path.exists():
            raise SystemExit(f"{verdict_path} is missing. Judge set {judge_set} has not judged {task['tile']}.")
        given = json.loads(verdict_path.read_text(encoding="utf-8"))["items"]
        missing = [tag for tag in task["items"] if given.get(tag) not in VERDICTS]
        if missing:
            raise SystemExit(f"{verdict_path} lacks a verdict for {missing[:3]}.")
        verdicts[task["tile"]] = (task, given)
    return verdicts


def cmd_merge(_args):
    judgments = load_judgments()
    first, second = (read_verdicts(judge_set) for judge_set in JUDGE_SETS)
    controls_by_tile = json.loads(CONTROLS_FILE.read_text(encoding="utf-8"))
    unpaired = sorted(set(first) ^ set(second))
    agreed = disagreed = control_agreed = control_total = 0
    disagreements = []

    if unpaired:
        raise SystemExit(f"Only one judge set has tasks for {', '.join(unpaired)}. Run tasks again for both sets.")
    for key, (task, first_given) in sorted(first.items()):
        second_given = second[key][1]
        stored = judgments.setdefault(key, {"items": {}, "positions": {}})["items"]
        for tag in task["items"]:
            if tag in controls_by_tile[key]:
                control_total += 2
                control_agreed += (first_given[tag] == stored[tag]) + (second_given[tag] == stored[tag])
                continue
            if first_given[tag] == second_given[tag]:
                stored[tag] = first_given[tag]
                agreed += 1
            else:
                disagreed += 1
                disagreements.append(f"{key} {tag!r}: {first_given[tag]} / {second_given[tag]}")
    JUDGMENTS.write_text(json.dumps(judgments, indent=1, sort_keys=True), encoding="utf-8")
    print("\n".join(disagreements))
    print(f"judges agreed on {agreed} of {agreed + disagreed} candidates, stored in {JUDGMENTS}. "
          f"Control verdicts matched the stored ones {control_agreed} of {control_total} times.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    tasks = commands.add_parser("tasks")
    tasks.add_argument("--arm", default="prod-fp8", help="the record whose candidates are judged")
    commands.add_parser("merge")
    args = parser.parse_args()
    {"tasks": cmd_tasks, "merge": cmd_merge}[args.command](args)


if __name__ == "__main__":
    main()
