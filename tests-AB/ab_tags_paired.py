"""Score tile arms against the judged verdicts, paired with a baseline arm tile by tile, with bootstrap intervals.

usage: ab_tags_paired.py [--base prod-fp8] [--arms a,b,...] [--thresholds 0.9,0.95,...]
A unit is one tile under one prompt. Rates are per unit. A threshold keeps an arm's items whose verify score reaches
it, as the bench report's v filter does.
"""
import argparse
import json
import random
from pathlib import Path

BENCH = Path(__file__).resolve().parent / "cache" / "tags_bench"
MAIN = {"cyber8k", "hangar8k", "dragon4k", "market4k", "face3k"}
VERDICTS = ("yes", "no", "vague", "unsure")


def load(arm):
    return json.loads((BENCH / "runs" / f"{arm}.json").read_text(encoding="utf-8"))


def units(record, judgments, threshold=None):
    out = {}
    for scene, by_prompt in record.items():
        for prompt, result in by_prompt.items():
            for index, tile in result["tiles"].items():
                entry = judgments.get(f"{scene}|{int(index):02d}", {"items": {}, "positions": {}})
                scores = {c: s for c, _o, s in tile.get("candidates", [])}
                counts = dict.fromkeys((*VERDICTS, "unjudged", "pos_right", "pos_wrong"), 0)
                for item, term in tile["items"]:
                    if threshold is not None and (scores.get(item) or 0.0) < threshold:
                        continue
                    verdict = entry["items"].get(item)
                    counts[verdict if verdict in VERDICTS else "unjudged"] += 1
                    if term and verdict == "yes":
                        position = entry["positions"].get(f"{item}|{term}")
                        if position in ("right", "wrong"):
                            counts[f"pos_{position}"] += 1
                out[(scene, prompt, index)] = counts
    return out


def interval(diffs, rounds=4000, seed=7):
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def summarize(label, arm_units, base_units):
    keys = sorted(set(arm_units) & set(base_units))
    n = len(keys)
    if n == 0:
        print(f"{label:<26}{0:>4}  no shared units with the base")
        return

    def rate(table, name):
        return sum(table[k][name] for k in keys) / n

    def wrong(table, k):
        return table[k]["no"] + table[k]["vague"]

    yes_diffs = [arm_units[k]["yes"] - base_units[k]["yes"] for k in keys]
    wrong_diffs = [wrong(arm_units, k) - wrong(base_units, k) for k in keys]
    judged = sum(arm_units[k][v] for k in keys for v in ("yes", "no", "vague"))
    precision = sum(arm_units[k]["yes"] for k in keys) / judged if judged else 0.0
    unjudged = sum(arm_units[k]["unjudged"] for k in keys)
    pos_right = sum(arm_units[k]["pos_right"] for k in keys)
    pos_total = pos_right + sum(arm_units[k]["pos_wrong"] for k in keys)
    ylo, yhi = interval(yes_diffs)
    wlo, whi = interval(wrong_diffs)
    print(f"{label:<26}{n:>4}{rate(arm_units, 'yes'):>7.2f}{rate(arm_units, 'no'):>6.2f}{rate(arm_units, 'vague'):>6.2f}"
          f"{rate(arm_units, 'unsure'):>6.2f}{precision:>7.3f}  dyes {sum(yes_diffs) / n:+.2f} [{ylo:+.2f},{yhi:+.2f}]"
          f"  dwrong {sum(wrong_diffs) / n:+.2f} [{wlo:+.2f},{whi:+.2f}]  pos {pos_right}/{pos_total}"
          f"  unjdg {unjudged}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="prod-fp8")
    parser.add_argument("--arms", default="", help="csv of arms, every fp8-* record by default")
    parser.add_argument("--thresholds", default="", help="csv of verify thresholds to apply offline")
    args = parser.parse_args()
    base = args.base
    arms = [a for a in args.arms.split(",") if a] or sorted(p.stem for p in (BENCH / "runs").glob("fp8-*.json"))
    thresholds = [float(t) for t in args.thresholds.split(",") if t]
    judgments = json.loads((BENCH / "judgments.json").read_text(encoding="utf-8"))
    base_record = load(base)

    for set_name, pick in (("main", lambda s: s in MAIN), ("holdout", lambda s: s not in MAIN), ("both", lambda s: True)):
        base_units = {k: v for k, v in units(base_record, judgments).items() if pick(k[0])}
        print(f"\n== {set_name}   per unit: yes no vague unsure, precision = yes/(yes+no+vague), d = arm - {base}")
        for arm in [base, *arms]:
            arm_units = {k: v for k, v in units(load(arm), judgments).items() if pick(k[0])}
            summarize(arm, arm_units, base_units)
            for threshold in thresholds:
                cut = {k: v for k, v in units(load(arm), judgments, threshold).items() if pick(k[0])}
                summarize(f"  {arm}@v{threshold}", cut, base_units)


if __name__ == "__main__":
    main()
