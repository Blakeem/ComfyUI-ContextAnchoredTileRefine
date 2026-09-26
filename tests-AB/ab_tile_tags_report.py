"""Score ab_tile_tags.py records with no GPU: the verify split and every location method.

    python tests-AB/ab_tile_tags_report.py tests-AB/cache/tile_tags/market-1mp [...]

Location is scored two ways. On real tiles against REF, the noul presence of the item in each
of the crop's 3x3 sub-crops (a cell is present at REF_THRESHOLD). On the synthetic boards
against the one cell the shrunk tile was pasted into.
"""

import json
import statistics
import sys
from pathlib import Path

ROWS = ("top", "center", "bottom")
COLS = ("left", "center", "right")
REF_THRESHOLD = 0.5
MARGINS = (0.0, 0.3, 0.5, 0.7)
NOUL_THRESHOLDS = (0.5, 0.8, 0.95)
MAX_TERMS = 2


def cell_name(row, col):
    if row == "center" and col == "center":
        return "center"
    if col == "center":
        return row
    return f"{row} {col}"


CELLS = [cell_name(r, c) for r in ROWS for c in COLS]
POSITION = {cell_name(r, c): (i, j) for i, r in enumerate(ROWS) for j, c in enumerate(COLS)}


def adjacent(a, b):
    (ra, ca), (rb, cb) = POSITION[a], POSITION[b]
    return max(abs(ra - rb), abs(ca - cb)) <= 1


def ranked(probabilities):
    return sorted(probabilities.items(), key=lambda pair: pair[1], reverse=True)


def within_margin(probabilities, margin):
    """The top option, plus the runner up when it holds at least `margin` of the top's mass."""
    order = ranked(probabilities)
    picks = [order[0][0]]
    if margin > 0 and len(order) > 1 and order[1][1] >= margin * order[0][1]:
        picks.append(order[1][0])
    return picks


def predict(method, location, margin):
    """The cells a method names for one item, or [] for 'no position'."""
    if method == "L9":
        return within_margin(location["L9"]["p"], margin)
    if method == "L10":
        picks = within_margin(location["L10"]["p"], margin)
        return [] if picks[0] == "entire image" else [p for p in picks if p != "entire image"]
    if method == "AX":
        rows = within_margin(location["AX"]["v"], margin)
        cols = within_margin(location["AX"]["h"], margin)
        joint = {cell_name(r, c): location["AX"]["v"][r] * location["AX"]["h"][c] for r in rows for c in cols}
        return [cell for cell, _p in ranked(joint)[:MAX_TERMS]]
    if method.startswith("N9@"):
        threshold = float(method[3:])
        passing = [(c, p) for c, p in ranked(location["N9"]) if p >= threshold]
        return [c for c, _p in passing[:MAX_TERMS]]
    raise ValueError(method)


def methods():
    return ["L9", "L10", "AX", *(f"N9@{t}" for t in NOUL_THRESHOLDS)]


def score_real(records):
    rows = []
    for record in records:
        for tile in record["tiles"]:
            for item, location in tile.get("locations", {}).items():
                present = {c for c, p in tile["reference"][item].items() if p >= REF_THRESHOLD}
                rows.append((item, location, present))
    print(f"\n== real tiles, {len(rows)} kept items, scored against sub-crop presence")
    spans = [len(p) for _i, _l, p in rows]
    print(f"reference cells per item: 0 cells {spans.count(0)}, 1-2 {sum(1 for s in spans if 1 <= s <= 2)}, "
          f"3-6 {sum(1 for s in spans if 3 <= s <= 6)}, 7-9 {sum(1 for s in spans if s >= 7)}")
    scorable = [(i, l, p) for i, l, p in rows if 1 <= len(p) <= 6]
    print(f"scored on the {len(scorable)} items the reference places in 1 to 6 cells")
    print(f"{'method':<10}{'margin':>7}{'terms':>7}{'precision':>11}{'top hit':>9}{'near hit':>10}{'none':>6}")
    for method in methods():
        for margin in (MARGINS if not method.startswith("N9") else (0.0,)):
            precision, top_hits, near_hits, terms, nothing = [], 0, 0, [], 0
            for _item, location, present in scorable:
                picks = predict(method, location, margin)
                if not picks:
                    nothing += 1
                    continue
                terms.append(len(picks))
                precision.append(sum(1 for c in picks if c in present) / len(picks))
                top_hits += picks[0] in present
                near_hits += any(adjacent(picks[0], c) for c in present)
            placed = len(scorable) - nothing
            if not placed:
                continue
            print(f"{method:<10}{margin:>7.1f}{statistics.mean(terms):>7.2f}{statistics.mean(precision):>11.3f}"
                  f"{top_hits / placed:>9.3f}{near_hits / placed:>10.3f}{nothing:>6}")
    wide = [(i, l) for i, l, p in rows if len(p) >= 7]
    if wide:
        entire = sum(1 for _i, l in wide if ranked(l["L10"]["p"])[0][0] == "entire image")
        print(f"items present in 7+ cells: L10 answers 'entire image' on {entire} of {len(wide)}")


def score_synthetic(records):
    trials = [trial for record in records for tile in record["tiles"] for trial in tile.get("synthetic", [])]
    if not trials:
        return
    print(f"\n== synthetic boards, {len(trials)} boards, the true cell is the pasted one")
    print(f"{'method':<10}{'top-1 accuracy':>15}{'adjacent':>10}{'items':>7}")
    for method in methods():
        hits, near, count = 0, 0, 0
        for trial in trials:
            for _item, location in trial["locations"].items():
                picks = predict(method, location, 0.0)
                count += 1
                if picks:
                    hits += picks[0] == trial["cell"]
                    near += adjacent(picks[0], trial["cell"])
        print(f"{method:<10}{hits / max(count, 1):>15.3f}{near / max(count, 1):>10.3f}{count:>7}")


def score_verify(records):
    print("== verify split")
    for record in records:
        values = [v["p"] for tile in record["tiles"] for v in tile["verify"]]
        middle = [v for v in values if 0.23 < v < 0.8]
        print(f"{record['scene']} {record['megapixels']:g} MP: {len(values)} items, "
              f"{sum(v >= 0.5 for v in values)} kept, {len(middle)} between 0.23 and 0.80")
        for tile in record["tiles"]:
            prompt_kept = [v["item"] for v in tile["verify"] if v["source"] == "prompt" and v["p"] >= 0.5]
            tag_dropped = [v["item"] for v in tile["verify"] if v["source"] == "tag" and v["p"] < 0.5]
            ms = f"propose {tile['propose_ms'] / 1000:.1f} s, verify {tile['verify_ms'] / 1000:.2f} s"
            print(f"  tile {tile['index']} r{tile['row']}c{tile['col']}: {len(tile['tags'])} tags, "
                  f"{len(tile['kept'])} kept, {ms}")
            print(f"    prompt fragments kept: {prompt_kept}")
            print(f"    proposed tags dropped: {tag_dropped}")


def _words(text):
    return set(text.lower().replace("-", " ").split()) - {"a", "an", "the", "of", "in", "and", "with"}


def match_label(label, items):
    """The kept item a grounding label names: exact, then the best word overlap of 0.3 or more."""
    if label.lower() in items:
        return label.lower()
    scored = []
    for item in items:
        a, b = _words(label), _words(item)
        if a and b:
            scored.append((len(a & b) / len(a | b), item))
    scored.sort(reverse=True)
    return scored[0][1] if scored and scored[0][0] >= 0.3 else None


def box_cells(boxes):
    """Cells the boxes cover (a fifth of the cell, or a third of a box), and the cell holding most box area."""
    covered, area = set(), dict.fromkeys(CELLS, 0.0)
    for x0, y0, x1, y1 in boxes:
        box_area = max((x1 - x0) * (y1 - y0), 1.0)
        for i, row in enumerate(ROWS):
            for j, col in enumerate(COLS):
                cx0, cx1, cy0, cy1 = j * 1000 / 3, (j + 1) * 1000 / 3, i * 1000 / 3, (i + 1) * 1000 / 3
                overlap = max(0.0, min(x1, cx1) - max(x0, cx0)) * max(0.0, min(y1, cy1) - max(y0, cy0))
                cell = cell_name(row, col)
                area[cell] += overlap
                if overlap / (1000 * 1000 / 9) >= 0.2 or overlap / box_area >= 0.33:
                    covered.add(cell)
    return covered, max(area, key=area.get)


def score_grounding(record_dirs):
    rows = []
    for record_dir in record_dirs:
        record = json.loads((Path(record_dir) / "record.json").read_text())
        tiles = {t["index"]: t for t in record["tiles"]}
        for path in sorted(Path(record_dir).glob("grounding*.json")):
            for g in json.loads(path.read_text())["tiles"]:
                tile = tiles[g["index"]]
                items = list(tile["locations"])
                found = {}
                for box in g["boxes"]:
                    item = match_label(box["label"], items)
                    if item is not None:
                        found.setdefault(item, []).append(box["box"])
                for item, boxes in found.items():
                    covered, primary = box_cells(boxes)
                    location = dict(tile["locations"][item], **g["arms"][item])
                    rows.append((item, location, covered, primary, tile["reference"][item]))
    if not rows:
        return
    print(f"\n== against trained grounding, {len(rows)} boxed items")
    spans = [len(c) for _i, _l, c, _p, _r in rows]
    print(f"covered cells per item: 1-2 {sum(1 for s in spans if s <= 2)}, 3-6 "
          f"{sum(1 for s in spans if 3 <= s <= 6)}, 7-9 {sum(1 for s in spans if s >= 7)}")
    scorable = [r for r in rows if len(r[2]) <= 6]
    print(f"{'method':<10}{'top = primary':>14}{'top covered':>13}{'set jaccard':>13}{'items':>7}")
    arms = ["L9", "L9P", "L9D", "L10", "AX", "N9@0.5", "N9@0.8", "REF@0.5", "REF@0.8", "REF@0.95"]
    for arm in arms:
        strict, lenient, jaccard, count = 0, 0, [], 0
        for _item, location, covered, primary, reference in scorable:
            if arm.startswith("REF@"):
                threshold = float(arm[4:])
                present = {c for c, p in reference.items() if p >= threshold}
                picks = [c for c, _p in ranked(reference)] if present else []
            elif arm in ("L9P", "L9D"):
                picks = [c for c, _p in ranked(location[arm]["p"])]
                present = {picks[0]}
            else:
                picks = predict(arm, location, 0.0)
                present = set(picks[:1])
            if not picks:
                continue
            count += 1
            strict += picks[0] == primary
            lenient += picks[0] in covered
            jaccard.append(len(present & covered) / len(present | covered))
        print(f"{arm:<10}{strict / max(count, 1):>14.3f}{lenient / max(count, 1):>13.3f}"
              f"{statistics.mean(jaccard) if jaccard else 0:>13.3f}{count:>7}")
    wide = [r for r in rows if len(r[2]) >= 7]
    ref_wide = sum(1 for r in wide if sum(p >= 0.5 for p in r[4].values()) >= 7)
    print(f"items the boxes spread over 7+ cells: REF@0.5 also finds 7+ cells on {ref_wide} of {len(wide)}")
    score_render_rules(rows)


def centroid_cell(reference, present):
    weight = sum(reference[c] for c in present)
    row = sum(POSITION[c][0] * reference[c] for c in present) / weight
    col = sum(POSITION[c][1] * reference[c] for c in present) / weight
    return cell_name(ROWS[round(row)], COLS[round(col)]), (row, col)


def render_terms(rule, reference, threshold, wide):
    """The 0 to 2 position terms a render rule writes from the sub-crop presence set."""
    present = {c for c, p in reference.items() if p >= threshold}
    if not present or len(present) >= wide:
        return []
    centre, (row, col) = centroid_cell(reference, present)
    if rule == "centroid":
        return [centre]
    near = sorted(present, key=lambda c: (-reference[c], abs(POSITION[c][0] - row) + abs(POSITION[c][1] - col)))
    if rule == "top2":
        return near[:2]
    if rule == "centroid+1":
        rest = [c for c in near if c != centre]
        return [centre, *rest[:1]] if len(present) >= 2 else [centre]
    raise ValueError(rule)


def score_render_rules(rows):
    print("\nrender rules on the sub-crop presence set, against the boxes")
    print(f"{'rule':<12}{'thresh':>7}{'wide':>5}{'terms':>7}{'precision':>11}{'first=primary':>15}"
          f"{'none narrow':>13}{'none wide':>11}")
    narrow = [r for r in rows if len(r[2]) <= 6]
    broad = [r for r in rows if len(r[2]) >= 7]
    for rule in ("centroid", "centroid+1", "top2"):
        for threshold in (0.5, 0.9):
            for wide in (6, 7, 10):
                terms, precision, first, none_narrow = [], [], 0, 0
                for _item, _location, covered, primary, reference in narrow:
                    picks = render_terms(rule, reference, threshold, wide)
                    if not picks:
                        none_narrow += 1
                        continue
                    terms.append(len(picks))
                    precision.append(sum(c in covered for c in picks) / len(picks))
                    first += picks[0] == primary
                none_wide = sum(1 for r in broad if not render_terms(rule, r[4], threshold, wide))
                placed = max(len(narrow) - none_narrow, 1)
                print(f"{rule:<12}{threshold:>7}{wide:>5}{statistics.mean(terms) if terms else 0:>7.2f}"
                      f"{statistics.mean(precision) if precision else 0:>11.3f}{first / placed:>15.3f}"
                      f"{none_narrow:>7}/{len(narrow):<5}{none_wide:>5}/{len(broad)}")


def main():
    record_dirs = sys.argv[1:]
    records = [json.loads((Path(arg) / "record.json").read_text()) for arg in record_dirs]
    score_verify(records)
    score_real(records)
    score_synthetic(records)
    score_grounding(record_dirs)


if __name__ == "__main__":
    main()
