"""Score ab_tile_position.py against the grounding judge, per axis, with no GPU.

Every method is turned into a position term the way the tile text would carry it: a row word
(top, center, bottom) and a column word (left, center, right), either of which may be left out.
A named row is right when the item's boxes cover a cell in that row, and exact when it is the
row of the cell holding most box area. Columns the same.

    python tests-AB/ab_tile_position_report.py tests-AB/cache/tile_tags/market-1mp [...]
"""

import json
import statistics
import sys
from pathlib import Path

from ab_tile_tags_report import POSITION, ROWS, box_cells, match_label

COLS = ("left", "center", "right")
PRESENT = 0.5


def five_term(probabilities, margin):
    """Your five way choice: the top pick, plus the runner up within `margin`, contradictions gated."""
    order = sorted(probabilities.items(), key=lambda pair: pair[1], reverse=True)
    picks = [order[0][0]]
    second, p2 = order[1]
    opposite = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
    if p2 >= margin * order[0][1] and opposite.get(picks[0]) != second:
        picks.append(second)
    row = next((p for p in picks if p in ("top", "bottom")), None)
    col = next((p for p in picks if p in ("left", "right")), None)
    if "center" in picks:
        if row is None and col is None:
            row, col = "center", "center"
        elif row is None:
            row = "center"
        elif col is None:
            col = "center"
    return row, col


def anchored_term(nouls, threshold):
    row = col = None
    top, bottom, left, right, center = (nouls[k] for k in ("top", "bottom", "left", "right", "center"))
    if max(top, bottom) >= threshold:
        row = "top" if top >= bottom else "bottom"
    elif center >= threshold:
        row = "center"
    if max(left, right) >= threshold:
        col = "left" if left >= right else "right"
    elif center >= threshold:
        col = "center"
    return row, col


def axis_value(present, names):
    """The p-weighted centre of the present positions on one axis, or None when all three hold it."""
    if not present or len(present) == 3:
        return None
    weight = sum(present.values())
    index = sum(names.index(name) * p for name, p in present.items()) / weight
    return names[round(index)]


def split_term(arm, kind):
    if kind == "G9":
        cells = {k: p for k, p in arm.items() if p >= PRESENT}
        rows, cols = {}, {}
        for key, p in cells.items():
            r, c = key.split("|")
            rows[r] = max(rows.get(r, 0.0), p)
            cols[c] = max(cols.get(c, 0.0), p)
    elif kind in ("G6", "G6M"):
        rows = {k.split("|")[1]: p for k, p in arm.items() if k.startswith("row|") and p >= PRESENT}
        cols = {k.split("|")[1]: p for k, p in arm.items() if k.startswith("col|") and p >= PRESENT}
    else:
        top = max(arm["top|left"], arm["top|right"]) >= PRESENT
        bottom = max(arm["bottom|left"], arm["bottom|right"]) >= PRESENT
        left = max(arm["top|left"], arm["bottom|left"]) >= PRESENT
        right = max(arm["top|right"], arm["bottom|right"]) >= PRESENT
        row = "top" if top and not bottom else "bottom" if bottom and not top else None
        col = "left" if left and not right else "right" if right and not left else None
        return row, col
    return axis_value(rows, ROWS), axis_value(cols, COLS)


def nine_term(probabilities):
    cell = max(probabilities, key=probabilities.get)
    if cell == "center":
        return "center", "center"
    return tuple(cell.split(" "))


def gated(term, spread):
    return (None, None) if spread else term


def terms(arms, item):
    out = {}
    for margin in (0.3, 0.5, 0.7, 1.01):
        tag = "top1" if margin > 1 else f"m{margin}"
        out[f"L5F {tag}"] = five_term(arms["L5F"][item], margin)
        out[f"L5I {tag}"] = five_term(arms["L5I"][item], margin)
        out[f"L5R {tag}"] = five_term(arms["L5R"][item], margin)
    for threshold in (0.5, 0.8, 0.95):
        out[f"N5F t{threshold}"] = anchored_term(arms["N5F"][item], threshold)
    ax = arms["AXF"][item]
    out["AXF"] = (max(ax["v"], key=ax["v"].get), max(ax["h"], key=ax["h"].get))
    for kind in ("G9", "G6", "G4", "G6M"):
        out[kind] = split_term(arms[kind][item], kind)
    if "L6F" not in arms:
        return out
    base = five_term(arms["L5F"][item], 1.01)
    six = arms["L6F"][item]
    five_of_six = {k: v for k, v in six.items() if k != "the whole frame"}
    out["L6F top1"] = (None, None) if max(six, key=six.get) == "the whole frame" else five_term(five_of_six, 1.01)
    six_r = arms["L6R"][item]
    out["L6R top1"] = (None, None) if max(six_r, key=six_r.get) == "the whole frame" else five_term(
        {k: v for k, v in six_r.items() if k != "the whole frame"}, 1.01)
    out["L9F"] = nine_term(arms["L9F"][item])
    for threshold in (0.5, 0.8, 0.95):
        out[f"L5F+SPN{threshold}"] = gated(base, arms["SPN"][item] >= threshold)
        out[f"L5F+SPW{threshold}"] = gated(base, arms["SPW"][item] >= threshold)
    g4 = arms["G4"][item]
    out["L5F+G4all"] = gated(base, min(g4.values()) >= PRESENT)
    g6 = arms["G6"][item]
    out["L5F+G6span"] = gated(base, all(g6[f"row|{r}"] >= PRESENT for r in ROWS)
                              and all(g6[f"col|{c}"] >= PRESENT for c in COLS))
    return out


def load_rows(record_dir):
    record_dir = Path(record_dir)
    record = json.loads((record_dir / "record.json").read_text())
    position = json.loads((record_dir / "position.json").read_text())["tiles"]
    tiles = {t["index"]: t for t in record["tiles"]}
    kind_path = record_dir / "kind.json"
    kinds = json.loads(kind_path.read_text())["tiles"] if kind_path.is_file() else {}
    rows, timing = [], []
    for path in sorted(record_dir.glob("grounding*.json")):
        for g in json.loads(path.read_text())["tiles"]:
            if str(g["index"]) not in position:
                continue
            items = tiles[g["index"]]["kept"]
            found = {}
            for box in g["boxes"]:
                item = match_label(box["label"], items)
                if item is not None:
                    found.setdefault(item, []).append(box["box"])
            arms = position[str(g["index"])]["arms"]
            for item, boxes in found.items():
                covered, primary = box_cells(boxes)
                extent = kinds.get(str(g["index"]), {}).get(item)
                named = terms(arms, item)
                if extent is not None and "L6F" in arms:
                    named["L5F+KIND0.5"] = gated(named["L5F top1"], extent >= 0.5)
                rows.append((record["scene"], covered, primary, named, extent))
    for index, entry in position.items():
        timing.append((record["scene"], int(index), entry["items"], entry["ms"]))
    return rows, timing


def score(rows, label):
    narrow = [r for r in rows if len(r[1]) <= 6]
    wide = [r for r in rows if len(r[1]) >= 7]
    print(f"\n== {label}: {len(narrow)} located items (1 to 6 cells), {len(wide)} spread items (7 to 9 cells)")
    print(f"{'method':<14}{'row named':>10}{'row right':>10}{'row exact':>10}{'col named':>10}{'col right':>10}"
          f"{'col exact':>10}{'all right':>10}{'spread named':>13}")
    for method in narrow[0][3]:
        stats = {"rn": 0, "rr": 0, "re": 0, "cn": 0, "cr": 0, "ce": 0, "all": 0, "placed": 0}
        for _scene, covered, primary, named, _extent in narrow:
            row, col = named[method]
            rows_true = {POSITION[c][0] for c in covered}
            cols_true = {POSITION[c][1] for c in covered}
            p_row, p_col = POSITION[primary]
            ok = True
            if row is not None:
                stats["rn"] += 1
                stats["rr"] += ROWS.index(row) in rows_true
                stats["re"] += ROWS.index(row) == p_row
                ok &= ROWS.index(row) in rows_true
            if col is not None:
                stats["cn"] += 1
                stats["cr"] += COLS.index(col) in cols_true
                stats["ce"] += COLS.index(col) == p_col
                ok &= COLS.index(col) in cols_true
            if row is not None or col is not None:
                stats["placed"] += 1
                stats["all"] += ok
        spread = sum(1 for r in wide if r[3][method] != (None, None))
        n = len(narrow)
        print(f"{method:<14}{stats['rn'] / n:>10.2f}{stats['rr'] / max(stats['rn'], 1):>10.3f}"
              f"{stats['re'] / max(stats['rn'], 1):>10.3f}{stats['cn'] / n:>10.2f}"
              f"{stats['cr'] / max(stats['cn'], 1):>10.3f}{stats['ce'] / max(stats['cn'], 1):>10.3f}"
              f"{stats['all'] / max(stats['placed'], 1):>10.3f}{spread:>8}/{len(wide)}")


def score_kinds(rows):
    """Whether the text-only object or extent question predicts which items spread over a tile."""
    judged = [r for r in rows if r[4] is not None]
    if not judged:
        return
    print(f"\n== object or extent, {len(judged)} judged items")
    for threshold in (0.5, 0.8):
        objects = [r for r in judged if r[4] < threshold]
        extents = [r for r in judged if r[4] >= threshold]
        spread_obj = sum(1 for r in objects if len(r[1]) >= 7)
        spread_ext = sum(1 for r in extents if len(r[1]) >= 7)
        print(f"  extent at p >= {threshold}: {len(objects)} objects ({spread_obj} spread), "
              f"{len(extents)} extents ({spread_ext} spread)")
    objects = [r for r in judged if r[4] < 0.5]
    if objects:
        score(objects, "objects only (extent p < 0.5)")


def main():
    rows, timing = [], []
    for arg in sys.argv[1:]:
        r, t = load_rows(arg)
        rows += r
        timing += t
    for scene in sorted({r[0] for r in rows}):
        score([r for r in rows if r[0] == scene], scene)
    score(rows, "all scenes")
    score_kinds(rows)
    print("\n== seconds per tile, mean over tiles, and per 10 items")
    methods = timing[0][3].keys()
    for method in methods:
        per_tile = statistics.mean(t[3][method] for t in timing) / 1000
        per_item = statistics.mean(t[3][method] / max(t[2], 1) for t in timing) / 100
        print(f"  {method:<6}{per_tile:>7.2f} s per tile{per_item:>8.2f} s per 10 items")
    print(f"  mean items per tile {statistics.mean(t[2] for t in timing):.1f} over {len(timing)} tiles")


if __name__ == "__main__":
    main()
