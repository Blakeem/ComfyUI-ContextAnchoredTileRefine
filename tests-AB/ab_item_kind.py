"""Can a text-only question pick the items worth locating, before any split pass?

Asks one choice per kept item with no image: a countable object, or an extent (a surface,
material, texture, light, background or the whole scene). Writes kind.json beside each record.
ab_tile_position_report.py --kinds scores whether the extent items are the ones that spread
over most of a tile, which is what would let the split pass skip them.

    python tests-AB/ab_item_kind.py tests-AB/cache/tile_tags/market-1mp [...]
"""

import json
import sys
import time
from pathlib import Path

import ab_env

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_tile_tags as tags_run

CRITERIA = {
    "object": "one countable object, person, animal or building, or a part of one",
    "extent": "a surface, material, texture, pattern, light, weather, background or the whole scene",
}


def main():
    ab_env.bootstrap()
    sys.path.insert(0, str(tags_run.TAGGER_ROOT))
    import torch
    from logit_classifier import ChoiceQuestion, SystemOneRequest
    from logit_tagger import tagging

    clip = tags_run.ab_env_load_clip()
    classifier = tagging.build_classifier(clip)
    with torch.inference_mode():
        for arg in sys.argv[1:]:
            record_dir = Path(arg)
            record = json.loads((record_dir / "record.json").read_text())
            kinds, times = {}, []
            for tile in record["tiles"]:
                items = tile["kept"]
                questions = {f"i{n}": ChoiceQuestion(instructions=f'What does the phrase "{item}" name in a picture',
                                                     criteria=CRITERIA) for n, item in enumerate(items)}
                started = time.perf_counter()
                response, _ = classifier.classify(SystemOneRequest(state="", questions=questions))
                times.append(time.perf_counter() - started)
                kinds[str(tile["index"])] = {item: response.answers[f"i{n}"].probabilities["extent"]
                                             for n, item in enumerate(items)}
            (record_dir / "kind.json").write_text(json.dumps({"tiles": kinds, "seconds": times}, indent=1))
            print(f"{record_dir.name}: {len(times)} tiles, {sum(times) / len(times):.2f} s per tile")


if __name__ == "__main__":
    main()
