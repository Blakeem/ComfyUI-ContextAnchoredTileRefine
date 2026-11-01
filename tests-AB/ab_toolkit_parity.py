"""Byte parity of both ComfyUI packs' toolkit stages on real images, recorded before and after a library change.

    python tests-AB/ab_toolkit_parity.py record --out tests-AB/cache/toolkit_parity/before.json
    python tests-AB/ab_toolkit_parity.py compare tests-AB/cache/toolkit_parity/before.json <after.json>

`record` loads the CLIP once and runs this pack's tags pass and caption set and the Logit Tagger's tag and style runs
through their run-level entry points. `--sections` picks from catr-tags, catr-captions, lt-tags and lt-style.
`compare` prints every difference outside the top-level "meta" block with its JSON path, and exits 1 on any. Floats
are compared by repr, so a one-ulp change counts. Two recordings of one checkout match byte for byte, so no tolerance
is needed.
"""

import argparse
import dataclasses
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import ab_env
import ab_tags_bench as bench

CATR_ROOT = ab_env.REPO_ROOT
LT_ROOT = CATR_ROOT.parent / "ComfyUI-LogitTagger"
LIBRARY_ROOT = ab_env.LIBRARY_HARNESS_DIR.parent
CLIP_NAME = "qwen3vl_4b_fp8_scaled.safetensors"
CLIP_TYPE = "krea2"

# The tags bench scenes, run with their long prompt.
TAG_SCENES = {key: (bench.SCENES[key].source, bench.SCENES[key].prompts["long"]) for key in ("dragon4k", "market4k")}
CAPTION_SCENE = "dragon4k"
# settings.toml ships only the tags preset, so the caption preset comes from the example file beside it.
CAPTION_SETTINGS = CATR_ROOT / "settings.user.example.toml"
CAPTION_PRESET = "artwork"

LT_RENDERS = LT_ROOT / "tests-AB" / "data" / "release-gates" / "renders"
CATR_BASE_IMAGE = CATR_ROOT / "tests-AB" / "inputs" / "krea2-00676-base-768x1024.png"
# (key, image path, manifest id whose prompt it runs with, or None for no prompt)
LT_TAG_IMAGES = (
    ("g03", LT_RENDERS / "g03.png", "g03"),
    ("g10", LT_RENDERS / "g10.png", "g10"),
    ("krea2-00676-base", CATR_BASE_IMAGE, None),
)
LT_STYLE_IMAGES = (
    ("g10-with-prompt", LT_RENDERS / "g10.png", "g10"),
    ("g03-no-prompt", LT_RENDERS / "g03.png", None),
)

CATR_TILE_FIELDS = ("reply", "echo_reply", "proposed", "candidates", "origins", "dropped", "scores", "verified",
                    "kept", "strips", "terms", "unplaced", "text")
CATR_PROMPT_FIELDS = ("text", "reply", "listed", "p_thing", "tags")
ABSENT = "<absent>"
SECTIONS = ("catr-tags", "catr-captions", "lt-tags", "lt-style")



def plain(value):
    """A JSON value for `value`: floats as their repr, dataclasses as dicts without timing fields."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float) or type(value).__name__ in ("float32", "float64", "float16"):
        return {"float": repr(float(value))}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        # Wall-clock timings change on every run and are not outputs.
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value) if not f.name.endswith("_ms")}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if type(value).__name__ == "Tensor":
        return {"tensor": tensor_digest(value)}
    raise TypeError(f"no JSON form for {type(value).__name__}: {value!r}")


def pick(obj, names):
    return {name: plain(getattr(obj, name, ABSENT)) for name in names}


def tensor_digest(tensor):
    pixels = tensor.detach().contiguous().cpu().float().numpy().tobytes()
    return {"sha256": hashlib.sha256(pixels).hexdigest(), "shape": list(tensor.shape), "dtype": str(tensor.dtype)}



def bootstrap():
    if str(LT_ROOT) not in sys.path:
        sys.path.insert(0, str(LT_ROOT))
    ab_env.bootstrap()
    import comfy_env

    return comfy_env


def git_head(path):
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True,
                               check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unknown ({error})"
    return out.stdout.strip() + (" +dirty" if dirty.stdout.strip() else "")


def load_rgb(path):
    """LoadImage's pixels: EXIF rotation applied, RGB, float32 in [0, 1], shape [1, H, W, 3]."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    with Image.open(path) as handle:
        rgb = ImageOps.exif_transpose(handle).convert("RGB")
    return torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0)[None,]


def render_prompts():
    manifest = json.loads((LT_RENDERS / "manifest.json").read_text(encoding="utf-8"))
    return {record["id"]: record["prompt"] for record in manifest}


class Runner:
    """Runs one item at a time as a fresh node execution: every in-process cache cleared first, inference mode on,
    and core's per-node cleanup after."""

    def __init__(self, comfy_env, torch):
        self.comfy_env = comfy_env
        self.torch = torch
        self.timings = {}

    def run(self, name, compute):
        from context_anchored_tile_refine import captions, tags

        captions.clear_caption_cache()
        tags.clear_tag_cache()
        gc.collect()
        started = time.perf_counter()
        with self.torch.inference_mode():
            result = compute()
        self.torch.cuda.synchronize()
        self.timings[name] = round(time.perf_counter() - started, 2)
        self.comfy_env.end_node_execution()
        print(f"  {name}: {self.timings[name]} s", flush=True)
        return result



def record_catr_tags(runner, clip):
    from context_anchored_tile_refine import captions, tags

    out = {}
    base_preset = captions.resolve_method(captions.default_vlm_method())
    for scene, (source, long_prompt) in TAG_SCENES.items():
        canvas = load_rgb(CATR_ROOT / source)
        layout = bench.solve_layout(canvas)
        tiles = list(layout.tiles)
        scene_out = {"canvas": tensor_digest(canvas), "grid_tiles": len(tiles),
                     "rects": [[t.crop_rect.x0, t.crop_rect.y0, t.crop_rect.x1, t.crop_rect.y1] for t in tiles],
                     "runs": {}}
        for prompt_key, prompt in (("none", ""), ("long", long_prompt)):
            preset = captions.with_prompt(base_preset, prompt)
            run = runner.run(f"catr-tags {scene} {prompt_key}",
                             lambda preset=preset, canvas=canvas, tiles=tiles:
                             tags.generate_tag_trace(clip, canvas, tiles, preset))
            scene_out["runs"][prompt_key] = {
                "prompt": prompt,
                "preset": plain(preset),
                "style_texts": plain(run.style_texts),
                "prompt_trace": None if run.prompt is None else pick(run.prompt, CATR_PROMPT_FIELDS),
                "tiles": [[pick(trace, CATR_TILE_FIELDS) for trace in rows] for rows in run.tiles],
                "tag_texts": plain(tags.tag_texts(run)),
            }
        out[scene] = scene_out
    return out


def record_catr_captions(runner, clip):
    from context_anchored_tile_refine import captions

    settings = captions.load_settings(CAPTION_SETTINGS)
    block = settings.presets[CAPTION_PRESET]
    preset = captions.with_prompt(
        captions.build_preset(captions.VLM_METHOD_VISION_CAPTIONS, CAPTION_PRESET, settings.vision,
                              block.get("tile_text", captions.TILE_TEXT_CAPTION), block), None)
    source, _prompt = TAG_SCENES[CAPTION_SCENE]
    canvas = load_rgb(CATR_ROOT / source)
    tiles = list(bench.solve_layout(canvas).tiles)
    style_texts, tile_captions = runner.run(
        f"catr-captions {CAPTION_SCENE}", lambda: captions.generate_caption_set(clip, canvas, tiles, preset))
    return {"scene": CAPTION_SCENE, "canvas": tensor_digest(canvas), "preset": plain(preset),
            "style_texts": plain(style_texts), "captions": plain(tile_captions),
            "joined": plain(captions.join_style_captions(style_texts, tile_captions))}


def record_lt_tags(runner, clip):
    from logit_tagger import tagging

    prompts = render_prompts()
    images = [load_rgb(path) for _key, path, _pid in LT_TAG_IMAGES]
    texts = ["" if pid is None else prompts[pid] for _key, _path, pid in LT_TAG_IMAGES]
    out = {}
    # One image per run, so each output is computed with nothing carried from another image.
    for (key, path, _pid), image, prompt in zip(LT_TAG_IMAGES, images, texts, strict=True):
        traces = runner.run(f"lt-tags {key}",
                            lambda image=image, prompt=prompt: tagging.tag_images(clip, [image], [prompt],
                                                                                  positions=True))
        out[key] = {"image": path.name, "pixels": tensor_digest(image), "prompt": prompt, "trace": plain(traces[0])}
    return out


def record_lt_style(runner, clip):
    from logit_tagger import style

    prompts = render_prompts()
    out = {}
    for key, path, pid in LT_STYLE_IMAGES:
        image = load_rgb(path)
        prompt = "" if pid is None else prompts[pid]
        traces = runner.run(f"lt-style {key}",
                            lambda image=image, prompt=prompt: style.extract_styles(clip, [image], [prompt]))
        out[key] = {"image": path.name, "pixels": tensor_digest(image), "prompt": prompt, "trace": plain(traces[0])}
    return out


RECORDERS = {"catr-tags": record_catr_tags, "catr-captions": record_catr_captions,
             "lt-tags": record_lt_tags, "lt-style": record_lt_style}


def count_items(results):
    counts = {}
    if "catr-tags" in results:
        runs = [run for scene in results["catr-tags"].values() for run in scene["runs"].values()]
        counts["catr-tags runs"] = len(runs)
        counts["catr-tags tile traces"] = sum(len(rows) for run in runs for rows in run["tiles"])
    if "catr-captions" in results:
        counts["catr-captions style"] = len(results["catr-captions"]["style_texts"])
        counts["catr-captions tiles"] = sum(len(rows) for rows in results["catr-captions"]["captions"])
    for name in ("lt-tags", "lt-style"):
        if name in results:
            counts[name] = len(results[name])
    return counts


def cmd_record(args):
    sections = [s for s in args.sections.split(",") if s]
    unknown = sorted(set(sections) - set(SECTIONS))
    if unknown:
        raise SystemExit(f"unknown sections {unknown}. Pick from {list(SECTIONS)}.")
    comfy_env = bootstrap()
    import logit_classifier
    import torch

    started = time.perf_counter()
    clip = comfy_env.load_clip(args.clip, CLIP_TYPE)
    runner = Runner(comfy_env, torch)
    torch.cuda.reset_peak_memory_stats()
    results = {}
    for name in sections:
        print(f"{name}:", flush=True)
        results[name] = RECORDERS[name](runner, clip)
    meta = {
        "clip": args.clip,
        "logit_classifier": logit_classifier.__version__,
        "git": {"logit-classifier": git_head(LIBRARY_ROOT), "catr": git_head(CATR_ROOT), "logit-tagger": git_head(LT_ROOT)},
        "seconds_total": round(time.perf_counter() - started, 1),
        "seconds_per_item": runner.timings,
        "peak_vram_gib": {"allocated": round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2),
                          "reserved": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)},
        "counts": count_items(results),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"meta": meta, **results}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=1))
    print(f"written {args.out}")



def short(value, limit=400):
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else f"{text[:limit]}... ({len(text)} chars)"


def diff(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in list(a) + [k for k in b if k not in a]:
            if key not in b:
                out.append((f"{path}.{key}", "only in before", short(a[key])))
            elif key not in a:
                out.append((f"{path}.{key}", "only in after", short(b[key])))
            else:
                diff(a[key], b[key], f"{path}.{key}", out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, "length", f"{len(a)} != {len(b)}"))
        for index, (x, y) in enumerate(zip(a, b, strict=False)):
            diff(x, y, f"{path}[{index}]", out)
        return
    if type(a) is not type(b) or a != b:
        out.append((path, "value", f"before {short(a)}\n      after  {short(b)}"))


def cmd_compare(args):
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    differences = []

    before.pop("meta", None)
    after.pop("meta", None)
    diff(before, after, "$", differences)
    for path, kind, detail in differences:
        print(f"{path} [{kind}]\n      {detail}")
    print(f"{len(differences)} differences")
    sys.exit(1 if differences else 0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("record")
    record.add_argument("--out", required=True)
    record.add_argument("--clip", default=CLIP_NAME)
    record.add_argument("--sections", default=",".join(SECTIONS))
    compare = sub.add_parser("compare")
    compare.add_argument("before")
    compare.add_argument("after")
    return parser.parse_args()


def main():
    args = parse_args()
    {"record": cmd_record, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    main()
