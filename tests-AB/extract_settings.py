"""Pull the A/B-relevant settings out of a ComfyUI PNG's embedded workflow.

ComfyUI stashes the API-format graph in the PNG's `prompt` tEXt chunk, so every output image
carries the exact settings that produced it. This reads that back into the flat dict the A/B
harness needs, so adding a new test image is "generate it in ComfyUI, point this at the PNG"
rather than "retype the settings and hope".

Only the knobs we actually vary are extracted; tile-resolution caps come along for the ride
because the harness needs them, but they are not an A/B axis.

Usage:
    python tests-AB/extract_settings.py <image.png> [more.png ...]      # human-readable
    python tests-AB/extract_settings.py --json <image.png> [...]        # machine-readable
    python tests-AB/extract_settings.py --dir <folder>                  # every PNG in a folder

As a library:
    from extract_settings import extract
    settings = extract(r"...\\AB-Test_00001_.png")
"""
import argparse
import json
import sys
from pathlib import Path

# The refine node we are testing. Its inputs are the settings that matter.
REFINE_CLASS = "ContextAnchoredTileRefine"

# Upstream nodes whose values we resolve by following links out of the refine node.
# (ComfyUI stores an input either as a literal or as a ["node_id", slot] reference.)
SCALAR_CLASSES = ("int", "float", "PrimitiveInt", "PrimitiveFloat", "PrimitiveString",
                  "PrimitiveStringMultiline", "MathExpression|pysssss")


def _load_graph(png_path):
    from PIL import Image

    with Image.open(png_path) as im:
        raw = im.info.get("prompt")
        size = im.size
    if not raw:
        raise ValueError("{}: no ComfyUI 'prompt' metadata (was it saved by SaveImage?)".format(png_path))
    return json.loads(raw), size


def _resolve(graph, value, _depth=0):
    """Follow a ComfyUI input reference to a concrete value, or return it unchanged."""
    if _depth > 12:                       # cycles shouldn't exist, but never hang on one
        return None
    if not (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)):
        return value                      # already a literal
    node = graph.get(value[0])
    if node is None:
        return None
    inputs = node.get("inputs", {})
    if node.get("class_type") in SCALAR_CLASSES:
        # A primitive/expression node: its own "value" (or evaluated expression) is the answer.
        if "value" in inputs:
            return _resolve(graph, inputs["value"], _depth + 1)
        if "expression" in inputs:
            return _eval_expression(graph, inputs, _depth)
    return None                           # a tensor-valued link (model/vae/image): not a setting


def _eval_expression(graph, inputs, depth):
    """Evaluate the tiny a*b / a*2 expressions the workflow uses for the tile caps."""
    expression = inputs.get("expression", "")
    scope = {}
    for name in ("a", "b", "c"):
        if name in inputs:
            resolved = _resolve(graph, inputs[name], depth + 1)
            if resolved is None:
                return None
            scope[name] = resolved
    try:
        # Arithmetic on resolved numbers only — no names beyond a/b/c, no builtins.
        return eval(expression, {"__builtins__": {}}, scope)  # noqa: S307
    except Exception:
        return None


def _find_refine_node(graph):
    for node_id, node in graph.items():
        if node.get("class_type") == REFINE_CLASS:
            return node_id, node
    return None, None


def _guider_settings(graph, guider_ref):
    """cfg / neg_scale / prompts from whichever guider feeds the refine node."""
    out = {}
    if not (isinstance(guider_ref, list) and guider_ref):
        return out
    guider = graph.get(guider_ref[0], {})
    inputs = guider.get("inputs", {})
    out["guider_class"] = guider.get("class_type")
    for key in ("cfg", "neg_scale"):
        if key in inputs:
            out[key] = _resolve(graph, inputs[key])
    for key, label in (("positive", "positive_prompt"), ("negative", "negative_prompt")):
        ref = inputs.get(key)
        if isinstance(ref, list) and ref:
            encode = graph.get(ref[0], {})
            out[label] = _resolve(graph, encode.get("inputs", {}).get("text"))
    return out


def _named(graph, ref, field):
    """A loader's filename widget (unet_name / clip_name / vae_name / model_name)."""
    if not (isinstance(ref, list) and ref):
        return None
    return graph.get(ref[0], {}).get("inputs", {}).get(field)


def extract(png_path):
    """Flat dict of every setting the A/B harness varies or needs, from one PNG."""
    graph, size = _load_graph(png_path)
    node_id, refine = _find_refine_node(graph)
    if refine is None:
        raise ValueError("{}: no {} node in the embedded workflow".format(png_path, REFINE_CLASS))
    inputs = refine.get("inputs", {})

    settings = {
        "source_png": str(png_path),
        "png_size": list(size),
        "refine_node_id": node_id,
    }

    # --- the node's own settings (the A/B surface) ---
    for key in ("context_anchor", "context_overlap", "seam_mode", "warp_amount", "warp_scale",
                "max_tile_width", "max_tile_height"):
        if key in inputs:
            settings[key] = _resolve(graph, inputs[key])

    # --- sampling settings for the refine pass (follow the refine node's own links, NOT the
    # base-generation pass, which uses a different sampler/scheduler/seed entirely) ---
    settings.update(_guider_settings(graph, inputs.get("guider")))

    sampler_ref = inputs.get("sampler")
    if isinstance(sampler_ref, list) and sampler_ref:
        settings["sampler_name"] = graph.get(sampler_ref[0], {}).get("inputs", {}).get("sampler_name")

    sigmas_ref = inputs.get("sigmas")
    if isinstance(sigmas_ref, list) and sigmas_ref:
        sched = graph.get(sigmas_ref[0], {}).get("inputs", {})
        for key in ("scheduler", "steps", "denoise"):
            if key in sched:
                settings[key] = _resolve(graph, sched[key])
        settings["unet_name"] = _named(graph, sched.get("model"), "unet_name")

    noise_ref = inputs.get("noise")
    if isinstance(noise_ref, list) and noise_ref:
        settings["noise_seed"] = graph.get(noise_ref[0], {}).get("inputs", {}).get("noise_seed")

    settings["vae_name"] = _named(graph, inputs.get("vae"), "vae_name")

    # --- the upscale chain feeding the node, so the harness can rebuild its input image ---
    image_ref = inputs.get("image")
    for _ in range(6):                    # walk back through resize/upscale wrappers
        if not (isinstance(image_ref, list) and image_ref):
            break
        node = graph.get(image_ref[0], {})
        if node.get("class_type") == "ImageUpscaleWithModel":
            settings["upscale_model"] = _named(graph, node["inputs"].get("upscale_model"), "model_name")
            break
        image_ref = node.get("inputs", {}).get("input") or node.get("inputs", {}).get("image")

    # CLIP is reached through whichever text-encode the guider used.
    for node in graph.values():
        if node.get("class_type") == "CLIPLoader":
            settings["clip_name"] = node["inputs"].get("clip_name")
            settings["clip_type"] = node["inputs"].get("type")
            break

    return settings


ORDER = ["source_png", "png_size", "positive_prompt", "negative_prompt", "noise_seed",
         "seam_mode", "warp_amount", "warp_scale", "context_anchor", "context_overlap",
         "max_tile_width", "max_tile_height", "sampler_name", "scheduler", "steps", "denoise",
         "guider_class", "cfg", "neg_scale", "unet_name", "clip_name", "clip_type",
         "vae_name", "upscale_model", "refine_node_id"]


def _print_human(settings):
    print("=" * 78)
    print(Path(settings["source_png"]).name)
    print("=" * 78)
    for key in ORDER + [k for k in settings if k not in ORDER]:
        if key in settings and key != "source_png":
            value = settings[key]
            if isinstance(value, str) and len(value) > 90:
                value = value[:87] + "..."
            print("  {:18s} {}".format(key, value))
    print()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("images", nargs="*", type=Path, help="PNG(s) saved by ComfyUI")
    parser.add_argument("--dir", type=Path, help="extract from every *.png in this folder")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    paths = list(args.images)
    if args.dir:
        paths.extend(sorted(args.dir.glob("*.png")))
    if not paths:
        parser.error("give at least one PNG, or --dir")

    results, failures = [], 0
    for path in paths:
        try:
            results.append(extract(path))
        except Exception as exc:                       # keep going across a batch
            failures += 1
            print("SKIP {}: {}".format(path.name, exc), file=sys.stderr)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for settings in results:
            _print_human(settings)
    return 1 if failures and not results else 0


if __name__ == "__main__":
    sys.exit(main())
