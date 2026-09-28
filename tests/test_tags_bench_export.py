"""tests-AB/ab_tags_bench.py's export: one manifest entry per tile and prompt, and PNGs that keep an
8-bit crop's pixels. Also its run's refusal to mix encoders in one record."""

import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

SCRIPTS = Path(__file__).resolve().parent.parent / "tests-AB"


@pytest.fixture(scope="module")
def bench():
    """The bench script loaded from its file. It imports its neighbours by name, so their folder sits
    on sys.path only while it loads."""
    saved_path = list(sys.path)
    spec = importlib.util.spec_from_file_location("ab_tags_bench_under_test", SCRIPTS / "ab_tags_bench.py")
    module = importlib.util.module_from_spec(spec)

    sys.path.insert(0, str(SCRIPTS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = saved_path
    return module


def test_export_manifest_holds_one_entry_per_tile_and_prompt(bench):
    street = bench.Scene("street", "unused.webp", (0, 12), {"none": "", "short": "a wet street"})
    harbor = bench.Scene("harbor", "unused.png", (3,), {"none": "", "tags": "boats, gulls"})
    sets = {"main": ({"street": street}, ("none", "short")), "holdout": ({"harbor": harbor}, ("none", "tags"))}
    digests = {"street-t00.png": "d00", "street-t12.png": "d12", "harbor-t03.png": "d03"}

    manifest = bench.export_manifest(sets, digests)

    assert manifest == [
        {"id": "tiles/street-t00", "file": "street-t00.png", "prompt": "", "prompt_name": "none",
         "scene": "street", "tile": 0, "set": "main", "sha256": "d00"},
        {"id": "tiles/street-t00", "file": "street-t00.png", "prompt": "a wet street", "prompt_name": "short",
         "scene": "street", "tile": 0, "set": "main", "sha256": "d00"},
        {"id": "tiles/street-t12", "file": "street-t12.png", "prompt": "", "prompt_name": "none",
         "scene": "street", "tile": 12, "set": "main", "sha256": "d12"},
        {"id": "tiles/street-t12", "file": "street-t12.png", "prompt": "a wet street", "prompt_name": "short",
         "scene": "street", "tile": 12, "set": "main", "sha256": "d12"},
        {"id": "tiles/harbor-t03", "file": "harbor-t03.png", "prompt": "", "prompt_name": "none",
         "scene": "harbor", "tile": 3, "set": "holdout", "sha256": "d03"},
        {"id": "tiles/harbor-t03", "file": "harbor-t03.png", "prompt": "boats, gulls", "prompt_name": "tags",
         "scene": "harbor", "tile": 3, "set": "holdout", "sha256": "d03"},
    ]


def test_export_manifest_names_every_bench_tile_and_prompt_once(bench):
    digests = {bench.export_png_name(scene.key, index): "digest"
               for registry, _prompt_keys in bench.SETS.values()
               for scene in registry.values() for index in scene.tiles}

    manifest = bench.export_manifest(bench.SETS, digests)
    keys = {(entry["id"], entry["prompt_name"]) for entry in manifest}

    # 5 main scenes x 3 tiles x 3 prompts, and 8 hold-out tiles x 3 prompts.
    assert len(manifest) == len(keys) == 69
    assert all(entry["prompt"] == "" for entry in manifest if entry["prompt_name"] == "none")


def test_export_png_keeps_an_8_bit_crop_byte_for_byte(bench):
    generator = torch.Generator().manual_seed(0)
    pixels = torch.randint(0, 256, (1, 5, 7, 3), generator=generator, dtype=torch.uint8)

    data = bench.png_bytes(pixels.float() / 255.0)

    with Image.open(io.BytesIO(data)) as png:
        assert (png.format, png.mode, png.size) == ("PNG", "RGB", (7, 5))
        assert np.array_equal(np.asarray(png), pixels[0].numpy())


def test_a_run_refuses_a_record_holding_another_encoders_results(bench):
    fp8, int8 = "qwen3vl_4b_fp8_scaled.safetensors", "qwen3-vl-4b-heretic_int8.safetensors"
    record = {"cyber8k": {"none": {"clip": fp8}, "long": {"clip": fp8}}}

    bench.check_record_encoder({}, fp8, "prod.json")
    bench.check_record_encoder(record, fp8, "prod.json")
    with pytest.raises(SystemExit, match=f"prod.json holds results from {fp8}, and this run uses {int8}"):
        bench.check_record_encoder(record, int8, "prod.json")
    # A result from before the stamp counts as another encoder.
    record["cyber8k"]["short"] = {}
    with pytest.raises(SystemExit, match=f"from {fp8}, unstamped, and .* Pass --fresh"):
        bench.check_record_encoder(record, fp8, "prod.json")
