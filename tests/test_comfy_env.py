import importlib

import pytest


@pytest.mark.comfy
def test_comfy_samplers_importable(comfy_env):
    # Safe under pytest: comfy.options.args_parsing is False outside main.py,
    # so comfy.cli_args parses an empty argv.
    assert (comfy_env / "comfy").is_dir()
    samplers = importlib.import_module("comfy.samplers")
    assert hasattr(samplers, "CFGGuider")
