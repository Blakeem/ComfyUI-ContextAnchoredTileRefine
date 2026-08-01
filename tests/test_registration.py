import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.resolve()


def test_loads_via_comfyui_directory_mechanism():
    # Replicates the 0.3.45 loader (nodes.py:2119-2124): spec from the package's
    # __init__.py, registered in sys.modules BEFORE exec_module so relative imports
    # resolve via the spec's submodule_search_locations.
    module_name = "ComfyUI-ContextAnchoredTileUpscale_x_test"
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)

        node_class = module.NODE_CLASS_MAPPINGS["ContextAnchoredTileRefine"]
        assert isinstance(node_class, type)
        assert module.NODE_CLASS_MAPPINGS == {"ContextAnchoredTileRefine": node_class}
        assert module.NODE_DISPLAY_NAME_MAPPINGS.keys() == module.NODE_CLASS_MAPPINGS.keys()
        assert (
            module.NODE_DISPLAY_NAME_MAPPINGS["ContextAnchoredTileRefine"]
            == "Context-Anchored Tile Refine"
        )
        assert "NODE_CLASS_MAPPINGS" in module.__all__
        assert "NODE_DISPLAY_NAME_MAPPINGS" in module.__all__
    finally:
        del sys.modules[module_name]


def test_node_module_never_imports_comfy():
    # Subprocess pins the constraint independent of test order or prior imports.
    code = (
        "import sys\n"
        "import context_anchored_tile_refine.node\n"
        "assert 'comfy' not in sys.modules, 'node.py imported comfy at module scope'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_sampling_module_never_imports_comfy():
    # A module-scope comfy/latent_preview import would pass on a machine where real
    # comfy is importable while silently breaking the stub-based pure suite.
    code = (
        "import sys\n"
        "import context_anchored_tile_refine.sampling\n"
        "assert 'comfy' not in sys.modules, 'sampling.py imported comfy at module scope'\n"
        "assert 'latent_preview' not in sys.modules, 'sampling.py imported latent_preview at module scope'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_conds_module_never_imports_comfy():
    # conds.py duck-types the control objects precisely so it needs no comfy import; a
    # module-scope one would break the stub-based pure suite the same way sampling.py's would.
    code = (
        "import sys\n"
        "import context_anchored_tile_refine.conds\n"
        "assert 'comfy' not in sys.modules, 'conds.py imported comfy at module scope'\n"
        "assert 'latent_preview' not in sys.modules, 'conds.py imported latent_preview at module scope'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_grid_module_never_imports_comfy():
    # grid.py must stay pure stdlib — not even torch.
    code = (
        "import sys\n"
        "import context_anchored_tile_refine.grid\n"
        "assert 'comfy' not in sys.modules, 'grid.py imported comfy at module scope'\n"
        "assert 'torch' not in sys.modules, 'grid.py imported torch at module scope'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
