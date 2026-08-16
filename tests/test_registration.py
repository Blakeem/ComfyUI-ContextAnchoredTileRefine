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
        vl_class = module.NODE_CLASS_MAPPINGS["ContextAnchoredTileRefineVL"]
        upscale_class = module.NODE_CLASS_MAPPINGS["ContextAnchoredTileUpscaleVL"]
        assert isinstance(node_class, type)
        assert isinstance(vl_class, type)
        assert isinstance(upscale_class, type)
        assert {
            "ContextAnchoredTileRefine": node_class,
            "ContextAnchoredTileRefineVL": vl_class,
            "ContextAnchoredTileUpscaleVL": upscale_class,
        } == module.NODE_CLASS_MAPPINGS
        assert module.NODE_DISPLAY_NAME_MAPPINGS.keys() == module.NODE_CLASS_MAPPINGS.keys()
        assert (
            module.NODE_DISPLAY_NAME_MAPPINGS["ContextAnchoredTileRefine"]
            == "Context-Anchored Tile Refine"
        )
        assert (
            module.NODE_DISPLAY_NAME_MAPPINGS["ContextAnchoredTileRefineVL"]
            == "Context-Anchored Tile Refine (VL)"
        )
        assert (
            module.NODE_DISPLAY_NAME_MAPPINGS["ContextAnchoredTileUpscaleVL"]
            == "Context-Anchored Tile Upscale (VL)"
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


def test_vl_module_never_imports_comfy():
    # vl.py holds the same lazy-import contract as sampling.py: torch at module scope,
    # comfy only inside functions.
    code = (
        "import sys\n"
        "import context_anchored_tile_refine.vl\n"
        "assert 'comfy' not in sys.modules, 'vl.py imported comfy at module scope'\n"
        "assert 'latent_preview' not in sys.modules, 'vl.py imported latent_preview at module scope'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_captions_module_never_imports_comfy():
    # captions.py holds the same lazy-import contract as vl.py, which it imports at module
    # scope: torch (and vl) at module scope, comfy only inside functions.
    code = (
        "import sys\n"
        "import context_anchored_tile_refine.captions\n"
        "assert 'comfy' not in sys.modules, 'captions.py imported comfy at module scope'\n"
        "assert 'latent_preview' not in sys.modules, 'captions.py imported latent_preview at module scope'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_upscale_module_never_imports_comfy():
    # upscale.py holds the same lazy-import contract as sampling.py / vl.py: torch at
    # module scope, comfy only inside functions.
    code = (
        "import sys\n"
        "import context_anchored_tile_refine.upscale\n"
        "assert 'comfy' not in sys.modules, 'upscale.py imported comfy at module scope'\n"
        "assert 'latent_preview' not in sys.modules, 'upscale.py imported latent_preview at module scope'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_stepper_module_never_imports_comfy():
    # stepper.py holds the same lazy-import contract as sampling.py: torch (and stdlib
    # threading) at module scope, comfy only inside functions.
    code = (
        "import sys\n"
        "import context_anchored_tile_refine.stepper\n"
        "assert 'comfy' not in sys.modules, 'stepper.py imported comfy at module scope'\n"
        "assert 'latent_preview' not in sys.modules, 'stepper.py imported latent_preview at module scope'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_sync_module_never_imports_comfy():
    # sync.py imports sampling/vl/captions/conds at module scope (no cycle — sampling imports
    # sync lazily) and must inherit their contract: torch at module scope, comfy only inside
    # functions, stepper.py lazily too.
    code = (
        "import sys\n"
        "import context_anchored_tile_refine.sync\n"
        "assert 'comfy' not in sys.modules, 'sync.py imported comfy at module scope'\n"
        "assert 'latent_preview' not in sys.modules, 'sync.py imported latent_preview at module scope'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_progress_module_never_imports_comfy():
    # progress.py is stdlib ONLY — it holds no tensors, so torch has no business there
    # either, and every comfy touch (the run's one ProgressBar, the scoped shim) is
    # function-scope. upscale.py and sync.py import it at THEIR module scope, so a comfy
    # import here would break their contracts too. `server` is held to the same rule and
    # matters more: it is ComfyUI's aiohttp web app, so importing it to write a status line
    # would drag a web server into every headless harness run.
    code = (
        "import sys\n"
        "import context_anchored_tile_refine.progress\n"
        "assert 'comfy' not in sys.modules, 'progress.py imported comfy at module scope'\n"
        "assert 'latent_preview' not in sys.modules, 'progress.py imported latent_preview at module scope'\n"
        "assert 'torch' not in sys.modules, 'progress.py imported torch at module scope'\n"
        "assert 'server' not in sys.modules, 'progress.py imported server at module scope'\n"
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
