# pytest support only (never imported by ComfyUI). The repo root carries __init__.py
# (the ComfyUI custom-node entry point) inside a hyphenated directory, so pytest cannot
# import it as a package and its default Package collection would exec it as a top-level
# "__init__" module, breaking the relative import. Collect the root as a plain directory.
# Registered as a global plugin because conftest-scoped hooks are never consulted for
# the rootdir itself (Session._collect_path proxies hooks through the PARENT directory).
import pytest


class _CollectRootAsPlainDir:
    @staticmethod
    def pytest_collect_directory(path, parent):
        if path == parent.config.rootpath:
            return pytest.Dir.from_parent(parent, path=path)
        return None


def pytest_configure(config):
    config.pluginmanager.register(_CollectRootAsPlainDir(), "collect-root-as-plain-dir")
