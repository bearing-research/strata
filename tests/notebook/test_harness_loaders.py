"""Smoke tests for the direct-file module loaders used by subprocess workers.

``harness.py`` and ``pool_worker.py`` load a small set of sibling modules
(``serializer.py``, ``immutability.py``, ``display/runtime.py``) via
``importlib.util.spec_from_file_location`` rather than the normal
``strata.notebook.*`` import path. That direct-file load path doesn't set
up package context, so any relative import inside those modules would
silently break the subprocess workers — and the breakage wouldn't surface
in unit tests that import the modules the normal way.

These tests guard the invariant: each loader target must remain importable
by file path with no relative imports.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_by_file(relative_path: str, module_name: str):
    """Mirror the loader pattern from harness.py / pool_worker.py."""
    from strata.notebook import display as _display_pkg  # noqa: F401

    # Anchor against the notebook package directory so the test mirrors
    # what ``Path(__file__).parent / relative_path`` does inside the
    # subprocess workers, regardless of where pytest's cwd is.
    notebook_dir = Path(_display_pkg.__file__).parent.parent
    module_path = notebook_dir / relative_path
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None, f"could not build spec for {module_path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_display_runtime_is_file_loadable():
    """``display/runtime.py`` loads via direct file path with no
    relative-import setup, mirroring the subprocess worker path."""
    mod = _load_by_file("display/runtime.py", "_test_nb_display_runtime")
    assert hasattr(mod, "Markdown")
    assert hasattr(mod, "DisplayCapture")
    markdown = mod.Markdown("# Hello")
    assert markdown._repr_markdown_() == "# Hello"


def test_serializer_is_file_loadable():
    """``serializer.py`` — the other large file-loaded target."""
    mod = _load_by_file("serializer.py", "_test_nb_serializer")
    # Sanity check: the public surface the harness reaches for.
    assert hasattr(mod, "to_serialization_safe")


def test_immutability_is_file_loadable():
    """``immutability.py`` — third file-loaded target."""
    mod = _load_by_file("immutability.py", "_test_nb_immutability")
    assert hasattr(mod, "snapshot_inputs")
    assert hasattr(mod, "detect_mutations")
