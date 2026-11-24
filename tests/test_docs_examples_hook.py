"""The docs hook that renders examples/ into the gallery's pages."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "export_examples", Path(__file__).parents[1] / "docs_hooks" / "export_examples.py"
)
assert _spec and _spec.loader
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)

PAGE = "# Data Viewer\n\nA two-cell notebook.\n"


def test_the_screenshot_goes_under_the_title_in_both_themes(tmp_path: Path) -> None:
    (tmp_path / "data_viewer-light.webp").write_bytes(b"")

    lines = hook.with_thumbnail(PAGE, "data_viewer", tmp_path).split("\n")

    assert lines[0] == "# Data Viewer"
    assert lines[2].endswith("(../assets/examples/data_viewer-light.webp#only-light)")
    assert lines[3].endswith("(../assets/examples/data_viewer-dark.webp#only-dark)")
    assert lines[2].startswith("![The Data Viewer notebook in the editor after a run]")
    assert lines[4:] == ["", "A two-cell notebook.", ""]


def test_an_example_with_no_screenshot_renders_unchanged(tmp_path: Path) -> None:
    assert hook.with_thumbnail(PAGE, "data_viewer", tmp_path) == PAGE
