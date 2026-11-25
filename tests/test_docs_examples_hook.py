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
    assert lines[2].startswith("![Data Viewer: what a run of the notebook shows]")
    assert lines[4:] == ["", "A two-cell notebook.", ""]


def test_an_example_with_no_screenshot_renders_unchanged(tmp_path: Path) -> None:
    assert hook.with_thumbnail(PAGE, "data_viewer", tmp_path) == PAGE


README = (
    "# Page a table\n\nWhat it makes, in two lines.\n\n## What it shows\n\n- One thing.\n"
    "\n### A detail\n\nMore.\n"
)
CELLS = (
    "# Page a table\n\nThe notebook's own intro.\n\n## load\n\n<sub>**kind** python</sub>\n\n"
    "```python\ndf\n```\n\n## sweep\n\n<sub>**kind** python  ·  **loop** max_iter=3 carry=x</sub>\n"
)


def test_a_page_is_the_readmes_opening_then_the_cells_then_its_notes() -> None:
    page = hook.render_page(README, hook.run_header("0" * 16) + CELLS, ran=True)
    lines = page.split("\n")

    assert lines[0] == "# Page a table"
    assert page.index("What it makes") < page.index("The notebook's own intro.")
    assert page.index("```python") < page.index("## About this example")
    # The README's sections move under "About this example", a level down.
    assert "### What it shows" in page
    assert "#### A detail" in page
    # One title: the first cell's own goes.
    assert [line for line in lines if line.startswith("# ")] == ["# Page a table"]
    assert "strata example run" not in page
    assert "No run is recorded" not in page


def test_a_cells_kind_label_goes_and_its_other_labels_stay() -> None:
    page = hook.render_page(README, CELLS, ran=False)

    assert "**kind**" not in page
    assert "<sub></sub>" not in page
    assert "<sub>**loop** max_iter=3 carry=x</sub>" in page


def test_a_page_with_no_recorded_run_says_it_shows_the_code_only() -> None:
    assert "No run is recorded" in hook.render_page(README, CELLS, ran=False)


def _notebook(root: Path, source: str) -> Path:
    (root / "cells").mkdir(parents=True)
    (root / "notebook.toml").write_text(
        'cells = [{ id = "a", file = "a.py", language = "python", order = 0 }]\n'
    )
    (root / "cells" / "a.py").write_text(source)
    return root


def test_a_recorded_run_is_used_only_while_the_cells_are_as_they_ran(tmp_path: Path) -> None:
    notebook = _notebook(tmp_path / "nb", "x = 1\n")
    run = tmp_path / "nb.md"
    run.write_text(hook.run_header(hook.source_fingerprint(notebook)) + CELLS)

    assert hook.recorded_run(run, hook.source_fingerprint(notebook)) == run.read_text()

    (notebook / "cells" / "a.py").write_text("x = 2\n")
    assert hook.recorded_run(run, hook.source_fingerprint(notebook)) is None
    assert hook.recorded_run(tmp_path / "missing.md", "0" * 16) is None


_REPO = Path(__file__).parents[1]


def test_every_recorded_run_ran_the_examples_cells_as_they_are() -> None:
    """Change an example's cells and its page shows code only until the run is recorded again:

    uv run python scripts/capture_docs_shots.py --only examples --example <name>
    """
    runs = sorted((_REPO / "docs" / hook.RUNS_DIR).glob("*.md"))
    assert runs, "no recorded runs"
    stale = [
        run.stem
        for run in runs
        if hook.recorded_run(run, hook.source_fingerprint(_REPO / "examples" / run.stem)) is None
    ]
    assert stale == []


def test_a_code_cells_name_is_marked_apart_from_the_prose_headings() -> None:
    cells = "## Step 1\n\nProse.\n\n## load\n\n<sub>**kind** python</sub>\n\n```python\ndf\n```\n"
    page = hook.render_page(README, cells, ran=False)

    assert "## load { .nb-cell }\n\n```python" in page
    assert "## Step 1\n" in page
