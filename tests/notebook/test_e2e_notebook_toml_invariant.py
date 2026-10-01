"""E2E invariant: running cells in a real example notebook does not churn ``notebook.toml``.

Drives a checked-in example through the real ``CellExecutor`` and asserts the committed
``notebook.toml`` is byte-identical afterwards.
"""

from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import SessionManager

EXAMPLE_DIR = Path(__file__).resolve().parents[2] / "examples" / "pandas_basics"


pytestmark = pytest.mark.integration


def _copy_example(dst: Path) -> Path:
    """Copy the pandas_basics example without its gitignored runtime dirs."""
    ignore = shutil.ignore_patterns(".strata", ".venv", "__pycache__")
    notebook_dir = dst / "pandas_basics"
    shutil.copytree(EXAMPLE_DIR, notebook_dir, ignore=ignore)
    return notebook_dir


def _inject_legacy_sections(notebook_toml: Path) -> None:
    """Append the empty pre-migration sections: artifacts, environment, cache."""
    with open(notebook_toml, "a", encoding="utf-8") as f:
        f.write("\n[artifacts]\n\n[environment]\n\n[cache]\n")


def test_copied_example_migrates_once_then_stays_stable(tmp_path: Path):
    """Opening a notebook with legacy sections rewrites it once to strip them; a second open is
    a no-op.
    """
    notebook_dir = _copy_example(tmp_path)
    notebook_toml = notebook_dir / "notebook.toml"
    _inject_legacy_sections(notebook_toml)

    before = notebook_toml.read_bytes()
    with open(notebook_toml, "rb") as f:
        raw = tomllib.load(f)
    assert "artifacts" in raw and "cache" in raw and "environment" in raw

    parse_notebook(notebook_dir)
    first_open = notebook_toml.read_bytes()
    assert first_open != before, "first open should strip legacy sections"

    with open(notebook_toml, "rb") as f:
        rewritten = tomllib.load(f)
    assert "artifacts" not in rewritten
    assert "cache" not in rewritten
    assert "environment" not in rewritten

    parse_notebook(notebook_dir)
    second_open = notebook_toml.read_bytes()
    assert second_open == first_open, "re-open must be a byte-identical no-op"


@pytest.mark.asyncio
async def test_executing_cell_in_copied_example_leaves_notebook_toml_untouched(
    tmp_path: Path,
):
    """All runtime state (display outputs, console, provenance, environment metadata) must land
    in ``.strata/``.
    """
    notebook_dir = _copy_example(tmp_path)
    notebook_toml = notebook_dir / "notebook.toml"

    session = SessionManager().open_notebook(notebook_dir)
    session.ensure_venv_synced()

    stable_bytes = notebook_toml.read_bytes()

    executor = CellExecutor(session)
    cell = next(c for c in session.notebook_state.cells if c.id == "create-data")
    result = await executor.execute_cell(cell.id, cell.source)
    assert result.success, f"cell failed: {result}"

    assert notebook_toml.read_bytes() == stable_bytes, (
        "runtime-state writes must not touch notebook.toml"
    )

    runtime_json = notebook_dir / ".strata" / "runtime.json"
    assert runtime_json.exists(), "expected runtime state to land in .strata/runtime.json"
    import json

    runtime = json.loads(runtime_json.read_text())
    assert "environment" in runtime
    cell_entry = runtime["cells"].get("create-data", {})
    assert cell_entry.get("last_provenance_hash"), "expected provenance hash to persist"
