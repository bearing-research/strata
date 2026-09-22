"""A failed cell's error, carried by everything that copies the notebook.

The live cell view shows a failure with its traceback (#820). Two other ways
out of a notebook dropped it: the Markdown and HTML export rendered a failed
cell's source and prints and nothing about the failure, and a snapshot
round trip brought the console across but not the error, so the imported copy
of a red cell opened idle.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from strata.notebook.export import ExportFormat, ExportOptions, export_notebook
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.snapshot import write_committed_files, write_snapshot
from strata.notebook.snapshot_import import import_snapshot
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

FAILING = 'print("Checking snapshot 7")\n1 / 0\n'


@pytest.fixture
def failed(tmp_path, capsys) -> Path:
    """A notebook with one good cell and one that has actually failed."""
    from strata.notebook.cli import run_main

    nb = create_notebook(tmp_path / "src", "Failures", initialize_environment=False)
    (nb / ".venv").mkdir(exist_ok=True)  # --no-sync placeholder
    add_cell_to_notebook(nb, "good", None)
    write_cell(nb, "good", "x = 1\nx\n")
    add_cell_to_notebook(nb, "diag", "good")
    write_cell(nb, "diag", FAILING)
    run_main([str(nb), "--no-sync", "--format", "json"])  # exits 1: diag fails
    capsys.readouterr()
    return nb


def _export(nb: Path, fmt: ExportFormat, **kwargs) -> str:
    return export_notebook(nb, ExportOptions(output_format=fmt, **kwargs))


def test_the_markdown_export_says_the_cell_failed(failed):
    text = _export(failed, ExportFormat.MARKDOWN)
    assert "Checking snapshot 7" in text  # the print that preceded it, as before
    assert "ZeroDivisionError" in text
    assert "Traceback" in text


def test_the_html_export_says_the_cell_failed(failed):
    assert "ZeroDivisionError" in _export(failed, ExportFormat.HTML)


def test_an_edited_cell_carries_no_traceback_for_code_it_no_longer_has(failed):
    write_cell(failed, "diag", 'print("fixed")\n')
    assert "ZeroDivisionError" not in _export(failed, ExportFormat.MARKDOWN)


def _snapshot(nb: Path, out: Path) -> Path:
    session = NotebookSession(parse_notebook(nb), nb)
    with zipfile.ZipFile(out, "w") as archive:
        write_committed_files(session, archive)
        write_snapshot(session, archive, include="all")
    return out


def test_a_snapshot_round_trip_keeps_the_failure(failed, tmp_path):
    from strata.notebook.ops import LocalNotebookOps

    imported = import_snapshot(_snapshot(failed, tmp_path / "snap.zip"), tmp_path / "dst")

    cell = parse_notebook(imported.notebook_dir).get_cell("diag")
    assert "ZeroDivisionError" in (cell.current_error() or "")
    assert "Checking snapshot 7" in cell.console_stdout

    # And the copy reads red, not idle, with nothing executed.
    view = LocalNotebookOps(imported.notebook_dir).get_cell("diag")
    assert view.status == "error"


def test_a_snapshot_written_before_the_error_pair_still_imports(failed, tmp_path):
    """Older bundles have no error keys; a missing pair is no recorded failure."""
    bundle = _snapshot(failed, tmp_path / "snap.zip")
    legacy = tmp_path / "legacy.zip"
    with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(legacy, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "artifacts.json":
                manifest = json.loads(data)
                for cell in manifest.get("cells", {}).values():
                    cell.pop("error", None)
                    cell.pop("error_source_hash", None)
                data = json.dumps(manifest).encode()
            dst.writestr(item, data)

    imported = import_snapshot(legacy, tmp_path / "dst")

    assert parse_notebook(imported.notebook_dir).get_cell("diag").current_error() is None
