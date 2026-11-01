"""Strata's vocabulary reports what ``validate_cell_annotations`` reports, cell by cell.

The traces cover what the scenarios exercise; these cells pin the checks they do
not: each is opened by a real session and analyzed by cellgraph, and the two must
agree on every diagnostic (code, severity, line) and on whether the cell runs.
"""

from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("the notebook subsystem is not run on Windows", allow_module_level=True)

from strata.notebook.annotation_validation import DiagnosticSeverity  # noqa: E402
from strata.notebook.parser import parse_notebook  # noqa: E402
from strata.notebook.session import NotebookSession  # noqa: E402
from tests.contract.cellgraph import replay  # noqa: E402

CASES = {
    "output_schema_malformed": [("p", "prompt", "# @output_schema {not json\nhello", None)],
    "stochastic_every_line": [("s", "python", "# @stochastic maybe\n# @stochastic\nx = 1", None)],
    "nocache_every_line": [("n", "python", "# @nocache maybe\n# @nocache\nx = 1", None)],
    "loop_carry_unknown_line": [("l", "python", "# @loop max_iter=3 carry=y\nx = 1", None)],
    "loop_every_problem": [("l", "python", "# @loop carry=x\n# @loop_until (\nx = 1", None)],
    "loop_bad_token_keeps_carry": [("l", "python", "# @loop max_iter=abc carry=x\nx = 1", None)],
    "prompt_name_line": [("p", "prompt", "# @name 1bad\nhello", None)],
    "per_variant_overwrite_line": [
        (
            "w",
            "python",
            "# @per_variant\n# @write_table y cat:ns.t overwrite\ny = 1",
            None,
        )
    ],
    "every_unreadable_mount": [("m", "python", "# @mount a\n# @mount b\nx = 1", None)],
    "cell_mount_is_not_notebook_mount": [
        (
            "m",
            "python",
            "# @mount d file:///tmp/x\nprint(d)",
            'mounts = [{ name = "d", uri = "file:///tmp/y", mode = "ro" }]',
        )
    ],
    "markdown_headings": [
        ("d", "markdown", "# @mentions are cool\n# @worker\n# @mount foo\nText.", None),
        ("x", "python", "y = 1", None),
    ],
    "markdown_reads_only_app": [
        ("d", "markdown", "# @app show\n# @after x\n# @variant g m\nText.", None),
        ("x", "python", "y = 1", None),
    ],
    "app_unreadable_lines": [("a", "python", "# @app show\n# @app hide\nx = 1", None)],
    "table_checks": [
        (
            "t",
            "python",
            "# @table a file:///w#events\n# @table b ns.t\n# @table b ns.u\na = 1\nprint(b)",
            None,
        )
    ],
}


def _write(notebook_dir, cells, notebook_mounts=None):
    (notebook_dir / "cells").mkdir(parents=True)
    rows = []
    for order, (cell_id, language, source, cell_toml) in enumerate(cells):
        (notebook_dir / "cells" / f"{cell_id}.py").write_text(source + "\n")
        extra = f", {cell_toml}" if cell_toml else ""
        rows.append(
            f'    {{ id = "{cell_id}", file = "{cell_id}.py", language = "{language}", '
            f"order = {order}{extra} }},"
        )
    mounts = f"mounts = {notebook_mounts}\n" if notebook_mounts else ""
    (notebook_dir / "notebook.toml").write_text(
        'notebook_id = "parity"\nname = "parity"\nworkers = []\n'
        + mounts
        + "cells = [\n"
        + "\n".join(rows)
        + "\n]\n"
    )
    (notebook_dir / "pyproject.toml").write_text(
        '[project]\nname = "parity"\nversion = "0.1.0"\nrequires-python = ">=3.12"\n'
        "dependencies = []\n"
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_cellgraph_reports_what_strata_reports(name, tmp_path):
    notebook_dir = tmp_path / name
    _write(notebook_dir, CASES[name])
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    analysis = replay.analyze(parse_notebook(notebook_dir))

    for cell in session.notebook_state.cells:
        strata = sorted((d.code, d.severity.value, d.line) for d in cell.annotation_diagnostics)
        ours = sorted(
            (d.code, d.severity.value, d.line) for d in analysis.cells[cell.id].diagnostics
        )
        assert ours == strata, cell.id
        runs = not any(d.severity == DiagnosticSeverity.ERROR for d in cell.annotation_diagnostics)
        assert analysis.cells[cell.id].plannable == runs, cell.id


def test_a_notebook_mount_shadowed_by_an_annotation(tmp_path):
    """The positive case of ``cell_mount_is_not_notebook_mount``."""
    notebook_dir = tmp_path / "shadow"
    _write(
        notebook_dir,
        [("m", "python", "# @mount d file:///tmp/x\nprint(d)", None)],
        notebook_mounts='[{ name = "d", uri = "file:///tmp/y", mode = "ro" }]',
    )
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    analysis = replay.analyze(parse_notebook(notebook_dir))
    (cell,) = session.notebook_state.cells
    strata = [(d.code, d.severity.value, d.line) for d in cell.annotation_diagnostics]
    ours = [(d.code, d.severity.value, d.line) for d in analysis.cells["m"].diagnostics]
    assert ours == strata == [("mount_shadows_notebook", "info", 1)]
