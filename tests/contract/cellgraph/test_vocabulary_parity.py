"""Strata's vocabulary reports what ``validate_cell_annotations`` reports, cell by cell.

The traces cover what the scenarios exercise; these cells pin the checks they do
not: each is opened by a real session and analyzed by cellgraph, and the two must
agree on every diagnostic (code, severity, line and message, in order) and on
whether the cell runs.
"""

from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("the notebook subsystem is not run on Windows", allow_module_level=True)

from strata.notebook.annotation_validation import DiagnosticSeverity  # noqa: E402
from strata.notebook.parser import parse_notebook  # noqa: E402
from strata.notebook.session import NotebookSession  # noqa: E402
from strata.notebook.vocabulary import strata_diagnostics  # noqa: E402
from tests.contract.cellgraph import record, replay  # noqa: E402

CASES = {
    "output_schema_malformed": [("p", "prompt", "# @output_schema {not json\nhello", None)],
    "stochastic_every_line": [("s", "python", "# @stochastic maybe\n# @stochastic\nx = 1", None)],
    "stochastic_then_nocache_off": [
        ("s", "python", "# @stochastic\n# @nocache off\nx = 1", None),
        ("d", "python", "print(x)", None),
    ],
    "nocache_off_then_stochastic": [("s", "r", "# @nocache off\n# @stochastic\nx <- 1", None)],
    "nocache_off_then_on_with_stochastic": [
        ("s", "python", "# @nocache off\n# @nocache\n# @stochastic\nx = 1", None)
    ],
    "stochastic_off_with_nocache_off": [
        ("s", "python", "# @stochastic off\n# @nocache off\nx = 1", None)
    ],
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

_SWITCH = '\n[[variant_group]]\ngroup = "g"\nmode = "switch"\n'
_SWEEP = '\n[[variant_group]]\ngroup = "g"\nmode = "sweep"\n'
_SWEEPS = _SWEEP + '\n[[variant_group]]\ngroup = "h"\nmode = "sweep"\n'


def _members(group="g", defines=("y", "y")):
    return [
        (f"v{i}", "python", f"# @variant {group} m{i}\n{name} = {i}", None)
        for i, name in enumerate(defines, start=1)
    ]


# The checks cellgraph's core makes (after, seeds, variants, fan-out, unknown keys),
# each with the notebook.toml it needs.
CORE_CASES = {
    "annotation_unknown": ([("w", "python", "# @wroker gpu\n# @zzz\nx = 1", None)], ""),
    "after_each_line": (
        [
            ("a", "python", "x = 1", None),
            ("c", "python", "# @after a\n# @after ghost c\nz = 1", None),
        ],
        "",
    ),
    "after_inactive_variant": (
        [*_members(), ("d", "python", "# @after v2\nprint(1)", None)],
        _SWITCH,
    ),
    "seed_unknown_cell": (
        [("l", "python", "# @loop max_iter=3 carry=x start_from=ghost@iter=1\nx = 1", None)],
        "",
    ),
    "seed_self": (
        [("l", "python", "# @loop max_iter=3 carry=x start_from=l@iter=1\nx = 1", None)],
        "",
    ),
    "seed_inactive_variant": (
        [
            *_members(defines=("x", "x")),
            ("l", "python", "# @loop max_iter=3 carry=x start_from=v2@iter=1\nx = x + 1", None),
        ],
        _SWITCH,
    ),
    "seed_without_the_carry": (
        [
            ("s", "python", "y = 1", None),
            ("l", "python", "# @loop max_iter=3 carry=x start_from=s@iter=1\nx = 1", None),
        ],
        "",
    ),
    "variant_malformed": ([("v", "python", "# @variant onlyone\nx = 1", None)], ""),
    "variant_contract_mismatch": (_members(defines=("y", "z")), _SWITCH),
    "variant_mode_invalid": (
        _members(),
        '\n[[variant_group]]\ngroup = "g"\nmode = "bogus"\n',
    ),
    "variant_active_redundant": (_members(), _SWEEP + 'active = "m2"\n'),
    "variant_active_unknown": (_members(), _SWITCH + 'active = "nope"\n'),
    "per_variant_on_variant_member": (
        [("v", "python", "# @variant g m\n# @per_variant\ny = 1", None)],
        _SWEEP,
    ),
    "per_variant_unknown_group": (
        [*_members(), ("f", "python", "# @per_variant other\nz = y", None)],
        _SWEEP,
    ),
    "per_variant_no_sweep_source": ([("f", "python", "# @per_variant\nz = 1", None)], ""),
    "per_variant_ambiguous_group": (
        [
            *_members(),
            ("h1", "python", "# @variant h n1\nw = 1", None),
            ("h2", "python", "# @variant h n2\nw = 2", None),
            ("f", "python", "# @per_variant\nz = y + w", None),
        ],
        _SWEEPS,
    ),
    "per_variant_group_of_one": (
        [
            ("v1", "python", "# @variant g m1\ny = 1", None),
            ("f", "python", "# @per_variant\nz = y", None),
        ],
        _SWEEP,
    ),
}


# The code a case exercises, where it is not the case's name.
CORE_CODES = {
    "after_each_line": "after_unknown_cell",
    "seed_unknown_cell": "loop_start_from_unknown",
    "seed_self": "loop_start_from_unknown",
    "seed_inactive_variant": "loop_start_from_unknown",
    "seed_without_the_carry": "loop_start_from_unknown",
}


def _write(notebook_dir, cells, notebook_mounts=None, extra_toml=""):
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
        + extra_toml
    )
    (notebook_dir / "pyproject.toml").write_text(
        '[project]\nname = "parity"\nversion = "0.1.0"\nrequires-python = ">=3.12"\n'
        "dependencies = []\n"
    )


_CONNECTIONS = """
[connections.good]
driver = "sqlite"
path = "db.sqlite"

[connections.broken]
path = "db.sqlite"

[connections.odd]
driver = "nosuch"

[connections.secret]
driver = "sqlite"
path = "db.sqlite"

[connections.secret.auth]
password = "hunter2"
"""

# A SQL cell named ``t``, read by a Python cell, and what Strata reports on it.
SQL_CASES = {
    "no_sql_line": ("select 1 as v", [("sql_connection_missing", "error", None)]),
    "unknown_connection": (
        "# @sql connection=elsewhere\nselect 1 as v",
        [("sql_connection_unknown", "warn", 2)],
    ),
    "malformed_connection": (
        "# @sql connection=broken\nselect 1 as v",
        [("connection_malformed", "error", 2)],
    ),
    "unknown_driver": (
        "# @sql connection=odd\nselect 1 as v",
        [("connection_driver_unknown", "error", 2)],
    ),
    "valid_connection": ("# @sql connection=good\nselect 1 as v", []),
    "auth_literal": (
        "# @sql connection=secret\nselect 1 as v",
        [("connection_auth_literal_secret", "warn", 2)],
    ),
    # The later line wins, so its line is the one reported.
    "later_sql_line_wins": (
        "# @sql connection=good\n# @sql connection=elsewhere\nselect 1 as v",
        [("sql_connection_unknown", "warn", 3)],
    ),
}


def _view(diagnostics):
    return [(d.code, d.severity.value, d.line, d.message) for d in diagnostics]


def _assert_parity(notebook_dir):
    """Every cell's diagnostics and whether it runs agree."""
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    analysis = replay.analyze(parse_notebook(notebook_dir))
    cells = {cell.id: cell for cell in session.notebook_state.cells}

    def runs(cell_id):
        # An error stops the cell, and Strata skips everything downstream of it.
        cell = cells[cell_id]
        if any(d.severity == DiagnosticSeverity.ERROR for d in cell.annotation_diagnostics):
            return False
        return all(runs(upstream) for upstream in cell.upstream_ids)

    for cell in cells.values():
        ours = _view(strata_diagnostics(analysis.cells[cell.id]))
        assert ours == _view(cell.annotation_diagnostics), cell.id
        # A broken DAG is ``dag_error``: Strata has no graph to say what runs.
        if session.dag_error is None:
            assert analysis.cells[cell.id].plannable == runs(cell.id), cell.id
    return session, analysis


@pytest.mark.parametrize("name", sorted(CASES))
def test_cellgraph_reports_what_strata_reports(name, tmp_path):
    notebook_dir = tmp_path / name
    _write(notebook_dir, CASES[name])
    _assert_parity(notebook_dir)


@pytest.mark.parametrize("name", sorted(CORE_CASES))
def test_cellgraph_reports_its_core_checks_as_strata_does(name, tmp_path):
    cells, extra_toml = CORE_CASES[name]
    notebook_dir = tmp_path / name
    _write(notebook_dir, cells, extra_toml=extra_toml)
    session, _analysis = _assert_parity(notebook_dir)
    codes = {d.code for cell in session.notebook_state.cells for d in cell.annotation_diagnostics}
    assert CORE_CODES.get(name, name) in codes


@pytest.mark.parametrize("name", sorted(SQL_CASES))
def test_cellgraph_reports_the_sql_connection_as_strata_does(name, tmp_path):
    source, expected = SQL_CASES[name]
    notebook_dir = tmp_path / name
    cells = [("q", "sql", "# @name t\n" + source, None), ("d", "python", "print(t)", None)]
    _write(notebook_dir, cells, extra_toml=_CONNECTIONS)
    session, _analysis = _assert_parity(notebook_dir)

    sql, reader = session.notebook_state.cells
    assert [(d.code, d.severity.value, d.line) for d in sql.annotation_diagnostics] == expected
    assert reader.upstream_ids == ["q"]


@pytest.mark.parametrize("name", [n for n in record.scenario_names() if n not in record.REQUIRES])
def test_every_scenario_step_reports_what_strata_reports(name, tmp_path):
    """The traces record codes and lines; this holds messages, order and ``dag_error`` too."""
    for _index, _step, _state in replay.notebooks(name, tmp_path):
        session, analysis = _assert_parity(tmp_path / name)
        structural = [d.message for d in analysis.structural_errors]
        assert session.dag_error == (structural[0] if structural else None)


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
    ours = [(d.code, d.severity.value, d.line) for d in strata_diagnostics(analysis.cells["m"])]
    assert ours == strata == [("mount_shadows_notebook", "info", 1)]
