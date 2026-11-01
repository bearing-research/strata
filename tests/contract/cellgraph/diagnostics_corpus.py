"""Notebooks whose diagnostics ``diagnostics_oracle.json`` holds, as Strata reported them
before validation ran on cellgraph.

Every annotation line, readable and not, in a cell of every language, beside a Python
cell it can read from. The oracle was recorded once, by the validators cellgraph
replaced; it is not re-recorded.
"""

from __future__ import annotations

from pathlib import Path

LANGUAGES = ("python", "r", "prompt", "sql", "widget")

# The body a cell of each language reads ``x`` with, after its annotation lines.
_BODIES = {
    "python": "y = x",
    "r": "y <- x",
    "prompt": "Use {{ x }}.",
    "sql": "select 1 as v",
    "widget": "y = slider(0, 1, default=0.5)",
}

LINES = (
    "# @worker gpu",
    "# @worker",
    "# @timeout 5",
    "# @timeout x",
    "# @env K=v",
    "# @env bad",
    "# @mount d file:///tmp/x",
    "# @mount d ftp://host/x",
    "# @mount a",
    "# @table a file:///w#events",
    "# @table b ns.t\n# @table b ns.u",
    "# @table x ns.t",
    "# @fetch f",
    "# @fetch f https://example.com/f.csv",
    "# @dataset",
    "# @write_table y cat:ns.t overwrite",
    "# @write_table bad",
    "# @output_dir out",
    "# @output_dir",
    "# @nocache",
    "# @nocache maybe",
    "# @stochastic",
    "# @stochastic maybe",
    "# @stochastic\n# @nocache off",
    "# @after p",
    "# @after ghost",
    "# @loop max_iter=3 carry=y",
    "# @loop max_iter=3 carry=z",
    "# @loop carry=y",
    "# @loop max_iter=abc carry=y",
    "# @loop max_iter=3 carry=y\n# @loop_until y >",
    "# @loop_until y > 1",
    "# @loop max_iter=3 carry=y start_from=zz@iter=1",
    "# @loop max_iter=3 carry=x start_from=p@iter=1",
    "# @name foo",
    "# @name 1bad",
    "# @name",
    "# @model m",
    "# @temperature x",
    "# @temperature 0.5",
    "# @max_tokens -1",
    "# @system be brief",
    "# @output json",
    "# @output xml",
    "# @output_schema {bad",
    '# @output_schema {"type": "object"}',
    "# @validate_retries x",
    "# @sql connection=good",
    "# @sql connection=elsewhere",
    "# @sql",
    "# @cache ttl=x",
    "# @cache forever",
    "# @live",
    "# @live maybe",
    "# @app hide",
    "# @app show",
    "# @variant g a",
    "# @variant onlyone",
    "# @per_variant",
    "# @per_variant a b",
    "# @zzz",
    "# @worker gpu\n# @timeout x\n# @loop carry=y\n# @mount a",
    # Checks that run in turn: the order is the validators'.
    "# @loop carry=y\n# @mount d ftp://host/x",
    "# @loop max_iter=3 carry=z\n# @table x ns.t\n# @mount d ftp://host/x",
    "# @loop carry=y start_from=zz@iter=1\n# @table a file:///w#events",
)

_TOML = """
[connections.good]
driver = "sqlite"
path = "db.sqlite"
"""

# With a cycle, so the session has no DAG.
_CYCLE = ("# @per_variant", "# @per_variant a b", "# @variant g a", "# @loop max_iter=3 carry=x")


def cases() -> dict[str, list[tuple[str, str, str]]]:
    """Each case's cells, as ``(id, language, source)``; ``p`` defines ``x``."""
    found: dict[str, list[tuple[str, str, str]]] = {}
    for language in LANGUAGES:
        for index, line in enumerate(LINES):
            found[f"{language}-{index:02d}"] = [
                ("p", "python", "x = 1"),
                ("c", language, f"{line}\n{_BODIES[language]}"),
            ]
    for index, line in enumerate(_CYCLE):
        found[f"cycle-{index}"] = [
            ("p", "python", f"{line}\n# @after c\nx = 1"),
            ("c", "python", "y = x"),
        ]
    return found


def write(notebook_dir: Path, cells: list[tuple[str, str, str]]) -> None:
    (notebook_dir / "cells").mkdir(parents=True)
    rows = [
        f'    {{ id = "{cell_id}", file = "{cell_id}.py", language = "{language}", '
        f"order = {order} }},"
        for order, (cell_id, language, _source) in enumerate(cells)
    ]
    for cell_id, _language, source in cells:
        (notebook_dir / "cells" / f"{cell_id}.py").write_text(source + "\n")
    (notebook_dir / "notebook.toml").write_text(
        'notebook_id = "oracle"\nname = "oracle"\nworkers = []\ncells = [\n'
        + "\n".join(rows)
        + "\n]\n"
        + _TOML
    )
    (notebook_dir / "pyproject.toml").write_text(
        '[project]\nname = "oracle"\nversion = "0.1.0"\nrequires-python = ">=3.12"\n'
        "dependencies = []\n"
    )


def observe(notebook_dir: Path) -> dict:
    """What a session opened on *notebook_dir* reports: each cell's diagnostics and
    upstream cells, and ``dag_error``."""
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession

    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    return {
        "dag_error": session.dag_error,
        "cells": {
            cell.id: [
                [d.code, d.severity.value, d.line, d.message] for d in cell.annotation_diagnostics
            ]
            for cell in session.notebook_state.cells
        },
        "upstream": {cell.id: list(cell.upstream_ids) for cell in session.notebook_state.cells},
    }
