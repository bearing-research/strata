"""A ``start_from`` seed is the only producer of a loop's carry."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from strata.notebook.dag import CellAnalysisWithId, NotebookDag
from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


def _cells(seed_id: str) -> list[CellAnalysisWithId]:
    return [
        CellAnalysisWithId(id="L", defines=["state"], references=[]),
        CellAnalysisWithId(id="P", defines=["state"], references=[]),
        CellAnalysisWithId(
            id="F", defines=["state"], references=["state"], seed_from=(seed_id, "state")
        ),
    ]


def test_the_seed_is_the_carrys_only_edge():
    dag = NotebookDag.from_cells(_cells("L"))

    assert [(e.from_cell_id, e.to_cell_id, e.variable) for e in dag.edges] == [("L", "F", "state")]
    assert dag.cell_upstream["F"] == ["L"]
    assert dag.read_producer("F", "state") == "L"
    assert dag.consumed_variables["P"] == set()


def test_an_unknown_seed_does_not_fall_back_to_the_ordinary_producer():
    dag = NotebookDag.from_cells(_cells("ghost"))

    assert dag.cell_upstream["F"] == []
    assert dag.edges == []


LOOP = "# @loop max_iter=2 carry=state\nstate = state + 1\n"
FORK = "# @loop max_iter=2 carry=state start_from=L@iter=0\nstate = state + 10\n"


def _session(tmp_path: Path, cells: dict[str, str]) -> NotebookSession:
    nb = create_notebook(tmp_path, "Seeded", initialize_environment=False)
    after = None
    for cell_id, source in cells.items():
        add_cell_to_notebook(nb, cell_id, after)
        write_cell(nb, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(nb), nb)
    session.venv_python = Path(sys.executable)
    return session


@pytest.mark.parametrize(
    "cells, why",
    [
        ({"F": FORK.replace("L@", "ghost@")}, "does not exist"),
        ({"L": "x = 1\n", "F": FORK}, "does not define the carry"),
        (
            # The first variant is the active one, so the second never runs.
            {
                "A": "# @variant g a\nstate = 1\n",
                "L": "# @variant g b\nstate = 2\n",
                "F": FORK,
            },
            "not active",
        ),
    ],
    ids=["unknown", "no-carry", "inactive"],
)
def test_an_invalid_seed_blocks_the_cell_without_an_ordinary_edge(tmp_path, cells, why):
    session = _session(tmp_path, {"P": "state = 0\n", **cells})
    assert session.dag.cell_upstream["F"] == []

    result = asyncio.run(
        CellExecutor(session).execute_cell("F", session.notebook_state.get_cell("F").source)
    )

    assert result.error_code == "loop_start_from_unknown"
    assert why in result.error


def test_a_fork_follows_its_seed_not_a_later_producer_of_the_carry(tmp_path, capsys):
    from strata.cli import main
    from tests.notebook.test_cli import _build_notebook, _mk_fake_venv

    nb = _build_notebook(
        tmp_path,
        cells=[
            ("init", "state = 0\n", None),
            ("L", LOOP, "init"),
            ("P", "state = 100\n", "L"),
            ("F", FORK, "P"),
            ("O", "print(state)\n", "F"),
        ],
    )
    _mk_fake_venv(nb)

    def run() -> dict[str, dict]:
        assert main(["run", str(nb), "--no-sync", "--format", "json"]) == 0
        return {c["id"]: c for c in json.loads(capsys.readouterr().out)["cells"]}

    # L's iteration 0 left state == 1, so the fork's two steps make 21 (P's 100 would make 120).
    assert run()["O"]["stdout"].strip() == "21"

    write_cell(nb, "P", "state = 200\n")
    after_p = run()
    assert after_p["P"]["cache_hit"] is False
    assert after_p["F"]["cache_hit"] is True

    write_cell(nb, "L", LOOP.replace("state + 1", "state + 2"))
    after_l = run()
    assert after_l["F"]["cache_hit"] is False
    assert after_l["O"]["stdout"].strip() == "22"
