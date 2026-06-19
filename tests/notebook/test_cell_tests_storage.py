"""Storage + parse round-trip for per-cell unit tests.

Covers the committed ``cells/{id}.test.py`` sibling (writer/parser) and the
``CellTestResult`` persistence in ``.strata/runtime.json`` (runtime_state),
all without spawning pytest — those are the executor/runner tests.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from strata.notebook.parser import parse_notebook
from strata.notebook.runtime_state import (
    load_runtime_state,
    persist_cell_test_result,
)
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    write_cell,
    write_cell_tests,
)


@pytest.fixture
def notebook_dir():
    with tempfile.TemporaryDirectory() as tmp:
        nb = create_notebook(Path(tmp), "test_notebook")
        add_cell_to_notebook(nb, "cell1")
        write_cell(nb, "cell1", "def add(a, b):\n    return a + b\n")
        yield nb


def test_write_and_parse_test_source(notebook_dir):
    test_src = "def test_add(cell):\n    assert cell.add(1, 2) == 3\n"
    write_cell_tests(notebook_dir, "cell1", test_src)

    assert (notebook_dir / "cells" / "cell1.test.py").read_text() == test_src

    state = parse_notebook(notebook_dir)
    cell = next(c for c in state.cells if c.id == "cell1")
    assert cell.test_source == test_src


def test_empty_test_source_removes_file(notebook_dir):
    write_cell_tests(notebook_dir, "cell1", "def test_x(cell): assert True\n")
    test_file = notebook_dir / "cells" / "cell1.test.py"
    assert test_file.exists()

    # Clearing the editor (whitespace only) deletes the file — no empty commits.
    write_cell_tests(notebook_dir, "cell1", "   \n")
    assert not test_file.exists()

    state = parse_notebook(notebook_dir)
    cell = next(c for c in state.cells if c.id == "cell1")
    assert cell.test_source == ""


def test_write_tests_unknown_cell_raises(notebook_dir):
    with pytest.raises(FileNotFoundError):
        write_cell_tests(notebook_dir, "nope", "def test_x(cell): assert True\n")


def test_test_result_persists_and_rehydrates(notebook_dir):
    result = {
        "passed": 2,
        "failed": 1,
        "errored": 0,
        "skipped": 0,
        "tests": [
            {
                "name": "test_a",
                "nodeid": "test_cell.py::test_a",
                "outcome": "passed",
                "message": "",
            },
            {
                "name": "test_b",
                "nodeid": "test_cell.py::test_b",
                "outcome": "failed",
                "message": "assert 3 == 5",
            },
        ],
        "cell_source_hash": "abc",
        "test_source_hash": "def",
        "input_fingerprint": "ghi",
        "ran_at": 1234,
        "pytest_unavailable": False,
    }
    persist_cell_test_result(notebook_dir, "cell1", result)

    # Survives a reload of the raw runtime state...
    reloaded = load_runtime_state(notebook_dir)
    assert reloaded.cells["cell1"].test_result == result

    # ...and rehydrates into the parsed CellState as a CellTestResult model.
    state = parse_notebook(notebook_dir)
    cell = next(c for c in state.cells if c.id == "cell1")
    assert cell.test_result is not None
    assert cell.test_result.passed == 2
    assert cell.test_result.failed == 1
    assert cell.test_result.tests[1].message == "assert 3 == 5"


def test_test_result_can_be_cleared(notebook_dir):
    persist_cell_test_result(notebook_dir, "cell1", {"passed": 1, "tests": []})
    persist_cell_test_result(notebook_dir, "cell1", None)

    reloaded = load_runtime_state(notebook_dir)
    # The entry carries nothing else, so it's pruned on save.
    assert "cell1" not in reloaded.cells
