"""Unit tests for the run-all batching partitioner (PR-b3 of issue #26).

Pure-logic tests — build a session, write cells with various
batchability-disqualifying shapes, and assert the partitioner
classifies each correctly.
"""

from __future__ import annotations

from pathlib import Path

from strata.notebook.executor import (
    CellExecutor,
    is_cell_batchable,
    partition_batchable_runs,
)
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


def _make_session(tmp_path: Path, cells: list[tuple[str, str]]) -> NotebookSession:
    notebook_dir = create_notebook(tmp_path, "PartTest")
    prev: str | None = None
    for cell_id, source in cells:
        add_cell_to_notebook(notebook_dir, cell_id, after_cell_id=prev)
        write_cell(notebook_dir, cell_id, source)
        prev = cell_id
    return NotebookSession(parse_notebook(notebook_dir), notebook_dir)


def test_plain_python_cell_is_batchable(tmp_path: Path):
    session = _make_session(tmp_path, [("c1", "x = 1\n")])
    executor = CellExecutor(session)
    cell = session.notebook_state.cells[0]
    assert is_cell_batchable(executor, cell)


def test_worker_annotation_blocks_batching(tmp_path: Path):
    session = _make_session(tmp_path, [("c1", "# @worker gpu\nx = 1\n")])
    executor = CellExecutor(session)
    cell = session.notebook_state.cells[0]
    assert not is_cell_batchable(executor, cell)


def test_loop_annotation_blocks_batching(tmp_path: Path):
    src = "# @loop max_iter=3 carry=state\nstate = state + 1\n"
    session = _make_session(tmp_path, [("c1", src)])
    executor = CellExecutor(session)
    cell = session.notebook_state.cells[0]
    assert not is_cell_batchable(executor, cell)


def test_timeout_annotation_blocks_batching(tmp_path: Path):
    src = "# @timeout 60\nx = 1\n"
    session = _make_session(tmp_path, [("c1", src)])
    executor = CellExecutor(session)
    cell = session.notebook_state.cells[0]
    assert not is_cell_batchable(executor, cell)


def test_rw_mount_blocks_batching(tmp_path: Path):
    mount_dir = tmp_path / "data"
    mount_dir.mkdir()
    src = f"# @mount scratch file://{mount_dir} rw\nx = 1\n"
    session = _make_session(tmp_path, [("c1", src)])
    executor = CellExecutor(session)
    cell = session.notebook_state.cells[0]
    assert not is_cell_batchable(executor, cell)


def test_ro_mount_does_not_block_batching(tmp_path: Path):
    mount_dir = tmp_path / "data"
    mount_dir.mkdir()
    src = f"# @mount data file://{mount_dir} ro\nx = 1\n"
    session = _make_session(tmp_path, [("c1", src)])
    executor = CellExecutor(session)
    cell = session.notebook_state.cells[0]
    assert is_cell_batchable(executor, cell)


def test_partition_three_python_cells_one_batch(tmp_path: Path):
    session = _make_session(
        tmp_path,
        [("c1", "a = 1\n"), ("c2", "b = a + 1\n"), ("c3", "c = b + 1\n")],
    )
    executor = CellExecutor(session)
    runs = partition_batchable_runs(executor, session.notebook_state.cells)
    assert len(runs) == 1
    kind, cells_in_run = runs[0]
    assert kind == "batch"
    assert [c.id for c in cells_in_run] == ["c1", "c2", "c3"]


def test_partition_worker_cell_splits_into_three_runs(tmp_path: Path):
    """``c1 → worker(c2) → c3``: each part its own run; the worker cell
    forces a process boundary."""
    session = _make_session(
        tmp_path,
        [
            ("c1", "a = 1\n"),
            ("c2", "# @worker gpu\nb = a + 1\n"),
            ("c3", "c = b + 1\n"),
        ],
    )
    executor = CellExecutor(session)
    runs = partition_batchable_runs(executor, session.notebook_state.cells)
    kinds = [k for k, _ in runs]
    cells_per_run = [[c.id for c in cs] for _, cs in runs]
    assert kinds == ["batch", "single", "batch"]
    assert cells_per_run == [["c1"], ["c2"], ["c3"]]


def test_partition_preserves_notebook_order(tmp_path: Path):
    """Topologically equivalent ordering must not be applied — partitioner
    walks the cells *in the order given*. (Important per issue #26 round-2
    finding #4 — notebook order, not topological order.)
    """
    session = _make_session(
        tmp_path,
        [
            ("c1", "a = 1\n"),
            ("c2", "# @worker gpu\nb = 1\n"),  # blocks
            ("c3", "c = 1\n"),
            ("c4", "# @worker gpu\nd = 1\n"),  # blocks
            ("c5", "e = 1\n"),
        ],
    )
    executor = CellExecutor(session)
    runs = partition_batchable_runs(executor, session.notebook_state.cells)
    kinds_and_ids = [(k, [c.id for c in cs]) for k, cs in runs]
    assert kinds_and_ids == [
        ("batch", ["c1"]),
        ("single", ["c2"]),
        ("batch", ["c3"]),
        ("single", ["c4"]),
        ("batch", ["c5"]),
    ]


def test_non_python_cell_blocks_batching(tmp_path: Path):
    """Prompt cells already run in-process — orthogonal to batching."""
    from strata.notebook.models import CellLanguage

    session = _make_session(tmp_path, [("c1", "a = 1\n")])
    executor = CellExecutor(session)
    cell = session.notebook_state.cells[0]
    # Manually flip to prompt so we don't have to set up an LLM.
    cell.language = CellLanguage.PROMPT
    assert not is_cell_batchable(executor, cell)


def test_notebook_level_worker_blocks_batching(tmp_path: Path):
    """When the notebook-level default worker is non-local, every cell
    inherits it via _resolve_effective_worker and becomes non-batchable
    (even with no per-cell annotation)."""
    session = _make_session(tmp_path, [("c1", "a = 1\n")])
    # Set notebook-level worker after session construction.
    session.notebook_state.worker = "gpu-fly"
    executor = CellExecutor(session)
    cell = session.notebook_state.cells[0]
    assert not is_cell_batchable(executor, cell)
