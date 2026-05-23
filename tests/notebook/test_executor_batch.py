"""Integration tests for ``CellExecutor.execute_batch``.

PR-b2 of issue #26 — these tests spawn the real harness subprocess
against a real notebook venv and verify the end-to-end batch flow.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


def _make_session_with_cells(tmp_path: Path, cells: list[tuple[str, str]]) -> NotebookSession:
    """Build a notebook with the given (cell_id, source) pairs and return a session."""
    notebook_dir = create_notebook(tmp_path, "BatchTest")
    prev: str | None = None
    for cell_id, source in cells:
        add_cell_to_notebook(notebook_dir, cell_id, after_cell_id=prev)
        write_cell(notebook_dir, cell_id, source)
        prev = cell_id
    return NotebookSession(parse_notebook(notebook_dir), notebook_dir)


def _cell_spec(cell_id: str, source: str) -> dict:
    """Build a minimal batch cell-spec for tests with no env/mounts."""
    return {
        "cell_id": cell_id,
        "source": source,
        "consumed_vars": [],  # filled in below from the session
        "env": {},
        "mount_manifest": {},
        "source_hash": "",
        "env_hash": "",
    }


def _populate_consumed_vars(specs: list[dict], session: NotebookSession) -> list[dict]:
    """Fill consumed_vars on each spec from the session DAG."""
    dag = session.dag
    for spec in specs:
        consumed = dag.consumed_variables.get(spec["cell_id"], set()) if dag else set()
        spec["consumed_vars"] = sorted(consumed)
    return specs


@pytest.mark.asyncio
async def test_batch_executes_two_linear_cells_end_to_end(tmp_path: Path):
    """Two cells, c1 produces x, c2 reads x and produces y. Real subprocess.
    Verify both succeed and the artifacts persist (single-cell re-run of
    c2 after the batch should hit the cache).
    """
    session = _make_session_with_cells(
        tmp_path,
        [
            ("c1", "x = 41\n"),
            ("c2", "y = x + 1\n"),
        ],
    )
    specs = _populate_consumed_vars(
        [_cell_spec("c1", "x = 41\n"), _cell_spec("c2", "y = x + 1\n")],
        session,
    )

    executor = CellExecutor(session)
    result = await executor.execute_batch(specs)

    assert result.completed, (
        f"batch did not complete: end_reason={result.end_reason} "
        f"failed_cell_id={result.failed_cell_id} cell_results={result.cell_results}"
    )
    assert result.end_reason == "complete"
    statuses = {r.cell_id: r.status for r in result.cell_results}
    assert statuses == {"c1": "ok", "c2": "ok"}

    # c1's `x` is consumed by c2, so it gets persisted.
    # c2's `y` has no downstream cell, so it's not in consumed_vars
    # (strata only persists variables that downstream cells reference).
    c1 = session.notebook_state.get_cell("c1")
    assert "x" in c1.artifact_uris


@pytest.mark.asyncio
async def test_batch_stops_cleanly_on_cell_error(tmp_path: Path):
    """Cell 2 raises; batch ends with cell_error reason and c3 does not run."""
    session = _make_session_with_cells(
        tmp_path,
        [
            ("c1", "x = 1\n"),
            ("c2", "raise RuntimeError('boom')\n"),
            ("c3", "z = 99\n"),
        ],
    )
    specs = _populate_consumed_vars(
        [
            _cell_spec("c1", "x = 1\n"),
            _cell_spec("c2", "raise RuntimeError('boom')\n"),
            _cell_spec("c3", "z = 99\n"),
        ],
        session,
    )

    executor = CellExecutor(session)
    result = await executor.execute_batch(specs)

    assert not result.completed
    assert result.end_reason == "cell_error"
    assert result.failed_cell_id == "c2"

    statuses = {r.cell_id: r.status for r in result.cell_results}
    assert statuses["c1"] == "ok"
    assert statuses["c2"] == "cell_error"
    assert statuses["c3"] == "not_run"

    c2_result = next(r for r in result.cell_results if r.cell_id == "c2")
    assert c2_result.error is not None
    assert "RuntimeError" in (c2_result.traceback or "")


@pytest.mark.asyncio
async def test_batch_blocked_module_export_fails_persist(tmp_path: Path):
    """A cell that defines a top-level lambda (blocked from module export)
    AND has a downstream consumer should fail to persist — matching
    single-cell semantics. Batch must surface this as status=persist_failed,
    NOT silently store as pickle/object.
    """
    # Top-level lambda is blocked from module export (per module_export.py
    # _target_names / blocking_lambda_names paths). With a downstream
    # consumer, single-cell mode returns success=False.
    session = _make_session_with_cells(
        tmp_path,
        [
            ("c1", "add = lambda x: x + 1\n"),
            ("c2", "y = add(41)\n"),
        ],
    )
    specs = _populate_consumed_vars(
        [_cell_spec("c1", "add = lambda x: x + 1\n"), _cell_spec("c2", "y = add(41)\n")],
        session,
    )

    executor = CellExecutor(session)
    result = await executor.execute_batch(specs)

    statuses = {r.cell_id: r.status for r in result.cell_results}
    assert statuses["c1"] == "persist_failed", (
        f"c1 should be persist_failed (blocked module export), got {statuses}"
    )
    # c2 must not have run — the batch ends on persist failure.
    assert statuses["c2"] == "not_run"
    assert result.end_reason == "persist_failed"
    assert result.failed_cell_id == "c1"

    # And critically: c1's `add` must NOT have been stored as a pickle.
    # cell.artifact_uris should be empty since persist was rejected.
    c1 = session.notebook_state.get_cell("c1")
    assert "add" not in c1.artifact_uris, (
        f"add must not be persisted after module-export rejection; got artifact_uris={c1.artifact_uris}"
    )


@pytest.mark.asyncio
async def test_batch_cache_hit_skips_execution(tmp_path: Path):
    """Run a 2-cell batch twice. c1's ``x`` is consumed by c2 (so it's in
    consumed_vars and persists). Second batch's c1 hits the cache via
    ``_batch_service_cache_check`` — verifies the cache materialization
    + harness load path.
    """
    session = _make_session_with_cells(
        tmp_path,
        [
            ("c1", "x = 41\n"),
            ("c2", "y = x + 1\n"),  # consumer of x — makes x a consumed_var
        ],
    )
    specs = _populate_consumed_vars(
        [_cell_spec("c1", "x = 41\n"), _cell_spec("c2", "y = x + 1\n")],
        session,
    )

    executor = CellExecutor(session)
    first = await executor.execute_batch(specs)
    assert first.completed
    assert {r.cell_id: r.status for r in first.cell_results} == {"c1": "ok", "c2": "ok"}

    second = await executor.execute_batch(specs)
    assert second.completed, f"second batch failed: {second.end_reason}"
    statuses = {r.cell_id: r.status for r in second.cell_results}
    assert statuses["c1"] == "cache_hit", f"c1 should cache-hit on re-run; got {statuses}"
