"""Outputs sharing a mutable object warn only when both are stored.

Stored as separate artifacts, two outputs that share an object reach downstream as
independent copies. A notebook stores what another cell reads, so ``fig, ax =
plt.subplots()`` in a cell nobody reads from is no reason to warn; a scratchpad, and
``strata run`` for a leaf, store the rest too, so there it is.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.models import WorkerBackendType, WorkerSpec
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

# ``trainer`` holds ``params`` itself, not a copy.
SHARING = "params = [1.0, 2.0]\ntrainer = {'params': params}\n"
READERS = {
    "both": "n = len(params) + len(trainer)\n",
    "one": "n = len(trainer)\n",
    "none": "n = 1\n",
}


def _session(tmp_path: Path, reader: str, *, scratchpad: bool = False) -> NotebookSession:
    notebook_dir = create_notebook(tmp_path, "Sharing", scratchpad=scratchpad)
    add_cell_to_notebook(notebook_dir, "up", None)
    write_cell(notebook_dir, "up", SHARING)
    add_cell_to_notebook(notebook_dir, "down", "up")
    write_cell(notebook_dir, "down", READERS[reader])
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


def _shared(warnings: list) -> list[str]:
    return [w["message"] for w in warnings if "share a mutable" in w["message"]]


def _expected(reader: str) -> int:
    return 1 if reader == "both" else 0


@pytest.mark.parametrize("reader", list(READERS))
async def test_a_cold_run(tmp_path, reader):
    result = await CellExecutor(_session(tmp_path, reader)).execute_cell("up", SHARING)

    assert result.success, result.error
    assert result.execution_method == "cold"
    assert len(_shared(result.mutation_warnings)) == _expected(reader)


@pytest.mark.integration
@pytest.mark.warm_pool
@pytest.mark.parametrize("reader", ["both", "none"])
async def test_a_warm_run(tmp_path, warm_pool, reader):
    session = _session(tmp_path, reader)
    cold = await CellExecutor(session).execute_cell("up", SHARING)
    pool = await warm_pool(session)

    # A rerun: the cold run's result would be a hit.
    result = await CellExecutor(session, pool).execute_cell_rerun("up", SHARING)

    assert result.success, result.error
    assert result.execution_method == "warm"
    assert len(_shared(result.mutation_warnings)) == _expected(reader)
    assert result.mutation_warnings == cold.mutation_warnings


@pytest.mark.parametrize("reader", ["both", "none"])
async def test_a_run_on_a_worker(tmp_path, notebook_executor_server, reader):
    session = _session(tmp_path, reader)
    session.notebook_state.workers = [
        WorkerSpec(
            name="remote",
            backend=WorkerBackendType.EXECUTOR,
            config={"url": notebook_executor_server["execute_url"]},
        )
    ]
    session.notebook_state.get_cell("up").worker = "remote"

    result = await CellExecutor(session).execute_cell("up", SHARING)

    assert result.success, result.error
    assert result.execution_method == "executor"
    assert len(_shared(result.mutation_warnings)) == _expected(reader)


async def test_a_scratchpad_keeps_what_nobody_reads_yet(tmp_path):
    """Its first reader loads the stored copies, decoupled."""
    session = _session(tmp_path, "none", scratchpad=True)

    result = await CellExecutor(session).execute_cell("up", SHARING)

    assert result.success, result.error
    assert len(_shared(result.mutation_warnings)) == 1


async def test_a_stochastic_scratchpad_cell_keeps_nothing_unread(tmp_path):
    """Its run is kept under its own key, which no reader finds: nothing unread is stored."""
    session = _session(tmp_path, "none", scratchpad=True)
    source = "# @stochastic\n" + SHARING
    write_cell(session.path, "up", source)
    session.notebook_state.get_cell("up").source = source
    session.re_analyze_cell("up")

    result = await CellExecutor(session).execute_cell("up", source)

    assert result.success, result.error
    assert _shared(result.mutation_warnings) == []


async def test_a_loops_carry_is_stored(tmp_path):
    """Each iteration keeps the carry: one it shares with a stored output decouples."""
    source = "# @loop max_iter=1 carry=state\nmodel = [0.0]\nstate = {'model': model}\n"
    notebook_dir = create_notebook(tmp_path, "Loop")
    add_cell_to_notebook(notebook_dir, "seed", None)
    write_cell(notebook_dir, "seed", "state = {}\n")
    add_cell_to_notebook(notebook_dir, "up", "seed")
    write_cell(notebook_dir, "up", source)
    add_cell_to_notebook(notebook_dir, "down", "up")
    write_cell(notebook_dir, "down", "n = len(model)\n")
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    executor = CellExecutor(session)
    assert (await executor.execute_cell("seed", "state = {}\n")).success

    result = await executor.execute_cell("up", source)

    assert result.success, result.error
    assert len(_shared(result.mutation_warnings)) == 1


async def test_strata_run_keeps_a_leafs_outputs(tmp_path):
    notebook_dir = create_notebook(tmp_path, "Leaf")
    add_cell_to_notebook(notebook_dir, "up", None)
    write_cell(notebook_dir, "up", SHARING)
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    executor = CellExecutor(session)
    executor.store_leaf_outputs = True

    result = await executor.execute_cell("up", SHARING)

    assert result.success, result.error
    assert len(_shared(result.mutation_warnings)) == 1


async def test_a_worker_refuses_stored_vars_that_are_not_names(monkeypatch):
    from strata.notebook import remote_executor
    from strata.notebook.remote_executor import create_notebook_executor_app

    async def no_harness(*_args, **_kwargs):
        raise AssertionError("the harness ran")

    monkeypatch.setattr(remote_executor, "_run_harness", no_harness)
    monkeypatch.delenv("STRATA_WORKER_TOKEN", raising=False)
    metadata = {
        "protocol_version": remote_executor.NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
        "source": "x = 1",
        "stored_vars": "x",
    }
    transport = httpx.ASGITransport(app=create_notebook_executor_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
        response = await client.post(
            "/v1/notebook-execute",
            files={"metadata": ("metadata.json", json.dumps(metadata), "application/json")},
        )

    assert response.status_code == 400
    assert "stored_vars" in response.text
