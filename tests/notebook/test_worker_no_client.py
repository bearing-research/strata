"""A cell on a worker has no ambient ``strata``, and says why rather than a bare ``NameError``."""

from __future__ import annotations

from pathlib import Path

from strata.notebook.executor import CellExecutor
from strata.notebook.models import WorkerBackendType, WorkerSpec
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

SCAN = (
    "# @worker remote\n"
    "strata.materialize(inputs=['file:///w#db.t'], transform={'executor': 'scan@v1'})\n"
)


def _session(tmp_path: Path, source: str, execute_url: str) -> NotebookSession:
    nb = create_notebook(tmp_path, "Worker")
    add_cell_to_notebook(nb, "c", None)
    write_cell(nb, "c", source)
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    session.notebook_state.workers = [
        WorkerSpec(
            name="remote",
            backend=WorkerBackendType.EXECUTOR,
            runtime_id="r",
            config={"url": execute_url},
        )
    ]
    return session


async def test_a_worker_cell_calling_strata_is_told_scans_run_on_the_server(
    tmp_path, notebook_executor_server
):
    session = _session(tmp_path, SCAN, notebook_executor_server["execute_url"])

    result = await CellExecutor(session).execute_cell("c", SCAN)

    assert not result.success
    assert result.execution_method == "executor"
    assert "cell on a worker has no ambient `strata` client" in (result.error or "")


async def test_another_undefined_name_gets_no_such_sentence(tmp_path, notebook_executor_server):
    source = "# @worker remote\nundefined_thing\n"
    session = _session(tmp_path, source, notebook_executor_server["execute_url"])

    result = await CellExecutor(session).execute_cell("c", source)

    assert not result.success
    assert "undefined_thing" in (result.error or "")
    assert "ambient" not in (result.error or "")


async def test_a_local_cell_that_deletes_strata_keeps_its_own_error(tmp_path):
    """Locally ``strata`` exists, so a ``NameError`` naming it is the cell's own doing."""
    source = "del strata\nstrata\n"
    nb = create_notebook(tmp_path, "Local")
    add_cell_to_notebook(nb, "c", None)
    write_cell(nb, "c", source)
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()

    result = await CellExecutor(session).execute_cell("c", source)

    assert not result.success
    assert "ambient" not in (result.error or "")
