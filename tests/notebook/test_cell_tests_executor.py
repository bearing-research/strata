"""Integration tests for cell unit tests: executor and WebSocket handler.

A real pytest subprocess runs with the test interpreter as the notebook venv. WS handlers
get a fake WebSocket, never ``TestClient.websocket_connect`` (it hangs on py3.12/macOS).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import cast

import pytest
from fastapi import WebSocket

from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.runtime_state import load_runtime_state
from strata.notebook.session import NotebookSession
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    write_cell,
    write_cell_tests,
)
from tests.notebook.e2e_fixtures import FakeNotebookWebSocket
from tests.notebook.e2e_fixtures import _reset_ws_globals as _reset_e2e_ws_globals


@pytest.fixture(autouse=True)
def _reset_ws_globals():
    _reset_e2e_ws_globals()
    yield
    _reset_e2e_ws_globals()


def _session_with(cells: list[tuple[str, str, str | None]]) -> NotebookSession:
    """A session whose notebook venv is the test interpreter.

    ``cells`` is a list of ``(cell_id, source, after_cell_id)``.
    """
    tmp = Path(tempfile.mkdtemp())
    nb = create_notebook(tmp, "test_notebook")
    for cell_id, source, after in cells:
        add_cell_to_notebook(nb, cell_id, after)
        write_cell(nb, cell_id, source)
    session = NotebookSession(parse_notebook(nb), nb)
    # Run the harness + pytest under the interpreter that has pytest + the
    # notebook extra, instead of bare "python" off PATH.
    session.venv_python = Path(sys.executable)
    return session


@pytest.mark.asyncio
async def test_run_cell_tests_pass_and_fail():
    session = _session_with([("cell1", "def add(a, b):\n    return a + b\n", None)])
    executor = CellExecutor(session)

    result = await executor.run_cell_tests(
        "cell1",
        "def test_pass(cell):\n    assert cell.add(1, 2) == 3\n"
        "def test_fail(cell):\n    assert cell.add(1, 2) == 5\n",
    )

    assert result.passed == 1
    assert result.failed == 1
    assert result.errored == 0
    assert result.cell_source_hash
    assert result.test_source_hash
    assert result.ran_at > 0
    # The failing assertion carries the rewritten diff.
    fail = next(t for t in result.tests if t.outcome == "failed")
    assert "assert 3 == 5" in fail.message


@pytest.mark.asyncio
async def test_run_cell_tests_persists_and_writes_test_file():
    session = _session_with([("cell1", "def add(a, b):\n    return a + b\n", None)])
    executor = CellExecutor(session)

    await executor.run_cell_tests("cell1", "def test_ok(cell):\n    assert cell.add(2, 2) == 4\n")

    # Persisted to runtime state...
    reloaded = load_runtime_state(session.path)
    assert reloaded.cells["cell1"].test_result["passed"] == 1
    # ...and surfaced on the in-memory cell.
    cell = session.notebook_state.get_cell("cell1")
    assert cell is not None and cell.test_result is not None
    assert cell.test_result.passed == 1


@pytest.mark.asyncio
async def test_run_cell_tests_injects_upstream_inputs():
    # cell_b references `factor` from cell_a: run_cell_tests must materialize
    # cell_a and inject its `factor` artifact as a test input.
    session = _session_with(
        [
            ("cell_a", "factor = 10\n", None),
            ("cell_b", "def scale(x):\n    return x * factor\n", "cell_a"),
        ]
    )
    executor = CellExecutor(session)

    result = await executor.run_cell_tests(
        "cell_b",
        "def test_scale(cell):\n    assert cell.scale(3) == 30\n",
    )

    assert result.passed == 1
    assert result.failed == 0
    assert result.input_fingerprint  # an upstream input was fingerprinted


@pytest.mark.asyncio
async def test_run_cell_tests_gets_fetch_mount_and_env_inputs(monkeypatch, tmp_path):
    """A test sees what a run of the cell sees: fetched files, mounts and notebook env."""
    from types import SimpleNamespace

    from tests.notebook.test_fetch import _Origin

    monkeypatch.setattr(
        "strata.server._state",
        SimpleNamespace(
            config=SimpleNamespace(
                deployment_mode="personal", notebook_fetch_allowed_hosts=["127.0.0.1"]
            )
        ),
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "rows.txt").write_text("a\nb\n")
    origin = _Origin()
    try:
        source = (
            f"# @fetch zones {origin.url()}\n"
            f"# @mount data {data_dir.as_uri()} ro\n"
            "# @env MODE=prod\n"
            "import os\n"
            "def summary():\n"
            "    rows = len(zones.read_text().splitlines())\n"
            "    text = (data / 'rows.txt').read_text()\n"
            "    return (rows, text, os.environ['MODE'], os.environ['TOKEN'])\n"
        )
        session = _session_with([("cell1", source, None)])
        session.notebook_state.get_cell("cell1").env = {"TOKEN": "t0"}
        expected = (len(origin.body.splitlines()), "a\nb\n", "prod", "t0")

        result = await CellExecutor(session).run_cell_tests(
            "cell1",
            f"def test_inputs(cell):\n    assert cell.summary() == {expected!r}\n",
        )
    finally:
        origin.close()

    assert (result.passed, result.failed, result.errored) == (1, 0, 0), result.tests


@pytest.mark.asyncio
async def test_run_cell_tests_gets_dataset_inputs(notebook_personal_server):
    """A ``@dataset`` name is bound in the test as in a run of the cell."""
    from tests.notebook.test_dataset import _json_version

    registry = notebook_personal_server["artifact_store"]
    registry.set_name("taxi/model", "taxi-model", _json_version(registry, {"t": 3}))
    source = "# @dataset model taxi/model\ndef score():\n    return model['t']\n"
    session = _session_with([("cell1", source, None)])

    result = await CellExecutor(session).run_cell_tests(
        "cell1", "def test_score(cell):\n    assert cell.score() == 3\n"
    )

    assert (result.passed, result.failed, result.errored) == (1, 0, 0), result.tests


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="pyiceberg local paths on Windows")
async def test_run_cell_tests_gets_table_inputs(tmp_path):
    """A ``@table`` binds its URI and snapshot id in the test as in a run of the cell."""
    import pyarrow as pa
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    catalog = SqlCatalog(
        "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=str(warehouse)
    )
    catalog.create_namespace("db")
    table = catalog.create_table(
        "db.events", Schema(NestedField(1, "id", LongType(), required=False))
    )
    table.append(pa.table({"id": pa.array([1, 2], type=pa.int64())}))
    snapshot_id = table.current_snapshot().snapshot_id
    uri = f"file://{warehouse}#db.events"
    session = _session_with(
        [("cell1", f"# @table events {uri}\ndef read():\n    return events_snapshot\n", None)]
    )

    result = await CellExecutor(session).run_cell_tests(
        "cell1",
        f"def test_read(cell):\n    assert cell.read() == {snapshot_id}\n"
        f"    assert cell.events == {uri!r}\n",
    )

    assert (result.passed, result.failed, result.errored) == (1, 0, 0), result.tests


@pytest.mark.asyncio
async def test_a_changed_mount_or_env_marks_the_test_stale(tmp_path):
    """The stale flag covers the mount and env inputs a test reads, not only upstreams."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "rows.txt").write_text("a\n")
    source = (
        f"# @mount data {data_dir.as_uri()} ro\n"
        "import os\n"
        "def summary():\n"
        "    return ((data / 'rows.txt').read_text(), os.environ['TOKEN'])\n"
    )
    session = _session_with([("cell1", source, None)])
    cell = session.notebook_state.get_cell("cell1")
    cell.env = {"TOKEN": "t0"}
    result = await CellExecutor(session).run_cell_tests(
        "cell1", "def test_summary(cell):\n    assert cell.summary() == ('a\\n', 't0')\n"
    )
    assert result.passed == 1, result.tests

    def stale() -> bool:
        # The inputs are read by the staleness pass every edit triggers, not by serializing.
        session.compute_staleness()
        return session.serialize_cell(cell)["test_result"]["stale"]

    assert stale() is False
    cell.env = {"TOKEN": "t1"}
    assert stale() is True
    cell.env = {"TOKEN": "t0"}
    assert stale() is False
    # Local mounts fingerprint sizes and mtimes.
    (data_dir / "rows.txt").write_text("a\nb\n")
    assert stale() is True


@pytest.mark.asyncio
@pytest.mark.skipif(sys.platform == "win32", reason="pyiceberg local paths on Windows")
async def test_a_passing_test_over_an_empty_table_stays_fresh(tmp_path, monkeypatch):
    """An empty table is keyed at random for caching; the test flag must not inherit that.

    Serializing the cell (every ``GET /cells``, from async routes) must not reach the catalog.
    """
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import LongType, NestedField

    from strata.notebook import tables

    warehouse = tmp_path / "warehouse"
    warehouse.mkdir()
    catalog = SqlCatalog(
        "strata", uri=f"sqlite:///{warehouse / 'catalog.db'}", warehouse=str(warehouse)
    )
    catalog.create_namespace("db")
    catalog.create_table("db.events", Schema(NestedField(1, "id", LongType(), required=False)))
    uri = f"file://{warehouse}#db.events"
    session = _session_with(
        [("cell1", f"# @table events {uri}\ndef read():\n    return events_snapshot\n", None)]
    )
    result = await CellExecutor(session).run_cell_tests(
        "cell1", "def test_read(cell):\n    assert cell.read() is None\n"
    )
    assert result.passed == 1, result.tests

    resolves: list[object] = []
    resolve = tables.resolve_table_snapshot
    monkeypatch.setattr(
        tables, "resolve_table_snapshot", lambda *a, **k: resolves.append(a) or resolve(*a, **k)
    )
    cell = session.notebook_state.get_cell("cell1")
    assert session.serialize_cell(cell)["test_result"]["stale"] is False
    session.serialize_cells()
    assert resolves == [], "serializing a cell read the catalog"

    session.compute_staleness()
    assert resolves, "the staleness pass is what reads the catalog"
    assert session.serialize_cell(cell)["test_result"]["stale"] is False


@pytest.mark.asyncio
async def test_the_stale_flag_is_computed_on_reopen():
    """Editing the cell, its tests or an upstream outside the editor shows stale after reopen."""
    session = _session_with(
        [
            ("cell_a", "factor = 10\n", None),
            ("cell_b", "def scale(x):\n    return x * factor\n", "cell_a"),
        ]
    )
    test_source = "def test_scale(cell):\n    assert cell.scale(3) == 30\n"
    write_cell_tests(session.path, "cell_b", test_source)
    await CellExecutor(session).run_cell_tests("cell_b", test_source)

    def reopened() -> NotebookSession:
        fresh = NotebookSession(parse_notebook(session.path), session.path)
        fresh.venv_python = Path(sys.executable)
        fresh.compute_staleness()  # as opening does
        return fresh

    def stale(s: NotebookSession) -> bool:
        return s.serialize_cell(s.notebook_state.get_cell("cell_b"))["test_result"]["stale"]

    assert stale(reopened()) is False

    write_cell(session.path, "cell_b", "def scale(x):\n    return x * factor * 1\n")
    assert stale(reopened()) is True
    write_cell(session.path, "cell_b", "def scale(x):\n    return x * factor\n")
    assert stale(reopened()) is False

    write_cell_tests(session.path, "cell_b", test_source + "# more\n")
    assert stale(reopened()) is True
    write_cell_tests(session.path, "cell_b", test_source)

    write_cell(session.path, "cell_a", "factor = 11\n")
    upstream_moved = reopened()
    result = await CellExecutor(upstream_moved).execute_cell("cell_a", "factor = 11\n")
    assert result.success, result.error
    assert stale(upstream_moved) is True


@pytest.mark.asyncio
async def test_run_cell_tests_auto_provisions_pytest_then_retries(monkeypatch):
    """A missing pytest auto-installs into the dev group and the run retries once."""
    from strata.notebook import cell_test_runner, dependencies
    from strata.notebook.dependencies import DependencyChangeResult

    session = _session_with([("cell1", "def add(a, b):\n    return a + b\n", None)])
    executor = CellExecutor(session)

    calls = {"n": 0}
    passed_run = {
        "passed": 1,
        "failed": 0,
        "errored": 0,
        "skipped": 0,
        "tests": [
            {
                "name": "test_ok",
                "nodeid": "test_cell.py::test_ok",
                "outcome": "passed",
                "message": "",
            }
        ],
    }

    def fake_run(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise cell_test_runner.PytestUnavailableError("pytest missing")
        return passed_run

    installs = {"n": 0}

    def fake_ensure(notebook_dir, tool, *, timeout=120):
        installs["n"] += 1
        assert tool == "pytest"
        return DependencyChangeResult(success=True, package=tool, action="add")

    monkeypatch.setattr(cell_test_runner, "run_cell_tests_in_dir", fake_run)
    monkeypatch.setattr(dependencies, "ensure_dev_tool", fake_ensure)

    result = await executor.run_cell_tests("cell1", "def test_ok(cell):\n    assert True\n")

    assert installs["n"] == 1  # provisioned exactly once
    assert calls["n"] == 2  # retried after the install
    assert result.auto_installed == ["pytest"]
    assert result.pytest_unavailable is False
    assert result.passed == 1


@pytest.mark.asyncio
async def test_run_cell_tests_auto_provision_failure_surfaces_unavailable(monkeypatch):
    """If the auto-install fails, fall back to the actionable pytest_unavailable flag."""
    from strata.notebook import cell_test_runner, dependencies
    from strata.notebook.dependencies import DependencyChangeResult

    session = _session_with([("cell1", "def add(a, b):\n    return a + b\n", None)])
    executor = CellExecutor(session)

    calls = {"n": 0}

    def fake_run(**_kwargs):
        calls["n"] += 1
        raise cell_test_runner.PytestUnavailableError("pytest missing")

    def fake_ensure(notebook_dir, tool, *, timeout=120):
        return DependencyChangeResult(success=False, package=tool, action="add", error="boom")

    monkeypatch.setattr(cell_test_runner, "run_cell_tests_in_dir", fake_run)
    monkeypatch.setattr(dependencies, "ensure_dev_tool", fake_ensure)

    result = await executor.run_cell_tests("cell1", "def test_ok(cell):\n    assert True\n")

    assert calls["n"] == 1  # no retry when the install fails
    assert result.pytest_unavailable is True
    assert result.auto_installed == []


# WebSocket handler


def _register_fake_ws(session: NotebookSession):
    from strata.notebook.ws import _ensure_execution_state, _notebook_connections

    fake = FakeNotebookWebSocket()
    _notebook_connections.setdefault(session.id, []).append(cast(WebSocket, fake))
    return fake, _ensure_execution_state(session.id)


@pytest.mark.asyncio
async def test_handle_cell_run_tests_broadcasts_status_and_results():
    from strata.notebook.ws import _handle_cell_run_tests

    session = _session_with([("cell1", "def add(a, b):\n    return a + b\n", None)])
    fake, execution_state = _register_fake_ws(session)

    await _handle_cell_run_tests(
        websocket=cast(WebSocket, fake),
        session=session,
        payload={
            "cell_id": "cell1",
            "test_source": "def test_pass(cell):\n    assert cell.add(1, 2) == 3\n",
        },
        execution_state=execution_state,
        notebook_id=session.id,
    )

    statuses = [f["payload"]["status"] for f in fake.frames_of("cell_test_status")]
    assert "running" in statuses
    assert statuses[-1] == "ready"

    results = fake.frames_of("cell_test_results")
    assert results
    payload = results[-1]["payload"]
    assert payload["cell_id"] == "cell1"
    assert payload["passed"] == 1
    assert payload["stale"] is False

    # Test source was committed to the sibling file.
    assert (session.path / "cells" / "cell1.test.py").exists()


@pytest.mark.asyncio
async def test_handle_cell_run_tests_failure_reports_error_status():
    from strata.notebook.ws import _handle_cell_run_tests

    session = _session_with([("cell1", "def add(a, b):\n    return a + b\n", None)])
    fake, execution_state = _register_fake_ws(session)

    await _handle_cell_run_tests(
        websocket=cast(WebSocket, fake),
        session=session,
        payload={
            "cell_id": "cell1",
            "test_source": "def test_fail(cell):\n    assert cell.add(1, 2) == 99\n",
        },
        execution_state=execution_state,
        notebook_id=session.id,
    )

    statuses = [f["payload"]["status"] for f in fake.frames_of("cell_test_status")]
    assert statuses[-1] == "error"
    assert fake.frames_of("cell_test_results")[-1]["payload"]["failed"] == 1


@pytest.mark.asyncio
async def test_handle_cell_run_tests_refuses_while_the_environment_is_not_ready():
    """Tests run the cell's code, so they wait for the environment as a run does rather than
    use whatever ``python`` is on PATH.
    """
    from strata.notebook.ws import _handle_cell_run_tests

    session = _session_with([("cell1", "def add(a, b):\n    return a + b\n", None)])
    session.venv_python = None
    fake, execution_state = _register_fake_ws(session)

    await _handle_cell_run_tests(
        websocket=cast(WebSocket, fake),
        session=session,
        payload={"cell_id": "cell1", "test_source": "def test_x(cell):\n    assert True\n"},
        execution_state=execution_state,
        notebook_id=session.id,
    )

    errors = fake.frames_of("error")
    assert errors[-1]["payload"]["code"] == "ENVIRONMENT_BUSY"
    assert "not ready" in errors[-1]["payload"]["error"]
    assert not fake.frames_of("cell_test_status")
    assert not fake.frames_of("cell_test_results")


@pytest.mark.asyncio
async def test_handle_cell_run_tests_rejects_non_python_cell():
    from strata.notebook.ws import _handle_cell_run_tests

    tmp = Path(tempfile.mkdtemp())
    nb = create_notebook(tmp, "test_notebook")
    add_cell_to_notebook(nb, "md1", None, language="markdown")
    session = NotebookSession(parse_notebook(nb), nb)
    session.venv_python = Path(sys.executable)
    fake, execution_state = _register_fake_ws(session)

    await _handle_cell_run_tests(
        websocket=cast(WebSocket, fake),
        session=session,
        payload={"cell_id": "md1", "test_source": "def test_x(cell): assert True\n"},
        execution_state=execution_state,
        notebook_id=session.id,
    )

    errors = fake.frames_of("error")
    assert errors
    assert "Python" in errors[-1]["payload"]["error"]
    assert not fake.frames_of("cell_test_results")
