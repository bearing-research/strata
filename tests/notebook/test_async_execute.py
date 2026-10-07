"""A worker that accepts a job and runs it later.

A worker, or a pool in front of one, may answer 202 with a job to poll: the wait for it to
start is bounded by the provisioning deadline, and the cell's timeout starts when it runs.
The facade stands in for a pool and runs the request on a real worker at ``finished``: a
manifest's result is the worker's JSON, a direct request's is its output bundle. A fake clock
advances on every status read, so deadlines are exercised without waiting.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading

import httpx
import pytest

import strata.notebook.executor as executor_module


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Facade:
    """Accepts the manifest with 202, walks a script of job states, one per
    status read, advancing the clock by ``step`` each read."""

    def __init__(self, clock: _Clock, script: list[str], worker_url: str, step: float = 10.0):
        self.clock = clock
        self.script = list(script)
        self.worker_url = worker_url
        self.step = step
        self.manifest: dict | None = None
        self.direct: tuple[bytes, str] | None = None
        self.cancelled: list[str] = []
        facade = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _json(self, status: int, payload: dict) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.path.endswith("/cancel"):
                    facade.cancelled.append(self.path)
                    self._json(200, {"cancelled": True})
                    return
                if self.headers["Content-Type"].startswith("multipart/form-data"):
                    facade.direct = (raw, self.headers["Content-Type"])
                else:
                    facade.manifest = json.loads(raw)
                self._json(202, {"job_url": "/jobs/1", "provisioning_deadline": 600})

            def do_GET(self):  # noqa: N802
                facade.clock.now += facade.step
                state = facade.script.pop(0) if len(facade.script) > 1 else facade.script[0]
                if state == "failed":
                    self._json(
                        200,
                        {"state": "failed", "status_code": 502, "error": "the machine was lost"},
                    )
                    return
                if state == "finished-without-bundle":
                    self._json(200, {"state": "finished", "status_code": 200, "result": {}})
                    return
                if state != "finished":
                    self._json(200, {"state": state})
                    return
                if facade.direct is not None:
                    raw, content_type = facade.direct
                    worker = httpx.post(
                        facade.worker_url,
                        content=raw,
                        headers={"Content-Type": content_type, "X-Strata-Executor-Protocol": "v1"},
                        timeout=120,
                    )
                    self.send_response(worker.status_code)
                    for name in (
                        "Content-Type",
                        "X-Strata-Executor-Protocol",
                        "X-Strata-Notebook-Executor-Protocol",
                    ):
                        self.send_header(name, worker.headers[name])
                    self.send_header("Content-Length", str(len(worker.content)))
                    self.end_headers()
                    self.wfile.write(worker.content)
                    return
                worker = httpx.post(facade.worker_url, json=facade.manifest, timeout=120)
                self._json(
                    200,
                    {
                        "state": "finished",
                        "status_code": worker.status_code,
                        "result": worker.json(),
                    },
                )

            def log_message(self, *args):
                return None

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"


@pytest.fixture
def run_cell(tmp_path, monkeypatch, notebook_executor_server, notebook_personal_server):
    from strata.notebook.executor import CellExecutor
    from strata.notebook.models import WorkerBackendType, WorkerSpec
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    clock = _Clock()
    monkeypatch.setattr(executor_module, "_monotonic", clock)
    monkeypatch.setattr(executor_module, "_JOB_POLL_SECONDS", 0)
    phases: list[str] = []
    cancel_at: list[str] = []

    async def record_phase(self, cell_id, worker_spec, phase):
        phases.append(phase)
        if phase in cancel_at:
            asyncio.current_task().cancel()

    monkeypatch.setattr(CellExecutor, "_broadcast_remote_phase", record_phase)
    facades: list[_Facade] = []

    async def _run(
        script: list[str],
        *,
        provisioning_limit: float = 600.0,
        transport: str = "signed",
        cancel_at_phase: str | None = None,
    ):
        notebook_personal_server["config"].worker_provisioning_timeout_seconds = provisioning_limit
        worker_url = notebook_executor_server[
            "execute_url" if transport == "direct" else "manifest_execute_url"
        ]
        facade = _Facade(clock, script, worker_url)
        facades.append(facade)
        source = "# @timeout 30\nx = 41 + 1"
        nb = create_notebook(tmp_path / f"nb{len(facades)}", "Async")
        add_cell_to_notebook(nb, "c1", None)
        write_cell(nb, "c1", source)
        session = NotebookSession(parse_notebook(nb), nb)
        session.notebook_state.workers = [
            WorkerSpec(
                name="pool",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="pool",
                config={
                    "url": f"{facade.base}/v1/execute",
                    "transport": transport,
                    "strata_url": notebook_personal_server["base_url"],
                },
            )
        ]
        session.notebook_state.worker = "pool"
        if cancel_at_phase is None:
            result = await CellExecutor(session).execute_cell("c1", source)
            return result, facade, phases
        cancel_at.append(cancel_at_phase)
        with pytest.raises(asyncio.CancelledError):
            await CellExecutor(session).execute_cell("c1", source)
        asyncio.current_task().uncancel()
        return None, facade, phases

    yield _run
    for facade in facades:
        facade.server.shutdown()


async def test_provisioning_does_not_spend_the_cells_timeout(run_cell):
    """50 s provisioning, 20 s running, a 30 s cell timeout: succeeds."""
    result, facade, phases = await run_cell(["provisioning"] * 5 + ["running"] * 2 + ["finished"])

    assert result.success, result.error
    assert result.outputs["x"]["preview"] == 42
    assert result.remote_build_state == "ready"
    assert phases == ["starting", "running"]
    assert facade.cancelled == []


async def test_a_job_that_never_starts_fails_at_the_provisioning_deadline_and_is_cancelled(
    run_cell,
):
    result, facade, _ = await run_cell(["provisioning"], provisioning_limit=40)

    assert result.success is False
    assert result.remote_error_code == "PROVISIONING_TIMEOUT"
    assert "STRATA_WORKER_PROVISIONING_TIMEOUT_SECONDS" in result.error
    assert len(facade.cancelled) == 1


async def test_the_cells_timeout_starts_when_the_job_runs(run_cell):
    """Provisioning well inside its limit, then 40 s running against a 30 s cell
    timeout: the running time is what times out."""
    result, facade, _ = await run_cell(["queued"] * 2 + ["running"])

    assert result.success is False
    assert result.remote_error_code == "TIMEOUT"
    assert len(facade.cancelled) == 1


async def test_a_direct_cell_survives_a_boot_longer_than_its_timeout(run_cell):
    """The direct transport follows the job URL too: 50 s provisioning against a
    30 s cell timeout, then the job URL answers the worker's output bundle."""
    result, facade, phases = await run_cell(
        ["provisioning"] * 5 + ["running", "finished"], transport="direct"
    )

    assert result.success, result.error
    assert result.outputs["x"]["preview"] == 42
    assert phases == ["starting", "running"]
    assert facade.direct is not None and facade.manifest is None
    assert facade.cancelled == []


async def test_a_failed_direct_job_fails_the_cell_with_the_jobs_error(run_cell):
    result, facade, _ = await run_cell(["provisioning", "failed"], transport="direct")

    assert result.success is False
    assert "the machine was lost" in result.error
    assert facade.cancelled == []


async def test_a_direct_job_that_never_starts_is_cancelled_by_its_build_id(run_cell):
    result, facade, _ = await run_cell(["provisioning"], provisioning_limit=40, transport="direct")

    assert result.success is False
    assert result.remote_error_code == "PROVISIONING_TIMEOUT"
    assert facade.direct is not None
    metadata = facade.direct[0].split(b"\r\n\r\n", 1)[1].split(b"\r\n--", 1)[0]
    assert facade.cancelled == [f"/v1/executions/{json.loads(metadata)['build_id']}/cancel"]


async def test_a_direct_job_finished_without_its_bundle_is_a_protocol_error(run_cell):
    result, _, _ = await run_cell(["running", "finished-without-bundle"], transport="direct")

    assert result.success is False
    assert result.remote_error_code == "PROTOCOL_ERROR"


async def test_cancelling_a_direct_cell_while_polling_cancels_the_job(run_cell):
    _, facade, _ = await run_cell(
        ["provisioning"], provisioning_limit=1e9, transport="direct", cancel_at_phase="starting"
    )

    assert len(facade.cancelled) == 1


async def test_the_remote_phase_reaches_the_sessions_sockets(tmp_path):
    """Sockets are keyed by the session id, which differs from the notebook.toml id."""
    from strata.notebook.executor import CellExecutor
    from strata.notebook.models import WorkerBackendType, WorkerSpec
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import create_notebook
    from strata.notebook.ws import _notebook_connections, forget_notebook_execution_state

    class _Socket:
        def __init__(self):
            self.sent: list[dict] = []

        async def send_text(self, text: str) -> None:
            self.sent.append(json.loads(text))

    nb = create_notebook(tmp_path / "nb", "Phases")
    session = NotebookSession(parse_notebook(nb), nb)
    assert session.id != session.notebook_state.id
    socket = _Socket()
    _notebook_connections[session.id] = [socket]
    worker = WorkerSpec(
        name="pool",
        backend=WorkerBackendType.EXECUTOR,
        config={"url": "https://pool.example/v1/execute", "transport": "signed"},
    )
    try:
        await CellExecutor(session)._broadcast_remote_phase("c1", worker, "starting")
    finally:
        _notebook_connections.pop(session.id, None)
        forget_notebook_execution_state(session.id)

    assert [frame["payload"] for frame in socket.sent] == [
        {
            "cell_id": "c1",
            "status": "running",
            "remote_worker": "pool",
            "remote_transport": "signed",
            "remote_build_state": "starting",
        }
    ]
