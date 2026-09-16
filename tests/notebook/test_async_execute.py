"""A worker that accepts a job and runs it later. Item 48.

The executor protocol was one synchronous request, so queueing, booting and
pulling an environment all spent the cell's own timeout. A worker, or a pool in
front of one, may now answer 202 with a job to poll: the wait for the job to
start is bounded by the provisioning deadline, and the cell's timeout starts
when it runs.

The facade here stands in for a pool: it accepts the manifest, reports scripted
states, and at ``finished`` runs the manifest on a real worker. Time is a fake
clock the facade advances on every status read, so deadlines are exercised
without waiting for them.
"""

from __future__ import annotations

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
                facade.manifest = json.loads(raw)
                self._json(202, {"job_url": "/jobs/1", "provisioning_deadline": 600})

            def do_GET(self):  # noqa: N802
                facade.clock.now += facade.step
                state = facade.script.pop(0) if len(facade.script) > 1 else facade.script[0]
                if state != "finished":
                    self._json(200, {"state": state})
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

    async def record_phase(self, cell_id, worker_spec, phase):
        phases.append(phase)

    monkeypatch.setattr(CellExecutor, "_broadcast_remote_phase", record_phase)
    facades: list[_Facade] = []

    async def _run(script: list[str], *, provisioning_limit: float = 600.0):
        notebook_personal_server["config"].worker_provisioning_timeout_seconds = provisioning_limit
        facade = _Facade(clock, script, notebook_executor_server["manifest_execute_url"])
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
                    "transport": "signed",
                    "strata_url": notebook_personal_server["base_url"],
                },
            )
        ]
        session.notebook_state.worker = "pool"
        result = await CellExecutor(session).execute_cell("c1", source)
        return result, facade, phases

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
