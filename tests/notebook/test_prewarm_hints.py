"""Pre-warm hints: the notebook tells a remote worker's dispatcher that a run is near."""

from __future__ import annotations

import http.server
import json
import threading
from pathlib import Path

import pytest

from strata.notebook import prewarm
from strata.notebook.models import WorkerBackendType, WorkerSpec
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from tests.notebook.test_cli import _build_notebook


class _Dispatcher:
    """A dispatcher in front of a pool: records hints, answers ``status`` to them."""

    def __init__(self, status: int = 202):
        self.hints: list[tuple[str, dict, dict]] = []
        dispatcher = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                dispatcher.hints.append((self.path, json.loads(body), dict(self.headers)))
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args):
                return None

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1/execute"

    def reasons(self) -> list[tuple[str, str]]:
        return [(body["worker"], body["reason"]) for _, body, _ in self.hints]


@pytest.fixture
def dispatcher():
    server = _Dispatcher()
    yield server
    server.server.shutdown()


@pytest.fixture(autouse=True)
def _fresh_rate_limit():
    prewarm._last_sent.clear()
    yield
    prewarm._last_sent.clear()


def _session(tmp_path: Path, url: str, cells=None) -> NotebookSession:
    nb = _build_notebook(
        tmp_path,
        cells=cells or [("up", "# @worker gpu\nx = 1", None), ("down", "y = x + 1", "up")],
    )
    session = NotebookSession(parse_notebook(nb), nb)
    session.notebook_state.workers = [
        WorkerSpec(
            name="gpu",
            backend=WorkerBackendType.EXECUTOR,
            runtime_id="gpu-a100",
            config={"url": url, "token": "dispatch-token"},
        )
    ]
    return session


@pytest.mark.parametrize(
    ("execute_url", "hints"),
    [
        ("http://pool.test/v1/execute", "http://pool.test/v1/hints"),
        ("http://pool.test/v1/execute-manifest?x=1", "http://pool.test/v1/hints"),
        ("http://pool.test/gpu/execute", "http://pool.test/gpu/v1/hints"),
        ("http://pool.test", "http://pool.test/v1/hints"),
    ],
)
def test_the_hints_route_sits_beside_the_execute_route(execute_url, hints):
    assert prewarm.hints_url(execute_url) == hints


@pytest.mark.asyncio
async def test_a_hint_names_the_worker_and_carries_its_token(tmp_path, dispatcher):
    session = _session(tmp_path, dispatcher.url)

    assert await prewarm.send_hint(session, "gpu", "focus")

    ((path, body, headers),) = dispatcher.hints
    assert path == "/v1/hints"
    assert body == {
        "worker": "gpu",
        "runtime_id": "gpu-a100",
        "notebook_id": session.notebook_state.id,
        "reason": "focus",
    }
    assert headers["Authorization"] == "Bearer dispatch-token"
    assert headers["X-Strata-Executor-Protocol"] == "v1"


@pytest.mark.asyncio
async def test_repeats_within_a_minute_send_nothing(tmp_path, dispatcher):
    session = _session(tmp_path, dispatcher.url)

    assert await prewarm.send_hint(session, "gpu", "focus")
    assert not await prewarm.send_hint(session, "gpu", "cascade")

    assert len(dispatcher.hints) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("worker", [None, "local", "nonexistent"])
async def test_no_hint_for_a_cell_that_runs_here_or_nowhere(tmp_path, dispatcher, worker):
    session = _session(tmp_path, dispatcher.url)

    assert not await prewarm.send_hint(session, worker, "focus")

    assert dispatcher.hints == []


@pytest.mark.asyncio
async def test_a_worker_without_a_hints_route_changes_nothing(tmp_path):
    """A plain strata-worker answers 404: the hint is dropped, nothing raises."""
    plain = _Dispatcher(status=404)
    try:
        session = _session(tmp_path, plain.url)
        assert not await prewarm.send_hint(session, "gpu", "focus")
    finally:
        plain.server.shutdown()

    unreachable = _session(tmp_path / "gone", "http://127.0.0.1:1/v1/execute")
    assert not await prewarm.send_hint(unreachable, "gpu", "focus")


@pytest.mark.asyncio
async def test_focusing_a_remote_cell_hints_its_worker(tmp_path, dispatcher):
    from strata.notebook.ws import _handle_cell_focus

    class Socket:
        async def send_text(self, text: str) -> None:
            return None

    session = _session(tmp_path, dispatcher.url)
    socket = Socket()
    session.presence.join(socket, "reviewer")

    await _handle_cell_focus(socket, session, {"cell_id": "down", "author": "reviewer"}, "nb")
    await prewarm.settled()
    assert dispatcher.hints == []  # a local cell: nothing to warm

    await _handle_cell_focus(socket, session, {"cell_id": "up", "author": "reviewer"}, "nb")
    await prewarm.settled()
    assert dispatcher.reasons() == [("gpu", "focus")]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["cascade", "run_all"])
async def test_a_cascade_or_run_all_hints_its_remote_workers(
    tmp_path, dispatcher, monkeypatch, entry
):
    from strata.notebook.cascade import CascadePlanner
    from strata.notebook.ws import NotebookExecutionState, _execute_cascade, _execute_run_all

    sent: list[tuple[list, str]] = []
    monkeypatch.setattr(
        prewarm,
        "hint_in_background",
        lambda session, workers, reason: sent.append((list(workers), reason)),
    )

    async def no_runs(self, cell_id, source, *args, **kwargs):
        raise RuntimeError("not running cells here")

    monkeypatch.setattr("strata.notebook.executor.CellExecutor.execute_cell", no_runs)
    monkeypatch.setattr("strata.notebook.executor.CellExecutor.execute_cell_rerun", no_runs)
    session = _session(tmp_path, dispatcher.url)

    if entry == "cascade":
        plan = CascadePlanner(session).plan("down")
        await _execute_cascade(None, session, plan, NotebookExecutionState(), session.id)
    else:
        await _execute_run_all(None, session, ["up", "down"], NotebookExecutionState(), session.id)

    workers, reason = sent[0]
    assert reason == entry
    assert "gpu" in workers


@pytest.mark.asyncio
async def test_picking_a_default_worker_hints_it(tmp_path, dispatcher, monkeypatch):
    from strata.notebook.routes import WorkerConfigRequest, update_notebook_worker_endpoint
    from strata.notebook.writer import update_notebook_workers

    sent: list[tuple[list, str]] = []
    monkeypatch.setattr(
        prewarm,
        "hint_in_background",
        lambda session, workers, reason: sent.append((list(workers), reason)),
    )
    session = _session(tmp_path, dispatcher.url)
    # Persisted, so the reload after the change keeps the worker defined.
    update_notebook_workers(session.path, session.notebook_state.workers)
    session.reload()

    await update_notebook_worker_endpoint("nb", session, WorkerConfigRequest(worker="gpu"))

    assert sent == [(["gpu"], "select")]
