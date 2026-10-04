"""One trace per remote cell run, across the server and the worker.

Spans are collected in memory; the worker runs in this process, so both sides export to the same
place.
"""

from __future__ import annotations

import http.server
import threading

import pytest

pytest.importorskip("opentelemetry.sdk")

from fastapi.testclient import TestClient  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

import strata.tracing as tracing  # noqa: E402


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.delenv("STRATA_TRACING_ENABLED", raising=False)
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("test"))
    monkeypatch.setattr(tracing, "_initialized", True)
    return exporter


def _one(exporter, name):
    (span,) = [s for s in exporter.get_finished_spans() if s.name == name]
    return span


@pytest.mark.parametrize("transport", ["direct", "signed"])
async def test_the_workers_execution_is_a_child_of_the_servers_dispatch(
    tmp_path, spans, transport, notebook_executor_server, notebook_personal_server
):
    from strata.notebook.executor import CellExecutor
    from strata.notebook.models import WorkerBackendType, WorkerSpec
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    nb = create_notebook(tmp_path / "nb", "Traced")
    add_cell_to_notebook(nb, "c1", None)
    write_cell(nb, "c1", "x = 1")
    session = NotebookSession(parse_notebook(nb), nb)
    config = {"url": notebook_executor_server["execute_url"], "transport": transport}
    if transport == "signed":
        config["strata_url"] = notebook_personal_server["base_url"]
    session.notebook_state.workers = [
        WorkerSpec(name="w", backend=WorkerBackendType.EXECUTOR, runtime_id="w", config=config)
    ]
    session.notebook_state.worker = "w"

    result = await CellExecutor(session).execute_cell("c1", "x = 1")

    assert result.success, result.error
    dispatch = _one(spans, "notebook.dispatch")
    execution = _one(spans, "worker.execute")
    assert execution.context.trace_id == dispatch.context.trace_id
    assert execution.parent.span_id == dispatch.context.span_id
    assert dispatch.attributes["cell_id"] == "c1"
    assert dispatch.attributes["notebook_id"] == session.notebook_state.id
    if transport == "signed":
        # The manifest names the cell, so the worker's span can too.
        assert execution.attributes["cell_id"] == "c1"
        assert execution.attributes["build_id"] == result.remote_build_id


class _Store(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args):
        return None


def test_a_dispatchers_header_context_is_the_nearer_parent_than_the_manifests(spans, monkeypatch):
    """A pool forwards its own span in headers while the manifest holds the server's; the worker
    belongs under the pool's.
    """
    from strata.notebook.remote_executor import (
        NOTEBOOK_EXECUTOR_MANIFEST_VERSION,
        NOTEBOOK_EXECUTOR_TRANSFORM_REF,
        create_notebook_executor_app,
    )

    monkeypatch.setenv("STRATA_WORKER_ALLOW_LOCAL_HOSTS", "1")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Store)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    with tracing.trace_span("server"):
        server_context = tracing.current_trace_context()
    with tracing.trace_span("pool"):
        pool_context = tracing.current_trace_context()

    manifest = {
        "schema_version": NOTEBOOK_EXECUTOR_MANIFEST_VERSION,
        "metadata": {
            "executor_ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            "params": {"source": "x = 1", "input_specs": {}, "mounts": [], "env": {}},
            "cell_id": "c9",
            **server_context,
        },
        "inputs": [],
        "output": {"url": f"{base}/upload"},
        "finalize_url": f"{base}/finalize",
    }
    try:
        with TestClient(create_notebook_executor_app()) as client:
            response = client.post("/execute", json=manifest, headers=pool_context)
    finally:
        server.shutdown()

    assert response.status_code == 200, response.text
    execution = _one(spans, "worker.execute")
    assert execution.parent.span_id == _one(spans, "pool").context.span_id
    assert execution.attributes["cell_id"] == "c9"


def test_with_tracing_off_nothing_is_added_and_no_span_is_made(monkeypatch):
    """The helpers run on every dispatch, so when off they add no header and no span."""
    monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

    assert tracing.current_trace_context() == {}
    with tracing.trace_span_from(
        "worker.execute", {"traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"}
    ) as span:
        assert isinstance(span, tracing.NoOpSpan)


async def test_one_trace_runs_from_the_servers_dispatch_through_the_pool_to_the_worker(
    tmp_path, spans, monkeypatch, notebook_build_server
):
    """A service-mode server, the pool service and a worker together: the pool's queueing, boot
    and execution sit under the server's dispatch, and the worker's execution under the pool's.
    """
    import httpx
    import uvicorn
    from starlette.requests import Request
    from starlette.responses import Response
    from strata_pool import MachineType, Pool, PoolStore
    from strata_pool.api import create_app
    from strata_pool.backend import ProvisionedWorker

    from strata.notebook.executor import CellExecutor
    from strata.notebook.models import WorkerBackendType, WorkerSpec
    from strata.notebook.parser import parse_notebook
    from strata.notebook.remote_executor import create_notebook_executor_app
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
    from tests.conftest import find_free_port, wait_for_server

    class Backend:
        name = "inline"

        async def start(self, spec, env=None):
            return ProvisionedWorker(backend_id="m1", endpoint="http://worker", region="here")

        async def stop(self, backend_id):
            return None

        async def health(self, endpoint):
            return True

    monkeypatch.setenv("STRATA_WORKER_ALLOW_LOCAL_HOSTS", "1")
    worker = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_notebook_executor_app()))
    pool = Pool(
        PoolStore(tmp_path / "pool.sqlite"),
        Backend(),
        [MachineType(name="cpu", image="w")],
        client=worker,
        health_poll_seconds=0,
        tracer=tracing.get_tracer(),
    )
    app = create_app(pool, api_token="t", scaler_interval_seconds=3600)

    # Stands in for the dispatcher above the server and the pool: it picks the machine type and
    # tenant, and forwards the body and the trace headers.
    async def dispatch(request: Request) -> Response:
        headers = {"Authorization": "Bearer t", "X-Strata-Tenant": "acme"}
        headers |= {k: v for k in ("traceparent", "tracestate") if (v := request.headers.get(k))}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://pool", timeout=60
        ) as client:
            reply = await client.post(
                "/v1/jobs/sync",
                params={"machine_type": "cpu"},
                content=await request.body(),
                headers=headers,
            )
        return Response(reply.content, status_code=reply.status_code, media_type="application/json")

    # A plain route: the module's postponed annotations would turn FastAPI's `request` into a query.
    app.add_route("/v1/execute-manifest", dispatch, methods=["POST"])

    port = find_free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    assert wait_for_server(port), "the pool service did not start"

    nb = create_notebook(tmp_path / "nb", "Pooled")
    add_cell_to_notebook(nb, "c1", None)
    write_cell(nb, "c1", "x = 1")
    session = NotebookSession(parse_notebook(nb), nb)
    config = {
        "url": f"http://127.0.0.1:{port}/v1/execute",
        "transport": "signed",
        "strata_url": notebook_build_server["base_url"],
    }
    notebook_build_server["config"].transforms_config["notebook_workers"] = [
        {"name": "pool", "backend": "executor", "runtime_id": "pool-cpu", "config": config}
    ]
    session.notebook_state.workers = [
        WorkerSpec(
            name="pool", backend=WorkerBackendType.EXECUTOR, runtime_id="pool-cpu", config=config
        )
    ]
    session.notebook_state.worker = "pool"

    try:
        result = await CellExecutor(session).execute_cell("c1", "x = 1")
    finally:
        server.should_exit = True
        thread.join(timeout=5.0)

    assert result.success, result.error
    assert result.remote_transport == "signed"
    dispatch_span = _one(spans, "notebook.dispatch")
    queued, boot, pooled, execution = (
        _one(spans, name) for name in ("pool.queue", "pool.boot", "pool.execute", "worker.execute")
    )
    assert {s.context.trace_id for s in (queued, boot, pooled, execution)} == {
        dispatch_span.context.trace_id
    }
    for span in (queued, boot, pooled):
        assert span.parent.span_id == dispatch_span.context.span_id
    assert execution.parent.span_id == pooled.context.span_id
    assert execution.attributes["cell_id"] == "c1"
    assert execution.attributes["build_id"] == result.remote_build_id
