"""One trace per remote cell run, across the server and the worker. Item 52.

The manifest carried no trace context, so a worker's spans started a trace of
their own and the two halves were matched by build id by hand. Spans are
collected in memory here; the worker runs in this process, so both sides
export to the same place.
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
    """A pool between the server and the worker forwards its own span in the
    headers while the manifest still holds the server's. The worker's span
    belongs under the pool's."""
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
    """The helpers run on every dispatch, so switched off they contribute no
    header and no span, whatever the caller sent."""
    monkeypatch.setenv("STRATA_TRACING_ENABLED", "false")

    assert tracing.current_trace_context() == {}
    with tracing.trace_span_from(
        "worker.execute", {"traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"}
    ) as span:
        assert isinstance(span, tracing.NoOpSpan)


async def test_through_a_pool_the_order_is_dispatch_then_pool_then_worker(spans, monkeypatch):
    """The chain a pooled remote cell takes: the dispatch's context reaches
    the pool as headers and in the manifest, the pool opens its span, and the
    worker runs under the pool's."""
    import json

    import httpx
    from strata_pool import MachineType, Pool, PoolStore
    from strata_pool.backend import ProvisionedWorker

    from strata.notebook.remote_executor import (
        NOTEBOOK_EXECUTOR_MANIFEST_VERSION,
        NOTEBOOK_EXECUTOR_TRANSFORM_REF,
        create_notebook_executor_app,
    )

    class Backend:
        name = "inline"

        async def start(self, spec, env=None):
            return ProvisionedWorker(backend_id="m1", endpoint="http://worker", region="here")

        async def stop(self, backend_id):
            return None

        async def health(self, endpoint):
            return True

    monkeypatch.setenv("STRATA_WORKER_ALLOW_LOCAL_HOSTS", "1")
    store_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Store)
    threading.Thread(target=store_server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{store_server.server_address[1]}"
    worker = httpx.AsyncClient(transport=httpx.ASGITransport(app=create_notebook_executor_app()))
    pool = Pool(
        PoolStore(":memory:"),
        Backend(),
        [MachineType(name="cpu", image="w")],
        client=worker,
        health_poll_seconds=0,
        tracer=tracing.get_tracer(),
    )

    try:
        with tracing.trace_span("notebook.dispatch", cell_id="c1"):
            context = tracing.current_trace_context()
            manifest = {
                "schema_version": NOTEBOOK_EXECUTOR_MANIFEST_VERSION,
                "metadata": {
                    "executor_ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
                    "params": {"source": "x = 1", "input_specs": {}, "mounts": [], "env": {}},
                    "cell_id": "c1",
                    **context,
                },
                "inputs": [],
                "output": {"url": f"{base}/upload"},
                "finalize_url": f"{base}/finalize",
            }
            job = await pool.submit(
                tenant_id="acme",
                machine_type="cpu",
                payload=json.dumps(manifest).encode(),
                trace_context=context,
            )
        done = await pool.wait(job.id, timeout=60)
    finally:
        await pool.aclose()
        await worker.aclose()
        store_server.shutdown()

    assert done.error is None, done.error
    dispatch = _one(spans, "notebook.dispatch")
    pooled = _one(spans, "pool.execute")
    execution = _one(spans, "worker.execute")
    assert pooled.parent.span_id == dispatch.context.span_id
    assert execution.parent.span_id == pooled.context.span_id
    assert {s.context.trace_id for s in (dispatch, pooled, execution)} == {
        dispatch.context.trace_id
    }
    assert pooled.attributes["job_id"] == job.id
