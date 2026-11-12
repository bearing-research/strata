"""A served blob's reader is closed once, however its response ends.

A client can be gone before the body starts, leaving the chunk generator unstarted, or
mid-send, leaving it parked at a yield. Either way the reader is otherwise open until the
collector finds it. Azure's reader is a generator's context manager, which cannot be
exited without being entered, or exited twice.
"""

from __future__ import annotations

import hashlib
from contextlib import AsyncExitStack

import pyarrow as pa
import pytest
from httpx import ASGITransport, AsyncClient
from starlette.requests import ClientDisconnect
from vestigo.provenance import TransformSpec
from vestigo.store import get_artifact_store

from tests.conftest import table_to_ipc_bytes


class StrictReader:
    """A blob reader that refuses an exit without an enter, or a second exit."""

    def __init__(self, inner):
        self._inner = inner
        self.entered = 0
        self.exited = 0

    def __enter__(self):
        self.entered += 1
        return self._inner.__enter__()

    def __exit__(self, *exc_info):
        if self.exited or not self.entered:
            raise RuntimeError("exited unentered, or twice")
        self.exited += 1
        return self._inner.__exit__(*exc_info)


@pytest.fixture
def served(in_process_server, monkeypatch):
    import strata.server as server_module

    state = in_process_server()
    store = get_artifact_store(state.config.artifact_dir)
    readers: list[StrictReader] = []
    open_reader = store.open_blob_reader

    def strict(*args, **kwargs):
        reader = open_reader(*args, **kwargs)
        if reader is None:
            return None
        readers.append(StrictReader(reader))
        return readers[-1]

    monkeypatch.setattr(store, "open_blob_reader", strict)
    return server_module.app, state, store, readers


async def _get_from_a_client_gone_at(app, path: str, gone_at: str | None) -> None:
    """GET *path*; the connection fails at the *gone_at* message (``None``: it never does)."""

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        if message["type"] == gone_at:
            raise OSError("connection reset")

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"test")],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
    }
    try:
        # The router, not the app: the app's HTTP middleware reads the whole body from the
        # route before it sends any, so a client gone early would never reach the response.
        # FastAPI's own middleware would give the scope this stack.
        async with AsyncExitStack() as stack:
            scope["fastapi_middleware_astack"] = stack
            await app.router(scope, receive, send)
    except ClientDisconnect:
        assert gone_at is not None


_ENDINGS = [
    pytest.param(None, id="read-whole"),
    pytest.param("http.response.start", id="gone-before-the-body"),
    pytest.param("http.response.body", id="gone-mid-send"),
]


@pytest.mark.parametrize("gone_at", _ENDINGS)
async def test_a_scan_stream_closes_its_blob_once(served, temp_warehouse, gone_at):
    app, state, _, readers = served
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/v1/materialize",
            json={
                "inputs": [temp_warehouse["table_uri"]],
                "transform": {"executor": "scan@v1", "params": {}},
                "mode": "stream",
            },
        )
    assert response.status_code == 200, response.text

    await _get_from_a_client_gone_at(app, response.json()["stream_url"], gone_at)
    # Its expiry timer is on this test's loop, gone by the time the shared teardown runs.
    state.streams.shutdown_cleanups()

    # The build, run by this request, reads its blob back as well as the response.
    assert readers
    assert {(r.entered, r.exited) for r in readers} == {(1, 1)}


@pytest.mark.parametrize("gone_at", _ENDINGS)
async def test_an_artifact_download_closes_its_blob_once(served, gone_at):
    app, _, store, readers = served
    data = table_to_ipc_bytes(pa.table({"x": [1, 2, 3]}))
    version = store.create_artifact(
        artifact_id="stored",
        provenance_hash=hashlib.sha256(b"stored").hexdigest(),
        transform_spec=TransformSpec(executor="local@v1", params={}, inputs=[]),
    )
    store.write_blob("stored", version, data)
    store.finalize_artifact(
        artifact_id="stored",
        version=version,
        schema_json="",
        row_count=3,
        byte_size=len(data),
        content_sha256=hashlib.sha256(data).hexdigest(),
    )

    await _get_from_a_client_gone_at(app, f"/v1/artifacts/stored/v/{version}/data", gone_at)

    assert [(r.entered, r.exited) for r in readers] == [(1, 1)]
