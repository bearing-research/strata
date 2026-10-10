"""Artifact and publication routes make their metadata-store calls off the event loop.

Each test holds one request's store call on an event, checks that another request completes
meanwhile, then releases it and checks the first finished correctly.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from types import SimpleNamespace

import pyarrow as pa
import pytest
from httpx import ASGITransport, AsyncClient

from strata.artifact_store import TransformSpec, get_artifact_store, reset_artifact_store
from strata.config import StrataConfig
from tests.conftest import table_to_ipc_bytes

ARROW = table_to_ipc_bytes(pa.table({"x": [1, 2, 3]}))


class Held:
    """Wraps one store method so that its first call waits until ``release`` is set."""

    def __init__(self, store, method: str, monkeypatch):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.done = threading.Event()
        inner = getattr(store, method)

        def held(*args, **kwargs):
            if not self.entered.is_set():
                self.entered.set()
                # A guard, so a call stuck on the loop fails the test rather than hanging it.
                self.release.wait(timeout=30)
                self.done.set()
            return inner(*args, **kwargs)

        monkeypatch.setattr(store, method, held)


async def _another_request_completes_while_held(client: AsyncClient, held: Held, call):
    """Run *call* with its store call held: whether ``/health`` completed meanwhile, and the
    result of *call*."""
    task = asyncio.ensure_future(call)
    try:
        assert await asyncio.to_thread(held.entered.wait, 30), "the held call was never made"
        assert (await client.get("/health")).status_code == 200
        completed_while_held = not held.done.is_set()
    finally:
        held.release.set()
    return completed_while_held, await task


@pytest.fixture
def served(tmp_path):
    """An in-process personal server and its artifact store."""
    import strata.server as server_module
    from strata.server import ServerState
    from strata.tenant_registry import reset_tenant_registry

    config = StrataConfig(
        host="127.0.0.1",
        deployment_mode="personal",
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
        metadata_db=tmp_path / "meta.sqlite",
        rate_limit_enabled=False,
    )
    reset_artifact_store()
    reset_tenant_registry()
    original = server_module._state
    state = ServerState(config)
    server_module._state = state
    try:
        yield SimpleNamespace(store=get_artifact_store(config.artifact_dir), app=server_module.app)
    finally:
        state.streams.shutdown_cleanups()
        server_module._state = original
        reset_artifact_store()
        reset_tenant_registry()


def _client(served) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=served.app), base_url="http://test")


def _files(metadata: dict) -> dict:
    return {
        "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
        "data": ("data.arrow", ARROW, "application/vnd.apache.arrow.stream"),
    }


def _ready(store, artifact_id: str) -> int:
    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=hashlib.sha256(artifact_id.encode()).hexdigest(),
        transform_spec=TransformSpec(
            executor="notebook/cell@v1", params={"content_type": "arrow/ipc"}, inputs=[]
        ),
    )
    store.write_blob(artifact_id, version, ARROW)
    store.finalize_artifact(
        artifact_id=artifact_id,
        version=version,
        schema_json="",
        row_count=3,
        byte_size=len(ARROW),
        content_sha256=hashlib.sha256(ARROW).hexdigest(),
    )
    return version


_PUT = {"inputs": [], "transform": {"executor": "local@v1", "params": {}}, "name": "the-put"}


async def test_put_looks_up_provenance_off_the_loop(served, monkeypatch):
    held = Held(served.store, "find_by_provenance", monkeypatch)
    async with _client(served) as client:
        completed, response = await _another_request_completes_while_held(
            client, held, client.put("/v1/artifacts", files=_files(_PUT))
        )

    assert completed
    assert response.status_code == 200, response.text
    assert response.json()["name_uri"] == "strata://name/the-put"
    named = served.store.resolve_name("the-put")
    assert response.json()["artifact_uri"] == f"strata://artifact/{named.id}@v={named.version}"


async def test_put_by_provenance_looks_up_provenance_off_the_loop(served, monkeypatch):
    held = Held(served.store, "find_by_provenance", monkeypatch)
    async with _client(served) as client:
        completed, response = await _another_request_completes_while_held(
            client,
            held,
            client.put(
                f"/v1/artifacts/by-provenance/{'b' * 64}",
                files=_files({"content_type": "arrow/ipc"}),
            ),
        )

    assert completed
    assert response.status_code == 200, response.text
    assert served.store.find_by_provenance("b" * 64).byte_size == len(ARROW)


async def test_import_checks_the_id_off_the_loop(served, monkeypatch):
    held = Held(served.store, "id_tenant", monkeypatch)
    record = {"id": "imported", "version": 1, "provenance_hash": "c" * 64, "created_at": 1.0}
    async with _client(served) as client:
        completed, response = await _another_request_completes_while_held(
            client, held, client.post("/v1/artifacts/import", files=_files(record))
        )

    assert completed
    assert response.status_code == 200, response.text
    assert served.store.get_artifact("imported", 1).state == "ready"


@pytest.mark.parametrize(
    ("method", "path", "status"),
    [
        ("DELETE", "/v1/artifacts/held/v/1", 200),
        ("GET", "/v1/artifacts/held/v/1/data", 200),
        # Not building: the route answers from the row it read.
        ("POST", "/v1/artifacts/upload/held/v/1", 400),
    ],
)
async def test_artifact_routes_read_the_row_off_the_loop(served, monkeypatch, method, path, status):
    _ready(served.store, "held")
    held = Held(served.store, "get_artifact", monkeypatch)
    async with _client(served) as client:
        completed, response = await _another_request_completes_while_held(
            client, held, client.request(method, path, content=b"x" if method == "POST" else None)
        )

    assert completed
    assert response.status_code == status, response.text


@pytest.mark.parametrize("suffix", ["", "/data", "/embed", "/verify", "/archive.zip"])
async def test_public_routes_resolve_the_token_off_the_loop(served, monkeypatch, suffix):
    store = served.store
    version = _ready(store, "published")
    token = store.publish_artifact("published", version).token
    held = Held(store, "get_publication", monkeypatch)
    async with _client(served) as client:
        completed, response = await _another_request_completes_while_held(
            client, held, client.get(f"/p/{token}{suffix}")
        )

    assert completed
    assert response.status_code == 200, response.text
    if suffix == "/data":
        assert response.content == ARROW
    if suffix == "/verify":
        assert response.json()["matches"] is True


async def test_a_named_put_hit_superseded_before_it_is_named_names_the_rebuild(served, monkeypatch):
    """A refresh supersedes the hit between the lookup and the name: the name lands on the
    rebuild, never silently nowhere."""
    store = served.store
    unnamed = {k: v for k, v in _PUT.items() if k != "name"}
    async with _client(served) as client:
        first = await client.put("/v1/artifacts", files=_files(unnamed))
        assert first.status_code == 200, first.text
        ref = first.json()["artifact_uri"].removeprefix("strata://artifact/")
        artifact_id, version = ref.split("@v=")
        provenance = store.get_artifact(artifact_id, int(version)).provenance_hash

        find = store.find_by_provenance
        rebuilt = []

        def find_then_rebuild(*args, **kwargs):
            found = find(*args, **kwargs)
            if not rebuilt:
                rebuilt.append(store.create_artifact(artifact_id, provenance))
                store.finalize_artifact(
                    artifact_id, rebuilt[0], "", 3, len(ARROW), content_sha256="e" * 64
                )
            return found

        monkeypatch.setattr(store, "find_by_provenance", find_then_rebuild)
        response = await client.put("/v1/artifacts", files=_files(_PUT))

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["hit"] is True
    assert body["name_uri"] == "strata://name/the-put"
    assert body["artifact_uri"] == f"strata://artifact/{artifact_id}@v={rebuilt[0]}"
    named = store.resolve_name("the-put")
    assert (named.id, named.version, named.state) == (artifact_id, rebuilt[0], "ready")
    assert store.get_artifact(artifact_id, int(version)).state == "superseded"
