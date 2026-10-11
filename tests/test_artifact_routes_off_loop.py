"""Artifact and publication routes make their metadata-store calls off the event loop.

Each test holds one request's store call on an event, checks that another request completes
meanwhile, then releases it and checks the first finished correctly.
"""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pyarrow as pa
import pytest
from httpx import ASGITransport, AsyncClient

from strata.artifact_store import TransformSpec, get_artifact_store
from tests.conftest import hold, ran_while_held, table_to_ipc_bytes

ARROW = table_to_ipc_bytes(pa.table({"x": [1, 2, 3]}))


@pytest.fixture
def served(in_process_server):
    """An in-process personal server and its artifact store."""
    import strata.server as server_module

    state = in_process_server()
    return SimpleNamespace(
        store=get_artifact_store(state.config.artifact_dir), app=server_module.app
    )


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
    held = hold(monkeypatch, served.store, "find_by_provenance", first_call_only=True)
    async with _client(served) as client:
        completed, response = await ran_while_held(
            held, client.put("/v1/artifacts", files=_files(_PUT)), client
        )

    assert completed
    assert response.status_code == 200, response.text
    assert response.json()["name_uri"] == "strata://name/the-put"
    named = served.store.resolve_name("the-put")
    assert response.json()["artifact_uri"] == f"strata://artifact/{named.id}@v={named.version}"


async def test_put_by_provenance_looks_up_provenance_off_the_loop(served, monkeypatch):
    held = hold(monkeypatch, served.store, "find_by_provenance", first_call_only=True)
    async with _client(served) as client:
        completed, response = await ran_while_held(
            held,
            client.put(
                f"/v1/artifacts/by-provenance/{'b' * 64}",
                files=_files({"content_type": "arrow/ipc"}),
            ),
            client,
        )

    assert completed
    assert response.status_code == 200, response.text
    assert served.store.find_by_provenance("b" * 64).byte_size == len(ARROW)


async def test_import_checks_the_id_off_the_loop(served, monkeypatch):
    held = hold(monkeypatch, served.store, "id_tenant", first_call_only=True)
    record = {"id": "imported", "version": 1, "provenance_hash": "c" * 64, "created_at": 1.0}
    async with _client(served) as client:
        completed, response = await ran_while_held(
            held, client.post("/v1/artifacts/import", files=_files(record)), client
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
    held = hold(monkeypatch, served.store, "get_artifact", first_call_only=True)
    async with _client(served) as client:
        completed, response = await ran_while_held(
            held, client.request(method, path, content=b"x" if method == "POST" else None), client
        )

    assert completed
    assert response.status_code == status, response.text


@pytest.mark.parametrize("suffix", ["", "/data", "/embed", "/verify", "/archive.zip"])
async def test_public_routes_resolve_the_token_off_the_loop(served, monkeypatch, suffix):
    store = served.store
    version = _ready(store, "published")
    token = store.publish_artifact("published", version).token
    held = hold(monkeypatch, store, "get_publication", first_call_only=True)
    async with _client(served) as client:
        completed, response = await ran_while_held(held, client.get(f"/p/{token}{suffix}"), client)

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
