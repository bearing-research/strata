"""Name, alias, tag, registry and lineage routes make their store calls off the event loop.

On a Postgres store each call is a network round trip, and a lock wait can last seconds; made
inline from an async route, it stalls every other request on the server. Each test blocks one
route's store call on an event and checks that another request completes meanwhile.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient

from strata.artifact_store import TransformSpec, get_artifact_store, reset_artifact_store
from strata.config import StrataConfig

PROTECTED = ["gold", "silver", "bronze"]


class Gate:
    """Blocks one store method until ``release`` is set."""

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.done = threading.Event()

    def wrap(self, method):
        def gated(*args, **kwargs):
            self.entered.set()
            # A guard, so a call stuck on the loop fails the test rather than hanging it.
            self.release.wait(timeout=30)
            self.done.set()
            return method(*args, **kwargs)

        return gated


def _ready(store, artifact_id: str) -> int:
    data = b"{}"
    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=hashlib.sha256(artifact_id.encode()).hexdigest(),
        transform_spec=TransformSpec(executor="notebook/cell@v1", params={}, inputs=[]),
    )
    store.write_blob(artifact_id, version, data)
    store.finalize_artifact(
        artifact_id=artifact_id,
        version=version,
        schema_json="",
        row_count=None,
        byte_size=len(data),
        content_sha256=hashlib.sha256(data).hexdigest(),
    )
    return version


@pytest.fixture
def served(tmp_path):
    """An in-process personal server with a name, aliases, a tag and two pending changes."""
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
        registry_protected_aliases=PROTECTED,
    )
    reset_artifact_store()
    reset_tenant_registry()
    original = server_module._state
    state = ServerState(config)
    server_module._state = state
    store = get_artifact_store(config.artifact_dir)
    version = _ready(store, "a1")
    store.set_name("n", "a1", version)
    store.set_alias("n", "prod", "a1", version)
    store.set_alias("n", "bronze", "a1", version)
    store.set_tag("a1", version, "k", "v")
    for alias in ("gold", "silver"):
        store.request_alias_change("n", alias, "set", artifact_id="a1", version=version)
    gate = Gate()
    try:
        yield SimpleNamespace(store=store, gate=gate, app=server_module.app)
    finally:
        gate.release.set()
        state.streams.shutdown_cleanups()
        server_module._state = original
        reset_artifact_store()
        reset_tenant_registry()


_ROUTES = [
    pytest.param(
        "PUT",
        "/v1/names/n/aliases/staging",
        {"artifact_id": "a1", "version": 1},
        "set_alias",
        "applied",
        id="set-alias",
    ),
    pytest.param(
        "PUT",
        "/v1/names/n/aliases/gold",
        {"artifact_id": "a1", "version": 1},
        "request_alias_change",
        "pending",
        id="set-protected-alias",
    ),
    pytest.param("GET", "/v1/names/n/aliases/prod", None, "resolve_alias", "a1", id="alias"),
    pytest.param(
        "DELETE", "/v1/names/n/aliases/prod", None, "delete_alias", "deleted", id="delete-alias"
    ),
    pytest.param(
        "DELETE",
        "/v1/names/n/aliases/bronze",
        None,
        "request_alias_change",
        "pending",
        id="delete-protected-alias",
    ),
    pytest.param("GET", "/v1/names/n/aliases", None, "list_aliases", "prod", id="aliases"),
    pytest.param(
        "PUT",
        "/v1/artifacts/a1/v/1/tags",
        {"key": "k2", "value": "v2"},
        "set_tag",
        "v2",
        id="set-tag",
    ),
    pytest.param("GET", "/v1/artifacts/a1/v/1/tags", None, "get_tags", "v", id="tags"),
    pytest.param(
        "DELETE", "/v1/artifacts/a1/v/1/tags/k", None, "delete_tag", "deleted", id="delete-tag"
    ),
    pytest.param("GET", "/v1/names/n", None, "get_name", "a1", id="name"),
    pytest.param(
        "POST",
        "/v1/names",
        {"name": "m", "artifact_id": "a1", "version": 1},
        "set_name",
        "strata://name/m",
        id="set-name",
    ),
    pytest.param("DELETE", "/v1/names/n", None, "delete_name", "deleted", id="delete-name"),
    pytest.param("GET", "/v1/names", None, "list_names", "a1", id="names"),
    pytest.param(
        "GET", "/v1/artifacts/names/n/status", None, "get_name_status", "a1", id="name-status"
    ),
    pytest.param("GET", "/v1/registry/audit", None, "read_audit", "alias_set", id="audit"),
    pytest.param("GET", "/v1/events", None, "read_events", "alias_set", id="events"),
    pytest.param("GET", "/v1/registry/summary", None, "list_all_names", "prod", id="summary"),
    pytest.param(
        "GET",
        "/v1/registry/artifacts?tag_key=k",
        None,
        "list_artifacts_with_tag_key",
        "a1",
        id="by-tag",
    ),
    pytest.param("GET", "/v1/registry/pending", None, "list_pending_changes", "gold", id="pending"),
    pytest.param(
        "POST",
        "/v1/registry/pending/approve",
        {"name": "n", "alias": "gold"},
        "approve_alias_change",
        "approved",
        id="approve",
    ),
    pytest.param(
        "POST",
        "/v1/registry/pending/reject",
        {"name": "n", "alias": "silver"},
        "reject_alias_change",
        "rejected",
        id="reject",
    ),
    pytest.param("GET", "/v1/artifacts/a1/v/1/lineage", None, "get_artifact", "a1", id="lineage"),
]


@pytest.mark.parametrize(("method", "path", "body", "store_method", "expected"), _ROUTES)
async def test_a_blocked_store_call_does_not_block_the_server(
    served, monkeypatch, method, path, body, store_method, expected
):
    gate = served.gate
    monkeypatch.setattr(served.store, store_method, gate.wrap(getattr(served.store, store_method)))

    async with AsyncClient(
        transport=ASGITransport(app=served.app), base_url="http://test"
    ) as client:
        blocked = asyncio.ensure_future(client.request(method, path, json=body))
        try:
            assert await asyncio.to_thread(gate.entered.wait, 30), f"{store_method} was not called"
            other = await client.get("/health")
            answered_while_blocked = not gate.done.is_set()
        finally:
            gate.release.set()
        response = await blocked

    assert other.status_code == 200
    assert answered_while_blocked
    assert response.status_code in (200, 202), response.text
    assert expected in response.text
