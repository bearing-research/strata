"""Artifact, build-status and publication routes make their store calls off the event loop.

These handlers await nothing, so they are plain ``def`` and FastAPI runs them on anyio's thread
limiter. A store call that waits (a SQLite write lock, a Postgres round trip) made inline would
stall every request the server is serving. Each test blocks one route's store call on an event
and checks that another request completes meanwhile.
"""

from __future__ import annotations

import asyncio
import hashlib
import threading
from types import SimpleNamespace

import anyio.to_thread
import pyarrow as pa
import pytest
from httpx import ASGITransport, AsyncClient

from strata.artifact_store import TransformSpec, get_artifact_store, reset_artifact_store
from strata.config import StrataConfig
from strata.transforms.build_store import reset_build_store
from tests.conftest import table_to_ipc_bytes

ARROW = table_to_ipc_bytes(pa.table({"x": [1, 2, 3]}))
PROVENANCE = hashlib.sha256(b"art").hexdigest()


def _gate(monkeypatch, obj, method: str) -> SimpleNamespace:
    """Make ``obj.method`` block until ``release`` is set; it then runs as before."""
    gate = SimpleNamespace(
        entered=threading.Event(), release=threading.Event(), done=threading.Event()
    )
    original = getattr(obj, method)

    def gated(*args, **kwargs):
        gate.entered.set()
        # A guard, so a call stuck on the loop fails the test rather than hanging it.
        gate.release.wait(timeout=30)
        try:
            return original(*args, **kwargs)
        finally:
            gate.done.set()

    monkeypatch.setattr(obj, method, gated)
    return gate


@pytest.fixture
def served(tmp_path):
    """A personal server holding one ready artifact, a pin, a publication and a build."""
    import strata.server as server_module
    from strata.api.dependencies import runtime_build_store
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
    reset_build_store()
    reset_tenant_registry()
    original = server_module._state
    state = ServerState(config)
    server_module._state = state
    store = get_artifact_store(config.artifact_dir)
    version = store.create_artifact(
        artifact_id="art",
        provenance_hash=PROVENANCE,
        transform_spec=TransformSpec(executor="notebook/cell@v1", params={}, inputs=[]),
    )
    store.write_blob("art", version, ARROW)
    store.finalize_artifact(
        artifact_id="art",
        version=version,
        schema_json="",
        row_count=3,
        byte_size=len(ARROW),
        content_sha256=hashlib.sha256(ARROW).hexdigest(),
    )
    store.pin_artifact("art", version, "kept")
    publication = store.publish_artifact("art", version)
    builds = runtime_build_store()
    builds.create_build("bld", "art", version, "local@v1")
    try:
        yield SimpleNamespace(
            app=server_module.app,
            store=store,
            builds=builds,
            version=version,
            token=publication.token,
        )
    finally:
        state.streams.shutdown_cleanups()
        server_module._state = original
        reset_build_store()
        reset_artifact_store()
        reset_tenant_registry()


# (target, gated method, HTTP method, path, request kwargs, check of the JSON or text body).
# ``{v}`` and ``{token}`` are filled from the fixture.
_ROUTES = [
    pytest.param(
        "store",
        "get_artifact",
        "GET",
        "/v1/artifacts/art/v/{v}",
        {},
        lambda r: r.json()["row_count"] == 3,
        id="artifact-info",
    ),
    pytest.param(
        "store",
        "find_by_provenance",
        "GET",
        f"/v1/artifacts/by-provenance/{PROVENANCE}",
        {},
        lambda r: r.json()["artifact_id"] == "art",
        id="by-provenance",
    ),
    pytest.param(
        "store",
        "stats",
        "GET",
        "/v1/artifacts/stats",
        {},
        lambda r: isinstance(r.json(), dict),
        id="stats",
    ),
    pytest.param(
        "store",
        "get_usage",
        "GET",
        "/v1/artifacts/usage",
        {},
        lambda r: isinstance(r.json(), dict),
        id="usage",
    ),
    pytest.param(
        "store",
        "list_artifacts",
        "GET",
        "/v1/artifacts",
        {},
        lambda r: [a["artifact_id"] for a in r.json()["artifacts"]] == ["art"],
        id="list",
    ),
    pytest.param(
        "store",
        "pin_artifact",
        "POST",
        "/v1/artifacts/art/v/{v}/pin",
        {"json": {"reason": "paper"}},
        lambda r: r.json()["reason"] == "paper",
        id="pin",
    ),
    pytest.param(
        "store",
        "unpin_artifact",
        "DELETE",
        "/v1/artifacts/art/v/{v}/pin",
        {"params": {"reason": "kept"}},
        lambda r: r.json()["unpinned"] is True,
        id="unpin",
    ),
    pytest.param(
        "store",
        "get_artifact",
        "GET",
        "/v1/artifacts/art/v/{v}/dependents",
        {},
        lambda r: r.json()["dependents"] == [],
        id="dependents",
    ),
    pytest.param(
        "builds",
        "get_build",
        "GET",
        "/v1/artifacts/builds/bld",
        {},
        lambda r: r.json()["state"] == "pending",
        id="build-status",
    ),
    pytest.param(
        "builds",
        "get_build",
        "GET",
        "/v1/builds/bld",
        {},
        lambda r: r.json()["build_id"] == "bld",
        id="build-status-compat",
    ),
    pytest.param(
        "store",
        "publish_artifact",
        "POST",
        "/v1/artifacts/art/v/{v}/publish",
        {},
        lambda r: r.json()["artifact_id"] == "art",
        id="publish",
    ),
    pytest.param(
        "store",
        "list_publications",
        "GET",
        "/v1/publications",
        {},
        lambda r: len(r.json()) == 1,
        id="list-publications",
    ),
    pytest.param(
        "store",
        "update_publication_credits",
        "PATCH",
        "/v1/publications/{token}",
        {"json": {"authors": [{"name": "Ada"}]}},
        lambda r: r.json()["authors"] == [{"name": "Ada"}],
        id="credits",
    ),
    pytest.param(
        "store",
        "revoke_publication",
        "DELETE",
        "/v1/publications/{token}",
        {},
        lambda r: r.json()["revoked"] is True,
        id="revoke",
    ),
    pytest.param(
        "store",
        "get_publication",
        "GET",
        "/v1/publications/{token}",
        {},
        lambda r: r.json()["publication"]["url"].startswith("/p/"),
        id="record",
    ),
    pytest.param(
        "store",
        "get_publication",
        "GET",
        "/oembed",
        {"params": {"url": "http://test/p/{token}"}},
        lambda r: r.json()["type"] == "rich",
        id="oembed",
    ),
    pytest.param(
        "store",
        "get_publication",
        "GET",
        "/p/{token}/ro-crate",
        {},
        lambda r: "@graph" in r.json(),
        id="ro-crate",
    ),
    pytest.param(
        "store",
        "get_publication",
        "GET",
        "/p/{token}/badge.svg",
        {},
        lambda r: r.text.startswith("<svg"),
        id="badge",
    ),
]


@pytest.mark.parametrize(("target", "gated", "method", "path", "kwargs", "check"), _ROUTES)
async def test_route_store_call_does_not_block_the_loop(
    served, monkeypatch, target, gated, method, path, kwargs, check
):
    def fill(value):
        if isinstance(value, str):
            return value.format(v=served.version, token=served.token)
        if isinstance(value, dict):
            return {k: fill(v) for k, v in value.items()}
        return value

    gate = _gate(monkeypatch, getattr(served, target), gated)
    async with AsyncClient(transport=ASGITransport(app=served.app), base_url="http://test") as c:
        blocked = asyncio.ensure_future(c.request(method, fill(path), **fill(kwargs)))
        try:
            assert await anyio.to_thread.run_sync(gate.entered.wait, 30), "store call not made"
            other = await c.get("/health")
            other_finished_first = not gate.done.is_set()
        finally:
            gate.release.set()
        response = await blocked

    assert other.status_code == 200
    assert other_finished_first, "the second request waited for the blocked store call"
    assert response.status_code == 200, response.text
    assert check(response)
