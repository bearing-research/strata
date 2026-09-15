"""Pins, and garbage collection a platform can run on a shared store. Item 9.

Publications already protected their chains. A platform also needs to hold
chains the store has no other reason to keep, a snapshot that must stay
restorable for instance, and to run the sweep in service mode and on a timer.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from strata.artifact_store import ArtifactStore


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


def _version(store, artifact_id: str, payload: bytes, inputs: dict | None = None, tenant=None):
    version = store.create_artifact(
        artifact_id,
        hashlib.sha256(artifact_id.encode() + payload).hexdigest(),
        input_versions=inputs,
        tenant=tenant,
    )
    with store.open_blob_writer(artifact_id, version) as writer:
        writer.write(payload)
    store.finalize_artifact(
        artifact_id, version, schema_json="", row_count=0, byte_size=len(payload)
    )
    return version


def _superseded_chain(store, tenant=None):
    """rows@v1 feeds figure@v1; both then gain a newer version, so neither is
    named, published or latest: exactly what a sweep collects."""
    rows = _version(store, "rows", b"[1]", tenant=tenant)
    ref = f"rows@v={rows}"
    figure = _version(
        store, "figure", b"PNG", inputs={f"strata://artifact/{ref}": ref}, tenant=tenant
    )
    _version(store, "rows", b"[1, 2]", tenant=tenant)
    _version(store, "figure", b"PNG2", tenant=tenant)
    return rows, figure


class TestPinsInTheStore:
    def test_a_pinned_version_and_its_ancestors_survive_a_sweep(self, store):
        rows, figure = _superseded_chain(store)
        store.pin_artifact("figure", figure, "snapshot:s1", pinned_by="amber")

        store.garbage_collect(max_age_days=0)

        assert store.get_artifact("figure", figure) is not None
        assert store.get_artifact("rows", rows) is not None

    def test_once_released_the_chain_is_collected(self, store):
        rows, figure = _superseded_chain(store)
        store.pin_artifact("figure", figure, "snapshot:s1")
        assert store.unpin_artifact("figure", figure, "snapshot:s1")

        result = store.garbage_collect(max_age_days=0)

        assert result["deleted_count"] == 2
        assert store.get_artifact("figure", figure) is None
        assert store.get_artifact("rows", rows) is None

    def test_two_holders_release_independently(self, store):
        rows, figure = _superseded_chain(store)
        store.pin_artifact("figure", figure, "snapshot:s1")
        store.pin_artifact("figure", figure, "review:42")
        # Pinning again under a reason refreshes it rather than stacking.
        store.pin_artifact("figure", figure, "snapshot:s1", pinned_by="again")

        assert store.unpin_artifact("figure", figure, "snapshot:s1")
        store.garbage_collect(max_age_days=0)

        assert store.get_artifact("figure", figure) is not None
        assert [pin["reason"] for pin in store.list_pins("figure", figure)] == ["review:42"]

    def test_a_pin_needs_a_version_that_exists_and_a_reason(self, store):
        _version(store, "rows", b"[1]")
        with pytest.raises(ValueError, match="not found"):
            store.pin_artifact("rows", 99, "snapshot:s1")
        with pytest.raises(ValueError, match="reason"):
            store.pin_artifact("rows", 1, "  ")


@pytest.fixture
def service(monkeypatch, tmp_path):
    """The real app over a service-mode, trusted-proxy server."""
    import strata.server as server_module
    from strata.artifact_store import get_artifact_store, reset_artifact_store
    from strata.config import StrataConfig
    from strata.server import ServerState, app

    artifact_dir = tmp_path / "service-artifacts"
    config = StrataConfig(
        deployment_mode="service",
        auth_mode="trusted_proxy",
        proxy_token="sekrit",
        artifact_dir=artifact_dir,
        cache_dir=tmp_path / "cache",
    )
    reset_artifact_store()
    monkeypatch.setattr(server_module, "_state", ServerState(config))
    store = get_artifact_store(artifact_dir)
    yield TestClient(app), store
    reset_artifact_store()


def _as(principal: str, tenant: str, scopes: str) -> dict[str, str]:
    return {
        "X-Strata-Proxy-Token": "sekrit",
        "X-Strata-Principal": principal,
        "X-Strata-Tenant": tenant,
        "X-Tenant-ID": tenant,
        "X-Strata-Scopes": scopes,
    }


class TestTheRoutesInServiceMode:
    def test_gc_needs_admin_and_is_scoped_to_the_callers_tenant(self, service):
        client, store = service
        acme_rows, _ = _superseded_chain(store, tenant="acme")
        _version(store, "other", b"x", tenant="globex")
        other_old = 1
        _version(store, "other", b"y", tenant="globex")

        refused = client.post(
            "/v1/artifacts/gc",
            params={"max_age_days": 0},
            headers=_as("m", "acme", "artifacts:read"),
        )
        swept = client.post(
            "/v1/artifacts/gc", params={"max_age_days": 0}, headers=_as("ops", "acme", "admin:*")
        )

        assert refused.status_code == 403
        assert swept.status_code == 200, swept.text
        assert store.get_artifact("rows", acme_rows) is None
        # admin:* is break-glass and unscoped, as on every other artifact
        # route, so its sweep reaches globex too.
        assert store.get_artifact("other", other_old) is None

    def test_a_pin_over_the_api_protects_the_chain_and_names_who_placed_it(self, service):
        client, store = service
        rows, figure = _superseded_chain(store, tenant="acme")

        refused = client.post(
            f"/v1/artifacts/figure/v/{figure}/pin",
            json={"reason": "snapshot:s1"},
            headers=_as("m", "acme", "artifacts:read"),
        )
        pinned = client.post(
            f"/v1/artifacts/figure/v/{figure}/pin",
            json={"reason": "snapshot:s1"},
            headers=_as("amber", "acme", "artifacts:pin"),
        )
        store.garbage_collect(max_age_days=0)

        assert refused.status_code == 403
        assert pinned.status_code == 200, pinned.text
        assert pinned.json()["pinned_by"] == "amber"
        assert store.get_artifact("rows", rows) is not None

        released = client.delete(
            f"/v1/artifacts/figure/v/{figure}/pin",
            params={"reason": "snapshot:s1"},
            headers=_as("amber", "acme", "artifacts:pin"),
        )
        assert released.status_code == 200
        assert store.list_pins("figure", figure) == []

    def test_another_tenants_version_cannot_be_pinned(self, service):
        client, store = service
        figure = _version(store, "figure", b"PNG", tenant="globex")

        response = client.post(
            f"/v1/artifacts/figure/v/{figure}/pin",
            json={"reason": "snapshot:s1"},
            headers=_as("amber", "acme", "artifacts:pin"),
        )

        assert response.status_code in (403, 404)
        assert store.list_pins("figure", figure) == []


def test_without_principal_auth_service_mode_refuses_gc(monkeypatch, tmp_path):
    import strata.server as server_module
    from strata.config import StrataConfig
    from strata.server import ServerState, app

    config = StrataConfig(deployment_mode="service", artifact_dir=tmp_path / "a")
    monkeypatch.setattr(server_module, "_state", ServerState(config))

    response = TestClient(app).post("/v1/artifacts/gc")

    assert response.status_code == 403


async def test_the_scheduled_sweep_runs_and_survives_a_failing_pass():
    from strata.server import _artifact_gc_loop

    calls: list[float] = []
    done = asyncio.Event()

    def garbage_collect(max_age_days):
        calls.append(max_age_days)
        if len(calls) == 1:
            raise RuntimeError("blob store down")
        if len(calls) == 3:
            done.set()
        return {"deleted_count": 0, "deleted_bytes": 0}

    task = asyncio.create_task(
        _artifact_gc_loop(SimpleNamespace(garbage_collect=garbage_collect), 0, 3.0)
    )
    await asyncio.wait_for(done.wait(), timeout=30)
    task.cancel()

    assert calls[:3] == [3.0, 3.0, 3.0]
