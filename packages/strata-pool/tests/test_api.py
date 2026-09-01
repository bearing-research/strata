"""The HTTP surface.

Driven through `httpx.ASGITransport` on the test's own event loop rather than
`TestClient`, which runs the app in a portal thread with its own loop. The
pool's background tasks belong to the loop that created them, and this repo
has already paid for one cross-loop task bug (#607).
"""

import os

import httpx
import pytest
from conftest import FakeBackend, FakeWorkers
from strata_pool import MachineType, Pool, PoolStore


def _require_server_extra() -> bool:
    try:
        import fastapi  # noqa: F401
    except ImportError:
        return False
    return True


_HAS_SERVER = _require_server_extra()

if os.environ.get("STRATA_POOL_REQUIRE_SERVER") == "1" and not _HAS_SERVER:
    # CI sets this. Otherwise a venv missing the extra would skip this entire
    # file and report green, which is coverage that examined nothing.
    raise RuntimeError("STRATA_POOL_REQUIRE_SERVER=1 but the `server` extra is not installed")

pytestmark = pytest.mark.skipif(not _HAS_SERVER, reason="needs the `server` extra")

TOKEN = "pool-token"
AUTH = {"Authorization": f"Bearer {TOKEN}", "X-Strata-Tenant": "acme"}


@pytest.fixture
async def api(tmp_path):
    """The app over a real pool, with lifespan run so the scaler starts."""
    from strata_pool.api import create_app

    backend = FakeBackend()
    store = PoolStore(tmp_path / "pool.sqlite")
    client_to_workers = httpx.AsyncClient(transport=httpx.MockTransport(FakeWorkers().handle))
    pool = Pool(
        store,
        backend,
        [MachineType(name="cpu", image="w"), MachineType(name="gpu", image="w")],
        client=client_to_workers,
        health_poll_seconds=0,
    )
    app = create_app(pool, api_token=TOKEN, scaler_interval_seconds=3600)

    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://pool"
        ) as client:
            client.pool = pool
            client.backend = backend
            yield client

    await client_to_workers.aclose()
    store.close()


async def test_a_job_runs_and_its_result_comes_back_as_bytes(api):
    response = await api.post("/v1/jobs/sync?machine_type=cpu", content=b"work", headers=AUTH)

    assert response.status_code == 200
    assert response.content == b"done:work"


async def test_the_payload_is_the_request_body_verbatim(api):
    response = await api.post(
        "/v1/jobs/sync?machine_type=cpu", content=b"\x00binary\xff", headers=AUTH
    )
    assert response.content == b"done:\x00binary\xff"


async def test_an_async_submit_returns_an_id_to_collect_later(api):
    accepted = await api.post("/v1/jobs?machine_type=cpu", content=b"work", headers=AUTH)
    assert accepted.status_code == 202
    job_id = accepted.json()["id"]

    await api.pool.wait(job_id)

    status = await api.get(f"/v1/jobs/{job_id}", headers=AUTH)
    assert status.json()["state"] == "completed"
    assert status.json()["has_result"] is True

    result = await api.get(f"/v1/jobs/{job_id}/result", headers=AUTH)
    assert result.content == b"done:work"


async def test_a_result_asked_for_too_early_is_a_conflict_not_a_lie(api):
    backend = FakeBackend(never_healthy=True)
    api.pool.backend = backend

    accepted = await api.post("/v1/jobs?machine_type=gpu", content=b"work", headers=AUTH)
    job_id = accepted.json()["id"]

    result = await api.get(f"/v1/jobs/{job_id}/result", headers=AUTH)
    assert result.status_code == 409
    assert "queued" in result.json()["detail"]


async def test_a_job_that_fails_on_the_worker_is_not_reported_as_a_pool_error(api):
    """The caller has to tell "your code raised" from "we could not run it"."""
    api.pool._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
    )

    response = await api.post("/v1/jobs/sync?machine_type=cpu", content=b"work", headers=AUTH)

    assert response.status_code == 502
    assert response.json()["state"] == "failed"
    assert "500" in response.json()["error"]


async def test_a_slow_job_hands_back_an_id_rather_than_hanging(api):
    backend = FakeBackend(never_healthy=True)
    api.pool.backend = backend

    response = await api.post(
        "/v1/jobs/sync?machine_type=gpu&wait_seconds=0.05", content=b"work", headers=AUTH
    )

    assert response.status_code == 202
    assert response.json()["state"] in ("queued", "dispatched", "running")


async def test_an_unknown_machine_type_is_the_callers_mistake(api):
    response = await api.post("/v1/jobs?machine_type=h100", content=b"work", headers=AUTH)

    assert response.status_code == 400
    assert "unknown machine type" in response.json()["detail"]


async def test_the_tenant_header_is_required(api):
    response = await api.post(
        "/v1/jobs?machine_type=cpu",
        content=b"work",
        headers={"Authorization": f"Bearer {TOKEN}"},
    )
    assert response.status_code == 400
    assert "X-Strata-Tenant" in response.json()["detail"]


async def test_the_tenant_header_decides_who_the_machine_belongs_to(api):
    await api.post(
        "/v1/jobs/sync?machine_type=cpu",
        content=b"work",
        headers={"Authorization": f"Bearer {TOKEN}", "X-Strata-Tenant": "globex"},
    )
    assert [w.tenant_id for w in api.pool.store.list_workers()] == ["globex"]


class TestAuth:
    async def test_submitting_without_a_token_is_rejected(self, api):
        response = await api.post(
            "/v1/jobs?machine_type=cpu", content=b"work", headers={"X-Strata-Tenant": "acme"}
        )
        assert response.status_code == 401
        assert api.backend.started == [], "an unauthenticated call must not start a machine"

    async def test_a_wrong_token_is_rejected(self, api):
        response = await api.get("/v1/workers", headers={"Authorization": "Bearer not-the-token"})
        assert response.status_code == 401

    async def test_health_is_reachable_without_a_token(self, api):
        response = await api.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestInspection:
    async def test_machine_types_are_listed_for_a_caller_to_resolve_against(self, api):
        response = await api.get("/v1/machine-types", headers=AUTH)
        assert {spec["name"] for spec in response.json()} == {"cpu", "gpu"}

    async def test_workers_are_listed_without_their_credentials(self, api):
        await api.post("/v1/jobs/sync?machine_type=cpu", content=b"work", headers=AUTH)

        listed = (await api.get("/v1/workers", headers=AUTH)).json()
        assert len(listed) == 1
        assert "auth_token" not in listed[0]
        assert TOKEN not in str(listed)
        stored = api.pool.store.list_workers()[0].auth_token
        assert stored not in str(listed), "the machine's own credential must not be served"

    async def test_usage_is_reported_per_tenant_for_billing(self, api):
        await api.post("/v1/jobs/sync?machine_type=cpu", content=b"a", headers=AUTH)
        await api.post(
            "/v1/jobs/sync?machine_type=cpu",
            content=b"b",
            headers={"Authorization": f"Bearer {TOKEN}", "X-Strata-Tenant": "globex"},
        )

        acme = (await api.get("/v1/usage?tenant_id=acme", headers=AUTH)).json()
        assert len(acme) == 1
        assert acme[0]["duration_ms"] > 0
        assert len((await api.get("/v1/usage", headers=AUTH)).json()) == 2


async def test_serving_the_pool_starts_the_scaler(tmp_path):
    """A deployment cannot forget the one call that stops it paying forever."""
    from strata_pool.api import create_app

    store = PoolStore(tmp_path / "pool.sqlite")
    pool = Pool(store, FakeBackend(), [MachineType(name="cpu", image="w")])
    reaped: list[float] = []
    pool.start_scaler = lambda interval: reaped.append(interval)

    app = create_app(pool, api_token=TOKEN, scaler_interval_seconds=42.0)
    async with app.router.lifespan_context(app):
        pass

    assert reaped == [42.0]
    store.close()


async def test_serving_the_pool_reconciles_what_the_last_process_left(tmp_path):
    from strata_pool.api import create_app

    store = PoolStore(tmp_path / "pool.sqlite")
    pool = Pool(store, FakeBackend(), [MachineType(name="cpu", image="w")])
    recovered: list[bool] = []
    original = pool.recover

    async def remember():
        recovered.append(True)
        await original()

    pool.recover = remember

    app = create_app(pool, api_token=TOKEN)
    async with app.router.lifespan_context(app):
        pass

    assert recovered == [True]
    store.close()
