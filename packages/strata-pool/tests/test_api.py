"""The HTTP surface.

Driven through `httpx.ASGITransport` on the test's own loop, not `TestClient`
(a portal thread with its own loop): the pool's tasks belong to their creating loop.
"""

import asyncio
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
    # CI sets this, so a venv missing the extra fails instead of skipping the whole file.
    raise RuntimeError("STRATA_POOL_REQUIRE_SERVER=1 but the `server` extra is not installed")

pytestmark = pytest.mark.skipif(not _HAS_SERVER, reason="needs the `server` extra")

TOKEN = "pool-token"
AUTH = {"Authorization": f"Bearer {TOKEN}", "X-Strata-Tenant": "acme"}
ADMIN_TOKEN = "operator-token"
ADMIN = {"Authorization": f"Bearer {ADMIN_TOKEN}"}


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
        [
            MachineType(name="cpu", image="w"),
            MachineType(
                name="gpu",
                image="w",
                env={"HF_TOKEN": "hf_operator_secret"},
                provider_options={"registryAuthId": "operator-registry-auth"},
            ),
        ],
        client=client_to_workers,
        health_poll_seconds=0,
    )
    app = create_app(pool, api_token=TOKEN, admin_token=ADMIN_TOKEN, scaler_interval_seconds=3600)

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


class TestJobReadsAreTenantScoped:
    async def _finished_job(self, api) -> str:
        accepted = await api.post("/v1/jobs?machine_type=cpu", content=b"work", headers=AUTH)
        job_id = accepted.json()["id"]
        await api.pool.wait(job_id)
        return job_id

    async def test_the_submitting_tenant_reads_status_and_result(self, api):
        job_id = await self._finished_job(api)

        status = await api.get(f"/v1/jobs/{job_id}", headers=AUTH)
        result = await api.get(f"/v1/jobs/{job_id}/result", headers=AUTH)

        assert status.status_code == 200
        assert status.json()["tenant_id"] == "acme"
        assert result.content == b"done:work"

    async def test_another_tenant_gets_404_not_403(self, api):
        """A 403 would confirm the id exists in someone else's tenant."""
        job_id = await self._finished_job(api)
        globex = {"Authorization": f"Bearer {TOKEN}", "X-Strata-Tenant": "globex"}

        status = await api.get(f"/v1/jobs/{job_id}", headers=globex)
        result = await api.get(f"/v1/jobs/{job_id}/result", headers=globex)

        assert status.status_code == 404
        assert result.status_code == 404
        assert b"done:work" not in result.content

    async def test_reading_without_the_tenant_header_is_refused(self, api):
        job_id = await self._finished_job(api)
        no_tenant = {"Authorization": f"Bearer {TOKEN}"}

        for path in (f"/v1/jobs/{job_id}", f"/v1/jobs/{job_id}/result"):
            response = await api.get(path, headers=no_tenant)
            assert response.status_code == 400
            assert "X-Strata-Tenant" in response.json()["detail"]


class TestCancel:
    async def _queued_job(self, api) -> str:
        api.pool.backend = FakeBackend(never_healthy=True)
        accepted = await api.post("/v1/jobs?machine_type=gpu", content=b"work", headers=AUTH)
        return accepted.json()["id"]

    async def test_a_queued_job_is_cancelled_and_reads_as_cancelled(self, api):
        job_id = await self._queued_job(api)

        response = await api.post(
            f"/v1/jobs/{job_id}/cancel", json={"build_id": "build-1"}, headers=AUTH
        )

        assert response.status_code == 200
        assert response.json()["state"] == "cancelled"
        status = await api.get(f"/v1/jobs/{job_id}", headers=AUTH)
        assert status.json()["state"] == "cancelled"
        result = await api.get(f"/v1/jobs/{job_id}/result", headers=AUTH)
        assert result.status_code == 409
        assert result.json()["state"] == "cancelled"

    async def test_cancelling_twice_answers_the_same(self, api):
        job_id = await self._queued_job(api)
        body = {"build_id": "build-1"}

        await api.post(f"/v1/jobs/{job_id}/cancel", json=body, headers=AUTH)
        again = await api.post(f"/v1/jobs/{job_id}/cancel", json=body, headers=AUTH)

        assert again.status_code == 200
        assert again.json()["state"] == "cancelled"

    async def test_another_tenant_cannot_cancel_a_job(self, api):
        job_id = await self._queued_job(api)
        globex = {"Authorization": f"Bearer {TOKEN}", "X-Strata-Tenant": "globex"}

        response = await api.post(
            f"/v1/jobs/{job_id}/cancel", json={"build_id": "build-1"}, headers=globex
        )

        assert response.status_code == 404
        assert api.pool.store.get_job(job_id).state.value == "queued"

    async def test_a_finished_job_is_a_conflict(self, api):
        accepted = await api.post("/v1/jobs?machine_type=cpu", content=b"work", headers=AUTH)
        job_id = accepted.json()["id"]
        await api.pool.wait(job_id)

        response = await api.post(
            f"/v1/jobs/{job_id}/cancel", json={"build_id": "build-1"}, headers=AUTH
        )

        assert response.status_code == 409
        assert "completed" in response.json()["detail"]
        result = await api.get(f"/v1/jobs/{job_id}/result", headers=AUTH)
        assert result.content == b"done:work"

    async def test_the_build_id_is_required_and_checked(self, api):
        job_id = await self._queued_job(api)

        missing = await api.post(f"/v1/jobs/{job_id}/cancel", json={}, headers=AUTH)
        escaping = await api.post(
            f"/v1/jobs/{job_id}/cancel", json={"build_id": "../execute"}, headers=AUTH
        )

        assert missing.status_code == 422
        assert escaping.status_code == 400
        assert api.pool.store.get_job(job_id).state.value == "queued"

    async def test_a_sync_submit_returns_as_soon_as_its_job_is_cancelled(self, api):
        """Even while the machine has not yet answered: the wait is for the job, not the machine."""
        executing = asyncio.Event()
        release = asyncio.Event()
        cancels: list[httpx.Request] = []

        async def worker(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/cancel"):
                cancels.append(request)
                return httpx.Response(200, json={"cancelled": True})
            executing.set()
            await release.wait()
            return httpx.Response(500, text="harness killed")

        api.pool._client = httpx.AsyncClient(transport=httpx.MockTransport(worker))
        waiting = asyncio.create_task(
            api.post("/v1/jobs/sync?machine_type=cpu&wait_seconds=30", content=b"x", headers=AUTH)
        )
        await executing.wait()
        [job] = api.pool.store.list_jobs()

        cancel = await api.post(
            f"/v1/jobs/{job.id}/cancel", json={"build_id": "build-1"}, headers=AUTH
        )
        response = await waiting

        assert cancel.status_code == 200
        assert response.status_code == 409
        assert response.json()["state"] == "cancelled"
        assert not release.is_set(), "the sync wait ended before the machine answered"
        assert [request.url.path for request in cancels] == ["/v1/executions/build-1/cancel"]
        release.set()


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


class TestCatalogueWrites:
    async def test_a_tenant_token_cannot_rewrite_the_catalogue(self, api):
        """Otherwise one tenant could point another's machine type at its own image."""
        response = await api.put(
            "/v1/machine-types", json=[{"name": "cpu", "image": "evil"}], headers=AUTH
        )

        assert response.status_code == 403
        assert api.pool.machine_types["cpu"].image == "w"
        assert api.pool.store.load_machine_types() is None

    async def test_the_admin_token_rewrites_the_catalogue(self, api):
        response = await api.put(
            "/v1/machine-types", json=[{"name": "cpu", "image": "w2"}], headers=ADMIN
        )

        assert response.status_code == 200
        assert api.pool.machine_types["cpu"].image == "w2"
        assert [spec.image for spec in api.pool.store.load_machine_types()] == ["w2"]

    async def test_the_admin_token_also_runs_jobs(self, api):
        response = await api.post(
            "/v1/jobs/sync?machine_type=cpu",
            content=b"work",
            headers={**ADMIN, "X-Strata-Tenant": "acme"},
        )
        assert response.status_code == 200

    async def test_without_an_admin_token_no_one_rewrites_the_catalogue(self, tmp_path):
        from strata_pool.api import create_app

        store = PoolStore(tmp_path / "pool.sqlite")
        pool = Pool(store, FakeBackend(), [MachineType(name="cpu", image="w")])
        app = create_app(pool, api_token=TOKEN, scaler_interval_seconds=3600)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://pool"
            ) as client:
                response = await client.put(
                    "/v1/machine-types", json=[{"name": "cpu", "image": "evil"}], headers=AUTH
                )

        assert response.status_code == 403
        assert "no admin token" in response.json()["detail"]
        assert pool.machine_types["cpu"].image == "w"
        store.close()


class TestInspection:
    async def test_machine_types_are_listed_for_a_caller_to_resolve_against(self, api):
        response = await api.get("/v1/machine-types", headers=AUTH)
        assert {spec["name"] for spec in response.json()} == {"cpu", "gpu"}

    async def test_a_tenant_sees_operator_settings_by_name_only(self, api):
        listed = (await api.get("/v1/machine-types", headers=AUTH)).json()

        gpu = next(spec for spec in listed if spec["name"] == "gpu")
        assert gpu["env"] == {"HF_TOKEN": "<redacted>"}
        assert gpu["provider_options"] == {"registryAuthId": "<redacted>"}
        assert "hf_operator_secret" not in str(listed)
        assert "operator-registry-auth" not in str(listed)

    async def test_the_admin_sees_the_whole_catalogue(self, api):
        listed = (await api.get("/v1/machine-types", headers=ADMIN)).json()

        gpu = next(spec for spec in listed if spec["name"] == "gpu")
        assert gpu["env"] == {"HF_TOKEN": "hf_operator_secret"}
        assert gpu["provider_options"] == {"registryAuthId": "operator-registry-auth"}

    async def test_workers_are_listed_without_their_credentials(self, api):
        await api.post("/v1/jobs/sync?machine_type=cpu", content=b"work", headers=AUTH)

        listed = (await api.get("/v1/workers", headers=AUTH)).json()
        assert len(listed) == 1
        assert "auth_token" not in listed[0]
        assert TOKEN not in str(listed)
        stored = api.pool.store.list_workers()[0].auth_token
        assert stored not in str(listed), "the machine's own credential must not be served"

    async def test_a_tenant_lists_only_its_own_machines(self, api):
        await api.post("/v1/jobs/sync?machine_type=cpu", content=b"a", headers=AUTH)
        await api.post(
            "/v1/jobs/sync?machine_type=cpu",
            content=b"b",
            headers={"Authorization": f"Bearer {TOKEN}", "X-Strata-Tenant": "globex"},
        )

        own = (await api.get("/v1/workers", headers=AUTH)).json()
        assert [worker["tenant_id"] for worker in own] == ["acme"]
        other = await api.get("/v1/workers?tenant_id=globex", headers=AUTH)
        assert other.status_code == 403

        fleet = (await api.get("/v1/workers", headers=ADMIN)).json()
        assert sorted(worker["tenant_id"] for worker in fleet) == ["acme", "globex"]
        globex = (await api.get("/v1/workers?tenant_id=globex", headers=ADMIN)).json()
        assert [worker["tenant_id"] for worker in globex] == ["globex"]

    async def test_usage_is_reported_per_tenant_for_billing(self, api):
        await api.post("/v1/jobs/sync?machine_type=cpu", content=b"a", headers=AUTH)
        await api.post(
            "/v1/jobs/sync?machine_type=cpu",
            content=b"b",
            headers={"Authorization": f"Bearer {TOKEN}", "X-Strata-Tenant": "globex"},
        )

        acme = (await api.get("/v1/usage?tenant_id=acme", headers=ADMIN)).json()
        assert len(acme) == 1
        assert acme[0]["duration_ms"] > 0
        assert len((await api.get("/v1/usage", headers=ADMIN)).json()) == 2

    async def test_a_tenant_reads_only_its_own_usage(self, api):
        await api.post("/v1/jobs/sync?machine_type=cpu", content=b"a", headers=AUTH)
        await api.post(
            "/v1/jobs/sync?machine_type=cpu",
            content=b"b",
            headers={"Authorization": f"Bearer {TOKEN}", "X-Strata-Tenant": "globex"},
        )

        own = (await api.get("/v1/usage", headers=AUTH)).json()
        assert [event["tenant_id"] for event in own] == ["acme"]
        other = await api.get("/v1/usage?tenant_id=globex", headers=AUTH)
        assert other.status_code == 403
        no_tenant = await api.get("/v1/usage", headers={"Authorization": f"Bearer {TOKEN}"})
        assert no_tenant.status_code == 400


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


async def test_the_callers_trace_headers_travel_with_the_job(api):
    """The caller's W3C trace context survives the store and reaches the machine."""
    traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    response = await api.post(
        "/v1/jobs",
        params={"machine_type": "cpu"},
        content=b"x",
        headers={**AUTH, "traceparent": traceparent, "tracestate": "k=v"},
    )

    job = api.pool.store.get_job(response.json()["id"])
    assert job.trace_context == {"traceparent": traceparent, "tracestate": "k=v"}
