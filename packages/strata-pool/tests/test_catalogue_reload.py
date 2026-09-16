"""Changing the machine-type catalogue without restarting the pool. Item 38.

Machine types used to be fixed at construction, and the image was read when a
machine started, so a new GPU type or a new image generation needed a pool
restart, and a new image reached only machines started after it.
"""

import os
import sqlite3

import httpx
import pytest
from conftest import FakeBackend, FakeWorkers
from strata_pool import JobState, MachineType, Pool, PoolStore, WorkerState


class Clock:
    def __init__(self, now: float = 1_000_000.0):
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


async def test_a_type_added_to_the_catalogue_takes_a_job_at_once(make_pool):
    pool = make_pool(machine_types=[MachineType(name="cpu", image="w:1")])
    with pytest.raises(ValueError):
        await pool.submit(tenant_id="acme", machine_type="gpu", payload=b"x")

    await pool.replace_machine_types(
        [MachineType(name="cpu", image="w:1"), MachineType(name="gpu", image="g:1")]
    )
    job = await pool.submit(tenant_id="acme", machine_type="gpu", payload=b"x")

    assert (await pool.wait(job.id)).state is JobState.COMPLETED


async def test_an_image_change_sends_new_work_to_the_new_image_and_retires_the_old(make_pool):
    clock = Clock()
    backend = FakeBackend()
    pool = make_pool(
        backend=backend,
        machine_types=[MachineType(name="cpu", image="w:1", cool_down_seconds=300)],
        wall=clock,
    )
    first = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"a")
    await pool.wait(first.id)
    (old,) = pool.store.list_workers()
    assert (old.image, old.state) == ("w:1", WorkerState.WARM)

    await pool.replace_machine_types([MachineType(name="cpu", image="w:2", cool_down_seconds=300)])
    clock.advance(200)
    second = await pool.submit(tenant_id="acme", machine_type="cpu", payload=b"b")
    await pool.wait(second.id)

    # Not the warm machine on the old image: a new one, on the new image.
    new = next(w for w in pool.store.list_workers() if w.id != old.id)
    assert new.image == "w:2"
    assert pool.store.get_job(second.id).worker_id == new.id

    # The old machine retires once idle past its cool-down; the new one, used
    # more recently, stays.
    clock.advance(101)
    assert await pool.reap_idle_workers() == 1
    assert backend.stopped == [old.backend_id]
    assert [w.id for w in pool.store.list_workers()] == [new.id]


async def test_a_removed_type_refuses_work_and_its_machines_drain_at_cool_down(make_pool):
    clock = Clock()
    backend = FakeBackend()
    pool = make_pool(
        backend=backend,
        machine_types=[
            MachineType(name="cpu", image="w"),
            MachineType(name="gpu", image="g", cool_down_seconds=300),
        ],
        wall=clock,
    )
    job = await pool.submit(tenant_id="acme", machine_type="gpu", payload=b"x")
    await pool.wait(job.id)

    await pool.replace_machine_types([MachineType(name="cpu", image="w")])

    with pytest.raises(ValueError):
        await pool.submit(tenant_id="acme", machine_type="gpu", payload=b"y")
    # The type's own cool-down still applies to the machine it left running.
    clock.advance(299)
    assert await pool.reap_idle_workers() == 0
    clock.advance(2)
    assert await pool.reap_idle_workers() == 1
    assert pool.store.list_workers() == []


async def test_queued_work_for_a_removed_type_fails_with_the_reason(make_pool):
    pool = make_pool(
        backend=FakeBackend(never_healthy=True),
        machine_types=[MachineType(name="cpu", image="w"), MachineType(name="gpu", image="g")],
    )
    job = await pool.submit(tenant_id="acme", machine_type="gpu", payload=b"x")
    assert pool.store.get_job(job.id).state is JobState.QUEUED

    await pool.replace_machine_types([MachineType(name="cpu", image="w")])

    failed = pool.store.get_job(job.id)
    assert failed.state is JobState.FAILED
    assert "removed from the catalogue" in failed.error


def test_a_store_from_before_images_were_recorded_gains_the_column(tmp_path):
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE workers (id TEXT PRIMARY KEY, machine_type TEXT NOT NULL, "
        "tenant_id TEXT NOT NULL, backend TEXT NOT NULL, backend_id TEXT, "
        "state TEXT NOT NULL, endpoint TEXT, region TEXT, session_id TEXT, "
        "current_job_id TEXT, created_at REAL NOT NULL, last_active_at REAL, auth_token TEXT)"
    )
    conn.execute(
        "INSERT INTO workers (id, machine_type, tenant_id, backend, state, created_at) "
        "VALUES ('w1', 'cpu', 'acme', 'fake', 'warm', 1.0)"
    )
    conn.commit()
    conn.close()

    store = PoolStore(path)

    (worker,) = store.list_workers()
    assert worker.image is None
    store.close()


def _has_server() -> bool:
    try:
        import fastapi  # noqa: F401
    except ImportError:
        return False
    return True


needs_server = pytest.mark.skipif(
    not _has_server() and os.environ.get("STRATA_POOL_REQUIRE_SERVER") != "1",
    reason="needs the `server` extra",
)


@needs_server
async def test_the_catalogue_set_over_the_api_serves_jobs_and_survives_a_restart(tmp_path):
    from strata_pool.api import create_app

    auth = {"Authorization": "Bearer t", "X-Strata-Tenant": "acme"}
    workers = httpx.AsyncClient(transport=httpx.MockTransport(FakeWorkers().handle))

    async def serve(store):
        pool = Pool(
            store,
            FakeBackend(),
            [MachineType(name="cpu", image="w")],
            client=workers,
            health_poll_seconds=0,
        )
        app = create_app(pool, api_token="t", scaler_interval_seconds=3600)
        return app

    store = PoolStore(tmp_path / "pool.sqlite")
    app = await serve(store)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://pool"
        ) as client:
            bad = await client.put("/v1/machine-types", json=[{"name": "x"}], headers=auth)
            assert bad.status_code == 400

            put = await client.put(
                "/v1/machine-types",
                json=[{"name": "cpu", "image": "w"}, {"name": "gpu", "image": "g", "gpu_count": 2}],
                headers=auth,
            )
            assert put.status_code == 200
            run = await client.post(
                "/v1/jobs/sync", params={"machine_type": "gpu"}, content=b"hi", headers=auth
            )
            assert run.status_code == 200
            assert run.content == b"done:hi"

    # A new process over the same store, constructed with the old catalogue.
    app = await serve(store)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://pool"
        ) as client:
            listed = (await client.get("/v1/machine-types", headers=auth)).json()
            assert [(t["name"], t["gpu_count"]) for t in listed] == [("cpu", 1), ("gpu", 2)]

    await workers.aclose()
    store.close()
