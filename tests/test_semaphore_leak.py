"""Resources are released when clients disconnect mid-stream, time out or are cancelled.

A leaked slot per disconnect eventually makes the server answer 503 to everything. Cleanup lives in
the generator's GeneratorExit and CancelledError handlers.
"""

import asyncio
import time

import httpx
import pyarrow as pa
import pytest
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, NestedField, StringType

from strata.config import StrataConfig
from tests.conftest import find_free_port, run_server


def build_materialize_request(table_uri: str, columns: list[str] | None = None) -> dict:
    params = {}
    if columns is not None:
        params["columns"] = columns
    return {
        "inputs": [table_uri],
        "transform": {"executor": "scan@v1", "params": params},
        "mode": "stream",
    }


@pytest.fixture
def large_warehouse(tmp_path):
    """A warehouse with enough data to make responses slow."""
    import sys

    if sys.platform == "win32":
        pytest.skip("pyiceberg + pyarrow LocalFileSystem path handling broken on Windows")

    warehouse_path = tmp_path / "warehouse"
    warehouse_path.mkdir()

    catalog = SqlCatalog(
        "strata",
        **{
            "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
            "warehouse": str(warehouse_path),
        },
    )

    catalog.create_namespace("test_db")

    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "payload", StringType(), required=False),
    )

    table = catalog.create_table("test_db.large_events", schema)

    # Enough data that responses take time: 10K rows of 1KB each, ~10MB.
    num_rows = 10000
    payload_size = 1000
    data = pa.table(
        {
            "id": pa.array(range(num_rows), type=pa.int64()),
            "payload": pa.array(["x" * payload_size for _ in range(num_rows)], type=pa.string()),
        }
    )
    table.append(data)

    return {
        "warehouse_path": warehouse_path,
        "table_uri": f"file://{warehouse_path}#test_db.large_events",
        "catalog": catalog,
        "table": table,
    }


class TestSemaphoreLeakRegression:
    """Semaphore slots must not leak under disconnects or timeouts."""

    def test_semaphore_released_on_client_timeout(self, large_warehouse, tmp_path):
        """The core case: a leaked slot per timeout eventually makes every request 503."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            max_concurrent_scans=5,  # Low limit, so a leak shows quickly
            scan_timeout_seconds=300.0,
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            table_uri = large_warehouse["table_uri"]

            # Phase 1: force client-side timeouts.
            timeout_count = 0
            for i in range(10):
                try:
                    with httpx.Client(timeout=0.001) as client:  # 1ms timeout
                        resp = client.post(
                            f"{base_url}/v1/materialize",
                            json=build_materialize_request(table_uri),
                            timeout=0.5,  # Longer timeout for the POST
                        )
                        if resp.status_code == 200:
                            stream_url = resp.json()["stream_url"]
                            try:
                                with client.stream(
                                    "GET",
                                    f"{base_url}{stream_url}",
                                    timeout=0.001,
                                ) as stream:
                                    for _ in stream.iter_bytes():
                                        pass
                            except httpx.TimeoutException:
                                timeout_count += 1
                except Exception:
                    timeout_count += 1

            assert timeout_count > 0, "Expected some client timeouts"

            # Phase 2: leaked semaphores would eventually show as 503s. Poll, since cleanup
            # can lag under load.
            with httpx.Client(timeout=30.0) as client:
                resp = client.get(f"{base_url}/health")
                assert resp.status_code == 200

                active_scans = None
                for attempt in range(10):  # Up to 5 seconds total
                    time.sleep(0.5)
                    resp = client.get(f"{base_url}/metrics")
                    assert resp.status_code == 200
                    metrics = resp.json()
                    limits = metrics.get("resource_limits", {})
                    active_scans = limits.get("active_scans", 0)
                    if active_scans == 0:
                        break

                # The key invariant: no active scans once every request is done.
                assert active_scans == 0, (
                    f"Semaphore leak detected! active_scans={active_scans} "
                    f"(should be 0 after all requests complete)"
                )

                # A new materialize succeeds (not 503).
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(table_uri),
                )
                assert resp.status_code == 200, (
                    f"Expected 200, got {resp.status_code}. "
                    "Server may have exhausted semaphore slots due to leak."
                )

    def test_semaphore_released_on_client_disconnect(self, large_warehouse, tmp_path):
        """The semaphore is released when the client disconnects mid-stream."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            max_concurrent_scans=5,
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            table_uri = large_warehouse["table_uri"]

            # Phase 1: disconnect after a partial read.
            for i in range(10):
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(
                        f"{base_url}/v1/materialize",
                        json=build_materialize_request(table_uri),
                    )
                    if resp.status_code != 200:
                        continue

                    stream_url = resp.json()["stream_url"]

                    try:
                        with client.stream(
                            "GET",
                            f"{base_url}{stream_url}",
                        ) as stream:
                            bytes_read = 0
                            for chunk in stream.iter_bytes(chunk_size=1024):
                                bytes_read += len(chunk)
                                if bytes_read > 1000:  # Disconnect after 1KB
                                    break
                    except Exception:
                        pass

            # Give the server time to clean up.
            time.sleep(0.5)

            # Phase 2: verify no leak.
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{base_url}/metrics")
                assert resp.status_code == 200
                metrics = resp.json()
                active_scans = metrics.get("resource_limits", {}).get("active_scans", 0)

                assert active_scans == 0, (
                    f"Semaphore leak on disconnect! active_scans={active_scans}"
                )

    def test_no_503_after_many_timeouts(self, large_warehouse, tmp_path):
        """After many client timeouts the server still accepts new requests."""
        port = find_free_port()
        max_scans = 3  # Very low limit
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            max_concurrent_scans=max_scans,
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            table_uri = large_warehouse["table_uri"]

            # Phase 1: more timeouts than slots; a leak exhausts the slots after max_scans.
            num_timeouts = max_scans * 3

            for i in range(num_timeouts):
                try:
                    with httpx.Client(timeout=0.001) as client:
                        resp = client.post(
                            f"{base_url}/v1/materialize",
                            json=build_materialize_request(table_uri),
                            timeout=1.0,
                        )
                        if resp.status_code == 200:
                            stream_url = resp.json()["stream_url"]
                            try:
                                with client.stream(
                                    "GET",
                                    f"{base_url}{stream_url}",
                                    timeout=0.001,
                                ) as stream:
                                    for _ in stream.iter_bytes():
                                        pass
                            except httpx.TimeoutException:
                                pass
                except Exception:
                    pass

            # Give the server time to clean up.
            time.sleep(0.5)

            # Phase 2: even after many timeouts, no 503 "Server at capacity".
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(
                    f"{base_url}/v1/materialize",
                    json=build_materialize_request(table_uri),
                )

                assert resp.status_code != 503, (
                    "Got 503 after timeouts - semaphore leak detected! "
                    "The fix for generator cleanup may have regressed."
                )
                assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_concurrent_disconnects_no_leak(self, large_warehouse, tmp_path):
        """Concurrent client disconnects do not leak semaphores."""
        port = find_free_port()
        max_scans = 10
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            max_concurrent_scans=max_scans,
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            table_uri = large_warehouse["table_uri"]

            async def disconnect_after_partial_read():
                """Start a materialize, read some data, then disconnect."""
                async with httpx.AsyncClient(timeout=10.0) as client:
                    try:
                        resp = await client.post(
                            f"{base_url}/v1/materialize",
                            json=build_materialize_request(table_uri),
                        )
                        if resp.status_code != 200:
                            return

                        stream_url = resp.json()["stream_url"]

                        try:
                            async with client.stream(
                                "GET",
                                f"{base_url}{stream_url}",
                            ) as stream:
                                bytes_read = 0
                                async for chunk in stream.aiter_bytes(chunk_size=512):
                                    bytes_read += len(chunk)
                                    if bytes_read > 500:
                                        break
                        except Exception:
                            pass
                    except Exception:
                        pass

            # Phase 1: many concurrent disconnects.
            tasks = [disconnect_after_partial_read() for _ in range(20)]
            await asyncio.gather(*tasks, return_exceptions=True)

            # Give the server time to clean up.
            await asyncio.sleep(0.5)

            # Phase 2: verify no leak.
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{base_url}/metrics")
                assert resp.status_code == 200
                metrics = resp.json()
                active_scans = metrics.get("resource_limits", {}).get("active_scans", 0)

                assert active_scans == 0, (
                    f"Semaphore leak under concurrent disconnects! active_scans={active_scans}"
                )


class TestSemaphoreInvariants:
    def test_active_scans_never_negative(self, large_warehouse, tmp_path):
        """active_scans never goes negative."""
        port = find_free_port()
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            max_concurrent_scans=5,
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            table_uri = large_warehouse["table_uri"]

            # Requests with mixed outcomes.
            for _ in range(20):
                with httpx.Client(timeout=5.0) as client:
                    try:
                        resp = client.post(
                            f"{base_url}/v1/materialize",
                            json=build_materialize_request(table_uri),
                        )
                        if resp.status_code == 200:
                            stream_url = resp.json()["stream_url"]
                            # Sometimes complete, sometimes disconnect.
                            try:
                                with client.stream(
                                    "GET",
                                    f"{base_url}{stream_url}",
                                ) as stream:
                                    for chunk in stream.iter_bytes():
                                        pass  # Complete the stream
                            except Exception:
                                pass
                    except Exception:
                        pass

                    # Check the invariant after each request.
                    try:
                        resp = client.get(f"{base_url}/metrics")
                        if resp.status_code == 200:
                            metrics = resp.json()
                            active = metrics.get("resource_limits", {}).get("active_scans", 0)
                            assert active >= 0, f"active_scans went negative: {active}"
                    except Exception:
                        pass

    def test_active_scans_bounded_by_max(self, large_warehouse, tmp_path):
        """active_scans never exceeds max_concurrent_scans."""
        port = find_free_port()
        max_scans = 3
        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            max_concurrent_scans=max_scans,
            deployment_mode="personal",
        )

        with run_server(config) as base_url:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{base_url}/metrics")
                assert resp.status_code == 200
                metrics = resp.json()
                limits = metrics.get("resource_limits", {})
                active = limits.get("active_scans", 0)
                max_allowed = limits.get("max_concurrent_scans", max_scans)

                assert active <= max_allowed, (
                    f"active_scans ({active}) exceeds max_concurrent_scans ({max_allowed})"
                )


@pytest.fixture
def in_process(tmp_path):
    """An in-process personal server over its own state, with stream cleanup torn down after."""
    from fastapi.testclient import TestClient

    import strata.server as server_module
    from strata.artifact_store import reset_artifact_store
    from strata.server import ServerState, app
    from strata.tenant_registry import reset_tenant_registry

    config = StrataConfig(
        host="127.0.0.1",
        deployment_mode="personal",
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
        metadata_db=tmp_path / "meta.sqlite",
    )
    reset_artifact_store()
    reset_tenant_registry()
    original = server_module._state
    state = ServerState(config)
    server_module._state = state
    try:
        yield TestClient(app, raise_server_exceptions=False), state
    finally:
        state.streams.shutdown_cleanups()
        server_module._state = original
        reset_artifact_store()
        reset_tenant_registry()


def _store_is_locked(self, artifact_id, version):
    import sqlite3

    raise sqlite3.OperationalError("database is locked")


class TestAStreamFetchThatFailsReleasesEverything:
    """A refused, cancelled or failed stream fetch frees its slots and keeps the stream expiring.

    Otherwise the stream, its plan and its ``building`` artifact stay until restart, and each
    stranded admission slot brings the server closer to answering every fetch with a 429.
    """

    def test_a_refused_fetch_rearms_the_stream_cleanup(self, in_process, temp_warehouse):
        from strata.streaming import QoSRejected

        client, state = in_process
        body = client.post(
            "/v1/materialize", json=build_materialize_request(temp_warehouse["table_uri"])
        ).json()

        async def refuse(plan, request, scan_id):
            raise QoSRejected("too_many_requests", "bulk", 1)

        state.qos.admit = refuse
        response = client.get(body["stream_url"])

        assert response.status_code == 429
        assert body["stream_id"] in state.streams._cleanup_tasks

    async def test_a_fetch_cancelled_while_queued_rearms_the_stream_cleanup(
        self, in_process, temp_warehouse
    ):
        from strata.api.routers.streams import get_stream

        client, state = in_process
        body = client.post(
            "/v1/materialize", json=build_materialize_request(temp_warehouse["table_uri"])
        ).json()

        async def cancelled_in_queue(plan, request, scan_id):
            raise asyncio.CancelledError

        state.qos.admit = cancelled_in_queue
        try:
            with pytest.raises(asyncio.CancelledError):
                await get_stream(body["stream_id"], request=None)
            assert body["stream_id"] in state.streams._cleanup_tasks
        finally:
            state.streams.shutdown_cleanups()

    def test_a_store_error_after_the_build_frees_the_admission_slot(
        self, in_process, temp_warehouse, monkeypatch
    ):
        from strata.artifact_store import ArtifactStore
        from strata.tenant_registry import get_tenant_registry

        client, state = in_process
        body = client.post(
            "/v1/materialize", json=build_materialize_request(temp_warehouse["table_uri"])
        ).json()

        monkeypatch.setattr(ArtifactStore, "get_artifact", _store_is_locked)
        response = client.get(body["stream_url"])

        assert response.status_code == 500
        interactive_in_use, _, bulk_in_use, _ = get_tenant_registry().aggregate_limiter_usage()
        assert (interactive_in_use, bulk_in_use) == (0, 0)
        assert state.qos.active_scans == 0
        assert body["stream_id"] in state.streams._cleanup_tasks

    @pytest.mark.parametrize("failing", ["open_store", "start_build"])
    def test_a_failure_before_the_build_frees_the_admission_slot(
        self, in_process, temp_warehouse, monkeypatch, failing
    ):
        import strata.artifact_store
        from strata.tenant_registry import get_tenant_registry

        client, state = in_process
        body = client.post(
            "/v1/materialize", json=build_materialize_request(temp_warehouse["table_uri"])
        ).json()

        def fail(*args, **kwargs):
            raise RuntimeError("injected")

        if failing == "open_store":
            monkeypatch.setattr(strata.artifact_store, "get_artifact_store", fail)
        else:
            monkeypatch.setattr(state.scan_builds, "build_identity_artifact", fail)
        response = client.get(body["stream_url"])

        assert response.status_code == 500
        interactive_in_use, _, bulk_in_use, _ = get_tenant_registry().aggregate_limiter_usage()
        assert (interactive_in_use, bulk_in_use) == (0, 0)
        assert state.qos.active_scans == 0
        assert body["stream_id"] in state.streams._cleanup_tasks

    async def test_a_store_error_after_a_build_frees_its_build_slot(
        self, in_process, temp_warehouse, monkeypatch
    ):
        import sqlite3

        from strata.artifact_store import ArtifactStore, get_artifact_store
        from strata.streaming import StreamState

        _, state = in_process
        plan = state.planner.plan(temp_warehouse["table_uri"])
        store = get_artifact_store(state.config.artifact_dir)
        version = store.create_artifact(artifact_id="built", provenance_hash="built")

        class Slot:
            released = False

            async def release(self):
                self.released = True

        slot = Slot()
        stream_state = StreamState(
            stream_id="built",
            plan=plan,
            artifact_id="built",
            artifact_version=version,
            created_at=time.time(),
            mode="artifact",
            build_slot=slot,
        )
        state.streams.register(stream_state)

        monkeypatch.setattr(ArtifactStore, "get_artifact", _store_is_locked)
        try:
            with pytest.raises(sqlite3.OperationalError):
                await state.scan_builds.build_identity_artifact(state, stream_state)
            assert slot.released
            assert "built" in state.streams._cleanup_tasks
        finally:
            state.streams.shutdown_cleanups()
