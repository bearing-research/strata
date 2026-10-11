"""The build, materialize and stream routes make their metadata-store calls off the event loop.

A store call that waits (a SQLite write lock, a Postgres round trip) made inline stalls every
request the server is serving. Each route test blocks one store call on an event and checks that
another request completes meanwhile, then that the blocked one completes correctly once released.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from types import SimpleNamespace

import pyarrow as pa
import pytest
from httpx import ASGITransport, AsyncClient

from strata.artifact_store import TransformSpec, get_artifact_store
from tests.conftest import hold, ran_while_held, table_to_ipc_bytes

ARROW = table_to_ipc_bytes(pa.table({"x": [1, 2, 3]}))


def _on_loop_recorder(monkeypatch, obj, method: str) -> list[bool]:
    """Record, per call of ``obj.method``, whether it ran on an event loop's thread."""
    on_loop: list[bool] = []
    original = getattr(obj, method)

    def recording(*args, **kwargs):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return original(*args, **kwargs)

    monkeypatch.setattr(obj, method, recording)
    return on_loop


@pytest.fixture
def served(in_process_server):
    """An in-process personal server with the build runtime's stores."""
    import strata.server as server_module
    from strata.transforms.build_store import get_build_store
    from strata.transforms.registry import (
        TransformDefinition,
        TransformRegistry,
        reset_transform_registry,
        set_transform_registry,
    )

    set_transform_registry(
        TransformRegistry(
            enabled=True,
            definitions=[TransformDefinition(ref="test_sql@*", executor_url="http://executor")],
        )
    )
    state = in_process_server()
    store = get_artifact_store(state.config.artifact_dir)
    build_store = get_build_store(
        state.config.artifact_dir / "artifacts.sqlite", dialect=store.dialect
    )
    yield SimpleNamespace(state=state, store=store, build_store=build_store, app=server_module.app)
    reset_transform_registry()


def _client(served) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=served.app), base_url="http://test")


def _ready(store, artifact_id: str, provenance_hash: str | None = None) -> int:
    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=provenance_hash or hashlib.sha256(artifact_id.encode()).hexdigest(),
        transform_spec=TransformSpec(executor="test_sql@v1", params={}, inputs=[]),
    )
    store.write_blob(artifact_id, version, ARROW)
    store.finalize_artifact(
        artifact_id=artifact_id,
        version=version,
        schema_json="",
        row_count=3,
        byte_size=len(ARROW),
    )
    return version


def _queued_build(served, artifact_id: str = "pulled") -> tuple[str, int]:
    version = served.store.create_artifact(artifact_id=artifact_id, provenance_hash="c" * 64)
    build_id = f"build-{artifact_id}"
    served.build_store.create_build(
        build_id=build_id,
        artifact_id=artifact_id,
        version=version,
        executor_ref="test_sql@v1",
        input_uris=[],
        params={},
    )
    return build_id, version


# --- builds ---


@pytest.mark.parametrize("method", ["get_build", "claim_build", "record_attempt"])
async def test_a_manifest_claims_the_build_off_the_loop(served, monkeypatch, method):
    build_id, _ = _queued_build(served)
    gate = hold(monkeypatch, served.build_store, method)

    async with _client(served) as client:
        served_meanwhile, response = await ran_while_held(
            gate, client.get(f"/v1/builds/{build_id}/manifest"), client
        )

    assert served_meanwhile
    assert response.status_code == 200, response.text
    assert response.json()["finalize_url"]
    assert served.build_store.get_build(build_id).state == "building"


@pytest.mark.parametrize("method", ["get_build", "complete_build"])
async def test_a_finalize_reads_and_completes_the_build_off_the_loop(served, monkeypatch, method):
    build_id, version = _queued_build(served)
    served.store.write_blob("pulled", version, ARROW)
    gate = hold(monkeypatch, served.build_store, method)

    async with _client(served) as client:
        served_meanwhile, response = await ran_while_held(
            gate, client.post(f"/v1/builds/{build_id}/finalize"), client
        )

    assert served_meanwhile
    assert response.status_code == 200, response.text
    assert served.build_store.get_build(build_id).state == "ready"
    assert served.store.get_artifact("pulled", version).state == "ready"


async def test_a_failing_finalize_fails_the_build_off_the_loop(served, monkeypatch):
    build_id, version = _queued_build(served)
    served.store.write_blob("pulled", version, b"not arrow")
    gate = hold(monkeypatch, served.build_store, "fail_build")

    async with _client(served) as client:
        served_meanwhile, response = await ran_while_held(
            gate, client.post(f"/v1/builds/{build_id}/finalize"), client
        )

    assert served_meanwhile
    assert response.status_code == 400, response.text
    assert served.build_store.get_build(build_id).error_code == "INVALID_ARROW_FORMAT"
    assert served.store.get_artifact("pulled", version).state == "failed"


async def test_a_signed_download_reads_the_artifact_off_the_loop(served, monkeypatch):
    version = _ready(served.store, "input")
    url = served.state.url_signer.generate_download_url(
        base_url="http://test", artifact_id="input", version=version, build_id="b"
    ).url
    gate = hold(monkeypatch, served.store, "get_artifact")

    async with _client(served) as client:
        served_meanwhile, response = await ran_while_held(gate, client.get(url), client)

    assert served_meanwhile
    assert response.status_code == 200, response.text
    assert response.content == ARROW


async def test_a_signed_upload_reads_the_build_off_the_loop(served, monkeypatch):
    build_id, version = _queued_build(served)
    async with _client(served) as client:
        manifest = (await client.get(f"/v1/builds/{build_id}/manifest")).json()
        gate = hold(monkeypatch, served.build_store, "get_build")
        served_meanwhile, response = await ran_while_held(
            gate, client.post(manifest["output"]["url"], content=ARROW), client
        )

    assert served_meanwhile
    assert response.status_code == 200, response.text


async def test_a_stale_executor_cannot_fail_a_build_a_newer_claim_holds(served, monkeypatch):
    """The manifest is re-issued while a stale executor's bad upload is being checked.

    Its finalize passed the lease check, but failing the build would take it from the newer
    claim, and failing the artifact would void what that claim is building.
    """
    from strata.api.routers.builds import _EXTERNAL_LEASE_OWNER

    build_id, version = _queued_build(served)
    async with _client(served) as client:
        manifest = (await client.get(f"/v1/builds/{build_id}/manifest")).json()
        upload = await client.post(manifest["output"]["url"], content=b"not arrow")
        assert upload.status_code == 200, upload.text

        blob_size = served.store.blob_size

        def reissued_meanwhile(*args, **kwargs):
            assert served.build_store.renew_lease(build_id, _EXTERNAL_LEASE_OWNER, 600.0)
            return blob_size(*args, **kwargs)

        monkeypatch.setattr(served.store, "blob_size", reissued_meanwhile)
        response = await client.post(manifest["finalize_url"])

    assert response.status_code == 409, response.text
    build = served.build_store.get_build(build_id)
    assert build.state == "building"
    assert build.lease_owner == _EXTERNAL_LEASE_OWNER
    assert served.store.get_artifact("pulled", version).state == "building"


async def test_a_manifest_refetch_whose_lease_was_reclaimed_mints_nothing(served, monkeypatch):
    """The runner reclaims the expired lease between the re-fetch's read and its renewal.

    URLs minted from the re-read row would carry the runner's claim, letting the executor
    finalize over the runner.
    """
    build_id, _ = _queued_build(served)
    build_store = served.build_store
    async with _client(served) as client:
        assert (await client.get(f"/v1/builds/{build_id}/manifest")).status_code == 200

        renew_lease = build_store.renew_lease

        def reclaimed_meanwhile(*args, **kwargs):
            conn = build_store._get_connection()
            try:
                conn.execute(
                    "UPDATE artifact_builds SET lease_expires_at = ? WHERE build_id = ?",
                    (time.time() - 1.0, build_id),
                )
                conn.commit()
            finally:
                conn.close()
            assert build_store.reclaim_expired_build(build_id, new_lease_owner="runner-9")
            return renew_lease(*args, **kwargs)

        monkeypatch.setattr(build_store, "renew_lease", reclaimed_meanwhile)
        response = await client.get(f"/v1/builds/{build_id}/manifest")

    assert response.status_code == 409, response.text
    assert build_store.get_build(build_id).lease_owner == "runner-9"


# --- materialize ---


def _transform_request(input_uri: str, **extra) -> dict:
    return {
        "inputs": [input_uri],
        "transform": {"executor": "test_sql@v1", "params": {"sql": "SELECT 1"}},
        **extra,
    }


@pytest.mark.parametrize(
    ("obj", "method"),
    [
        ("store", "get_artifact"),  # resolving the input
        ("store", "find_by_provenance"),
        ("store", "create_artifact"),
        ("build_store", "create_build"),
    ],
)
async def test_a_transform_materialize_uses_the_store_off_the_loop(
    served, monkeypatch, obj, method
):
    version = _ready(served.store, "input")
    gate = hold(monkeypatch, getattr(served, obj), method)

    async with _client(served) as client:
        served_meanwhile, response = await ran_while_held(
            gate,
            client.post(
                "/v1/artifacts/materialize",
                json=_transform_request(f"strata://artifact/input@v={version}"),
            ),
            client,
        )

    assert served_meanwhile
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["hit"] is False
    assert served.build_store.get_build(body["build_id"]).state == "pending"


async def test_an_explain_resolves_its_inputs_off_the_loop(served, monkeypatch):
    version = _ready(served.store, "input")
    gate = hold(monkeypatch, served.store, "get_artifact")

    async with _client(served) as client:
        served_meanwhile, response = await ran_while_held(
            gate,
            client.post(
                "/v1/artifacts/explain-materialize",
                json=_transform_request(f"strata://artifact/input@v={version}"),
            ),
            client,
        )

    assert served_meanwhile
    assert response.status_code == 200, response.text
    assert response.json()["resolved_input_versions"] == {
        f"strata://artifact/input@v={version}": f"input@v={version}"
    }


async def test_a_named_hit_takes_the_refresh_that_superseded_what_it_found(served, monkeypatch):
    """A refresh finalizes between the hit's lookup and its name write.

    The version found is superseded by then, so naming it would be refused: the name goes to
    the refresh's version instead, and the request is still a hit.
    """
    from strata.services.materialize import materialize_service

    store = served.store
    input_version = _ready(store, "input")
    input_uri = f"strata://artifact/input@v={input_version}"
    spec = TransformSpec(executor="test_sql@v1", params={"sql": "SELECT 1"}, inputs=[input_uri])
    provenance = materialize_service.compute_provenance(
        spec, {input_uri: f"input@v={input_version}"}
    )
    _ready(store, "output", provenance)
    find_by_provenance = store.find_by_provenance
    refreshed: list[int] = []

    def refreshed_after_the_find(*args, **kwargs):
        found = find_by_provenance(*args, **kwargs)
        monkeypatch.setattr(store, "find_by_provenance", find_by_provenance)
        refreshed.append(_ready(store, "output", provenance))
        return found

    monkeypatch.setattr(store, "find_by_provenance", refreshed_after_the_find)

    async with _client(served) as client:
        response = await client.post(
            "/v1/artifacts/materialize", json=_transform_request(input_uri, name="the-output")
        )

    assert response.status_code == 200, response.text
    assert response.json()["hit"] is True
    named = store.get_name("the-output")
    assert (named.artifact_id, named.version) == ("output", refreshed[0])
    assert response.json()["artifact_uri"] == f"strata://artifact/output@v={refreshed[0]}"


@pytest.mark.parametrize("method", ["find_by_provenance", "create_artifact"])
async def test_a_scan_materialize_uses_the_store_off_the_loop(
    served, temp_warehouse, monkeypatch, method
):
    gate = hold(monkeypatch, served.store, method)

    async with _client(served) as client:
        served_meanwhile, response = await ran_while_held(
            gate,
            client.post(
                "/v1/materialize",
                json={
                    "inputs": [temp_warehouse["table_uri"]],
                    "transform": {"executor": "scan@v1", "params": {}},
                    "mode": "stream",
                },
            ),
            client,
        )

    assert served_meanwhile
    assert response.status_code == 200, response.text
    assert response.json()["stream_url"]


async def test_a_named_scan_hit_names_off_the_loop(served, temp_warehouse, monkeypatch):
    body = {
        "inputs": [temp_warehouse["table_uri"]],
        "transform": {"executor": "scan@v1", "params": {}},
        "mode": "artifact",
    }
    async with _client(served) as client:
        first = (await client.post("/v1/materialize", json=body)).json()
        await served.state.streams.get(first["build_id"]).background_task
        gate = hold(monkeypatch, served.store, "find_ready_and_set_name")
        served_meanwhile, response = await ran_while_held(
            gate, client.post("/v1/materialize", json={**body, "name": "scanned"}), client
        )

    assert served_meanwhile
    assert response.status_code == 200, response.text
    assert response.json()["hit"] is True
    named = served.store.get_name("scanned")
    assert response.json()["artifact_uri"] == (
        f"strata://artifact/{named.artifact_id}@v={named.version}"
    )


# --- streams ---


async def test_a_stream_reads_its_artifact_off_the_loop(served, temp_warehouse, monkeypatch):
    state = served.state
    async with _client(served) as client:
        materialized = (
            await client.post(
                "/v1/materialize",
                json={
                    "inputs": [temp_warehouse["table_uri"]],
                    "transform": {"executor": "scan@v1", "params": {}},
                    "mode": "stream",
                },
            )
        ).json()
        # Built first, so the gated read is the route's own.
        stream_state = state.streams.get(materialized["stream_id"])
        stream_state.background_task = asyncio.create_task(
            state.scan_builds.build_identity_artifact(state, stream_state)
        )
        await stream_state.background_task
        gate = hold(monkeypatch, served.store, "get_artifact")

        served_meanwhile, response = await ran_while_held(
            gate, client.get(materialized["stream_url"]), client
        )

    assert served_meanwhile
    assert response.status_code == 200, response.text
    assert response.headers["X-Arrow-Row-Count"] == "500"


@pytest.fixture
def two_nodes(served, tmp_path):
    """``served`` as node A of a multi-node deployment sharing the ownership table."""
    from strata.streaming.ownership import (
        get_stream_ownership_store,
        reset_stream_ownership_store,
    )

    reset_stream_ownership_store()
    served.state.config = served.state.config.model_copy(
        update={"node_advertised_url": "http://node-a"}
    )
    owners = get_stream_ownership_store(
        tmp_path / "artifacts" / "artifacts.sqlite", dialect=served.store.dialect
    )
    yield owners
    reset_stream_ownership_store()


async def test_a_stream_claim_is_written_off_the_loop(
    served, two_nodes, temp_warehouse, monkeypatch
):
    gate = hold(monkeypatch, two_nodes, "claim")

    async with _client(served) as client:
        served_meanwhile, response = await ran_while_held(
            gate,
            client.post(
                "/v1/materialize",
                json={
                    "inputs": [temp_warehouse["table_uri"]],
                    "transform": {"executor": "scan@v1", "params": {}},
                    "mode": "stream",
                },
            ),
            client,
        )

    assert served_meanwhile
    assert response.status_code == 200, response.text
    stream_id = response.json()["stream_id"]
    assert two_nodes.resolve(stream_id, exclude_node_url="http://node-b") == "http://node-a"


async def test_a_stream_owner_lookup_runs_off_the_loop(served, two_nodes, monkeypatch):
    two_nodes.claim("elsewhere", "http://node-b", ttl_seconds=60)
    gate = hold(monkeypatch, two_nodes, "resolve")

    async with _client(served) as client:
        served_meanwhile, response = await ran_while_held(
            gate, client.get("/v1/streams/elsewhere", follow_redirects=False), client
        )

    assert served_meanwhile
    assert response.status_code == 307
    assert response.headers["location"] == "http://node-b/v1/streams/elsewhere"


# --- background work ---


def _scan_stream(served, temp_warehouse, **plan_changes):
    from dataclasses import replace

    from strata.streaming import StreamState

    plan = served.state.planner.plan(temp_warehouse["table_uri"])
    if plan_changes:
        plan = replace(plan, **plan_changes)
    version = served.store.create_artifact(artifact_id="scanned", provenance_hash="f" * 64)
    stream_state = StreamState(
        stream_id="scanned",
        plan=plan,
        artifact_id="scanned",
        artifact_version=version,
        created_at=time.time(),
        mode="artifact",
    )
    served.state.streams.register(stream_state)
    return stream_state, version


async def test_a_scan_build_reads_its_result_off_the_loop(served, temp_warehouse, monkeypatch):
    stream_state, version = _scan_stream(served, temp_warehouse)
    on_loop = _on_loop_recorder(monkeypatch, served.store, "get_artifact")

    await served.state.scan_builds.build_identity_artifact(served.state, stream_state)

    assert on_loop
    assert not any(on_loop)
    assert served.store.get_artifact("scanned", version).state == "ready"


async def test_a_failed_scan_build_fails_its_artifact_off_the_loop(
    served, temp_warehouse, monkeypatch
):
    stream_state, version = _scan_stream(served, temp_warehouse)
    on_loop = _on_loop_recorder(monkeypatch, served.store, "fail_artifact")
    served.state._draining = True

    await served.state.scan_builds.build_identity_artifact(served.state, stream_state)

    assert on_loop == [False]
    assert served.store.get_artifact("scanned", version).state == "failed"


async def test_a_scan_finalize_failure_fails_its_artifact_off_the_loop(
    served, temp_warehouse, monkeypatch
):
    stream_state, version = _scan_stream(served, temp_warehouse)
    on_loop = _on_loop_recorder(monkeypatch, served.store, "fail_artifact")

    # The integrity gate refuses a row count the blob does not hold.
    await served.state.scan_builds.finalize_written_blob(served.state, stream_state, 7, 0)

    assert served.store.get_artifact("scanned", version).state == "failed"
    assert on_loop[0] is False


async def test_an_expired_unfetched_stream_fails_its_artifact_off_the_loop(
    served, temp_warehouse, monkeypatch
):
    stream_state, version = _scan_stream(served, temp_warehouse)
    on_loop = _on_loop_recorder(monkeypatch, served.store, "fail_artifact")
    served.state.streams._ttl_seconds = 0.0

    served.state.streams.schedule_cleanup("scanned", stream_state.plan.scan_id)
    await served.state.streams._cleanup_tasks["scanned"]

    assert served.store.get_artifact("scanned", version).state == "failed"
    assert on_loop == [False]
