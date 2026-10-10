"""Blob store calls run off the event loop.

On S3, GCS or Azure each blob read or write is network I/O; made inline from an async route or
build, it stalls every other request on the server. Each test blocks one blob call on an event and
checks that the loop runs another coroutine meanwhile.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import time
import uuid
from contextlib import contextmanager
from types import SimpleNamespace

import anyio.to_thread
import pyarrow as pa
import pytest
from httpx import ASGITransport, AsyncClient

from strata.artifact_store import TransformSpec, get_artifact_store, reset_artifact_store
from strata.blob_store import LocalBlobStore
from strata.config import StrataConfig
from tests.conftest import table_to_ipc_bytes

ARROW = table_to_ipc_bytes(pa.table({"x": [1, 2, 3]}))
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


class GatedBlobStore(LocalBlobStore):
    """A local store whose ``gated`` call blocks until ``release`` is set."""

    def __init__(self, blobs_dir):
        super().__init__(blobs_dir)
        self.gated: str | None = None
        self.entered = threading.Event()
        self.release = threading.Event()
        self.done = threading.Event()

    def _gate(self, name: str) -> None:
        if name != self.gated:
            return
        self.entered.set()
        # A guard, so a call stuck on the loop fails the test rather than hanging it.
        self.release.wait(timeout=30)
        self.done.set()

    def publish_blob_from_path(self, artifact_id, version, source_path):
        self._gate("publish")
        super().publish_blob_from_path(artifact_id, version, source_path)

    def open_blob_reader(self, artifact_id, version):
        self._gate("read")
        return super().open_blob_reader(artifact_id, version)

    def blob_exists(self, artifact_id, version):
        self._gate("exists")
        return super().blob_exists(artifact_id, version)

    def blob_size(self, artifact_id, version):
        self._gate("size")
        return super().blob_size(artifact_id, version)

    def delete_blob(self, artifact_id, version):
        self._gate("delete")
        return super().delete_blob(artifact_id, version)

    @contextmanager
    def open_blob_writer(self, artifact_id, version):
        with super().open_blob_writer(artifact_id, version) as out:
            yield out
            self._gate("commit")


async def _loop_ran_while_blocked(blobs: GatedBlobStore, gate: str, call):
    """Run *call* with *gate* blocked: whether this coroutine ran meanwhile, and the result."""
    blobs.gated = gate
    task = asyncio.ensure_future(call)
    try:
        assert await asyncio.to_thread(blobs.entered.wait, 30), f"no {gate} call was made"
        ran_while_blocked = not blobs.done.is_set()
    finally:
        blobs.release.set()
    return ran_while_blocked, await task


@pytest.fixture
def served(tmp_path):
    """An in-process personal server whose artifact store's blob calls can be blocked."""
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
    store = get_artifact_store(config.artifact_dir)
    blobs = GatedBlobStore(store.blobs_dir)
    store.blob_store = blobs
    try:
        yield SimpleNamespace(state=state, store=store, blobs=blobs, app=server_module.app)
    finally:
        blobs.release.set()
        state.streams.shutdown_cleanups()
        server_module._state = original
        reset_artifact_store()
        reset_tenant_registry()


def _client(served) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=served.app), base_url="http://test")


def _files(metadata: dict, data: bytes = ARROW) -> dict:
    return {
        "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
        "data": ("data.arrow", data, "application/vnd.apache.arrow.stream"),
    }


def _ready(store, artifact_id: str, data: bytes, content_type: str | None = None) -> int:
    params = {"content_type": content_type} if content_type else {}
    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=hashlib.sha256(artifact_id.encode()).hexdigest(),
        transform_spec=TransformSpec(executor="notebook/cell@v1", params=params, inputs=[]),
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


_UPLOADS = [
    pytest.param(
        "PUT",
        "/v1/artifacts",
        {"inputs": [], "transform": {"executor": "local@v1", "params": {}}},
        id="put",
    ),
    pytest.param(
        "PUT",
        f"/v1/artifacts/by-provenance/{'b' * 64}",
        {"content_type": "arrow/ipc"},
        id="by-provenance",
    ),
    pytest.param(
        "POST",
        "/v1/artifacts/import",
        {"id": "imported", "version": 1, "provenance_hash": "c" * 64, "created_at": 1.0},
        id="import",
    ),
]


@pytest.mark.parametrize(("method", "path", "metadata"), _UPLOADS)
async def test_an_upload_publishes_its_blob_off_the_loop(served, method, path, metadata):
    async with _client(served) as client:
        ran, response = await _loop_ran_while_blocked(
            served.blobs,
            "publish",
            client.request(method, path, files=_files(metadata)),
        )

    assert response.status_code == 200, response.text
    assert ran


async def test_finalize_hashes_the_blob_off_the_loop(served):
    """Finalize with no digest reads the whole blob back to hash it."""
    store = served.store
    version = store.create_artifact(artifact_id="uploaded", provenance_hash="d" * 64)
    store.write_blob("uploaded", version, ARROW)
    body = {"artifact_id": "uploaded", "version": version, "arrow_schema": "", "row_count": 3}

    async with _client(served) as client:
        ran, response = await _loop_ran_while_blocked(
            served.blobs, "read", client.post("/v1/artifacts/finalize", json=body)
        )

    assert response.status_code == 200, response.text
    assert ran
    assert store.get_artifact("uploaded", version).content_sha256 == (
        hashlib.sha256(ARROW).hexdigest()
    )


@pytest.mark.parametrize("gate", ["exists", "size"])
async def test_finalize_probes_the_blob_off_the_loop(served, gate):
    store = served.store
    version = store.create_artifact(artifact_id="uploaded", provenance_hash="d" * 64)
    store.write_blob("uploaded", version, ARROW)
    body = {"artifact_id": "uploaded", "version": version, "arrow_schema": "", "row_count": 3}

    async with _client(served) as client:
        ran, response = await _loop_ran_while_blocked(
            served.blobs, gate, client.post("/v1/artifacts/finalize", json=body)
        )

    assert response.status_code == 200, response.text
    assert ran
    assert store.get_artifact("uploaded", version).state == "ready"


async def test_a_delete_removes_its_blob_off_the_loop(served):
    store = served.store
    version = _ready(store, "doomed", ARROW)

    async with _client(served) as client:
        ran, response = await _loop_ran_while_blocked(
            served.blobs, "delete", client.delete(f"/v1/artifacts/doomed/v/{version}")
        )

    assert response.status_code == 200, response.text
    assert ran
    assert store.get_artifact("doomed", version) is None
    assert not store.blob_exists("doomed", version)


async def test_an_import_retry_checks_the_held_blob_off_the_loop(served):
    """A JSON import with nothing staged is a retry of one that went through."""
    version = _ready(served.store, "held", ARROW)
    record = {
        "id": "held",
        "version": version,
        "provenance_hash": hashlib.sha256(b"held").hexdigest(),
        "created_at": 1.0,
        "content_sha256": hashlib.sha256(ARROW).hexdigest(),
    }

    async with _client(served) as client:
        ran, response = await _loop_ran_while_blocked(
            served.blobs, "exists", client.post("/v1/artifacts/import", json=record)
        )

    assert response.status_code == 200, response.text
    assert ran
    assert response.json()["id"] == "held"


def _queued_build(store, build_store, artifact_id: str) -> tuple[str, int]:
    version = store.create_artifact(artifact_id=artifact_id, provenance_hash="c" * 64)
    build_id = str(uuid.uuid4())
    build_store.create_build(
        build_id=build_id,
        artifact_id=artifact_id,
        version=version,
        executor_ref="duckdb_sql@v1",
        input_uris=[],
        params={},
    )
    return build_id, version


@pytest.mark.parametrize("gate", ["exists", "size"])
async def test_a_build_finalize_probes_the_blob_off_the_loop(served, gate):
    from strata.api.dependencies import runtime_build_store

    store = served.store
    build_store = runtime_build_store()
    build_id, version = _queued_build(store, build_store, "pulled")
    store.write_blob("pulled", version, ARROW)

    async with _client(served) as client:
        ran, response = await _loop_ran_while_blocked(
            served.blobs, gate, client.post(f"/v1/builds/{build_id}/finalize")
        )

    assert response.status_code == 200, response.text
    assert ran
    assert build_store.get_build(build_id).state == "ready"


async def test_a_build_finalize_that_lost_its_lease_drops_its_attempt_off_the_loop(
    served, monkeypatch
):
    """The lease moved on mid-finalize: the attempt's bytes go, and nothing is published."""
    from strata.api.dependencies import runtime_build_store

    store = served.store
    build_store = runtime_build_store()
    build_id, version = _queued_build(store, build_store, "pulled")
    finalize_and_set_name = store.finalize_and_set_name

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
        return finalize_and_set_name(*args, **kwargs)

    async with _client(served) as client:
        manifest = (await client.get(f"/v1/builds/{build_id}/manifest")).json()
        upload = await client.post(manifest["output"]["url"], content=ARROW)
        assert upload.status_code == 200, upload.text
        attempt_blobs = {p.name for p in store.blobs_dir.rglob("*.arrow")}
        assert len(attempt_blobs) == 1
        monkeypatch.setattr(store, "finalize_and_set_name", reclaimed_meanwhile)

        ran, response = await _loop_ran_while_blocked(
            served.blobs, "delete", client.post(manifest["finalize_url"])
        )

    assert response.status_code == 409, response.text
    assert ran
    assert not list(store.blobs_dir.rglob("*.arrow"))
    assert store.get_artifact("pulled", version).state == "building"


@pytest.mark.parametrize("suffix", ["", "/embed", "/data"])
async def test_a_publication_reads_its_blob_off_the_loop(served, suffix):
    store = served.store
    version = _ready(store, "figure", PNG, content_type="image/png")
    token = store.publish_artifact("figure", version).token

    async with _client(served) as client:
        ran, response = await _loop_ran_while_blocked(
            served.blobs, "read", client.get(f"/p/{token}{suffix}")
        )

    assert response.status_code == 200, response.text
    assert ran


@pytest.fixture
def runner(served, tmp_path):
    from strata.transforms.build_store import get_build_store, reset_build_store
    from strata.transforms.registry import TransformDefinition, TransformRegistry
    from strata.transforms.runner import BuildRunner, RunnerConfig

    reset_build_store()
    build_store = get_build_store(tmp_path / "artifacts" / "artifacts.sqlite")
    registry = TransformRegistry(
        enabled=True,
        definitions=[TransformDefinition(ref="test_sql@*", executor_url="http://executor")],
    )
    yield BuildRunner(
        config=RunnerConfig(),
        artifact_store=served.store,
        build_store=build_store,
        transform_registry=registry,
        artifact_dir=tmp_path / "artifacts",
    )
    reset_build_store()


@pytest.mark.parametrize("by_name", [False, True])
async def test_a_build_input_is_read_off_the_loop(served, runner, by_name):
    version = _ready(served.store, "input", ARROW)
    uri = "strata://name/the-input" if by_name else f"strata://artifact/input@v={version}"
    if by_name:
        served.store.set_name("the-input", "input", version)
    temp_files = []

    ran, path = await _loop_ran_while_blocked(
        served.blobs, "read", runner._acquire_input(uri, temp_files)
    )

    assert ran
    assert path.read_bytes() == ARROW
    path.unlink()


async def test_an_input_copy_cancelled_midway_leaves_its_file_registered(served, runner, tmp_path):
    """A stopping runner cancels the build while the copy thread runs on.

    The build's cleanup can only remove files it knows about, so the file must be
    registered before the copy starts.
    """
    version = _ready(served.store, "input", ARROW)
    temp_files: list = []
    served.blobs.gated = "read"
    task = asyncio.ensure_future(
        runner._acquire_input(f"strata://artifact/input@v={version}", temp_files)
    )
    try:
        assert await asyncio.to_thread(served.blobs.entered.wait, 30), "the copy never started"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        served.blobs.release.set()
    assert await asyncio.to_thread(served.blobs.done.wait, 30)

    on_disk = set((tmp_path / "artifacts").glob("tmp*.arrow"))
    assert on_disk <= set(temp_files)
    assert temp_files, "nothing was registered for cleanup"


@pytest.mark.parametrize("gate", ["publish", "read"])
async def test_a_build_publishes_and_finalizes_off_the_loop(served, runner, tmp_path, gate):
    """``publish`` is the output upload; ``read`` is finalize hashing it back."""
    store = served.store
    version = store.create_artifact(
        artifact_id="built",
        provenance_hash="e" * 64,
        transform_spec=TransformSpec(executor="service://test_sql@v1", params={}, inputs=[]),
        input_versions={},
    )
    build_id = str(uuid.uuid4())
    runner.build_store.create_build(
        build_id=build_id,
        artifact_id="built",
        version=version,
        executor_ref="test_sql@v1",
        executor_url="http://executor",
    )
    output = tmp_path / "output.arrow"

    async def executor(**_kwargs):
        output.write_bytes(ARROW)
        return output, None

    runner._call_executor = executor

    ran, _ = await _loop_ran_while_blocked(
        served.blobs, gate, runner._execute_build(runner.build_store.get_build(build_id))
    )

    assert ran
    assert runner.build_store.get_build(build_id).state == "ready"
    assert store.get_artifact("built", version).state == "ready"


@pytest.mark.parametrize("empty", [False, True])
async def test_a_scan_build_commits_its_blob_off_the_loop(served, temp_warehouse, empty):
    from dataclasses import replace

    from strata.streaming import StreamState

    state, store = served.state, served.store
    plan = state.planner.plan(temp_warehouse["table_uri"])
    if empty:
        plan = replace(plan, tasks=[])
    version = store.create_artifact(artifact_id="scanned", provenance_hash="f" * 64)
    stream_state = StreamState(
        stream_id="scanned",
        plan=plan,
        artifact_id="scanned",
        artifact_version=version,
        created_at=time.time(),
        mode="artifact",
    )
    state.streams.register(stream_state)

    ran, _ = await _loop_ran_while_blocked(
        served.blobs, "commit", state.scan_builds.build_identity_artifact(state, stream_state)
    )

    assert ran
    assert stream_state.error_message is None
    assert store.get_artifact("scanned", version).state == "ready"


async def test_a_scan_build_hashes_its_blob_off_the_loop(served, temp_warehouse, monkeypatch):
    from strata.streaming import StreamState

    state, store = served.state, served.store
    plan = state.planner.plan(temp_warehouse["table_uri"])
    version = store.create_artifact(artifact_id="scanned", provenance_hash="f" * 64)
    stream_state = StreamState(
        stream_id="scanned",
        plan=plan,
        artifact_id="scanned",
        artifact_version=version,
        created_at=time.time(),
        mode="artifact",
    )
    state.streams.register(stream_state)
    on_loop: list[bool] = []
    blob_digest = store.blob_digest

    def recording(*args, **kwargs):
        try:
            asyncio.get_running_loop()
            on_loop.append(True)
        except RuntimeError:
            on_loop.append(False)
        return blob_digest(*args, **kwargs)

    monkeypatch.setattr(store, "blob_digest", recording)

    await state.scan_builds.build_identity_artifact(state, stream_state)

    assert store.get_artifact("scanned", version).state == "ready"
    assert on_loop == [False]


async def test_a_named_build_is_ready_only_with_its_name(served, runner, tmp_path, monkeypatch):
    """The build completes and its name moves in one commit.

    Finalize runs in a thread, so the loop serves requests the moment it commits: a client that
    saw the build ready and then asked for the name must find it.
    """
    store = served.store
    version = store.create_artifact(
        artifact_id="named",
        provenance_hash="f" * 64,
        transform_spec=TransformSpec(executor="service://test_sql@v1", params={}, inputs=[]),
        input_versions={},
    )
    build_id = str(uuid.uuid4())
    runner.build_store.create_build(
        build_id=build_id,
        artifact_id="named",
        version=version,
        executor_ref="test_sql@v1",
        executor_url="http://executor",
        name="the-output",
    )
    output = tmp_path / "output.arrow"

    async def executor(**_kwargs):
        output.write_bytes(ARROW)
        return output, None

    runner._call_executor = executor
    seen_at_commit = []

    def observed(finalize):
        def call(*args, **kwargs):
            result = finalize(*args, **kwargs)
            seen_at_commit.append(
                (runner.build_store.get_build(build_id).state, store.get_name("the-output"))
            )
            return result

        return call

    for method in ("finalize_artifact", "finalize_and_set_name"):
        monkeypatch.setattr(store, method, observed(getattr(store, method)))

    await runner._execute_build(runner.build_store.get_build(build_id))

    [(state, name)] = seen_at_commit
    assert state == "ready"
    assert name is not None and (name.artifact_id, name.version) == ("named", version)


async def test_an_import_that_loses_a_race_leaves_the_winner_intact(served):
    """Two imports of one id@v with different computations, the second landing mid-first.

    The import writes its bytes in a thread, so the other import can commit before the
    first inserts its row. The loser is a 409 and the winner's row reads its own bytes.
    """
    first_bytes = table_to_ipc_bytes(pa.table({"x": [1, 2, 3]}))
    second_bytes = table_to_ipc_bytes(pa.table({"x": [7, 8, 9, 10]}))
    first_meta = {"id": "shared", "version": 1, "provenance_hash": "a" * 64, "created_at": 1.0}
    second_meta = {"id": "shared", "version": 1, "provenance_hash": "b" * 64, "created_at": 1.0}
    blobs = served.blobs
    blobs.gated = "publish"
    async with _client(served) as client:
        first = asyncio.ensure_future(
            client.post("/v1/artifacts/import", files=_files(first_meta, first_bytes))
        )
        try:
            assert await asyncio.to_thread(blobs.entered.wait, 30), "the first import never wrote"
            blobs.gated = None
            second = await client.post(
                "/v1/artifacts/import", files=_files(second_meta, second_bytes)
            )
        finally:
            blobs.release.set()
        first = await first

    assert second.status_code == 200
    assert first.status_code == 409
    row = served.store.get_artifact("shared", 1)
    assert row.provenance_hash == "b" * 64
    assert served.store.read_blob("shared", 1) == second_bytes
    assert row.content_sha256 == hashlib.sha256(second_bytes).hexdigest()
    # The loser wrote under a key of its own, and removed it.
    winner_key = served.store._blob_key("shared", 1)[0]
    assert [p.name for p in served.store.blobs_dir.rglob("*.arrow")] == [f"{winner_key}@v=1.arrow"]


async def test_an_uploaded_artifact_is_ready_only_with_its_name(served, monkeypatch):
    """Finalize runs in a thread, so a reader can look the moment it commits."""
    store = served.store
    version = store.create_artifact(artifact_id="uploaded", provenance_hash="d" * 64)
    store.write_blob("uploaded", version, ARROW)
    seen_at_commit = []

    def observed(finalize):
        def call(*args, **kwargs):
            result = finalize(*args, **kwargs)
            seen_at_commit.append(store.get_name("the-upload"))
            return result

        return call

    for method in ("finalize_artifact", "finalize_and_set_name"):
        monkeypatch.setattr(store, method, observed(getattr(store, method)))
    body = {
        "artifact_id": "uploaded",
        "version": version,
        "arrow_schema": "",
        "row_count": 3,
        "name": "the-upload",
    }

    async with _client(served) as client:
        response = await client.post("/v1/artifacts/finalize", json=body)

    assert response.status_code == 200, response.text
    assert response.json()["name_uri"] == "strata://name/the-upload"
    [name] = seen_at_commit
    assert (name.artifact_id, name.version) == ("uploaded", version)


_NAMED_PUT = {
    "inputs": [],
    "transform": {"executor": "local@v1", "params": {}},
    "data": {"x": [1, 2, 3]},
    "name": "the-put",
}


async def test_a_put_artifact_is_ready_only_with_its_name(served, monkeypatch):
    """The ready state and the name commit together, so no reader sees one without the other."""
    store = served.store
    seen_at_commit = []

    def observed(finalize):
        def call(*args, **kwargs):
            result = finalize(*args, **kwargs)
            seen_at_commit.append((result.state, store.get_name("the-put")))
            return result

        return call

    for method in ("finalize_artifact", "finalize_and_set_name"):
        monkeypatch.setattr(store, method, observed(getattr(store, method)))

    async with _client(served) as client:
        response = await client.put("/v1/artifacts", json=_NAMED_PUT)

    assert response.status_code == 200, response.text
    assert response.json()["name_uri"] == "strata://name/the-put"
    [(state, name)] = seen_at_commit
    assert state == "ready"
    assert name is not None
    uri = f"strata://artifact/{name.artifact_id}@v={name.version}"
    assert response.json()["artifact_uri"] == uri


async def test_a_put_whose_name_cannot_be_written_is_not_left_ready(served, monkeypatch):
    import sqlite3

    store = served.store

    def name_write_fails(*args, **kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(store, "_set_name_in_connection", name_write_fails)
    transport = ASGITransport(app=served.app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.put("/v1/artifacts", json=_NAMED_PUT)

    assert response.status_code == 500
    conn = store._get_connection()
    try:
        states = [row["state"] for row in conn.execute("SELECT state FROM artifact_versions")]
    finally:
        conn.close()
    assert states == ["building"]


async def test_a_put_overtaken_at_finalize_names_the_artifact_that_won(served, monkeypatch):
    """Another writer finalizes the same provenance after the PUT's lookup missed."""
    store = served.store
    publish = store.publish_blob_from_path

    def another_writer_finishes_first(artifact_id, version, path):
        publish(artifact_id, version, path)
        provenance = store.get_artifact(artifact_id, version).provenance_hash
        winner = store.create_artifact(artifact_id="winner", provenance_hash=provenance)
        store.write_blob("winner", winner, ARROW)
        store.finalize_artifact("winner", winner, "", row_count=3, byte_size=len(ARROW))

    monkeypatch.setattr(store, "publish_blob_from_path", another_writer_finishes_first)

    async with _client(served) as client:
        response = await client.put("/v1/artifacts", json=_NAMED_PUT)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["hit"] is True
    assert body["artifact_uri"] == "strata://artifact/winner@v=1"
    name = store.get_name("the-put")
    assert (name.artifact_id, name.version) == ("winner", 1)


async def test_an_offload_borrows_a_server_thread_token_and_keeps_the_request_context(
    served, monkeypatch
):
    """Route offloads count against anyio's default limiter, the one the Postgres pool is
    sized to, and the thread still sees the request's context (its log fields)."""
    from strata.logging import get_request_context

    store = served.store
    version = store.create_artifact(artifact_id="uploaded", provenance_hash="d" * 64)
    store.write_blob("uploaded", version, ARROW)
    body = {"artifact_id": "uploaded", "version": version, "arrow_schema": "", "row_count": 3}
    limiter = anyio.to_thread.current_default_thread_limiter()
    seen: dict = {}
    open_reader = served.blobs.open_blob_reader

    def recording_reader(artifact_id, version):
        seen["request_id"] = get_request_context().get("request_id")
        return open_reader(artifact_id, version)

    monkeypatch.setattr(served.blobs, "open_blob_reader", recording_reader)
    served.blobs.gated = "read"

    async with _client(served) as client:
        task = asyncio.ensure_future(
            client.post("/v1/artifacts/finalize", json=body, headers={"X-Request-ID": "offloaded"})
        )
        try:
            # Waits in the loop's default executor, outside the limiter it inspects.
            assert await asyncio.to_thread(served.blobs.entered.wait, 30), "no read was made"
            borrowed = limiter.borrowed_tokens
        finally:
            served.blobs.release.set()
        response = await task

    assert response.status_code == 200, response.text
    assert borrowed == 1
    assert seen["request_id"] == "offloaded"
