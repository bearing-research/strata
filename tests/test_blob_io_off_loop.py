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
