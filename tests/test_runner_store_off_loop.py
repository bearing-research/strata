"""The build runner's metadata and blob calls run off the event loop.

The runner shares the server's loop. A store call that waits (a SQLite write lock, a Postgres
round trip, a blob delete on S3) made inline stalls every request the server is serving. Each
test blocks one call on an event and checks that the loop runs another coroutine meanwhile.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from types import SimpleNamespace

import anyio.to_thread
import pyarrow as pa
import pytest

from strata.artifact_store import TransformSpec, get_artifact_store, reset_artifact_store
from strata.transforms.build_store import get_build_store, reset_build_store
from strata.transforms.registry import TransformDefinition, TransformRegistry
from strata.transforms.runner import BuildRunner, RunnerConfig
from tests.conftest import table_to_ipc_bytes

ARROW = table_to_ipc_bytes(pa.table({"x": [1, 2, 3]}))


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
            # After the call, so a test that waits on done sees what it wrote.
            gate.done.set()

    monkeypatch.setattr(obj, method, gated)
    return gate


async def _loop_ran_while_blocked(gate: SimpleNamespace, call):
    """Run *call* with the gated method blocked: whether this coroutine ran meanwhile."""
    task = asyncio.ensure_future(call)
    try:
        assert await asyncio.to_thread(gate.entered.wait, 30), "the gated call was never made"
        ran_while_blocked = not gate.done.is_set()
    finally:
        gate.release.set()
    return ran_while_blocked, task


@pytest.fixture
def runner(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    reset_artifact_store()
    reset_build_store()
    store = get_artifact_store(artifact_dir)
    registry = TransformRegistry(
        enabled=True,
        definitions=[TransformDefinition(ref="test_sql@*", executor_url="http://executor")],
    )
    runner = BuildRunner(
        config=RunnerConfig(poll_interval_ms=10),
        artifact_store=store,
        build_store=get_build_store(artifact_dir / "artifacts.sqlite", dialect=store.dialect),
        transform_registry=registry,
        artifact_dir=artifact_dir,
    )
    yield runner
    reset_build_store()
    reset_artifact_store()


def _queue_build(runner: BuildRunner, *, name: str | None = None) -> tuple[str, int]:
    store = runner.artifact_store
    artifact_id = uuid.uuid4().hex
    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=uuid.uuid4().hex * 2,
        transform_spec=TransformSpec(executor="service://test_sql@v1", params={}, inputs=[]),
        input_versions={},
    )
    build_id = str(uuid.uuid4())
    runner.build_store.create_build(
        build_id=build_id,
        artifact_id=artifact_id,
        version=version,
        executor_ref="test_sql@v1",
        executor_url="http://executor",
        name=name,
    )
    return build_id, version


def _succeeding_executor(runner: BuildRunner, tmp_path):
    output = tmp_path / "output.arrow"

    async def executor(**_kwargs):
        output.write_bytes(ARROW)
        return output, None

    runner._call_executor = executor


async def _stopped(runner: BuildRunner, task: asyncio.Task) -> None:
    runner._running = False
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.parametrize("method", ["list_pending_builds", "list_expired_leases"])
async def test_the_poll_reads_the_build_store_off_the_loop(runner, monkeypatch, method):
    gate = _gate(monkeypatch, runner.build_store, method)
    runner._running = True

    ran, task = await _loop_ran_while_blocked(gate, runner._run_loop())

    await _stopped(runner, task)
    assert ran


async def test_the_poll_borrows_a_server_thread_token(runner, monkeypatch):
    """The runner's store calls count against anyio's default limiter, the one the Postgres
    pool is sized to, not the loop's separate default executor."""
    gate = _gate(monkeypatch, runner.build_store, "list_pending_builds")
    limiter = anyio.to_thread.current_default_thread_limiter()
    runner._running = True

    task = asyncio.ensure_future(runner._run_loop())
    try:
        assert await asyncio.to_thread(gate.entered.wait, 30), "the poll never read the store"
        borrowed = limiter.borrowed_tokens
    finally:
        gate.release.set()

    await _stopped(runner, task)
    assert borrowed == 1


async def test_an_orphan_is_reclaimed_off_the_loop(runner, monkeypatch):
    build_id, _ = _queue_build(runner)
    # A lease that has already lapsed: the orphan of a runner that died.
    assert runner.build_store.claim_build(build_id, "runner-dead", lease_duration_seconds=-1)
    submitted = []
    monkeypatch.setattr(
        runner, "_submit_build", lambda build, already_claimed=False: submitted.append(build)
    )
    gate = _gate(monkeypatch, runner.build_store, "reclaim_expired_build")
    runner._running = True

    ran, task = await _loop_ran_while_blocked(gate, runner._run_loop())
    while not submitted:
        await asyncio.sleep(0)

    await _stopped(runner, task)
    assert ran
    assert [b.build_id for b in submitted] == [build_id]
    assert runner.build_store.get_build(build_id).lease_owner == runner._runner_id


async def test_the_heartbeat_renews_leases_off_the_loop(runner, monkeypatch):
    build_id, _ = _queue_build(runner)
    assert runner.build_store.claim_build(build_id, runner._runner_id, lease_duration_seconds=1)
    before = runner.build_store.get_build(build_id).lease_expires_at
    runner._running_builds.add(build_id)
    gate = _gate(monkeypatch, runner.build_store, "renew_lease")
    runner._running = True

    ran, task = await _loop_ran_while_blocked(gate, runner._heartbeat_loop())
    await asyncio.to_thread(gate.done.wait, 30)

    await _stopped(runner, task)
    assert ran
    assert runner.build_store.get_build(build_id).lease_expires_at > before


@pytest.mark.parametrize("finished", [False, True])
async def test_a_build_that_finishes_mid_renewal_is_not_a_lost_lease(
    runner, monkeypatch, caplog, finished
):
    """The loop runs while a renewal is in flight, so the build can finish meanwhile."""
    build_id, _ = _queue_build(runner)
    assert runner.build_store.claim_build(build_id, runner._runner_id)
    runner._running_builds.add(build_id)
    renew_lease = runner.build_store.renew_lease

    def renew_as_the_build_completes(*args, **kwargs):
        runner.build_store.complete_build(build_id)
        if finished:
            runner._running_builds.discard(build_id)
        runner._running = False  # one heartbeat
        return renew_lease(*args, **kwargs)

    monkeypatch.setattr(runner.build_store, "renew_lease", renew_as_the_build_completes)
    runner.config.heartbeat_interval_seconds = 0
    runner._running = True

    await runner._heartbeat_loop()

    lost = [r for r in caplog.records if "Failed to renew lease" in r.getMessage()]
    assert len(lost) == (0 if finished else 1)


@pytest.mark.parametrize(
    ("store", "method"),
    [
        ("build_store", "claim_build"),
        ("build_store", "get_build"),
        ("artifact_store", "get_artifact"),
        ("build_store", "record_attempt"),
        ("build_store", "complete_within"),
    ],
)
async def test_a_build_reads_and_writes_the_store_off_the_loop(
    runner, monkeypatch, tmp_path, store, method
):
    build_id, version = _queue_build(runner)
    queued = runner.build_store.get_build(build_id)
    _succeeding_executor(runner, tmp_path)
    gate = _gate(monkeypatch, getattr(runner, store), method)

    ran, task = await _loop_ran_while_blocked(gate, runner._execute_build(queued))
    await task

    assert ran
    build = runner.build_store.get_build(build_id)
    assert build.state == "ready"
    assert runner.artifact_store.get_artifact(build.artifact_id, version).state == "ready"


@pytest.mark.parametrize(
    ("store", "method"), [("build_store", "fail_build"), ("artifact_store", "fail_artifact")]
)
async def test_a_failed_build_is_recorded_off_the_loop(
    runner, monkeypatch, tmp_path, store, method
):
    build_id, version = _queue_build(runner)

    async def executor(**_kwargs):
        raise ValueError("the executor failed")

    runner._call_executor = executor
    gate = _gate(monkeypatch, getattr(runner, store), method)

    ran, task = await _loop_ran_while_blocked(
        gate, runner._execute_build(runner.build_store.get_build(build_id))
    )
    await task

    assert ran
    assert runner.build_store.get_build(build_id).state == "failed"
    artifact_id = runner.build_store.get_build(build_id).artifact_id
    assert runner.artifact_store.get_artifact(artifact_id, version).state == "failed"


async def test_a_failure_cancelled_midway_still_fails_the_artifact(runner, monkeypatch, tmp_path):
    """A stopping runner can cancel a build while its failure is being recorded.

    The build and its artifact fail together: a failed build over a ``building``
    artifact would leave the artifact building for good.
    """
    build_id, version = _queue_build(runner)

    async def executor(**_kwargs):
        raise ValueError("the executor failed")

    runner._call_executor = executor
    failed_artifact = threading.Event()
    fail_artifact = runner.artifact_store.fail_artifact

    def recording(*args, **kwargs):
        fail_artifact(*args, **kwargs)
        failed_artifact.set()

    monkeypatch.setattr(runner.artifact_store, "fail_artifact", recording)
    gate = _gate(monkeypatch, runner.build_store, "fail_build")
    task = asyncio.ensure_future(runner._execute_build(runner.build_store.get_build(build_id)))
    try:
        assert await asyncio.to_thread(gate.entered.wait, 30), "the failure was never recorded"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        gate.release.set()
    assert await asyncio.to_thread(failed_artifact.wait, 30), "the artifact was left building"

    artifact_id = runner.build_store.get_build(build_id).artifact_id
    assert runner.build_store.get_build(build_id).state == "failed"
    assert runner.artifact_store.get_artifact(artifact_id, version).state == "failed"


async def test_a_lost_lease_drops_its_attempt_off_the_loop(runner, monkeypatch, tmp_path):
    """Another runner took the build over mid-run: this attempt's bytes are deleted."""
    build_id, version = _queue_build(runner)
    output = tmp_path / "output.arrow"

    async def executor(**_kwargs):
        conn = runner.build_store._get_connection()
        try:
            conn.execute(
                "UPDATE artifact_builds SET lease_owner = 'runner-b' WHERE build_id = ?",
                (build_id,),
            )
            conn.commit()
        finally:
            conn.close()
        output.write_bytes(ARROW)
        return output, None

    runner._call_executor = executor
    deleted = []
    delete_attempt_blob = runner.artifact_store.delete_attempt_blob

    def recording(artifact_id, version, attempt):
        deleted.append((artifact_id, version, attempt))
        delete_attempt_blob(artifact_id, version, attempt)

    monkeypatch.setattr(runner.artifact_store, "delete_attempt_blob", recording)
    gate = _gate(monkeypatch, runner.artifact_store, "delete_attempt_blob")

    ran, task = await _loop_ran_while_blocked(
        gate, runner._execute_build(runner.build_store.get_build(build_id))
    )
    await task

    assert ran
    [(artifact_id, deleted_version, attempt)] = deleted
    assert not runner.artifact_store.blob_exists(artifact_id, deleted_version, attempt=attempt)
    build = runner.build_store.get_build(build_id)
    assert (build.state, build.lease_owner) == ("building", "runner-b")


async def test_a_name_input_is_resolved_off_the_loop(runner, monkeypatch):
    store = runner.artifact_store
    version = store.create_artifact(artifact_id="input", provenance_hash="a" * 64)
    store.write_blob("input", version, ARROW)
    store.finalize_artifact(
        artifact_id="input", version=version, schema_json="", row_count=3, byte_size=len(ARROW)
    )
    store.set_name("the-input", "input", version)
    gate = _gate(monkeypatch, store, "resolve_name")
    temp_files: list = []

    ran, task = await _loop_ran_while_blocked(
        gate, runner._acquire_input("strata://name/the-input", temp_files)
    )
    path = await task

    assert ran
    assert path.read_bytes() == ARROW
    path.unlink()
