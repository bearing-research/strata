"""Replay BuildLease.tla's runner-path counterexamples against BuildRunner.

Each test passes while the bug exists: it asserts the violating outcome.
Once a fix lands, invert the assertion and move the test into ``tests/``.

    uv run pytest formal/ -v
"""

# Fixtures are imported from tests/ and then requested by name.
# ruff: noqa: F811

from __future__ import annotations

import time

import pyarrow as pa
import pytest

from strata.transforms.runner import BuildRunner, RunnerConfig
from tests.test_build_runner import (  # noqa: F401  (fixtures)
    artifact_dir,
    artifact_store,
    build_store,
    create_arrow_ipc_bytes,
    create_test_artifact,
    transform_registry,
)


def _runner(runner_id, artifact_store, build_store, transform_registry, artifact_dir):
    return BuildRunner(
        config=RunnerConfig(runner_id=runner_id),
        artifact_store=artifact_store,
        build_store=build_store,
        transform_registry=transform_registry,
        artifact_dir=artifact_dir,
    )


def _steal_lease(build_store, build_id, new_owner):
    """The lease runs out (GC pause, blocked loop) and another node reclaims it."""
    conn = build_store._get_connection()
    conn.execute(
        "UPDATE artifact_builds SET lease_expires_at = ? WHERE build_id = ?",
        (time.time() - 1.0, build_id),
    )
    conn.commit()
    conn.close()
    assert build_store.reclaim_expired_build(build_id, new_lease_owner=new_owner)


def _executor(tmp_path, name, values, before_return=None):
    """Stand-in for _call_executor: optional side effect, then an Arrow file."""

    async def call(**_kwargs):
        if before_return is not None:
            before_return()
        path = tmp_path / f"{name}.arrow"
        path.write_bytes(create_arrow_ipc_bytes({"x": values}))
        return path, ""

    return call


@pytest.fixture
def two_runners(artifact_store, build_store, transform_registry, artifact_dir):
    args = (artifact_store, build_store, transform_registry, artifact_dir)
    return _runner("A", *args), _runner("B", *args)


async def test_stale_runner_publishes_then_winner_rewrites_ready_bytes(
    two_runners, artifact_store, build_store, tmp_path
):
    """Build_RunnerPath / ReadyBytesStable and NoStaleBytesPublished.

    A loses its lease mid-build, but publish_blob_from_path and
    finalize_artifact are unfenced, so its bytes become the ready artifact
    and their digest is recorded. Only complete_build is fenced, and it
    rejects A after the fact. B, the rightful owner, then writes its own
    bytes over the ready artifact; finalize is an idempotent no-op, so the
    recorded digest stays A's.
    """
    a, b = two_runners
    artifact_id, version, build_id = create_test_artifact(artifact_store, build_store)

    a._call_executor = _executor(
        tmp_path, "a", [1], before_return=lambda: _steal_lease(build_store, build_id, "B")
    )
    await a._execute_build(build_store.get_build(build_id))

    # A was told it lost, yet its result is already the ready artifact.
    assert build_store.get_build(build_id).state == "building"
    assert artifact_store.get_artifact(artifact_id, version).state == "ready"
    assert _values(artifact_store, artifact_id, version) == [1]

    b._call_executor = _executor(tmp_path, "b", [2])
    await b._execute_build(build_store.get_build(build_id), already_claimed=True)

    assert build_store.get_build(build_id).state == "ready"
    # Same ready (id, version), different bytes, and the store notices.
    assert _values(artifact_store, artifact_id, version) == [2]
    problems = {
        f["problem"] for f in artifact_store.verify_artifacts() if f["artifact_id"] == artifact_id
    }
    assert problems == {"digest_mismatch"}


async def test_stale_runner_failure_kills_the_takeover(
    two_runners, artifact_store, build_store, tmp_path
):
    """Build_RunnerPath / OnlyLeaseHolderFails.

    A's executor times out after B has reclaimed the build. fail_build and
    fail_artifact don't check the lease, so A fails the build B owns, and
    B, which would have succeeded, finds it failed and gives up.
    """
    a, b = two_runners
    artifact_id, version, build_id = create_test_artifact(artifact_store, build_store)

    def steal_then_time_out():
        _steal_lease(build_store, build_id, "B")
        raise TimeoutError("executor timed out")

    a._call_executor = _executor(tmp_path, "a", [1], before_return=steal_then_time_out)
    await a._execute_build(build_store.get_build(build_id))

    build = build_store.get_build(build_id)
    assert build.state == "failed"
    assert build.lease_owner == "B"  # failed by a runner that did not hold it

    b._call_executor = _executor(tmp_path, "b", [2])
    await b._execute_build(build_store.get_build(build_id), already_claimed=True)

    assert build_store.get_build(build_id).state == "failed"
    assert artifact_store.get_artifact(artifact_id, version).state == "failed"


def _values(artifact_store, artifact_id, version):
    data = artifact_store.read_blob(artifact_id, version)
    return pa.ipc.open_stream(data).read_all().column("x").to_pylist()
