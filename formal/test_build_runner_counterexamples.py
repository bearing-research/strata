"""Replay BuildLease.tla's runner-path counterexamples against BuildRunner.

Each test passes while the bug exists: it asserts the violating outcome.
Once a fix lands, invert the assertion and move the test into ``tests/``.

    uv run pytest formal/ -v
"""

# Fixtures are imported from tests/ and then requested by name.
# ruff: noqa: F811

from __future__ import annotations

import pyarrow as pa

from tests.test_build_runner import (  # noqa: F401  (fixtures)
    artifact_dir,
    artifact_store,
    build_store,
    create_test_artifact,
    fake_executor,
    steal_lease,
    transform_registry,
    two_runners,
)


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

    a._call_executor = fake_executor(
        tmp_path, "a", [1], before_return=lambda: steal_lease(build_store, build_id, "B")
    )
    await a._execute_build(build_store.get_build(build_id))

    # A was told it lost, yet its result is already the ready artifact.
    assert build_store.get_build(build_id).state == "building"
    assert artifact_store.get_artifact(artifact_id, version).state == "ready"
    assert _values(artifact_store, artifact_id, version) == [1]

    b._call_executor = fake_executor(tmp_path, "b", [2])
    await b._execute_build(build_store.get_build(build_id), already_claimed=True)

    assert build_store.get_build(build_id).state == "ready"
    # Same ready (id, version), different bytes, and the store notices.
    assert _values(artifact_store, artifact_id, version) == [2]
    problems = {
        f["problem"] for f in artifact_store.verify_artifacts() if f["artifact_id"] == artifact_id
    }
    assert problems == {"digest_mismatch"}


def _values(artifact_store, artifact_id, version):
    data = artifact_store.read_blob(artifact_id, version)
    return pa.ipc.open_stream(data).read_all().column("x").to_pylist()
