"""Build reliability: lease claiming, heartbeat renewal, orphan recovery, idempotent finalize.

Also checks finalize-and-name is atomic.
"""

import asyncio
import time
import uuid
from unittest.mock import patch

import pytest

from strata.artifact_store import (
    TransformSpec,
    get_artifact_store,
    reset_artifact_store,
)
from strata.transforms.build_store import reset_build_store
from strata.transforms.registry import (
    TransformDefinition,
    TransformRegistry,
    reset_transform_registry,
    set_transform_registry,
)
from strata.transforms.runner import BuildRunner, RunnerConfig
from tests.conftest import seed_build_targets


@pytest.fixture
def artifact_dir(tmp_path):
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir(parents=True)
    return artifact_path


@pytest.fixture
def artifact_store(artifact_dir):
    reset_artifact_store()
    store = get_artifact_store(artifact_dir)
    yield store
    reset_artifact_store()


@pytest.fixture
def build_store(artifact_dir):
    reset_build_store()
    from strata.transforms.build_store import get_build_store

    seed_build_targets(artifact_dir)
    db_path = artifact_dir / "artifacts.sqlite"
    store = get_build_store(db_path)
    yield store
    reset_build_store()


@pytest.fixture
def clock_build_store(artifact_dir):
    """A build store whose clock the test advances by hand.

    Sleeping past a lease measures timer resolution plus a SQLite round-trip, which flakes on
    Windows' ~15ms ``time.time`` granularity.
    """
    from strata.transforms.build_store import BuildStore

    # The shared database, not a file of its own: build rows carry a foreign key into
    # artifact_versions, and the artifacts come from the artifact_store fixture.
    now = [1000.0]
    store = BuildStore(artifact_dir / "artifacts.sqlite", clock=lambda: now[0])

    def advance(seconds: float) -> None:
        now[0] += seconds

    store.advance = advance  # type: ignore[attr-defined]
    return store


@pytest.fixture
def transform_registry():
    reset_transform_registry()
    registry = TransformRegistry(
        enabled=True,
        definitions=[
            TransformDefinition(
                ref="test_executor@v1",
                executor_url="http://localhost:9999",
                timeout_seconds=30.0,
                max_output_bytes=1024 * 1024,
            ),
        ],
    )
    set_transform_registry(registry)
    yield registry
    reset_transform_registry()


class TestLeaseBasedClaiming:
    def test_claim_build_sets_lease_owner(self, build_store, artifact_store):
        """claim_build sets lease_owner and lease_expires_at."""
        artifact_id = str(uuid.uuid4())
        version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash="test-hash",
            transform_spec=TransformSpec(
                executor="test_executor@v1",
                params={},
                inputs=[],
            ),
        )
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id=artifact_id,
            version=version,
            executor_ref="test_executor@v1",
        )

        runner_id = "runner-1"
        lease_duration = 60.0
        result = build_store.claim_build(build_id, runner_id, lease_duration)

        assert result is True
        build = build_store.get_build(build_id)
        assert build.state == "building"
        assert build.lease_owner == runner_id
        assert build.lease_expires_at is not None
        assert build.lease_expires_at > time.time()

    def test_claim_build_fails_if_already_claimed(self, build_store, artifact_store):
        artifact_id = str(uuid.uuid4())
        version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash="test-hash",
            transform_spec=TransformSpec(
                executor="test_executor@v1",
                params={},
                inputs=[],
            ),
        )
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id=artifact_id,
            version=version,
            executor_ref="test_executor@v1",
        )

        result1 = build_store.claim_build(build_id, "runner-1", 60.0)
        assert result1 is True

        result2 = build_store.claim_build(build_id, "runner-2", 60.0)
        assert result2 is False

        build = build_store.get_build(build_id)
        assert build.lease_owner == "runner-1"

    def test_renew_lease_extends_expiry(self, clock_build_store, artifact_store):
        artifact_id = str(uuid.uuid4())
        version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash="test-hash",
        )
        build_id = str(uuid.uuid4())
        clock_build_store.create_build(
            build_id=build_id,
            artifact_id=artifact_id,
            version=version,
            executor_ref="test_executor@v1",
        )
        clock_build_store.claim_build(build_id, "runner-1", 60.0)

        build = clock_build_store.get_build(build_id)
        initial_expiry = build.lease_expires_at

        clock_build_store.advance(30.0)
        result = clock_build_store.renew_lease(build_id, "runner-1", 120.0)

        assert result is True
        build = clock_build_store.get_build(build_id)
        assert build.lease_expires_at > initial_expiry

    def test_renew_lease_fails_for_wrong_owner(self, build_store, artifact_store):
        """renew_lease fails when the caller is not the lease owner."""
        artifact_id = str(uuid.uuid4())
        version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash="test-hash",
        )
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id=artifact_id,
            version=version,
            executor_ref="test_executor@v1",
        )
        build_store.claim_build(build_id, "runner-1", 60.0)

        result = build_store.renew_lease(build_id, "runner-2", 60.0)
        assert result is False


class TestOrphanRecovery:
    def test_list_expired_leases_finds_orphans(self, clock_build_store, artifact_store):
        artifact_id = str(uuid.uuid4())
        version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash="test-hash",
        )
        build_id = str(uuid.uuid4())
        clock_build_store.create_build(
            build_id=build_id,
            artifact_id=artifact_id,
            version=version,
            executor_ref="test_executor@v1",
        )

        clock_build_store.claim_build(build_id, "runner-1", 60.0)

        # Inside the lease: nothing is an orphan yet.
        assert clock_build_store.list_expired_leases() == []
        clock_build_store.advance(59.0)
        assert clock_build_store.list_expired_leases() == []

        # Past it: the build is reclaimable.
        clock_build_store.advance(2.0)
        expired = clock_build_store.list_expired_leases()
        assert len(expired) == 1
        assert expired[0].build_id == build_id

    def test_reclaim_expired_build(self, clock_build_store, artifact_store):
        artifact_id = str(uuid.uuid4())
        version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash="test-hash",
        )
        build_id = str(uuid.uuid4())
        clock_build_store.create_build(
            build_id=build_id,
            artifact_id=artifact_id,
            version=version,
            executor_ref="test_executor@v1",
        )
        clock_build_store.claim_build(build_id, "runner-1", 60.0)

        clock_build_store.advance(61.0)

        result = clock_build_store.reclaim_expired_build(build_id, "runner-2", 60.0)
        assert result is True

        # New owner, with a lease that runs from the reclaim.
        build = clock_build_store.get_build(build_id)
        assert build.lease_owner == "runner-2"
        assert build.lease_expires_at == pytest.approx(1000.0 + 61.0 + 60.0)

    def test_reclaim_fails_for_non_expired_lease(self, build_store, artifact_store):
        artifact_id = str(uuid.uuid4())
        version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash="test-hash",
        )
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id=artifact_id,
            version=version,
            executor_ref="test_executor@v1",
        )
        build_store.claim_build(build_id, "runner-1", 60.0)

        # Lease not expired yet.
        result = build_store.reclaim_expired_build(build_id, "runner-2", 60.0)
        assert result is False

        build = build_store.get_build(build_id)
        assert build.lease_owner == "runner-1"


class TestIdempotentFinalize:
    def test_finalize_same_artifact_twice_is_idempotent(self, artifact_store):
        artifact_id = str(uuid.uuid4())
        provenance_hash = f"hash-{uuid.uuid4()}"

        version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash=provenance_hash,
            tenant="team-a",
        )

        result1 = artifact_store.finalize_artifact(
            artifact_id=artifact_id,
            version=version,
            schema_json='{"type": "struct"}',
            row_count=100,
            byte_size=1000,
        )
        assert result1 is not None
        assert result1.state == "ready"

        result2 = artifact_store.finalize_artifact(
            artifact_id=artifact_id,
            version=version,
            schema_json='{"type": "struct"}',
            row_count=100,
            byte_size=1000,
        )
        assert result2 is not None
        assert result2.id == result1.id
        assert result2.version == result1.version

    def test_duplicate_provenance_returns_existing(self, artifact_store):
        """An existing (tenant, provenance_hash) returns the existing artifact."""
        provenance_hash = f"hash-{uuid.uuid4()}"
        tenant = "team-a"

        artifact1_id = str(uuid.uuid4())
        version1 = artifact_store.create_artifact(
            artifact_id=artifact1_id,
            provenance_hash=provenance_hash,
            tenant=tenant,
        )
        result1 = artifact_store.finalize_artifact(
            artifact_id=artifact1_id,
            version=version1,
            schema_json='{"type": "struct"}',
            row_count=100,
            byte_size=1000,
        )
        assert result1.state == "ready"

        artifact2_id = str(uuid.uuid4())
        version2 = artifact_store.create_artifact(
            artifact_id=artifact2_id,
            provenance_hash=provenance_hash,
            tenant=tenant,
        )

        result2 = artifact_store.finalize_artifact(
            artifact_id=artifact2_id,
            version=version2,
            schema_json='{"type": "struct"}',
            row_count=100,
            byte_size=1000,
        )
        assert result2 is not None
        assert result2.id == artifact1_id  # Returns the first artifact
        assert result2.version == version1

        # The second artifact is overtaken, not failed: it finished.
        artifact2 = artifact_store.get_artifact(artifact2_id, version2)
        assert artifact2.state == "superseded"


class TestAtomicFinalizeAndName:
    def test_finalize_and_set_name_atomic(self, artifact_store):
        artifact_id = str(uuid.uuid4())
        provenance_hash = f"hash-{uuid.uuid4()}"
        tenant = "team-a"
        name = "my-report"

        version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash=provenance_hash,
            tenant=tenant,
        )

        result = artifact_store.finalize_and_set_name(
            artifact_id=artifact_id,
            version=version,
            schema_json='{"type": "struct"}',
            row_count=100,
            byte_size=1000,
            name=name,
            tenant=tenant,
        )

        assert result is not None
        assert result.state == "ready"

        resolved = artifact_store.resolve_name(name, tenant=tenant)
        assert resolved is not None
        assert resolved.id == artifact_id
        assert resolved.version == version

    def test_finalize_and_set_name_points_to_existing_on_duplicate(self, artifact_store):
        """On duplicate provenance the name points to the existing artifact."""
        provenance_hash = f"hash-{uuid.uuid4()}"
        tenant = "team-a"
        name = "my-report"

        artifact1_id = str(uuid.uuid4())
        version1 = artifact_store.create_artifact(
            artifact_id=artifact1_id,
            provenance_hash=provenance_hash,
            tenant=tenant,
        )
        artifact_store.finalize_artifact(
            artifact_id=artifact1_id,
            version=version1,
            schema_json='{"type": "struct"}',
            row_count=100,
            byte_size=1000,
        )

        artifact2_id = str(uuid.uuid4())
        version2 = artifact_store.create_artifact(
            artifact_id=artifact2_id,
            provenance_hash=provenance_hash,
            tenant=tenant,
        )

        result = artifact_store.finalize_and_set_name(
            artifact_id=artifact2_id,
            version=version2,
            schema_json='{"type": "struct"}',
            row_count=100,
            byte_size=1000,
            name=name,
            tenant=tenant,
        )

        assert result is not None
        assert result.id == artifact1_id  # Returns the existing one

        resolved = artifact_store.resolve_name(name, tenant=tenant)
        assert resolved is not None
        assert resolved.id == artifact1_id


class TestBuildRunnerLeaseIntegration:
    """The build runner with leases."""

    @pytest.fixture
    def runner_config(self):
        return RunnerConfig(
            poll_interval_ms=50,
            max_concurrent_builds=5,
            max_builds_per_tenant=2,
            default_timeout_seconds=30.0,
            default_max_output_bytes=1024 * 1024,
            lease_duration_seconds=2.0,
            heartbeat_interval_seconds=0.5,
            runner_id="test-runner-1",
        )

    @pytest.fixture
    def build_runner(
        self, runner_config, artifact_store, build_store, transform_registry, artifact_dir
    ):
        runner = BuildRunner(
            config=runner_config,
            artifact_store=artifact_store,
            build_store=build_store,
            transform_registry=transform_registry,
            artifact_dir=artifact_dir,
        )
        return runner

    def test_runner_has_unique_id(self, build_runner):
        assert build_runner._runner_id == "test-runner-1"

    def test_runner_config_has_lease_settings(self, runner_config):
        assert runner_config.lease_duration_seconds == 2.0
        assert runner_config.heartbeat_interval_seconds == 0.5
        assert runner_config.runner_id == "test-runner-1"

    def test_runner_generates_id_if_not_provided(
        self, artifact_store, build_store, transform_registry, artifact_dir
    ):
        config = RunnerConfig()
        runner = BuildRunner(
            config=config,
            artifact_store=artifact_store,
            build_store=build_store,
            transform_registry=transform_registry,
            artifact_dir=artifact_dir,
        )
        assert runner._runner_id.startswith("runner-")
        assert len(runner._runner_id) > 7  # "runner-" + UUID fragment

    @pytest.mark.asyncio
    async def test_heartbeat_renews_leases(self, build_runner, artifact_store, build_store):
        """The heartbeat loop renews leases for running builds."""
        artifact_id = str(uuid.uuid4())
        version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash=f"hash-{uuid.uuid4()}",
            transform_spec=TransformSpec(
                executor="test_executor@v1",
                params={},
                inputs=[],
            ),
        )
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id=artifact_id,
            version=version,
            executor_ref="test_executor@v1",
            executor_url="http://localhost:9999",
        )

        original_renew = build_store.renew_lease
        renew_calls = []

        def track_renew(bid, owner, duration):
            result = original_renew(bid, owner, duration)
            renew_calls.append((bid, owner, result))
            return result

        with patch.object(build_store, "renew_lease", track_renew):
            # Run slowly so the heartbeat fires.
            async def slow_execute(build, already_claimed=False):
                if not already_claimed:
                    build_store.claim_build(
                        build.build_id,
                        build_runner._runner_id,
                        build_runner.config.lease_duration_seconds,
                    )
                await asyncio.sleep(0.8)  # Let the heartbeat run.
                build_store.complete_build(build.build_id)

            with patch.object(build_runner, "_execute_build", slow_execute):
                await build_runner.start()
                await asyncio.sleep(1.2)
                await build_runner.stop()

        successful_renewals = [c for c in renew_calls if c[2]]
        assert len(successful_renewals) >= 1
