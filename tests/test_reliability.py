"""Tests for build reliability features.

These tests verify:
1. Lease-based claiming prevents duplicate execution
2. Heartbeat renewal keeps leases alive
3. Orphan recovery reclaims expired leases
4. Idempotent finalize prevents duplicate artifacts
5. Atomic finalize-and-name ensures consistency
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
    """Create a temporary artifact directory."""
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir(parents=True)
    return artifact_path


@pytest.fixture
def artifact_store(artifact_dir):
    """Create a temporary artifact store."""
    reset_artifact_store()
    store = get_artifact_store(artifact_dir)
    yield store
    reset_artifact_store()


@pytest.fixture
def build_store(artifact_dir):
    """Create a temporary build store."""
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

    Lease expiry compares two stored timestamps, so a test that sleeps past a
    lease is really asserting the runner's timer resolution plus the cost of a
    SQLite round-trip. On Windows (~15ms ``time.time`` granularity) that flaked
    both directions: the pre-sleep "nothing has expired yet" check failed
    because more than the lease duration had already elapsed, and widening the
    lease only moved which of the two assertions broke (#627).
    """
    from strata.transforms.build_store import BuildStore

    # The shared database, not a file of its own: build rows carry a foreign
    # key into artifact_versions, and these tests create their artifacts
    # through the artifact_store fixture. A separate file puts the two halves
    # in different databases, where the constraint cannot resolve at all.
    now = [1000.0]
    store = BuildStore(artifact_dir / "artifacts.sqlite", clock=lambda: now[0])

    def advance(seconds: float) -> None:
        now[0] += seconds

    store.advance = advance  # type: ignore[attr-defined]
    return store


@pytest.fixture
def transform_registry():
    """Create a test transform registry."""
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
    """Tests for lease-based build claiming."""

    def test_claim_build_sets_lease_owner(self, build_store, artifact_store):
        """claim_build should set lease_owner and lease_expires_at."""
        # Create artifact and build
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

        # Claim the build
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
        """claim_build should fail if build is already claimed."""
        # Create artifact and build
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

        # First claim succeeds
        result1 = build_store.claim_build(build_id, "runner-1", 60.0)
        assert result1 is True

        # Second claim fails
        result2 = build_store.claim_build(build_id, "runner-2", 60.0)
        assert result2 is False

        # Original claim still valid
        build = build_store.get_build(build_id)
        assert build.lease_owner == "runner-1"

    def test_renew_lease_extends_expiry(self, clock_build_store, artifact_store):
        """renew_lease should extend the lease expiry time."""
        # Create and claim build
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

        # Get initial expiry
        build = clock_build_store.get_build(build_id)
        initial_expiry = build.lease_expires_at

        clock_build_store.advance(30.0)
        result = clock_build_store.renew_lease(build_id, "runner-1", 120.0)

        assert result is True
        build = clock_build_store.get_build(build_id)
        assert build.lease_expires_at > initial_expiry

    def test_renew_lease_fails_for_wrong_owner(self, build_store, artifact_store):
        """renew_lease should fail if caller is not the lease owner."""
        # Create and claim build
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

        # Different runner tries to renew
        result = build_store.renew_lease(build_id, "runner-2", 60.0)
        assert result is False


class TestOrphanRecovery:
    """Tests for orphan build recovery."""

    def test_list_expired_leases_finds_orphans(self, clock_build_store, artifact_store):
        """list_expired_leases should find builds with expired leases."""
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
        """reclaim_expired_build should take over orphaned builds."""
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

        # Reclaim as different runner
        result = clock_build_store.reclaim_expired_build(build_id, "runner-2", 60.0)
        assert result is True

        # Verify new owner, with a lease that runs from the reclaim.
        build = clock_build_store.get_build(build_id)
        assert build.lease_owner == "runner-2"
        assert build.lease_expires_at == pytest.approx(1000.0 + 61.0 + 60.0)

    def test_reclaim_fails_for_non_expired_lease(self, build_store, artifact_store):
        """reclaim_expired_build should fail if lease is still valid."""
        # Create and claim build with long lease
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

        # Try to reclaim immediately (lease not expired)
        result = build_store.reclaim_expired_build(build_id, "runner-2", 60.0)
        assert result is False

        # Original owner still valid
        build = build_store.get_build(build_id)
        assert build.lease_owner == "runner-1"


class TestIdempotentFinalize:
    """Tests for idempotent artifact finalization."""

    def test_finalize_same_artifact_twice_is_idempotent(self, artifact_store):
        """Calling finalize_artifact twice should be idempotent."""
        artifact_id = str(uuid.uuid4())
        provenance_hash = f"hash-{uuid.uuid4()}"

        version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash=provenance_hash,
            tenant="team-a",
        )

        # First finalize
        result1 = artifact_store.finalize_artifact(
            artifact_id=artifact_id,
            version=version,
            schema_json='{"type": "struct"}',
            row_count=100,
            byte_size=1000,
        )
        assert result1 is not None
        assert result1.state == "ready"

        # Second finalize should return same artifact
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
        """If same (tenant, provenance_hash) exists, return existing artifact."""
        provenance_hash = f"hash-{uuid.uuid4()}"
        tenant = "team-a"

        # Create and finalize first artifact
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

        # Create second artifact with same provenance
        artifact2_id = str(uuid.uuid4())
        version2 = artifact_store.create_artifact(
            artifact_id=artifact2_id,
            provenance_hash=provenance_hash,
            tenant=tenant,
        )

        # Finalize should return existing artifact
        result2 = artifact_store.finalize_artifact(
            artifact_id=artifact2_id,
            version=version2,
            schema_json='{"type": "struct"}',
            row_count=100,
            byte_size=1000,
        )
        assert result2 is not None
        assert result2.id == artifact1_id  # Returns first artifact
        assert result2.version == version1

        # Second artifact should be marked failed
        artifact2 = artifact_store.get_artifact(artifact2_id, version2)
        assert artifact2.state == "failed"


class TestAtomicFinalizeAndName:
    """Tests for atomic finalize-and-set-name."""

    def test_finalize_and_set_name_atomic(self, artifact_store):
        """finalize_and_set_name should set name atomically with finalize."""
        artifact_id = str(uuid.uuid4())
        provenance_hash = f"hash-{uuid.uuid4()}"
        tenant = "team-a"
        name = "my-report"

        version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash=provenance_hash,
            tenant=tenant,
        )

        # Finalize and set name
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

        # Name should be set
        resolved = artifact_store.resolve_name(name, tenant=tenant)
        assert resolved is not None
        assert resolved.id == artifact_id
        assert resolved.version == version

    def test_finalize_and_set_name_points_to_existing_on_duplicate(self, artifact_store):
        """If duplicate provenance, name should point to existing artifact."""
        provenance_hash = f"hash-{uuid.uuid4()}"
        tenant = "team-a"
        name = "my-report"

        # Create and finalize first artifact
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

        # Create second artifact with same provenance
        artifact2_id = str(uuid.uuid4())
        version2 = artifact_store.create_artifact(
            artifact_id=artifact2_id,
            provenance_hash=provenance_hash,
            tenant=tenant,
        )

        # Finalize and set name - should point to existing
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
        assert result.id == artifact1_id  # Returns existing

        # Name should point to existing artifact
        resolved = artifact_store.resolve_name(name, tenant=tenant)
        assert resolved is not None
        assert resolved.id == artifact1_id


class TestBuildRunnerLeaseIntegration:
    """Integration tests for build runner with leases."""

    @pytest.fixture
    def runner_config(self):
        """Create a test runner config."""
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
        """Create a test build runner."""
        runner = BuildRunner(
            config=runner_config,
            artifact_store=artifact_store,
            build_store=build_store,
            transform_registry=transform_registry,
            artifact_dir=artifact_dir,
        )
        return runner

    def test_runner_has_unique_id(self, build_runner):
        """Build runner should have a unique ID."""
        assert build_runner._runner_id == "test-runner-1"

    def test_runner_config_has_lease_settings(self, runner_config):
        """Runner config should have lease settings."""
        assert runner_config.lease_duration_seconds == 2.0
        assert runner_config.heartbeat_interval_seconds == 0.5
        assert runner_config.runner_id == "test-runner-1"

    def test_runner_generates_id_if_not_provided(
        self, artifact_store, build_store, transform_registry, artifact_dir
    ):
        """Runner should generate unique ID if not provided."""
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
        """Heartbeat loop should renew leases for running builds."""
        # Create artifact and build
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

        # Track lease renewals
        original_renew = build_store.renew_lease
        renew_calls = []

        def track_renew(bid, owner, duration):
            result = original_renew(bid, owner, duration)
            renew_calls.append((bid, owner, result))
            return result

        with patch.object(build_store, "renew_lease", track_renew):
            # Mock execute to run slowly, allowing heartbeats
            async def slow_execute(build, already_claimed=False):
                # Actually claim the build first if not already claimed
                if not already_claimed:
                    build_store.claim_build(
                        build.build_id,
                        build_runner._runner_id,
                        build_runner.config.lease_duration_seconds,
                    )
                await asyncio.sleep(0.8)  # Allow heartbeat to run
                build_store.complete_build(build.build_id)

            with patch.object(build_runner, "_execute_build", slow_execute):
                await build_runner.start()
                await asyncio.sleep(1.2)
                await build_runner.stop()

        # Verify renew_lease was called
        successful_renewals = [c for c in renew_calls if c[2]]
        assert len(successful_renewals) >= 1
