"""Tests for build state store (server-mode transforms)."""

import time
import uuid

import pytest

from strata.transforms.build_store import (
    BuildStore,
    get_build_store,
    reset_build_store,
)
from tests.conftest import seed_build_targets


@pytest.fixture
def build_store(tmp_path):
    """Create a temporary build store over a database that also holds artifacts.

    ``artifact_builds`` has a foreign key into ``artifact_versions``, so the
    build store cannot live in a file of its own: the constraint would have
    nothing to resolve against. Production shares one database for exactly
    this reason.
    """
    seed_build_targets(tmp_path)
    store = BuildStore(tmp_path / "artifacts.sqlite")
    yield store


class TestBuildStore:
    """Tests for BuildStore."""

    def test_create_build(self, build_store):
        """Create a new build record."""
        build_id = str(uuid.uuid4())
        state = build_store.create_build(
            build_id=build_id,
            artifact_id="art-123",
            version=1,
            executor_ref="duckdb_sql@v1",
            executor_url="http://executor:8080",
            tenant_id="tenant-1",
            principal_id="user-1",
        )

        assert state.build_id == build_id
        assert state.artifact_id == "art-123"
        assert state.version == 1
        assert state.state == "pending"
        assert state.executor_ref == "duckdb_sql@v1"
        assert state.executor_url == "http://executor:8080"
        assert state.tenant_id == "tenant-1"
        assert state.principal_id == "user-1"
        assert state.created_at > 0
        assert state.started_at is None
        assert state.completed_at is None

    def test_get_build(self, build_store):
        """Get build by ID."""
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id="art-123",
            version=1,
            executor_ref="duckdb_sql@v1",
        )

        retrieved = build_store.get_build(build_id)
        assert retrieved is not None
        assert retrieved.build_id == build_id
        assert retrieved.state == "pending"

    def test_update_build_output(self, build_store):
        """Build output can be repointed to the canonical artifact."""
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id="art-123",
            version=1,
            executor_ref="duckdb_sql@v1",
        )

        success = build_store.update_build_output(build_id, "canonical-artifact", 7)
        assert success

        updated = build_store.get_build(build_id)
        assert updated is not None
        assert updated.artifact_id == "canonical-artifact"
        assert updated.version == 7

    def test_get_build_not_found(self, build_store):
        """Get build returns None for missing ID."""
        result = build_store.get_build("nonexistent")
        assert result is None

    def test_start_build(self, build_store):
        """Start a pending build."""
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id="art-123",
            version=1,
            executor_ref="duckdb_sql@v1",
        )

        success = build_store.start_build(build_id)
        assert success

        state = build_store.get_build(build_id)
        assert state.state == "building"
        assert state.started_at is not None

    def test_start_build_wrong_state(self, build_store):
        """Start fails for non-pending builds."""
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id="art-123",
            version=1,
            executor_ref="duckdb_sql@v1",
        )

        # Start once
        build_store.start_build(build_id)

        # Try to start again (should fail)
        success = build_store.start_build(build_id)
        assert not success

    def test_complete_build(self, build_store):
        """Complete a building build."""
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id="art-123",
            version=1,
            executor_ref="duckdb_sql@v1",
        )
        build_store.start_build(build_id)

        success = build_store.complete_build(build_id, output_byte_count=1024)
        assert success

        state = build_store.get_build(build_id)
        assert state.state == "ready"
        assert state.completed_at is not None
        assert state.output_byte_count == 1024

    def test_complete_build_wrong_state(self, build_store):
        """Complete fails for non-building builds."""
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id="art-123",
            version=1,
            executor_ref="duckdb_sql@v1",
        )

        # Try to complete without starting
        success = build_store.complete_build(build_id)
        assert not success

    def test_fail_build(self, build_store):
        """Fail a building build."""
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id="art-123",
            version=1,
            executor_ref="duckdb_sql@v1",
        )
        build_store.start_build(build_id)

        success = build_store.fail_build(
            build_id,
            error_message="Executor timeout",
            error_code="EXECUTOR_TIMEOUT",
        )
        assert success

        state = build_store.get_build(build_id)
        assert state.state == "failed"
        assert state.completed_at is not None
        assert state.error_message == "Executor timeout"
        assert state.error_code == "EXECUTOR_TIMEOUT"

    def test_fail_pending_build(self, build_store):
        """Can fail a pending build (before it starts)."""
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id="art-123",
            version=1,
            executor_ref="duckdb_sql@v1",
        )

        success = build_store.fail_build(
            build_id,
            error_message="Transform not allowed",
            error_code="TRANSFORM_NOT_ALLOWED",
        )
        assert success

        state = build_store.get_build(build_id)
        assert state.state == "failed"

    def test_list_pending_builds(self, build_store):
        """List pending builds in order."""
        # Create 3 builds
        ids = []
        for i in range(3):
            build_id = str(uuid.uuid4())
            ids.append(build_id)
            build_store.create_build(
                build_id=build_id,
                artifact_id=f"art-{i}",
                version=1,
                executor_ref="duckdb_sql@v1",
            )
            time.sleep(0.01)  # Ensure different timestamps

        # Start one build
        build_store.start_build(ids[1])

        # List pending (should only have 2)
        pending = build_store.list_pending_builds()
        assert len(pending) == 2
        assert pending[0].build_id == ids[0]  # Oldest first
        assert pending[1].build_id == ids[2]

    def test_list_builds_by_tenant(self, build_store):
        """List builds for a specific tenant."""
        # Create builds for different tenants
        build_store.create_build(
            build_id=str(uuid.uuid4()),
            artifact_id="art-1",
            version=1,
            executor_ref="duckdb_sql@v1",
            tenant_id="tenant-1",
        )
        build_store.create_build(
            build_id=str(uuid.uuid4()),
            artifact_id="art-2",
            version=1,
            executor_ref="duckdb_sql@v1",
            tenant_id="tenant-2",
        )
        build_store.create_build(
            build_id=str(uuid.uuid4()),
            artifact_id="art-3",
            version=1,
            executor_ref="duckdb_sql@v1",
            tenant_id="tenant-1",
        )

        # List for tenant-1
        builds = build_store.list_builds_by_tenant("tenant-1")
        assert len(builds) == 2

        # List for tenant-2
        builds = build_store.list_builds_by_tenant("tenant-2")
        assert len(builds) == 1

    def test_list_builds_by_tenant_with_state_filter(self, build_store):
        """List builds with state filter."""
        build_id_1 = str(uuid.uuid4())
        build_id_2 = str(uuid.uuid4())

        build_store.create_build(
            build_id=build_id_1,
            artifact_id="art-1",
            version=1,
            executor_ref="duckdb_sql@v1",
            tenant_id="tenant-1",
        )
        build_store.create_build(
            build_id=build_id_2,
            artifact_id="art-2",
            version=1,
            executor_ref="duckdb_sql@v1",
            tenant_id="tenant-1",
        )

        # Start one build
        build_store.start_build(build_id_1)

        # Filter by state
        pending = build_store.list_builds_by_tenant("tenant-1", state="pending")
        assert len(pending) == 1
        assert pending[0].build_id == build_id_2

        building = build_store.list_builds_by_tenant("tenant-1", state="building")
        assert len(building) == 1
        assert building[0].build_id == build_id_1

    def test_cleanup_old_builds(self, build_store):
        """Cleanup old completed/failed builds."""
        # Create and complete a build
        old_build = str(uuid.uuid4())
        build_store.create_build(
            build_id=old_build,
            artifact_id="art-1",
            version=1,
            executor_ref="duckdb_sql@v1",
        )
        build_store.start_build(old_build)
        build_store.complete_build(old_build)

        # Manually update completed_at to be old
        conn = build_store._get_connection()
        try:
            old_time = time.time() - (8 * 86400)  # 8 days ago
            conn.execute(
                "UPDATE artifact_builds SET completed_at = ? WHERE build_id = ?",
                (old_time, old_build),
            )
            conn.commit()
        finally:
            conn.close()

        # Create a recent build
        recent_build = str(uuid.uuid4())
        build_store.create_build(
            build_id=recent_build,
            artifact_id="art-2",
            version=1,
            executor_ref="duckdb_sql@v1",
        )
        build_store.start_build(recent_build)
        build_store.complete_build(recent_build)

        # Cleanup (7 days default)
        deleted = build_store.cleanup_old_builds()
        assert deleted == 1

        # Old build should be gone
        assert build_store.get_build(old_build) is None

        # Recent build should remain
        assert build_store.get_build(recent_build) is not None

    def test_get_stats(self, build_store):
        """Get build statistics."""
        # Create builds in various states
        b1 = str(uuid.uuid4())
        b2 = str(uuid.uuid4())
        b3 = str(uuid.uuid4())
        b4 = str(uuid.uuid4())

        build_store.create_build(build_id=b1, artifact_id="a", version=1, executor_ref="x")
        build_store.create_build(build_id=b2, artifact_id="a", version=2, executor_ref="x")
        build_store.create_build(build_id=b3, artifact_id="a", version=3, executor_ref="x")
        build_store.create_build(build_id=b4, artifact_id="a", version=4, executor_ref="x")

        build_store.start_build(b2)
        build_store.start_build(b3)
        build_store.complete_build(b3)
        build_store.start_build(b4)
        build_store.fail_build(b4, "error")

        stats = build_store.get_stats()
        assert stats["total"] == 4
        assert stats["pending"] == 1
        assert stats["building"] == 1
        assert stats["ready"] == 1
        assert stats["failed"] == 1

    def test_build_state_to_dict(self, build_store):
        """BuildState.to_dict() returns expected format."""
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id="art-123",
            version=1,
            executor_ref="duckdb_sql@v1",
        )

        state = build_store.get_build(build_id)
        d = state.to_dict()

        assert d["build_id"] == build_id
        assert d["artifact_id"] == "art-123"
        assert d["version"] == 1
        assert d["state"] == "pending"
        assert d["executor_ref"] == "duckdb_sql@v1"
        assert "created_at" in d


class TestSingletons:
    """Tests for module-level singleton functions."""

    def setup_method(self):
        """Reset singleton before each test."""
        reset_build_store()

    def teardown_method(self):
        """Reset singleton after each test."""
        reset_build_store()

    def test_get_build_store_uninitialized(self):
        """get_build_store returns None if not initialized."""
        store = get_build_store()
        assert store is None

    def test_get_build_store_initialized(self, tmp_path):
        """get_build_store returns store after initialization."""
        db_path = tmp_path / "test.sqlite"
        store = get_build_store(db_path)

        assert store is not None

        # Subsequent calls return same instance
        store2 = get_build_store()
        assert store2 is store

    def test_reset_clears_singleton(self, tmp_path):
        """reset_build_store clears the singleton."""
        db_path = tmp_path / "test.sqlite"
        get_build_store(db_path)

        reset_build_store()

        # After reset, returns None
        store = get_build_store()
        assert store is None


class TestCompletionIsLeaseChecked:
    """``claim_build`` decides which runner *starts* a build; completion has to
    decide which one is allowed to *publish* its result.

    When a lease expires (a GC pause or a partition longer than
    ``lease_duration_seconds``) ``reclaim_expired_build`` hands the build to
    another runner. The original runner keeps executing — that is deliberate,
    the heartbeat loop declines to cancel it — and it used to be able to walk
    all the way through completion, marking the build ready and repointing the
    requested registry name at its own output, after the takeover.
    """

    def test_a_runner_that_lost_its_lease_cannot_complete(self, build_store):
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id="art-123",
            version=1,
            executor_ref="duckdb_sql@v1",
        )
        build_store.claim_build(build_id, "runner-a", lease_duration_seconds=-1)
        assert build_store.reclaim_expired_build(build_id, "runner-b") is True

        # runner-a is still executing and reaches completion.
        assert build_store.complete_build(build_id, lease_owner="runner-a") is False
        assert build_store.get_build(build_id).state == "building"

    def test_the_current_lease_holder_can_complete(self, build_store):
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id="art-123",
            version=1,
            executor_ref="duckdb_sql@v1",
        )
        build_store.claim_build(build_id, "runner-a", lease_duration_seconds=-1)
        build_store.reclaim_expired_build(build_id, "runner-b")

        assert build_store.complete_build(build_id, output_byte_count=7, lease_owner="runner-b")
        state = build_store.get_build(build_id)
        assert state.state == "ready"
        assert state.output_byte_count == 7

    def test_omitting_the_owner_keeps_the_unchecked_behaviour(self, build_store):
        # The deprecated ``start_build`` path leaves lease_owner NULL, so the
        # check stays opt-in rather than breaking builds that never had a lease.
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id="art-123",
            version=1,
            executor_ref="duckdb_sql@v1",
        )
        build_store.start_build(build_id)

        assert build_store.complete_build(build_id) is True
        assert build_store.get_build(build_id).state == "ready"
