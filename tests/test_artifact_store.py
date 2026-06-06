"""Tests for artifact store.

These tests verify:
1. Artifact creation, finalization, and state transitions
2. Provenance hash computation and deduplication
3. Name pointer CRUD operations
4. Blob I/O (write/read)
5. Cleanup of failed artifacts
"""

import json

import pytest

from strata.artifact_store import (
    ArtifactStore,
    TransformSpec,
    compute_provenance_hash,
    reset_artifact_store,
)


@pytest.fixture
def artifact_dir(tmp_path):
    """Create a temporary artifact directory."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    yield artifact_dir
    reset_artifact_store()


@pytest.fixture
def store(artifact_dir):
    """Create an artifact store."""
    return ArtifactStore(artifact_dir)


class TestTransformSpec:
    """Tests for TransformSpec serialization."""

    def test_to_json(self):
        """TransformSpec serializes to JSON."""
        spec = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT * FROM input"},
            inputs=["strata://table/db.events"],
        )
        json_str = spec.to_json()
        data = json.loads(json_str)
        assert data["executor"] == "local://duckdb_sql@v1"
        assert data["params"] == {"sql": "SELECT * FROM input"}
        assert data["inputs"] == ["strata://table/db.events"]

    def test_from_json(self):
        """TransformSpec deserializes from JSON."""
        json_str = json.dumps(
            {
                "executor": "local://duckdb_sql@v1",
                "params": {"sql": "SELECT 1"},
                "inputs": [],
            }
        )
        spec = TransformSpec.from_json(json_str)
        assert spec.executor == "local://duckdb_sql@v1"
        assert spec.params == {"sql": "SELECT 1"}
        assert spec.inputs == []

    def test_roundtrip(self):
        """TransformSpec survives JSON roundtrip."""
        original = TransformSpec(
            executor="local://polars_expr@v1",
            params={"expr": "col('a') + 1"},
            inputs=["input1", "input2"],
        )
        restored = TransformSpec.from_json(original.to_json())
        assert restored.executor == original.executor
        assert restored.params == original.params
        assert restored.inputs == original.inputs


class TestProvenanceHash:
    """Tests for provenance hash computation."""

    def test_deterministic(self):
        """Provenance hash is deterministic."""
        spec = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT 1"},
            inputs=[],
        )
        hash1 = compute_provenance_hash(["abc", "def"], spec)
        hash2 = compute_provenance_hash(["abc", "def"], spec)
        assert hash1 == hash2

    def test_input_order_independent(self):
        """Provenance hash is independent of input order."""
        spec = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT 1"},
            inputs=[],
        )
        hash1 = compute_provenance_hash(["abc", "def"], spec)
        hash2 = compute_provenance_hash(["def", "abc"], spec)
        assert hash1 == hash2

    def test_different_inputs_different_hash(self):
        """Different inputs produce different hashes."""
        spec = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT 1"},
            inputs=[],
        )
        hash1 = compute_provenance_hash(["abc"], spec)
        hash2 = compute_provenance_hash(["xyz"], spec)
        assert hash1 != hash2

    def test_different_transform_different_hash(self):
        """Different transforms produce different hashes."""
        spec1 = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT 1"},
            inputs=[],
        )
        spec2 = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT 2"},
            inputs=[],
        )
        hash1 = compute_provenance_hash(["abc"], spec1)
        hash2 = compute_provenance_hash(["abc"], spec2)
        assert hash1 != hash2


class TestArtifactCRUD:
    """Tests for artifact CRUD operations."""

    def test_create_artifact(self, store):
        """Create artifact starts in building state."""
        version = store.create_artifact(
            artifact_id="test-id",
            provenance_hash="hash123",
        )
        assert version == 1

        artifact = store.get_artifact("test-id", version)
        assert artifact is not None
        assert artifact.id == "test-id"
        assert artifact.version == 1
        assert artifact.state == "building"
        assert artifact.provenance_hash == "hash123"

    def test_create_increments_version(self, store):
        """Each create increments the version number."""
        v1 = store.create_artifact("test-id", "hash1")
        v2 = store.create_artifact("test-id", "hash2")
        v3 = store.create_artifact("test-id", "hash3")

        assert v1 == 1
        assert v2 == 2
        assert v3 == 3

    def test_create_with_transform_spec(self, store):
        """Create artifact with transform spec."""
        spec = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT 1"},
            inputs=[],
        )
        version = store.create_artifact(
            artifact_id="test-id",
            provenance_hash="hash123",
            transform_spec=spec,
        )
        artifact = store.get_artifact("test-id", version)
        assert artifact.transform_spec == spec.to_json()

    def test_finalize_artifact(self, store):
        """Finalize transitions to ready state."""
        version = store.create_artifact("test-id", "hash123")
        store.finalize_artifact(
            artifact_id="test-id",
            version=version,
            schema_json='{"fields": []}',
            row_count=100,
            byte_size=1024,
        )

        artifact = store.get_artifact("test-id", version)
        assert artifact.state == "ready"
        assert artifact.schema_json == '{"fields": []}'
        assert artifact.row_count == 100
        assert artifact.byte_size == 1024

    def test_finalize_nonexistent_raises(self, store):
        """Finalize nonexistent artifact raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            store.finalize_artifact("nonexistent", 1, "{}", 0, 0)
        assert "not found" in str(exc_info.value)

    def test_finalize_already_ready_is_idempotent(self, store):
        """Finalize already-ready artifact is idempotent (returns existing)."""
        version = store.create_artifact("test-id", "hash123")
        first_result = store.finalize_artifact("test-id", version, "{}", 0, 0)

        # Calling finalize again should return the same artifact (idempotent)
        second_result = store.finalize_artifact("test-id", version, "{}", 0, 0)
        assert second_result is not None
        assert second_result.id == first_result.id
        assert second_result.version == first_result.version
        assert second_result.state == "ready"

    def test_fail_artifact(self, store):
        """Fail transitions to failed state."""
        version = store.create_artifact("test-id", "hash123")
        store.fail_artifact("test-id", version)

        artifact = store.get_artifact("test-id", version)
        assert artifact.state == "failed"

    def test_get_nonexistent(self, store):
        """Get nonexistent artifact returns None."""
        result = store.get_artifact("nonexistent", 1)
        assert result is None

    def test_get_latest_version(self, store):
        """Get latest ready version."""
        # Create v1 (ready)
        v1 = store.create_artifact("test-id", "hash1")
        store.finalize_artifact("test-id", v1, "{}", 100, 1000)

        # Create v2 (ready)
        v2 = store.create_artifact("test-id", "hash2")
        store.finalize_artifact("test-id", v2, "{}", 200, 2000)

        # Create v3 (still building)
        store.create_artifact("test-id", "hash3")

        # Latest should be v2 (highest ready version)
        latest = store.get_latest_version("test-id")
        assert latest is not None
        assert latest.version == 2
        assert latest.row_count == 200


class TestProvenanceLookup:
    """Tests for provenance-based deduplication."""

    def test_find_by_provenance(self, store):
        """Find artifact by provenance hash."""
        version = store.create_artifact("test-id", "unique-hash")
        store.finalize_artifact("test-id", version, "{}", 100, 1000)

        found = store.find_by_provenance("unique-hash")
        assert found is not None
        assert found.id == "test-id"
        assert found.version == version

    def test_find_by_provenance_not_found(self, store):
        """Find returns None for unknown provenance."""
        found = store.find_by_provenance("unknown-hash")
        assert found is None

    def test_find_by_provenance_ignores_building(self, store):
        """Find ignores artifacts in building state."""
        store.create_artifact("test-id", "hash123")
        # Not finalized, so should not be found
        found = store.find_by_provenance("hash123")
        assert found is None

    def test_find_by_provenance_ignores_failed(self, store):
        """Find ignores artifacts in failed state."""
        version = store.create_artifact("test-id", "hash123")
        store.fail_artifact("test-id", version)

        found = store.find_by_provenance("hash123")
        assert found is None


class TestBlobIO:
    """Tests for blob I/O operations."""

    def test_write_and_read_blob(self, store):
        """Write and read blob data."""
        version = store.create_artifact("test-id", "hash123")
        data = b"test arrow data"

        store.write_blob("test-id", version, data)
        result = store.read_blob("test-id", version)

        assert result == data

    def test_read_nonexistent_blob(self, store):
        """Read nonexistent blob returns None."""
        result = store.read_blob("nonexistent", 1)
        assert result is None

    def test_blob_exists(self, store):
        """Check if blob exists."""
        version = store.create_artifact("test-id", "hash123")

        assert store.blob_exists("test-id", version) is False

        store.write_blob("test-id", version, b"data")

        assert store.blob_exists("test-id", version) is True

    def test_write_blob_atomic(self, store, artifact_dir):
        """Write blob is atomic (no partial writes)."""
        version = store.create_artifact("test-id", "hash123")
        data = b"x" * 10000

        store.write_blob("test-id", version, data)

        # No temp files should remain
        temp_files = list(artifact_dir.glob("**/*.tmp"))
        assert len(temp_files) == 0

        # Blob should be complete
        assert store.read_blob("test-id", version) == data


class TestNamePointers:
    """Tests for name pointer operations."""

    def test_set_and_resolve_name(self, store):
        """Set and resolve a name pointer."""
        version = store.create_artifact("test-id", "hash123")
        store.finalize_artifact("test-id", version, "{}", 100, 1000)

        store.set_name("my-artifact", "test-id", version)

        resolved = store.resolve_name("my-artifact")
        assert resolved is not None
        assert resolved.id == "test-id"
        assert resolved.version == version

    def test_resolve_nonexistent_name(self, store):
        """Resolve nonexistent name returns None."""
        resolved = store.resolve_name("nonexistent")
        assert resolved is None

    def test_set_name_requires_ready(self, store):
        """Set name requires target to be ready."""
        version = store.create_artifact("test-id", "hash123")
        # Not finalized

        with pytest.raises(ValueError) as exc_info:
            store.set_name("my-artifact", "test-id", version)
        assert "not ready" in str(exc_info.value)

    def test_set_name_requires_exists(self, store):
        """Set name requires target to exist."""
        with pytest.raises(ValueError) as exc_info:
            store.set_name("my-artifact", "nonexistent", 1)
        assert "not found" in str(exc_info.value)

    def test_update_name(self, store):
        """Update name to point to new version."""
        # Create v1
        v1 = store.create_artifact("test-id", "hash1")
        store.finalize_artifact("test-id", v1, "{}", 100, 1000)
        store.set_name("my-artifact", "test-id", v1)

        # Create v2
        v2 = store.create_artifact("test-id", "hash2")
        store.finalize_artifact("test-id", v2, "{}", 200, 2000)
        store.set_name("my-artifact", "test-id", v2)

        # Should now resolve to v2
        resolved = store.resolve_name("my-artifact")
        assert resolved.version == v2

    def test_get_name(self, store):
        """Get name pointer metadata."""
        version = store.create_artifact("test-id", "hash123")
        store.finalize_artifact("test-id", version, "{}", 100, 1000)
        store.set_name("my-artifact", "test-id", version)

        name_info = store.get_name("my-artifact")
        assert name_info is not None
        assert name_info.name == "my-artifact"
        assert name_info.artifact_id == "test-id"
        assert name_info.version == version
        assert name_info.updated_at > 0

    def test_delete_name(self, store):
        """Delete a name pointer."""
        version = store.create_artifact("test-id", "hash123")
        store.finalize_artifact("test-id", version, "{}", 100, 1000)
        store.set_name("my-artifact", "test-id", version)

        assert store.delete_name("my-artifact") is True
        assert store.resolve_name("my-artifact") is None

    def test_delete_nonexistent_name(self, store):
        """Delete nonexistent name returns False."""
        assert store.delete_name("nonexistent") is False

    def test_list_names(self, store):
        """List all name pointers."""
        # Create artifacts and names
        for i in range(3):
            v = store.create_artifact(f"id-{i}", f"hash-{i}")
            store.finalize_artifact(f"id-{i}", v, "{}", i * 100, i * 1000)
            store.set_name(f"name-{i}", f"id-{i}", v)

        names = store.list_names()
        assert len(names) == 3
        assert [n.name for n in names] == ["name-0", "name-1", "name-2"]


class TestCleanup:
    """Tests for cleanup operations."""

    def test_cleanup_failed(self, store, artifact_dir):
        """Cleanup removes failed artifacts older than max age."""
        # Create a failed artifact
        version = store.create_artifact("test-id", "hash123")
        store.write_blob("test-id", version, b"data")
        store.fail_artifact("test-id", version)

        # Should not be cleaned up yet (too recent)
        count = store.cleanup_failed(max_age_seconds=3600)
        assert count == 0

        # Cleanup with 0 age should remove it
        count = store.cleanup_failed(max_age_seconds=0)
        assert count == 1

        # Artifact and blob should be gone
        assert store.get_artifact("test-id", version) is None
        assert store.blob_exists("test-id", version) is False

    def test_cleanup_preserves_ready(self, store):
        """Cleanup preserves ready artifacts."""
        version = store.create_artifact("test-id", "hash123")
        store.finalize_artifact("test-id", version, "{}", 100, 1000)

        # Even with 0 age, ready artifacts should not be removed
        count = store.cleanup_failed(max_age_seconds=0)
        assert count == 0
        assert store.get_artifact("test-id", version) is not None


class TestStats:
    """Tests for statistics."""

    def test_stats_empty(self, store):
        """Stats on empty store."""
        stats = store.stats()
        assert stats["total_versions"] == 0
        assert stats["ready_versions"] == 0
        assert stats["building_versions"] == 0
        assert stats["failed_versions"] == 0
        assert stats["total_bytes"] == 0
        assert stats["total_rows"] == 0
        assert stats["name_count"] == 0

    def test_stats_with_data(self, store):
        """Stats with artifacts."""
        # Create ready artifact
        v1 = store.create_artifact("id-1", "hash1")
        store.finalize_artifact("id-1", v1, "{}", 100, 1000)

        # Create building artifact
        store.create_artifact("id-2", "hash2")

        # Create failed artifact
        v3 = store.create_artifact("id-3", "hash3")
        store.fail_artifact("id-3", v3)

        # Create name
        store.set_name("my-name", "id-1", v1)

        stats = store.stats()
        assert stats["total_versions"] == 3
        assert stats["ready_versions"] == 1
        assert stats["building_versions"] == 1
        assert stats["failed_versions"] == 1
        assert stats["total_bytes"] == 1000
        assert stats["total_rows"] == 100
        assert stats["name_count"] == 1


def _ipc_bytes(num_rows: int) -> bytes:
    """Build a single valid Arrow IPC stream with ``num_rows`` rows."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, pa.schema([("id", pa.int64())])) as writer:
        writer.write_batch(pa.RecordBatch.from_pydict({"id": list(range(num_rows))}))
    return sink.getvalue().to_pybytes()


class TestRefreshSupersede:
    """Refresh rebuilds become new versions of the same artifact (#123)."""

    def test_finalize_supersedes_older_ready_version(self, store):
        """Finalizing v2 with v1's provenance demotes v1 to superseded."""
        store.create_artifact("art-1", "prov-x")
        store.finalize_artifact("art-1", 1, "{}", 10, 100)

        version = store.create_artifact("art-1", "prov-x")
        assert version == 2
        finalized = store.finalize_artifact("art-1", 2, "{}", 12, 120)

        assert finalized.version == 2
        assert finalized.state == "ready"
        assert store.get_artifact("art-1", 1).state == "superseded"

    def test_provenance_lookup_returns_rebuild(self, store):
        """After supersede, dedup lookups resolve the new version."""
        store.create_artifact("art-1", "prov-x")
        store.finalize_artifact("art-1", 1, "{}", 10, 100)
        store.create_artifact("art-1", "prov-x")
        store.finalize_artifact("art-1", 2, "{}", 12, 120)

        found = store.find_by_provenance("prov-x")
        assert found is not None
        assert (found.id, found.version) == ("art-1", 2)

    def test_different_id_still_dedup_fails(self, store):
        """The cross-id duplicate race keeps its existing semantics."""
        store.create_artifact("art-1", "prov-x")
        store.finalize_artifact("art-1", 1, "{}", 10, 100)

        store.create_artifact("art-2", "prov-x")
        result = store.finalize_artifact("art-2", 1, "{}", 10, 100)

        # Returns the existing artifact; the duplicate is failed
        assert (result.id, result.version) == ("art-1", 1)
        assert store.get_artifact("art-2", 1).state == "failed"
        assert store.get_artifact("art-1", 1).state == "ready"

    def test_finalize_and_set_name_supersedes(self, store):
        """The atomic finalize+name path supersedes the same way."""
        store.create_artifact("art-1", "prov-x")
        store.finalize_and_set_name("art-1", 1, "{}", 10, 100, name="model")
        store.create_artifact("art-1", "prov-x")
        finalized = store.finalize_and_set_name("art-1", 2, "{}", 12, 120, name="model")

        assert finalized.version == 2
        assert store.get_artifact("art-1", 1).state == "superseded"
        resolved = store.resolve_name("model")
        assert (resolved.id, resolved.version) == ("art-1", 2)


class TestZombieSweep:
    """Stale building artifacts are demoted to failed (#123)."""

    def test_old_building_demoted(self, store):
        store.create_artifact("zombie", "prov-z")
        swept = store.sweep_zombie_builds(max_age_seconds=0)
        assert swept == 1
        assert store.get_artifact("zombie", 1).state == "failed"

    def test_recent_building_kept(self, store):
        store.create_artifact("fresh", "prov-f")
        swept = store.sweep_zombie_builds(max_age_seconds=3600)
        assert swept == 0
        assert store.get_artifact("fresh", 1).state == "building"

    def test_ready_untouched(self, store):
        store.create_artifact("done", "prov-d")
        store.finalize_artifact("done", 1, "{}", 1, 10)
        swept = store.sweep_zombie_builds(max_age_seconds=0)
        assert swept == 0
        assert store.get_artifact("done", 1).state == "ready"


class TestVerifyArtifacts:
    """Store-wide blob/metadata consistency check (#123)."""

    def _make_ready(self, store, artifact_id: str, provenance: str, rows: int) -> None:
        store.create_artifact(artifact_id, provenance)
        store.write_blob(artifact_id, 1, _ipc_bytes(rows))
        store.finalize_artifact(artifact_id, 1, "{}", rows, 100)

    def test_consistent_store_is_clean(self, store):
        self._make_ready(store, "good", "prov-g", 5)
        assert store.verify_artifacts() == []

    def test_row_count_mismatch_detected(self, store):
        store.create_artifact("short", "prov-s")
        store.write_blob("short", 1, _ipc_bytes(3))
        store.finalize_artifact("short", 1, "{}", 99, 100)  # lies about rows

        findings = store.verify_artifacts()
        assert len(findings) == 1
        assert findings[0]["problem"] == "row_count_mismatch"
        assert findings[0]["artifact_id"] == "short"

    def test_concatenated_streams_detected(self, store):
        """The #121 corruption shape is flagged as invalid_stream."""
        store.create_artifact("concat", "prov-c")
        store.write_blob("concat", 1, _ipc_bytes(3) + _ipc_bytes(3))
        store.finalize_artifact("concat", 1, "{}", 6, 100)

        findings = store.verify_artifacts()
        assert len(findings) == 1
        assert findings[0]["problem"] == "invalid_stream"

    def test_missing_blob_detected(self, store):
        store.create_artifact("ghost", "prov-gh")
        store.finalize_artifact("ghost", 1, "{}", 1, 10)  # never wrote a blob

        findings = store.verify_artifacts()
        assert len(findings) == 1
        assert findings[0]["problem"] == "missing_blob"


class TestArtifactVerifyCli:
    """`strata artifact verify` surfaces store inconsistencies (#123)."""

    def _run(self, artifact_dir, fmt="human"):
        import argparse

        from strata.artifact_cli import cmd_verify

        args = argparse.Namespace(artifact_dir=str(artifact_dir), format=fmt)
        return cmd_verify(args)

    def test_clean_store_exits_zero(self, store, artifact_dir, capsys):
        store.create_artifact("good", "prov-g")
        store.write_blob("good", 1, _ipc_bytes(5))
        store.finalize_artifact("good", 1, "{}", 5, 100)

        assert self._run(artifact_dir) == 0
        assert "consistent" in capsys.readouterr().out

    def test_problems_exit_one_with_json(self, store, artifact_dir, capsys):
        store.create_artifact("bad", "prov-b")
        store.write_blob("bad", 1, _ipc_bytes(3) + _ipc_bytes(3))
        store.finalize_artifact("bad", 1, "{}", 6, 100)

        assert self._run(artifact_dir, fmt="json") == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["findings"][0]["problem"] == "invalid_stream"

    def test_missing_dir_exits_two(self, tmp_path):
        assert self._run(tmp_path / "nope") == 2


class TestLegacyDefaultTenantNames:
    """Artifacts stamped with legacy '_default' stay nameable (friction #7)."""

    def test_tenantless_request_can_name_default_artifact(self, store):
        store.create_artifact("legacy", "prov-l", tenant="_default")
        store.finalize_artifact("legacy", 1, "{}", 1, 10)

        # Pre-#126 PUT uploads stamped "_default"; the names routes resolve
        # single-tenant requests to None. That combination must not strand
        # the artifact as unnameable.
        store.set_name("legacy-name", "legacy", 1, tenant=None)  # must not raise
        resolved = store.resolve_name("legacy-name")
        assert (resolved.id, resolved.version) == ("legacy", 1)

    def test_real_tenant_mismatch_still_rejected(self, store):
        store.create_artifact("owned", "prov-o", tenant="acme")
        store.finalize_artifact("owned", 1, "{}", 1, 10)

        with pytest.raises(ValueError, match="cannot assign name"):
            store.set_name("steal", "owned", 1, tenant="globex")
