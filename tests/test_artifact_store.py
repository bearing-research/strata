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


def _make_ready_artifact(store, artifact_id: str, provenance: str) -> None:
    store.create_artifact(artifact_id, provenance)
    store.write_blob(artifact_id, 1, _ipc_bytes(1))
    store.finalize_artifact(artifact_id, 1, "{}", 1, 10)


class TestAliases:
    """Registry aliases: intent pointers on a name (#129)."""

    def test_set_and_resolve(self, store):
        _make_ready_artifact(store, "model-a", "prov-a")
        store.set_alias("demo/model", "champion", "model-a", 1)
        resolved = store.resolve_alias("demo/model", "champion")
        assert (resolved.id, resolved.version) == ("model-a", 1)

    def test_many_aliases_per_name(self, store):
        _make_ready_artifact(store, "model-a", "prov-a")
        _make_ready_artifact(store, "model-b", "prov-b")
        store.set_alias("demo/model", "champion", "model-a", 1)
        store.set_alias("demo/model", "candidate", "model-b", 1)

        aliases = store.list_aliases("demo/model")
        assert [(a.alias, a.artifact_id) for a in aliases] == [
            ("candidate", "model-b"),
            ("champion", "model-a"),
        ]

    def test_alias_move_keeps_old_version_reachable(self, store):
        """The friction-#9 scenario: promotion must not lose the old champion."""
        _make_ready_artifact(store, "model-a", "prov-a")
        _make_ready_artifact(store, "model-b", "prov-b")
        store.set_alias("demo/model", "champion", "model-a", 1)
        store.set_alias("demo/model", "champion", "model-b", 1)  # promote

        # Champion moved...
        assert store.resolve_alias("demo/model", "champion").id == "model-b"
        # ...and the audit answers "what was champion before?"
        moves = [e for e in store.read_audit(name="demo/model") if e["action"] == "alias_set"]
        assert len(moves) == 2
        latest = moves[0]
        assert latest["artifact_id"] == "model-b"
        assert latest["from_version"] == 1  # pointed at model-a v1 before

    def test_unknown_artifact_rejected(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.set_alias("demo/model", "champion", "ghost", 1)

    def test_superseded_artifact_allowed(self, store):
        _make_ready_artifact(store, "model-a", "prov-x")
        store.create_artifact("model-a", "prov-x")
        store.write_blob("model-a", 2, _ipc_bytes(1))
        store.finalize_artifact("model-a", 2, "{}", 1, 10)  # supersedes v1

        # An alias may pin the superseded version (still immutable + readable)
        store.set_alias("demo/model", "baseline", "model-a", 1)
        assert store.resolve_alias("demo/model", "baseline").version == 1

    def test_delete_alias(self, store):
        _make_ready_artifact(store, "model-a", "prov-a")
        store.set_alias("demo/model", "champion", "model-a", 1)
        assert store.delete_alias("demo/model", "champion") is True
        assert store.resolve_alias("demo/model", "champion") is None
        assert store.delete_alias("demo/model", "champion") is False


class TestTags:
    """Version tags: facts about one artifact build (#129)."""

    def test_set_get_tags(self, store):
        _make_ready_artifact(store, "model-a", "prov-a")
        store.set_tag("model-a", 1, "auc", "0.91")
        store.set_tag("model-a", 1, "validated_by", "fangchen")
        assert store.get_tags("model-a", 1) == {"auc": "0.91", "validated_by": "fangchen"}

    def test_tag_overwrite(self, store):
        _make_ready_artifact(store, "model-a", "prov-a")
        store.set_tag("model-a", 1, "auc", "0.91")
        store.set_tag("model-a", 1, "auc", "0.93")
        assert store.get_tags("model-a", 1) == {"auc": "0.93"}

    def test_tag_unknown_artifact_rejected(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.set_tag("ghost", 1, "k", "v")

    def test_delete_tag(self, store):
        _make_ready_artifact(store, "model-a", "prov-a")
        store.set_tag("model-a", 1, "auc", "0.91")
        assert store.delete_tag("model-a", 1, "auc") is True
        assert store.get_tags("model-a", 1) == {}


class TestRegistryAudit:
    """Append-only audit of name/alias/tag mutations (#129)."""

    def test_name_moves_audited_with_history(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        _make_ready_artifact(store, "m2", "prov-2")
        store.set_name("demo/model", "m1", 1)
        store.set_name("demo/model", "m2", 1)  # the silent-swap, now recorded

        entries = store.read_audit(name="demo/model")
        assert [e["action"] for e in entries] == ["name_set", "name_set"]
        assert entries[0]["artifact_id"] == "m2"
        assert entries[0]["from_version"] == 1
        assert entries[1]["from_version"] is None  # first set had no previous

    def test_name_delete_audited(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        store.set_name("demo/model", "m1", 1)
        store.delete_name("demo/model")
        actions = [e["action"] for e in store.read_audit(name="demo/model")]
        assert actions[0] == "name_delete"

    def test_finalize_promotion_audited(self, store):
        """Names set through finalize_and_set_name land in the audit too."""
        store.create_artifact("m1", "prov-f")
        store.write_blob("m1", 1, _ipc_bytes(1))
        store.finalize_and_set_name("m1", 1, "{}", 1, 10, name="auto/model")
        entries = store.read_audit(name="auto/model")
        assert entries and entries[0]["action"] == "name_set"

    def test_tag_audit_carries_key_value(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        store.set_tag("m1", 1, "auc", "0.91", actor="ci-bot")
        entry = store.read_audit(artifact_id="m1")[0]
        assert entry["action"] == "tag_set"
        assert (entry["key"], entry["value"]) == ("auc", "0.91")
        assert entry["actor"] == "ci-bot"

    def test_audit_filters_and_limit(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        for i in range(5):
            store.set_tag("m1", 1, f"k{i}", "v")
        assert len(store.read_audit(artifact_id="m1", limit=3)) == 3
        assert store.read_audit(name="unrelated") == []


class TestPendingAliasChanges:
    """Approval-gate mechanics: request / approve / reject (#129 follow-up)."""

    def test_request_and_approve_set(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        store.request_alias_change(
            "demo/model", "champion", "set", artifact_id="m1", version=1, actor="requester"
        )

        # Not applied yet
        assert store.resolve_alias("demo/model", "champion") is None
        pending = store.list_pending_changes()
        assert len(pending) == 1 and pending[0]["action"] == "set"

        applied = store.approve_alias_change("demo/model", "champion", actor="approver")
        assert applied["artifact_id"] == "m1"
        assert store.resolve_alias("demo/model", "champion").id == "m1"
        assert store.list_pending_changes() == []

        # Audit trail: request -> approved -> the applied alias_set
        actions = [e["action"] for e in store.read_audit(name="demo/model")]
        assert actions[:3] == ["alias_set", "alias_approved", "alias_request_set"]
        approved = next(
            e for e in store.read_audit(name="demo/model") if e["action"] == "alias_approved"
        )
        assert approved["actor"] == "approver"

    def test_reject_discards(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        store.request_alias_change("demo/model", "champion", "set", artifact_id="m1", version=1)
        rejected = store.reject_alias_change("demo/model", "champion", actor="reviewer")
        assert rejected["artifact_id"] == "m1"
        assert store.resolve_alias("demo/model", "champion") is None
        assert store.list_pending_changes() == []
        actions = [e["action"] for e in store.read_audit(name="demo/model")]
        assert actions[0] == "alias_rejected"

    def test_request_delete_then_approve(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        store.set_alias("demo/model", "champion", "m1", 1)
        store.request_alias_change("demo/model", "champion", "delete")
        # still resolvable until approved
        assert store.resolve_alias("demo/model", "champion") is not None
        store.approve_alias_change("demo/model", "champion")
        assert store.resolve_alias("demo/model", "champion") is None

    def test_new_request_replaces_previous(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        _make_ready_artifact(store, "m2", "prov-2")
        store.request_alias_change("demo/model", "champion", "set", artifact_id="m1", version=1)
        store.request_alias_change("demo/model", "champion", "set", artifact_id="m2", version=1)
        pending = store.list_pending_changes()
        assert len(pending) == 1
        assert pending[0]["artifact_id"] == "m2"

    def test_approve_without_pending_raises(self, store):
        with pytest.raises(ValueError, match="No pending change"):
            store.approve_alias_change("demo/model", "champion")

    def test_request_set_validates_artifact(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.request_alias_change(
                "demo/model", "champion", "set", artifact_id="ghost", version=1
            )


class TestAliasedArtifactProtection:
    """GC and delete respect alias pointers (review finding C1)."""

    def test_gc_spares_aliased_artifact(self, store):
        _make_ready_artifact(store, "pinned", "prov-p")
        store.set_alias("demo/model", "champion", "pinned", 1)

        result = store.garbage_collect(max_age_days=0)
        assert store.get_artifact("pinned", 1) is not None, result
        assert store.resolve_alias("demo/model", "champion") is not None

    def test_gc_spares_aliased_superseded_version(self, store):
        """The C1 failure sequence: champion pins a superseded version."""
        _make_ready_artifact(store, "model", "prov-s")
        store.create_artifact("model", "prov-s")
        store.write_blob("model", 2, _ipc_bytes(1))
        store.finalize_artifact("model", 2, "{}", 1, 10)  # supersedes v1
        store.set_alias("demo/model", "champion", "model", 1)
        store.set_name("demo/model", "model", 2)  # name guards v2

        store.garbage_collect(max_age_days=0)

        champion = store.resolve_alias("demo/model", "champion")
        assert champion is not None and champion.version == 1

    def test_gc_still_collects_unreferenced(self, store):
        _make_ready_artifact(store, "loose", "prov-l")
        result = store.garbage_collect(max_age_days=0)
        assert result["deleted_count"] == 1
        assert store.get_artifact("loose", 1) is None

    def test_delete_artifact_cleans_aliases_and_tags(self, store):
        _make_ready_artifact(store, "doomed", "prov-d")
        store.set_alias("demo/model", "candidate", "doomed", 1)
        store.set_tag("doomed", 1, "auc", "0.5")

        assert store.delete_artifact("doomed", 1) is True
        assert store.resolve_alias("demo/model", "candidate") is None
        assert store.get_tags("doomed", 1) == {}
        # The forced alias removal is auditable
        entries = store.read_audit(name="demo/model")
        assert entries[0]["action"] == "alias_delete"


class TestApproveAtomicity:
    """approve_alias_change is one transaction (review finding M1)."""

    def test_dead_target_keeps_pending_intact(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        store.request_alias_change("demo/model", "champion", "set", artifact_id="m1", version=1)
        store.delete_artifact("m1", 1)  # target dies between request and approve

        with pytest.raises(ValueError, match="no longer available"):
            store.approve_alias_change("demo/model", "champion")

        # The pending entry survives for an explicit reject
        assert len(store.list_pending_changes()) == 1
        # And no phantom approval landed in the audit
        actions = [e["action"] for e in store.read_audit(name="demo/model")]
        assert "alias_approved" not in actions

    def test_approve_applies_and_audits_in_order(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        store.request_alias_change(
            "demo/model", "champion", "set", artifact_id="m1", version=1, actor="req"
        )
        store.approve_alias_change("demo/model", "champion", actor="approver")

        assert store.resolve_alias("demo/model", "champion").id == "m1"
        entries = store.read_audit(name="demo/model")
        assert [e["action"] for e in entries] == [
            "alias_set",
            "alias_approved",
            "alias_request_set",
        ]
        assert entries[0]["actor"] == "approver"


class TestCreateArtifactVersionRace:
    """Concurrent creates for one artifact id allocate distinct versions (M2)."""

    def test_threaded_creates_never_collide(self, store):
        import threading

        _make_ready_artifact(store, "contended", "prov-base")
        errors: list[Exception] = []
        versions: list[int] = []
        barrier = threading.Barrier(4)

        def create(n):
            try:
                barrier.wait()
                versions.append(store.create_artifact("contended", f"prov-{n}"))
            except Exception as e:  # noqa: BLE001 — collecting for assertion
                errors.append(e)

        threads = [threading.Thread(target=create, args=(i,)) for i in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == []
        assert sorted(versions) == [2, 3, 4, 5]


class TestAliasIdempotence:
    """Identical-target alias writes are no-ops (dogfood finding D4)."""

    def test_set_alias_same_target_is_noop(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        assert store.set_alias("demo/model", "champion", "m1", 1) is True
        before = len(store.read_audit(name="demo/model"))
        assert store.set_alias("demo/model", "champion", "m1", 1) is False
        assert len(store.read_audit(name="demo/model")) == before  # no audit spam

    def test_request_for_live_pointer_is_noop(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        store.set_alias("demo/model", "champion", "m1", 1)
        assert (
            store.request_alias_change("demo/model", "champion", "set", artifact_id="m1", version=1)
            is False
        )
        assert store.list_pending_changes() == []

    def test_request_for_new_target_still_queues(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        _make_ready_artifact(store, "m2", "prov-2")
        store.set_alias("demo/model", "champion", "m1", 1)
        assert (
            store.request_alias_change("demo/model", "champion", "set", artifact_id="m2", version=1)
            is True
        )
        assert len(store.list_pending_changes()) == 1


class TestAuditTenantScoping:
    """read_audit isolates by tenant for request-serving callers (Vuln 1)."""

    def _seed(self, store, tenant, name):
        store.create_artifact(f"{tenant}-a", f"prov-{tenant}", tenant=tenant)
        store.write_blob(f"{tenant}-a", 1, _ipc_bytes(1))
        store.finalize_artifact(f"{tenant}-a", 1, "{}", 1, 10)
        store.set_name(name, f"{tenant}-a", 1, tenant=tenant)

    def test_tenant_filter_isolates_history(self, store):
        self._seed(store, "acme", "acme/model")
        self._seed(store, "globex", "globex/model")

        acme = store.read_audit(tenant="acme")
        assert acme, "expected acme audit rows"
        assert all(e["tenant"] == "acme" for e in acme)
        assert not any(e["name"] == "globex/model" for e in acme)

        globex = store.read_audit(tenant="globex")
        assert all(e["tenant"] == "globex" for e in globex)

    def test_default_sentinel_returns_all_tenants(self, store):
        """No tenant arg = whole-store view (CLI / admin)."""
        self._seed(store, "acme", "acme/model")
        self._seed(store, "globex", "globex/model")
        tenants = {e["tenant"] for e in store.read_audit()}
        assert {"acme", "globex"} <= tenants

    def test_none_tenant_filters_to_default(self, store):
        """Explicit tenant=None scopes to the '' default tenant, not all."""
        _make_ready_artifact(store, "m1", "prov-1")  # tenant None -> ''
        store.set_name("default/model", "m1", 1)
        self._seed(store, "acme", "acme/model")

        default_rows = store.read_audit(tenant=None)
        assert default_rows
        assert all(e["tenant"] in (None, "") for e in default_rows)
        assert not any(e["name"] == "acme/model" for e in default_rows)


class TestApproveSeparationOfDuty:
    """approve_alias_change can forbid self-approval (Vuln 2)."""

    def _pending(self, store, requester):
        _make_ready_artifact(store, "m1", "prov-1")
        store.request_alias_change(
            "demo/model", "champion", "set", artifact_id="m1", version=1, actor=requester
        )

    def test_self_approve_blocked_when_required(self, store):
        self._pending(store, "alice")
        with pytest.raises(ValueError, match="Separation of duty"):
            store.approve_alias_change(
                "demo/model", "champion", actor="alice", require_distinct_approver=True
            )
        # Pending survives the rejected self-approval
        assert len(store.list_pending_changes()) == 1
        assert store.resolve_alias("demo/model", "champion") is None

    def test_distinct_approver_allowed(self, store):
        self._pending(store, "alice")
        store.approve_alias_change(
            "demo/model", "champion", actor="bob", require_distinct_approver=True
        )
        assert store.resolve_alias("demo/model", "champion").id == "m1"

    def test_self_approve_allowed_when_not_required(self, store):
        """Break-glass / personal mode: separation of duty off."""
        self._pending(store, "alice")
        store.approve_alias_change(
            "demo/model", "champion", actor="alice", require_distinct_approver=False
        )
        assert store.resolve_alias("demo/model", "champion").id == "m1"


class TestTagAndNameLookups:
    """Reverse lookups used by the per-cell published-artifacts route:
    find artifacts a cell published (``nb_cell`` tag) and the names that
    point at a version."""

    def test_list_artifacts_by_tag(self, store):
        for aid, h in (("a1", "h1"), ("a2", "h2"), ("a3", "h3")):
            store.create_artifact(aid, h)
            store.finalize_artifact(aid, 1, "{}", 0, 0)
        store.set_tag("a1", 1, "nb_cell", "cellX")
        store.set_tag("a3", 1, "nb_cell", "cellX")
        store.set_tag("a2", 1, "nb_cell", "cellY")

        assert store.list_artifacts_by_tag("nb_cell", "cellX") == [("a1", 1), ("a3", 1)]
        assert store.list_artifacts_by_tag("nb_cell", "cellY") == [("a2", 1)]
        assert store.list_artifacts_by_tag("nb_cell", "absent") == []

    def test_names_for_artifact(self, store):
        store.create_artifact("a1", "h1")
        store.finalize_artifact("a1", 1, "{}", 0, 0)
        store.set_name("team/model", "a1", 1)

        assert store.names_for_artifact("a1", 1) == ["team/model"]
        assert store.names_for_artifact("a1", 2) == []
        assert store.names_for_artifact("nope", 1) == []
