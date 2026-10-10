"""Tests for the artifact store: lifecycle, provenance dedup, names, blob I/O, cleanup."""

import json

import pytest

from strata.artifact_store import (
    ArtifactNotFoundError,
    ArtifactStore,
    TransformSpec,
    compute_provenance_hash,
    reset_artifact_store,
)
from tests.conftest import race_after_pending_read


@pytest.fixture
def artifact_dir(tmp_path):
    """Create a temporary artifact directory."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    yield artifact_dir
    reset_artifact_store()


@pytest.fixture
def store(artifact_dir):
    """An artifact store."""
    return ArtifactStore(artifact_dir)


class TestTransformSpec:
    def test_to_json(self):
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
    def test_deterministic(self):
        spec = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT 1"},
            inputs=[],
        )
        hash1 = compute_provenance_hash(["abc", "def"], spec)
        hash2 = compute_provenance_hash(["abc", "def"], spec)
        assert hash1 == hash2

    def test_input_order_independent(self):
        spec = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT 1"},
            inputs=[],
        )
        hash1 = compute_provenance_hash(["abc", "def"], spec)
        hash2 = compute_provenance_hash(["def", "abc"], spec)
        assert hash1 == hash2

    def test_different_inputs_different_hash(self):
        spec = TransformSpec(
            executor="local://duckdb_sql@v1",
            params={"sql": "SELECT 1"},
            inputs=[],
        )
        hash1 = compute_provenance_hash(["abc"], spec)
        hash2 = compute_provenance_hash(["xyz"], spec)
        assert hash1 != hash2

    def test_different_transform_different_hash(self):
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
    def test_create_artifact(self, store):
        """A created artifact starts in building state."""
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
        v1 = store.create_artifact("test-id", "hash1")
        v2 = store.create_artifact("test-id", "hash2")
        v3 = store.create_artifact("test-id", "hash3")

        assert v1 == 1
        assert v2 == 2
        assert v3 == 3

    def test_create_with_transform_spec(self, store):
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
        with pytest.raises(ValueError) as exc_info:
            store.finalize_artifact("nonexistent", 1, "{}", 0, 0)
        assert "not found" in str(exc_info.value)

    def test_finalize_already_ready_is_idempotent(self, store):
        """Finalizing an already-ready artifact returns the existing one."""
        version = store.create_artifact("test-id", "hash123")
        first_result = store.finalize_artifact("test-id", version, "{}", 0, 0)

        # Finalize is idempotent.
        second_result = store.finalize_artifact("test-id", version, "{}", 0, 0)
        assert second_result is not None
        assert second_result.id == first_result.id
        assert second_result.version == first_result.version
        assert second_result.state == "ready"

    def test_fail_artifact(self, store):
        version = store.create_artifact("test-id", "hash123")
        store.fail_artifact("test-id", version)

        artifact = store.get_artifact("test-id", version)
        assert artifact.state == "failed"

    def test_get_nonexistent(self, store):
        result = store.get_artifact("nonexistent", 1)
        assert result is None

    def test_get_latest_version(self, store):
        """Returns the latest ready version."""
        v1 = store.create_artifact("test-id", "hash1")
        store.finalize_artifact("test-id", v1, "{}", 100, 1000)

        v2 = store.create_artifact("test-id", "hash2")
        store.finalize_artifact("test-id", v2, "{}", 200, 2000)

        # v3 is still building.
        store.create_artifact("test-id", "hash3")

        # Latest is v2, the highest ready version.
        latest = store.get_latest_version("test-id")
        assert latest is not None
        assert latest.version == 2
        assert latest.row_count == 200


class TestProvenanceLookup:
    def test_find_by_provenance(self, store):
        version = store.create_artifact("test-id", "unique-hash")
        store.finalize_artifact("test-id", version, "{}", 100, 1000)

        found = store.find_by_provenance("unique-hash")
        assert found is not None
        assert found.id == "test-id"
        assert found.version == version

    def test_find_by_provenance_not_found(self, store):
        found = store.find_by_provenance("unknown-hash")
        assert found is None

    def test_find_by_provenance_ignores_building(self, store):
        store.create_artifact("test-id", "hash123")
        # Not finalized, so not found.
        found = store.find_by_provenance("hash123")
        assert found is None

    def test_find_by_provenance_ignores_failed(self, store):
        version = store.create_artifact("test-id", "hash123")
        store.fail_artifact("test-id", version)

        found = store.find_by_provenance("hash123")
        assert found is None


class TestBlobIO:
    def test_write_and_read_blob(self, store):
        version = store.create_artifact("test-id", "hash123")
        data = b"test arrow data"

        store.write_blob("test-id", version, data)
        result = store.read_blob("test-id", version)

        assert result == data

    def test_read_nonexistent_blob(self, store):
        result = store.read_blob("nonexistent", 1)
        assert result is None

    def test_blob_exists(self, store):
        version = store.create_artifact("test-id", "hash123")

        assert store.blob_exists("test-id", version) is False

        store.write_blob("test-id", version, b"data")

        assert store.blob_exists("test-id", version) is True

    def test_write_blob_atomic(self, store, artifact_dir):
        """Blob writes are atomic (no partial writes)."""
        version = store.create_artifact("test-id", "hash123")
        data = b"x" * 10000

        store.write_blob("test-id", version, data)

        temp_files = list(artifact_dir.glob("**/*.tmp"))
        assert len(temp_files) == 0

        assert store.read_blob("test-id", version) == data


class TestNamePointers:
    def test_set_and_resolve_name(self, store):
        version = store.create_artifact("test-id", "hash123")
        store.finalize_artifact("test-id", version, "{}", 100, 1000)

        store.set_name("my-artifact", "test-id", version)

        resolved = store.resolve_name("my-artifact")
        assert resolved is not None
        assert resolved.id == "test-id"
        assert resolved.version == version

    def test_resolve_nonexistent_name(self, store):
        resolved = store.resolve_name("nonexistent")
        assert resolved is None

    def test_set_name_requires_ready(self, store):
        version = store.create_artifact("test-id", "hash123")
        # Not finalized

        with pytest.raises(ValueError) as exc_info:
            store.set_name("my-artifact", "test-id", version)
        assert "not ready" in str(exc_info.value)

    def test_set_name_requires_exists(self, store):
        with pytest.raises(ValueError) as exc_info:
            store.set_name("my-artifact", "nonexistent", 1)
        assert "not found" in str(exc_info.value)

    def test_update_name(self, store):
        v1 = store.create_artifact("test-id", "hash1")
        store.finalize_artifact("test-id", v1, "{}", 100, 1000)
        store.set_name("my-artifact", "test-id", v1)

        v2 = store.create_artifact("test-id", "hash2")
        store.finalize_artifact("test-id", v2, "{}", 200, 2000)
        store.set_name("my-artifact", "test-id", v2)

        resolved = store.resolve_name("my-artifact")
        assert resolved.version == v2

    def test_get_name(self, store):
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
        version = store.create_artifact("test-id", "hash123")
        store.finalize_artifact("test-id", version, "{}", 100, 1000)
        store.set_name("my-artifact", "test-id", version)

        assert store.delete_name("my-artifact") is True
        assert store.resolve_name("my-artifact") is None

    def test_delete_nonexistent_name(self, store):
        assert store.delete_name("nonexistent") is False

    def test_list_names(self, store):
        for i in range(3):
            v = store.create_artifact(f"id-{i}", f"hash-{i}")
            store.finalize_artifact(f"id-{i}", v, "{}", i * 100, i * 1000)
            store.set_name(f"name-{i}", f"id-{i}", v)

        names = store.list_names()
        assert len(names) == 3
        assert [n.name for n in names] == ["name-0", "name-1", "name-2"]


class TestCleanup:
    def test_cleanup_failed(self, store, artifact_dir):
        """Cleanup removes failed artifacts older than max age."""
        version = store.create_artifact("test-id", "hash123")
        store.write_blob("test-id", version, b"data")
        store.fail_artifact("test-id", version)

        # Too recent to clean up yet.
        count = store.cleanup_failed(max_age_seconds=3600)
        assert count == 0

        count = store.cleanup_failed(max_age_seconds=0)
        assert count == 1

        assert store.get_artifact("test-id", version) is None
        assert store.blob_exists("test-id", version) is False

    def test_cleanup_failed_takes_a_failed_build_row_with_it(self, store):
        """A server build that failed leaves a build row, whose foreign key refused the delete."""
        from strata.transforms.build_store import BuildStore

        builds = BuildStore(store.db_path, dialect=store.dialect)
        version = store.create_artifact("built", "hash-built")
        builds.create_build("build-1", "built", version, "exec@v1")
        builds.fail_build("build-1", "boom")
        store.fail_artifact("built", version)

        assert store.cleanup_failed(max_age_seconds=-10) == 1
        assert store.get_artifact("built", version) is None
        assert builds.get_build("build-1") is None

    def test_cleanup_preserves_ready(self, store):
        version = store.create_artifact("test-id", "hash123")
        store.finalize_artifact("test-id", version, "{}", 100, 1000)

        # Ready artifacts survive even with 0 age.
        count = store.cleanup_failed(max_age_seconds=0)
        assert count == 0
        assert store.get_artifact("test-id", version) is not None


class TestStats:
    def test_stats_empty(self, store):
        stats = store.stats()
        assert stats["total_versions"] == 0
        assert stats["ready_versions"] == 0
        assert stats["building_versions"] == 0
        assert stats["failed_versions"] == 0
        assert stats["total_bytes"] == 0
        assert stats["total_rows"] == 0
        assert stats["name_count"] == 0

    def test_stats_with_data(self, store):
        v1 = store.create_artifact("id-1", "hash1")
        store.finalize_artifact("id-1", v1, "{}", 100, 1000)

        store.create_artifact("id-2", "hash2")

        v3 = store.create_artifact("id-3", "hash3")
        store.fail_artifact("id-3", v3)

        store.set_name("my-name", "id-1", v1)

        stats = store.stats()
        assert stats["total_versions"] == 3
        assert stats["ready_versions"] == 1
        assert stats["building_versions"] == 1
        assert stats["failed_versions"] == 1
        assert stats["total_bytes"] == 1000
        assert stats["total_rows"] == 100
        assert stats["name_count"] == 1

    def test_byte_totals_count_what_versions_still_hold_as_the_sweep_does(self, store):
        """An older version a refresh superseded keeps its bytes until a sweep takes them."""
        for size in (1000, 300):
            version = store.create_artifact("id-1", "hash1")
            store.finalize_artifact("id-1", version, "{}", 1, size)
        store.create_artifact("id-2", "hash1")
        store.finalize_artifact("id-2", 1, "{}", 1, 300)  # reads id-1@v=2's bytes

        swept = store.garbage_collect(dry_run=True)["store_bytes"]
        assert store.stats()["total_bytes"] == swept == 1300
        assert store.get_usage()["total_bytes"] == 1300

    def test_superseded_versions_are_counted_so_the_states_add_up(self, store):
        """A refresh and a dedup each leave a superseded version, which no other count holds."""
        for _ in range(2):
            version = store.create_artifact("id-1", "hash1")
            store.finalize_artifact("id-1", version, "{}", 1, 10)
        store.create_artifact("id-2", "hash1")
        store.finalize_artifact("id-2", 1, "{}", 1, 10)
        store.create_artifact("id-3", "hash3")

        for counts in (store.stats(), store.get_usage()):
            assert counts["superseded_versions"] == 2
            assert counts["total_versions"] == sum(
                counts[f"{state}_versions"]
                for state in ("ready", "building", "superseded", "failed")
            )

    def test_name_prefix_treats_an_underscore_literally(self, store):
        for name, artifact_id in (("taxi_model", "a"), ("taxi/model", "b")):
            version = store.create_artifact(artifact_id, f"hash-{artifact_id}")
            store.finalize_artifact(artifact_id, version, "{}", 1, 1)
            store.set_name(name, artifact_id, version)

        assert [a.id for a in store.list_artifacts(name_prefix="taxi_")] == ["a"]


def _ipc_bytes(num_rows: int) -> bytes:
    """A single valid Arrow IPC stream with ``num_rows`` rows."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, pa.schema([("id", pa.int64())])) as writer:
        writer.write_batch(pa.RecordBatch.from_pydict({"id": list(range(num_rows))}))
    return sink.getvalue().to_pybytes()


class TestForeignKeyEnforcement:
    """Declared foreign keys are enforced, and deletes clean up so enforcement does not break
    collection.

    SQLite ignores FOREIGN KEY clauses unless ``PRAGMA foreign_keys=ON`` is set per connection;
    Postgres always enforces them.
    """

    def _ready(self, store, artifact_id, provenance):
        version = store.create_artifact(artifact_id, provenance)
        store.blob_store.write_blob(artifact_id, version, b"x")
        store.finalize_artifact(artifact_id, version, "{}", 1, 1)
        return version

    def test_a_name_cannot_point_at_a_version_that_does_not_exist(self, store):
        conn = store._get_connection()
        try:
            with pytest.raises(store.dialect.integrity_error):
                conn.execute(
                    "INSERT INTO artifact_names (name, artifact_id, version, updated_at, tenant) "
                    "VALUES (?, ?, ?, ?, ?)",
                    ("ghost", "art-missing", 1, 0.0, ""),
                )
                conn.commit()
        finally:
            conn.close()

    def test_a_version_cannot_be_deleted_out_from_under_its_name(self, store):
        self._ready(store, "art-1", "prov-a")
        store.set_name("team/model", "art-1", 1)

        conn = store._get_connection()
        try:
            with pytest.raises(store.dialect.integrity_error):
                conn.execute(
                    "DELETE FROM artifact_versions WHERE id = ? AND version = ?", ("art-1", 1)
                )
                conn.commit()
        finally:
            conn.close()

        assert store.get_artifact("art-1", 1) is not None

    def test_delete_artifact_takes_the_build_row_with_it(self, tmp_path, store):
        from strata.transforms.build_store import BuildStore

        builds = BuildStore(store.db_path)
        self._ready(store, "art-1", "prov-a")
        builds.create_build("build-1", "art-1", 1, "exec@v1")

        # Without the cascade this raises: every server-built artifact has a
        # build row, so the foreign key would make it undeletable.
        assert store.delete_artifact("art-1", 1) is True
        assert builds.get_build("build-1") is None

    def test_garbage_collect_takes_build_rows_with_it(self, tmp_path, store):
        from strata.transforms.build_store import BuildStore

        builds = BuildStore(store.db_path)
        self._ready(store, "art-1", "prov-a")
        self._ready(store, "art-1", "prov-b")
        builds.create_build("build-1", "art-1", 1, "exec@v1")
        # Age v1 past the cutoff; v2 stays as the artifact's current value.
        conn = store._get_connection()
        try:
            conn.execute(
                "UPDATE artifact_versions SET created_at = 0, last_used_at = 0 "
                "WHERE id = ? AND version = 1",
                ("art-1",),
            )
            conn.commit()
        finally:
            conn.close()

        stats = store.garbage_collect(max_idle_days=1.0)

        assert stats["deleted_count"] == 1
        assert store.get_artifact("art-1", 1) is None
        assert builds.get_build("build-1") is None


class TestVersionPromotion:
    """Re-pointing ``latest`` at a result the artifact already holds.

    Consumers read through ``get_latest_version``, so returning to a previous provenance is a cache
    hit only if promotion re-records the older version.
    """

    def _ready(self, store, artifact_id, provenance, blob):
        version = store.create_artifact(artifact_id, provenance)
        store.blob_store.write_blob(artifact_id, version, blob)
        store.finalize_artifact(artifact_id, version, "{}", 1, len(blob))
        return version

    def test_finds_a_superseded_version_of_the_same_artifact(self, store):
        # v1 is superseded by the time we look for it: an intervening rebuild
        # under the same provenance demotes it. That row is the one worth
        # finding, so the search cannot be restricted to ready.
        self._ready(store, "art-1", "prov-a", b"a1")
        self._ready(store, "art-1", "prov-a", b"a2")
        assert store.get_artifact("art-1", 1).state == "superseded"

        found = store.find_version_by_provenance("art-1", "prov-a")
        assert found is not None
        assert found.version == 2

    def test_does_not_reach_into_another_artifact(self, store):
        # find_by_provenance answers "does anyone hold this"; this answers
        # "does *this* artifact hold it", and must not confuse the two.
        self._ready(store, "art-other", "prov-a", b"a1")

        assert store.find_version_by_provenance("art-1", "prov-a") is None

    def test_promotion_makes_an_older_version_current(self, store):
        self._ready(store, "art-1", "prov-a", b"first")
        self._ready(store, "art-1", "prov-b", b"second")
        assert store.get_latest_version("art-1").provenance_hash == "prov-b"

        promoted = store.promote_version("art-1", 1)

        assert promoted is not None
        assert promoted.id == "art-1"
        assert promoted.version == 3
        assert promoted.provenance_hash == "prov-a"
        # What a consumer sees, which is the whole point.
        latest = store.get_latest_version("art-1")
        assert latest.version == 3
        assert latest.provenance_hash == "prov-a"
        assert store.read_blob("art-1", 3) == b"first"
        # And the row it was copied from steps aside, exactly as a rebuild
        # under the same provenance would leave it.
        assert store.get_artifact("art-1", 1).state == "superseded"

    def test_promotion_reports_failure_when_the_blob_is_gone(self, store):
        # A GC pass can take the blob while the row survives. The caller falls
        # back to recomputing, so this must say so rather than register a
        # version pointing at nothing.
        self._ready(store, "art-1", "prov-a", b"first")
        self._ready(store, "art-1", "prov-b", b"second")
        store.blob_store.delete_blob("art-1", 1)

        assert store.promote_version("art-1", 1) is None
        assert store.get_latest_version("art-1").provenance_hash == "prov-b"
        # And nothing was registered on the way out: opening the source before
        # inserting the row is what keeps a failed promotion from stranding a
        # building version that points at no blob.
        assert store.get_artifact("art-1", 3) is None

    def test_promotion_refuses_a_version_that_was_never_ready(self, store):
        # A building row may be a partial or abandoned write and a failed one
        # was rejected on purpose. Neither is a result to make current.
        self._ready(store, "art-1", "prov-a", b"first")
        building = store.create_artifact("art-1", "prov-c")
        store.blob_store.write_blob("art-1", building, b"partial")

        assert store.promote_version("art-1", building) is None
        assert store.get_latest_version("art-1").provenance_hash == "prov-a"

    def test_lookup_does_not_cross_tenants(self, store):
        # An artifact id is not an isolation boundary, so a tenantless lookup must not
        # select a row another tenant wrote (why find_by_provenance is tenant-scoped).
        version = store.create_artifact("art-1", "prov-a", tenant="acme")
        store.blob_store.write_blob("art-1", version, b"theirs")
        store.finalize_artifact("art-1", version, "{}", 1, 6)

        assert store.find_version_by_provenance("art-1", "prov-a") is None
        assert store.find_version_by_provenance("art-1", "prov-a", tenant="acme") is not None

    def test_promotion_copies_a_blob_larger_than_one_stream_chunk(self, store):
        # The copy is streamed rather than read into memory; a payload spanning
        # several chunks is what tells a chunk-boundary bug from a working one.
        from strata.blob_store import BLOB_STREAM_CHUNK_BYTES

        payload = bytes(range(256)) * ((BLOB_STREAM_CHUNK_BYTES * 2) // 256 + 3)
        self._ready(store, "art-1", "prov-a", payload)
        self._ready(store, "art-1", "prov-b", b"small")

        promoted = store.promote_version("art-1", 1)

        assert promoted is not None
        assert store.read_blob("art-1", promoted.version) == payload
        assert promoted.byte_size == len(payload)

    def test_promotion_keeps_the_canonical_id_when_dedup_diverts(self, store):
        # Another artifact already holds this provenance in ready state, so
        # finalize_artifact would hand back *its* id and fail ours. Callers
        # resolve by id, so the promotion has to end with our id current.
        self._ready(store, "art-1", "prov-a", b"first")
        self._ready(store, "art-1", "prov-b", b"second")
        self._ready(store, "art-twin", "prov-a", b"first")

        promoted = store.promote_version("art-1", 1)

        assert promoted is not None
        assert promoted.id == "art-1"
        assert store.get_latest_version("art-1").provenance_hash == "prov-a"

    def test_promotion_keeps_its_own_bytes_when_another_id_holds_the_provenance(self, store):
        """A non-deterministic cell's twin holds other bytes; the reverted cell gets its own."""
        from strata.artifact_store import StagedVersion

        self._ready(store, "art-1", "prov-a", b"mine-bytes")
        self._ready(store, "art-1", "prov-b", b"second")
        twin = store.create_artifact("art-twin", "prov-a")
        store.write_blob("art-twin", twin, b"twin-bytes-differ")
        store.finalize_canonical_together([StagedVersion("art-twin", twin, "{}", 1, 17, "digest")])

        promoted = store.promote_version("art-1", 1)

        assert promoted is not None
        assert (promoted.id, promoted.state) == ("art-1", "ready")
        assert store.read_blob("art-1", promoted.version) == b"mine-bytes"
        assert promoted.byte_size == len(b"mine-bytes")
        assert store.read_blob("art-twin", twin) == b"twin-bytes-differ"


class TestRefreshSupersede:
    """Refresh rebuilds become new versions of the same artifact."""

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

    def test_different_id_dedup_supersedes_the_duplicate(self, store):
        """The cross-id duplicate race keeps one ready row; the loser is overtaken, not failed.

        It finished, so it keeps its metadata and stays readable by its own id and version,
        and usage counts no failure. Its own bytes go at once: it reads the canonical's, the
        store holds them once, and retention collects it without touching them.
        """
        store.create_artifact("art-1", "prov-x")
        store.write_blob("art-1", 1, _ipc_bytes(10))
        store.finalize_artifact("art-1", 1, "{}", 10, 100)

        store.create_artifact("art-2", "prov-x")
        store.write_blob("art-2", 1, _ipc_bytes(10))
        result = store.finalize_artifact("art-2", 1, "{}", 10, 100)

        assert (result.id, result.version) == ("art-1", 1)
        assert store.get_artifact("art-1", 1).state == "ready"
        overtaken = store.get_artifact("art-2", 1)
        assert overtaken.state == "superseded"
        assert (overtaken.row_count, overtaken.byte_size) == (10, 100)
        assert overtaken.content_sha256 == store.get_artifact("art-1", 1).content_sha256
        assert store.read_blob("art-2", 1) == _ipc_bytes(10)
        assert not store.blob_store.blob_exists("art-2", 1)
        assert store.get_usage()["failed_versions"] == 0
        assert store.get_usage()["total_bytes"] == 100
        assert store.garbage_collect(dry_run=True)["store_bytes"] == 100
        assert store.verify_artifacts() == []

        store.set_name("canonical", "art-1", 1)
        collected = store.garbage_collect(max_idle_days=0, collect_latest=True)
        assert collected["deleted_count"] == 1
        assert collected["deleted_bytes"] == 0
        assert store.get_artifact("art-2", 1) is None
        assert store.read_blob("art-1", 1) == _ipc_bytes(10)
        assert store.verify_artifacts() == []

    def test_an_overtaken_version_whose_canonical_went_is_reported(self, store):
        """A deleted canonical leaves the pointer dangling; verify names it rather than
        reporting an orphan blob."""
        store.create_artifact("art-1", "prov-x")
        store.write_blob("art-1", 1, _ipc_bytes(10))
        store.finalize_artifact("art-1", 1, "{}", 10, 100)
        store.create_artifact("art-2", "prov-x")
        store.write_blob("art-2", 1, _ipc_bytes(10))
        store.finalize_artifact("art-2", 1, "{}", 10, 100)

        assert store.delete_artifact("art-1", 1)

        assert store.read_blob("art-2", 1) is None
        [finding] = store.verify_artifacts()
        assert (finding["artifact_id"], finding["problem"]) == ("art-2", "missing_blob")
        assert "art-1@v=1" in finding["detail"]

    @pytest.mark.parametrize("hold", ["publish", "pin"])
    def test_deleting_a_canonical_hands_its_bytes_to_a_held_version_reading_them(self, store, hold):
        """A published or pinned version is a link somebody holds; the delete must not empty it."""
        store.create_artifact("art-1", "prov-x")
        store.write_blob("art-1", 1, _ipc_bytes(10))
        store.finalize_artifact("art-1", 1, "{}", 10, 100)
        store.create_artifact("art-2", "prov-x")
        store.write_blob("art-2", 1, _ipc_bytes(10))
        store.finalize_artifact("art-2", 1, "{}", 10, 100)
        if hold == "publish":
            store.publish_artifact("art-2", 1)
        else:
            store.pin_artifact("art-2", 1, "review")

        assert store.delete_artifact("art-1", 1)

        assert store.read_blob("art-2", 1) == _ipc_bytes(10)
        assert store.blob_store.blob_exists("art-2", 1)
        assert store.verify_artifacts() == []

    @pytest.mark.parametrize("hold", ["publish", "pin"])
    def test_deleting_a_canonical_whose_blob_is_gone_under_a_held_reader_refuses(self, store, hold):
        """With no bytes to hand over, the delete would leave the held link empty: it stops."""
        store.create_artifact("art-1", "prov-x")
        store.write_blob("art-1", 1, _ipc_bytes(10))
        store.finalize_artifact("art-1", 1, "{}", 10, 100)
        store.create_artifact("art-2", "prov-x")
        store.write_blob("art-2", 1, _ipc_bytes(10))
        store.finalize_artifact("art-2", 1, "{}", 10, 100)
        if hold == "publish":
            store.publish_artifact("art-2", 1)
        else:
            store.pin_artifact("art-2", 1, "review")
        store.blob_store.delete_blob("art-1", 1)

        with pytest.raises(ValueError, match="whose blob is gone"):
            store.delete_artifact("art-1", 1)

        assert store.get_artifact("art-1", 1) is not None
        conn = store._get_connection()
        try:
            held = conn.execute(
                "SELECT state, superseded_by FROM artifact_versions WHERE id = ? AND version = 1",
                ("art-2",),
            ).fetchone()
        finally:
            conn.close()
        assert (held["state"], held["superseded_by"]) == ("superseded", "art-1@v=1")

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
    """Stale building artifacts are demoted to failed."""

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
    """Store-wide blob/metadata consistency check."""

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
        """Concatenated IPC streams are flagged as invalid_stream."""
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
    """`strata artifact verify` surfaces store inconsistencies."""

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

    def test_it_names_the_metadata_and_blobs_it_checks(self, store, artifact_dir, capsys):
        assert self._run(artifact_dir) == 0
        out = capsys.readouterr().out
        assert f"metadata {store.db_path}" in out
        assert f"blobs {store.blob_store.blobs_dir}" in out

    def test_a_remote_store_is_named_by_its_dsn_and_bucket(self, tmp_path):
        """The artifact dir is not what a Postgres + S3 store verifies."""
        from strata.artifact_cli import _store_location
        from strata.blob_store import S3BlobStore
        from strata.sql_backend import PostgresDialect

        remote = ArtifactStore.__new__(ArtifactStore)
        remote._dialect = PostgresDialect("postgresql://strata:s3cret@db:5432/strata")
        remote.blob_store = S3BlobStore("lake", prefix="artifacts", region="us-east-1")

        metadata, blobs = _store_location(remote)

        assert metadata == "postgresql://strata:***@db:5432/strata"
        assert blobs == "s3://lake/artifacts"


class TestLegacyDefaultTenantNames:
    """Artifacts stamped with legacy '_default' stay nameable."""

    def test_tenantless_request_can_name_default_artifact(self, store):
        store.create_artifact("legacy", "prov-l", tenant="_default")
        store.finalize_artifact("legacy", 1, "{}", 1, 10)

        # Older PUT uploads stamped "_default" while the names routes resolve single-tenant
        # requests to None; that combination must not leave the artifact unnameable.
        store.set_name("legacy-name", "legacy", 1, tenant=None)  # must not raise
        resolved = store.resolve_name("legacy-name")
        assert (resolved.id, resolved.version) == ("legacy", 1)

    def test_real_tenant_mismatch_still_rejected(self, store):
        store.create_artifact("owned", "prov-o", tenant="acme")
        store.finalize_artifact("owned", 1, "{}", 1, 10)

        # Answered as a missing artifact: the error must not confirm it exists, or name its tenant.
        with pytest.raises(ArtifactNotFoundError, match="not found") as excinfo:
            store.set_name("steal", "owned", 1, tenant="globex")
        assert "acme" not in str(excinfo.value)


def _make_ready_artifact(store, artifact_id: str, provenance: str) -> None:
    store.create_artifact(artifact_id, provenance)
    store.write_blob(artifact_id, 1, _ipc_bytes(1))
    store.finalize_artifact(artifact_id, 1, "{}", 1, 10)


class TestAliases:
    """Registry aliases: intent pointers on a name."""

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
        """Promotion must not lose the old champion."""
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
    """Version tags: facts about one artifact build."""

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
    """Append-only audit of name/alias/tag mutations."""

    def test_name_moves_audited_with_history(self, store):
        _make_ready_artifact(store, "m1", "prov-1")
        _make_ready_artifact(store, "m2", "prov-2")
        store.set_name("demo/model", "m1", 1)
        store.set_name("demo/model", "m2", 1)  # the silent swap, recorded

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
    """Approval-gate mechanics: request, approve, reject."""

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
    """GC and delete respect alias pointers."""

    def test_gc_spares_aliased_artifact(self, store):
        _make_ready_artifact(store, "pinned", "prov-p")
        store.set_alias("demo/model", "champion", "pinned", 1)

        result = store.garbage_collect(max_idle_days=0)
        assert store.get_artifact("pinned", 1) is not None, result
        assert store.resolve_alias("demo/model", "champion") is not None

    def test_gc_spares_aliased_superseded_version(self, store):
        """An alias pinning a superseded version keeps it from GC."""
        _make_ready_artifact(store, "model", "prov-s")
        store.create_artifact("model", "prov-s")
        store.write_blob("model", 2, _ipc_bytes(1))
        store.finalize_artifact("model", 2, "{}", 1, 10)  # supersedes v1
        store.set_alias("demo/model", "champion", "model", 1)
        store.set_name("demo/model", "model", 2)  # name guards v2

        store.garbage_collect(max_idle_days=0)

        champion = store.resolve_alias("demo/model", "champion")
        assert champion is not None and champion.version == 1

    def test_gc_still_collects_unreferenced(self, store):
        """A lone unnamed artifact is a current value; reclaiming it takes the explicit opt-in."""
        _make_ready_artifact(store, "loose", "prov-l")
        result = store.garbage_collect(max_idle_days=0, collect_latest=True)
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
    """approve_alias_change is one transaction."""

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


def _no_wait_store(artifact_dir):
    """A second store on the same database (as another process would open) that never waits."""
    competitor = ArtifactStore(artifact_dir)
    connect = competitor._get_connection

    def no_wait():
        conn = connect()
        conn.execute("PRAGMA busy_timeout = 0")
        return conn

    competitor._get_connection = no_wait
    return competitor


_DECISIONS = {
    "approve": lambda s: s.approve_alias_change("demo/model", "champion"),
    "reject": lambda s: s.reject_alias_change("demo/model", "champion"),
}


class TestPendingChangeRaces:
    """Reading a pending change and consuming it is one step, even across connections."""

    @pytest.mark.parametrize(("first", "second"), [("approve", "reject"), ("reject", "approve")])
    def test_an_approve_and_a_reject_of_one_change_have_one_winner(
        self, store, artifact_dir, first, second
    ):
        _make_ready_artifact(store, "m1", "prov-1")
        store.request_alias_change("demo/model", "champion", "set", artifact_id="m1", version=1)

        won, lost = race_after_pending_read(
            store, _no_wait_store(artifact_dir), _DECISIONS[first], _DECISIONS[second]
        )

        assert won["artifact_id"] == "m1"
        assert isinstance(lost, ValueError) and "No pending change" in str(lost)
        actions = [e["action"] for e in store.read_audit(name="demo/model")]
        assert ("alias_approved" in actions, "alias_rejected" in actions) == (
            first == "approve",
            first == "reject",
        )
        assert (store.resolve_alias("demo/model", "champion") is not None) == (first == "approve")

    def test_an_approval_leaves_a_newer_request_pending(self, store, artifact_dir):
        _make_ready_artifact(store, "m1", "prov-1")
        _make_ready_artifact(store, "m2", "prov-2")
        store.request_alias_change("demo/model", "champion", "set", artifact_id="m1", version=1)

        applied, requested = race_after_pending_read(
            store,
            _no_wait_store(artifact_dir),
            _DECISIONS["approve"],
            lambda s: s.request_alias_change(
                "demo/model", "champion", "set", artifact_id="m2", version=1
            ),
        )

        assert applied["artifact_id"] == "m1" and requested is True
        assert store.resolve_alias("demo/model", "champion").id == "m1"
        audit = store.read_audit(name="demo/model")
        assert [e["artifact_id"] for e in audit if e["action"] == "alias_approved"] == ["m1"]
        assert [p["artifact_id"] for p in store.list_pending_changes()] == ["m2"]


class TestCreateArtifactVersionRace:
    """Concurrent creates for one artifact id allocate distinct versions."""

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
            except Exception as e:  # noqa: BLE001 (collecting for assertion)
                errors.append(e)

        threads = [threading.Thread(target=create, args=(i,)) for i in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert errors == []
        assert sorted(versions) == [2, 3, 4, 5]


class TestAliasIdempotence:
    """Identical-target alias writes are no-ops."""

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
    """read_audit isolates by tenant for request-serving callers."""

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
        """No tenant arg gives the whole-store view (CLI / admin)."""
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
    """approve_alias_change can forbid self-approval."""

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
    """Reverse lookups for the per-cell published-artifacts route: artifacts a cell published
    (``nb_cell`` tag) and the names pointing at a version.
    """

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


class TestListArtifactsSortAndSince:
    """The since / sort / order filters behind the artifacts dashboard."""

    @staticmethod
    def _add(store, aid, byte_size):
        store.create_artifact(aid, f"{aid}-hash")
        store.finalize_artifact(aid, 1, "{}", 0, byte_size)

    def test_sort_by_byte_size(self, store):
        self._add(store, "small", 100)
        self._add(store, "big", 300)
        self._add(store, "mid", 200)
        asc = [a.id for a in store.list_artifacts(sort="byte_size", order="asc")]
        assert asc == ["small", "mid", "big"]
        desc = [a.id for a in store.list_artifacts(sort="byte_size", order="desc")]
        assert desc == ["big", "mid", "small"]

    def test_unknown_sort_falls_back_safely(self, store):
        # A non-whitelisted sort (or an injection attempt) falls back to
        # created_at rather than reaching the SQL.
        self._add(store, "a", 1)
        result = store.list_artifacts(sort="; DROP TABLE artifact_versions;--")
        assert [a.id for a in result] == ["a"]

    def test_since_filter(self, store):
        import time

        self._add(store, "a", 1)
        assert len(store.list_artifacts(since=0)) == 1
        assert store.list_artifacts(since=time.time() + 3600) == []

    def test_names_for_artifact(self, store):
        store.create_artifact("a1", "h1")
        store.finalize_artifact("a1", 1, "{}", 0, 0)
        store.set_name("team/model", "a1", 1)

        assert store.names_for_artifact("a1", 1) == ["team/model"]
        assert store.names_for_artifact("a1", 2) == []
        assert store.names_for_artifact("nope", 1) == []


class TestTenantNormalization:
    """Tenant is stored as '' (never NULL) for tenantless rows, and the isolation and uniqueness
    guarantees that depend on it.
    """

    def test_tenantless_provenance_uniqueness_enforced(self, store):
        """Two tenantless artifacts with one provenance cannot both be ready; NULL would be
        distinct.
        """
        store.create_artifact("art-1", "dup-prov")
        store.finalize_artifact("art-1", 1, "{}", 1, 10)
        store.create_artifact("art-2", "dup-prov")
        result = store.finalize_artifact("art-2", 1, "{}", 1, 10)
        assert result.id == "art-1"  # deduped to the first

        import sqlite3

        conn = sqlite3.connect(str(store.db_path))
        ready = conn.execute(
            "SELECT COUNT(*) FROM artifact_versions "
            "WHERE provenance_hash = 'dup-prov' AND state = 'ready'"
        ).fetchone()[0]
        conn.close()
        assert ready == 1

    def test_tenantless_finalize_does_not_dedup_cross_tenant(self, store):
        """A tenantless finalize must not dedup against a tenant's artifact and name it."""
        store.create_artifact("team-art", "shared-prov", tenant="team-a")
        store.finalize_artifact("team-art", 1, "{}", 1, 10)

        store.create_artifact("none-art", "shared-prov")  # tenantless
        result = store.finalize_and_set_name(
            "none-art", 1, "{}", 1, 10, name="my-name", tenant=None
        )
        # It finalizes as its own tenantless artifact, NOT team-a's.
        assert result.id == "none-art"
        resolved = store.resolve_name("my-name")
        assert resolved.id == "none-art"

    @staticmethod
    def _staged_over_a_ready_equal(store, provenance):
        """``canonical`` staged with the provenance ``equiv`` already holds ready, in team-a."""
        from strata.artifact_store import StagedVersion

        store.create_artifact("equiv", provenance, tenant="team-a")
        store.write_blob("equiv", 1, b"x" * 10)
        store.finalize_artifact("equiv", 1, "{}", 1, 10)
        store.create_artifact("canonical", provenance, tenant="team-a")
        store.write_blob("canonical", 1, b"x" * 10)
        return StagedVersion("canonical", 1, "{}", 1, 10, "digest")

    def test_canonical_finalize_retries_past_a_losing_race(self, store):
        """A competing writer can commit our provenance between our supersede and promote; the retry
        must still promote.

        SQLite's BEGIN IMMEDIATE prevents this in production, so the conflict is injected; the
        retry loop is dialect-agnostic.
        """
        import sqlite3

        staged = self._staged_over_a_ready_equal(store, "race-prov")

        real_get_connection = store._get_connection
        # Counting commits, not connections: the trailing get_artifact opens
        # one more connection but never commits.
        commits = []

        class _FailsCommitOnce:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def commit(self):
                commits.append(1)
                if len(commits) == 1:
                    raise sqlite3.IntegrityError("injected UNIQUE constraint failure")
                return self._inner.commit()

        store._get_connection = lambda: _FailsCommitOnce(real_get_connection())
        try:
            [promoted] = store.finalize_canonical_together([staged])
        finally:
            store._get_connection = real_get_connection

        assert len(commits) == 2, "expected exactly one retry after the conflict"
        assert promoted is not None
        assert promoted.id == "canonical", "must never hand back the foreign winner"
        assert promoted.state == "ready"

    def test_canonical_finalize_surfaces_a_persistent_conflict(self, store):
        """Sustained contention must surface, not report a promotion that did not happen."""
        import sqlite3

        staged = self._staged_over_a_ready_equal(store, "race-prov")

        real_get_connection = store._get_connection

        class _AlwaysFailsCommit:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def commit(self):
                raise sqlite3.IntegrityError("injected UNIQUE constraint failure")

        store._get_connection = lambda: _AlwaysFailsCommit(real_get_connection())
        try:
            with pytest.raises(sqlite3.IntegrityError):
                store.finalize_canonical_together([staged])
        finally:
            store._get_connection = real_get_connection

    def test_canonical_finalize_supersedes_conflicting_ready_row(self, store):
        """One ready row per (tenant, provenance) afterwards, for tenant-scoped rows too."""
        staged = self._staged_over_a_ready_equal(store, "canon-prov")

        [promoted] = store.finalize_canonical_together([staged])
        assert promoted is not None and promoted.state == "ready"

        import sqlite3

        conn = sqlite3.connect(str(store.db_path))
        ready_ids = [
            r[0]
            for r in conn.execute(
                "SELECT id FROM artifact_versions "
                "WHERE provenance_hash = 'canon-prov' AND state = 'ready'"
            ).fetchall()
        ]
        conn.close()
        assert ready_ids == ["canonical"]  # exactly one ready, the canonical
        # The equivalent is still fetchable by id+version.
        assert store.get_artifact("equiv", 1).state == "superseded"

    def test_set_tag_rejects_cross_tenant(self, store):
        """set_tag enforces ownership like set_alias."""
        store.create_artifact("owned", "p", tenant="acme")
        store.finalize_artifact("owned", 1, "{}", 1, 10)
        with pytest.raises(ArtifactNotFoundError, match="not found") as excinfo:
            store.set_tag("owned", 1, "k", "v", tenant="globex")
        assert "acme" not in str(excinfo.value)

    def test_set_tag_rejects_non_readable(self, store):
        """Tags are only allowed on readable (ready/superseded) artifacts."""
        store.create_artifact("building", "p")  # still 'building'
        with pytest.raises(ValueError, match="not readable"):
            store.set_tag("building", 1, "k", "v")

    def test_delete_artifact_refuses_other_tenant(self, store):
        """delete_artifact refuses to delete another tenant's artifact."""
        store.create_artifact("owned", "p", tenant="acme")
        store.finalize_artifact("owned", 1, "{}", 1, 10)
        assert store.delete_artifact("owned", 1, tenant="globex") is False
        assert store.get_artifact("owned", 1) is not None  # not deleted
        assert store.delete_artifact("owned", 1, tenant="acme") is True

    def test_migration_normalizes_null_tenant_and_dedups(self, artifact_dir):
        """A legacy store with NULL tenants and a duplicate ready row normalizes to '' and collapses
        the duplicate so the uniqueness index holds.
        """
        import sqlite3

        store = ArtifactStore(artifact_dir)
        store.create_artifact("a1", "h")
        store.finalize_artifact("a1", 1, "{}", 1, 10)

        # Rebuild artifact_versions as the legacy nullable-tenant table, then plant NULL
        # tenants and a second ready row with the same provenance (the duplicate the old
        # NULL-distinct index permitted).
        conn = sqlite3.connect(str(store.db_path))
        conn.executescript(
            """
            DROP INDEX IF EXISTS idx_tenant_provenance_unique;
            ALTER TABLE artifact_versions RENAME TO _av_old;
            CREATE TABLE artifact_versions (
                id TEXT NOT NULL, version INTEGER NOT NULL,
                state TEXT NOT NULL DEFAULT 'building', provenance_hash TEXT NOT NULL,
                schema_json TEXT, row_count INTEGER, byte_size INTEGER,
                created_at REAL NOT NULL, transform_spec TEXT, input_versions TEXT,
                tenant TEXT, principal TEXT, PRIMARY KEY (id, version)
            );
            -- Columns named, not ``SELECT *``: this table is deliberately the
            -- legacy shape, so every column added to the current schema would
            -- otherwise break the copy and this test would fail on each new
            -- migration for a reason that has nothing to do with tenants.
            INSERT INTO artifact_versions
                (id, version, state, provenance_hash, schema_json, row_count,
                 byte_size, created_at, transform_spec, input_versions,
                 tenant, principal)
            SELECT id, version, state, provenance_hash, schema_json, row_count,
                   byte_size, created_at, transform_spec, input_versions,
                   tenant, principal
            FROM _av_old;
            DROP TABLE _av_old;
            """
        )
        conn.execute("UPDATE artifact_versions SET tenant = NULL")
        conn.execute(
            "INSERT INTO artifact_versions "
            "(id, version, state, provenance_hash, created_at, tenant) "
            "VALUES ('a2', 1, 'ready', 'h', 1.0, NULL)"
        )
        conn.commit()
        conn.close()

        # Reopen -> _init_schema migration runs.
        ArtifactStore(artifact_dir)

        conn = sqlite3.connect(str(store.db_path))
        n_null = conn.execute(
            "SELECT COUNT(*) FROM artifact_versions WHERE tenant IS NULL"
        ).fetchone()[0]
        n_ready = conn.execute(
            "SELECT COUNT(*) FROM artifact_versions WHERE provenance_hash = 'h' AND state = 'ready'"
        ).fetchone()[0]
        conn.close()
        assert n_null == 0  # all normalized to ''
        assert n_ready == 1  # duplicate collapsed


class TestGcSparesCurrentValues:
    """GC must not delete the current value of an artifact id.

    Notebook cell outputs (``nb_{notebook}_cell_{cell}_var_{name}``) are never named, so
    ``get_latest_version`` is their only handle; treating unnamed as garbage would delete every
    notebook's state after the GC cutoff.
    """

    def test_gc_spares_an_unnamed_notebook_cell_artifact(self, store):
        _make_ready_artifact(store, "nb_abc_cell_def_var_model", "prov-nb")

        result = store.garbage_collect(max_idle_days=0)

        assert result["deleted_count"] == 0
        assert store.get_latest_version("nb_abc_cell_def_var_model") is not None

    def test_gc_still_collects_superseded_versions(self, store):
        """Older versions are still collected."""
        _make_ready_artifact(store, "model", "prov-v1")
        store.create_artifact("model", "prov-v2")
        store.write_blob("model", 2, _ipc_bytes(1))
        store.finalize_artifact("model", 2, "{}", 1, 10)  # supersedes v1

        result = store.garbage_collect(max_idle_days=0)

        assert result["deleted_count"] == 1
        assert store.get_artifact("model", 1) is None
        assert store.get_artifact("model", 2) is not None  # current value kept

    def test_gc_spares_the_current_value_while_a_rebuild_is_in_flight(self, store):
        """A rebuild's ``building`` row is the highest version, so "spare MAX(version)" would leave
        the readable value unprotected during the build and after it fails.
        """
        artifact_id = "nb_x_cell_c1_var_df"
        _make_ready_artifact(store, artifact_id, "prov-v1")
        rebuild = store.create_artifact(artifact_id, "prov-v2")

        assert store.garbage_collect(max_idle_days=0)["deleted_count"] == 0
        assert store.get_latest_version(artifact_id).version == 1

        store.fail_artifact(artifact_id, rebuild)
        store.garbage_collect(max_idle_days=0)
        assert store.get_latest_version(artifact_id).version == 1

    def test_collect_latest_opt_in_still_reclaims(self, store):
        _make_ready_artifact(store, "loose", "prov-loose")
        assert store.garbage_collect(max_idle_days=0)["deleted_count"] == 0
        assert store.garbage_collect(max_idle_days=0, collect_latest=True)["deleted_count"] == 1


class TestGcDeletesMetadataBeforeBlobs:
    """Metadata is deleted and committed before best-effort blob removal, so a mid-GC crash leaves
    orphaned bytes, not ready rows without blobs.
    """

    def test_blob_failure_does_not_abort_gc_or_strand_ready_rows(self, store, monkeypatch):
        _make_ready_artifact(store, "model", "prov-v1")
        store.create_artifact("model", "prov-v2")
        store.write_blob("model", 2, _ipc_bytes(1))
        store.finalize_artifact("model", 2, "{}", 1, 10)

        def boom(artifact_id, version):
            raise OSError("backend unavailable")

        monkeypatch.setattr(store.blob_store, "delete_blob", boom)

        result = store.garbage_collect(max_idle_days=0)

        # The run completed and the metadata is gone: no ready row survives pointing at a
        # blob we may or may not have removed.
        assert result["deleted_count"] == 1
        assert store.get_artifact("model", 1) is None


class TestTenantlessArtifactsAreVisibleToScopedQueries:
    """Tenantless rows are stored as ``''``, so tenant filters must match ``''``, not only NULL.

    Otherwise dedup still resolves these rows while listing, usage, stats, verify and GC cannot see
    them: a permanent leak.
    """

    def _seed(self, store):
        _make_ready_artifact(store, "legacy", "prov-legacy")  # tenantless ('')
        version = store.create_artifact("scoped", "prov-scoped", tenant="team-a")
        store.write_blob("scoped", version, _ipc_bytes(1))
        store.finalize_artifact("scoped", version, "{}", 1, 10)

    def test_tenantless_rows_are_stored_as_empty_string_not_null(self, store):
        self._seed(store)
        artifact = store.get_artifact("legacy", 1)
        assert artifact.tenant == "", "schema stores '' for tenantless rows"

    def test_list_artifacts_includes_legacy_rows(self, store):
        self._seed(store)
        ids = {a.id for a in store.list_artifacts(tenant="team-a")}
        assert "scoped" in ids
        assert "legacy" in ids, "legacy tenantless artifact was invisible to the tenant"

    def test_usage_and_stats_count_legacy_rows(self, store):
        self._seed(store)
        assert store.get_usage(tenant="team-a")["total_versions"] >= 2
        assert store.stats(tenant="team-a")["total_versions"] >= 2

    def test_verify_sees_legacy_rows(self, store):
        self._seed(store)
        # verify walks the same filter; it must not skip legacy rows.
        store.verify_artifacts(tenant="team-a")

    def test_gc_can_finally_reach_legacy_rows(self, store):
        """Asserts reachability rather than an exact count, so the GC policy is not encoded here."""
        self._seed(store)
        store.create_artifact("legacy", "prov-legacy-v2")  # supersede v1
        store.write_blob("legacy", 2, _ipc_bytes(1))
        store.finalize_artifact("legacy", 2, "{}", 1, 10)

        store.garbage_collect(max_idle_days=0, tenant="team-a")

        assert store.get_artifact("legacy", 1) is None, (
            "tenant-scoped GC still cannot reach a tenantless artifact"
        )


class TestAnIdTwoComputationsClaim:
    """``import_artifact`` keeps the given id, and notebook ids derive from notebook and cell ids,
    so two people from one repo can send the same id for different cells.
    """

    @staticmethod
    def _record(provenance: str):
        from strata.artifact_store import ArtifactVersion

        return ArtifactVersion(
            id="nb_shared_cell_c1_var_rows",
            version=1,
            state="ready",
            provenance_hash=provenance,
            schema_json="",
            row_count=0,
            byte_size=5,
            created_at=1.0,
            transform_spec=json.dumps({"executor": "notebook/cell@v1", "params": {}}),
            input_versions="{}",
            principal=None,
            tenant=None,
            content_sha256=None,
        )

    def test_a_repeated_import_of_the_same_computation_is_a_no_op(self, store):
        store.import_artifact(self._record("a" * 64), b"ALICE")

        again = store.import_artifact(self._record("a" * 64), b"ALICE")

        assert again.written is False

    def test_a_different_computation_under_that_id_is_refused(self, store):
        """Keeping the existing row would leave the caller's name, tags and descendants on someone
        else's bytes.
        """
        from strata.artifact_store import ArtifactImportConflict

        store.import_artifact(self._record("a" * 64), b"ALICE")

        with pytest.raises(ArtifactImportConflict):
            store.import_artifact(self._record("b" * 64), b"BOBBY")

        assert store.read_blob("nb_shared_cell_c1_var_rows", 1) == b"ALICE"


class TestAnIdIsHeldByOneTenant:
    """A version under another tenant's id becomes that id's latest, which its notebook reads.

    Refused inside the write, since a route's earlier look can go stale before the insert.
    """

    @pytest.mark.parametrize("holder,caller", [("team-b", "team-a"), ("team-b", None), (None, "a")])
    def test_create_refuses_an_id_another_tenant_holds(self, store, holder, caller):
        from strata.artifact_store import ArtifactIdTaken

        store.create_artifact("nb_x", "a" * 64, tenant=holder)

        with pytest.raises(ArtifactIdTaken):
            store.create_artifact("nb_x", "b" * 64, tenant=caller)

        assert store.get_artifact("nb_x", 2) is None

    @pytest.mark.parametrize("tenant,again", [("team-a", "team-a"), (None, ""), ("", None)])
    def test_create_appends_to_its_own_tenants_id(self, store, tenant, again):
        store.create_artifact("nb_x", "a" * 64, tenant=tenant)

        assert store.create_artifact("nb_x", "b" * 64, tenant=again) == 2

    @pytest.mark.parametrize("version", [1, 2])
    def test_import_refuses_an_id_another_tenant_holds(self, store, version):
        """Version 1 is the same computation, which used to resolve onto the other tenant's row."""
        from strata.artifact_store import ArtifactImportConflict, ArtifactVersion

        held = store.create_artifact("nb_x", "a" * 64, tenant="team-b")
        store.write_blob("nb_x", held, b"TEAMB")
        store.finalize_artifact("nb_x", held, "", row_count=None, byte_size=5)
        record = ArtifactVersion(
            id="nb_x",
            version=version,
            state="ready",
            provenance_hash="a" * 64,
            schema_json="",
            row_count=None,
            byte_size=5,
            created_at=1.0,
            transform_spec=None,
            input_versions=json.dumps({"strata://artifact/up@v=1": "up@v=1"}),
            principal=None,
            tenant="team-a",
            content_sha256=None,
        )

        with pytest.raises(ArtifactImportConflict):
            store.import_artifact(record, b"TEAMA")

        assert store.get_artifact("nb_x", 2) is None
        assert store.read_blob("nb_x", 1) == b"TEAMB"
        assert store.get_artifact("nb_x", 1).input_versions is None


class TestTwoIdsOneComputation:
    """A duplicated notebook yields one provenance under two ids.

    Finalizing the second id's run supersedes the first id's row, which keeps its bytes and stays
    that id's current value for its downstream cells.
    """

    A = "nb_A_cell_c1_var_df"
    B = "nb_B_cell_c1_var_df"

    def _both_finalized(self, store):
        import hashlib

        from strata.artifact_store import StagedVersion

        _make_ready_artifact(store, self.A, "same-prov")
        vb = store.create_artifact(self.B, "same-prov")
        data = _ipc_bytes(1)
        store.write_blob(self.B, vb, data)
        digest = hashlib.sha256(data).hexdigest()
        # What finalize_cell_outputs does with a cell run's outputs.
        store.finalize_canonical_together([StagedVersion(self.B, vb, "{}", 1, len(data), digest)])
        assert store.get_artifact(self.A, 1).state == "superseded"

    def test_promoting_one_id_leaves_the_other_its_value(self, store):
        self._both_finalized(store)

        a = store.get_latest_version(self.A)
        assert a is not None and a.version == 1
        assert store.read_blob(self.A, a.version) is not None
        assert store.get_latest_version(self.B) is not None
        assert [v.id for v in store.list_latest_by_id_prefix("nb_A_")] == [self.A]

    def test_the_promoted_id_owns_its_bytes(self, store):
        """A is now the superseded one and may be collected; B must not read through it."""
        self._both_finalized(store)

        assert store.blob_store.blob_exists(self.B, 1)
        assert store.delete_artifact(self.A, 1)
        assert store.read_blob(self.B, 1) == _ipc_bytes(1)
        assert store.verify_artifacts() == []

    def test_gc_spares_a_superseded_current_value_during_a_rebuild(self, store):
        self._both_finalized(store)
        store.create_artifact(self.A, "prov-a-edited")  # A reruns after an edit

        assert store.garbage_collect(max_idle_days=0)["deleted_count"] == 0
        assert store.get_latest_version(self.A).version == 1


class TestFinalizeTogether:
    """``finalize_canonical_together``: one cell run's outputs become current all at once."""

    @staticmethod
    def _staged(store, artifact_id: str, provenance: str):
        from strata.artifact_store import StagedVersion

        version = store.create_artifact(artifact_id, provenance)
        store.write_blob(artifact_id, version, b"x")
        return StagedVersion(artifact_id, version, "{}", 1, 1, "digest")

    def test_each_becomes_the_canonical_row_under_its_own_id(self, store):
        _make_ready_artifact(store, "nb_A_cell_c_var_x", "prov-x")  # another id, same result
        _make_ready_artifact(store, "nb_B_cell_c_var_y", "prov-y")  # this id's earlier run
        staged = [
            self._staged(store, "nb_B_cell_c_var_x", "prov-x"),
            self._staged(store, "nb_B_cell_c_var_y", "prov-y"),
        ]

        finalized = store.finalize_canonical_together(staged)

        assert [(v.id, v.version, v.state) for v in finalized] == [
            ("nb_B_cell_c_var_x", 1, "ready"),
            ("nb_B_cell_c_var_y", 2, "ready"),
        ]
        assert store.get_artifact("nb_A_cell_c_var_x", 1).state == "superseded"
        assert store.get_artifact("nb_B_cell_c_var_y", 1).state == "superseded"

    def test_one_that_cannot_be_finalized_finalizes_none(self, store):
        first = self._staged(store, "nb_B_cell_c_var_x", "prov-x")
        second = self._staged(store, "nb_B_cell_c_var_y", "prov-y")
        store.fail_artifact(second.artifact_id, second.version)

        with pytest.raises(ValueError, match="building"):
            store.finalize_canonical_together([first, second])

        assert store.get_artifact(first.artifact_id, first.version).state == "building"


def _rebuild_after_first_lookup(store, monkeypatch, artifact_id: str, provenance: str) -> None:
    """Land a rebuild of ``artifact_id``, superseding what the first provenance lookup returns."""
    find = store.find_by_provenance
    landed = []

    def find_then_rebuild(*args, **kwargs):
        found = find(*args, **kwargs)
        if not landed:
            landed.append(True)
            version = store.create_artifact(artifact_id, provenance)
            store.finalize_artifact(artifact_id, version, "{}", 1, 10, content_sha256="e" * 64)
        return found

    monkeypatch.setattr(store, "find_by_provenance", find_then_rebuild)


class TestFindReadyAndSetName:
    def test_it_names_the_ready_version(self, store):
        _make_ready_artifact(store, "a1", "prov-1")

        found = store.find_ready_and_set_name("prov-1", "the-name")

        assert (found.id, found.version) == ("a1", 1)
        name = store.get_name("the-name")
        assert (name.artifact_id, name.version) == ("a1", 1)

    def test_nothing_ready_names_nothing(self, store):
        store.create_artifact("a1", "prov-1")

        assert store.find_ready_and_set_name("prov-1", "the-name") is None
        assert store.get_name("the-name") is None

    def test_a_rebuild_between_the_lookup_and_the_name_gets_the_name(self, store, monkeypatch):
        _make_ready_artifact(store, "a1", "prov-1")
        _rebuild_after_first_lookup(store, monkeypatch, "a1", "prov-1")

        found = store.find_ready_and_set_name("prov-1", "the-name")

        assert (found.id, found.version, found.state) == ("a1", 2, "ready")
        name = store.get_name("the-name")
        assert (name.artifact_id, name.version) == ("a1", 2)
        assert store.get_artifact("a1", 1).state == "superseded"
