"""Retention as data (ADR 002 §7.4): roots say why a version is kept, edges what it needs.

The store reads no blob to learn either. A writer declares what its version requires with
the row; a holder retains a version under a kind and a holder of its own.
"""

from __future__ import annotations

import sqlite3
import time
import uuid

import pytest

from vestigo.provenance import TransformSpec
from vestigo.store import RETENTION_SCHEMA_VERSION, ArtifactStore, ArtifactVersion, _add_retention


@pytest.fixture
def store(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


def _ready(
    store: ArtifactStore,
    artifact_id: str | None = None,
    *,
    tenant: str | None = None,
    requires=(),
    content_type: str = "application/octet-stream",
) -> tuple[str, int]:
    """A ready, idle, minted version: what a sweep takes unless something holds it."""
    artifact_id = artifact_id or str(uuid.uuid4())
    version = store.create_artifact(
        artifact_id,
        f"prov-{uuid.uuid4()}",
        TransformSpec(executor="x@v1", params={"content_type": content_type}, inputs=[]),
        tenant=tenant,
        minted=True,
        requires=requires,
    )
    store.write_blob(artifact_id, version, b"x" * 100)
    store.finalize_artifact(artifact_id, version, "", None, 100)
    _idle(store, artifact_id, version)
    return artifact_id, version


def _idle(store: ArtifactStore, artifact_id: str, version: int) -> None:
    conn = store._get_connection()
    try:
        conn.execute(
            "UPDATE artifact_versions SET created_at = 0, last_used_at = ? "
            "WHERE id = ? AND version = ?",
            (time.time() - 40 * 86400, artifact_id, version),
        )
        conn.commit()
    finally:
        conn.close()


def _sweep(store: ArtifactStore) -> int:
    return store.garbage_collect(max_idle_days=30, collect_latest=True)["deleted_count"]


def _edges(store: ArtifactStore) -> set[tuple[str, int, str, int]]:
    conn = store._get_connection()
    try:
        return {
            (r["from_artifact_id"], r["from_version"], r["to_artifact_id"], r["to_version"])
            for r in conn.execute("SELECT * FROM retention_edges").fetchall()
        }
    finally:
        conn.close()


class TestRoots:
    def test_a_root_is_returned_and_listed(self, store):
        key = _ready(store)

        root = store.retain(*key, kind="pin", holder="review:7", created_by="amber")

        assert (root["kind"], root["holder"], root["created_by"]) == ("pin", "review:7", "amber")
        assert root["created_at"] == pytest.approx(time.time(), abs=5)
        assert store.retention_roots(*key) == [root]
        assert store.retention_roots(kind="pin") == [root]
        assert store.retention_roots(kind="head") == []

    def test_retaining_again_refreshes_who_and_when(self, store):
        key = _ready(store)
        first = store.retain(*key, kind="pin", holder="review", created_by="a")

        again = store.retain(*key, kind="pin", holder="review", created_by="b")

        assert again["created_by"] == "b"
        assert again["created_at"] >= first["created_at"]
        assert len(store.retention_roots(*key)) == 1

    def test_a_root_holds_its_version_and_its_chain(self, store):
        upstream = _ready(store)
        ref = f"{upstream[0]}@v={upstream[1]}"
        held = store.create_artifact(
            "held", "prov-h", input_versions={f"vestigo://artifact/{ref}": ref}, minted=True
        )
        store.write_blob("held", held, b"h")
        store.finalize_artifact("held", held, "", None, 1)
        _idle(store, "held", held)
        store.retain("held", held, kind="pin", holder="review")

        assert _sweep(store) == 0
        assert store.release("held", held, kind="pin", holder="review")
        assert _sweep(store) == 2

    def test_two_holders_release_independently(self, store):
        key = _ready(store)
        store.retain(*key, kind="pin", holder="review")
        store.retain(*key, kind="head", holder="latest")

        assert store.release(*key, kind="pin", holder="review")
        assert not store.release(*key, kind="pin", holder="review")
        assert [r["kind"] for r in store.retention_roots(*key)] == ["head"]
        assert _sweep(store) == 0

    def test_a_root_needs_a_version_a_kind_and_a_holder(self, store):
        key = _ready(store)
        with pytest.raises(ValueError, match="not found"):
            store.retain(key[0], 99, kind="pin", holder="x")
        with pytest.raises(ValueError, match="holder"):
            store.retain(*key, kind="pin", holder=" ")
        with pytest.raises(ValueError, match="holder"):
            store.retain(*key, kind="", holder="x")

    def test_another_tenants_version_cannot_be_retained(self, store):
        theirs = _ready(store, tenant="b")

        with pytest.raises(ValueError, match="not found"):
            store.retain(*theirs, kind="pin", holder="x", tenant="a")
        assert store.retain(*theirs, kind="pin", holder="x", tenant="b")["tenant"] == "b"
        assert not store.release(*theirs, kind="pin", holder="x", tenant="a")
        assert store.retention_roots(tenant="a") == []

    def test_a_held_version_is_not_deleted_until_released(self, store):
        """A hold is a promise to restore the version: an explicit delete is refused,
        as it is for a publication, until the holder lets go."""
        key = _ready(store)
        store.retain(*key, kind="pin", holder="review")

        with pytest.raises(ValueError, match="held \\(pin review\\)"):
            store.delete_artifact(*key)

        assert store.get_artifact(*key) is not None
        assert store.retention_roots(*key) != []
        store.release(*key, kind="pin", holder="review")
        assert store.delete_artifact(*key)
        assert store.retention_roots(*key) == []

    def test_a_root_held_without_a_tenant_is_the_versions_tenants(self, store):
        """So the tenant that owns the version can release and list it."""
        key = _ready(store, tenant="a")

        root = store.retain(*key, kind="pin", holder="r")

        assert root["tenant"] == "a"
        assert store.retention_roots(*key, tenant="a") == [root]
        assert store.release(*key, kind="pin", holder="r", tenant="a")
        assert store.retention_roots(*key) == []


class TestHoldLandingMidSweep:
    """A hold landing on a reader after the selection keeps the reader's inputs: the
    sweep reads reachability again under the write lock before each batch."""

    def _reader_of(self, store, upstream):
        ref = f"{upstream[0]}@v={upstream[1]}"
        version = store.create_artifact(
            "reader", "prov-r", input_versions={f"vestigo://artifact/{ref}": ref}, minted=True
        )
        store.write_blob("reader", version, b"r")
        store.finalize_artifact("reader", version, "", None, 1)
        _idle(store, "reader", version)
        return "reader", version

    def _sweep_with(self, store, monkeypatch, action):
        """A full sweep, with *action* run after the candidates are chosen."""
        chosen_then = ArtifactStore._without_version_gaps

        def land(conn, chosen):
            action()
            return chosen_then(conn, chosen)

        monkeypatch.setattr(ArtifactStore, "_without_version_gaps", staticmethod(land))
        return _sweep(store)

    @pytest.mark.parametrize("hold", ["name", "pin"])
    def test_a_hold_on_a_reader_keeps_its_input(self, store, monkeypatch, hold):
        upstream = _ready(store)
        reader = self._reader_of(store, upstream)
        land = {
            "name": lambda: store.set_name("published-model", *reader),
            "pin": lambda: store.retain(*reader, kind="pin", holder="review"),
        }[hold]

        assert self._sweep_with(store, monkeypatch, land) == 0

        assert store.get_artifact(*reader) is not None
        assert store.get_artifact(*upstream) is not None

    def test_a_requirer_created_mid_sweep_keeps_its_file(self, store, monkeypatch):
        """A manifest reusing a file the sweep chose: the claim sees the new edge."""
        file = _ready(store)
        created = {}

        def reuse():
            created["manifest"] = store.create_artifact("manifest", "prov-m", requires=[file])

        assert self._sweep_with(store, monkeypatch, reuse) == 0

        assert store.get_artifact(*file) is not None
        store.write_blob("manifest", created["manifest"], b"m")
        store.finalize_artifact("manifest", created["manifest"], "", None, 1)
        assert store.requirements_met(store.get_artifact("manifest", created["manifest"]))

    def test_an_id_another_sweep_emptied_is_dropped_from_the_choice(self, store):
        key = _ready(store)
        row = {"id": key[0], "version": key[1]}
        store.delete_artifact(*key)
        conn = store._get_connection()
        try:
            assert ArtifactStore._without_version_gaps(conn, {key: row}) == {}
        finally:
            conn.close()


class TestRequirements:
    def test_what_a_version_requires_is_recorded_with_its_row(self, store):
        parts = [_ready(store) for _ in range(2)]

        whole = _ready(store, "whole", requires=[*parts, *parts, ("whole", 1)])

        assert whole == ("whole", 1)
        assert _edges(store) == {(*whole, *part) for part in parts}

    def test_a_held_version_keeps_what_it_requires(self, store):
        parts = [_ready(store) for _ in range(2)]
        whole = _ready(store, requires=parts)
        store.retain(*whole, kind="pin", holder="review")

        assert _sweep(store) == 0
        assert all(store.get_artifact(*part) is not None for part in parts)

    def test_an_idle_version_goes_with_what_it_requires(self, store):
        parts = [_ready(store) for _ in range(2)]
        whole = _ready(store, requires=parts)

        assert _sweep(store) == 3
        assert store.get_artifact(*whole) is None
        assert _edges(store) == set()

    def test_a_requirement_stays_while_its_requirer_is_used(self, store):
        parts = [_ready(store) for _ in range(2)]
        whole = _ready(store, requires=parts)
        store.record_use(*whole)

        assert _sweep(store) == 0

    def test_a_requirement_of_another_tenant_is_not_kept(self, store):
        theirs = _ready(store, tenant="b")
        mine = _ready(store, tenant="a")
        whole = _ready(store, tenant="a", requires=[theirs, mine])
        store.record_use(*whole)

        assert _sweep(store) == 1
        assert store.get_artifact(*theirs) is None
        assert store.get_artifact(*mine) is not None
        assert not store.requirements_met(store.get_artifact(*whole))

    def test_requirements_met_reads_the_stored_versions(self, store):
        part = _ready(store)
        whole = _ready(store, requires=[part])
        plain = _ready(store)

        assert store.requirements_met(store.get_artifact(*whole))
        assert store.requirements_met(store.get_artifact(*plain))
        store.delete_artifact(*part)
        assert not store.requirements_met(store.get_artifact(*whole))

    def test_requirements_can_be_declared_after_the_fact(self, store):
        part = _ready(store)
        whole = _ready(store)

        store.require(*whole, [part])
        store.require(*whole, [part])

        assert _edges(store) == {(*whole, *part)}
        with pytest.raises(ValueError, match="not found"):
            store.require("nope", 1, [part])

    def test_many_declare_in_one_transaction_or_not_at_all(self, store):
        part = _ready(store)
        first = _ready(store)
        second = _ready(store)

        with pytest.raises(ValueError, match="nope"):
            store.require_many({first: [part], ("nope", 1): [part]})
        assert _edges(store) == set()

        store.require_many({first: [part], second: [part, first]})
        assert _edges(store) == {(*first, *part), (*second, *part), (*second, *first)}

    def test_a_consumer_upgrade_step_is_recorded_once(self, store, tmp_path):
        """A step the store cannot do itself (it reads no blob) is marked in the store."""
        assert not store.upgrade_applied("folders")

        store.mark_upgrade_applied("folders")
        store.mark_upgrade_applied("folders")

        assert store.upgrade_applied("folders")
        assert ArtifactStore(tmp_path / "artifacts").upgrade_applied("folders")
        assert not store.upgrade_applied("checkpoints")

    def test_an_import_records_what_the_record_requires(self, store, tmp_path):
        part = _ready(store)
        record = ArtifactVersion(
            id="copied",
            version=3,
            state="ready",
            provenance_hash="prov-copied",
            schema_json="",
            row_count=None,
            byte_size=1,
            created_at=1.0,
            transform_spec=None,
            input_versions=None,
            tenant="",
            principal=None,
            content_sha256=None,
        )

        landed = store.import_artifact(record, b"m", requires=[part])

        assert (landed.id, landed.version, landed.written) == ("copied", 3, True)
        assert _edges(store) == {("copied", 3, *part)}

    def test_versions_declaring_a_content_type(self, store):
        manifest = _ready(store, content_type="dir/manifest")
        _ready(store, content_type="dir/manifesto")
        _ready(store)

        store.create_artifact(
            "still-building",
            "prov-building",
            TransformSpec(executor="x@v1", params={"content_type": "dir/manifest"}, inputs=[]),
        )

        declared = store.versions_declaring("dir/manifest")

        assert [(v.id, v.version) for v in declared] == [manifest]


class TestTheMigration:
    """A store from before this schema holds pins in a table of their own and no edges."""

    def _before_retention(self, store: ArtifactStore, pins) -> None:
        conn = sqlite3.connect(store.db_path)
        try:
            conn.executescript(
                "DROP TABLE retention_roots; DROP TABLE retention_edges;"
                "CREATE TABLE artifact_pins (artifact_id TEXT NOT NULL, version INTEGER NOT NULL,"
                " reason TEXT NOT NULL, tenant TEXT NOT NULL DEFAULT '', pinned_by TEXT,"
                " pinned_at REAL NOT NULL, PRIMARY KEY (artifact_id, version, reason));"
            )
            conn.executemany(
                "INSERT INTO artifact_pins VALUES (?, ?, ?, ?, ?, ?)",
                [(a, v, reason, tenant, by, 123.0) for a, v, reason, tenant, by in pins],
            )
            conn.execute(
                "DELETE FROM schema_version WHERE version >= ?", (RETENTION_SCHEMA_VERSION,)
            )
            conn.execute(
                "INSERT INTO schema_version (version, applied_at) VALUES (?, 0)",
                (RETENTION_SCHEMA_VERSION - 1,),
            )
            conn.commit()
        finally:
            conn.close()

    def test_pins_become_roots_and_their_table_goes(self, store, tmp_path):
        key = _ready(store, tenant="t")
        self._before_retention(store, [(*key, "snapshot", "t", "ops"), (*key, "review", "t", None)])

        reopened = ArtifactStore(tmp_path / "artifacts")

        assert {
            (r["holder"], r["tenant"], r["created_by"], r["created_at"])
            for r in reopened.retention_roots(*key)
        } == {
            ("review", "t", None, 123.0),
            ("snapshot", "t", "ops", 123.0),
        }
        conn = sqlite3.connect(store.db_path)
        try:
            names = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
        finally:
            conn.close()
        assert "artifact_pins" not in names

    def test_the_migration_is_idempotent(self, store):
        """Two processes opening an old store together both come out at the same place."""
        key = _ready(store, tenant="t")
        self._before_retention(store, [(*key, "snapshot", "t", "ops")])

        conn = store._get_connection()
        try:
            _add_retention(conn, store._dialect)
            _add_retention(conn, store._dialect)
            conn.commit()
        finally:
            conn.close()

        assert [r["holder"] for r in store.retention_roots(*key)] == ["snapshot"]
