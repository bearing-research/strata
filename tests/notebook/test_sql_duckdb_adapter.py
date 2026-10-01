"""Tests for the DuckDB DriverAdapter.

DuckDB is local and free, so we don't need testcontainers — the
real-DB integration tests below exercise the full open → probe →
write → reprobe cycle end-to-end. The mocks at the top cover the
identity / path-resolution surface where exercising the engine
adds nothing.

Note on cursors: DuckDB's ``conn.cursor()`` returns an *independent*
child connection rather than a SQL-DBAPI cursor that shares
transaction state with the parent. That's a real footgun for
read-only enforcement — see the ``_ReadOnlyDuckDB`` proxy in the
adapter and the in-memory RO tests below.
"""

from __future__ import annotations

import os

import pytest

from strata.notebook.models import ConnectionSpec
from strata.notebook.sql import FreshnessToken, QualifiedTable, SchemaFingerprint
from strata.notebook.sql.drivers.duckdb import DuckDBAdapter, _ReadOnlyDuckDB

duckdb = pytest.importorskip("duckdb")

# --- capability flags -------------------------------------------------------


def test_capabilities_match_design_doc():
    a = DuckDBAdapter()
    assert a.name == "duckdb"
    assert a.sqlglot_dialect == "duckdb"
    # PRAGMA database_size is DB-wide, not per-table.
    assert a.capabilities.per_table_freshness is False
    assert a.capabilities.supports_snapshot is False
    # Probes are statement-level, no transaction-frozen views.
    assert a.capabilities.needs_separate_probe_conn is False


# --- canonicalize_connection_id -------------------------------------------


def test_connection_id_canonicalizes_relative_paths(tmp_path, monkeypatch):
    """Two specs that resolve to the same absolute path produce the
    same id, regardless of how the path was written."""
    a = DuckDBAdapter()
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.chdir(tmp_path)

    absolute = ConnectionSpec(
        name="db",
        driver="duckdb",
        path=os.path.abspath(str(nested / "events.duckdb")),
    )
    relative = ConnectionSpec(
        name="db",
        driver="duckdb",
        path="nested/events.duckdb",
    )
    assert a.canonicalize_connection_id(absolute) == a.canonicalize_connection_id(relative)


def test_connection_id_distinguishes_paths():
    a = DuckDBAdapter()
    a_spec = ConnectionSpec(name="db", driver="duckdb", path="/tmp/a.duckdb")
    b_spec = ConnectionSpec(name="db", driver="duckdb", path="/tmp/b.duckdb")
    assert a.canonicalize_connection_id(a_spec) != a.canonicalize_connection_id(b_spec)


def test_connection_id_memory_distinct_from_file():
    a = DuckDBAdapter()
    mem = ConnectionSpec(name="db", driver="duckdb", path=":memory:")
    file = ConnectionSpec(name="db", driver="duckdb", path="/tmp/mem.duckdb")
    assert a.canonicalize_connection_id(mem) != a.canonicalize_connection_id(file)


def test_connection_id_read_only_flag_no_op():
    """DuckDB embedded has no read/write principal split, so the
    ``read_only`` kwarg in the Protocol is a no-op here. Both
    sides must produce the same id."""
    a = DuckDBAdapter()
    spec = ConnectionSpec(name="db", driver="duckdb", path="/tmp/x.duckdb")
    assert a.canonicalize_connection_id(spec, read_only=True) == a.canonicalize_connection_id(
        spec, read_only=False
    )


# --- open() (mocked test seam) --------------------------------------------


def test_open_passes_read_only_flag_for_existing_file(tmp_path):
    """File-backed read_only=True opens with the file flag — the
    primary security boundary for file DBs."""
    target = tmp_path / "existing.duckdb"
    # Create the file first so ``read_only=True`` is permitted.
    duckdb.connect(str(target)).close()
    captured: list[tuple[str, bool]] = []

    def fake_connect(path, *, read_only):
        captured.append((path, read_only))
        return duckdb.connect(":memory:", read_only=False)

    a = DuckDBAdapter(connect_fn=fake_connect)
    spec = ConnectionSpec(name="db", driver="duckdb", path=str(target))
    a.open(spec, read_only=True)

    assert captured, "fake connect_fn was not invoked"
    assert captured[0][0] == str(target)
    assert captured[0][1] is True


def test_open_falls_back_to_writable_for_missing_file(tmp_path):
    """``duckdb.connect(path, read_only=True)`` errors if the file
    doesn't exist; falling back to a writable handle keeps a
    "first run" notebook executable. The RO transaction proxy is
    what enforces read-only in that case."""
    target = tmp_path / "not_yet.duckdb"
    captured: list[tuple[str, bool]] = []

    def fake_connect(path, *, read_only):
        captured.append((path, read_only))
        return duckdb.connect(":memory:", read_only=False)

    a = DuckDBAdapter(connect_fn=fake_connect)
    spec = ConnectionSpec(name="db", driver="duckdb", path=str(target))
    a.open(spec, read_only=True)

    assert captured[0][1] is False, "expected fallback to writable handle"


def test_open_memory_db_uses_writable_handle():
    """``:memory:`` cannot be opened ``read_only=True`` (the database
    is created on demand). The proxy + RO transaction is the only
    enforcement mechanism for memory DBs."""
    captured: list[tuple[str, bool]] = []

    def fake_connect(path, *, read_only):
        captured.append((path, read_only))
        return duckdb.connect(":memory:", read_only=False)

    a = DuckDBAdapter(connect_fn=fake_connect)
    spec = ConnectionSpec(name="db", driver="duckdb", path=":memory:")
    a.open(spec, read_only=True)

    assert captured[0][0] == ":memory:"
    assert captured[0][1] is False


def test_open_returns_proxy_when_read_only(tmp_path):
    """``read_only=True`` always returns a ``_ReadOnlyDuckDB``
    proxy, so cursor spawning runs through our RO-transaction
    interception. ``read_only=False`` returns the bare connection."""
    a = DuckDBAdapter()
    db = tmp_path / "p.duckdb"
    spec = ConnectionSpec(name="db", driver="duckdb", path=str(db))

    rw = a.open(spec, read_only=False)
    assert not isinstance(rw, _ReadOnlyDuckDB)
    rw.close()

    ro = a.open(spec, read_only=True)
    assert isinstance(ro, _ReadOnlyDuckDB)
    ro.close()


def test_open_raises_when_path_missing():
    a = DuckDBAdapter()
    spec = ConnectionSpec(name="db", driver="duckdb")
    with pytest.raises(RuntimeError, match="path"):
        a.open(spec, read_only=True)


# --- real-DB: read-only enforcement ----------------------------------------


def test_real_open_read_only_rejects_write_on_file_db(tmp_path):
    """File-backed ``read_only=True`` opens with ``read_only=True``
    in the engine. Writes are blocked at the file level."""
    db = tmp_path / "ro.duckdb"
    a = DuckDBAdapter()
    rw_spec = ConnectionSpec(name="db", driver="duckdb", path=str(db))

    rw = a.open(rw_spec, read_only=False)
    rw.execute("CREATE TABLE t(x INT)")
    rw.execute("INSERT INTO t VALUES (1), (2), (3)")
    rw.close()

    ro = a.open(rw_spec, read_only=True)
    cur = ro.cursor()
    cur.execute("SELECT count(*) FROM t")
    assert cur.fetchone() == (3,)
    with pytest.raises(Exception):
        cur.execute("INSERT INTO t VALUES (99)")
    ro.close()


def test_real_open_read_only_rejects_write_on_memory_db():
    """Memory DBs can't open with the file ``read_only=True`` flag
    (no file). The ``_ReadOnlyDuckDB`` proxy + ``BEGIN TRANSACTION
    READ ONLY`` per-cursor is the only barrier — and it must hold."""
    a = DuckDBAdapter()
    spec = ConnectionSpec(name="db", driver="duckdb", path=":memory:")
    ro = a.open(spec, read_only=True)
    cur = ro.cursor()
    cur.execute("SELECT 1+1")
    assert cur.fetchone() == (2,)
    # Writes are blocked by the per-cursor RO transaction.
    with pytest.raises(Exception):
        cur.execute("CREATE TABLE m(x INT)")
    ro.close()


def test_real_write_path_still_allows_dml(tmp_path):
    """Sanity: ``read_only=False`` lets writes through and the
    proxy is bypassed entirely."""
    db = tmp_path / "w.duckdb"
    a = DuckDBAdapter()
    spec = ConnectionSpec(name="db", driver="duckdb", path=str(db))
    rw = a.open(spec, read_only=False)
    assert not isinstance(rw, _ReadOnlyDuckDB)
    rw.execute("CREATE TABLE t(x INT)")
    rw.execute("INSERT INTO t VALUES (1), (2)")
    cur = rw.cursor()
    cur.execute("SELECT count(*) FROM t")
    assert cur.fetchone() == (2,)
    rw.close()


# --- real-DB: probes -------------------------------------------------------


def test_real_probe_freshness_changes_after_dml(tmp_path):
    """``PRAGMA database_size`` advances when blocks flip to dirty
    and a checkpoint persists them. Two distinct on-disk states
    must produce two distinct tokens."""
    db = tmp_path / "f.duckdb"
    a = DuckDBAdapter()
    spec = ConnectionSpec(name="db", driver="duckdb", path=str(db))

    # Seed and checkpoint.
    rw = a.open(spec, read_only=False)
    rw.execute("CREATE TABLE t(x INT)")
    rw.execute("INSERT INTO t SELECT range FROM range(0, 100)")
    rw.execute("CHECKPOINT")
    rw.close()

    ro = a.open(spec, read_only=True)
    t1 = a.probe_freshness(ro, [QualifiedTable(catalog=None, schema=None, name="t")])
    ro.close()

    rw = a.open(spec, read_only=False)
    rw.execute("INSERT INTO t SELECT range FROM range(0, 50000)")
    rw.execute("CHECKPOINT")
    rw.close()

    ro = a.open(spec, read_only=True)
    t2 = a.probe_freshness(ro, [QualifiedTable(catalog=None, schema=None, name="t")])
    ro.close()

    assert isinstance(t1, FreshnessToken)
    assert isinstance(t2, FreshnessToken)
    assert not t1.is_session_only
    assert not t2.is_session_only
    assert t1.value != t2.value


def test_real_probe_freshness_ignores_table_list_db_wide(tmp_path):
    """Capability flag says ``per_table_freshness=False`` — two
    different table sets against the same DB state must produce
    the same token."""
    db = tmp_path / "g.duckdb"
    a = DuckDBAdapter()
    spec = ConnectionSpec(name="db", driver="duckdb", path=str(db))

    rw = a.open(spec, read_only=False)
    rw.execute("CREATE TABLE a(x INT); CREATE TABLE b(y INT)")
    rw.execute("INSERT INTO a VALUES (1)")
    rw.execute("INSERT INTO b VALUES (2)")
    rw.execute("CHECKPOINT")
    rw.close()

    ro = a.open(spec, read_only=True)
    t_a = a.probe_freshness(ro, [QualifiedTable(catalog=None, schema=None, name="a")])
    t_ab = a.probe_freshness(
        ro,
        [
            QualifiedTable(catalog=None, schema=None, name="a"),
            QualifiedTable(catalog=None, schema=None, name="b"),
        ],
    )
    ro.close()
    assert t_a.value == t_ab.value


def test_real_probe_schema_changes_on_add_column(tmp_path):
    """Per-table fingerprint catches metadata-only ADD COLUMN even
    if the freshness probe (block-aligned) hasn't moved."""
    db = tmp_path / "s.duckdb"
    a = DuckDBAdapter()
    spec = ConnectionSpec(name="db", driver="duckdb", path=str(db))

    rw = a.open(spec, read_only=False)
    rw.execute("CREATE TABLE t(x INT)")
    rw.close()
    ro = a.open(spec, read_only=True)
    fp1 = a.probe_schema(ro, [QualifiedTable(catalog=None, schema="main", name="t")])
    ro.close()

    rw = a.open(spec, read_only=False)
    rw.execute("ALTER TABLE t ADD COLUMN y VARCHAR")
    rw.close()
    ro = a.open(spec, read_only=True)
    fp2 = a.probe_schema(ro, [QualifiedTable(catalog=None, schema="main", name="t")])
    ro.close()

    assert isinstance(fp1, SchemaFingerprint)
    assert fp1.value != fp2.value


def test_real_probe_schema_empty_tables_yields_empty_token(tmp_path):
    db = tmp_path / "e.duckdb"
    a = DuckDBAdapter()
    spec = ConnectionSpec(name="db", driver="duckdb", path=str(db))
    rw = a.open(spec, read_only=False)
    rw.close()
    ro = a.open(spec, read_only=True)
    fp = a.probe_schema(ro, [])
    ro.close()
    assert fp.value == b""


def test_probe_schema_rejects_invalid_schema_name(tmp_path):
    """Identifier validation defends against splice in the
    ``duckdb_columns()`` predicates. Even though the adapter uses
    bind parameters for the values, the upstream ``QualifiedTable``
    is user-influenced — fail fast on garbage input."""
    db = tmp_path / "q.duckdb"
    a = DuckDBAdapter()
    spec = ConnectionSpec(name="db", driver="duckdb", path=str(db))
    rw = a.open(spec, read_only=False)
    rw.execute("CREATE TABLE t(x INT)")
    rw.close()

    ro = a.open(spec, read_only=True)
    with pytest.raises(RuntimeError, match="schema name"):
        a.probe_schema(
            ro,
            [QualifiedTable(catalog=None, schema='main"; DROP TABLE t; --', name="t")],
        )
    ro.close()


# --- real-DB: list_schema --------------------------------------------------


def test_real_list_schema_returns_tables_and_columns(tmp_path):
    """``duckdb_tables()`` + ``duckdb_columns()`` enumerate the
    user-visible surface, with internal databases (``system``,
    ``temp``) and internal schemas filtered out."""
    db = tmp_path / "ls.duckdb"
    a = DuckDBAdapter()
    spec = ConnectionSpec(name="db", driver="duckdb", path=str(db))

    rw = a.open(spec, read_only=False)
    rw.execute("CREATE SCHEMA analytics")
    rw.execute("CREATE TABLE main.users(id INT NOT NULL, name VARCHAR)")
    rw.execute("CREATE TABLE analytics.events(id INT, payload JSON)")
    rw.execute("CREATE VIEW analytics.recent_events AS SELECT * FROM analytics.events")
    rw.close()

    ro = a.open(spec, read_only=True)
    schemas = a.list_schema(ro)
    ro.close()

    by_full_name = {(s.schema, s.name): s for s in schemas}
    assert ("main", "users") in by_full_name
    assert ("analytics", "events") in by_full_name
    assert ("analytics", "recent_events") in by_full_name

    users = by_full_name[("main", "users")]
    cols = {c.name: c for c in users.columns}
    assert set(cols) == {"id", "name"}
    assert cols["id"].nullable is False
    assert cols["name"].nullable is True


def test_real_list_schema_empty_database(tmp_path):
    db = tmp_path / "empty.duckdb"
    a = DuckDBAdapter()
    spec = ConnectionSpec(name="db", driver="duckdb", path=str(db))
    rw = a.open(spec, read_only=False)
    rw.close()
    ro = a.open(spec, read_only=True)
    schemas = a.list_schema(ro)
    ro.close()
    assert schemas == []


# --- registry integration --------------------------------------------------


def test_duckdb_adapter_is_auto_registered():
    from strata.notebook.sql.registry import (
        _restore_defaults_for_tests,
        get_adapter,
    )

    _restore_defaults_for_tests()
    a = get_adapter("duckdb")
    assert a.name == "duckdb"
    assert a.sqlglot_dialect == "duckdb"
