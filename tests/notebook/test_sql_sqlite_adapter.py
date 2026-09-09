"""Tests for the SQLite DriverAdapter.

Includes a real-DB integration block at the bottom — SQLite is local
and free, so we don't need testcontainers to exercise the full open →
probe → write → reprobe cycle.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

import pytest

from strata.notebook.models import ConnectionSpec
from strata.notebook.sql import FreshnessToken, QualifiedTable, SchemaFingerprint
from strata.notebook.sql.drivers.sqlite import SqliteAdapter

# --- mock-conn helper -------------------------------------------------------


class _FakeCursor:
    """Scripted cursor: maps query-substring → result.

    Each ``execute`` records its SQL + params; ``fetchone`` and
    ``fetchall`` return the most recently matched script entry.
    """

    def __init__(self, scripts: list[tuple[str, object]]):
        self._scripts = scripts
        self._last_match: object = None
        self.executions: list[tuple[str, tuple]] = []

    def execute(self, sql, params=()):
        self.executions.append((sql, tuple(params)))
        for needle, value in self._scripts:
            if needle in sql:
                self._last_match = value
                return
        self._last_match = None

    def fetchone(self):
        last = self._last_match
        if isinstance(last, list):
            return last.pop(0) if last else None
        return last

    def fetchall(self):
        last = self._last_match
        if isinstance(last, list):
            return last
        return [last] if last is not None else []


class _FakeConn:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor

    @contextmanager
    def cursor(self):
        yield self._cursor


# --- capability flags -------------------------------------------------------


@contextmanager
def cursor_expected_to_fail_on_close(conn):
    """A cursor for a statement ADBC will reject when the statement closes.

    That is how a write to a read-only connection is reported, and
    ``Cursor.close()`` sets ``_closed = True`` only *after* ``_stmt.close()``
    returns — so a close that raises leaves the cursor marked open and its
    ``__del__`` closes it a second time, underflowing the driver's child count
    and surfacing as an unraisable exception attached to whatever ran next.

    Production code corrects the same flag in ``cell_executor._safely_close``.
    A test that drives a cursor itself has to do it itself.
    """
    cursor = conn.cursor()
    try:
        yield cursor
    finally:
        try:
            cursor.close()
        finally:
            if getattr(cursor, "_closed", None) is False:
                cursor._closed = True


def test_capabilities_match_design_doc():
    a = SqliteAdapter()
    assert a.name == "sqlite"
    assert a.sqlglot_dialect == "sqlite"
    # data_version / schema_version are DB-wide, not per-table.
    assert a.capabilities.per_table_freshness is False
    assert a.capabilities.supports_snapshot is False
    # Pragmas don't have transaction-frozen semantics — same
    # connection can probe and query.
    assert a.capabilities.needs_separate_probe_conn is False


# --- canonicalize_connection_id -------------------------------------------


def test_connection_id_canonicalizes_relative_paths(tmp_path, monkeypatch):
    """Two specs that resolve to the same absolute path produce the
    same id, regardless of how the path was written. macOS's
    /var → /private/var symlink dance is the reason we use the
    fixture-supplied tmp_path and resolve both sides through abspath
    on the same root."""
    a = SqliteAdapter()
    nested = tmp_path / "nested"
    nested.mkdir()
    monkeypatch.chdir(tmp_path)

    absolute = ConnectionSpec(
        name="db",
        driver="sqlite",
        path=os.path.abspath(str(nested / "events.sqlite")),
    )
    relative = ConnectionSpec(
        name="db",
        driver="sqlite",
        path="nested/events.sqlite",
    )
    assert a.canonicalize_connection_id(absolute) == a.canonicalize_connection_id(relative)


def test_connection_id_distinguishes_paths():
    a = SqliteAdapter()
    a_spec = ConnectionSpec(name="db", driver="sqlite", path="/tmp/a.sqlite")
    b_spec = ConnectionSpec(name="db", driver="sqlite", path="/tmp/b.sqlite")
    assert a.canonicalize_connection_id(a_spec) != a.canonicalize_connection_id(b_spec)


def test_connection_id_strips_read_only_query_params():
    """``mode=ro`` and ``immutable=1`` change *how* we open, not which
    objects we see — must not perturb the cache key."""
    a = SqliteAdapter()
    base = ConnectionSpec(name="db", driver="sqlite", uri="file:/tmp/db.sqlite")
    with_ro = ConnectionSpec(name="db", driver="sqlite", uri="file:/tmp/db.sqlite?mode=ro")
    with_immutable = ConnectionSpec(
        name="db", driver="sqlite", uri="file:/tmp/db.sqlite?immutable=1"
    )
    cid = a.canonicalize_connection_id(base)
    assert a.canonicalize_connection_id(with_ro) == cid
    assert a.canonicalize_connection_id(with_immutable) == cid


def test_connection_id_memory_distinct_from_file():
    a = SqliteAdapter()
    mem = ConnectionSpec(name="db", driver="sqlite", path=":memory:")
    file = ConnectionSpec(name="db", driver="sqlite", path="/tmp/mem.sqlite")
    assert a.canonicalize_connection_id(mem) != a.canonicalize_connection_id(file)


def test_connection_id_preserves_identity_shaping_query_params():
    """Codex review fix: the canonicalizer must NOT drop all query
    params — ``cache=shared``, ``mode=memory``, ``vfs=`` etc. all
    affect which database is opened, and collapsing them onto the
    same id would alias distinct connections.

    Before the fix, ``file:memdb1?mode=memory&cache=shared`` and
    ``file:memdb1`` produced the same id; they describe completely
    different databases (a shared in-memory DB vs a literal file
    named ``memdb1`` in cwd)."""
    a = SqliteAdapter()
    shared_memory = ConnectionSpec(
        name="db",
        driver="sqlite",
        uri="file:memdb1?mode=memory&cache=shared",
    )
    file_named_memdb1 = ConnectionSpec(name="db", driver="sqlite", uri="file:memdb1")
    private_memory = ConnectionSpec(
        name="db",
        driver="sqlite",
        uri="file:memdb1?mode=memory",
    )
    cid_shared = a.canonicalize_connection_id(shared_memory)
    cid_file = a.canonicalize_connection_id(file_named_memdb1)
    cid_private = a.canonicalize_connection_id(private_memory)
    assert cid_shared != cid_file
    assert cid_shared != cid_private
    assert cid_file != cid_private


def test_connection_id_distinguishes_named_memory_dbs():
    """Two named in-memory DBs (different names) must produce
    different connection_ids — they're distinct logical databases."""
    a = SqliteAdapter()
    db1 = ConnectionSpec(
        name="db",
        driver="sqlite",
        uri="file:db1?mode=memory&cache=shared",
    )
    db2 = ConnectionSpec(
        name="db",
        driver="sqlite",
        uri="file:db2?mode=memory&cache=shared",
    )
    assert a.canonicalize_connection_id(db1) != a.canonicalize_connection_id(db2)


def test_connection_id_preserves_vfs_param():
    """``vfs=`` selects an alternate VFS implementation — affects
    *which* file is opened in some configurations. Identity-shaping."""
    a = SqliteAdapter()
    base = ConnectionSpec(name="db", driver="sqlite", uri="file:/tmp/db.sqlite")
    custom = ConnectionSpec(name="db", driver="sqlite", uri="file:/tmp/db.sqlite?vfs=unix-dotfile")
    assert a.canonicalize_connection_id(base) != a.canonicalize_connection_id(custom)


# --- open() URI building --------------------------------------------------


def test_open_with_read_only_appends_mode_ro_to_path():
    captured: dict[str, str] = {}

    def fake_connect(uri):
        captured["uri"] = uri
        return _FakeConn(_FakeCursor(scripts=[]))

    a = SqliteAdapter(connect_fn=fake_connect)
    spec = ConnectionSpec(name="db", driver="sqlite", path="/tmp/events.sqlite")
    a.open(spec, read_only=True)
    assert captured["uri"] == "file:/tmp/events.sqlite?mode=ro"


def test_open_without_read_only_uses_bare_path():
    captured: dict[str, str] = {}

    def fake_connect(uri):
        captured["uri"] = uri
        return _FakeConn(_FakeCursor(scripts=[]))

    a = SqliteAdapter(connect_fn=fake_connect)
    spec = ConnectionSpec(name="db", driver="sqlite", path="/tmp/events.sqlite")
    a.open(spec, read_only=False)
    assert captured["uri"] == "/tmp/events.sqlite"


def test_open_memory_db_passes_through_unchanged_uri():
    """``:memory:`` databases can't take ``mode=ro`` in the URI; the
    pragma in ``open()`` is what enforces read-only for them.
    The URI itself stays as the literal."""
    captured_uri: dict[str, str] = {}
    cursor = _FakeCursor(scripts=[])
    conn = _FakeConn(cursor)

    def fake_connect(uri):
        captured_uri["uri"] = uri
        return conn

    a = SqliteAdapter(connect_fn=fake_connect)
    spec = ConnectionSpec(name="db", driver="sqlite", path=":memory:")
    a.open(spec, read_only=True)
    assert captured_uri["uri"] == ":memory:"


def test_open_read_only_always_issues_query_only_pragma():
    """The session-level guard is the universal read-only enforcement —
    runs after every ``open(read_only=True)`` regardless of URI form,
    so that file-handle ``mode=ro`` failure (or its absence on memory
    DBs) is backstopped at the engine."""
    cases = [
        ConnectionSpec(name="db", driver="sqlite", path="/tmp/a.sqlite"),
        ConnectionSpec(name="db", driver="sqlite", path=":memory:"),
        ConnectionSpec(name="db", driver="sqlite", uri="file:/tmp/b.sqlite"),
        ConnectionSpec(name="db", driver="sqlite", uri="file:/tmp/c.sqlite?mode=rwc"),
        ConnectionSpec(
            name="db",
            driver="sqlite",
            uri="file:memdb1?mode=memory&cache=shared",
        ),
    ]
    for spec in cases:
        cursor = _FakeCursor(scripts=[])
        conn = _FakeConn(cursor)
        a = SqliteAdapter(connect_fn=lambda uri: conn)
        a.open(spec, read_only=True)
        assert any("PRAGMA query_only" in sql for sql, _ in cursor.executions), (
            f"{spec!r} did not issue PRAGMA query_only"
        )


def test_open_does_not_issue_query_only_when_not_read_only():
    cursor = _FakeCursor(scripts=[])
    conn = _FakeConn(cursor)
    a = SqliteAdapter(connect_fn=lambda uri: conn)
    spec = ConnectionSpec(name="db", driver="sqlite", path="/tmp/x.sqlite")
    a.open(spec, read_only=False)
    assert not any("PRAGMA query_only" in sql for sql, _ in cursor.executions)


def test_open_existing_uri_appends_mode_ro_when_missing():
    captured: dict[str, str] = {}

    def fake_connect(uri):
        captured["uri"] = uri
        return _FakeConn(_FakeCursor(scripts=[]))

    a = SqliteAdapter(connect_fn=fake_connect)
    spec = ConnectionSpec(name="db", driver="sqlite", uri="file:/tmp/events.sqlite")
    a.open(spec, read_only=True)
    assert "mode=ro" in captured["uri"]


def test_open_existing_uri_overrides_user_supplied_mode_rwc():
    """Codex review fix: ``mode=rwc`` (or any user-supplied
    access-mode) MUST be overridden to ``mode=ro`` when read_only=True.
    Letting the user's writability hint win would silently break the
    read-only contract."""
    captured: dict[str, str] = {}

    def fake_connect(uri):
        captured["uri"] = uri
        return _FakeConn(_FakeCursor(scripts=[]))

    a = SqliteAdapter(connect_fn=fake_connect)
    spec = ConnectionSpec(
        name="db",
        driver="sqlite",
        uri="file:/tmp/events.sqlite?mode=rwc",
    )
    a.open(spec, read_only=True)
    # User wrote mode=rwc; under read_only, we override.
    assert "mode=ro" in captured["uri"]
    assert "mode=rwc" not in captured["uri"]


def test_open_existing_uri_with_mode_memory_keeps_memory():
    """``mode=memory`` and ``mode=ro`` are mutually exclusive in
    SQLite. The URI stays as ``mode=memory``; ``PRAGMA query_only``
    is the read-only enforcement."""
    captured: dict[str, str] = {}

    def fake_connect(uri):
        captured["uri"] = uri
        return _FakeConn(_FakeCursor(scripts=[]))

    a = SqliteAdapter(connect_fn=fake_connect)
    spec = ConnectionSpec(
        name="db",
        driver="sqlite",
        uri="file:memdb1?mode=memory&cache=shared",
    )
    a.open(spec, read_only=True)
    assert "mode=memory" in captured["uri"]
    assert "mode=ro" not in captured["uri"]


def test_open_raises_when_neither_path_nor_uri_set():
    a = SqliteAdapter(connect_fn=lambda uri: None)
    spec = ConnectionSpec(name="db", driver="sqlite")
    with pytest.raises(RuntimeError, match="path"):
        a.open(spec, read_only=True)


# --- probe_freshness (mocked) --------------------------------------------


def test_probe_freshness_combines_data_and_schema_version():
    a = SqliteAdapter()
    cursor = _FakeCursor(
        scripts=[
            ("data_version", [(123,)]),
            ("schema_version", [(7,)]),
        ]
    )
    conn = _FakeConn(cursor)
    token = a.probe_freshness(conn, [QualifiedTable(None, None, "users")])
    assert isinstance(token, FreshnessToken)
    assert not token.is_session_only
    assert token.value != b""


def test_probe_freshness_token_changes_on_data_version():
    a = SqliteAdapter()

    def make_token(data_v: int, schema_v: int) -> bytes:
        cursor = _FakeCursor(
            scripts=[
                ("data_version", [(data_v,)]),
                ("schema_version", [(schema_v,)]),
            ]
        )
        return a.probe_freshness(_FakeConn(cursor), [QualifiedTable(None, None, "t")]).value

    base = make_token(1, 1)
    assert make_token(1, 1) == base
    assert make_token(2, 1) != base  # data_version moved → DML happened
    assert make_token(1, 2) != base  # schema_version moved → DDL happened


def test_probe_freshness_ignores_table_list_db_wide():
    """Capability flag says per_table_freshness=False — the token is
    DB-wide, so different table sets must produce the same token."""
    a = SqliteAdapter()

    def make_token(tables):
        cursor = _FakeCursor(
            scripts=[
                ("data_version", [(99,)]),
                ("schema_version", [(5,)]),
            ]
        )
        return a.probe_freshness(_FakeConn(cursor), tables).value

    a_only = make_token([QualifiedTable(None, None, "users")])
    a_and_b = make_token(
        [QualifiedTable(None, None, "users"), QualifiedTable(None, None, "orders")]
    )
    assert a_only == a_and_b


# --- probe_schema (mocked) ----------------------------------------------


def test_probe_schema_token_reflects_columns():
    a = SqliteAdapter()

    def make_token(rows):
        # PRAGMA table_info row shape: (cid via ordering, name, type,
        # notnull, dflt, pk). Our SCHEMA_QUERY projects just
        # (name, type, notnull, dflt_value, pk).
        cursor = _FakeCursor(scripts=[("pragma_table_info", list(rows))])
        return a.probe_schema(_FakeConn(cursor), [QualifiedTable(None, None, "users")]).value

    base = make_token([("id", "INTEGER", 1, None, 1), ("name", "TEXT", 0, None, 0)])
    assert make_token([("id", "INTEGER", 1, None, 1), ("name", "TEXT", 0, None, 0)]) == base
    # ADD COLUMN
    assert (
        make_token(
            [
                ("id", "INTEGER", 1, None, 1),
                ("name", "TEXT", 0, None, 0),
                ("age", "INTEGER", 0, None, 0),
            ]
        )
        != base
    )
    # Type change
    assert make_token([("id", "BIGINT", 1, None, 1), ("name", "TEXT", 0, None, 0)]) != base
    # Nullability flip
    assert make_token([("id", "INTEGER", 1, None, 1), ("name", "TEXT", 1, None, 0)]) != base


def test_probe_schema_empty_tables_returns_empty_token():
    a = SqliteAdapter()
    token = a.probe_schema(_FakeConn(_FakeCursor(scripts=[])), [])
    assert isinstance(token, SchemaFingerprint)
    assert token.value == b""


def test_probe_schema_uses_pragma_table_info():
    a = SqliteAdapter()
    cursor = _FakeCursor(scripts=[("pragma_table_info", [])])
    conn = _FakeConn(cursor)
    a.probe_schema(conn, [QualifiedTable(None, None, "events")])
    sql, params = cursor.executions[0]
    assert "pragma_table_info" in sql
    assert params == ("events",)


def test_probe_schema_qualifies_attached_database():
    """Codex review fix: when ``QualifiedTable.schema`` is set (the
    SQLite attached-database name), the pragma must run against THAT
    database, not the default ``main``. Otherwise ``aux.events``
    silently fingerprints ``main.events`` (or whatever the search
    order resolves)."""
    a = SqliteAdapter()
    cursor = _FakeCursor(scripts=[("pragma_table_info", [])])
    conn = _FakeConn(cursor)
    a.probe_schema(conn, [QualifiedTable(None, "aux", "events")])
    sql, params = cursor.executions[0]
    assert '"aux".pragma_table_info' in sql, sql
    assert params == ("events",)


def test_probe_schema_unqualified_uses_default_pragma():
    """Without an attached-DB schema, the default-DB pragma form
    runs — preserves backward compatibility with simple cases."""
    a = SqliteAdapter()
    cursor = _FakeCursor(scripts=[("pragma_table_info", [])])
    conn = _FakeConn(cursor)
    a.probe_schema(conn, [QualifiedTable(None, None, "events")])
    sql, _ = cursor.executions[0]
    # Bare pragma form — no "<schema>".pragma_table_info qualifier.
    assert "pragma_table_info" in sql
    assert '".pragma_table_info' not in sql


def test_probe_schema_distinguishes_same_table_in_different_attached_dbs():
    """The fingerprint must differ between ``aux.events`` and
    ``main.events`` even when both happen to have identical column
    structure right now — the qualified pragma queries different
    objects, and the QualifiedTable.render() in the hash already
    differs. This belt-and-suspenders test guards against a
    regression that probed both against ``main``."""
    a = SqliteAdapter()

    rows = [("id", "INTEGER", 1, None, 1)]

    def make_token(table):
        cursor = _FakeCursor(scripts=[("pragma_table_info", list(rows))])
        return a.probe_schema(_FakeConn(cursor), [table]).value

    aux_token = make_token(QualifiedTable(None, "aux", "events"))
    main_token = make_token(QualifiedTable(None, "main", "events"))
    bare_token = make_token(QualifiedTable(None, None, "events"))
    assert aux_token != main_token
    assert aux_token != bare_token
    assert main_token != bare_token


def test_probe_schema_rejects_invalid_attached_db_name():
    """Pragma functions don't accept bind params in the schema
    position, so the schema name is splice-inlined. Identifier
    validation is the injection guard — anything that doesn't match
    the SQLite identifier pattern raises before SQL hits the
    connection."""
    a = SqliteAdapter()
    conn = _FakeConn(_FakeCursor(scripts=[]))
    bad_table = QualifiedTable(None, '"; DROP TABLE x; --', "events")
    with pytest.raises(RuntimeError, match="identifier"):
        a.probe_schema(conn, [bad_table])


# --- registry integration -------------------------------------------------


def test_sqlite_adapter_is_auto_registered():
    from strata.notebook.sql import get_adapter, known_drivers

    assert "sqlite" in known_drivers()
    assert get_adapter("sqlite").name == "sqlite"


# --- real-DB integration --------------------------------------------------
# These tests use ADBC against a real SQLite file. Skipped if the
# package isn't installed, otherwise execute the full open → probe →
# write → reprobe cycle to catch protocol-level mistakes the mock
# tests miss.


def _adbc_sqlite_available() -> bool:
    try:
        import adbc_driver_sqlite.dbapi  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark_real_db = pytest.mark.skipif(
    not _adbc_sqlite_available(), reason="adbc-driver-sqlite not installed"
)


@pytestmark_real_db
def test_real_open_read_only_rejects_write(tmp_path):
    """Read-only enforcement must be the engine's job, not text
    filtering. A DML statement issued through a read-only connection
    has to error at the SQLite engine — though the error may surface
    on cursor close rather than on execute, depending on whether the
    failed statement is reported synchronously."""
    db_path = tmp_path / "rw.sqlite"
    seed = sqlite3.connect(db_path)
    seed.execute("CREATE TABLE t (x INTEGER)")
    seed.execute("INSERT INTO t VALUES (1)")
    seed.commit()
    seed.close()

    a = SqliteAdapter()
    spec = ConnectionSpec(name="db", driver="sqlite", path=str(db_path))
    conn = a.open(spec, read_only=True)
    try:
        # Wrap the whole cursor lifecycle, not just .execute(): ADBC
        # SQLite surfaces the read-only failure during statement
        # finalize/close, not at execute time.
        with pytest.raises(Exception, match="readonly|read.only"):
            with cursor_expected_to_fail_on_close(conn) as cursor:
                cursor.execute("INSERT INTO t VALUES (2)")
    finally:
        conn.close()


@pytestmark_real_db
def test_real_read_only_rejects_write_for_mode_rwc_uri(tmp_path):
    """Codex review fix verification: a user-supplied URI with
    ``mode=rwc`` must NOT remain writable when read_only=True.
    Both the URI override (mode=ro) and the ``PRAGMA query_only``
    fallback exist to make sure this can't slip through."""
    db_path = tmp_path / "rwc.sqlite"
    seed = sqlite3.connect(db_path)
    seed.execute("CREATE TABLE t (x INTEGER)")
    seed.commit()
    seed.close()

    a = SqliteAdapter()
    spec = ConnectionSpec(
        name="db",
        driver="sqlite",
        uri=f"file:{db_path}?mode=rwc",
    )
    conn = a.open(spec, read_only=True)
    try:
        with pytest.raises(Exception, match="readonly|read.only"):
            with cursor_expected_to_fail_on_close(conn) as cursor:
                cursor.execute("INSERT INTO t VALUES (1)")
    finally:
        conn.close()


@pytestmark_real_db
def test_real_query_only_pragma_alone_rejects_writes(tmp_path):
    """Codex review fix verification: ``PRAGMA query_only = ON`` is
    the universal session-level enforcement that backstops ``mode=ro``
    URI flag — for in-memory DBs and any future quirk where ``mode=ro``
    might not propagate. Verify the pragma alone rejects writes when
    applied to an otherwise-writable file connection.

    DDL is also blocked by ``query_only``, so we can't open a fresh
    DB inside the read-only session — seed it externally first."""
    db_path = tmp_path / "qo.sqlite"
    seed = sqlite3.connect(db_path)
    seed.execute("CREATE TABLE t (x INTEGER)")
    seed.commit()
    seed.close()

    a = SqliteAdapter()
    spec = ConnectionSpec(name="db", driver="sqlite", path=str(db_path))
    # Open without URI mode=ro so this isn't covered by the file-handle
    # path; rely on the pragma alone.
    conn = a.open(spec, read_only=False)
    try:
        with conn.cursor() as cursor:
            cursor.execute("PRAGMA query_only = ON")
        with pytest.raises(Exception, match="query_only|readonly|read.only"):
            with cursor_expected_to_fail_on_close(conn) as cursor:
                cursor.execute("INSERT INTO t VALUES (1)")
    finally:
        conn.close()


@pytestmark_real_db
def test_real_schema_version_changes_on_ddl(tmp_path):
    """``schema_version`` is the more reliable cross-implementation
    signal — Python's stdlib ``sqlite3`` and ADBC's bundled SQLite
    are different builds, so ``data_version``'s "writes by another
    connection" semantics aren't guaranteed across them. Schema
    changes always bump ``schema_version`` and must move the
    freshness token."""
    db_path = tmp_path / "ddl.sqlite"
    seed = sqlite3.connect(db_path)
    seed.execute("CREATE TABLE t (x INTEGER)")
    seed.commit()
    seed.close()

    a = SqliteAdapter()
    spec = ConnectionSpec(name="db", driver="sqlite", path=str(db_path))

    probe = a.open(spec, read_only=True)
    try:
        before = a.probe_freshness(probe, [QualifiedTable(None, None, "t")]).value
    finally:
        probe.close()

    altering = sqlite3.connect(db_path)
    altering.execute("ALTER TABLE t ADD COLUMN y TEXT")
    altering.commit()
    altering.close()

    probe2 = a.open(spec, read_only=True)
    try:
        after = a.probe_freshness(probe2, [QualifiedTable(None, None, "t")]).value
    finally:
        probe2.close()

    assert before != after, "schema_version should advance after DDL"


@pytestmark_real_db
def test_real_schema_fingerprint_changes_on_add_column(tmp_path):
    """A schema-only edit (no row changes) must move the schema
    fingerprint even when freshness sees the new ``schema_version``
    too — both probes must catch this."""
    db_path = tmp_path / "schema.sqlite"
    seed = sqlite3.connect(db_path)
    seed.execute("CREATE TABLE t (x INTEGER)")
    seed.commit()
    seed.close()

    a = SqliteAdapter()
    spec = ConnectionSpec(name="db", driver="sqlite", path=str(db_path))

    probe = a.open(spec, read_only=True)
    try:
        before = a.probe_schema(probe, [QualifiedTable(None, None, "t")]).value
    finally:
        probe.close()

    altering = sqlite3.connect(db_path)
    altering.execute("ALTER TABLE t ADD COLUMN y TEXT")
    altering.commit()
    altering.close()

    probe2 = a.open(spec, read_only=True)
    try:
        after = a.probe_schema(probe2, [QualifiedTable(None, None, "t")]).value
    finally:
        probe2.close()

    assert before != after


# --- list_schema (schema-discovery sidebar) -----------------------------


def test_sqlite_list_schema_returns_tables_and_columns(tmp_path):
    """list_schema enumerates tables + columns from sqlite_master /
    pragma_table_info. Powers the schema sidebar so users can see
    what's available before writing SQL."""
    import sqlite3

    db = tmp_path / "schema.db"
    with sqlite3.connect(db) as raw:
        raw.executescript(
            """
            CREATE TABLE products (
                sku TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                price REAL
            );
            CREATE TABLE orders (
                id INTEGER PRIMARY KEY,
                customer TEXT NOT NULL,
                amount REAL
            );
            CREATE VIEW expensive_orders AS
                SELECT * FROM orders WHERE amount > 100;
            """
        )

    pytest.importorskip("adbc_driver_sqlite")

    from strata.notebook.models import ConnectionSpec
    from strata.notebook.sql.drivers.sqlite import SqliteAdapter

    adapter = SqliteAdapter()
    spec = ConnectionSpec(name="db", driver="sqlite", path=str(db))
    conn = adapter.open(spec, read_only=True)
    try:
        schema = adapter.list_schema(conn)
    finally:
        conn.close()

    by_name = {t.name: t for t in schema}
    assert set(by_name) == {"products", "orders", "expensive_orders"}

    products = by_name["products"]
    assert [c.name for c in products.columns] == ["sku", "category", "price"]
    # category is NOT NULL in the schema; price isn't.
    cat = next(c for c in products.columns if c.name == "category")
    assert cat.nullable is False
    price = next(c for c in products.columns if c.name == "price")
    assert price.nullable is True

    # System tables (sqlite_*) are excluded from the listing.
    assert not any(t.name.startswith("sqlite_") for t in schema)


def test_sqlite_list_schema_empty_database(tmp_path):
    """A brand-new SQLite file has no tables — list_schema returns []."""
    pytest.importorskip("adbc_driver_sqlite")

    from strata.notebook.models import ConnectionSpec
    from strata.notebook.sql.drivers.sqlite import SqliteAdapter

    db = tmp_path / "empty.db"
    db.touch()
    adapter = SqliteAdapter()
    spec = ConnectionSpec(name="db", driver="sqlite", path=str(db))
    conn = adapter.open(spec, read_only=True)
    try:
        schema = adapter.list_schema(conn)
    finally:
        conn.close()
    assert schema == []
