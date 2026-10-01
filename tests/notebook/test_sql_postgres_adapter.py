"""PostgreSQL DriverAdapter contract and fingerprint shape, with mocked ADBC connections."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from strata.notebook.models import ConnectionSpec
from strata.notebook.sql import FreshnessToken, QualifiedTable, SchemaFingerprint
from strata.notebook.sql.drivers.postgresql import PostgresAdapter, _splice_userinfo

# --- mock-conn helper -------------------------------------------------------


class _FakeCursor:
    """Minimal DBAPI cursor that returns scripted rows.

    ``rows`` is a list of (query_substring, result) pairs: a fetch returns the result whose
    substring matches the last executed SQL. ``executions`` records every (sql, params).
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
        if last is None:
            return None
        return last

    def fetchall(self):
        last = self._last_match
        if isinstance(last, list):
            return last
        return [last] if last is not None else []


class _FakeConn:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor
        self.commits = 0

    @contextmanager
    def cursor(self):
        yield self._cursor

    def commit(self):
        self.commits += 1


# --- capability flags -------------------------------------------------------


def test_capabilities_match_design_doc():
    a = PostgresAdapter()
    assert a.name == "postgresql"
    assert a.sqlglot_dialect == "postgres"
    assert a.capabilities.per_table_freshness is True
    assert a.capabilities.supports_snapshot is False
    # Postgres stats freeze inside an open txn, so the probe needs its own connection.
    assert a.capabilities.needs_separate_probe_conn is True


# --- canonicalize_connection_id --------------------------------------------


def test_connection_id_stable_across_url_and_components():
    """The same connection via uri and via discrete fields produces the same id."""
    a = PostgresAdapter()
    via_uri = ConnectionSpec(
        name="x",
        driver="postgresql",
        uri="postgresql://reader@db.host:5432/events",
    )
    via_components = ConnectionSpec(
        name="x",
        driver="postgresql",
        host="db.host",
        port=5432,
        database="events",
        user="reader",
    )
    assert a.canonicalize_connection_id(via_uri) == a.canonicalize_connection_id(via_components)


def test_connection_id_excludes_password_and_runtime_tunables():
    """Password, application_name and connect_timeout don't change object visibility."""
    a = PostgresAdapter()
    base = ConnectionSpec(
        name="x",
        driver="postgresql",
        uri="postgresql://reader@db.host:5432/events",
    )
    with_pw = ConnectionSpec(
        name="x",
        driver="postgresql",
        uri="postgresql://reader:changeit@db.host:5432/events",
    )
    with_appname = ConnectionSpec(
        name="x",
        driver="postgresql",
        uri="postgresql://reader@db.host:5432/events",
        options={"application_name": "strata"},
    )
    with_timeout = ConnectionSpec(
        name="x",
        driver="postgresql",
        uri="postgresql://reader@db.host:5432/events",
        options={"connect_timeout": 5},
    )
    cid = a.canonicalize_connection_id(base)
    assert a.canonicalize_connection_id(with_pw) == cid
    assert a.canonicalize_connection_id(with_appname) == cid
    assert a.canonicalize_connection_id(with_timeout) == cid


def test_connection_id_changes_on_identity_shaping_fields():
    """Host, port, database, user, role and search_path all change object visibility."""
    a = PostgresAdapter()
    base = ConnectionSpec(
        name="x",
        driver="postgresql",
        host="db.host",
        port=5432,
        database="events",
        user="reader",
    )
    cid = a.canonicalize_connection_id(base)
    cases = [
        base.model_copy(update={"host": "other.host"}),
        base.model_copy(update={"port": 5433}),
        base.model_copy(update={"database": "metrics"}),
        base.model_copy(update={"user": "admin"}),
    ]
    for variant in cases:
        assert a.canonicalize_connection_id(variant) != cid

    # Role and search_path live in extras / options.
    role_variant = ConnectionSpec(
        name="x",
        driver="postgresql",
        host="db.host",
        port=5432,
        database="events",
        user="reader",
        role="ro_role",
    )
    assert a.canonicalize_connection_id(role_variant) != cid

    sp_variant = base.model_copy(update={"options": {"search_path": "analytics,public"}})
    assert a.canonicalize_connection_id(sp_variant) != cid


def test_connection_id_resolves_auth_user_indirection(monkeypatch):
    """Same ``auth.user`` env var gives the same id; changing its value changes the id."""
    a = PostgresAdapter()
    monkeypatch.setenv("PGUSER", "alice")
    spec = ConnectionSpec(
        name="x",
        driver="postgresql",
        host="h",
        port=5432,
        database="d",
        auth={"user": "${PGUSER}"},
    )
    cid_alice = a.canonicalize_connection_id(spec)

    monkeypatch.setenv("PGUSER", "bob")
    cid_bob = a.canonicalize_connection_id(spec)

    assert cid_alice != cid_bob


def test_connection_id_falls_back_when_auth_var_missing(monkeypatch):
    """An unset auth env var falls back to the raw spec value, so the id stays stable."""
    a = PostgresAdapter()
    monkeypatch.delenv("PGUSER", raising=False)
    spec = ConnectionSpec(
        name="x",
        driver="postgresql",
        host="h",
        port=5432,
        database="d",
        auth={"user": "${PGUSER}"},
    )
    cid = a.canonicalize_connection_id(spec)
    assert isinstance(cid, str) and len(cid) == 64


# --- open() and read-only enforcement --------------------------------------


def test_open_with_read_only_sets_session_read_only():
    """Read-only is the security boundary: the SET runs before the cell can query."""
    cursor = _FakeCursor(scripts=[])
    conn = _FakeConn(cursor)
    a = PostgresAdapter(connect_fn=lambda uri: conn)

    spec = ConnectionSpec(name="x", driver="postgresql", uri="postgresql://x@h/d")
    result = a.open(spec, read_only=True)

    assert result is conn
    assert any("default_transaction_read_only" in sql for sql, _ in cursor.executions)
    assert conn.commits == 1


def test_open_without_read_only_skips_set():
    """Read-only mode is for SQL cells, not general adapter use that might write."""
    cursor = _FakeCursor(scripts=[])
    conn = _FakeConn(cursor)
    a = PostgresAdapter(connect_fn=lambda uri: conn)

    spec = ConnectionSpec(name="x", driver="postgresql", uri="postgresql://x@h/d")
    a.open(spec, read_only=False)

    assert not any("default_transaction_read_only" in sql for sql, _ in cursor.executions)
    assert conn.commits == 0


def test_open_resolves_auth_indirection_into_uri(monkeypatch):
    monkeypatch.setenv("PGPASS", "s3cret")
    captured: dict[str, str] = {}

    def fake_connect(uri):
        captured["uri"] = uri
        return _FakeConn(_FakeCursor(scripts=[]))

    a = PostgresAdapter(connect_fn=fake_connect)
    spec = ConnectionSpec(
        name="x",
        driver="postgresql",
        uri="postgresql://reader@db.host:5432/events",
        auth={"password": "${PGPASS}"},
    )
    a.open(spec, read_only=True)
    assert "s3cret" in captured["uri"]
    assert "reader" in captured["uri"]


def test_open_raises_when_auth_var_missing(monkeypatch):
    monkeypatch.delenv("PGPASS", raising=False)
    a = PostgresAdapter(connect_fn=lambda uri: None)
    spec = ConnectionSpec(
        name="x",
        driver="postgresql",
        uri="postgresql://reader@db.host:5432/events",
        auth={"password": "${PGPASS}"},
    )
    with pytest.raises(RuntimeError, match="PGPASS"):
        a.open(spec, read_only=True)


def test_open_builds_uri_from_components(monkeypatch):
    monkeypatch.setenv("PGUSER", "alice")
    monkeypatch.setenv("PGPASS", "s3cret")
    captured: dict[str, str] = {}

    def fake_connect(uri):
        captured["uri"] = uri
        return _FakeConn(_FakeCursor(scripts=[]))

    a = PostgresAdapter(connect_fn=fake_connect)
    spec = ConnectionSpec(
        name="x",
        driver="postgresql",
        host="db.host",
        port=5432,
        database="events",
        auth={"user": "${PGUSER}", "password": "${PGPASS}"},
    )
    a.open(spec, read_only=True)
    uri = captured["uri"]
    assert uri.startswith("postgresql://")
    assert "alice:s3cret" in uri
    assert "db.host:5432" in uri
    assert uri.endswith("/events")


def test_splice_userinfo_preserves_path_and_port():
    out = _splice_userinfo(
        "postgresql://existing@db.host:5432/events?sslmode=require",
        "alice",
        "s3cret",
    )
    assert out.startswith("postgresql://alice:s3cret@db.host:5432/events")
    assert "sslmode=require" in out


def test_splice_userinfo_handles_special_chars():
    """Passwords with @ / : / # are percent-encoded, not broken by the parser."""
    out = _splice_userinfo(
        "postgresql://existing@h/d",
        "user@home",
        "p@ss:word#1",
    )
    # The host portion must still be `h/d`, not parsed as part of the
    # password.
    assert "@h/d" in out
    assert "p%40ss%3Aword%231" in out


# --- probe_freshness -------------------------------------------------------


def test_probe_freshness_empty_tables_returns_empty_token():
    a = PostgresAdapter()
    token = a.probe_freshness(_FakeConn(_FakeCursor(scripts=[])), [])
    assert isinstance(token, FreshnessToken)
    assert token.value == b""
    assert not token.is_session_only


def test_probe_freshness_token_reflects_dml_relfilenode_and_schema():
    """Any of (dml, relfilenode, resolved_schema) changing flips the token.

    Resolved schema is folded in so an unqualified name that resolves elsewhere differs.
    """
    a = PostgresAdapter()

    def make_token(dml: int, relfilenode: int, resolved_schema: str = "public") -> bytes:
        cursor = _FakeCursor(scripts=[("to_regclass", (dml, relfilenode, resolved_schema))])
        conn = _FakeConn(cursor)
        return a.probe_freshness(
            conn, [QualifiedTable(catalog=None, schema=None, name="users")]
        ).value

    base = make_token(123, 456)
    assert make_token(123, 456) == base
    assert make_token(124, 456) != base  # DML moved
    assert make_token(123, 457) != base  # relfilenode moved (rewrite-style DDL)
    assert make_token(123, 456, "analytics") != base  # search_path resolved differently


def test_probe_freshness_is_order_invariant():
    """The token doesn't depend on the order the SQL parser yielded the tables."""
    a = PostgresAdapter()
    # Map identifier-string → (dml, relfilenode, resolved_schema).
    rows = {
        '"public"."users"': (10, 100, "public"),
        '"public"."orders"': (20, 200, "public"),
    }

    def conn_factory():
        cursor = MagicMock()
        cursor.execute = MagicMock()

        def fetchone():
            last_call = cursor.execute.call_args
            if last_call is None:
                return None
            params = last_call[0][1]
            return rows.get(params[0])

        cursor.fetchone = fetchone
        conn = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cursor
        conn.cursor.return_value.__exit__.return_value = False
        return conn

    t_users = QualifiedTable(catalog=None, schema="public", name="users")
    t_orders = QualifiedTable(catalog=None, schema="public", name="orders")
    a_b = a.probe_freshness(conn_factory(), [t_users, t_orders]).value
    b_a = a.probe_freshness(conn_factory(), [t_orders, t_users]).value
    assert a_b == b_a


def test_probe_freshness_missing_table_marks_session_only():
    """A NULL ``to_regclass`` (missing or unresolvable table) gives a session-only token."""
    a = PostgresAdapter()
    cursor = _FakeCursor(scripts=[("to_regclass", None)])
    conn = _FakeConn(cursor)
    token = a.probe_freshness(conn, [QualifiedTable(catalog=None, schema=None, name="ghost")])
    assert token.is_session_only is True


def test_probe_freshness_uses_to_regclass_for_qualified_name():
    """A qualified table goes to ``to_regclass`` as ``"schema"."name"``, an unqualified one bare."""
    a = PostgresAdapter()
    cursor = _FakeCursor(scripts=[("to_regclass", (1, 2, "analytics"))])
    conn = _FakeConn(cursor)
    a.probe_freshness(
        conn,
        [QualifiedTable(catalog=None, schema="analytics", name="events")],
    )
    sql, params = cursor.executions[0]
    assert "to_regclass" in sql
    assert params == ('"analytics"."events"',)


def test_probe_freshness_lets_search_path_resolve_unqualified_name():
    """Unqualified names resolve via the live ``search_path``, not a hardcoded ``public``."""
    a = PostgresAdapter()
    cursor = _FakeCursor(scripts=[("to_regclass", (1, 2, "analytics"))])
    conn = _FakeConn(cursor)
    a.probe_freshness(conn, [QualifiedTable(catalog=None, schema=None, name="events")])
    sql, params = cursor.executions[0]
    assert "to_regclass" in sql
    # A single-component identifier resolves through the connection's search_path,
    # not a hardcoded "public".
    assert params == ('"events"',)


def test_probe_freshness_resolved_schema_distinguishes_unqualified_collisions():
    """Different search_paths probing one unqualified name produce different tokens."""
    a = PostgresAdapter()

    def token_for_resolved(resolved_schema: str) -> bytes:
        cursor = _FakeCursor(scripts=[("to_regclass", (1, 2, resolved_schema))])
        conn = _FakeConn(cursor)
        return a.probe_freshness(
            conn, [QualifiedTable(catalog=None, schema=None, name="events")]
        ).value

    assert token_for_resolved("public") != token_for_resolved("analytics")


# --- probe_schema ---------------------------------------------------------


def test_probe_schema_token_reflects_columns():
    """Adding or removing a column, or changing a type, changes the token."""
    a = PostgresAdapter()

    def make_token(rows):
        # Schema query reads pg_attribute with to_regclass.
        cursor = _FakeCursor(scripts=[("pg_attribute", list(rows))])
        conn = _FakeConn(cursor)
        return a.probe_schema(
            conn, [QualifiedTable(catalog=None, schema="public", name="users")]
        ).value

    base = make_token([("id", "integer", False), ("name", "text", True)])
    assert make_token([("id", "integer", False), ("name", "text", True)]) == base
    # ADD COLUMN
    assert (
        make_token(
            [
                ("id", "integer", False),
                ("name", "text", True),
                ("age", "integer", True),
            ]
        )
        != base
    )
    # Type change (format_type output differs)
    assert make_token([("id", "bigint", False), ("name", "text", True)]) != base
    # Nullability flip
    assert make_token([("id", "integer", False), ("name", "text", False)]) != base


def test_probe_schema_uses_to_regclass():
    """The schema probe resolves names with ``to_regclass``, like the freshness probe."""
    a = PostgresAdapter()
    cursor = _FakeCursor(scripts=[("pg_attribute", [])])
    conn = _FakeConn(cursor)
    a.probe_schema(conn, [QualifiedTable(catalog=None, schema=None, name="events")])
    sql, params = cursor.executions[0]
    assert "to_regclass" in sql
    assert params == ('"events"',)


def test_probes_ask_for_the_names_postgres_stores():
    """Postgres folds unquoted identifiers to lowercase; the probe must ask for that.

    ``FROM Analytics.Events`` reads ``analytics.events``; probing ``"Analytics"."Events"``
    finds nothing and the token never moves. A quoted name is stored as written.
    """
    from strata.notebook.sql.analyzer import analyze_sql_cell

    analysis = analyze_sql_cell(
        '# @sql connection=db\nSELECT * FROM Analytics.Events JOIN "MixedCase" USING (id)',
        dialect="postgres",
    )
    a = PostgresAdapter()
    cursor = _FakeCursor(scripts=[("to_regclass", (1, 2, "analytics"))])
    a.probe_freshness(_FakeConn(cursor), analysis.tables)
    assert sorted(params for _sql, params in cursor.executions) == [
        ('"MixedCase"',),
        ('"analytics"."events"',),
    ]

    cursor = _FakeCursor(scripts=[("pg_attribute", [])])
    a.probe_schema(_FakeConn(cursor), analysis.tables)
    assert sorted(params for _sql, params in cursor.executions) == [
        ('"MixedCase"',),
        ('"analytics"."events"',),
    ]


def test_probe_schema_empty_tables_returns_empty_token():
    a = PostgresAdapter()
    token = a.probe_schema(_FakeConn(_FakeCursor(scripts=[])), [])
    assert isinstance(token, SchemaFingerprint)
    assert token.value == b""


# --- role / search_path applied during open() -----------------------------


def test_open_applies_role():
    """``role`` is in connection_id, so open() must apply it to the live session."""
    cursor = _FakeCursor(scripts=[])
    conn = _FakeConn(cursor)
    a = PostgresAdapter(connect_fn=lambda uri: conn)

    spec = ConnectionSpec(
        name="x",
        driver="postgresql",
        uri="postgresql://x@h/d",
        role="readers",
    )
    a.open(spec, read_only=True)

    assert any('SET ROLE "readers"' in sql for sql, _ in cursor.executions)


def test_open_rejects_role_with_invalid_chars():
    """``SET ROLE`` takes no bind parameters, so the role is validated strictly."""
    a = PostgresAdapter(connect_fn=lambda uri: _FakeConn(_FakeCursor(scripts=[])))
    spec = ConnectionSpec(
        name="x",
        driver="postgresql",
        uri="postgresql://x@h/d",
        role='"; DROP TABLE users; --',
    )
    with pytest.raises(RuntimeError, match="role"):
        a.open(spec, read_only=True)


def test_open_applies_search_path_string_form():
    cursor = _FakeCursor(scripts=[])
    conn = _FakeConn(cursor)
    a = PostgresAdapter(connect_fn=lambda uri: conn)

    spec = ConnectionSpec(
        name="x",
        driver="postgresql",
        uri="postgresql://x@h/d",
        options={"search_path": "analytics, public"},
    )
    a.open(spec, read_only=True)

    set_sp = next(
        (sql for sql, _ in cursor.executions if sql.startswith("SET search_path")),
        None,
    )
    assert set_sp is not None
    assert '"analytics"' in set_sp
    assert '"public"' in set_sp


def test_open_applies_search_path_list_form():
    cursor = _FakeCursor(scripts=[])
    conn = _FakeConn(cursor)
    a = PostgresAdapter(connect_fn=lambda uri: conn)

    spec = ConnectionSpec(
        name="x",
        driver="postgresql",
        uri="postgresql://x@h/d",
        options={"search_path": ["analytics", "public"]},
    )
    a.open(spec, read_only=True)
    set_sp = next(
        (sql for sql, _ in cursor.executions if sql.startswith("SET search_path")),
        None,
    )
    assert set_sp is not None
    assert '"analytics"' in set_sp
    assert '"public"' in set_sp


def test_open_rejects_search_path_with_invalid_entries():
    a = PostgresAdapter(connect_fn=lambda uri: _FakeConn(_FakeCursor(scripts=[])))
    spec = ConnectionSpec(
        name="x",
        driver="postgresql",
        uri="postgresql://x@h/d",
        options={"search_path": "analytics, ; DROP TABLE x"},
    )
    with pytest.raises(RuntimeError, match="search_path"):
        a.open(spec, read_only=True)


# --- registry integration --------------------------------------------------


def test_postgres_adapter_is_auto_registered():
    """Importing the sql package registers the driver so SQL cell validation knows it."""
    from strata.notebook.sql import get_adapter, known_drivers

    assert "postgresql" in known_drivers()
    assert get_adapter("postgresql").name == "postgresql"


def test_every_advertised_builtin_driver_registers():
    """Every name in ``_BUILTIN_DRIVERS`` has a module exposing ``register()``."""
    from strata.notebook.sql import known_drivers
    from strata.notebook.sql.drivers import (
        builtin_driver_names,
        register_default_adapters,
    )
    from strata.notebook.sql.registry import _reset_for_tests, _restore_defaults_for_tests

    _reset_for_tests()
    try:
        register_default_adapters()
        registered = set(known_drivers())
        for module_name in builtin_driver_names():
            # Every built-in driver registers under its module name in drivers/.
            assert module_name in registered, (
                f"built-in driver module {module_name!r} did not "
                f"register its adapter; known after register: {registered}"
            )
    finally:
        _restore_defaults_for_tests()
