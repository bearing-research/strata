"""Snowflake DriverAdapter tests with mocked ADBC connections: contract, identity hash, probes."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from strata.notebook.models import ConnectionSpec
from strata.notebook.sql import FreshnessToken, QualifiedTable, SchemaFingerprint
from strata.notebook.sql.drivers.snowflake import SnowflakeAdapter

# --- mock-conn helper -------------------------------------------------------


class _FakeCursor:
    """Minimal DBAPI cursor that returns scripted rows.

    ``scripts`` holds (query_substring, result) pairs; a fetch returns the result whose substring
    matches the last executed SQL. Every (sql, params) pair is recorded.
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

    def close(self):
        pass


class _FakeConn:
    def __init__(self, cursor: _FakeCursor):
        self._cursor = cursor
        self.commits = 0

    @contextmanager
    def cursor(self):
        yield self._cursor

    def commit(self):
        self.commits += 1

    def close(self):
        pass


# --- capabilities + dialect ------------------------------------------------


def test_capabilities_match_design():
    a = SnowflakeAdapter()
    assert a.name == "snowflake"
    assert a.sqlglot_dialect == "snowflake"
    assert a.capabilities.per_table_freshness is True
    # Time travel: a snapshot is a timestamp (time_travel.py).
    assert a.capabilities.supports_snapshot is True
    # INFORMATION_SCHEMA isn't frozen inside a transaction the
    # way Postgres's pg_stat_* views are.
    assert a.capabilities.needs_separate_probe_conn is False


# --- canonicalize_connection_id -------------------------------------------


def test_connection_id_includes_all_identity_shaping_fields():
    """Account, user, role, warehouse, database and schema each change the id."""
    a = SnowflakeAdapter()
    base = ConnectionSpec(
        name="x",
        driver="snowflake",
        account="ACME-PROD",
        user="reader",
        role="ANALYTICS_RO",
        warehouse="WH_SMALL",
        database="EVENTS",
        schema="public",
    )
    cid = a.canonicalize_connection_id(base)
    for field, new_value in [
        ("account", "ACME-DEV"),
        ("user", "writer"),
        ("role", "ANALYTICS_RW"),
        ("warehouse", "WH_LARGE"),
        ("database", "METRICS"),
        ("schema", "staging"),
    ]:
        variant = base.model_copy(update={field: new_value})
        assert a.canonicalize_connection_id(variant) != cid, (
            f"changing {field} should change the connection id"
        )


def test_connection_id_excludes_password():
    a = SnowflakeAdapter()
    base = ConnectionSpec(
        name="x",
        driver="snowflake",
        account="ACME",
        user="reader",
        auth={"user": "reader", "password": "${SF_PASS}"},
    )
    with_different_pw = ConnectionSpec(
        name="x",
        driver="snowflake",
        account="ACME",
        user="reader",
        auth={"user": "reader", "password": "${OTHER_PASS}"},
    )
    assert a.canonicalize_connection_id(base) == a.canonicalize_connection_id(with_different_pw)


def test_connection_id_resolves_auth_user_var(monkeypatch):
    """A literal user and a ``${VAR}`` resolving to the same user produce the same id."""
    monkeypatch.setenv("SF_USER", "reader")
    a = SnowflakeAdapter()
    via_literal = ConnectionSpec(
        name="x",
        driver="snowflake",
        account="ACME",
        user="reader",
    )
    via_var = ConnectionSpec(
        name="x",
        driver="snowflake",
        account="ACME",
        auth={"user": "${SF_USER}"},
    )
    assert a.canonicalize_connection_id(via_literal) == a.canonicalize_connection_id(via_var)


# --- open() session-shaping --------------------------------------------------


def test_open_applies_role_warehouse_database_schema_in_order():
    """Each identity field issues its own ``USE ...`` before the connection is returned, in spec
    order.
    """
    cur = _FakeCursor([])
    a = SnowflakeAdapter(connect_fn=lambda _uri: _FakeConn(cur))

    spec = ConnectionSpec(
        name="x",
        driver="snowflake",
        account="ACME",
        role="ANALYTICS_RO",
        warehouse="WH_SMALL",
        database="EVENTS",
        schema="PUBLIC",
    )
    a.open(spec, read_only=True)

    issued = [sql for sql, _ in cur.executions]
    assert issued == [
        'USE ROLE "ANALYTICS_RO"',
        'USE WAREHOUSE "WH_SMALL"',
        'USE DATABASE "EVENTS"',
        'USE SCHEMA "PUBLIC"',
    ]


def test_open_skips_unset_fields():
    """A spec with only account and role must not ``USE WAREHOUSE`` an empty string."""
    cur = _FakeCursor([])
    a = SnowflakeAdapter(connect_fn=lambda _uri: _FakeConn(cur))
    a.open(
        ConnectionSpec(name="x", driver="snowflake", account="ACME", role="ANALYTICS_RO"),
        read_only=True,
    )
    assert [sql for sql, _ in cur.executions] == ['USE ROLE "ANALYTICS_RO"']


def test_open_rejects_invalid_identifier():
    """``USE`` takes no bind parameters, so spliced identifiers must be validated before any SQL
    runs.
    """
    cur = _FakeCursor([])
    a = SnowflakeAdapter(connect_fn=lambda _uri: _FakeConn(cur))
    with pytest.raises(RuntimeError, match="Snowflake identifier"):
        a.open(
            ConnectionSpec(
                name="x",
                driver="snowflake",
                account="ACME",
                role="evil; DROP TABLE secret",
            ),
            read_only=True,
        )
    # No SQL was issued before the validation error.
    assert cur.executions == []


# --- _build_uri --------------------------------------------------------------


def test_build_uri_from_components():
    a = SnowflakeAdapter()
    spec = ConnectionSpec(
        name="x",
        driver="snowflake",
        account="ACME-PROD",
        database="EVENTS",
        schema="PUBLIC",
        warehouse="WH_SMALL",
        role="ANALYTICS_RO",
        auth={"user": "reader", "password": "p@ss/w0rd"},
    )
    uri = a._build_uri(spec)
    assert uri.startswith("snowflake://reader:p%40ss%2Fw0rd@ACME-PROD/EVENTS/PUBLIC")
    assert "warehouse=WH_SMALL" in uri
    assert "role=ANALYTICS_RO" in uri


def test_build_uri_passes_through_explicit_uri():
    """A hand-rolled URI (private-key auth, custom params) is used as is."""
    a = SnowflakeAdapter()
    spec = ConnectionSpec(
        name="x",
        driver="snowflake",
        uri="snowflake://reader@ACME/EVENTS?authenticator=externalbrowser",
    )
    assert a._build_uri(spec) == ("snowflake://reader@ACME/EVENTS?authenticator=externalbrowser")


def test_build_uri_requires_account_or_uri():
    a = SnowflakeAdapter()
    with pytest.raises(RuntimeError, match="account"):
        a._build_uri(ConnectionSpec(name="x", driver="snowflake"))


# --- probe_freshness --------------------------------------------------------


def test_probe_freshness_groups_by_database():
    """INFORMATION_SCHEMA is per database, so one query runs per touched database."""
    cur = _FakeCursor(
        [
            ("INFORMATION_SCHEMA.TABLES", ("PUBLIC", "events", "2026-05-01 12:00:00")),
        ]
    )
    a = SnowflakeAdapter()
    tables = [
        QualifiedTable(catalog="EVENTS", schema="PUBLIC", name="events"),
        QualifiedTable(catalog="METRICS", schema="PUBLIC", name="counters"),
    ]
    token = a.probe_freshness(_FakeConn(cur), tables)
    assert isinstance(token, FreshnessToken)
    assert token.value  # non-empty hash

    # The adapter issued one INFORMATION_SCHEMA query per database.
    db_queries = [sql for sql, _ in cur.executions if "INFORMATION_SCHEMA" in sql]
    assert len(db_queries) == 2
    # Each query targets the right database (spliced as
    # "<DB>".INFORMATION_SCHEMA…).
    assert any('"EVENTS".INFORMATION_SCHEMA' in q for q in db_queries)
    assert any('"METRICS".INFORMATION_SCHEMA' in q for q in db_queries)


def test_probe_freshness_uses_current_database_for_unqualified_tables():
    cur = _FakeCursor(
        [
            ("CURRENT_DATABASE", ("MAIN_DB",)),
            ("INFORMATION_SCHEMA.TABLES", ("PUBLIC", "events", "ts")),
        ]
    )
    a = SnowflakeAdapter()
    tables = [QualifiedTable(catalog=None, schema="PUBLIC", name="events")]
    a.probe_freshness(_FakeConn(cur), tables)

    queries = [sql for sql, _ in cur.executions]
    # First we resolve the current database, then use it.
    assert "CURRENT_DATABASE" in queries[0]
    assert any('"MAIN_DB".INFORMATION_SCHEMA.TABLES' in q for q in queries)


def test_probe_freshness_token_changes_on_last_altered():
    a = SnowflakeAdapter()
    tables = [QualifiedTable(catalog="EVENTS", schema="PUBLIC", name="orders")]

    cur1 = _FakeCursor([("INFORMATION_SCHEMA.TABLES", ("PUBLIC", "orders", "2026-05-01 12:00:00"))])
    token1 = a.probe_freshness(_FakeConn(cur1), tables)

    cur2 = _FakeCursor([("INFORMATION_SCHEMA.TABLES", ("PUBLIC", "orders", "2026-05-01 12:01:00"))])
    token2 = a.probe_freshness(_FakeConn(cur2), tables)
    assert token1.value != token2.value


def test_probe_freshness_missing_table_distinct_from_present():
    """Otherwise a permission lapse masquerades as unchanged."""
    a = SnowflakeAdapter()
    tables = [QualifiedTable(catalog="EVENTS", schema="PUBLIC", name="orders")]

    cur_present = _FakeCursor([("INFORMATION_SCHEMA.TABLES", ("PUBLIC", "orders", "2026-05-01"))])
    cur_missing = _FakeCursor([("INFORMATION_SCHEMA.TABLES", None)])
    a_present = a.probe_freshness(_FakeConn(cur_present), tables)
    a_missing = a.probe_freshness(_FakeConn(cur_missing), tables)
    assert a_present.value != a_missing.value


def test_probes_ask_for_the_names_snowflake_stores():
    """Unquoted identifiers are stored uppercased and INFORMATION_SCHEMA compares exactly.

    Probing the lowercase name found nothing, so the token never changed and the cell never saw a
    write. A quoted name is stored as written.
    """
    from strata.notebook.sql.analyzer import analyze_sql_cell

    src = '# @sql connection=db\nSELECT * FROM mydb.public.events JOIN "MixedCase" USING (id)'
    tables = analyze_sql_cell(src, dialect=SnowflakeAdapter.sqlglot_dialect).tables
    a = SnowflakeAdapter()

    def probe_params(probe) -> tuple[list[str], list[tuple]]:
        cur = _FakeCursor([("CURRENT_DATABASE", ("MAIN_DB", "PUBLIC")), ("RETENTION_TIME", (1,))])
        probe(_FakeConn(cur), tables)
        info = [(sql, params) for sql, params in cur.executions if "INFORMATION_SCHEMA" in sql]
        return [sql for sql, _ in info], [params for _, params in info]

    for probe in (a.probe_freshness, a.probe_schema):
        queries, params = probe_params(probe)
        assert any('"MYDB".INFORMATION_SCHEMA' in q for q in queries)
        assert sorted(params) == [("PUBLIC", "EVENTS"), ("PUBLIC", "MixedCase")]

    _, params = probe_params(
        lambda conn, tables: a.retention_until(conn, tables, "2026-05-01T00:00:00Z")
    )
    assert sorted(params) == [("PUBLIC", "EVENTS"), ("PUBLIC", "MixedCase")]


def test_probe_freshness_empty_tables():
    """No tables touched gives an empty token and issues no queries."""
    cur = _FakeCursor([])
    a = SnowflakeAdapter()
    token = a.probe_freshness(_FakeConn(cur), [])
    assert token.value == b""
    assert cur.executions == []


# --- probe_schema -----------------------------------------------------------


def test_probe_schema_walks_columns():
    """Reads INFORMATION_SCHEMA.COLUMNS per touched table, grouped by database."""
    cur = _FakeCursor(
        [
            (
                "INFORMATION_SCHEMA.COLUMNS",
                [
                    ("id", "NUMBER", "NO"),
                    ("label", "TEXT", "YES"),
                ],
            ),
        ]
    )
    a = SnowflakeAdapter()
    tables = [QualifiedTable(catalog="EVENTS", schema="PUBLIC", name="orders")]
    fp = a.probe_schema(_FakeConn(cur), tables)
    assert isinstance(fp, SchemaFingerprint)
    assert fp.value


def test_probe_schema_fingerprint_changes_on_column_set():
    """Adding a column changes the fingerprint even if LAST_ALTERED did not move."""
    a = SnowflakeAdapter()
    tables = [QualifiedTable(catalog="EVENTS", schema="PUBLIC", name="orders")]

    cur1 = _FakeCursor([("INFORMATION_SCHEMA.COLUMNS", [("id", "NUMBER", "NO")])])
    cur2 = _FakeCursor(
        [
            (
                "INFORMATION_SCHEMA.COLUMNS",
                [("id", "NUMBER", "NO"), ("label", "TEXT", "YES")],
            )
        ]
    )
    fp1 = a.probe_schema(_FakeConn(cur1), tables)
    fp2 = a.probe_schema(_FakeConn(cur2), tables)
    assert fp1.value != fp2.value


# --- list_schema ------------------------------------------------------------


def test_list_schema_uses_current_database():
    """Discovery is scoped to the current database; columns come back in ordinal_position order."""
    cur = _FakeCursor(
        [
            ("CURRENT_DATABASE", ("EVENTS",)),
            (
                "INFORMATION_SCHEMA.TABLES",
                [
                    ("EVENTS", "PUBLIC", "orders", "id", "NUMBER", "NO"),
                    ("EVENTS", "PUBLIC", "orders", "amount", "NUMBER", "YES"),
                    ("EVENTS", "PUBLIC", "products", "sku", "TEXT", "NO"),
                ],
            ),
        ]
    )
    a = SnowflakeAdapter()
    schema = a.list_schema(_FakeConn(cur))
    by_name = {t.name: t for t in schema}
    assert set(by_name) == {"orders", "products"}

    orders = by_name["orders"]
    assert [c.name for c in orders.columns] == ["id", "amount"]
    id_col = orders.columns[0]
    assert id_col.type == "NUMBER"
    assert id_col.nullable is False
    amount_col = orders.columns[1]
    assert amount_col.nullable is True


def test_list_schema_empty_when_no_database():
    """A role with no default database returns [] instead of running an unscoped query."""
    cur = _FakeCursor([("CURRENT_DATABASE", (None,))])
    a = SnowflakeAdapter()
    assert a.list_schema(_FakeConn(cur)) == []


# --- URI identity, current schema, write role ---


def test_connection_id_includes_uri_components():
    """Two URI-only connections to different databases get different ids.

    The adapter parses ``spec.uri`` and folds account, database, schema, warehouse and role into the
    identity.
    """
    a = SnowflakeAdapter()
    db_a = ConnectionSpec(
        name="a",
        driver="snowflake",
        uri="snowflake://reader@ACME/DB_A/PUBLIC?warehouse=W&role=R",
    )
    db_b = ConnectionSpec(
        name="b",
        driver="snowflake",
        uri="snowflake://reader@ACME/DB_B/PUBLIC?warehouse=W&role=R",
    )
    different_role = ConnectionSpec(
        name="c",
        driver="snowflake",
        uri="snowflake://reader@ACME/DB_A/PUBLIC?warehouse=W&role=OTHER",
    )

    assert a.canonicalize_connection_id(db_a) != a.canonicalize_connection_id(db_b)
    assert a.canonicalize_connection_id(db_a) != a.canonicalize_connection_id(different_role)


def test_connection_id_discrete_fields_override_uri():
    """Discrete fields override the URI: ``spec.role = "OVERRIDE"`` beats a role in the URI."""
    a = SnowflakeAdapter()
    base_uri = "snowflake://reader@ACME/EVENTS/PUBLIC?warehouse=WH&role=URI_ROLE"
    spec = ConnectionSpec(
        name="x",
        driver="snowflake",
        uri=base_uri,
        role="DISCRETE_ROLE",
    )
    same_explicit = ConnectionSpec(
        name="x",
        driver="snowflake",
        account="ACME",
        database="EVENTS",
        schema="PUBLIC",
        warehouse="WH",
        role="DISCRETE_ROLE",
        auth={"user": "reader"},
    )
    # Same effective identity → same id.
    assert a.canonicalize_connection_id(spec) == a.canonicalize_connection_id(same_explicit)


def test_probe_freshness_uses_current_schema_for_unqualified_tables():
    """Unqualified tables resolve via ``CURRENT_SCHEMA()``, not a hard-coded ``PUBLIC``."""
    cur = _FakeCursor(
        [
            ("CURRENT_DATABASE()", ("EVENTS", "ANALYTICS")),
            ("INFORMATION_SCHEMA.TABLES", ("ANALYTICS", "orders", "ts")),
        ]
    )
    a = SnowflakeAdapter()
    # Unqualified table, which relies on CURRENT_SCHEMA().
    tables = [QualifiedTable(catalog=None, schema=None, name="orders")]
    a.probe_freshness(_FakeConn(cur), tables)

    # Find the INFORMATION_SCHEMA query and its schema bind value.
    info_calls = [(sql, params) for sql, params in cur.executions if "INFORMATION_SCHEMA" in sql]
    assert len(info_calls) == 1
    _, params = info_calls[0]
    # First param is the schema; should be ANALYTICS, not PUBLIC.
    assert params[0] == "ANALYTICS", (
        f"probe should resolve unqualified tables against CURRENT_SCHEMA, got {params[0]!r}"
    )


def test_probe_freshness_no_schema_does_not_pretend_to_match_public():
    """With no schema anywhere the probe folds a sentinel and skips, rather than assuming PUBLIC."""
    cur = _FakeCursor(
        [
            # CURRENT_SCHEMA returns null (no default schema).
            ("CURRENT_DATABASE()", ("EVENTS", None)),
        ]
    )
    a = SnowflakeAdapter()
    tables = [QualifiedTable(catalog=None, schema=None, name="orders")]
    token = a.probe_freshness(_FakeConn(cur), tables)
    assert token.value  # non-empty (sentinel)
    info_calls = [sql for sql, _ in cur.executions if "INFORMATION_SCHEMA" in sql]
    # No INFORMATION_SCHEMA query was issued: sentinel only.
    assert info_calls == []


def test_open_uses_write_role_when_read_only_false():
    """Read cells apply ``role`` and write cells apply ``write_role``, so ``# @sql write=true``
    means something.
    """
    cur = _FakeCursor([])
    a = SnowflakeAdapter(connect_fn=lambda _uri: _FakeConn(cur))
    spec = ConnectionSpec(
        name="x",
        driver="snowflake",
        account="ACME",
        role="ANALYTICS_RO",
        write_role="ANALYTICS_RW",
    )

    a.open(spec, read_only=True)
    assert [sql for sql, _ in cur.executions] == ['USE ROLE "ANALYTICS_RO"']

    cur2 = _FakeCursor([])
    a2 = SnowflakeAdapter(connect_fn=lambda _uri: _FakeConn(cur2))
    a2.open(spec, read_only=False)
    assert [sql for sql, _ in cur2.executions] == ['USE ROLE "ANALYTICS_RW"']


def test_open_falls_back_to_role_when_write_role_unset():
    """Without ``write_role``, write cells use ``role``; warehouse policy decides if the write
    succeeds.
    """
    cur = _FakeCursor([])
    a = SnowflakeAdapter(connect_fn=lambda _uri: _FakeConn(cur))
    spec = ConnectionSpec(
        name="x",
        driver="snowflake",
        account="ACME",
        role="ANALYTICS_GENERAL",
    )

    a.open(spec, read_only=False)
    assert [sql for sql, _ in cur.executions] == ['USE ROLE "ANALYTICS_GENERAL"']


def test_connection_id_write_role_only_in_write_identity():
    """Read cells never apply ``write_role``, so it must not churn their cache; write cells do."""
    a = SnowflakeAdapter()
    base = ConnectionSpec(
        name="x",
        driver="snowflake",
        account="ACME",
        role="RO",
    )
    with_write = base.model_copy(update={"write_role": "RW"})

    # Read identity: ignores write_role, so swapping it leaves
    # the read connection_id unchanged.
    assert a.canonicalize_connection_id(base, read_only=True) == a.canonicalize_connection_id(
        with_write, read_only=True
    )

    # Write identity: write_role joins, so swapping invalidates.
    assert a.canonicalize_connection_id(base, read_only=False) != a.canonicalize_connection_id(
        with_write, read_only=False
    )
