"""Tests for the SQL dialect seam."""

from __future__ import annotations

import struct
from pathlib import Path

from strata.sql_backend import (
    PostgresDialect,
    SqliteDialect,
    advisory_lock_id,
    translate_placeholders,
)


class TestTranslatePlaceholders:
    """Qmark to pyformat, without corrupting data that looks like syntax."""

    def test_bare_placeholders_convert(self):
        assert (
            translate_placeholders("SELECT * FROM t WHERE id = ? AND v = ?")
            == "SELECT * FROM t WHERE id = %s AND v = %s"
        )

    def test_statement_without_placeholders_is_unchanged(self):
        sql = "SELECT COUNT(*) FROM artifact_versions"
        assert translate_placeholders(sql) == sql

    def test_question_mark_inside_a_string_literal_is_data(self):
        assert (
            translate_placeholders("SELECT * FROM t WHERE name = 'why?' AND id = ?")
            == "SELECT * FROM t WHERE name = 'why?' AND id = %s"
        )

    def test_question_mark_inside_a_quoted_identifier_is_data(self):
        assert (
            translate_placeholders('SELECT "odd?col" FROM t WHERE id = ?')
            == 'SELECT "odd?col" FROM t WHERE id = %s'
        )

    def test_doubled_quote_does_not_end_the_literal(self):
        # 'it''s ?' is one literal containing an apostrophe, so its question
        # mark stays data. Treating '' as a close would convert it.
        assert (
            translate_placeholders("SELECT * FROM t WHERE s = 'it''s ?' AND id = ?")
            == "SELECT * FROM t WHERE s = 'it''s ?' AND id = %s"
        )

    def test_percent_is_escaped_outside_literals(self):
        assert translate_placeholders("SELECT 5 % 2") == "SELECT 5 %% 2"

    def test_percent_is_escaped_inside_literals(self):
        # The driver scans the whole statement, so a percent is a substitution
        # marker even inside quotes.
        assert (
            translate_placeholders("SELECT * FROM t WHERE n LIKE 'nb_%'")
            == "SELECT * FROM t WHERE n LIKE 'nb_%%'"
        )

    def test_line_comment_contents_are_not_translated(self):
        sql = "SELECT 1 -- is this a ? placeholder\nWHERE id = ?"
        assert translate_placeholders(sql) == "SELECT 1 -- is this a ? placeholder\nWHERE id = %s"

    def test_block_comment_contents_are_not_translated(self):
        sql = "SELECT /* a ? here */ 1 WHERE id = ?"
        assert translate_placeholders(sql) == "SELECT /* a ? here */ 1 WHERE id = %s"

    def test_unterminated_literal_does_not_raise(self):
        # A translator should not be the thing that reports a syntax error --
        # the driver gives a far better message. It just must not crash.
        assert translate_placeholders("SELECT 'oops") == "SELECT 'oops"

    def test_real_statement_from_the_store(self):
        # artifact_store.py:1231, the one place the store mixes a placeholder
        # with a quoted literal in the same statement.
        sql = "WHERE id LIKE ? ESCAPE '\\' AND state = 'ready'"
        assert translate_placeholders(sql) == "WHERE id LIKE %s ESCAPE '\\' AND state = 'ready'"

    def test_multiline_ddl(self):
        sql = """
            INSERT INTO artifact_versions
                (id, version, state, provenance_hash)
            VALUES (?, ?, 'building', ?)
        """
        translated = translate_placeholders(sql)
        assert translated.count("%s") == 3
        # The inline state literal is data and must survive intact.
        assert "'building'" in translated


class TestSqliteDialectPreservesTodaysBehavior:
    """Personal mode must not notice the seam exists."""

    def test_adapt_ddl_is_identity(self):
        # The schema is already written in this dialect; rewriting it would be
        # churn.
        ddl = "CREATE TABLE t (seq INTEGER PRIMARY KEY AUTOINCREMENT, at REAL)"
        assert SqliteDialect(Path("x.sqlite")).adapt_ddl(ddl) == ddl

    def test_connect_sets_wal_and_a_name_indexable_row_factory(self, tmp_path):
        dialect = SqliteDialect(tmp_path / "artifacts.sqlite")
        conn = dialect.connect()
        try:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
            conn.execute("CREATE TABLE t (tenant TEXT)")
            conn.execute("INSERT INTO t VALUES ('acme')")
            # The store reads columns by name throughout.
            assert conn.execute("SELECT tenant FROM t").fetchone()["tenant"] == "acme"
        finally:
            conn.close()

    def test_begin_write_serializes_writers(self, tmp_path):
        dialect = SqliteDialect(tmp_path / "artifacts.sqlite")
        conn = dialect.connect()
        try:
            # SQLite locks the whole file, so the key is accepted and ignored.
            dialect.begin_write(conn, "a1")
            assert conn.in_transaction
        finally:
            conn.rollback()
            conn.close()

    def test_legacy_migration_is_supported(self):
        # SQLite is the only backend with deployed history to upgrade.
        assert SqliteDialect(Path("x.sqlite")).supports_legacy_migration is True


class _FakeInner:
    """Records what the wrapper would send to psycopg.

    Lets the connection adapter be tested without a server, so these stay in
    the main CI job rather than the container-backed integration one.
    """

    def __init__(self):
        self.executed: list[tuple[str, tuple]] = []

    def execute(self, sql, params=()):
        self.executed.append((sql, tuple(params)))
        return self


class TestPostgresConnectionAdapter:
    """The wrapper is what keeps the store's 103 call sites unchanged."""

    def _wrap(self):
        from strata.sql_backend import _PostgresConnection

        inner = _FakeInner()
        return _PostgresConnection(inner), inner

    def test_execute_translates_placeholders(self):
        conn, inner = self._wrap()
        conn.execute("SELECT * FROM t WHERE id = ?", ("a1",))
        assert inner.executed == [("SELECT * FROM t WHERE id = %s", ("a1",))]

    def test_execute_accepts_a_list_of_params(self):
        # The store builds params as a list where filters are optional.
        conn, inner = self._wrap()
        conn.execute("SELECT ? , ?", ["a", "b"])
        assert inner.executed[0][1] == ("a", "b")

    def test_executescript_sends_ddl_unchanged(self):
        # Multi-statement schema DDL, no parameters, no percent signs.
        conn, inner = self._wrap()
        conn.executescript("CREATE TABLE a (x TEXT); CREATE TABLE b (y TEXT);")
        assert inner.executed == [("CREATE TABLE a (x TEXT); CREATE TABLE b (y TEXT);", ())]

    def test_begin_write_takes_an_advisory_lock_on_the_key(self):
        conn, inner = self._wrap()
        PostgresDialect("postgresql:///x").begin_write(conn, "a1")
        sql, params = inner.executed[0]
        assert "pg_advisory_xact_lock" in sql
        assert params == (advisory_lock_id("a1"),)


class TestRowBehavesLikeSqliteRow:
    """The store reads results by name, by position, and via dict()."""

    def _row(self):
        from strata.sql_backend import _Row

        return _Row(("id", "tenant"), ("a1", "acme"))

    def test_indexable_by_name(self):
        assert self._row()["tenant"] == "acme"

    def test_indexable_by_position(self):
        # create_artifact reads `cursor.fetchone()[0]`.
        assert self._row()[0] == "a1"

    def test_convertible_to_dict(self):
        # read_audit and list_pending_changes do `dict(row)`.
        assert dict(self._row()) == {"id": "a1", "tenant": "acme"}


class TestPostgresDialectRendering:
    """What SQL to emit."""

    def test_adapt_ddl_rewrites_the_types_that_differ(self):
        ddl = "CREATE TABLE t (seq INTEGER PRIMARY KEY AUTOINCREMENT, at REAL NOT NULL)"
        adapted = PostgresDialect("postgresql:///x").adapt_ddl(ddl)
        assert "BIGSERIAL PRIMARY KEY" in adapted
        assert "at DOUBLE PRECISION NOT NULL" in adapted
        assert "AUTOINCREMENT" not in adapted

    def test_adapt_ddl_leaves_other_types_alone(self):
        ddl = "CREATE TABLE t (id TEXT NOT NULL, version INTEGER NOT NULL)"
        assert PostgresDialect("postgresql:///x").adapt_ddl(ddl) == ddl

    def test_autoincrement_differs_from_sqlite(self):
        assert PostgresDialect("postgresql:///x").autoincrement_pk == "BIGSERIAL PRIMARY KEY"
        assert SqliteDialect(Path("x")).autoincrement_pk == "INTEGER PRIMARY KEY AUTOINCREMENT"

    def test_float_type_is_double_precision(self):
        # Not REAL: see test_real_would_lose_timestamp_precision below.
        assert PostgresDialect("postgresql:///x").float_type == "DOUBLE PRECISION"

    def test_real_would_lose_timestamp_precision(self):
        # Why float_type exists. Postgres REAL is single precision, and the
        # store keeps epoch seconds in these columns. Round-tripping a
        # realistic created_at through 32 bits loses the sub-second part
        # entirely -- silently, and only visible much later as artifacts that
        # appear to have been created at the same instant.
        created_at = 1787000000.123456
        through_single = struct.unpack("f", struct.pack("f", created_at))[0]
        assert through_single != created_at
        assert abs(through_single - created_at) > 1.0

    def test_legacy_migration_is_not_supported(self):
        # No deployed Postgres history exists, so the sqlite_master / PRAGMA /
        # rowid migration path never runs there.
        assert PostgresDialect("postgresql:///x").supports_legacy_migration is False

    def test_lock_ids_are_stable_and_distinct(self):
        # Recomputed at every call site, so drift would silently stop
        # serializing the writers the lock exists to serialize.
        assert advisory_lock_id("a1") == advisory_lock_id("a1")
        assert advisory_lock_id("a1") != advisory_lock_id("a2")
        assert -(2**63) <= advisory_lock_id("a1") < 2**63
