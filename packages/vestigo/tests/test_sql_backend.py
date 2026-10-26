"""Tests for the SQL dialect seam."""

from __future__ import annotations

import gc
import logging
import os
import threading
from pathlib import Path

import pytest

from vestigo import sql_backend
from vestigo.sql_backend import (
    SERVER_THREAD_LIMIT,
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
        # 'it''s ?' is one literal containing an apostrophe, so its question mark stays
        # data. Treating '' as a close would convert it.
        assert (
            translate_placeholders("SELECT * FROM t WHERE s = 'it''s ?' AND id = ?")
            == "SELECT * FROM t WHERE s = 'it''s ?' AND id = %s"
        )

    def test_percent_is_escaped_outside_literals(self):
        assert translate_placeholders("SELECT 5 % 2") == "SELECT 5 %% 2"

    def test_percent_is_escaped_inside_literals(self):
        # The driver scans the whole statement, so a percent is a substitution marker even
        # inside quotes.
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
        # The driver gives a far better syntax error; the translator just must not crash.
        assert translate_placeholders("SELECT 'oops") == "SELECT 'oops"

    def test_real_statement_from_the_store(self):
        # The one place the store mixes a placeholder with a quoted literal in a statement.
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
        # The schema is already written in this dialect.
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
    """Records what the wrapper would send to psycopg, so no server is needed."""

    def __init__(self):
        self.executed: list[tuple[str, tuple]] = []
        self.statusmessage = None

    def execute(self, sql, params=()):
        self.executed.append((sql, tuple(params)))
        # The command tag psycopg exposes on the cursor, from the statement's verb.
        self.statusmessage = sql.split(None, 1)[0].upper()
        return self

    def commit(self):
        return None

    def rollback(self):
        return None


class TestPostgresConnectionAdapter:
    """The wrapper keeps the store's call sites unchanged."""

    def _wrap(self):
        from vestigo.sql_backend import _Lease, _PostgresConnection

        inner = _FakeInner()
        released = []
        lease = _Lease(inner)
        return _PostgresConnection(lease, lambda lease: released.append(1)), inner, released

    def test_execute_translates_placeholders(self):
        conn, inner, _ = self._wrap()
        conn.execute("SELECT * FROM t WHERE id = ?", ("a1",))
        assert inner.executed == [("SELECT * FROM t WHERE id = %s", ("a1",))]

    def test_execute_accepts_a_list_of_params(self):
        # The store builds params as a list where filters are optional.
        conn, inner, _ = self._wrap()
        conn.execute("SELECT ? , ?", ["a", "b"])
        assert inner.executed[0][1] == ("a", "b")

    def test_executescript_sends_ddl_unchanged(self):
        # Multi-statement schema DDL, no parameters, no percent signs.
        conn, inner, _ = self._wrap()
        conn.executescript("CREATE TABLE a (x TEXT); CREATE TABLE b (y TEXT);")
        assert inner.executed == [("CREATE TABLE a (x TEXT); CREATE TABLE b (y TEXT);", ())]

    def test_begin_write_takes_an_advisory_lock_on_the_key(self):
        conn, inner, _ = self._wrap()
        with conn:
            PostgresDialect("postgresql:///x").begin_write(conn, "a1")
        sql, params = inner.executed[0]
        assert "pg_advisory_xact_lock" in sql
        assert params == (advisory_lock_id("a1"),)

    def test_close_returns_the_connection_instead_of_closing_it(self):
        # The store calls close() to mean "done with this"; under a pool that must mean
        # "give it back", or the pool drains to nothing.
        conn, inner, released = self._wrap()
        conn.close()
        assert released == [1]
        assert not hasattr(inner, "closed")

    def test_double_close_releases_once(self):
        # A second release would hand the same connection to two threads.
        conn, _, released = self._wrap()
        conn.close()
        conn.close()
        assert released == [1]

    def test_the_context_manager_releases_on_exit(self):
        conn, _, released = self._wrap()
        with conn as entered:
            assert entered is conn
            assert released == []
        assert released == [1]

    def test_a_dropped_wrapper_releases_itself_with_a_warning(self, caplog):
        # The ledger closes in a finally everywhere; this is for a consumer that does not.
        conn, _, released = self._wrap()
        with caplog.at_level(logging.WARNING, logger="vestigo.sql_backend"):
            del conn
            gc.collect()
        assert released == [1]
        assert any("without close()" in r.getMessage() for r in caplog.records)

    def test_writes_mark_the_lease_dirty_until_commit_or_rollback(self):
        # Decided by the server's command tag: the advisory lock is a SELECT and not a write.
        conn, _, _ = self._wrap()
        with conn:
            conn.execute("SELECT pg_advisory_xact_lock(?)", (1,))
            assert conn._lease.dirty is False
            conn.execute("INSERT INTO t VALUES (?)", (1,))
            assert conn._lease.dirty is True
            conn.commit()
            assert conn._lease.dirty is False
            conn.executescript("CREATE TABLE a (x TEXT);")
            assert conn._lease.dirty is True
            conn.rollback()
            assert conn._lease.dirty is False


class TestRowBehavesLikeSqliteRow:
    """The store reads results by name, by position, and via dict()."""

    def _row(self):
        from vestigo.sql_backend import _Row

        return _Row(("id", "tenant"), ("a1", "acme"))

    def test_indexable_by_name(self):
        assert self._row()["tenant"] == "acme"

    def test_indexable_by_position(self):
        # create_artifact reads `cursor.fetchone()[0]`.
        assert self._row()[0] == "a1"

    def test_convertible_to_dict(self):
        # read_audit and list_pending_changes do `dict(row)`.
        assert dict(self._row()) == {"id": "a1", "tenant": "acme"}

    def test_get_returns_the_default_for_a_missing_column(self):
        # Mapping's mixin get() catches KeyError only, so a raw ValueError out
        # of list.index would propagate instead of returning the default.
        assert self._row().get("principal", "fallback") == "fallback"

    def test_membership_test_does_not_raise(self):
        assert "principal" not in self._row()
        assert "tenant" in self._row()


class TestPostgresDialectRendering:
    """What SQL to emit."""

    def test_adapt_ddl_rewrites_the_types_that_differ(self):
        ddl = (
            "CREATE TABLE t (seq INTEGER PRIMARY KEY AUTOINCREMENT, "
            "at REAL NOT NULL, byte_size INTEGER)"
        )
        adapted = PostgresDialect("postgresql:///x").adapt_ddl(ddl)
        assert "BIGSERIAL PRIMARY KEY" in adapted
        assert "at DOUBLE PRECISION NOT NULL" in adapted
        assert "byte_size BIGINT" in adapted
        assert "AUTOINCREMENT" not in adapted

    def test_autoincrement_is_rewritten_before_bare_integer(self):
        # Order matters: the autoincrement rule must consume its own INTEGER first, or the
        # bare-INTEGER rule makes "BIGINT PRIMARY KEY AUTOINCREMENT" and the rule never matches.
        adapted = PostgresDialect("postgresql:///x").adapt_ddl(
            "CREATE TABLE t (seq INTEGER PRIMARY KEY AUTOINCREMENT)"
        )
        assert adapted == "CREATE TABLE t (seq BIGSERIAL PRIMARY KEY)"

    def test_adapt_ddl_leaves_portable_types_alone(self):
        ddl = "CREATE TABLE t (id TEXT NOT NULL, tenant TEXT DEFAULT '')"
        assert PostgresDialect("postgresql:///x").adapt_ddl(ddl) == ddl

    def test_integer_widens_to_bigint(self):
        # Postgres INTEGER is int4 (max 2147483647), smaller than byte_size for any artifact
        # of 2 GiB or more.
        assert PostgresDialect("postgresql:///x").integer_type == "BIGINT"
        assert SqliteDialect(Path("x")).integer_type == "INTEGER"

    def test_autoincrement_differs_from_sqlite(self):
        assert PostgresDialect("postgresql:///x").autoincrement_pk == "BIGSERIAL PRIMARY KEY"
        assert SqliteDialect(Path("x")).autoincrement_pk == "INTEGER PRIMARY KEY AUTOINCREMENT"

    def test_float_type_is_double_precision(self):
        # Not REAL, which is single precision and drops a timestamp's sub-second part;
        # TestTimestampPrecision in the Postgres suite checks that against a live server.
        assert PostgresDialect("postgresql:///x").float_type == "DOUBLE PRECISION"

    def test_legacy_migration_is_not_supported(self):
        # No deployed Postgres history exists, so the sqlite_master / PRAGMA /
        # rowid migration path never runs there.
        assert PostgresDialect("postgresql:///x").supports_legacy_migration is False

    def test_lock_ids_are_stable_and_distinct(self):
        # Recomputed at every call site, so drift would silently stop serializing the
        # writers the lock exists for.
        assert advisory_lock_id("a1") == advisory_lock_id("a1")
        assert advisory_lock_id("a1") != advisory_lock_id("a2")
        assert -(2**63) <= advisory_lock_id("a1") < 2**63


class TestThreadLimitAndPoolAreOneNumber:
    """The pool holds a connection for every thread that may call the store at once."""

    def test_the_pool_covers_every_thread_token_and_the_event_loop(self):
        assert PostgresDialect("postgresql:///x").max_size == SERVER_THREAD_LIMIT + 1


def _docker_daemon_reachable() -> bool:
    try:
        import docker

        docker.from_env().ping()
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def postgres_dsn():
    """A live Postgres for the pooled-connection tests; skipped without Docker."""
    if not _docker_daemon_reachable():
        if os.environ.get("VESTIGO_REQUIRE_DOCKER"):
            raise RuntimeError("VESTIGO_REQUIRE_DOCKER is set and the Docker daemon is unreachable")
        pytest.skip("Docker daemon is not running")
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer("postgres:18-alpine") as container:
        yield container.get_connection_url().replace("postgresql+psycopg2://", "postgresql://")


@pytest.fixture
def dialect(postgres_dsn):
    dialect = PostgresDialect(postgres_dsn)
    with dialect.connect() as conn:
        conn.executescript("DROP TABLE IF EXISTS t; CREATE TABLE t (x INT)")
        conn.commit()
    yield dialect
    dialect.close()


def _count(dialect) -> int:
    with dialect.connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM t").fetchone()[0]


@pytest.mark.integration
@pytest.mark.slow
class TestPooledConnectionLifecycle:
    """What a thread's pooled connection does across wrappers, threads and failures."""

    def test_a_wrapper_dropped_after_a_failed_statement_does_not_poison_the_thread(
        self, dialect, caplog
    ):
        import psycopg

        def forgets_to_close() -> None:
            conn = dialect.connect()
            try:
                conn.execute("SELECT * FROM no_such_table")
            except psycopg.Error:
                return

        with caplog.at_level(logging.WARNING, logger="vestigo.sql_backend"):
            forgets_to_close()
            gc.collect()

        assert any("without close()" in r.getMessage() for r in caplog.records)
        # The next acquisition on this thread is a clean connection, not the failed one.
        with dialect.connect() as again:
            assert again.execute("SELECT 1").fetchone()[0] == 1

    def test_a_nested_connect_on_a_failed_transaction_is_refused(self, dialect):
        import psycopg

        outer = dialect.connect()
        try:
            with pytest.raises(psycopg.Error):
                outer.execute("SELECT * FROM no_such_table")
            with pytest.raises(RuntimeError, match="failed transaction"):
                dialect.connect()
            outer.rollback()
            with dialect.connect() as inner:
                assert inner.execute("SELECT 1").fetchone()[0] == 1
        finally:
            outer.close()

    def test_a_nested_connect_over_uncommitted_writes_is_refused(self, dialect):
        # Joined, the inner's rollback would discard the outer's INSERT and its commit would
        # publish it; refused, the outer's own decision stands.
        outer = dialect.connect()
        try:
            outer.execute("INSERT INTO t VALUES (1)")
            with pytest.raises(RuntimeError, match="uncommitted writes"):
                dialect.connect()
            outer.commit()
            # The store nests reads after its commit; those stay allowed.
            with dialect.connect() as inner:
                inner.execute("SELECT 1").fetchone()
                inner.rollback()
            outer.execute("INSERT INTO t VALUES (2)")
            outer.rollback()
        finally:
            outer.close()
        assert _count(dialect) == 1

    def test_a_nested_connect_after_reads_shares_the_connection(self, dialect):
        outer = dialect.connect()
        try:
            outer.execute("SELECT 1").fetchone()
            dialect.begin_write(outer, "a1")
            inner = dialect.connect()
            try:
                assert inner._inner is outer._inner
            finally:
                inner.close()
            # The inner's close keeps the outer's connection checked out.
            assert dialect._local.lease.depth == 1
        finally:
            outer.close()
        assert dialect._local.lease is None

    def test_a_close_from_another_thread_returns_the_connection(self, postgres_dsn, monkeypatch):
        # Released through the record captured at connect, not the closing thread's state.
        monkeypatch.setattr(sql_backend, "_POOL_TIMEOUT_SECONDS", 2.0)
        dialect = PostgresDialect(postgres_dsn, max_size=1)
        try:
            box: dict[str, object] = {}
            thread = threading.Thread(target=lambda: box.__setitem__("conn", dialect.connect()))
            thread.start()
            thread.join()
            box["conn"].close()
            with dialect.connect() as conn:
                assert conn.execute("SELECT 1").fetchone()[0] == 1
        finally:
            dialect.close()

    def test_a_connection_the_server_dropped_is_not_handed_out(self, dialect, postgres_dsn):
        # After a restart or failover every idle pooled connection is dead; the pool checks
        # before handing one out, so no store call fails for it.
        import psycopg

        with dialect.connect() as conn:
            pid = conn.execute("SELECT pg_backend_pid()").fetchone()[0]
        with psycopg.connect(postgres_dsn, autocommit=True) as killer:
            assert killer.execute("SELECT pg_terminate_backend(%s)", (pid,)).fetchone()[0]
        with dialect.connect() as conn:
            assert conn.execute("SELECT pg_backend_pid()").fetchone()[0] != pid
