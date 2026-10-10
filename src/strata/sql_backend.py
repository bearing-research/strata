"""SQL dialect seam for the artifact store: SQLite (personal mode) or Postgres.

What differs between the two lives here, so ``artifact_store.py`` keeps one
copy of its SQL: placeholder style (``?`` vs ``%s``), autoincrement, numeric
widths (Postgres ``REAL`` is single precision and ``INTEGER`` 32-bit, which
would truncate timestamps and overflow ``byte_size`` at 2 GiB), writer
serialization (:meth:`SqlDialect.begin_write`), and the integrity-error type.
The legacy-migration path (``PRAGMA``, ``sqlite_master``, ``rowid``) runs only on SQLite.

``LIKE`` is deliberately not unified: case-insensitive for ASCII on SQLite,
case-sensitive on Postgres. Callers needing case-insensitive prefix matching
should normalize explicitly. :class:`SqliteDialect` reproduces the store's
original behavior exactly.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

# Mirrors ``sqlite3.connect(timeout=30.0)``: a contended writer gives up, not blocks forever.
_LOCK_TIMEOUT_MS = 30_000
_CONNECT_TIMEOUT_SECONDS = 10

# Worker threads that may call the store at once: the server sets anyio's default thread
# limiter to this, and every store offload and sync route borrows a token from it.
SERVER_THREAD_LIMIT = 40

# max_size is per process: one connection per thread token, plus the event loop thread, which
# still calls the store inline at startup, at shutdown and from notebook sessions. A thread then
# never waits for a connection, only for a token.
# The store nests acquisition two deep; re-entrant sharing (``PostgresDialect.connect``) keeps
# that at one connection per thread.
_POOL_MIN_SIZE = 0
_POOL_MAX_SIZE = SERVER_THREAD_LIMIT + 1
# Without it an exhausted pool blocks forever.
_POOL_TIMEOUT_SECONDS = 30.0


logger = logging.getLogger(__name__)

# Characters that open a region where ``?`` is data, not a placeholder.
_SINGLE_QUOTE = "'"
_DOUBLE_QUOTE = '"'

# Column-type tokens that differ by engine; applied only to this package's own DDL.
# Order matters: the autoincrement rule must consume its INTEGER before the bare-INTEGER one.
_DDL_TYPE_REWRITES = (
    ("INTEGER PRIMARY KEY AUTOINCREMENT", "autoincrement_pk"),
    (r"\bREAL\b", "float_type"),
    (r"\bINTEGER\b", "integer_type"),
)


def translate_placeholders(sql: str) -> str:
    """Rewrite qmark ``?`` placeholders to the pyformat ``%s`` style Postgres drivers expect.

    Placeholders convert only outside single-quoted literals, double-quoted
    identifiers and ``--``/``/* */`` comments, where a ``?`` is data. Every ``%`` is
    doubled, because the driver scans the whole string for substitutions.
    """
    out: list[str] = []
    i = 0
    n = len(sql)

    while i < n:
        ch = sql[i]

        if ch == _SINGLE_QUOTE or ch == _DOUBLE_QUOTE:
            end = _scan_quoted(sql, i, ch)
            out.append(sql[i:end].replace("%", "%%"))
            i = end
        elif sql.startswith("--", i):
            end = sql.find("\n", i)
            end = n if end == -1 else end
            out.append(sql[i:end].replace("%", "%%"))
            i = end
        elif sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            end = n if end == -1 else end + 2
            out.append(sql[i:end].replace("%", "%%"))
            i = end
        elif ch == "?":
            out.append("%s")
            i += 1
        elif ch == "%":
            out.append("%%")
            i += 1
        else:
            out.append(ch)
            i += 1

    return "".join(out)


def _scan_quoted(sql: str, start: int, quote: str) -> int:
    """Return the index just past the quoted region opening at ``start``.

    A doubled quote is an escape, not a terminator. An unterminated region runs to
    the end, leaving the syntax error for the driver to report.
    """
    i = start + 1
    n = len(sql)
    while i < n:
        if sql[i] == quote:
            if i + 1 < n and sql[i + 1] == quote:
                i += 2
                continue
            return i + 1
        i += 1
    return n


def advisory_lock_id(key: str) -> int:
    """Map a lock key to the signed 64-bit integer Postgres advisory locks use.

    Computed here rather than with ``hashtext()``, whose output is not stable
    across Postgres major versions; a changed id would silently stop serializing writers.
    """
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


class StoreConnection(Protocol):
    """The connection surface ``artifact_store`` uses, shaped like ``sqlite3.Connection``.

    :class:`_PostgresConnection` adapts psycopg to it, so the store's call sites stay unchanged.
    """

    # Positional-only so sqlite3.Connection matches the protocol.
    def execute(self, sql: str, params: Sequence[Any] = (), /) -> Any: ...

    def executescript(self, sql: str, /) -> Any: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


class SqlDialect(Protocol):
    """The parts of ``artifact_store`` that differ between databases.

    Each member has a concrete divergence behind it; everything else is portable SQL.
    """

    name: str

    def connect(self) -> StoreConnection:
        """Open a connection configured the way the store expects.

        Rows must be indexable by both column name and position.
        """
        ...

    def adapt_ddl(self, sql: str) -> str:
        """Rewrite column types in this package's own schema DDL."""
        ...

    @property
    def autoincrement_pk(self) -> str:
        """Column type for a synthetic, monotonically increasing primary key."""
        ...

    @property
    def float_type(self) -> str:
        """Column type for a 64-bit float (``REAL`` is single precision on Postgres)."""
        ...

    @property
    def integer_type(self) -> str:
        """Column type for a 64-bit integer (a 2 GiB ``byte_size`` overflows 32 bits)."""
        ...

    @property
    def integrity_error(self) -> type[Exception]:
        """Exception raised when a constraint is violated.

        The store catches it to make finalize idempotent, so it must be the driver's own type.
        """
        ...

    @property
    def operational_error(self) -> type[Exception]:
        """Exception raised when the database cannot take a write right now.

        Caught where a write is advisory (recording use), so a read never fails for want of it.
        """
        ...

    @property
    def rejectable_errors(self) -> tuple[type[Exception], ...]:
        """Errors a bulk copy attributes to one row rather than the whole run.

        Constraint violations, plus values a stricter engine rejects outright (e.g. a
        NUL byte in TEXT, reachable via executor logs or tag values).
        """
        ...

    @property
    def supports_legacy_migration(self) -> bool:
        """Whether pre-tenant-column databases can exist for this backend (only SQLite).

        Keeps ``sqlite_master``/``PRAGMA``/``rowid`` out of the Postgres path.
        """
        ...

    def column_exists(self, conn: StoreConnection, table: str, column: str) -> bool:
        """Whether ``table`` has ``column``.

        Migrations ask first because ``ADD COLUMN IF NOT EXISTS`` is not portable to SQLite.
        """
        ...

    def schema_exists(self, conn: StoreConnection, table: str = "artifact_versions") -> bool:
        """Whether ``table`` is already present.

        Lets ``_init_schema`` skip the global schema lock once the schema exists.
        """
        ...

    def close(self) -> None:
        """Release any resources the dialect holds. Idempotent; a no-op on SQLite."""
        ...

    def resync_autoincrement(self, conn: StoreConnection, table: str, column: str) -> None:
        """Point a synthetic key's generator past the largest existing value.

        Needed after inserts with explicit ids (migration): a Postgres ``BIGSERIAL``
        sequence does not advance on them and would collide. SQLite needs nothing.
        """
        ...

    def begin_write(self, conn: StoreConnection, key: str) -> None:
        """Open a transaction that serializes writers contending on ``key``.

        Used where a read-modify-write must not race: ``create_artifact`` computing
        ``MAX(version)+1``, and ``finalize_canonical_together``. ``key`` lets a backend
        lock narrowly rather than globally.
        """
        ...


class SqliteDialect:
    """SQLite dialect: the store's PRAGMAs, row factory, timeout and ``BEGIN IMMEDIATE``."""

    name = "sqlite"

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def connect(self) -> StoreConnection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def adapt_ddl(self, sql: str) -> str:
        return sql

    @property
    def autoincrement_pk(self) -> str:
        return "INTEGER PRIMARY KEY AUTOINCREMENT"

    @property
    def float_type(self) -> str:
        # SQLite REAL is an IEEE 754 double.
        return "REAL"

    @property
    def integer_type(self) -> str:
        # SQLite INTEGER widens to 8 bytes as needed.
        return "INTEGER"

    @property
    def integrity_error(self) -> type[Exception]:
        return sqlite3.IntegrityError

    @property
    def operational_error(self) -> type[Exception]:
        return sqlite3.OperationalError

    @property
    def rejectable_errors(self) -> tuple[type[Exception], ...]:
        return (sqlite3.IntegrityError, sqlite3.DataError)

    @property
    def supports_legacy_migration(self) -> bool:
        return True

    def column_exists(self, conn: StoreConnection, table: str, column: str) -> bool:
        # PRAGMA takes no bind parameters; every caller passes a literal table name.
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        return any(row["name"] == column for row in rows)

    def schema_exists(self, conn: StoreConnection, table: str = "artifact_versions") -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (table,)
        ).fetchone()
        return row is not None

    def close(self) -> None:
        # Nothing pooled: callers close their own connections.
        return

    def resync_autoincrement(self, conn: StoreConnection, table: str, column: str) -> None:
        # SQLite derives the next rowid from the table; no sequence to drift.
        return

    def begin_write(self, conn: StoreConnection, key: str) -> None:
        # SQLite locks the whole file, so the key is unused.
        conn.execute("BEGIN IMMEDIATE")


class _Row(Mapping):
    """A psycopg row that behaves like ``sqlite3.Row``.

    Supports both ``row["col"]`` and ``row[0]``; implementing ``Mapping`` makes ``dict(row)`` work.
    """

    __slots__ = ("_columns", "_values")

    def __init__(self, columns: tuple[str, ...], values: tuple[Any, ...]) -> None:
        self._columns = columns
        self._values = values

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._values[key]
        try:
            return self._values[self._columns.index(key)]
        except ValueError:
            # Mapping's get() and __contains__ catch only KeyError.
            raise KeyError(key) from None

    def __iter__(self):
        return iter(self._columns)

    def __len__(self) -> int:
        return len(self._columns)

    def __repr__(self) -> str:
        return f"_Row({dict(zip(self._columns, self._values, strict=True))!r})"


def _row_factory(cursor: Any) -> Any:
    """psycopg row factory producing :class:`_Row`."""
    description = cursor.description
    if description is None:
        return lambda values: values
    columns = tuple(column.name for column in description)
    return lambda values: _Row(columns, tuple(values))


class _PostgresConnection:
    """Adapts a psycopg connection to the surface the store expects.

    ``execute`` translates qmark to pyformat; ``executescript`` (absent in psycopg)
    runs the store's own multi-statement DDL.
    """

    def __init__(self, inner: Any, release: Any) -> None:
        self._inner = inner
        self._release = release
        self._closed = False

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        return self._inner.execute(translate_placeholders(sql), params)

    def executescript(self, sql: str) -> Any:
        # No parameters, so skip translate_placeholders' percent-escaping.
        return self._inner.execute(sql)

    def commit(self) -> None:
        self._inner.commit()

    def rollback(self) -> None:
        self._inner.rollback()

    def close(self) -> None:
        """Return the connection to the pool rather than closing it.

        Guarded against a double release, which would hand one connection to two threads.
        """
        if self._closed:
            return
        self._closed = True
        self._release()


class PostgresDialect:
    """Postgres backing for the artifact store (requires the ``postgres`` extra)."""

    name = "postgres"

    def __init__(self, dsn: str, *, max_size: int = _POOL_MAX_SIZE) -> None:
        self.dsn = dsn
        self.max_size = max_size
        self._pool: Any = None
        self._pool_lock = threading.Lock()
        self._closed = False
        # Per-thread (connection, depth) for re-entrant acquisition.
        self._local = threading.local()

    def _get_pool(self) -> Any:
        """Build the pool on first use, so constructing a dialect opens no sockets."""
        if self._closed:
            raise RuntimeError(
                "PostgresDialect is closed. Reopening on demand would defeat "
                "the connection bound that close() exists to enforce, so this "
                "raises instead of quietly building a second pool."
            )
        if self._pool is None:
            with self._pool_lock:
                if self._pool is None:
                    from psycopg_pool import ConnectionPool

                    # ``pg_advisory_xact_lock`` waits forever, so a node stalled holding a
                    # lock (paused container, "idle in transaction") would hang every other
                    # node. No ``statement_timeout``: it would break slow maintenance like
                    # ``garbage_collect``.
                    self._pool = ConnectionPool(
                        self.dsn,
                        min_size=_POOL_MIN_SIZE,
                        max_size=self.max_size,
                        timeout=_POOL_TIMEOUT_SECONDS,
                        kwargs={
                            "row_factory": _row_factory,
                            "connect_timeout": _CONNECT_TIMEOUT_SECONDS,
                            "options": f"-c lock_timeout={_LOCK_TIMEOUT_MS}",
                        },
                        open=True,
                    )
        return self._pool

    def connect(self) -> StoreConnection:
        """Take a pooled connection, re-entrantly per thread.

        A thread already holding one gets it back, and it returns to the pool only
        when the outermost holder closes it. The store acquires nested connections in
        several places, so without this a bounded pool deadlocks.

        A nested call joins the outer transaction. That is sound only while every
        nesting runs after the outer work was committed or rolled back; a nested write
        inside uncommitted outer work would break it.
        """
        state = self._local
        conn = getattr(state, "conn", None)
        if conn is not None:
            state.depth += 1
            return _PostgresConnection(conn, self._release)

        raw = self._get_pool().getconn()
        state.conn = raw
        state.depth = 1
        return _PostgresConnection(raw, self._release)

    def _release(self) -> None:
        state = self._local
        if getattr(state, "conn", None) is None:
            # Release from a non-owning thread (cross-thread close or double release).
            # Decrementing would drive depth negative and putconn(None), losing the real
            # connection from a bounded pool.
            logger.warning("ignoring artifact-store connection release from a non-owning thread")
            return

        state.depth -= 1
        if state.depth > 0:
            return

        raw = state.conn
        state.conn = None

        pool = self._pool
        if self._closed or pool is None:
            # Pool disposed while checked out: putconn would rebuild a pool and reject this
            # connection as foreign.
            raw.close()
            return

        # Every read leaves a transaction open (psycopg begins one implicitly). The pool would
        # roll it back too, but with a WARNING per returned connection.
        import psycopg
        from psycopg.pq import TransactionStatus

        if raw.info.transaction_status in (TransactionStatus.INTRANS, TransactionStatus.INERROR):
            try:
                raw.rollback()
            except psycopg.Error as exc:
                # putconn discards a connection it cannot reset.
                logger.debug("rolling back a returned connection failed: %s", exc)
        pool.putconn(raw)

    def close(self) -> None:
        """Dispose the pool. Terminal: a closed dialect will not reopen.

        An unclosed pool leaks its worker threads and raises at interpreter shutdown.
        """
        with self._pool_lock:
            self._closed = True
            if self._pool is not None:
                self._pool.close()
                self._pool = None

    def column_exists(self, conn: StoreConnection, table: str, column: str) -> bool:
        row = conn.execute(
            "SELECT 1 FROM information_schema.columns WHERE table_name = ? AND column_name = ?",
            (table, column),
        ).fetchone()
        return row is not None

    def schema_exists(self, conn: StoreConnection, table: str = "artifact_versions") -> bool:
        row = conn.execute("SELECT to_regclass(?)", (table,)).fetchone()
        return row is not None and row[0] is not None

    def adapt_ddl(self, sql: str) -> str:
        for pattern, attribute in _DDL_TYPE_REWRITES:
            sql = re.sub(pattern, getattr(self, attribute), sql)
        return sql

    @property
    def autoincrement_pk(self) -> str:
        return "BIGSERIAL PRIMARY KEY"

    @property
    def float_type(self) -> str:
        # Postgres REAL is single precision: too coarse for sub-second time.time() values.
        return "DOUBLE PRECISION"

    @property
    def integer_type(self) -> str:
        # Postgres INTEGER is int4; byte_size reaches 2 GiB.
        return "BIGINT"

    @property
    def integrity_error(self) -> type[Exception]:
        import psycopg

        return psycopg.IntegrityError

    @property
    def operational_error(self) -> type[Exception]:
        import psycopg

        return psycopg.OperationalError

    @property
    def rejectable_errors(self) -> tuple[type[Exception], ...]:
        import psycopg

        return (psycopg.IntegrityError, psycopg.DataError)

    @property
    def supports_legacy_migration(self) -> bool:
        return False

    def resync_autoincrement(self, conn: StoreConnection, table: str, column: str) -> None:
        # is_called=false so the next nextval() returns exactly this value; coalesce starts an
        # empty table at 1.
        conn.execute(
            f"SELECT setval(pg_get_serial_sequence('{table}', '{column}'), "  # noqa: S608
            f"COALESCE((SELECT MAX({column}) FROM {table}), 0) + 1, false)"  # noqa: S608
        )

    def begin_write(self, conn: StoreConnection, key: str) -> None:
        """Serialize writers with a transaction-scoped advisory lock keyed on ``key``.

        Preferred over ``SERIALIZABLE``: no retry loop, released at commit or rollback,
        and keyed so unrelated artifacts do not queue. It is weaker than SQLite's
        whole-file lock: ``finalize_canonical_together`` can still race writers on other
        ids sharing a provenance hash, which ``idx_tenant_provenance_unique`` rejects and
        the caller resolves by catching :attr:`integrity_error`.
        """
        conn.execute("SELECT pg_advisory_xact_lock(?)", (advisory_lock_id(key),))
