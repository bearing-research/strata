"""SQL dialect seam for the artifact store.

The artifact store is the platform's system of record, and it is SQLite on a
local file. That caps a deployment at one node. This module is the seam that
lets a second backend exist without forking ``artifact_store.py``, whose ~50
public methods issue 105 SQL statements between them.

The port is far smaller than that count suggests. Surveying the store turns up
only a handful of genuinely dialect-specific constructs, and almost all of them
sit in the *legacy-migration* path (``PRAGMA table_info``, ``sqlite_master``,
``rowid``) that exists to upgrade databases written by older Strata versions. A
Postgres deployment has no such history, so it never runs that path. What
remains is:

- **Placeholders.** The store writes ``?``; Postgres drivers want ``%s``.
  Handled by :func:`translate_placeholders` inside the connection wrapper, so
  none of the 103 ``conn.execute`` call sites change.
- **Autoincrement.** One ``INTEGER PRIMARY KEY AUTOINCREMENT`` in the audit
  table.
- **Integer and float width.** Both of the store's numeric column types are
  narrower in Postgres than in SQLite, and both lose data silently at the
  boundary rather than failing early.

  ``REAL`` is a 64-bit double in SQLite and *single* precision in Postgres,
  which cannot hold a ``time.time()`` value without losing sub-second
  resolution.

  ``INTEGER`` is up to 64 bits in SQLite and exactly 32 in Postgres, so
  ``byte_size`` overflows on any artifact at or above 2 GiB. That one is
  worse than it first looks: Postgres raises ``NumericValueOutOfRange``,
  which is a ``DataError`` and *not* an ``IntegrityError``, so the store's
  finalize handler does not catch it -- the blob is already written and the
  row is left in ``building`` forever.
- **Writer serialization.** ``BEGIN IMMEDIATE`` guards two read-modify-writes.
  Postgres has no such statement; see :meth:`SqlDialect.begin_write`.
- **The integrity-error type**, which the store catches by name.

One divergence is deliberately *not* papered over. SQLite's ``LIKE`` is
case-insensitive for ASCII; Postgres's is case-sensitive. So a prefix search
for ``"Model"`` matches a stored ``model_v1`` on SQLite and nothing on
Postgres (``list_artifacts``, ``find_dependents``,
``list_latest_by_id_prefix``). Postgres is the stricter and more predictable
side, and forcing either engine to imitate the other would change behavior
personal mode has today. Callers that need case-insensitive matching should
normalize explicitly rather than rely on the backend.

Personal mode keeps SQLite and is unaffected: :class:`SqliteDialect` reproduces
today's behavior exactly, including the connection PRAGMAs.
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

# Mirrors ``sqlite3.connect(timeout=30.0)``: a contended writer gives up after
# the same interval rather than blocking forever.
_LOCK_TIMEOUT_MS = 30_000
_CONNECT_TIMEOUT_SECONDS = 10

# Pool defaults. max_size is per process, and the store nests connection
# acquisition two deep in places, so re-entrant sharing (see
# ``PostgresDialect.connect``) is what keeps this from having to be sized at
# twice the concurrency.
_POOL_MIN_SIZE = 0
_POOL_MAX_SIZE = 16
# Bounds the wait for a free connection. Without it an exhausted pool blocks
# forever, which is the failure this whole class exists to avoid.
_POOL_TIMEOUT_SECONDS = 30.0


# Characters that open a region where ``?`` is data, not a placeholder.
logger = logging.getLogger(__name__)

_SINGLE_QUOTE = "'"
_DOUBLE_QUOTE = '"'

# Column-type tokens that mean different things to different engines. Applied
# only to this package's own DDL constants, never to arbitrary SQL.
#
# Order matters: the autoincrement rewrite has to consume its own INTEGER
# before the bare-INTEGER rule runs, or it would never match.
_DDL_TYPE_REWRITES = (
    ("INTEGER PRIMARY KEY AUTOINCREMENT", "autoincrement_pk"),
    (r"\bREAL\b", "float_type"),
    (r"\bINTEGER\b", "integer_type"),
)


def translate_placeholders(sql: str) -> str:
    """Rewrite ``?`` placeholders to the ``%s`` style Postgres drivers expect.

    A blind ``sql.replace("?", "%s")`` is wrong in two directions, and both
    occur in real queries:

    1. A ``?`` inside a string literal or a quoted identifier is data. The
       store already ships ``LIKE ? ESCAPE '\\'``, where a naive replace would
       not corrupt anything today but would the moment a literal contains a
       question mark.
    2. Drivers using pyformat treat ``%`` as the start of a substitution, so
       any literal percent in the SQL has to be doubled. This is latent rather
       than live -- every ``LIKE`` in the store binds its pattern as a
       parameter -- but a future inline ``LIKE 'nb\\_%'`` would otherwise fail
       at execute time with an opaque driver error.

    So this walks the statement tracking whether it is inside a single-quoted
    literal, a double-quoted identifier, a ``--`` line comment, or a ``/* */``
    block comment. Placeholders convert only outside those regions; percent
    signs are escaped everywhere, because the driver scans the whole string.
    Doubled quotes (``''`` and ``""``) are SQL's own escape and stay inside
    their region.

    Parameters
    ----------
    sql : str
        A statement written in qmark style.

    Returns
    -------
    str
        The statement in pyformat style.
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

    SQL escapes a quote by doubling it, so ``'it''s'`` is one literal rather
    than two. An unterminated region runs to the end of the statement: this is
    a translator, not a validator, and letting the driver report the syntax
    error keeps the failure legible.
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

    Deliberately computed here rather than with Postgres's ``hashtext()``: that
    function is an internal whose output is not contracted across major
    versions, and a lock id that changes under an upgrade would silently stop
    serializing the writers it was added to serialize.
    """
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


class StoreConnection(Protocol):
    """The connection surface ``artifact_store`` actually uses.

    Narrower than either driver's real API, and deliberately shaped like
    ``sqlite3.Connection`` because that is what the store was written against.
    :class:`_PostgresConnection` adapts psycopg to fit, which is what keeps the
    103 ``conn.execute`` call sites unchanged.
    """

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any: ...

    def executescript(self, sql: str) -> Any: ...

    def commit(self) -> None: ...

    def rollback(self) -> None: ...

    def close(self) -> None: ...


class SqlDialect(Protocol):
    """The parts of ``artifact_store`` that differ between databases.

    Everything not named here is portable SQL and stays written once. The
    protocol is deliberately small: each member earns its place by having a
    concrete divergence behind it, not by anticipating one.
    """

    name: str

    def connect(self) -> StoreConnection:
        """Open a connection configured the way the store expects.

        Returns rows indexable by both column name and position, since the
        store uses ``row["tenant"]`` and ``fetchone()[0]`` interchangeably.
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
        """Column type for a 64-bit float.

        Named rather than hardcoded because ``REAL`` silently means different
        widths in SQLite and Postgres, and the store stores epoch timestamps
        in these columns.
        """
        ...

    @property
    def integer_type(self) -> str:
        """Column type for a 64-bit integer.

        Same hazard as :attr:`float_type`. ``byte_size`` and ``row_count`` are
        declared with it, and a 2 GiB artifact exceeds a 32-bit column.
        """
        ...

    @property
    def integrity_error(self) -> type[Exception]:
        """Exception raised when a constraint is violated.

        The store catches this to make finalize idempotent, so it has to be a
        driver-specific type rather than a bare ``Exception``.
        """
        ...

    @property
    def supports_legacy_migration(self) -> bool:
        """Whether pre-tenant-column databases can exist for this backend.

        Only SQLite has deployed history to migrate. Guarding the migration on
        this keeps ``sqlite_master`` / ``PRAGMA`` / ``rowid`` out of the
        Postgres path instead of forcing portable rewrites of code that runs
        exactly once per legacy database.
        """
        ...

    def schema_exists(self, conn: StoreConnection, table: str = "artifact_versions") -> bool:
        """Whether ``table`` is already present.

        Lets ``_init_schema`` skip the global schema lock on every subsequent
        construction; several call sites build a store per session, so taking
        it unconditionally would funnel them all through one mutex.
        """
        ...

    def close(self) -> None:
        """Release any resources the dialect holds. Idempotent.

        On SQLite there are none; connections are per-operation file handles.
        Declared here anyway so a caller can shut a store down without
        knowing which backend is underneath.
        """
        ...

    def resync_autoincrement(self, conn: StoreConnection, table: str, column: str) -> None:
        """Point a synthetic key's generator past the largest existing value.

        Only matters after rows are inserted with explicit ids -- migration.
        Postgres keeps a sequence *beside* a ``BIGSERIAL`` column, and an
        explicit insert does not advance it, so a migrated table hands out
        ids starting at 1 again and every insert collides until it catches up.
        SQLite derives the next rowid from the table itself and needs nothing.
        """
        ...

    def begin_write(self, conn: StoreConnection, key: str) -> None:
        """Open a transaction that serializes writers contending on ``key``.

        Two call sites need this. ``create_artifact`` reads ``MAX(version)+1``
        and then inserts that version; two concurrent rebuilds of the same
        artifact must not both read ``N`` and collide on the ``(id, version)``
        primary key, a bug this store has already been bitten by once.
        ``force_finalize_canonical`` promotes one row to canonical while
        superseding others.

        ``key`` names the contended resource so a backend can lock narrowly
        rather than globally.
        """
        ...


class SqliteDialect:
    """Today's behavior, unchanged.

    Personal mode runs on this and must not notice the seam exists: the
    PRAGMAs, the row factory, the timeout, and ``BEGIN IMMEDIATE`` are all
    reproduced exactly as ``ArtifactStore._get_connection`` set them.
    """

    name = "sqlite"

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def connect(self) -> StoreConnection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn  # type: ignore[return-value]

    def adapt_ddl(self, sql: str) -> str:
        # The DDL is already written in this dialect.
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
    def supports_legacy_migration(self) -> bool:
        return True

    def schema_exists(self, conn: StoreConnection, table: str = "artifact_versions") -> bool:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (table,)
        ).fetchone()
        return row is not None

    def close(self) -> None:
        # Nothing pooled: each connection is a file handle closed by its caller.
        return

    def resync_autoincrement(self, conn: StoreConnection, table: str, column: str) -> None:
        # SQLite derives the next rowid from the table, so there is nothing
        # standing beside it to fall out of step.
        return

    def begin_write(self, conn: StoreConnection, key: str) -> None:
        # SQLite locks the whole database file, so the key is not needed: any
        # writer excludes any other regardless of what it intends to touch.
        conn.execute("BEGIN IMMEDIATE")


class _Row(Mapping):
    """A psycopg row that behaves like ``sqlite3.Row``.

    The store reads results both ways -- ``row["tenant"]`` in most places,
    ``cursor.fetchone()[0]`` where the query selects a single computed value --
    and converts two of them with ``dict(row)``. psycopg's stock factories give
    one access style or the other, so this supplies both. Implementing
    ``Mapping`` is what makes ``dict(row)`` work without a special case.
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
            # Mapping's mixin get() and __contains__ catch KeyError only, so
            # letting list.index's ValueError escape would make row.get("x")
            # raise instead of returning the default.
            raise KeyError(key) from None

    def keys(self) -> tuple[str, ...]:  # type: ignore[override]
        return self._columns

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

    Two adaptations, both mechanical:

    - ``execute`` translates qmark to pyformat on the way through, which is why
      the store's SQL does not have to be rewritten.
    - ``executescript`` exists on ``sqlite3.Connection`` but not on psycopg.
      The store uses it only for its own multi-statement schema DDL.
    """

    def __init__(self, inner: Any, release: Any) -> None:
        self._inner = inner
        self._release = release
        self._closed = False

    def execute(self, sql: str, params: Sequence[Any] = ()) -> Any:
        return self._inner.execute(translate_placeholders(sql), params)

    def executescript(self, sql: str) -> Any:
        # No parameters, so the percent-escaping in translate_placeholders
        # would be the only effect; the DDL contains none. Sent as-is.
        return self._inner.execute(sql)

    def commit(self) -> None:
        self._inner.commit()

    def rollback(self) -> None:
        self._inner.rollback()

    def close(self) -> None:
        """Hand the connection back rather than closing it.

        The store calls ``close`` 41 times and means "I am done with this";
        under a pool that has to mean "return it". Guarded against a double
        release, which would hand the same connection to two threads.
        """
        if self._closed:
            return
        self._closed = True
        self._release()


class PostgresDialect:
    """Postgres backing for the artifact store.

    Requires the ``postgres`` extra (``pip install strata-notebook[postgres]``).
    """

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
        """Build the pool on first use.

        Lazy so that constructing a dialect -- which tests and the config
        factory both do freely -- costs no sockets until something queries.
        """
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

                    # Timeouts, because the SQLite dialect had them and this
                    # one inherited none. ``sqlite3.connect(timeout=30.0)``
                    # made a contended writer fail loudly after 30s;
                    # ``pg_advisory_xact_lock`` waits forever, so a node that
                    # stalls while holding a lock -- a paused container, a
                    # partition leaving the backend "idle in transaction" --
                    # would hang every other node behind it with no bound.
                    #
                    # Deliberately no ``statement_timeout``: the hazard is
                    # waiting on a lock, not running a long query, and a query
                    # cap would break slow maintenance like ``garbage_collect``.
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
        """Take a pooled connection, re-entrantly.

        A thread that already holds one gets the same connection back with a
        bumped depth, and the connection returns to the pool only when the
        outermost holder closes it.

        This is not an optimization, it is what makes a bounded pool safe
        here. ``ArtifactStore`` acquires two deep in six places -- most of
        them ``return self.get_artifact(...)`` evaluated inside a ``try``
        whose ``finally`` has not yet released the outer connection. Without
        re-entrancy, ``max_size`` threads each holding one and waiting for a
        second deadlock until the pool timeout fires, and the pool would have
        to be sized at twice peak concurrency to be safe.

        Sharing a connection means a nested call joins the outer transaction.
        That is sound for every nesting that exists today: each one runs after
        the outer work has been committed or rolled back, so there is no
        pending state for a nested ``commit`` to publish early. The nested
        ``set_name`` at ``artifact_store.py:1060`` is the only nested *write*,
        and the line above it rolls the outer transaction back. Adding a
        nested write while the outer holds uncommitted work would break that,
        which is why it is written down here.
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
            # A release from a thread that is not the one holding this
            # connection: a cross-thread close, or a double release that got
            # past the wrapper's own guard. Returning here keeps the failure
            # from compounding -- decrementing blindly would drive depth
            # negative and then call putconn(None), which psycopg_pool rejects
            # while the real connection stays lost from a bounded pool.
            logger.warning("ignoring artifact-store connection release from a non-owning thread")
            return

        state.depth -= 1
        if state.depth > 0:
            return

        raw = state.conn
        state.conn = None

        pool = self._pool
        if self._closed or pool is None:
            # The pool was disposed while this connection was checked out.
            # putconn would rebuild a pool and then reject the connection as
            # foreign, so dispose it directly instead.
            raw.close()
            return

        # putconn rolls back anything still open, so a caller that raised
        # before committing cannot leak a transaction into the next borrower.
        pool.putconn(raw)

    def close(self) -> None:
        """Dispose the pool. Terminal -- a closed dialect will not reopen.

        Every ``ConnectionPool`` also runs worker threads, so a dialect left
        unclosed leaks those for the process lifetime and its ``__del__``
        raises ``PythonFinalizationError`` at interpreter shutdown.
        """
        with self._pool_lock:
            self._closed = True
            if self._pool is not None:
                self._pool.close()
                self._pool = None

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
        # Postgres REAL is single precision (~7 significant digits), which
        # cannot represent a time.time() value to sub-second resolution.
        return "DOUBLE PRECISION"

    @property
    def integer_type(self) -> str:
        # Postgres INTEGER is int4, capped at 2147483647 -- smaller than
        # byte_size for any artifact at or above 2 GiB.
        return "BIGINT"

    @property
    def integrity_error(self) -> type[Exception]:
        import psycopg

        return psycopg.IntegrityError

    @property
    def supports_legacy_migration(self) -> bool:
        return False

    def resync_autoincrement(self, conn: StoreConnection, table: str, column: str) -> None:
        # setval with is_called=false so the next nextval() returns exactly
        # this value; coalesce covers an empty table, where the sequence must
        # start at 1 rather than 0.
        conn.execute(
            f"SELECT setval(pg_get_serial_sequence('{table}', '{column}'), "  # noqa: S608
            f"COALESCE((SELECT MAX({column}) FROM {table}), 0) + 1, false)"  # noqa: S608
        )

    def begin_write(self, conn: StoreConnection, key: str) -> None:
        """Serialize writers with a transaction-scoped advisory lock.

        Postgres has no ``BEGIN IMMEDIATE``. Of the two candidates, an advisory
        lock beats ``SERIALIZABLE`` here: it needs no retry loop (the second
        writer blocks rather than aborting), it releases automatically at
        commit or rollback, and it is keyed, so writers touching different
        artifacts do not queue behind each other the way SQLite's whole-file
        lock makes them.

        Scope, stated precisely: the lock serializes writers passing the *same*
        key, and nothing else. ``force_finalize_canonical`` also touches rows
        sharing a provenance hash across *other* artifact ids, which those
        writers do not contend on. The ``idx_tenant_provenance_unique`` partial
        index rejects that case, and the caller catches
        :attr:`integrity_error` and returns the row that won.

        Note this is genuinely weaker than SQLite, not merely different: under
        ``BEGIN IMMEDIATE`` the whole file is locked, so a second writer is
        never in flight and the index never fires. Narrower locking is the
        price of not serializing every artifact-store write in the cluster
        behind one mutex, and it is why that handler had to be added.
        """
        conn.execute("SELECT pg_advisory_xact_lock(?)", (advisory_lock_id(key),))
