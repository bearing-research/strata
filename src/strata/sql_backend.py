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
  Handled by :func:`translate_placeholders` rather than by rewriting 99 call
  sites, so the diff stays proportional to the dialect difference instead of to
  the number of queries.
- **Autoincrement.** One ``INTEGER PRIMARY KEY AUTOINCREMENT`` in the audit
  table.
- **Float width.** The store declares timestamps ``REAL``. In SQLite that is a
  64-bit double; in Postgres ``REAL`` is *single* precision, which cannot hold
  a ``time.time()`` value without losing sub-second resolution. Mapping this
  correctly is the least visible and most damaging difference of the set.
- **Writer serialization.** ``BEGIN IMMEDIATE`` guards the ``MAX(version)+1``
  read-modify-write in ``create_artifact``. Postgres has no such statement and
  needs a different strategy; see :meth:`SqlDialect.begin_write`.

Personal mode keeps SQLite and is unaffected: :class:`SqliteDialect` reproduces
today's behavior exactly, including the connection PRAGMAs.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Protocol

# Characters that open a region where ``?`` is data, not a placeholder.
_SINGLE_QUOTE = "'"
_DOUBLE_QUOTE = '"'


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


class SqlDialect(Protocol):
    """The parts of ``artifact_store`` that differ between databases.

    Everything not named here is portable SQL and stays written once. The
    protocol is deliberately small: each member earns its place by having a
    concrete divergence behind it, not by anticipating one.
    """

    name: str

    def connect(self) -> sqlite3.Connection:
        """Open a connection configured the way the store expects.

        Returns rows indexable by column name, since the store reads
        ``row["tenant"]`` throughout.
        """
        ...

    def sql(self, statement: str) -> str:
        """Adapt a qmark-style statement to this dialect."""
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
    def supports_legacy_migration(self) -> bool:
        """Whether pre-tenant-column databases can exist for this backend.

        Only SQLite has deployed history to migrate. Guarding the migration on
        this keeps ``sqlite_master`` / ``PRAGMA`` / ``rowid`` out of the
        Postgres path instead of forcing portable rewrites of code that runs
        exactly once per legacy database.
        """
        ...

    def begin_write(self, conn: sqlite3.Connection) -> None:
        """Open a transaction that serializes concurrent writers.

        ``create_artifact`` reads ``MAX(version)+1`` and then inserts that
        version. Two concurrent rebuilds of the same artifact must not both
        read ``N`` and collide on the ``(id, version)`` primary key, which is
        a bug this store has already been bitten by once.
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

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def sql(self, statement: str) -> str:
        # The store is already written in this dialect.
        return statement

    @property
    def autoincrement_pk(self) -> str:
        return "INTEGER PRIMARY KEY AUTOINCREMENT"

    @property
    def float_type(self) -> str:
        # SQLite REAL is an IEEE 754 double.
        return "REAL"

    @property
    def supports_legacy_migration(self) -> bool:
        return True

    def begin_write(self, conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")


class PostgresDialect:
    """Rendering rules for Postgres.

    Connection handling is deliberately absent: it needs a driver dependency
    and a pool, and landing those is a separate reviewable step. Everything
    that decides *what SQL to emit* is here and unit-testable now, so the
    rendering can be settled before the infrastructure arrives.
    """

    name = "postgres"

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def connect(self) -> sqlite3.Connection:
        raise NotImplementedError(
            "PostgresDialect.connect is not wired yet. The driver dependency "
            "and connection pool land separately; this class currently exists "
            "so SQL rendering can be settled and tested first."
        )

    def sql(self, statement: str) -> str:
        return translate_placeholders(statement)

    @property
    def autoincrement_pk(self) -> str:
        return "BIGSERIAL PRIMARY KEY"

    @property
    def float_type(self) -> str:
        # Postgres REAL is single precision (~7 significant digits), which
        # cannot represent a time.time() value to sub-second resolution.
        return "DOUBLE PRECISION"

    @property
    def supports_legacy_migration(self) -> bool:
        return False

    def begin_write(self, conn: sqlite3.Connection) -> None:
        raise NotImplementedError(
            "Writer serialization for Postgres is unresolved. BEGIN IMMEDIATE "
            "has no equivalent; the candidates are a transaction-scoped "
            "advisory lock keyed on the artifact id, or SERIALIZABLE with a "
            "retry on serialization failure. The choice belongs with the "
            "connection work, since it depends on the pool's retry behavior."
        )
