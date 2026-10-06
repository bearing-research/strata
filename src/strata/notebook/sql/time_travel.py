"""Durable snapshots for warehouses with time travel (``# @cache snapshot``).

Snowflake and BigQuery have no per-table snapshot id, so a snapshot is a
timestamp: the first run reads the warehouse clock and pins the query there, and
the timestamp is recorded on the artifact as the freshness token. Later runs
reuse it, so they hit the cache or replay the same state; only a change to the
cell or a rerun takes a new one. Each pin records ``valid_until``, the end of
the warehouse's retention window.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

import sqlglot
from sqlglot import exp

from strata.notebook.sql.adapter import QualifiedTable

# Artifact transform params a snapshot cell records.
PARAM_BASIS = "sql_snapshot_basis"
PARAM_AT = "sql_snapshot_at"
PARAM_VALID_UNTIL = "sql_snapshot_valid_until"


@dataclass(frozen=True)
class SnapshotPin:
    """The timestamp a snapshot cell's query is pinned to."""

    at: str
    valid_until: str | None


class TimeTravelAdapter(Protocol):
    """What a driver adds to support ``# @cache snapshot`` through time travel."""

    def snapshot_timestamp(self, conn: Any) -> str:
        """The warehouse's current timestamp, as ISO-8601 UTC."""
        ...

    def retention_until(self, conn: Any, tables: list[QualifiedTable], at: str) -> str | None:
        """Until when *tables* can still be queried as of *at*, as ISO-8601 UTC."""
        ...

    def pin_query(self, sql: str, at: str) -> str:
        """*sql* with every table read as of *at*."""
        ...


def supports_time_travel(adapter: Any) -> bool:
    return callable(getattr(adapter, "pin_query", None))


def iso_utc(value: Any) -> str:
    """A driver's timestamp (datetime or string) as ISO-8601 in UTC."""
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    else:
        moment = datetime.fromisoformat(str(value))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC).isoformat()


def plus(at: str, delta: timedelta) -> str:
    return (datetime.fromisoformat(at) + delta).astimezone(UTC).isoformat()


def pin_tables(sql: str, dialect: str, clause: str, clause_key: str, *, after_alias: bool) -> str:
    """Write the time-travel *clause* after every table *sql* reads.

    *clause_key* is where sqlglot keeps such a clause on a ``Table`` (``when`` for
    Snowflake, ``version`` for BigQuery). Tables are chosen scope by scope as the
    analyzer does, so a base table sharing a name with a CTE elsewhere is still
    pinned. A table the author already pinned keeps its moment.
    """
    from strata.notebook.sql.analyzer import base_table_nodes

    tree = sqlglot.parse_one(sql, read=dialect)
    tables = [t for t in base_table_nodes(tree, dialect) if not t.args.get(clause_key)]
    return splice_after_tables(sql, [(t, clause) for t in tables], after_alias=after_alias)


def splice_after_tables(sql: str, clauses: list[tuple[Any, str]], *, after_alias: bool) -> str:
    """*sql* with each clause written after its parsed ``exp.Table``, all else as written.

    Not sqlglot's regeneration of the query, which can compute something else
    (DuckDB ``MAP {a: b}`` becomes ``MAP {'a': b}``, BigQuery ``NUMERIC(10,2)``
    loses its precision). The tables must come from parsing *sql* itself.
    """
    points: list[tuple[int, str]] = []
    for table, clause in clauses:
        alias = table.args.get("alias") if after_alias else None
        anchor = alias.find_all(exp.Identifier) if alias is not None else table.parts
        ends = [part.meta["end"] for part in anchor if "end" in part.meta]
        if not ends:
            raise ValueError(f"cannot place a snapshot clause after {table.sql()}")
        end = max(ends) + 1
        if alias is not None and alias.columns:
            end = sql.index(")", end) + 1
        points.append((end, clause))
    for end, clause in sorted(points, reverse=True):
        sql = f"{sql[:end]} {clause}{sql[end:]}"
    return sql
