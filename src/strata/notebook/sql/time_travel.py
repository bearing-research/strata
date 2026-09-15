"""Durable snapshots for warehouses with time travel (``# @cache snapshot``).

Snowflake and BigQuery expose no per-table snapshot id, but both can query a
table as it stood at a timestamp: ``AT (TIMESTAMP => ...)`` and ``FOR
SYSTEM_TIME AS OF ...``. So a snapshot here is a timestamp. The first run of a
snapshot cell reads the warehouse clock and runs its query pinned there; the
timestamp is the freshness token, and it is recorded on the artifact. Later
runs of the same query find it there and reuse it, so they hit the cache, and a
run with the cache off replays the same state even after new rows land. Only a
change to the cell (query, binds, connection, upstream inputs) or a rerun takes
a new timestamp.

A timestamp stays queryable only within the warehouse's retention window, so
each pin records how long it is good for (``valid_until``).
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


def pin_tables(sql: str, dialect: str, template: str, clause_key: str) -> str:
    """Attach the time-travel clause of *template* to every table *sql* reads.

    *template* is a one-table query in *dialect* carrying the clause, and
    *clause_key* is where sqlglot keeps it on a ``Table`` (``when`` for
    Snowflake's ``AT``, ``version`` for BigQuery's ``FOR SYSTEM_TIME``). A
    reference to a CTE is not a table and is left alone.
    """
    table = sqlglot.parse_one(template, read=dialect).find(exp.Table)
    assert table is not None
    clause = table.args[clause_key]
    tree = sqlglot.parse_one(sql, read=dialect)
    ctes = {cte.alias_or_name for cte in tree.find_all(exp.CTE)}
    for reference in tree.find_all(exp.Table):
        if not reference.db and reference.name in ctes:
            continue
        reference.set(clause_key, clause.copy())
    return tree.sql(dialect=dialect)
