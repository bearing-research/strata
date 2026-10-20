"""DuckDB SQL transform (duckdb_sql@v1).

Inputs are registered by position as ``input0``, ``input1`` and so on. Example::

    client.materialize(
        inputs=["file:///warehouse#db.events", "file:///warehouse#db.users"],
        transform={
            "executor": "duckdb_sql@v1",
            "params": {"sql": "SELECT e.*, u.name FROM input0 e JOIN input1 u ON e.user_id = u.id"},
        },
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, field_validator

from strata.transforms.base import Transform, register_transform

if TYPE_CHECKING:
    import pyarrow as pa


class DuckDBSQLParams(BaseModel):
    """Parameters for duckdb_sql@v1; ``sql`` must be non-empty and is stripped."""

    sql: str

    @field_validator("sql")
    @classmethod
    def validate_sql(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("SQL query cannot be empty")
        return v.strip()


@register_transform("duckdb_sql@v1")
class DuckDBSQLTransform(Transform[DuckDBSQLParams]):
    """Run SQL in an in-memory DuckDB, inputs registered by position as ``input0``, ``input1``."""

    Params = DuckDBSQLParams

    def validate(self, inputs: list[pa.Table], params: DuckDBSQLParams) -> None:
        """Accept any inputs: SQL may need none, and DuckDB reports a missing ``inputN``."""
        # No strict validation: DuckDB reports missing table references itself.
        pass

    def execute(self, inputs: list[pa.Table], params: DuckDBSQLParams) -> pa.Table:
        """Execute the SQL and return the result as an Arrow table; DuckDB errors propagate."""
        return run_sql(params.sql, dict(zip(self.get_input_names(len(inputs)), inputs)))


def run_sql(
    sql: str, tables: dict[str, pa.Table], config: dict[str, Any] | None = None
) -> pa.Table:
    """Run *sql* in a fresh in-memory DuckDB over *tables*, each registered under its key.

    *config* adds connect-time settings, such as a memory limit. DuckDB errors propagate.
    """
    import duckdb

    # No files, network or extensions, and the SQL cannot turn them back on: the
    # registered tables are all it reads, so its result is a function of its inputs.
    conn = duckdb.connect(
        ":memory:",
        config={
            "enable_external_access": False,
            "autoinstall_known_extensions": False,
            "autoload_known_extensions": False,
            **(config or {}),
        },
    )
    for name, table in tables.items():
        conn.register(name, table)
    lock_down(conn)
    return conn.execute(sql).to_arrow_table()


def lock_down(conn: Any) -> None:
    """Cut *conn* off from files, network and extensions, and lock its configuration.

    Call it after the connection's own setup, which may need that access; SQL run
    after it cannot turn either back on.
    """
    conn.execute("SET enable_external_access = false")
    conn.execute("SET lock_configuration = true")


def build_duckdb_sql_transform(sql: str) -> dict[str, Any]:
    """Build a duckdb_sql@v1 transform spec for ``materialize()``.

    Examples
    --------
    >>> build_duckdb_sql_transform("SELECT 1")
    {'executor': 'duckdb_sql@v1', 'params': {'sql': 'SELECT 1'}}
    """
    return {"executor": "duckdb_sql@v1", "params": {"sql": sql}}
