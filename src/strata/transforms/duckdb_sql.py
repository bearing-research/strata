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
        import duckdb

        # Runs in the server process: no files, network or extensions, and the
        # SQL cannot turn them back on. The registered inputs are all it reads.
        conn = duckdb.connect(
            ":memory:",
            config={
                "enable_external_access": False,
                "autoinstall_known_extensions": False,
                "autoload_known_extensions": False,
            },
        )

        input_names = self.get_input_names(len(inputs))
        for name, table in zip(input_names, inputs):
            conn.register(name, table)
        conn.execute("SET lock_configuration = true")

        result = conn.execute(params.sql).to_arrow_table()
        return result


def build_duckdb_sql_transform(sql: str) -> dict[str, Any]:
    """Build a duckdb_sql@v1 transform spec for ``materialize()``.

    Examples
    --------
    >>> build_duckdb_sql_transform("SELECT 1")
    {'executor': 'duckdb_sql@v1', 'params': {'sql': 'SELECT 1'}}
    """
    return {"executor": "duckdb_sql@v1", "params": {"sql": sql}}
