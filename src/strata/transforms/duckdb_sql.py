"""DuckDB SQL transform (duckdb_sql@v1).

Execute SQL queries over input tables using DuckDB. Input tables are
registered as "input0", "input1", etc. and can be queried using SQL.

Example:
    client.materialize(
        inputs=[
            "file:///warehouse#db.events",
            "file:///warehouse#db.users",
        ],
        transform={
            "executor": "duckdb_sql@v1",
            "params": {
                "sql": "SELECT e.*, u.name FROM input0 e JOIN input1 u ON e.user_id = u.id"
            }
        },
    )

The transform:
1. Fetches all input tables (in parallel if possible)
2. Registers them as input0, input1, ... in DuckDB
3. Executes the SQL query
4. Returns the result as an Arrow table
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, field_validator

from strata.transforms.base import Transform, register_transform

if TYPE_CHECKING:
    import pyarrow as pa


class DuckDBSQLParams(BaseModel):
    """Parameters for the duckdb_sql@v1 transform.

    Attributes:
        sql: SQL query to execute. Use input0, input1, etc. to reference inputs.
    """

    sql: str

    @field_validator("sql")
    @classmethod
    def validate_sql(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("SQL query cannot be empty")
        return v.strip()


@register_transform("duckdb_sql@v1")
class DuckDBSQLTransform(Transform[DuckDBSQLParams]):
    """Execute SQL queries using DuckDB.

    DuckDB is a high-performance analytical database that works directly
    with Arrow data. This transform registers input tables in an in-memory
    DuckDB instance and executes the provided SQL query.

    Input Naming:
        Inputs are registered as "input0", "input1", etc. based on their
        position in the inputs list. Use these names in your SQL query.

    Example:
        # Single table aggregation
        transform = {
            "executor": "duckdb_sql@v1",
            "params": {"sql": "SELECT category, SUM(amount) FROM input0 GROUP BY 1"}
        }

        # Two-table join
        transform = {
            "executor": "duckdb_sql@v1",
            "params": {
                "sql": '''
                    SELECT e.event_type, u.name, COUNT(*)
                    FROM input0 e
                    JOIN input1 u ON e.user_id = u.id
                    GROUP BY 1, 2
                '''
            }
        }

    Requirements:
        - DuckDB must be installed: pip install duckdb
    """

    Params = DuckDBSQLParams

    def validate(self, inputs: list[pa.Table], params: DuckDBSQLParams) -> None:
        """Validate inputs before execution.

        Note: DuckDB can execute queries without inputs (e.g., SELECT 1),
        so we don't require inputs here. If the SQL references input0, etc.
        but no inputs are provided, DuckDB will raise an error at execution time.

        Args:
            inputs: List of input Arrow tables
            params: Validated parameters
        """
        # No strict validation: DuckDB reports missing table references itself.
        pass

    def execute(self, inputs: list[pa.Table], params: DuckDBSQLParams) -> pa.Table:
        """Execute the SQL query.

        Args:
            inputs: List of input Arrow tables
            params: Validated parameters with SQL query

        Returns:
            Result Arrow table

        Raises:
            ImportError: If DuckDB is not installed
            Exception: If SQL execution fails
        """
        try:
            import duckdb
        except ImportError:
            raise ImportError(
                "DuckDB is required for duckdb_sql@v1. Install with: pip install duckdb"
            )

        conn = duckdb.connect(":memory:")

        input_names = self.get_input_names(len(inputs))
        for name, table in zip(input_names, inputs):
            conn.register(name, table)

        result = conn.execute(params.sql).to_arrow_table()
        return result


def build_duckdb_sql_transform(sql: str) -> dict[str, Any]:
    """Build a duckdb_sql@v1 transform specification.

    Args:
        sql: SQL query to execute

    Returns:
        Transform spec dict for materialize()

    Example:
        transform = build_duckdb_sql_transform(
            "SELECT category, SUM(amount) FROM input0 GROUP BY 1"
        )
        client.materialize(inputs=[table_uri], transform=transform)
    """
    return {"executor": "duckdb_sql@v1", "params": {"sql": sql}}
