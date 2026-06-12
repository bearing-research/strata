#!/usr/bin/env python3
"""
Example 5: DuckDB Integration

This example shows how to use Strata with DuckDB for SQL queries.
Data is fetched from Strata once at registration time and materialized
as an Arrow table that DuckDB can query with SQL.

What you'll learn:
    - How to register Strata tables as DuckDB views
    - How to run SQL queries against Iceberg tables via Strata
    - Difference between Strata-side and DuckDB-side filtering

Important: DuckDB SQL filters (WHERE clauses) are applied *after* data is
fetched from Strata. For Strata-side pruning, pass filters to register().
"""

import duckdb

from strata_client import gt
from strata_client.integration.duckdb import StrataScanner, register_strata_scan, strata_query

# Define your table URI
table_uri = "file:///path/to/warehouse#my_db.events"

# Method 1: Register a single table
conn = duckdb.connect()
register_strata_scan(
    conn,
    name="events",
    table_uri=table_uri,
    columns=["id", "value", "category"],  # Column projection (Strata-side)
    filters=[gt("value", 10.0)],  # Row-group pruning (Strata-side)
)

# Query using SQL - WHERE clause is DuckDB-side (after fetch)
result = conn.execute("""
    SELECT
        category,
        COUNT(*) as count,
        AVG(value) as avg_value
    FROM events
    WHERE value > 50
    GROUP BY category
    ORDER BY count DESC
    LIMIT 10
""").fetchdf()

print(result)
conn.close()

# Method 2: Use StrataScanner for multiple tables
with StrataScanner() as scanner:
    # Register with Strata-side pruning
    scanner.register("events", table_uri, columns=["id", "value", "category"])
    scanner.register("users", "file:///warehouse#my_db.users")

    # Join tables with SQL
    result = scanner.query("""
        SELECT e.*, u.name
        FROM events e
        JOIN users u ON e.user_id = u.id
        WHERE e.value > 100
    """)
    print(result)

# Method 3: One-shot query with strata_query()
result = strata_query(
    "SELECT id, value FROM events WHERE id < 1000",
    tables={
        "events": {
            "table_uri": table_uri,
            "columns": ["id", "value"],
            "filters": [gt("value", 100)],  # Strata-side pruning
        }
    },
)
print(result)
