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

import sys
from pathlib import Path

import duckdb

from strata_client import gt
from strata_client.integration.duckdb import StrataScanner, register_strata_scan, strata_query

# The table examples/setup_demo.py creates; pass another table URI as the first argument.
DEMO_WAREHOUSE = Path(__file__).resolve().parent.parent / "demo-warehouse"
table_uri = sys.argv[1] if len(sys.argv) > 1 else f"file://{DEMO_WAREHOUSE}#analytics.events"

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
    # The same table again, pruned to the row groups that can hold value > 900
    scanner.register(
        "high_value", table_uri, columns=["category", "value"], filters=[gt("value", 900.0)]
    )

    # Join the two with SQL; pruning keeps whole row groups, so filter rows here too
    result = scanner.query("""
        SELECT e.category, COUNT(*) AS events, h.high_value_events
        FROM events e
        JOIN (
            SELECT category, COUNT(*) AS high_value_events
            FROM high_value WHERE value > 900 GROUP BY category
        ) h USING (category)
        GROUP BY e.category, h.high_value_events
        ORDER BY e.category
        LIMIT 5
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
