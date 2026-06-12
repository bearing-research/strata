#!/usr/bin/env python3
"""
Example 3: Filtering with Predicates

This example shows how to use filters to reduce data at the source.
Filters enable two-tier pruning that can dramatically reduce I/O:

1. **Iceberg file-level pruning**: Uses manifest statistics (per-file min/max)
   to skip entire Parquet files before reading any file metadata.

2. **Parquet row-group pruning**: Uses row group statistics (per-chunk min/max)
   to skip row groups within files that don't match the filter.

What you'll learn:
    - How to use comparison filters (gt, lt, eq, etc.)
    - How to combine multiple filters
    - How filters enable two-tier pruning
"""

from strata_client import StrataClient

client = StrataClient(base_url="http://127.0.0.1:8765")
table_uri = "file:///path/to/warehouse#my_db.events"


# Helper to build filter specs for the transform params
def make_filter(column: str, op: str, value) -> dict:
    """Create a filter dict for the transform params."""
    return {"column": column, "op": op, "value": value}


# Filter with greater-than
# Two-tier pruning:
#   1. Iceberg skips files where max(timestamp) < 1704067200000000
#   2. Parquet skips row groups where max(timestamp) < 1704067200000000
artifact = client.materialize(
    inputs=[table_uri],
    transform={
        "executor": "scan@v1",
        "params": {
            "filters": [make_filter("timestamp", ">", 1704067200000000)],  # After 2024-01-01
        },
    },
)
table = client.fetch(artifact.uri)
print(f"Rows after timestamp filter: {table.num_rows}")

# Filter with less-than
artifact = client.materialize(
    inputs=[table_uri],
    transform={
        "executor": "scan@v1",
        "params": {"filters": [make_filter("value", "<", 100.0)]},
    },
)
table = client.fetch(artifact.uri)
print(f"Rows with value < 100: {table.num_rows}")

# Filter with equality
artifact = client.materialize(
    inputs=[table_uri],
    transform={
        "executor": "scan@v1",
        "params": {"filters": [make_filter("category", "=", "electronics")]},
    },
)
table = client.fetch(artifact.uri)
print(f"Rows in electronics category: {table.num_rows}")

# Combine multiple filters (AND logic)
# All conditions must be true for a row to be included
artifact = client.materialize(
    inputs=[table_uri],
    transform={
        "executor": "scan@v1",
        "params": {
            "columns": ["id", "value", "category"],
            "filters": [
                make_filter("value", ">=", 10.0),
                make_filter("value", "<=", 100.0),
                make_filter("category", "=", "electronics"),
            ],
        },
    },
)
table = client.fetch(artifact.uri)
print(f"Rows matching all filters: {table.num_rows}")

# Check server metrics for pruning stats
metrics = client.metrics()
print(f"Row groups pruned: {metrics.get('row_groups_pruned', 0)}")

client.close()
