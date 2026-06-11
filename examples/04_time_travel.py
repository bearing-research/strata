#!/usr/bin/env python3
"""
Example 4: Time Travel (Snapshot Queries)

This example shows how to query historical snapshots of a table.
Iceberg maintains a history of table snapshots, and Strata can read any of them.

What you'll learn:
    - How to query a specific snapshot by ID
    - When to use time travel
"""

from strata_client import StrataClient

client = StrataClient(base_url="http://127.0.0.1:8765")
table_uri = "file:///path/to/warehouse#my_db.events"

# Query the current (latest) snapshot
# No snapshot_id in params means "use the latest"
artifact = client.materialize(
    inputs=[table_uri],
    transform={"executor": "scan@v1", "params": {}},
)
current_table = client.fetch(artifact.uri)
print(f"Current snapshot: {current_table.num_rows} rows")

# Query a specific historical snapshot
# You can get snapshot IDs from Iceberg table metadata
historical_snapshot_id = 1234567890123456789

artifact = client.materialize(
    inputs=[table_uri],
    transform={
        "executor": "scan@v1",
        "params": {"snapshot_id": historical_snapshot_id},
    },
)
historical_table = client.fetch(artifact.uri)
print(f"Historical snapshot: {historical_table.num_rows} rows")

# Use case: Compare current vs historical data
# This is useful for:
# - Auditing changes
# - Reproducing ML training data
# - Debugging data issues

client.close()
