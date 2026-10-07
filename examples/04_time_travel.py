#!/usr/bin/env python3
"""
Example 4: Time Travel (Snapshot Queries)

This example shows how to query historical snapshots of a table.
Iceberg maintains a history of table snapshots, and Strata can read any of them.

What you'll learn:
    - How to query a specific snapshot by ID
    - When to use time travel

Runs against the table examples/setup_demo.py creates, which commits its rows
in two appends and so has two snapshots. Strata does not list a table's
snapshots; this reads them from the demo's catalog with pyiceberg.
"""

from pathlib import Path

from pyiceberg.catalog.sql import SqlCatalog

from strata_client import StrataClient

DEMO_WAREHOUSE = Path(__file__).resolve().parent.parent / "demo-warehouse"
table_uri = f"file://{DEMO_WAREHOUSE}#analytics.events"

client = StrataClient(base_url="http://127.0.0.1:8765")

# Query the current (latest) snapshot
# No snapshot_id in params means "use the latest"
artifact = client.materialize(
    inputs=[table_uri],
    transform={"executor": "scan@v1", "params": {}},
)
current_table = client.fetch(artifact.uri)
print(f"Current snapshot: {current_table.num_rows} rows")

# Query a specific historical snapshot: here the table's first one
catalog = SqlCatalog(
    "strata",
    uri=f"sqlite:///{DEMO_WAREHOUSE / 'catalog.db'}",
    warehouse=str(DEMO_WAREHOUSE),
)
historical_snapshot_id = catalog.load_table("analytics.events").snapshots()[0].snapshot_id

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
