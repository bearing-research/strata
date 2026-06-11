#!/usr/bin/env python3
"""
Example 2: Column Projection

This example shows how to select specific columns to reduce data transfer.

What you'll learn:
    - How to specify which columns to read
    - Why projection improves performance
"""

from strata_client import StrataClient

client = StrataClient(base_url="http://127.0.0.1:8765")
table_uri = "file:///path/to/warehouse#my_db.my_table"

# Read only specific columns using the scan transform params
# This reduces network transfer and memory usage
artifact = client.materialize(
    inputs=[table_uri],
    transform={
        "executor": "scan@v1",
        "params": {"columns": ["user_id", "event_type", "timestamp"]},
    },
)

# Fetch the data
table = client.fetch(artifact.uri)

# Verify we only got the requested columns
print(f"Columns returned: {table.schema.names}")
# Output: ['user_id', 'event_type', 'timestamp']

client.close()
