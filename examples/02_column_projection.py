#!/usr/bin/env python3
"""
Example 2: Column Projection

This example shows how to select specific columns to reduce data transfer.

What you'll learn:
    - How to specify which columns to read
    - Why projection improves performance
"""

import sys
from pathlib import Path

from strata_client import StrataClient

client = StrataClient(base_url="http://127.0.0.1:8765")
# The table examples/setup_demo.py creates; pass another table URI as the first argument.
DEMO_WAREHOUSE = Path(__file__).resolve().parent.parent / "demo-warehouse"
table_uri = sys.argv[1] if len(sys.argv) > 1 else f"file://{DEMO_WAREHOUSE}#analytics.events"

# Read only specific columns using the scan transform params
# This reduces network transfer and memory usage
artifact = client.materialize(
    inputs=[table_uri],
    transform={
        "executor": "scan@v1",
        "params": {"columns": ["id", "category", "timestamp"]},
    },
)

# Fetch the data
table = client.fetch(artifact.uri)

# Verify we only got the requested columns
print(f"Columns returned: {table.schema.names}")
# Output: ['id', 'category', 'timestamp']

client.close()
