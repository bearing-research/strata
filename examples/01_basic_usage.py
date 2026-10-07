#!/usr/bin/env python3
"""
Example 1: Basic Strata Usage

This example shows the simplest way to use Strata to query an Iceberg table.

Prerequisites:
    1. Start the Strata server: strata-notebook
    2. Create the demo table: uv run python examples/setup_demo.py

What you'll learn:
    - How to connect to a Strata server
    - How to materialize a table and fetch data as Arrow
    - How to convert results to pandas
"""

import sys
from pathlib import Path

from strata_client import StrataClient

# Connect to Strata server
client = StrataClient(base_url="http://127.0.0.1:8765")

# Table URI format: file://<warehouse_path>#<namespace>.<table>
# The table examples/setup_demo.py creates; pass another table URI as the first argument.
DEMO_WAREHOUSE = Path(__file__).resolve().parent.parent / "demo-warehouse"
table_uri = sys.argv[1] if len(sys.argv) > 1 else f"file://{DEMO_WAREHOUSE}#analytics.events"

# Materialize the table - returns an Artifact with metadata
artifact = client.materialize(
    inputs=[table_uri],
    transform={"executor": "scan@v1", "params": {}},
)

print(f"Artifact URI: {artifact.uri}")
print(f"Cache hit: {artifact.cache_hit}")

# Fetch the data as an Arrow table
table = client.fetch(artifact.uri)
print(f"Got table with {table.num_rows} rows")
print(f"Columns: {table.schema.names}")

# Option 1: Use the Artifact's helper method
df = artifact.to_pandas()
print(df.head())

# Option 2: Convert Arrow table to pandas directly
df = table.to_pandas()
print(df.head())

# Always close the client when done
client.close()
