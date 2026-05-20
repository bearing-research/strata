#!/usr/bin/env python3
"""
Example 1: Basic Strata Usage

This example shows the simplest way to use Strata to query an Iceberg table.

Prerequisites:
    1. Start the Strata server: strata-notebook
    2. Have an Iceberg table accessible via file:// URI

What you'll learn:
    - How to connect to a Strata server
    - How to materialize a table and fetch data as Arrow
    - How to convert results to pandas
"""

from strata.client import StrataClient

# Connect to Strata server
client = StrataClient(base_url="http://127.0.0.1:8765")

# Define your table URI
# Format: file://<warehouse_path>#<namespace>.<table>
table_uri = "file:///path/to/warehouse#my_db.my_table"

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
