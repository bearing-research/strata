#!/usr/bin/env python3
"""
Example 9: S3 Storage Backend

This example shows how to use Strata with Iceberg tables stored in S3.
Strata supports S3-compatible storage (AWS S3, MinIO, LocalStack) read-only.

What you'll learn:
    - Where S3 credentials go: on the Strata server, not the client
    - How to use S3 table URIs

Prerequisites:
    - An Iceberg table in S3 (created with PyIceberg or Spark)
    - A Strata server started with S3 credentials (or MinIO/LocalStack for
      local testing)
"""

from strata_client import StrataClient

# The server reads the table, so S3 credentials belong to the server process.
# The client only needs the server's URL. Start the server with, for example:
#   STRATA_S3_REGION or AWS_REGION
#   STRATA_S3_ACCESS_KEY or AWS_ACCESS_KEY_ID
#   STRATA_S3_SECRET_KEY or AWS_SECRET_ACCESS_KEY
#   STRATA_S3_ENDPOINT_URL (for MinIO/LocalStack)
#   STRATA_S3_ANONYMOUS=true (for public buckets)
# or the matching s3_* keys under [tool.strata] in its pyproject.toml.

# S3 table URI format: s3://bucket/path/to/warehouse#namespace.table
table_uri = "s3://my-data-lake/warehouse#analytics.events"

# Use the client normally - S3 is transparent
client = StrataClient(base_url="http://127.0.0.1:8765")


# Helper to build filter specs for the transform params
def make_filter(column: str, op: str, value) -> dict:
    """Create a filter dict for the transform params."""
    return {"column": column, "op": op, "value": value}


# Materialize with filters (row-group pruning works with S3 too!)
artifact = client.materialize(
    inputs=[table_uri],
    transform={
        "executor": "scan@v1",
        "params": {
            "columns": ["id", "value", "timestamp"],
            "filters": [make_filter("value", ">", 100.0)],
        },
    },
)

# Fetch the data
table = client.fetch(artifact.uri)
print(f"Retrieved {table.num_rows} rows from S3")
print(f"Columns: {table.schema.names}")

# Collect to Arrow table with different columns
artifact = client.materialize(
    inputs=[table_uri],
    transform={
        "executor": "scan@v1",
        "params": {"columns": ["id", "value"]},
    },
)
table = client.fetch(artifact.uri)
print(f"\nTotal rows: {table.num_rows}")

client.close()

# --- Using with MinIO (local S3-compatible storage) ---
#
# For local development, MinIO provides S3-compatible storage:
#
#   docker run -p 9000:9000 -p 9001:9001 minio/minio server /data --console-address ":9001"
#
# Then start the Strata server with:
#   export STRATA_S3_ENDPOINT_URL="http://localhost:9000"
#   export AWS_ACCESS_KEY_ID="minioadmin"
#   export AWS_SECRET_ACCESS_KEY="minioadmin"
#
# Create your Iceberg table using PyIceberg with the same S3 config,
# then Strata can serve it.

# --- Public S3 Buckets ---
#
# For public data (read-only), start the server with anonymous access:
#   export STRATA_S3_ANONYMOUS=true
