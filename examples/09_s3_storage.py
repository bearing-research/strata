#!/usr/bin/env python3
"""
Example 9: S3 Storage Backend

This example shows how to use Strata with Iceberg tables stored in S3.
Strata supports S3-compatible storage (AWS S3, MinIO, LocalStack) read-only.

What you'll learn:
    - How to configure S3 credentials
    - How to use S3 table URIs
    - Using environment variables for AWS credentials

Prerequisites:
    - An Iceberg table in S3 (created with PyIceberg or Spark)
    - AWS credentials configured (or MinIO/LocalStack for local testing)
"""

import os

from strata_client import StrataClient
from strata.config import StrataConfig

# Method 1: Configure S3 via StrataConfig
# Useful for programmatic configuration
config = StrataConfig(
    s3_region="us-east-1",
    s3_access_key="your-access-key",
    s3_secret_key="your-secret-key",
    # For MinIO or LocalStack, set endpoint:
    # s3_endpoint_url="http://localhost:9000",
)

# Method 2: Use environment variables (recommended for production)
# Strata reads from AWS standard variables:
#   AWS_REGION or STRATA_S3_REGION
#   AWS_ACCESS_KEY_ID or STRATA_S3_ACCESS_KEY
#   AWS_SECRET_ACCESS_KEY or STRATA_S3_SECRET_KEY
#   STRATA_S3_ENDPOINT_URL (for MinIO/LocalStack)
#   STRATA_S3_ANONYMOUS=true (for public buckets)

# Example with environment variables:
os.environ["AWS_REGION"] = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"] = "your-access-key"
os.environ["AWS_SECRET_ACCESS_KEY"] = "your-secret-key"

# Load config from environment
config = StrataConfig.load()

# S3 table URI format: s3://bucket/path/to/warehouse#namespace.table
table_uri = "s3://my-data-lake/warehouse#analytics.events"

# Use the client normally - S3 is transparent
client = StrataClient(config=config, base_url="http://127.0.0.1:8765")


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
# Then configure Strata:
#   export STRATA_S3_ENDPOINT_URL="http://localhost:9000"
#   export AWS_ACCESS_KEY_ID="minioadmin"
#   export AWS_SECRET_ACCESS_KEY="minioadmin"
#
# Create your Iceberg table using PyIceberg with the same S3 config,
# then Strata can serve it.

# --- Public S3 Buckets ---
#
# For public data (read-only), use anonymous access:
#   config = StrataConfig(s3_anonymous=True)
#
# Or via environment:
#   export STRATA_S3_ANONYMOUS=true
