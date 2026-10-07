#!/usr/bin/env python3
"""
Setup script for Strata demo.

Creates a demo Iceberg warehouse with sample data that can be used
with docker-compose or the hello_world.py script.

Usage (from a Strata checkout):
  uv run python examples/setup_demo.py
"""

import sys
from datetime import UTC, datetime
from pathlib import Path

try:
    import pyarrow as pa
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField, StringType
except ImportError:
    print("Missing dependencies. Run it from a Strata checkout with:")
    print("  uv run python examples/setup_demo.py")
    sys.exit(1)


def main():
    # Create warehouse in project root
    project_root = Path(__file__).parent.parent
    warehouse_path = project_root / "demo-warehouse"
    warehouse_path.mkdir(parents=True, exist_ok=True)

    print(f"Creating demo warehouse at {warehouse_path}...")

    # Create catalog
    # Use "strata" to match PyIcebergCatalog's expected name
    catalog = SqlCatalog(
        "strata",
        **{
            "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
            "warehouse": str(warehouse_path),
        },
    )

    # Create namespace
    try:
        catalog.create_namespace("analytics")
    except Exception:
        pass

    # Define schema
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
        NestedField(3, "category", StringType(), required=False),
        NestedField(4, "timestamp", LongType(), required=False),
    )

    # Create or replace table
    table_id = "analytics.events"
    try:
        catalog.drop_table(table_id)
    except Exception:
        pass

    table = catalog.create_table(table_id, schema)

    # Create sample data
    num_rows = 100_000
    base_ts = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000)

    print(f"Generating {num_rows:,} rows...")
    data = pa.table(
        {
            "id": pa.array(range(num_rows), type=pa.int64()),
            "value": pa.array([float(i * 0.01) for i in range(num_rows)], type=pa.float64()),
            "category": pa.array([f"cat_{i % 100}" for i in range(num_rows)], type=pa.string()),
            "timestamp": pa.array(
                [base_ts + i * 1000 for i in range(num_rows)],
                type=pa.int64(),
            ),
        }
    )

    # Two appends, so the table has two snapshots for examples/04_time_travel.py.
    half = num_rows // 2
    table.append(data.slice(0, half))
    table.append(data.slice(half))

    table_uri = f"file://{warehouse_path}#analytics.events"

    print("\nDemo warehouse created!")
    print(f"  Location: {warehouse_path}")
    print(f"  Table:    {table_id}")
    print(f"  Rows:     {num_rows:,}")
    print("\nTable URI for Strata:")
    print(f"  {table_uri}")
    print("\nNext steps:")
    print("  1. Start the server: strata-notebook")
    print("  2. Run a script:     uv run python examples/01_basic_usage.py")


if __name__ == "__main__":
    main()
