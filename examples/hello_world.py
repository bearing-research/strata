#!/usr/bin/env python3
"""
Strata Hello World Demo

This script demonstrates Strata's caching capabilities:
1. Creates a sample Iceberg table with 100K rows
2. Measures COLD run (no cache)
3. Measures WARM run (cache hit)
4. Simulates RESTART (clears in-memory cache, keeps disk cache)

Expected output:
  Cold run:     ~500ms (reading from Parquet files)
  Warm run:     ~50ms  (reading from Arrow IPC cache)
  Restart run:  ~60ms  (reading from disk cache)

Usage:
  # With local server:
  python examples/hello_world.py

  # With Docker:
  docker compose up -d
  python examples/hello_world.py --url http://localhost:8765
"""

import argparse
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

# Check dependencies
try:
    import pyarrow as pa
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField, StringType
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install pyiceberg[sql-sqlite] pyarrow")
    sys.exit(1)


def create_demo_warehouse(warehouse_path: Path) -> str:
    """Create a sample Iceberg warehouse with test data."""
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
        pass  # Already exists

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

    # Create sample data: 100K rows, ~10 row groups
    num_rows = 100_000
    base_ts = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000)

    print(f"  Generating {num_rows:,} rows...")
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

    table.append(data)
    print(f"  Created table with {num_rows:,} rows")

    return f"file://{warehouse_path}#{table_id}"


def run_benchmark(server_url: str, table_uri: str) -> dict:
    """Run the benchmark and return timing results."""
    # Import here to allow script to show helpful error if strata not installed
    try:
        from strata_client import StrataClient
    except ImportError:
        print("Strata not installed. Install with:")
        print("  pip install -e .")
        sys.exit(1)

    import httpx

    results = {}

    # Create client
    client = StrataClient(base_url=server_url)

    try:
        # Verify server is running
        try:
            _ = client.health()
            print(f"Connected to Strata server at {server_url}")
        except httpx.ConnectError:
            print(f"ERROR: Cannot connect to Strata server at {server_url}")
            print("Start the server with: strata-notebook")
            sys.exit(1)

        # Clear cache for fair comparison
        print("\nClearing cache for cold run...")
        client.clear_cache()

        # Helper to fetch table data using the unified materialize API
        def fetch_table():
            artifact = client.materialize(
                inputs=[table_uri],
                transform={"executor": "scan@v1", "params": {}},
            )
            return artifact.to_table()

        # COLD RUN
        print("\n[1/3] COLD RUN (no cache)...")
        start = time.perf_counter()
        table = fetch_table()
        cold_time = (time.perf_counter() - start) * 1000
        total_rows = table.num_rows
        results["cold_ms"] = cold_time
        print(f"  Time: {cold_time:.1f}ms")
        print(f"  Rows: {total_rows:,}")

        # WARM RUN
        print("\n[2/3] WARM RUN (cache hit)...")
        start = time.perf_counter()
        table = fetch_table()
        warm_time = (time.perf_counter() - start) * 1000
        results["warm_ms"] = warm_time
        print(f"  Time: {warm_time:.1f}ms")

        # Check metrics
        metrics = client.metrics()
        print(f"  Cache hits: {metrics.get('cache_hits', 0)}")

        # RESTART SIMULATION
        # We can't easily restart the server, but we can clear the in-memory
        # metadata caches while keeping the disk cache
        print("\n[3/3] RESTART RUN (disk cache only)...")
        # Clear metadata stats endpoint tells server to refresh caches
        # This simulates the effect of a restart where disk cache persists
        import requests

        requests.post(f"{server_url}/v1/metadata/cleanup")

        start = time.perf_counter()
        table = fetch_table()
        restart_time = (time.perf_counter() - start) * 1000
        results["restart_ms"] = restart_time
        print(f"  Time: {restart_time:.1f}ms")

    finally:
        client.close()

    return results


def print_summary(results: dict):
    """Print a summary of the benchmark results."""
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)

    cold = results.get("cold_ms", 0)
    warm = results.get("warm_ms", 0)
    restart = results.get("restart_ms", 0)

    print(f"\n  Cold run:     {cold:>8.1f} ms  (no cache)")
    print(f"  Warm run:     {warm:>8.1f} ms  (in-memory cache)")
    print(f"  Restart run:  {restart:>8.1f} ms  (disk cache)")

    if cold > 0 and warm > 0:
        speedup = cold / warm
        print(f"\n  Cache speedup: {speedup:.1f}x faster")

    print("\n" + "=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Strata Hello World Demo")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8765",
        help="Strata server URL (default: http://127.0.0.1:8765)",
    )
    parser.add_argument(
        "--warehouse",
        default=None,
        help="Path to warehouse directory (default: creates temp dir)",
    )
    parser.add_argument(
        "--table-uri",
        default=None,
        help="Use existing table URI instead of creating demo data",
    )
    args = parser.parse_args()

    print("=" * 50)
    print("STRATA HELLO WORLD DEMO")
    print("=" * 50)

    # Create demo warehouse if no table URI provided
    if args.table_uri:
        table_uri = args.table_uri
    else:
        if args.warehouse:
            warehouse_path = Path(args.warehouse)
            warehouse_path.mkdir(parents=True, exist_ok=True)
        else:
            # Use a consistent temp directory so restarts work
            warehouse_path = Path(tempfile.gettempdir()) / "strata-demo-warehouse"
            warehouse_path.mkdir(parents=True, exist_ok=True)

        table_uri = create_demo_warehouse(warehouse_path)

    print(f"\nTable URI: {table_uri}")
    print(f"Server:    {args.url}")

    # Run benchmark
    results = run_benchmark(args.url, table_uri)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
