#!/usr/bin/env python3
"""
Example 11: Async Client

This example shows how to use Strata's async client for non-blocking
operations. The async client is ideal for:
- Web applications (FastAPI, aiohttp, etc.)
- Concurrent data fetching
- High-throughput data pipelines

What you'll learn:
    - How to use AsyncStrataClient
    - Concurrent fetches
    - Async context managers
"""

import asyncio
import sys
from pathlib import Path

from strata_client import AsyncStrataClient

# The table examples/setup_demo.py creates; pass another table URI as the first argument.
DEMO_WAREHOUSE = Path(__file__).resolve().parent.parent / "demo-warehouse"
table_uri = sys.argv[1] if len(sys.argv) > 1 else f"file://{DEMO_WAREHOUSE}#analytics.events"


# Helper to build filter specs for the transform params
def make_filter(column: str, op: str, value) -> dict:
    """Create a filter dict for the transform params."""
    return {"column": column, "op": op, "value": value}


async def basic_async_fetch():
    """Basic async fetch example."""
    async with AsyncStrataClient(base_url="http://127.0.0.1:8765") as client:
        # Check server health
        health = await client.health()
        print(f"Server status: {health['status']}")

        # Materialize with filters
        artifact = await client.materialize(
            inputs=[table_uri],
            transform={
                "executor": "scan@v1",
                "params": {
                    "columns": ["id", "value", "category"],
                    "filters": [make_filter("value", ">", 100.0)],
                },
            },
        )

        # Fetch the data
        table = await client.fetch(artifact.uri)
        print(f"Total rows: {table.num_rows}")

        # Or use the artifact's async helper
        df = await artifact.to_pandas()
        print(f"DataFrame shape: {df.shape}")


async def concurrent_fetches():
    """Fetch several projections of the table concurrently."""
    async with AsyncStrataClient() as client:
        # Define the projections to fetch
        tables = [
            (table_uri, ["id", "value"]),
            (table_uri, ["id", "category"]),
            (table_uri, ["id", "timestamp"]),
        ]

        # Create fetch tasks
        async def fetch_table(uri: str, columns: list[str]):
            artifact = await client.materialize(
                inputs=[uri],
                transform={
                    "executor": "scan@v1",
                    "params": {"columns": columns},
                },
            )
            return await client.fetch(artifact.uri)

        # Run concurrently
        results = await asyncio.gather(
            *[fetch_table(uri, cols) for uri, cols in tables],
            return_exceptions=True,
        )

        for (_, cols), result in zip(tables, results):
            if isinstance(result, Exception):
                print(f"{cols}: Error - {result}")
            else:
                print(f"{cols}: {result.num_rows} rows")


async def async_with_timeout():
    """Async fetch with custom timeout."""
    async with AsyncStrataClient() as client:
        try:
            # Set timeout on the materialize call
            artifact = await asyncio.wait_for(
                client.materialize(
                    inputs=[table_uri],
                    transform={"executor": "scan@v1", "params": {}},
                ),
                timeout=10.0,  # 10 second timeout
            )
            table = await client.fetch(artifact.uri)
            print(f"Got {table.num_rows} rows")
        except TimeoutError:
            print("Fetch timed out - try adding filters")


async def process_artifact_data():
    """Process artifact data asynchronously."""
    async with AsyncStrataClient() as client:
        # Materialize and fetch
        artifact = await client.materialize(
            inputs=[table_uri],
            transform={
                "executor": "scan@v1",
                "params": {"columns": ["value"]},
            },
        )
        table = await client.fetch(artifact.uri)

        # Process the data
        import pyarrow.compute as pc

        total_value = pc.sum(table.column("value")).as_py() or 0.0
        row_count = table.num_rows

        print(f"Processed {row_count} rows")
        print(f"Total value: {total_value:,.2f}")


async def main():
    """Run all examples."""
    print("=== Basic Async Fetch ===")
    await basic_async_fetch()

    print("\n=== Concurrent Fetches ===")
    await concurrent_fetches()

    print("\n=== Async with Timeout ===")
    await async_with_timeout()

    print("\n=== Process Artifact Data ===")
    await process_artifact_data()


if __name__ == "__main__":
    asyncio.run(main())
