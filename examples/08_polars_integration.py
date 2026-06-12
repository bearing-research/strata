#!/usr/bin/env python3
"""
Example 8: Polars Integration

This example shows how to use Strata with Polars for DataFrame operations.
Polars is Arrow-native, so the integration is typically zero-copy.

What you'll learn:
    - How to use the Polars extension module
    - Zero-copy Arrow interop
    - Using LazyFrames for deferred operations
    - Streaming with scan_batches()

Important: Polars filter operations (e.g., df.filter(...)) are applied
*after* data is fetched from Strata. For Strata-side pruning, pass
filters to the scan functions.
"""

import polars as pl

from strata_client import gt
from strata_client.integration.polars import StrataPolarsScanner, scan_to_lazy, scan_to_polars

table_uri = "file:///path/to/warehouse#my_db.events"

# Method 1: Simple one-shot scan
df = scan_to_polars(
    table_uri,
    columns=["id", "value", "category", "timestamp"],
    filters=[gt("value", 10.0)],  # Strata-side pruning
)

print("=== DataFrame from Strata ===")
print(df.head())

# Method 2: LazyFrame for deferred operations
# Note: Data is fetched eagerly, but downstream ops are lazy
lf = scan_to_lazy(
    table_uri,
    columns=["id", "value", "category"],
)

result = (
    lf.filter(pl.col("value") > 50)  # Polars-side filtering
    .group_by("category")
    .agg(
        [
            pl.count().alias("count"),
            pl.col("value").mean().alias("avg_value"),
        ]
    )
    .sort("count", descending=True)
    .head(10)
    .collect()  # Triggers Polars computation
)

print("\n=== Aggregation Result ===")
print(result)

# Method 3: Reusable scanner for multiple tables
with StrataPolarsScanner() as scanner:
    events = scanner.scan(
        table_uri,
        columns=["id", "category", "value"],
    )

    # Polars operations
    summary = events.group_by("category").agg(
        [
            pl.count().alias("count"),
            pl.col("value").sum().alias("total_value"),
        ]
    )
    print("\n=== Category Summary ===")
    print(summary.head())

# Method 4: Streaming with scan_batches()
# Batches are yielded incrementally for memory-efficient processing
with StrataPolarsScanner() as scanner:
    total_rows = 0
    for batch in scanner.scan_batches(table_uri, columns=["id", "value"]):
        # Process each batch incrementally
        df = pl.from_arrow(batch)
        total_rows += len(df)
    print(f"\n=== Streaming: processed {total_rows} rows ===")
