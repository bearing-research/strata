#!/usr/bin/env python3
"""Append month 2 to the warehouse (creates snapshot S2). Run AFTER the
month-1 pipeline has trained + promoted, to exercise staleness detection.

Usage: uv run python experiments/ml-dogfood-taxi/append_month2.py
"""

from pyiceberg.catalog.sql import SqlCatalog

from taxi_schema import DATA_DIR, TABLE_ID, WAREHOUSE, load_month


def main() -> None:
    catalog = SqlCatalog(
        "strata",
        uri=f"sqlite:///{WAREHOUSE / 'catalog.db'}",
        warehouse=str(WAREHOUSE),
    )
    table = catalog.load_table(TABLE_ID)
    before = table.current_snapshot().snapshot_id

    month2 = load_month(DATA_DIR / "yellow_tripdata_2024-02.parquet")
    table.append(month2)

    after = table.current_snapshot().snapshot_id
    print(f"Appended month 2: {month2.num_rows:,} rows")
    print(f"Snapshot S1 (before): {before}")
    print(f"Snapshot S2 (after):  {after}")


if __name__ == "__main__":
    main()
