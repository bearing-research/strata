#!/usr/bin/env python3
"""Build the local Iceberg warehouse and load month 1 (creates snapshot S1).

Usage: uv run python experiments/ml-dogfood-taxi/setup_warehouse.py
"""

from pyiceberg.catalog.sql import SqlCatalog

from taxi_schema import DATA_DIR, TABLE_ID, TARGET, WAREHOUSE, load_month


def main() -> None:
    WAREHOUSE.mkdir(parents=True, exist_ok=True)
    catalog = SqlCatalog(
        "strata",
        uri=f"sqlite:///{WAREHOUSE / 'catalog.db'}",
        warehouse=str(WAREHOUSE),
    )
    try:
        catalog.create_namespace("nyc")
    except Exception:
        pass
    try:
        catalog.drop_table(TABLE_ID)
    except Exception:
        pass

    table = catalog.create_table(TABLE_ID, schema=TARGET)

    month1 = load_month(DATA_DIR / "yellow_tripdata_2024-01.parquet")
    table.append(month1)

    snapshot = table.current_snapshot()
    print(f"Loaded month 1: {month1.num_rows:,} rows")
    print(f"Snapshot S1: {snapshot.snapshot_id}")
    print(f"Table URI for Strata: file://{WAREHOUSE}#{TABLE_ID}")


if __name__ == "__main__":
    main()
