"""Shared schema + load helper for the NYC taxi dogfood warehouse."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"
WAREHOUSE = HERE / "warehouse"
TABLE_ID = "nyc.trips"

# Raw TLC column -> our canonical name
RENAMES = {
    "tpep_pickup_datetime": "pickup_at",
    "tpep_dropoff_datetime": "dropoff_at",
    "passenger_count": "passenger_count",
    "trip_distance": "trip_distance",
    "PULocationID": "pu_location_id",
    "DOLocationID": "do_location_id",
    "payment_type": "payment_type",
    "fare_amount": "fare_amount",
    "tip_amount": "tip_amount",
    "total_amount": "total_amount",
}

TARGET = pa.schema(
    [
        ("pickup_at", pa.timestamp("us")),
        ("dropoff_at", pa.timestamp("us")),
        ("passenger_count", pa.float64()),
        ("trip_distance", pa.float64()),
        ("pu_location_id", pa.int64()),
        ("do_location_id", pa.int64()),
        ("payment_type", pa.int64()),
        ("fare_amount", pa.float64()),
        ("tip_amount", pa.float64()),
        ("total_amount", pa.float64()),
    ]
)


def load_month(parquet_path: Path) -> pa.Table:
    """Read a raw TLC month, select + rename + cast to the canonical schema."""
    table = pq.read_table(parquet_path, columns=list(RENAMES))
    table = table.rename_columns([RENAMES[name] for name in table.column_names])
    return table.cast(TARGET)
