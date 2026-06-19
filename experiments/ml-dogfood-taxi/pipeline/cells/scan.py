# @name scan_trips
# @timeout 600
# @table trips file:///Users/fangchenli/Workspace/strata/experiments/ml-dogfood-taxi/warehouse#nyc.trips
"""Materialize the snapshot-pinned scan, then feature-engineer locally.

The @table annotation makes this cell lake-aware: new data in nyc.trips
moves the snapshot, the cell goes stale, and a plain run (no --force)
recomputes everything downstream. `trips` / `trips_snapshot` are injected.
"""


COLUMNS = [
    "pickup_at",
    "dropoff_at",
    "passenger_count",
    "trip_distance",
    "pu_location_id",
    "do_location_id",
    "payment_type",
    "fare_amount",
    "tip_amount",
]
SAMPLE_ROWS = 300_000
SEED = 42


scan_art = strata.materialize(
    inputs=[trips],
    transform={
        "executor": "scan@v1",
        "params": {"columns": COLUMNS, "snapshot_id": trips_snapshot},
    },
    name="taxi/trips-raw",
)
trips_df = scan_art.to_pandas()
# Injected table variables live only in this cell's namespace — export the
# snapshot as a real define so downstream cells (promote's tags) can use it.
scanned_snapshot = trips_snapshot
print(f"scan artifact: {scan_art.uri} (cache_hit={scan_art.cache_hit})")
print(f"scanned {len(trips_df):,} rows at snapshot {trips_snapshot}")

# --- local feature engineering (card payments only: cash tips unrecorded) ---
mask = (
    (trips_df.payment_type == 1)
    & trips_df.fare_amount.between(1, 200)
    & trips_df.trip_distance.between(0.1, 100)
    & trips_df.tip_amount.between(0, 100)
    & (trips_df.dropoff_at > trips_df.pickup_at)
)
clean = trips_df[mask]

features_df = clean.assign(
    pickup_hour=clean.pickup_at.dt.hour,
    pickup_dow=clean.pickup_at.dt.dayofweek,
    duration_min=(clean.dropoff_at - clean.pickup_at).dt.total_seconds() / 60.0,
)[
    [
        "pickup_hour",
        "pickup_dow",
        "duration_min",
        "passenger_count",
        "trip_distance",
        "pu_location_id",
        "do_location_id",
        "fare_amount",
        "tip_amount",
    ]
]
if len(features_df) > SAMPLE_ROWS:
    features_df = features_df.sample(n=SAMPLE_ROWS, random_state=SEED)
features_df = features_df.reset_index(drop=True)

# persist features with provenance back to the scan artifact
feat_art = strata.put(
    inputs=[scan_art.uri],
    transform={
        "executor": "feature_eng@v1",
        "params": {"sample_rows": SAMPLE_ROWS, "seed": SEED, "card_only": True},
    },
    data=features_df,
    name="taxi/features",
)
features_uri = feat_art.uri

print(f"features artifact: {features_uri} (cache_hit={feat_art.cache_hit})")
print(f"training frame: {features_df.shape[0]:,} rows x {features_df.shape[1]} cols")
features_df.head()
