# @name train_model
# @timeout 900
"""Train a tip-amount regressor on the materialized features, one checkpoint per stage.

Each stage adds trees to the same model and publishes a checkpoint, so a run
that dies (crash, cancel, server restart) continues from its last checkpoint on
the next run instead of starting over, and the best stage so far is the `model`
head of the run.
"""

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

FEATURES = [
    "pickup_hour",
    "pickup_dow",
    "duration_min",
    "passenger_count",
    "trip_distance",
    "pu_location_id",
    "do_location_id",
    "fare_amount",
]
TARGET = "tip_amount"
STAGES, TREES_PER_STAGE = 8, 25

params = {
    "learning_rate": 0.1,
    "max_leaf_nodes": 63,
    "random_state": 42,
    "warm_start": True,
}

X_train, X_test, y_train, y_test = train_test_split(
    features_df[FEATURES],
    features_df[TARGET],
    test_size=0.2,
    random_state=42,
)

resumed = strata.restore()
if resumed is None:
    model = HistGradientBoostingRegressor(max_iter=TREES_PER_STAGE, **params)
    metrics = {"stage": 0, "trees": 0, "r2": None, "best_r2": None}
else:
    model, metrics = resumed["model"], resumed["metrics"]
    print(f"resumed after stage {metrics['stage']} ({metrics['trees']} trees)")

for stage in range(metrics["stage"] + 1, STAGES + 1):
    model.max_iter = stage * TREES_PER_STAGE
    model.fit(X_train, y_train)
    r2 = float(model.score(X_test, y_test))
    best = metrics["best_r2"] is None or r2 > metrics["best_r2"]
    metrics = {
        "stage": stage,
        "trees": int(model.n_iter_),
        "r2": r2,
        "best_r2": r2 if best else metrics["best_r2"],
    }
    sequence = strata.checkpoint(model=model, metrics=metrics)
    if best:
        strata.head("model", sequence)
    print(f"stage {stage}: {metrics['trees']} trees, holdout r2 {r2:.4f}")

print(f"trained on {len(X_train):,} rows, holdout {len(X_test):,} rows")
print(f"params: {params}")
