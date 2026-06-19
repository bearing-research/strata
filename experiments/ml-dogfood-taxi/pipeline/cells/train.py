# @name train_model
# @timeout 900
"""Train a tip-amount regressor on the materialized features."""

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

params = {
    "learning_rate": 0.1,
    "max_iter": 200,
    "max_leaf_nodes": 63,
    "random_state": 42,
}

X_train, X_test, y_train, y_test = train_test_split(
    features_df[FEATURES],
    features_df[TARGET],
    test_size=0.2,
    random_state=42,
)

model = HistGradientBoostingRegressor(**params).fit(X_train, y_train)

print(f"trained on {len(X_train):,} rows, holdout {len(X_test):,} rows")
print(f"params: {params}")
