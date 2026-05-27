# @name Synthesize a small housing dataset
#
# 240 rows, four features (sqft, bedrooms, age, location), one target
# (price). Generated with a known linear-with-noise structure so both
# R and sklearn should recover similar coefficients — the demo is
# about *how* each toolkit expresses the fit, not which is more
# accurate.
#
# Split into train (200) + test (40). The test split is held out so
# downstream cells can compare R's predicted prices against sklearn's
# on the same observations.

import numpy as np
import pandas as pd

rng = np.random.default_rng(seed=42)
n = 240

sqft = rng.uniform(600, 3200, size=n).round(0)
bedrooms = rng.integers(1, 6, size=n)
age = rng.uniform(0, 80, size=n).round(1)
location = rng.choice(
    ["downtown", "suburb", "rural"],
    size=n,
    p=[0.35, 0.5, 0.15],
)

# Known generating coefficients (in $1k units): intercept 50,
# sqft 0.18, bedrooms 15, age -1.2, downtown +85, suburb 0,
# rural -45. Gaussian noise σ=25.
location_premium = pd.Series(location).map(
    {"downtown": 85.0, "suburb": 0.0, "rural": -45.0}
).to_numpy()
price_thousands = (
    50
    + 0.18 * sqft
    + 15 * bedrooms
    - 1.2 * age
    + location_premium
    + rng.normal(0, 25, size=n)
).round(2)

housing = pd.DataFrame(
    {
        "sqft": sqft,
        "bedrooms": bedrooms,
        "age": age,
        "location": location,
        "price": price_thousands,
    }
)

# Hold out the last 40 rows as a test set. Deterministic split — no
# need to re-shuffle since the rows are already in random order.
housing_train = housing.iloc[:200].reset_index(drop=True)
housing_test = housing.iloc[200:].reset_index(drop=True)

print(f"train: {len(housing_train)} rows, test: {len(housing_test)} rows")
print(housing_train.head())
