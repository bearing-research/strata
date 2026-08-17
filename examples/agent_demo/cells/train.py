# The expensive step: fit the model. Strata caches this by provenance, so a
# downstream edit reuses the trained model instead of retraining it.
import numpy as np
from sklearn.ensemble import RandomForestClassifier

rng = np.random.default_rng(0)
X = rng.normal(size=(8_000, 20))
y = (X[:, 0] + X[:, 1] > 0).astype(int)

model = RandomForestClassifier(
    n_estimators=100, max_depth=8, n_jobs=-1, random_state=0
).fit(X, y)
