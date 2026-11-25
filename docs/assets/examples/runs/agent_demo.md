<!-- strata example run, source eec478c7249d0988 -->
# Agent demo: a coding agent builds this notebook

A random forest, trained once and cached. Because Strata is content-addressed,
editing the evaluation below recomputes only that cell, the model is never
retrained.

## train

<sub>**kind** python</sub>

```python
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
```

## evaluate

<sub>**kind** python</sub>

```python
# Downstream of `model`. Edit this cell and only this cell recomputes; the
# model above stays cached (no retraining).
from sklearn.metrics import confusion_matrix

accuracy = float(model.score(X, y))
conf_matrix = confusion_matrix(y, model.predict(X))
```
