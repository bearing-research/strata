# Model the survivors

We've engineered features from the raw passenger data. Now train
three classifiers (logistic regression, random forest, gradient
boosting), evaluate the best one, and compare feature importance.

Each classifier runs once and gets cached. Tweaking
`features.py` invalidates all three downstream cells; the
provenance hash on each training cell changes because its `X_train`
input changed.
