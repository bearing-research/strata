# @variant classifier gbm
# Gradient boosting — sequential weak learners.
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print(f"Trained {type(model).__name__} (train acc {model.score(X_train, y_train):.3f})")
