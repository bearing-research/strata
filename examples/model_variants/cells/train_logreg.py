# @variant classifier logreg
# Logistic regression baseline.
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

print(f"Trained {type(model).__name__} (train acc {model.score(X_train, y_train):.3f})")
