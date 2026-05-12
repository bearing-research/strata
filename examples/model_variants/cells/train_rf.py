# @variant classifier rf
# @name Random forest
# Random forest — more capacity, slower to train.
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print(f"Trained {type(model).__name__} (train acc {model.score(X_train, y_train):.3f})")
