# @name Evaluate
# Per-class precision, recall, and f1 against the test set.
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
