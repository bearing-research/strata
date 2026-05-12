# @name Evaluate active classifier
# Test-set classification report — works for any classifier.
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
test_acc = model.score(X_test, y_test)

print(f"=== {type(model).__name__} ===")
print(f"Test accuracy: {test_acc:.3f}\n")
print(classification_report(y_test, y_pred))
