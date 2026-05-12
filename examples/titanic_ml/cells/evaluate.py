# @name Evaluate best model
# Detailed evaluation of the best model.
from sklearn.metrics import classification_report

best_name = max(results, key=lambda k: results[k]["test"])
best_model = results[best_name]["model"]
y_pred = best_model.predict(X_test)

print(f"Best model: {best_name} (test accuracy: {results[best_name]['test']:.3f})\n")
print(classification_report(y_test, y_pred, target_names=["Did not survive", "Survived"]))
