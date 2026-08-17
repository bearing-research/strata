# Downstream of `model`. Edit this cell and only this cell recomputes — the
# model above stays cached (no retraining).
from sklearn.metrics import confusion_matrix

accuracy = float(model.score(X, y))
conf_matrix = confusion_matrix(y, model.predict(X))
