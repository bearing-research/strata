# @name Confusion Matrix
# @worker local
# Visualize the confusion matrix on the held-out test set.
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

X = embeddings
y = sampled_papers["topic"].to_numpy()
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

classes = sorted(set(y))
predictions = classifier.predict(X_test)
cm = confusion_matrix(y_test, predictions, labels=classes)

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay(cm, display_labels=classes).plot(
    ax=ax, cmap="Blues", colorbar=False, xticks_rotation=30
)
ax.set_title("arXiv Topic Classification: Confusion Matrix")
plt.tight_layout()
fig
