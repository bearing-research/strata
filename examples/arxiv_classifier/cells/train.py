# @name Train Classifier
# @worker local
# Train a logistic regression classifier: embeddings → topic label.
# Runs locally. Logistic regression on 3K × 384 takes <1s on CPU, so
# there is no reason to ship the data to a GPU worker and back.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = embeddings
y = sampled_papers["topic"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
classifier = LogisticRegression(max_iter=1000, n_jobs=-1)
classifier.fit(X_train, y_train)

train_acc = classifier.score(X_train, y_train)
test_acc = classifier.score(X_test, y_test)
print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy:  {test_acc:.3f}")
print(f"Classes: {list(classifier.classes_)}")

train_test_split_info = {
    "train_size": len(X_train),
    "test_size": len(X_test),
    "train_acc": round(train_acc, 4),
    "test_acc": round(test_acc, 4),
}
train_test_split_info
