# @variant model majority
# Predict the majority label of the training set.
majority_label = 1 if sum(y_true) * 2 >= len(y_true) else 0
preds = [majority_label] * len(y_true)
