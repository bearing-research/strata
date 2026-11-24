# @variant model majority
# Predict the majority label of the training set. The helper name is private
# (leading underscore): every variant in a group must define the same names.
_majority = 1 if sum(y_true) * 2 >= len(y_true) else 0
preds = [_majority] * len(y_true)
