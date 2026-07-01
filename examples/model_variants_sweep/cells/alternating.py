# @variant model alternating
# Alternate 1, 0, 1, 0, ... regardless of the data.
preds = [i % 2 for i in range(len(y_true))]
