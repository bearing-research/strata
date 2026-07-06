# A plain (non-fan-out) downstream cell collapses the per-variant `accuracy`
# back into a {variant: value} dict — the same shape the `compare` cell gets,
# but produced by fanning the scoring out first. Pick the winner:
best_model = max(accuracy, key=accuracy.get)
print("accuracy by variant:", accuracy)
print("best model:", best_model)
