# `preds` is a dict {variant_name: predictions} because the `model` group is in
# sweep mode — every variant ran, and this cell consumes them all at once.
accuracy = {
    name: sum(p == t for p, t in zip(pred, y_true)) / len(y_true)
    for name, pred in preds.items()
}
print("accuracy by variant:", accuracy)
