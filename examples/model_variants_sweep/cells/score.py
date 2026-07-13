# @per_variant
# Fan-out: runs once per model variant. `preds` is bound to THAT variant's
# predictions (a scalar list), not the whole {variant: preds} dict — so this
# cell computes a single accuracy. In a real workload each instance could carry
# `# @worker gpu` and run as an independent job.
accuracy = sum(p == t for p, t in zip(preds, y_true)) / len(y_true)
