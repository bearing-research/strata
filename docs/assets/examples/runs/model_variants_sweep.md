<!-- strata example run, source dff863471efa4ab1 -->
## data

<sub>**kind** python</sub>

```python
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
```

## baseline

<sub>**kind** python  ·  **variant** always_one of model</sub>

```python
# @variant model always_one
# A trivial baseline "classifier": predict 1 for everything.
preds = [1] * len(y_true)
```

## compare

<sub>**kind** python</sub>

```python
# `preds` is a dict {variant_name: predictions} because the `model` group is in
# sweep mode: every variant ran, and this cell consumes them all at once.
accuracy = {
    name: sum(p == t for p, t in zip(pred, y_true)) / len(y_true)
    for name, pred in preds.items()
}
print("accuracy by variant:", accuracy)
```

```text title="stdout"
accuracy by variant: {'alternating': 0.5, 'always_one': 0.6, 'majority': 0.6}
```

## score

<sub>**kind** python</sub>

```python
# @per_variant
# Fan-out: runs once per model variant. `preds` is bound to THAT variant's
# predictions (a scalar list), not the whole {variant: preds} dict, so this
# cell computes a single accuracy. In a real workload each instance could carry
# `# @worker gpu` and run as an independent job.
accuracy = sum(p == t for p, t in zip(preds, y_true)) / len(y_true)
```

## best

<sub>**kind** python</sub>

```python
# A plain (non-fan-out) downstream cell collapses the per-variant `accuracy`
# back into a {variant: value} dict, the same shape the `compare` cell gets,
# but produced by fanning the scoring out first. Pick the winner:
best_model = max(accuracy, key=accuracy.get)
print("accuracy by variant:", accuracy)
print("best model:", best_model)
```

```text title="stdout"
accuracy by variant: {'alternating': 0.5, 'always_one': 0.6, 'majority': 0.6}
best model: always_one
```
