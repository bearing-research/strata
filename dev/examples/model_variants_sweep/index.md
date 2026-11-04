# Model Variants Sweep

A minimal example of variant **sweep mode**: run every variant of a group and
compare them all in one downstream cell.

The `model` group has three variant cells, `always_one`, `alternating`,
`majority`: each producing `preds` a different way. Because the group is in
**sweep mode** (`mode = "sweep"` in `notebook.toml`), all three run on every
execution, and the downstream `compare` cell receives `preds` as a
`{variant_name: predictions}` dict:

```
accuracy by variant: {'alternating': 0.5, 'always_one': 0.6, 'majority': 0.6}
```

## Fan-out with `@per_variant`

The `compare` cell consumes the whole dict at once. The `score` cell shows the
other option, **fan-out**: `# @per_variant` runs it once per variant with
`preds` bound to that variant's list (a scalar, not the dict), so each variant
is scored independently (in a real workload, each could carry `# @worker gpu`
and run as its own job). A plain downstream cell, `best`, then collapses the
per-variant `accuracy` back to a dict and picks the winner:

```
accuracy by variant: {'alternating': 0.5, 'always_one': 0.6, 'majority': 0.6}
best model: always_one
```

Same numbers as `compare`, produced by fanning the scoring out first. See
[Variant Cells → Fan-out](https://bearingresearch.com/strata/latest/notebook/annotations/#fan-out-run-a-downstream-cell-once-per-variant).

## Run it

```bash
strata run examples/model_variants_sweep
```

Or open it in the UI (`python -m strata`), the `model` group renders as a tab
strip with a **sweep** badge; clicking a tab shows that variant's source (all
of them still run), and the group has a run-all button + a readiness rollup.

## Switch vs sweep

Flip `mode = "sweep"` to `mode = "switch"` (or drop the line) in
`notebook.toml` and only the active variant runs; `compare` would then see a
single `preds` value instead of the dict. Sweep is for *comparing* variants on
a shared downstream; switch is for *picking* one. See
[Variant Cells → Sweep mode](https://bearingresearch.com/strata/latest/notebook/annotations/#sweep-mode-compare-all-variants-at-once).

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
