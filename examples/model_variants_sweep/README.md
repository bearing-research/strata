# Model Variants Sweep

A minimal example of variant **sweep mode**: run every variant of a group and
compare them all in one downstream cell.

The `model` group has three variant cells — `always_one`, `alternating`,
`majority` — each producing `preds` a different way. Because the group is in
**sweep mode** (`mode = "sweep"` in `notebook.toml`), all three run on every
execution, and the downstream `compare` cell receives `preds` as a
`{variant_name: predictions}` dict:

```
accuracy by variant: {'alternating': 0.5, 'always_one': 0.6, 'majority': 0.6}
```

## Fan-out with `@per_variant`

The `compare` cell consumes the whole dict at once. The `score` cell shows the
other option — **fan-out**: `# @per_variant` runs it once per variant with
`preds` bound to that variant's list (a scalar, not the dict), so each variant
is scored independently (in a real workload, each could carry `# @worker gpu`
and run as its own job). A plain downstream cell, `best`, then collapses the
per-variant `accuracy` back to a dict and picks the winner:

```
accuracy by variant: {'alternating': 0.5, 'always_one': 0.6, 'majority': 0.6}
best model: always_one
```

Same numbers as `compare`, produced by fanning the scoring out first. See
[Variant Cells → Fan-out](https://bearing-research.github.io/strata/notebook/annotations/#fan-out--run-a-downstream-cell-once-per-variant).

## Run it

```bash
strata run examples/model_variants_sweep
```

Or open it in the UI (`python -m strata`) — the `model` group renders as a tab
strip with a **sweep** badge; clicking a tab shows that variant's source (all
of them still run), and the group has a run-all button + a readiness rollup.

## Switch vs sweep

Flip `mode = "sweep"` to `mode = "switch"` (or drop the line) in
`notebook.toml` and only the active variant runs; `compare` would then see a
single `preds` value instead of the dict. Sweep is for *comparing* variants on
a shared downstream; switch is for *picking* one. See
[Variant Cells → Sweep mode](https://bearing-research.github.io/strata/notebook/annotations/#sweep-mode--compare-all-variants-at-once).
