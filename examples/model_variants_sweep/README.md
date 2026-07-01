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
[Variant Cells → Sweep mode](../../docs/notebook/annotations.md#sweep-mode--compare-all-variants-at-once).
