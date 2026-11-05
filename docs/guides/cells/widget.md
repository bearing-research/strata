# Widget cells

A **widget cell** is a declarative control panel - one control per line - that
feeds values into the DAG. Drag a slider or pick from a dropdown, and the cells
that read that value go stale; run them to see the result. Add one with the
**+** menu (**Widget**), or declare `language = "widget"` in `notebook.toml`.

```
# @name Controls
alpha = slider(0, 1, step=0.01, default=0.5)
epochs = number(default=10, min=1, max=200)
optimizer = dropdown(["adam", "sgd"], default="adam")
augment = checkbox(default=False)
label = text(default="baseline")
```

Each line `name = control(...)` **defines a variable** downstream cells consume,
exactly like a Python cell's output:

```python
# @name Train
model = train(data, lr=alpha, epochs=epochs, optimizer=optimizer)
report = evaluate(model)
report
```

## Controls

| Control | Value | Signature |
|---|---|---|
| `slider` | number | `slider(min, max, step=, default=)` |
| `number` | number | `number(default=, min=, max=)` |
| `dropdown` | string | `dropdown([...options], default=)` |
| `checkbox` | bool | `checkbox(default=)` |
| `text` | string | `text(default=)` |

Arguments must be literals. Omit `default=` and it's inferred (a slider's `min`,
a dropdown's first option, `False`, `""`). Omit a slider's `step=` and it's
derived from the range - roughly `(max - min) / 100`, snapped to a readable
`1`/`2`/`5` × 10ⁿ increment (so `slider(0, 1)` steps by `0.01`, `slider(0, 100)`
by `1`). Pass `step=` explicitly to override.

## How it works

A widget cell runs **no code and no subprocess** - it's declarative. Each
control's current value is stored as a content-addressed artifact, so:

- **Changing a control** re-materializes its value and marks downstream cells
  **stale** - the same signal as editing an upstream cell. Run the stale cells
  (or accept the cascade) to propagate the change.
- **Returning a control to a previous value is a cache hit** - downstream cells
  that already computed for that value don't recompute.
- The **declaration** (`slider(0, 1, …)`) is the committed cell file,
  `cells/<id>.widget`; the **current value** is runtime state in
  `.strata/runtime.json` (a drag never churns a committed file).
  It travels in a [snapshot](../../notebook/export.md#snapshots), so an imported copy computes
  the scenario the bundle was taken from rather than the declared defaults.

In the web UI a widget cell shows its controls in place of the code editor;
click **✎ Edit controls** to edit the declaration. Widget cells render in the
web UI (the CLI and read-only TUI show them as their declaration).

## Live mode

By default a control change marks downstream cells stale and you run them. Turn
on **⚡ Live** (a `# @live` annotation on the widget cell) and changing a control
**auto-runs** the cheap downstream cells instead. Cells whose last run was
expensive (over a cost threshold) are left stale rather than re-run on every
drag - so a slider feeding a quick plot updates live, while one feeding a long
training cell waits for a manual run.
