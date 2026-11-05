# Loop and variant annotations

A Python cell carrying `@loop` is executed iteratively. The body runs once per
iteration and the `carry` variable threads state between them.

## `@loop`

```python
# @loop max_iter=50 carry=state
# @loop_until state["converged"]
state = step(state)
```

Iteration 0 reads `state` from the upstream cell that defines it (or from
`start_from=`, below); the body must rebind it, or update it in place, every
iteration.

Key/value parameters:

- `max_iter=<N>`, hard upper bound on iterations.
- `carry=<var>`, the variable threaded between iterations.
- `start_from=<cell-id>@iter=<k>`, (optional) resume from another loop cell's
  stored iteration `k`. Useful for forking a converged run to explore a
  variant. `<cell-id>` is the upstream loop cell's `id` in
  `notebook.toml` (not its `@name`) - see [Cell IDs](python.md#cell-ids). The fork
  depends on that cell: when its result changes, the fork runs again. It reads
  the carry from there only, so another cell that defines the carry is not an
  input of the fork. A `start_from` that names no cell, a variant that is not
  active, or a cell that does not define the carry is an error
  (`loop_start_from_unknown`), and the fork does not run.
  Iteration `k` must come from that cell's latest run: if the latest run
  stopped before `k` (a lower `max_iter`, an earlier `@loop_until` exit, a
  failure), the fork fails rather than seed from a step an older run left behind.
- `until=<expr>`, (optional) the same early exit as `@loop_until`, inline.
  Parameters are split on whitespace, so the expression cannot contain a
  space (`until=acc["loss"]<0.05`); use `@loop_until` for anything longer.

## `@loop_until`

Python expression evaluated after each iteration in the cell's namespace. When
it returns truthy, the loop exits early. A name it reads that the body does not
bind (a tolerance set upstream) is an input of the cell, like any other
reference. An expression that raises fails the cell at that iteration.

```python
# @loop max_iter=100 carry=acc
# @loop_until acc["loss"] < 0.05
```

Each iteration's carry state is stored as `…@iter=k` artifacts; the final
iteration becomes the cell's canonical artifact. Progress is broadcast over
WebSocket as `cell_iteration_progress` messages. The `@loop` and `@loop_until`
parameters are part of the cell's cache key, so editing one (say `max_iter`)
runs the loop again rather than returning the old result.

## `@variant`

`# @variant <group> <name>` marks the cell as one alternative in a group, so
several cells can fill the same slot in the DAG. See
[Variant Cells](../../guides/cells/loops-and-variants.md#variant-cells).

## `@per_variant`

`# @per_variant` runs a downstream cell once per variant, bound to that variant's
value instead of the whole `{variant: value}` dict. See
[Fan-out](../../guides/cells/loops-and-variants.md#fan-out-run-a-downstream-cell-once-per-variant).
