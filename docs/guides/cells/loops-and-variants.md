# Loops and variants

## Loop cells

A loop cell is a regular Python cell with a `# @loop` annotation. The body runs N times, with a **carry variable** threaded between iterations. Each iteration's state is stored as its own artifact, so you can inspect any intermediate step.

Use loop cells for iterative refinement (hill climbing, MCMC, training loops with checkpoints), simulations, and anything where you'd want to pause and inspect intermediate states, or fork a new run from a promising one.

### Minimal example

Two cells: a seed and a loop.

```python
# seed cell, initial carry state
state = {"x": 0.0, "best_score": float("inf"), "iter": 0}
```

```python
# loop cell
# @loop max_iter=40 carry=state
# @loop_until state["best_score"] < 1e-3
import random

# Each iteration: read `state`, compute the next step, rebind `state`.
candidate = state["x"] + random.uniform(-0.1, 0.1)
score = candidate**2  # some objective
if score < state["best_score"]:
    state = {**state, "x": candidate, "best_score": score, "iter": state["iter"] + 1}
else:
    state = {**state, "iter": state["iter"] + 1}
```

After execution, `state` holds the final iteration's value and every intermediate iteration is queryable.

The cell's display output (a figure, `display(...)`, a trailing expression) is what its final iteration displayed. It is saved like any Python cell's, so it is still there after a reopen, in `strata cell show` and in `strata export`. Earlier iterations' displays are not kept.

### Required directives

| Directive            | What it does                                                                                                                                         |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `# @loop max_iter=N` | Hard cap on iterations. Required: it is the safety bound on the loop.                                                                               |
| `# @loop carry=VAR`  | The variable threaded between iterations. Required. Must be re-bound (or updated in place) by the cell body each iteration, and seeded by an upstream cell on iteration 0. |

These can be on the same line: `# @loop max_iter=40 carry=state`.

### Optional directives

| Directive                          | What it does                                                                        |
| ---------------------------------- | ----------------------------------------------------------------------------------- |
| `# @loop_until <expr>`             | Early termination when `<expr>` is truthy (evaluated against the current `state`)   |
| `# @loop start_from=<cell>@iter=k` | Seed iteration 0 from a specific prior iteration's artifact, for forking runs |

### Per-iteration artifacts

Every iteration's carry value becomes its own artifact with an `@iter=k` suffix:

```
strata://artifact/nb_..._cell_<loop_id>_var_state@iter=0@v=1
strata://artifact/nb_..._cell_<loop_id>_var_state@iter=1@v=1
...
```

The inspect panel shows an iteration picker so you can scrub through the intermediate states. The **final** iteration's artifact is also the cell's canonical output (no `@iter` suffix); downstream cells read it via the normal DAG path.

### Output folders

A loop cell's [`# @output_dir`](../../reference/annotations/python.md#output_dir) folder lasts the
whole loop: each iteration sees what the earlier ones wrote, so a training loop
can add one checkpoint per iteration. After every iteration the folder is
stored as that iteration's `@iter=k` version, and the last one is the cell's
output. A file unchanged since the last iteration (same size, times and inode)
is not read again. When an iteration fails, what the folder held is kept as a
failed version. A loop started with `start_from=<cell>@iter=k` gets that
iteration's folder, when the run its carry came from kept one.

### Forking a loop

Intermediate iterations are first-class artifacts, so you can branch a new
run from any step of an old one without re-running the expensive prefix.

**Scenario.** You ran a hill-climbing search for 50 iterations. Glancing at
the inspect panel, iteration 17 looked like it was about to find a better
local optimum before the sampler drifted away. You want to explore what
happens if you push harder from that exact state with a different step size.

1. Open the loop cell's **Inspect** panel, scrub to iteration 17, copy its
   artifact URI. It'll look like
   `strata://artifact/nb_..._cell_hill_climb_var_state@iter=17@v=1`.
2. Add a new loop cell below. Reference the original cell's ID (not the full
   URI) in `start_from`:

   ```python
   # new loop cell, continues from iteration 17 of the previous run
   # @loop max_iter=20 carry=state start_from=hill_climb@iter=17
   state["step_size"] *= 0.5  # smaller steps from here on
   state = sample_and_score(state)
   ```

3. Run the new cell. It reads iteration 17's carry value as its seed, runs up
   to 20 more iterations under the modified strategy, and stores those
   iterations as its own artifact chain. The original run stays untouched.

You now have two parallel forks materialized in the artifact store. Either
one can be forked further, and the inspect panel shows both chains. The fork
is downstream of the cell it starts from: when that cell's result changes, the
fork and everything reading it run again.

This is the escape hatch for "that intermediate state looked promising, let
me explore from there": the thing that's hard to do in a plain for-loop
once you've thrown away the intermediates.

### When not to use a loop cell

- Tight `for` loops over short collections: a regular Python cell with a `for` loop is simpler and the extra per-iteration artifact overhead isn't worth it.
- Loops where intermediate state is genuinely disposable: store only the final answer in a regular Python cell.
- Anything that needs to branch out into multiple parallel runs: loop cells are sequential by design. Use separate cells, or model the fan-out in Python.

A loop cell checks the cache like any other cell, so running something downstream of a finished loop does not re-run it. A loop cell cannot declare `@fetch` or `@dataset`; read those in an upstream cell. Its `@table` names are bound in every iteration, at the snapshot its provenance records. A `@write_table` on a loop cell writes the final value once, after the last iteration.

A loop cell runs on the notebook's own server, never on a [remote worker](../../notebook/workers.md): one whose `# @worker`, or the notebook's default worker, names another fails before it starts. Add `# @worker local` to the loop cell, or put the work that needs the worker in an ordinary cell upstream.

Reach for loop cells when **being able to inspect or fork from iteration k matters**. That's the feature you're paying for.

## Variant cells

Variant cells let you keep multiple alternative implementations of the
same DAG slot side by side and switch between them. The canonical use
case is "we want to try three models for this experiment": three
training cells all produce a `model` variable, and downstream cells
reference `model` without caring which variant produced it.

```python
# @variant classifier logreg
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
```

```python
# @variant classifier rf
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
```

Both cells declare `# @variant <group> <name>` with the same group
(`classifier`) and different names (`logreg`, `rf`). At any given time
exactly one variant is **active**; only the active variant participates
in the DAG, so downstream cells see one producer for `model`. In the UI
the group renders as a tab strip: clicking a tab switches the active
variant, and the cell editor shows that variant's source.

### Switching variants

The active variant per group is persisted in `notebook.toml`:

```toml
[[variant_group]]
group = "classifier"
active = "rf"
```

Switching is a one-line diff. Each variant carries its own provenance
hash, so re-running a variant you've already trained is a cache hit:
flip-flopping between two variants is free after each has run once.
Downstream cells go stale on switch (their input artifact comes from
a different upstream cell) but become cache hits on the way back.

If `notebook.toml` doesn't pin a selection, or pins a name no cell
provides, the DAG falls back to the **first variant in source order**.
A `variant_active_unknown` diagnostic surfaces in the UI when the
selection drifts (e.g. you renamed a variant in source without updating
the toml entry).

### Sweep mode - compare all variants at once

Switch mode answers "which model do I pick?". **Sweep mode** answers
"how do all three compare on the same downstream pipeline?" - it runs
**every** variant on each group execution and hands a downstream cell the
whole set as a `{variant_name: value}` dict.

```toml
[[variant_group]]
group = "classifier"
mode = "sweep"        # default is "switch"; `active` is ignored in sweep
```

The variant cells are **unchanged** - same `# @variant <group> <name>`
annotation, same defines contract. Only the consumer changes: instead of
seeing one `model`, a downstream cell receives a dict keyed by variant
name and can compare them in one pass:

```python
# downstream cell - `preds` is {"logreg": ..., "rf": ..., "gbm": ...}
scores = {name: accuracy(p, y_test) for name, p in preds.items()}
```

Migrating switch → sweep is a one-line toml edit (no cell changes). In
the UI, the tab strip becomes a **display selector** (clicking a tab just
shows that variant's source; it no longer changes what runs), with a
group-level status rollup. Provenance includes the sorted
variant→artifact map, so **adding a variant only runs that one** and
renaming/removing a variant restalens the downstream as expected.

Caveats worth knowing: a variant that fails stops the downstream. Running a
consumer materialises every variant first, and the first one to fail fails the
consumer's run with that variant's error; the consumer does not run on a
partial dict. Fix or remove the failing variant and run again. A sweep group of
one is legal but pointless - you'd get a one-key dict. Sweep-group
members and their downstream consumers always run as **single-cell**
executions - they're excluded from run-all's shared-namespace batching, so a
sweep notebook doesn't get that speedup. This is deliberate: batching would let
one variant's in-namespace state leak into a sibling's, corrupting its
provenance, so sweep trades the batch speedup for per-variant isolation.

### Fan-out - run a downstream cell once per variant

Sweep mode hands a downstream cell the whole `{variant: value}` dict. Sometimes
you'd rather run the *downstream work itself* once per variant - score each
model on its own, tune each independently, dispatch each to its own GPU worker.
That's `# @per_variant`:

```python
# @per_variant
# Runs once per model variant. `preds` is bound to THAT variant's list
# (a scalar), not the whole dict - so this cell computes one accuracy.
accuracy = sum(p == t for p, t in zip(preds, y_true)) / len(y_true)
```

The cell runs N times (once per variant of its upstream sweep group), each
instance with the variant's scalar value bound. A cell that defines nothing,
one that only prints, plots or writes, fans out the same way. Its own output becomes
per-variant too, so a **downstream cell decides how to consume it**:

- Another `# @per_variant` cell continues the fan-out, zipping by variant name
  (instance `logreg` reads the upstream `logreg` value).
- A plain cell **collapses** it back to a `{variant: value}` dict - the same v1
  sweep machinery:

```python
# plain downstream - `accuracy` is {"logreg": 0.9, "rf": 0.88, ...}
best = max(accuracy, key=accuracy.get)
```

Each instance is an independent `materialize`: its own provenance, its own
cache entry, its own worker dispatch - so `# @per_variant` + `# @worker gpu`
fans out to N independent jobs. An instance's provenance covers only its own
variant's value, so adding a variant only runs the new instance and editing one
reruns only that variant's instance. A group the cell reads as a dict is an input
of every instance, so changing any of its variants reruns them all.

**Choosing the group.** Bare `# @per_variant` infers the group when the cell
reads from exactly one sweep group. It reads what the cells above it last
defined, so after a plain cell redefines `preds` there is no sweep left to fan
out over. If it reads from two or more, name the one
to fan out over - `# @per_variant model` - and the others collapse to dicts. A
cell fans out over **one** group; comparing across two groups (cartesian) isn't
supported. The validator flags the mistakes: `per_variant_no_sweep_source`
(nothing to fan out over), `per_variant_ambiguous_group` (bare form, ≥2 groups),
`per_variant_on_variant_member` (a cell can't both be a variant and fan out),
`per_variant_write_table_overwrite` (every variant would overwrite the same
table). The named group is part of the cell's cache key, so switching it runs the
instances again.

### Defines contract

All variants in a group must produce the same set of top-level
bindings. The validator compares each variant's `defines` against its
siblings and flags `variant_contract_mismatch` on any outlier. If
`logreg` exposes only `model` and `rf` exposes `model + feature_importance`,
downstream cells that depend on the missing name would break under one
selection but not the other.

Imports don't count toward the contract: they're scaffolding, not
interface. The variants above each bring in a different sklearn class,
which is fine; only the *values* the cells produce need to match.

### Mixing cell kinds

A variant group can mix any cell kinds. A Python variant and a prompt
variant can sit in the same group as long as they both produce the
contract names, e.g. one variant calls a deterministic regex
classifier, another asks an AI model to classify.

### Adding and removing variants

The variant tab strip carries a `+` button that clones the active
variant as a sibling. The new cell starts as a copy of the active
body with the `# @variant` line rewritten to an auto-generated name
(`<active>_copy`, then `_copy2`, `_copy3`, …). Rename happens by
editing the annotation line in source, the standard
annotation-as-truth pattern, no separate rename UI.

Deleting a variant tab removes only that variant. If you delete the
active one, the next variant in source order auto-promotes. Deleting
the last variant in a group removes the cell entirely *and* drops the
`[[variant_group]]` entry, the group dissolves.

### Bootstrapping

The first variant of a new group is created by typing the annotation:
add `# @variant <new_group> <variant_name>` to any existing cell, save
it, then use the `+` tab to add siblings. (There's no UI affordance for
the bootstrap step, source is the only place a group comes into
existence, which keeps `notebook.toml` honest about *which* groups
exist.)
