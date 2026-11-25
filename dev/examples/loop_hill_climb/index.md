# Hill-climb Himmelblau's function, one cached step at a time

Four-code-cell notebook (plus a markdown cell) that exercises every moving part of Strata's loop
cell feature: the `@loop` / `@loop_until` annotations, per-iteration
artifacts, the progress badge, the iteration picker, and the
`@loop start_from` fork. The cell body is greedy hill climbing on
[Himmelblau's function](https://en.wikipedia.org/wiki/Himmelblau%27s_function),
small enough to reason about and visually interesting because there
are four equal-valued minima, so different seeds (and different forks)
converge to different basins.

---

## Helpers { .nb-cell }

```python
# @name Helpers
# A "module cell": it contains only imports and definitions, no top-level
# runtime state. Strata treats this kind of cell as a shareable module
# so downstream cells can ``import random`` / call ``himmelblau`` just
# by referencing the names. Mixing imports/defs with runtime state
# (``x = random.uniform(...)``) in the same cell blocks the share.
import random


def himmelblau(x: float, y: float) -> float:
    """The classic Himmelblau function. Four equal-valued minima at
    roughly (3, 2), (-2.8, 3.1), (-3.8, -3.3), (3.6, -1.8)."""
    return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2
```

## Seed State { .nb-cell }

```python
# @name Seed State
# Seed the hill-climb search from a random point on Himmelblau's
# surface. ``random`` and ``himmelblau`` come from the helpers cell
# via the DAG. Reusing them here keeps the example DRY and shows how
# a module cell's definitions flow through to runtime cells.
random.seed(42)
x = random.uniform(-5, 5)
y = random.uniform(-5, 5)
score = himmelblau(x, y)

state = {
    "x": x,
    "y": y,
    "best_score": score,
    # Step size shrinks each accepted move so the search sharpens as
    # it approaches a minimum.
    "step": 1.0,
    # history accumulates every proposal (accepted or rejected) so
    # the summary cell can show the whole trajectory, not just the
    # kept moves.
    "history": [{"iter": 0, "x": x, "y": y, "score": score, "accepted": True}],
    # iteration counter lets each loop iteration deterministically
    # seed its own RNG, so running the notebook twice reproduces the
    # same trajectory.
    "iter": 0,
}

print(f"Seed at ({x:.3f}, {y:.3f}) with score {score:.3f}")
state
```

```json title="Output"
{
  "best_score": 478.10046069787757,
  "history": [
    {
      "accepted": true,
      "iter": 0,
      "score": 478.10046069787757,
      "x": 1.3942679845788373,
      "y": -4.74989244777333
    }
  ],
  "iter": 0,
  "step": 1.0,
  "x": 1.3942679845788373,
  "y": -4.74989244777333
}
```

```text title="stdout"
Seed at (1.394, -4.750) with score 478.100
```

# The loop cell

The cell below carries `# @loop max_iter=40 carry=state`: Strata's
loop primitive. The cell body runs up to 40 times. On each iteration:

1. Strata reads the current `state` from the previous iteration's
   stored artifact (or from the upstream `seed` cell on iter 0).
2. The body executes, rebinding `state`.
3. Strata stores the new `state` as `…@iter=k`, so every step is a
   first-class artifact you can scrub back to.

The `# @loop_until` predicate fires when `state["best_score"]` drops
below `1e-3`, terminating early. Every intermediate iteration stays
queryable in the inspect panel, and a new loop cell can `start_from`
any of them to fork the search without re-running the prefix.

## Hill Climb { .nb-cell }

<sub>**loop** max_iter=40 carry=state</sub>

```python
# @name Hill Climb
# @loop max_iter=40 carry=state
# @loop_until state["best_score"] < 1e-3
#
# One iteration of greedy hill climbing on Himmelblau's function:
# perturb (x, y) by a small random step, keep the move if it strictly
# improves the score, otherwise record the proposal and keep the
# current best. The step size shrinks after every accepted move so the
# search naturally sharpens as it approaches a minimum.
#
# Every iteration stores the full `state` as its own artifact with an
# ``@iter=k`` suffix. The inspect panel's iteration picker lets you
# open any of them, and the returned URI can be pasted into a new
# loop cell's ``# @loop start_from=<cell>@iter=<k>`` to fork a
# different search strategy from a promising mid-iter state.
#
# `random` and `himmelblau` come from the helpers cell via the DAG, so
# there is no need to re-import or redefine them here.
next_iter = state["iter"] + 1
# Deterministic per-iteration RNG so the whole notebook is reproducible.
# random.Random only accepts int/str/bytes seeds on Python 3.11+, so we
# hash a per-iteration tuple down to an int.
rng = random.Random(hash((next_iter, round(state["x"], 6), round(state["y"], 6))))

step = state["step"]
candidate_x = state["x"] + rng.uniform(-step, step)
candidate_y = state["y"] + rng.uniform(-step, step)
candidate_score = himmelblau(candidate_x, candidate_y)

accepted = candidate_score < state["best_score"]

new_history_entry = {
    "iter": next_iter,
    "x": candidate_x,
    "y": candidate_y,
    "score": candidate_score,
    "accepted": accepted,
}

if accepted:
    state = {
        **state,
        "x": candidate_x,
        "y": candidate_y,
        "best_score": candidate_score,
        # Shrink the step on accepted moves to converge cleanly.
        "step": step * 0.9,
        "history": state["history"] + [new_history_entry],
        "iter": next_iter,
    }
    print(f"iter {next_iter}: accept → score {candidate_score:.4f} at ({candidate_x:.3f}, {candidate_y:.3f})")
else:
    state = {
        **state,
        "history": state["history"] + [new_history_entry],
        "iter": next_iter,
    }
    print(
        f"iter {next_iter}: reject (score {candidate_score:.4f} ≥ "
        f"best {state['best_score']:.4f})"
    )
```

```text title="stdout"
iter 1: accept → score 318.2285 at (2.284, -4.400)
iter 2: accept → score 152.7328 at (2.324, -3.596)
iter 3: accept → score 89.7289 at (2.470, -3.099)
iter 4: reject (score 166.4612 ≥ best 89.7289)
iter 5: accept → score 25.1563 at (3.148, -2.677)
iter 6: reject (score 48.2847 ≥ best 25.1563)
iter 7: reject (score 25.9849 ≥ best 25.1563)
iter 8: reject (score 51.4171 ≥ best 25.1563)
iter 9: reject (score 32.0542 ≥ best 25.1563)
iter 10: accept → score 12.2973 at (3.760, -2.589)
iter 11: reject (score 36.9591 ≥ best 12.2973)
iter 12: accept → score 2.0014 at (3.622, -2.189)
iter 13: reject (score 6.6491 ≥ best 2.0014)
iter 14: reject (score 2.0510 ≥ best 2.0014)
iter 15: reject (score 7.4480 ≥ best 2.0014)
iter 16: reject (score 6.0646 ≥ best 2.0014)
iter 17: reject (score 5.8389 ≥ best 2.0014)
iter 18: accept → score 0.3362 at (3.509, -1.887)
iter 19: reject (score 5.0110 ≥ best 0.3362)
iter 20: reject (score 3.1250 ≥ best 0.3362)
iter 21: reject (score 2.1808 ≥ best 0.3362)
iter 22: reject (score 17.0027 ≥ best 0.3362)
iter 23: reject (score 4.4482 ≥ best 0.3362)
iter 24: reject (score 0.8571 ≥ best 0.3362)
iter 25: reject (score 1.3200 ≥ best 0.3362)
iter 26: reject (score 7.9796 ≥ best 0.3362)
iter 27: reject (score 2.2638 ≥ best 0.3362)
iter 28: reject (score 0.8615 ≥ best 0.3362)
iter 29: reject (score 10.0107 ≥ best 0.3362)
iter 30: reject (score 3.7942 ≥ best 0.3362)
iter 31: reject (score 4.0234 ≥ best 0.3362)
iter 32: reject (score 4.0880 ≥ best 0.3362)
iter 33: accept → score 0.1114 at (3.609, -1.780)
iter 34: reject (score 11.7741 ≥ best 0.1114)
iter 35: reject (score 7.7162 ≥ best 0.1114)
iter 36: reject (score 6.7401 ≥ best 0.1114)
iter 37: reject (score 0.3459 ≥ best 0.1114)
iter 38: reject (score 2.5042 ≥ best 0.1114)
iter 39: reject (score 7.5657 ≥ best 0.1114)
iter 40: reject (score 5.6943 ≥ best 0.1114)
```

## Convergence Summary { .nb-cell }

```python
# @name Convergence Summary
# Read the loop cell's final state and render a compact convergence
# table. This is a regular downstream cell: it sees the *final*
# iteration's carry artifact via the normal DAG input path, identical
# to how any downstream cell reads any upstream variable.
accepted = [h for h in state["history"] if h.get("accepted")]
rejected = [h for h in state["history"] if not h.get("accepted")]

print(f"Final (x, y) = ({state['x']:.4f}, {state['y']:.4f})")
print(f"Final score  = {state['best_score']:.6f}")
print(f"Iterations   = {state['iter']}")
print(f"Accepted     = {len(accepted)}")
print(f"Rejected     = {len(rejected)}")
print()
print(f"{'iter':>4} {'x':>9} {'y':>9} {'score':>12} {'outcome':>10}")
for entry in state["history"][-10:]:
    outcome = "accept" if entry.get("accepted") else "reject"
    print(
        f"{entry['iter']:>4d} {entry['x']:>9.3f} {entry['y']:>9.3f} "
        f"{entry['score']:>12.4f} {outcome:>10}"
    )

# The last expression becomes the cell's display output.
{
    "final_x": state["x"],
    "final_y": state["y"],
    "final_score": state["best_score"],
    "iterations": state["iter"],
    "accepted": len(accepted),
    "rejected": len(rejected),
}
```

```json title="Output"
{
  "accepted": 9,
  "final_score": 0.1114213543102833,
  "final_x": 3.6094555292987467,
  "final_y": -1.7797668397786437,
  "iterations": 40,
  "rejected": 32
}
```

```text title="stdout"
Final (x, y) = (3.6095, -1.7798)
Final score  = 0.111421
Iterations   = 40
Accepted     = 9
Rejected     = 32

iter         x         y        score    outcome
  31     3.852    -1.861       4.0234     reject
  32     3.856    -1.904       4.0880     reject
  33     3.609    -1.780       0.1114     accept
  34     3.981    -1.502      11.7741     reject
  35     3.245    -2.158       7.7162     reject
  36     3.209    -1.626       6.7401     reject
  37     3.646    -1.762       0.3459     reject
  38     3.795    -1.827       2.5042     reject
  39     3.929    -1.692       7.5657     reject
  40     3.330    -2.192       5.6943     reject
```

---

## About this example

### What to look at

- **`helpers`** is a module cell: imports (``random``) and a
  reusable function (``himmelblau``). It declares no top-level
  runtime state, so Strata treats it as a shareable module whose
  names are directly referenceable from every downstream cell.
- **`seed`** bootstraps the search at a random `(x, y)` with an initial
  score, using ``random`` and ``himmelblau`` from ``helpers``. Its
  output variable `state` is the upstream carry for the loop cell
  below.
- **`evolve`** is the loop cell.
  - `# @loop max_iter=40 carry=state` caps the search at 40 iterations
    and tells Strata that the variable `state` is threaded between
    iterations.
  - `# @loop_until state["best_score"] < 1e-3` terminates the loop
    early as soon as the cell finds a minimum.
  - Each iteration proposes a small random perturbation of `(x, y)` and
    keeps the move if it improves the score. The step size shrinks on
    every accepted move so the search sharpens.
  - ``random`` and ``himmelblau`` come from the ``helpers`` cell via
    the DAG, with no duplicate imports or redefinitions here.
  - Every iteration stores the full `state` as its own artifact
    (`nb_loop-hill-climb-001_cell_evolve_var_state@iter={k}`).
- **`summary`** is a regular downstream cell. It reads the final
  iteration's `state` via the normal DAG input path and prints the
  convergence table.

### Running

From the project root:

```bash
uv run strata-notebook --notebook-dir examples
```

Open **Hill Climb Loop** from the notebook list on the Strata home page, then run the four code
cells in order. While `evolve` is running you should see:

- A `iter k/40` progress badge next to the cell title, with a spinner
  while it is still running.
- The badge settles to a green "done" state when `@loop_until` fires
  or `max_iter` is reached.

Click the inspect icon on `evolve` to open the inspect panel. The new
iteration picker lists every stored iteration with its size and
content type. Copy any iteration's URI to the clipboard for the fork
demo below.

### Try the fork

1. Add a new cell below `evolve` (the `+` button in the UI).
2. Paste the following, replacing `<iter-K>` with a mid-iter iteration
   you find interesting (e.g., iter 5 if the search is still making
   progress there):

```python
# @name Alt Search (forked)
# @loop max_iter=30 carry=state start_from=evolve@iter=<iter-K>
# @loop_until state["best_score"] < 1e-3
import random

next_iter = state["iter"] + 1
# Numbers only: hash() of a str changes from one process to the next.
rng = random.Random(hash((next_iter, round(state["x"], 6), round(state["y"], 6), 2)))

# Larger exploration step so the fork is meaningfully different.
step = state["step"] * 3.0
cx = state["x"] + rng.uniform(-step, step)
cy = state["y"] + rng.uniform(-step, step)
cs = (cx**2 + cy - 11) ** 2 + (cx + cy**2 - 7) ** 2

accepted = cs < state["best_score"]
entry = {"iter": next_iter, "x": cx, "y": cy, "score": cs, "accepted": accepted}
state = {
    **state,
    "x": cx if accepted else state["x"],
    "y": cy if accepted else state["y"],
    "best_score": min(cs, state["best_score"]),
    "step": state["step"] * (0.9 if accepted else 1.0),
    "history": state["history"] + [entry],
    "iter": next_iter,
}
print(f"alt iter {next_iter}: score {state['best_score']:.4f}")
```

The forked cell seeds iter 0 from `evolve`'s iter K and iterates from
there with a larger exploration step. The two cells share a history
prefix (iters 0..K) and diverge after.

### What this demo verifies

- `@loop` dispatching (non-loop cells unaffected).
- Carry seeding from an upstream cell on iter 0.
- Per-iteration carry seeding for k > 0 (each iter sees the previous
  iter's `state`, not the original seed).
- `@loop_until` early termination.
- Per-iteration artifacts stored with `@iter=k` suffix.
- WebSocket `cell_iteration_progress` messages driving the UI badge.
- `@loop start_from=<cell>@iter=<k>` seed resolution for forking.
- Inspect-panel iteration picker listing iterations and copying URIs.

### Notes

- Each iteration runs in a fresh subprocess (no warm-pool reuse). For
  a ~40-iteration loop expect ~40 × subprocess-startup cost (~1s
  each on this machine). For agentic LLM loops the per-iteration
  network call dwarfs that; for tight numerical loops the pool reuse
  is a future optimization.
- A loop cell runs on the notebook's own server, never on a remote
  worker: one whose `# @worker` (or the notebook's default worker) names
  another fails before it starts.
