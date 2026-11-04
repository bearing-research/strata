# Loop Hill Climb: demo for the `@loop` cell primitive

Four-code-cell notebook (plus a markdown cell) that exercises every moving part of Strata's loop
cell feature: the `@loop` / `@loop_until` annotations, per-iteration
artifacts, the progress badge, the iteration picker, and the
`@loop start_from` fork. The cell body is greedy hill climbing on
[Himmelblau's function](https://en.wikipedia.org/wiki/Himmelblau%27s_function) , 
small enough to reason about and visually interesting because there
are four equal-valued minima, so different seeds (and different forks)
converge to different basins.

## What to look at

- **`helpers`** is a module cell, imports (``random``) and a
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
    the DAG, no duplicate imports or redefinitions here.
  - Every iteration stores the full `state` as its own artifact
    (`nb_loop-hill-climb-001_cell_evolve_var_state@iter={k}`).
- **`summary`** is a regular downstream cell. It reads the final
  iteration's `state` via the normal DAG input path and prints the
  convergence table.

## Running

```bash
uv run strata-notebook
```

Open the notebook from the Strata home page, then run the four code
cells in order. While `evolve` is running you should see:

- A `iter k/40` progress badge next to the cell title, with a spinner
  while it is still running.
- The badge settles to a green "done" state when `@loop_until` fires
  or `max_iter` is reached.

Click the inspect icon on `evolve` to open the inspect panel. The new
iteration picker lists every stored iteration with its size and
content type. Copy any iteration's URI to the clipboard for the fork
demo below.

## Try the fork

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

## What this demo verifies

- `@loop` dispatching (non-loop cells unaffected).
- Carry seeding from an upstream cell on iter 0.
- Per-iteration carry seeding for k > 0 (each iter sees the previous
  iter's `state`, not the original seed).
- `@loop_until` early termination.
- Per-iteration artifacts stored with `@iter=k` suffix.
- WebSocket `cell_iteration_progress` messages driving the UI badge.
- `@loop start_from=<cell>@iter=<k>` seed resolution for forking.
- Inspect-panel iteration picker listing iterations and copying URIs.

## Notes

- Each iteration runs in a fresh subprocess (no warm-pool reuse). For
  a ~40-iteration loop expect ~40 × subprocess-startup cost (~1s
  each on this machine). For agentic LLM loops the per-iteration
  network call dwarfs that; for tight numerical loops the pool reuse
  is a future optimization.
- The loop cell runs only on the `local` worker in Phase 1. Remote
  worker support is a later phase.

## Helpers

<sub>**kind** python</sub>

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

## Seed State

<sub>**kind** python</sub>

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

## Hill Climb

<sub>**kind** python  ·  **loop** max_iter=40 carry=state</sub>

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

## Convergence Summary

<sub>**kind** python</sub>

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
