# Loop Hill Climb — demo for the `@loop` cell primitive

Three-cell notebook that exercises every moving part of Strata's loop
cell feature: the `@loop` / `@loop_until` annotations, per-iteration
artifacts, the progress badge, the iteration picker, and the
`@loop start_from` fork. The cell body is greedy hill climbing on
[Himmelblau's function](https://en.wikipedia.org/wiki/Himmelblau%27s_function) —
small enough to reason about and visually interesting because there
are four equal-valued minima, so different seeds (and different forks)
converge to different basins.

## What to look at

- **`helpers`** is a module cell — imports (``random``) and a
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
    the DAG — no duplicate imports or redefinitions here.
  - Every iteration stores the full `state` as its own artifact
    (`nb_loop-hill-climb-001_cell_evolve_var_state@iter={k}`).
- **`summary`** is a regular downstream cell. It reads the final
  iteration's `state` via the normal DAG input path and prints the
  convergence table.

## Running

```bash
uv run strata-notebook --host 127.0.0.1 --port 8765
```

Open the notebook from the Strata home page, then run the three cells
in order. While `evolve` is running you should see:

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
rng = random.Random((next_iter, round(state["x"], 6), round(state["y"], 6), "alt"))

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
