# The loop cell

The cell below carries `# @loop max_iter=40 carry=state` — Strata's
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
