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
# `random` and `himmelblau` come from the seed cell via the DAG, so
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
