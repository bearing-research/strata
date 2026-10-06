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
