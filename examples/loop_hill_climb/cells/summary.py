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
