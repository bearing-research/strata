# Mixed cells (the new payoff)

The next cell mixes runtime setup with reusable helpers. Before slicing
this whole cell would have been blocked because of the runtime
assignments, the user would have to split them into separate cells just
to share `clamp` downstream.

After slicing, the planner drops the runtime lines from the synthetic
module and exports the helper. The runtime variables still flow through
the regular artifact path, so downstream cells see them too.
