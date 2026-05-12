# Pick a classifier

The three cells below are alternatives. They share the same `model`
defines contract — each one trains a different classifier on the same
inputs and binds the result to `model`. Only the active variant
participates in the DAG; the others render as inactive tabs in the
notebook UI.

Switching variants is cheap. Each variant has its own provenance
hash, so re-running an already-trained variant is a cache hit. The
downstream cells (`Evaluate active classifier`, `Confusion matrix`)
re-cascade against whichever variant is active.
