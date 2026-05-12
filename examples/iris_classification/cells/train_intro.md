# Train and evaluate

The exploration cells above don't feed the model directly — they're
sanity checks. The cells below are the ML pipeline proper: split the
data, fit a random forest, score it against the held-out test set,
and visualize the confusion matrix.

Edit any cell in this section and the downstream cells go stale.
Re-running the *test set* cell after an upstream change cascades the
whole subtree in topological order, hitting cache for everything that
hasn't changed.
