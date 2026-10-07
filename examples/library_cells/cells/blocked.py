# @name Closure over a runtime value (intentionally blocked)
#
# ``runtime_threshold`` is computed at runtime, so it gets dropped from
# the slice. ``is_outlier`` references it as a free variable, which
# leaves the synthetic module unable to resolve the name at call time.
#
# Running this cell errors at execution time. The diagnostic message
# names both the function (``is_outlier``) and the unresolved variable
# (``runtime_threshold``), pointing the user straight at the fix.
#
# To unblock: move ``runtime_threshold`` into its own cell so the slice
# only contains ``is_outlier``, and let the threshold flow through the
# regular artifact path.

import math

runtime_threshold = math.sqrt(9)


def is_outlier(value: float) -> bool:
    return value > runtime_threshold
