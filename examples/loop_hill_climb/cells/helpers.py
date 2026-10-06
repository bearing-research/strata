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
