# @name Pure module cell (baseline)
#
# This is the classic shape: imports, a literal constant, and a couple
# of helpers. No runtime work at module scope. The cell is exported
# verbatim (no slicing), so comments and formatting survive.

import math

CIRCLE_PRECISION = 4


def area(radius: float) -> float:
    return round(math.pi * radius * radius, CIRCLE_PRECISION)


def perimeter(radius: float) -> float:
    return round(2 * math.pi * radius, CIRCLE_PRECISION)
