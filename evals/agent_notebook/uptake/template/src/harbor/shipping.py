"""Shipping times."""

import math

from fleet.geo import distance


def eta_days(origin: str, dest: str) -> int:
    """Days to ship from *origin* to *dest*: one to pack, then 400 km a day."""
    return 1 + math.ceil(distance(origin, dest) / 400)
