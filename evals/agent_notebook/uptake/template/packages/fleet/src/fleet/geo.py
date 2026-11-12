"""Where the regions are, in km on a flat map."""

import math

COORDS = {
    "north": (0.0, 900.0),
    "south": (150.0, -700.0),
    "east": (850.0, 100.0),
    "west": (-950.0, 50.0),
    "central": (0.0, 0.0),
}


def distance(a: str, b: str) -> float:
    """Kilometres between two regions."""
    (ax, ay), (bx, by) = COORDS[a], COORDS[b]
    return round(math.hypot(ax - bx, ay - by), 1)
