"""What a region costs, relative to central."""

MULTIPLIERS = {"north": 1.08, "south": 0.97, "east": 1.02, "west": 1.11, "central": 1.0}


def region_multiplier(region: str) -> float:
    return MULTIPLIERS[region]
