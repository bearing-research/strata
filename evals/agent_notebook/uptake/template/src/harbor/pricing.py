"""Price quotes."""

from fleet.rates import region_multiplier

BASE_PRICES = {f"p{i:02d}": round(5 + (i * 37 % 50) * 1.5, 2) for i in range(50)}


def quote(product: str, qty: int, region: str) -> float:
    """What *qty* units of *product* cost in *region*, after the volume discount."""
    discount = 0.1 if qty >= 20 else 0.05 if qty >= 10 else 0.0
    return round(BASE_PRICES[product] * qty * (1 - discount) * region_multiplier(region), 2)
