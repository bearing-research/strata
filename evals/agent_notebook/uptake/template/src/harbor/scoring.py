"""Customer risk."""

import hashlib


def risk_score(customer_id: int) -> float:
    """A customer's risk, from 0 (none) to 1."""
    digest = hashlib.sha256(f"harbor-customer-{customer_id}".encode()).digest()
    return round(int.from_bytes(digest[:4]) / 2**32, 4)
