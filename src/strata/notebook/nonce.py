"""The random part of a key component for an input that could not be resolved.

Such a component matches no earlier run, so the cell never hits the cache. Every call
site takes it from here, so a recorder can replace this one function with a counter.
"""

from __future__ import annotations

import hashlib
import os

# Starts every nonce, so an explanation can tell an input that could not be resolved from
# one that changed. Datasets keep the first 32 digits.
_MARKER = "0a7e50150a7e5015"


def unresolved_nonce() -> str:
    """64 hex digits no other call returns."""
    return _MARKER + hashlib.sha256(os.urandom(32)).hexdigest()[len(_MARKER) :]


def unread_nonce() -> str:
    """64 hex digits no other call returns, for a file not read yet or read while it
    changed: unlike :func:`unresolved_nonce`, its input shows as changed, not unavailable."""
    return hashlib.sha256(b"unread" + os.urandom(32)).hexdigest()


def is_unresolved(component: str) -> bool:
    """Whether this key component holds a nonce: its input could not be resolved."""
    return _MARKER in component
