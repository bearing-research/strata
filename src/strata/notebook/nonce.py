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
_UNHASHED = "c4ec4ed0c4ec4ed0"


def unresolved_nonce() -> str:
    """64 hex digits no other call returns."""
    return _MARKER + hashlib.sha256(os.urandom(32)).hexdigest()[len(_MARKER) :]


def unread_nonce() -> str:
    """64 hex digits no other call returns, for a file read while it changed: unlike
    :func:`unresolved_nonce`, its input shows as changed, not unavailable."""
    return hashlib.sha256(b"unread" + os.urandom(32)).hexdigest()


def unhashed_nonce() -> str:
    """64 hex digits no other call returns, for a local mount whose files are not hashed
    yet: its input shows as being checked."""
    return _UNHASHED + hashlib.sha256(os.urandom(32)).hexdigest()[len(_UNHASHED) :]


def is_unresolved(component: str) -> bool:
    """Whether this key component holds a nonce: its input could not be resolved."""
    return _MARKER in component


def is_unhashed(component: str) -> bool:
    """Whether this key component holds :func:`unhashed_nonce`'s nonce."""
    return _UNHASHED in component
