"""The random part of a key component for an input that could not be resolved.

Such a component matches no earlier run, so the cell never hits the cache. Every call
site takes it from here, so a recorder can replace this one function with a counter.
"""

from __future__ import annotations

import hashlib
import os


def unresolved_nonce() -> str:
    """64 hex digits no other call returns."""
    return hashlib.sha256(os.urandom(32)).hexdigest()
