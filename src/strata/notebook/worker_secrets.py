"""Process-local runtime tokens for dynamically provisioned workers.

An SSH-tunneled worker's token is generated at provisioning time and must never
reach ``notebook.toml``, so ``RemoteWorkerSupervisor`` stashes it here, keyed by
worker name, for the executor to find. Stdlib only: the executor imports it.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_tokens: dict[str, str] = {}


def set_runtime_worker_token(name: str, token: str) -> None:
    """Record the live bearer token for worker *name* (overwrites any prior)."""
    with _lock:
        _tokens[name] = token


def get_runtime_worker_token(name: str) -> str | None:
    """Return the live token for worker *name*, or ``None`` if none is registered."""
    with _lock:
        return _tokens.get(name)


def clear_runtime_worker_token(name: str) -> None:
    """Forget worker *name*'s token (on teardown). A no-op if absent."""
    with _lock:
        _tokens.pop(name, None)
