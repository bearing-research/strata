"""Process-local runtime tokens for dynamically provisioned workers.

An SSH-tunneled worker's token is generated at provisioning time and must never
reach ``notebook.toml``, so ``RemoteWorkerSupervisor`` stashes it here, keyed by
worker name, for the executor to find. Stdlib only: the executor imports it.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_tokens: dict[str, str] = {}
# Workers whose tunnel is not seen reaching them: nothing is sent to them.
_withheld: set[str] = set()


def set_runtime_worker_token(name: str, token: str) -> None:
    """Record the live bearer token for worker *name* (overwrites any prior)."""
    with _lock:
        _tokens[name] = token
        _withheld.discard(name)


def withhold_runtime_worker_token(name: str) -> None:
    """Forget *name*'s token, and refuse to send it anything, until it is set again."""
    with _lock:
        _tokens.pop(name, None)
        _withheld.add(name)


def runtime_worker_withheld(name: str) -> bool:
    """Whether worker *name*'s tunnel was last found not reaching it."""
    with _lock:
        return name in _withheld


def get_runtime_worker_token(name: str) -> str | None:
    """Return the live token for worker *name*, or ``None`` if none is registered."""
    with _lock:
        return _tokens.get(name)


def clear_runtime_worker_token(name: str) -> None:
    """Forget worker *name*'s token (on teardown). A no-op if absent."""
    with _lock:
        _tokens.pop(name, None)
        _withheld.discard(name)
