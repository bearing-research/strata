"""Process-local runtime tokens for dynamically-provisioned workers.

An SSH-tunneled worker's bearer token is generated at provisioning time and held
in the server process — it must never land in the committed ``notebook.toml``
(which is why a plain ``config.token`` won't do, and there's no operator-exported
env var to name in ``config.token_env``). This tiny registry is where the
:class:`~strata.notebook.remote_worker_supervisor.RemoteWorkerSupervisor` stashes
the token and where the executor's token resolution finds it, keyed by worker
name and cleared on teardown.

Deliberately dependency-free (stdlib only): the executor imports it, and the
executor is core-deps-only.
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
