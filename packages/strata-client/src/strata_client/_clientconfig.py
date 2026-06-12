"""Slim server-URL resolution for the client.

The full :class:`strata.config.StrataConfig` pulls in pydantic, pydantic-settings,
and notebook submodules — far more than a client needs just to find the server.
This module replicates *only* the server-URL resolution, standard library only,
so ``StrataClient()`` can locate the server without importing the server's config
stack. ``StrataClient(config=...)`` still accepts a full ``StrataConfig`` (it has
a ``server_url``); this is just the default path.

Resolution precedence (highest wins):

1. ``STRATA_SERVER_URL`` env var (a full URL — the simplest knob for pointing a
   client at a remote server; not read by the server config itself).
2. ``STRATA_HOST`` / ``STRATA_PORT`` env vars.
3. ``[tool.strata]`` ``host`` / ``port`` in the nearest ``pyproject.toml``.
4. Defaults ``127.0.0.1:8765`` — matching ``StrataConfig`` defaults.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Protocol

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8765


class HasServerUrl(Protocol):
    """Anything carrying a ``server_url`` — e.g. a full ``StrataConfig``."""

    @property
    def server_url(self) -> str: ...


def _find_pyproject() -> Path | None:
    """Find pyproject.toml in the current or a parent directory."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        candidate = parent / "pyproject.toml"
        if candidate.exists():
            return candidate
    return None


def _pyproject_strata() -> dict:
    """Return the ``[tool.strata]`` table, or ``{}`` if none is found."""
    path = _find_pyproject()
    if path is None:
        return {}
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data.get("tool", {}).get("strata", {})


def resolve_server_url() -> str:
    """Resolve the Strata server URL for a client (see module docstring)."""
    direct = os.environ.get("STRATA_SERVER_URL")
    if direct:
        return direct

    strata_table = _pyproject_strata()
    host = os.environ.get("STRATA_HOST") or strata_table.get("host") or _DEFAULT_HOST
    port = os.environ.get("STRATA_PORT") or strata_table.get("port") or _DEFAULT_PORT
    return f"http://{host}:{port}"
