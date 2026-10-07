"""What a cell subprocess may see of the server's environment.

Unset, ``notebook_harness_env_allowlist`` passes the whole environment through
(right for a laptop). Set, the harness gets only the listed names plus a few
that any subprocess needs; ``STRATA_*`` names (proxy token, remote-store
headers, worker tokens) pass only when listed exactly, never by prefix.
``[env]`` and mount credentials travel in the manifest, so they are unaffected.
"""

from __future__ import annotations

import os

from strata.notebook.writer import _is_sensitive_env_key

# Without these a subprocess cannot start or cannot find its interpreter,
# temp directory or locale. They are the floor, not a judgement about cells.
_ESSENTIAL_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "LOGNAME",
        "USER",
        "SHELL",
        "TMPDIR",
        "TEMP",
        "TMP",
        "PWD",
        "TZ",
        # Windows: the interpreter cannot start without these.
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "COMSPEC",
        "PATHEXT",
        "WINDIR",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "PROGRAMFILES",
        "PROGRAMDATA",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
    }
)

# ``uv run`` finds its cache and the notebook's interpreter through UV_*, Python
# start-up reads PYTHON*, and R and RStudio tools read R_* and RSTUDIO_*.
_ESSENTIAL_PREFIXES = ("UV_", "LC_", "PYTHON", "VIRTUAL_ENV", "R_", "RSTUDIO_")

_SECRET_PREFIX = "STRATA_"


def _essential(name: str) -> bool:
    """In the floor, minus credentials: a private index's login or a publish token is
    for the server's ``uv sync``, not for every cell. Listed by name, one still passes."""
    if name in _ESSENTIAL_NAMES:
        return True
    if not name.startswith(_ESSENTIAL_PREFIXES) or _is_sensitive_env_key(name):
        return False
    if name.startswith("UV_PUBLISH_"):
        return False
    return not (name.startswith("UV_INDEX_") and name.endswith("_USERNAME"))


def _allowed(name: str, allowlist: list[str]) -> bool:
    if _essential(name):
        return True
    for entry in allowlist:
        if entry.endswith("*"):
            # Prefix rules deliberately cannot reach STRATA_*: a rule broad enough to
            # catch a credential by accident is what this setting exists to prevent.
            if name.startswith(entry[:-1]) and not name.startswith(_SECRET_PREFIX):
                return True
        elif name == entry:
            return True
    return False


def harness_env(allowlist: list[str] | None, extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment to spawn a cell subprocess with.

    An empty or ``None`` ``allowlist`` returns the server's environment unchanged.
    ``extra`` is applied after filtering (internal fd hand-offs to the batch
    harness, not subject to the allowlist).
    """
    if not allowlist:
        return {**os.environ, **(extra or {})}

    kept = {name: value for name, value in os.environ.items() if _allowed(name, allowlist)}
    kept.update(extra or {})
    return kept


def configured_allowlist() -> list[str]:
    """The allowlist from the running server, or from a freshly loaded config.

    The warm pool has no executor or session to read config through, so it
    reads the setting here rather than skip the filter.
    """
    try:
        from strata.server import get_state

        config = get_state().config
    except RuntimeError:
        from strata.config import StrataConfig

        config = StrataConfig.load()
    return list(getattr(config, "notebook_harness_env_allowlist", []) or [])
