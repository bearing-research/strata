"""What a cell subprocess is allowed to see of the server's environment.

A cell is arbitrary Python, and it has always been spawned with the server's
whole environment. On one's own laptop that is the right default and there is
nothing to protect. On a shared server it means every member who can run a cell
can read ``STRATA_NOTEBOOK_REMOTE_STORE_HEADERS``, ``STRATA_PROXY_TOKEN``,
worker tokens and every data-source credential the server holds — from
``os.environ``, or from ``/proc/<pid>/environ``, or from any file the server
process can open.

``notebook_harness_env_allowlist`` narrows it. Unset, nothing changes; set, the
harness receives the names it lists plus the handful without which no
subprocess runs at all, and ``STRATA_*`` is dropped unless named exactly — an
operator who writes the name means it, while a prefix rule that happened to
match should not sweep the server's secrets along with it.

A cell's own configuration does not come through here. ``[env]`` in
``notebook.toml`` and mount credentials travel in the manifest and are applied
inside the harness, so an allowlist can be short without taking anything away
from the notebook.
"""

from __future__ import annotations

import os

# Without these a subprocess does not start, or starts and cannot find its
# interpreter, its temp directory or its locale. They are the floor, not a
# judgement about what a cell ought to have.
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

# ``uv run`` resolves the notebook's interpreter, and Python's own start-up
# reads PYTHON*. Both belong to running the cell rather than to the server's
# secrets.
_ESSENTIAL_PREFIXES = ("UV_", "LC_", "PYTHON", "VIRTUAL_ENV", "CONDA_", "R_", "RSTUDIO_")

_SECRET_PREFIX = "STRATA_"


def _allowed(name: str, allowlist: list[str]) -> bool:
    if name in _ESSENTIAL_NAMES or name.startswith(_ESSENTIAL_PREFIXES):
        return True
    for entry in allowlist:
        if entry.endswith("*"):
            # A prefix rule is a convenience for a family of related names. It
            # deliberately cannot reach STRATA_*: the caller who wants one of
            # those has to name it, because a rule broad enough to catch a
            # credential by accident is the failure this setting exists to
            # prevent.
            if name.startswith(entry[:-1]) and not name.startswith(_SECRET_PREFIX):
                return True
        elif name == entry:
            return True
    return False


def harness_env(allowlist: list[str] | None, extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment to spawn a cell subprocess with.

    ``allowlist`` empty or ``None`` returns the server's environment unchanged,
    which is what every deployment got before this existed and what a personal
    one should keep getting.

    ``extra`` is set after filtering: the batch harness is told which file
    descriptors to use through the environment, and those are this code
    talking to itself rather than anything the allowlist is about.
    """
    if not allowlist:
        return {**os.environ, **(extra or {})}

    kept = {name: value for name, value in os.environ.items() if _allowed(name, allowlist)}
    kept.update(extra or {})
    return kept


def configured_allowlist() -> list[str]:
    """The allowlist from the running server, or from a freshly loaded config.

    The warm pool has no executor and no session to read config through, and it
    spawns the process that runs the cell on the default WebSocket path — so it
    has to reach the setting itself or be the one place the filter does not
    apply.
    """
    try:
        from strata.server import get_state

        config = get_state().config
    except RuntimeError:
        from strata.config import StrataConfig

        config = StrataConfig.load()
    return list(getattr(config, "notebook_harness_env_allowlist", []) or [])
