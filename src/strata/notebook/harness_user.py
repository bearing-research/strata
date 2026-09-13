"""Who a cell subprocess runs as, and whether it may run on this host at all.

``notebook_harness_env_allowlist`` filters what a cell is *given*. It does not
change who the cell *is*: the harness ran as the server's own OS user, so a cell
on a server with the allowlist set could still read every secret from
``/proc/<server pid>/environ`` — mode 0400, owned by the process owner, and the
cell was the process owner — along with the server's config and every other
notebook on disk.

So a service-mode server refuses to run cell code on its own host unless the
operator has said how that is safe:

1. Run cells somewhere else: a server-managed worker on another machine. That is
   isolation which already exists, and the recommended answer.
2. Drop privileges: ``notebook_harness_user`` names an OS user the harness runs
   as. Dropping privileges requires having them, so the server runs as root.

Personal mode is unaffected. One person on their own machine has nothing to
isolate a cell from.

The refusal is made where cell code would start, not at startup: which worker a
cell uses is decided per cell, from its annotation, then ``notebook.toml``, and a
server cannot know at boot which notebooks will ask for the local one. A cache
hit starts nothing, so it is never refused.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REFUSAL = (
    "This server runs in service mode, where a cell run on the server's own host "
    "could read the server's credentials. Assign the cell a server-managed worker "
    "so it runs on another machine, or set STRATA_NOTEBOOK_HARNESS_USER so cells "
    "on this host run as a separate OS user."
)


class LocalExecutionRefused(RuntimeError):
    """Cell code may not start on this host; the message says what to change."""


@dataclass(frozen=True)
class HarnessUser:
    """The OS user a cell subprocess is started as."""

    name: str
    uid: int
    gid: int
    home: str


def _running_server_config() -> Any | None:
    """The live server's config, or ``None`` outside a server.

    Deliberately not ``StrataConfig.load()`` as a fallback: its default mode is
    service, and a CLI run or a test with no server has no server credentials
    for a cell to read.
    """
    try:
        from strata.server import get_state

        return get_state().config
    except RuntimeError:
        return None


def resolve_harness_user(config: Any | None = None) -> HarnessUser | None:
    """Who to start a cell subprocess as, ``None`` for "as this process".

    Raises:
        LocalExecutionRefused: in service mode with no harness user, or when the
            configured user cannot be switched to here.
    """
    if config is None:
        config = _running_server_config()
    if config is None:
        return None

    name = str(getattr(config, "notebook_harness_user", None) or "").strip()
    if not name:
        if getattr(config, "deployment_mode", None) == "service":
            raise LocalExecutionRefused(REFUSAL)
        return None

    if os.name == "nt":
        raise LocalExecutionRefused(
            "STRATA_NOTEBOOK_HARNESS_USER is not supported on Windows; run cells on a "
            "server-managed worker instead."
        )

    import pwd

    try:
        entry = pwd.getpwnam(name)
    except KeyError:
        raise LocalExecutionRefused(
            f"STRATA_NOTEBOOK_HARNESS_USER names {name!r}, which is not a user on this host."
        ) from None

    if entry.pw_uid != os.geteuid() and os.geteuid() != 0:
        # Popen(user=) would fail with a bare PermissionError naming nothing.
        raise LocalExecutionRefused(
            f"Running cells as {name!r} (STRATA_NOTEBOOK_HARNESS_USER) needs the server "
            "to run as root, so it can switch to that user."
        )
    return HarnessUser(name=name, uid=entry.pw_uid, gid=entry.pw_gid, home=entry.pw_dir)


def spawn_kwargs(user: HarnessUser | None) -> dict[str, Any]:
    """``user=`` / ``group=`` for ``create_subprocess_exec`` or ``subprocess.run``.

    Supplementary groups are cleared too when the server is root, or the cell
    would keep root's groups and whatever files they open. Clearing them needs
    the privilege, which only root has, and a server switching to its own user
    has no other groups to shed.
    """
    if user is None:
        return {}
    kwargs: dict[str, Any] = {"user": user.uid, "group": user.gid}
    if os.geteuid() == 0:
        kwargs["extra_groups"] = []
    return kwargs


def identity_env(env: dict[str, str] | None, user: HarnessUser | None) -> dict[str, str] | None:
    """The spawn environment with ``HOME`` / ``USER`` / ``LOGNAME`` naming the user.

    ``Popen(user=)`` changes the uid and nothing else, so a cell would otherwise
    be told its home is root's: every library that caches under ``~`` fails to
    write there, and the names disagree with ``os.getuid()``.
    """
    if user is None:
        return env
    merged = dict(os.environ if env is None else env)
    merged.update({"HOME": user.home, "USER": user.name, "LOGNAME": user.name})
    return merged


def hand_over(path: Path, user: HarnessUser | None) -> None:
    """Give the harness user a per-run directory it has to write into.

    Run directories are made with ``mkdtemp`` / ``TemporaryDirectory``, which
    are private to their creator — right for the server, and unwritable for a
    cell running as someone else. The server keeps access either way: it is
    root.
    """
    if user is None or user.uid == os.geteuid():
        return
    os.chown(path, user.uid, user.gid)
    for root, dirs, files in os.walk(path):
        for name in (*dirs, *files):
            os.chown(os.path.join(root, name), user.uid, user.gid, follow_symlinks=False)
