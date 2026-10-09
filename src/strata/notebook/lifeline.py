"""Kill a cell subprocess's process group when the process that spawned it dies.

The spawner keeps the only write end of a pipe and the child inherits the read end
(``pass_fds``), so the child reads EOF exactly when every copy of the write end is
gone: the spawner exited or was killed. ``getppid`` cannot tell, because a cold
harness's parent is ``uv run``, not the server.

Stdlib only: the harness and the pool worker load it by path from the notebook venv.
"""

from __future__ import annotations

import os
import signal
import stat
import sys
import threading

LIFELINE_FD_ENV = "STRATA_LIFELINE_FD"

_pipe: tuple[int, int] | None = None


def lifeline_handoff() -> tuple[tuple[int, ...], dict[str, str]]:
    """The ``pass_fds`` and env entries that hand a child the lifeline.

    One pipe per process, never written to. Empty on Windows, which keeps no
    lifeline. The child must lead its own process group: on EOF it kills that group.
    """
    global _pipe
    if sys.platform == "win32":
        return (), {}
    if _pipe is None:
        _pipe = os.pipe()  # both ends non-inheritable, so other children never get them
    return (_pipe[0],), {LIFELINE_FD_ENV: str(_pipe[0])}


def watch_lifeline() -> None:
    """Child side: start a daemon thread that kills this process group on EOF."""
    raw = os.environ.pop(LIFELINE_FD_ENV, None)
    if raw is None:
        return
    fd = int(raw)
    try:
        is_pipe = stat.S_ISFIFO(os.fstat(fd).st_mode)
    except OSError:
        return  # not inherited (an intermediate launcher closed it)
    if not is_pipe:
        return  # the number was reused by an unrelated file; reading it would kill us
    threading.Thread(target=_await_eof, args=(fd,), name="strata-lifeline", daemon=True).start()


def _await_eof(fd: int) -> None:
    while os.read(fd, 1):  # nobody writes; only EOF matters
        continue
    os.killpg(os.getpgrp(), signal.SIGKILL)
