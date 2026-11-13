"""One cell run's harness process and the tasks that serve it, ended together.

A run starts a harness and, beside it, tasks that read its pipes (stdout and stderr
drains, the event relay). Each run path used to end them by hand on the exits it
named, cancellation and timeout, so any other exception (a failing callback, a broken
pipe) left the harness running the user's code, unreaped, and its readers pending.
:class:`RunScope` ends them on every exit.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from types import TracebackType
from typing import Any

from strata.notebook import process_tree

logger = logging.getLogger(__name__)


class RunScope:
    """``async with RunScope("Cell execution") as scope``: owns a harness and its helpers.

    Leaving the scope cancels every helper still running and waits for it; the caller
    awaits the helpers whose result it needs before then. Leaving it by an exception,
    cancellation included, also stops the harness's whole process tree first.
    """

    def __init__(self, what: str) -> None:
        self._what = what
        self._proc: asyncio.subprocess.Process | None = None
        self._tasks: list[asyncio.Task[Any]] = []

    def own(self, proc: asyncio.subprocess.Process) -> asyncio.subprocess.Process:
        """Make *proc*, spawned in its own process group, the harness this scope stops."""
        self._proc = proc
        return proc

    def start[T](self, coro: Coroutine[Any, Any, T]) -> asyncio.Task[T]:
        """Run *coro* as a helper of this run, ended with it."""
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        return task

    async def __aenter__(self) -> RunScope:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pending = [task for task in self._tasks if not task.done()]
        for task in pending:
            task.cancel()
        proc = self._proc
        if exc_type is not None and proc is not None:
            logger.info(
                "%s ended by %s; stopping harness subprocess tree pid=%s",
                self._what,
                exc_type.__name__,
                proc.pid,
            )
            try:
                # Shielded: a second cancel must not leave the tree half stopped.
                await asyncio.shield(process_tree.terminate_subprocess_tree(proc))
            except Exception:
                logger.exception("Failed to terminate harness subprocess tree pid=%s", proc.pid)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
