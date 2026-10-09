"""Warm process pool: pre-spawned Python processes with common imports loaded.

Cuts per-execution startup from ~1.5s to ~50ms.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from strata.notebook.harness_env import configured_allowlist, harness_env
from strata.notebook.harness_user import (
    LocalExecutionRefused,
    hand_over,
    identity_env,
    resolve_harness_user,
    spawn_kwargs,
)
from strata.notebook.lifeline import lifeline_command, lifeline_handoff
from strata.notebook.process_tree import (
    SUBPROCESS_LINE_LIMIT,
    kill_subprocess_tree_nowait,
    subprocess_kwargs_for_new_group,
    terminate_subprocess_tree,
)

if TYPE_CHECKING:
    import asyncio.subprocess

logger = logging.getLogger(__name__)


@dataclass
class WarmProcess:
    """A pre-spawned Python process ready for work."""

    process: asyncio.subprocess.Process
    created_at: float
    ready: bool = False


class WarmProcessPool:
    """Pool of pre-spawned processes for fast cell execution.

    Each worker is single-shot: it runs one manifest and is killed by
    ``release_and_replace``, which spawns a replacement in the background. The win
    is imports loaded ahead of time, not interpreter reuse; single-shot keeps cells
    isolated (no ``sys.path``, ``os.environ`` or module state leaks between them).
    """

    def __init__(
        self,
        notebook_dir: Path,
        pool_size: int = 2,
        python_executable: str | Path = "python",
        worker_command: list[str] | None = None,
        ready_timeout_seconds: float = 10.0,
    ):
        """Initialize the warm process pool.

        Args:
            python_executable: Ignored when ``worker_command`` is given.
            worker_command: Full worker argv; defaults to the Python pool worker. The
                R pool passes ``[Rscript, pool_worker.R, notebook_dir]``.
            ready_timeout_seconds: Wait for the worker's "ready" line. R workers pay
                renv activation at startup, so their pools pass a larger value.
        """
        self.notebook_dir = Path(notebook_dir)
        self.pool_size = pool_size
        self.python_executable = str(python_executable)
        self.worker_command = list(worker_command) if worker_command else None
        self.ready_timeout_seconds = ready_timeout_seconds
        self._available: asyncio.Queue[WarmProcess] = asyncio.Queue()
        self._warming: int = 0  # Processes currently starting up
        self._started: bool = False
        self._lock = asyncio.Lock()
        # So drain() can cancel them
        self._background_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        """Spawn the initial ``pool_size`` warm processes in parallel."""
        async with self._lock:
            if self._started:
                return
            self._started = True

        tasks = [self._spawn_warm_process() for _ in range(self.pool_size)]
        await asyncio.gather(*tasks, return_exceptions=True)

    def track_background_task(self, task: asyncio.Task) -> None:
        """Track a task so shutdown paths can cancel it."""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _spawn_warm_process(self) -> None:
        """Spawn a process that imports common deps and waits for one manifest on stdin."""
        try:
            harness_user = resolve_harness_user()
        except LocalExecutionRefused as exc:
            # The cold path tells the user why.
            logger.debug("Not spawning a warm worker: %s", exc)
            return
        self._warming += 1
        try:
            if self.worker_command is not None:
                command = lifeline_command(self.worker_command)
            else:
                worker_script = Path(__file__).parent / "pool_worker.py"
                command = [
                    self.python_executable,
                    str(worker_script),
                    str(self.notebook_dir),
                ]

            # Process-group leader so cancel/drain kills the whole tree
            # (DataLoader children, fork servers, ...).
            # limit= lifts the 64 KiB line cap: the one result line embeds the
            # cell's full stdout, and readline() raises on a longer line.
            # The env allowlist applies here too: this is the default path, so
            # skipping it would expose server secrets to most cells.
            allowlist = configured_allowlist()
            lifeline_fds, lifeline_vars = lifeline_handoff()
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.notebook_dir),
                limit=SUBPROCESS_LINE_LIMIT,
                env=identity_env(harness_env(allowlist, lifeline_vars), harness_user),
                pass_fds=lifeline_fds,
                **spawn_kwargs(harness_user),
                **subprocess_kwargs_for_new_group(),
            )

            try:
                assert process.stdout is not None
                ready_line = await asyncio.wait_for(
                    process.stdout.readline(), timeout=self.ready_timeout_seconds
                )
                if ready_line and b"ready" in ready_line.lower():
                    warm_proc = WarmProcess(
                        process=process,
                        created_at=time.time(),
                        ready=True,
                    )
                    await self._available.put(warm_proc)
                    logger.debug(f"Warm process spawned and ready (pid={process.pid})")
                else:
                    logger.warning("Warm process did not send ready signal, killing")
                    await terminate_subprocess_tree(process)
            except TimeoutError:
                logger.warning("Warm process startup timed out, killing process")
                await terminate_subprocess_tree(process)

        except Exception as e:
            logger.error(f"Failed to spawn warm process: {e}")
        finally:
            self._warming -= 1

    async def acquire(self) -> WarmProcess | None:
        """Return a warm process, or None if the pool is empty or not started."""
        # A background spawn finishing just after drain() could otherwise be
        # handed out and then killed by the next invalidate cycle.
        if not self._started:
            return None
        try:
            warm_proc = self._available.get_nowait()
            return warm_proc
        except asyncio.QueueEmpty:
            return None

    async def release_and_replace(self, process: WarmProcess) -> None:
        """Kill a used process and spawn a replacement in the background."""
        if process.process and process.process.returncode is None:
            await terminate_subprocess_tree(process.process)

        task = asyncio.create_task(self._spawn_warm_process())
        self.track_background_task(task)

    async def drain(self) -> None:
        """Cancel pending spawns and kill every queued process."""
        async with self._lock:
            self._started = False

        # Cancel spawns first so they don't refill the queue after the drain.
        for task in list(self._background_tasks):
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()

        while True:
            try:
                proc = self._available.get_nowait()
                if proc.process and proc.process.returncode is None:
                    await terminate_subprocess_tree(proc.process)
            except asyncio.QueueEmpty:
                break

    async def invalidate(self) -> None:
        """Drain and respawn after the environment changed (``uv.lock`` moved)."""
        logger.info("Invalidating warm process pool due to env change")
        await self.drain()
        await self.start()

    def shutdown_nowait(self) -> None:
        """Best-effort synchronous shutdown for non-async callers."""
        self._started = False
        for task in list(self._background_tasks):
            task.cancel()
        self._background_tasks.clear()

        while True:
            try:
                proc = self._available.get_nowait()
            except asyncio.QueueEmpty:
                break
            if proc.process and proc.process.returncode is None:
                kill_subprocess_tree_nowait(proc.process)


class PooledCellExecutor:
    """Runs cells on the warm pool for ``CellExecutor`` when one is available."""

    @staticmethod
    async def execute_with_pool(
        pool: WarmProcessPool,
        manifest_path: Path,
        notebook_dir: Path,
        timeout_seconds: float = 30,
    ) -> dict | None:
        """Execute a cell using a warm process from the pool.

        Returns None when no warm process is available; the caller then cold-spawns.

        Raises:
            TimeoutError: the cell exceeded ``timeout_seconds`` in the worker. This is
                a real cell timeout, not a pool miss, so there is no cold re-run.
        """
        try:
            harness_user = resolve_harness_user()
        except LocalExecutionRefused:
            # Fall through to the cold path, which refuses with the reason.
            return None

        warm_proc = await pool.acquire()
        if warm_proc is None:
            return None
        # Outputs land beside the manifest, in a run dir private to the server.
        hand_over(manifest_path.parent, harness_user)

        try:
            assert warm_proc.process.stdin is not None
            assert warm_proc.process.stdout is not None
            manifest_str = (str(manifest_path) + "\n").encode()
            warm_proc.process.stdin.write(manifest_str)
            await warm_proc.process.stdin.drain()

            result_json = await asyncio.wait_for(
                warm_proc.process.stdout.readline(),
                timeout=timeout_seconds,
            )

            if result_json:
                result_data = json.loads(result_json.decode())
                return result_data
            else:
                logger.warning("Warm process returned empty result")
                return None

        except asyncio.CancelledError:
            logger.info(
                "Warm process execution cancelled; killing worker pid=%s",
                warm_proc.process.pid,
            )
            if warm_proc.process.returncode is None:
                try:
                    await asyncio.shield(terminate_subprocess_tree(warm_proc.process))
                except Exception:
                    logger.exception(
                        "Failed terminating cancelled warm worker tree pid=%s",
                        warm_proc.process.pid,
                    )
            raise
        except TimeoutError:
            # A real cell timeout. Returning None ("pool unavailable") would make
            # the caller re-run the cell cold: twice the timeout, and side
            # effects run twice.
            logger.warning("Warm process execution timed out")
            raise
        except Exception as e:
            logger.error(f"Error executing with warm process: {e}")
            return None
        finally:
            await asyncio.shield(pool.release_and_replace(warm_proc))
