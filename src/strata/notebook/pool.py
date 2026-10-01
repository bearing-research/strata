"""Warm process pool for fast cell execution.

The pool pre-spawns Python processes with common imports already loaded,
reducing the startup overhead from ~1.5s to ~50ms per execution.
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
    """Pool of pre-spawned Python processes for fast cell execution.

    Each worker is *single-shot*: it runs one manifest and is killed by
    ``release_and_replace``. A background replacement is spawned so the
    queue stays primed. The win is parallel warm-up — common imports
    are loaded ahead of time so cells don't pay the ~1.5s import cost
    when they execute — not interpreter reuse. Single-shot also
    preserves Strata's per-cell isolation invariant (no sys.path,
    os.environ, or module-state leakage between cells).

    Attributes:
        notebook_dir: Path to the notebook directory
        pool_size: Number of warm processes to maintain
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
            notebook_dir: Path to the notebook directory
            pool_size: Number of warm processes to maintain
            python_executable: Python interpreter used for warm workers
                (ignored when ``worker_command`` is given)
            worker_command: Full argv for the warm worker. Defaults to the
                Python pool worker; the R pool passes
                ``[Rscript, pool_worker.R, notebook_dir]`` here. The
                stdin/stdout frame protocol is language-agnostic.
            ready_timeout_seconds: How long to wait for the worker's
                "ready" line. R workers pay renv activation at startup,
                so their pools pass a larger value.
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
        """Spawn initial pool of warm processes.

        Spawns pool_size processes in parallel.
        """
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
        """Spawn a process that imports common deps and waits for work.

        Uses a pool_worker.py script that:
        1. Imports common packages
        2. Sends a 'ready' signal
        3. Waits for a manifest path on stdin
        4. Runs the harness logic
        5. Exits (one-shot)
        """
        try:
            harness_user = resolve_harness_user()
        except LocalExecutionRefused as exc:
            # The cold path tells the user why.
            logger.debug("Not spawning a warm worker: %s", exc)
            return
        self._warming += 1
        try:
            if self.worker_command is not None:
                command = self.worker_command
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
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.notebook_dir),
                limit=SUBPROCESS_LINE_LIMIT,
                env=identity_env(harness_env(allowlist) if allowlist else None, harness_user),
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
        """Get a warm process. If none available, return None (caller uses cold spawn).

        Returns:
            WarmProcess if available, None if pool is empty or not started
        """
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
        """Kill used process and spawn a replacement in background.

        Args:
            process: The WarmProcess to kill
        """
        if process.process and process.process.returncode is None:
            await terminate_subprocess_tree(process.process)

        task = asyncio.create_task(self._spawn_warm_process())
        self.track_background_task(task)

    async def drain(self) -> None:
        """Kill all processes in the pool (on env change or shutdown).

        Cancels pending background spawn tasks, then drains and kills
        all queued processes.
        """
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
        """Environment changed — drain and respawn.

        Called when uv.lock has changed, indicating dependencies changed.
        """
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
    """Wrapper that uses the warm process pool for execution.

    This is a helper class for CellExecutor to use the pool when available.
    """

    @staticmethod
    async def execute_with_pool(
        pool: WarmProcessPool,
        manifest_path: Path,
        notebook_dir: Path,
        timeout_seconds: float = 30,
    ) -> dict | None:
        """Execute a cell using a warm process from the pool.

        Args:
            pool: The WarmProcessPool instance
            manifest_path: Path to the execution manifest
            notebook_dir: Path to the notebook directory
            timeout_seconds: Execution timeout

        Returns:
            Result dict if successful, None if pool not available (caller should use cold)

        Raises:
            TimeoutError: the cell exceeded ``timeout_seconds`` in the warm
                worker — a real cell timeout, not a pool-availability miss;
                the caller's timeout handler surfaces it (no cold re-run).
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
