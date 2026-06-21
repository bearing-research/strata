"""Inspect REPL — on-demand interactive exploration of cell artifacts.

Spawns a subprocess with a cell's input variables pre-loaded, accepts eval
expressions, and returns results. The subprocess stays alive until explicitly
closed, allowing multiple evaluations without re-loading. The subprocess body
lives in the sibling :mod:`strata.notebook.inspect_harness`, which documents
the JSON line protocol.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strata.notebook.session import NotebookSession

logger = logging.getLogger(__name__)

# The inspect subprocess runs this sibling script. It loads serializer.py from
# its own directory (this package), so — unlike a temp-file harness — neither
# the script nor serializer.py needs copying into the per-session temp dir.
_HARNESS_PATH = Path(__file__).parent / "inspect_harness.py"


class InspectSession:
    """An active inspect REPL session for a cell.

    Spawns a subprocess with the cell's input artifacts pre-loaded, then accepts
    eval commands and returns results.

    Attributes
    ----------
    cell_id : str
        Cell being inspected.
    process : asyncio.subprocess.Process or None
        The subprocess running the REPL.
    ready : bool
        Whether the subprocess has finished loading.
    """

    def __init__(self, cell_id: str):
        self.cell_id = cell_id
        self.process: asyncio.subprocess.Process | None = None
        self.ready = False
        self._manifest_dir: Path | None = None

    async def start(
        self,
        session: NotebookSession,
        timeout_seconds: float = 15,
    ) -> str:
        """Start the inspect subprocess.

        Resolves the cell's upstream inputs, writes them to a temp dir, then
        spawns a Python process with those inputs loaded.

        Parameters
        ----------
        session : NotebookSession
            Notebook session whose cell is being inspected.
        timeout_seconds : float, optional
            Startup timeout (default 15).

        Returns
        -------
        str
            ``"ready"`` on success, otherwise an error message.
        """
        from strata.notebook.executor import CellExecutor

        # Create temp dir for input files (persists for the session lifetime).
        self._manifest_dir = Path(tempfile.mkdtemp(prefix="strata_inspect_"))

        # Materialise upstreams then load input blobs for this cell.
        executor = CellExecutor(session, session.warm_pool)
        await executor._materialize_upstreams(self.cell_id)
        input_specs = executor._load_input_blobs(self.cell_id, self._manifest_dir)

        # Write manifest.
        manifest = {
            "inputs": input_specs,
            "output_dir": str(self._manifest_dir),
        }
        manifest_path = self._manifest_dir / "inspect_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f)

        # Spawn subprocess with the notebook interpreter used for normal runs.
        python_executable = session.venv_python or Path("python")
        cmd = [
            str(python_executable),
            str(_HARNESS_PATH),
            str(manifest_path),
        ]

        self.process = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(session.path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait for the "ready" signal.
        try:
            assert self.process.stdout is not None
            line = await asyncio.wait_for(
                self.process.stdout.readline(),
                timeout=timeout_seconds,
            )
            msg = json.loads(line.decode().strip())
            if msg.get("ok") and msg.get("result") == "ready":
                self.ready = True
                return "ready"
            return msg.get("error", "Unknown startup error")
        except TimeoutError:
            await self.close()
            return "Inspect process timed out during startup"
        except (OSError, ValueError, json.JSONDecodeError) as e:
            await self.close()
            return f"Inspect startup failed: {e}"

    async def evaluate(self, expr: str, timeout_seconds: float = 10) -> dict[str, Any]:
        """Evaluate an expression in the inspect subprocess.

        Parameters
        ----------
        expr : str
            Python expression or statement to evaluate.
        timeout_seconds : float, optional
            Evaluation timeout (default 10).

        Returns
        -------
        dict
            Response with ``ok`` and either ``result``/``type``/``stdout`` or
            ``error``.
        """
        if not self.ready or self.process is None:
            return {"ok": False, "error": "Inspect session not ready"}

        if self.process.returncode is not None:
            self.ready = False
            return {"ok": False, "error": "Inspect process has exited"}

        try:
            assert self.process.stdin is not None
            assert self.process.stdout is not None
            cmd = json.dumps({"expr": expr}) + "\n"
            self.process.stdin.write(cmd.encode())
            await self.process.stdin.drain()

            line = await asyncio.wait_for(
                self.process.stdout.readline(),
                timeout=timeout_seconds,
            )
            if not line:
                self.ready = False
                return {"ok": False, "error": "Inspect process closed unexpectedly"}

            return json.loads(line.decode().strip())

        except TimeoutError:
            return {"ok": False, "error": f"Evaluation timed out after {timeout_seconds}s"}
        except (OSError, ValueError, json.JSONDecodeError) as e:
            return {"ok": False, "error": f"Evaluation failed: {e}"}

    async def close(self) -> None:
        """Close the inspect subprocess and clean up its temp dir."""
        if self.process is not None:
            try:
                if self.process.returncode is None:
                    # Ask it to exit gracefully, then kill if it lingers.
                    assert self.process.stdin is not None
                    cmd = json.dumps({"cmd": "close"}) + "\n"
                    self.process.stdin.write(cmd.encode())
                    await self.process.stdin.drain()
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=2)
                    except TimeoutError:
                        self.process.kill()
                        await self.process.wait()
            except (OSError, ValueError):
                # stdin already closed / process gone — force termination.
                self.process.kill()
                await self.process.wait()

        self.ready = False
        self.process = None

        # Clean up temp dir.
        if self._manifest_dir and self._manifest_dir.exists():
            shutil.rmtree(self._manifest_dir, ignore_errors=True)
            self._manifest_dir = None


class InspectManager:
    """Manages inspect sessions across notebooks.

    One inspect session can be open per cell at a time.
    """

    def __init__(self):
        self._sessions: dict[str, InspectSession] = {}

    async def open_session(
        self,
        cell_id: str,
        notebook_session: NotebookSession,
    ) -> tuple[InspectSession, str]:
        """Open an inspect session for a cell.

        If a session already exists for this cell, close it first.

        Parameters
        ----------
        cell_id : str
            Cell to inspect.
        notebook_session : NotebookSession
            Parent notebook session.

        Returns
        -------
        tuple of (InspectSession, str)
            The session and a status message.
        """
        # Close existing session for this cell.
        if cell_id in self._sessions:
            await self._sessions[cell_id].close()
            del self._sessions[cell_id]

        inspect = InspectSession(cell_id)
        status = await inspect.start(notebook_session)

        if inspect.ready:
            self._sessions[cell_id] = inspect

        return inspect, status

    async def get_session(self, cell_id: str) -> InspectSession | None:
        """Return an active inspect session for a cell, or ``None``.

        Parameters
        ----------
        cell_id : str
            Cell ID.

        Returns
        -------
        InspectSession or None
            The live session, or ``None`` if none exists or it has died.
        """
        session = self._sessions.get(cell_id)
        if session and not session.ready:
            # Session died — clean up.
            await session.close()
            del self._sessions[cell_id]
            return None
        return session

    async def close_session(self, cell_id: str) -> None:
        """Close an inspect session.

        Parameters
        ----------
        cell_id : str
            Cell ID.
        """
        session = self._sessions.pop(cell_id, None)
        if session:
            await session.close()

    async def close_all(self) -> None:
        """Close all inspect sessions."""
        for session in self._sessions.values():
            await session.close()
        self._sessions.clear()
