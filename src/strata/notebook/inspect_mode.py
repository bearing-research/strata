"""Inspect mode for interactive exploration of cell artifacts.

Provides on-demand REPL for exploring a cell's input artifacts without
running the full cell.

NOTE: This module is not used in production. The WebSocket handler uses
inspect_repl.py instead. Kept for potential future use.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strata.notebook.session import NotebookSession

logger = logging.getLogger(__name__)


class ArtifactProxy:
    """Lazy-loading proxy for artifact data.

    Wraps an artifact reference. Loads actual data only on first
    attribute access, not eagerly. This keeps inspect mode startup fast.

    For v1, we actually load eagerly since implementing true lazy proxies
    across process boundaries is complex. This is documented for future enhancement.
    """

    def __init__(self, artifact_uri: str, content_type: str, artifact_manager: Any):
        """Initialize artifact proxy.

        Args:
            artifact_uri: URI of the artifact (strata://artifact/{id}@v={version})
            content_type: Content type (arrow/ipc, json/object, pickle/object)
            artifact_manager: NotebookArtifactManager instance
        """
        self._uri = artifact_uri
        self._content_type = content_type
        self._artifact_manager = artifact_manager
        self._data = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        """Load the artifact data on first access."""
        if not self._loaded:
            try:
                # Parse URI to get artifact ID and version
                # Format: strata://artifact/{id}@v={version}
                parts = self._uri.split("@v=")
                if len(parts) == 2:
                    artifact_id = parts[0].split("/")[-1]
                    version = int(parts[1])
                    self._data = self._artifact_manager.fetch_artifact(artifact_id, version)
                else:
                    self._data = None
                self._loaded = True
            except Exception as e:
                logger.error(f"Failed to load artifact {self._uri}: {e}")
                self._data = None
                self._loaded = True

    def __getattr__(self, name: str) -> Any:
        """Lazy-load on first attribute access."""
        if name in ("_uri", "_content_type", "_artifact_manager", "_data", "_loaded"):
            return object.__getattribute__(self, name)
        self._ensure_loaded()
        if self._data is None:
            raise AttributeError("Artifact failed to load")
        return getattr(self._data, name)

    def __repr__(self) -> str:
        """String representation."""
        if self._loaded:
            return repr(self._data)
        return f"<ArtifactProxy({self._content_type}) — not yet loaded>"

    def __str__(self) -> str:
        """String conversion."""
        if self._loaded:
            return str(self._data)
        return f"<ArtifactProxy({self._content_type})>"


class InspectSession:
    """Interactive REPL session for exploring artifacts.

    Spawns a subprocess with a cell's input artifacts loaded.
    Accepts eval expressions via WebSocket.
    """

    def __init__(self, session: NotebookSession, cell_id: str, artifact_manager: Any):
        """Initialize inspect session.

        Args:
            session: NotebookSession instance
            cell_id: ID of the cell to inspect
            artifact_manager: NotebookArtifactManager instance
        """
        self.session = session
        self.cell_id = cell_id
        self.artifact_manager = artifact_manager
        self.process: asyncio.subprocess.Process | None = None
        self._input_vars: dict[str, str] = {}  # var_name -> artifact_uri

    async def start(self) -> None:
        """Spawn subprocess with cell inputs loaded.

        This finds all input artifacts for the cell and loads them
        into the inspect subprocess namespace.
        """
        # Find the cell
        cell = self.session.notebook_state.get_cell(self.cell_id)
        if cell is None:
            raise ValueError(f"Cell {self.cell_id} not found")

        # Get input artifact URIs from upstream cells
        for upstream_id in cell.upstream_ids:
            upstream_cell = self.session.notebook_state.get_cell(upstream_id)
            if upstream_cell and upstream_cell.artifact_uri:
                # Extract variable name (we'll use the upstream cell ID as var name)
                # For now, use a simple naming scheme
                var_name = upstream_id.replace("-", "_")
                self._input_vars[var_name] = upstream_cell.artifact_uri

        # Start the inspect worker process
        worker_script = Path(__file__).parent / "inspect_worker.py"

        self.process = await asyncio.create_subprocess_exec(
            "python",
            str(worker_script),
            stdout=asyncio.subprocess.PIPE,
            stdin=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self.session.path),
        )

        # Send input variable info to the worker
        input_info = {
            "variables": self._input_vars,
            "artifact_manager_type": "notebook",
        }
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        input_str = json.dumps(input_info) + "\n"
        self.process.stdin.write(input_str.encode())
        await self.process.stdin.drain()

        # Wait for ready signal
        try:
            ready_line = await asyncio.wait_for(self.process.stdout.readline(), timeout=10.0)
            if not (ready_line and b"ready" in ready_line.lower()):
                raise RuntimeError("Inspect worker did not send ready signal")
        except TimeoutError:
            raise RuntimeError("Inspect worker startup timed out")

        logger.debug(f"Inspect session started for cell {self.cell_id}")

    async def evaluate(self, expression: str) -> dict[str, Any]:
        """Evaluate expression in inspect namespace.

        Args:
            expression: Python expression to evaluate

        Returns:
            Dict with: {result: str, type: str, error: str | None}
        """
        if self.process is None:
            raise RuntimeError("Inspect session not started")

        try:
            # Send expression to worker
            assert self.process.stdin is not None
            assert self.process.stdout is not None
            req = {"expr": expression}
            req_str = json.dumps(req) + "\n"
            self.process.stdin.write(req_str.encode())
            await self.process.stdin.drain()

            # Read response
            result_line = await asyncio.wait_for(self.process.stdout.readline(), timeout=30.0)

            if result_line:
                result = json.loads(result_line.decode())
                return result
            else:
                return {
                    "result": None,
                    "type": None,
                    "error": "Worker process ended unexpectedly",
                }

        except TimeoutError:
            return {
                "result": None,
                "type": None,
                "error": "Evaluation timed out",
            }
        except json.JSONDecodeError as e:
            return {
                "result": None,
                "type": None,
                "error": f"Invalid response from worker: {e}",
            }
        except Exception as e:
            return {
                "result": None,
                "type": None,
                "error": f"Evaluation error: {e}",
            }

    async def close(self) -> None:
        """Kill inspect subprocess."""
        if self.process and self.process.returncode is None:
            self.process.kill()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=2.0)
            except TimeoutError:
                logger.warning("Inspect process kill timeout")


class InspectSessionManager:
    """Manages multiple inspect sessions for different cells."""

    def __init__(self):
        """Initialize session manager."""
        self._sessions: dict[str, InspectSession] = {}

    async def open_inspect(
        self,
        session_id: str,
        cell_id: str,
        notebook_session: NotebookSession,
        artifact_manager: Any,
    ) -> InspectSession:
        """Open an inspect session for a cell.

        Args:
            session_id: Notebook session ID
            cell_id: Cell ID to inspect
            notebook_session: NotebookSession instance
            artifact_manager: NotebookArtifactManager instance

        Returns:
            InspectSession instance
        """
        key = f"{session_id}:{cell_id}"

        # Close existing session if any
        if key in self._sessions:
            await self._sessions[key].close()

        # Create and start new session
        inspect_session = InspectSession(notebook_session, cell_id, artifact_manager)
        await inspect_session.start()

        self._sessions[key] = inspect_session
        return inspect_session

    async def get_inspect(self, session_id: str, cell_id: str) -> InspectSession | None:
        """Get an open inspect session.

        Args:
            session_id: Notebook session ID
            cell_id: Cell ID

        Returns:
            InspectSession or None if not open
        """
        key = f"{session_id}:{cell_id}"
        return self._sessions.get(key)

    async def close_inspect(self, session_id: str, cell_id: str) -> None:
        """Close an inspect session.

        Args:
            session_id: Notebook session ID
            cell_id: Cell ID
        """
        key = f"{session_id}:{cell_id}"
        if key in self._sessions:
            await self._sessions[key].close()
            del self._sessions[key]
