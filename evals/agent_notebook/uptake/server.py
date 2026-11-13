"""Arm C' (design-agent-uptake §6, P1): a scratchpad-profile Strata server per trial.

One notebook, open on a server of its own whose MCP endpoint lists ``run`` and
``scratchpad``; the trial's config dir gets a user-scope HTTP entry for it, as the
installer will write in P4. The notebook's environment has the project installed
editable, so a cell can import it (P1 cells run in the notebook's environment,
in its ``work/``, not the project's).
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import httpx

# MCP entries in the trial's config dir are named this; tools are mcp__strata__<tool>.
SERVER_NAME = "strata"
TOOLS = (f"mcp__{SERVER_NAME}__run", f"mcp__{SERVER_NAME}__scratchpad")
# What a cell here first imports: the harness's own needs and the project's.
_WARM = "import cloudpickle, orjson, pyarrow, pandas, matplotlib.pyplot, harbor.data"


class SetupError(RuntimeError):
    """The arm's server could not be brought up; the trial is discarded."""


def server_env(root: Path, port: int, load_log: Path, load_seconds: float) -> dict[str, str]:
    """The server's environment: everything it stores under *root*, nothing of the host's
    Strata settings or API keys, and the load counter its cells inherit."""

    def uv_dir(*args: str) -> str:
        return subprocess.run(
            ["uv", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("STRATA_", "ANTHROPIC_", "CLAUDE", "HARBOR_", "VIRTUAL_ENV"))
    }
    env.update(
        # Some paths fall back to ~/.strata whatever the settings say.
        HOME=str(root / "home"),
        # uv's real cache and interpreters, so nothing is downloaded again.
        UV_CACHE_DIR=uv_dir("cache", "dir"),
        UV_PYTHON_INSTALL_DIR=uv_dir("python", "dir"),
        STRATA_HOST="127.0.0.1",
        STRATA_PORT=str(port),
        STRATA_MCP_ENABLED="true",
        STRATA_MCP_PROFILE="scratchpad",
        STRATA_CACHE_DIR=str(root / "cache"),
        STRATA_ARTIFACT_DIR=str(root / "artifacts"),
        STRATA_METADATA_DB=str(root / "metadata.sqlite"),
        STRATA_NOTEBOOK_STORAGE_DIR=str(root / "notebooks"),
        STRATA_WORKER_ENV_ROOT=str(root / "worker-envs"),
        HARBOR_LOAD_LOG=str(load_log),
        HARBOR_LOAD_SECONDS=str(load_seconds),
    )
    return env


def mcp_entry_command(claude: str, url: str) -> list[str]:
    """The ``claude mcp add-json`` call writing the server's user-scope entry.

    ``add-json`` keeps ``alwaysLoad`` (checked on Claude Code 2.1.323, §7)."""
    entry = {"type": "http", "url": f"{url}/mcp/", "alwaysLoad": True}
    return [claude, "mcp", "add-json", "-s", "user", SERVER_NAME, json.dumps(entry)]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _until(ready, what: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SetupError(f"the server exited ({proc.returncode}) waiting for {what}")
        try:
            if ready():
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise SetupError(f"timed out waiting for {what}")


@contextmanager
def scratchpad_server(
    root: Path,
    project: Path,
    agent_env: dict[str, str],
    *,
    claude: str,
    load_log: Path,
    load_seconds: float,
    timeout: float = 120.0,
) -> Iterator[str]:
    """A running scratchpad-profile server over one open notebook under *root* that
    can import *project*; its MCP entry is written to the agent's config dir. Yields
    the server's URL and stops the server on exit."""
    from strata.notebook.writer import create_notebook

    # The server opens notebooks only inside its storage dir.
    storage = root / "notebooks"
    storage.mkdir(parents=True)
    notebook = create_notebook(storage, "scratch")
    added = subprocess.run(
        ["uv", "add", "--quiet", "--editable", str(project)],
        cwd=notebook,
        capture_output=True,
        text=True,
    )
    if added.returncode:
        raise SetupError(f"uv add of the project failed: {added.stderr.strip()[-300:]}")
    # macOS checks each binary of a fresh venv on its first load, 10 s and more in all; a
    # user's own environment is past that, so it is paid here, before the agent's clock.
    warmed = subprocess.run(
        [str(notebook / ".venv" / "bin" / "python"), "-c", _WARM], capture_output=True, text=True
    )
    if warmed.returncode:
        raise SetupError(f"the notebook's environment cannot import: {warmed.stderr[-300:]}")

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    with open(root / "server.log", "w") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "strata"],
            cwd=root,
            env=server_env(root, port, load_log, load_seconds),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            # Its own group, so stopping it takes its cells and pools along.
            start_new_session=True,
        )
        try:
            _until(lambda: httpx.get(f"{url}/health").is_success, "health", proc, timeout)
            opened = httpx.post(
                f"{url}/v1/notebooks/open", json={"path": str(notebook)}, timeout=timeout
            )
            if not opened.is_success:
                raise SetupError(f"opening the notebook failed: {opened.text[:300]}")
            session_id = opened.json()["session_id"]

            def synced() -> bool:
                payload = httpx.get(f"{url}/v1/notebooks/{session_id}/environment").json()
                state = payload["environment"]["sync_state"]
                if state in ("failed", "fallback"):
                    raise SetupError(f"the notebook's environment is {state}")
                return state == "ready"

            _until(synced, "the notebook's environment", proc, timeout)
            registered = subprocess.run(
                mcp_entry_command(claude, url), env=agent_env, capture_output=True, text=True
            )
            if registered.returncode:
                raise SetupError(f"claude mcp add-json failed: {registered.stderr.strip()}")
            yield url
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
