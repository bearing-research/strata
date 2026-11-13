"""``strata agent <path>``: one command to let a coding agent drive a notebook.

Creates or opens the notebook, starts (or reuses) a server with MCP enabled,
opens a warm session (``list_notebooks`` only sees open sessions), writes
``.mcp.json`` plus a CLAUDE.md working agreement, and launches the read-only TUI.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
from vestigo.file_modes import private_file

# ANSI helpers, tty-gated (mirrors strata.notebook.cli).
_USE_COLOR = sys.stdout.isatty()


def _color(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _green(text: str) -> str:
    return _color("32", text)


def _dim(text: str) -> str:
    return _color("90", text)


# Managed block markers: a re-run rewrites its block in place and never touches the
# user's own CLAUDE.md content.
_BLOCK_START = "<!-- strata:agent:start -->"
_BLOCK_END = "<!-- strata:agent:end -->"


def _server_endpoints(server_url: str) -> tuple[str, int]:
    """Return ``(host, port)`` to bind a spawned server on, from *server_url*."""
    parsed = urlparse(server_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 8765)
    return host, port


def _server_alive(server_url: str) -> bool:
    """True if a server answers ``/health`` at *server_url*."""
    try:
        response = httpx.get(f"{server_url}/health", timeout=2.0)
    except httpx.HTTPError:
        return False
    return response.status_code < 500


def _mcp_mounted(server_url: str) -> bool:
    """True if the MCP endpoint is mounted (i.e. MCP is enabled).

    Probes ``/mcp/`` (``/mcp`` falls through to the SPA). A mounted MCP app
    answers a bare GET with a JSON 4xx; an unmounted path hits the SPA catch-all
    (``200 text/html``) or a plain 404, so this keys off content type, not status.
    """
    try:
        response = httpx.get(f"{server_url}/mcp/", timeout=5.0)
    except httpx.HTTPError:
        return False
    if response.status_code == 404:
        return False
    return "text/html" not in response.headers.get("content-type", "")


def _resolve_notebook_dir(
    path_arg: str, python_version: str | None, initialize_environment: bool
) -> Path:
    """Return an existing notebook dir, or scaffold one and return its path.

    :func:`create_notebook` slugifies the name, so the returned directory can
    differ from the requested path.
    """
    path = Path(path_arg).expanduser().resolve()
    if (path / "notebook.toml").is_file():
        return path

    from strata.notebook.writer import create_notebook

    return create_notebook(
        path.parent,
        path.name,
        python_version,
        initialize_environment=initialize_environment,
    )


def _server_log_path(notebook_dir: Path) -> Path:
    """Where a spawned server's output goes: the notebook's gitignored runtime dir."""
    return notebook_dir / ".strata" / "server.log"


def _spawn_server(host: str, port: int, notebook_dir: Path) -> subprocess.Popen:
    """Start a notebook server with MCP enabled in personal mode.

    MCP and the mode go in the child's environment because the ``/mcp`` mount is
    decided at import time. The storage root is *notebook_dir*'s parent, since
    ``POST /open`` rejects notebooks outside the server's storage root. Its output
    goes to a log file: on this terminal it would interleave with the launcher's own
    lines and draw over the TUI.
    """
    env = dict(os.environ)
    env["STRATA_MCP_ENABLED"] = "true"
    env["STRATA_DEPLOYMENT_MODE"] = "personal"
    env["STRATA_HOST"] = host
    env["STRATA_PORT"] = str(port)
    env["STRATA_NOTEBOOK_STORAGE_DIR"] = str(notebook_dir.parent)
    log_path = _server_log_path(notebook_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    private_file(log_path)
    with open(log_path, "wb") as log:
        return subprocess.Popen(
            [sys.executable, "-m", "strata"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def _await_health(server_url: str, proc: subprocess.Popen, timeout: float = 30.0) -> bool:
    """Poll ``/health`` until the spawned server answers, dies, or *timeout*."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False  # server exited before becoming healthy
        if _server_alive(server_url):
            return True
        time.sleep(0.3)
    return False


class _OpenError(RuntimeError):
    """``POST /open`` failed; carries a human-readable reason."""


def _open_session(server_url: str, notebook_dir: Path) -> str:
    """``POST /v1/notebooks/open`` for *notebook_dir*; return its ``session_id``."""
    return _post_for_session(
        f"{server_url}/v1/notebooks/open", {"path": str(notebook_dir)}, server_url
    )


def _post_for_session(
    url: str, body: dict[str, str] | None, server_url: str, *, timeout: float = 60.0
) -> str:
    """POST to a route that opens a notebook; return its ``session_id``.

    Raises :class:`_OpenError` carrying the server's ``detail``, e.g. the 400 a
    reused server gives when the notebook is outside its storage root.
    """
    try:
        response = httpx.post(url, json=body, timeout=timeout)
    except httpx.HTTPError as exc:
        raise _OpenError(f"cannot reach {server_url}: {exc}") from exc
    if response.is_error:
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = str(body.get("detail") or "")
        except ValueError:
            detail = response.text.strip()
        suffix = f": {detail}" if detail else ""
        raise _OpenError(f"server returned {response.status_code}{suffix}")
    return response.json()["session_id"]


def _establish_ssh_worker(server_url: str, session_id: str, ssh_target: str) -> None:
    """Provision, tunnel and register an SSH worker via the running server.

    Best-effort: a failure warns rather than aborts. The long timeout covers a
    first connect that installs ``strata-worker`` on the box.
    """
    print(_dim(f"connecting ssh worker {ssh_target} (first connect may install strata-worker)…"))
    try:
        response = httpx.post(
            f"{server_url}/v1/notebooks/{session_id}/workers/ssh",
            json={"ssh_target": ssh_target, "set_default": True},
            timeout=600.0,
        )
    except httpx.HTTPError as exc:
        print(f"warning: could not connect ssh worker {ssh_target}: {exc}", file=sys.stderr)
        return
    if response.is_error:
        detail = ""
        try:
            body = response.json()
            if isinstance(body, dict):
                detail = str(body.get("detail") or "")
        except ValueError:
            detail = response.text.strip()
        print(
            f"warning: ssh worker {ssh_target} not connected "
            f"({response.status_code}{f': {detail}' if detail else ''})",
            file=sys.stderr,
        )
        return
    name = (response.json().get("worker") or {}).get("name", ssh_target)
    print(_dim(f"ssh worker '{name}' ready; cells run on {ssh_target} by default"))


def _interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def _terminate(proc: subprocess.Popen) -> None:
    """Stop a spawned server, escalating to kill if it ignores SIGTERM."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def _agent_guidance(session_id: str) -> str:
    """The working-agreement body written into the notebook's CLAUDE.md."""
    from strata.agent_guidance import section

    return section("notebook_claude_md").replace("{session_id}", session_id)


def _write_agent_config(notebook_dir: Path, server_url: str, session_id: str) -> None:
    """Write ``.mcp.json`` and the CLAUDE.md working agreement into *notebook_dir*.

    ``.mcp.json`` is overwritten. In CLAUDE.md only the region between our
    managed markers is rewritten, so the user's own instructions survive.
    """
    # Trailing slash required: Starlette 307-redirects ``/mcp`` to ``/mcp/`` and MCP HTTP
    # clients (Claude Code) drop the POST body across the redirect, so the handshake fails.
    mcp_url = f"{server_url}/mcp/"
    mcp_config = {"mcpServers": {"strata-notebook": {"type": "http", "url": mcp_url}}}
    (notebook_dir / ".mcp.json").write_text(
        json.dumps(mcp_config, indent=2) + "\n", encoding="utf-8"
    )

    block = f"{_BLOCK_START}\n{_agent_guidance(session_id)}{_BLOCK_END}\n"
    claude_md = notebook_dir / "CLAUDE.md"
    if claude_md.exists():
        text = claude_md.read_text(encoding="utf-8")
        if _BLOCK_START in text and _BLOCK_END in text:
            before = text[: text.index(_BLOCK_START)]
            after = text[text.index(_BLOCK_END) + len(_BLOCK_END) :]
            new_text = before + block.rstrip("\n") + after
        else:
            new_text = text.rstrip("\n") + "\n\n" + block
    else:
        new_text = block
    claude_md.write_text(new_text, encoding="utf-8")


def _print_ready(notebook_dir: Path, server_url: str, session_id: str, *, tui: bool = True) -> None:
    check = _green("✓")
    print()
    print(f"{check} notebook  {notebook_dir}")
    print(f"{check} server    {server_url}  ({_dim('MCP at ' + server_url + '/mcp')})")
    print(f"{check} session   {session_id}")
    print()
    print("Point your coding agent at it, in another terminal:")
    print()
    print(f"    cd {notebook_dir} && claude")
    print()
    print(_dim("The agent auto-connects via .mcp.json and drives this notebook;"))
    if tui:
        print(_dim("watch it happen live in the TUI below. Quit the TUI to stop."))
    else:
        # --no-tui: there is no viewer below to watch or quit.
        print(_dim(f"watch it happen live in the web UI at {server_url}."))
    # A script reading a pipe needs the session id now, not when the buffer fills.
    print(flush=True)


def _launch_tui(server_url: str, session_id: str) -> None:
    """Attach the read-only TUI spectator to *session_id* (blocks until quit)."""
    from strata.notebook.tui.app import NotebookTUI
    from strata.notebook.tui.client import TuiClient

    client = TuiClient(server_url=server_url)
    NotebookTUI(client=client, session_id=session_id, notebook_path=None).run()


def agent_main(args: argparse.Namespace) -> int:
    """Entry point for ``strata agent``."""
    server_url = args.server.rstrip("/")

    try:
        notebook_dir = _resolve_notebook_dir(args.path, args.python_version, not args.no_env)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    spawned: subprocess.Popen | None = None
    # SIGTERM unwinds like Ctrl-C, so the finally below stops the server we started.
    previous_sigterm = signal.signal(signal.SIGTERM, _interrupt)
    try:
        if _server_alive(server_url):
            if not _mcp_mounted(server_url):
                print(
                    f"error: a server is already running at {server_url} but its MCP "
                    f"endpoint is disabled.\n"
                    f"hint: stop it and let `strata agent` start one, or restart it "
                    f"with STRATA_MCP_ENABLED=true",
                    file=sys.stderr,
                )
                return 2
            print(_dim(f"reusing server at {server_url}"))
        else:
            host, port = _server_endpoints(server_url)
            log_path = _server_log_path(notebook_dir)
            print(
                _dim(f"starting server on {host}:{port} (MCP enabled), logging to {log_path}…"),
                flush=True,
            )
            spawned = _spawn_server(host, port, notebook_dir)
            if not _await_health(server_url, spawned):
                print(
                    f"error: server failed to start at {server_url}; see {log_path}",
                    file=sys.stderr,
                )
                return 2
            if not _mcp_mounted(server_url):
                print(
                    "error: the MCP endpoint did not mount; the [mcp] extra is "
                    "likely missing.\nhint: in a checkout, `uv sync --extra mcp`; "
                    "for an installed tool, `uv tool install 'strata-notebook[mcp,tui]'`",
                    file=sys.stderr,
                )
                return 2

        try:
            session_id = _open_session(server_url, notebook_dir)
        except (_OpenError, KeyError) as exc:
            print(f"error: failed to open notebook session: {exc}", file=sys.stderr)
            if spawned is None:
                print(
                    _dim(
                        "hint: the reused server may confine notebooks to its own "
                        "storage root; stop it so `strata agent` can start one "
                        "scoped to this notebook."
                    ),
                    file=sys.stderr,
                )
            return 2

        _write_agent_config(notebook_dir, server_url, session_id)
        if args.worker_ssh:
            _establish_ssh_worker(server_url, session_id, args.worker_ssh)
        _print_ready(notebook_dir, server_url, session_id, tui=not args.no_tui)

        if args.no_tui:
            if spawned is not None:
                print(_dim("server running; press Ctrl-C to stop."), flush=True)
                try:
                    spawned.wait()
                except KeyboardInterrupt:
                    return 0  # Ctrl-C or SIGTERM: the finally stops the server
            return 0

        _launch_tui(server_url, session_id)
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if spawned is not None:
            _terminate(spawned)


def add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach ``agent`` subcommand arguments to an existing parser."""
    parser.add_argument(
        "path",
        help="Notebook directory to drive (created if it doesn't exist yet)",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("STRATA_TUI_SERVER", "http://localhost:8765"),
        help="Server base URL; reused if already running, else started (default: :8765)",
    )
    parser.add_argument(
        "--python",
        dest="python_version",
        default=None,
        help="Python major.minor for a newly created notebook's venv",
    )
    parser.add_argument(
        "--no-env",
        action="store_true",
        help="When creating a notebook, skip building its venv now",
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help="Set up and open the session but don't launch the TUI spectator",
    )
    parser.add_argument(
        "--worker-ssh",
        dest="worker_ssh",
        default=None,
        metavar="SSH_TARGET",
        help="Provision a remote worker over SSH (user@host) and route cells to it",
    )
