"""``strata agent <path>`` — one-command on-ramp for driving a notebook with a coding agent.

Everything needed to let a coding agent (Claude Code) drive a Strata notebook
while a human watches it live already exists — the MCP server, the CLI ops, the
TUI — but assembling them by hand is a fiddly, ordered dance that nobody does.
This command collapses it into one step:

1. create-or-open the notebook directory,
2. start (or reuse) a notebook server with the MCP endpoint enabled,
3. open a warm session — the piece the agent *cannot* do itself
   (``list_notebooks`` only sees already-open sessions),
4. drop a ``.mcp.json`` + a ``CLAUDE.md`` working agreement into the notebook so
   Claude Code auto-connects and knows to drive cells instead of writing scratch
   scripts,
5. launch the read-only TUI attached to that session.

The human then runs ``claude`` in the notebook directory in another pane; it
discovers the MCP server, reads the working agreement, and drives cells that
light up live in the TUI.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx

# ANSI helpers, tty-gated (mirrors strata.notebook.cli).
_USE_COLOR = sys.stdout.isatty()


def _color(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _green(text: str) -> str:
    return _color("32", text)


def _dim(text: str) -> str:
    return _color("90", text)


# Managed block markers so a re-run rewrites its guidance in place instead of
# appending a duplicate, and never clobbers a user's own CLAUDE.md content.
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
    """True if the MCP endpoint is actually mounted (i.e. MCP is enabled).

    Probe ``/mcp/`` — the real endpoint (trailing slash; ``/mcp`` without it
    falls through to the SPA). A mounted FastMCP app answers a bare GET with a
    JSON 4xx (e.g. 406 Not Acceptable). When MCP is *not* mounted — the
    ``[mcp]`` extra is missing, or the server is in service mode — the path hits
    the SPA catch-all, which returns ``200 text/html``. Status alone can't tell
    them apart (both non-404), so key off content type: the SPA is HTML, the MCP
    endpoint is JSON. (Without a bundled frontend the miss is a plain ``404``,
    also excluded.)
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

    If ``<path>/notebook.toml`` exists we reuse it as-is. Otherwise the notebook
    is created via :func:`create_notebook`, which slugifies the name — so we
    return whatever directory it actually wrote, not the requested path.
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


def _spawn_server(host: str, port: int, notebook_dir: Path) -> subprocess.Popen:
    """Start a notebook server with MCP enabled in personal mode.

    ``STRATA_MCP_ENABLED`` and the mode must be set before the child imports
    ``strata.server`` — the ``/mcp`` mount decision is made at import time — so
    they go into the child's environment, not toggled after boot.

    The storage root is scoped to *notebook_dir*'s parent so ``POST /open``
    accepts the notebook: the open handler confines notebook paths to the
    server's storage root, so a notebook living anywhere on disk would 400
    against the default ``~/.strata/notebooks``.
    """
    env = dict(os.environ)
    env["STRATA_MCP_ENABLED"] = "true"
    env["STRATA_DEPLOYMENT_MODE"] = "personal"
    env["STRATA_HOST"] = host
    env["STRATA_PORT"] = str(port)
    env["STRATA_NOTEBOOK_STORAGE_DIR"] = str(notebook_dir.parent)
    return subprocess.Popen([sys.executable, "-m", "strata"], env=env)


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
    """``POST /v1/notebooks/open`` for *notebook_dir*; return its ``session_id``.

    Raises :class:`_OpenError` with the server's ``detail`` surfaced — most
    usefully the "must be inside configured notebook storage" 400 that a reused
    server raises when the notebook lives outside its storage root.
    """
    try:
        response = httpx.post(
            f"{server_url}/v1/notebooks/open",
            json={"path": str(notebook_dir)},
            timeout=60.0,
        )
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
    return f"""# Driving this Strata notebook

This directory is a **Strata notebook**. Operate on it through the connected
**`strata-notebook` MCP server** (auto-configured in `.mcp.json`) — not by
writing throwaway `.py` scripts and running them.

Work done here is captured as versioned, content-addressed cells: identical
computations are deduplicated and never re-run, lineage is explicit, and a human
is watching this session live in a terminal UI. Scratch scripts throw all of
that away, and the watcher can't see them.

**How to work:**

1. Call `list_notebooks` to get the open `session_id` (it is `{session_id}` for
   this launch, but confirm — a new id is minted each time).
2. Use `get_notebook` / `status` / `dag` to see the current cells and state.
3. Build by adding and running cells: `add_cell`, `edit_cell`, `run_cell` (and
   `add_dependency` for packages). Every cell you run appears live in the
   watcher's UI.
4. Use `note` to narrate what you're about to do so the person watching can
   follow along.

**Do not** run Python here with the Bash tool (`python …`, `uv run …`, a REPL,
or a scratch script). That work is invisible to the notebook and to the person
watching, and it isn't cached. Put the code in a cell and `run_cell` it instead
— **this is your scratchpad**. Even a throwaway `print`-only diagnostic cell is
cached by provenance, so re-running it unchanged replays its output instantly
rather than recomputing; leave those cells in place instead of deleting them.

**One exception — side effects and fresh values:** a cell whose point is a side
effect (writing a file, calling an API, mutating external state) or a fresh
value (the clock, `random`, a live endpoint) must not replay a cached result.
Put `# @nocache` on the first line of such a cell so it always re-executes.

Prefer editing an existing cell over re-adding one; prefer `run_cell` over
re-deriving a value you already computed — the cache makes an unchanged rerun
instant.
"""


def _write_agent_config(notebook_dir: Path, server_url: str, session_id: str) -> None:
    """Write ``.mcp.json`` + the CLAUDE.md working agreement into *notebook_dir*.

    ``.mcp.json`` is overwritten wholesale (it's ours). CLAUDE.md is edited
    surgically: only the region between our managed markers is (re)written, so a
    user's own instructions in the same file survive.
    """
    # Trailing slash is required: the server mounts the MCP app at ``/mcp`` and
    # Starlette 307-redirects ``/mcp`` → ``/mcp/``. MCP HTTP clients (Claude
    # Code) drop the POST body across that redirect, so the handshake fails and
    # the notebook tools never register — the agent then can't drive the
    # notebook natively. Point straight at ``/mcp/`` and there's no redirect.
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


def _print_ready(notebook_dir: Path, server_url: str, session_id: str) -> None:
    check = _green("✓")
    print()
    print(f"{check} notebook  {notebook_dir}")
    print(f"{check} server    {server_url}  ({_dim('MCP at ' + server_url + '/mcp')})")
    print(f"{check} session   {session_id}")
    print()
    print("Point your coding agent at it — in another terminal:")
    print()
    print(f"    cd {notebook_dir} && claude")
    print()
    print(_dim("The agent auto-connects via .mcp.json and drives this notebook;"))
    print(_dim("watch it happen live in the TUI below. Quit the TUI to stop."))
    print()


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
        print(_dim(f"starting server on {host}:{port} (MCP enabled)…"))
        spawned = _spawn_server(host, port, notebook_dir)
        if not _await_health(server_url, spawned):
            print(f"error: server failed to start at {server_url}", file=sys.stderr)
            _terminate(spawned)
            return 2
        if not _mcp_mounted(server_url):
            print(
                "error: the MCP endpoint did not mount — the [mcp] extra is "
                "likely missing.\nhint: install it with `uv sync --extra mcp` "
                "(or `--all-extras`)",
                file=sys.stderr,
            )
            _terminate(spawned)
            return 2

    try:
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
        _print_ready(notebook_dir, server_url, session_id)

        if args.no_tui:
            if spawned is not None:
                print(_dim("server running; press Ctrl-C to stop."))
                try:
                    spawned.wait()
                except KeyboardInterrupt:
                    pass
            return 0

        _launch_tui(server_url, session_id)
        return 0
    finally:
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
