"""``strata tutorial [name]``: open a copy of a bundled sample, or list them."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

from vestigo.file_modes import private_file

from strata.notebook.agent_launch import (
    _await_health,
    _dim,
    _green,
    _interrupt,
    _OpenError,
    _post_for_session,
    _server_alive,
    _server_endpoints,
    _terminate,
)
from strata.notebook.samples import SAMPLES, get_sample

# Syncing a fresh venv and importing into it can take minutes on a slow connection.
_OPEN_TIMEOUT_SECONDS = 600.0


def _print_samples() -> None:
    print("Samples:")
    print()
    width = max(len(sample.name) for sample in SAMPLES)
    for sample in SAMPLES:
        print(f"  {sample.name.ljust(width)}  {sample.description}")
    print()
    print(f"Open a copy with: strata tutorial {SAMPLES[0].name}")


def _spawn_server(host: str, port: int) -> tuple[subprocess.Popen, Path]:
    """Start a server on its default storage root; its output would bury the URL."""
    from strata.config import StrataConfig

    env = dict(os.environ)
    env["STRATA_HOST"] = host
    env["STRATA_PORT"] = str(port)
    log_path = Path(StrataConfig.load().notebook_storage_dir).parent / "tutorial-server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    private_file(log_path)
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "strata"], env=env, stdout=log, stderr=subprocess.STDOUT
        )
    return proc, log_path


def tutorial_main(args: argparse.Namespace) -> int:
    """Entry point for ``strata tutorial``."""
    if args.name is None:
        _print_samples()
        return 0
    sample = get_sample(args.name)
    if sample is None:
        print(f"error: no sample named {args.name!r}", file=sys.stderr)
        _print_samples()
        return 2

    server_url = args.server.rstrip("/")
    spawned: subprocess.Popen | None = None
    # SIGTERM unwinds like Ctrl-C, so the finally below stops the server we started.
    previous_sigterm = signal.signal(signal.SIGTERM, _interrupt)
    try:
        if not _server_alive(server_url):
            host, port = _server_endpoints(server_url)
            spawned, log_path = _spawn_server(host, port)
            print(_dim(f"starting server on {host}:{port}, logging to {log_path}…"), flush=True)
            if not _await_health(server_url, spawned):
                print(
                    f"error: server failed to start at {server_url}; see {log_path}",
                    file=sys.stderr,
                )
                return 2

        print(_dim("copying the sample and preparing its environment…"), flush=True)
        try:
            session_id = _post_for_session(
                f"{server_url}/v1/notebooks/samples/{sample.name}/open",
                None,
                server_url,
                timeout=_OPEN_TIMEOUT_SECONDS,
            )
        except (_OpenError, KeyError) as exc:
            print(f"error: failed to open {sample.name}: {exc}", file=sys.stderr)
            return 2

        print()
        print(f"{_green('✓')} {sample.title}")
        print(f"  {server_url}/#/notebook/{session_id}")
        print(flush=True)
        if spawned is not None:
            print(_dim("server running; press Ctrl-C to stop."), flush=True)
            try:
                spawned.wait()
            except KeyboardInterrupt:
                return 0  # Ctrl-C or SIGTERM: the finally stops the server
        return 0
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)
        if spawned is not None:
            _terminate(spawned)


def add_tutorial_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach ``tutorial`` subcommand arguments to an existing parser."""
    parser.add_argument("name", nargs="?", default=None, help="Sample to open; omit to list them")
    parser.add_argument(
        "--server",
        default=os.environ.get("STRATA_TUI_SERVER", "http://localhost:8765"),
        help="Server base URL; reused if already running, else started (default: :8765)",
    )
