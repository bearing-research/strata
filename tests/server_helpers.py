"""Stopping a Strata server a test ran on a thread."""

import sys
import threading

import uvicorn


def shutdown_wait_seconds() -> float:
    """How long to wait for a server's shutdown: past its drain of active scans."""
    from strata.server import DRAIN_TIMEOUT_SECONDS

    return DRAIN_TIMEOUT_SECONDS + 15


def stop_strata_server(server: uvicorn.Server, thread: threading.Thread, port: int) -> None:
    """Stop *server* and wait out its shutdown; raise if it outlives the wait.

    The lifespan shutdown drains scans for up to ``DRAIN_TIMEOUT_SECONDS``, then stops the
    build runner, and ends by clearing ``strata.server._state``: a server still in it would
    clear the next test's. The raise only reports that: the thread lives on. It is not raised
    over an exception already on its way out, which would hide the test's own failure.
    """
    wait = shutdown_wait_seconds()
    server.should_exit = True
    thread.join(timeout=wait)
    if thread.is_alive() and sys.exc_info()[1] is None:
        raise RuntimeError(
            f"Strata server on port {port} still shutting down after {wait} s;"
            " a later test's server state may be cleared by it"
        )
