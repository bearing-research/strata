"""Route console chunks from a running remote build to the notebook watching it.

The worker posts chunks to a signed log URL; the dispatching executor registers
which notebook and cell a build belongs to, and the log route asks here.
Process-local on purpose: the dispatching server holds the session's socket. A
chunk for a build this process is not running is dropped: the bundle stays the
record, and a wrong cell's console is worse than none.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# build_id -> (notebook_id, cell_id)
_routes: dict[str, tuple[str, str]] = {}

# (notebook_id, cell_id) -> chars of each stream already shown while running.
# The final report sends only the rest: the frontend appends, so resending all
# would duplicate, and sending none would lose a dropped chunk.
_streamed: dict[tuple[str, str], dict[str, int]] = {}


def register(build_id: str, notebook_id: str, cell_id: str) -> None:
    """Say where a build's console output should be delivered."""
    _routes[build_id] = (notebook_id, cell_id)


def unregister(build_id: str) -> None:
    """Forget a build. Safe to call for one that was never registered."""
    _routes.pop(build_id, None)


def streamed(notebook_id: str, cell_id: str) -> dict[str, int] | None:
    """How much of each stream this cell showed while it ran, or None.

    None means nothing was streamed and the whole console is still to be sent.
    """
    return _streamed.get((notebook_id, cell_id))


def clear_streamed(notebook_id: str, cell_id: str) -> None:
    """Forget that a cell streamed, once its run is fully reported."""
    _streamed.pop((notebook_id, cell_id), None)


async def deliver(build_id: str, stream: str, text: str) -> bool:
    """Broadcast one console chunk to the notebook running *build_id*.

    Returns False when this process is not running that build (a stale worker, or
    a replica that did not dispatch it).
    """
    route = _routes.get(build_id)
    if route is None or not text:
        return False
    notebook_id, cell_id = route

    from strata.notebook.protocol import MessageType
    from strata.notebook.ws import (
        _broadcast_message,
        _make_message,
        next_notebook_sequence,
    )
    from strata.notebook.ws_payloads import CellConsolePayload

    delivered = _streamed.setdefault((notebook_id, cell_id), {})
    delivered[stream] = delivered.get(stream, 0) + len(text)
    await _broadcast_message(
        notebook_id,
        _make_message(
            MessageType.CELL_CONSOLE,
            next_notebook_sequence(notebook_id),
            CellConsolePayload(
                cell_id=cell_id,
                stream="stderr" if stream == "stderr" else "stdout",
                text=text,
            ).model_dump(mode="json"),
        ),
    )
    return True
