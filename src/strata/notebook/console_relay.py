"""Route console chunks from a running remote build to the notebook watching it.

A cell dispatched to a worker used to be silent until it finished: output
arrived in the result bundle, and the only live signal was the ``cell_status``
frame naming the machine. For a training loop that is most of an hour with
nothing on screen, and for one that dies at hour three the tail is the whole
diagnostic.

The worker posts chunks to a signed log URL as they are produced. This is the
piece in the middle: the dispatching executor registers which notebook and
cell a build belongs to, and the log route asks here where a chunk should go.

Process-local on purpose. The registration and the WebSocket it feeds live in
the same process, because the server dispatching the cell is the one holding
the session's socket. A chunk for a build this process is not running is
dropped rather than guessed at — console is advisory, the bundle stays the
record, and a wrong cell's console is worse than none.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# build_id -> (notebook_id, cell_id)
_routes: dict[str, tuple[str, str]] = {}

# (notebook_id, cell_id) that streamed at least one chunk during this run.
# The finished-execution broadcast sends the complete stdout and stderr, which
# would show everything a second time under what was already streamed; it
# consults this and skips what it has already shown.
# How much of each stream a cell has already shown while it ran, so the
# report at the end can send what did not make it rather than all of it again
# (the frontend appends) or nothing (a dropped chunk would be lost for good).
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

    Returns whether it was delivered. False means this process is not running
    that build — a stale worker, or a replica that did not dispatch it.
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
