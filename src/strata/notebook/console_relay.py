"""Route console chunks from a running remote build to the notebook watching it.

The worker posts numbered chunks to a signed log URL; the dispatching executor
registers which session and cell a build belongs to, and the log route asks here.
The session's sockets live on the node that dispatched the build, so a chunk that
lands on another node goes through the shared build store, which the dispatching
node polls (multi-node only). A chunk for a build no node is running is dropped:
the bundle stays the record, and a wrong cell's console is worse than none.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata.transforms.build_store import BuildStore

logger = logging.getLogger(__name__)

# The end of each stream kept for a viewer who opens the notebook mid-run.
_TAIL_CHARS = 64 * 1024
# Chunks held behind a missing one. A lost chunk never arrives, so past this the
# rest waits for the final report, which sends whatever was not shown.
_MAX_PENDING_CHUNKS = 64
# How often the dispatching node looks for chunks another node received.
_SHARED_POLL_SECONDS = 0.5


@dataclass
class _Stream:
    next_seq: int = 0
    pending: dict[int, str] = field(default_factory=dict)
    tail: str = ""


@dataclass
class _Run:
    notebook_id: str
    cell_id: str
    streams: dict[str, _Stream] = field(default_factory=dict)


# build_id -> the run it belongs to
_runs: dict[str, _Run] = {}

# (notebook_id, cell_id) -> chars of each stream already shown while running.
# The final report sends only the rest: the frontend appends, so resending all
# would duplicate, and sending none would lose a dropped chunk.
_streamed: dict[tuple[str, str], dict[str, int]] = {}


def register(build_id: str, notebook_id: str, cell_id: str) -> None:
    """Say where a build's console output should be delivered (*notebook_id* is the session id)."""
    _runs[build_id] = _Run(notebook_id, cell_id)


def unregister(build_id: str) -> None:
    """Forget a build. Safe to call for one that was never registered."""
    _runs.pop(build_id, None)


def streamed(notebook_id: str, cell_id: str) -> dict[str, int] | None:
    """How much of each stream this cell showed while it ran, or None.

    None means nothing was streamed and the whole console is still to be sent.
    """
    return _streamed.get((notebook_id, cell_id))


def clear_streamed(notebook_id: str, cell_id: str) -> None:
    """Forget that a cell streamed, once its run is fully reported."""
    _streamed.pop((notebook_id, cell_id), None)


def live_console(notebook_id: str, cell_id: str) -> dict[str, str]:
    """The last part of each stream a running cell has shown, by stream name.

    Empty when the cell is not streaming. A stream that has shown nothing yet is
    absent, so its last run's text stays as every connected viewer still sees it.
    """
    for run in _runs.values():
        if run.notebook_id == notebook_id and run.cell_id == cell_id:
            return {name: s.tail for name, s in run.streams.items() if s.next_seq > 0}
    return {}


async def deliver(build_id: str, stream: str, seq: int, text: str) -> bool:
    """Broadcast one numbered console chunk to the notebook running *build_id*.

    Chunks are shown in ``seq`` order, each once: a repeat is ignored and one that
    arrives early waits for the gap, so what was shown stays a prefix of the stream.
    Returns False when this process is not running that build (a stale worker, or
    a node that did not dispatch it).
    """
    run = _runs.get(build_id)
    if run is None:
        return False
    stream = "stderr" if stream == "stderr" else "stdout"
    state = run.streams.setdefault(stream, _Stream())
    if seq < state.next_seq or seq in state.pending:
        return True
    if seq > state.next_seq:
        if len(state.pending) < _MAX_PENDING_CHUNKS:
            state.pending[seq] = text
        return True

    ready = [(seq, text)]
    state.next_seq += 1
    while state.next_seq in state.pending:
        ready.append((state.next_seq, state.pending.pop(state.next_seq)))
        state.next_seq += 1

    from strata.notebook.protocol import MessageType
    from strata.notebook.ws import (
        _broadcast_message,
        _make_message,
        next_notebook_sequence,
    )
    from strata.notebook.ws_payloads import CellConsolePayload

    delivered = _streamed.setdefault((run.notebook_id, run.cell_id), {})
    for chunk_seq, chunk in ready:
        delivered[stream] = delivered.get(stream, 0) + len(chunk)
        state.tail = (state.tail + chunk)[-_TAIL_CHARS:]
        await _broadcast_message(
            run.notebook_id,
            _make_message(
                MessageType.CELL_CONSOLE,
                next_notebook_sequence(run.notebook_id),
                CellConsolePayload(
                    cell_id=run.cell_id,
                    stream=stream,
                    text=chunk,
                    chunk_seq=chunk_seq,
                ).model_dump(mode="json"),
            ),
        )
    return True


async def deliver_shared(build_id: str, store: BuildStore) -> None:
    """Deliver the chunks of *build_id* that other nodes left in the shared store."""
    try:
        chunks = await asyncio.to_thread(store.take_console_chunks, build_id)
    except Exception:
        # Console is advisory: a failed read costs live lines, never the cell.
        logger.warning("Could not read relayed console for build %s", build_id, exc_info=True)
        return
    for stream, seq, text in chunks:
        await deliver(build_id, stream, seq, text)


@contextlib.asynccontextmanager
async def relaying(
    build_id: str,
    notebook_id: str,
    cell_id: str,
    shared_store: BuildStore | None = None,
    poll_seconds: float = _SHARED_POLL_SECONDS,
) -> AsyncIterator[None]:
    """Deliver a build's console to a cell for the duration of the block.

    With *shared_store* (a multi-node deployment) also poll it for chunks that
    reached another node, and clear what is left at the end.
    """
    register(build_id, notebook_id, cell_id)

    async def _follow(store: BuildStore) -> None:
        while True:
            await asyncio.sleep(poll_seconds)
            await deliver_shared(build_id, store)

    follower = asyncio.create_task(_follow(shared_store)) if shared_store is not None else None
    try:
        yield
    finally:
        unregister(build_id)
        if follower is not None and shared_store is not None:
            follower.cancel()
            await asyncio.gather(follower, return_exceptions=True)
            try:
                await asyncio.to_thread(shared_store.delete_console_chunks, build_id)
            except Exception:
                logger.warning(
                    "Could not clear relayed console for build %s", build_id, exc_info=True
                )
