"""Stream registry: the live ``stream_id -> StreamState`` table and per-stream TTL cleanup.

Held on ``ServerState`` as ``state.streams``. Scan-side cleanup (prefetch discard, scan
pop) is injected as the ``on_expire`` callback so this module stays unaware of prefetch.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio.to_thread

if TYPE_CHECKING:
    from strata.transforms.build_qos import BuildSlot
    from strata.types import ReadPlan


@dataclass
class StreamState:
    """State of a stream-mode materialize: read plan, streaming progress and artifact metadata."""

    stream_id: str
    plan: ReadPlan
    artifact_id: str  # Being built
    artifact_version: int
    created_at: float  # Unix timestamp
    mode: str = "stream"  # "stream" for client streaming, "artifact" for background build
    name: str | None = None
    tenant: str | None = None
    executor_ref: str = "scan@v1"
    started: bool = False
    completed: bool = False
    bytes_streamed: int = 0
    started_at: float | None = None
    completed_at: float | None = None
    error_message: str | None = None
    background_task: asyncio.Task[None] | None = None
    build_slot: BuildSlot | None = None
    qos_tenant_id: str | None = None


class StreamRegistry:
    """The live ``stream_id -> StreamState`` table plus TTL cleanup scheduling.

    ``on_expire``, injected by ``ServerState``, runs the scan-side cleanup for a stream whose
    TTL elapsed; ``on_drop`` gets the dropped stream itself.
    """

    def __init__(
        self,
        ttl_seconds: float,
        *,
        on_expire: Callable[[str], None] | None = None,
        on_claim: Callable[[str, float], None] | None = None,
        on_release: Callable[[str], None] | None = None,
        on_drop: Callable[[StreamState], None] | None = None,
    ) -> None:
        self._streams: dict[str, StreamState] = {}
        self._cleanup_tasks: dict[str, asyncio.Task[None]] = {}
        self._ttl_seconds = ttl_seconds
        self._on_expire = on_expire
        # Injected so this module stays unaware of where ownership is recorded; both None
        # (and free) on a single node.
        self._on_claim = on_claim
        self._on_release = on_release
        # Told of each stream its TTL drops, to settle an artifact nobody built.
        self._on_drop = on_drop

    def get(self, stream_id: str) -> StreamState | None:
        return self._streams.get(stream_id)

    def __contains__(self, stream_id: str) -> bool:
        return stream_id in self._streams

    def active_streams(self) -> list[StreamState]:
        """Snapshot of the live streams (for graceful-shutdown cancellation)."""
        return list(self._streams.values())

    def register(self, stream_state: StreamState) -> None:
        """Add a stream to the table; :meth:`claim` then advertises it to other nodes."""
        self._streams[stream_state.stream_id] = stream_state

    async def claim(self, stream_id: str) -> None:
        """Record this node as the one serving ``stream_id``, before its URL is handed out."""
        if self._on_claim is not None:
            # A database write on a multi-node deployment.
            await anyio.to_thread.run_sync(self._on_claim, stream_id, self._ttl_seconds)

    def pop(self, stream_id: str) -> StreamState | None:
        popped = self._streams.pop(stream_id, None)
        if popped is not None and self._on_release is not None:
            self._on_release(stream_id)
        return popped

    def cancel_cleanup(self, stream_id: str) -> None:
        """Cancel any pending cleanup task for a stream."""
        task = self._cleanup_tasks.pop(stream_id, None)
        if task is not None:
            task.cancel()

    def schedule_cleanup(self, stream_id: str, scan_id: str | None = None) -> None:
        """Remove completed or abandoned stream state after the configured TTL.

        Always pass ``scan_id`` when one exists: this replaces any pending cleanup, and
        ``on_expire`` (the only path that frees the scan and its prefetch) fires only with a
        ``scan_id``, so omitting it leaks the ReadPlan for the life of the process.
        """
        if scan_id is None and stream_id in self._streams:
            existing = self._streams[stream_id]
            plan = getattr(existing, "plan", None)
            if plan is not None:
                scan_id = getattr(plan, "scan_id", None)
        self.cancel_cleanup(stream_id)

        async def _cleanup() -> None:
            try:
                await asyncio.sleep(self._ttl_seconds)
            except asyncio.CancelledError:
                # cancel_cleanup and shutdown_cleanups drop the entry themselves.
                return
            # Not in a ``finally``: a task stranded by a closed loop is closed
            # outside any loop, where current_task() raises.
            if self._cleanup_tasks.get(stream_id) is asyncio.current_task():
                self._cleanup_tasks.pop(stream_id, None)

            if scan_id is not None and self._on_expire is not None:
                self._on_expire(scan_id)
            dropped = self._streams.pop(stream_id, None)
            if dropped is not None:
                # Both can write the store, so off the loop.
                await anyio.to_thread.run_sync(self._settle_dropped, dropped)

        self._cleanup_tasks[stream_id] = asyncio.create_task(_cleanup())

    def _settle_dropped(self, dropped: StreamState) -> None:
        if self._on_release is not None:
            # Bypasses pop(), so release the claim here or an expired stream keeps
            # advertising this node until its row expires.
            self._on_release(dropped.stream_id)
        if self._on_drop is not None:
            self._on_drop(dropped)

    def shutdown_cleanups(self) -> None:
        """Cancel and drop all pending cleanup tasks (graceful shutdown)."""
        for task in list(self._cleanup_tasks.values()):
            task.cancel()
        self._cleanup_tasks.clear()
