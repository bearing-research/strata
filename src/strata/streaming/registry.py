"""Stream registry — the live stream/cleanup tables for unified materialize.

Owns the ``stream_id -> StreamState`` table and the per-stream TTL cleanup tasks
that ``server.py`` used to hold as ``_streams`` / ``_stream_cleanup_tasks`` and
mutate through the free ``_schedule_stream_cleanup`` / ``_cancel_stream_cleanup``
helpers. Held on ``ServerState`` as ``state.streams`` and shared by the
materialize/streams handlers and the builds router (its identity build-status
fast-path reads the registry).

Phase 1 of the stream-runtime extraction (#302). The scan table (``scans``) and
the prefetch counters stay on ``ServerState`` for now; the scan-side cleanup
(prefetch discard + scan pop) is injected as the ``on_expire`` callback so this
module stays unaware of prefetch. See
``docs/internal/design-stream-runtime-extraction.md``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata.transforms.build_qos import BuildSlot
    from strata.types import ReadPlan


@dataclass
class StreamState:
    """State for a streaming materialize operation.

    Tracks the read plan, streaming progress, and artifact metadata
    for a unified materialize request in stream mode.
    """

    stream_id: str
    plan: ReadPlan  # The underlying scan plan
    artifact_id: str  # Artifact being built
    artifact_version: int
    created_at: float  # Unix timestamp
    mode: str = "stream"  # "stream" for client streaming, "artifact" for background build
    name: str | None = None
    tenant: str | None = None
    executor_ref: str = "scan@v1"
    started: bool = False  # True once streaming has begun
    completed: bool = False  # True once streaming finished
    bytes_streamed: int = 0  # Bytes streamed to client so far
    started_at: float | None = None
    completed_at: float | None = None
    error_message: str | None = None
    background_task: asyncio.Task[None] | None = None
    build_slot: BuildSlot | None = None
    qos_tenant_id: str | None = None


class StreamRegistry:
    """The live ``stream_id -> StreamState`` table plus TTL cleanup scheduling.

    ``on_expire`` runs the scan-side cleanup (prefetch discard + scan-table pop)
    for a stream whose TTL elapsed; it is injected by ``ServerState`` so the
    registry stays unaware of the prefetch/scan state those concerns own until a
    later extraction phase folds them in too.
    """

    def __init__(
        self,
        ttl_seconds: float,
        *,
        on_expire: Callable[[str], None] | None = None,
    ) -> None:
        self._streams: dict[str, StreamState] = {}
        self._cleanup_tasks: dict[str, asyncio.Task[None]] = {}
        self._ttl_seconds = ttl_seconds
        self._on_expire = on_expire

    # --- lookup -------------------------------------------------------------

    def get(self, stream_id: str) -> StreamState | None:
        return self._streams.get(stream_id)

    def __contains__(self, stream_id: str) -> bool:
        return stream_id in self._streams

    def active_streams(self) -> list[StreamState]:
        """Snapshot of the live streams (for graceful-shutdown cancellation)."""
        return list(self._streams.values())

    # --- mutation -----------------------------------------------------------

    def register(self, stream_state: StreamState) -> None:
        self._streams[stream_state.stream_id] = stream_state

    def pop(self, stream_id: str) -> StreamState | None:
        return self._streams.pop(stream_id, None)

    # --- cleanup lifecycle --------------------------------------------------

    def cancel_cleanup(self, stream_id: str) -> None:
        """Cancel any pending cleanup task for a stream."""
        task = self._cleanup_tasks.pop(stream_id, None)
        if task is not None:
            task.cancel()

    def schedule_cleanup(self, stream_id: str, scan_id: str | None = None) -> None:
        """Remove completed or abandoned stream state after the configured TTL."""
        self.cancel_cleanup(stream_id)

        async def _cleanup() -> None:
            try:
                await asyncio.sleep(self._ttl_seconds)
            except asyncio.CancelledError:
                return
            finally:
                if self._cleanup_tasks.get(stream_id) is asyncio.current_task():
                    self._cleanup_tasks.pop(stream_id, None)

            if scan_id is not None and self._on_expire is not None:
                self._on_expire(scan_id)
            self._streams.pop(stream_id, None)

        self._cleanup_tasks[stream_id] = asyncio.create_task(_cleanup())

    def shutdown_cleanups(self) -> None:
        """Cancel and drop all pending cleanup tasks (graceful shutdown)."""
        for task in list(self._cleanup_tasks.values()):
            task.cancel()
        self._cleanup_tasks.clear()
