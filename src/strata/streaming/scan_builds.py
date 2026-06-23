"""Scan-build manager — the active scan table + opportunistic prefetch.

Owns the ``scan_id -> ReadPlan`` table and the best-effort first-row-group
prefetch that ``server.py`` held as ``ServerState.scans`` / ``_prefetch_*`` and
mutated through the free ``_start_prefetch`` / ``_discard_prefetch`` helpers.
Held on ``ServerState`` as ``state.scan_builds``.

Phase 2a of the stream-runtime extraction (#302): the scan table + prefetch.
The background build (``_build_identity_artifact`` / ``_finalize_written_blob``)
still lives in ``server.py`` and calls this manager's API; a later phase folds it
in too. The prefetch methods that need shared infra (fetcher, fetch executor,
draining flag) take ``state`` per-call, so the manager owns the prefetch *state*
without a back-reference to ``ServerState``. See
``docs/internal/design-stream-runtime-extraction.md``.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from strata.logging import get_logger

if TYPE_CHECKING:
    from strata.server import ServerState
    from strata.types import ReadPlan

logger = get_logger(__name__)


class ScanBuildManager:
    """The active scan table plus opportunistic first-row-group prefetch."""

    def __init__(self, prefetch_concurrency: int = 4) -> None:
        # Active scans (scan_id -> ReadPlan), registered when a client will stream.
        self.scans: dict[str, ReadPlan] = {}

        # Prefetch: limit concurrent prefetches so clients spamming POST /scan
        # without consuming the streams can't exhaust resources (independent of
        # streaming concurrency).
        self._prefetch_semaphore = asyncio.Semaphore(prefetch_concurrency)
        self._prefetch_futures: dict[str, asyncio.Task[None]] = {}
        self._started = 0  # Total prefetches started
        self._used = 0  # Prefetches consumed by streaming
        self._wasted = 0  # Prefetches discarded (scan deleted/abandoned)
        self._skipped = 0  # Prefetches skipped (server busy)
        self._in_flight = 0  # Prefetches actively fetching

    # --- scan table ---------------------------------------------------------

    def register_scan(self, plan: ReadPlan) -> None:
        self.scans[plan.scan_id] = plan

    def get_scan(self, scan_id: str) -> ReadPlan | None:
        return self.scans.get(scan_id)

    def pop_scan(self, scan_id: str) -> ReadPlan | None:
        return self.scans.pop(scan_id, None)

    def __contains__(self, scan_id: str) -> bool:
        return scan_id in self.scans

    # --- prefetch -----------------------------------------------------------

    def discard_prefetch(self, scan_id: str, *, count_wasted: bool) -> None:
        """Cancel or discard any prefetched first chunk for a scan."""
        plan = self.scans.get(scan_id)
        prefetched_ready = plan is not None and plan.prefetched_first is not None
        task = self._prefetch_futures.pop(scan_id, None)

        if task is not None and not task.done():
            task.cancel()
            if count_wasted:
                self._wasted += 1
        elif prefetched_ready and count_wasted:
            self._wasted += 1

        if plan is not None:
            plan.prefetched_first = None

    def start_prefetch(self, state: ServerState, plan: ReadPlan) -> None:
        """Best-effort prefetch of the first row group for stream-mode reads."""
        if (
            not plan.tasks
            or plan.scan_id in self._prefetch_futures
            or plan.prefetched_first is not None
        ):
            return

        async def _prefetch() -> None:
            if state._draining:
                return

            # Prefetch is opportunistic; skip instead of queueing behind work.
            if getattr(self._prefetch_semaphore, "_value", 0) <= 0:
                self._skipped += 1
                return

            await self._prefetch_semaphore.acquire()
            self._started += 1
            self._in_flight += 1
            try:
                loop = asyncio.get_running_loop()
                plan.prefetched_first = await loop.run_in_executor(
                    state._fetch_executor,
                    state.fetcher.fetch_as_stream_bytes,
                    plan.tasks[0],
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("prefetch_failed", scan_id=plan.scan_id, error=str(e))
            finally:
                self._in_flight -= 1
                self._prefetch_semaphore.release()

        task = asyncio.create_task(_prefetch())
        self._prefetch_futures[plan.scan_id] = task

        def _cleanup_prefetch(done_task: asyncio.Task[None]) -> None:
            if self._prefetch_futures.get(plan.scan_id) is done_task:
                self._prefetch_futures.pop(plan.scan_id, None)

        task.add_done_callback(_cleanup_prefetch)

    async def consume_prefetched_first(self, plan: ReadPlan, scan_id: str) -> bytes | None:
        """Return the prefetched first row group if one is (or becomes) warm, else None.

        Called by the background build at row-group index 0. Consumes the eagerly
        prefetched chunk (counting it ``used``), briefly waiting on an in-flight
        prefetch; if none materializes, discards it (counting it ``wasted``) so the
        build falls back to a direct fetch.
        """
        if plan.prefetched_first is not None:
            chunk = plan.prefetched_first
            plan.prefetched_first = None
            self._used += 1
            return chunk

        prefetch_task = self._prefetch_futures.get(scan_id)
        if prefetch_task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(prefetch_task), timeout=0.05)
            except TimeoutError:
                pass
            if plan.prefetched_first is not None:
                chunk = plan.prefetched_first
                plan.prefetched_first = None
                self._used += 1
                return chunk
            self.discard_prefetch(scan_id, count_wasted=True)
        return None

    def prefetch_metrics(self) -> dict[str, int]:
        """Prefetch counters for observability (``/metrics`` + prometheus)."""
        return {
            "started": self._started,
            "used": self._used,
            "wasted": self._wasted,
            "skipped": self._skipped,
            "in_flight": self._in_flight,
        }

    # --- cleanup callback ---------------------------------------------------

    def expire_scan(self, scan_id: str) -> None:
        """Scan-side cleanup for an expired stream (the registry's ``on_expire``).

        Discards any prefetched first chunk and drops the scan from the table.
        """
        self.discard_prefetch(scan_id, count_wasted=True)
        self.pop_scan(scan_id)
