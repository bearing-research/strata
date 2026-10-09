"""Scan-build manager: active scan table, first-row-group prefetch, and the background build.

Held on ``ServerState`` as ``state.scan_builds``. Methods that need shared
infra take ``state`` per call, so there is no back-reference to
``ServerState``. The manager creates the background build task; the handler
shields it.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.ipc as ipc

from strata.fast_io import (
    IncrementalIpcMerger,
    validate_ipc_stream,
    validate_ipc_stream_reader,
)
from strata.logging import get_logger
from strata.pool_metrics import get_pool_tracker

if TYPE_CHECKING:
    from strata.server import ServerState
    from strata.streaming.registry import StreamState
    from strata.types import ReadPlan

logger = get_logger(__name__)


def record_scan_complete(
    state: ServerState,
    plan: ReadPlan,
    *,
    rows_returned: int,
    fetch_time_ms: float,
    artifact_bytes: int | None = None,
) -> None:
    """Count a finished ``scan@v1`` in the scan, per-table and per-tenant metrics.

    *artifact_bytes* marks a scan answered from a stored artifact: every row group
    counts as a cache hit. Otherwise the plan's tasks say how each row group was read.
    """
    from strata.metrics import ScanMetrics
    from strata.tenant import get_tenant_id
    from strata.tenant_registry import get_tenant_registry

    tasks = plan.tasks
    if artifact_bytes is not None:
        hits, misses, from_cache, from_storage = len(tasks), 0, artifact_bytes, 0
    else:
        hits = sum(1 for t in tasks if t.cached)
        misses = len(tasks) - hits
        from_cache = sum(t.bytes_read for t in tasks if t.cached)
        from_storage = sum(t.bytes_read for t in tasks if not t.cached)

    state.metrics.log_scan_complete(
        ScanMetrics(
            scan_id=plan.scan_id,
            snapshot_id=plan.snapshot_id,
            table_id=str(plan.table_identity),
            planning_time_ms=plan.planning_time_ms,
            fetch_time_ms=fetch_time_ms,
            total_time_ms=plan.planning_time_ms + fetch_time_ms,
            cache_hits=hits,
            cache_misses=misses,
            bytes_from_cache=from_cache,
            bytes_from_storage=from_storage,
            total_row_groups=plan.total_row_groups,
            pruned_row_groups=plan.pruned_row_groups,
            rows_returned=rows_returned,
        )
    )
    get_tenant_registry().record_scan(
        get_tenant_id(), hits, misses, from_cache, from_storage, rows_returned
    )


class ScanBuildManager:
    """The active scan table plus opportunistic first-row-group prefetch."""

    def __init__(self, prefetch_concurrency: int = 4) -> None:
        # Registered when a client will stream
        self.scans: dict[str, ReadPlan] = {}

        # Separate from streaming concurrency, so clients spamming POST /scan without
        # consuming can't exhaust resources.
        self._prefetch_semaphore = asyncio.Semaphore(prefetch_concurrency)
        self._prefetch_futures: dict[str, asyncio.Task[None]] = {}
        self._started = 0
        self._used = 0  # Consumed by streaming
        self._wasted = 0  # Discarded (scan deleted/abandoned)
        self._skipped = 0  # Server busy
        self._in_flight = 0

    # --- scan table ---

    def register_scan(self, plan: ReadPlan) -> None:
        self.scans[plan.scan_id] = plan

    def get_scan(self, scan_id: str) -> ReadPlan | None:
        return self.scans.get(scan_id)

    def pop_scan(self, scan_id: str) -> ReadPlan | None:
        return self.scans.pop(scan_id, None)

    def __contains__(self, scan_id: str) -> bool:
        return scan_id in self.scans

    # --- prefetch ---

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
                with get_pool_tracker().track("fetch"):
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

        Waits briefly on an in-flight prefetch and counts it ``used``; otherwise
        discards it as ``wasted`` and the build fetches directly.
        """
        if plan.prefetched_first is not None:
            chunk = plan.prefetched_first
            plan.prefetched_first = None
            self._used += 1
            return chunk

        prefetch_task = self._prefetch_futures.get(scan_id)
        if prefetch_task is not None:
            # A timeout just means not ready yet; fall through to consume-or-discard.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(prefetch_task), timeout=0.05)
            if plan.prefetched_first is not None:
                chunk = plan.prefetched_first
                plan.prefetched_first = None
                self._used += 1
                return chunk
            self.discard_prefetch(scan_id, count_wasted=True)
        return None

    def prefetch_metrics(self) -> dict[str, int]:
        """Prefetch counters for ``/metrics`` and Prometheus."""
        return {
            "started": self._started,
            "used": self._used,
            "wasted": self._wasted,
            "skipped": self._skipped,
            "in_flight": self._in_flight,
        }

    # --- cleanup callback ---

    def expire_scan(self, scan_id: str) -> None:
        """Discard any prefetch and drop the scan (the registry's ``on_expire`` hook)."""
        self.discard_prefetch(scan_id, count_wasted=True)
        self.pop_scan(scan_id)

    # --- background build ---

    def mark_stream_artifact_failed(self, state: ServerState, stream_state: StreamState) -> None:
        """Best-effort transition a stream-backed artifact to failed state."""
        from strata.artifact_store import get_artifact_store

        store = get_artifact_store(state.config.artifact_dir)
        if store is None:
            return

        try:
            store.fail_artifact(stream_state.artifact_id, stream_state.artifact_version)
        except Exception as e:
            # Best-effort on an already-failed build; raising would mask the original error.
            logger.debug(
                "mark_stream_artifact_failed_error",
                artifact_id=stream_state.artifact_id,
                error=str(e),
            )

    async def build_identity_artifact(self, state: ServerState, stream_state: StreamState) -> None:
        """Build a scan@v1 artifact in the background.

        Runs as a shielded task decoupled from readers: it writes row group by row
        group straight to the blob (bounded memory) and finalizes ready/failed on its
        own, so a slow or dropped reader cannot poison the cache entry.
        """
        from strata.artifact_store import get_artifact_store
        from strata.transforms.build_qos import record_build_output_bytes

        plan = stream_state.plan

        stream_state.started = True
        stream_state.started_at = time.time()
        build_started = time.perf_counter()

        try:
            store = get_artifact_store(state.config.artifact_dir)
            if store is None:
                return  # service mode has no artifact store

            if not plan.tasks:
                if plan.schema is not None:
                    sink = pa.BufferOutputStream()
                    writer = ipc.new_stream(sink, plan.schema)
                    writer.close()
                    empty_stream = sink.getvalue().to_pybytes()
                else:
                    empty_stream = b""

                await asyncio.to_thread(
                    store.write_blob,
                    stream_state.artifact_id,
                    stream_state.artifact_version,
                    empty_stream,
                )
                await self.finalize_written_blob(state, stream_state, 0, len(empty_stream))
                stream_state.bytes_streamed = len(empty_stream)
                stream_state.completed = True
                return

            loop = asyncio.get_running_loop()
            scan_id = plan.scan_id
            row_count = 0
            byte_size = 0
            start_time = time.perf_counter()

            # Write-through for bounded memory. The merger emits one IPC stream across row
            # groups so standard readers see every row. Writes go to a local staging file; the
            # commit, an upload on a remote store, runs off the loop.
            merger = IncrementalIpcMerger() if len(plan.tasks) > 1 else None
            writer = store.open_blob_writer(stream_state.artifact_id, stream_state.artifact_version)
            blob = writer.__enter__()
            try:
                for index, task in enumerate(plan.tasks):
                    if state._draining:
                        raise RuntimeError("Server is shutting down")

                    # A runaway scan fails the artifact rather than holding resources.
                    if time.perf_counter() - start_time > state.config.scan_timeout_seconds:
                        state.metrics.record_stream_abort_timeout()
                        raise RuntimeError(
                            f"Scan timed out after {state.config.scan_timeout_seconds}s"
                        )

                    # Stream mode may have prefetched the first row group; artifact mode never does.
                    chunk: bytes | None = None
                    if index == 0:
                        chunk = await self.consume_prefetched_first(plan, scan_id)

                    if chunk is None:
                        with get_pool_tracker().track("fetch"):
                            chunk = await loop.run_in_executor(
                                state._fetch_executor,
                                state.fetcher.fetch_as_stream_bytes,
                                task,
                            )
                    out = merger.feed(chunk) if merger is not None else chunk
                    if out:
                        blob.write(out)
                        byte_size += len(out)
                    # With equality deletes a task's num_rows is only an upper bound.
                    row_count += validate_ipc_stream(chunk)

                if merger is not None:
                    tail = merger.finish()
                    if tail:
                        blob.write(tail)
                        byte_size += len(tail)
            except BaseException as exc:
                writer.__exit__(type(exc), exc, exc.__traceback__)
                raise
            await asyncio.to_thread(writer.__exit__, None, None, None)

            await self.finalize_written_blob(state, stream_state, row_count, byte_size)
            stream_state.bytes_streamed = byte_size
            stream_state.completed = True
        except asyncio.CancelledError:
            stream_state.error_message = "Build cancelled"
            self.mark_stream_artifact_failed(state, stream_state)
            raise
        except Exception as e:
            stream_state.error_message = str(e)
            logger.error(
                "identity_artifact_build_error",
                artifact_id=stream_state.artifact_id,
                error=str(e),
            )
            self.mark_stream_artifact_failed(state, stream_state)
        finally:
            # The store read can raise; the slot and the cleanup must not depend on it.
            try:
                artifact = None
                store = get_artifact_store(state.config.artifact_dir)
                if store is not None:
                    artifact = store.get_artifact(
                        stream_state.artifact_id, stream_state.artifact_version
                    )

                if (
                    stream_state.completed
                    and stream_state.error_message is None
                    and artifact is not None
                    and artifact.state == "ready"
                ):
                    await record_build_output_bytes(
                        stream_state.qos_tenant_id,
                        stream_state.bytes_streamed,
                    )
                    record_scan_complete(
                        state,
                        plan,
                        rows_returned=artifact.row_count or 0,
                        fetch_time_ms=(time.perf_counter() - build_started) * 1000,
                    )
            finally:
                if stream_state.build_slot is not None:
                    await stream_state.build_slot.release()
                stream_state.completed_at = time.time()
                # Pass the scan_id: schedule_cleanup() replaces the pending cleanup, and only a
                # scan-aware one runs ``expire_scan``; without it the ReadPlan leaks for good.
                state.streams.schedule_cleanup(stream_state.stream_id, stream_state.plan.scan_id)

    async def finalize_written_blob(
        self,
        state: ServerState,
        stream_state: StreamState,
        row_count: int,
        byte_size: int,
    ) -> None:
        """Finalize a scan artifact whose blob is already written, then mark it ``ready``.

        Re-reads the blob one record batch at a time in a worker thread for the
        integrity gate. The sole finalizer for scan builds.
        """
        from strata.artifact_store import get_artifact_store

        store = get_artifact_store(state.config.artifact_dir)
        if store is None:
            return  # service mode has no artifact store

        try:
            # Integrity gate: the blob must be one readable IPC stream with the plan's row total.
            if byte_size == 0:
                readable_rows, schema_json = 0, ""
            else:

                def _read_and_validate() -> tuple[int, str]:
                    with store.open_blob_reader(
                        stream_state.artifact_id, stream_state.artifact_version
                    ) as blob:
                        return validate_ipc_stream_reader(blob)

                readable_rows, schema_json = await asyncio.to_thread(_read_and_validate)
            if readable_rows != row_count:
                raise ValueError(
                    f"Artifact blob integrity check failed: stream yields "
                    f"{readable_rows} rows, build reported {row_count}"
                )

            # Marks ready and sets metadata and the name pointer atomically. It reads the
            # blob back to hash it, so it runs off the loop.
            finalized_artifact = await asyncio.to_thread(
                store.finalize_and_set_name,
                artifact_id=stream_state.artifact_id,
                version=stream_state.artifact_version,
                schema_json=schema_json,
                row_count=row_count,
                byte_size=byte_size,
                name=stream_state.name,
                tenant=stream_state.tenant,
            )
            if finalized_artifact is not None:
                stream_state.artifact_id = finalized_artifact.id
                stream_state.artifact_version = finalized_artifact.version

            logger.info(
                "stream_artifact_finalized",
                artifact_id=stream_state.artifact_id,
                version=stream_state.artifact_version,
                byte_size=byte_size,
                row_count=row_count,
            )
        except Exception as e:
            logger.error(
                "stream_artifact_finalize_error",
                artifact_id=stream_state.artifact_id,
                error=str(e),
            )
            try:
                store.fail_artifact(stream_state.artifact_id, stream_state.artifact_version)
            except Exception as fail_err:
                # Best-effort: the finalize failure is already logged; don't raise.
                logger.debug(
                    "stream_artifact_fail_cleanup_error",
                    artifact_id=stream_state.artifact_id,
                    error=str(fail_err),
                )
