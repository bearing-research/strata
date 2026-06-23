"""Scan-build manager — active scan table, prefetch, and the background build.

Owns the ``scan_id -> ReadPlan`` table, the best-effort first-row-group prefetch,
and the write-through scan@v1 background build that ``server.py`` held as
``ServerState.scans`` / ``_prefetch_*`` and the free ``_start_prefetch`` /
``_discard_prefetch`` / ``_build_identity_artifact`` / ``_finalize_written_blob``
/ ``_mark_stream_artifact_failed`` helpers. Held on ``ServerState`` as
``state.scan_builds``.

Phases 2a + 2b of the stream-runtime extraction (#302). Methods that need shared
infra (config, fetcher, fetch executor, draining flag, artifact store, the stream
registry) take ``state`` per-call, so the manager owns the scan/prefetch *state*
without a back-reference to ``ServerState``. The build stays the SHIELDED,
decoupled task (#165) — the manager creates it, the handler shields it. See
``docs/internal/design-stream-runtime-extraction.md``.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.ipc as ipc

from strata.fast_io import IncrementalIpcMerger, validate_ipc_stream_reader
from strata.logging import get_logger

if TYPE_CHECKING:
    from strata.server import ServerState
    from strata.streaming.registry import StreamState
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

    # --- background build ---------------------------------------------------

    def mark_stream_artifact_failed(self, state: ServerState, stream_state: StreamState) -> None:
        """Best-effort transition a stream-backed artifact to failed state."""
        from strata.artifact_store import get_artifact_store

        store = get_artifact_store(state.config.artifact_dir)
        if store is None:
            return

        try:
            store.fail_artifact(stream_state.artifact_id, stream_state.artifact_version)
        except Exception:
            pass

    async def build_identity_artifact(self, state: ServerState, stream_state: StreamState) -> None:
        """Build a scan@v1 artifact in the background for artifact-mode requests.

        The build is the SHIELDED, decoupled task (#165): it scans row group by
        row group straight to the blob (write-through, bounded memory) and
        finalizes ready/failed on its own merits, so a slow or dropped reader can
        never poison the cache entry.
        """
        from strata.artifact_store import get_artifact_store
        from strata.transforms.build_qos import record_build_output_bytes

        plan = stream_state.plan

        stream_state.started = True
        stream_state.started_at = time.time()

        try:
            store = get_artifact_store(state.config.artifact_dir)
            if store is None:
                return  # No artifact store in service mode

            if not plan.tasks:
                if plan.schema is not None:
                    sink = pa.BufferOutputStream()
                    writer = ipc.new_stream(sink, plan.schema)
                    writer.close()
                    empty_stream = sink.getvalue().to_pybytes()
                else:
                    empty_stream = b""

                store.write_blob(
                    stream_state.artifact_id, stream_state.artifact_version, empty_stream
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

            # Write each merged row-group chunk straight to the blob (write-through,
            # bounded memory — no full-result buffer). The IncrementalIpcMerger emits
            # one valid IPC stream across the row groups so standard readers see
            # every row. The blob commits atomically when the writer context exits.
            merger = IncrementalIpcMerger() if len(plan.tasks) > 1 else None
            with store.open_blob_writer(
                stream_state.artifact_id, stream_state.artifact_version
            ) as blob:
                for index, task in enumerate(plan.tasks):
                    if state._draining:
                        raise RuntimeError("Server is shutting down")

                    # Bound the build's wall-clock the same way the old streaming
                    # generator did — a runaway scan marks the artifact failed
                    # rather than holding resources indefinitely.
                    if time.perf_counter() - start_time > state.config.scan_timeout_seconds:
                        state.metrics.record_stream_abort_timeout()
                        raise RuntimeError(
                            f"Scan timed out after {state.config.scan_timeout_seconds}s"
                        )

                    # Consume the eagerly-prefetched first row group when stream
                    # mode warmed one (no-op in artifact mode, which never prefetches).
                    chunk: bytes | None = None
                    if index == 0:
                        chunk = await self.consume_prefetched_first(plan, scan_id)

                    if chunk is None:
                        chunk = await loop.run_in_executor(
                            state._fetch_executor,
                            state.fetcher.fetch_as_stream_bytes,
                            task,
                        )
                    out = merger.feed(chunk) if merger is not None else chunk
                    if out:
                        blob.write(out)
                        byte_size += len(out)
                    row_count += task.num_rows

                if merger is not None:
                    tail = merger.finish()
                    if tail:
                        blob.write(tail)
                        byte_size += len(tail)

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
            artifact_ready = False
            store = get_artifact_store(state.config.artifact_dir)
            if store is not None:
                artifact = store.get_artifact(
                    stream_state.artifact_id, stream_state.artifact_version
                )
                artifact_ready = artifact is not None and artifact.state == "ready"

            if stream_state.completed and stream_state.error_message is None and artifact_ready:
                await record_build_output_bytes(
                    stream_state.qos_tenant_id,
                    stream_state.bytes_streamed,
                )
            if stream_state.build_slot is not None:
                await stream_state.build_slot.release()
            stream_state.completed_at = time.time()
            state.streams.schedule_cleanup(stream_state.stream_id)

    async def finalize_written_blob(
        self,
        state: ServerState,
        stream_state: StreamState,
        row_count: int,
        byte_size: int,
    ) -> None:
        """Finalize a scan artifact whose blob was already **written through**.

        The blob is on disk by the time this runs; re-read it in bounded memory
        (one record batch at a time, in a worker thread) for the integrity gate
        (#124), then flip state to ``ready``. The sole finalizer for scan builds —
        both the stream and artifact materialize paths route through the
        write-through build. See docs/internal/design-streaming-decouple.md.
        """
        from strata.artifact_store import get_artifact_store

        store = get_artifact_store(state.config.artifact_dir)
        if store is None:
            return  # No artifact store in service mode

        try:
            # Integrity gate (#124): bounded re-read confirms the persisted blob is
            # exactly one readable IPC stream whose row total matches the plan.
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

            # The blob is already persisted; finalize_and_set_name flips state to
            # ready and records metadata + the requested name pointer atomically.
            finalized_artifact = store.finalize_and_set_name(
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
            except Exception:
                pass
