"""Background build runner for server-mode transforms.

Polls for pending builds, acquires inputs (artifacts or Iceberg scans), runs the executor,
and persists the output as a new artifact version. Concurrency is bounded globally and per
tenant; each build has its transform's timeout.

Executor protocol v1 (push; schemas in ``strata.types``):

    POST {executor_url}/v1/execute, multipart/form-data, X-Strata-Executor-Protocol: v1
    parts: metadata (application/json, ExecutorRequestMetadata),
           input0, input1, ... (application/vnd.apache.arrow.stream)
    200: Arrow IPC stream body; optional X-Strata-Logs (base64 executor logs)
    4xx/5xx: application/json ExecutorResponse with success=false
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from strata.artifact_store import BuildLeaseLost

if TYPE_CHECKING:
    from strata.artifact_store import ArtifactStore
    from strata.cache import CachedFetcher
    from strata.config import StrataConfig
    from strata.planner import ReadPlanner
    from strata.transforms.build_store import BuildState, BuildStore
    from strata.transforms.registry import TransformRegistry

logger = logging.getLogger(__name__)


def _is_runner_managed_build(build: BuildState) -> bool:
    """Return whether the generic build runner should execute this build."""
    params = build.params or {}
    if not isinstance(params, dict):
        return True
    return params.get("_dispatch_mode") != "external"


@dataclass
class RunnerConfig:
    """Configuration for the build runner.

    Timeout and max-output defaults apply when the registry entry sets none. A lease must be
    renewed every ``heartbeat_interval_seconds`` or it lapses after ``lease_duration_seconds``.
    """

    poll_interval_ms: int = 500
    max_concurrent_builds: int = 10
    max_builds_per_tenant: int = 3
    default_timeout_seconds: float = 300.0
    default_max_output_bytes: int = 1024 * 1024 * 1024  # 1 GB
    lease_duration_seconds: float = 60.0
    heartbeat_interval_seconds: float = 15.0
    runner_id: str | None = None


# How often the runner's loop sweeps settled build attempts.
_SWEEP_INTERVAL_SECONDS = 60.0
_SWEEP_BATCH = 100


def sweep_settled_attempts(build_store: BuildStore, artifact_store: ArtifactStore) -> None:
    """Delete what settled build attempts wrote, except the published one.

    Blob deletes are network I/O, so the runner loop calls this off the event loop.
    """
    while True:
        batch = build_store.settled_attempts(limit=_SWEEP_BATCH)
        for build_id, artifact_id, version, attempt, promoted in batch:
            if not promoted:
                artifact_store.delete_attempt_blob(artifact_id, version, attempt)
            build_store.forget_attempt(build_id, attempt)
        if len(batch) < _SWEEP_BATCH:
            return


@dataclass
class BuildRunner:
    """Background runner that claims pending builds and executes them.

    Builds are claimed under a lease renewed by a heartbeat; if a runner dies, its leases
    expire and another runner reclaims the builds. Call ``start()`` and ``stop()`` around the
    server's lifetime.
    """

    config: RunnerConfig
    artifact_store: ArtifactStore
    build_store: BuildStore
    transform_registry: TransformRegistry
    artifact_dir: Path
    runtime_config: StrataConfig | None = None
    scan_planner: ReadPlanner | None = None
    scan_fetcher: CachedFetcher | None = None

    _running: bool = field(default=False, init=False)
    _task: asyncio.Task | None = field(default=None, init=False)
    _heartbeat_task: asyncio.Task | None = field(default=None, init=False)
    _global_sem: asyncio.Semaphore = field(init=False)
    _tenant_sems: dict[str, asyncio.Semaphore] = field(default_factory=dict, init=False)
    _running_builds: set[str] = field(default_factory=set, init=False)
    _build_tasks: dict[str, asyncio.Task] = field(default_factory=dict, init=False)
    _runner_id: str = field(init=False)
    _next_sweep_at: float = field(default=0.0, init=False)

    def __post_init__(self):
        self._global_sem = asyncio.Semaphore(self.config.max_concurrent_builds)
        self._runner_id = self.config.runner_id or f"runner-{uuid.uuid4().hex[:8]}"

    async def start(self) -> None:
        """Start the build runner background loop."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(
            "Build runner started",
            extra={
                "runner_id": self._runner_id,
                "max_concurrent": self.config.max_concurrent_builds,
                "max_per_tenant": self.config.max_builds_per_tenant,
                "poll_interval_ms": self.config.poll_interval_ms,
                "lease_duration_seconds": self.config.lease_duration_seconds,
                "heartbeat_interval_seconds": self.config.heartbeat_interval_seconds,
            },
        )

    async def stop(self) -> None:
        """Stop the build runner and cancel pending tasks."""
        if not self._running:
            return

        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        for build_id, task in list(self._build_tasks.items()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            # Only a build this runner still holds: one taken over by another runner
            # is still running there.
            self.build_store.fail_build(
                build_id,
                error_message="Build cancelled due to server shutdown",
                error_code="SERVER_SHUTDOWN",
                lease_owner=self._runner_id,
            )

        self._build_tasks.clear()
        self._running_builds.clear()
        logger.info("Build runner stopped", extra={"runner_id": self._runner_id})

    async def _run_loop(self) -> None:
        """Main polling loop for pending builds."""
        poll_interval = self.config.poll_interval_ms / 1000.0

        while self._running:
            try:
                pending = self.build_store.list_pending_builds(limit=50)

                for build in pending:
                    if not _is_runner_managed_build(build):
                        continue
                    if build.build_id in self._running_builds:
                        continue

                    self._submit_build(build)

                # Nothing waits on the space attempts free, so once a minute is enough.
                if time.monotonic() >= self._next_sweep_at:
                    self._next_sweep_at = time.monotonic() + _SWEEP_INTERVAL_SECONDS
                    await asyncio.to_thread(
                        sweep_settled_attempts, self.build_store, self.artifact_store
                    )

                # Orphans: expired leases from crashed runners.
                expired = self.build_store.list_expired_leases(limit=10)
                for build in expired:
                    if build.build_id not in self._running_builds:
                        if self.build_store.reclaim_expired_build(
                            build.build_id,
                            self._runner_id,
                            self.config.lease_duration_seconds,
                        ):
                            logger.info(
                                f"Reclaimed orphaned build {build.build_id}",
                                extra={
                                    "runner_id": self._runner_id,
                                    "previous_owner": build.lease_owner,
                                },
                            )
                            self._submit_build(build, already_claimed=True)

            except Exception as e:
                logger.error(f"Error in build runner loop: {e}")

            await asyncio.sleep(poll_interval)

    async def _heartbeat_loop(self) -> None:
        """Periodically renew leases on running builds.

        If this loop stops, the leases expire and other runners can reclaim the builds.
        """
        heartbeat_interval = self.config.heartbeat_interval_seconds

        while self._running:
            try:
                for build_id in list(self._running_builds):
                    if not self.build_store.renew_lease(
                        build_id,
                        self._runner_id,
                        self.config.lease_duration_seconds,
                    ):
                        # Let the task finish; completion is fenced on the lease anyway.
                        logger.warning(
                            f"Failed to renew lease for build {build_id}",
                            extra={"runner_id": self._runner_id},
                        )

            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

            await asyncio.sleep(heartbeat_interval)

    def _submit_build(self, build: BuildState, already_claimed: bool = False) -> None:
        """Submit a build for async execution.

        ``already_claimed`` skips claiming, for builds already reclaimed.
        """
        if build.build_id in self._running_builds:
            return

        self._running_builds.add(build.build_id)
        task = asyncio.create_task(self._execute_build_with_semaphores(build, already_claimed))
        self._build_tasks[build.build_id] = task

        def cleanup(t):
            self._running_builds.discard(build.build_id)
            self._build_tasks.pop(build.build_id, None)

        task.add_done_callback(cleanup)

    async def _execute_build_with_semaphores(
        self, build: BuildState, already_claimed: bool = False
    ) -> None:
        """Execute a build under the global and per-tenant concurrency limits."""
        tenant_id = build.tenant_id or "__default__"

        if tenant_id not in self._tenant_sems:
            self._tenant_sems[tenant_id] = asyncio.Semaphore(self.config.max_builds_per_tenant)

        tenant_sem = self._tenant_sems[tenant_id]

        async with self._global_sem:
            async with tenant_sem:
                await self._execute_build(build, already_claimed)

    async def _execute_build(self, build: BuildState, already_claimed: bool = False) -> None:
        """Execute one build: claim, acquire inputs, run the executor, persist, update state.

        ``already_claimed`` skips claiming (reclaimed builds).
        """
        import time as time_mod

        from strata.logging import BuildContext
        from strata.transforms.build_metrics import get_build_metrics

        build_id = build.build_id
        temp_files: list[Path] = []
        start_time = time_mod.time()
        build_started_recorded = False

        with BuildContext(
            build_id=build_id,
            tenant_id=build.tenant_id,
            transform_ref=build.executor_ref,
        ):
            try:
                # Skip claiming when already building (retry / reclaim).
                if build.state == "pending" and not already_claimed:
                    if not self.build_store.claim_build(
                        build_id,
                        self._runner_id,
                        self.config.lease_duration_seconds,
                    ):
                        logger.warning(f"Build {build_id} already claimed or completed")
                        return

                fresh_build = self.build_store.get_build(build_id)
                if fresh_build is None or fresh_build.state not in ("pending", "building"):
                    return
                build = fresh_build

                if build.lease_owner and build.lease_owner != self._runner_id:
                    logger.warning(
                        f"Build {build_id} claimed by another runner",
                        extra={"owner": build.lease_owner, "runner_id": self._runner_id},
                    )
                    return

                metrics = get_build_metrics()
                if metrics is not None:
                    queue_wait_ms = None
                    if build.created_at:
                        queue_wait_ms = (start_time - build.created_at) * 1000.0
                    metrics.record_started(
                        build_id=build_id,
                        tenant_id=build.tenant_id,
                        transform_ref=build.executor_ref,
                        queue_wait_ms=queue_wait_ms,
                    )
                    build_started_recorded = True

                transform_defn = self.transform_registry.get(build.executor_ref)
                if transform_defn is None:
                    raise ValueError(f"Transform not found in registry: {build.executor_ref}")

                artifact = self.artifact_store.get_artifact(build.artifact_id, build.version)
                if artifact is None:
                    raise ValueError(f"Artifact not found: {build.artifact_id}@v={build.version}")

                if artifact.transform_spec is None:
                    raise ValueError("Artifact has no transform spec")

                transform_data = json.loads(artifact.transform_spec)
                input_uris = transform_data.get("inputs", [])

                input_files: list[tuple[str, Path]] = []
                for i, input_uri in enumerate(input_uris):
                    input_name = f"input{i}"
                    input_path = await self._acquire_input(
                        input_uri,
                        temp_files,
                        tenant_id=build.tenant_id,
                    )
                    input_files.append((input_name, input_path))

                executor_url = transform_defn.executor_url
                timeout = transform_defn.timeout_seconds or self.config.default_timeout_seconds
                max_output = transform_defn.max_output_bytes or self.config.default_max_output_bytes

                max_input = transform_defn.max_input_bytes or 0
                if max_input > 0:
                    total_input_bytes = sum(p.stat().st_size for _, p in input_files)
                    if total_input_bytes > max_input:
                        raise ValueError(
                            f"Total input size {total_input_bytes} exceeds "
                            f"max_input_bytes {max_input} for {build.executor_ref}"
                        )

                from strata.types import EXECUTOR_PROTOCOL_VERSION

                metadata = {
                    "protocol_version": EXECUTOR_PROTOCOL_VERSION,
                    "build_id": build_id,
                    "tenant": build.tenant_id,
                    "principal": build.principal_id,
                    "provenance_hash": artifact.provenance_hash,
                    "transform": {
                        "ref": build.executor_ref,
                        "code_hash": hashlib.sha256(artifact.transform_spec.encode()).hexdigest()[
                            :16
                        ],
                        "params": transform_data.get("params", {}),
                    },
                    "inputs": [
                        {"name": name, "format": "arrow_ipc_stream"} for name, _ in input_files
                    ],
                }

                output_path, executor_logs = await self._call_executor(
                    executor_url=executor_url,
                    metadata=metadata,
                    input_files=input_files,
                    timeout=timeout,
                    max_output_bytes=max_output,
                    temp_files=temp_files,
                )

                output_bytes = output_path.stat().st_size
                schema_json, row_count = self._read_arrow_metadata(output_path)

                # Through the blob store, not a rename into ``_blob_path``: with a remote
                # backend that local directory is never read. Under this attempt's own id,
                # not the version's: a runner whose lease was taken over keeps executing,
                # and writing the shared key could replace bytes already published as ready.
                attempt = uuid.uuid4().hex
                # Recorded first, so a crash before finalize leaves bytes the sweep can find.
                self.build_store.record_attempt(
                    build_id,
                    build.artifact_id,
                    build.version,
                    attempt,
                    writable_until=time.time(),
                )
                self.artifact_store.publish_blob_from_path(
                    build.artifact_id, build.version, output_path, attempt=attempt
                )

                # Publishing the attempt and completing the build are one transaction,
                # fenced on the lease: the commit point that decides which runner won,
                # mirroring ``claim_build`` deciding which one started.
                def _complete(conn: Any, artifact_id: str, version: int) -> bool:
                    return self.build_store.complete_within(
                        conn,
                        build_id,
                        artifact_id=artifact_id,
                        version=version,
                        lease_owner=self._runner_id,
                        output_byte_count=output_bytes,
                        logs=executor_logs,
                    )

                try:
                    finalized_artifact = self.artifact_store.finalize_artifact(
                        artifact_id=build.artifact_id,
                        version=build.version,
                        schema_json=schema_json,
                        row_count=row_count,
                        byte_size=output_bytes,
                        blob_attempt=attempt,
                        fence=_complete,
                    )
                except BuildLeaseLost:
                    self.artifact_store.delete_attempt_blob(
                        build.artifact_id, build.version, attempt
                    )
                    logger.warning(
                        f"Build {build_id} finished after its lease moved on; "
                        "discarding the result and leaving the name pointer alone",
                        extra={"runner_id": self._runner_id},
                    )
                    return
                if finalized_artifact is None:
                    raise ValueError(
                        f"Failed to finalize build artifact {build.artifact_id}@v={build.version}"
                    )
                # Deduplicated to an artifact that already existed: the build points at it, and
                # finalize dropped this attempt's bytes, leaving the superseded version reading
                # the canonical's by the URI the materialize response handed out.

                # Set here because the materialize endpoint can't: the build is async.
                if build.name:
                    self.artifact_store.set_name(
                        build.name,
                        finalized_artifact.id,
                        finalized_artifact.version,
                        tenant=build.tenant_id,
                    )
                from strata.transforms.build_qos import get_build_qos

                build_qos = get_build_qos()
                if build_qos is not None:
                    await build_qos.record_bytes(build.tenant_id or "__default__", output_bytes)

                metrics = get_build_metrics()
                if metrics is not None and build_started_recorded:
                    duration_ms = (time_mod.time() - start_time) * 1000.0
                    input_bytes = sum(f.stat().st_size if f.exists() else 0 for _, f in input_files)
                    metrics.record_succeeded(
                        build_id=build_id,
                        tenant_id=build.tenant_id,
                        transform_ref=build.executor_ref,
                        duration_ms=duration_ms,
                        bytes_in=input_bytes,
                        bytes_out=output_bytes,
                    )

                logger.info(
                    f"Build {build_id} completed successfully",
                    extra={
                        "artifact_id": build.artifact_id,
                        "version": build.version,
                        "output_bytes": output_bytes,
                        "row_count": row_count,
                    },
                )

            except asyncio.CancelledError:
                # Not failed: a cancelled build is retried.
                raise

            except Exception as e:
                error_msg = str(e)
                error_code = type(e).__name__

                if len(error_msg) > 500:
                    error_msg = error_msg[:500] + "..."

                logger.error(
                    f"Build {build_id} failed: {error_msg}",
                    extra={"traceback": traceback.format_exc()},
                )

                metrics = get_build_metrics()
                if metrics is not None and build_started_recorded:
                    duration_ms = (time_mod.time() - start_time) * 1000.0
                    metrics.record_failed(
                        build_id=build_id,
                        tenant_id=build.tenant_id,
                        transform_ref=build.executor_ref,
                        duration_ms=duration_ms,
                        error_code=error_code,
                    )

                # Only if this runner still holds the lease; a runner whose lease was taken
                # over keeps executing, and failing here would fail its successor's build.
                # The artifact row has no lease of its own, so its failure is gated on this.
                failed = self.build_store.fail_build(
                    build_id=build_id,
                    error_message=error_msg,
                    error_code=error_code,
                    lease_owner=self._runner_id,
                )
                if failed:
                    self.artifact_store.fail_artifact(build.artifact_id, build.version)
                else:
                    logger.info(
                        f"Build {build_id} failed here after its lease moved on; "
                        "leaving it to the runner that holds it",
                        extra={"runner_id": self._runner_id},
                    )

            finally:
                for temp_file in temp_files:
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except Exception:
                        pass

    async def _acquire_input(
        self,
        input_uri: str,
        temp_files: list[Path],
        tenant_id: str | None = None,
    ) -> Path:
        """Write an input to a temp file as an Arrow IPC stream and return its path.

        Accepts ``strata://artifact/{id}@v={version}``, ``strata://name/{name}``, and ``file://``
        or ``s3://`` table URIs (scanned). Anything else raises ``ValueError``.
        """
        if input_uri.startswith("strata://artifact/"):
            import re

            match = re.match(r"strata://artifact/([^@]+)@v=(\d+)", input_uri)
            if not match:
                raise ValueError(f"Invalid artifact URI: {input_uri}")

            artifact_id = match.group(1)
            version = int(match.group(2))

            blob = self.artifact_store.read_blob(artifact_id, version)
            if blob is None:
                raise ValueError(f"Artifact blob not found: {input_uri}")

            _fd, _tmp_path = tempfile.mkstemp(suffix=".arrow", dir=self.artifact_dir)
            os.close(_fd)  # Windows: handle must be closed before rename
            temp_file = Path(_tmp_path)
            temp_file.write_bytes(blob)
            temp_files.append(temp_file)
            return temp_file

        if input_uri.startswith("strata://name/"):
            name = input_uri.split("/", 3)[-1]
            artifact = self.artifact_store.resolve_name(name, tenant=tenant_id)
            if artifact is None:
                raise ValueError(f"Name not found: {name}")

            blob = self.artifact_store.read_blob(artifact.id, artifact.version)
            if blob is None:
                raise ValueError(f"Artifact blob not found for name: {name}")

            _fd, _tmp_path = tempfile.mkstemp(suffix=".arrow", dir=self.artifact_dir)
            os.close(_fd)  # Windows: handle must be closed before rename
            temp_file = Path(_tmp_path)
            temp_file.write_bytes(blob)
            temp_files.append(temp_file)
            return temp_file

        if input_uri.startswith("strata://"):
            raise ValueError(f"Unsupported input URI: {input_uri}")

        # Any other URI is a table in a form the planner reads, as admission resolved it.
        return await self._scan_to_file(input_uri, temp_files)

    async def _scan_to_file(
        self,
        table_uri: str,
        temp_files: list[Path],
    ) -> Path:
        """Run an Iceberg scan through the internal scan pipeline into a temp Arrow IPC file."""

        _fd, _tmp_path = tempfile.mkstemp(suffix=".arrow", dir=self.artifact_dir)
        os.close(_fd)  # Windows: handle must be closed before rename
        temp_file = Path(_tmp_path)
        temp_files.append(temp_file)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            self._scan_to_file_sync,
            table_uri,
            temp_file,
        )

        return temp_file

    def _scan_to_file_sync(self, table_uri: str, output_path: Path) -> None:
        """Synchronous helper to scan a table and write to file."""
        from strata.cache import CachedFetcher
        from strata.config import StrataConfig
        from strata.planner import ReadPlanner

        planner = self.scan_planner
        fetcher = self.scan_fetcher

        if planner is None or fetcher is None:
            runtime_config = self.runtime_config or StrataConfig.load()
            planner = planner or ReadPlanner(runtime_config)
            fetcher = fetcher or CachedFetcher(runtime_config)

        plan = planner.plan(
            table_uri=table_uri,
            snapshot_id=None,  # Current snapshot
            columns=None,  # All columns
            filters=None,
        )

        import pyarrow as pa

        with pa.OSFile(str(output_path), "wb") as sink:
            writer = None
            try:
                for task in plan.tasks:
                    batch = fetcher.fetch(task)
                    if writer is None:
                        writer = pa.ipc.new_stream(sink, batch.schema)
                    writer.write_batch(batch)

                if writer is None:
                    schema = plan.schema or pa.schema([])
                    writer = pa.ipc.new_stream(sink, schema)
            finally:
                if writer is not None:
                    writer.close()

    async def _call_executor(
        self,
        executor_url: str,
        metadata: dict,
        input_files: list[tuple[str, Path]],
        timeout: float,
        max_output_bytes: int,
        temp_files: list[Path],
    ) -> tuple[Path, str | None]:
        """Run the executor in-process (``embedded://local`` or empty URL) or over HTTP.

        Returns ``(output path, executor logs or None)``. Temp files are appended to
        ``temp_files`` for the caller to clean up.

        Raises:
            ValueError: output exceeds ``max_output_bytes``.
            httpx.HTTPStatusError: the executor returned an error status.
            asyncio.TimeoutError: the call timed out.
        """
        if executor_url == "embedded://local" or not executor_url:
            return await self._call_embedded_executor(
                metadata, input_files, timeout, max_output_bytes, temp_files
            )

        return await self._call_http_executor(
            executor_url, metadata, input_files, timeout, max_output_bytes, temp_files
        )

    async def _call_embedded_executor(
        self,
        metadata: dict,
        input_files: list[tuple[str, Path]],
        timeout: float,
        max_output_bytes: int,
        temp_files: list[Path],
    ) -> tuple[Path, str | None]:
        """Run the transform in-process; returns ``(output path, None)``."""
        import io

        import pyarrow.ipc as ipc

        from strata.transforms.base import _run_transform

        inputs = []
        for name, path in sorted(input_files, key=lambda x: x[0]):
            with ipc.open_stream(str(path)) as reader:
                inputs.append(reader.read_all())

        transform = metadata.get("transform", {})
        transform_ref = transform.get("ref", "")
        params = transform.get("params", {})

        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: _run_transform(transform_ref, inputs, params),
            ),
            timeout=timeout,
        )

        output_buffer = io.BytesIO()
        with ipc.new_stream(output_buffer, result.schema) as writer:
            writer.write_table(result)
        output_bytes = output_buffer.getvalue()

        if len(output_bytes) > max_output_bytes:
            raise ValueError(
                f"Output exceeds maximum size: {len(output_bytes)} > {max_output_bytes}"
            )

        _fd, _tmp_path = tempfile.mkstemp(suffix=".arrow", dir=self.artifact_dir)
        os.close(_fd)  # Windows: handle must be closed before rename
        output_path = Path(_tmp_path)
        temp_files.append(output_path)
        output_path.write_bytes(output_bytes)

        return output_path, None

    async def _call_http_executor(
        self,
        executor_url: str,
        metadata: dict,
        input_files: list[tuple[str, Path]],
        timeout: float,
        max_output_bytes: int,
        temp_files: list[Path],
    ) -> tuple[Path, str | None]:
        """Call an external executor over HTTP (protocol v1).

        Inputs stream from disk and the response streams to a temp file, enforcing
        ``max_output_bytes`` as it arrives. Returns ``(output path, decoded X-Strata-Logs or
        None)``; raises as ``_call_executor`` documents.
        """
        from strata.types import EXECUTOR_PROTOCOL_HEADER, EXECUTOR_PROTOCOL_VERSION

        # Inputs go to httpx as open file objects, not ``read_bytes()``, so N inputs
        # don't cost ~2x their total size in RAM.
        files: dict[str, tuple[str, Any, str]] = {
            "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
        }

        _fd, _tmp_path = tempfile.mkstemp(suffix=".arrow", dir=self.artifact_dir)
        os.close(_fd)  # Windows: handle must be closed before rename
        output_path = Path(_tmp_path)
        temp_files.append(output_path)

        headers = {
            EXECUTOR_PROTOCOL_HEADER: EXECUTOR_PROTOCOL_VERSION,
        }

        # ``client.stream``, not ``post``: post() buffers the whole body before the
        # size check can run, so an oversized executor response could OOM the server.
        with contextlib.ExitStack() as input_handles:
            for name, path in input_files:
                files[name] = (
                    f"{name}.arrow",
                    input_handles.enter_context(open(path, "rb")),
                    "application/vnd.apache.arrow.stream",
                )

            async with httpx.AsyncClient(timeout=timeout) as client:
                request = client.build_request(
                    "POST",
                    f"{executor_url}/v1/execute",
                    files=files,
                    headers=headers,
                )
                response = await asyncio.wait_for(
                    client.send(request, stream=True),
                    timeout=timeout,
                )

                try:
                    # A streamed error response has no body yet; read it so the
                    # raised HTTPStatusError carries the executor's message.
                    if response.status_code >= 400:
                        await response.aread()
                        raise httpx.HTTPStatusError(
                            f"Executor returned HTTP {response.status_code}: "
                            f"{_executor_error_text(response)}",
                            request=response.request,
                            response=response,
                        )

                    # Executors may send base64-encoded logs in EXECUTOR_LOGS_HEADER.
                    from strata.types import EXECUTOR_LOGS_HEADER

                    executor_logs = None
                    logs_header = response.headers.get(EXECUTOR_LOGS_HEADER)
                    if logs_header:
                        import base64

                        try:
                            executor_logs = base64.b64decode(logs_header).decode("utf-8")
                        except Exception:
                            executor_logs = logs_header

                    # Size limit enforced during transfer, aborting an oversized download.
                    bytes_written = 0
                    with open(output_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            bytes_written += len(chunk)
                            if bytes_written > max_output_bytes:
                                raise ValueError(
                                    f"Output exceeds maximum size: "
                                    f"{bytes_written} > {max_output_bytes}"
                                )
                            f.write(chunk)
                finally:
                    await response.aclose()

        return output_path, executor_logs

    def _read_arrow_metadata(self, path: Path) -> tuple[str, int]:
        """Read an Arrow IPC file's ``(schema_json, row_count)``."""
        import pyarrow.ipc as ipc

        with ipc.open_stream(str(path)) as reader:
            schema = reader.schema
            row_count = 0
            for batch in reader:
                row_count += batch.num_rows

        schema_json = schema.to_string()

        return schema_json, row_count


def _executor_error_text(response: httpx.Response, limit: int = 500) -> str:
    """The executor's reason from an error body: ``error_message``, ``detail``, or the text."""
    try:
        body = response.json()
    except ValueError:
        body = None
    message = (body.get("error_message") or body.get("detail")) if isinstance(body, dict) else None
    return str(message or response.text or "(empty body)")[:limit]


_runner: BuildRunner | None = None


def get_build_runner() -> BuildRunner | None:
    """Get the build runner singleton."""
    return _runner


def set_build_runner(runner: BuildRunner | None) -> None:
    """Set the build runner singleton."""
    global _runner
    _runner = runner


def reset_build_runner() -> None:
    """Reset the build runner singleton (for testing)."""
    global _runner
    _runner = None
