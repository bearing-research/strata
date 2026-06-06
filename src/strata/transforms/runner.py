"""Background build runner for server-mode transforms.

The build runner is responsible for:
1. Polling for pending builds
2. Acquiring inputs (artifacts or Iceberg scans)
3. Streaming inputs to external executors
4. Persisting outputs as new artifact versions
5. Updating build state on success/failure

Concurrency controls:
- Global semaphore limits total concurrent builds
- Per-tenant semaphore limits builds per tenant
- Build-level timeout from transform definition

Executor HTTP Protocol v1 (Push Model):
    See strata.types for protocol definitions:
    - ExecutorRequestMetadata: JSON metadata schema
    - ExecutorInputDescriptor: Input descriptor schema
    - ExecutorTransformSpec: Transform specification schema

    Request: POST {executor_url}/v1/execute
    Headers:
        Content-Type: multipart/form-data
        X-Strata-Executor-Protocol: v1

    Parts:
    1. metadata (application/json): ExecutorRequestMetadata
    2. input0, input1, ... (application/vnd.apache.arrow.stream)

    Response:
    - Status: 200
    - Content-Type: application/vnd.apache.arrow.stream
    - X-Strata-Logs: base64-encoded executor logs (optional)
    - Body: Output Arrow IPC stream bytes

    Error Response (4xx/5xx):
    - Content-Type: application/json
    - Body: ExecutorResponse with success=false
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

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

    Attributes:
        poll_interval_ms: How often to poll for pending builds (default 500ms)
        max_concurrent_builds: Global limit on concurrent builds
        max_builds_per_tenant: Per-tenant limit on concurrent builds
        default_timeout_seconds: Default build timeout if not in registry
        default_max_output_bytes: Default max output size if not in registry
        lease_duration_seconds: How long a build lease is valid (default 60s)
        heartbeat_interval_seconds: How often to renew leases (default 15s)
        runner_id: Unique identifier for this runner instance
    """

    poll_interval_ms: int = 500
    max_concurrent_builds: int = 10
    max_builds_per_tenant: int = 3
    default_timeout_seconds: float = 300.0
    default_max_output_bytes: int = 1024 * 1024 * 1024  # 1 GB
    lease_duration_seconds: float = 60.0
    heartbeat_interval_seconds: float = 15.0
    runner_id: str | None = None  # Auto-generated if not provided


@dataclass
class BuildRunner:
    """Background runner for server-mode builds.

    The runner polls for pending builds and executes them asynchronously
    using external executors. It manages concurrency limits and handles
    errors gracefully.

    Reliability features:
    - Lease-based claiming: Each build is claimed with a lease that must be
      renewed periodically. If the runner crashes, another runner can reclaim
      the build after the lease expires.
    - Heartbeat: Leases are renewed periodically during execution.
    - Orphan recovery: Builds with expired leases are reclaimed automatically.

    Usage:
        runner = BuildRunner(config, artifact_store, build_store, registry)
        await runner.start()
        # ... server runs ...
        await runner.stop()
    """

    config: RunnerConfig
    artifact_store: ArtifactStore
    build_store: BuildStore
    transform_registry: TransformRegistry
    artifact_dir: Path
    runtime_config: StrataConfig | None = None
    scan_planner: ReadPlanner | None = None
    scan_fetcher: CachedFetcher | None = None

    # Internal state
    _running: bool = field(default=False, init=False)
    _task: asyncio.Task | None = field(default=None, init=False)
    _heartbeat_task: asyncio.Task | None = field(default=None, init=False)
    _global_sem: asyncio.Semaphore = field(init=False)
    _tenant_sems: dict[str, asyncio.Semaphore] = field(default_factory=dict, init=False)
    _running_builds: set[str] = field(default_factory=set, init=False)
    _build_tasks: dict[str, asyncio.Task] = field(default_factory=dict, init=False)
    _runner_id: str = field(init=False)

    def __post_init__(self):
        self._global_sem = asyncio.Semaphore(self.config.max_concurrent_builds)
        # Generate unique runner ID if not provided
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

        # Cancel the heartbeat loop
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Cancel the main loop
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Cancel all running build tasks
        for build_id, task in list(self._build_tasks.items()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            # Mark as failed due to shutdown
            self.build_store.fail_build(
                build_id,
                error_message="Build cancelled due to server shutdown",
                error_code="SERVER_SHUTDOWN",
            )

        self._build_tasks.clear()
        self._running_builds.clear()
        logger.info("Build runner stopped", extra={"runner_id": self._runner_id})

    async def _run_loop(self) -> None:
        """Main polling loop for pending builds."""
        poll_interval = self.config.poll_interval_ms / 1000.0

        while self._running:
            try:
                # Fetch pending builds
                pending = self.build_store.list_pending_builds(limit=50)

                for build in pending:
                    if not _is_runner_managed_build(build):
                        continue
                    # Skip if already running
                    if build.build_id in self._running_builds:
                        continue

                    # Submit build for execution
                    self._submit_build(build)

                # Recover orphaned builds (expired leases from crashed runners)
                expired = self.build_store.list_expired_leases(limit=10)
                for build in expired:
                    if build.build_id not in self._running_builds:
                        # Try to reclaim the build
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

        This keeps builds alive while they're executing. If this loop
        stops (e.g., runner crashes), the leases will expire and the
        builds can be reclaimed by other runners.
        """
        heartbeat_interval = self.config.heartbeat_interval_seconds

        while self._running:
            try:
                # Renew leases for all running builds
                for build_id in list(self._running_builds):
                    if not self.build_store.renew_lease(
                        build_id,
                        self._runner_id,
                        self.config.lease_duration_seconds,
                    ):
                        # Lease renewal failed - we may have lost the lease
                        logger.warning(
                            f"Failed to renew lease for build {build_id}",
                            extra={"runner_id": self._runner_id},
                        )
                        # Don't cancel the task - let it finish if possible
                        # The build will fail at completion if lease is lost

            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")

            await asyncio.sleep(heartbeat_interval)

    def _submit_build(self, build: BuildState, already_claimed: bool = False) -> None:
        """Submit a build for async execution.

        Args:
            build: Build to execute
            already_claimed: If True, skip claiming (already claimed via reclaim)
        """
        if build.build_id in self._running_builds:
            return

        self._running_builds.add(build.build_id)
        task = asyncio.create_task(self._execute_build_with_semaphores(build, already_claimed))
        self._build_tasks[build.build_id] = task

        # Clean up when done
        def cleanup(t):
            self._running_builds.discard(build.build_id)
            self._build_tasks.pop(build.build_id, None)

        task.add_done_callback(cleanup)

    async def _execute_build_with_semaphores(
        self, build: BuildState, already_claimed: bool = False
    ) -> None:
        """Execute a build with concurrency controls.

        Args:
            build: Build to execute
            already_claimed: If True, skip claiming (already claimed via reclaim)
        """
        tenant_id = build.tenant_id or "__default__"

        # Get or create per-tenant semaphore
        if tenant_id not in self._tenant_sems:
            self._tenant_sems[tenant_id] = asyncio.Semaphore(self.config.max_builds_per_tenant)

        tenant_sem = self._tenant_sems[tenant_id]

        # Acquire both semaphores
        async with self._global_sem:
            async with tenant_sem:
                await self._execute_build(build, already_claimed)

    async def _execute_build(self, build: BuildState, already_claimed: bool = False) -> None:
        """Execute a single build.

        This is the main build execution logic:
        1. Claim build with lease (or skip if already claimed)
        2. Get transform definition from registry
        3. Load artifact metadata to get inputs and transform
        4. Acquire inputs (artifacts or Iceberg scans)
        5. Stream inputs to executor
        6. Persist output as new artifact version
        7. Update build state on success/failure

        Args:
            build: Build to execute
            already_claimed: If True, skip claiming (already claimed via reclaim)
        """
        import time as time_mod

        from strata.logging import BuildContext
        from strata.transforms.build_metrics import get_build_metrics

        build_id = build.build_id
        temp_files: list[Path] = []
        start_time = time_mod.time()
        build_started_recorded = False

        # Wrap entire build execution in BuildContext for structured logging
        with BuildContext(
            build_id=build_id,
            tenant_id=build.tenant_id,
            transform_ref=build.executor_ref,
        ):
            try:
                # Claim build with lease (skip if already building from retry/reclaim)
                if build.state == "pending" and not already_claimed:
                    if not self.build_store.claim_build(
                        build_id,
                        self._runner_id,
                        self.config.lease_duration_seconds,
                    ):
                        logger.warning(f"Build {build_id} already claimed or completed")
                        return

                # Get fresh build state
                fresh_build = self.build_store.get_build(build_id)
                if fresh_build is None or fresh_build.state not in ("pending", "building"):
                    return
                build = fresh_build

                # Verify we still own the lease (in case of race)
                if build.lease_owner and build.lease_owner != self._runner_id:
                    logger.warning(
                        f"Build {build_id} claimed by another runner",
                        extra={"owner": build.lease_owner, "runner_id": self._runner_id},
                    )
                    return

                # Record build started metric
                metrics = get_build_metrics()
                if metrics is not None:
                    # Calculate queue wait time (time from created_at to now)
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

                # Get transform definition
                transform_defn = self.transform_registry.get(build.executor_ref)
                if transform_defn is None:
                    raise ValueError(f"Transform not found in registry: {build.executor_ref}")

                # Get artifact metadata to retrieve inputs and transform
                artifact = self.artifact_store.get_artifact(build.artifact_id, build.version)
                if artifact is None:
                    raise ValueError(f"Artifact not found: {build.artifact_id}@v={build.version}")

                # Parse inputs and transform from artifact metadata
                if artifact.transform_spec is None:
                    raise ValueError("Artifact has no transform spec")

                transform_data = json.loads(artifact.transform_spec)
                input_uris = transform_data.get("inputs", [])

                # Prepare input files (materialize each input to temp file)
                input_files: list[tuple[str, Path]] = []
                for i, input_uri in enumerate(input_uris):
                    input_name = f"input{i}"
                    input_path = await self._acquire_input(
                        input_uri,
                        temp_files,
                        tenant_id=build.tenant_id,
                    )
                    input_files.append((input_name, input_path))

                # Get executor URL and timeout
                executor_url = transform_defn.executor_url
                timeout = transform_defn.timeout_seconds or self.config.default_timeout_seconds
                max_output = transform_defn.max_output_bytes or self.config.default_max_output_bytes

                # Prepare executor request metadata (protocol v1)
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

                # Execute transform via external executor
                output_path, executor_logs = await self._call_executor(
                    executor_url=executor_url,
                    metadata=metadata,
                    input_files=input_files,
                    timeout=timeout,
                    max_output_bytes=max_output,
                    temp_files=temp_files,
                )

                # Read output schema and row count
                output_bytes = output_path.stat().st_size
                schema_json, row_count = self._read_arrow_metadata(output_path)

                # Move output to final artifact location
                final_path = self.artifact_store._blob_path(build.artifact_id, build.version)
                output_path.rename(final_path)

                # Finalize artifact
                finalized_artifact = self.artifact_store.finalize_artifact(
                    artifact_id=build.artifact_id,
                    version=build.version,
                    schema_json=schema_json,
                    row_count=row_count,
                    byte_size=output_bytes,
                )
                if finalized_artifact is None:
                    raise ValueError(
                        f"Failed to finalize build artifact {build.artifact_id}@v={build.version}"
                    )
                if (
                    finalized_artifact.id != build.artifact_id
                    or finalized_artifact.version != build.version
                ):
                    self.build_store.update_build_output(
                        build.build_id,
                        finalized_artifact.id,
                        finalized_artifact.version,
                    )

                # Set the requested name pointer now that the artifact is
                # ready — the materialize endpoint can't (the build is async).
                if build.name:
                    self.artifact_store.set_name(
                        build.name,
                        finalized_artifact.id,
                        finalized_artifact.version,
                        tenant=build.tenant_id,
                    )

                # Mark build as complete (include executor logs for debugging)
                self.build_store.complete_build(
                    build_id=build_id,
                    output_byte_count=output_bytes,
                    logs=executor_logs,
                )
                from strata.transforms.build_qos import get_build_qos

                build_qos = get_build_qos()
                if build_qos is not None:
                    await build_qos.record_bytes(build.tenant_id or "__default__", output_bytes)

                # Record success metric
                metrics = get_build_metrics()
                if metrics is not None and build_started_recorded:
                    duration_ms = (time_mod.time() - start_time) * 1000.0
                    # Calculate input bytes from input files
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
                # Don't mark as failed if cancelled - will be retried
                raise

            except Exception as e:
                error_msg = str(e)
                error_code = type(e).__name__

                # Truncate long error messages
                if len(error_msg) > 500:
                    error_msg = error_msg[:500] + "..."

                logger.error(
                    f"Build {build_id} failed: {error_msg}",
                    extra={"traceback": traceback.format_exc()},
                )

                # Record failure metric
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

                self.build_store.fail_build(
                    build_id=build_id,
                    error_message=error_msg,
                    error_code=error_code,
                )

                # Also mark artifact as failed
                self.artifact_store.fail_artifact(build.artifact_id, build.version)

            finally:
                # Clean up temp files
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
        """Acquire an input and write it to a temp file.

        Supports:
        - strata://artifact/{id}@v={version} - read artifact blob
        - strata://name/{name} - resolve name and read artifact blob
        - file:// or s3:// - Iceberg table scan (to be implemented)

        Returns:
            Path to temp file containing Arrow IPC stream
        """
        # Handle artifact URIs
        if input_uri.startswith("strata://artifact/"):
            # Parse artifact URI: strata://artifact/{id}@v={version}
            import re

            match = re.match(r"strata://artifact/([^@]+)@v=(\d+)", input_uri)
            if not match:
                raise ValueError(f"Invalid artifact URI: {input_uri}")

            artifact_id = match.group(1)
            version = int(match.group(2))

            # Read artifact blob
            blob = self.artifact_store.read_blob(artifact_id, version)
            if blob is None:
                raise ValueError(f"Artifact blob not found: {input_uri}")

            # Write to temp file
            _fd, _tmp_path = tempfile.mkstemp(suffix=".arrow", dir=self.artifact_dir)
            os.close(_fd)  # Windows: handle must be closed before rename
            temp_file = Path(_tmp_path)
            temp_file.write_bytes(blob)
            temp_files.append(temp_file)
            return temp_file

        # Handle name URIs
        if input_uri.startswith("strata://name/"):
            name = input_uri.split("/", 3)[-1]
            artifact = self.artifact_store.resolve_name(name, tenant=tenant_id)
            if artifact is None:
                raise ValueError(f"Name not found: {name}")

            # Read artifact blob
            blob = self.artifact_store.read_blob(artifact.id, artifact.version)
            if blob is None:
                raise ValueError(f"Artifact blob not found for name: {name}")

            # Write to temp file
            _fd, _tmp_path = tempfile.mkstemp(suffix=".arrow", dir=self.artifact_dir)
            os.close(_fd)  # Windows: handle must be closed before rename
            temp_file = Path(_tmp_path)
            temp_file.write_bytes(blob)
            temp_files.append(temp_file)
            return temp_file

        # Handle Iceberg table URIs (file:// or s3://)
        if input_uri.startswith("file://") or input_uri.startswith("s3://"):
            # For Iceberg scans, we need to run the internal scan pipeline
            # and write the output to a temp file
            temp_file = await self._scan_to_file(input_uri, temp_files)
            return temp_file

        raise ValueError(f"Unsupported input URI: {input_uri}")

    async def _scan_to_file(
        self,
        table_uri: str,
        temp_files: list[Path],
    ) -> Path:
        """Run an Iceberg scan and write output to a temp file.

        This uses the internal scan pipeline to read the table and
        write the Arrow IPC stream to a file.
        """
        # Import here to avoid circular imports

        # Create a temp file for output
        _fd, _tmp_path = tempfile.mkstemp(suffix=".arrow", dir=self.artifact_dir)
        os.close(_fd)  # Windows: handle must be closed before rename
        temp_file = Path(_tmp_path)
        temp_files.append(temp_file)

        # Run in thread pool to avoid blocking event loop
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
        """Call executor - either embedded or via HTTP.

        For embedded execution (executor_url == "embedded://local" or empty),
        runs the transform directly in-process. Otherwise, makes an HTTP call
        to an external executor service.

        Args:
            executor_url: Base URL of the executor, or "embedded://local" for embedded
            metadata: Build metadata JSON
            input_files: List of (name, path) tuples for inputs
            timeout: Request timeout in seconds
            max_output_bytes: Maximum output size in bytes
            temp_files: List to append temp files for cleanup

        Returns:
            Tuple of (Path to temp file containing output Arrow IPC stream,
                      executor logs from X-Strata-Logs header or None)

        Raises:
            ValueError: If output exceeds max_output_bytes
            httpx.HTTPStatusError: If executor returns non-200 status
            asyncio.TimeoutError: If request times out
        """
        # Check for embedded executor (special URL or empty)
        if executor_url == "embedded://local" or not executor_url:
            return await self._call_embedded_executor(
                metadata, input_files, timeout, max_output_bytes, temp_files
            )

        # External executor via HTTP
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
        """Execute transform locally using embedded executor.

        This runs the transform directly in-process, without HTTP overhead.
        Used for local deployment where no external executor service is needed.

        Args:
            metadata: Build metadata JSON with transform ref and params
            input_files: List of (name, path) tuples for inputs
            timeout: Execution timeout in seconds
            max_output_bytes: Maximum output size in bytes
            temp_files: List to append temp files for cleanup

        Returns:
            Tuple of (Path to output file, None for logs)
        """
        import io

        import pyarrow.ipc as ipc

        from strata.transforms.base import _run_transform

        # Parse inputs from files
        inputs = []
        for name, path in sorted(input_files, key=lambda x: x[0]):
            with ipc.open_stream(str(path)) as reader:
                inputs.append(reader.read_all())

        # Get transform info from metadata
        transform = metadata.get("transform", {})
        transform_ref = transform.get("ref", "")
        params = transform.get("params", {})

        # Execute in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: _run_transform(transform_ref, inputs, params),
            ),
            timeout=timeout,
        )

        # Serialize result to Arrow IPC
        output_buffer = io.BytesIO()
        with ipc.new_stream(output_buffer, result.schema) as writer:
            writer.write_table(result)
        output_bytes = output_buffer.getvalue()

        # Check output size
        if len(output_bytes) > max_output_bytes:
            raise ValueError(
                f"Output exceeds maximum size: {len(output_bytes)} > {max_output_bytes}"
            )

        # Write to temp file
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
        """Call external executor via HTTP.

        Uses multipart/form-data to stream inputs to the executor.
        The executor returns an Arrow IPC stream which is written to
        a temp file with size enforcement.

        Args:
            executor_url: Base URL of the executor
            metadata: Build metadata JSON
            input_files: List of (name, path) tuples for inputs
            timeout: Request timeout in seconds
            max_output_bytes: Maximum output size in bytes
            temp_files: List to append temp files for cleanup

        Returns:
            Tuple of (Path to temp file containing output Arrow IPC stream,
                      executor logs from X-Strata-Logs header or None)

        Raises:
            ValueError: If output exceeds max_output_bytes
            httpx.HTTPStatusError: If executor returns non-200 status
            asyncio.TimeoutError: If request times out
        """
        # Import protocol constants
        from strata.types import EXECUTOR_PROTOCOL_HEADER, EXECUTOR_PROTOCOL_VERSION

        # Prepare multipart files
        files: dict[str, tuple[str, str | bytes, str]] = {
            "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
        }

        for name, path in input_files:
            files[name] = (
                f"{name}.arrow",
                path.read_bytes(),
                "application/vnd.apache.arrow.stream",
            )

        # Create output temp file
        _fd, _tmp_path = tempfile.mkstemp(suffix=".arrow", dir=self.artifact_dir)
        os.close(_fd)  # Windows: handle must be closed before rename
        output_path = Path(_tmp_path)
        temp_files.append(output_path)

        # Protocol version header for executor compatibility
        headers = {
            EXECUTOR_PROTOCOL_HEADER: EXECUTOR_PROTOCOL_VERSION,
        }

        # Make request with timeout
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await asyncio.wait_for(
                client.post(
                    f"{executor_url}/v1/execute",
                    files=files,
                    headers=headers,
                ),
                timeout=timeout,
            )

            response.raise_for_status()

            # Capture executor logs from response header (if present)
            # Executors can include logs in EXECUTOR_LOGS_HEADER (base64 encoded)
            from strata.types import EXECUTOR_LOGS_HEADER

            executor_logs = None
            logs_header = response.headers.get(EXECUTOR_LOGS_HEADER)
            if logs_header:
                import base64

                try:
                    executor_logs = base64.b64decode(logs_header).decode("utf-8")
                except Exception:
                    # If we can't decode, store the raw header value
                    executor_logs = logs_header

            # Stream response to file with size limit
            bytes_written = 0
            with open(output_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    bytes_written += len(chunk)
                    if bytes_written > max_output_bytes:
                        raise ValueError(
                            f"Output exceeds maximum size: {bytes_written} > {max_output_bytes}"
                        )
                    f.write(chunk)

        return output_path, executor_logs

    def _read_arrow_metadata(self, path: Path) -> tuple[str, int]:
        """Read Arrow IPC file metadata.

        Returns:
            Tuple of (schema_json, row_count)
        """
        import pyarrow.ipc as ipc

        with ipc.open_stream(str(path)) as reader:
            schema = reader.schema
            row_count = 0
            for batch in reader:
                row_count += batch.num_rows

        # Convert schema to JSON
        schema_json = schema.to_string()

        return schema_json, row_count


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

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
