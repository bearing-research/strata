"""Tests for background build runner (server-mode transforms).

These tests verify:
1. Successful build execution with mocked executor
2. Build failure when max_output_bytes is exceeded
3. Build failure when executor returns non-200 status
4. Build failure when executor times out
5. Concurrency controls (global and per-tenant semaphores)
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from strata.artifact_store import (
    TransformSpec,
    get_artifact_store,
    reset_artifact_store,
)
from strata.transforms.build_qos import (
    BuildQoS,
    BuildQoSConfig,
    reset_build_qos,
    set_build_qos,
)
from strata.transforms.build_store import get_build_store, reset_build_store
from strata.transforms.registry import (
    TransformDefinition,
    TransformRegistry,
    reset_transform_registry,
    set_transform_registry,
)
from strata.transforms.runner import (
    BuildRunner,
    RunnerConfig,
    reset_build_runner,
    set_build_runner,
)
from strata.types import CacheKey, ReadPlan, TableIdentity, Task


@pytest.fixture
def artifact_dir(tmp_path):
    """Create a temporary artifact directory."""
    artifact_path = tmp_path / "artifacts"
    artifact_path.mkdir(parents=True)
    (artifact_path / "blobs").mkdir()
    return artifact_path


@pytest.fixture
def artifact_store(artifact_dir):
    """Create a temporary artifact store."""
    reset_artifact_store()
    store = get_artifact_store(artifact_dir)
    yield store
    reset_artifact_store()


@pytest.fixture
def build_store(artifact_dir):
    """Create a temporary build store."""
    reset_build_store()
    db_path = artifact_dir / "artifacts.sqlite"
    store = get_build_store(db_path)
    yield store
    reset_build_store()


@pytest.fixture
def transform_registry():
    """Create a transform registry with test transforms."""
    reset_transform_registry()
    registry = TransformRegistry(
        enabled=True,
        definitions=[
            TransformDefinition(
                ref="test_sql@*",
                executor_url="http://test-executor:8080",
                timeout_seconds=30.0,
                max_output_bytes=10 * 1024 * 1024,  # 10 MB
            ),
            TransformDefinition(
                ref="slow_transform@v1",
                executor_url="http://slow-executor:8080",
                timeout_seconds=0.1,  # Very short timeout for testing
                max_output_bytes=10 * 1024 * 1024,
            ),
            TransformDefinition(
                ref="small_output@v1",
                executor_url="http://test-executor:8080",
                timeout_seconds=30.0,
                max_output_bytes=100,  # Very small limit for testing
            ),
        ],
    )
    set_transform_registry(registry)
    yield registry
    reset_transform_registry()


@pytest.fixture
def runner_config():
    """Create runner configuration for testing."""
    return RunnerConfig(
        poll_interval_ms=50,  # Fast polling for tests
        max_concurrent_builds=5,
        max_builds_per_tenant=2,
        default_timeout_seconds=30.0,
        default_max_output_bytes=10 * 1024 * 1024,
    )


@pytest.fixture
def build_runner(runner_config, artifact_store, build_store, transform_registry, artifact_dir):
    """Create a build runner for testing."""
    reset_build_runner()
    runner = BuildRunner(
        config=runner_config,
        artifact_store=artifact_store,
        build_store=build_store,
        transform_registry=transform_registry,
        artifact_dir=artifact_dir,
    )
    set_build_runner(runner)
    yield runner
    reset_build_runner()


def create_arrow_ipc_bytes(data: dict) -> bytes:
    """Create Arrow IPC stream bytes from a dictionary.

    Args:
        data: Dictionary mapping column names to lists of values

    Returns:
        Arrow IPC stream bytes
    """
    table = pa.table(data)
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        for batch in table.to_batches():
            writer.write_batch(batch)
    return sink.getvalue().to_pybytes()


def create_test_artifact(artifact_store, build_store, executor_ref="test_sql@v1", tenant_id=None):
    """Create a test artifact in building state with a pending build.

    Returns:
        Tuple of (artifact_id, version, build_id)
    """
    artifact_id = str(uuid.uuid4())

    # Create transform spec with inputs
    transform_spec = TransformSpec(
        executor=f"service://{executor_ref}",
        params={"query": "SELECT * FROM input0"},
        inputs=[],  # No inputs for simplicity
    )

    # Create artifact in building state
    provenance_hash = f"test-hash-{artifact_id}"
    version = artifact_store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=provenance_hash,
        transform_spec=transform_spec,
        input_versions={},
    )

    # Create pending build
    build_id = str(uuid.uuid4())
    build_store.create_build(
        build_id=build_id,
        artifact_id=artifact_id,
        version=version,
        executor_ref=executor_ref,
        executor_url="http://test-executor:8080",
        tenant_id=tenant_id,
        principal_id="test-user",
    )

    return artifact_id, version, build_id


class TestBuildRunnerBasics:
    """Basic build runner tests."""

    @pytest.mark.asyncio
    async def test_start_stop(self, build_runner):
        """Test runner start and stop lifecycle."""
        assert not build_runner._running

        await build_runner.start()
        assert build_runner._running
        assert build_runner._task is not None

        await build_runner.stop()
        assert not build_runner._running
        assert build_runner._task is None

    @pytest.mark.asyncio
    async def test_double_start(self, build_runner):
        """Starting twice should be idempotent."""
        await build_runner.start()
        task1 = build_runner._task

        await build_runner.start()
        task2 = build_runner._task

        # Should be the same task
        assert task1 is task2

        await build_runner.stop()

    @pytest.mark.asyncio
    async def test_runner_ignores_external_dispatch_builds(
        self,
        artifact_store,
        build_store,
        build_runner,
    ):
        """Externally driven builds should stay pending for their own controller."""
        artifact_id = str(uuid.uuid4())
        version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash=f"prov-{artifact_id}",
            transform_spec=TransformSpec(
                executor="notebook_cell@v1",
                params={"_dispatch_mode": "external"},
                inputs=[],
            ),
        )
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id=artifact_id,
            version=version,
            executor_ref="notebook_cell@v1",
            executor_url="http://notebook-executor:8080",
            params={"_dispatch_mode": "external"},
        )

        await build_runner.start()
        try:
            await asyncio.sleep(build_runner.config.poll_interval_ms / 1000.0 * 3)
            build = build_store.get_build(build_id)
            assert build is not None
            assert build.state == "pending"
        finally:
            await build_runner.stop()

    @pytest.mark.asyncio
    async def test_double_stop(self, build_runner):
        """Stopping twice should be idempotent."""
        await build_runner.start()
        await build_runner.stop()
        # Should not raise
        await build_runner.stop()


def _send_from_post(mock_post):
    """Adapt a ``client.post`` mock to the streaming ``client.send`` call.

    The runner streams the executor response (``build_request`` +
    ``send(..., stream=True)``) so the output size limit is enforced during
    transfer rather than after the whole body is already in memory. These
    mocks predate that and script ``post``; this bridges them.
    """

    async def _send(request, *args, **kwargs):
        response = await mock_post()
        # Always assign: these are MagicMocks, so hasattr is unconditionally
        # True and a guard would leave a non-awaitable in place.
        response.aclose = AsyncMock()
        response.aread = AsyncMock(return_value=b"")
        return response

    return _send


class TestBuildExecution:
    """Tests for build execution with mocked executor."""

    @pytest.mark.asyncio
    async def test_successful_build(self, build_runner, artifact_store, build_store, artifact_dir):
        """Test successful build execution with mocked executor."""
        # Create a pending build
        artifact_id, version, build_id = create_test_artifact(artifact_store, build_store)

        # Create mock response with Arrow data
        output_data = {"id": [1, 2, 3], "value": ["a", "b", "c"]}
        output_bytes = create_arrow_ipc_bytes(output_data)

        # Mock the httpx response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        # Mock headers to return None for logs header (avoid MagicMock leaking to SQLite)
        mock_response.headers = {}

        async def mock_aiter_bytes(chunk_size=65536):
            yield output_bytes

        mock_response.aiter_bytes = mock_aiter_bytes

        async def mock_post(*args, **kwargs):
            return mock_response

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.build_request = MagicMock(return_value=MagicMock())
            mock_client.send = _send_from_post(mock_post)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            # Execute the build
            build = build_store.get_build(build_id)
            await build_runner._execute_build(build)

        # Verify build completed successfully
        build = build_store.get_build(build_id)
        assert build.state == "ready"
        assert build.completed_at is not None
        assert build.output_byte_count == len(output_bytes)
        assert build.error_message is None

        # Verify artifact was finalized
        artifact = artifact_store.get_artifact(artifact_id, version)
        assert artifact.state == "ready"
        assert artifact.row_count == 3

    @pytest.mark.asyncio
    async def test_successful_build_records_quota_bytes(
        self, build_runner, artifact_store, build_store
    ):
        """Successful server-mode builds should update per-tenant byte quotas."""
        qos = BuildQoS(BuildQoSConfig(bytes_per_day_limit=10 * 1024 * 1024))
        set_build_qos(qos)

        try:
            artifact_id, version, build_id = create_test_artifact(
                artifact_store,
                build_store,
                tenant_id="tenant-quota",
            )

            output_bytes = create_arrow_ipc_bytes({"id": [1, 2], "value": ["a", "b"]})
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_response.headers = {}

            async def mock_aiter_bytes(chunk_size=65536):
                yield output_bytes

            mock_response.aiter_bytes = mock_aiter_bytes

            async def mock_post(*args, **kwargs):
                return mock_response

            with patch("httpx.AsyncClient") as MockClient:
                mock_client = AsyncMock()
                mock_client.post = mock_post
                mock_client.build_request = MagicMock(return_value=MagicMock())
                mock_client.send = _send_from_post(mock_post)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                MockClient.return_value = mock_client

                build = build_store.get_build(build_id)
                await build_runner._execute_build(build)

            tenant_metrics = qos.get_tenant_metrics("tenant-quota")
            assert tenant_metrics is not None
            assert tenant_metrics["quota"]["bytes_today"] == len(output_bytes)

            build = build_store.get_build(build_id)
            assert build is not None
            assert build.state == "ready"
            artifact = artifact_store.get_artifact(artifact_id, version)
            assert artifact is not None
            assert artifact.state == "ready"
        finally:
            reset_build_qos()

    @pytest.mark.asyncio
    async def test_duplicate_finalize_repoints_build_to_existing_artifact(
        self, build_runner, artifact_store, build_store
    ):
        """Duplicate provenance should complete against the canonical artifact."""
        provenance_hash = f"duplicate-hash-{uuid.uuid4()}"

        existing_artifact_id = str(uuid.uuid4())
        existing_version = artifact_store.create_artifact(existing_artifact_id, provenance_hash)
        artifact_store.finalize_artifact(existing_artifact_id, existing_version, "{}", 1, 1)

        duplicate_artifact_id = str(uuid.uuid4())
        transform_spec = TransformSpec(
            executor="service://test_sql@v1",
            params={"query": "SELECT * FROM input0"},
            inputs=[],
        )
        duplicate_version = artifact_store.create_artifact(
            artifact_id=duplicate_artifact_id,
            provenance_hash=provenance_hash,
            transform_spec=transform_spec,
            input_versions={},
        )
        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id=duplicate_artifact_id,
            version=duplicate_version,
            executor_ref="test_sql@v1",
            executor_url="http://test-executor:8080",
            principal_id="test-user",
        )

        output_bytes = create_arrow_ipc_bytes({"id": [1], "value": ["a"]})
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_response.headers = {}

        async def mock_aiter_bytes(chunk_size=65536):
            yield output_bytes

        mock_response.aiter_bytes = mock_aiter_bytes

        async def mock_post(*args, **kwargs):
            return mock_response

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.build_request = MagicMock(return_value=MagicMock())
            mock_client.send = _send_from_post(mock_post)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            build = build_store.get_build(build_id)
            await build_runner._execute_build(build)

        completed_build = build_store.get_build(build_id)
        assert completed_build is not None
        assert completed_build.state == "ready"
        assert completed_build.artifact_id == existing_artifact_id
        assert completed_build.version == existing_version

        duplicate_artifact = artifact_store.get_artifact(duplicate_artifact_id, duplicate_version)
        assert duplicate_artifact is not None
        assert duplicate_artifact.state == "failed"

    @pytest.mark.asyncio
    async def test_build_max_output_bytes_exceeded(self, build_runner, artifact_store, build_store):
        """Test build failure when output exceeds max_output_bytes."""
        # Create a pending build with small_output transform (100 byte limit)
        artifact_id, version, build_id = create_test_artifact(
            artifact_store, build_store, executor_ref="small_output@v1"
        )

        # Create mock response with large output (>100 bytes)
        output_data = {"id": list(range(1000)), "value": ["x" * 100] * 1000}
        output_bytes = create_arrow_ipc_bytes(output_data)

        # Mock the httpx response that streams large output
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        # Mock headers to return None for logs header (avoid MagicMock leaking to SQLite)
        mock_response.headers = {}

        async def mock_aiter_bytes(chunk_size=65536):
            # Stream in chunks
            for i in range(0, len(output_bytes), chunk_size):
                yield output_bytes[i : i + chunk_size]

        mock_response.aiter_bytes = mock_aiter_bytes

        async def mock_post(*args, **kwargs):
            return mock_response

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.build_request = MagicMock(return_value=MagicMock())
            mock_client.send = _send_from_post(mock_post)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            # Execute the build
            build = build_store.get_build(build_id)
            await build_runner._execute_build(build)

        # Verify build failed
        build = build_store.get_build(build_id)
        assert build.state == "failed"
        assert build.error_message is not None
        assert "exceeds maximum size" in build.error_message

        # Verify artifact was marked as failed
        artifact = artifact_store.get_artifact(artifact_id, version)
        assert artifact.state == "failed"

    @pytest.mark.asyncio
    async def test_build_executor_non_200(self, build_runner, artifact_store, build_store):
        """Test build failure when executor returns non-200 status."""
        # Create a pending build
        artifact_id, version, build_id = create_test_artifact(artifact_store, build_store)

        # Mock the httpx response with 500 status
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.headers = {}  # Avoid MagicMock leaking to SQLite
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                "Internal Server Error",
                request=MagicMock(),
                response=MagicMock(status_code=500),
            )
        )

        async def mock_post(*args, **kwargs):
            return mock_response

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.build_request = MagicMock(return_value=MagicMock())
            mock_client.send = _send_from_post(mock_post)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            # Execute the build
            build = build_store.get_build(build_id)
            await build_runner._execute_build(build)

        # Verify build failed
        build = build_store.get_build(build_id)
        assert build.state == "failed"
        assert build.error_message is not None
        assert build.error_code == "HTTPStatusError"

        # Verify artifact was marked as failed
        artifact = artifact_store.get_artifact(artifact_id, version)
        assert artifact.state == "failed"

    @pytest.mark.asyncio
    async def test_build_executor_timeout(self, build_runner, artifact_store, build_store):
        """Test build failure when executor times out."""
        # Create a pending build with slow_transform (0.1s timeout)
        artifact_id, version, build_id = create_test_artifact(
            artifact_store, build_store, executor_ref="slow_transform@v1"
        )

        # Mock the httpx client to simulate timeout
        async def mock_post(*args, **kwargs):
            # Sleep longer than timeout
            await asyncio.sleep(1.0)
            return MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.build_request = MagicMock(return_value=MagicMock())
            mock_client.send = _send_from_post(mock_post)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            # Execute the build
            build = build_store.get_build(build_id)
            await build_runner._execute_build(build)

        # Verify build failed due to timeout
        build = build_store.get_build(build_id)
        assert build.state == "failed"
        assert build.error_message is not None
        # The error could be TimeoutError or asyncio.TimeoutError
        assert "Timeout" in build.error_code or "timeout" in build.error_message.lower()

        # Verify artifact was marked as failed
        artifact = artifact_store.get_artifact(artifact_id, version)
        assert artifact.state == "failed"

    @pytest.mark.asyncio
    async def test_build_transform_not_in_registry(self, build_runner, artifact_store, build_store):
        """Test build failure when transform is not in registry."""
        # Create a pending build with unknown transform
        artifact_id = str(uuid.uuid4())

        transform_spec = TransformSpec(
            executor="service://unknown_transform@v1",
            params={},
            inputs=[],
        )

        provenance_hash = f"test-hash-{artifact_id}"
        version = artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash=provenance_hash,
            transform_spec=transform_spec,
            input_versions={},
        )

        build_id = str(uuid.uuid4())
        build_store.create_build(
            build_id=build_id,
            artifact_id=artifact_id,
            version=version,
            executor_ref="unknown_transform@v1",  # Not in registry
            executor_url=None,
        )

        # Execute the build
        build = build_store.get_build(build_id)
        await build_runner._execute_build(build)

        # Verify build failed
        build = build_store.get_build(build_id)
        assert build.state == "failed"
        assert "not found in registry" in build.error_message


class TestConcurrencyControls:
    """Tests for concurrency controls (semaphores)."""

    @pytest.mark.asyncio
    async def test_global_concurrency_limit(self, build_runner, artifact_store, build_store):
        """Test that global concurrency limit is respected."""
        # Create more builds than global limit
        num_builds = 10
        builds = []
        for i in range(num_builds):
            artifact_id, version, build_id = create_test_artifact(artifact_store, build_store)
            builds.append(build_id)

        # Track concurrent executions
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def counting_execute(build):
            nonlocal max_concurrent, current_concurrent
            async with lock:
                current_concurrent += 1
                max_concurrent = max(max_concurrent, current_concurrent)
            try:
                await asyncio.sleep(0.05)  # Simulate work
            finally:
                async with lock:
                    current_concurrent -= 1

        # Patch _execute_build to track concurrency
        with patch.object(build_runner, "_execute_build", counting_execute):
            # Start the runner and wait for builds
            await build_runner.start()
            await asyncio.sleep(0.5)  # Let builds run
            await build_runner.stop()

        # Should not exceed global limit
        assert max_concurrent <= build_runner.config.max_concurrent_builds

    @pytest.mark.asyncio
    async def test_per_tenant_concurrency_limit(self, build_runner, artifact_store, build_store):
        """Test that per-tenant concurrency limit is respected."""
        tenant_id = "test-tenant"
        num_builds = 5

        # Create builds for same tenant
        builds = []
        for i in range(num_builds):
            artifact_id, version, build_id = create_test_artifact(
                artifact_store, build_store, tenant_id=tenant_id
            )
            builds.append(build_id)

        # Track concurrent executions per tenant
        tenant_concurrent = {}
        max_tenant_concurrent = 0
        lock = asyncio.Lock()

        async def counting_execute(build):
            nonlocal max_tenant_concurrent
            tid = build.tenant_id or "__default__"
            async with lock:
                tenant_concurrent[tid] = tenant_concurrent.get(tid, 0) + 1
                max_tenant_concurrent = max(max_tenant_concurrent, tenant_concurrent[tid])
            try:
                await asyncio.sleep(0.05)  # Simulate work
            finally:
                async with lock:
                    tenant_concurrent[tid] -= 1

        # Patch _execute_build to track tenant concurrency
        with patch.object(build_runner, "_execute_build", counting_execute):
            await build_runner.start()
            await asyncio.sleep(0.5)
            await build_runner.stop()

        # Should not exceed per-tenant limit
        assert max_tenant_concurrent <= build_runner.config.max_builds_per_tenant


class TestInputAcquisition:
    """Tests for input acquisition (_acquire_input)."""

    @pytest.mark.asyncio
    async def test_acquire_artifact_input(
        self, build_runner, artifact_store, build_store, artifact_dir
    ):
        """Test acquiring an artifact as input."""
        # Create a ready artifact as input
        input_artifact_id = str(uuid.uuid4())
        input_data = {"x": [1, 2, 3]}
        input_bytes = create_arrow_ipc_bytes(input_data)

        # Create and finalize input artifact
        input_version = artifact_store.create_artifact(
            artifact_id=input_artifact_id,
            provenance_hash=f"input-hash-{input_artifact_id}",
            transform_spec=None,
            input_versions={},
        )
        blob_path = artifact_dir / "blobs" / f"{input_artifact_id}@v={input_version}.arrow"
        blob_path.write_bytes(input_bytes)
        artifact_store.finalize_artifact(
            artifact_id=input_artifact_id,
            version=input_version,
            schema_json="{}",
            row_count=3,
            byte_size=len(input_bytes),
        )

        # Acquire the input
        temp_files = []
        input_uri = f"strata://artifact/{input_artifact_id}@v={input_version}"
        result_path = await build_runner._acquire_input(input_uri, temp_files)

        # Verify temp file was created
        assert result_path.exists()
        assert result_path in temp_files
        assert result_path.read_bytes() == input_bytes

        # Clean up
        for f in temp_files:
            if f.exists():
                f.unlink()

    @pytest.mark.asyncio
    async def test_acquire_invalid_uri(self, build_runner):
        """Test that invalid input URI raises error."""
        temp_files = []
        with pytest.raises(ValueError, match="Unsupported input URI"):
            await build_runner._acquire_input("invalid://uri", temp_files)

    @pytest.mark.asyncio
    async def test_acquire_named_input_uses_build_tenant(self, build_runner, artifact_store):
        """Name URIs should resolve within the owning build tenant."""
        bytes_a = create_arrow_ipc_bytes({"tenant": ["a"]})
        bytes_b = create_arrow_ipc_bytes({"tenant": ["b"]})

        version_a = artifact_store.create_artifact(
            artifact_id="tenant-a-input",
            provenance_hash="tenant-a-input",
            tenant="team-a",
        )
        artifact_store.write_blob("tenant-a-input", version_a, bytes_a)
        artifact_store.finalize_artifact("tenant-a-input", version_a, "{}", 1, len(bytes_a))
        artifact_store.set_name("shared-input", "tenant-a-input", version_a, tenant="team-a")

        version_b = artifact_store.create_artifact(
            artifact_id="tenant-b-input",
            provenance_hash="tenant-b-input",
            tenant="team-b",
        )
        artifact_store.write_blob("tenant-b-input", version_b, bytes_b)
        artifact_store.finalize_artifact("tenant-b-input", version_b, "{}", 1, len(bytes_b))
        artifact_store.set_name("shared-input", "tenant-b-input", version_b, tenant="team-b")

        temp_files = []
        result_path = await build_runner._acquire_input(
            "strata://name/shared-input",
            temp_files,
            tenant_id="team-a",
        )

        assert result_path.read_bytes() == bytes_a

    def test_scan_to_file_sync_uses_fetch_pipeline_on_cold_cache(self, build_runner, artifact_dir):
        """Table inputs should fetch through the planner/fetcher path, not raw cache files."""
        batch = pa.record_batch([pa.array([1, 2, 3])], names=["id"])
        task = Task(
            file_path="file:///warehouse/data.parquet",
            row_group_id=0,
            cache_key=CacheKey(
                tenant_id="team-a",
                table_identity=TableIdentity(catalog="strata", namespace="ns", table="tbl"),
                snapshot_id=1,
                file_path="file:///warehouse/data.parquet",
                row_group_id=0,
                projection_fingerprint="*",
            ),
            num_rows=3,
        )
        plan = ReadPlan(
            table_uri="file:///warehouse#ns.tbl",
            table_identity=TableIdentity(catalog="strata", namespace="ns", table="tbl"),
            snapshot_id=1,
            tasks=[task],
            schema=batch.schema,
        )

        build_runner.scan_planner = MagicMock()
        build_runner.scan_planner.plan.return_value = plan
        build_runner.scan_fetcher = MagicMock()
        build_runner.scan_fetcher.fetch.return_value = batch

        output_path = artifact_dir / "scan-output.arrow"
        build_runner._scan_to_file_sync("file:///warehouse#ns.tbl", output_path)

        build_runner.scan_planner.plan.assert_called_once()
        build_runner.scan_fetcher.fetch.assert_called_once_with(task)

        with pa.ipc.open_stream(output_path) as reader:
            table = reader.read_all()
        assert table.column("id").to_pylist() == [1, 2, 3]


class TestBuildPolling:
    """Tests for build polling loop."""

    @pytest.mark.asyncio
    async def test_pending_builds_picked_up(self, build_runner, artifact_store, build_store):
        """Test that pending builds are picked up by the polling loop."""
        # Create a pending build
        artifact_id, version, build_id = create_test_artifact(artifact_store, build_store)

        # Mock successful execution
        executed_builds = []

        async def mock_execute(build, already_claimed=False):
            executed_builds.append(build.build_id)
            build_store.start_build(build.build_id)
            build_store.complete_build(build.build_id)

        with patch.object(build_runner, "_execute_build", mock_execute):
            await build_runner.start()
            await asyncio.sleep(0.2)  # Wait for polling
            await build_runner.stop()

        # Verify build was executed
        assert build_id in executed_builds

    @pytest.mark.asyncio
    async def test_shutdown_cancels_builds(self, build_runner, artifact_store, build_store):
        """Test that shutdown cancels in-progress builds."""
        # Create a pending build
        artifact_id, version, build_id = create_test_artifact(artifact_store, build_store)

        # Mock slow execution
        async def slow_execute(build, already_claimed=False):
            build_store.start_build(build.build_id)
            await asyncio.sleep(10.0)  # Long sleep

        with patch.object(build_runner, "_execute_build", slow_execute):
            await build_runner.start()
            await asyncio.sleep(0.1)  # Let it start
            await build_runner.stop()

        # Build should be marked as failed due to shutdown
        build = build_store.get_build(build_id)
        assert build.state == "failed"
        assert "shutdown" in build.error_message.lower()


class TestArrowMetadataExtraction:
    """Tests for Arrow metadata extraction."""

    def test_read_arrow_metadata(self, build_runner, artifact_dir):
        """Test reading Arrow IPC metadata."""
        # Create test Arrow file
        data = {"id": [1, 2, 3], "name": ["a", "b", "c"]}
        arrow_bytes = create_arrow_ipc_bytes(data)

        temp_file = artifact_dir / "test.arrow"
        temp_file.write_bytes(arrow_bytes)

        # Read metadata
        schema_json, row_count = build_runner._read_arrow_metadata(temp_file)

        assert row_count == 3
        assert "id" in schema_json
        assert "name" in schema_json

        temp_file.unlink()


class TestV1PushMemoryBounds:
    """The push protocol must not buffer unbounded data.

    ``client.post`` reads the entire response body into memory before
    returning, so the ``max_output_bytes`` check ran only *after* the whole
    payload had been allocated — an executor returning 50 GB OOM'd the server
    despite a 1 GB limit. Streaming makes the check fire during transfer.
    """

    @pytest.mark.asyncio
    async def test_output_limit_aborts_mid_transfer(self, build_runner, artifact_dir):
        """The download stops at the limit instead of consuming the whole body."""
        chunks_yielded = 0

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()

        async def endless_chunks(chunk_size=65536):
            nonlocal chunks_yielded
            while True:
                chunks_yielded += 1
                yield b"x" * 1024

        mock_response.aiter_bytes = endless_chunks

        async def mock_post(*args, **kwargs):
            return mock_response

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.build_request = MagicMock(return_value=MagicMock())
            mock_client.send = _send_from_post(mock_post)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            with pytest.raises(ValueError, match="exceeds maximum size"):
                await build_runner._call_executor(
                    executor_url="http://executor:8080",
                    metadata={},
                    input_files=[],
                    timeout=30.0,
                    max_output_bytes=4096,
                    temp_files=[],
                )

        # Aborted after a handful of 1 KiB chunks rather than draining an
        # unbounded body.
        assert chunks_yielded < 20, f"read {chunks_yielded} chunks past the 4 KiB limit"

    @pytest.mark.asyncio
    async def test_inputs_are_streamed_not_slurped(self, build_runner, artifact_dir, monkeypatch):
        """Inputs are handed to httpx as file objects; the bytes form loaded
        every input fully into RAM and httpx then built the whole multipart
        body on top of that."""
        payload = artifact_dir / "big-input.arrow"
        payload.write_bytes(b"y" * 4096)

        from pathlib import Path as _Path

        monkeypatch.setattr(
            _Path, "read_bytes", lambda self: pytest.fail(f"input slurped into memory: {self}")
        )

        captured = {}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.raise_for_status = MagicMock()

        async def one_chunk(chunk_size=65536):
            yield b""

        mock_response.aiter_bytes = one_chunk

        async def mock_post(*args, **kwargs):
            return mock_response

        def build_request(method, url, files=None, headers=None):
            captured["files"] = files
            return MagicMock()

        with patch("httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.build_request = build_request
            mock_client.send = _send_from_post(mock_post)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            await build_runner._call_executor(
                executor_url="http://executor:8080",
                metadata={},
                input_files=[("input0", payload)],
                timeout=30.0,
                max_output_bytes=1024 * 1024,
                temp_files=[],
            )

        _name, handle, _ctype = captured["files"]["input0"]
        assert hasattr(handle, "read"), "input should be a file object, not bytes"
