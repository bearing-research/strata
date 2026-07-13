"""Tests for Strata Executor Protocol v1.

These tests verify:
1. Protocol type definitions
2. Reference executor implementation
3. Protocol header handling
4. Input/output serialization
5. HTTP integration with executor server
"""

import json
import threading

import httpx
import pyarrow as pa
import pytest
import uvicorn

from strata.types import (
    EXECUTOR_LOGS_HEADER,
    EXECUTOR_PROTOCOL_HEADER,
    EXECUTOR_PROTOCOL_VERSION,
    ExecutorCapabilities,
    ExecutorHealthResponse,
    ExecutorInputDescriptor,
    ExecutorManifest,
    ExecutorManifestInput,
    ExecutorRequestMetadata,
    ExecutorResponse,
    ExecutorTransformSpec,
)
from tests.conftest import find_free_port, ipc_bytes_to_table, table_to_ipc_bytes, wait_for_server


class TestProtocolTypes:
    """Tests for protocol type definitions."""

    def test_protocol_version_constant(self):
        """Protocol version is 'v1'."""
        assert EXECUTOR_PROTOCOL_VERSION == "v1"

    def test_protocol_headers(self):
        """Protocol headers are correctly defined."""
        assert EXECUTOR_PROTOCOL_HEADER == "X-Strata-Executor-Protocol"
        assert EXECUTOR_LOGS_HEADER == "X-Strata-Logs"

    def test_executor_input_descriptor(self):
        """ExecutorInputDescriptor serializes correctly."""
        desc = ExecutorInputDescriptor(
            name="input0",
            format="arrow_ipc_stream",
            uri="strata://artifact/abc123@v=1",
            byte_size=1024,
        )
        data = desc.model_dump()
        assert data["name"] == "input0"
        assert data["format"] == "arrow_ipc_stream"
        assert data["uri"] == "strata://artifact/abc123@v=1"
        assert data["byte_size"] == 1024

    def test_executor_transform_spec(self):
        """ExecutorTransformSpec serializes correctly."""
        spec = ExecutorTransformSpec(
            ref="duckdb_sql@v1",
            code_hash="abc123",
            params={"sql": "SELECT * FROM input0"},
        )
        data = spec.model_dump()
        assert data["ref"] == "duckdb_sql@v1"
        assert data["code_hash"] == "abc123"
        assert data["params"]["sql"] == "SELECT * FROM input0"

    def test_executor_request_metadata(self):
        """ExecutorRequestMetadata serializes correctly."""
        meta = ExecutorRequestMetadata(
            build_id="build-123",
            tenant="acme",
            principal="user@example.com",
            provenance_hash="sha256...",
            transform=ExecutorTransformSpec(
                ref="duckdb_sql@v1",
                code_hash="abc123",
                params={"sql": "SELECT 1"},
            ),
            inputs=[
                ExecutorInputDescriptor(name="input0"),
                ExecutorInputDescriptor(name="input1"),
            ],
        )
        data = meta.model_dump()
        assert data["protocol_version"] == "v1"
        assert data["build_id"] == "build-123"
        assert data["tenant"] == "acme"
        assert len(data["inputs"]) == 2

    def test_executor_response_success(self):
        """ExecutorResponse for success case."""
        resp = ExecutorResponse(
            success=True,
            duration_ms=150.5,
            output_rows=1000,
            output_bytes=50000,
        )
        data = resp.model_dump()
        assert data["success"] is True
        assert data["error_code"] is None
        assert data["output_rows"] == 1000

    def test_executor_response_error(self):
        """ExecutorResponse for error case."""
        resp = ExecutorResponse(
            success=False,
            error_code="SQL_ERROR",
            error_message="Syntax error near 'SELEC'",
            logs="Error at line 1",
        )
        data = resp.model_dump()
        assert data["success"] is False
        assert data["error_code"] == "SQL_ERROR"
        assert "Syntax error" in data["error_message"]

    def test_executor_manifest(self):
        """ExecutorManifest serializes correctly for pull model."""
        manifest = ExecutorManifest(
            build_id="build-456",
            metadata={"transform": {"ref": "duckdb_sql@v1"}},
            inputs=[
                ExecutorManifestInput(
                    name="input0",
                    download_url="https://strata.example.com/download?sig=...",
                    byte_size=1024,
                ),
            ],
            upload_url="https://strata.example.com/upload?sig=...",
            finalize_url="https://strata.example.com/finalize",
            max_output_bytes=1073741824,
            timeout_seconds=300.0,
        )
        data = manifest.model_dump()
        assert data["protocol_version"] == "v1"
        assert data["build_id"] == "build-456"
        assert len(data["inputs"]) == 1
        assert data["max_output_bytes"] == 1073741824

    def test_executor_capabilities(self):
        """ExecutorCapabilities serializes correctly."""
        caps = ExecutorCapabilities(
            transform_refs=["duckdb_sql@v1", "pandas_transform@v1"],
            max_input_bytes=10 * 1024 * 1024 * 1024,
            max_output_bytes=1024 * 1024 * 1024,
            max_concurrent_executions=10,
            features={"streaming": True},
        )
        data = caps.model_dump()
        assert "v1" in data["protocol_versions"]
        assert "duckdb_sql@v1" in data["transform_refs"]
        assert data["max_concurrent_executions"] == 10

    def test_executor_health_response(self):
        """ExecutorHealthResponse serializes correctly."""
        health = ExecutorHealthResponse(
            status="healthy",
            capabilities=ExecutorCapabilities(
                transform_refs=["duckdb_sql@v1"],
            ),
            version="1.2.3",
            uptime_seconds=3600.0,
            active_executions=3,
        )
        data = health.model_dump()
        assert data["status"] == "healthy"
        assert data["version"] == "1.2.3"
        assert data["active_executions"] == 3


class TestReferenceExecutor:
    """Tests for the reference DuckDB executor."""

    def test_duckdb_executor_init(self):
        """DuckDBExecutor initializes correctly."""
        from strata.transforms.reference_executor import DuckDBExecutor

        executor = DuckDBExecutor(max_memory_mb=512)
        assert executor.max_memory_mb == 512
        assert "duckdb_sql@v1" in executor.get_transform_refs()

    def test_duckdb_executor_health_check(self):
        """DuckDBExecutor returns correct health check."""
        from strata.transforms.reference_executor import DuckDBExecutor

        executor = DuckDBExecutor()
        health = executor.health_check()

        assert health["status"] == "healthy"
        assert "v1" in health["capabilities"]["protocol_versions"]
        assert "duckdb_sql@v1" in health["capabilities"]["transform_refs"]

    def test_duckdb_executor_simple_query(self):
        """DuckDBExecutor executes simple SQL query."""
        from strata.transforms.reference_executor import DuckDBExecutor, ExecutorInput

        executor = DuckDBExecutor()

        # Create input table
        input_table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        input_data = table_to_ipc_bytes(input_table)

        result = executor.execute(
            transform_ref="duckdb_sql@v1",
            params={"sql": "SELECT x * 2 as doubled FROM input0"},
            inputs=[ExecutorInput(name="input0", data=input_data)],
        )

        assert result.success is True
        assert result.output_bytes is not None
        assert result.output_rows == 3
        assert result.logs is not None

        # Verify output
        output_table = ipc_bytes_to_table(result.output_bytes)
        assert output_table.num_rows == 3
        assert output_table.column("doubled").to_pylist() == [2, 4, 6]

    def test_duckdb_executor_multiple_inputs(self):
        """DuckDBExecutor handles multiple inputs correctly."""
        from strata.transforms.reference_executor import DuckDBExecutor, ExecutorInput

        executor = DuckDBExecutor()

        # Create input tables
        users = pa.table({"id": [1, 2], "name": ["Alice", "Bob"]})
        orders = pa.table({"user_id": [1, 1, 2], "amount": [100, 200, 150]})

        result = executor.execute(
            transform_ref="duckdb_sql@v1",
            params={
                "sql": """
                SELECT u.name, SUM(o.amount) as total
                FROM input0 u
                JOIN input1 o ON u.id = o.user_id
                GROUP BY u.name
                ORDER BY u.name
                """
            },
            inputs=[
                ExecutorInput(name="input0", data=table_to_ipc_bytes(users)),
                ExecutorInput(name="input1", data=table_to_ipc_bytes(orders)),
            ],
        )

        assert result.success is True
        assert result.output_bytes is not None
        output_table = ipc_bytes_to_table(result.output_bytes)
        assert output_table.num_rows == 2
        assert output_table.column("name").to_pylist() == ["Alice", "Bob"]
        assert output_table.column("total").to_pylist() == [300, 150]

    def test_duckdb_executor_missing_sql(self):
        """DuckDBExecutor returns error for missing SQL."""
        from strata.transforms.reference_executor import DuckDBExecutor

        executor = DuckDBExecutor()

        result = executor.execute(
            transform_ref="duckdb_sql@v1",
            params={},  # Missing SQL
            inputs=[],
        )

        assert result.success is False
        assert result.error_code == "INVALID_PARAMS"
        assert result.error_message is not None
        assert "sql" in result.error_message.lower()

    def test_duckdb_executor_sql_error(self):
        """DuckDBExecutor handles SQL errors gracefully."""
        from strata.transforms.reference_executor import DuckDBExecutor, ExecutorInput

        executor = DuckDBExecutor()

        input_table = pa.table({"x": [1, 2, 3]})

        result = executor.execute(
            transform_ref="duckdb_sql@v1",
            params={"sql": "SELEC * FROM input0"},  # Invalid SQL
            inputs=[ExecutorInput(name="input0", data=table_to_ipc_bytes(input_table))],
        )

        assert result.success is False
        assert result.error_code is not None
        assert result.logs is not None


class TestUtilityFunctions:
    """Tests for executor utility functions."""

    def test_parse_arrow_inputs(self):
        """parse_arrow_inputs extracts and sorts inputs."""
        from strata.transforms.reference_executor import parse_arrow_inputs

        table1 = pa.table({"a": [1]})
        table2 = pa.table({"b": [2]})

        file_parts = {
            "metadata": b"{}",  # Should be ignored
            "input1": table_to_ipc_bytes(table2),
            "input0": table_to_ipc_bytes(table1),
        }

        inputs = parse_arrow_inputs(file_parts)

        assert len(inputs) == 2
        assert inputs[0].name == "input0"
        assert inputs[1].name == "input1"

    def test_serialize_arrow_output(self):
        """serialize_arrow_output produces valid IPC bytes."""
        from strata.transforms.reference_executor import serialize_arrow_output

        table = pa.table({"x": [1, 2, 3]})
        data = serialize_arrow_output(table)

        # Verify we can read it back
        result = ipc_bytes_to_table(data)
        assert result.equals(table)

    def test_encode_decode_logs_header(self):
        """encode_logs_header and decode_logs_header are inverses."""
        from strata.transforms.reference_executor import (
            decode_logs_header,
            encode_logs_header,
        )

        original = "Hello, World!\nLine 2"
        encoded = encode_logs_header(original)
        decoded = decode_logs_header(encoded)

        assert decoded == original

    def test_encode_logs_header_unicode(self):
        """encode_logs_header handles Unicode correctly."""
        from strata.transforms.reference_executor import (
            decode_logs_header,
            encode_logs_header,
        )

        original = "日本語テスト: こんにちは"
        encoded = encode_logs_header(original)
        decoded = decode_logs_header(encoded)

        assert decoded == original


class TestBaseExecutorInterface:
    """Tests for the BaseExecutor abstract interface."""

    def test_custom_executor(self):
        """Custom executor can be implemented."""
        from strata.transforms.reference_executor import (
            BaseExecutor,
            ExecutionResult,
            ExecutorInput,
        )

        class IdentityExecutor(BaseExecutor):
            """Simple executor that returns first input unchanged."""

            def get_transform_refs(self) -> list[str]:
                return ["scan@v1"]

            def execute(
                self,
                transform_ref: str,
                params: dict,
                inputs: list[ExecutorInput],
            ) -> ExecutionResult:
                if not inputs:
                    return ExecutionResult(
                        success=False,
                        error_code="NO_INPUT",
                        error_message="At least one input required",
                    )
                return ExecutionResult(
                    success=True,
                    output_bytes=inputs[0].data,
                    output_rows=1,  # Simplified
                )

        executor = IdentityExecutor()
        assert executor.get_transform_refs() == ["scan@v1"]

        table = pa.table({"x": [42]})
        result = executor.execute(
            transform_ref="scan@v1",
            params={},
            inputs=[ExecutorInput(name="input0", data=table_to_ipc_bytes(table))],
        )

        assert result.success is True
        assert result.output_bytes is not None
        output = ipc_bytes_to_table(result.output_bytes)
        assert output.column("x").to_pylist() == [42]


# ---------------------------------------------------------------------------
# HTTP Integration Tests (Full Server)
# ---------------------------------------------------------------------------


@pytest.fixture
def executor_server():
    """Start a reference executor server for integration tests."""
    from strata.transforms.reference_executor import create_executor_app

    port = find_free_port()
    app = create_executor_app()

    server_config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        # Avoid uvicorn's legacy websockets backend (default ws="auto"), which
        # emits a DeprecationWarning and is broken on CPython 3.14.
        ws="websockets-sansio",
    )
    server_instance = uvicorn.Server(server_config)
    thread = threading.Thread(target=server_instance.run, daemon=True)
    thread.start()

    if not wait_for_server(port):
        raise RuntimeError(f"Executor server failed to start on port {port}")

    try:
        yield {"port": port, "base_url": f"http://127.0.0.1:{port}"}
    finally:
        server_instance.should_exit = True
        thread.join(timeout=2.0)


class TestExecutorHTTPIntegration:
    """HTTP integration tests for the executor server."""

    def test_health_endpoint(self, executor_server):
        """Health endpoint returns capabilities."""
        resp = httpx.get(f"{executor_server['base_url']}/health")
        assert resp.status_code == 200

        data = resp.json()
        assert data["status"] == "healthy"
        assert "v1" in data["capabilities"]["protocol_versions"]
        assert "duckdb_sql@v1" in data["capabilities"]["transform_refs"]

    def test_execute_simple_query(self, executor_server):
        """Execute simple SQL query via HTTP."""
        base_url = executor_server["base_url"]

        # Create input table
        input_table = pa.table({"value": [10, 20, 30]})
        input_bytes = table_to_ipc_bytes(input_table)

        # Build multipart request
        metadata = json.dumps(
            {
                "protocol_version": "v1",
                "build_id": "test-build-001",
                "provenance_hash": "test-hash",
                "transform": {
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT SUM(value) as total FROM input0"},
                },
                "inputs": [
                    {"name": "input0", "size_bytes": len(input_bytes)},
                ],
            }
        )

        files = {
            "metadata": ("metadata.json", metadata, "application/json"),
            "input0": ("input0.arrow", input_bytes, "application/vnd.apache.arrow.stream"),
        }

        resp = httpx.post(f"{base_url}/v1/execute", files=files)
        assert resp.status_code == 200
        assert resp.headers.get(EXECUTOR_PROTOCOL_HEADER) == EXECUTOR_PROTOCOL_VERSION

        # Parse response as Arrow IPC
        output_table = ipc_bytes_to_table(resp.content)
        assert output_table.num_rows == 1
        assert output_table.column("total").to_pylist() == [60]

    def test_execute_with_multiple_inputs(self, executor_server):
        """Execute query with multiple inputs via HTTP."""
        base_url = executor_server["base_url"]

        # Create input tables
        products = pa.table({"id": [1, 2], "name": ["Widget", "Gadget"], "price": [9.99, 19.99]})
        orders = pa.table({"product_id": [1, 1, 2], "qty": [2, 1, 3]})

        products_bytes = table_to_ipc_bytes(products)
        orders_bytes = table_to_ipc_bytes(orders)

        metadata = json.dumps(
            {
                "protocol_version": "v1",
                "build_id": "test-build-002",
                "provenance_hash": "test-hash-2",
                "transform": {
                    "ref": "duckdb_sql@v1",
                    "params": {
                        "sql": """
                        SELECT p.name, SUM(o.qty * p.price) as revenue
                        FROM input0 p
                        JOIN input1 o ON p.id = o.product_id
                        GROUP BY p.name
                        ORDER BY p.name
                    """
                    },
                },
                "inputs": [
                    {"name": "input0", "size_bytes": len(products_bytes)},
                    {"name": "input1", "size_bytes": len(orders_bytes)},
                ],
            }
        )

        files = {
            "metadata": ("metadata.json", metadata, "application/json"),
            "input0": ("input0.arrow", products_bytes, "application/vnd.apache.arrow.stream"),
            "input1": ("input1.arrow", orders_bytes, "application/vnd.apache.arrow.stream"),
        }

        resp = httpx.post(f"{base_url}/v1/execute", files=files)
        assert resp.status_code == 200

        output_table = ipc_bytes_to_table(resp.content)
        assert output_table.num_rows == 2
        assert output_table.column("name").to_pylist() == ["Gadget", "Widget"]
        # Widget: 2*9.99 + 1*9.99 = 29.97
        # Gadget: 3*19.99 = 59.97
        revenues = output_table.column("revenue").to_pylist()
        assert abs(revenues[0] - 59.97) < 0.01  # Gadget
        assert abs(revenues[1] - 29.97) < 0.01  # Widget

    def test_execute_logs_in_header(self, executor_server):
        """Executor logs are returned in header."""
        from strata.transforms.reference_executor import decode_logs_header

        base_url = executor_server["base_url"]

        input_table = pa.table({"x": [1, 2, 3]})
        input_bytes = table_to_ipc_bytes(input_table)

        metadata = json.dumps(
            {
                "protocol_version": "v1",
                "build_id": "test-build-003",
                "provenance_hash": "test-hash-3",
                "transform": {
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT COUNT(*) as cnt FROM input0"},
                },
                "inputs": [
                    {"name": "input0", "size_bytes": len(input_bytes)},
                ],
            }
        )

        files = {
            "metadata": ("metadata.json", metadata, "application/json"),
            "input0": ("input0.arrow", input_bytes, "application/vnd.apache.arrow.stream"),
        }

        resp = httpx.post(f"{base_url}/v1/execute", files=files)
        assert resp.status_code == 200

        # Check logs header
        logs_header = resp.headers.get(EXECUTOR_LOGS_HEADER)
        assert logs_header is not None

        logs = decode_logs_header(logs_header)
        assert "Executing SQL" in logs
        assert "input0" in logs
        assert "Result:" in logs

    def test_execute_invalid_sql(self, executor_server):
        """Invalid SQL returns error response."""
        base_url = executor_server["base_url"]

        input_table = pa.table({"x": [1]})
        input_bytes = table_to_ipc_bytes(input_table)

        metadata = json.dumps(
            {
                "protocol_version": "v1",
                "build_id": "test-build-004",
                "provenance_hash": "test-hash-4",
                "transform": {
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELEC * FROM invalid_table"},  # Typo in SELECT
                },
                "inputs": [
                    {"name": "input0", "size_bytes": len(input_bytes)},
                ],
            }
        )

        files = {
            "metadata": ("metadata.json", metadata, "application/json"),
            "input0": ("input0.arrow", input_bytes, "application/vnd.apache.arrow.stream"),
        }

        resp = httpx.post(f"{base_url}/v1/execute", files=files)
        assert resp.status_code == 400

        data = resp.json()
        assert data["success"] is False
        assert data["error_code"] is not None
        assert data["error_message"] is not None

    def test_execute_missing_sql_param(self, executor_server):
        """Missing SQL param returns error response."""
        base_url = executor_server["base_url"]

        input_table = pa.table({"x": [1]})
        input_bytes = table_to_ipc_bytes(input_table)

        metadata = json.dumps(
            {
                "protocol_version": "v1",
                "build_id": "test-build-005",
                "provenance_hash": "test-hash-5",
                "transform": {
                    "ref": "duckdb_sql@v1",
                    "params": {},  # Missing sql
                },
                "inputs": [
                    {"name": "input0", "size_bytes": len(input_bytes)},
                ],
            }
        )

        files = {
            "metadata": ("metadata.json", metadata, "application/json"),
            "input0": ("input0.arrow", input_bytes, "application/vnd.apache.arrow.stream"),
        }

        resp = httpx.post(f"{base_url}/v1/execute", files=files)
        assert resp.status_code == 400

        data = resp.json()
        assert data["success"] is False
        assert data["error_code"] == "INVALID_PARAMS"
        assert data["error_message"] is not None
        assert "sql" in data["error_message"].lower()

    def test_execute_unsupported_transform(self, executor_server):
        """Unsupported transform returns error."""
        base_url = executor_server["base_url"]

        metadata = json.dumps(
            {
                "protocol_version": "v1",
                "build_id": "test-build-006",
                "provenance_hash": "test-hash-6",
                "transform": {
                    "ref": "pandas_transform@v1",  # Not supported by DuckDBExecutor
                    "params": {"code": "df.head()"},
                },
                "inputs": [],
            }
        )

        files = {
            "metadata": ("metadata.json", metadata, "application/json"),
        }

        resp = httpx.post(f"{base_url}/v1/execute", files=files)
        assert resp.status_code == 400
        assert "Unsupported transform" in resp.json()["detail"]

    def test_execute_invalid_protocol_version(self, executor_server):
        """Invalid protocol version returns error."""
        base_url = executor_server["base_url"]

        metadata = json.dumps(
            {
                "protocol_version": "v99",  # Invalid version
                "build_id": "test-build-007",
                "provenance_hash": "test-hash-7",
                "transform": {
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 1"},
                },
                "inputs": [],
            }
        )

        files = {
            "metadata": ("metadata.json", metadata, "application/json"),
        }

        resp = httpx.post(f"{base_url}/v1/execute", files=files)
        assert resp.status_code == 400
        assert "protocol version" in resp.json()["detail"].lower()
