"""Reference executor implementation for Strata Protocol v1.

This module provides a reference implementation of a Strata executor that can
be used as a template for building external executors. It demonstrates:

1. Protocol v1 request/response handling
2. Multipart form parsing for inputs
3. Arrow IPC stream processing
4. Error handling and logging

Usage:
    # Run as a standalone server (for development/testing)
    uvicorn strata.transforms.reference_executor:app --port 8080

    # Or import the executor class for embedding in your own application
    from strata.transforms.reference_executor import DuckDBExecutor
    executor = DuckDBExecutor()
    result = executor.execute(metadata, inputs)

Protocol v1 Endpoints:
    POST /v1/execute      - Execute a transform (multipart/form-data)
    GET  /health          - Health check with capabilities

See strata.types for protocol type definitions:
    - ExecutorRequestMetadata: Request metadata schema
    - ExecutorInputDescriptor: Input descriptor schema
    - ExecutorTransformSpec: Transform specification schema
    - ExecutorResponse: Error response schema
    - ExecutorHealthResponse: Health check response schema
"""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# Import at module level to avoid annotation resolution issues with FastAPI
from fastapi import Request as FastAPIRequest
from fastapi import UploadFile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Executor Interface (Abstract Base Class)
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """Result from executor execution.

    Attributes:
        success: Whether execution succeeded
        output_bytes: Arrow IPC stream bytes (on success)
        error_code: Machine-readable error code (on failure)
        error_message: Human-readable error message (on failure)
        logs: Executor logs (stdout/stderr)
        duration_ms: Execution time in milliseconds
        output_rows: Number of rows in output
    """

    success: bool
    output_bytes: bytes | None = None
    error_code: str | None = None
    error_message: str | None = None
    logs: str | None = None
    duration_ms: float | None = None
    output_rows: int | None = None


@dataclass
class ExecutorInput:
    """Input data for executor.

    Attributes:
        name: Input name (e.g., "input0", "input1")
        data: Arrow IPC stream bytes
    """

    name: str
    data: bytes


class BaseExecutor(ABC):
    """Abstract base class for Strata executors.

    Implement this interface to create a new executor type.
    The execute() method receives parsed metadata and input data,
    and returns an ExecutionResult.

    Example:
        class MyCustomExecutor(BaseExecutor):
            def get_transform_refs(self) -> list[str]:
                return ["my_transform@v1", "my_transform@v2"]

            def execute(
                self,
                transform_ref: str,
                params: dict[str, Any],
                inputs: list[ExecutorInput],
            ) -> ExecutionResult:
                # Your transform logic here
                ...
    """

    @abstractmethod
    def get_transform_refs(self) -> list[str]:
        """Return list of transform references this executor supports.

        Examples: ["duckdb_sql@v1"], ["pandas_transform@v1", "pandas_transform@v2"]
        """
        pass

    @abstractmethod
    def execute(
        self,
        transform_ref: str,
        params: dict[str, Any],
        inputs: list[ExecutorInput],
    ) -> ExecutionResult:
        """Execute the transform.

        Args:
            transform_ref: Transform reference (e.g., "duckdb_sql@v1")
            params: Transform parameters (e.g., {"sql": "SELECT ..."})
            inputs: List of input data (Arrow IPC stream bytes)

        Returns:
            ExecutionResult with output or error
        """
        pass

    def health_check(self) -> dict:
        """Return health status and capabilities.

        Override this to add custom health checks.
        """
        from strata.types import EXECUTOR_PROTOCOL_VERSION

        return {
            "status": "healthy",
            "capabilities": {
                "protocol_versions": [EXECUTOR_PROTOCOL_VERSION],
                "transform_refs": self.get_transform_refs(),
            },
        }


# ---------------------------------------------------------------------------
# DuckDB SQL Executor (Reference Implementation)
# ---------------------------------------------------------------------------


class DuckDBExecutor(BaseExecutor):
    """DuckDB SQL executor - reference implementation.

    This executor runs DuckDB SQL queries on Arrow input tables.
    Inputs are registered as 'input0', 'input1', etc. in DuckDB.

    Example SQL:
        SELECT a.id, b.value
        FROM input0 a
        JOIN input1 b ON a.id = b.id
        WHERE a.timestamp > '2024-01-01'
    """

    def __init__(self, max_memory_mb: int = 1024):
        """Initialize DuckDB executor.

        Args:
            max_memory_mb: Maximum memory for DuckDB (default 1GB)
        """
        self.max_memory_mb = max_memory_mb

    def get_transform_refs(self) -> list[str]:
        return ["duckdb_sql@v1"]

    def execute(
        self,
        transform_ref: str,
        params: dict[str, Any],
        inputs: list[ExecutorInput],
    ) -> ExecutionResult:
        """Execute DuckDB SQL query.

        Args:
            transform_ref: Must be "duckdb_sql@v1"
            params: Must contain "sql" key with the query
            inputs: Arrow IPC stream inputs

        Returns:
            ExecutionResult with Arrow IPC output
        """
        import io
        import time

        start_time = time.time()
        logs_buffer = io.StringIO()

        try:
            import duckdb
            import pyarrow.ipc as ipc
        except ImportError as e:
            return ExecutionResult(
                success=False,
                error_code="IMPORT_ERROR",
                error_message=f"Missing dependency: {e}",
            )

        try:
            # Extract SQL from params
            sql = params.get("sql")
            if not sql:
                return ExecutionResult(
                    success=False,
                    error_code="INVALID_PARAMS",
                    error_message="Missing 'sql' in params",
                )

            logs_buffer.write(f"Executing SQL: {sql[:100]}...\n")

            # Create DuckDB connection with memory limit
            conn = duckdb.connect(":memory:")
            conn.execute(f"SET memory_limit='{self.max_memory_mb}MB'")

            # Sort inputs by name to ensure consistent ordering
            sorted_inputs = sorted(inputs, key=lambda x: x.name)

            # Register each input as a table
            for inp in sorted_inputs:
                reader = ipc.open_stream(io.BytesIO(inp.data))
                table = reader.read_all()
                conn.register(inp.name, table)
                logs_buffer.write(
                    f"Registered {inp.name}: {table.num_rows} rows, {table.num_columns} columns\n"
                )

            # Execute query
            result = conn.execute(sql).fetch_arrow_table()
            logs_buffer.write(f"Result: {result.num_rows} rows\n")

            # Serialize to Arrow IPC stream
            output_buffer = io.BytesIO()
            with ipc.new_stream(output_buffer, result.schema) as writer:
                writer.write_table(result)

            duration_ms = (time.time() - start_time) * 1000

            return ExecutionResult(
                success=True,
                output_bytes=output_buffer.getvalue(),
                logs=logs_buffer.getvalue(),
                duration_ms=duration_ms,
                output_rows=result.num_rows,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logs_buffer.write(f"Error: {e}\n")

            return ExecutionResult(
                success=False,
                error_code=type(e).__name__,
                error_message=str(e),
                logs=logs_buffer.getvalue(),
                duration_ms=duration_ms,
            )


# ---------------------------------------------------------------------------
# FastAPI Application (Standalone Server)
# ---------------------------------------------------------------------------


def create_executor_app(executor: BaseExecutor | None = None):
    """Create FastAPI application for an executor.

    Args:
        executor: Executor instance to use. Defaults to DuckDBExecutor.

    Returns:
        FastAPI application
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import Response

    from strata.types import (
        EXECUTOR_LOGS_HEADER,
        EXECUTOR_PROTOCOL_HEADER,
        EXECUTOR_PROTOCOL_VERSION,
    )

    if executor is None:
        executor = DuckDBExecutor()

    app = FastAPI(
        title="Strata Executor",
        description="Reference executor for Strata Protocol v1",
        version="1.0.0",
    )

    @app.get("/health")
    async def health():
        """Health check endpoint returning executor capabilities."""
        return executor.health_check()

    @app.post("/v1/execute")
    async def execute(http_request: FastAPIRequest):
        """Execute a transform with the given inputs.

        Request:
            Content-Type: multipart/form-data
            X-Strata-Executor-Protocol: v1

            Parts:
                - metadata: JSON with ExecutorRequestMetadata schema
                - input0, input1, ...: Arrow IPC stream bytes

        Response:
            - 200: Arrow IPC stream (application/vnd.apache.arrow.stream)
            - 4xx/5xx: JSON error (ExecutorResponse schema)
        """
        import json

        # Parse multipart form data
        form = await http_request.form()

        # Get metadata
        metadata_file = form.get("metadata")
        if not isinstance(metadata_file, UploadFile):
            raise HTTPException(status_code=400, detail="Missing metadata")

        # Parse metadata
        try:
            metadata_bytes = await metadata_file.read()
            meta = json.loads(metadata_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid metadata: {e}")

        # Validate protocol version
        protocol_version = meta.get("protocol_version", "v1")
        if protocol_version != EXECUTOR_PROTOCOL_VERSION:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported protocol version: {protocol_version}. "
                f"Expected: {EXECUTOR_PROTOCOL_VERSION}",
            )

        # Extract transform info
        transform = meta.get("transform", {})
        transform_ref = transform.get("ref", "")
        params = transform.get("params", {})

        # Validate transform reference
        supported_refs = executor.get_transform_refs()
        if not any(transform_ref.startswith(ref.split("@")[0]) for ref in supported_refs):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported transform: {transform_ref}. Supported: {supported_refs}",
            )

        # Collect inputs from form data
        inputs: list[ExecutorInput] = []
        for name in ["input0", "input1", "input2", "input3", "input4"]:
            upload = form.get(name)
            if isinstance(upload, UploadFile):
                data = await upload.read()
                if data:
                    inputs.append(ExecutorInput(name=name, data=data))

        # Execute transform
        result = executor.execute(transform_ref, params, inputs)

        if not result.success:
            # Return error as JSON
            return Response(
                content=json.dumps(
                    {
                        "success": False,
                        "error_code": result.error_code,
                        "error_message": result.error_message,
                        "logs": result.logs,
                        "duration_ms": result.duration_ms,
                    }
                ),
                status_code=400,
                media_type="application/json",
            )

        # Build response headers
        headers = {
            EXECUTOR_PROTOCOL_HEADER: EXECUTOR_PROTOCOL_VERSION,
        }

        # Add logs as base64-encoded header if present
        if result.logs:
            headers[EXECUTOR_LOGS_HEADER] = base64.b64encode(result.logs.encode("utf-8")).decode(
                "ascii"
            )

        return Response(
            content=result.output_bytes,
            media_type="application/vnd.apache.arrow.stream",
            headers=headers,
        )

    return app


# Lazy app creation for uvicorn (only when running as main)
# Don't create at import time to avoid dependency issues in tests
def get_app():
    """Get the default executor app (lazy initialization)."""
    return create_executor_app()


# For uvicorn: use `uvicorn strata.transforms.reference_executor:get_app --factory`
# Or use the create_executor_app() function directly


# ---------------------------------------------------------------------------
# Utility Functions for Building Executors
# ---------------------------------------------------------------------------


def parse_arrow_inputs(
    file_parts: dict[str, bytes],
) -> list[ExecutorInput]:
    """Parse Arrow IPC inputs from multipart form data.

    Args:
        file_parts: Mapping of part name -> bytes

    Returns:
        List of ExecutorInput sorted by name
    """
    inputs = []
    for name, data in file_parts.items():
        if name.startswith("input") and data:
            inputs.append(ExecutorInput(name=name, data=data))

    return sorted(inputs, key=lambda x: x.name)


def serialize_arrow_output(table) -> bytes:
    """Serialize Arrow table to IPC stream bytes.

    Args:
        table: PyArrow Table

    Returns:
        Arrow IPC stream bytes
    """
    import io

    import pyarrow.ipc as ipc

    buffer = io.BytesIO()
    with ipc.new_stream(buffer, table.schema) as writer:
        writer.write_table(table)
    return buffer.getvalue()


def encode_logs_header(logs: str) -> str:
    """Encode logs for X-Strata-Logs header.

    Args:
        logs: Log text

    Returns:
        Base64-encoded string
    """
    return base64.b64encode(logs.encode("utf-8")).decode("ascii")


def decode_logs_header(header: str) -> str:
    """Decode X-Strata-Logs header.

    Args:
        header: Base64-encoded log header

    Returns:
        Decoded log text
    """
    return base64.b64decode(header).decode("utf-8")
