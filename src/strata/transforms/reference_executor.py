"""Reference executor for Strata executor protocol v1, a template for external executors.

Serves ``POST /v1/execute`` (multipart: ``metadata`` JSON plus ``input0``...) and
``GET /health``. Run with ``uvicorn strata.transforms.reference_executor:get_app
--factory --port 8080``. Protocol types live in ``strata.types`` (``Executor*``).
"""

from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

# Import at module level to avoid annotation resolution issues with FastAPI
from fastapi import Request as FastAPIRequest
from starlette.datastructures import UploadFile

logger = logging.getLogger(__name__)


# --- Executor interface ---


@dataclass
class ExecutionResult:
    """Outcome of one execution: Arrow IPC ``output_bytes`` on success, error fields otherwise."""

    success: bool
    output_bytes: bytes | None = None
    error_code: str | None = None
    error_message: str | None = None
    logs: str | None = None
    duration_ms: float | None = None
    output_rows: int | None = None


@dataclass
class ExecutorInput:
    """One named input (``input0``, ...) as Arrow IPC stream bytes."""

    name: str
    data: bytes


class BaseExecutor(ABC):
    """Base class for executors: implement ``get_transform_refs`` and ``execute``."""

    @abstractmethod
    def get_transform_refs(self) -> list[str]:
        """Return the transform refs this executor supports, e.g. ``["duckdb_sql@v1"]``."""
        pass

    @abstractmethod
    def execute(
        self,
        transform_ref: str,
        params: dict[str, Any],
        inputs: list[ExecutorInput],
    ) -> ExecutionResult:
        """Run ``transform_ref`` with ``params`` over ``inputs``; report failure in the result."""
        pass

    def health_check(self) -> dict:
        """Return health status and capabilities; override to add checks."""
        from strata.types import EXECUTOR_PROTOCOL_VERSION

        return {
            "status": "healthy",
            "capabilities": {
                "protocol_versions": [EXECUTOR_PROTOCOL_VERSION],
                "transform_refs": self.get_transform_refs(),
            },
        }


# --- DuckDB SQL executor (reference implementation) ---


class DuckDBExecutor(BaseExecutor):
    """Reference executor running DuckDB SQL over inputs registered as ``input0``, ``input1``."""

    def __init__(self, max_memory_mb: int = 1024):
        """Create the executor with a DuckDB memory limit in megabytes."""
        self.max_memory_mb = max_memory_mb

    def get_transform_refs(self) -> list[str]:
        return ["duckdb_sql@v1"]

    def execute(
        self,
        transform_ref: str,
        params: dict[str, Any],
        inputs: list[ExecutorInput],
    ) -> ExecutionResult:
        """Run ``params["sql"]`` in a fresh in-memory DuckDB; errors become a failed result."""
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
            sql = params.get("sql")
            if not sql:
                return ExecutionResult(
                    success=False,
                    error_code="INVALID_PARAMS",
                    error_message="Missing 'sql' in params",
                )

            logs_buffer.write(f"Executing SQL: {sql[:100]}...\n")

            conn = duckdb.connect(":memory:")
            conn.execute(f"SET memory_limit='{self.max_memory_mb}MB'")

            # Sort by name for a stable input order.
            sorted_inputs = sorted(inputs, key=lambda x: x.name)

            for inp in sorted_inputs:
                reader = ipc.open_stream(io.BytesIO(inp.data))
                table = reader.read_all()
                conn.register(inp.name, table)
                logs_buffer.write(
                    f"Registered {inp.name}: {table.num_rows} rows, {table.num_columns} columns\n"
                )

            result = conn.execute(sql).to_arrow_table()
            logs_buffer.write(f"Result: {result.num_rows} rows\n")

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


# --- FastAPI application (standalone server) ---


def create_executor_app(executor: BaseExecutor | None = None):
    """Create the FastAPI app serving ``executor`` (a DuckDBExecutor by default)."""
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
        """Execute a transform from a multipart ``metadata`` part and ``input0``..``input4``.

        Returns an Arrow IPC stream, or a 400 JSON error (``ExecutorResponse`` shape).
        """
        import json

        form = await http_request.form()

        metadata_file = form.get("metadata")
        if not isinstance(metadata_file, UploadFile):
            raise HTTPException(status_code=400, detail="Missing metadata")

        try:
            metadata_bytes = await metadata_file.read()
            meta = json.loads(metadata_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid metadata: {e}")

        protocol_version = meta.get("protocol_version", "v1")
        if protocol_version != EXECUTOR_PROTOCOL_VERSION:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported protocol version: {protocol_version}. "
                f"Expected: {EXECUTOR_PROTOCOL_VERSION}",
            )

        transform = meta.get("transform", {})
        transform_ref = transform.get("ref", "")
        params = transform.get("params", {})

        supported_refs = executor.get_transform_refs()
        if not any(transform_ref.startswith(ref.split("@")[0]) for ref in supported_refs):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported transform: {transform_ref}. Supported: {supported_refs}",
            )

        inputs: list[ExecutorInput] = []
        for name in ["input0", "input1", "input2", "input3", "input4"]:
            upload = form.get(name)
            if isinstance(upload, UploadFile):
                data = await upload.read()
                if data:
                    inputs.append(ExecutorInput(name=name, data=data))

        result = executor.execute(transform_ref, params, inputs)

        if not result.success:
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

        headers = {
            EXECUTOR_PROTOCOL_HEADER: EXECUTOR_PROTOCOL_VERSION,
        }

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


# Lazy so that importing this module (e.g. in tests) does not build the app.
def get_app():
    """Build the default executor app (uvicorn ``--factory`` entry point)."""
    return create_executor_app()


# For uvicorn: `uvicorn strata.transforms.reference_executor:get_app --factory`


# --- Utilities for building executors ---


def parse_arrow_inputs(
    file_parts: dict[str, bytes],
) -> list[ExecutorInput]:
    """Collect non-empty ``input*`` parts as ExecutorInputs, sorted by name."""
    inputs = []
    for name, data in file_parts.items():
        if name.startswith("input") and data:
            inputs.append(ExecutorInput(name=name, data=data))

    return sorted(inputs, key=lambda x: x.name)


def serialize_arrow_output(table) -> bytes:
    """Serialize a PyArrow table to IPC stream bytes."""
    import io

    import pyarrow.ipc as ipc

    buffer = io.BytesIO()
    with ipc.new_stream(buffer, table.schema) as writer:
        writer.write_table(table)
    return buffer.getvalue()


def encode_logs_header(logs: str) -> str:
    """Base64-encode log text for the logs header."""
    return base64.b64encode(logs.encode("utf-8")).decode("ascii")


def decode_logs_header(header: str) -> str:
    """Decode a base64 logs header back to text."""
    return base64.b64decode(header).decode("utf-8")
