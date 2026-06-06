"""Shared pytest fixtures and helpers for Strata tests.

This module provides:
- Common utility functions (find_free_port, wait_for_server, etc.)
- IPC conversion helpers (table_to_ipc_bytes, ipc_bytes_to_table)
- Server context managers for running test servers
- Base fixtures (temp_warehouse, strata_config, server_with_client)
"""

import io
import socket
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

# Python 3.10 compatibility: UTC is in timezone module
try:
    from datetime import UTC
except ImportError:
    UTC = UTC

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import uvicorn
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import (
    DoubleType,
    LongType,
    NestedField,
    StringType,
)

from strata.client import StrataClient
from strata.config import StrataConfig

# =============================================================================
# Common Utility Functions
# =============================================================================


def _reset_transform_singletons() -> None:
    """Reset build store + runner globals between test servers.

    The personal-mode lifespan now starts an embedded build runner; the
    build store singleton caches its first db_path forever, so without a
    reset every later test server would create build records in the FIRST
    server's database while its own runner polls a different one — builds
    sit in pending until the wait times out.
    """
    from strata.transforms.build_store import reset_build_store
    from strata.transforms.runner import reset_build_runner

    reset_build_store()
    reset_build_runner()


def find_free_port() -> int:
    """Find an available port on localhost.

    Uses SO_REUSEADDR to avoid "address already in use" errors.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(port: int, timeout: float = 5.0) -> bool:
    """Wait for server to be ready by polling /health endpoint.

    Args:
        port: Port the server is running on
        timeout: Maximum time to wait in seconds

    Returns:
        True if server is ready, False if timeout
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


# =============================================================================
# IPC Conversion Helpers
# =============================================================================


def table_to_ipc_bytes(table: pa.Table) -> bytes:
    """Convert Arrow table to IPC stream bytes.

    Args:
        table: PyArrow Table to convert

    Returns:
        Bytes representing the Arrow IPC stream
    """
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def ipc_bytes_to_table(data: bytes) -> pa.Table:
    """Convert IPC stream bytes to Arrow table.

    Args:
        data: Arrow IPC stream bytes

    Returns:
        PyArrow Table
    """
    reader = ipc.open_stream(io.BytesIO(data))
    return reader.read_all()


# =============================================================================
# Server Context Managers
# =============================================================================


@dataclass
class ServerContext:
    """Context for a running test server.

    Attributes:
        config: StrataConfig used by the server
        port: Port the server is running on
        base_url: Base URL for HTTP requests
        server_instance: The uvicorn Server instance (if using uvicorn.Server)
        thread: The thread running the server
    """

    config: StrataConfig
    port: int
    base_url: str
    server_instance: uvicorn.Server | None = None
    thread: threading.Thread | None = None


@contextmanager
def run_server(config: StrataConfig, reset_caches: bool = False) -> Iterator[str]:
    """Run a Strata server in a background thread using uvicorn.run.

    This is the basic server context manager that yields the base URL.
    Server runs as a daemon thread and is killed on exit.

    Args:
        config: StrataConfig with host/port settings
        reset_caches: If True, reset global metadata caches before starting

    Yields:
        Base URL string (e.g., "http://127.0.0.1:8765")

    Raises:
        RuntimeError: If server fails to start within 5 seconds
    """
    import strata.server as server_module
    from strata.artifact_store import reset_artifact_store
    from strata.server import ServerState, app

    # Reset global caches if requested (for test isolation)
    if reset_caches:
        from strata.metadata_cache import reset_caches as do_reset_caches

        do_reset_caches()

    # Reset artifact store singleton for test isolation across server runs.
    reset_artifact_store()
    _reset_transform_singletons()

    # Initialize server state
    server_module._state = ServerState(config)

    # Start server in background thread
    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={
            "app": app,
            "host": config.host,
            "port": config.port,
            "log_level": "error",
        },
        daemon=True,
    )
    server_thread.start()

    # Wait for server to be ready
    base_url = f"http://{config.host}:{config.port}"
    for _ in range(50):  # 5 second timeout
        try:
            with httpx.Client() as client:
                resp = client.get(f"{base_url}/health", timeout=1.0)
                if resp.status_code == 200:
                    break
        except Exception:
            pass
        time.sleep(0.1)
    else:
        raise RuntimeError("Server failed to start")

    try:
        yield base_url
    finally:
        # Server thread is daemon, will be killed on exit
        server_module._state = None
        reset_artifact_store()
        _reset_transform_singletons()


@contextmanager
def run_server_with_context(
    cache_dir,
    artifact_dir=None,
    deployment_mode: Literal["personal", "service"] = "personal",
    **config_overrides,
) -> Iterator[ServerContext]:
    """Run a server with full context including graceful shutdown.

    This context manager provides more control than run_server():
    - Returns ServerContext with server instance for graceful shutdown
    - Supports artifact_dir configuration
    - Resets artifact store on cleanup

    Args:
        cache_dir: Path for cache directory
        artifact_dir: Optional path for artifact storage
        deployment_mode: "personal" or "service"

    Yields:
        ServerContext with server details

    Raises:
        RuntimeError: If server fails to start
    """
    from strata import server
    from strata.artifact_store import reset_artifact_store
    from strata.server import ServerState, app

    port = find_free_port()

    config = StrataConfig(
        host="127.0.0.1",
        port=port,
        cache_dir=cache_dir,
        deployment_mode=deployment_mode,
        artifact_dir=artifact_dir,
        **config_overrides,
    )
    reset_artifact_store()
    _reset_transform_singletons()
    server._state = ServerState(config)

    server_config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
    )
    server_instance = uvicorn.Server(server_config)
    thread = threading.Thread(target=server_instance.run, daemon=True)
    thread.start()

    if not wait_for_server(port):
        raise RuntimeError(f"Server failed to start on port {port}")

    try:
        yield ServerContext(
            config=config,
            port=port,
            base_url=f"http://127.0.0.1:{port}",
            server_instance=server_instance,
            thread=thread,
        )
    finally:
        server_instance.should_exit = True
        thread.join(timeout=2.0)
        server._state = None
        reset_artifact_store()
        _reset_transform_singletons()


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_warehouse(tmp_path):
    """Create a temporary warehouse with a sample Iceberg table.

    Skipped on Windows: pyiceberg's PyArrowFileIO strips ``file://``
    to a path like ``/C:/...`` which Windows pyarrow LocalFileSystem
    can't resolve. The stack is pyiceberg + pyarrow upstream; working
    around it here would mean bypassing the normal catalog code path.
    Iceberg scanning on Windows is a tier-2 target — skip the tests
    that need a real warehouse.
    """
    if sys.platform == "win32":
        pytest.skip("pyiceberg + pyarrow LocalFileSystem path handling broken on Windows")
    warehouse_path = tmp_path / "warehouse"
    warehouse_path.mkdir()

    warehouse_uri = warehouse_path.as_uri()
    catalog_db = (warehouse_path / "catalog.db").as_posix()
    catalog = SqlCatalog(
        "strata",
        **{
            "uri": f"sqlite:///{catalog_db}",
            "warehouse": warehouse_uri,
        },
    )

    # Create namespace
    catalog.create_namespace("test_db")

    # Define schema - use optional fields to match PyArrow defaults
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
        NestedField(3, "name", StringType(), required=False),
        NestedField(4, "timestamp", LongType(), required=False),  # Epoch micros
    )

    # Create table
    table = catalog.create_table("test_db.events", schema)

    # Create sample data with multiple row groups
    num_rows = 500
    base_ts = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000)
    data = pa.table(
        {
            "id": pa.array(range(num_rows), type=pa.int64()),
            "value": pa.array([float(i * 1.5) for i in range(num_rows)], type=pa.float64()),
            "name": pa.array([f"item_{i}" for i in range(num_rows)], type=pa.string()),
            "timestamp": pa.array(
                [base_ts + i * 3600_000_000 for i in range(num_rows)],  # micros
                type=pa.int64(),
            ),
        }
    )

    # Append data to table
    table.append(data)

    return {
        "warehouse_path": warehouse_path,
        "table_uri": f"{warehouse_uri}#test_db.events",
        "catalog": catalog,
        "table": table,
    }


@pytest.fixture
def strata_config(tmp_path):
    """Create a test configuration."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return StrataConfig(cache_dir=cache_dir)


@pytest.fixture
def server_with_client(temp_warehouse, tmp_path):
    """Start a server and provide a client.

    Yields a dict with:
        - client: StrataClient connected to the running server
        - config: StrataConfig used by the server
        - warehouse: temp_warehouse dict with table_uri, catalog, etc.
    """
    import strata.server as server_module
    from strata.server import ServerState, app

    port = find_free_port()

    config = StrataConfig(
        host="127.0.0.1",
        port=port,
        cache_dir=tmp_path / "cache",
        deployment_mode="personal",
    )

    # Initialize state manually for testing
    server_module._state = ServerState(config)

    server_thread = threading.Thread(
        target=uvicorn.run,
        kwargs={
            "app": app,
            "host": config.host,
            "port": config.port,
            "log_level": "error",
        },
        daemon=True,
    )
    server_thread.start()

    # Wait for server to start
    if not wait_for_server(port):
        raise RuntimeError("Server failed to start")

    client = StrataClient(base_url=f"http://127.0.0.1:{port}")

    yield {
        "client": client,
        "config": config,
        "warehouse": temp_warehouse,
    }

    client.close()
