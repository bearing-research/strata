#!/usr/bin/env python3
"""Production workload simulator for Strata.

This benchmark simulates a realistic multi-user workload against a local Strata server:
- Multiple concurrent users with async httpx
- Zipf-ish table access distribution (hot/warm/cold)
- Dashboard (80%) vs analyst (20%) query types
- Phases: cold → warm → churn → restart → disconnect

Usage:
    # Quick test (5 requests)
    python benchmarks/prod_sim_local.py --dry-run

    # Full benchmark with default settings
    python benchmarks/prod_sim_local.py

    # Custom configuration
    python benchmarks/prod_sim_local.py --users 20 --duration 30 --start-server

    # Specific phases only
    python benchmarks/prod_sim_local.py --phases cold,warm

    # Help
    python benchmarks/prod_sim_local.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa

# =============================================================================
# Configuration
# =============================================================================


class Phase(Enum):
    """Benchmark phases."""

    COLD = "cold"
    WARM = "warm"
    CHURN = "churn"
    RESTART = "restart"
    DISCONNECT = "disconnect"


class QueryType(Enum):
    """Query type classification."""

    DASHBOARD = "dashboard"  # Narrow projection, selective filters
    ANALYST = "analyst"  # Wide projection, less selective


class RequestStatus(Enum):
    """Request outcome status."""

    SUCCESS = "success"
    ABORT_TIMEOUT = "abort_timeout"
    ABORT_SIZE = "abort_size"
    DISCONNECT = "disconnect"
    HTTP_ERROR = "http_error"
    CLIENT_ERROR = "client_error"


@dataclass
class BenchmarkConfig:
    """Configuration for the production workload simulator."""

    # Server settings
    base_url: str = "http://127.0.0.1:8765"
    start_server: bool = True  # Start server as subprocess
    server_host: str = "127.0.0.1"
    server_port: int = 8765

    # Warehouse settings
    warehouse_dir: Path | None = None  # Auto-create if None
    cache_dir: Path | None = None  # Auto-create if None
    keep_dirs: bool = False  # Keep tmp dirs after run

    # Data generation settings
    num_tables: int = 10  # Total tables to create
    rows_hot: int = 1_000_000  # Rows per hot table (1)
    rows_warm: int = 500_000  # Rows per warm table (2)
    rows_cold: int = 100_000  # Rows per cold table (7)
    row_groups_per_file: int = 10  # Row groups per data file
    files_per_table: int = 5  # Data files per table
    payload_bytes: int = 100  # Size of payload string per row

    # Workload settings
    users: int = 10  # Concurrent users
    duration_s: float = 60.0  # Duration per phase
    phases: list[Phase] = field(
        default_factory=lambda: [
            Phase.COLD,
            Phase.WARM,
            Phase.CHURN,
            Phase.RESTART,
            Phase.DISCONNECT,
        ]
    )

    # Table access distribution (Zipf-ish)
    hot_table_weight: float = 0.6  # 1 table gets 60%
    warm_tables_weight: float = 0.3  # 2 tables get 30% total
    cold_tables_weight: float = 0.1  # 7 tables get 10% total

    # Query type distribution
    dashboard_ratio: float = 0.8  # 80% dashboard queries
    latest_snapshot_ratio: float = 0.1  # 10% use latest snapshot

    # Phase-specific settings
    churn_cache_size_bytes: int = 50 * 1024 * 1024  # 50MB (force eviction)
    disconnect_ratio: float = 0.1  # 10% disconnect mid-stream

    # Metrics collection
    metrics_interval_s: float = 5.0  # Sample /metrics every 5s
    results_dir: Path = field(default_factory=lambda: Path("benchmarks/results"))

    # Misc
    seed: int = 42
    request_timeout_s: float = 10.0  # Overall request timeout (shorter to avoid blocking)
    connect_timeout_s: float = 2.0  # Connection timeout
    max_connections: int = 100

    # Dry run mode
    dry_run: bool = False  # Run only 5 requests

    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data Structures for Metrics
# =============================================================================


@dataclass
class RequestResult:
    """Result from a single request with detailed timing breakdown.

    Timing breakdown:
    - latency_planning_ms: Time for POST /scan (planning phase)
    - latency_ttfb_ms: Time from stream start to first byte received
    - latency_streaming_ms: Time to read all remaining bytes after first byte
    - latency_total_ms: Total end-to-end time

    This breakdown helps identify bottlenecks:
    - High planning = catalog/manifest overhead
    - High TTFB = cache miss / first row group fetch
    - High streaming = large response / slow throughput
    """

    request_id: str
    phase: str
    user_id: int
    scan_id: str | None
    table_name: str
    query_type: str
    status: str
    bytes_read: int
    latency_total_ms: float
    timestamp: float
    # Detailed timing breakdown
    latency_planning_ms: float = 0.0  # POST /scan time
    latency_ttfb_ms: float = 0.0  # Time to first byte of stream
    latency_streaming_ms: float = 0.0  # Time to read all bytes after first
    # Additional metadata from planning
    num_tasks: int = 0  # Row groups to read
    estimated_bytes: int = 0  # Estimated response size
    planning_time_server_ms: float = 0.0  # Server-reported planning time
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "phase": self.phase,
            "user_id": self.user_id,
            "scan_id": self.scan_id,
            "table_name": self.table_name,
            "query_type": self.query_type,
            "status": self.status,
            "bytes_read": self.bytes_read,
            "latency_total_ms": self.latency_total_ms,
            "latency_planning_ms": self.latency_planning_ms,
            "latency_ttfb_ms": self.latency_ttfb_ms,
            "latency_streaming_ms": self.latency_streaming_ms,
            "num_tasks": self.num_tasks,
            "estimated_bytes": self.estimated_bytes,
            "planning_time_server_ms": self.planning_time_server_ms,
            "timestamp": self.timestamp,
            "error": self.error,
        }


@dataclass
class MetricsSample:
    """Sampled metrics from server."""

    timestamp: float
    phase: str
    cache_hits: int
    cache_misses: int
    bytes_from_cache: int
    bytes_from_storage: int
    scan_count: int
    active_scans: int

    def to_dict(self) -> dict:
        return {
            "type": "metrics_sample",
            "timestamp": self.timestamp,
            "phase": self.phase,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "bytes_from_cache": self.bytes_from_cache,
            "bytes_from_storage": self.bytes_from_storage,
            "scan_count": self.scan_count,
            "active_scans": self.active_scans,
        }


@dataclass
class PhaseStats:
    """Aggregated statistics for a phase."""

    phase: str
    requests: int = 0
    success: int = 0
    aborts_timeout: int = 0
    aborts_size: int = 0
    disconnects: int = 0
    http_errors: int = 0
    client_errors: int = 0
    total_bytes: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    # Detailed latency breakdown
    latencies_planning_ms: list[float] = field(default_factory=list)
    latencies_ttfb_ms: list[float] = field(default_factory=list)
    latencies_streaming_ms: list[float] = field(default_factory=list)
    duration_s: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


# =============================================================================
# Dataset Generator
# =============================================================================


def generate_warehouse(config: BenchmarkConfig) -> dict[str, Any]:
    """Generate a local Iceberg warehouse with multiple tables.

    Creates tables with different "temperatures":
    - 1 hot table (60% of traffic)
    - 2 warm tables (30% of traffic)
    - 7 cold tables (10% of traffic)

    Returns:
        Dict with table metadata and URIs.
    """
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import (
        DoubleType,
        IntegerType,
        LongType,
        NestedField,
        StringType,
    )

    warehouse_path = config.warehouse_dir
    warehouse_path.mkdir(parents=True, exist_ok=True)

    # Create SQL catalog
    catalog = SqlCatalog(
        "strata",
        **{
            "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
            "warehouse": str(warehouse_path),
        },
    )

    # Create namespace
    try:
        catalog.create_namespace("benchmark")
    except Exception:
        pass

    # Schema for all tables
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "ts", LongType(), required=False),  # Epoch micros
        NestedField(3, "user_id", IntegerType(), required=False),
        NestedField(4, "category", StringType(), required=False),
        NestedField(5, "value", DoubleType(), required=False),
        NestedField(6, "payload", StringType(), required=False),
    )

    tables_info = []
    categories = ["electronics", "clothing", "food", "books", "sports", "home", "auto"]

    random.seed(config.seed)

    for i in range(config.num_tables):
        table_name = f"table_{i:02d}"
        table_id = f"benchmark.{table_name}"

        # Determine table temperature
        if i == 0:
            temperature = "hot"
            num_rows = config.rows_hot
        elif i < 3:
            temperature = "warm"
            num_rows = config.rows_warm
        else:
            temperature = "cold"
            num_rows = config.rows_cold

        print(f"  Creating {table_name} ({temperature}, {num_rows:,} rows)...")

        try:
            table = catalog.load_table(table_id)
        except Exception:
            table = catalog.create_table(table_id, schema)

            # Write data in chunks to create multiple files and row groups
            rows_per_file = num_rows // config.files_per_table
            rows_per_chunk = rows_per_file // config.row_groups_per_file

            # Use minimum chunk size to avoid too many tiny files
            rows_per_chunk = max(rows_per_chunk, 1000)

            base_ts = 1704067200000000  # 2024-01-01 00:00:00 UTC in micros
            row_offset = 0

            for file_idx in range(config.files_per_table):
                # Create chunks for this file
                file_rows = (
                    rows_per_file
                    if file_idx < config.files_per_table - 1
                    else num_rows - row_offset
                )

                for chunk_start in range(0, file_rows, rows_per_chunk):
                    chunk_size = min(rows_per_chunk, file_rows - chunk_start)
                    if chunk_size <= 0:
                        break

                    start_id = row_offset + chunk_start

                    # Generate data
                    data = pa.table(
                        {
                            "id": pa.array(range(start_id, start_id + chunk_size), type=pa.int64()),
                            "ts": pa.array(
                                [base_ts + (start_id + j) * 1000000 for j in range(chunk_size)],
                                type=pa.int64(),
                            ),
                            "user_id": pa.array(
                                [random.randint(1, 10000) for _ in range(chunk_size)],
                                type=pa.int32(),
                            ),
                            "category": pa.array(
                                [
                                    categories[random.randint(0, len(categories) - 1)]
                                    for _ in range(chunk_size)
                                ],
                                type=pa.string(),
                            ),
                            "value": pa.array(
                                [random.uniform(0.0, 1000.0) for _ in range(chunk_size)],
                                type=pa.float64(),
                            ),
                            "payload": pa.array(
                                [
                                    f"data_{start_id + j:08d}_" + "x" * config.payload_bytes
                                    for j in range(chunk_size)
                                ],
                                type=pa.string(),
                            ),
                        }
                    )
                    table.append(data)

                row_offset += file_rows

        # Get snapshot ID
        snapshot_id = table.current_snapshot().snapshot_id

        tables_info.append(
            {
                "name": table_name,
                "table_id": table_id,
                "table_uri": f"file://{warehouse_path}#benchmark.{table_name}",
                "temperature": temperature,
                "num_rows": num_rows,
                "snapshot_id": snapshot_id,
            }
        )

    return {
        "catalog": catalog,
        "warehouse_path": warehouse_path,
        "tables": tables_info,
    }


# =============================================================================
# Server Management
# =============================================================================


class ServerProcess:
    """Manages a Strata server as a subprocess."""

    def __init__(
        self,
        host: str,
        port: int,
        cache_dir: Path,
        max_cache_size_bytes: int | None = None,
    ):
        self.host = host
        self.port = port
        self.cache_dir = cache_dir
        self.max_cache_size_bytes = max_cache_size_bytes
        self._process: subprocess.Popen | None = None

    def start(self, timeout: float = 30.0):
        """Start the server as a subprocess."""
        env = os.environ.copy()
        env["STRATA_HOST"] = self.host
        env["STRATA_PORT"] = str(self.port)
        env["STRATA_CACHE_DIR"] = str(self.cache_dir)
        env["STRATA_METRICS_ENABLED"] = "true"

        if self.max_cache_size_bytes is not None:
            env["STRATA_MAX_CACHE_SIZE_BYTES"] = str(self.max_cache_size_bytes)

        self._process = subprocess.Popen(
            [sys.executable, "-m", "strata.server"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self._wait_for_ready(timeout)

    def _wait_for_ready(self, timeout: float = 30.0):
        """Wait for the server to respond to health checks."""
        start = time.perf_counter()
        url = f"http://{self.host}:{self.port}/health"

        while time.perf_counter() - start < timeout:
            try:
                response = httpx.get(url, timeout=1.0)
                if response.status_code == 200:
                    return
            except Exception:
                pass
            time.sleep(0.1)

        raise TimeoutError(f"Server did not start within {timeout}s")

    def stop(self):
        """Stop the server subprocess."""
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

    def clear_cache(self):
        """Clear the data cache directory, preserving metadata store.

        NOTE: We only clear the versioned cache subdirectories (v1/, v2/, etc.),
        not the metadata.sqlite file. This preserves manifest and parquet
        metadata while forcing data cache misses.
        """
        if not self.cache_dir.exists():
            return

        # Only remove versioned cache subdirectories (v1, v2, etc.)
        # This preserves the metadata.sqlite file
        for entry in self.cache_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("v"):
                shutil.rmtree(entry)
            elif entry.suffix == ".arrowstream" or entry.suffix == ".meta.json":
                entry.unlink()


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# =============================================================================
# Workload Generator
# =============================================================================


class WorkloadGenerator:
    """Generates realistic workload patterns."""

    def __init__(self, config: BenchmarkConfig, tables_info: list[dict]):
        self.config = config
        self.tables_info = tables_info
        self.rng = random.Random(config.seed)

        # Build weighted table selection
        self._build_table_weights()

        # Column groups for different query types
        self.dashboard_columns = ["id", "ts", "value"]  # Narrow
        self.analyst_columns = ["id", "ts", "user_id", "category", "value", "payload"]  # Wide

        # Categories for filters
        self.categories = ["electronics", "clothing", "food", "books", "sports", "home", "auto"]

    def _build_table_weights(self):
        """Build weighted table selection based on temperature."""
        self.table_weights = []

        for t in self.tables_info:
            if t["temperature"] == "hot":
                weight = self.config.hot_table_weight
            elif t["temperature"] == "warm":
                weight = self.config.warm_tables_weight / 2  # Split among 2 warm
            else:
                weight = self.config.cold_tables_weight / 7  # Split among 7 cold
            self.table_weights.append(weight)

        # Normalize
        total = sum(self.table_weights)
        self.table_weights = [w / total for w in self.table_weights]

    def choose_table(self) -> dict:
        """Choose a table based on weighted distribution."""
        return self.rng.choices(self.tables_info, weights=self.table_weights, k=1)[0]

    def choose_query_type(self) -> QueryType:
        """Choose dashboard (80%) or analyst (20%) query type."""
        return (
            QueryType.DASHBOARD
            if self.rng.random() < self.config.dashboard_ratio
            else QueryType.ANALYST
        )

    def choose_snapshot(self, table_info: dict) -> int | None:
        """Choose snapshot ID (90% pinned, 10% latest)."""
        if self.rng.random() < self.config.latest_snapshot_ratio:
            return None  # Latest
        return table_info["snapshot_id"]

    def generate_query(self) -> dict[str, Any]:
        """Generate a complete query specification."""
        table_info = self.choose_table()
        query_type = self.choose_query_type()
        snapshot_id = self.choose_snapshot(table_info)

        if query_type == QueryType.DASHBOARD:
            columns = self.dashboard_columns
            # Selective filter: specific category
            # Use FilterOp enum values: "=" not "=="
            filters = [{"column": "category", "op": "=", "value": self.rng.choice(self.categories)}]
        else:
            columns = self.analyst_columns
            # Less selective: value range
            # Use FilterOp enum values: ">=" and "<="
            min_val = self.rng.uniform(0, 500)
            max_val = min_val + self.rng.uniform(100, 500)
            filters = [
                {"column": "value", "op": ">=", "value": min_val},
                {"column": "value", "op": "<=", "value": max_val},
            ]

        return {
            "table_uri": table_info["table_uri"],
            "table_name": table_info["name"],
            "snapshot_id": snapshot_id,
            "columns": columns,
            "filters": filters,
            "query_type": query_type.value,
        }


# =============================================================================
# Async Load Driver
# =============================================================================


class AsyncLoadDriver:
    """Async HTTP client for driving load against Strata."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._client: httpx.AsyncClient | None = None
        self._request_counter = 0
        self._lock = asyncio.Lock()

    async def start(self):
        """Start the HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            limits=httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_connections // 2,
            ),
            timeout=httpx.Timeout(
                connect=self.config.connect_timeout_s,
                read=self.config.request_timeout_s,
                write=self.config.request_timeout_s,
                pool=self.config.connect_timeout_s,
            ),
        )

    async def stop(self):
        """Stop the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _get_request_id(self) -> str:
        async with self._lock:
            self._request_counter += 1
            return f"req_{self._request_counter:06d}"

    async def execute_scan(
        self,
        query: dict[str, Any],
        phase: str,
        user_id: int,
        disconnect_early: bool = False,
    ) -> RequestResult:
        """Execute a single scan request with detailed timing breakdown.

        Captures three timing phases:
        1. Planning: POST /scan - catalog lookup, manifest parsing, metadata fetch
        2. TTFB: Time from GET /batches to first byte - first row group fetch
        3. Streaming: Time to read remaining bytes - sustained throughput
        """
        request_id = await self._get_request_id()
        start_time = time.perf_counter()
        timestamp = time.time()
        scan_id = None
        bytes_read = 0
        status = RequestStatus.SUCCESS
        error = None

        # Detailed timing
        planning_end_time = None
        ttfb_time = None
        streaming_start_time = None
        num_tasks = 0
        estimated_bytes = 0
        planning_time_server_ms = 0.0

        try:
            # Phase 1: POST /v1/materialize (scan@v1 planning)
            request_body = {
                "inputs": [query["table_uri"]],
                "transform": {
                    "executor": "scan@v1",
                    "params": {
                        "snapshot_id": query["snapshot_id"],
                        "columns": query["columns"],
                        "filters": query["filters"],
                    },
                },
                "mode": "stream",
            }

            response = await self._client.post("/v1/materialize", json=request_body)
            planning_end_time = time.perf_counter()
            response.raise_for_status()
            scan_info = response.json()
            stream_url = scan_info["stream_url"]
            # num_tasks / estimated_bytes / server planning time aren't in the
            # materialize response; they stay at their 0 defaults.

            # Phase 2 & 3: GET the stream (TTFB + streaming)
            async with self._client.stream("GET", stream_url) as stream:
                stream.raise_for_status()

                first_chunk = True
                if disconnect_early:
                    # Read a small portion then disconnect (64KB chunks)
                    async for chunk in stream.aiter_bytes(chunk_size=65536):
                        if first_chunk:
                            ttfb_time = time.perf_counter()
                            streaming_start_time = ttfb_time
                            first_chunk = False
                        bytes_read += len(chunk)
                        if bytes_read > 10000:  # Disconnect after 10KB
                            status = RequestStatus.DISCONNECT
                            break
                else:
                    # Read all data (1MB chunks to measure server throughput, not client)
                    async for chunk in stream.aiter_bytes(chunk_size=1024 * 1024):
                        if first_chunk:
                            ttfb_time = time.perf_counter()
                            streaming_start_time = ttfb_time
                            first_chunk = False
                        bytes_read += len(chunk)

        except httpx.TimeoutException:
            status = RequestStatus.ABORT_TIMEOUT
            error = "Request timed out"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 413:
                status = RequestStatus.ABORT_SIZE
                error = "Response too large"
            else:
                status = RequestStatus.HTTP_ERROR
                # Include response body for debugging
                try:
                    detail = e.response.json().get("detail", "")[:200]
                    error = f"HTTP {e.response.status_code}: {detail}"
                except Exception:
                    error = f"HTTP {e.response.status_code}"
        except Exception as e:
            status = RequestStatus.CLIENT_ERROR
            error = str(e)
        # No explicit cleanup: materialize streams are freed server-side by the
        # stream-state TTL once consumed (the old /v1/scan DELETE is gone).
        end_time = time.perf_counter()

        # Calculate detailed latencies
        latency_total_ms = (end_time - start_time) * 1000
        latency_planning_ms = (
            (planning_end_time - start_time) * 1000 if planning_end_time else latency_total_ms
        )
        latency_ttfb_ms = (
            (ttfb_time - planning_end_time) * 1000 if ttfb_time and planning_end_time else 0.0
        )
        latency_streaming_ms = (
            (end_time - streaming_start_time) * 1000 if streaming_start_time else 0.0
        )

        return RequestResult(
            request_id=request_id,
            phase=phase,
            user_id=user_id,
            scan_id=scan_id,
            table_name=query["table_name"],
            query_type=query["query_type"],
            status=status.value,
            bytes_read=bytes_read,
            latency_total_ms=latency_total_ms,
            latency_planning_ms=latency_planning_ms,
            latency_ttfb_ms=latency_ttfb_ms,
            latency_streaming_ms=latency_streaming_ms,
            num_tasks=num_tasks,
            estimated_bytes=estimated_bytes,
            planning_time_server_ms=planning_time_server_ms,
            timestamp=timestamp,
            error=error,
        )

    async def get_metrics(self) -> dict[str, Any]:
        """Get server metrics."""
        try:
            response = await self._client.get("/metrics")
            response.raise_for_status()
            return response.json()
        except Exception:
            return {}

    async def clear_cache(self) -> bool:
        """Clear server cache."""
        try:
            response = await self._client.post("/v1/cache/clear")
            return response.status_code == 200
        except Exception:
            return False

    async def health_check(self) -> bool:
        """Check server health."""
        try:
            response = await self._client.get("/health")
            return response.status_code == 200
        except Exception:
            return False


# =============================================================================
# Phase Execution
# =============================================================================


class PhaseExecutor:
    """Executes benchmark phases."""

    def __init__(
        self,
        config: BenchmarkConfig,
        driver: AsyncLoadDriver,
        workload: WorkloadGenerator,
        server: ServerProcess | None = None,
    ):
        self.config = config
        self.driver = driver
        self.workload = workload
        self.server = server
        self.results: list[RequestResult] = []
        self.metrics_samples: list[MetricsSample] = []
        self._stop_event = asyncio.Event()

    async def _user_loop(
        self,
        user_id: int,
        phase: str,
        duration_s: float,
        disconnect_early: bool = False,
    ) -> list[RequestResult]:
        """User loop that generates requests until duration expires."""
        results = []
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration_s and not self._stop_event.is_set():
            query = self.workload.generate_query()

            # In disconnect phase, randomly disconnect early
            do_disconnect = disconnect_early and random.random() < self.config.disconnect_ratio

            result = await self.driver.execute_scan(
                query=query,
                phase=phase,
                user_id=user_id,
                disconnect_early=do_disconnect,
            )
            results.append(result)

            # Small delay to avoid hammering
            await asyncio.sleep(0.001)

        return results

    async def _metrics_loop(self, phase: str, duration_s: float):
        """Periodically sample server metrics."""
        start_time = time.perf_counter()

        while time.perf_counter() - start_time < duration_s and not self._stop_event.is_set():
            await asyncio.sleep(self.config.metrics_interval_s)

            metrics = await self.driver.get_metrics()
            if metrics:
                sample = MetricsSample(
                    timestamp=time.time(),
                    phase=phase,
                    cache_hits=metrics.get("cache_hits", 0),
                    cache_misses=metrics.get("cache_misses", 0),
                    bytes_from_cache=metrics.get("bytes_from_cache", 0),
                    bytes_from_storage=metrics.get("bytes_from_storage", 0),
                    scan_count=metrics.get("scan_count", 0),
                    active_scans=metrics.get("resource_limits", {}).get("active_scans", 0),
                )
                self.metrics_samples.append(sample)

    async def run_phase(
        self,
        phase: Phase,
        duration_s: float | None = None,
    ) -> PhaseStats:
        """Run a single benchmark phase."""
        phase_name = phase.value
        duration = duration_s or self.config.duration_s
        disconnect_early = phase == Phase.DISCONNECT

        # Special handling for different phases
        if phase == Phase.COLD:
            # Clear cache via HTTP API - this only clears data cache, not metadata
            print("    Clearing cache for cold start...")
            cleared = await self.driver.clear_cache()
            if not cleared:
                print("    WARNING: Cache clear failed")

        elif phase == Phase.WARM:
            # Restart server to ensure clean state (workaround for server bug with timeouts)
            if self.server:
                print("    Restarting server for clean warm phase...")
                self.server.stop()
                time.sleep(1)
                self.server.start()
                # Recreate client connection
                await self.driver.stop()
                await self.driver.start()
            else:
                # No server control, just reset client
                print("    Resetting HTTP client...")
                await self.driver.stop()
                await self.driver.start()
            # Verify server is healthy
            if not await self.driver.health_check():
                print("    WARNING: Server health check failed!")
            else:
                print("    Server health check passed")

        elif phase == Phase.CHURN:
            # Restart server with reduced cache size to force eviction
            if self.server:
                print("    Restarting server with reduced cache size...")
                self.server.stop()
                time.sleep(1)
                # Use reduced cache size
                self.server.max_cache_size_bytes = self.config.churn_cache_size_bytes
                self.server.start()
                await self.driver.stop()
                await self.driver.start()
            else:
                print("    Note: Cannot reduce cache size without server control")

        elif phase == Phase.RESTART:
            print("    Restarting server...")
            if self.server:
                self.server.stop()
                time.sleep(1)
                self.server.start()
                await self.driver.stop()
                await self.driver.start()

        elif phase == Phase.DISCONNECT:
            # Restart server first (workaround for server bug with timeouts)
            if self.server:
                print("    Restarting server for disconnect test...")
                self.server.stop()
                time.sleep(1)
                self.server.start()
                await self.driver.stop()
                await self.driver.start()

        # Get initial metrics
        initial_metrics = await self.driver.get_metrics()

        print(f"    Running {self.config.users} users for {duration:.1f}s...")

        start_time = time.perf_counter()
        self._stop_event.clear()

        # Launch user tasks and metrics task
        user_tasks = [
            asyncio.create_task(self._user_loop(user_id, phase_name, duration, disconnect_early))
            for user_id in range(self.config.users)
        ]
        metrics_task = asyncio.create_task(self._metrics_loop(phase_name, duration))

        # Wait for all users to complete
        all_results = await asyncio.gather(*user_tasks)

        # Stop metrics collection
        self._stop_event.set()
        await metrics_task

        actual_duration = time.perf_counter() - start_time

        # Flatten results
        phase_results = [r for user_results in all_results for r in user_results]
        self.results.extend(phase_results)

        # Get final metrics
        final_metrics = await self.driver.get_metrics()

        # Compute stats
        stats = PhaseStats(phase=phase_name, duration_s=actual_duration)
        stats.requests = len(phase_results)

        for r in phase_results:
            stats.latencies_ms.append(r.latency_total_ms)
            stats.latencies_planning_ms.append(r.latency_planning_ms)
            stats.latencies_ttfb_ms.append(r.latency_ttfb_ms)
            stats.latencies_streaming_ms.append(r.latency_streaming_ms)
            stats.total_bytes += r.bytes_read

            if r.status == RequestStatus.SUCCESS.value:
                stats.success += 1
            elif r.status == RequestStatus.ABORT_TIMEOUT.value:
                stats.aborts_timeout += 1
            elif r.status == RequestStatus.ABORT_SIZE.value:
                stats.aborts_size += 1
            elif r.status == RequestStatus.DISCONNECT.value:
                stats.disconnects += 1
            elif r.status == RequestStatus.HTTP_ERROR.value:
                stats.http_errors += 1
            else:
                stats.client_errors += 1

        # Cache stats delta
        if initial_metrics and final_metrics:
            stats.cache_hits = final_metrics.get("cache_hits", 0) - initial_metrics.get(
                "cache_hits", 0
            )
            stats.cache_misses = final_metrics.get("cache_misses", 0) - initial_metrics.get(
                "cache_misses", 0
            )

        return stats


# =============================================================================
# Reporting
# =============================================================================


def compute_percentiles(values: list[float], percentiles: list[float]) -> dict[str, float]:
    """Compute percentiles from a list of values."""
    if not values:
        return {f"p{int(p * 100)}": 0.0 for p in percentiles}

    sorted_values = sorted(values)
    n = len(sorted_values)

    result = {}
    for p in percentiles:
        idx = int(p * n)
        idx = min(idx, n - 1)
        result[f"p{int(p * 100)}"] = sorted_values[idx]

    return result


def format_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024**3):.2f} GB"
    elif n >= 1024 * 1024:
        return f"{n / (1024**2):.2f} MB"
    elif n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


def print_phase_summary(phase_stats: list[PhaseStats]):
    """Print a summary table for all phases."""
    print("\n" + "=" * 120)
    print("PHASE SUMMARY")
    print("=" * 120)

    # Header - overall latency
    print(
        f"\n{'Phase':<12} {'Requests':>10} {'Success':>10} {'Errors':>10} "
        f"{'p50(ms)':>10} {'p95(ms)':>10} {'p99(ms)':>10} {'MB/s':>10} {'Hit Rate':>10}"
    )
    print("-" * 120)

    for stats in phase_stats:
        pcts = compute_percentiles(stats.latencies_ms, [0.5, 0.95, 0.99])

        # Throughput
        mb_per_s = (
            (stats.total_bytes / (1024 * 1024)) / stats.duration_s if stats.duration_s > 0 else 0
        )

        # Cache hit rate
        total_cache_ops = stats.cache_hits + stats.cache_misses
        hit_rate = stats.cache_hits / total_cache_ops if total_cache_ops > 0 else 0

        # Error count
        errors = stats.aborts_timeout + stats.aborts_size + stats.http_errors + stats.client_errors

        print(
            f"{stats.phase:<12} {stats.requests:>10} {stats.success:>10} {errors:>10} "
            f"{pcts['p50']:>10.1f} {pcts['p95']:>10.1f} {pcts['p99']:>10.1f} "
            f"{mb_per_s:>10.2f} {hit_rate:>10.1%}"
        )

    print("-" * 120)

    # Latency breakdown by phase (planning vs TTFB vs streaming)
    print("\nLatency Breakdown (p50 in ms):")
    print(
        f"{'Phase':<12} {'Planning':>12} {'TTFB':>12} {'Streaming':>12} "
        f"{'Total':>12}    {'Bottleneck':<20}"
    )
    print("-" * 90)

    for stats in phase_stats:
        p50_total = compute_percentiles(stats.latencies_ms, [0.5])["p50"]
        p50_planning = compute_percentiles(stats.latencies_planning_ms, [0.5])["p50"]
        p50_ttfb = compute_percentiles(stats.latencies_ttfb_ms, [0.5])["p50"]
        p50_streaming = compute_percentiles(stats.latencies_streaming_ms, [0.5])["p50"]

        # Identify bottleneck
        max_phase = max(p50_planning, p50_ttfb, p50_streaming)
        if max_phase == 0:
            bottleneck = "N/A"
        elif max_phase == p50_planning:
            bottleneck = "PLANNING (catalog/meta)"
        elif max_phase == p50_ttfb:
            bottleneck = "TTFB (first fetch)"
        else:
            bottleneck = "STREAMING (throughput)"

        print(
            f"{stats.phase:<12} {p50_planning:>12.1f} {p50_ttfb:>12.1f} {p50_streaming:>12.1f} "
            f"{p50_total:>12.1f}    {bottleneck:<20}"
        )

    print("-" * 90)

    # Detailed error breakdown
    print("\nError Breakdown:")
    print(
        f"{'Phase':<12} {'Timeouts':>10} {'Size':>10} {'Disconnect':>10} "
        f"{'HTTP Err':>10} {'Client Err':>10} {'Total Bytes':>15}"
    )
    print("-" * 90)

    for stats in phase_stats:
        print(
            f"{stats.phase:<12} {stats.aborts_timeout:>10} {stats.aborts_size:>10} "
            f"{stats.disconnects:>10} {stats.http_errors:>10} {stats.client_errors:>10} "
            f"{format_bytes(stats.total_bytes):>15}"
        )

    print("=" * 120)


def write_jsonl_results(
    results: list[RequestResult],
    metrics_samples: list[MetricsSample],
    phase_stats: list[PhaseStats],
    output_path: Path,
):
    """Write results to JSONL file."""
    with open(output_path, "w") as f:
        # Write request results
        for r in results:
            f.write(json.dumps({"type": "request", **r.to_dict()}) + "\n")

        # Write metrics samples
        for m in metrics_samples:
            f.write(json.dumps(m.to_dict()) + "\n")

        # Write phase summaries
        for stats in phase_stats:
            pcts = compute_percentiles(stats.latencies_ms, [0.5, 0.95, 0.99])
            pcts_planning = compute_percentiles(stats.latencies_planning_ms, [0.5, 0.95, 0.99])
            pcts_ttfb = compute_percentiles(stats.latencies_ttfb_ms, [0.5, 0.95, 0.99])
            pcts_streaming = compute_percentiles(stats.latencies_streaming_ms, [0.5, 0.95, 0.99])
            total_cache_ops = stats.cache_hits + stats.cache_misses
            hit_rate = stats.cache_hits / total_cache_ops if total_cache_ops > 0 else 0

            summary = {
                "type": "phase_summary",
                "phase": stats.phase,
                "requests": stats.requests,
                "success": stats.success,
                "errors": stats.aborts_timeout
                + stats.aborts_size
                + stats.http_errors
                + stats.client_errors,
                "disconnects": stats.disconnects,
                "total_bytes": stats.total_bytes,
                "duration_s": stats.duration_s,
                # Overall latency
                "p50_ms": pcts["p50"],
                "p95_ms": pcts["p95"],
                "p99_ms": pcts["p99"],
                # Detailed latency breakdown
                "planning_p50_ms": pcts_planning["p50"],
                "planning_p95_ms": pcts_planning["p95"],
                "planning_p99_ms": pcts_planning["p99"],
                "ttfb_p50_ms": pcts_ttfb["p50"],
                "ttfb_p95_ms": pcts_ttfb["p95"],
                "ttfb_p99_ms": pcts_ttfb["p99"],
                "streaming_p50_ms": pcts_streaming["p50"],
                "streaming_p95_ms": pcts_streaming["p95"],
                "streaming_p99_ms": pcts_streaming["p99"],
                # Cache stats
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "cache_hit_rate": hit_rate,
            }
            f.write(json.dumps(summary) + "\n")


# =============================================================================
# Main Execution
# =============================================================================


async def run_benchmark(config: BenchmarkConfig) -> list[PhaseStats]:
    """Run the complete benchmark."""
    print("=" * 100)
    print("STRATA PRODUCTION WORKLOAD SIMULATOR")
    print("=" * 100)
    print("\nConfiguration:")
    print(f"  Users: {config.users}")
    print(f"  Duration per phase: {config.duration_s}s")
    print(f"  Phases: {', '.join(p.value for p in config.phases)}")
    print(f"  Tables: {config.num_tables}")
    print(f"  Dry run: {config.dry_run}")

    # Setup directories
    temp_dir = None
    if config.warehouse_dir is None or config.cache_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="strata_bench_"))
        if config.warehouse_dir is None:
            config.warehouse_dir = temp_dir / "warehouse"
        if config.cache_dir is None:
            config.cache_dir = temp_dir / "cache"
        config.cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate warehouse
        print(f"\n[1/4] Generating warehouse at {config.warehouse_dir}...")
        warehouse_info = generate_warehouse(config)
        tables_info = warehouse_info["tables"]
        print(f"  Created {len(tables_info)} tables")

        # Start server if needed
        server = None
        if config.start_server:
            print("\n[2/4] Starting Strata server...")
            if config.server_port == 0:
                config.server_port = find_free_port()
            config.base_url = f"http://{config.server_host}:{config.server_port}"

            server = ServerProcess(
                host=config.server_host,
                port=config.server_port,
                cache_dir=config.cache_dir,
            )
            server.start()
            print(f"  Server running at {config.base_url}")
        else:
            print(f"\n[2/4] Using existing server at {config.base_url}")

        # Initialize load driver
        print("\n[3/4] Initializing load driver...")
        driver = AsyncLoadDriver(config)
        await driver.start()

        # Health check
        if not await driver.health_check():
            raise RuntimeError(f"Server at {config.base_url} is not healthy")
        print("  Server health check passed")

        # Initialize workload generator
        workload = WorkloadGenerator(config, tables_info)

        # Initialize phase executor
        executor = PhaseExecutor(config, driver, workload, server)

        # Run phases
        print("\n[4/4] Running benchmark phases...")
        phase_stats = []

        for phase in config.phases:
            print(f"\n  Phase: {phase.value.upper()}")

            # In dry run mode, only run for 5 requests
            if config.dry_run:
                # Override duration to be very short
                stats = await executor.run_phase(phase, duration_s=2.0)
            else:
                stats = await executor.run_phase(phase)

            phase_stats.append(stats)
            print(f"    Completed: {stats.requests} requests, {stats.success} success")

        # Stop driver
        await driver.stop()

        # Stop server if we started it
        if server:
            print("\nStopping server...")
            server.stop()

        # Print summary
        print_phase_summary(phase_stats)

        # Write results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = config.results_dir / f"prod_sim_local_{timestamp}.jsonl"
        write_jsonl_results(
            executor.results,
            executor.metrics_samples,
            phase_stats,
            output_path,
        )
        print(f"\nResults written to: {output_path}")

        return phase_stats

    finally:
        # Cleanup
        if temp_dir and not config.keep_dirs:
            print(f"\nCleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args() -> BenchmarkConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Production workload simulator for Strata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Server settings
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8765",
        help="Base URL of Strata server (default: http://127.0.0.1:8765)",
    )
    parser.add_argument(
        "--start-server",
        action="store_true",
        default=True,
        help="Start server as subprocess (default: true)",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_false",
        dest="start_server",
        help="Use existing server (don't start subprocess)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Server port (default: auto-find free port)",
    )

    # Workload settings
    parser.add_argument(
        "--users",
        type=int,
        default=10,
        help="Number of concurrent users (default: 10)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Duration per phase in seconds (default: 60)",
    )
    parser.add_argument(
        "--phases",
        type=str,
        default="cold,warm,churn,restart,disconnect",
        help="Comma-separated list of phases (default: cold,warm,churn,restart,disconnect)",
    )

    # Data settings
    parser.add_argument(
        "--rows-hot",
        type=int,
        default=500_000,
        help="Rows per hot table (default: 500000)",
    )
    parser.add_argument(
        "--rows-warm",
        type=int,
        default=200_000,
        help="Rows per warm table (default: 200000)",
    )
    parser.add_argument(
        "--rows-cold",
        type=int,
        default=50_000,
        help="Rows per cold table (default: 50000)",
    )
    parser.add_argument(
        "--payload-bytes",
        type=int,
        default=100,
        help="Payload size per row (default: 100)",
    )
    parser.add_argument(
        "--num-tables",
        type=int,
        default=10,
        help="Number of tables (default: 10)",
    )

    # Directory settings
    parser.add_argument(
        "--warehouse-dir",
        type=str,
        help="Warehouse directory (default: auto-create temp)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Cache directory (default: auto-create temp)",
    )
    parser.add_argument(
        "--keep-dirs",
        action="store_true",
        help="Keep temporary directories after run",
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only a few requests for testing",
    )

    args = parser.parse_args()

    # Parse phases
    phase_names = [p.strip() for p in args.phases.split(",")]
    phases = []
    for name in phase_names:
        try:
            phases.append(Phase(name))
        except ValueError:
            parser.error(
                f"Unknown phase: {name}. Valid phases: {', '.join(p.value for p in Phase)}"
            )

    config = BenchmarkConfig(
        base_url=args.base_url,
        start_server=args.start_server,
        server_port=args.port,
        users=args.users,
        duration_s=args.duration,
        phases=phases,
        rows_hot=args.rows_hot,
        rows_warm=args.rows_warm,
        rows_cold=args.rows_cold,
        payload_bytes=args.payload_bytes,
        num_tables=args.num_tables,
        warehouse_dir=Path(args.warehouse_dir) if args.warehouse_dir else None,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        keep_dirs=args.keep_dirs,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    return config


def main():
    """Main entry point."""
    config = parse_args()

    # Handle signals for graceful shutdown
    def signal_handler(signum, frame):
        print("\nInterrupted, cleaning up...")
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run benchmark
    asyncio.run(run_benchmark(config))


if __name__ == "__main__":
    main()
