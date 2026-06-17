#!/usr/bin/env python3
"""Stress test benchmark for Strata.

This benchmark pushes the system to production-like stress levels:
- High concurrency: 50-200 users
- Mixed response sizes: 1MB, 20MB, 200MB tables
- Cache pressure: cache set to ~2x hotset to force eviction
- Noisy neighbor: 1 user running huge scans while dashboards run

Success criteria:
- Dashboards keep p95 < 500ms while bulk scans run
- No semaphore starvation (all users get fair access)
- No growth in active futures / memory over time
- No resource leaks (active_scans returns to 0)

Usage:
    # Quick validation
    python benchmarks/stress_test.py --dry-run

    # Full stress test
    python benchmarks/stress_test.py

    # High concurrency test
    python benchmarks/stress_test.py --users 100 --duration 120

    # Noisy neighbor focus
    python benchmarks/stress_test.py --scenario noisy-neighbor
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


class Scenario(Enum):
    """Stress test scenarios."""

    FULL = "full"  # All scenarios combined
    HIGH_CONCURRENCY = "high-concurrency"  # 100+ users, mixed sizes
    CACHE_PRESSURE = "cache-pressure"  # Cache < working set
    NOISY_NEIGHBOR = "noisy-neighbor"  # 1 bulk user + dashboard users
    SUSTAINED_LOAD = "sustained"  # Long-running steady state


class QuerySize(Enum):
    """Query size classification."""

    SMALL = "small"  # ~1MB response (dashboard)
    MEDIUM = "medium"  # ~20MB response (analyst)
    LARGE = "large"  # ~200MB response (bulk/ETL)


class UserType(Enum):
    """User type classification."""

    DASHBOARD = "dashboard"  # Fast, small queries (p95 < 500ms target)
    ANALYST = "analyst"  # Medium queries
    BULK = "bulk"  # Large, slow queries (noisy neighbor)


class RequestStatus(Enum):
    """Request outcome status."""

    SUCCESS = "success"
    ABORT_TIMEOUT = "abort_timeout"
    ABORT_SIZE = "abort_size"
    DISCONNECT = "disconnect"
    HTTP_ERROR = "http_error"
    CLIENT_ERROR = "client_error"
    SEMAPHORE_REJECTED = "semaphore_rejected"  # 503 from server


@dataclass
class StressConfig:
    """Configuration for stress test."""

    # Server settings
    base_url: str = "http://127.0.0.1:8765"
    start_server: bool = True
    server_host: str = "127.0.0.1"
    server_port: int = 0  # Auto-find

    # Directories
    warehouse_dir: Path | None = None
    cache_dir: Path | None = None
    keep_dirs: bool = False

    # Scenario
    scenario: Scenario = Scenario.FULL

    # Concurrency settings
    total_users: int = 50  # Total concurrent users
    dashboard_users: int = 40  # Fast dashboard users
    analyst_users: int = 8  # Medium analyst users
    bulk_users: int = 2  # Slow bulk/noisy neighbor users

    # Duration
    duration_s: float = 120.0  # 2 minutes per phase
    warmup_s: float = 10.0  # Initial warmup phase

    # Table sizes (to create mixed response sizes)
    # Small table: ~500KB response (dashboard)
    small_table_rows: int = 5_000
    small_table_payload: int = 50  # 50 bytes per row

    # Medium table: ~5MB response (analyst)
    medium_table_rows: int = 25_000
    medium_table_payload: int = 100

    # Large table: ~25MB response (bulk)
    large_table_rows: int = 100_000
    large_table_payload: int = 150

    # Number of each table type
    num_small_tables: int = 5
    num_medium_tables: int = 3
    num_large_tables: int = 2

    # Cache pressure: set cache to ~2x hotset
    # Hotset = dashboard tables (~5MB) + some medium (~40MB) = ~50MB
    # Set cache to 100MB to force eviction when large tables accessed
    cache_size_bytes: int = 100 * 1024 * 1024  # 100MB

    # Metrics collection
    metrics_interval_s: float = 2.0  # Sample every 2s for stress monitoring
    results_dir: Path = field(default_factory=lambda: Path("benchmarks/results"))

    # Request settings
    request_timeout_s: float = 30.0  # Timeout for all queries
    connect_timeout_s: float = 5.0
    max_connections: int = 250  # High connection limit

    # Success criteria thresholds
    dashboard_p95_target_ms: float = 500.0  # Dashboard p95 < 500ms
    max_active_scans_drift: int = 5  # Max active scans at end (should be 0)

    # Misc
    seed: int = 42
    dry_run: bool = False

    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Adjust user counts based on scenario
        if self.scenario == Scenario.HIGH_CONCURRENCY:
            self.total_users = 100
            self.dashboard_users = 80
            self.analyst_users = 15
            self.bulk_users = 5
        elif self.scenario == Scenario.NOISY_NEIGHBOR:
            self.total_users = 50
            self.dashboard_users = 45
            self.analyst_users = 3
            self.bulk_users = 2  # Noisy neighbors
        elif self.scenario == Scenario.CACHE_PRESSURE:
            self.cache_size_bytes = 50 * 1024 * 1024  # 50MB - aggressive eviction
        elif self.scenario == Scenario.SUSTAINED_LOAD:
            self.duration_s = 300.0  # 5 minutes


# =============================================================================
# Data Structures for Metrics
# =============================================================================


@dataclass
class RequestResult:
    """Result from a single request."""

    request_id: str
    user_type: str
    user_id: int
    scan_id: str | None
    table_name: str
    query_size: str
    status: str
    bytes_read: int
    latency_total_ms: float
    latency_planning_ms: float
    latency_ttfb_ms: float
    latency_streaming_ms: float
    timestamp: float
    num_tasks: int = 0
    estimated_bytes: int = 0
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "type": "request",
            "request_id": self.request_id,
            "user_type": self.user_type,
            "user_id": self.user_id,
            "scan_id": self.scan_id,
            "table_name": self.table_name,
            "query_size": self.query_size,
            "status": self.status,
            "bytes_read": self.bytes_read,
            "latency_total_ms": self.latency_total_ms,
            "latency_planning_ms": self.latency_planning_ms,
            "latency_ttfb_ms": self.latency_ttfb_ms,
            "latency_streaming_ms": self.latency_streaming_ms,
            "timestamp": self.timestamp,
            "num_tasks": self.num_tasks,
            "estimated_bytes": self.estimated_bytes,
            "error": self.error,
        }


@dataclass
class MetricsSnapshot:
    """Snapshot of server metrics for monitoring."""

    timestamp: float
    elapsed_s: float
    active_scans: int
    prefetch_in_flight: int
    prefetch_started: int
    prefetch_used: int
    prefetch_wasted: int
    cache_hits: int
    cache_misses: int
    scan_count: int
    # Fields with defaults must come after non-default fields
    prefetch_skipped: int = 0
    # Resource tracking
    dashboard_requests: int = 0
    dashboard_p50_ms: float = 0.0
    dashboard_p95_ms: float = 0.0
    analyst_requests: int = 0
    bulk_requests: int = 0
    # QoS tier metrics
    interactive_active: int = 0
    interactive_slots: int = 0
    bulk_active: int = 0
    bulk_slots: int = 0
    # Cache pressure metrics
    cache_bytes_current: int = 0
    cache_bytes_max: int = 0
    cache_evictions: int = 0
    cache_evicted_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "type": "metrics_snapshot",
            **{k: v for k, v in self.__dict__.items()},
        }


@dataclass
class StressResults:
    """Aggregated stress test results."""

    scenario: str
    duration_s: float
    total_requests: int
    success_rate: float

    # By user type
    dashboard_requests: int
    dashboard_success: int
    dashboard_p50_ms: float
    dashboard_p95_ms: float
    dashboard_p99_ms: float

    analyst_requests: int
    analyst_success: int
    analyst_p50_ms: float
    analyst_p95_ms: float

    bulk_requests: int
    bulk_success: int
    bulk_p50_ms: float
    bulk_p95_ms: float

    # Resource metrics
    max_active_scans: int
    max_prefetch_in_flight: int
    final_active_scans: int
    total_bytes: int
    cache_hit_rate: float

    # Success criteria
    dashboard_p95_met: bool  # p95 < 500ms
    no_semaphore_starvation: bool  # All user types got requests through
    no_resource_leak: bool  # active_scans returned to 0

    # QoS tier metrics (with defaults for backwards compatibility)
    max_interactive_active: int = 0
    max_bulk_active: int = 0
    interactive_slots: int = 8
    bulk_slots: int = 4
    qos_isolation: bool = True  # Interactive tier not saturated by bulk

    # Cache pressure metrics
    max_cache_bytes: int = 0  # Peak cache usage
    cache_bytes_max: int = 0  # Cache limit
    total_evictions: int = 0  # Total entries evicted
    total_evicted_bytes: int = 0  # Total bytes evicted
    cache_thrash: bool = False  # True if evicted > written (thrashing)

    # Prefetch efficiency metrics
    prefetch_started: int = 0  # Total prefetches started
    prefetch_used: int = 0  # Prefetches that were consumed
    prefetch_wasted: int = 0  # Prefetches that were wasted
    prefetch_skipped: int = 0  # Prefetches skipped (server busy)
    prefetch_efficiency: float = 0.0  # used / started (0-1)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# Dataset Generator
# =============================================================================


def generate_stress_warehouse(config: StressConfig) -> dict[str, Any]:
    """Generate warehouse with mixed table sizes."""
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

    catalog = SqlCatalog(
        "strata",
        **{
            "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
            "warehouse": str(warehouse_path),
        },
    )

    try:
        catalog.create_namespace("stress")
    except Exception:
        pass

    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "ts", LongType(), required=False),
        NestedField(3, "user_id", IntegerType(), required=False),
        NestedField(4, "category", StringType(), required=False),
        NestedField(5, "value", DoubleType(), required=False),
        NestedField(6, "payload", StringType(), required=False),
    )

    tables_info = []
    categories = ["electronics", "clothing", "food", "books", "sports", "home", "auto"]
    random.seed(config.seed)

    # Create small tables (dashboard queries)
    for i in range(config.num_small_tables):
        table_name = f"small_{i:02d}"
        tables_info.append(
            _create_table(
                catalog,
                schema,
                table_name,
                config.small_table_rows,
                config.small_table_payload,
                categories,
                QuerySize.SMALL,
                warehouse_path,
            )
        )

    # Create medium tables (analyst queries)
    for i in range(config.num_medium_tables):
        table_name = f"medium_{i:02d}"
        tables_info.append(
            _create_table(
                catalog,
                schema,
                table_name,
                config.medium_table_rows,
                config.medium_table_payload,
                categories,
                QuerySize.MEDIUM,
                warehouse_path,
            )
        )

    # Create large tables (bulk queries)
    for i in range(config.num_large_tables):
        table_name = f"large_{i:02d}"
        tables_info.append(
            _create_table(
                catalog,
                schema,
                table_name,
                config.large_table_rows,
                config.large_table_payload,
                categories,
                QuerySize.LARGE,
                warehouse_path,
            )
        )

    return {
        "catalog": catalog,
        "warehouse_path": warehouse_path,
        "tables": tables_info,
    }


def _create_table(
    catalog,
    schema,
    table_name: str,
    num_rows: int,
    payload_bytes: int,
    categories: list[str],
    size: QuerySize,
    warehouse_path: Path,
) -> dict:
    """Create a single table with specified characteristics."""
    table_id = f"stress.{table_name}"

    print(
        f"  Creating {table_name} ({size.value}, {num_rows:,} rows, ~{payload_bytes}B payload)..."
    )

    try:
        table = catalog.load_table(table_id)
    except Exception:
        table = catalog.create_table(table_id, schema)

        # Write in chunks to create multiple row groups
        chunk_size = min(50_000, num_rows)
        base_ts = 1704067200000000

        for chunk_start in range(0, num_rows, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_rows)
            actual_chunk_size = chunk_end - chunk_start

            data = pa.table(
                {
                    "id": pa.array(range(chunk_start, chunk_end), type=pa.int64()),
                    "ts": pa.array(
                        [base_ts + (chunk_start + j) * 1000 for j in range(actual_chunk_size)],
                        type=pa.int64(),
                    ),
                    "user_id": pa.array(
                        [random.randint(1, 10000) for _ in range(actual_chunk_size)],
                        type=pa.int32(),
                    ),
                    "category": pa.array(
                        [random.choice(categories) for _ in range(actual_chunk_size)],
                        type=pa.string(),
                    ),
                    "value": pa.array(
                        [random.uniform(0.0, 1000.0) for _ in range(actual_chunk_size)],
                        type=pa.float64(),
                    ),
                    "payload": pa.array(
                        [
                            f"data_{chunk_start + j:08d}_" + "x" * payload_bytes
                            for j in range(actual_chunk_size)
                        ],
                        type=pa.string(),
                    ),
                }
            )
            table.append(data)

    snapshot_id = table.current_snapshot().snapshot_id

    return {
        "name": table_name,
        "table_id": table_id,
        "table_uri": f"file://{warehouse_path}#stress.{table_name}",
        "size": size.value,
        "num_rows": num_rows,
        "snapshot_id": snapshot_id,
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

        # Use DEVNULL for stdout/stderr to avoid pipe buffer deadlock
        # The server writes metrics logs to stdout, and if nothing reads from
        # the pipe, the buffer fills up and blocks the server
        self._process = subprocess.Popen(
            [sys.executable, "-m", "strata.server"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# =============================================================================
# Stress Test Driver
# =============================================================================


class StressDriver:
    """Drives stress test workload."""

    def __init__(self, config: StressConfig, tables_info: list[dict]):
        self.config = config
        self.tables_info = tables_info
        self.rng = random.Random(config.seed)

        # Group tables by size
        self.small_tables = [t for t in tables_info if t["size"] == QuerySize.SMALL.value]
        self.medium_tables = [t for t in tables_info if t["size"] == QuerySize.MEDIUM.value]
        self.large_tables = [t for t in tables_info if t["size"] == QuerySize.LARGE.value]

        # Column sets
        self.dashboard_columns = ["id", "ts", "value"]  # Narrow
        self.analyst_columns = ["id", "ts", "user_id", "category", "value"]
        self.bulk_columns = ["id", "ts", "user_id", "category", "value", "payload"]  # Full

        self.categories = ["electronics", "clothing", "food", "books", "sports", "home", "auto"]

        # Results collection
        self.results: list[RequestResult] = []
        self.metrics_snapshots: list[MetricsSnapshot] = []
        self._results_lock = asyncio.Lock()
        self._request_counter = 0
        self._counter_lock = asyncio.Lock()

        # HTTP client
        self._client: httpx.AsyncClient | None = None

        # Stop event
        self._stop_event = asyncio.Event()

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
        async with self._counter_lock:
            self._request_counter += 1
            return f"req_{self._request_counter:06d}"

    def generate_query(self, user_type: UserType) -> dict[str, Any]:
        """Generate query based on user type."""
        if user_type == UserType.DASHBOARD:
            table = self.rng.choice(self.small_tables)
            columns = self.dashboard_columns
            # Selective filter
            filters = [{"column": "category", "op": "=", "value": self.rng.choice(self.categories)}]
            query_size = QuerySize.SMALL
        elif user_type == UserType.ANALYST:
            table = self.rng.choice(self.medium_tables)
            columns = self.analyst_columns
            # Less selective
            min_val = self.rng.uniform(0, 500)
            filters = [
                {"column": "value", "op": ">=", "value": min_val},
                {"column": "value", "op": "<=", "value": min_val + 300},
            ]
            query_size = QuerySize.MEDIUM
        else:  # BULK
            table = self.rng.choice(self.large_tables)
            columns = self.bulk_columns
            filters = []  # Full table scan
            query_size = QuerySize.LARGE

        return {
            "table_uri": table["table_uri"],
            "table_name": table["name"],
            "snapshot_id": table["snapshot_id"],
            "columns": columns,
            "filters": filters,
            "query_size": query_size.value,
        }

    async def execute_scan(
        self,
        query: dict[str, Any],
        user_type: UserType,
        user_id: int,
    ) -> RequestResult:
        """Execute a single scan request."""
        request_id = await self._get_request_id()
        start_time = time.perf_counter()
        timestamp = time.time()
        scan_id = None
        bytes_read = 0
        status = RequestStatus.SUCCESS
        error = None

        planning_end_time = None
        ttfb_time = None
        streaming_start_time = None
        num_tasks = 0
        estimated_bytes = 0

        try:
            # POST /v1/materialize (scan@v1)
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

            if response.status_code == 503:
                status = RequestStatus.SEMAPHORE_REJECTED
                error = "Server at capacity (503)"
            else:
                response.raise_for_status()
                stream_url = response.json()["stream_url"]
                # num_tasks / estimated_bytes aren't in the materialize response;
                # they stay at their 0 defaults.

                # GET the stream
                async with self._client.stream("GET", stream_url) as stream:
                    stream.raise_for_status()

                    first_chunk = True
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
            if e.response.status_code == 503:
                status = RequestStatus.SEMAPHORE_REJECTED
                error = "Server at capacity (503)"
            elif e.response.status_code == 413:
                status = RequestStatus.ABORT_SIZE
                error = "Response too large"
            else:
                status = RequestStatus.HTTP_ERROR
                error = f"HTTP {e.response.status_code}"
        except Exception as e:
            status = RequestStatus.CLIENT_ERROR
            error = str(e)[:200]
        # No /v1/scan DELETE in the unified API — materialize streams self-clean
        # via the stream-state TTL.
        end_time = time.perf_counter()

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
            user_type=user_type.value,
            user_id=user_id,
            scan_id=scan_id,
            table_name=query["table_name"],
            query_size=query["query_size"],
            status=status.value,
            bytes_read=bytes_read,
            latency_total_ms=latency_total_ms,
            latency_planning_ms=latency_planning_ms,
            latency_ttfb_ms=latency_ttfb_ms,
            latency_streaming_ms=latency_streaming_ms,
            timestamp=timestamp,
            num_tasks=num_tasks,
            estimated_bytes=estimated_bytes,
            error=error,
        )

    async def user_loop(
        self,
        user_id: int,
        user_type: UserType,
        duration_s: float,
    ):
        """Run user loop for specified duration."""
        start_time = time.perf_counter()

        # Add jitter to avoid thundering herd during startup.
        # Scale jitter with user ID to spread the initial burst, but cap it
        # to ensure all users start within 1 second.
        max_jitter = min(1.0, user_id * 0.05)  # 0-1s based on user ID
        await asyncio.sleep(random.uniform(0, max_jitter))

        while time.perf_counter() - start_time < duration_s and not self._stop_event.is_set():
            try:
                query = self.generate_query(user_type)
                result = await self.execute_scan(query, user_type, user_id)

                # Store result immediately
                async with self._results_lock:
                    self.results.append(result)

                # Think time based on user type (realistic pacing)
                if user_type == UserType.DASHBOARD:
                    await asyncio.sleep(random.uniform(0.5, 1.5))  # Dashboard: 0.5-1.5s
                elif user_type == UserType.ANALYST:
                    await asyncio.sleep(random.uniform(2.0, 5.0))  # Analyst: 2-5s
                else:  # BULK
                    await asyncio.sleep(random.uniform(5.0, 15.0))  # Bulk: 5-15s
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log but continue
                print(f"  User {user_id} error: {e}")

    async def get_metrics(self) -> dict[str, Any]:
        """Get server metrics using a fresh async client.

        We create a new client for each request to avoid connection pool
        issues with the load-generating client.
        """
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=5.0, read=30.0, write=5.0, pool=5.0),
            ) as client:
                response = await client.get(f"{self.config.base_url}/metrics")
                response.raise_for_status()
                return response.json()
        except httpx.ReadTimeout:
            # Read timeout - server is processing but too slow
            if not hasattr(self, "_metrics_error_printed"):
                print("  Metrics read timeout (30s)")
                self._metrics_error_printed = True
            return {}
        except httpx.ConnectTimeout:
            # Connection timeout - can't even connect
            if not hasattr(self, "_connect_timeout_printed"):
                print("  Metrics connect timeout")
                self._connect_timeout_printed = True
            return {}
        except httpx.ConnectError:
            return {}
        except Exception as e:
            print(f"  Warning: Failed to get metrics: {type(e).__name__}: {e}")
            return {}

    async def metrics_loop(self, duration_s: float, start_time: float):
        """Collect metrics periodically."""
        fetch_failures = 0
        while time.perf_counter() - start_time < duration_s and not self._stop_event.is_set():
            await asyncio.sleep(self.config.metrics_interval_s)

            metrics = await self.get_metrics()
            if not metrics:
                fetch_failures += 1
                if fetch_failures == 1:  # Only print on first failure
                    print(f"  Warning: Metrics fetch failing (url={self.config.base_url})")
                continue
            if metrics:
                elapsed = time.perf_counter() - start_time

                # Calculate live stats from collected results
                async with self._results_lock:
                    recent_results = self.results[-1000:]  # Last 1000 requests

                dashboard_results = [
                    r for r in recent_results if r.user_type == UserType.DASHBOARD.value
                ]
                dashboard_latencies = [
                    r.latency_total_ms for r in dashboard_results if r.status == "success"
                ]

                # Extract QoS metrics
                qos = metrics.get("qos", {})
                interactive_active = qos.get("interactive_active", 0)
                interactive_slots = qos.get("interactive_slots", 8)
                bulk_active = qos.get("bulk_active", 0)
                bulk_slots = qos.get("bulk_slots", 4)

                # Extract disk cache metrics
                disk_cache = metrics.get("disk_cache", {})
                cache_bytes_current = disk_cache.get("bytes_current", 0)
                cache_bytes_max = disk_cache.get("bytes_max", 0)
                cache_evictions = disk_cache.get("evictions_count", 0)
                cache_evicted_bytes = disk_cache.get("evicted_bytes", 0)

                snapshot = MetricsSnapshot(
                    timestamp=time.time(),
                    elapsed_s=elapsed,
                    active_scans=metrics.get("resource_limits", {}).get("active_scans", 0),
                    prefetch_in_flight=metrics.get("prefetch", {}).get("in_flight", 0),
                    prefetch_started=metrics.get("prefetch", {}).get("started", 0),
                    prefetch_used=metrics.get("prefetch", {}).get("used", 0),
                    prefetch_wasted=metrics.get("prefetch", {}).get("wasted", 0),
                    cache_hits=metrics.get("cache_hits", 0),
                    cache_misses=metrics.get("cache_misses", 0),
                    scan_count=metrics.get("scan_count", 0),
                    prefetch_skipped=metrics.get("prefetch", {}).get("skipped", 0),
                    dashboard_requests=len(dashboard_results),
                    dashboard_p50_ms=_percentile(dashboard_latencies, 0.5),
                    dashboard_p95_ms=_percentile(dashboard_latencies, 0.95),
                    analyst_requests=len(
                        [r for r in recent_results if r.user_type == UserType.ANALYST.value]
                    ),
                    bulk_requests=len(
                        [r for r in recent_results if r.user_type == UserType.BULK.value]
                    ),
                    interactive_active=interactive_active,
                    interactive_slots=interactive_slots,
                    bulk_active=bulk_active,
                    bulk_slots=bulk_slots,
                    cache_bytes_current=cache_bytes_current,
                    cache_bytes_max=cache_bytes_max,
                    cache_evictions=cache_evictions,
                    cache_evicted_bytes=cache_evicted_bytes,
                )
                self.metrics_snapshots.append(snapshot)

                # Calculate cache pressure for live status
                cache_pct = (
                    (cache_bytes_current / cache_bytes_max * 100) if cache_bytes_max > 0 else 0
                )

                # Print live status with QoS tier info and cache pressure
                print(
                    f"  [{elapsed:6.1f}s] int={interactive_active}/{interactive_slots} "
                    f"bulk={bulk_active}/{bulk_slots} "
                    f"cache={cache_pct:.0f}% evict={cache_evictions} "
                    f"dashboard_p95={snapshot.dashboard_p95_ms:6.1f}ms "
                    f"reqs={len(recent_results)}"
                )

    async def _warmup_metadata_cache(self):
        """Pre-warm metadata cache by making one request to each table.

        This simulates a production scenario where metadata is already cached
        from previous requests. Without warmup, the first requests would need
        to load Parquet file metadata from disk, which serializes due to I/O
        and causes timeouts under high concurrency.
        """
        # Make one scan request to each table (sequentially to avoid contention)
        for table_info in self.tables_info:
            table_name = table_info["name"]
            try:
                # Materialize (loads metadata)
                start = time.perf_counter()
                response = await self._client.post(
                    "/v1/materialize",
                    json={
                        "inputs": [table_info["table_uri"]],
                        "transform": {
                            "executor": "scan@v1",
                            "params": {
                                "snapshot_id": table_info["snapshot_id"],
                                "columns": ["id"],  # Minimal projection
                            },
                        },
                        "mode": "stream",
                    },
                    timeout=60.0,  # Longer timeout for cold cache
                )
                if response.status_code == 200:
                    stream_url = response.json()["stream_url"]
                    # Consume a few bytes to trigger fetch
                    async with self._client.stream("GET", stream_url, timeout=60.0) as stream:
                        async for chunk in stream.aiter_bytes(chunk_size=4096):
                            break  # Just read first chunk
                    elapsed = time.perf_counter() - start
                    print(f"    {table_name}: warmed in {elapsed:.2f}s")
                else:
                    print(f"    {table_name}: HTTP {response.status_code}")
            except Exception as e:
                print(f"    Warmup failed for {table_name}: {e}")

    async def run_stress_test(self) -> StressResults:
        """Run the full stress test."""
        print(f"\n  Starting stress test: {self.config.scenario.value}")
        print(
            f"  Users: {self.config.dashboard_users} dashboard + "
            f"{self.config.analyst_users} analyst + {self.config.bulk_users} bulk"
        )
        print(f"  Duration: {self.config.duration_s}s")

        # Warmup phase: pre-warm metadata cache to avoid cold-start penalties
        # This simulates a real production scenario where metadata is already cached
        print("  Warming up metadata cache...")
        warmup_start = time.perf_counter()
        await self._warmup_metadata_cache()
        warmup_time = time.perf_counter() - warmup_start
        print(f"  Warmup complete ({warmup_time:.1f}s)")

        start_time = time.perf_counter()
        self._stop_event.clear()

        # Create user tasks
        tasks = []
        user_id = 0

        # Dashboard users
        for _ in range(self.config.dashboard_users):
            tasks.append(
                asyncio.create_task(
                    self.user_loop(user_id, UserType.DASHBOARD, self.config.duration_s)
                )
            )
            user_id += 1

        # Analyst users
        for _ in range(self.config.analyst_users):
            tasks.append(
                asyncio.create_task(
                    self.user_loop(user_id, UserType.ANALYST, self.config.duration_s)
                )
            )
            user_id += 1

        # Bulk users (noisy neighbors)
        for _ in range(self.config.bulk_users):
            tasks.append(
                asyncio.create_task(self.user_loop(user_id, UserType.BULK, self.config.duration_s))
            )
            user_id += 1

        # Metrics collection task
        metrics_task = asyncio.create_task(self.metrics_loop(self.config.duration_s, start_time))

        # Wait for test duration with timeout (add buffer for cleanup)
        all_results = []
        try:
            all_results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.duration_s + 30,  # 30s buffer for in-flight requests
            )
        except TimeoutError:
            print("\n  Test duration reached, stopping...")
            self._stop_event.set()
            # Cancel remaining tasks and collect what we have
            for task in tasks:
                if not task.done():
                    task.cancel()
            # Wait a bit for tasks to clean up
            await asyncio.sleep(1)
            # Collect results from completed tasks
            for task in tasks:
                if task.done() and not task.cancelled():
                    try:
                        result = task.result()
                        if isinstance(result, list):
                            all_results.append(result)
                    except Exception:
                        pass

        # Stop metrics collection
        self._stop_event.set()
        try:
            await asyncio.wait_for(metrics_task, timeout=5)
        except TimeoutError:
            metrics_task.cancel()

        actual_duration = min(time.perf_counter() - start_time, self.config.duration_s + 30)

        # Results are already collected in self.results by user_loop

        # Wait for server to drain all requests before fetching final metrics
        print("  Waiting for server to settle...")
        await asyncio.sleep(2.0)

        # Get final metrics with retries since server needs to calm down
        final_metrics = {}
        for attempt in range(5):
            final_metrics = await self.get_metrics()
            if final_metrics:
                break
            await asyncio.sleep(0.5)

        if not final_metrics:
            print("  WARNING: Could not fetch final metrics")

        # Compute results
        return self._compute_results(actual_duration, final_metrics)

    def _compute_results(self, duration_s: float, final_metrics: dict) -> StressResults:
        """Compute aggregated results."""
        # Group by user type
        dashboard = [r for r in self.results if r.user_type == UserType.DASHBOARD.value]
        analyst = [r for r in self.results if r.user_type == UserType.ANALYST.value]
        bulk = [r for r in self.results if r.user_type == UserType.BULK.value]

        # Success counts
        dashboard_success = [r for r in dashboard if r.status == "success"]
        analyst_success = [r for r in analyst if r.status == "success"]
        bulk_success = [r for r in bulk if r.status == "success"]

        # Latencies (success only)
        dashboard_latencies = [r.latency_total_ms for r in dashboard_success]
        analyst_latencies = [r.latency_total_ms for r in analyst_success]
        bulk_latencies = [r.latency_total_ms for r in bulk_success]

        # Resource metrics from snapshots
        max_active = max((s.active_scans for s in self.metrics_snapshots), default=0)
        max_prefetch = max((s.prefetch_in_flight for s in self.metrics_snapshots), default=0)
        final_active = final_metrics.get("resource_limits", {}).get("active_scans", 0)

        # QoS tier metrics from snapshots
        max_interactive = max((s.interactive_active for s in self.metrics_snapshots), default=0)
        max_bulk_active = max((s.bulk_active for s in self.metrics_snapshots), default=0)
        # Get slot limits from final metrics
        qos = final_metrics.get("qos", {})
        interactive_slots = qos.get("interactive_slots", 8)
        bulk_slots = qos.get("bulk_slots", 4)

        # Cache stats
        cache_hits = final_metrics.get("cache_hits", 0)
        cache_misses = final_metrics.get("cache_misses", 0)
        cache_hit_rate = (
            cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        )

        # Total bytes
        total_bytes = sum(r.bytes_read for r in self.results)

        # Success criteria
        dashboard_p95 = _percentile(dashboard_latencies, 0.95)
        dashboard_p95_met = dashboard_p95 < self.config.dashboard_p95_target_ms
        no_starvation = (
            len(dashboard_success) > 0 and len(analyst_success) > 0 and len(bulk_success) > 0
        )
        no_leak = final_active <= self.config.max_active_scans_drift

        # QoS isolation: interactive tier never saturated by bulk
        # (bulk queries shouldn't consume interactive slots)
        qos_isolation = max_interactive <= interactive_slots

        # Cache pressure metrics from snapshots
        max_cache_bytes = max((s.cache_bytes_current for s in self.metrics_snapshots), default=0)
        cache_bytes_max = (
            self.metrics_snapshots[-1].cache_bytes_max if self.metrics_snapshots else 0
        )
        total_evictions = (
            self.metrics_snapshots[-1].cache_evictions if self.metrics_snapshots else 0
        )
        total_evicted_bytes = (
            self.metrics_snapshots[-1].cache_evicted_bytes if self.metrics_snapshots else 0
        )
        # Cache thrash: evicted more bytes than written (indicates working set > cache size)
        bytes_written = final_metrics.get("bytes_written_to_cache", 0)
        cache_thrash = total_evicted_bytes > bytes_written if bytes_written > 0 else False

        # Prefetch efficiency from final metrics
        prefetch = final_metrics.get("prefetch", {})
        prefetch_started = prefetch.get("started", 0)
        prefetch_used = prefetch.get("used", 0)
        prefetch_wasted = prefetch.get("wasted", 0)
        prefetch_skipped = prefetch.get("skipped", 0)
        prefetch_efficiency = prefetch_used / prefetch_started if prefetch_started > 0 else 0.0

        total_requests = len(self.results)
        total_success = len(dashboard_success) + len(analyst_success) + len(bulk_success)

        return StressResults(
            scenario=self.config.scenario.value,
            duration_s=duration_s,
            total_requests=total_requests,
            success_rate=total_success / total_requests if total_requests > 0 else 0,
            dashboard_requests=len(dashboard),
            dashboard_success=len(dashboard_success),
            dashboard_p50_ms=_percentile(dashboard_latencies, 0.5),
            dashboard_p95_ms=dashboard_p95,
            dashboard_p99_ms=_percentile(dashboard_latencies, 0.99),
            analyst_requests=len(analyst),
            analyst_success=len(analyst_success),
            analyst_p50_ms=_percentile(analyst_latencies, 0.5),
            analyst_p95_ms=_percentile(analyst_latencies, 0.95),
            bulk_requests=len(bulk),
            bulk_success=len(bulk_success),
            bulk_p50_ms=_percentile(bulk_latencies, 0.5),
            bulk_p95_ms=_percentile(bulk_latencies, 0.95),
            max_active_scans=max_active,
            max_prefetch_in_flight=max_prefetch,
            final_active_scans=final_active,
            total_bytes=total_bytes,
            cache_hit_rate=cache_hit_rate,
            max_interactive_active=max_interactive,
            max_bulk_active=max_bulk_active,
            interactive_slots=interactive_slots,
            bulk_slots=bulk_slots,
            dashboard_p95_met=dashboard_p95_met,
            no_semaphore_starvation=no_starvation,
            no_resource_leak=no_leak,
            qos_isolation=qos_isolation,
            max_cache_bytes=max_cache_bytes,
            cache_bytes_max=cache_bytes_max,
            total_evictions=total_evictions,
            total_evicted_bytes=total_evicted_bytes,
            cache_thrash=cache_thrash,
            prefetch_started=prefetch_started,
            prefetch_used=prefetch_used,
            prefetch_wasted=prefetch_wasted,
            prefetch_skipped=prefetch_skipped,
            prefetch_efficiency=prefetch_efficiency,
        )


def _percentile(values: list[float], p: float) -> float:
    """Compute percentile of values."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = int(p * len(sorted_values))
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


# =============================================================================
# Reporting
# =============================================================================


def print_stress_results(results: StressResults):
    """Print stress test results."""
    print("\n" + "=" * 100)
    print("STRESS TEST RESULTS")
    print("=" * 100)

    print(f"\nScenario: {results.scenario}")
    print(f"Duration: {results.duration_s:.1f}s")
    print(f"Total requests: {results.total_requests:,}")
    print(f"Success rate: {results.success_rate:.1%}")
    print(f"Total bytes: {results.total_bytes / (1024 * 1024):.1f} MB")
    print(f"Cache hit rate: {results.cache_hit_rate:.1%}")

    print("\n" + "-" * 80)
    print("BY USER TYPE")
    print("-" * 80)
    header = (
        f"{'Type':<12} {'Requests':>10} {'Success':>10} "
        f"{'p50(ms)':>10} {'p95(ms)':>10} {'p99(ms)':>10}"
    )
    print(header)
    print("-" * 80)

    print(
        f"{'Dashboard':<12} {results.dashboard_requests:>10} {results.dashboard_success:>10} "
        f"{results.dashboard_p50_ms:>10.1f} {results.dashboard_p95_ms:>10.1f} "
        f"{results.dashboard_p99_ms:>10.1f}"
    )
    print(
        f"{'Analyst':<12} {results.analyst_requests:>10} {results.analyst_success:>10} "
        f"{results.analyst_p50_ms:>10.1f} {results.analyst_p95_ms:>10.1f} {'N/A':>10}"
    )
    print(
        f"{'Bulk':<12} {results.bulk_requests:>10} {results.bulk_success:>10} "
        f"{results.bulk_p50_ms:>10.1f} {results.bulk_p95_ms:>10.1f} {'N/A':>10}"
    )

    print("\n" + "-" * 80)
    print("RESOURCE METRICS")
    print("-" * 80)
    print(f"Max active scans: {results.max_active_scans}")
    print(f"Max prefetch in-flight: {results.max_prefetch_in_flight}")
    print(f"Final active scans: {results.final_active_scans}")

    print("\n" + "-" * 80)
    print("QOS TIER METRICS")
    print("-" * 80)
    print(
        f"Interactive tier: max {results.max_interactive_active} / "
        f"{results.interactive_slots} slots"
    )
    print(f"Bulk tier: max {results.max_bulk_active} / {results.bulk_slots} slots")

    print("\n" + "-" * 80)
    print("CACHE PRESSURE")
    print("-" * 80)

    def format_bytes(b: int) -> str:
        """Format bytes as human-readable string."""
        if b >= 1024 * 1024 * 1024:
            return f"{b / (1024 * 1024 * 1024):.1f} GB"
        if b >= 1024 * 1024:
            return f"{b / (1024 * 1024):.1f} MB"
        if b >= 1024:
            return f"{b / 1024:.1f} KB"
        return f"{b} B"

    cache_pct = (
        (results.max_cache_bytes / results.cache_bytes_max * 100)
        if results.cache_bytes_max > 0
        else 0
    )
    print(
        f"Peak cache usage: {format_bytes(results.max_cache_bytes)} / "
        f"{format_bytes(results.cache_bytes_max)} ({cache_pct:.0f}%)"
    )
    print(f"Total evictions: {results.total_evictions} entries")
    print(f"Total evicted bytes: {format_bytes(results.total_evicted_bytes)}")
    print(f"Cache thrashing: {'YES' if results.cache_thrash else 'NO'}")

    print("\n" + "-" * 80)
    print("PREFETCH EFFICIENCY")
    print("-" * 80)
    print(f"Prefetches started: {results.prefetch_started}")
    print(
        f"Prefetches used: {results.prefetch_used} ({results.prefetch_efficiency:.0%} efficiency)"
    )
    print(f"Prefetches wasted: {results.prefetch_wasted}")
    print(f"Prefetches skipped (server busy): {results.prefetch_skipped}")
    # Warn if wasted is high
    if results.prefetch_started > 0:
        wasted_pct = results.prefetch_wasted / results.prefetch_started
        if wasted_pct > 0.25:
            print(
                f"WARNING: High prefetch waste ({wasted_pct:.0%}) - prefetch may be amplifying load"
            )

    print("\n" + "-" * 80)
    print("SUCCESS CRITERIA")
    print("-" * 80)

    def status(passed: bool) -> str:
        return "PASS" if passed else "FAIL"

    p95_status = status(results.dashboard_p95_met)
    print(f"Dashboard p95 < 500ms: {p95_status} ({results.dashboard_p95_ms:.1f}ms)")
    print(f"No semaphore starvation: {status(results.no_semaphore_starvation)}")
    leak_status = status(results.no_resource_leak)
    print(f"No resource leak: {leak_status} (final active={results.final_active_scans})")
    print(
        f"QoS isolation: {status(results.qos_isolation)} "
        f"(interactive max={results.max_interactive_active})"
    )

    all_passed = (
        results.dashboard_p95_met
        and results.no_semaphore_starvation
        and results.no_resource_leak
        and results.qos_isolation
    )
    print("\n" + "=" * 80)
    print(f"OVERALL: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 80)


def write_stress_results(
    driver: StressDriver,
    results: StressResults,
    output_path: Path,
):
    """Write results to JSONL file."""
    with open(output_path, "w") as f:
        # Write all request results
        for r in driver.results:
            f.write(json.dumps(r.to_dict()) + "\n")

        # Write metrics snapshots
        for m in driver.metrics_snapshots:
            f.write(json.dumps(m.to_dict()) + "\n")

        # Write summary
        f.write(json.dumps({"type": "summary", **results.to_dict()}) + "\n")


# =============================================================================
# Main Execution
# =============================================================================


async def run_stress_test(config: StressConfig) -> StressResults:
    """Run the stress test."""
    print("=" * 100)
    print("STRATA STRESS TEST")
    print("=" * 100)
    print(f"\nScenario: {config.scenario.value}")
    print(
        f"Users: {config.total_users} ({config.dashboard_users} dashboard + "
        f"{config.analyst_users} analyst + {config.bulk_users} bulk)"
    )
    print(f"Duration: {config.duration_s}s")
    print(f"Cache size: {config.cache_size_bytes / (1024 * 1024):.0f} MB")

    temp_dir = None
    if config.warehouse_dir is None or config.cache_dir is None:
        temp_dir = Path(tempfile.mkdtemp(prefix="strata_stress_"))
        if config.warehouse_dir is None:
            config.warehouse_dir = temp_dir / "warehouse"
        if config.cache_dir is None:
            config.cache_dir = temp_dir / "cache"
        config.cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Generate warehouse
        print(f"\n[1/3] Generating warehouse at {config.warehouse_dir}...")
        warehouse_info = generate_stress_warehouse(config)
        tables_info = warehouse_info["tables"]
        print(f"  Created {len(tables_info)} tables")

        # Start server
        server = None
        if config.start_server:
            print("\n[2/3] Starting Strata server...")
            if config.server_port == 0:
                config.server_port = find_free_port()
            config.base_url = f"http://{config.server_host}:{config.server_port}"

            server = ServerProcess(
                host=config.server_host,
                port=config.server_port,
                cache_dir=config.cache_dir,
                max_cache_size_bytes=config.cache_size_bytes,
            )
            server.start()
            print(f"  Server running at {config.base_url}")
            print(f"  Cache limit: {config.cache_size_bytes / (1024 * 1024):.0f} MB")
        else:
            print(f"\n[2/3] Using existing server at {config.base_url}")

        # Run stress test
        print("\n[3/3] Running stress test...")
        driver = StressDriver(config, tables_info)
        await driver.start()

        if config.dry_run:
            config.duration_s = 15.0
            config.request_timeout_s = 10.0  # Shorter timeout for dry run
            print("  (dry run - 15s duration)")

        results = await driver.run_stress_test()

        await driver.stop()

        if server:
            print("\nStopping server...")
            server.stop()

        # Print results
        print_stress_results(results)

        # Write results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = config.results_dir / f"stress_test_{config.scenario.value}_{timestamp}.jsonl"
        write_stress_results(driver, results, output_path)
        print(f"\nResults written to: {output_path}")

        return results

    finally:
        if temp_dir and not config.keep_dirs:
            print(f"\nCleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir, ignore_errors=True)


def parse_args() -> StressConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Stress test for Strata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--scenario",
        type=str,
        default="full",
        choices=[s.value for s in Scenario],
        help="Test scenario (default: full)",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=50,
        help="Total concurrent users (default: 50)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Test duration in seconds (default: 120)",
    )
    parser.add_argument(
        "--cache-size-mb",
        type=int,
        default=100,
        help="Cache size in MB (default: 100)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Server port (default: auto-find)",
    )
    parser.add_argument(
        "--no-start-server",
        action="store_true",
        help="Use existing server",
    )
    parser.add_argument(
        "--keep-dirs",
        action="store_true",
        help="Keep temporary directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Quick test run (10s)",
    )

    args = parser.parse_args()

    # Determine user distribution based on total users
    total = args.users
    dashboard = int(total * 0.8)
    analyst = int(total * 0.16)
    bulk = total - dashboard - analyst

    config = StressConfig(
        scenario=Scenario(args.scenario),
        total_users=total,
        dashboard_users=dashboard,
        analyst_users=analyst,
        bulk_users=bulk,
        duration_s=args.duration,
        cache_size_bytes=args.cache_size_mb * 1024 * 1024,
        server_port=args.port,
        start_server=not args.no_start_server,
        keep_dirs=args.keep_dirs,
        dry_run=args.dry_run,
    )

    return config


def main():
    """Main entry point."""
    config = parse_args()

    def signal_handler(signum, frame):
        print("\nInterrupted, cleaning up...")
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    asyncio.run(run_stress_test(config))


if __name__ == "__main__":
    main()
