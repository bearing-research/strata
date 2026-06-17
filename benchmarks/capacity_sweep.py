#!/usr/bin/env python3
"""Capacity sweep benchmark for Strata.

Maps the "safe operating envelope" by running load sweeps at increasing
concurrency levels and measuring throughput, latency, and error rates.

Goal: Identify the "knee" where latency explodes or 429s start.

Plots (conceptual):
- 2xx throughput vs offered load
- p95 latency (2xx only) vs offered load
- 429 rate vs offered load
- Other failure rate vs offered load

Usage:
    # Quick sweep (5 levels, 2 min each)
    python benchmarks/capacity_sweep.py

    # Detailed sweep (10 levels, 3 min each)
    python benchmarks/capacity_sweep.py --levels 10 --duration 180

    # Custom load range
    python benchmarks/capacity_sweep.py --min-users 10 --max-users 100

    # Connect to external server
    python benchmarks/capacity_sweep.py --no-server --base-url http://localhost:8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SweepConfig:
    """Configuration for capacity sweep."""

    # Server settings
    base_url: str = "http://127.0.0.1:8765"
    start_server: bool = True
    server_host: str = "127.0.0.1"
    server_port: int = 0  # Auto-find

    # Directories
    warehouse_dir: Path | None = None
    cache_dir: Path | None = None
    keep_dirs: bool = False

    # Sweep parameters
    num_levels: int = 6  # Number of load levels to test
    min_users: int = 5  # Starting concurrency
    max_users: int = 60  # Maximum concurrency
    duration_per_level_s: float = 120.0  # 2 minutes per level
    warmup_s: float = 15.0  # Warmup before each level
    cooldown_s: float = 10.0  # Cooldown between levels

    # Workload mix (dashboard-heavy to stress interactive tier)
    dashboard_ratio: float = 0.80  # 80% dashboard queries
    analyst_ratio: float = 0.15  # 15% analyst queries
    bulk_ratio: float = 0.05  # 5% bulk queries

    # Table configuration
    num_tables: int = 6
    rows_per_table: int = 30_000
    payload_bytes: int = 100

    # Cache settings
    cache_size_bytes: int = 150 * 1024 * 1024  # 150MB

    # QoS settings (default server settings)
    interactive_slots: int = 8
    bulk_slots: int = 4

    # Metrics collection
    metrics_interval_s: float = 5.0  # Sample every 5s
    results_dir: Path = field(default_factory=lambda: Path("benchmarks/results"))

    # Request settings
    request_timeout_s: float = 60.0
    connect_timeout_s: float = 5.0
    max_connections: int = 200

    # Misc
    seed: int = 42
    dry_run: bool = False

    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def get_load_levels(self) -> list[int]:
        """Generate list of load levels to test."""
        if self.num_levels == 1:
            return [self.max_users]

        step = (self.max_users - self.min_users) / (self.num_levels - 1)
        levels = [int(self.min_users + i * step) for i in range(self.num_levels)]
        # Ensure max is included
        levels[-1] = self.max_users
        return levels


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class RequestResult:
    """Result of a single request with per-phase timing breakdown."""

    # Total end-to-end latency
    total_latency_ms: float
    user_type: str  # dashboard, analyst, bulk
    status_code: int
    bytes_read: int

    # Per-phase latency breakdown (to identify bottlenecks)
    post_ms: float = 0.0  # POST /v1/scan latency
    get_ms: float = 0.0  # GET /v1/scan/{id}/batches latency
    delete_ms: float = 0.0  # DELETE /v1/scan/{id} latency

    # Error classification (explicit types for debugging under load)
    error_type: str | None = None  # timeout, connection, cancelled, decode, other
    error_detail: str | None = None  # Additional error info

    timestamp: float = 0.0

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def is_rate_limited(self) -> bool:
        return self.status_code == 429

    @property
    def is_5xx(self) -> bool:
        return 500 <= self.status_code < 600

    @property
    def is_timeout(self) -> bool:
        return self.error_type == "timeout"

    @property
    def is_connection_error(self) -> bool:
        return self.error_type == "connection"

    @property
    def is_cancelled(self) -> bool:
        return self.error_type == "cancelled"


@dataclass
class LevelMetrics:
    """Metrics for a single load level."""

    level_num: int
    concurrent_users: int
    target_duration_s: float  # Requested duration
    actual_duration_s: float = 0.0  # Actual wall time (after hard cutoff)

    # Request counts
    total_requests: int = 0  # Completed requests (with results)
    attempted_requests: int = 0  # All attempts (includes in-flight at cutoff)
    success_2xx: int = 0
    rate_limited_429: int = 0
    server_error_5xx: int = 0

    # Detailed error breakdown (for debugging under load)
    timeout_errors: int = 0
    connection_errors: int = 0
    cancelled_errors: int = 0  # Client-cancelled (task cancellation)
    other_errors: int = 0

    # Throughput: offered vs achieved (based on attempted for true offered load)
    offered_rps: float = 0.0  # Attempted requests / second (true offered load)
    goodput_rps: float = 0.0  # Successful 2xx / second (achieved throughput)

    # Latency (2xx only) - total end-to-end
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_max_ms: float = 0.0

    # Per-phase p95 latency (to identify bottlenecks)
    post_p95_ms: float = 0.0  # POST /v1/scan
    get_p95_ms: float = 0.0  # GET /v1/scan/{id}/batches
    delete_p95_ms: float = 0.0  # DELETE /v1/scan/{id}

    # Rates
    success_rate: float = 0.0
    rate_limited_rate: float = 0.0
    error_rate: float = 0.0

    # By user type (total and per-phase p95)
    dashboard_count: int = 0
    dashboard_success: int = 0
    dashboard_p95_ms: float = 0.0
    dashboard_post_p95_ms: float = 0.0
    dashboard_get_p95_ms: float = 0.0

    analyst_count: int = 0
    analyst_success: int = 0
    analyst_p95_ms: float = 0.0
    analyst_post_p95_ms: float = 0.0
    analyst_get_p95_ms: float = 0.0

    bulk_count: int = 0
    bulk_success: int = 0
    bulk_p95_ms: float = 0.0
    bulk_post_p95_ms: float = 0.0
    bulk_get_p95_ms: float = 0.0

    # Server metrics (sampled within level window)
    avg_active_scans: float = 0.0
    max_active_scans: int = 0
    final_active_scans: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class SweepResults:
    """Overall sweep results."""

    levels: list[LevelMetrics]
    total_duration_s: float
    total_requests: int

    # Identified capacity points
    saturation_users: int | None = None  # Where 429 > 5%
    latency_knee_users: int | None = None  # Where p95 > 2x baseline
    error_threshold_users: int | None = None  # Where errors > 1%

    # Baseline (lowest load level)
    baseline_p95_ms: float = 0.0
    baseline_goodput_rps: float = 0.0

    # Peak performance
    peak_goodput_rps: float = 0.0
    peak_goodput_users: int = 0

    # Operational recommendation
    recommended_max_users: int | None = None  # Safe operating point
    recommended_goodput_rps: float = 0.0  # Expected throughput at safe point

    def to_dict(self) -> dict:
        return {
            "levels": [level.to_dict() for level in self.levels],
            "total_duration_s": self.total_duration_s,
            "total_requests": self.total_requests,
            "saturation_users": self.saturation_users,
            "latency_knee_users": self.latency_knee_users,
            "error_threshold_users": self.error_threshold_users,
            "baseline_p95_ms": self.baseline_p95_ms,
            "baseline_goodput_rps": self.baseline_goodput_rps,
            "peak_goodput_rps": self.peak_goodput_rps,
            "peak_goodput_users": self.peak_goodput_users,
            "recommended_max_users": self.recommended_max_users,
            "recommended_goodput_rps": self.recommended_goodput_rps,
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
        log_dir: Path | None = None,
        interactive_slots: int | None = None,
        bulk_slots: int | None = None,
    ):
        self.host = host
        self.port = port
        self.cache_dir = cache_dir
        self.max_cache_size_bytes = max_cache_size_bytes
        self.log_dir = log_dir
        self.interactive_slots = interactive_slots
        self.bulk_slots = bulk_slots
        self._process: subprocess.Popen | None = None
        self._stdout_file: Any = None
        self._stderr_file: Any = None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process else None

    def start(self, timeout: float = 30.0):
        """Start the server as a subprocess."""
        env = os.environ.copy()
        env["STRATA_HOST"] = self.host
        env["STRATA_PORT"] = str(self.port)
        env["STRATA_CACHE_DIR"] = str(self.cache_dir)
        env["STRATA_METRICS_ENABLED"] = "true"
        env["STRATA_LOG_FORMAT"] = "json"
        env["STRATA_LOG_LEVEL"] = "INFO"

        if self.max_cache_size_bytes is not None:
            env["STRATA_MAX_CACHE_SIZE_BYTES"] = str(self.max_cache_size_bytes)

        if self.interactive_slots is not None:
            env["STRATA_INTERACTIVE_SLOTS"] = str(self.interactive_slots)
        if self.bulk_slots is not None:
            env["STRATA_BULK_SLOTS"] = str(self.bulk_slots)

        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._stdout_file = open(self.log_dir / "server_stdout.log", "w")
            self._stderr_file = open(self.log_dir / "server_stderr.log", "w")
            stdout_dest = self._stdout_file
            stderr_dest = self._stderr_file
        else:
            stdout_dest = subprocess.DEVNULL
            stderr_dest = subprocess.DEVNULL

        self._process = subprocess.Popen(
            [sys.executable, "-m", "strata.server"],
            env=env,
            stdout=stdout_dest,
            stderr=stderr_dest,
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
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()

            self._process = None

        if self._stdout_file:
            self._stdout_file.close()
        if self._stderr_file:
            self._stderr_file.close()


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# =============================================================================
# Dataset Generation
# =============================================================================


def generate_warehouse(config: SweepConfig) -> dict[str, Any]:
    """Generate warehouse with test tables."""
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
        catalog.create_namespace("sweep")
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
    categories = ["electronics", "clothing", "food", "books", "sports"]
    random.seed(config.seed)

    for i in range(config.num_tables):
        table_name = f"events_{i:02d}"
        table_id = f"sweep.{table_name}"

        try:
            catalog.drop_table(table_id)
        except Exception:
            pass

        table = catalog.create_table(table_id, schema)

        # Generate data
        data = pa.table(
            {
                "id": pa.array(range(config.rows_per_table), type=pa.int64()),
                "ts": pa.array(
                    [
                        int(time.time() * 1000) - random.randint(0, 86400000)
                        for _ in range(config.rows_per_table)
                    ],
                    type=pa.int64(),
                ),
                "user_id": pa.array(
                    [random.randint(1, 1000) for _ in range(config.rows_per_table)],
                    type=pa.int32(),
                ),
                "category": pa.array(
                    [random.choice(categories) for _ in range(config.rows_per_table)],
                    type=pa.string(),
                ),
                "value": pa.array(
                    [random.random() * 100 for _ in range(config.rows_per_table)],
                    type=pa.float64(),
                ),
                "payload": pa.array(
                    ["x" * config.payload_bytes for _ in range(config.rows_per_table)],
                    type=pa.string(),
                ),
            }
        )
        table.append(data)

        tables_info.append(
            {
                "name": table_name,
                "uri": f"file://{warehouse_path}#sweep.{table_name}",
                "rows": config.rows_per_table,
            }
        )

    return {
        "catalog": catalog,
        "warehouse_path": warehouse_path,
        "tables": tables_info,
    }


# =============================================================================
# Load Driver
# =============================================================================


def _percentile(data: list[float], p: float) -> float:
    """Compute percentile of sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


class LoadDriver:
    """Drives load at a specified concurrency level."""

    def __init__(
        self,
        config: SweepConfig,
        tables_info: list[dict],
        num_users: int,
    ):
        self.config = config
        self.tables_info = tables_info
        self.num_users = num_users
        self.rng = random.Random(config.seed)

        # Calculate user counts by type (allow 0 bulk users at low load)
        self.dashboard_users = int(num_users * config.dashboard_ratio)
        self.analyst_users = int(num_users * config.analyst_ratio)
        self.bulk_users = num_users - self.dashboard_users - self.analyst_users
        # Ensure at least 1 dashboard user
        if self.dashboard_users == 0 and num_users > 0:
            self.dashboard_users = 1
            self.analyst_users = max(0, self.analyst_users - 1)

        # Results collection
        self.results: list[RequestResult] = []
        self._lock = asyncio.Lock()
        self._stop_event = asyncio.Event()

        # Attempt counter for true offered load (incremented before each request)
        self.attempted_requests: int = 0

        # HTTP client
        self._client: httpx.AsyncClient | None = None

        # Metrics samples with timestamps for filtering
        self.metrics_samples: list[dict] = []
        self._level_start_time: float = 0.0
        self._level_end_time: float = 0.0

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
        self._stop_event.set()
        if self._client:
            await self._client.aclose()
            self._client = None

    async def run_level(
        self, duration_s: float, hard_cutoff: bool = True
    ) -> tuple[list[RequestResult], float, float, float, int]:
        """Run load for specified duration.

        Args:
            duration_s: How long to run the level
            hard_cutoff: If True, cancel tasks at end. If False, let them drain.

        Returns:
            Tuple of (results, actual_duration_s, start_time, end_time, attempted)
        """
        self.results = []
        self.metrics_samples = []
        self.attempted_requests = 0
        self._stop_event.clear()

        level_start = time.perf_counter()
        level_start_time = time.time()

        # Start user tasks
        tasks = []
        user_id = 0

        for i in range(self.dashboard_users):
            tasks.append(asyncio.create_task(self._user_loop("dashboard", user_id)))
            user_id += 1

        for i in range(self.analyst_users):
            tasks.append(asyncio.create_task(self._user_loop("analyst", user_id)))
            user_id += 1

        for i in range(self.bulk_users):
            tasks.append(asyncio.create_task(self._user_loop("bulk", user_id)))
            user_id += 1

        # Start metrics collector
        metrics_task = asyncio.create_task(self._collect_metrics())

        # Run for duration then signal stop
        await asyncio.sleep(duration_s)
        self._stop_event.set()

        if hard_cutoff:
            # Hard cutoff: cancel user tasks (they'll record cancelled results)
            for t in tasks:
                t.cancel()

        # Wait for user tasks to complete (cancelled or natural)
        await asyncio.gather(*tasks, return_exceptions=True)

        # Let metrics task exit naturally via stop_event (don't cancel)
        # Give it a brief moment to collect final sample
        try:
            await asyncio.wait_for(metrics_task, timeout=1.0)
        except TimeoutError:
            metrics_task.cancel()
            await asyncio.gather(metrics_task, return_exceptions=True)

        level_end_time = time.time()
        actual_duration = time.perf_counter() - level_start

        return (
            self.results,
            actual_duration,
            level_start_time,
            level_end_time,
            self.attempted_requests,
        )

    async def _user_loop(self, user_type: str, user_id: int):
        """Simulate a user making requests."""
        request_start: float | None = None
        try:
            while not self._stop_event.is_set():
                # Track when request starts for cancelled recording
                request_start = time.perf_counter()
                async with self._lock:
                    self.attempted_requests += 1

                result = await self._execute_request(user_type)
                request_start = None  # Request completed, clear tracking

                async with self._lock:
                    self.results.append(result)

                # Think time based on user type
                if user_type == "dashboard":
                    await asyncio.sleep(self.rng.uniform(0.5, 1.5))
                elif user_type == "analyst":
                    await asyncio.sleep(self.rng.uniform(2.0, 5.0))
                else:  # bulk
                    await asyncio.sleep(self.rng.uniform(5.0, 10.0))
        except asyncio.CancelledError:
            # Hard cutoff - record in-flight request as cancelled with timing
            if request_start is not None:
                elapsed_ms = (time.perf_counter() - request_start) * 1000
                async with self._lock:
                    self.results.append(
                        RequestResult(
                            total_latency_ms=elapsed_ms,
                            user_type=user_type,
                            status_code=0,
                            bytes_read=0,
                            error_type="cancelled",
                            timestamp=time.time(),
                        )
                    )
            raise  # Re-raise to properly propagate cancellation

    async def _execute_request(self, user_type: str) -> RequestResult:
        """Execute a single scan request with per-phase timing."""
        request_start = time.perf_counter()
        table = self.rng.choice(self.tables_info)

        # Build request
        if user_type == "dashboard":
            columns = ["id", "ts", "value"]
            filters = [{"column": "category", "op": "=", "value": "electronics"}]
        elif user_type == "analyst":
            columns = ["id", "ts", "user_id", "category", "value"]
            filters = []
        else:  # bulk
            columns = None  # All columns
            filters = []

        stream_url = None
        status_code = 0
        bytes_read = 0
        error_type = None
        error_detail = None

        # Per-phase timing (set in finally blocks for accuracy on errors)
        post_ms = 0.0
        get_ms = 0.0
        delete_ms = 0.0
        post_start: float | None = None
        get_start: float | None = None

        try:
            # POST /v1/materialize (scan@v1 planning)
            post_start = time.perf_counter()
            try:
                resp = await self._client.post(
                    "/v1/materialize",
                    json={
                        "inputs": [table["uri"]],
                        "transform": {
                            "executor": "scan@v1",
                            "params": {"columns": columns, "filters": filters},
                        },
                        "mode": "stream",
                    },
                )
                status_code = resp.status_code
            finally:
                post_ms = (time.perf_counter() - post_start) * 1000

            if resp.status_code != 200:
                return RequestResult(
                    total_latency_ms=(time.perf_counter() - request_start) * 1000,
                    user_type=user_type,
                    status_code=status_code,
                    bytes_read=0,
                    post_ms=post_ms,
                    timestamp=time.time(),
                )

            stream_url = resp.json()["stream_url"]

            # GET the stream (TTFB + streaming)
            get_start = time.perf_counter()
            try:
                async with self._client.stream("GET", stream_url) as stream:
                    status_code = stream.status_code
                    if stream.status_code == 200:
                        async for chunk in stream.aiter_bytes():
                            bytes_read += len(chunk)
            finally:
                get_ms = (time.perf_counter() - get_start) * 1000

        except asyncio.CancelledError:
            error_type = "cancelled"
            raise  # Re-raise to propagate cancellation
        except httpx.TimeoutException as e:
            error_type = "timeout"
            error_detail = str(e)
            status_code = 0
        except httpx.ConnectError as e:
            error_type = "connection"
            error_detail = str(e)
            status_code = 0
        except Exception as e:
            error_type = "other"
            error_detail = f"{type(e).__name__}: {e}"
            status_code = 0
        # No explicit cleanup: a materialize stream is freed server-side by the
        # stream-state TTL once consumed (the old /v1/scan DELETE is gone), so
        # delete_ms stays 0.

        return RequestResult(
            total_latency_ms=(time.perf_counter() - request_start) * 1000,
            user_type=user_type,
            status_code=status_code,
            bytes_read=bytes_read,
            post_ms=post_ms,
            get_ms=get_ms,
            delete_ms=delete_ms,
            error_type=error_type,
            error_detail=error_detail,
            timestamp=time.time(),
        )

    async def _collect_metrics(self):
        """Periodically collect server metrics with timestamps."""
        try:
            while not self._stop_event.is_set():
                try:
                    resp = await self._client.get("/metrics")
                    if resp.status_code == 200:
                        sample = resp.json()
                        # Add timestamp for filtering to level window
                        sample["timestamp"] = time.time()
                        self.metrics_samples.append(sample)
                except Exception:
                    pass
                await asyncio.sleep(self.config.metrics_interval_s)
        except asyncio.CancelledError:
            pass  # Clean exit on cancellation


# =============================================================================
# Sweep Runner
# =============================================================================


def compute_level_metrics(
    level_num: int,
    num_users: int,
    target_duration_s: float,
    actual_duration_s: float,
    results: list[RequestResult],
    metrics_samples: list[dict],
    level_start_time: float = 0.0,
    level_end_time: float = 0.0,
    attempted_requests: int = 0,
) -> LevelMetrics:
    """Compute metrics for a load level.

    Args:
        level_num: Level number (1-indexed)
        num_users: Number of concurrent users
        target_duration_s: Requested duration
        actual_duration_s: Actual wall time (after hard cutoff)
        results: List of request results
        metrics_samples: List of server metrics samples (with timestamps)
        level_start_time: Unix timestamp when level started
        level_end_time: Unix timestamp when level ended
        attempted_requests: Total attempts started (for true offered load)
    """
    metrics = LevelMetrics(
        level_num=level_num,
        concurrent_users=num_users,
        target_duration_s=target_duration_s,
        actual_duration_s=actual_duration_s,
    )

    # Always set attempted_requests (even if no results completed)
    metrics.attempted_requests = attempted_requests

    if not results:
        # Still compute offered_rps from attempts even with no completed results
        if actual_duration_s > 0:
            metrics.offered_rps = attempted_requests / actual_duration_s
        return metrics

    metrics.total_requests = len(results)
    # Use attempted count for true offered load (falls back to total if not provided)
    metrics.attempted_requests = attempted_requests if attempted_requests > 0 else len(results)

    # Classify results
    success_results = [r for r in results if r.is_success]
    metrics.success_2xx = len(success_results)
    metrics.rate_limited_429 = sum(1 for r in results if r.is_rate_limited)
    metrics.server_error_5xx = sum(1 for r in results if r.is_5xx)

    # Detailed error breakdown
    metrics.timeout_errors = sum(1 for r in results if r.error_type == "timeout")
    metrics.connection_errors = sum(1 for r in results if r.error_type == "connection")
    metrics.cancelled_errors = sum(1 for r in results if r.error_type == "cancelled")
    metrics.other_errors = sum(
        1
        for r in results
        if r.error_type is not None and r.error_type not in ("timeout", "connection", "cancelled")
    )

    # Throughput: offered uses attempted count, goodput uses success count
    if actual_duration_s > 0:
        metrics.offered_rps = metrics.attempted_requests / actual_duration_s
        metrics.goodput_rps = metrics.success_2xx / actual_duration_s
    else:
        metrics.offered_rps = 0.0
        metrics.goodput_rps = 0.0

    # Latency (2xx only) - total end-to-end
    success_latencies = [r.total_latency_ms for r in success_results]
    if success_latencies:
        metrics.latency_p50_ms = _percentile(success_latencies, 0.50)
        metrics.latency_p95_ms = _percentile(success_latencies, 0.95)
        metrics.latency_p99_ms = _percentile(success_latencies, 0.99)
        metrics.latency_max_ms = max(success_latencies)

    # Per-phase p95 latency (2xx only, to identify bottlenecks)
    post_latencies = [r.post_ms for r in success_results if r.post_ms > 0]
    get_latencies = [r.get_ms for r in success_results if r.get_ms > 0]
    delete_latencies = [r.delete_ms for r in success_results if r.delete_ms > 0]

    if post_latencies:
        metrics.post_p95_ms = _percentile(post_latencies, 0.95)
    if get_latencies:
        metrics.get_p95_ms = _percentile(get_latencies, 0.95)
    if delete_latencies:
        metrics.delete_p95_ms = _percentile(delete_latencies, 0.95)

    # Rates
    metrics.success_rate = metrics.success_2xx / metrics.total_requests
    metrics.rate_limited_rate = metrics.rate_limited_429 / metrics.total_requests
    metrics.error_rate = (
        metrics.server_error_5xx
        + metrics.timeout_errors
        + metrics.connection_errors
        + metrics.cancelled_errors
        + metrics.other_errors
    ) / metrics.total_requests

    # By user type (with per-phase p95 to identify tier-specific bottlenecks)
    for user_type in ["dashboard", "analyst", "bulk"]:
        type_results = [r for r in results if r.user_type == user_type]
        type_success = [r for r in type_results if r.is_success]
        type_latencies = [r.total_latency_ms for r in type_success]

        count = len(type_results)
        success = len(type_success)
        p95 = _percentile(type_latencies, 0.95) if type_latencies else 0.0

        # Per-phase p95 for this user type
        type_post = [r.post_ms for r in type_success if r.post_ms > 0]
        type_get = [r.get_ms for r in type_success if r.get_ms > 0]
        post_p95 = _percentile(type_post, 0.95) if type_post else 0.0
        get_p95 = _percentile(type_get, 0.95) if type_get else 0.0

        if user_type == "dashboard":
            metrics.dashboard_count = count
            metrics.dashboard_success = success
            metrics.dashboard_p95_ms = p95
            metrics.dashboard_post_p95_ms = post_p95
            metrics.dashboard_get_p95_ms = get_p95
        elif user_type == "analyst":
            metrics.analyst_count = count
            metrics.analyst_success = success
            metrics.analyst_p95_ms = p95
            metrics.analyst_post_p95_ms = post_p95
            metrics.analyst_get_p95_ms = get_p95
        else:
            metrics.bulk_count = count
            metrics.bulk_success = success
            metrics.bulk_p95_ms = p95
            metrics.bulk_post_p95_ms = post_p95
            metrics.bulk_get_p95_ms = get_p95

    # Server metrics - filter to level window if timestamps available
    if metrics_samples:
        # Filter samples to level window if we have timestamps
        if level_start_time > 0 and level_end_time > 0:
            window_samples = [
                m
                for m in metrics_samples
                if level_start_time <= m.get("timestamp", 0) <= level_end_time
            ]
            # Fall back to all samples if filtering removes everything
            if not window_samples:
                window_samples = metrics_samples
        else:
            window_samples = metrics_samples

        active_scans = [m.get("resource_limits", {}).get("active_scans", 0) for m in window_samples]
        if active_scans:
            metrics.avg_active_scans = sum(active_scans) / len(active_scans)
            metrics.max_active_scans = max(active_scans)
            metrics.final_active_scans = active_scans[-1]

    return metrics


def analyze_sweep(levels: list[LevelMetrics]) -> SweepResults:
    """Analyze sweep results to find capacity points and operational recommendation."""
    if not levels:
        return SweepResults(
            levels=[],
            total_duration_s=0,
            total_requests=0,
        )

    total_duration = sum(level.actual_duration_s for level in levels)
    total_requests = sum(level.total_requests for level in levels)

    # Baseline from first level
    baseline = levels[0]
    baseline_p95 = baseline.latency_p95_ms
    baseline_goodput = baseline.goodput_rps

    # Find peak goodput
    peak_level = max(levels, key=lambda x: x.goodput_rps)
    peak_goodput = peak_level.goodput_rps
    peak_users = peak_level.concurrent_users

    # Find saturation point (429 > 5%)
    saturation_users = None
    saturation_idx = None
    for i, level in enumerate(levels):
        if level.rate_limited_rate > 0.05:
            saturation_users = level.concurrent_users
            saturation_idx = i
            break

    # Find latency knee (p95 > 2x baseline)
    latency_knee_users = None
    latency_knee_idx = None
    if baseline_p95 > 0:
        for i, level in enumerate(levels):
            if level.latency_p95_ms > baseline_p95 * 2:
                latency_knee_users = level.concurrent_users
                latency_knee_idx = i
                break

    # Find error threshold (errors > 1%)
    error_threshold_users = None
    error_threshold_idx = None
    for i, level in enumerate(levels):
        if level.error_rate > 0.01:
            error_threshold_users = level.concurrent_users
            error_threshold_idx = i
            break

    # Compute operational max: one level before the first limit is hit
    # Find the minimum index where a limit was hit
    limit_indices = [
        idx for idx in [saturation_idx, latency_knee_idx, error_threshold_idx] if idx is not None
    ]

    recommended_max_users = None
    recommended_goodput_rps = 0.0

    if limit_indices:
        first_limit_idx = min(limit_indices)
        # Recommend one level before (or the first level if limit hit at level 0)
        safe_idx = max(0, first_limit_idx - 1)
        safe_level = levels[safe_idx]
        recommended_max_users = safe_level.concurrent_users
        recommended_goodput_rps = safe_level.goodput_rps
    else:
        # No limits hit - can recommend highest tested level
        # But note this means we haven't found the ceiling yet
        recommended_max_users = levels[-1].concurrent_users
        recommended_goodput_rps = levels[-1].goodput_rps

    return SweepResults(
        levels=levels,
        total_duration_s=total_duration,
        total_requests=total_requests,
        saturation_users=saturation_users,
        latency_knee_users=latency_knee_users,
        error_threshold_users=error_threshold_users,
        baseline_p95_ms=baseline_p95,
        baseline_goodput_rps=baseline_goodput,
        peak_goodput_rps=peak_goodput,
        peak_goodput_users=peak_users,
        recommended_max_users=recommended_max_users,
        recommended_goodput_rps=recommended_goodput_rps,
    )


async def _wait_for_server_drain(base_url: str, timeout_s: float = 5.0) -> None:
    """Wait for server active_scans to drain to 0."""
    async with httpx.AsyncClient(timeout=2.0) as client:
        start = time.perf_counter()
        while time.perf_counter() - start < timeout_s:
            try:
                resp = await client.get(f"{base_url}/metrics")
                if resp.status_code == 200:
                    metrics = resp.json()
                    active = metrics.get("resource_limits", {}).get("active_scans", 0)
                    if active == 0:
                        return
            except Exception:
                pass
            await asyncio.sleep(0.2)


async def run_sweep(config: SweepConfig, tables_info: list[dict]) -> SweepResults:
    """Run the capacity sweep."""
    load_levels = config.get_load_levels()
    level_results: list[LevelMetrics] = []

    print(f"\n  Running capacity sweep with {len(load_levels)} levels")
    print(f"  Load levels: {load_levels}")
    print(f"  Duration per level: {config.duration_per_level_s}s")

    for i, num_users in enumerate(load_levels):
        print(f"\n  --- Level {i + 1}/{len(load_levels)}: {num_users} users ---")

        driver = LoadDriver(config, tables_info, num_users)
        # Print driver's actual computed mix (after adjustments)
        print(
            f"  User mix: {driver.dashboard_users} dashboard, "
            f"{driver.analyst_users} analyst, {driver.bulk_users} bulk"
        )

        await driver.start()

        # Warmup (don't hard-cutoff to let requests complete)
        print(f"  Warming up ({config.warmup_s}s)...")
        await driver.run_level(config.warmup_s, hard_cutoff=False)

        # Drain step: wait for server to clear any residual active scans
        print("  Draining...")
        await _wait_for_server_drain(config.base_url)

        # Clear warmup results for measurement
        driver.results = []
        driver.metrics_samples = []
        driver.attempted_requests = 0

        # Measurement period (hard cutoff for precise timing)
        print(f"  Measuring ({config.duration_per_level_s}s)...")
        results, actual_duration, start_time, end_time, attempted = await driver.run_level(
            config.duration_per_level_s, hard_cutoff=True
        )

        await driver.stop()

        # Compute metrics
        metrics = compute_level_metrics(
            level_num=i + 1,
            num_users=num_users,
            target_duration_s=config.duration_per_level_s,
            actual_duration_s=actual_duration,
            results=results,
            metrics_samples=driver.metrics_samples,
            level_start_time=start_time,
            level_end_time=end_time,
            attempted_requests=attempted,
        )
        level_results.append(metrics)

        # Print summary with new metrics
        print(f"  Results (actual duration: {actual_duration:.1f}s):")
        print(
            f"    Offered: {metrics.offered_rps:.1f} req/s | "
            f"Goodput: {metrics.goodput_rps:.1f} req/s"
        )
        print(
            f"    Success: {metrics.success_2xx}/{metrics.total_requests} "
            f"({metrics.success_rate * 100:.1f}%)"
        )
        print(f"    429s: {metrics.rate_limited_429} ({metrics.rate_limited_rate * 100:.1f}%)")
        error_total = (
            metrics.timeout_errors
            + metrics.connection_errors
            + metrics.cancelled_errors
            + metrics.other_errors
        )
        print(
            f"    Errors: {error_total} ({metrics.error_rate * 100:.2f}%) "
            f"[timeout:{metrics.timeout_errors} conn:{metrics.connection_errors} "
            f"cancel:{metrics.cancelled_errors} other:{metrics.other_errors}]"
        )
        print(f"    p95 latency: {metrics.latency_p95_ms:.1f}ms total")
        print(
            f"      POST:{metrics.post_p95_ms:.1f}ms "
            f"GET:{metrics.get_p95_ms:.1f}ms DEL:{metrics.delete_p95_ms:.1f}ms"
        )

        # Cooldown between levels
        if i < len(load_levels) - 1:
            print(f"  Cooling down ({config.cooldown_s}s)...")
            await asyncio.sleep(config.cooldown_s)

    return analyze_sweep(level_results)


# =============================================================================
# Output
# =============================================================================


def print_results(results: SweepResults):
    """Print sweep results."""
    print("\n" + "=" * 80)
    print("CAPACITY SWEEP RESULTS")
    print("=" * 80)

    print(f"\nTotal duration: {results.total_duration_s / 60:.1f} minutes")
    print(f"Total requests: {results.total_requests:,}")

    # Capacity curve table with offered vs goodput
    print("\n" + "-" * 80)
    print("CAPACITY CURVE (offered vs goodput)")
    print("-" * 80)
    print(
        f"{'Users':>6} | {'Offered':>10} | {'Goodput':>10} | {'p95 (ms)':>10} | "
        f"{'Success':>8} | {'429s':>6} | {'Err':>6}"
    )
    print("-" * 80)

    for level in results.levels:
        print(
            f"{level.concurrent_users:>6} | "
            f"{level.offered_rps:>8.1f}/s | "
            f"{level.goodput_rps:>8.1f}/s | "
            f"{level.latency_p95_ms:>10.1f} | "
            f"{level.success_rate * 100:>7.1f}% | "
            f"{level.rate_limited_rate * 100:>5.1f}% | "
            f"{level.error_rate * 100:>5.2f}%"
        )

    # Per-phase latency breakdown
    print("\n" + "-" * 80)
    print("PER-PHASE p95 LATENCY (ms)")
    print("-" * 80)
    print(f"{'Users':>6} | {'Total':>10} | {'POST':>10} | {'GET':>10} | {'DELETE':>10}")
    print("-" * 80)
    for level in results.levels:
        print(
            f"{level.concurrent_users:>6} | "
            f"{level.latency_p95_ms:>10.1f} | "
            f"{level.post_p95_ms:>10.1f} | "
            f"{level.get_p95_ms:>10.1f} | "
            f"{level.delete_p95_ms:>10.1f}"
        )

    # Key findings
    print("\n" + "-" * 80)
    print("KEY FINDINGS")
    print("-" * 80)

    print(f"\nBaseline (at {results.levels[0].concurrent_users} users):")
    print(f"  p95 latency: {results.baseline_p95_ms:.1f}ms")
    print(f"  Goodput:     {results.baseline_goodput_rps:.1f} req/s")

    print("\nPeak performance:")
    print(
        f"  Goodput: {results.peak_goodput_rps:.1f} req/s (at {results.peak_goodput_users} users)"
    )

    print("\nCapacity limits:")
    if results.saturation_users:
        print(f"  Saturation (429 > 5%):      {results.saturation_users} users")
    else:
        print("  Saturation (429 > 5%):      Not reached")

    if results.latency_knee_users:
        print(f"  Latency knee (p95 > 2x):    {results.latency_knee_users} users")
    else:
        print("  Latency knee (p95 > 2x):    Not reached")

    if results.error_threshold_users:
        print(f"  Error threshold (err > 1%): {results.error_threshold_users} users")
    else:
        print("  Error threshold (err > 1%): Not reached")

    # Operational recommendation
    print("\n" + "-" * 80)
    print("OPERATIONAL RECOMMENDATION")
    print("-" * 80)
    if results.recommended_max_users:
        any_limit_hit = (
            results.saturation_users or results.latency_knee_users or results.error_threshold_users
        )
        if any_limit_hit:
            print(f"\n  Recommended max users: {results.recommended_max_users}")
            print(f"  Expected goodput:      {results.recommended_goodput_rps:.1f} req/s")
            print("\n  This is one level below the first capacity limit.")
        else:
            print("\n  No capacity limits reached in this sweep.")
            print(
                f"  Highest tested: {results.recommended_max_users} users "
                f"({results.recommended_goodput_rps:.1f} req/s)"
            )
            print("  Consider running with higher --max-users to find the ceiling.")
    else:
        print("\n  Insufficient data to make recommendation.")

    # Per-tier breakdown for each level (total p95)
    print("\n" + "-" * 80)
    print("BY USER TYPE - TOTAL p95 (ms)")
    print("-" * 80)
    print(f"{'Users':>6} | {'Dashboard':>12} | {'Analyst':>12} | {'Bulk':>12}")
    print("-" * 80)
    for level in results.levels:
        print(
            f"{level.concurrent_users:>6} | "
            f"{level.dashboard_p95_ms:>10.1f}ms | "
            f"{level.analyst_p95_ms:>10.1f}ms | "
            f"{level.bulk_p95_ms:>10.1f}ms"
        )

    # Per-tier POST p95 (reveals QoS queueing bottlenecks)
    print("\n" + "-" * 80)
    print("BY USER TYPE - POST p95 (ms) [reveals QoS queue delays]")
    print("-" * 80)
    print(f"{'Users':>6} | {'Dashboard':>12} | {'Analyst':>12} | {'Bulk':>12}")
    print("-" * 80)
    for level in results.levels:
        print(
            f"{level.concurrent_users:>6} | "
            f"{level.dashboard_post_p95_ms:>10.1f}ms | "
            f"{level.analyst_post_p95_ms:>10.1f}ms | "
            f"{level.bulk_post_p95_ms:>10.1f}ms"
        )

    # Per-tier GET p95 (reveals streaming/cache bottlenecks)
    print("\n" + "-" * 80)
    print("BY USER TYPE - GET p95 (ms) [reveals streaming bottlenecks]")
    print("-" * 80)
    print(f"{'Users':>6} | {'Dashboard':>12} | {'Analyst':>12} | {'Bulk':>12}")
    print("-" * 80)
    for level in results.levels:
        print(
            f"{level.concurrent_users:>6} | "
            f"{level.dashboard_get_p95_ms:>10.1f}ms | "
            f"{level.analyst_get_p95_ms:>10.1f}ms | "
            f"{level.bulk_get_p95_ms:>10.1f}ms"
        )

    # ASCII chart
    print("\n" + "-" * 80)
    print("GOODPUT vs LOAD (ASCII)")
    print("-" * 80)
    _print_ascii_chart(results)

    print("\n" + "=" * 80)


def _print_ascii_chart(results: SweepResults):
    """Print simple ASCII chart of goodput vs load."""
    if not results.levels:
        return

    max_goodput = max(level.goodput_rps for level in results.levels)
    if max_goodput == 0:
        return

    chart_width = 50

    for level in results.levels:
        bar_len = int(level.goodput_rps / max_goodput * chart_width)
        bar = "█" * bar_len

        # Mark special points
        marker = ""
        if results.saturation_users and level.concurrent_users >= results.saturation_users:
            marker = " ← 429s"
        elif results.latency_knee_users and level.concurrent_users >= results.latency_knee_users:
            marker = " ← latency"

        print(f"{level.concurrent_users:>4}u | {bar}{marker}")


# =============================================================================
# Main
# =============================================================================


async def main():
    parser = argparse.ArgumentParser(description="Capacity sweep benchmark for Strata")
    parser.add_argument("--levels", type=int, default=6, help="Number of load levels")
    parser.add_argument("--min-users", type=int, default=5, help="Minimum concurrent users")
    parser.add_argument("--max-users", type=int, default=60, help="Maximum concurrent users")
    parser.add_argument("--duration", type=float, default=120, help="Seconds per load level")
    parser.add_argument("--warmup", type=float, default=15, help="Warmup seconds per level")
    parser.add_argument("--no-server", action="store_true", help="Don't start server")
    parser.add_argument("--base-url", type=str, help="Server base URL")
    parser.add_argument("--keep-dirs", action="store_true", help="Keep temp directories")
    parser.add_argument("--dry-run", action="store_true", help="Quick validation run")
    args = parser.parse_args()

    # Build config
    config = SweepConfig(
        num_levels=args.levels,
        min_users=args.min_users,
        max_users=args.max_users,
        duration_per_level_s=args.duration,
        warmup_s=args.warmup,
        start_server=not args.no_server,
        keep_dirs=args.keep_dirs,
        dry_run=args.dry_run,
    )

    if args.base_url:
        config.base_url = args.base_url

    if args.dry_run:
        config.num_levels = 3
        config.min_users = 5
        config.max_users = 15
        config.duration_per_level_s = 30
        config.warmup_s = 5
        config.cooldown_s = 5
        config.num_tables = 3
        config.rows_per_table = 5000

    # Setup directories
    if config.start_server:
        temp_dir = Path(tempfile.mkdtemp(prefix="strata_sweep_"))
        config.warehouse_dir = temp_dir / "warehouse"
        config.cache_dir = temp_dir / "cache"
    else:
        temp_dir = None

    server = None
    try:
        # Generate warehouse
        if config.start_server:
            print("Generating test warehouse...")
            warehouse_info = generate_warehouse(config)
            tables_info = warehouse_info["tables"]

            # Start server
            port = find_free_port()
            config.server_port = port
            config.base_url = f"http://{config.server_host}:{port}"

            print(f"Starting server on port {port}...")
            server = ServerProcess(
                host=config.server_host,
                port=port,
                cache_dir=config.cache_dir,
                max_cache_size_bytes=config.cache_size_bytes,
                log_dir=config.results_dir,
                interactive_slots=config.interactive_slots,
                bulk_slots=config.bulk_slots,
            )
            server.start()
        else:
            # Connect to external server - need tables info
            print("Connecting to external server - using dummy tables info")
            tables_info = [
                {"name": f"table_{i}", "uri": f"table_{i}", "rows": 10000}
                for i in range(config.num_tables)
            ]

        # Run sweep
        print("\nStarting capacity sweep...")
        results = await run_sweep(config, tables_info)

        # Print results
        print_results(results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = config.results_dir / f"capacity_sweep_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        print(f"\nResults saved to: {results_file}")

    finally:
        if server:
            print("\nStopping server...")
            server.stop()

        if temp_dir and not config.keep_dirs:
            print("Cleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
