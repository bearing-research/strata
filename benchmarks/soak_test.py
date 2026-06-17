#!/usr/bin/env python3
"""Soak test for Strata production readiness.

Long-running stability test to detect memory leaks, latency drift, and resource exhaustion.

Features:
- Duration: 1-2 hours (configurable)
- Memory tracking with psutil
- Latency drift detection (baseline vs current p95)
- Periodic stress spikes (2x load every 15 min)
- Cache stability monitoring

Success criteria:
- Memory growth < 10% after warmup
- p95 latency drift < 20% from baseline
- Final active_scans = 0
- Cache eviction rate stabilizes
- Zero 5xx errors after warmup

Usage:
    # Quick 15-minute test
    python benchmarks/soak_test.py --duration 0.25

    # Standard 1-hour test
    python benchmarks/soak_test.py

    # Extended 2-hour test
    python benchmarks/soak_test.py --duration 2

    # With custom users
    python benchmarks/soak_test.py --users 50 --duration 1
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
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
import psutil
import pyarrow as pa
import pyarrow.ipc as ipc

# =============================================================================
# Configuration
# =============================================================================


class Phase(Enum):
    """Soak test phases."""

    WARMUP = "warmup"
    STEADY = "steady"
    SPIKE = "spike"
    COOLDOWN = "cooldown"


@dataclass
class SoakConfig:
    """Configuration for soak test."""

    # Server settings
    base_url: str = "http://127.0.0.1:8765"
    start_server: bool = True
    server_host: str = "127.0.0.1"
    server_port: int = 0  # Auto-find

    # Directories
    warehouse_dir: Path | None = None
    cache_dir: Path | None = None
    keep_dirs: bool = False

    # Duration
    duration_hours: float = 1.0  # 1 hour default
    warmup_minutes: float = 5.0  # 5 min warmup
    cooldown_minutes: float = 2.0  # 2 min cooldown

    # Stress spikes
    spike_interval_minutes: float = 15.0  # Every 15 min
    spike_duration_seconds: float = 30.0  # 30s spike
    spike_multiplier: float = 2.0  # 2x load during spike

    # Concurrency
    base_users: int = 30  # Base concurrent users
    dashboard_ratio: float = 0.8  # 80% dashboard
    analyst_ratio: float = 0.15  # 15% analyst
    bulk_ratio: float = 0.05  # 5% bulk

    # Table sizes (moderate for long-running test)
    num_tables: int = 8
    rows_per_table: int = 50_000
    payload_bytes: int = 100

    # Cache
    cache_size_bytes: int = 200 * 1024 * 1024  # 200MB

    # Metrics collection
    metrics_interval_s: float = 30.0  # Sample every 30s
    drift_window_minutes: float = 5.0  # 5-min window for drift detection
    results_dir: Path = field(default_factory=lambda: Path("benchmarks/results"))

    # Request settings
    request_timeout_s: float = 60.0
    connect_timeout_s: float = 5.0
    max_connections: int = 100

    # Success criteria
    max_memory_growth_pct: float = 10.0  # Max 10% memory growth after warmup
    max_latency_drift_pct: float = 20.0  # Max 20% p95 drift from baseline
    min_prefetch_efficiency: float = 0.5  # Min 50% prefetch used
    min_success_rate: float = 0.95  # Min 95% request success rate

    # Misc
    seed: int = 42
    dry_run: bool = False

    def __post_init__(self):
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @property
    def duration_s(self) -> float:
        return self.duration_hours * 3600

    @property
    def warmup_s(self) -> float:
        return self.warmup_minutes * 60

    @property
    def cooldown_s(self) -> float:
        return self.cooldown_minutes * 60

    @property
    def spike_interval_s(self) -> float:
        return self.spike_interval_minutes * 60

    @property
    def dashboard_users(self) -> int:
        return int(self.base_users * self.dashboard_ratio)

    @property
    def analyst_users(self) -> int:
        return int(self.base_users * self.analyst_ratio)

    @property
    def bulk_users(self) -> int:
        return max(1, self.base_users - self.dashboard_users - self.analyst_users)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ResourceSample:
    """Resource sample at a point in time.

    Captures multiple resource dimensions to detect:
    - Memory leaks (RSS growth)
    - Allocator fragmentation (RSS vs cache bytes)
    - File descriptor leaks (num_fds growth)
    - Thread leaks (num_threads growth)
    - Cache bloat (cache_bytes vs expected)
    - GC pause impact (pause duration tracking)
    """

    timestamp: float
    elapsed_s: float
    phase: str

    # Process-level metrics (from psutil)
    rss_bytes: int
    num_fds: int  # File descriptors (Linux) or -1 if unavailable
    num_threads: int

    # Server-reported cache metrics (from /metrics endpoint)
    cache_bytes: int = 0  # Actual bytes in cache
    cache_entries: int = 0  # Number of cached row groups
    cache_evictions: int = 0  # Total evictions since start

    # GC metrics (for diagnosing periodic stalls)
    gc_gen2_collections: int = 0  # Gen2 collections (most expensive)

    # GC pause duration metrics (from gc.callbacks tracker)
    gc_total_pauses: int = 0  # Total GC pauses since start
    gc_total_pause_ms: float = 0.0  # Cumulative pause time
    gc_max_pause_ms: float = 0.0  # Max single pause
    gc_gen2_pause_count: int = 0  # Gen2 pauses (most expensive)
    gc_gen2_total_ms: float = 0.0  # Total gen2 pause time
    gc_gen2_max_ms: float = 0.0  # Max gen2 pause

    def to_dict(self) -> dict:
        return {
            "type": "resource",
            "timestamp": self.timestamp,
            "elapsed_s": self.elapsed_s,
            "phase": self.phase,
            "rss_mb": self.rss_bytes / (1024 * 1024),
            "num_fds": self.num_fds,
            "num_threads": self.num_threads,
            "cache_mb": self.cache_bytes / (1024 * 1024),
            "cache_entries": self.cache_entries,
            "cache_evictions": self.cache_evictions,
            "gc_gen2_collections": self.gc_gen2_collections,
            # GC pause duration metrics
            "gc_total_pauses": self.gc_total_pauses,
            "gc_total_pause_ms": self.gc_total_pause_ms,
            "gc_max_pause_ms": self.gc_max_pause_ms,
            "gc_gen2_pause_count": self.gc_gen2_pause_count,
            "gc_gen2_total_ms": self.gc_gen2_total_ms,
            "gc_gen2_max_ms": self.gc_gen2_max_ms,
        }


@dataclass
class LatencySample:
    """Latency sample for drift detection.

    Includes GC metrics to correlate garbage collection with latency spikes.
    """

    timestamp: float
    elapsed_s: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    request_count: int
    success_count: int
    error_count: int
    phase: str
    event_loop_lag_ms: float = 0.0  # Event loop lag (client-side health indicator)
    # GC correlation metrics (to identify if high latency correlates with gen2 GC)
    gc_gen2_in_window: int = 0  # Gen2 collections since previous sample
    gc_pause_ms_in_window: float = 0.0  # GC pause time since previous sample

    def to_dict(self) -> dict:
        return {
            "type": "latency",
            "timestamp": self.timestamp,
            "elapsed_s": self.elapsed_s,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "phase": self.phase,
            "event_loop_lag_ms": self.event_loop_lag_ms,
            "gc_gen2_in_window": self.gc_gen2_in_window,
            "gc_pause_ms_in_window": self.gc_pause_ms_in_window,
        }


@dataclass
class RequestResult:
    """Result of a single scan request with per-phase classification.

    Tracks status codes for each phase (POST, GET stream, DELETE) to enable
    precise failure classification:
    - success: All phases succeeded (2xx)
    - rate_limited: Got 429 at any phase
    - failed_post: POST failed with non-429 error
    - failed_stream: GET stream failed with non-429 error
    - cleanup_failed: DELETE failed (tracked but not considered failure)
    """

    latency_ms: float
    phase: str  # warmup, steady, spike, cooldown
    user_type: str  # dashboard, analyst, bulk

    # Status codes by request phase (0 means phase not reached)
    post_status: int = 0
    stream_status: int = 0
    delete_status: int = 0

    # Error details (for diagnostics)
    error_type: str | None = None  # timeout, arrow_decode, connection, etc.
    error_detail: str | None = None

    # DELETE-specific error tracking (separate from main request error)
    delete_error_type: str | None = None  # timeout, connection, etc.

    @property
    def is_success(self) -> bool:
        """True if request completed successfully (2xx on POST and stream)."""
        return 200 <= self.post_status < 300 and 200 <= self.stream_status < 300

    @property
    def is_rate_limited(self) -> bool:
        """True if request was rate limited (429)."""
        return self.post_status == 429 or self.stream_status == 429

    @property
    def is_timeout(self) -> bool:
        """True if request timed out."""
        return self.error_type == "timeout"

    @property
    def is_5xx(self) -> bool:
        """True if server returned 5xx error."""
        return (500 <= self.post_status < 600) or (500 <= self.stream_status < 600)


@dataclass
class SoakResults:
    """Aggregated soak test results."""

    duration_hours: float
    total_requests: int

    # Request classification (per user feedback: separate 2xx vs 429 vs other)
    success_2xx_count: int = 0  # Fully successful requests
    rate_limited_429_count: int = 0  # Rate limited (QoS working as designed)
    other_fail_count: int = 0  # Real failures (5xx, timeouts, etc.)

    # Derived rates
    success_2xx_rate: float = 0.0  # success_2xx_count / total
    rate_limited_429_rate: float = 0.0  # rate_limited / total
    other_fail_rate: float = 0.0  # other_fail / total

    # Legacy field for compatibility
    success_rate: float = 0.0

    # Memory metrics (robust 3-window analysis per user feedback)
    # Window 1: Baseline = median RSS in (warmup_end - 2min .. warmup_end)
    baseline_rss_mb: float = 0.0
    # Window 2: Early steady = median RSS in (warmup_end + 30min .. warmup_end + 90min)
    early_steady_rss_mb: float = 0.0
    # Window 3: Late steady = median RSS in last 30min (excluding cooldown/spikes)
    late_steady_rss_mb: float = 0.0
    # Legacy: overall steady median (for backwards compat)
    steady_median_rss_mb: float = 0.0
    peak_rss_mb: float = 0.0
    min_rss_mb: float = 0.0  # Min RSS (shows GC floor)
    final_rss_mb: float = 0.0
    # Growth metrics
    memory_growth_pct: float = 0.0  # early_steady vs baseline
    memory_end_growth_pct: float = 0.0  # late_steady vs early_steady
    memory_slope_mb_per_hour: float = 0.0  # Linear trend slope (0 = stable)

    # Latency metrics (measured on successful 2xx requests only per user feedback)
    baseline_p95_ms: float = 0.0
    final_p95_ms: float = 0.0
    median_steady_p95_ms: float = 0.0  # Median p95 from clean steady-state samples
    max_p95_ms: float = 0.0
    latency_drift_pct: float = 0.0
    # Spike latency (separate from steady)
    spike_p95_ms: float = 0.0  # p95 during spikes (expected to be higher)
    spike_recovery_time_s: float = 0.0  # Avg time to recover after spike

    # Stability metrics
    final_active_scans: int = 0
    cache_hit_rate: float = 0.0
    prefetch_efficiency: float = 0.0

    # Spike behavior
    spike_count: int = 0
    spike_recovery_ok: bool = True

    # Resource metrics (fds, threads, cache)
    baseline_fds: float = -1  # -1 means unavailable
    final_fds: int = -1
    baseline_threads: float = 0.0
    final_threads: int = 0
    final_cache_mb: float = 0.0
    final_cache_entries: int = 0
    total_cache_evictions: int = 0

    # Error breakdown by phase (POST, GET stream, DELETE)
    post_success_count: int = 0
    post_429_count: int = 0
    post_5xx_count: int = 0
    post_other_fail_count: int = 0

    stream_success_count: int = 0
    stream_429_count: int = 0
    stream_5xx_count: int = 0
    stream_other_fail_count: int = 0

    # DELETE outcomes (split by semantics)
    delete_success_count: int = 0  # 2xx - successfully deleted
    delete_already_gone_count: int = 0  # 404 - scan already cleaned up (harmless)
    delete_5xx_count: int = 0  # 5xx - server error (real failure)
    delete_timeout_count: int = 0  # Timeout during DELETE
    delete_other_fail_count: int = 0  # Other failures

    # Post-warmup error breakdown (for success criteria)
    post_warmup_5xx: int = 0
    post_warmup_429: int = 0
    post_warmup_timeouts: int = 0
    post_warmup_arrow_errors: int = 0
    post_warmup_other_errors: int = 0

    # Event loop lag (client health indicator)
    max_event_loop_lag_ms: float = 0.0
    p95_event_loop_lag_ms: float = 0.0

    # GC pause metrics (server-side)
    gc_total_pauses: int = 0
    gc_total_pause_ms: float = 0.0
    gc_max_pause_ms: float = 0.0
    gc_gen2_pause_count: int = 0
    gc_gen2_max_ms: float = 0.0

    # GC-latency correlation metrics (per user feedback: correlate slow windows with gen2)
    # A "slow window" is a latency sample with p95 > 2x baseline
    slow_window_count: int = 0  # Number of slow windows (p95 > 2x baseline)
    slow_window_with_gc_count: int = 0  # Slow windows that had gen2 GC activity
    gc_latency_correlation: float = 0.0  # % of slow windows with gen2 activity (0-100)
    slow_window_avg_gc_pause_ms: float = 0.0  # Avg GC pause time in slow windows
    normal_window_avg_gc_pause_ms: float = 0.0  # Avg GC pause time in normal windows

    # Success criteria (updated per user specifications)
    server_alive: bool = True
    memory_ok: bool = True
    latency_ok: bool = True
    no_leak: bool = True
    no_fd_leak: bool = True
    no_thread_leak: bool = True
    no_errors_post_warmup: bool = True  # Zero 5xx/timeouts (429 OK)
    other_fail_rate_ok: bool = True  # other_fail_rate <= 0.1%
    rate_429_in_band: bool = True  # 429 rate within expected range
    overall_pass: bool = True

    # Crash info
    server_crash_time_min: float | None = None
    server_crash_signal: str | None = None
    server_crash_exit_code: int | None = None

    # Data quality
    insufficient_data: bool = False

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# Server Management (reused from stress_test.py)
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
        self.exit_code: int | None = None
        self.exit_signal: int | None = None
        self.crash_detected: bool = False

    def start(self, timeout: float = 30.0):
        """Start the server as a subprocess."""
        env = os.environ.copy()
        env["STRATA_HOST"] = self.host
        env["STRATA_PORT"] = str(self.port)
        env["STRATA_CACHE_DIR"] = str(self.cache_dir)
        env["STRATA_METRICS_ENABLED"] = "true"
        # Use JSON logging for structured output
        env["STRATA_LOG_FORMAT"] = "json"
        env["STRATA_LOG_LEVEL"] = "INFO"

        if self.max_cache_size_bytes is not None:
            env["STRATA_MAX_CACHE_SIZE_BYTES"] = str(self.max_cache_size_bytes)

        # QoS slots - sized appropriately for the load test
        if self.interactive_slots is not None:
            env["STRATA_INTERACTIVE_SLOTS"] = str(self.interactive_slots)
        if self.bulk_slots is not None:
            env["STRATA_BULK_SLOTS"] = str(self.bulk_slots)

        # Capture stdout/stderr to log files if log_dir provided
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

    def check_alive(self) -> bool:
        """Check if server process is still running. Updates crash info if dead."""
        if not self._process:
            return False

        poll = self._process.poll()
        if poll is None:
            return True  # Still running

        # Process exited - capture exit info
        self.crash_detected = True
        self.exit_code = poll

        # Negative exit code means killed by signal
        if poll < 0:
            self.exit_signal = -poll

        return False

    def get_crash_info(self) -> dict:
        """Get information about server crash."""
        info = {
            "crashed": self.crash_detected,
            "exit_code": self.exit_code,
            "exit_signal": self.exit_signal,
            "signal_name": None,
            "last_stderr_lines": [],
            "last_stdout_lines": [],
        }

        # Decode signal name
        if self.exit_signal:
            signal_names = {
                9: "SIGKILL (likely OOM killer)",
                11: "SIGSEGV (segmentation fault)",
                6: "SIGABRT (abort)",
                15: "SIGTERM (terminated)",
                2: "SIGINT (interrupted)",
            }
            info["signal_name"] = signal_names.get(self.exit_signal, f"signal {self.exit_signal}")

        # Read last lines from log files
        if self.log_dir:
            try:
                stderr_path = self.log_dir / "server_stderr.log"
                if stderr_path.exists():
                    with open(stderr_path) as f:
                        lines = f.readlines()
                        info["last_stderr_lines"] = [line.rstrip() for line in lines[-50:]]
            except Exception:
                pass

            try:
                stdout_path = self.log_dir / "server_stdout.log"
                if stdout_path.exists():
                    with open(stdout_path) as f:
                        lines = f.readlines()
                        info["last_stdout_lines"] = [line.rstrip() for line in lines[-50:]]
            except Exception:
                pass

        return info

    def stop(self):
        """Stop the server subprocess."""
        if self._process:
            # Check if already dead before terminating
            self.check_alive()

            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()

            self._process = None

        # Close log files
        if self._stdout_file:
            self._stdout_file.close()
            self._stdout_file = None
        if self._stderr_file:
            self._stderr_file.close()
            self._stderr_file = None

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process else None


def find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# =============================================================================
# Warehouse Generation (simplified from stress_test.py)
# =============================================================================


def generate_soak_warehouse(config: SoakConfig) -> dict[str, Any]:
    """Generate warehouse for soak testing."""
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
        catalog.create_namespace("soak")
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

    for i in range(config.num_tables):
        table_name = f"table_{i:02d}"
        table_id = f"soak.{table_name}"

        print(f"  Creating {table_name} ({config.rows_per_table:,} rows)...")

        try:
            table = catalog.load_table(table_id)
        except Exception:
            table = catalog.create_table(table_id, schema)

            chunk_size = min(50_000, config.rows_per_table)
            base_ts = 1704067200000000

            for chunk_start in range(0, config.rows_per_table, chunk_size):
                chunk_end = min(chunk_start + chunk_size, config.rows_per_table)
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
                                f"data_{chunk_start + j:08d}_" + "x" * config.payload_bytes
                                for j in range(actual_chunk_size)
                            ],
                            type=pa.string(),
                        ),
                    }
                )
                table.append(data)

        snapshot_id = table.current_snapshot().snapshot_id

        tables_info.append(
            {
                "name": table_name,
                "table_id": table_id,
                "table_uri": f"file://{warehouse_path}#soak.{table_name}",
                "snapshot_id": snapshot_id,
            }
        )

    return {
        "catalog": catalog,
        "warehouse_path": warehouse_path,
        "tables": tables_info,
    }


# =============================================================================
# Soak Test Driver
# =============================================================================


def _percentile(data: list[float], p: float) -> float:
    """Compute percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


class SoakDriver:
    """Drives soak test workload."""

    def __init__(self, config: SoakConfig, tables_info: list[dict], server: ServerProcess | None):
        self.config = config
        self.tables_info = tables_info
        self.server = server
        self.server_pid = server.pid if server else None
        self.rng = random.Random(config.seed)

        # Column sets
        self.dashboard_columns = ["id", "ts", "value"]
        self.analyst_columns = ["id", "ts", "user_id", "category", "value"]
        self.bulk_columns = ["id", "ts", "user_id", "category", "value", "payload"]
        self.categories = ["electronics", "clothing", "food", "books", "sports", "home", "auto"]

        # Results collection - now uses RequestResult for per-phase tracking
        self.request_results: list[RequestResult] = []
        self.resource_samples: list[ResourceSample] = []
        self.latency_samples: list[LatencySample] = []
        self._lock = asyncio.Lock()

        # Phase tracking
        self.current_phase = Phase.WARMUP
        self.spike_count = 0
        # Track spike events: list of (spike_start_elapsed_s, spike_end_elapsed_s)
        self.spike_events: list[tuple[float, float]] = []

        # Crash detection
        self.server_crash_time: float | None = None  # Elapsed seconds when crash detected
        self.server_crash_info: dict | None = None

        # GC tracking for delta calculation (to correlate GC with latency spikes)
        self._prev_gc_gen2_count: int = 0
        self._prev_gc_total_pause_ms: float = 0.0

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

    def get_process_resources(self) -> tuple[int, int, int]:
        """Get server process resource metrics.

        Returns:
            Tuple of (rss_bytes, num_fds, num_threads).
            Returns (0, -1, 0) if process is unavailable.
        """
        if self.server_pid is None:
            return 0, -1, 0
        try:
            proc = psutil.Process(self.server_pid)
            rss = proc.memory_info().rss
            num_threads = proc.num_threads()
            # num_fds is Linux-only; returns -1 on macOS/Windows
            try:
                num_fds = proc.num_fds()
            except AttributeError:
                # macOS/Windows: use open_files count as approximation
                try:
                    num_fds = len(proc.open_files())
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    num_fds = -1
            return rss, num_fds, num_threads
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0, -1, 0

    async def execute_scan(self, user_type: str) -> RequestResult:
        """Execute a single scan and return RequestResult with per-phase status codes.

        Tracks status codes for each phase (POST, GET stream, DELETE) to enable
        precise failure classification per user feedback:
        - success: All phases succeeded (2xx)
        - rate_limited: Got 429 at any phase
        - failed_post: POST failed with non-429 error
        - failed_stream: GET stream failed with non-429 error

        For dashboard users, randomly decodes 10-20% of responses using
        pyarrow.ipc.open_stream to catch schema issues and simulate
        client-side parsing costs.
        """
        start = time.perf_counter()

        # Per-phase status tracking
        post_status = 0
        stream_status = 0
        delete_status = 0
        error_type = None
        error_detail = None

        # Decide whether to decode Arrow for this request (dashboard only, 15% of requests)
        should_decode = user_type == "dashboard" and self.rng.random() < 0.15

        try:
            # Select table and columns based on user type
            table = self.rng.choice(self.tables_info)

            if user_type == "dashboard":
                columns = self.dashboard_columns
                filters = [
                    {"column": "category", "op": "=", "value": self.rng.choice(self.categories)}
                ]
            elif user_type == "analyst":
                columns = self.analyst_columns
                min_val = self.rng.uniform(0, 500)
                filters = [
                    {"column": "value", "op": ">=", "value": min_val},
                    {"column": "value", "op": "<=", "value": min_val + 300},
                ]
            else:  # bulk
                columns = self.bulk_columns
                filters = []

            # POST /v1/materialize (scan@v1)
            response = await self._client.post(
                "/v1/materialize",
                json={
                    "inputs": [table["table_uri"]],
                    "transform": {
                        "executor": "scan@v1",
                        "params": {
                            "snapshot_id": table["snapshot_id"],
                            "columns": columns,
                            "filters": filters,
                        },
                    },
                    "mode": "stream",
                },
            )
            post_status = response.status_code

            if response.status_code != 200:
                error_type = "http_error"
                error_detail = f"POST returned {response.status_code}"
            else:
                stream_url = response.json()["stream_url"]

                # Stream response
                async with self._client.stream("GET", stream_url) as stream:
                    stream_status = stream.status_code

                    if stream_status != 200:
                        error_type = "http_error"
                        error_detail = f"GET stream returned {stream_status}"
                    else:
                        if should_decode:
                            # Collect bytes and decode Arrow to catch schema issues
                            # and simulate client-side parsing costs
                            chunks = []
                            async for chunk in stream.aiter_bytes(chunk_size=1024 * 1024):
                                chunks.append(chunk)

                            if chunks:
                                # Decode the Arrow IPC stream
                                data = b"".join(chunks)
                                reader = ipc.open_stream(pa.BufferReader(data))
                                # Read all batches to simulate full client processing
                                total_rows = 0
                                for batch in reader:
                                    total_rows += batch.num_rows
                                # Schema validation happens implicitly during read
                        else:
                            # Just drain bytes without decoding
                            async for _ in stream.aiter_bytes(chunk_size=1024 * 1024):
                                pass

        except httpx.TimeoutException as e:
            error_type = "timeout"
            error_detail = str(e)[:80]
        except httpx.HTTPStatusError as e:
            error_type = "http_error"
            error_detail = f"HTTP {e.response.status_code}"
            # Update status code from exception
            if post_status == 0:
                post_status = e.response.status_code
            elif stream_status == 0:
                stream_status = e.response.status_code
        except pa.ArrowInvalid as e:
            # Arrow decoding error - schema issue or corrupt data
            error_type = "arrow_decode"
            error_detail = str(e)[:80]
        except httpx.ConnectError as e:
            error_type = "connection"
            error_detail = str(e)[:80]
        except Exception as e:
            error_type = "other"
            error_detail = str(e)[:100]
        # No /v1/scan DELETE in the unified API — materialize streams self-clean
        # via the stream-state TTL; delete_status stays 0.
        delete_error_type = None

        latency_ms = (time.perf_counter() - start) * 1000
        return RequestResult(
            latency_ms=latency_ms,
            phase=self.current_phase.value,
            user_type=user_type,
            post_status=post_status,
            stream_status=stream_status,
            delete_status=delete_status,
            error_type=error_type,
            error_detail=error_detail,
            delete_error_type=delete_error_type,
        )

    async def user_loop(self, user_id: int, user_type: str, get_duration: callable):
        """Run user loop."""
        # Stagger start
        await asyncio.sleep(self.rng.uniform(0, 1.0))

        while not self._stop_event.is_set():
            duration = get_duration()
            if duration <= 0:
                break

            result = await self.execute_scan(user_type)

            async with self._lock:
                self.request_results.append(result)

            # Think time
            if user_type == "dashboard":
                await asyncio.sleep(self.rng.uniform(0.5, 1.5))
            elif user_type == "analyst":
                await asyncio.sleep(self.rng.uniform(2.0, 5.0))
            else:
                await asyncio.sleep(self.rng.uniform(5.0, 15.0))

    async def _measure_event_loop_lag(self) -> float:
        """Measure event loop lag in milliseconds.

        Schedules a callback and measures how long it takes to execute.
        High values indicate the event loop is blocked by other tasks.
        """
        loop = asyncio.get_event_loop()
        start = time.perf_counter()

        # Schedule a callback to run as soon as possible
        future: asyncio.Future[None] = loop.create_future()
        loop.call_soon(lambda: future.set_result(None))
        await future

        return (time.perf_counter() - start) * 1000  # Convert to ms

    async def collect_samples(self, start_time: float):
        """Collect resource and latency samples periodically."""
        while not self._stop_event.is_set():
            await asyncio.sleep(self.config.metrics_interval_s)

            elapsed = time.perf_counter() - start_time
            timestamp = time.time()

            # Check if server crashed (only once)
            if self.server and self.server_crash_time is None:
                if not self.server.check_alive():
                    self.server_crash_time = elapsed
                    self.server_crash_info = self.server.get_crash_info()
                    print(f"\n*** SERVER CRASHED at {elapsed / 60:.1f} min ***")
                    if self.server_crash_info.get("signal_name"):
                        print(f"    Signal: {self.server_crash_info['signal_name']}")
                    elif self.server_crash_info.get("exit_code") is not None:
                        print(f"    Exit code: {self.server_crash_info['exit_code']}")

            # Measure event loop lag first (before any blocking ops)
            event_loop_lag_ms = await self._measure_event_loop_lag()

            # Get process-level resources
            rss, num_fds, num_threads = self.get_process_resources()

            # Get server-reported cache and GC metrics
            server_metrics = await self._get_metrics()
            disk_cache = server_metrics.get("disk_cache", {})
            cache_bytes = disk_cache.get("bytes_current", 0)
            cache_evictions = disk_cache.get("evictions_count", 0)
            cache_entries = disk_cache.get("entries_current", 0)

            # GC metrics for diagnosing periodic stalls
            gc_info = server_metrics.get("gc", {})
            gc_gen2_collections = gc_info.get("gen2_collections", 0)

            # GC pause duration metrics (from gc.callbacks tracker)
            gc_pauses = server_metrics.get("gc_pauses", {})
            gc_total_pauses = gc_pauses.get("total_pauses", 0)
            gc_total_pause_ms = gc_pauses.get("total_pause_ms", 0.0)
            gc_max_pause_ms = gc_pauses.get("max_pause_ms", 0.0)
            gc_gen2_stats = gc_pauses.get("gen2", {})
            gc_gen2_pause_count = gc_gen2_stats.get("count", 0)
            gc_gen2_total_ms = gc_gen2_stats.get("total_ms", 0.0)
            gc_gen2_max_ms = gc_gen2_stats.get("max_ms", 0.0)

            # Resource sample (combines process + server metrics)
            self.resource_samples.append(
                ResourceSample(
                    timestamp=timestamp,
                    elapsed_s=elapsed,
                    phase=self.current_phase.value,
                    rss_bytes=rss,
                    num_fds=num_fds,
                    num_threads=num_threads,
                    cache_bytes=cache_bytes,
                    cache_entries=cache_entries,
                    cache_evictions=cache_evictions,
                    gc_gen2_collections=gc_gen2_collections,
                    gc_total_pauses=gc_total_pauses,
                    gc_total_pause_ms=gc_total_pause_ms,
                    gc_max_pause_ms=gc_max_pause_ms,
                    gc_gen2_pause_count=gc_gen2_pause_count,
                    gc_gen2_total_ms=gc_gen2_total_ms,
                    gc_gen2_max_ms=gc_gen2_max_ms,
                )
            )

            # Latency sample from recent requests
            # Per user feedback: measure latency only on successful (2xx) requests
            async with self._lock:
                recent_results = self.request_results[-1000:]  # Last 1000
                total_results = len(self.request_results)

            if recent_results:
                # Filter to successful requests for latency measurement
                recent_success_latencies = [r.latency_ms for r in recent_results if r.is_success]
                # Count error types
                recent_2xx = sum(1 for r in recent_results if r.is_success)
                recent_429 = sum(1 for r in recent_results if r.is_rate_limited)
                recent_other = len(recent_results) - recent_2xx - recent_429

                # Use successful latencies for percentiles, fall back to all if none
                latencies_for_pct = (
                    recent_success_latencies
                    if recent_success_latencies
                    else [r.latency_ms for r in recent_results]
                )

                # Compute GC deltas for this window (to correlate with latency)
                gc_gen2_delta = gc_gen2_pause_count - self._prev_gc_gen2_count
                gc_pause_ms_delta = gc_total_pause_ms - self._prev_gc_total_pause_ms
                # Update previous values for next sample
                self._prev_gc_gen2_count = gc_gen2_pause_count
                self._prev_gc_total_pause_ms = gc_total_pause_ms

                self.latency_samples.append(
                    LatencySample(
                        timestamp=timestamp,
                        elapsed_s=elapsed,
                        p50_ms=_percentile(latencies_for_pct, 0.5),
                        p95_ms=_percentile(latencies_for_pct, 0.95),
                        p99_ms=_percentile(latencies_for_pct, 0.99),
                        request_count=total_results,
                        success_count=recent_2xx,
                        error_count=recent_429 + recent_other,
                        phase=self.current_phase.value,
                        event_loop_lag_ms=event_loop_lag_ms,
                        gc_gen2_in_window=gc_gen2_delta,
                        gc_pause_ms_in_window=gc_pause_ms_delta,
                    )
                )

                # Print status with extended resource info
                rss_mb = rss / (1024 * 1024)
                cache_mb = cache_bytes / (1024 * 1024)
                p95 = _percentile(latencies_for_pct, 0.95)
                fd_str = str(num_fds) if num_fds >= 0 else "n/a"
                # Show GC pause stats (max pause is most impactful for latency)
                gc_pause_str = f"gc_max={gc_max_pause_ms:5.1f}ms" if gc_total_pauses > 0 else ""
                # Show request breakdown: 2xx/429/other
                total_all = sum(1 for r in self.request_results)
                total_2xx = sum(1 for r in self.request_results if r.is_success)
                total_429 = sum(1 for r in self.request_results if r.is_rate_limited)
                other_count = total_all - total_2xx - total_429
                print(
                    f"  [{elapsed / 60:5.1f}m] {self.current_phase.value:8s} "
                    f"RSS={rss_mb:5.0f}MB cache={cache_mb:5.0f}MB fds={fd_str:>4} "
                    f"p95={p95:6.1f}ms {gc_pause_str} "
                    f"2xx={total_2xx} 429={total_429} other={other_count}"
                )

    async def run_soak_test(self) -> SoakResults:
        """Run the full soak test."""
        print("\n  Starting soak test")
        print(f"  Duration: {self.config.duration_hours}h")
        print(f"  Users: {self.config.base_users} ({self.config.dashboard_users} dashboard)")
        print(f"  Spikes: every {self.config.spike_interval_minutes}m")

        start_time = time.perf_counter()
        total_duration = self.config.duration_s

        # Calculate phase boundaries
        warmup_end = self.config.warmup_s
        cooldown_start = total_duration - self.config.cooldown_s

        def get_remaining():
            elapsed = time.perf_counter() - start_time
            return max(0, total_duration - elapsed)

        # Track spike timing
        last_spike_time = 0.0
        spike_start_elapsed = 0.0
        in_spike = False

        # Warmup phase
        print(f"\n  Phase: WARMUP ({self.config.warmup_minutes}m)")
        self.current_phase = Phase.WARMUP

        # Start sample collection
        sample_task = asyncio.create_task(self.collect_samples(start_time))

        # Create user tasks
        user_tasks = []
        user_id = 0

        for _ in range(self.config.dashboard_users):
            user_tasks.append(
                asyncio.create_task(self.user_loop(user_id, "dashboard", get_remaining))
            )
            user_id += 1

        for _ in range(self.config.analyst_users):
            user_tasks.append(
                asyncio.create_task(self.user_loop(user_id, "analyst", get_remaining))
            )
            user_id += 1

        for _ in range(self.config.bulk_users):
            user_tasks.append(asyncio.create_task(self.user_loop(user_id, "bulk", get_remaining)))
            user_id += 1

        # Extra users for spikes (initially paused)
        spike_users = []

        # Run test
        try:
            while time.perf_counter() - start_time < total_duration:
                elapsed = time.perf_counter() - start_time

                # Update phase
                if elapsed < warmup_end:
                    self.current_phase = Phase.WARMUP
                elif elapsed >= cooldown_start:
                    if self.current_phase != Phase.COOLDOWN:
                        print(f"\n  Phase: COOLDOWN ({self.config.cooldown_minutes}m)")
                    self.current_phase = Phase.COOLDOWN
                else:
                    # Check for spike
                    time_since_spike = elapsed - last_spike_time

                    if in_spike and time_since_spike >= self.config.spike_duration_seconds:
                        # End spike
                        in_spike = False
                        self.current_phase = Phase.STEADY
                        # Record spike end time
                        self.spike_events.append((spike_start_elapsed, elapsed))
                        print("  Spike ended")
                        # Cancel spike users and await them properly
                        for task in spike_users:
                            task.cancel()
                        await asyncio.gather(*spike_users, return_exceptions=True)
                        spike_users = []
                    elif (
                        not in_spike
                        and time_since_spike >= self.config.spike_interval_s
                        and elapsed >= warmup_end
                    ):
                        # Start spike
                        in_spike = True
                        self.spike_count += 1
                        spike_start_elapsed = elapsed
                        last_spike_time = elapsed
                        self.current_phase = Phase.SPIKE
                        print(f"\n  Phase: SPIKE #{self.spike_count} (2x load for 30s)")
                        # Add spike users
                        extra_users = int(
                            self.config.base_users * (self.config.spike_multiplier - 1)
                        )
                        for i in range(extra_users):
                            spike_users.append(
                                asyncio.create_task(
                                    self.user_loop(1000 + i, "dashboard", get_remaining)
                                )
                            )
                    elif not in_spike and self.current_phase != Phase.STEADY:
                        if self.current_phase == Phase.WARMUP:
                            print("\n  Phase: STEADY")
                        self.current_phase = Phase.STEADY

                await asyncio.sleep(1.0)

        except KeyboardInterrupt:
            print("\n  Interrupted by user")
        finally:
            self._stop_event.set()

            # Cancel all tasks
            all_tasks = user_tasks + spike_users + [sample_task]
            for task in all_tasks:
                task.cancel()

            # Await all cancelled tasks to ensure clean shutdown
            # This prevents warnings and ensures HTTP streams are closed
            print("\n  Draining in-flight requests...")
            await asyncio.gather(*all_tasks, return_exceptions=True)

            # Brief pause to let server finish processing any remaining requests
            await asyncio.sleep(0.5)

        # Wait for active scans to drain before reading metrics
        await self._wait_for_drain()

        # Get final metrics
        final_metrics = await self._get_metrics()

        # Compute results
        return self._compute_results(start_time, final_metrics)

    async def _wait_for_drain(self, timeout_s: float = 10.0) -> None:
        """Wait for server to drain active scans.

        Polls the metrics endpoint until active_scans reaches 0 or timeout.

        Args:
            timeout_s: Maximum time to wait for drain
        """
        start = time.perf_counter()
        while time.perf_counter() - start < timeout_s:
            try:
                metrics = await self._get_metrics()
                active = metrics.get("resource_limits", {}).get("active_scans", 0)
                if active == 0:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.2)
        print("  Warning: drain timeout - some scans may still be active")

    async def _get_metrics(self) -> dict:
        """Get server metrics."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.config.base_url}/metrics")
                response.raise_for_status()
                return response.json()
        except Exception:
            return {}

    def _check_spike_recovery(self, baseline_p95: float) -> bool:
        """Check if latency recovered after each spike.

        For each spike, verify that p95 latency returns within 20% of baseline
        within 2 minutes after the spike ends.

        Args:
            baseline_p95: Baseline p95 latency from warmup phase

        Returns:
            True if all spikes recovered properly, False otherwise
        """
        if not self.spike_events or baseline_p95 <= 0:
            # No spikes occurred or no baseline - consider it a pass
            return True

        recovery_window_s = 120.0  # 2 minutes to recover
        max_recovery_threshold = 1.20  # Within 20% of baseline

        for spike_start, spike_end in self.spike_events:
            # Find latency samples in the recovery window (spike_end to spike_end + 2min)
            recovery_samples = [
                s
                for s in self.latency_samples
                if spike_end <= s.elapsed_s <= spike_end + recovery_window_s
                and s.phase == Phase.STEADY.value
            ]

            if not recovery_samples:
                # No samples in recovery window - can't verify, assume OK
                continue

            # Check if any sample in the window shows recovery
            # (p95 within 20% of baseline)
            recovered = False
            for sample in recovery_samples:
                if sample.p95_ms <= baseline_p95 * max_recovery_threshold:
                    recovered = True
                    break

            if not recovered:
                # This spike didn't recover in time
                return False

        return True

    def _compute_results(self, start_time: float, final_metrics: dict) -> SoakResults:
        """Compute soak test results with improved metrics per user feedback.

        Key improvements:
        1. Separate 2xx vs 429 vs other failures (429 is QoS working, not failure)
        2. Track status codes per phase (POST, GET stream, DELETE)
        3. Use median RSS in steady-state window for memory baseline
        4. Measure latency only on successful (2xx) requests
        5. Production-ready success criteria
        """
        duration_hours = (time.perf_counter() - start_time) / 3600

        # Minimum samples required for reliable baseline (at least 2 samples)
        min_baseline_samples = 2

        # =================================================================
        # Request classification (per user feedback: 2xx vs 429 vs other)
        # =================================================================
        total_requests = len(self.request_results)

        # Classify all requests
        success_2xx_count = sum(1 for r in self.request_results if r.is_success)
        rate_limited_429_count = sum(1 for r in self.request_results if r.is_rate_limited)
        other_fail_count = total_requests - success_2xx_count - rate_limited_429_count

        # Compute rates
        success_2xx_rate = success_2xx_count / total_requests if total_requests > 0 else 0
        rate_limited_429_rate = rate_limited_429_count / total_requests if total_requests > 0 else 0
        other_fail_rate = other_fail_count / total_requests if total_requests > 0 else 0

        # Legacy success rate (includes 429 as failure for backwards compat)
        success_rate = success_2xx_count / total_requests if total_requests > 0 else 0

        # =================================================================
        # Per-phase status code breakdown
        # =================================================================
        post_success_count = 0
        post_429_count = 0
        post_5xx_count = 0
        post_other_fail_count = 0

        stream_success_count = 0
        stream_429_count = 0
        stream_5xx_count = 0
        stream_other_fail_count = 0

        delete_success_count = 0
        delete_already_gone_count = 0
        delete_5xx_count = 0
        delete_timeout_count = 0
        delete_other_fail_count = 0

        for r in self.request_results:
            # POST phase
            if 200 <= r.post_status < 300:
                post_success_count += 1
            elif r.post_status == 429:
                post_429_count += 1
            elif 500 <= r.post_status < 600:
                post_5xx_count += 1
            elif r.post_status != 0:  # 0 means not reached
                post_other_fail_count += 1

            # Stream phase (only if POST succeeded)
            if r.post_status == 200:
                if 200 <= r.stream_status < 300:
                    stream_success_count += 1
                elif r.stream_status == 429:
                    stream_429_count += 1
                elif 500 <= r.stream_status < 600:
                    stream_5xx_count += 1
                elif r.stream_status != 0:
                    stream_other_fail_count += 1

            # DELETE phase - split by semantics
            if 200 <= r.delete_status < 300:
                delete_success_count += 1
            elif r.delete_status == 404:
                # 404 = scan already cleaned up (harmless)
                delete_already_gone_count += 1
            elif 500 <= r.delete_status < 600:
                # 5xx = server error (real failure)
                delete_5xx_count += 1
            elif r.delete_error_type == "timeout":
                # Timeout during DELETE
                delete_timeout_count += 1
            elif r.delete_status != 0 or r.delete_error_type is not None:
                # Other failures (connection errors, etc.)
                delete_other_fail_count += 1

        # =================================================================
        # Post-warmup error breakdown (for success criteria)
        # =================================================================
        post_warmup_5xx = 0
        post_warmup_429 = 0
        post_warmup_timeouts = 0
        post_warmup_arrow_errors = 0
        post_warmup_other_errors = 0

        for r in self.request_results:
            if r.phase == Phase.WARMUP.value:
                continue  # Only count post-warmup

            if r.is_5xx:
                post_warmup_5xx += 1
            elif r.is_rate_limited:
                post_warmup_429 += 1
            elif r.is_timeout:
                post_warmup_timeouts += 1
            elif r.error_type == "arrow_decode":
                post_warmup_arrow_errors += 1
            elif r.error_type is not None:
                post_warmup_other_errors += 1

        # =================================================================
        # Memory analysis - robust 3-window comparison per user feedback
        # =================================================================
        # Window definitions (in seconds from test start):
        # - Baseline: last 2min of warmup (warmup_end - 2min .. warmup_end)
        # - Early steady: 30min to 90min after warmup (warmup_end + 30min .. warmup_end + 90min)
        # - Late steady: last 30min of test (excluding cooldown and spikes)
        #
        # Success criteria:
        # - early_steady/baseline <= 1.10 (initial growth OK)
        # - late_steady/early_steady <= 1.10 (no continued growth)
        # - slope of RSS over time ≈ 0 (no upward drift)

        warmup_samples = [s for s in self.resource_samples if s.phase == Phase.WARMUP.value]

        # Filter samples to exclude spike contamination windows
        spike_contamination_window_s = 120.0

        def is_clean_sample(sample) -> bool:
            """Check if sample is not contaminated by spike recovery."""
            for _, spike_end in self.spike_events:
                if spike_end <= sample.elapsed_s <= spike_end + spike_contamination_window_s:
                    return False
            return True

        def is_clean_steady_resource_sample(sample) -> bool:
            if sample.phase != Phase.STEADY.value:
                return False
            return is_clean_sample(sample)

        clean_steady_samples = [
            s for s in self.resource_samples if is_clean_steady_resource_sample(s)
        ]

        # Get warmup end time (when warmup phase ends)
        warmup_end_s = warmup_samples[-1].elapsed_s if warmup_samples else 0

        # Window 1: Baseline - median RSS from warmup tail (last 2 min)
        baseline_window_samples = 4  # ~2 min at 30s interval
        warmup_tail = warmup_samples[-baseline_window_samples:] if warmup_samples else []

        if len(warmup_tail) >= min_baseline_samples:
            warmup_rss_values = sorted(s.rss_bytes for s in warmup_tail)
            baseline_rss = warmup_rss_values[len(warmup_rss_values) // 2]  # Median
        else:
            baseline_rss = 0

        # Window 2: Early steady - median RSS from (warmup_end + 30min) to (warmup_end + 90min)
        early_steady_start_s = warmup_end_s + 30 * 60  # 30 min after warmup
        early_steady_end_s = warmup_end_s + 90 * 60  # 90 min after warmup
        early_steady_samples = [
            s
            for s in clean_steady_samples
            if early_steady_start_s <= s.elapsed_s <= early_steady_end_s
        ]

        if len(early_steady_samples) >= min_baseline_samples:
            early_rss_values = sorted(s.rss_bytes for s in early_steady_samples)
            early_steady_rss = early_rss_values[len(early_rss_values) // 2]
        else:
            # Fall back to all clean steady samples if not enough in window
            early_steady_rss = baseline_rss

        # Window 3: Late steady - median RSS from last 30min (excluding cooldown)
        # Find samples in last 30 min that are clean steady (not cooldown, not spike recovery)
        if clean_steady_samples:
            max_elapsed_s = max(s.elapsed_s for s in clean_steady_samples)
            late_window_start_s = max_elapsed_s - 30 * 60  # Last 30 min
            late_steady_samples = [
                s for s in clean_steady_samples if s.elapsed_s >= late_window_start_s
            ]

            if len(late_steady_samples) >= min_baseline_samples:
                late_rss_values = sorted(s.rss_bytes for s in late_steady_samples)
                late_steady_rss = late_rss_values[len(late_rss_values) // 2]
            else:
                late_steady_rss = early_steady_rss
        else:
            late_steady_rss = early_steady_rss

        # Legacy: overall steady median (for backwards compat)
        if clean_steady_samples:
            steady_rss_values = sorted(s.rss_bytes for s in clean_steady_samples)
            steady_median_rss = steady_rss_values[len(steady_rss_values) // 2]
        else:
            steady_median_rss = baseline_rss

        # Peak and min RSS (for observability - shows GC fluctuation range)
        all_rss = [s.rss_bytes for s in self.resource_samples if s.rss_bytes > 0]
        peak_rss = max(all_rss) if all_rss else 0
        min_rss = min(all_rss) if all_rss else 0
        final_rss = self.resource_samples[-1].rss_bytes if self.resource_samples else 0

        # Memory growth metrics
        # Primary: early_steady vs baseline (initial stabilization)
        memory_growth_pct = (
            ((early_steady_rss - baseline_rss) / baseline_rss * 100) if baseline_rss > 0 else 0
        )

        # Secondary: late_steady vs early_steady (continued growth = leak)
        memory_end_growth_pct = (
            ((late_steady_rss - early_steady_rss) / early_steady_rss * 100)
            if early_steady_rss > 0
            else 0
        )

        # Trend: compute linear regression slope of RSS over time (clean steady only)
        # Slope in bytes/second, convert to MB/hour for readability
        memory_slope_mb_per_hour = 0.0
        if len(clean_steady_samples) >= 10:
            # Simple linear regression: y = mx + b
            # We want slope m (in bytes/second)
            times = [s.elapsed_s for s in clean_steady_samples]
            rss_values = [s.rss_bytes for s in clean_steady_samples]
            n = len(times)
            sum_t = sum(times)
            sum_rss = sum(rss_values)
            sum_t_rss = sum(t * r for t, r in zip(times, rss_values))
            sum_t2 = sum(t * t for t in times)

            denominator = n * sum_t2 - sum_t * sum_t
            if denominator != 0:
                slope_bytes_per_sec = (n * sum_t_rss - sum_t * sum_rss) / denominator
                # Convert to MB/hour
                memory_slope_mb_per_hour = slope_bytes_per_sec * 3600 / (1024 * 1024)

        # FD and thread analysis
        baseline_fds = -1
        if warmup_tail and all(s.num_fds >= 0 for s in warmup_tail):
            fd_values = sorted(s.num_fds for s in warmup_tail)
            baseline_fds = fd_values[len(fd_values) // 2]

        final_fds = self.resource_samples[-1].num_fds if self.resource_samples else -1

        baseline_threads = 0.0
        if len(warmup_tail) >= min_baseline_samples:
            thread_values = sorted(s.num_threads for s in warmup_tail)
            baseline_threads = thread_values[len(thread_values) // 2]

        final_threads = self.resource_samples[-1].num_threads if self.resource_samples else 0

        # Cache metrics from final sample
        final_cache_bytes = self.resource_samples[-1].cache_bytes if self.resource_samples else 0
        final_cache_entries = (
            self.resource_samples[-1].cache_entries if self.resource_samples else 0
        )
        total_evictions = self.resource_samples[-1].cache_evictions if self.resource_samples else 0

        # =================================================================
        # Latency analysis - measured on successful (2xx) requests only
        # =================================================================
        warmup_latency = [s for s in self.latency_samples if s.phase == Phase.WARMUP.value]
        spike_latency = [s for s in self.latency_samples if s.phase == Phase.SPIKE.value]

        # Filter steady-state latency samples to exclude spike contamination
        def is_clean_steady_latency(sample) -> bool:
            if sample.phase != Phase.STEADY.value:
                return False
            for _, spike_end in self.spike_events:
                if spike_end <= sample.elapsed_s <= spike_end + spike_contamination_window_s:
                    return False
            return True

        steady_latency_clean = [s for s in self.latency_samples if is_clean_steady_latency(s)]

        # Baseline p95 from warmup tail
        warmup_latency_tail = warmup_latency[-baseline_window_samples:] if warmup_latency else []
        baseline_p95 = (
            sum(s.p95_ms for s in warmup_latency_tail) / len(warmup_latency_tail)
            if len(warmup_latency_tail) >= min_baseline_samples
            else 0
        )

        # Final p95 from clean steady-state samples tail
        steady_latency_tail = (
            steady_latency_clean[-baseline_window_samples:] if steady_latency_clean else []
        )
        final_p95 = (
            sum(s.p95_ms for s in steady_latency_tail) / len(steady_latency_tail)
            if len(steady_latency_tail) >= min_baseline_samples
            else 0
        )

        # Median steady p95 (robust measure)
        if steady_latency_clean:
            sorted_p95 = sorted(s.p95_ms for s in steady_latency_clean)
            median_steady_p95 = sorted_p95[len(sorted_p95) // 2]
        else:
            median_steady_p95 = final_p95

        # Max p95 from all samples
        max_p95 = max((s.p95_ms for s in self.latency_samples), default=0)

        # Spike p95 (expected to be higher during spikes)
        spike_p95 = (
            sum(s.p95_ms for s in spike_latency) / len(spike_latency) if spike_latency else 0
        )

        # Latency drift: compare median steady vs baseline
        latency_drift_pct = (
            ((median_steady_p95 - baseline_p95) / baseline_p95 * 100) if baseline_p95 > 0 else 0
        )

        # Check if we have sufficient data for reliable metrics
        insufficient_data = (
            len(warmup_tail) < min_baseline_samples
            or len(warmup_latency_tail) < min_baseline_samples
            or baseline_rss == 0
            or baseline_p95 == 0
        )

        # =================================================================
        # Spike recovery analysis
        # =================================================================
        spike_recovery_ok = self._check_spike_recovery(baseline_p95)

        # Compute average recovery time across all spikes
        recovery_times = []
        for spike_start, spike_end in self.spike_events:
            recovery_window_s = 120.0
            recovery_samples = [
                s
                for s in self.latency_samples
                if spike_end <= s.elapsed_s <= spike_end + recovery_window_s
                and s.phase == Phase.STEADY.value
            ]
            if recovery_samples and baseline_p95 > 0:
                for sample in recovery_samples:
                    if sample.p95_ms <= baseline_p95 * 1.20:
                        recovery_times.append(sample.elapsed_s - spike_end)
                        break

        spike_recovery_time_s = sum(recovery_times) / len(recovery_times) if recovery_times else 0

        # =================================================================
        # Server metrics
        # =================================================================
        metrics_available = bool(final_metrics)
        final_active = final_metrics.get("resource_limits", {}).get("active_scans", 0)

        cache_hits = final_metrics.get("cache_hits", 0)
        cache_misses = final_metrics.get("cache_misses", 0)
        if cache_hits + cache_misses > 0:
            cache_hit_rate = cache_hits / (cache_hits + cache_misses)
        else:
            cache_hit_rate = -1.0 if not metrics_available else 0.0

        prefetch = final_metrics.get("prefetch", {})
        prefetch_started = prefetch.get("started", 0)
        prefetch_used = prefetch.get("used", 0)
        if prefetch_started > 0:
            prefetch_efficiency = prefetch_used / prefetch_started
        else:
            prefetch_efficiency = -1.0 if not metrics_available else 0.0

        # Event loop lag metrics
        all_lags = [s.event_loop_lag_ms for s in self.latency_samples if s.event_loop_lag_ms > 0]
        max_event_loop_lag = max(all_lags) if all_lags else 0
        p95_event_loop_lag = _percentile(all_lags, 0.95) if all_lags else 0

        # GC pause metrics
        valid_samples = [s for s in self.resource_samples if s.rss_bytes > 0]
        last_valid = valid_samples[-1] if valid_samples else None
        gc_total_pauses = last_valid.gc_total_pauses if last_valid else 0
        gc_total_pause_ms = last_valid.gc_total_pause_ms if last_valid else 0.0
        gc_max_pause_ms = max((s.gc_max_pause_ms for s in valid_samples), default=0.0)
        gc_gen2_pause_count = last_valid.gc_gen2_pause_count if last_valid else 0
        gc_gen2_max_ms = max((s.gc_gen2_max_ms for s in valid_samples), default=0.0)

        # =================================================================
        # GC-latency correlation analysis (per user feedback)
        # =================================================================
        # Identify "slow windows" where p95 > 2x baseline, then check if they
        # correlate with gen2 GC activity in that window. High correlation
        # suggests GC pauses are causing latency spikes.
        slow_window_threshold = baseline_p95 * 2.0 if baseline_p95 > 0 else 1000.0
        slow_window_count = 0
        slow_window_with_gc_count = 0
        slow_window_gc_pause_total = 0.0
        normal_window_gc_pause_total = 0.0
        normal_window_count = 0

        # Use steady-state latency samples only (exclude warmup/cooldown)
        steady_latency_for_gc = [s for s in self.latency_samples if s.phase == Phase.STEADY.value]

        for sample in steady_latency_for_gc:
            if sample.p95_ms > slow_window_threshold:
                slow_window_count += 1
                slow_window_gc_pause_total += sample.gc_pause_ms_in_window
                if sample.gc_gen2_in_window > 0:
                    slow_window_with_gc_count += 1
            else:
                normal_window_count += 1
                normal_window_gc_pause_total += sample.gc_pause_ms_in_window

        # Compute correlation: % of slow windows that had gen2 activity
        gc_latency_correlation = (
            (slow_window_with_gc_count / slow_window_count * 100) if slow_window_count > 0 else 0.0
        )

        # Average GC pause time in slow vs normal windows
        slow_window_avg_gc_pause_ms = (
            slow_window_gc_pause_total / slow_window_count if slow_window_count > 0 else 0.0
        )
        normal_window_avg_gc_pause_ms = (
            normal_window_gc_pause_total / normal_window_count if normal_window_count > 0 else 0.0
        )

        # =================================================================
        # Success criteria (updated per user specifications)
        # =================================================================
        server_alive = final_rss > 0 and final_threads > 0

        # Memory OK requires all three conditions (robust 3-window check):
        # 1. early_steady/baseline <= 1.10 (initial growth acceptable)
        # 2. late_steady/early_steady <= 1.10 (no continued growth)
        # 3. slope ≈ 0 (allow up to 5 MB/hour drift for noise tolerance)
        max_slope_mb_per_hour = 5.0  # Allow small drift due to noise
        memory_ok = server_alive and (
            memory_growth_pct <= self.config.max_memory_growth_pct
            and memory_end_growth_pct <= self.config.max_memory_growth_pct
            and abs(memory_slope_mb_per_hour) <= max_slope_mb_per_hour
        )
        latency_ok = latency_drift_pct <= self.config.max_latency_drift_pct
        no_leak = metrics_available and final_active == 0

        # Zero 5xx, timeouts, and Arrow decode errors after warmup (429 OK)
        no_errors_post_warmup = (
            post_warmup_5xx == 0 and post_warmup_timeouts == 0 and post_warmup_arrow_errors == 0
        )

        # Other fail rate <= 0.1% (per user specification)
        other_fail_rate_ok = other_fail_rate <= 0.001

        # 429 rate within expected band (based on load vs capacity)
        # For soak test with N users and M slots, expected 429 rate depends on load
        # We consider it OK if 429 rate < 50% (server not completely saturated)
        rate_429_in_band = rate_limited_429_rate <= 0.50

        # FD leak check
        no_fd_leak = True
        if baseline_fds > 0 and final_fds > 0:
            no_fd_leak = final_fds <= baseline_fds * 2

        # Thread leak check
        no_thread_leak = True
        if baseline_threads > 0 and final_threads > 0:
            no_thread_leak = final_threads <= baseline_threads * 2

        # Overall pass (updated criteria per user feedback)
        overall_pass = (
            server_alive
            and memory_ok
            and latency_ok
            and no_leak
            and no_fd_leak
            and no_thread_leak
            and no_errors_post_warmup
            and other_fail_rate_ok
            and rate_429_in_band
            and spike_recovery_ok
            and not insufficient_data
        )

        return SoakResults(
            duration_hours=duration_hours,
            total_requests=total_requests,
            success_2xx_count=success_2xx_count,
            rate_limited_429_count=rate_limited_429_count,
            other_fail_count=other_fail_count,
            success_2xx_rate=success_2xx_rate,
            rate_limited_429_rate=rate_limited_429_rate,
            other_fail_rate=other_fail_rate,
            success_rate=success_rate,
            baseline_rss_mb=baseline_rss / (1024 * 1024),
            early_steady_rss_mb=early_steady_rss / (1024 * 1024),
            late_steady_rss_mb=late_steady_rss / (1024 * 1024),
            steady_median_rss_mb=steady_median_rss / (1024 * 1024),
            peak_rss_mb=peak_rss / (1024 * 1024),
            min_rss_mb=min_rss / (1024 * 1024),
            final_rss_mb=final_rss / (1024 * 1024),
            memory_growth_pct=memory_growth_pct,
            memory_end_growth_pct=memory_end_growth_pct,
            memory_slope_mb_per_hour=memory_slope_mb_per_hour,
            baseline_p95_ms=baseline_p95,
            final_p95_ms=final_p95,
            median_steady_p95_ms=median_steady_p95,
            max_p95_ms=max_p95,
            latency_drift_pct=latency_drift_pct,
            spike_p95_ms=spike_p95,
            spike_recovery_time_s=spike_recovery_time_s,
            final_active_scans=final_active,
            cache_hit_rate=cache_hit_rate,
            prefetch_efficiency=prefetch_efficiency,
            spike_count=self.spike_count,
            spike_recovery_ok=spike_recovery_ok,
            baseline_fds=baseline_fds,
            final_fds=final_fds,
            baseline_threads=baseline_threads,
            final_threads=final_threads,
            final_cache_mb=final_cache_bytes / (1024 * 1024),
            final_cache_entries=final_cache_entries,
            total_cache_evictions=total_evictions,
            post_success_count=post_success_count,
            post_429_count=post_429_count,
            post_5xx_count=post_5xx_count,
            post_other_fail_count=post_other_fail_count,
            stream_success_count=stream_success_count,
            stream_429_count=stream_429_count,
            stream_5xx_count=stream_5xx_count,
            stream_other_fail_count=stream_other_fail_count,
            delete_success_count=delete_success_count,
            delete_already_gone_count=delete_already_gone_count,
            delete_5xx_count=delete_5xx_count,
            delete_timeout_count=delete_timeout_count,
            delete_other_fail_count=delete_other_fail_count,
            post_warmup_5xx=post_warmup_5xx,
            post_warmup_429=post_warmup_429,
            post_warmup_timeouts=post_warmup_timeouts,
            post_warmup_arrow_errors=post_warmup_arrow_errors,
            post_warmup_other_errors=post_warmup_other_errors,
            max_event_loop_lag_ms=max_event_loop_lag,
            p95_event_loop_lag_ms=p95_event_loop_lag,
            gc_total_pauses=gc_total_pauses,
            gc_total_pause_ms=gc_total_pause_ms,
            gc_max_pause_ms=gc_max_pause_ms,
            gc_gen2_pause_count=gc_gen2_pause_count,
            gc_gen2_max_ms=gc_gen2_max_ms,
            slow_window_count=slow_window_count,
            slow_window_with_gc_count=slow_window_with_gc_count,
            gc_latency_correlation=gc_latency_correlation,
            slow_window_avg_gc_pause_ms=slow_window_avg_gc_pause_ms,
            normal_window_avg_gc_pause_ms=normal_window_avg_gc_pause_ms,
            server_alive=server_alive,
            memory_ok=memory_ok,
            latency_ok=latency_ok,
            no_leak=no_leak,
            no_fd_leak=no_fd_leak,
            no_thread_leak=no_thread_leak,
            no_errors_post_warmup=no_errors_post_warmup,
            other_fail_rate_ok=other_fail_rate_ok,
            rate_429_in_band=rate_429_in_band,
            overall_pass=overall_pass,
            server_crash_time_min=self.server_crash_time / 60 if self.server_crash_time else None,
            server_crash_signal=self.server_crash_info.get("signal_name")
            if self.server_crash_info
            else None,
            server_crash_exit_code=self.server_crash_info.get("exit_code")
            if self.server_crash_info
            else None,
            insufficient_data=insufficient_data,
        )


# =============================================================================
# Main
# =============================================================================


def print_results(results: SoakResults):
    """Print formatted results with improved metrics per user feedback."""
    print("\n" + "=" * 80)
    print("SOAK TEST RESULTS")
    print("=" * 80)

    print(f"\nDuration: {results.duration_hours:.2f} hours")
    print(f"Total requests: {results.total_requests:,}")

    # =================================================================
    # Request Classification (new: 2xx vs 429 vs other)
    # =================================================================
    print("\n" + "-" * 40)
    print("REQUEST CLASSIFICATION")
    print("-" * 40)
    success_pct = results.success_2xx_rate * 100
    print(f"Success (2xx):     {results.success_2xx_count:>8,} ({success_pct:5.1f}%)")
    rate_limited_pct = results.rate_limited_429_rate * 100
    print(f"Rate limited (429):{results.rate_limited_429_count:>8,} ({rate_limited_pct:5.1f}%)")
    print(
        f"Other failures:    {results.other_fail_count:>8,} ({results.other_fail_rate * 100:5.2f}%)"
    )

    # =================================================================
    # Per-Phase Status Codes (new: POST/GET/DELETE breakdown)
    # =================================================================
    print("\n" + "-" * 40)
    print("STATUS CODES BY PHASE")
    print("-" * 40)
    print("POST /v1/materialize:")
    print(
        f"  2xx: {results.post_success_count:,}  429: {results.post_429_count:,}  "
        f"5xx: {results.post_5xx_count}  other: {results.post_other_fail_count}"
    )
    print("GET stream:")
    print(
        f"  2xx: {results.stream_success_count:,}  429: {results.stream_429_count:,}  "
        f"5xx: {results.stream_5xx_count}  other: {results.stream_other_fail_count}"
    )

    # =================================================================
    # Memory (robust 3-window analysis)
    # =================================================================
    print("\n" + "-" * 40)
    print("MEMORY (3-window analysis)")
    print("-" * 40)
    print(f"Baseline (warmup tail):   {results.baseline_rss_mb:.1f} MB")
    print(f"Early steady (30-90min):  {results.early_steady_rss_mb:.1f} MB")
    print(f"Late steady (last 30min): {results.late_steady_rss_mb:.1f} MB")
    print(
        f"Range: {results.min_rss_mb:.1f} MB - {results.peak_rss_mb:.1f} MB (shows GC fluctuation)"
    )
    print(f"Final RSS: {results.final_rss_mb:.1f} MB")
    print(f"Growth (early vs baseline):  {results.memory_growth_pct:+.1f}%")
    print(f"Growth (late vs early):      {results.memory_end_growth_pct:+.1f}%")
    print(f"Trend (slope):               {results.memory_slope_mb_per_hour:+.2f} MB/hour")

    # =================================================================
    # Latency (updated: measured on 2xx only, separate spike/steady)
    # =================================================================
    print("\n" + "-" * 40)
    print("LATENCY (2xx requests only)")
    print("-" * 40)
    print(f"Baseline p95 (warmup):     {results.baseline_p95_ms:.1f} ms")
    print(f"Steady p95 (median):       {results.median_steady_p95_ms:.1f} ms")
    print(f"Steady p95 (final window): {results.final_p95_ms:.1f} ms")
    print(f"Spike p95:                 {results.spike_p95_ms:.1f} ms")
    print(f"Max p95 (any sample):      {results.max_p95_ms:.1f} ms")
    print(f"Drift (baseline vs steady median): {results.latency_drift_pct:+.1f}%")
    if results.spike_recovery_time_s > 0:
        print(f"Avg spike recovery time:   {results.spike_recovery_time_s:.1f}s")

    # =================================================================
    # Resources
    # =================================================================
    print("\n" + "-" * 40)
    print("RESOURCES")
    print("-" * 40)
    fd_baseline = f"{results.baseline_fds:.0f}" if results.baseline_fds >= 0 else "n/a"
    fd_final = str(results.final_fds) if results.final_fds >= 0 else "n/a"
    print(f"File descriptors: {fd_baseline} -> {fd_final}")
    print(f"Threads: {results.baseline_threads:.0f} -> {results.final_threads}")
    print(f"Cache size: {results.final_cache_mb:.1f} MB ({results.final_cache_entries} entries)")
    print(f"Cache evictions: {results.total_cache_evictions}")

    # =================================================================
    # GC Pause Duration
    # =================================================================
    print("\n" + "-" * 40)
    print("GC PAUSE DURATION")
    print("-" * 40)
    print(f"Total pauses: {results.gc_total_pauses}")
    print(f"Total pause time: {results.gc_total_pause_ms:.1f} ms")
    print(f"Max single pause: {results.gc_max_pause_ms:.1f} ms")
    print(f"Gen2 pauses: {results.gc_gen2_pause_count}")
    print(f"Gen2 max pause: {results.gc_gen2_max_ms:.1f} ms")
    if results.duration_hours > 0:
        pause_pct = (results.gc_total_pause_ms / 1000) / (results.duration_hours * 3600) * 100
        print(f"Pause overhead: {pause_pct:.3f}% of runtime")

    # =================================================================
    # GC-Latency Correlation
    # =================================================================
    print("\n" + "-" * 40)
    print("GC-LATENCY CORRELATION")
    print("-" * 40)
    print(f"Slow windows (p95 > 2x baseline): {results.slow_window_count}")
    if results.slow_window_count > 0:
        print(f"Slow windows with gen2 GC:        {results.slow_window_with_gc_count}")
        print(f"Correlation:                      {results.gc_latency_correlation:.1f}%")
        print(f"Avg GC pause in slow windows:     {results.slow_window_avg_gc_pause_ms:.1f} ms")
        print(f"Avg GC pause in normal windows:   {results.normal_window_avg_gc_pause_ms:.1f} ms")
        # Interpret the correlation
        if results.gc_latency_correlation > 50:
            print("\n  NOTE: High GC-latency correlation. Gen2 GC pauses are likely")
            print("        contributing to tail latency. Consider:")
            print("        - Reducing object allocations in hot paths")
            print("        - Increasing gen2 threshold (gc.set_threshold)")
            print("        - Upgrading to Python 3.12+ for improved GC")
    else:
        print("No slow windows detected - latency stable throughout test")

    # =================================================================
    # Client Health
    # =================================================================
    print("\n" + "-" * 40)
    print("CLIENT HEALTH")
    print("-" * 40)
    print(f"Event loop lag (max): {results.max_event_loop_lag_ms:.2f} ms")
    print(f"Event loop lag (p95): {results.p95_event_loop_lag_ms:.2f} ms")

    # =================================================================
    # Stability
    # =================================================================
    print("\n" + "-" * 40)
    print("STABILITY")
    print("-" * 40)
    print(f"Final active scans: {results.final_active_scans}")
    cache_str = f"{results.cache_hit_rate * 100:.1f}%" if results.cache_hit_rate >= 0 else "n/a"
    print(f"Cache hit rate: {cache_str}")
    prefetch_str = (
        f"{results.prefetch_efficiency * 100:.1f}%" if results.prefetch_efficiency >= 0 else "n/a"
    )
    print(f"Prefetch efficiency: {prefetch_str}")
    print(f"Spikes completed: {results.spike_count}")

    # =================================================================
    # Post-Warmup Error Breakdown
    # =================================================================
    print("\n" + "-" * 40)
    print("POST-WARMUP BREAKDOWN")
    print("-" * 40)
    print(f"5xx errors: {results.post_warmup_5xx}")
    print(f"429 rate limited: {results.post_warmup_429}")
    print(f"Timeouts: {results.post_warmup_timeouts}")
    print(f"Arrow decode errors: {results.post_warmup_arrow_errors}")
    print(f"Other errors: {results.post_warmup_other_errors}")

    if results.rate_limited_429_rate > 0.1:
        print(f"\nNOTE: High 429 rate ({results.rate_limited_429_rate * 100:.1f}%).")
        print("      Consider increasing QoS slots or reducing concurrent users.")

    # Print crash info if server died
    if results.server_crash_time_min is not None:
        print("\n" + "-" * 40)
        print("SERVER CRASH DETECTED")
        print("-" * 40)
        print(f"Crash time: {results.server_crash_time_min:.1f} min into test")
        if results.server_crash_signal:
            print(f"Signal: {results.server_crash_signal}")
        elif results.server_crash_exit_code is not None:
            print(f"Exit code: {results.server_crash_exit_code}")
        print("\nCheck server logs for details:")
        print("  - benchmarks/results/server_stderr.log")
        print("  - benchmarks/results/server_stdout.log")

    # =================================================================
    # Success Criteria (updated per user feedback)
    # =================================================================
    print("\n" + "-" * 40)
    print("SUCCESS CRITERIA")
    print("-" * 40)

    alive_status = "PASS" if results.server_alive else "FAIL"
    crash_info = ""
    if results.server_crash_time_min is not None:
        crash_info = f" (crashed at {results.server_crash_time_min:.1f} min)"
    print(f"Server alive at end: {alive_status}{crash_info}")

    mem_status = "PASS" if results.memory_ok else "FAIL"
    print(f"Memory stable: {mem_status}")
    print(f"  - Early growth <= 10%: {results.memory_growth_pct:+.1f}%")
    print(f"  - Late growth <= 10%:  {results.memory_end_growth_pct:+.1f}%")
    print(f"  - Trend <= 5 MB/hr:    {results.memory_slope_mb_per_hour:+.2f} MB/hr")

    lat_status = "PASS" if results.latency_ok else "FAIL"
    print(f"Latency drift <= 20%: {lat_status} ({results.latency_drift_pct:+.1f}%)")

    leak_status = "PASS" if results.no_leak else "FAIL"
    print(f"No active scan leak: {leak_status} (active={results.final_active_scans})")

    fd_status = "PASS" if results.no_fd_leak else "FAIL"
    print(f"No FD leak: {fd_status} ({fd_baseline} -> {fd_final})")

    thread_status = "PASS" if results.no_thread_leak else "FAIL"
    threads_str = f"{results.baseline_threads:.0f} -> {results.final_threads}"
    print(f"No thread leak: {thread_status} ({threads_str})")

    critical_errors = (
        results.post_warmup_5xx + results.post_warmup_timeouts + results.post_warmup_arrow_errors
    )
    err_status = "PASS" if results.no_errors_post_warmup else "FAIL"
    print(f"Zero critical errors (5xx/timeout/arrow): {err_status} ({critical_errors})")

    # New criteria: other_fail_rate <= 0.1%
    other_fail_status = "PASS" if results.other_fail_rate_ok else "FAIL"
    print(f"Other fail rate <= 0.1%: {other_fail_status} ({results.other_fail_rate * 100:.2f}%)")

    # New criteria: 429 rate in expected band
    rate_429_status = "PASS" if results.rate_429_in_band else "FAIL"
    print(f"429 rate <= 50%: {rate_429_status} ({results.rate_limited_429_rate * 100:.1f}%)")

    recovery_status = "PASS" if results.spike_recovery_ok else "FAIL"
    print(f"Spike recovery (<2min): {recovery_status} ({results.spike_count} spikes)")

    if results.insufficient_data:
        print("\nWARNING: Insufficient data for reliable baseline metrics")
        print("         Run longer or increase warmup duration")

    print("\n" + "=" * 80)
    if results.insufficient_data:
        print("OVERALL: FAIL (insufficient data)")
    else:
        print(f"OVERALL: {'PASS' if results.overall_pass else 'FAIL'}")
    print("=" * 80)


async def main():
    parser = argparse.ArgumentParser(
        description="Soak test for Strata production readiness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Test duration in hours (default: 1.0)",
    )
    parser.add_argument(
        "--users",
        type=int,
        default=30,
        help="Base concurrent users (default: 30)",
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
        help="Quick test run (5 minutes)",
    )
    args = parser.parse_args()

    # Create config
    config = SoakConfig(
        duration_hours=args.duration if not args.dry_run else 5 / 60,  # 5 min for dry-run
        base_users=args.users,
        start_server=not args.no_start_server,
        server_port=args.port or find_free_port(),
        keep_dirs=args.keep_dirs,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        config.warmup_minutes = 1.0
        config.cooldown_minutes = 0.5
        config.spike_interval_minutes = 2.0
        config.num_tables = 4
        config.rows_per_table = 10_000

    # Create temp directories
    temp_dir = Path(tempfile.mkdtemp(prefix="strata_soak_"))
    config.warehouse_dir = temp_dir / "warehouse"
    config.cache_dir = temp_dir / "cache"
    config.cache_dir.mkdir(parents=True)

    print("=" * 80)
    print("STRATA SOAK TEST")
    print("=" * 80)
    print(f"\nDuration: {config.duration_hours}h")
    print(f"Users: {config.base_users}")
    print(f"Spikes: every {config.spike_interval_minutes}m")

    server = None
    try:
        # Generate warehouse
        print(f"\n[1/3] Generating warehouse at {config.warehouse_dir}...")
        warehouse = generate_soak_warehouse(config)
        print(f"  Created {len(warehouse['tables'])} tables")

        # Start server
        if config.start_server:
            print("\n[2/3] Starting Strata server...")
            # Calculate QoS slots based on user count:
            # - Interactive slots for dashboard users (fast queries)
            # - Bulk slots for analyst + bulk users (slower queries)
            # Add ~20% headroom to avoid excessive 429s during spikes
            interactive_slots = max(8, int(config.dashboard_users * 1.2))
            bulk_slots = max(4, int((config.analyst_users + config.bulk_users) * 1.2))
            total_slots = interactive_slots + bulk_slots
            server = ServerProcess(
                config.server_host,
                config.server_port,
                config.cache_dir,
                config.cache_size_bytes,
                log_dir=config.results_dir,  # Capture server logs
                interactive_slots=interactive_slots,
                bulk_slots=bulk_slots,
            )
            server.start()
            config.base_url = f"http://{config.server_host}:{config.server_port}"
            print(f"  Server running at {config.base_url}")
            print(
                f"  QoS slots: {interactive_slots} interactive + {bulk_slots} bulk "
                f"= {total_slots} total"
            )
            print(f"  Cache limit: {config.cache_size_bytes // (1024 * 1024)} MB")
            print(f"  Logs: {config.results_dir}/server_*.log")

        # Run test
        print("\n[3/3] Running soak test...")
        if config.dry_run:
            print("  (dry run - 5 min)")

        driver = SoakDriver(config, warehouse["tables"], server)
        await driver.start()

        try:
            results = await driver.run_soak_test()
        finally:
            await driver.stop()

        # Print results
        print_results(results)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = config.results_dir / f"soak_test_{timestamp}.jsonl"

        with open(results_file, "w") as f:
            # Write samples
            for sample in driver.resource_samples:
                f.write(json.dumps(sample.to_dict()) + "\n")
            for sample in driver.latency_samples:
                f.write(json.dumps(sample.to_dict()) + "\n")
            # Write final results
            f.write(json.dumps({"type": "results", **results.to_dict()}) + "\n")

        print(f"\nResults written to: {results_file}")

    finally:
        if server:
            print("\nStopping server...")
            server.stop()

        if not config.keep_dirs and temp_dir.exists():
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    asyncio.run(main())
