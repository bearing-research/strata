#!/usr/bin/env python3
"""Benchmark: Cache persistence across server restarts.

This benchmark proves Strata's core thesis:
1. Cold start: Data read from Parquet (cache miss)
2. Warm cache: Data served from cache (cache hit, faster)
3. After restart: Cache persists on disk (cache hit, still fast)

This is the slide you show everyone.

Note: Each append creates a separate Parquet file with 1 row group.
So --data-files 5 creates 5 files × 1 row group = 5 cache tasks.

Usage:
    # Single scale run
    python benchmarks/bench_restart.py --rows 100000
    python benchmarks/bench_restart.py --rows 1000000 --data-files 10

    # Multi-scale comparison (the slide you show everyone)
    python benchmarks/bench_restart.py --scale
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import pyarrow as pa


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    total_latency_ms: float  # End-to-end including planning + fetch + cleanup
    planning_latency_ms: float  # POST /scan only
    fetch_latency_ms: float  # GET /batches only (data plane)
    cache_hits: int
    cache_misses: int
    bytes_from_cache: int
    bytes_from_storage: int
    rows: int

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0


def create_sample_table(warehouse_path: Path, num_rows: int, num_data_files: int) -> str:
    """Create a sample Iceberg table with test data.

    Each append creates a separate Parquet data file with 1 row group.
    """
    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField, StringType

    catalog = SqlCatalog(
        "strata",
        **{
            "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
            "warehouse": str(warehouse_path),
        },
    )

    try:
        catalog.create_namespace("bench")
    except Exception:
        pass

    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
        NestedField(3, "category", StringType(), required=False),
        NestedField(4, "payload", StringType(), required=False),
    )

    table_id = "bench.events"
    try:
        table = catalog.load_table(table_id)
    except Exception:
        table = catalog.create_table(table_id, schema)

        # Write data in chunks to create multiple data files
        rows_per_file = num_rows // num_data_files
        categories = ["A", "B", "C", "D", "E"]

        for file_idx in range(num_data_files):
            start_id = file_idx * rows_per_file
            chunk_rows = rows_per_file if file_idx < num_data_files - 1 else num_rows - start_id

            data = pa.table(
                {
                    "id": pa.array(range(start_id, start_id + chunk_rows), type=pa.int64()),
                    "value": pa.array(
                        [float(i * 1.5) for i in range(chunk_rows)], type=pa.float64()
                    ),
                    "category": pa.array(
                        [categories[i % len(categories)] for i in range(chunk_rows)],
                        type=pa.string(),
                    ),
                    "payload": pa.array(
                        [f"data_{i:08d}_" + "x" * 100 for i in range(chunk_rows)],
                        type=pa.string(),
                    ),
                }
            )
            table.append(data)

    return f"file://{warehouse_path}#bench.events"


class ServerProcess:
    """Manages a Strata server as a subprocess.

    Uses subprocess.Popen to truly restart the server process,
    ensuring cache persistence is tested across genuine process boundaries.
    """

    def __init__(self, host: str, port: int, cache_dir: Path):
        self.host = host
        self.port = port
        self.cache_dir = cache_dir
        self._process: subprocess.Popen | None = None

    def start(self):
        """Start the server as a subprocess."""
        env = os.environ.copy()
        env["STRATA_HOST"] = self.host
        env["STRATA_PORT"] = str(self.port)
        env["STRATA_CACHE_DIR"] = str(self.cache_dir)
        env["STRATA_METRICS_ENABLED"] = "false"  # Suppress JSON logs

        self._process = subprocess.Popen(
            [sys.executable, "-m", "strata.server"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to be ready
        self._wait_for_ready()

    def _wait_for_ready(self, timeout: float = 10.0):
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

        # Print server stderr on timeout for debugging
        if self._process and self._process.stderr:
            import select

            if select.select([self._process.stderr], [], [], 0)[0]:
                print(f"SERVER STDERR: {self._process.stderr.read().decode()[:1000]}")
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
            # Print any server errors for debugging
            if self._process.stderr:
                stderr = self._process.stderr.read()
                if stderr and b"Error" in stderr:
                    print(f"SERVER STDERR: {stderr.decode()[:500]}")
            self._process = None


def run_scan(client, table_uri: str, metrics_before: dict | None = None) -> BenchmarkResult:
    """Run a scan and collect metrics with phase-level timing.

    Measures:
    - planning_latency_ms: POST /scan (planning phase)
    - fetch_latency_ms: GET /batches (data plane)
    - total_latency_ms: End-to-end including cleanup

    If metrics_before is provided, calculates deltas.
    """
    import pyarrow.ipc as ipc

    total_start = time.perf_counter()

    # Phase 1: Planning (POST /scan)
    plan_start = time.perf_counter()
    request_body = {"table_uri": table_uri}
    response = client._client.post("/v1/scan", json=request_body)
    response.raise_for_status()
    scan_info = response.json()
    scan_id = scan_info["scan_id"]
    planning_latency_ms = (time.perf_counter() - plan_start) * 1000

    # Phase 2: Data fetch (GET /batches)
    fetch_start = time.perf_counter()
    response = client._client.get(f"/v1/scan/{scan_id}/batches")
    response.raise_for_status()
    batches = []
    if response.content:
        reader = ipc.open_stream(pa.BufferReader(response.content))
        batches = list(reader)
    fetch_latency_ms = (time.perf_counter() - fetch_start) * 1000

    # Cleanup
    try:
        client._client.delete(f"/v1/scan/{scan_id}")
    except Exception:
        pass

    total_latency_ms = (time.perf_counter() - total_start) * 1000
    total_rows = sum(b.num_rows for b in batches)
    metrics_after = client.metrics()

    if metrics_before:
        cache_hits = metrics_after["cache_hits"] - metrics_before["cache_hits"]
        cache_misses = metrics_after["cache_misses"] - metrics_before["cache_misses"]
        bytes_from_cache = metrics_after["bytes_from_cache"] - metrics_before["bytes_from_cache"]
        bytes_from_storage = (
            metrics_after["bytes_from_storage"] - metrics_before["bytes_from_storage"]
        )
    else:
        cache_hits = metrics_after["cache_hits"]
        cache_misses = metrics_after["cache_misses"]
        bytes_from_cache = metrics_after["bytes_from_cache"]
        bytes_from_storage = metrics_after["bytes_from_storage"]

    return BenchmarkResult(
        name="",
        total_latency_ms=total_latency_ms,
        planning_latency_ms=planning_latency_ms,
        fetch_latency_ms=fetch_latency_ms,
        cache_hits=cache_hits,
        cache_misses=cache_misses,
        bytes_from_cache=bytes_from_cache,
        bytes_from_storage=bytes_from_storage,
        rows=total_rows,
    )


def format_bytes(n: int) -> str:
    """Format bytes as human-readable string."""
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024**3):.2f} GB"
    elif n >= 1024 * 1024:
        return f"{n / (1024**2):.2f} MB"
    elif n >= 1024:
        return f"{n / 1024:.2f} KB"
    return f"{n} B"


def print_results_table(results: list[BenchmarkResult]):
    """Print results as a formatted table."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS: Cache Persistence Across Server Restarts")
    print("=" * 80)

    # Header with phase breakdown
    print(f"\n{'Phase':<18} {'Total':>10} {'Plan':>10} {'Fetch':>10} {'Hits':>8} {'Miss':>8}")
    print("-" * 80)

    # Data rows
    for r in results:
        print(
            f"{r.name:<18} {r.total_latency_ms:>8.1f}ms "
            f"{r.planning_latency_ms:>8.1f}ms {r.fetch_latency_ms:>8.1f}ms "
            f"{r.cache_hits:>8} {r.cache_misses:>8}"
        )

    print("-" * 80)

    # Summary
    if len(results) >= 3:
        cold = results[0]
        warm = results[1]
        restart = results[2]

        print(f"\n{'Metric':<24} {'Cold Start':>15} {'Warm Cache':>15} {'Post-Restart':>15}")
        print("-" * 72)
        print(
            f"{'Total Latency (ms)':<24} "
            f"{cold.total_latency_ms:>15.1f} {warm.total_latency_ms:>15.1f} "
            f"{restart.total_latency_ms:>15.1f}"
        )
        print(
            f"{'  Planning (ms)':<24} "
            f"{cold.planning_latency_ms:>15.1f} {warm.planning_latency_ms:>15.1f} "
            f"{restart.planning_latency_ms:>15.1f}"
        )
        print(
            f"{'  Fetch (ms)':<24} "
            f"{cold.fetch_latency_ms:>15.1f} {warm.fetch_latency_ms:>15.1f} "
            f"{restart.fetch_latency_ms:>15.1f}"
        )
        cold_pq = format_bytes(cold.bytes_from_storage)
        warm_pq = format_bytes(warm.bytes_from_storage)
        restart_pq = format_bytes(restart.bytes_from_storage)
        print(f"{'Bytes from Parquet':<24} {cold_pq:>15} {warm_pq:>15} {restart_pq:>15}")
        cold_cache = format_bytes(cold.bytes_from_cache)
        warm_cache = format_bytes(warm.bytes_from_cache)
        restart_cache = format_bytes(restart.bytes_from_cache)
        print(f"{'Bytes from Cache':<24} {cold_cache:>15} {warm_cache:>15} {restart_cache:>15}")
        print(f"{'Rows':<24} {cold.rows:>15,} {warm.rows:>15,} {restart.rows:>15,}")

        # Speedups - show both total and fetch-only
        print("\n" + "-" * 72)
        warm_total = (
            cold.total_latency_ms / warm.total_latency_ms if warm.total_latency_ms > 0 else 0
        )
        restart_total = (
            cold.total_latency_ms / restart.total_latency_ms if restart.total_latency_ms > 0 else 0
        )
        warm_fetch = (
            cold.fetch_latency_ms / warm.fetch_latency_ms if warm.fetch_latency_ms > 0 else 0
        )
        restart_fetch = (
            cold.fetch_latency_ms / restart.fetch_latency_ms if restart.fetch_latency_ms > 0 else 0
        )

        print(f"\n{'Speedup':<30} {'Warm Cache':>15} {'Post-Restart':>15}")
        print(f"{'  Total (end-to-end)':<30} {warm_total:>14.1f}x {restart_total:>14.1f}x")
        print(f"{'  Fetch (data plane)':<30} {warm_fetch:>14.1f}x {restart_fetch:>14.1f}x")

        # Thesis validation
        print("\n" + "=" * 80)
        if restart.cache_misses == 0 and restart.cache_hits > 0:
            print("THESIS VALIDATED: Cache persisted across server restart!")
            print(f"  - All {restart.cache_hits} cache tasks served from Arrow cache")
            print("  - Zero Parquet decoding required")
        else:
            print("WARNING: Cache did not fully persist")
            print(f"  - Cache hits: {restart.cache_hits}, misses: {restart.cache_misses}")
        print("=" * 80)


@dataclass
class ScaleResult:
    """Results from a single scale point."""

    label: str
    rows: int
    data_size: str  # Human-readable
    cold_fetch_ms: float
    warm_fetch_ms: float
    restart_fetch_ms: float
    cold_total_ms: float
    warm_total_ms: float
    restart_total_ms: float
    warm_speedup: float
    restart_speedup: float


def find_free_port() -> int:
    """Find a free port on localhost."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_single_benchmark(
    num_rows: int,
    num_data_files: int,
    host: str = "127.0.0.1",
    port: int | None = None,
    verbose: bool = True,
) -> list[BenchmarkResult]:
    """Run a single benchmark with the given configuration.

    Returns list of [cold, warm, restart] BenchmarkResult.
    """
    from strata_client import StrataClient

    from strata.cache import CACHE_FILE_EXTENSION

    # Find a free port if not specified
    if port is None:
        port = find_free_port()

    with tempfile.TemporaryDirectory() as tmpdir:
        warehouse_path = Path(tmpdir) / "warehouse"
        warehouse_path.mkdir()
        cache_dir = Path(tmpdir) / "cache"
        cache_dir.mkdir()

        # Create sample table
        if verbose:
            print("\n[1/6] Creating sample Iceberg table...")
        table_uri = create_sample_table(warehouse_path, num_rows, num_data_files)

        base_url = f"http://{host}:{port}"
        results = []

        # === Phase 1: Cold start ===
        if verbose:
            print("\n[2/6] Starting server (first instance)...")
        server = ServerProcess(host, port, cache_dir)
        server.start()

        client = StrataClient(base_url=base_url)

        # Clear cache to ensure cold start
        client.clear_cache()

        if verbose:
            print("\n[3/6] Running cold start scan (cache miss expected)...")
        metrics_before = client.metrics()
        result1 = run_scan(client, table_uri, metrics_before)
        result1.name = "1. Cold Start"
        results.append(result1)
        if verbose:
            print(
                f"       Total: {result1.total_latency_ms:.1f}ms "
                f"(plan: {result1.planning_latency_ms:.1f}ms, "
                f"fetch: {result1.fetch_latency_ms:.1f}ms)"
            )
            print(f"       Cache misses: {result1.cache_misses}")

        # === Phase 2: Warm cache ===
        if verbose:
            print("\n[4/6] Running warm cache scan (cache hit expected)...")
        metrics_before = client.metrics()
        result2 = run_scan(client, table_uri, metrics_before)
        result2.name = "2. Warm Cache"
        results.append(result2)
        if verbose:
            print(
                f"       Total: {result2.total_latency_ms:.1f}ms "
                f"(plan: {result2.planning_latency_ms:.1f}ms, "
                f"fetch: {result2.fetch_latency_ms:.1f}ms)"
            )
            print(f"       Cache hits: {result2.cache_hits}")

        client.close()

        # === Phase 3: Restart ===
        if verbose:
            print("\n[5/6] Stopping and restarting server...")
        server.stop()
        time.sleep(1)

        # Verify cache files exist
        cache_files = list(cache_dir.rglob(f"*{CACHE_FILE_EXTENSION}"))
        if verbose:
            print(f"       Cache files on disk: {len(cache_files)}")

        # Start new server instance
        server = ServerProcess(host, port, cache_dir)
        server.start()

        client = StrataClient(base_url=base_url)

        if verbose:
            print("\n[6/6] Running post-restart scan (cache hit expected)...")
        metrics_before = client.metrics()
        result3 = run_scan(client, table_uri, metrics_before)
        result3.name = "3. Post-Restart"
        results.append(result3)
        if verbose:
            print(
                f"       Total: {result3.total_latency_ms:.1f}ms "
                f"(plan: {result3.planning_latency_ms:.1f}ms, "
                f"fetch: {result3.fetch_latency_ms:.1f}ms)"
            )
            print(f"       Cache hits: {result3.cache_hits}")

        client.close()
        server.stop()

        return results


def print_scale_comparison(scale_results: list[ScaleResult]):
    """Print a comparison table across multiple scale points."""
    print("\n" + "=" * 100)
    print("SCALE COMPARISON: Cache Performance Across Data Sizes")
    print("=" * 100)

    # Header
    print(
        f"\n{'Scale':<12} {'Data Size':>10} {'Cold':>12} {'Warm':>12} "
        f"{'Restart':>12} {'Warm':>10} {'Restart':>10}"
    )
    print(
        f"{'':12} {'':>10} {'Fetch(ms)':>12} {'Fetch(ms)':>12} "
        f"{'Fetch(ms)':>12} {'Speedup':>10} {'Speedup':>10}"
    )
    print("-" * 100)

    for r in scale_results:
        print(
            f"{r.label:<12} {r.data_size:>10} {r.cold_fetch_ms:>12.1f} "
            f"{r.warm_fetch_ms:>12.1f} {r.restart_fetch_ms:>12.1f} "
            f"{r.warm_speedup:>9.1f}x {r.restart_speedup:>9.1f}x"
        )

    print("-" * 100)

    # Observations
    print("\nObservations:")
    if len(scale_results) >= 2:
        first = scale_results[0]
        last = scale_results[-1]
        cold_growth = last.cold_fetch_ms / first.cold_fetch_ms if first.cold_fetch_ms > 0 else 0
        warm_growth = last.warm_fetch_ms / first.warm_fetch_ms if first.warm_fetch_ms > 0 else 0
        restart_growth = (
            last.restart_fetch_ms / first.restart_fetch_ms if first.restart_fetch_ms > 0 else 0
        )
        print(f"  - Cold fetch time grew {cold_growth:.1f}x from smallest to largest scale")
        print(f"  - Warm fetch time grew {warm_growth:.1f}x (cache serving is efficient)")
        print(f"  - Post-restart time grew {restart_growth:.1f}x (cache persists across restart)")

    # Thesis validation
    all_valid = all(r.restart_speedup > 1.0 for r in scale_results)
    print("\n" + "=" * 100)
    if all_valid:
        print("THESIS VALIDATED ACROSS ALL SCALES:")
        print("  - Cache persistence provides speedup at every data size")
        print("  - Post-restart performance tracks warm cache, not cold start")
    else:
        print("WARNING: Some scale points did not show expected speedup")
    print("=" * 100)


def run_scale_benchmark():
    """Run benchmark at multiple scale points."""
    # Scale points: rows, data_files, label
    scale_points = [
        (50_000, 5, "50K rows"),
        (500_000, 10, "500K rows"),
        (5_000_000, 20, "5M rows"),
    ]

    print("=" * 100)
    print("STRATA BENCHMARK: Multi-Scale Cache Persistence Test")
    print("=" * 100)
    print("\nThis benchmark runs at multiple data sizes to demonstrate that:")
    print("  1. Cold start time grows with data size")
    print("  2. Warm cache time grows much more slowly")
    print("  3. Post-restart time stays near warm cache time")

    scale_results = []

    for i, (num_rows, num_files, label) in enumerate(scale_points):
        print(f"\n{'=' * 100}")
        print(f"SCALE POINT {i + 1}/{len(scale_points)}: {label}")
        print(f"  Rows: {num_rows:,}")
        print(f"  Data files: {num_files}")
        print("=" * 100)

        results = run_single_benchmark(num_rows, num_files, verbose=True)
        cold, warm, restart = results

        # Calculate data size from cold bytes
        data_size = format_bytes(cold.bytes_from_storage)

        # Calculate speedups (fetch-only for accuracy)
        warm_speedup = cold.fetch_latency_ms / warm.fetch_latency_ms if warm.fetch_latency_ms else 0
        restart_speedup = (
            cold.fetch_latency_ms / restart.fetch_latency_ms if restart.fetch_latency_ms else 0
        )

        scale_results.append(
            ScaleResult(
                label=label,
                rows=num_rows,
                data_size=data_size,
                cold_fetch_ms=cold.fetch_latency_ms,
                warm_fetch_ms=warm.fetch_latency_ms,
                restart_fetch_ms=restart.fetch_latency_ms,
                cold_total_ms=cold.total_latency_ms,
                warm_total_ms=warm.total_latency_ms,
                restart_total_ms=restart.total_latency_ms,
                warm_speedup=warm_speedup,
                restart_speedup=restart_speedup,
            )
        )

        # Print single-run results
        print_results_table(results)

    # Print comparison table
    print_scale_comparison(scale_results)


def main():
    parser = argparse.ArgumentParser(description="Benchmark cache persistence across restarts")
    parser.add_argument("--rows", type=int, default=50000, help="Number of rows (default: 50000)")
    parser.add_argument(
        "--data-files", type=int, default=5, help="Number of data files (default: 5)"
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Run multi-scale benchmark (50K, 500K, 5M rows)",
    )
    args = parser.parse_args()

    if args.scale:
        run_scale_benchmark()
        return

    print("=" * 80)
    print("STRATA BENCHMARK: Cache Persistence Across Server Restarts")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Rows: {args.rows:,}")
    print(f"  Data files: {args.data_files} (1 row group each = {args.data_files} cache tasks)")

    results = run_single_benchmark(args.rows, args.data_files, verbose=True)
    print_results_table(results)


if __name__ == "__main__":
    main()
