"""Benchmark the cache-hit hot path: file read, mmap read and IPC concat.

The cache stores Arrow IPC stream bytes, so a hit is a plain file read with no
Arrow parsing.

Run with: uv run python benchmarks/bench_hot_path.py
"""

import statistics
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

from strata import fast_io


def create_test_data(num_rows: int, num_columns: int = 10) -> pa.RecordBatch:
    """Create a test record batch with specified size."""
    columns = {}
    for i in range(num_columns):
        if i % 3 == 0:
            columns[f"int_col_{i}"] = list(range(num_rows))
        elif i % 3 == 1:
            columns[f"float_col_{i}"] = [float(j) * 1.5 for j in range(num_rows)]
        else:
            columns[f"str_col_{i}"] = [f"value_{j}" for j in range(num_rows)]
    return pa.RecordBatch.from_pydict(columns)


def create_stream_file(path: Path, batch: pa.RecordBatch) -> None:
    """Create an Arrow IPC Stream format file (like our cache)."""
    sink = pa.BufferOutputStream()
    writer = ipc.new_stream(sink, batch.schema)
    writer.write_batch(batch)
    writer.close()
    path.write_bytes(sink.getvalue().to_pybytes())


def benchmark_raw_file_read(file_path: Path, num_iterations: int = 100) -> dict:
    """Benchmark raw file read (Python Path.read_bytes())."""
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = file_path.read_bytes()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_mmap_file_read(file_path: Path, num_iterations: int = 100) -> dict:
    """Benchmark mmap file read (Rust mmap via fast_io)."""
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = fast_io.read_file_mmap(str(file_path))
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_parse_and_serve(file_path: Path, num_iterations: int = 100) -> dict:
    """Benchmark read + parse + reserialize, the baseline the raw read avoids."""
    times = []

    for _ in range(num_iterations):
        start = time.perf_counter()

        stream_bytes = file_path.read_bytes()
        reader = ipc.open_stream(pa.BufferReader(stream_bytes))
        batches = list(reader)

        if batches:
            sink = pa.BufferOutputStream()
            writer = ipc.new_stream(sink, batches[0].schema)
            for batch in batches:
                writer.write_batch(batch)
            writer.close()
            _ = sink.getvalue().to_pybytes()

        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def benchmark_concat_stream_bytes(segments: list[bytes], num_iterations: int = 100) -> dict:
    """Benchmark concat_stream_bytes."""
    times = []
    for _ in range(num_iterations):
        start = time.perf_counter()
        _ = fast_io.concat_stream_bytes(segments)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "stdev_ms": statistics.stdev(times) if len(times) > 1 else 0,
        "min_ms": min(times),
        "max_ms": max(times),
    }


def format_results(name: str, results: dict) -> str:
    """Format benchmark results for display."""
    return (
        f"{name}:\n"
        f"  Mean:   {results['mean_ms']:>8.3f} ms\n"
        f"  Median: {results['median_ms']:>8.3f} ms\n"
        f"  Stdev:  {results['stdev_ms']:>8.3f} ms\n"
        f"  Min:    {results['min_ms']:>8.3f} ms\n"
        f"  Max:    {results['max_ms']:>8.3f} ms"
    )


def main():
    print("=" * 60)
    print("Strata Hot Path Benchmark")
    print("=" * 60)
    print()
    print("Cache format: Arrow IPC Stream (.arrowstream)")
    print("Hot path: raw file read (zero Arrow parsing)")
    print()

    configs = [
        {"num_rows": 1_000, "num_columns": 10, "label": "Small (1K rows, 10 cols)"},
        {"num_rows": 10_000, "num_columns": 10, "label": "Medium (10K rows, 10 cols)"},
        {"num_rows": 100_000, "num_columns": 10, "label": "Large (100K rows, 10 cols)"},
        {"num_rows": 100_000, "num_columns": 50, "label": "Wide (100K rows, 50 cols)"},
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for config in configs:
            print("-" * 60)
            print(f"Benchmarking: {config['label']}")
            print("-" * 60)

            batch = create_test_data(config["num_rows"], config["num_columns"])
            file_path = tmpdir / f"test_{config['num_rows']}_{config['num_columns']}.arrowstream"

            # Same format the cache writes.
            create_stream_file(file_path, batch)

            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"File size: {file_size_mb:.2f} MB")
            print()

            # Warm up
            for _ in range(5):
                _ = file_path.read_bytes()

            num_iterations = 50

            print("Cache Hit Performance:")
            raw_read = benchmark_raw_file_read(file_path, num_iterations)
            print(format_results("  Python read_bytes()", raw_read))
            print()

            mmap_read = benchmark_mmap_file_read(file_path, num_iterations)
            print(format_results("  Rust mmap read", mmap_read))
            print()

            parse_serve = benchmark_parse_and_serve(file_path, num_iterations)
            print(format_results("  Old path (parse + serialize)", parse_serve))
            print()

            if raw_read["mean_ms"] > 0:
                mmap_vs_raw = raw_read["mean_ms"] / mmap_read["mean_ms"]
                print(f"  Mmap vs read_bytes: {mmap_vs_raw:.2f}x")
            speedup = parse_serve["mean_ms"] / mmap_read["mean_ms"]
            print(f"  Mmap vs parse+serialize: {speedup:.1f}x faster")
            print()

        print("-" * 60)
        print("Benchmarking: Concat Multiple Segments")
        print("-" * 60)

        num_segments = 10
        segments = []
        for i in range(num_segments):
            batch = create_test_data(10_000, 10)
            file_path = tmpdir / f"segment_{i}.arrowstream"
            create_stream_file(file_path, batch)
            segments.append(file_path.read_bytes())

        total_size_mb = sum(len(s) for s in segments) / (1024 * 1024)
        print(f"Total segments: {num_segments}, Total size: {total_size_mb:.2f} MB")
        print()

        # Warm up
        for _ in range(5):
            fast_io.concat_stream_bytes(segments)

        num_iterations = 50

        print("Concat Stream Bytes:")
        concat_results = benchmark_concat_stream_bytes(segments, num_iterations)
        print(format_results("  PyArrow C++ concat", concat_results))
        print()

    print("=" * 60)
    print("Benchmark complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
