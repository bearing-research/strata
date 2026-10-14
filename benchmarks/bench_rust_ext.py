"""A/B benchmark of the Rust extension (`_strata_core`) against its Python fallback.

Covers ``read_file_bytes`` (mmap read vs ``Path.read_bytes()``). Unlike
``bench_hot_path.py``, each implementation is forced, and the pair is checked
for equivalent output before it is timed.

Run with: uv run python benchmarks/bench_rust_ext.py
"""

from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc

try:
    from strata import _strata_core

    RUST = _strata_core
except ImportError:
    RUST = None


def _timeit(fn, iterations: int) -> dict:
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "min_ms": min(times),
    }


def _make_stream(num_rows: int, num_cols: int = 10) -> bytes:
    cols = {}
    for i in range(num_cols):
        if i % 3 == 0:
            cols[f"int_{i}"] = list(range(num_rows))
        elif i % 3 == 1:
            cols[f"flt_{i}"] = [j * 1.5 for j in range(num_rows)]
        else:
            cols[f"str_{i}"] = [f"v_{j}" for j in range(num_rows)]
    batch = pa.RecordBatch.from_pydict(cols)
    sink = pa.BufferOutputStream()
    writer = ipc.new_stream(sink, batch.schema)
    writer.write_batch(batch)
    writer.close()
    return sink.getvalue().to_pybytes()


def _fmt(name: str, r: dict) -> str:
    return f"    {name:<28} median {r['median_ms']:>8.4f} ms   (min {r['min_ms']:>8.4f})"


# --------------------------------------------------------------------------
# 1. mmap read vs Path.read_bytes()
# --------------------------------------------------------------------------


def bench_read(tmpdir: Path) -> None:
    print("-" * 68)
    print("1. Cache read: Rust mmap vs Python read_bytes()")
    print("-" * 68)
    configs = [
        (10_000, "Small  ~0.7 MB"),
        (100_000, "Medium  ~7 MB"),
        (1_000_000, "Large  ~70 MB"),
    ]
    for num_rows, label in configs:
        data = _make_stream(num_rows)
        path = tmpdir / f"read_{num_rows}.arrowstream"
        path.write_bytes(data)
        size_mb = path.stat().st_size / (1024 * 1024)

        py = path.read_bytes()
        if RUST is not None:
            rs = bytes(RUST.read_file_bytes(str(path)))
            assert rs == py, "mmap read produced different bytes"

        # Warm the OS page cache; both paths benefit equally.
        for _ in range(5):
            path.read_bytes()

        iters = 50
        print(f"\n  {label}  ({size_mb:.1f} MB actual)")
        py_r = _timeit(lambda p=path: p.read_bytes(), iters)
        print(_fmt("Python read_bytes()", py_r))
        if RUST is not None:
            rs_r = _timeit(lambda p=path: bytes(RUST.read_file_bytes(str(p))), iters)
            print(_fmt("Rust mmap read", rs_r))
            print(f"    -> speedup {py_r['median_ms'] / rs_r['median_ms']:.2f}x")
        else:
            print("    Rust unavailable — skipped")


def main() -> None:
    print("=" * 68)
    print("Existing Rust extension — head-to-head A/B")
    print(f"Rust module available: {RUST is not None}")
    print("=" * 68)
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        bench_read(tmpdir)
    print()
    print("=" * 68)
    print("Done. Speedup >1 means Rust is faster.")
    print("=" * 68)


if __name__ == "__main__":
    main()
