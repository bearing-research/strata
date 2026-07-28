"""Decision benchmark: is row-group pruning worth moving to Rust?

The candidate is ``ReadPlanner._should_prune_row_group`` — a Python loop that
runs once per (row group x filter) on every *cold* scan. Before writing any
Rust we need to know two things:

1. How much of cold planning is actually the prunable Python loop, versus the
   PyArrow C++ footer parse that Rust would NOT replace? If the loop is a sliver
   of parse time, Rust buys nothing.
2. How cheap is the *warm* path already? The persisted metadata cache hands the
   loop plain ``RowGroupMeta`` dataclasses (Python dict lookups) instead of
   PyArrow ``RowGroupMetaData`` objects (each ``.column().statistics.min`` is an
   FFI hop). If the dataclass loop is already fast, warm scans don't need Rust.

This isolates the exact code that would move to Rust — no Iceberg, no catalog —
so the numbers speak only to the pruning decision.

Run with: uv run python benchmarks/bench_pruning.py
"""

from __future__ import annotations

import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from strata.filters import Filter, FilterOp
from strata.metadata_cache import ColumnChunkMeta, ColumnStatistics, RowGroupMeta
from strata.planner import ReadPlanner, _build_column_index_map, _compile_filters

ROWS_PER_GROUP = 1_000


def _timeit(fn, iterations: int) -> dict:
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(times), "min_ms": min(times)}


def _write_parquet(path: Path, num_groups: int) -> None:
    """A Parquet file with ``num_groups`` row groups, each 1000 sorted rows.

    ``val`` is globally sorted so per-row-group min/max stats partition cleanly
    — a selective filter can prune most groups, exactly the shape pruning is for.
    """
    n = num_groups * ROWS_PER_GROUP
    table = pa.table(
        {
            "val": pa.array(np.arange(n, dtype=np.int64)),
            "id": pa.array(np.arange(n, dtype=np.int64)[::-1].copy()),
            "score": pa.array(np.arange(n, dtype=np.float64) * 1.5),
            "grp": pa.array(np.arange(n, dtype=np.int64) % 7),
        }
    )
    pq.write_table(table, path, row_group_size=ROWS_PER_GROUP)


def _to_dataclass_repr(meta: pq.FileMetaData, col_index_map: dict[str, int]) -> list[RowGroupMeta]:
    """Convert PyArrow metadata to the warm-cache ``RowGroupMeta`` representation.

    Mirrors what the persisted metadata cache stores and reloads. The cost of
    this conversion itself is the one-time price the warm cache pays; we time it
    separately so it isn't hidden.
    """
    rgs: list[RowGroupMeta] = []
    for i in range(meta.num_row_groups):
        rg = meta.row_group(i)
        cols: dict[int, ColumnChunkMeta] = {}
        for idx in col_index_map.values():
            cm = rg.column(idx)
            if cm.is_stats_set and cm.statistics is not None:
                s = cm.statistics
                cols[idx] = ColumnChunkMeta(
                    is_stats_set=True,
                    statistics=ColumnStatistics(
                        has_min_max=s.has_min_max,
                        min=s.min,
                        max=s.max,
                        null_count=s.null_count,
                    ),
                )
            else:
                cols[idx] = ColumnChunkMeta(is_stats_set=False, statistics=None)
        rgs.append(
            RowGroupMeta(num_rows=rg.num_rows, total_byte_size=rg.total_byte_size, _columns=cols)
        )
    return rgs


def _filters(num_filters: int, num_groups: int) -> list[Filter]:
    n = num_groups * ROWS_PER_GROUP
    # `val >= 90th percentile` prunes ~90% of groups: the realistic selective case.
    hi = int(n * 0.9)
    all_filters = [
        Filter(column="val", op=FilterOp.GE, value=hi),
        Filter(column="id", op=FilterOp.LE, value=hi),
        Filter(column="score", op=FilterOp.GT, value=float(hi)),
        Filter(column="grp", op=FilterOp.LT, value=5),
    ]
    return all_filters[:num_filters]


def _prune_loop(planner: ReadPlanner, rgs, compiled) -> int:
    pruned = 0
    for rg in rgs:
        if planner._should_prune_row_group(rg, compiled):
            pruned += 1
    return pruned


def main() -> None:
    print("=" * 78)
    print("Row-group pruning — Rust decision benchmark")
    print(f"(each row group = {ROWS_PER_GROUP} rows; `val` sorted so pruning is meaningful)")
    print("=" * 78)

    # Bare planner: _should_prune_row_group / _convert_stats never touch config.
    planner = ReadPlanner.__new__(ReadPlanner)

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for num_groups in (200, 1_000, 5_000):
            path = tmpdir / f"rg_{num_groups}.parquet"
            _write_parquet(path, num_groups)
            size_mb = path.stat().st_size / (1024 * 1024)

            meta0 = pq.read_metadata(str(path))
            col_index_map = _build_column_index_map(meta0.schema)

            print("\n" + "-" * 78)
            print(
                f"{num_groups} row groups  ({num_groups * ROWS_PER_GROUP:,} rows, {size_mb:.1f} MB)"
            )
            print("-" * 78)

            # Cost 1: PyArrow footer parse (C++, NOT replaced by Rust pruning).
            parse = _timeit(lambda p=path: pq.read_metadata(str(p)), 20)

            for num_filters in (1, 4):
                compiled_pa = _compile_filters(_filters(num_filters, num_groups), col_index_map)
                pa_rgs = [meta0.row_group(i) for i in range(meta0.num_row_groups)]
                dc_rgs = _to_dataclass_repr(meta0, col_index_map)

                # sanity: both representations prune identically
                p1 = _prune_loop(planner, pa_rgs, compiled_pa)
                p2 = _prune_loop(planner, dc_rgs, compiled_pa)
                assert p1 == p2, f"repr disagreement: {p1} vs {p2}"

                convert = _timeit(lambda m=meta0, c=col_index_map: _to_dataclass_repr(m, c), 10)
                loop_pa = _timeit(lambda: _prune_loop(planner, pa_rgs, compiled_pa), 20)
                loop_dc = _timeit(lambda: _prune_loop(planner, dc_rgs, compiled_pa), 20)

                cold = parse["median_ms"] + loop_pa["median_ms"]
                frac = loop_pa["median_ms"] / cold * 100

                print(f"\n  {num_filters} filter(s) — pruned {p1}/{num_groups} groups")
                print(f"    footer parse [C++]     {parse['median_ms']:>8.3f} ms")
                print(f"    prune loop [PyArrow]   {loop_pa['median_ms']:>8.3f} ms  (cold, target)")
                print(f"    prune loop [dataclass] {loop_dc['median_ms']:>8.3f} ms  (warm)")
                print(f"    convert to warm        {convert['median_ms']:>8.3f} ms")
                print(f"    => loop = {frac:.1f}% of cold planning (parse + loop)")

    print("\n" + "=" * 78)
    print("Read: if the prune loop is a small % of cold planning, Rust pruning")
    print("buys little; the warm dataclass loop shows what steady-state costs.")
    print("=" * 78)


if __name__ == "__main__":
    main()
