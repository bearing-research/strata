"""Memory profiling utilities for Strata.

This module provides detailed memory profiling information for diagnosing
memory-related performance issues, including:
- PyArrow memory pool allocations
- Python memory statistics
- Process-level memory usage
- Memory allocation patterns

These are exposed via /v1/debug/memory for operational diagnostics.
"""

import gc
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

import pyarrow as pa


class ArrowSnapshotDict(TypedDict):
    """Arrow memory snapshot payload."""

    bytes_allocated: int
    max_memory: int
    pool_backend: str
    allocated_mb: float
    max_mb: float


class PythonSnapshotDict(TypedDict):
    """Python GC snapshot payload."""

    gc_tracked_objects: int
    gc_objects_by_gen: list[int]


class ProcessSnapshotDict(TypedDict):
    """Process memory snapshot payload."""

    rss_bytes: int | None
    vms_bytes: int | None
    rss_mb: float | None
    vms_mb: float | None


class MemorySnapshotDict(TypedDict):
    """Top-level memory snapshot payload."""

    arrow: ArrowSnapshotDict
    python: PythonSnapshotDict
    process: ProcessSnapshotDict


class DefaultArrowPoolDict(TypedDict):
    """Default Arrow pool allocation payload."""

    backend: str
    bytes_allocated: int
    max_memory: int


class AvailableArrowPoolDict(TypedDict):
    """Available Arrow pool allocation payload."""

    name: str
    bytes_allocated: int
    max_memory: int


class ArrowAllocationsDict(TypedDict):
    """Detailed Arrow allocation payload."""

    default_pool: DefaultArrowPoolDict
    available_pools: list[AvailableArrowPoolDict]


class GcThresholdsDict(TypedDict):
    """GC thresholds payload."""

    gen0: int
    gen1: int
    gen2: int


class GcGenerationStatsDict(TypedDict):
    """GC generation statistics payload."""

    generation: int
    collections: int
    collected: int
    uncollectable: int


class TopObjectTypeDict(TypedDict):
    """Top Python object type count payload."""

    type: str
    count: int


class PythonMemoryStatsDict(TypedDict):
    """Detailed Python memory payload."""

    gc_thresholds: GcThresholdsDict
    gc_stats: list[GcGenerationStatsDict]
    gc_is_enabled: bool
    gc_freeze_count: int | None
    top_object_types: list[TopObjectTypeDict]
    total_objects: int


class DetailedMemoryReportDict(TypedDict):
    """Top-level detailed memory report payload."""

    snapshot: MemorySnapshotDict
    arrow_details: ArrowAllocationsDict
    python_details: PythonMemoryStatsDict
    recommendations: list[str]


@dataclass
class MemorySnapshot:
    """Point-in-time memory snapshot."""

    arrow_bytes_allocated: int
    arrow_max_memory: int
    arrow_pool_backend: str

    python_gc_tracked: int  # Objects tracked by GC
    python_gc_objects_by_gen: list[int]  # Count per generation

    process_rss_bytes: int | None
    process_vms_bytes: int | None

    def to_dict(self) -> MemorySnapshotDict:
        return {
            "arrow": {
                "bytes_allocated": self.arrow_bytes_allocated,
                "max_memory": self.arrow_max_memory,
                "pool_backend": self.arrow_pool_backend,
                "allocated_mb": round(self.arrow_bytes_allocated / (1024 * 1024), 2),
                "max_mb": round(self.arrow_max_memory / (1024 * 1024), 2),
            },
            "python": {
                "gc_tracked_objects": self.python_gc_tracked,
                "gc_objects_by_gen": self.python_gc_objects_by_gen,
            },
            "process": {
                "rss_bytes": self.process_rss_bytes,
                "vms_bytes": self.process_vms_bytes,
                "rss_mb": (
                    round(self.process_rss_bytes / (1024 * 1024), 2)
                    if self.process_rss_bytes
                    else None
                ),
                "vms_mb": (
                    round(self.process_vms_bytes / (1024 * 1024), 2)
                    if self.process_vms_bytes
                    else None
                ),
            },
        }


def get_memory_snapshot() -> MemorySnapshot:
    """Capture current memory state across Arrow, Python, and process levels.

    This is a relatively cheap operation suitable for periodic sampling.

    Returns:
        MemorySnapshot with current memory statistics
    """
    pool = pa.default_memory_pool()
    arrow_bytes = pool.bytes_allocated()
    arrow_max = pool.max_memory()
    arrow_backend = pool.backend_name

    gc_stats = gc.get_stats()
    gc_tracked = sum(s["collections"] for s in gc_stats)
    gc_objects = [len(gc.get_objects(i)) for i in range(3)]

    rss_bytes = None
    vms_bytes = None
    try:
        import psutil

        process = psutil.Process()
        mem_info = process.memory_info()
        rss_bytes = mem_info.rss
        vms_bytes = mem_info.vms
    except ImportError:
        # No psutil: fall back to /proc/self/statm (Linux).
        try:
            with open("/proc/self/statm") as f:
                parts = f.read().split()
                page_size = 4096  # Typical page size
                vms_bytes = int(parts[0]) * page_size
                rss_bytes = int(parts[1]) * page_size
        except (FileNotFoundError, IndexError, ValueError):
            pass

    return MemorySnapshot(
        arrow_bytes_allocated=arrow_bytes,
        arrow_max_memory=arrow_max,
        arrow_pool_backend=arrow_backend,
        python_gc_tracked=gc_tracked,
        python_gc_objects_by_gen=gc_objects,
        process_rss_bytes=rss_bytes,
        process_vms_bytes=vms_bytes,
    )


def get_arrow_allocations() -> ArrowAllocationsDict:
    """Get detailed Arrow memory allocation information.

    Returns information about all available Arrow memory pools
    and their current allocation state.
    """
    result: ArrowAllocationsDict = {
        "default_pool": {
            "backend": pa.default_memory_pool().backend_name,
            "bytes_allocated": pa.default_memory_pool().bytes_allocated(),
            "max_memory": pa.default_memory_pool().max_memory(),
        },
        "available_pools": [],
    }

    pools_to_check = [
        ("system", pa.system_memory_pool),
    ]

    try:
        pools_to_check.append(("jemalloc", pa.jemalloc_memory_pool))
    except AttributeError:
        pass

    try:
        pools_to_check.append(("mimalloc", pa.mimalloc_memory_pool))
    except AttributeError:
        pass

    for name, pool_fn in pools_to_check:
        try:
            pool = pool_fn()
            result["available_pools"].append(
                {
                    "name": name,
                    "bytes_allocated": pool.bytes_allocated(),
                    "max_memory": pool.max_memory(),
                }
            )
        except Exception:
            pass

    return result


def get_python_memory_stats() -> PythonMemoryStatsDict:
    """Get detailed Python memory statistics.

    Includes information about:
    - Object counts by type (top types)
    - GC thresholds and counts
    - Reference cycle information
    """
    gc_stats = gc.get_stats()
    thresholds = gc.get_threshold()

    # Can be expensive for large heaps.
    type_counts: dict[str, int] = {}
    try:
        for obj in gc.get_objects():
            type_name = type(obj).__name__
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
    except Exception:
        pass

    top_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:20]

    result: PythonMemoryStatsDict = {
        "gc_thresholds": {
            "gen0": thresholds[0],
            "gen1": thresholds[1],
            "gen2": thresholds[2],
        },
        "gc_stats": [
            {
                "generation": i,
                "collections": s["collections"],
                "collected": s["collected"],
                "uncollectable": s["uncollectable"],
            }
            for i, s in enumerate(gc_stats)
        ],
        "gc_is_enabled": gc.isenabled(),
        "gc_freeze_count": gc.get_freeze_count() if hasattr(gc, "get_freeze_count") else None,
        "top_object_types": [{"type": t, "count": c} for t, c in top_types],
        "total_objects": sum(type_counts.values()),
    }
    return result


def get_detailed_memory_report() -> DetailedMemoryReportDict:
    """Get comprehensive memory report for debugging.

    This is more expensive than get_memory_snapshot() and should
    only be called on-demand (not for periodic sampling).

    Returns:
        Dictionary with Arrow, Python, and process memory details
    """
    snapshot = get_memory_snapshot()
    arrow_details = get_arrow_allocations()
    python_details = get_python_memory_stats()
    return {
        "snapshot": snapshot.to_dict(),
        "arrow_details": arrow_details,
        "python_details": python_details,
        "recommendations": _get_memory_recommendations(snapshot, python_details),
    }


def _get_memory_recommendations(
    snapshot: MemorySnapshot,
    python_stats: Mapping[str, object] | int,
) -> list[str]:
    """Generate memory-related recommendations based on current state."""
    recommendations = []
    total_objects = (
        python_stats
        if isinstance(python_stats, int)
        else _read_int_value(python_stats.get("total_objects"))
    )

    if snapshot.arrow_bytes_allocated > 1024 * 1024 * 1024:  # > 1GB
        recommendations.append(
            f"Arrow has {snapshot.arrow_bytes_allocated / (1024**3):.1f}GB allocated. "
            "Consider checking for retained batches or memory leaks."
        )

    # Max far above current suggests fragmentation.
    if snapshot.arrow_max_memory > 2 * snapshot.arrow_bytes_allocated > 0:
        recommendations.append(
            f"Arrow max memory ({snapshot.arrow_max_memory / (1024**3):.1f}GB) is much higher "
            f"than current ({snapshot.arrow_bytes_allocated / (1024**3):.1f}GB). "
            "This may indicate memory fragmentation."
        )

    if total_objects > 1_000_000:
        recommendations.append(
            f"Python GC is tracking {total_objects:,} objects. "
            "High object count can slow down GC. Consider object pooling."
        )

    if snapshot.process_rss_bytes and snapshot.process_rss_bytes > 4 * 1024**3:
        recommendations.append(
            f"Process RSS is {snapshot.process_rss_bytes / (1024**3):.1f}GB. "
            "Consider increasing memory limits or optimizing memory usage."
        )

    if not recommendations:
        recommendations.append("Memory usage looks healthy.")

    return recommendations


def _read_int_value(value: object | None) -> int:
    """Safely read an integer from a loosely typed payload."""
    return value if isinstance(value, int) else 0
