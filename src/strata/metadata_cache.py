"""Metadata caches for planning: Parquet footers per file path, manifest resolution per
(table identity, snapshot).

Each is an in-memory LRU over an optional SQLite store that persists across restarts.
"""

import json
from collections import OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, overload

import pyarrow as pa

if TYPE_CHECKING:
    import pyarrow.fs

    from strata.metadata_store import MetadataStore, PersistedParquetMeta


class LRUCache[K: Hashable, V]:
    """Thread-safe LRU cache bounded by entry count.

    Suits entries of roughly similar size (schemas, statistics); use a byte-bounded cache for
    variable-size data.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._cache: OrderedDict[K, V] = OrderedDict()
        self._max_size = max_size
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
        self._updates = 0
        self._evictions = 0

    @overload
    def get(self, key: K) -> V | None: ...

    @overload
    def get(self, key: K, default: V) -> V: ...

    def get(self, key: K, default: V | None = None) -> V | None:
        """Get a value from the cache, returning default if not found."""
        with self._lock:
            try:
                value = self._cache.pop(key)
            except KeyError:
                self._misses += 1
                return default
            self._cache[key] = value
            self._hits += 1
            return value

    def put(self, key: K, value: V) -> None:
        """Put a value in the cache, evicting oldest if at capacity."""
        if self._max_size <= 0:
            return
        with self._lock:
            if key in self._cache:
                self._cache[key] = value
                self._cache.move_to_end(key)
                self._updates += 1
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                    self._evictions += 1
                self._cache[key] = value

    def get_or_put(self, key: K, factory: Callable[[], V]) -> V:
        """Return the cached value, or compute it with *factory* and cache it.

        The factory runs outside the lock, so concurrent misses may compute the same value; only
        one is cached. Factories must be idempotent.
        """
        cached = self.get(key)
        if cached is not None:
            return cached

        # Compute outside the lock
        value = factory()

        with self._lock:
            if key in self._cache:
                # Another thread won the race; use its value
                self._cache.move_to_end(key)
                return self._cache[key]
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1
            self._cache[key] = value
        return value

    def resize(self, new_max_size: int) -> None:
        """Resize the cache, evicting oldest entries if needed."""
        with self._lock:
            self._max_size = new_max_size
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

    def __contains__(self, key: K) -> bool:
        """Check if a key is in the cache (does not update LRU order)."""
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._updates = 0
            self._evictions = 0

    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "updates": self._updates,
                "evictions": self._evictions,
            }

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


@dataclass
class ColumnStatistics:
    """Minimal column statistics for pruning."""

    has_min_max: bool = False
    min: object = None
    max: object = None
    null_count: int | None = None


@dataclass
class ColumnChunkMeta:
    """Minimal column chunk metadata for pruning and size estimation.

    ``total_uncompressed_size`` mirrors pyarrow's field so the planner reads either type;
    0 means "not recorded", which the planner treats as unknown.
    """

    is_stats_set: bool
    statistics: ColumnStatistics | None
    total_uncompressed_size: int = 0


@dataclass
class RowGroupMeta:
    """Minimal row group metadata, compatible with the pyarrow ``RowGroupMetaData`` interface
    the planner uses.
    """

    num_rows: int
    total_byte_size: int  # For pre-flight estimates
    _columns: dict  # column index -> ColumnChunkMeta

    def column(self, idx: int) -> ColumnChunkMeta:
        """Get column metadata by index."""
        # Unknown column: empty stats
        if idx in self._columns:
            return self._columns[idx]
        return ColumnChunkMeta(is_stats_set=False, statistics=None)


class _ColumnPath(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def path(self) -> str: ...


class _SchemaColumns(Protocol):
    """What callers read off a Parquet schema: pyarrow's ``ParquetSchema`` or the one below,
    rebuilt from the persisted cache.
    """

    def __len__(self) -> int: ...

    def column(self, i: int, /) -> _ColumnPath: ...


class _Column(NamedTuple):
    name: str
    path: str


@dataclass
class ParquetSchema:
    """Minimal schema for column lookups."""

    _column_names: list[str]

    def __len__(self) -> int:
        return len(self._column_names)

    def column(self, idx: int) -> _Column:
        """Get column info by index.

        ``_column_names`` holds dotted paths (``user.id``); ``name`` is the leaf, matching
        pyarrow's ``ColumnSchema``. The planner relies on the dot in ``path`` to skip nested
        columns.
        """
        path = self._column_names[idx]
        return _Column(name=path.rsplit(".", 1)[-1], path=path)


@dataclass
class ParquetMetadata:
    """Cached Parquet file metadata: everything planning needs without re-reading the file."""

    arrow_schema: pa.Schema
    num_row_groups: int
    row_group_metadata: list  # RowGroupMeta or pq.RowGroupMetaData
    parquet_schema: _SchemaColumns  # ParquetSchema or pq.ParquetSchema


@dataclass(frozen=True)
class DeleteFileEntry:
    """A positional delete file, or a deletion vector, that applies to a data file."""

    file_path: str  # As the manifest names it; read through the table's FileIO
    file_format: str  # "PARQUET" or "PUFFIN"


@dataclass(frozen=True)
class EqualityDeleteEntry:
    """An equality delete file that applies to a data file (see iceberg_equality)."""

    file_path: str  # As the manifest names it
    actual_path: str  # Resolved for reading, like a data file's
    equality_ids: tuple[int, ...]  # Rows equal on these field ids are deleted
    record_count: int  # For the equality-delete limit
    # Per key field from the manifest: (field id, lower, upper, null count), bounds encoded
    # by iceberg_equality.encode_bound.
    bounds: tuple[tuple[int, str | None, str | None, int | None], ...] = ()

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "EqualityDeleteEntry":
        """Rebuild an entry ``dataclasses.asdict`` wrote (JSON turns tuples into lists)."""
        return cls(
            file_path=data["file_path"],
            actual_path=data["actual_path"],
            equality_ids=tuple(data["equality_ids"]),
            record_count=data["record_count"],
            bounds=tuple(tuple(bound) for bound in data["bounds"]),
        )


@dataclass
class ManifestEntry:
    """A resolved data file from an Iceberg manifest, with the deletes that apply to it."""

    file_path: str  # As the manifest names it
    actual_path: str  # Resolved for reading
    # Positional deletes; which apply is pyiceberg's call (sequence numbers, partition,
    # referenced file).
    delete_files: tuple[DeleteFileEntry, ...] = ()
    # Sequence number and partition already matched; see iceberg_equality.
    equality_deletes: tuple[EqualityDeleteEntry, ...] = ()
    # Identity-partition values, which a source column the file omits reads as:
    # (source field id, Iceberg type, single-value encoding as hex).
    partition_values: tuple[tuple[int, str, str], ...] = ()


@dataclass
class ManifestResolution:
    """Cached manifest resolution for a snapshot: its data files."""

    data_files: list[ManifestEntry]


def _json_safe_stat_value(value: object) -> object:
    """Convert a statistics value to a JSON-serializable representation."""
    as_py = getattr(value, "as_py", None)
    if callable(as_py):
        value = as_py()
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def _persisted_parquet_meta_from_loaded(metadata: ParquetMetadata) -> "PersistedParquetMeta":
    """Convert loaded Parquet metadata to the persisted representation."""
    from strata.metadata_store import (
        PersistedParquetMeta,
        PersistedRowGroupMeta,
        serialize_arrow_schema,
    )

    # Dotted paths, not leaf names: a struct field ``user.id`` would collide with a top-level
    # ``id``, and pruning would compare against the wrong column's min/max and drop rows.
    column_names = [
        metadata.parquet_schema.column(i).path for i in range(len(metadata.parquet_schema))
    ]
    row_groups = []
    for row_group in metadata.row_group_metadata:
        column_stats: dict[str, dict[str, object]] = {}
        column_sizes: dict[str, int] = {}
        for idx, column_name in enumerate(column_names):
            column_meta = row_group.column(idx)
            # Every column, stats or not: the pre-flight estimate needs projected sizes.
            column_sizes[column_name] = column_meta.total_uncompressed_size
            if not column_meta.is_stats_set or column_meta.statistics is None:
                continue

            stats = column_meta.statistics
            stat_dict: dict[str, object] = {}
            if getattr(stats, "has_min_max", False):
                stat_dict["min"] = _json_safe_stat_value(stats.min)
                stat_dict["max"] = _json_safe_stat_value(stats.max)
            null_count = getattr(stats, "null_count", None)
            if null_count is not None:
                stat_dict["null_count"] = null_count
            if stat_dict:
                column_stats[column_name] = stat_dict

        row_groups.append(
            PersistedRowGroupMeta(
                num_rows=row_group.num_rows,
                total_byte_size=row_group.total_byte_size,
                column_stats=column_stats,
                column_sizes=column_sizes,
            )
        )

    return PersistedParquetMeta(
        arrow_schema_bytes=serialize_arrow_schema(metadata.arrow_schema),
        num_row_groups=metadata.num_row_groups,
        row_groups=row_groups,
        column_names=column_names,
    )


def _persisted_meta_is_legacy_leaf_named(persisted: "PersistedParquetMeta") -> bool:
    """True when a persisted row stored Parquet leaf names rather than paths in ``column_names``.

    With nested columns leaf names collide (``user.id`` and ``id`` both ``"id"``), which made
    pruning compare a filter against the wrong column's stats and drop rows. Real paths are
    unique, so a duplicate marks such a row; it is treated as a miss and re-read.
    """
    names = persisted.column_names
    return len(set(names)) != len(names)


class ParquetMetadataCache:
    """Parquet footer cache keyed by file path, with optional SQLite persistence.

    Roughly 10-50 MB per 1000 files, depending on schema. ``s3_filesystem``, when given, is
    used to open ``s3://`` paths.
    """

    def __init__(
        self,
        max_size: int = 1000,
        store: "MetadataStore | None" = None,
        s3_filesystem: "pa.fs.S3FileSystem | None" = None,
        max_workers: int = 8,
    ) -> None:
        self._cache: LRUCache[str, ParquetMetadata] = LRUCache(max_size)
        self._store = store
        self._s3_filesystem = s3_filesystem
        self._max_workers = max_workers

    def get(self, file_path: str) -> ParquetMetadata | None:
        """Get cached metadata for a file."""
        return self._cache.get(file_path)

    def get_or_load(self, file_path: str) -> ParquetMetadata:
        """Get metadata from memory, then the SQLite store, then the Parquet file."""
        cached = self._cache.get(file_path)
        if cached is not None:
            return cached

        if self._store is not None:
            persisted = self._load_from_store(file_path)
            if persisted is not None:
                self._cache.put(file_path, persisted)
                return persisted

        metadata = self._load_metadata(file_path)
        self._cache.put(file_path, metadata)

        if self._store is not None:
            self._save_to_store(file_path, metadata)

        return metadata

    def get_or_load_many(self, file_paths: list[str]) -> dict[str, ParquetMetadata]:
        """Get metadata for many files, keyed by path, loading misses in parallel.

        Batches the SQLite lookups and reads missing footers on a thread pool; this dominates
        cold-table planning time.
        """
        if not file_paths:
            return {}

        result: dict[str, ParquetMetadata] = {}
        missing_from_memory: list[str] = []

        for fp in file_paths:
            cached = self._cache.get(fp)
            if cached is not None:
                result[fp] = cached
            else:
                missing_from_memory.append(fp)

        if not missing_from_memory:
            return result

        missing_from_store: list[str] = []
        if self._store is not None:
            persisted_batch = self._store.get_parquet_meta_many(missing_from_memory)
            for fp in missing_from_memory:
                if fp in persisted_batch:
                    meta = self._convert_persisted(persisted_batch[fp])
                    if meta is not None:
                        self._cache.put(fp, meta)
                        result[fp] = meta
                    else:
                        missing_from_store.append(fp)
                else:
                    missing_from_store.append(fp)
        else:
            missing_from_store = missing_from_memory

        if not missing_from_store:
            return result

        # Parallel footer reads are the main win here
        loaded: dict[str, ParquetMetadata] = {}
        errors: dict[str, Exception] = {}

        num_workers = min(self._max_workers, len(missing_from_store))

        if num_workers == 1:
            # Skip the pool for one file
            fp = missing_from_store[0]
            try:
                loaded[fp] = self._load_metadata(fp)
            except Exception as e:
                errors[fp] = e
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_path = {
                    executor.submit(self._load_metadata, fp): fp for fp in missing_from_store
                }
                for future in as_completed(future_to_path):
                    fp = future_to_path[future]
                    try:
                        loaded[fp] = future.result()
                    except Exception as e:
                        errors[fp] = e

        for fp, metadata in loaded.items():
            self._cache.put(fp, metadata)
            result[fp] = metadata

        if errors:
            first_path, first_error = next(iter(errors.items()))
            raise RuntimeError(f"Failed to load Parquet metadata for {first_path}: {first_error}")

        if loaded and self._store is not None:
            to_persist: list[tuple[str, PersistedParquetMeta]] = []

            for fp, metadata in loaded.items():
                try:
                    persisted = _persisted_parquet_meta_from_loaded(metadata)
                    to_persist.append((fp, persisted))
                except Exception:
                    pass

            if to_persist:
                try:
                    self._store.put_parquet_meta_many(to_persist)
                except Exception:
                    pass

        return result

    def _convert_persisted(self, persisted: "PersistedParquetMeta") -> ParquetMetadata | None:
        """Convert persisted metadata to ParquetMetadata."""
        from strata.metadata_store import deserialize_arrow_schema

        if _persisted_meta_is_legacy_leaf_named(persisted):
            return None

        try:
            arrow_schema = deserialize_arrow_schema(persisted.arrow_schema_bytes)

            row_group_meta = []
            for rg in persisted.row_groups:
                columns = {}
                for idx, col_name in enumerate(persisted.column_names):
                    if col_name in rg.column_stats:
                        stats_dict = rg.column_stats[col_name]
                        stats = ColumnStatistics(
                            has_min_max="min" in stats_dict and "max" in stats_dict,
                            min=stats_dict.get("min"),
                            max=stats_dict.get("max"),
                            null_count=stats_dict.get("null_count"),
                        )
                        columns[idx] = ColumnChunkMeta(
                            is_stats_set=True,
                            statistics=stats,
                            total_uncompressed_size=rg.column_sizes.get(col_name, 0),
                        )
                    else:
                        columns[idx] = ColumnChunkMeta(
                            is_stats_set=False,
                            statistics=None,
                            total_uncompressed_size=rg.column_sizes.get(col_name, 0),
                        )

                row_group_meta.append(
                    RowGroupMeta(
                        num_rows=rg.num_rows,
                        total_byte_size=rg.total_byte_size,
                        _columns=columns,
                    )
                )

            return ParquetMetadata(
                arrow_schema=arrow_schema,
                num_row_groups=persisted.num_row_groups,
                row_group_metadata=row_group_meta,
                parquet_schema=ParquetSchema(_column_names=persisted.column_names),
            )
        except Exception:
            return None

    def _load_from_store(self, file_path: str) -> ParquetMetadata | None:
        """Load metadata from persistent store without reading the file."""
        from strata.metadata_store import deserialize_arrow_schema

        if self._store is None:
            return None
        persisted = self._store.get_parquet_meta(file_path)
        if persisted is None:
            return None

        if _persisted_meta_is_legacy_leaf_named(persisted):
            return None

        try:
            arrow_schema = deserialize_arrow_schema(persisted.arrow_schema_bytes)

            row_group_meta = []
            for rg in persisted.row_groups:
                # Keyed by column position
                columns = {}
                for idx, col_name in enumerate(persisted.column_names):
                    if col_name in rg.column_stats:
                        stats_dict = rg.column_stats[col_name]
                        stats = ColumnStatistics(
                            has_min_max="min" in stats_dict and "max" in stats_dict,
                            min=stats_dict.get("min"),
                            max=stats_dict.get("max"),
                            null_count=stats_dict.get("null_count"),
                        )
                        columns[idx] = ColumnChunkMeta(
                            is_stats_set=True,
                            statistics=stats,
                            total_uncompressed_size=rg.column_sizes.get(col_name, 0),
                        )
                    else:
                        columns[idx] = ColumnChunkMeta(
                            is_stats_set=False,
                            statistics=None,
                            total_uncompressed_size=rg.column_sizes.get(col_name, 0),
                        )

                row_group_meta.append(
                    RowGroupMeta(
                        num_rows=rg.num_rows,
                        total_byte_size=rg.total_byte_size,
                        _columns=columns,
                    )
                )

            return ParquetMetadata(
                arrow_schema=arrow_schema,
                num_row_groups=persisted.num_row_groups,
                row_group_metadata=row_group_meta,
                parquet_schema=ParquetSchema(_column_names=persisted.column_names),
            )
        except Exception:
            return None

    def _save_to_store(self, file_path: str, metadata: ParquetMetadata) -> None:
        """Save metadata to persistent store."""
        if self._store is None:
            return
        try:
            persisted = _persisted_parquet_meta_from_loaded(metadata)
            self._store.put_parquet_meta(file_path, persisted)
        except Exception:
            pass  # Persistence is best-effort

    def _load_metadata(self, file_path: str) -> ParquetMetadata:
        """Load metadata from a Parquet file."""
        from strata.lake_files import open_parquet

        pq_file = open_parquet(file_path, self._s3_filesystem)

        # References, not copies
        row_group_meta = []
        for i in range(pq_file.metadata.num_row_groups):
            row_group_meta.append(pq_file.metadata.row_group(i))

        return ParquetMetadata(
            arrow_schema=pq_file.schema_arrow,
            num_row_groups=pq_file.metadata.num_row_groups,
            row_group_metadata=row_group_meta,
            parquet_schema=pq_file.metadata.schema,
        )

    def put(self, file_path: str, metadata: ParquetMetadata) -> None:
        """Manually put metadata in the cache."""
        self._cache.put(file_path, metadata)

    def clear(self) -> None:
        """Clear all cached metadata."""
        self._cache.clear()

    def stats(self) -> dict:
        """Get cache statistics."""
        return self._cache.stats()


class ManifestCache:
    """Cache of Iceberg manifest resolutions, keyed by snapshot so entries never go stale.

    Unfiltered entries (all files) persist to SQLite; filtered entries (keyed also by filter
    fingerprint) hold file-pruned results in memory only.
    """

    def __init__(self, max_size: int = 100, store: "MetadataStore | None" = None) -> None:
        # (catalog, table, snapshot) -> all files
        self._cache: LRUCache[tuple[str, str, int], ManifestResolution] = LRUCache(max_size)
        # (catalog, table, snapshot, filter_fp) -> pruned files
        self._filtered_cache: LRUCache[tuple[str, str, int, str], ManifestResolution] = LRUCache(
            max_size * 2
        )
        self._store = store

    def get(
        self,
        catalog_name: str,
        table_identity: str,
        snapshot_id: int,
        filter_fingerprint: str = "nofilter",
    ) -> ManifestResolution | None:
        """Get a cached manifest resolution, or ``None``.

        Tries the filtered entry (when ``filter_fingerprint != "nofilter"``), then the unfiltered
        in-memory entry, then the SQLite store. An unfiltered result is a correct superset for any
        filter.
        """
        if filter_fingerprint != "nofilter":
            cached = self._filtered_cache.get(
                (catalog_name, table_identity, snapshot_id, filter_fingerprint)
            )
            if cached is not None:
                return cached

        # Unfiltered files are a correct superset for any filter
        cached = self._cache.get((catalog_name, table_identity, snapshot_id))
        if cached is not None:
            return cached

        if self._store is not None:
            persisted = self._store.get_manifest(catalog_name, table_identity, snapshot_id)
            if persisted is not None:
                resolution = ManifestResolution(
                    data_files=[
                        ManifestEntry(
                            file_path=entry["file_path"],
                            actual_path=entry["actual_path"],
                            delete_files=tuple(
                                DeleteFileEntry(**delete) for delete in entry["delete_files"]
                            ),
                            equality_deletes=tuple(
                                EqualityDeleteEntry.from_json(delete)
                                for delete in entry["equality_deletes"]
                            ),
                            partition_values=tuple(
                                tuple(value) for value in entry["partition_values"]
                            ),
                        )
                        for entry in persisted
                    ]
                )
                self._cache.put((catalog_name, table_identity, snapshot_id), resolution)
                return resolution

        return None

    def put(
        self,
        catalog_name: str,
        table_identity: str,
        snapshot_id: int,
        resolution: ManifestResolution,
        filter_fingerprint: str = "nofilter",
    ) -> None:
        """Cache a manifest resolution; ``"nofilter"`` stores it as the unfiltered entry."""
        if filter_fingerprint != "nofilter":
            # In memory only, not persisted
            self._filtered_cache.put(
                (catalog_name, table_identity, snapshot_id, filter_fingerprint), resolution
            )
        else:
            self._cache.put((catalog_name, table_identity, snapshot_id), resolution)

            if self._store is not None:
                try:
                    data_files = [asdict(entry) for entry in resolution.data_files]
                    self._store.put_manifest(catalog_name, table_identity, snapshot_id, data_files)
                except Exception:
                    pass  # Persistence is best-effort

    def clear(self) -> None:
        """Clear all cached resolutions."""
        self._cache.clear()
        self._filtered_cache.clear()

    def stats(self) -> dict:
        """Get cache statistics."""
        unfiltered = self._cache.stats()
        filtered = self._filtered_cache.stats()
        return {
            "unfiltered": unfiltered,
            "filtered": filtered,
        }


# Lazy process-wide singletons; override with set_*_cache().

_parquet_cache: ParquetMetadataCache | None = None
_manifest_cache: ManifestCache | None = None
_metadata_store: "MetadataStore | None" = None
_cache_lock = Lock()  # Guards all the singletons


def get_metadata_store(cache_dir: Path | None = None) -> "MetadataStore":
    """Get the global metadata store, creating it (or replacing it for a new ``cache_dir``).

    The whole body holds ``_cache_lock`` once: two acquisitions would let a no-arg caller
    clobber a store another caller just created for the configured ``cache_dir``. The cache
    getters call this before taking the lock, so this cannot deadlock.
    """
    global _metadata_store
    from strata.metadata_store import MetadataStore

    with _cache_lock:
        if cache_dir is None:
            # No cache_dir means "the store in use", not the personal-mode default:
            # resolving the default here would swap the singleton out from under a server
            # with another cache_dir on every /health/ready probe.
            if _metadata_store is not None:
                return _metadata_store
            cache_dir = Path.home() / ".strata" / "cache"

        cache_dir.mkdir(parents=True, exist_ok=True)
        expected_db_path = cache_dir / "metadata.sqlite"

        if _metadata_store is None or _metadata_store.db_path != expected_db_path:
            _metadata_store = MetadataStore(expected_db_path)

        return _metadata_store


def get_parquet_cache(
    max_size: int = 1000,
    cache_dir: Path | None = None,
    s3_filesystem: "pa.fs.S3FileSystem | None" = None,
    max_workers: int = 8,
) -> ParquetMetadataCache:
    """Get the global Parquet metadata cache, creating it if needed (thread-safe).

    ``cache_dir=None`` disables persistence; a ``cache_dir`` whose store differs from the
    existing cache's replaces the cache.
    """
    global _parquet_cache

    # Outside the lock to avoid nested locking
    store = None
    expected_db_path = None
    if cache_dir is not None:
        store = get_metadata_store(cache_dir)
        expected_db_path = cache_dir / "metadata.sqlite"

    with _cache_lock:
        if cache_dir is not None:
            if _parquet_cache is not None:
                if (
                    _parquet_cache._store is None
                    or _parquet_cache._store.db_path != expected_db_path
                ):
                    _parquet_cache = ParquetMetadataCache(
                        max_size,
                        store=store,
                        s3_filesystem=s3_filesystem,
                        max_workers=max_workers,
                    )
                elif s3_filesystem is not None and _parquet_cache._s3_filesystem is None:
                    _parquet_cache._s3_filesystem = s3_filesystem
            else:
                _parquet_cache = ParquetMetadataCache(
                    max_size, store=store, s3_filesystem=s3_filesystem, max_workers=max_workers
                )
        elif _parquet_cache is None:
            _parquet_cache = ParquetMetadataCache(
                max_size, store=None, s3_filesystem=s3_filesystem, max_workers=max_workers
            )
        elif s3_filesystem is not None and _parquet_cache._s3_filesystem is None:
            _parquet_cache._s3_filesystem = s3_filesystem

        return _parquet_cache


def get_manifest_cache(max_size: int = 100, cache_dir: Path | None = None) -> ManifestCache:
    """Get the global manifest cache, creating it if needed (thread-safe).

    ``cache_dir=None`` disables persistence; a ``cache_dir`` whose store differs from the
    existing cache's replaces the cache.
    """
    global _manifest_cache

    # Outside the lock to avoid nested locking
    store = None
    expected_db_path = None
    if cache_dir is not None:
        store = get_metadata_store(cache_dir)
        expected_db_path = cache_dir / "metadata.sqlite"

    with _cache_lock:
        if cache_dir is not None:
            if _manifest_cache is not None:
                if (
                    _manifest_cache._store is None
                    or _manifest_cache._store.db_path != expected_db_path
                ):
                    _manifest_cache = ManifestCache(max_size, store=store)
            else:
                _manifest_cache = ManifestCache(max_size, store=store)
        elif _manifest_cache is None:
            _manifest_cache = ManifestCache(max_size, store=None)

        return _manifest_cache


def clear_all_caches() -> None:
    """Clear all global metadata caches."""
    with _cache_lock:
        if _parquet_cache is not None:
            _parquet_cache.clear()
        if _manifest_cache is not None:
            _manifest_cache.clear()
        if _metadata_store is not None:
            _metadata_store.clear()


def reset_caches() -> None:
    """Reset global caches (for testing)."""
    global _parquet_cache, _manifest_cache, _metadata_store
    with _cache_lock:
        _parquet_cache = None
        _manifest_cache = None
        _metadata_store = None
