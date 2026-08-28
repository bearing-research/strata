"""Metadata caches for planning optimization.

These caches reduce planning time by avoiding redundant metadata reads:
- Parquet metadata: Cached per file path (schema, row group info, statistics)
- Manifest resolution: Cached per (table_identity, snapshot_id)

Architecture:
- In-memory LRU cache for fast access during normal operation
- SQLite backing store for persistence across restarts
- On cache miss: check SQLite, then load from source
- Parallel I/O for loading multiple files (configurable worker count)

Both use simple LRU eviction with configurable sizes.
"""

import json
from collections import OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, overload

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from strata.metadata_store import MetadataStore, PersistedParquetMeta


class LRUCache[K: Hashable, V]:
    """Thread-safe LRU cache with configurable max size (entry count).

    Suitable for metadata caches where entries are roughly similar size
    (schemas, statistics, manifest entries). For variable-size data
    (Arrow batches, row groups), use a byte-based cache instead.
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
            # Reinsert at end (most recently used)
            self._cache[key] = value
            self._hits += 1
            return value

    def put(self, key: K, value: V) -> None:
        """Put a value in the cache, evicting oldest if at capacity."""
        if self._max_size <= 0:
            return
        with self._lock:
            if key in self._cache:
                # Update existing and move to end
                self._cache[key] = value
                self._cache.move_to_end(key)
                self._updates += 1
            else:
                # Evict one if at capacity (O(1) since we only insert one at a time)
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                    self._evictions += 1
                self._cache[key] = value

    def get_or_put(self, key: K, factory: Callable[[], V]) -> V:
        """Get value if cached, otherwise compute and cache it.

        This avoids thundering herd by:
        1. Check cache under lock, return if hit
        2. Release lock, compute value (I/O happens here)
        3. Re-acquire lock, insert if still absent

        Note: In high-concurrency scenarios, multiple threads may compute
        the same value simultaneously, but only one will be cached.
        This is acceptable for idempotent factories.
        """
        # Fast path: check if already cached
        cached = self.get(key)
        if cached is not None:
            return cached

        # Slow path: compute outside lock
        value = factory()

        # Insert if still absent (another thread may have inserted)
        with self._lock:
            if key in self._cache:
                # Another thread beat us, use their value
                self._cache.move_to_end(key)
                return self._cache[key]
            # Evict one if at capacity
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
    """Minimal column chunk metadata for pruning."""

    is_stats_set: bool
    statistics: ColumnStatistics | None


@dataclass
class RowGroupMeta:
    """Minimal row group metadata for pruning.

    Compatible with PyArrow's RowGroupMetaData interface used in planner.
    """

    num_rows: int
    total_byte_size: int  # Size from Parquet metadata for pre-flight estimates
    _columns: dict  # column_name -> ColumnChunkMeta

    def column(self, idx: int) -> ColumnChunkMeta:
        """Get column metadata by index."""
        # Map index to column name and return metadata
        # If we don't have this column, return empty stats
        if idx in self._columns:
            return self._columns[idx]
        return ColumnChunkMeta(is_stats_set=False, statistics=None)


@dataclass
class ParquetSchema:
    """Minimal schema for column lookups."""

    _column_names: list[str]

    def __len__(self) -> int:
        return len(self._column_names)

    def column(self, idx: int):
        """Get column info by index.

        ``_column_names`` holds dotted **paths** (``user.id``); ``name`` is the
        leaf segment, matching pyarrow's own ``ColumnSchema`` split. The
        planner relies on ``path`` carrying the dot so it can skip nested
        columns when building its column-index map.
        """
        path = self._column_names[idx]
        return type("Col", (), {"name": path.rsplit(".", 1)[-1], "path": path})()


@dataclass
class ParquetMetadata:
    """Cached Parquet file metadata.

    Contains everything needed for planning without re-reading the file:
    - Arrow schema for type information
    - Number of row groups
    - Per-row-group metadata (num_rows, statistics)
    """

    arrow_schema: pa.Schema
    num_row_groups: int
    row_group_metadata: list  # List of RowGroupMeta or pq.RowGroupMetaData objects
    parquet_schema: object  # ParquetSchema or pq.ParquetSchema for column lookups


@dataclass
class ManifestEntry:
    """A single file entry from manifest resolution.

    Stores the resolved data file information from Iceberg manifest.
    """

    file_path: str  # Original file path from manifest
    actual_path: str  # Resolved path for reading


@dataclass
class ManifestResolution:
    """Cached manifest resolution result for a snapshot.

    Contains the list of data files from resolving Iceberg manifests.
    """

    data_files: list[ManifestEntry]


def _json_safe_stat_value(value: object) -> object:
    """Convert a statistics value to a JSON-serializable representation."""
    if hasattr(value, "as_py"):
        value = value.as_py()  # type: ignore[union-attr]
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

    # Dotted PATHS, not leaf names. A struct field ``user.id`` has leaf name
    # ``id``; keying by that collides with a top-level ``id`` — the nested
    # column's stats overwrote the top-level column's, and on restore the
    # planner's ``"." in col.path`` guard could no longer tell them apart, so
    # pruning compared a filter against the WRONG column's min/max and dropped
    # matching rows.
    column_names = [
        metadata.parquet_schema.column(i).path  # type: ignore[union-attr]
        for i in range(len(metadata.parquet_schema))  # type: ignore[arg-type]
    ]
    row_groups = []
    for row_group in metadata.row_group_metadata:
        column_stats: dict[str, dict[str, object]] = {}
        for idx, column_name in enumerate(column_names):
            column_meta = row_group.column(idx)
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
            )
        )

    return PersistedParquetMeta(
        arrow_schema_bytes=serialize_arrow_schema(metadata.arrow_schema),
        num_row_groups=metadata.num_row_groups,
        row_groups=row_groups,
        column_names=column_names,
    )


def _persisted_meta_is_legacy_leaf_named(persisted: "PersistedParquetMeta") -> bool:
    """True when a persisted row predates path-keyed column stats.

    Older rows stored Parquet **leaf** names in ``column_names``. For a flat
    schema leaf name == path, so those rows stay valid. But when a file has
    nested columns the leaf names collide (a struct ``user.id`` and a
    top-level ``id`` both stored as ``"id"``), which both dropped one
    column's stats and defeated the planner's nested-column guard — pruning
    then compared a filter against the wrong column's min/max and silently
    dropped rows.

    Duplicates in the list are exactly that signature: real paths are unique
    per physical column, so a duplicate can only come from a legacy
    leaf-named row. Such rows are treated as a miss and re-read (which
    re-persists them correctly) rather than trusted.
    """
    names = persisted.column_names
    return len(set(names)) != len(names)


class ParquetMetadataCache:
    """Cache for Parquet file metadata with optional SQLite persistence.

    Avoids re-reading Parquet file footers on every scan.
    Key: file path (string)
    Value: ParquetMetadata

    Architecture:
    - In-memory LRU cache for fast access
    - Optional SQLite store for persistence across restarts
    - Parallel I/O when loading multiple files (get_or_load_many)

    Typical size: 1000 files = ~10-50 MB depending on schema complexity.

    S3 Support:
    - Pass an S3FileSystem to read from S3 paths (s3://bucket/path)
    - S3 filesystem is created lazily if not provided but S3 paths are accessed
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
        """Get cached metadata or load from file.

        This is the primary API - it transparently handles cache misses.
        Lookup order: in-memory cache -> SQLite store -> Parquet file
        """
        # Check in-memory cache first
        cached = self._cache.get(file_path)
        if cached is not None:
            return cached

        # Check persistent store if available
        if self._store is not None:
            persisted = self._load_from_store(file_path)
            if persisted is not None:
                self._cache.put(file_path, persisted)
                return persisted

        # Load from file
        metadata = self._load_metadata(file_path)
        self._cache.put(file_path, metadata)

        # Persist to store if available
        if self._store is not None:
            self._save_to_store(file_path, metadata)

        return metadata

    def get_or_load_many(self, file_paths: list[str]) -> dict[str, ParquetMetadata]:
        """Get cached metadata for multiple files, loading missing ones in parallel.

        More efficient than calling get_or_load() in a loop:
        - Batches SQLite queries for persistence layer
        - Uses ThreadPoolExecutor for parallel file I/O on cache misses

        This is critical for cold table performance where we need to read
        many Parquet file footers. Sequential reads of 50 files × 50ms = 2.5s,
        but parallel reads can reduce this to ~500ms (5x speedup).

        Returns dict mapping file_path -> ParquetMetadata for all requested files.
        """
        if not file_paths:
            return {}

        result: dict[str, ParquetMetadata] = {}
        missing_from_memory: list[str] = []

        # Check in-memory cache first
        for fp in file_paths:
            cached = self._cache.get(fp)
            if cached is not None:
                result[fp] = cached
            else:
                missing_from_memory.append(fp)

        if not missing_from_memory:
            return result

        # Batch lookup from persistent store
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

        # Load remaining from files IN PARALLEL
        # This is the key optimization: parallel I/O for Parquet footer reads
        loaded: dict[str, ParquetMetadata] = {}
        errors: dict[str, Exception] = {}

        # Use min of max_workers and number of files to avoid thread overhead
        num_workers = min(self._max_workers, len(missing_from_store))

        if num_workers == 1:
            # Single file: no thread overhead
            fp = missing_from_store[0]
            try:
                loaded[fp] = self._load_metadata(fp)
            except Exception as e:
                errors[fp] = e
        else:
            # Multiple files: parallel loading
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

        # Update cache and result with loaded metadata
        for fp, metadata in loaded.items():
            self._cache.put(fp, metadata)
            result[fp] = metadata

        # Re-raise first error if any files failed to load
        if errors:
            first_path, first_error = next(iter(errors.items()))
            raise RuntimeError(f"Failed to load Parquet metadata for {first_path}: {first_error}")

        # Batch persist to store without rereading the Parquet files.
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

            # Build row group metadata from persisted data
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
                        columns[idx] = ColumnChunkMeta(is_stats_set=True, statistics=stats)
                    else:
                        columns[idx] = ColumnChunkMeta(is_stats_set=False, statistics=None)

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

        # Convert persisted metadata to our compatible types
        try:
            arrow_schema = deserialize_arrow_schema(persisted.arrow_schema_bytes)

            # Build row group metadata from persisted data
            row_group_meta = []
            for rg in persisted.row_groups:
                # Convert column stats to our format, indexed by column position
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
                        columns[idx] = ColumnChunkMeta(is_stats_set=True, statistics=stats)
                    else:
                        columns[idx] = ColumnChunkMeta(is_stats_set=False, statistics=None)

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
            pass  # Don't fail if persistence fails

    def _load_metadata(self, file_path: str) -> ParquetMetadata:
        """Load metadata from a Parquet file."""
        # Handle S3 paths
        if file_path.startswith("s3://"):
            if self._s3_filesystem is None:
                # Create default S3 filesystem on demand
                import pyarrow.fs as pafs

                self._s3_filesystem = pafs.S3FileSystem()
            # Strip s3:// prefix for PyArrow filesystem
            s3_path = file_path[5:]
            pq_file = pq.ParquetFile(s3_path, filesystem=self._s3_filesystem)
        else:
            pq_file = pq.ParquetFile(file_path)

        # Extract row group metadata (we store references, not copies)
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
    """Cache for Iceberg manifest resolution results with optional persistence.

    Avoids re-resolving manifests on every scan for the same snapshot.

    Two-level caching:
    - Unfiltered: Key is (catalog, table, snapshot) -> all files
    - Filtered: Key is (catalog, table, snapshot, filter_fingerprint) -> pruned files

    The unfiltered cache is used for persistence and as a fallback.
    The filtered cache stores results of Iceberg file-level pruning.

    Architecture:
    - In-memory LRU cache for fast access
    - Optional SQLite store for persistence across restarts (unfiltered only)

    Note: This cache is invalidated when a new snapshot is created,
    since the key includes snapshot_id.
    """

    def __init__(self, max_size: int = 100, store: "MetadataStore | None" = None) -> None:
        # Unfiltered cache: (catalog, table, snapshot) -> all files
        self._cache: LRUCache[tuple[str, str, int], ManifestResolution] = LRUCache(max_size)
        # Filtered cache: (catalog, table, snapshot, filter_fp) -> pruned files
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
        """Get cached manifest resolution.

        Args:
            catalog_name: Catalog name
            table_identity: Table identity string
            snapshot_id: Snapshot ID
            filter_fingerprint: Filter fingerprint for filtered queries (default: "nofilter")

        Lookup order:
        - If filter_fingerprint != "nofilter": check filtered cache
        - Check unfiltered in-memory cache as a correctness-preserving fallback
        - Check SQLite store for the persisted unfiltered resolution
        """
        # For filtered queries, check filtered cache first
        if filter_fingerprint != "nofilter":
            cached = self._filtered_cache.get(
                (catalog_name, table_identity, snapshot_id, filter_fingerprint)
            )
            if cached is not None:
                return cached

        # Check unfiltered cache as the correctness-preserving fallback.
        cached = self._cache.get((catalog_name, table_identity, snapshot_id))
        if cached is not None:
            return cached

        if self._store is not None:
            persisted = self._store.get_manifest(catalog_name, table_identity, snapshot_id)
            if persisted is not None:
                resolution = ManifestResolution(
                    data_files=[ManifestEntry(file_path=fp, actual_path=ap) for fp, ap in persisted]
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
        """Cache manifest resolution.

        Args:
            catalog_name: Catalog name
            table_identity: Table identity string
            snapshot_id: Snapshot ID
            resolution: Manifest resolution to cache
            filter_fingerprint: Filter fingerprint (default: "nofilter" for unfiltered)
        """
        if filter_fingerprint != "nofilter":
            # Cache filtered result (in-memory only, not persisted)
            self._filtered_cache.put(
                (catalog_name, table_identity, snapshot_id, filter_fingerprint), resolution
            )
        else:
            # Cache unfiltered result
            self._cache.put((catalog_name, table_identity, snapshot_id), resolution)

            # Persist to store if available (unfiltered only)
            if self._store is not None:
                try:
                    data_files = [
                        (entry.file_path, entry.actual_path) for entry in resolution.data_files
                    ]
                    self._store.put_manifest(catalog_name, table_identity, snapshot_id, data_files)
                except Exception:
                    pass  # Don't fail if persistence fails

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


# Global singleton caches for use across the application.
# These are created lazily and can be configured via set_*_cache().

_parquet_cache: ParquetMetadataCache | None = None
_manifest_cache: ManifestCache | None = None
_metadata_store: "MetadataStore | None" = None
_cache_lock = Lock()  # Protects all global cache singletons


def get_metadata_store(cache_dir: Path | None = None) -> "MetadataStore":
    """Get the global metadata store (creates if needed).

    If cache_dir is provided and differs from existing store's path,
    a new store is created for the new path.

    Thread-safe: uses a lock to prevent multiple threads from racing
    to create MetadataStore instances simultaneously.
    """
    global _metadata_store
    from strata.metadata_store import MetadataStore

    if cache_dir is None:
        cache_dir = Path.home() / ".strata" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected_db_path = cache_dir / "metadata.sqlite"

    with _cache_lock:
        # Check if existing store uses the same path
        if _metadata_store is not None and _metadata_store.db_path != expected_db_path:
            # Different path requested, create new store
            _metadata_store = MetadataStore(expected_db_path)
        elif _metadata_store is None:
            _metadata_store = MetadataStore(expected_db_path)

        return _metadata_store


def get_parquet_cache(
    max_size: int = 1000,
    cache_dir: Path | None = None,
    s3_filesystem: "pa.fs.S3FileSystem | None" = None,
    max_workers: int = 8,
) -> ParquetMetadataCache:
    """Get the global Parquet metadata cache (creates if needed).

    Args:
        max_size: Maximum number of entries in LRU cache
        cache_dir: Directory for SQLite persistence (None to disable persistence)
        s3_filesystem: Optional S3 filesystem for reading from S3 paths
        max_workers: Maximum threads for parallel file I/O (default 8)

    Note: If cache_dir is provided and differs from existing cache's store path,
    a new cache with the correct store will be created.

    Thread-safe: uses a lock to prevent race conditions during creation.
    """
    global _parquet_cache

    # Get metadata store outside the lock to avoid nested locking
    store = None
    expected_db_path = None
    if cache_dir is not None:
        store = get_metadata_store(cache_dir)
        expected_db_path = cache_dir / "metadata.sqlite"

    with _cache_lock:
        if cache_dir is not None:
            # Check if existing cache uses different store path
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
                    # Update existing cache with S3 filesystem
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
            # Update existing cache with S3 filesystem
            _parquet_cache._s3_filesystem = s3_filesystem

        return _parquet_cache


def get_manifest_cache(max_size: int = 100, cache_dir: Path | None = None) -> ManifestCache:
    """Get the global manifest cache (creates if needed).

    Args:
        max_size: Maximum number of entries in LRU cache
        cache_dir: Directory for SQLite persistence (None to disable persistence)

    Note: If cache_dir is provided and differs from existing cache's store path,
    a new cache with the correct store will be created.

    Thread-safe: uses a lock to prevent race conditions during creation.
    """
    global _manifest_cache

    # Get metadata store outside the lock to avoid nested locking
    store = None
    expected_db_path = None
    if cache_dir is not None:
        store = get_metadata_store(cache_dir)
        expected_db_path = cache_dir / "metadata.sqlite"

    with _cache_lock:
        if cache_dir is not None:
            # Check if existing cache uses different store path
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
