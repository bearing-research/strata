"""Persistent metadata store using SQLite.

Persists planning metadata to survive server restarts:
- Manifest resolution: (table_identity, snapshot_id) -> list of data files
- Parquet metadata: file_path -> (schema, row_groups, stats)

This makes post-restart planning fast by avoiding:
- Re-resolving Iceberg manifests
- Re-reading Parquet file footers
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    import pyarrow.fs as pafs


@dataclass
class PersistedRowGroupMeta:
    """Serializable row group metadata for persistence."""

    num_rows: int
    total_byte_size: int
    # Column statistics: {col_name: {min: val, max: val, null_count: int}}
    column_stats: dict[str, dict]


@dataclass
class PersistedParquetMeta:
    """Serializable Parquet file metadata."""

    arrow_schema_bytes: bytes  # Serialized Arrow schema
    num_row_groups: int
    row_groups: list[PersistedRowGroupMeta]
    column_names: list[str]


def _local_path_for_stat(file_path: str) -> Path | None:
    """Return a local path for stat-based validation, or None for remote URIs.

    Windows paths like ``C:\\Users\\...`` parse with scheme ``"c"``
    because ``urlparse`` treats single letters followed by ``:`` as a
    scheme. Detect that pattern first and treat it as a local path.
    """
    # Windows drive letter, e.g. "C:\..." — a real URI scheme is always
    # >= 2 chars, so a 1-char "scheme" can only be a Windows drive.
    if len(file_path) >= 2 and file_path[1] == ":" and file_path[0].isalpha():
        return Path(file_path)
    parsed = urlparse(file_path)
    if not parsed.scheme:
        return Path(file_path)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return None


def _stat_identity(file_path: str) -> tuple[float | None, int | None]:
    """Return (mtime, size) for local files, or (None, None) for remote URIs."""
    local_path = _local_path_for_stat(file_path)
    if local_path is None:
        return None, None
    stat = local_path.stat()
    return stat.st_mtime, stat.st_size


class MetadataStore:
    """SQLite-backed persistent metadata store.

    Stores:
    - manifest_cache: (table_identity, snapshot_id) -> JSON list of file entries
    - parquet_meta: file_path -> serialized ParquetMeta

    Thread-safe via connection-per-thread pattern (WAL mode).

    Future optimizations if JSON blobs become a bottleneck:
    - Compress row_groups_json with zstd/gzip for large files
    - Normalize to separate table: parquet_row_group(file_path, rg_idx, ...)
      for partial updates and row-group-level queries
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        # Counters for observability
        self.manifest_hits = 0
        self.manifest_misses = 0
        self.parquet_meta_hits = 0
        self.parquet_meta_misses = 0
        self.stale_invalidations = 0
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a connection (creates new one per call for thread safety).

        Uses connection-per-call pattern rather than connection pooling.
        Each connection is used with a context manager and closed automatically.
        No close() method needed on MetadataStore itself.

        If switching to per-thread connection pooling for performance,
        add a close() method to clean up thread-local connections.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        # Performance and concurrency pragmas
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA busy_timeout=30000")  # 30s timeout
        return conn

    def _init_db(self) -> None:
        """Initialize database schema, migrating if needed."""
        with self._get_conn() as conn:
            # Check if we need to migrate manifest_cache (add catalog_name)
            cursor = conn.execute("PRAGMA table_info(manifest_cache)")
            columns = {row[1] for row in cursor.fetchall()}
            if columns and "catalog_name" not in columns:
                # Old schema without catalog_name - drop and recreate
                conn.execute("DROP TABLE IF EXISTS manifest_cache")
                conn.execute("DROP INDEX IF EXISTS idx_manifest_snapshot")

            # Check if we need to migrate parquet_meta (add file_size)
            cursor = conn.execute("PRAGMA table_info(parquet_meta)")
            columns = {row[1] for row in cursor.fetchall()}
            if columns and "file_size" not in columns:
                # Old schema without file_size - drop and recreate
                conn.execute("DROP TABLE IF EXISTS parquet_meta")

            conn.executescript("""
                CREATE TABLE IF NOT EXISTS manifest_cache (
                    catalog_name TEXT NOT NULL,
                    table_identity TEXT NOT NULL,
                    snapshot_id INTEGER NOT NULL,
                    data_files_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (catalog_name, table_identity, snapshot_id)
                );

                CREATE TABLE IF NOT EXISTS parquet_meta (
                    file_path TEXT PRIMARY KEY,
                    schema_ipc BLOB NOT NULL,
                    num_row_groups INTEGER NOT NULL,
                    row_groups_json TEXT NOT NULL,
                    column_names_json TEXT NOT NULL,
                    file_mtime REAL,
                    file_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_manifest_lookup
                ON manifest_cache(catalog_name, table_identity, snapshot_id);
            """)

    def get_manifest(
        self, catalog_name: str, table_identity: str, snapshot_id: int
    ) -> list[tuple[str, str]] | None:
        """Get cached manifest resolution.

        Returns list of (file_path, actual_path) tuples, or None if not cached.
        """
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT data_files_json FROM manifest_cache
                   WHERE catalog_name = ? AND table_identity = ? AND snapshot_id = ?""",
                (catalog_name, table_identity, snapshot_id),
            ).fetchone()

            if row is None:
                self.manifest_misses += 1
                return None

            self.manifest_hits += 1
            entries = json.loads(row["data_files_json"])
            return [(e["file_path"], e["actual_path"]) for e in entries]

    def put_manifest(
        self,
        catalog_name: str,
        table_identity: str,
        snapshot_id: int,
        data_files: list[tuple[str, str]],
    ) -> None:
        """Store manifest resolution.

        Args:
            catalog_name: Catalog name (e.g., 'default', 'prod')
            table_identity: Canonical table identity string
            snapshot_id: Iceberg snapshot ID
            data_files: List of (file_path, actual_path) tuples
        """
        entries = [{"file_path": fp, "actual_path": ap} for fp, ap in data_files]
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO manifest_cache
                   (catalog_name, table_identity, snapshot_id, data_files_json)
                   VALUES (?, ?, ?, ?)""",
                (catalog_name, table_identity, snapshot_id, json.dumps(entries)),
            )
            conn.commit()
        finally:
            conn.close()

    def get_parquet_meta(self, file_path: str) -> PersistedParquetMeta | None:
        """Get cached Parquet metadata.

        Returns PersistedParquetMeta or None if not cached or stale.
        Validates file (mtime, size) to detect stale entries.

        Note: Does not delete stale entries inline to avoid write locks
        during reads. Stale entries are overwritten on next put() or
        cleaned up via cleanup_stale_parquet_meta().
        """
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM parquet_meta WHERE file_path = ?",
                (file_path,),
            ).fetchone()

            if row is None:
                self.parquet_meta_misses += 1
                return None

            # Check if file has been modified using (mtime, size) tuple
            # This handles: mtime going backwards, same mtime with different content
            local_path = _local_path_for_stat(file_path)
            if local_path is not None:
                try:
                    stat = local_path.stat()
                    current_mtime = stat.st_mtime
                    current_size = stat.st_size
                    stored_mtime = row["file_mtime"]
                    stored_size = row["file_size"]
                    if stored_mtime is not None and stored_size is not None:
                        if current_mtime != stored_mtime or current_size != stored_size:
                            # Stale entry - return None, let put() overwrite later
                            self.stale_invalidations += 1
                            self.parquet_meta_misses += 1
                            return None
                except OSError:
                    # File doesn't exist - return None, cleanup will handle it
                    self.stale_invalidations += 1
                    self.parquet_meta_misses += 1
                    return None

            self.parquet_meta_hits += 1
            # Deserialize row groups
            row_groups_data = json.loads(row["row_groups_json"])
            row_groups = [
                PersistedRowGroupMeta(
                    num_rows=rg["num_rows"],
                    total_byte_size=rg["total_byte_size"],
                    column_stats=rg.get("column_stats", {}),
                )
                for rg in row_groups_data
            ]

            return PersistedParquetMeta(
                arrow_schema_bytes=row["schema_ipc"],
                num_row_groups=row["num_row_groups"],
                row_groups=row_groups,
                column_names=json.loads(row["column_names_json"]),
            )

    def put_parquet_meta(self, file_path: str, meta: PersistedParquetMeta) -> None:
        """Store Parquet metadata."""
        try:
            file_mtime, file_size = _stat_identity(file_path)
        except OSError:
            file_mtime = None
            file_size = None

        row_groups_json = json.dumps(
            [
                {
                    "num_rows": rg.num_rows,
                    "total_byte_size": rg.total_byte_size,
                    "column_stats": rg.column_stats,
                }
                for rg in meta.row_groups
            ]
        )

        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT OR REPLACE INTO parquet_meta
                   (file_path, schema_ipc, num_row_groups, row_groups_json,
                    column_names_json, file_mtime, file_size)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    file_path,
                    meta.arrow_schema_bytes,
                    meta.num_row_groups,
                    row_groups_json,
                    json.dumps(meta.column_names),
                    file_mtime,
                    file_size,
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def get_parquet_meta_many(self, file_paths: list[str]) -> dict[str, PersistedParquetMeta]:
        """Get cached Parquet metadata for multiple files in one query.

        More efficient than calling get_parquet_meta() in a loop.
        Returns dict mapping file_path -> metadata for found (non-stale) entries.
        """
        if not file_paths:
            return {}

        result: dict[str, PersistedParquetMeta] = {}
        with self._get_conn() as conn:
            # Use WHERE IN with placeholders
            placeholders = ",".join("?" * len(file_paths))
            rows = conn.execute(
                f"SELECT * FROM parquet_meta WHERE file_path IN ({placeholders})",
                file_paths,
            ).fetchall()

            for row in rows:
                file_path = row["file_path"]
                local_path = _local_path_for_stat(file_path)

                # Remote URIs have no local stat identity to validate against.
                if local_path is not None:
                    try:
                        stat = local_path.stat()
                        stored_mtime = row["file_mtime"]
                        stored_size = row["file_size"]
                        if stored_mtime is not None and stored_size is not None:
                            if stat.st_mtime != stored_mtime or stat.st_size != stored_size:
                                self.stale_invalidations += 1
                                self.parquet_meta_misses += 1
                                continue
                    except OSError:
                        self.stale_invalidations += 1
                        self.parquet_meta_misses += 1
                        continue

                self.parquet_meta_hits += 1
                # Deserialize
                row_groups_data = json.loads(row["row_groups_json"])
                row_groups = [
                    PersistedRowGroupMeta(
                        num_rows=rg["num_rows"],
                        total_byte_size=rg["total_byte_size"],
                        column_stats=rg.get("column_stats", {}),
                    )
                    for rg in row_groups_data
                ]
                result[file_path] = PersistedParquetMeta(
                    arrow_schema_bytes=row["schema_ipc"],
                    num_row_groups=row["num_row_groups"],
                    row_groups=row_groups,
                    column_names=json.loads(row["column_names_json"]),
                )

        # Count misses for paths not found
        self.parquet_meta_misses += len(file_paths) - len(result)
        return result

    def put_parquet_meta_many(self, items: list[tuple[str, PersistedParquetMeta]]) -> None:
        """Store multiple Parquet metadata entries in one transaction.

        More efficient than calling put_parquet_meta() in a loop.
        Args:
            items: List of (file_path, metadata) tuples
        """
        if not items:
            return

        rows = []
        for file_path, meta in items:
            try:
                file_mtime, file_size = _stat_identity(file_path)
            except OSError:
                file_mtime = None
                file_size = None

            row_groups_json = json.dumps(
                [
                    {
                        "num_rows": rg.num_rows,
                        "total_byte_size": rg.total_byte_size,
                        "column_stats": rg.column_stats,
                    }
                    for rg in meta.row_groups
                ]
            )
            rows.append(
                (
                    file_path,
                    meta.arrow_schema_bytes,
                    meta.num_row_groups,
                    row_groups_json,
                    json.dumps(meta.column_names),
                    file_mtime,
                    file_size,
                )
            )

        conn = self._get_conn()
        try:
            conn.executemany(
                """INSERT OR REPLACE INTO parquet_meta
                   (file_path, schema_ipc, num_row_groups, row_groups_json,
                    column_names_json, file_mtime, file_size)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
        finally:
            conn.close()

    def clear(self) -> None:
        """Clear all cached data."""
        conn = self._get_conn()
        try:
            conn.execute("DELETE FROM manifest_cache")
            conn.execute("DELETE FROM parquet_meta")
            conn.commit()
        finally:
            conn.close()

    def stats(self) -> dict:
        """Get store statistics."""
        with self._get_conn() as conn:
            manifest_count = conn.execute("SELECT COUNT(*) FROM manifest_cache").fetchone()[0]
            parquet_count = conn.execute("SELECT COUNT(*) FROM parquet_meta").fetchone()[0]

        return {
            "manifest_entries": manifest_count,
            "parquet_entries": parquet_count,
            "manifest_hits": self.manifest_hits,
            "manifest_misses": self.manifest_misses,
            "parquet_meta_hits": self.parquet_meta_hits,
            "parquet_meta_misses": self.parquet_meta_misses,
            "stale_invalidations": self.stale_invalidations,
            "db_path": str(self.db_path),
        }

    def cleanup_stale_parquet_meta(self) -> int:
        """Remove stale parquet_meta entries where files no longer exist or changed.

        Returns the number of entries removed. Call periodically (e.g., on startup
        or via background task) to reclaim space from stale entries.
        """
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT file_path, file_mtime, file_size FROM parquet_meta"
            ).fetchall()

            stale_paths = []
            for row in rows:
                file_path = row["file_path"]
                local_path = _local_path_for_stat(file_path)
                if local_path is None:
                    continue
                try:
                    stat = local_path.stat()
                    stored_mtime = row["file_mtime"]
                    stored_size = row["file_size"]
                    if stored_mtime is not None and stored_size is not None:
                        if stat.st_mtime != stored_mtime or stat.st_size != stored_size:
                            stale_paths.append(file_path)
                except OSError:
                    # File doesn't exist
                    stale_paths.append(file_path)

            if stale_paths:
                conn.executemany(
                    "DELETE FROM parquet_meta WHERE file_path = ?",
                    [(p,) for p in stale_paths],
                )
                conn.commit()

            return len(stale_paths)
        finally:
            conn.close()


def serialize_arrow_schema(schema: pa.Schema) -> bytes:
    """Serialize Arrow schema to IPC format bytes."""
    sink = pa.BufferOutputStream()
    # Write empty batch to capture schema
    batch = pa.RecordBatch.from_pydict({name: [] for name in schema.names}, schema=schema)
    writer = pa.ipc.new_stream(sink, schema)
    writer.write_batch(batch)
    writer.close()
    return sink.getvalue().to_pybytes()


def deserialize_arrow_schema(schema_bytes: bytes) -> pa.Schema:
    """Deserialize Arrow schema from IPC format bytes."""
    reader = pa.ipc.open_stream(pa.BufferReader(schema_bytes))
    return reader.schema


def extract_parquet_meta(
    file_path: str, s3_filesystem: "pafs.S3FileSystem | None" = None
) -> PersistedParquetMeta:
    """Extract metadata from a Parquet file for persistence.

    Args:
        file_path: Path to the Parquet file (local or s3://)
        s3_filesystem: Optional S3 filesystem for reading S3 paths

    Returns:
        PersistedParquetMeta containing serializable metadata
    """
    # Handle S3 paths
    if file_path.startswith("s3://"):
        if s3_filesystem is None:
            import pyarrow.fs as pafs

            s3_filesystem = pafs.S3FileSystem()
        # Strip s3:// prefix for PyArrow filesystem
        s3_path = file_path[5:]
        pq_file = pq.ParquetFile(s3_path, filesystem=s3_filesystem)
    else:
        pq_file = pq.ParquetFile(file_path)
    metadata = pq_file.metadata

    # Serialize schema
    schema_bytes = serialize_arrow_schema(pq_file.schema_arrow)

    # Extract row group metadata
    row_groups = []
    for i in range(metadata.num_row_groups):
        rg = metadata.row_group(i)
        column_stats = {}

        for j in range(rg.num_columns):
            col = rg.column(j)
            # Key stats by the column's dotted PATH, not its leaf name: a
            # struct field ``user.id`` has leaf name ``id``, which collides
            # with a top-level ``id`` and silently overwrote its stats.
            col_name = metadata.schema.column(j).path

            if col.is_stats_set:
                stats = col.statistics
                stat_dict = {}
                if stats.has_min_max:
                    # Convert to Python types for JSON serialization
                    min_val = stats.min
                    max_val = stats.max
                    if hasattr(min_val, "as_py"):
                        min_val = min_val.as_py()
                    if hasattr(max_val, "as_py"):
                        max_val = max_val.as_py()

                    # Handle non-JSON-serializable types
                    try:
                        json.dumps(min_val)
                        stat_dict["min"] = min_val
                    except (TypeError, ValueError):
                        stat_dict["min"] = str(min_val)

                    try:
                        json.dumps(max_val)
                        stat_dict["max"] = max_val
                    except (TypeError, ValueError):
                        stat_dict["max"] = str(max_val)

                if stats.null_count is not None:
                    stat_dict["null_count"] = stats.null_count

                if stat_dict:
                    column_stats[col_name] = stat_dict

        row_groups.append(
            PersistedRowGroupMeta(
                num_rows=rg.num_rows,
                total_byte_size=rg.total_byte_size,
                column_stats=column_stats,
            )
        )

    return PersistedParquetMeta(
        arrow_schema_bytes=schema_bytes,
        num_row_groups=metadata.num_row_groups,
        row_groups=row_groups,
        # Dotted paths (unique per physical column), not leaf names — see
        # the column_stats comment above.
        column_names=[metadata.schema.column(i).path for i in range(len(metadata.schema))],
    )


# Global singleton store - moved to metadata_cache.py to avoid duplicate singletons
# Use get_metadata_store from metadata_cache.py instead


def reset_metadata_store() -> None:
    """Reset global metadata store (for testing).

    Note: This is a compatibility shim. Use reset_caches() from metadata_cache.py instead.
    """
    from strata.metadata_cache import reset_caches

    reset_caches()
