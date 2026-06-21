"""Read planner: builds ReadPlan from snapshot + filters + projection."""

import time
from pathlib import Path

import pyarrow.parquet as pq

from strata.config import StrataConfig
from strata.iceberg import PyIcebergCatalog
from strata.metadata_cache import (
    ManifestCache,
    ManifestEntry,
    ManifestResolution,
    ParquetMetadataCache,
    RowGroupMeta,
    get_manifest_cache,
    get_parquet_cache,
)
from strata.tenant import get_tenant_id
from strata.timing import elapsed_ms
from strata.tracing import trace_span
from strata.types import (
    CacheKey,
    Filter,
    ReadPlan,
    TableIdentity,
    Task,
    compute_filter_fingerprint,
    filters_to_iceberg_expression,
)

# Type alias for compiled filters: list of (parquet_column_index, filter)
CompiledFilters = list[tuple[int, Filter]]


def _build_column_index_map(schema) -> dict[str, int]:
    """Build a mapping from column name to Parquet leaf column index.

    Only includes flat (non-nested) columns that can be reliably used
    for row group pruning via statistics.

    Args:
        schema: Parquet schema from file metadata

    Returns:
        Dict mapping column name to its physical column index
    """
    col_map: dict[str, int] = {}
    for i in range(len(schema)):
        col = schema.column(i)
        # Skip nested/repeated fields - path contains '.' for nested
        if "." in col.path:
            continue
        col_map[col.name] = i
    return col_map


def _compile_filters(filters: list[Filter], col_index_map: dict[str, int]) -> CompiledFilters:
    """Compile filters into (column_index, filter) pairs for fast evaluation.

    Filters referencing columns not in the map (nested or missing) are dropped.

    Args:
        filters: List of Filter objects
        col_index_map: Mapping from column name to Parquet column index

    Returns:
        List of (column_index, filter) tuples for columns that exist
    """
    compiled: CompiledFilters = []
    for f in filters:
        col_idx = col_index_map.get(f.column)
        if col_idx is not None:
            compiled.append((col_idx, f))
    return compiled


def _normalize_s3_path(path: str) -> str:
    """Normalize an S3 path by removing redundant slashes and path components.

    Args:
        path: S3 URI (s3://bucket/path)

    Returns:
        Normalized S3 path with clean path components
    """
    if not path.startswith("s3://"):
        return path

    # Split into bucket and path
    without_prefix = path[5:]
    if "/" not in without_prefix:
        return path  # Just bucket name

    bucket_end = without_prefix.index("/")
    bucket = without_prefix[:bucket_end]
    key = without_prefix[bucket_end + 1 :]

    # Normalize the key path: split, filter empty/dot components, rejoin
    parts = key.split("/")
    normalized_parts = []
    for part in parts:
        if part == "" or part == ".":
            continue
        if part == ".." and normalized_parts:
            normalized_parts.pop()
        elif part != "..":
            normalized_parts.append(part)

    normalized_key = "/".join(normalized_parts)
    return f"s3://{bucket}/{normalized_key}" if normalized_key else f"s3://{bucket}"


def _join_s3_path(base: str, relative: str) -> str:
    """Join an S3 base path with a relative path.

    Args:
        base: S3 base URI (s3://bucket/path)
        relative: Relative path to append

    Returns:
        Joined and normalized S3 path
    """
    # Strip trailing slash from base
    base = base.rstrip("/")
    # Strip leading slash from relative
    relative = relative.lstrip("/")
    # Join and normalize
    return _normalize_s3_path(f"{base}/{relative}")


class ReadPlanner:
    """Plans reads from Iceberg tables with row-group pruning.

    Uses metadata caches to avoid redundant reads:
    - ParquetMetadataCache: Caches Parquet file metadata (schema, row groups, stats)
    - ManifestCache: Caches Iceberg manifest resolution per snapshot

    When cache_dir is provided, metadata is persisted to SQLite for fast
    planning after server restarts.
    """

    def __init__(
        self,
        config: StrataConfig,
        parquet_cache: ParquetMetadataCache | None = None,
        manifest_cache: ManifestCache | None = None,
    ) -> None:
        self.config = config
        self.catalog = PyIcebergCatalog(config)
        # Enable persistence by passing cache_dir
        cache_dir = config.cache_dir

        # Create S3 filesystem if any S3 config is provided
        s3_filesystem = None
        if (
            config.s3_region
            or config.s3_access_key
            or config.s3_anonymous
            or config.s3_endpoint_url
        ):
            s3_filesystem = config.get_s3_filesystem()

        self.parquet_cache = parquet_cache or get_parquet_cache(
            cache_dir=cache_dir, s3_filesystem=s3_filesystem
        )
        self.manifest_cache = manifest_cache or get_manifest_cache(cache_dir=cache_dir)

    def plan(
        self,
        table_uri: str,
        snapshot_id: int | None = None,
        columns: list[str] | None = None,
        filters: list[Filter] | None = None,
    ) -> ReadPlan:
        """Create a read plan for the given table and options.

        Args:
            table_uri: Table identifier (path#namespace.table or just namespace.table)
            snapshot_id: Specific snapshot to read (None for current)
            columns: Columns to project (None for all)
            filters: Filters for row-group pruning

        Returns:
            ReadPlan with tasks for each row group to read
        """
        start_time = time.perf_counter()
        filters = filters or []

        # Parse table URI and build canonical TableIdentity
        # table_uri is treated as input only; table_identity is the canonical ID
        warehouse_path, table_id = self.catalog.parse_table_uri(table_uri)
        identity_catalog_name = self.config.catalog_name if warehouse_path is None else "strata"
        manifest_catalog_name = (
            self.config.catalog_name if warehouse_path is None else warehouse_path
        )
        table_identity = TableIdentity.from_table_id(table_id, catalog=identity_catalog_name)

        # Load table and resolve snapshot
        table = self.catalog.load_table(table_uri)
        resolved_snapshot_id = self.catalog.get_snapshot_id(table, snapshot_id)

        # Get the snapshot's manifest
        snapshot = table.snapshot_by_id(resolved_snapshot_id)
        if snapshot is None:
            raise ValueError(f"Snapshot {resolved_snapshot_id} not found")

        # Compute projection fingerprint
        proj_fingerprint = CacheKey.compute_projection_fingerprint(columns)

        # Compute filter fingerprint for cache keying
        filter_fingerprint = compute_filter_fingerprint(filters)

        # Collect all data files from the snapshot
        plan = ReadPlan(
            table_uri=table_uri,
            table_identity=table_identity,
            snapshot_id=resolved_snapshot_id,
            columns=columns,
            filters=filters,
        )

        # Get data files from manifest cache or resolve fresh
        # Two-level lookup: try filtered cache first, then compute with Iceberg pruning
        table_identity_str = str(table_identity)
        manifest_resolution = self.manifest_cache.get(
            manifest_catalog_name,
            table_identity_str,
            resolved_snapshot_id,
            filter_fingerprint,
        )

        if manifest_resolution is None:
            # Cache miss: resolve manifests with Iceberg file-level pruning
            with trace_span(
                "resolve_manifests",
                table_id=table_identity_str,
                snapshot_id=resolved_snapshot_id,
            ) as span:
                iceberg_expr = filters_to_iceberg_expression(filters)

                try:
                    # Use Iceberg's file-level pruning if we have filters
                    if iceberg_expr is not None:
                        scan = table.scan(snapshot_id=resolved_snapshot_id, row_filter=iceberg_expr)
                    else:
                        scan = table.scan(snapshot_id=resolved_snapshot_id)
                    data_files = list(scan.plan_files())
                except Exception:
                    # If Iceberg expression fails (type mismatch, unsupported column, etc.),
                    # fall back to unfiltered scan - row-group pruning will still work
                    scan = table.scan(snapshot_id=resolved_snapshot_id)
                    data_files = list(scan.plan_files())

                # Build manifest entries with resolved paths
                entries = []
                for file_task in data_files:
                    file_path = file_task.file.file_path
                    actual_path = self._resolve_file_path(table_uri, file_path)
                    entries.append(ManifestEntry(file_path=file_path, actual_path=actual_path))

                manifest_resolution = ManifestResolution(data_files=entries)
                span.set_attribute("files_count", len(entries))

            self.manifest_cache.put(
                manifest_catalog_name,
                table_identity_str,
                resolved_snapshot_id,
                manifest_resolution,
                filter_fingerprint,
            )

        total_row_groups = 0
        pruned_row_groups = 0
        arrow_schema = None
        estimated_bytes = 0

        # Batch load Parquet metadata for all files
        actual_paths = [entry.actual_path for entry in manifest_resolution.data_files]
        try:
            pq_meta_batch = self.parquet_cache.get_or_load_many(actual_paths)
        except Exception as e:
            raise RuntimeError(f"Failed to read Parquet metadata: {e}") from e

        for entry in manifest_resolution.data_files:
            file_path = entry.file_path
            actual_path = entry.actual_path

            pq_meta = pq_meta_batch.get(actual_path)
            if pq_meta is None:
                raise RuntimeError(f"Failed to load Parquet metadata for {actual_path}")

            # Capture schema from first file (all files should have same schema)
            if arrow_schema is None:
                arrow_schema = pq_meta.arrow_schema

            # Build column index map once per file and compile filters
            # This avoids O(num_columns × num_filters × num_row_groups) scanning
            col_index_map = _build_column_index_map(pq_meta.parquet_schema)
            compiled_filters = _compile_filters(filters, col_index_map)

            for rg_idx in range(pq_meta.num_row_groups):
                total_row_groups += 1
                rg_meta = pq_meta.row_group_metadata[rg_idx]

                # Check if we can prune this row group using compiled filters
                if self._should_prune_row_group(rg_meta, compiled_filters):
                    pruned_row_groups += 1
                    continue

                cache_key = CacheKey(
                    tenant_id=get_tenant_id(),
                    table_identity=table_identity,
                    snapshot_id=resolved_snapshot_id,
                    file_path=file_path,
                    row_group_id=rg_idx,
                    projection_fingerprint=proj_fingerprint,
                )

                # Get estimated size from Parquet metadata
                # Works with both our RowGroupMeta and PyArrow's RowGroupMetaData
                rg_size = getattr(rg_meta, "total_byte_size", 0)

                task = Task(
                    file_path=actual_path,
                    row_group_id=rg_idx,
                    cache_key=cache_key,
                    num_rows=rg_meta.num_rows,
                    columns=columns,
                    estimated_bytes=rg_size,
                )
                plan.tasks.append(task)
                estimated_bytes += rg_size

        plan.total_row_groups = total_row_groups
        plan.pruned_row_groups = pruned_row_groups
        plan.estimated_bytes = estimated_bytes
        plan.planning_time_ms = elapsed_ms(start_time)

        # Set schema: prefer Parquet schema (may have column projection),
        # fall back to Iceberg table schema for empty tables/scans
        if arrow_schema is not None:
            plan.schema = arrow_schema
        else:
            # No data files - get schema from Iceberg table metadata
            # This ensures empty scans still have a valid schema
            plan.schema = table.schema().as_arrow()

        return plan

    def _resolve_file_path(self, table_uri: str, file_path: str) -> str:
        """Resolve a file path from the table metadata to an actual path.

        Args:
            table_uri: Table URI in format warehouse_path#namespace.table
            file_path: File path from Iceberg manifest (absolute or relative)

        Returns:
            Resolved absolute path (local or S3)
        """
        # Handle S3 paths - normalize and return
        if file_path.startswith("s3://"):
            return _normalize_s3_path(file_path)

        # Handle file:// prefix
        if file_path.startswith("file://"):
            return file_path[7:]

        # If it's already absolute, use it
        if file_path.startswith("/"):
            return file_path

        # Try to resolve relative to warehouse
        if "#" in table_uri:
            warehouse_path = table_uri.split("#")[0]
            # S3 relative paths
            if warehouse_path.startswith("s3://"):
                return _join_s3_path(warehouse_path, file_path)
            # Local filesystem relative paths
            warehouse_path = warehouse_path.replace("file://", "")
            candidate = Path(warehouse_path) / file_path
            if candidate.exists():
                return str(candidate)

        return file_path

    def _should_prune_row_group(
        self,
        rg_meta: RowGroupMeta | pq.RowGroupMetaData,
        compiled_filters: CompiledFilters,
    ) -> bool:
        """Check if a row group can be pruned based on compiled filters and stats.

        Uses pre-compiled filters with resolved column indices for efficient
        row group pruning. Column index mapping is done once per file, not per
        row group.

        Args:
            rg_meta: Row group metadata from Parquet file (PyArrow RowGroupMetaData)
            compiled_filters: List of (column_index, Filter) tuples pre-compiled
                by _compile_filters()

        Returns:
            True if the row group can be safely pruned (no matching rows),
            False if the row group should be read.

        Note:
            Limitations:
            - Only flat, primitive columns are supported for pruning
            - Complex Parquet timestamps may not convert correctly; use int64 epoch
            - Filters use AND semantics (all must match for row to be included)

            If pruning cannot be safely determined, we err on the side of NOT
            pruning (i.e., we read the row group rather than risk missing data).
        """
        if not compiled_filters:
            return False

        for col_idx, f in compiled_filters:
            try:
                col_meta = rg_meta.column(col_idx)
                if not col_meta.is_stats_set:
                    continue

                stats = col_meta.statistics
                if stats is None:
                    continue

                min_val = stats.min
                max_val = stats.max

                # Convert to comparable types if needed
                min_val, max_val = self._convert_stats(min_val, max_val)

                if not f.matches_stats(min_val, max_val):
                    return True

            except Exception:
                # If we can't get stats, don't prune (safe default)
                continue

        return False

    def _convert_stats(self, min_val, max_val):
        """Convert statistics to comparable types.

        Converts PyArrow scalar values to Python types for comparison with
        filter values during row group pruning.

        Args:
            min_val: Minimum value from Parquet column statistics (may be PyArrow scalar)
            max_val: Maximum value from Parquet column statistics (may be PyArrow scalar)

        Returns:
            Tuple of (min_val, max_val) converted to Python types

        Note:
            Limitations:
            - Only basic type conversions are supported
            - Numeric types (int, float) work directly with Parquet stats
            - String comparisons work if both filter and stats are strings
            - Timestamp pruning:
              * For int64 epoch micros (recommended): use int filter values
              * For datetime filters: stats must return datetime-compatible objects
              * Type mismatches will raise and be caught (no pruning, safe)
            - Decimals, bytes, and complex types may not compare correctly

            Future: use Iceberg schema for type-aware conversions.
        """
        # Convert PyArrow scalars to Python types if needed
        if hasattr(min_val, "as_py"):
            min_val = min_val.as_py()
        if hasattr(max_val, "as_py"):
            max_val = max_val.as_py()

        return min_val, max_val
