"""Read planner: builds ReadPlan from snapshot + filters + projection."""

import time
from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from strata import lake_files
from strata.config import StrataConfig
from strata.iceberg import (
    CatalogProvider,
    PyIcebergCatalog,
    named_catalog,
    table_identity_for,
)
from strata.iceberg_deletes import READABLE_FORMATS, DeletedRows, in_row_group
from strata.iceberg_equality import encode_bound, may_apply, plan_files
from strata.iceberg_schema import (
    Column,
    UnsupportedTableFormatError,
    file_columns,
    snapshot_arrow_schema,
    stored_columns,
)
from strata.metadata_cache import (
    DeleteFileEntry,
    EqualityDeleteEntry,
    LRUCache,
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


def _delete_files(positional_deletes, table_identity: str) -> tuple[DeleteFileEntry, ...]:
    """A data file's positional deletes: Parquet delete files or Puffin deletion vectors.

    ORC and Avro delete files are valid Iceberg but not readable here, and
    skipping one would return the rows it deletes.
    """
    entries = []
    for delete_file in positional_deletes:
        file_format = delete_file.file_format.value
        if file_format not in READABLE_FORMATS:
            raise UnsupportedTableFormatError(
                f"Table {table_identity} has a {file_format} delete file "
                f"({delete_file.file_path}). Strata applies Parquet positional "
                "deletes and Puffin deletion vectors only. Compact the table "
                "(rewrite_data_files), or scan a snapshot taken before the deletes."
            )
        entries.append(DeleteFileEntry(file_path=delete_file.file_path, file_format=file_format))
    return tuple(sorted(entries, key=lambda entry: entry.file_path))


def _equality_deletes(
    equality_deletes, table, table_identity: str, resolve: Callable[[str], str]
) -> tuple[EqualityDeleteEntry, ...]:
    """A data file's equality deletes, with what planning needs to prune and limit them.

    Keys are matched by field id against the data file's top-level columns; a
    key inside a struct is refused, as is a delete file Strata cannot read.
    """
    top_level = {
        field.field_id: field.field_type
        for schema in table.metadata.schemas
        for field in schema.fields
    }
    entries = []
    for delete_file in equality_deletes:
        file_format = delete_file.file_format.value
        if file_format != "PARQUET":
            raise UnsupportedTableFormatError(
                f"Table {table_identity} has a {file_format} equality delete file "
                f"({delete_file.file_path}). Strata reads Parquet equality deletes only. "
                "Compact the table (rewrite_data_files), or scan a snapshot taken "
                "before the deletes."
            )
        equality_ids = tuple(delete_file.equality_ids or ())
        nested = [field_id for field_id in equality_ids if field_id not in top_level]
        if not equality_ids or nested:
            raise UnsupportedTableFormatError(
                f"Table {table_identity} has an equality delete file "
                f"({delete_file.file_path}) keyed on field id(s) {nested or 'none'} "
                "that are not top-level columns. Strata matches top-level keys only. "
                "Compact the table (rewrite_data_files)."
            )
        lower = delete_file.lower_bounds or {}
        upper = delete_file.upper_bounds or {}
        nulls = delete_file.null_value_counts or {}
        entries.append(
            EqualityDeleteEntry(
                file_path=delete_file.file_path,
                actual_path=resolve(delete_file.file_path),
                equality_ids=equality_ids,
                record_count=delete_file.record_count,
                bounds=tuple(
                    (
                        field_id,
                        encode_bound(top_level[field_id], lower.get(field_id)),
                        encode_bound(top_level[field_id], upper.get(field_id)),
                        nulls.get(field_id),
                    )
                    for field_id in equality_ids
                ),
            )
        )
    return tuple(sorted(entries, key=lambda entry: entry.file_path))


def _assert_projection_exists(
    columns: list[str] | None,
    table_schema,
    table_identity: str,
) -> None:
    """Raise when a requested column is absent from the table schema.

    Nothing checked this once: a single-file table validated nothing at all.
    The projection was then applied by ``CachedFetcher._project_batch``
    via ``schema.get_field_index(name)``, which returns ``-1`` for a name it
    does not know, and ``batch.column(-1)`` is the LAST column. So scanning a
    column that does not exist returned the last column's data under the
    requested name, silently and with no error: exactly the "quietly wrong
    rows" outcome this codebase refuses everywhere else.
    """
    if not columns:
        return
    known = set(table_schema.names)
    missing = [c for c in columns if c not in known]
    if missing:
        raise ValueError(
            f"Table {table_identity} has no column(s) {missing}. "
            f"Available columns: {sorted(known)}."
        )


def _project_schema(schema, columns: list[str] | None):
    """Return ``schema`` narrowed to ``columns``, in the requested order."""
    if not columns or schema is None:
        return schema
    if list(schema.names) == columns:
        return schema
    # ``field(name)`` raises for a name it does not know; ``get_field_index``
    # would return -1 and silently pick the last field instead.
    return pa.schema([schema.field(name) for name in columns])


def _estimate_row_group_bytes(
    rg_meta, columns: list[str] | None, col_index_map: dict[str, int]
) -> int:
    """Estimate the bytes a row group contributes to the response.

    This feeds the pre-flight 413, so it has to describe the response the
    limit governs. It used to be ``total_byte_size``, the size of the WHOLE
    row group, which ignores the projection entirely: scanning two columns of
    a forty-column table was estimated as if all forty were read, and a
    perfectly reasonable projected scan was rejected as oversized.

    Summing only the projected columns' chunks fixes that exactly. It falls
    back to the whole-row-group size whenever the per-column sizes are not
    available -- a nested projection (absent from ``col_index_map``), or a
    metadata cache entry written before the sizes were recorded -- because
    over-estimating is the safe direction for a guard.

    Note this is Parquet's uncompressed-but-ENCODED size, which still
    under-states Arrow's in-memory size for dictionary-encoded columns.
    Narrowing that gap is a separate question about how strict the limit
    should be, not something to settle inside an estimator.
    """
    total = getattr(rg_meta, "total_byte_size", 0)
    if not columns:
        return total

    projected = 0
    for name in columns:
        idx = col_index_map.get(name)
        if idx is None:
            return total
        size = getattr(rg_meta.column(idx), "total_uncompressed_size", 0)
        if not size:
            return total
        projected += size
    return projected


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
        catalog: CatalogProvider | None = None,
    ) -> None:
        self.config = config
        # Injectable: a deployment with its own catalog access (or a test)
        # supplies the provider; the default reads Strata's catalog settings.
        self.catalog = catalog if catalog is not None else PyIcebergCatalog(config)
        lake_files.configure(config)
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
        self._deleted_rows = DeletedRows()
        # Each data file's column layout (None when it needs none) and its
        # columns by field id, per (table, schema): files never change, and a
        # schema change is a new schema id.
        self._file_columns: LRUCache[
            tuple[str, int, str], tuple[tuple[Column, ...] | None, dict[int, str]]
        ] = LRUCache(10_000)

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
        named, named_table_id = named_catalog(table_uri, self.config)
        if named is not None:
            table_id = named_table_id
            manifest_catalog_name = named
        else:
            warehouse_path, table_id = PyIcebergCatalog.parse_table_uri(table_uri)
            manifest_catalog_name = (
                self.config.catalog_name if warehouse_path is None else warehouse_path
            )
        table_identity = table_identity_for(table_uri, self.config)

        # Load table and resolve snapshot
        table = self.catalog.load_table(table_uri)
        if named is not None:
            # A catalog that vends credentials hands them to the table's FileIO;
            # the fetcher reads the data files, so it needs them too.
            lake_files.register_vended_credentials(table.location(), dict(table.io.properties))
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
                    # Iceberg's file-level pruning when there are filters.
                    # Strata plans the files itself (iceberg_equality), since
                    # pyiceberg refuses a table with equality deletes.
                    data_files = plan_files(table, resolved_snapshot_id, iceberg_expr)
                except Exception:
                    # If Iceberg expression fails (type mismatch, unsupported column, etc.),
                    # fall back to unfiltered scan - row-group pruning will still work
                    data_files = plan_files(table, resolved_snapshot_id)

                # Build manifest entries with resolved paths
                entries = []
                for planned in data_files:
                    file_path = planned.data_file.file_path
                    actual_path = self._resolve_file_path(table_uri, file_path)
                    entries.append(
                        ManifestEntry(
                            file_path=file_path,
                            actual_path=actual_path,
                            delete_files=_delete_files(
                                planned.positional_deletes, table_identity_str
                            ),
                            equality_deletes=_equality_deletes(
                                planned.equality_deletes,
                                table,
                                table_identity_str,
                                lambda path: self._resolve_file_path(table_uri, path),
                            ),
                        )
                    )

                manifest_resolution = ManifestResolution(data_files=entries)
                span.set_attribute("files_count", len(entries))

            self.manifest_cache.put(
                manifest_catalog_name,
                table_identity_str,
                resolved_snapshot_id,
                manifest_resolution,
                filter_fingerprint,
            )

        # The schema to read as. A scan of the current table reads the current
        # schema; one that names a snapshot reads that snapshot's, as pyiceberg
        # does. A schema change makes no snapshot, so the cache key and the
        # scan's provenance carry the schema as well as the snapshot.
        if snapshot_id is None:
            snapshot_schema = table.schema()
        else:
            snapshot_schema = table.scan(snapshot_id=resolved_snapshot_id).projection()
        plan.schema_id = snapshot_schema.schema_id
        name_mapping = table.name_mapping()
        initial_defaults = {
            field.field_id: field.initial_default for field in snapshot_schema.fields
        }
        table_arrow_schema = snapshot_arrow_schema(snapshot_schema)
        _assert_projection_exists(columns, table_arrow_schema, table_identity_str)

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

            # Iceberg schema evolution rewrites no data files, so an older
            # file may name, hold or type a column differently from the
            # snapshot. Columns are matched by field id; ``layout`` says how to
            # read this file as the snapshot's schema, None when it already is.
            layout_key = (table_identity_str, snapshot_schema.schema_id, actual_path)
            cached_layout = self._file_columns.get(layout_key)
            if cached_layout is None:
                cached_layout = (
                    file_columns(
                        pq_meta.arrow_schema,
                        snapshot_schema,
                        name_mapping,
                        table_identity=table_identity_str,
                        file_path=file_path,
                        format_version=table.metadata.format_version,
                    ),
                    stored_columns(
                        pq_meta.arrow_schema,
                        snapshot_schema,
                        name_mapping,
                        table.metadata.format_version,
                    ),
                )
                self._file_columns.put(layout_key, cached_layout)
            layout, stored = cached_layout
            if arrow_schema is None:
                arrow_schema = (
                    pq_meta.arrow_schema
                    if layout is None
                    else pa.schema([column.field for column in layout])
                )

            # Build column index map once per file and compile filters
            # This avoids O(num_columns × num_filters × num_row_groups) scanning
            col_index_map = _build_column_index_map(pq_meta.parquet_schema)
            # Equality delete keys are looked up by field id in the file itself:
            # a key column may have been renamed or dropped since.
            key_index = {field_id: col_index_map.get(name) for field_id, name in stored.items()}
            if layout is not None:
                # Statistics by the snapshot's names. A column the file lacks
                # has none, and a dropped column's name must not lend it its own.
                col_index_map = {
                    column.name: col_index_map[column.source]
                    for column in layout
                    if column.source is not None and column.source in col_index_map
                }
            compiled_filters = _compile_filters(filters, col_index_map)

            # Merge-on-read: rows this snapshot deleted from the file. The
            # fetcher drops them, so the row group is cached without them,
            # under a key that names the snapshot and so its deletes.
            deleted = (
                self._deleted_rows.for_data_file(table.io, file_path, entry.delete_files)
                if entry.delete_files
                else None
            )
            row_group_start = 0

            for rg_idx in range(pq_meta.num_row_groups):
                total_row_groups += 1
                rg_meta = pq_meta.row_group_metadata[rg_idx]
                start = row_group_start
                row_group_start += rg_meta.num_rows

                # Check if we can prune this row group using compiled filters
                if self._should_prune_row_group(rg_meta, compiled_filters):
                    pruned_row_groups += 1
                    continue

                num_rows = rg_meta.num_rows
                deleted_rows = (
                    in_row_group(deleted, start, num_rows) if deleted is not None else None
                )
                if deleted_rows is not None:
                    num_rows -= len(deleted_rows)
                    if num_rows == 0:
                        # Every row deleted: nothing to read.
                        pruned_row_groups += 1
                        continue

                # Equality deletes whose key range can meet this row group's.
                # Applying them needs every one of their rows in memory, so a
                # row group over the limit is refused rather than read.
                equality = tuple(
                    delete
                    for delete in entry.equality_deletes
                    if may_apply(
                        delete,
                        lambda field_id: self._key_stats(rg_meta, key_index.get(field_id)),
                    )
                )
                pending = sum(delete.record_count for delete in equality)
                if pending > self.config.max_equality_delete_rows:
                    raise UnsupportedTableFormatError(
                        f"Table {table_identity_str} has {pending:,} pending equality "
                        f"deletes for a row group of {file_path}, above the limit of "
                        f"{self.config.max_equality_delete_rows:,} "
                        "(max_equality_delete_rows). Compact the table "
                        "(rewrite_data_files), or raise the limit."
                    )
                key_ids = sorted(
                    {field_id for delete in equality for field_id in delete.equality_ids}
                )

                cache_key = CacheKey(
                    tenant_id=get_tenant_id(),
                    table_identity=table_identity,
                    snapshot_id=resolved_snapshot_id,
                    file_path=file_path,
                    row_group_id=rg_idx,
                    projection_fingerprint=proj_fingerprint,
                    schema_id=plan.schema_id,
                )

                # Estimated size, projection-aware. Works with both our
                # RowGroupMeta and PyArrow's RowGroupMetaData.
                rg_size = _estimate_row_group_bytes(rg_meta, columns, col_index_map)

                task = Task(
                    file_path=actual_path,
                    row_group_id=rg_idx,
                    cache_key=cache_key,
                    num_rows=num_rows,
                    columns=columns,
                    estimated_bytes=rg_size,
                    deleted_rows=deleted_rows,
                    file_columns=layout,
                    equality_deletes=equality,
                    equality_columns=tuple(
                        (field_id, stored.get(field_id), initial_defaults.get(field_id))
                        for field_id in key_ids
                    ),
                    downcast_ns=table.metadata.format_version <= 2,
                )
                plan.tasks.append(task)
                estimated_bytes += rg_size

        plan.total_row_groups = total_row_groups
        plan.pruned_row_groups = pruned_row_groups
        plan.estimated_bytes = estimated_bytes
        plan.planning_time_ms = elapsed_ms(start_time)

        # Set schema: the first file's, as the snapshot reads it, when there
        # was a file to read; the snapshot's schema for empty tables /
        # fully-pruned scans.
        #
        # Then apply the projection. Neither source is projected on its own,
        # and ``plan.schema`` IS the response schema when there are no tasks:
        # a scan for ``columns=["id"]`` that matched rows streamed one column,
        # while the same scan matching none streamed every column. Same query,
        # different shape depending on the data — which breaks anything that
        # concatenates partitioned scans or asserts on the schema.
        base_schema = arrow_schema if arrow_schema is not None else table_arrow_schema
        plan.schema = _project_schema(base_schema, columns)

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

    def _key_stats(
        self, rg_meta: RowGroupMeta | pq.RowGroupMetaData, column_index: int | None
    ) -> tuple[object, object, int | None] | None:
        """A key column's ``(min, max, null_count)`` in a row group, when recorded."""
        if column_index is None:
            return None
        column = rg_meta.column(column_index)
        if not column.is_stats_set or column.statistics is None:
            return None
        low, high = self._convert_stats(column.statistics.min, column.statistics.max)
        return low, high, getattr(column.statistics, "null_count", None)

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
