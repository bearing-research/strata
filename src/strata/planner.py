"""Read planner: builds ReadPlan from snapshot + filters + projection."""

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pyiceberg.conversions import from_bytes, to_bytes
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import IcebergType

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

# (parquet_column_index, filter)
CompiledFilters = list[tuple[int, Filter]]


def _build_column_index_map(schema) -> dict[str, int]:
    """Map column name to Parquet leaf column index, for flat columns only (prunable by stats)."""
    col_map: dict[str, int] = {}
    for i in range(len(schema)):
        col = schema.column(i)
        # Nested/repeated fields have a dotted path; skip them
        if "." in col.path:
            continue
        col_map[col.name] = i
    return col_map


def _compile_filters(filters: list[Filter], col_index_map: dict[str, int]) -> CompiledFilters:
    """Compile filters into ``(column_index, filter)`` pairs, dropping those on unmapped columns."""
    compiled: CompiledFilters = []
    for f in filters:
        col_idx = col_index_map.get(f.column)
        if col_idx is not None:
            compiled.append((col_idx, f))
    return compiled


def _normalize_s3_path(path: str) -> str:
    """Normalize an ``s3://`` URI by removing redundant slashes and path components."""
    if not path.startswith("s3://"):
        return path

    without_prefix = path[5:]
    if "/" not in without_prefix:
        return path  # bucket only

    bucket_end = without_prefix.index("/")
    bucket = without_prefix[:bucket_end]
    key = without_prefix[bucket_end + 1 :]

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
    """Join an S3 base URI with a relative path and normalize the result."""
    base = base.rstrip("/")
    relative = relative.lstrip("/")
    return _normalize_s3_path(f"{base}/{relative}")


def _delete_files(positional_deletes, table_identity: str) -> tuple[DeleteFileEntry, ...]:
    """Return a data file's positional deletes: Parquet delete files or Puffin deletion vectors.

    ORC and Avro delete files are refused, since skipping one would return the rows it deletes.
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
    """Return a data file's equality deletes, with what planning needs to prune and limit them.

    Keys match by field id against top-level columns; a key inside a struct, or an
    unreadable delete file, is refused.
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


def _partition_values(
    data_file, table, top_level: dict[int, IcebergType]
) -> tuple[tuple[int, str, str], ...]:
    """Return *data_file*'s identity-partition values, for a source column the file omits.

    Hive-layout files registered with add_files often omit the partition column;
    pyiceberg reads it as the partition value. Values keep their type in Iceberg's
    single-value encoding. *top_level* maps field id to type; nested source columns
    are not filled.
    """
    spec = table.metadata.specs()[data_file.spec_id or 0]
    values = []
    for position, partition_field in enumerate(spec.fields):
        value = data_file.partition[position]
        field_type = top_level.get(partition_field.source_id)
        if (
            isinstance(partition_field.transform, IdentityTransform)
            and value is not None
            and field_type is not None
        ):
            encoded = to_bytes(field_type, value).hex()
            values.append((partition_field.source_id, str(field_type), encoded))
    return tuple(values)


def _decode_partition_values(values: tuple[tuple[int, str, str], ...]) -> dict[int, Any]:
    """Return the values ``_partition_values`` kept, by source field id."""
    return {
        field_id: from_bytes(IcebergType.model_validate(type_name), bytes.fromhex(encoded))
        for field_id, type_name, encoded in values
    }


class ColumnNotFound(ValueError):
    """A read projects a column the table's schema does not have."""


def _assert_projection_exists(
    columns: list[str] | None,
    table_schema,
    table_identity: str,
) -> None:
    """Raise when a requested column is absent from the table schema.

    Without this, ``schema.get_field_index`` returns -1 for an unknown name and
    ``batch.column(-1)`` silently returns the last column under the requested name.
    """
    if not columns:
        return
    known = set(table_schema.names)
    missing = [c for c in columns if c not in known]
    if missing:
        raise ColumnNotFound(
            f"Table {table_identity} has no column(s) {missing}. "
            f"Available columns: {sorted(known)}."
        )


def _project_schema(schema, columns: list[str] | None):
    """Return ``schema`` narrowed to ``columns``, in the requested order."""
    if not columns or schema is None:
        return schema
    if list(schema.names) == columns:
        return schema
    # ``field(name)`` raises on an unknown name; ``get_field_index`` would return -1 and
    # silently pick the last field.
    return pa.schema([schema.field(name) for name in columns])


def _estimate_row_group_bytes(
    rg_meta, columns: list[str] | None, col_index_map: dict[str, int]
) -> int:
    """Estimate the bytes a row group contributes to the response, for the pre-flight 413.

    Sums only the projected columns' chunks, falling back to the whole row group
    when per-column sizes are unavailable (nested projection, older cache entry),
    since over-estimating is the safe direction. This is Parquet's encoded size,
    which can under-state Arrow's in-memory size for dictionary-encoded columns.
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

    Caches Parquet file metadata and manifest resolution per snapshot; with
    ``cache_dir`` they persist to SQLite across restarts.
    """

    def __init__(
        self,
        config: StrataConfig,
        parquet_cache: ParquetMetadataCache | None = None,
        manifest_cache: ManifestCache | None = None,
        catalog: CatalogProvider | None = None,
    ) -> None:
        self.config = config
        # Injectable for deployments with their own catalog access (and tests).
        self.catalog = catalog if catalog is not None else PyIcebergCatalog(config)
        lake_files.configure(config)
        cache_dir = config.cache_dir

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
        # Per (table, schema id, file): column layout (None when it needs none), columns by
        # field id, identity-partition values. Safe to cache: data files never change.
        self._file_columns: LRUCache[
            tuple[str, int, str],
            tuple[tuple[Column, ...] | None, dict[int, str], dict[int, Any]],
        ] = LRUCache(10_000)

    def plan(
        self,
        table_uri: str,
        snapshot_id: int | None = None,
        columns: list[str] | None = None,
        filters: list[Filter] | None = None,
    ) -> ReadPlan:
        """Create a read plan with one task per row group to read.

        ``table_uri`` is ``path#namespace.table`` or ``namespace.table``;
        ``snapshot_id`` None reads the current snapshot, ``columns`` None reads all.
        """
        start_time = time.perf_counter()
        filters = filters or []

        # table_uri is input only; table_identity is the canonical ID
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

        table = self.catalog.load_table(table_uri)
        if named is not None:
            # Vended credentials go to the table's FileIO; the fetcher needs them too.
            lake_files.register_vended_credentials(table.location(), dict(table.io.properties))
        resolved_snapshot_id = self.catalog.get_snapshot_id(table, snapshot_id)

        snapshot = table.snapshot_by_id(resolved_snapshot_id)
        if snapshot is None:
            raise ValueError(f"Snapshot {resolved_snapshot_id} not found")

        proj_fingerprint = CacheKey.compute_projection_fingerprint(columns)

        filter_fingerprint = compute_filter_fingerprint(filters)

        plan = ReadPlan(
            table_uri=table_uri,
            table_identity=table_identity,
            snapshot_id=resolved_snapshot_id,
            columns=columns,
            filters=filters,
        )

        table_identity_str = str(table_identity)
        manifest_resolution = self.manifest_cache.get(
            manifest_catalog_name,
            table_identity_str,
            resolved_snapshot_id,
            filter_fingerprint,
        )

        if manifest_resolution is None:
            with trace_span(
                "resolve_manifests",
                table_id=table_identity_str,
                snapshot_id=resolved_snapshot_id,
            ) as span:
                iceberg_expr = filters_to_iceberg_expression(filters)

                try:
                    # Strata plans files itself (iceberg_equality): pyiceberg refuses a
                    # table with equality deletes.
                    data_files = plan_files(table, resolved_snapshot_id, iceberg_expr)
                except Exception:
                    # Expression unsupported (type mismatch, unknown column): scan unfiltered;
                    # row-group pruning still applies.
                    data_files = plan_files(table, resolved_snapshot_id)

                entries = []
                top_level = {
                    field.field_id: field.field_type
                    for schema in table.metadata.schemas
                    for field in schema.fields
                }
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
                            partition_values=_partition_values(planned.data_file, table, top_level),
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

        # Read the current schema, or the named snapshot's (as pyiceberg does). A schema
        # change makes no snapshot, so the cache key and provenance carry the schema too.
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

            # Schema evolution rewrites no files, so an older file may name, hold or type a
            # column differently. Columns match by field id; ``layout`` says how to read
            # this file as the snapshot's schema (None when it already is).
            layout_key = (table_identity_str, snapshot_schema.schema_id, actual_path)
            cached_layout = self._file_columns.get(layout_key)
            if cached_layout is None:
                partition = _decode_partition_values(entry.partition_values)
                cached_layout = (
                    file_columns(
                        pq_meta.arrow_schema,
                        snapshot_schema,
                        name_mapping,
                        table_identity=table_identity_str,
                        file_path=file_path,
                        format_version=table.metadata.format_version,
                        partition_values=partition,
                    ),
                    stored_columns(
                        pq_meta.arrow_schema,
                        snapshot_schema,
                        name_mapping,
                        table.metadata.format_version,
                    ),
                    partition,
                )
                self._file_columns.put(layout_key, cached_layout)
            layout, stored, partition = cached_layout
            if arrow_schema is None:
                arrow_schema = (
                    pq_meta.arrow_schema
                    if layout is None
                    else pa.schema([column.field for column in layout])
                )

            # Once per file, not per row group and filter
            col_index_map = _build_column_index_map(pq_meta.parquet_schema)
            # Equality delete keys resolve by field id in the file: a key column may have
            # been renamed or dropped since.
            key_index = {field_id: col_index_map.get(name) for field_id, name in stored.items()}
            if layout is not None:
                # Stats by the snapshot's names. A column the file lacks has none, and a
                # dropped column's name must not lend it its own.
                col_index_map = {
                    column.name: col_index_map[column.source]
                    for column in layout
                    if column.source is not None and column.source in col_index_map
                }
            compiled_filters = _compile_filters(filters, col_index_map)

            # Merge-on-read: the fetcher drops these rows, so the row group is cached without
            # them under a key that names the snapshot (and so its deletes).
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
                        pruned_row_groups += 1
                        continue

                # Equality deletes whose key range can meet this row group's. Applying them
                # holds all their rows in memory, so a row group over the limit is refused.
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

                # Projection-aware; takes our RowGroupMeta or PyArrow's RowGroupMetaData.
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
                        (
                            field_id,
                            stored.get(field_id),
                            partition.get(field_id, initial_defaults.get(field_id)),
                        )
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

        # The first file's schema (as the snapshot reads it), else the snapshot's for empty or
        # fully pruned scans. Always project: ``plan.schema`` is the response schema when
        # there are no tasks, and the same query must have the same shape with or without rows.
        base_schema = arrow_schema if arrow_schema is not None else table_arrow_schema
        plan.schema = _project_schema(base_schema, columns)

        return plan

    def _resolve_file_path(self, table_uri: str, file_path: str) -> str:
        """Resolve a manifest file path (absolute or relative) to a local or S3 path."""
        if file_path.startswith("s3://"):
            return _normalize_s3_path(file_path)

        if file_path.startswith("file://"):
            return file_path[7:]

        if file_path.startswith("/"):
            return file_path

        # Relative to the warehouse
        if "#" in table_uri:
            warehouse_path = table_uri.split("#")[0]
            if warehouse_path.startswith("s3://"):
                return _join_s3_path(warehouse_path, file_path)
            warehouse_path = warehouse_path.replace("file://", "")
            candidate = Path(warehouse_path) / file_path
            if candidate.exists():
                return str(candidate)

        return file_path

    def _key_stats(
        self, rg_meta: RowGroupMeta | pq.RowGroupMetaData, column_index: int | None
    ) -> tuple[object, object, int | None] | None:
        """Return a key column's ``(min, max, null_count)`` in a row group, when recorded."""
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
        """Return True only if stats prove the row group has no matching rows.

        Filters are ANDed and only flat primitive columns prune. When safety cannot be
        determined the row group is read, never pruned.
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

                min_val, max_val = self._convert_stats(min_val, max_val)

                if not f.matches_stats(min_val, max_val):
                    return True

            except Exception:
                # No stats: don't prune
                continue

        return False

    def _convert_stats(self, min_val, max_val):
        """Convert Parquet min/max statistics (possibly PyArrow scalars) to Python values.

        Numbers and strings compare directly; for timestamps prefer int64 epoch
        micros with int filters. Mismatched types raise and the caller does not prune.
        Decimals, bytes and complex types may not compare correctly.
        """
        if hasattr(min_val, "as_py"):
            min_val = min_val.as_py()
        if hasattr(max_val, "as_py"):
            max_val = max_val.as_py()

        return min_val, max_val
