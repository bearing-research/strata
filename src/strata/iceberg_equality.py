"""Iceberg equality deletes: rows deleted by value, not by position.

A streaming writer (Flink upserts, CDC sinks) deletes a row by its key rather
than by where it sits: an equality delete file holds key values, and every
older row with those values is deleted, in whichever data file it is.

pyiceberg (0.12) refuses to plan a scan once a manifest lists one, so
``plan_files`` plans it here. The manifest entries still come from pyiceberg's
``ManifestGroupPlanner`` (partition and metrics pruning included), positional
deletes still go through pyiceberg's ``DeleteFileIndex``, and equality deletes
go through ``EqualityDeleteIndex``.

Which equality deletes apply to a data file (Iceberg spec, "Scan Planning"):
those with a larger data sequence number, in the data file's partition (spec
id and value), or in an unpartitioned spec, where they apply to every data
file. A row is deleted when its values for the delete's equality field ids all
equal a delete row's, with null equal to null. Field ids, not names: a column
dropped from the schema since still matches, and a column the data file
predates reads as nulls.
"""

import threading
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as _pc
from pyiceberg.expressions import AlwaysTrue, BooleanExpression
from pyiceberg.manifest import DataFile, DataFileContent, ManifestEntry
from pyiceberg.partitioning import PartitionSpec
from pyiceberg.table import ManifestGroupPlanner, Table
from pyiceberg.table.delete_file_index import DeleteFileIndex
from pyiceberg.table.metadata import INITIAL_SEQUENCE_NUMBER
from pyiceberg.typedef import Record

from strata import lake_files
from strata.metadata_cache import EqualityDeleteEntry

# pyarrow.compute registers its kernels at import time, so ty does not know
# members like ``is_in``/``or_``. Cast through Any.
pc = cast(Any, _pc)

_FIELD_ID = b"PARQUET:field_id"


@dataclass
class PlannedFile:
    """A data file to scan, with the delete files that apply to it."""

    data_file: DataFile
    positional_deletes: set[DataFile]
    equality_deletes: list[DataFile]


class EqualityDeleteIndex:
    """Equality delete files by partition, answering which apply to a data file."""

    def __init__(self, specs: dict[int, PartitionSpec]) -> None:
        self._specs = specs
        self._global: list[tuple[int, DataFile]] = []
        self._by_partition: dict[tuple[int, Record], list[tuple[int, DataFile]]] = {}

    def add_delete_file(self, manifest_entry: ManifestEntry) -> None:
        delete_file = manifest_entry.data_file
        sequence_number = manifest_entry.sequence_number or INITIAL_SEQUENCE_NUMBER
        spec_id = delete_file.spec_id or 0
        if self._specs[spec_id].is_unpartitioned():
            self._global.append((sequence_number, delete_file))
        else:
            key = (spec_id, delete_file.partition)
            self._by_partition.setdefault(key, []).append((sequence_number, delete_file))

    def for_data_file(self, sequence_number: int, data_file: DataFile) -> list[DataFile]:
        """The equality deletes that apply to *data_file*, added at *sequence_number*.

        Strictly later ones only: an upsert writes its delete and its new row
        in one snapshot, and the delete must not remove that row.
        """
        key = (data_file.spec_id or 0, data_file.partition)
        return [
            delete_file
            for delete_sequence, delete_file in self._global + self._by_partition.get(key, [])
            if delete_sequence > sequence_number
        ]


def plan_files(
    table: Table, snapshot_id: int, row_filter: BooleanExpression | None = None
) -> list[PlannedFile]:
    """The data files of *snapshot_id* matching *row_filter*, with their deletes."""
    snapshot = table.snapshot_by_id(snapshot_id)
    if snapshot is None:
        raise ValueError(f"Snapshot {snapshot_id} not found")
    planner = ManifestGroupPlanner(
        table_metadata=table.metadata, io=table.io, row_filter=row_filter or AlwaysTrue()
    )
    data_entries: list[ManifestEntry] = []
    positional = DeleteFileIndex()
    equality = EqualityDeleteIndex(table.metadata.specs())
    for entries in planner.plan_manifest_entries(snapshot.manifests(table.io)):
        for entry in entries:
            content = entry.data_file.content
            if content == DataFileContent.DATA:
                data_entries.append(entry)
            elif content == DataFileContent.POSITION_DELETES:
                positional.add_delete_file(entry, partition_key=entry.data_file.partition)
            else:
                equality.add_delete_file(entry)
    planned = []
    for entry in data_entries:
        sequence_number = entry.sequence_number or INITIAL_SEQUENCE_NUMBER
        data_file = entry.data_file
        planned.append(
            PlannedFile(
                data_file=data_file,
                positional_deletes=positional.for_data_file(
                    sequence_number, data_file, partition_key=data_file.partition
                ),
                equality_deletes=equality.for_data_file(sequence_number, data_file),
            )
        )
    return planned


def encode_bound(field_type: Any, raw: bytes | None) -> str | None:
    """A manifest bound kept for pruning: ``int:<hex>`` or ``str:<hex>``, else None.

    Iceberg stores a bound in its single-value encoding. Only int and long
    (little-endian) and string (UTF-8) bounds are kept; they cover the usual
    key types and compare directly with Parquet's row-group statistics.
    """
    from pyiceberg.types import IntegerType, LongType, StringType

    if raw is None:
        return None
    if isinstance(field_type, IntegerType | LongType):
        return f"int:{raw.hex()}"
    if isinstance(field_type, StringType):
        return f"str:{raw.hex()}"
    return None


def _decode_bound(raw: str | None) -> int | str | None:
    """A bound kept by ``encode_bound``, as the int or str it encodes."""
    if raw is None:
        return None
    kind, _, value = raw.partition(":")
    data = bytes.fromhex(value)
    if kind == "int":
        return int.from_bytes(data, "little", signed=True)
    if kind == "str":
        return data.decode("utf-8")
    return None


def may_apply(
    delete: EqualityDeleteEntry,
    row_group_stats: Callable[[int], tuple[object, object, int | None] | None],
) -> bool:
    """False when *delete* provably deletes nothing in a row group.

    *row_group_stats(field_id)* gives the row group's ``(min, max, null_count)``
    for that key, or None when unknown. A delete with no null keys whose range
    of values lies wholly outside the row group's range matches none of its
    rows. Anything unknown or of an undecoded type keeps the delete.
    """
    for field_id, lower_raw, upper_raw, null_count in delete.bounds:
        if null_count is None or null_count > 0:
            continue
        lower, upper = _decode_bound(lower_raw), _decode_bound(upper_raw)
        stats = row_group_stats(field_id)
        if lower is None or upper is None or stats is None:
            continue
        low, high, _ = stats
        if type(low) is not type(lower) or type(high) is not type(upper):
            continue
        if cast(Any, upper) < low or cast(Any, lower) > high:
            return False
    return True


class EqualityDeleteSets:
    """Each equality delete file's key values, read once (delete files never change).

    Holds at most *max_rows* delete rows, least recently used out first: the
    bound planning already puts on the deletes one row group may need.
    """

    def __init__(self, s3_filesystem: Any = None, max_rows: int = 10_000_000) -> None:
        self._s3_filesystem = s3_filesystem
        self._max_rows = max_rows
        self._files: OrderedDict[str, pa.Table] = OrderedDict()
        self._rows = 0
        self._lock = threading.Lock()

    def keys(self, delete: EqualityDeleteEntry) -> pa.Table:
        """*delete*'s rows, one column per equality field id, named by that id."""
        with self._lock:
            cached = self._files.get(delete.actual_path)
            if cached is not None:
                self._files.move_to_end(delete.actual_path)
                return cached
        # Read outside the lock: the fetch pool shares this cache.
        keys = self._read(delete)
        with self._lock:
            if delete.actual_path not in self._files:
                self._files[delete.actual_path] = keys
                self._rows += keys.num_rows
                while self._rows > self._max_rows and len(self._files) > 1:
                    _, evicted = self._files.popitem(last=False)
                    self._rows -= evicted.num_rows
        return keys

    def _read(self, delete: EqualityDeleteEntry) -> pa.Table:
        parquet = lake_files.open_parquet(delete.actual_path, self._s3_filesystem)
        by_id = {
            int(field.metadata[_FIELD_ID]): field.name
            for field in parquet.schema_arrow
            if field.metadata and _FIELD_ID in field.metadata
        }
        missing = [field_id for field_id in delete.equality_ids if field_id not in by_id]
        if missing:
            raise ValueError(
                f"Equality delete file {delete.file_path} has no column for field id(s) {missing}"
            )
        read = parquet.read(columns=[by_id[field_id] for field_id in delete.equality_ids])
        return pa.table(
            {str(field_id): read.column(i) for i, field_id in enumerate(delete.equality_ids)}
        )


def _placeholder(data_type: pa.DataType) -> pa.Scalar:
    """Any valid value of *data_type*, to stand in for a null in a join key."""
    if pa.types.is_string(data_type) or pa.types.is_large_string(data_type):
        return pa.scalar("", data_type)
    if pa.types.is_binary(data_type) or pa.types.is_large_binary(data_type):
        return pa.scalar(b"", data_type)
    if pa.types.is_fixed_size_binary(data_type):
        return pa.scalar(b"\0" * data_type.byte_width, data_type)
    if pa.types.is_boolean(data_type):
        return pa.scalar(False)
    return pa.array([0]).cast(data_type)[0]


def _null_safe(keys: pa.Table) -> pa.Table:
    """*keys* with each column split in two, so an Arrow join treats null as equal to null.

    A hash join never matches null to null; Iceberg equality does. Each key
    becomes its value with nulls filled plus an is-null flag, and two rows match
    on those exactly when they match under Iceberg's rule.
    """
    columns = {}
    for name in keys.column_names:
        column = keys.column(name)
        columns[f"{name}.null"] = column.is_null()
        columns[name] = column.fill_null(_placeholder(column.type)) if column.null_count else column
    return pa.table(columns)


def _matching_rows(data_keys: pa.Table, delete_keys: pa.Table) -> pa.Array:
    """For each row of *data_keys*, whether some row of *delete_keys* equals it."""
    rows = pa.array(range(data_keys.num_rows), pa.int64())
    left = _null_safe(data_keys).append_column("row", rows)
    right = _null_safe(delete_keys)
    hits = left.join(right, keys=right.column_names, join_type="left semi").column("row")
    return pc.is_in(rows, value_set=hits)


def deleted_mask(
    table: pa.Table,
    key_columns: dict[int, str | None],
    deletes: Iterable[EqualityDeleteEntry],
    keys: Callable[[EqualityDeleteEntry], pa.Table],
) -> pa.Array | None:
    """For each row of *table*, whether an equality delete in *deletes* removes it.

    *key_columns* maps each equality field id to *table*'s column holding it,
    or None when the data file predates the column (its values are then null).
    Deletes are grouped by their equality ids; a row goes if any group matches.
    """
    groups: dict[tuple[int, ...], list[EqualityDeleteEntry]] = defaultdict(list)
    for delete in deletes:
        groups[delete.equality_ids].append(delete)
    deleted = None
    for field_ids, group in groups.items():
        names = [str(field_id) for field_id in field_ids]
        delete_keys = pa.concat_tables(
            [keys(delete).select(names) for delete in group], promote_options="permissive"
        )
        data_keys = pa.table(
            {
                name: table.column(column)
                if (column := key_columns.get(field_id)) is not None
                else pa.nulls(table.num_rows, delete_keys.schema.field(name).type)
                for field_id, name in zip(field_ids, names, strict=True)
            }
        )
        # One type per key on both sides (a key promoted since, say int to long).
        common = pa.unify_schemas(
            [data_keys.schema, delete_keys.schema], promote_options="permissive"
        )
        hit = _matching_rows(data_keys.cast(common), delete_keys.cast(common))
        deleted = hit if deleted is None else pc.or_(deleted, hit)
    return deleted
