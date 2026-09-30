"""Iceberg schema evolution: reading an older data file as the snapshot's schema.

Iceberg names a column by field id, not by name, and changing a table's schema
rewrites no data files. So an older file may lack a column the table added
since (it reads as nulls), hold a column the table renamed under its old name,
hold a dropped column whose name a new column reuses (that is not the new
column, which also reads as nulls), or store a narrower type than the one the
table promoted it to (int to long, float to double, a wider decimal). The same
holds for the fields inside a struct, list or map column. Iceberg's Parquet
files record the field id of every column and nested field, which is what
matches a file's columns to the snapshot's; a file without them is matched
through the table's name mapping, or by name when it has none.
"""

from typing import Any, NamedTuple, cast

import pyarrow as pa
import pyarrow.compute as _pc
from pyiceberg.exceptions import ResolveError
from pyiceberg.io.pyarrow import pyarrow_to_schema, schema_to_pyarrow
from pyiceberg.schema import Schema, promote
from pyiceberg.table.name_mapping import NameMapping, create_mapping_from_schema
from pyiceberg.types import IcebergType, ListType, MapType, NestedField, PrimitiveType, StructType

# pyarrow.compute registers its kernels at import time, so ty does not know
# members like ``subtract``. Cast through Any.
pc = cast(Any, _pc)


class UnsupportedTableFormatError(RuntimeError):
    """The table uses an Iceberg feature Strata cannot read correctly."""


class Column(NamedTuple):
    """One column of the snapshot's schema, as a data file holds it."""

    name: str  # in the snapshot's schema
    source: str | None  # the file's column holding it; None when the file predates it
    field: pa.Field  # what the scan returns
    # (file type, table type) of a struct, list or map column whose nested
    # fields changed since the file was written; None otherwise.
    reshape: tuple[IcebergType, IcebergType] | None = None
    # What a column the file predates reads as: its Iceberg v3 initial-default,
    # or None for nulls.
    default: Any = None


def _small(data_type: pa.DataType) -> pa.DataType:
    """*data_type* with Arrow's 32-bit offset types, which is what Parquet reads give."""
    if pa.types.is_large_string(data_type):
        return pa.string()
    if pa.types.is_large_binary(data_type):
        return pa.binary()
    if pa.types.is_large_list(data_type) or pa.types.is_list(data_type):
        return pa.list_(data_type.value_field.with_type(_small(data_type.value_type)))
    if pa.types.is_map(data_type):
        return pa.map_(
            data_type.key_field.with_type(_small(data_type.key_type)),
            data_type.item_field.with_type(_small(data_type.item_type)),
        )
    if pa.types.is_struct(data_type):
        return pa.struct([field.with_type(_small(field.type)) for field in data_type])
    return data_type


def _children(iceberg_type: IcebergType) -> list[NestedField]:
    if isinstance(iceberg_type, StructType):
        return list(iceberg_type.fields)
    if isinstance(iceberg_type, ListType):
        return [iceberg_type.element_field]
    if isinstance(iceberg_type, MapType):
        return [iceberg_type.key_field, iceberg_type.value_field]
    return []


def _same_type(file_type: IcebergType, table_type: IcebergType) -> bool:
    """Whether the file stores exactly the table's type: same nested ids, names and types.

    Nullability is not compared: making a column optional rewrites no file.
    """
    if isinstance(file_type, PrimitiveType) or isinstance(table_type, PrimitiveType):
        return file_type == table_type
    if type(file_type) is not type(table_type):
        return False
    file_children, table_children = _children(file_type), _children(table_type)
    return len(file_children) == len(table_children) and all(
        f.field_id == t.field_id and f.name == t.name and _same_type(f.field_type, t.field_type)
        for f, t in zip(file_children, table_children, strict=True)
    )


def _unreadable(file_type: IcebergType, table_type: IcebergType) -> str | None:
    """Why a file's *file_type* cannot be read as *table_type*, or None when it can.

    Nested fields are matched by id: one the file lacks reads as nulls, and
    each one it has must be readable in turn.
    """
    if isinstance(file_type, PrimitiveType) and isinstance(table_type, PrimitiveType):
        if file_type == table_type:
            return None
        try:
            promote(file_type, table_type)
        except ResolveError:
            return f"{file_type} cannot be read as {table_type}"
        return None
    if type(file_type) is not type(table_type):
        return f"{file_type} cannot be read as {table_type}"
    stored = {child.field_id: child for child in _children(file_type)}
    for child in _children(table_type):
        held = stored.get(child.field_id)
        if held is not None and (why := _unreadable(held.field_type, child.field_type)):
            return why
    return None


def _reshape(
    array: pa.Array, file_type: IcebergType, table_type: IcebergType, target: pa.DataType
) -> pa.Array:
    """*array*, stored as *file_type*, rebuilt as *table_type* (in Arrow, *target*)."""
    if isinstance(table_type, PrimitiveType):
        return array.cast(target)
    mask = array.is_null()
    if isinstance(table_type, StructType):
        assert isinstance(file_type, StructType)
        stored = {child.field_id: i for i, child in enumerate(file_type.fields)}
        children = []
        for child, field in zip(table_type.fields, target, strict=True):
            i = stored.get(child.field_id)
            children.append(
                pa.nulls(len(array), field.type)
                if i is None
                else _reshape(
                    array.field(i), file_type.fields[i].field_type, child.field_type, field.type
                )
            )
        return pa.StructArray.from_arrays(children, fields=list(target), mask=mask)
    # A sliced list's offsets point into its whole child array, and pyarrow
    # takes no null mask with sliced offsets: rebase them to this slice.
    start, end = array.offsets[0].as_py(), array.offsets[-1].as_py()
    offsets = pc.subtract(array.offsets, start)
    if isinstance(table_type, ListType):
        assert isinstance(file_type, ListType)
        values = _reshape(
            array.values.slice(start, end - start),
            file_type.element_type,
            table_type.element_type,
            target.value_type,
        )
        return pa.ListArray.from_arrays(offsets, values, type=target, mask=mask)
    assert isinstance(file_type, MapType) and isinstance(table_type, MapType)
    keys = _reshape(
        array.keys.slice(start, end - start),
        file_type.key_type,
        table_type.key_type,
        target.key_type,
    )
    items = _reshape(
        array.items.slice(start, end - start),
        file_type.value_type,
        table_type.value_type,
        target.item_type,
    )
    return pa.MapArray.from_arrays(offsets, keys, items, type=target, mask=mask)


def snapshot_arrow_field(field: NestedField) -> pa.Field:
    """The Arrow field a column the file lacks reads as: nulls of the table's type."""
    data_type = _small(schema_to_pyarrow(field.field_type, include_field_ids=False))
    return pa.field(field.name, data_type, nullable=True)


def snapshot_arrow_schema(schema: Schema) -> pa.Schema:
    """*schema* in Arrow, typed as a Parquet read gives it."""
    return pa.schema(
        pa.field(
            field.name,
            _small(schema_to_pyarrow(field.field_type, include_field_ids=False)),
            nullable=not field.required,
        )
        for field in schema.fields
    )


def file_columns(
    file_schema: pa.Schema,
    snapshot_schema: Schema,
    name_mapping: NameMapping | None,
    *,
    table_identity: str,
    file_path: str,
    format_version: int,
) -> tuple[Column, ...] | None:
    """How a file with *file_schema* reads as *snapshot_schema*, or None when it already does.

    Every column takes its nullability from the snapshot, so files written on
    either side of a required column becoming optional stream as one schema.

    Raises ``UnsupportedTableFormatError`` for a type change Iceberg does not
    allow, at any depth.
    """
    stored = {
        field.field_id: field
        for field in _file_schema(file_schema, snapshot_schema, name_mapping, format_version).fields
    }
    columns = []
    for field in snapshot_schema.fields:
        nullable = not field.required
        held = stored.get(field.field_id)
        if held is None:
            columns.append(
                Column(
                    field.name,
                    None,
                    snapshot_arrow_field(field).with_nullable(nullable),
                    default=field.initial_default,
                )
            )
            continue
        physical = file_schema.field(held.name)
        if _same_type(held.field_type, field.field_type):
            read_as = physical.with_name(field.name).with_nullable(nullable)
            if pa.types.is_timestamp(physical.type):
                # A v1 or v2 table allows only microseconds, but a file can hold
                # nanoseconds (or INT96, read as nanoseconds): read it at the
                # table's unit, truncating, as pyiceberg does.
                read_as = read_as.with_type(
                    _small(schema_to_pyarrow(field.field_type, include_field_ids=False))
                )
            columns.append(Column(field.name, held.name, read_as))
            continue
        why = _unreadable(held.field_type, field.field_type)
        if why is not None:
            raise UnsupportedTableFormatError(
                f"Table {table_identity}: {file_path} stores column {field.name!r} in a "
                f"way Iceberg cannot read as the table's schema ({why})."
            )
        target = _small(schema_to_pyarrow(field.field_type, include_field_ids=False))
        # A promoted primitive is cast by read_as_snapshot's schema; a nested
        # column is rebuilt field by field.
        reshape = (
            None
            if isinstance(field.field_type, PrimitiveType)
            else (held.field_type, field.field_type)
        )
        columns.append(
            Column(
                field.name,
                held.name,
                physical.with_name(field.name).with_type(target).with_nullable(nullable),
                reshape,
            )
        )

    # Compare sources and reshapes, not just fields: Arrow's type equality
    # ignores field ids, so a re-added column or nested field of the old one's
    # name and type looks the same.
    as_stored = [(field.name, field.name, field, None, None) for field in file_schema]
    if as_stored == [(c.name, c.source, c.field, c.reshape, c.default) for c in columns]:
        return None
    return tuple(columns)


def _file_schema(
    file_schema: pa.Schema,
    snapshot_schema: Schema,
    name_mapping: NameMapping | None,
    format_version: int,
) -> Schema:
    """The Iceberg schema a data file holds (by the name mapping when it has no ids).

    Nanosecond timestamps are read as microseconds in a v1 or v2 table, the
    only unit those formats allow; pyiceberg's own reader does the same.
    """
    return pyarrow_to_schema(
        file_schema,
        name_mapping or create_mapping_from_schema(snapshot_schema),
        downcast_ns_timestamp_to_us=format_version <= 2,
        format_version=cast(Any, format_version),
    )


def stored_columns(
    file_schema: pa.Schema,
    snapshot_schema: Schema,
    name_mapping: NameMapping | None,
    format_version: int,
) -> dict[int, str]:
    """The file's top-level columns by field id."""
    schema = _file_schema(file_schema, snapshot_schema, name_mapping, format_version)
    return {field.field_id: field.name for field in schema.fields}


def absent_column(num_rows: int, data_type: pa.DataType, default: Any) -> pa.Array:
    """A column the data file predates: its Iceberg v3 initial-default, else nulls."""
    if default is None:
        return pa.nulls(num_rows, data_type)
    return pa.repeat(pa.scalar(default, type=data_type), num_rows)


def read_as_snapshot(
    table: pa.Table, columns: tuple[Column, ...], names: list[str] | None
) -> pa.Table:
    """*table*, read from the file, as the snapshot's *names* (all of them when None)."""
    wanted = [c for c in columns if names is None or c.name in names]
    if names is not None:
        order = {name: i for i, name in enumerate(names)}
        wanted.sort(key=lambda c: order[c.name])
    arrays = []
    for column in wanted:
        if column.source is None:
            arrays.append(absent_column(table.num_rows, column.field.type, column.default))
        elif column.reshape is not None:
            chunks = table.column(column.source).chunks
            arrays.append(
                pa.chunked_array(
                    [_reshape(chunk, *column.reshape, column.field.type) for chunk in chunks],
                    type=column.field.type,
                )
            )
        else:
            data = table.column(column.source)
            if data.type != column.field.type:
                # Widen a promoted column; truncate a finer timestamp to the
                # table's unit, as pyiceberg does (hence not a safe cast).
                data = data.cast(column.field.type, safe=False)
            arrays.append(data)
    return pa.table(arrays, schema=pa.schema([c.field for c in wanted]))


def source_columns(columns: tuple[Column, ...], names: list[str] | None) -> list[str]:
    """The file's columns to read for the snapshot's *names* (all of them when None)."""
    return [
        c.source for c in columns if c.source is not None and (names is None or c.name in names)
    ]
