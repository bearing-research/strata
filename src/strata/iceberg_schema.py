"""Iceberg schema evolution: reading an older data file as the snapshot's schema.

Iceberg names a column by field id, not by name, and changing a table's schema
rewrites no data files. So an older file may lack a column the table added
since (it reads as nulls), hold a column the table renamed under its old name,
hold a dropped column whose name a new column reuses (that is not the new
column, which also reads as nulls), or store a narrower type than the one the
table promoted it to (int to long, float to double, a wider decimal). Iceberg's
Parquet files record each column's field id, which is what matches a file's
columns to the snapshot's; a file without them is matched through the table's
name mapping, or by name when it has none.
"""

from typing import NamedTuple

import pyarrow as pa
from pyiceberg.exceptions import ResolveError
from pyiceberg.io.pyarrow import pyarrow_to_schema, schema_to_pyarrow
from pyiceberg.schema import Schema, promote
from pyiceberg.table.name_mapping import NameMapping, create_mapping_from_schema
from pyiceberg.types import IcebergType, ListType, MapType, NestedField, PrimitiveType, StructType


class UnsupportedTableFormatError(RuntimeError):
    """The table uses an Iceberg feature Strata cannot read correctly."""


class Column(NamedTuple):
    """One column of the snapshot's schema, as a data file holds it."""

    name: str  # in the snapshot's schema
    source: str | None  # the file's column holding it; None when the file predates it
    field: pa.Field  # what the scan returns


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
) -> tuple[Column, ...] | None:
    """How a file with *file_schema* reads as *snapshot_schema*, or None when it already does.

    Raises ``UnsupportedTableFormatError`` for a change that needs more than
    renaming, filling or widening a top-level column: a nested field added,
    dropped or renamed, or a type change Iceberg does not allow.
    """
    mapping = name_mapping or create_mapping_from_schema(snapshot_schema)
    stored = {field.field_id: field for field in pyarrow_to_schema(file_schema, mapping).fields}
    columns = []
    for field in snapshot_schema.fields:
        held = stored.get(field.field_id)
        if held is None:
            columns.append(Column(field.name, None, snapshot_arrow_field(field)))
            continue
        physical = file_schema.field(held.name)
        if _same_type(held.field_type, field.field_type):
            columns.append(Column(field.name, held.name, physical.with_name(field.name)))
            continue
        if isinstance(held.field_type, PrimitiveType) and isinstance(
            field.field_type, PrimitiveType
        ):
            try:
                promote(held.field_type, field.field_type)
            except ResolveError as e:
                raise UnsupportedTableFormatError(
                    f"Table {table_identity}: {file_path} stores column {field.name!r} as "
                    f"{held.field_type}, which Iceberg cannot read as {field.field_type}."
                ) from e
            widened = _small(schema_to_pyarrow(field.field_type, include_field_ids=False))
            columns.append(
                Column(field.name, held.name, physical.with_name(field.name).with_type(widened))
            )
            continue
        raise UnsupportedTableFormatError(
            f"Table {table_identity}: {file_path} predates a change inside nested column "
            f"{field.name!r} ({held.field_type} in the file, {field.field_type} in the table). "
            "Strata reconciles top-level columns only; compact the table "
            "(rewrite_data_files) or scan a snapshot from before the change."
        )

    # Compare sources, not just fields: Arrow's field equality ignores the
    # field id, so a re-added column of the old one's name and type looks the same.
    as_stored = [(field.name, field.name, field) for field in file_schema]
    if as_stored == [(column.name, column.source, column.field) for column in columns]:
        return None
    return tuple(columns)


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
            arrays.append(pa.nulls(table.num_rows, column.field.type))
        else:
            arrays.append(table.column(column.source))
    # Built to the fields' schema, which casts a promoted column to its new width.
    return pa.table(arrays, schema=pa.schema([c.field for c in wanted]))


def source_columns(columns: tuple[Column, ...], names: list[str] | None) -> list[str]:
    """The file's columns to read for the snapshot's *names* (all of them when None)."""
    return [
        c.source for c in columns if c.source is not None and (names is None or c.name in names)
    ]
