"""Hand-built Iceberg snapshots for what pyiceberg cannot write: delete files.

pyiceberg writes data manifests only, so ``commit_files`` writes the Parquet
files, a DATA and a DELETES manifest, a manifest list and the snapshot itself,
the way Flink's upsert sink commits a row delta. Real-writer coverage is in
tests/test_lake_catalogs_integration.py.
"""

from __future__ import annotations

import random
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from pyiceberg.manifest import (
    DataFile,
    DataFileContent,
    FileFormat,
    ManifestContent,
    ManifestEntry,
    ManifestEntryStatus,
    ManifestWriterV2,
    write_manifest_list,
)
from pyiceberg.table import Table
from pyiceberg.table.snapshots import Operation, Snapshot, Summary
from pyiceberg.table.update import AddSnapshotUpdate, AssertRefSnapshotId, SetSnapshotRefUpdate
from pyiceberg.typedef import Record

_FIELD_ID = b"PARQUET:field_id"


class _DeleteManifestWriter(ManifestWriterV2):
    def content(self) -> ManifestContent:
        return ManifestContent.DELETES


def _write(table: Table, rows: pa.Table, prefix: str) -> tuple[str, int]:
    """*rows* as a Parquet file in the table's data directory, columns tagged with field ids.

    Each column's name is its field's name in the current schema, or
    ``"<id>"`` to give an id directly (a column dropped from the schema).
    """
    schema = table.schema()
    fields = []
    for field in rows.schema:
        field_id = (
            int(field.name) if field.name.isdigit() else schema.find_field(field.name).field_id
        )
        fields.append(field.with_metadata({_FIELD_ID: str(field_id).encode()}))
    path = (
        Path(table.location().removeprefix("file://")) / "data" / f"{prefix}-{uuid.uuid4()}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(rows.cast(pa.schema(fields)), path)
    return path.as_uri(), path.stat().st_size


def equality_delete(
    table: Table,
    rows: pa.Table,
    *,
    partition: Record | None = None,
    file_format: FileFormat = FileFormat.PARQUET,
    lower_bounds: dict[int, bytes] | None = None,
    upper_bounds: dict[int, bytes] | None = None,
    null_value_counts: dict[int, int] | None = None,
) -> DataFile:
    """An equality delete file keyed on *rows*' columns (all of them)."""
    uri, size = _write(table, rows, "eq-delete")
    schema = table.schema()
    equality_ids = [
        int(name) if name.isdigit() else schema.find_field(name).field_id
        for name in rows.column_names
    ]
    delete_file = DataFile.from_args(
        content=DataFileContent.EQUALITY_DELETES,
        file_path=uri,
        file_format=file_format,
        partition=partition or Record(),
        record_count=rows.num_rows,
        file_size_in_bytes=size,
        equality_ids=equality_ids,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        null_value_counts=null_value_counts,
    )
    delete_file.spec_id = table.metadata.default_spec_id
    return delete_file


def positional_delete(table: Table, rows: dict[str, list[int]]) -> DataFile:
    """A positional delete file deleting *rows* (data file path -> positions)."""
    deleted = pa.table(
        {
            "file_path": pa.array([p for p, positions in rows.items() for _ in positions]),
            "pos": pa.array([n for positions in rows.values() for n in positions], pa.int64()),
        }
    )
    deleted = deleted.cast(
        pa.schema(
            [
                pa.field("file_path", pa.string(), metadata={_FIELD_ID: b"2147483546"}),
                pa.field("pos", pa.int64(), metadata={_FIELD_ID: b"2147483545"}),
            ]
        )
    )
    path = (
        Path(table.location().removeprefix("file://"))
        / "data"
        / f"pos-delete-{uuid.uuid4()}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(deleted, path)
    delete_file = DataFile.from_args(
        content=DataFileContent.POSITION_DELETES,
        file_path=path.as_uri(),
        file_format=FileFormat.PARQUET,
        partition=Record(),
        record_count=deleted.num_rows,
        file_size_in_bytes=path.stat().st_size,
    )
    delete_file.spec_id = table.metadata.default_spec_id
    return delete_file


def data_file(table: Table, rows: pa.Table, *, partition: Record | None = None) -> DataFile:
    """A data file holding *rows*."""
    uri, size = _write(table, rows, "data")
    written = DataFile.from_args(
        content=DataFileContent.DATA,
        file_path=uri,
        file_format=FileFormat.PARQUET,
        partition=partition or Record(),
        record_count=rows.num_rows,
        file_size_in_bytes=size,
    )
    written.spec_id = table.metadata.default_spec_id
    return written


def commit_files(table: Table, *files: DataFile) -> int:
    """Commit *files* (data and delete files) as one new snapshot; its sequence number."""
    metadata = table.metadata
    parent = table.current_snapshot()
    assert parent is not None
    snapshot_id = random.randint(1, 2**62)
    sequence_number = metadata.last_sequence_number + 1
    metadata_dir = Path(table.location().removeprefix("file://")) / "metadata"
    manifests = list(parent.manifests(table.io))
    for writer_class, content in (
        (ManifestWriterV2, DataFileContent.DATA),
        (_DeleteManifestWriter, DataFileContent.POSITION_DELETES),
        (_DeleteManifestWriter, DataFileContent.EQUALITY_DELETES),
    ):
        batch = [file for file in files if file.content == content]
        if not batch:
            continue
        output = table.io.new_output(str(metadata_dir / f"{uuid.uuid4()}-m0.avro"))
        with writer_class(metadata.spec(), metadata.schema(), output, snapshot_id, "deflate") as w:
            for file in batch:
                w.add(
                    ManifestEntry.from_args(
                        status=ManifestEntryStatus.ADDED,
                        snapshot_id=snapshot_id,
                        sequence_number=sequence_number,
                        file_sequence_number=sequence_number,
                        data_file=file,
                    )
                )
        manifests.append(w.to_manifest_file())
    list_path = metadata_dir / f"snap-{snapshot_id}-{uuid.uuid4()}.avro"
    with write_manifest_list(
        2,
        table.io.new_output(str(list_path)),
        snapshot_id,
        parent.snapshot_id,
        sequence_number,
        "deflate",
    ) as writer:
        writer.add_manifests(manifests)
    snapshot = Snapshot(
        snapshot_id=snapshot_id,
        parent_snapshot_id=parent.snapshot_id,
        sequence_number=sequence_number,
        manifest_list=list_path.as_uri(),
        summary=Summary(Operation.OVERWRITE),
        schema_id=metadata.current_schema_id,
    )
    transaction = table.transaction()
    transaction._apply(
        (
            AddSnapshotUpdate(snapshot=snapshot),
            SetSnapshotRefUpdate(ref_name="main", type="branch", snapshot_id=snapshot_id),
        ),
        (AssertRefSnapshotId(ref="main", snapshot_id=parent.snapshot_id),),
    )
    transaction.commit_transaction()
    return sequence_number
