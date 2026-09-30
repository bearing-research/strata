"""Iceberg positional deletes: which rows of a data file a snapshot has deleted.

A merge-on-read ``DELETE`` leaves the data files alone and records the deleted
rows beside them: a positional delete file (Parquet rows of ``file_path, pos``,
format v2) or a deletion vector (a Puffin blob per data file, format v3).
pyiceberg decides which delete files apply to which data file, by sequence
number and partition; this module reads them.

Positions are file-absolute, 0-based row numbers, so a row group's share is the
positions in ``[start, start + num_rows)`` shifted down by ``start``.
"""

import bisect
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as _pc
import pyarrow.parquet as pq
from pyiceberg.io import FileIO
from pyiceberg.table.deletion_vector import deletion_vectors_from_puffin_file
from pyiceberg.table.puffin import PuffinFile

from strata.metadata_cache import DeleteFileEntry, LRUCache

# pyarrow.compute registers its kernels at import time, so ty does not know
# members like ``is_in``/``unique``. Cast through Any.
pc = cast(Any, _pc)

# What ``_read`` can parse. ORC and Avro delete files are refused while planning.
READABLE_FORMATS = frozenset({"PARQUET", "PUFFIN"})


def _read(io: FileIO, delete_file: DeleteFileEntry) -> dict[str, pa.ChunkedArray]:
    """The positions *delete_file* deletes, per data file it names."""
    with io.new_input(delete_file.file_path).open() as stream:
        if delete_file.file_format == "PUFFIN":
            vectors = deletion_vectors_from_puffin_file(PuffinFile(stream.read()))
            return {vector.referenced_data_file: vector.to_vector() for vector in vectors}
        table = pq.read_table(stream, columns=["file_path", "pos"])
    paths = table.column("file_path")
    if pa.types.is_dictionary(paths.type):
        # A file pyarrow wrote from a dictionary column reads back as one, and
        # neither the sort nor the run encoding below takes dictionaries.
        table = table.set_column(
            table.schema.get_field_index("file_path"),
            "file_path",
            paths.cast(paths.type.value_type),
        )
    # One stable sort, then a slice per run of equal paths. Filtering the whole
    # table once per path was quadratic: a Spark delete file of 2M rows over
    # 2000 data files took longer to parse than the plan timeout.
    table = table.sort_by("file_path")
    runs = pc.run_end_encode(table.column("file_path").combine_chunks())
    positions = table.column("pos")
    by_data_file: dict[str, pa.ChunkedArray] = {}
    start = 0
    for path, end in zip(runs.values.to_pylist(), runs.run_ends.to_pylist(), strict=True):
        by_data_file[path] = positions.slice(start, end - start)
        start = end
    return by_data_file


class DeletedRows:
    """Deleted positions per data file, reading each delete file once.

    Delete files are immutable, like data files, so a parsed one stays valid
    for every snapshot that references it; re-planning a table does not
    re-read its deletes.
    """

    def __init__(self, max_files: int = 256) -> None:
        self._files: LRUCache[str, dict[str, pa.ChunkedArray]] = LRUCache(max_files)

    def for_data_file(
        self, io: FileIO, data_file_path: str, delete_files: tuple[DeleteFileEntry, ...]
    ) -> pa.Array | None:
        """Sorted, distinct file-absolute positions deleted from *data_file_path*."""
        chunks: list[pa.Array] = []
        for delete_file in delete_files:
            by_data_file = self._files.get(delete_file.file_path)
            if by_data_file is None:
                by_data_file = _read(io, delete_file)
                self._files.put(delete_file.file_path, by_data_file)
            positions = by_data_file.get(data_file_path)
            if positions is not None:
                chunks.extend(chunk.cast(pa.int64()) for chunk in positions.chunks)
        if not chunks:
            return None
        # Two delete files may both delete a row.
        positions = pc.unique(pa.chunked_array(chunks, pa.int64()))
        return positions.take(pc.sort_indices(positions))


def in_row_group(positions: pa.Array, start: int, num_rows: int) -> pa.Array | None:
    """Row-group-relative positions of the deleted rows among ``[start, start + num_rows)``."""

    def key(scalar: pa.Scalar) -> int:
        return scalar.as_py()

    low = bisect.bisect_left(positions, start, key=key)
    high = bisect.bisect_left(positions, start + num_rows, lo=low, key=key)
    if low == high:
        return None
    return pc.subtract(positions.slice(low, high - low), start)
