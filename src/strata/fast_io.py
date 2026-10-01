"""Arrow IPC stream I/O: cache reads, concatenation and validation.

Prefer :func:`stream_concat_ipc_segments` for large responses: memory stays
O(segment), not O(response). ``STRATA_FAST_CONCAT`` picks the buffered concat
backend: ``rust`` (zero-parse, default when the extension is built) or ``pyarrow``.
"""

import os
import time
from collections.abc import Iterable, Iterator

import pyarrow as pa
import pyarrow.ipc as ipc

type _BytesLike = bytes | bytearray | memoryview

_RUST_AVAILABLE = False
_rust_module = None

try:
    from strata import _strata_core

    _rust_module = _strata_core
    _RUST_AVAILABLE = True
except ImportError:
    pass

# Concat implementation selection via environment variable
# "rust" = use Rust byte manipulation (zero-parse, fastest)
# "pyarrow" = use PyArrow parsing (slower but handles edge cases)
_FAST_CONCAT_MODE = os.environ.get("STRATA_FAST_CONCAT", "rust" if _RUST_AVAILABLE else "pyarrow")


def is_rust_available() -> bool:
    """Check if Rust acceleration module is available."""
    return _RUST_AVAILABLE


def get_concat_mode() -> str:
    """Return current concat mode ('rust' or 'pyarrow')."""
    return _FAST_CONCAT_MODE


# Below this size, Python's read_bytes() beats the Rust mmap path: mmap syscall + FFI + PyBytes copy
# overhead is not amortized. Measured crossover is ~4-6 MB (benchmarks/bench_rust_ext.py). Override
# with STRATA_MMAP_MIN_BYTES (0 forces mmap always).
MMAP_MIN_BYTES = int(os.environ.get("STRATA_MMAP_MIN_BYTES", 4 * 1024 * 1024))


def read_file_mmap(path: str) -> bytes:
    """Read a file, via Rust mmap at or above ``MMAP_MIN_BYTES``.

    Smaller files and any mmap error fall back to ``Path.read_bytes()``.
    """
    if _RUST_AVAILABLE and _rust_module is not None:
        try:
            if os.stat(path).st_size >= MMAP_MIN_BYTES:
                return bytes(_rust_module.read_file_bytes(path))
        except Exception:
            # Small file, missing file, or mmap failure: fall through to the
            # plain read below, which reproduces the real error if there is one.
            pass

    from pathlib import Path

    return Path(path).read_bytes()


def _concat_stream_bytes_pyarrow(segments: list[_BytesLike]) -> bytes:
    """Concatenate by parsing and re-serializing batches; slower but handles every case."""
    # Single pass straight to the output buffer; an intermediate list would add ~1x memory.
    sink = pa.BufferOutputStream()
    writer = None

    for segment in segments:
        if not segment:
            continue
        reader = ipc.open_stream(pa.BufferReader(segment))
        if writer is None:
            writer = ipc.new_stream(sink, reader.schema)
        for batch in reader:
            writer.write_batch(batch)

    if writer is None:
        return b""

    writer.close()
    return sink.getvalue().to_pybytes()


def _concat_stream_bytes_rust(segments: list[_BytesLike]) -> bytes:
    """Concatenate at the byte level without parsing Arrow data.

    Takes C-contiguous bytes-like segments to avoid a ``bytes()`` copy. Falls back
    to PyArrow if the extension is missing or errors.
    """
    if not _RUST_AVAILABLE or _rust_module is None:
        return _concat_stream_bytes_pyarrow(segments)

    try:
        return bytes(_rust_module.concat_ipc_streams(segments))
    except Exception:
        return _concat_stream_bytes_pyarrow(segments)


def concat_stream_bytes(segments: Iterable[_BytesLike]) -> bytes:
    """Concatenate Arrow IPC stream segments into one buffered stream.

    Empty segments are skipped; the backend follows ``STRATA_FAST_CONCAT``.
    """
    segment_list = [s for s in segments if s]
    if not segment_list:
        return b""

    if len(segment_list) == 1:
        return bytes(segment_list[0])

    if _FAST_CONCAT_MODE == "rust":
        return _concat_stream_bytes_rust(segment_list)
    else:
        return _concat_stream_bytes_pyarrow(segment_list)


class _StreamingBuffer:
    """File-like sink Arrow writes into and the caller drains with ``read_new()``.

    Append-only chunk list (no BytesIO seek/truncate); ``tell()`` reports the
    logical write position, which draining does not reset.
    """

    # Bounds list growth from many small writes.
    _COMPACT_THRESHOLD = 64

    def __init__(self) -> None:
        self._chunks: list[bytes] = []
        self._write_pos = 0  # Logical write position (for Arrow's tell())
        self._pending_bytes = 0  # Bytes available to drain

    def write(self, data: bytes) -> int:
        """Write data to buffer (called by Arrow)."""
        if data:
            self._chunks.append(data)
            self._write_pos += len(data)
            self._pending_bytes += len(data)
        return len(data)

    def pending_bytes(self) -> int:
        """Return number of bytes available to read."""
        return self._pending_bytes

    def read_new(self) -> bytes:
        """Drain all pending bytes as a single bytes object."""
        if not self._chunks:
            return b""

        if len(self._chunks) == 1:
            result = self._chunks[0]
        else:
            result = b"".join(self._chunks)

        self._chunks.clear()
        self._pending_bytes = 0
        return result

    def tell(self) -> int:
        """Return current write position (logical, for Arrow)."""
        return self._write_pos

    def seek(self, pos: int, whence: int = 0) -> int:
        """Report a position without moving; Arrow's IPC writer never seeks backwards."""
        if whence == 2:  # SEEK_END
            return self._write_pos
        elif whence == 0:  # SEEK_SET
            return pos
        elif whence == 1:  # SEEK_CUR
            return self._write_pos + pos
        return self._write_pos

    def flush(self) -> None:
        """No-op; the buffer is in memory."""
        pass

    @property
    def closed(self) -> bool:
        """Always False; the buffer is never closed."""
        return False


def validate_ipc_stream(data: _BytesLike) -> int:
    """Validate that ``data`` is exactly one Arrow IPC stream; return its row count.

    Concatenated streams parse cleanly but readers drop every row after the
    first end-of-stream marker, so trailing bytes are an error.

    Raises:
        ValueError: If bytes remain after the first stream's EOS marker.
        pyarrow.lib.ArrowInvalid: If the data is not a parseable IPC stream.
    """
    if not data:
        return 0

    buf = pa.BufferReader(pa.py_buffer(data))
    reader = ipc.open_stream(buf)
    rows = sum(batch.num_rows for batch in reader)
    if buf.tell() != len(data):
        raise ValueError(
            f"Trailing bytes after IPC stream end: {len(data) - buf.tell()} of "
            f"{len(data)} bytes unread (concatenated streams?)"
        )
    return rows


def validate_ipc_stream_reader(source) -> tuple[int, str]:
    """Bounded-memory :func:`validate_ipc_stream` over a readable file-like.

    Reads one batch at a time. Returns ``(row_count, schema_json)``.

    Raises:
        ValueError: If bytes remain after the first stream's EOS marker.
        pyarrow.lib.ArrowInvalid: If the data is not a parseable IPC stream.
    """
    reader = ipc.open_stream(source)
    schema_json = reader.schema.to_string()
    rows = sum(batch.num_rows for batch in reader)
    # The reader stops at the first EOS; trailing bytes mean concatenated streams, which standard
    # readers silently truncate.
    trailing = source.read()
    if trailing:
        raise ValueError(
            f"Trailing bytes after IPC stream end: {len(trailing)} bytes (concatenated streams?)"
        )
    return rows, schema_json


class IncrementalIpcMerger:
    """Push-style merge of complete IPC stream segments, for async producers.

    The outputs of ``feed`` and ``finish``, concatenated in order, form one valid
    IPC stream. ``feed`` raises ``ValueError`` on a schema mismatch.
    """

    def __init__(self) -> None:
        self._buf = _StreamingBuffer()
        self._writer: ipc.RecordBatchStreamWriter | None = None
        self._schema: pa.Schema | None = None

    def feed(self, segment: _BytesLike) -> bytes:
        """Merge one complete IPC stream segment; return bytes ready to emit."""
        if not segment:
            return b""

        reader = ipc.open_stream(pa.BufferReader(segment))
        if self._writer is None:
            self._schema = reader.schema
            self._writer = ipc.new_stream(self._buf, self._schema)
        elif not reader.schema.equals(self._schema):
            raise ValueError(
                f"Schema mismatch across segments: expected {self._schema}, got {reader.schema}"
            )

        for batch in reader:
            self._writer.write_batch(batch)
        return self._buf.read_new()

    def finish(self) -> bytes:
        """Close the output stream; return the EOS marker and any remainder."""
        if self._writer is None:
            return b""
        self._writer.close()
        self._writer = None
        return self._buf.read_new()


# Smaller chunks pay syscall overhead; larger ones cost memory.
DEFAULT_MIN_CHUNK_SIZE = 256 * 1024


class StreamLimitExceeded(RuntimeError):
    """Output size limit hit; aborts the stream so clients see an error, not truncation."""

    pass


class StreamDeadlineExceeded(RuntimeError):
    """Deadline hit; aborts the stream so clients see an error, not truncation."""

    pass


def stream_concat_ipc_segments(
    segments: Iterable[bytes],
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
    max_output_bytes: int | None = None,
    deadline: float | None = None,
) -> Iterator[bytes]:
    """Stream-concatenate complete IPC stream segments into one output stream.

    Peak memory is about one segment plus ``min_chunk_size``. Chunks are yielded
    at ``min_chunk_size``, or at a segment boundary past a quarter of it. The caller
    should abort the HTTP connection on a limit error rather than truncate.

    Args:
        max_output_bytes: Total yielded bytes allowed; None for no limit.
        deadline: ``time.monotonic()`` value after which to abort.

    Raises:
        StreamLimitExceeded: If ``max_output_bytes`` would be exceeded.
        StreamDeadlineExceeded: If ``deadline`` passes.
        ValueError: If schemas differ across segments.
    """
    buf = _StreamingBuffer()
    writer = None
    expected_schema = None
    total_bytes_yielded = 0

    def check_limits() -> None:
        """Raise ``StreamDeadlineExceeded`` once the deadline has passed."""
        if deadline is not None and time.monotonic() > deadline:
            raise StreamDeadlineExceeded(
                f"Stream deadline exceeded (deadline={deadline:.2f}s monotonic)"
            )

    def yield_chunk(chunk: bytes) -> bytes:
        """Count ``chunk`` against the size limit and return it."""
        nonlocal total_bytes_yielded
        if max_output_bytes is not None:
            if total_bytes_yielded + len(chunk) > max_output_bytes:
                raise StreamLimitExceeded(
                    f"Stream size limit exceeded: "
                    f"{total_bytes_yielded + len(chunk)} > {max_output_bytes} bytes"
                )
        total_bytes_yielded += len(chunk)
        return chunk

    try:
        for segment in segments:
            check_limits()

            if not segment:
                continue

            reader = ipc.open_stream(pa.BufferReader(segment))

            if writer is None:
                expected_schema = reader.schema
                writer = ipc.new_stream(buf, expected_schema)
                # Yield the schema immediately so the client can start processing.
                schema_bytes = buf.read_new()
                if schema_bytes:
                    yield yield_chunk(schema_bytes)
            else:
                # Fail here with a clear error rather than with a confusing Arrow decode error on
                # the client.
                if not reader.schema.equals(expected_schema):
                    raise ValueError(
                        f"Schema mismatch across segments: "
                        f"expected {expected_schema}, got {reader.schema}"
                    )

            for batch in reader:
                check_limits()
                writer.write_batch(batch)
                if buf.pending_bytes() >= min_chunk_size:
                    chunk = buf.read_new()
                    if chunk:
                        yield yield_chunk(chunk)

            # At a segment boundary, yield only past a quarter chunk to avoid tiny chunks for narrow
            # tables with many small batches. The remainder is flushed at the end regardless.
            boundary_threshold = min_chunk_size // 4
            if buf.pending_bytes() >= boundary_threshold:
                chunk = buf.read_new()
                if chunk:
                    yield yield_chunk(chunk)

        # Only on normal completion; exceptions skip this and go to finally.
        if writer is not None:
            check_limits()
            writer.close()
            writer = None  # Mark as closed so finally doesn't double-close
            final_bytes = buf.read_new()
            if final_bytes:
                yield yield_chunk(final_bytes)

    finally:
        # Don't yield in finally; let the exception propagate cleanly.
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass  # Ignore close errors during exception handling
