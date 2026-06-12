"""Fast I/O utilities for Arrow IPC stream operations.

This module provides optimized functions for Arrow IPC stream handling.
The cache now stores data in stream format, so the hot path for cache
hits is simply reading raw bytes (no Arrow parsing needed).

For concatenating multiple streams (multi-row-group scans), we provide:
- concat_stream_bytes: Buffered concatenation (returns all bytes at once)
- stream_concat_ipc_segments: True streaming (yields chunks incrementally)

The streaming version is preferred for large responses as it keeps memory
usage bounded to O(single segment) instead of O(total response).

Performance tuning:
- STRATA_FAST_CONCAT env var controls concat implementation:
  - "rust": Use Rust byte manipulation (zero-parse, fastest)
  - "pyarrow": Use PyArrow parsing (slower but handles edge cases)
  - Default: "rust" if available, else "pyarrow"
"""

import os
import time
from collections.abc import Iterable, Iterator

import pyarrow as pa
import pyarrow.ipc as ipc

type _BytesLike = bytes | bytearray | memoryview

# Try to import Rust module for fast byte manipulation
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


def read_file_mmap(path: str) -> bytes:
    """Read file using memory-mapping for faster cache hits.

    Uses Rust mmap implementation when available, falling back to Python
    read_bytes() otherwise. Memory-mapping is faster for large files and
    for repeated access to the same file (OS page cache reuse).

    Args:
        path: Path to the file to read

    Returns:
        bytes: File contents
    """
    if _RUST_AVAILABLE and _rust_module is not None:
        try:
            return bytes(_rust_module.read_file_bytes(path))
        except Exception:
            # Fall back to Python on any error
            pass

    # Fallback: standard Python file read
    from pathlib import Path

    return Path(path).read_bytes()


def _concat_stream_bytes_pyarrow(segments: list[_BytesLike]) -> bytes:
    """PyArrow implementation of concat_stream_bytes.

    Parses each segment and re-serializes batches. Slower but handles
    all edge cases and schema variations.
    """
    # Single-pass: stream batches directly to output buffer
    # Avoids intermediate list that would add ~1× memory overhead
    sink = pa.BufferOutputStream()
    writer = None

    for segment in segments:
        if not segment:
            continue
        reader = ipc.open_stream(pa.BufferReader(segment))
        if writer is None:
            writer = ipc.new_stream(sink, reader.schema)
        # Write each batch directly to output - no intermediate storage
        for batch in reader:
            writer.write_batch(batch)

    if writer is None:
        return b""

    writer.close()
    return sink.getvalue().to_pybytes()


def _concat_stream_bytes_rust(segments: list[_BytesLike]) -> bytes:
    """Rust implementation of concat_stream_bytes.

    Uses byte manipulation to concatenate streams without parsing Arrow data.
    Much faster for cache hits since it avoids deserialize/reserialize overhead.
    Accepts generic C-contiguous bytes-like segments so callers can hand Rust
    existing buffers without forcing an extra Python-side ``bytes()`` copy.

    Falls back to PyArrow if Rust module unavailable or on error.
    """
    if not _RUST_AVAILABLE or _rust_module is None:
        return _concat_stream_bytes_pyarrow(segments)

    try:
        # Rust concat_ipc_streams handles byte manipulation directly
        return bytes(_rust_module.concat_ipc_streams(segments))
    except Exception:
        # Fall back to PyArrow on any error (malformed data, etc.)
        return _concat_stream_bytes_pyarrow(segments)


def concat_stream_bytes(segments: Iterable[_BytesLike]) -> bytes:
    """Concatenate multiple Arrow IPC stream segments into one.

    When serving multiple cached row groups, we need to combine them
    into a single response stream for the client.

    Implementation selection:
    - STRATA_FAST_CONCAT=rust: Zero-parse byte manipulation (fastest)
    - STRATA_FAST_CONCAT=pyarrow: Full Arrow parsing (slower, handles edge cases)

    Args:
        segments: Iterable of Arrow IPC stream segments. Each segment may be
            ``bytes``, ``bytearray``, or ``memoryview``.

    Returns:
        bytes: Single combined IPC stream
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
    """A buffer that allows Arrow to write and us to read incrementally.

    Arrow's IPC writer needs a file-like object to write to. This buffer
    uses a list-based accumulator (append-only) to avoid the overhead of
    repeated seek(0)/truncate() calls on BytesIO.

    The buffer tracks a logical write position for Arrow's tell() calls,
    and drains accumulated chunks on read_new() without modifying the
    underlying storage structure.
    """

    # Compact the chunks list when it exceeds this many entries
    # Prevents unbounded list growth from many small writes
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
        """Seek in buffer.

        Arrow's IPC writer only uses tell() for position tracking and
        doesn't seek backwards. We support seek(0, 2) for append mode
        which is effectively a no-op since we're always at the end.
        """
        if whence == 2:  # SEEK_END
            return self._write_pos
        elif whence == 0:  # SEEK_SET
            return pos
        elif whence == 1:  # SEEK_CUR
            return self._write_pos + pos
        return self._write_pos

    def flush(self) -> None:
        """Flush buffer (no-op, we're in memory)."""
        pass

    @property
    def closed(self) -> bool:
        """Return False - buffer is never closed."""
        return False


def validate_ipc_stream(data: _BytesLike) -> int:
    """Validate that ``data`` is exactly one Arrow IPC stream; return its row count.

    Catches the #121 corruption class at write time: multiple complete IPC
    streams butted together parse "successfully" with a standard reader but
    silently drop every row after the first end-of-stream marker.

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
    """Bounded-memory variant of :func:`validate_ipc_stream` over a file-like.

    Reads ``source`` (any object with ``read``) incrementally — one record batch
    at a time — so a multi-GB artifact never has to be held whole in memory. Used
    to validate a write-through-persisted scan blob by re-reading it from the blob
    store. Same guarantees as ``validate_ipc_stream``: exactly one IPC stream, no
    trailing bytes after EOS (the #121 concatenation class).

    Returns ``(row_count, schema_json)``.

    Raises:
        ValueError: If bytes remain after the first stream's EOS marker.
        pyarrow.lib.ArrowInvalid: If the data is not a parseable IPC stream.
    """
    reader = ipc.open_stream(source)
    schema_json = reader.schema.to_string()
    rows = sum(batch.num_rows for batch in reader)
    # The reader stops at the first EOS; any remaining bytes mean concatenated
    # streams (the #121 corruption that standard readers silently truncate).
    trailing = source.read()
    if trailing:
        raise ValueError(
            f"Trailing bytes after IPC stream end: {len(trailing)} bytes (concatenated streams?)"
        )
    return rows, schema_json


class IncrementalIpcMerger:
    """Merge complete IPC stream segments into one stream, one segment at a time.

    Async producers (e.g. the scan streaming endpoint, which awaits each
    row-group fetch) cannot drive the synchronous ``stream_concat_ipc_segments``
    generator. This class exposes the same merge as push-style calls:
    ``feed(segment)`` returns merged bytes ready to send, ``finish()`` returns
    the EOS marker plus any remaining buffered data.

    The output of ``feed``/``finish`` calls, concatenated in order, is a
    single valid Arrow IPC stream (one schema header, all batches, one EOS).
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


# Default minimum chunk size for streaming (256 KB)
# Smaller chunks hurt throughput due to syscall overhead
# Larger chunks increase memory but improve throughput
DEFAULT_MIN_CHUNK_SIZE = 256 * 1024


class StreamLimitExceeded(RuntimeError):
    """Raised when a streaming limit is exceeded.

    This exception aborts the stream, ensuring clients receive an error
    instead of silently truncated data.
    """

    pass


class StreamDeadlineExceeded(RuntimeError):
    """Raised when a streaming deadline is exceeded.

    This exception aborts the stream, ensuring clients receive an error
    instead of silently truncated data.
    """

    pass


def stream_concat_ipc_segments(
    segments: Iterable[bytes],
    min_chunk_size: int = DEFAULT_MIN_CHUNK_SIZE,
    max_output_bytes: int | None = None,
    deadline: float | None = None,
) -> Iterator[bytes]:
    """Stream-concatenate multiple IPC stream segments into one output stream.

    This is the memory-efficient alternative to concat_stream_bytes().
    Instead of buffering the entire response, it yields chunks as they're
    produced, keeping memory usage bounded to O(single segment).

    Each input segment is a complete Arrow IPC stream (schema + batches + EOS).
    Output is a single IPC stream with all batches from all segments combined.

    Chunking strategy:
    - Buffer writes until we have at least min_chunk_size bytes
    - Yield on segment boundary only if buffer >= min_chunk_size/4 (avoids tiny chunks)
    - Always flush remaining data at end

    This avoids tiny yields (1-4 KB) that kill throughput while keeping
    memory bounded. The boundary threshold prevents small chunk overhead
    for narrow tables with tiny batches.

    Memory usage:
    - Peak: ~1 segment + min_chunk_size buffer
    - Output chunks are yielded when buffer threshold reached

    Enforcement hooks:
    - max_output_bytes: If total yielded bytes exceeds this, raises StreamLimitExceeded
    - deadline: If time.monotonic() exceeds this, raises StreamDeadlineExceeded

    These hooks ensure the stream is aborted (not silently truncated) when
    limits are exceeded. The caller should catch these exceptions and
    abort the HTTP connection.

    Use with FastAPI's StreamingResponse:
        return StreamingResponse(
            stream_concat_ipc_segments(segment_iterator),
            media_type="application/vnd.apache.arrow.stream"
        )

    Args:
        segments: Iterator of Arrow IPC stream bytes (e.g., cached row groups)
        min_chunk_size: Minimum bytes to buffer before yielding (default 256KB)
        max_output_bytes: Maximum total bytes to yield before aborting (None = no limit)
        deadline: Monotonic time deadline (from time.monotonic()) after which to abort

    Yields:
        bytes: Chunks of the combined IPC stream (schema, batches, EOS marker)

    Raises:
        StreamLimitExceeded: If max_output_bytes is exceeded
        StreamDeadlineExceeded: If deadline is exceeded
        ValueError: If schemas don't match across segments
    """
    buf = _StreamingBuffer()
    writer = None
    expected_schema = None
    total_bytes_yielded = 0

    def check_limits() -> None:
        """Check enforcement limits and raise if exceeded."""
        if deadline is not None and time.monotonic() > deadline:
            raise StreamDeadlineExceeded(
                f"Stream deadline exceeded (deadline={deadline:.2f}s monotonic)"
            )

    def yield_chunk(chunk: bytes) -> bytes:
        """Yield a chunk, checking size limit first."""
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

            # Read batches from this segment's IPC stream
            reader = ipc.open_stream(pa.BufferReader(segment))

            if writer is None:
                # Initialize output writer with schema from first segment
                expected_schema = reader.schema
                writer = ipc.new_stream(buf, expected_schema)
                # Always yield schema immediately so client can start processing
                schema_bytes = buf.read_new()
                if schema_bytes:
                    yield yield_chunk(schema_bytes)
            else:
                # Validate schema matches first segment
                # This catches bugs early with a clear error instead of
                # confusing Arrow decode errors on the client
                if not reader.schema.equals(expected_schema):
                    raise ValueError(
                        f"Schema mismatch across segments: "
                        f"expected {expected_schema}, got {reader.schema}"
                    )

            # Stream each batch from input to output
            for batch in reader:
                check_limits()
                writer.write_batch(batch)
                # Only yield if we've accumulated enough bytes
                if buf.pending_bytes() >= min_chunk_size:
                    chunk = buf.read_new()
                    if chunk:
                        yield yield_chunk(chunk)

            # At segment boundary: yield only if we have meaningful data
            # Use threshold (min_chunk_size/4) to avoid tiny chunk overhead
            # for narrow tables with many small batches.
            # Note: remaining data will be flushed at end regardless of size.
            boundary_threshold = min_chunk_size // 4
            if buf.pending_bytes() >= boundary_threshold:
                chunk = buf.read_new()
                if chunk:
                    yield yield_chunk(chunk)

            # Segment processed - its memory can be freed by GC

        # Close writer and yield EOS marker + any remaining buffered data
        # Only on normal completion - exceptions skip this and go to finally
        if writer is not None:
            check_limits()
            writer.close()
            writer = None  # Mark as closed so finally doesn't double-close
            # Always flush remaining data at end, regardless of size
            final_bytes = buf.read_new()
            if final_bytes:
                yield yield_chunk(final_bytes)

    finally:
        # Ensure writer is closed on any exception
        # Don't yield in finally - let exception propagate cleanly
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass  # Ignore close errors during exception handling
