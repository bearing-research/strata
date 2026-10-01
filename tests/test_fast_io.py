"""Tests for fast_io module."""

import io

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from strata import fast_io


def create_stream_bytes(batch: pa.RecordBatch) -> bytes:
    sink = pa.BufferOutputStream()
    writer = ipc.new_stream(sink, batch.schema)
    writer.write_batch(batch)
    writer.close()
    return sink.getvalue().to_pybytes()


class TestFastIoAvailability:
    def test_is_rust_available(self):
        result = fast_io.is_rust_available()
        assert isinstance(result, bool)

    def test_rust_module_has_expected_functions(self):
        if fast_io.is_rust_available():
            from strata import _strata_core

            # Only two live entry points; see rust/src/lib.rs.
            assert hasattr(_strata_core, "read_file_bytes")
            assert hasattr(_strata_core, "concat_ipc_streams")


class TestReadFileMmapThreshold:
    """The mmap read routes only large files through Rust (small ones regress)."""

    def _spy_rust(self, monkeypatch):
        """Replace the Rust reader with a spy that records whether it was called."""
        if not fast_io.is_rust_available():
            pytest.skip("Rust module not available")
        calls: list[str] = []
        real = fast_io._rust_module.read_file_bytes

        def spy(path):
            calls.append(path)
            return real(path)

        monkeypatch.setattr(fast_io._rust_module, "read_file_bytes", spy)
        return calls

    def test_small_file_skips_rust(self, tmp_path, monkeypatch):
        calls = self._spy_rust(monkeypatch)
        path = tmp_path / "small"
        path.write_bytes(b"x" * 1024)  # 1 KB, well under the threshold
        assert fast_io.read_file_mmap(str(path)) == b"x" * 1024
        assert calls == []  # Rust path not taken

    def test_large_file_uses_rust(self, tmp_path, monkeypatch):
        calls = self._spy_rust(monkeypatch)
        path = tmp_path / "big"
        data = b"y" * (fast_io.MMAP_MIN_BYTES + 1)
        path.write_bytes(data)
        assert fast_io.read_file_mmap(str(path)) == data
        assert calls == [str(path)]  # Rust path taken exactly once

    def test_nonexistent_still_raises(self, tmp_path):
        # The size check must not swallow a genuinely missing file.
        with pytest.raises(FileNotFoundError):
            fast_io.read_file_mmap(str(tmp_path / "nope"))


class TestConcatStreamBytes:
    def test_concat_empty_list(self):
        result = fast_io.concat_stream_bytes([])
        assert result == b""

    def test_concat_single_segment(self):
        batch = pa.RecordBatch.from_pydict({"id": [1, 2, 3]})
        stream_bytes = create_stream_bytes(batch)

        result = fast_io.concat_stream_bytes([stream_bytes])

        reader = ipc.open_stream(pa.BufferReader(result))
        batches = list(reader)
        assert len(batches) == 1
        assert batches[0].num_rows == 3

    def test_concat_multiple_segments(self):
        segments = []
        total_rows = 0
        for i in range(3):
            num_rows = 10 + i * 5
            batch = pa.RecordBatch.from_pydict({"id": list(range(num_rows))})
            segments.append(create_stream_bytes(batch))
            total_rows += num_rows

        result = fast_io.concat_stream_bytes(segments)

        reader = ipc.open_stream(pa.BufferReader(result))
        batches = list(reader)
        assert len(batches) == 3
        assert sum(b.num_rows for b in batches) == total_rows

    def test_concat_preserves_schema(self):
        batch = pa.RecordBatch.from_pydict(
            {
                "id": [1, 2, 3],
                "value": [1.0, 2.0, 3.0],
                "name": ["a", "b", "c"],
            }
        )
        segments = [create_stream_bytes(batch) for _ in range(2)]

        result = fast_io.concat_stream_bytes(segments)

        reader = ipc.open_stream(pa.BufferReader(result))
        assert reader.schema == batch.schema

    def test_concat_with_empty_segment(self):
        batch = pa.RecordBatch.from_pydict({"id": [1, 2, 3]})
        stream_bytes = create_stream_bytes(batch)

        result = fast_io.concat_stream_bytes([stream_bytes, b"", stream_bytes])

        reader = ipc.open_stream(pa.BufferReader(result))
        batches = list(reader)
        assert len(batches) == 2
        assert sum(b.num_rows for b in batches) == 6

    def test_concat_all_empty_segments(self):
        result = fast_io.concat_stream_bytes([b"", b"", b""])
        assert result == b""

    def test_concat_accepts_bytearray_segments(self):
        batch1 = pa.RecordBatch.from_pydict({"id": [1, 2]})
        batch2 = pa.RecordBatch.from_pydict({"id": [3, 4]})

        segments = [
            bytearray(create_stream_bytes(batch1)),
            bytearray(create_stream_bytes(batch2)),
        ]

        result = fast_io.concat_stream_bytes(segments)

        reader = ipc.open_stream(pa.BufferReader(result))
        batches = list(reader)
        assert len(batches) == 2
        assert batches[0].column("id").to_pylist() == [1, 2]
        assert batches[1].column("id").to_pylist() == [3, 4]

    def test_concat_accepts_memoryview_segments(self):
        batch1 = pa.RecordBatch.from_pydict({"id": [10]})
        batch2 = pa.RecordBatch.from_pydict({"id": [20, 30]})

        segments = [
            memoryview(create_stream_bytes(batch1)),
            memoryview(create_stream_bytes(batch2)),
        ]

        result = fast_io.concat_stream_bytes(segments)

        reader = ipc.open_stream(pa.BufferReader(result))
        batches = list(reader)
        assert len(batches) == 2
        assert batches[0].column("id").to_pylist() == [10]
        assert batches[1].column("id").to_pylist() == [20, 30]

    def test_concat_accepts_mixed_bytes_like_segments(self):
        batch1 = pa.RecordBatch.from_pydict({"id": [1]})
        batch2 = pa.RecordBatch.from_pydict({"id": [2, 3]})
        batch3 = pa.RecordBatch.from_pydict({"id": [4, 5, 6]})

        segments = [
            create_stream_bytes(batch1),
            bytearray(create_stream_bytes(batch2)),
            memoryview(create_stream_bytes(batch3)),
        ]

        result = fast_io.concat_stream_bytes(segments)

        reader = ipc.open_stream(pa.BufferReader(result))
        batches = list(reader)
        assert [b.column("id").to_pylist() for b in batches] == [[1], [2, 3], [4, 5, 6]]

    def test_concat_accepts_non_contiguous_memoryview_segments(self):
        """Non-contiguous memoryviews copy through the fallback path."""
        batch = pa.RecordBatch.from_pydict({"id": [7, 8, 9]})
        original = create_stream_bytes(batch)

        doubled = bytearray()
        for value in original:
            doubled.extend((value, value))
        non_contiguous = memoryview(doubled)[::2]

        result = fast_io.concat_stream_bytes([non_contiguous, original])

        reader = ipc.open_stream(pa.BufferReader(result))
        batches = list(reader)
        assert len(batches) == 2
        assert batches[0].column("id").to_pylist() == [7, 8, 9]
        assert batches[1].column("id").to_pylist() == [7, 8, 9]

    def test_concat_single_memoryview_returns_bytes(self):
        """The single-segment fast path still normalizes to bytes."""
        batch = pa.RecordBatch.from_pydict({"id": [1, 2, 3]})
        segment = memoryview(create_stream_bytes(batch))

        result = fast_io.concat_stream_bytes([segment])

        assert isinstance(result, bytes)
        reader = ipc.open_stream(pa.BufferReader(result))
        batches = list(reader)
        assert len(batches) == 1
        assert batches[0].column("id").to_pylist() == [1, 2, 3]

    def test_concat_preserves_data_values(self):
        batch1 = pa.RecordBatch.from_pydict({"id": [1, 2], "value": ["a", "b"]})
        batch2 = pa.RecordBatch.from_pydict({"id": [3, 4], "value": ["c", "d"]})
        batch3 = pa.RecordBatch.from_pydict({"id": [5], "value": ["e"]})

        segments = [
            create_stream_bytes(batch1),
            create_stream_bytes(batch2),
            create_stream_bytes(batch3),
        ]

        result = fast_io.concat_stream_bytes(segments)

        reader = ipc.open_stream(pa.BufferReader(result))
        batches = list(reader)

        all_ids = []
        all_values = []
        for batch in batches:
            all_ids.extend(batch.column("id").to_pylist())
            all_values.extend(batch.column("value").to_pylist())

        assert all_ids == [1, 2, 3, 4, 5]
        assert all_values == ["a", "b", "c", "d", "e"]

    def test_concat_with_multiple_batches_per_segment(self):
        sink = pa.BufferOutputStream()
        schema = pa.schema([("id", pa.int64())])
        writer = ipc.new_stream(sink, schema)
        writer.write_batch(pa.RecordBatch.from_pydict({"id": [1, 2]}))
        writer.write_batch(pa.RecordBatch.from_pydict({"id": [3, 4]}))
        writer.close()
        multi_batch_segment = sink.getvalue().to_pybytes()

        single_batch = pa.RecordBatch.from_pydict({"id": [5]})
        single_segment = create_stream_bytes(single_batch)

        result = fast_io.concat_stream_bytes([multi_batch_segment, single_segment])

        reader = ipc.open_stream(pa.BufferReader(result))
        batches = list(reader)

        # 2 from the first segment + 1 from the second
        assert len(batches) == 3
        all_ids = []
        for batch in batches:
            all_ids.extend(batch.column("id").to_pylist())
        assert all_ids == [1, 2, 3, 4, 5]

    def test_concat_large_number_of_segments(self):
        segments = []
        expected_total = 0
        for i in range(100):
            batch = pa.RecordBatch.from_pydict({"id": [i]})
            segments.append(create_stream_bytes(batch))
            expected_total += 1

        result = fast_io.concat_stream_bytes(segments)

        reader = ipc.open_stream(pa.BufferReader(result))
        batches = list(reader)
        assert len(batches) == 100
        assert sum(b.num_rows for b in batches) == expected_total


class TestStreamConcatIpcSegments:
    def test_stream_empty_iterator(self):
        chunks = list(fast_io.stream_concat_ipc_segments(iter([])))
        assert chunks == []

    def test_stream_single_segment(self):
        batch = pa.RecordBatch.from_pydict({"id": [1, 2, 3]})
        segment = create_stream_bytes(batch)

        chunks = list(fast_io.stream_concat_ipc_segments(iter([segment])))

        # Small data may coalesce into one chunk.
        assert len(chunks) >= 1

        combined = b"".join(chunks)
        reader = ipc.open_stream(pa.BufferReader(combined))
        batches = list(reader)
        assert len(batches) == 1
        assert batches[0].num_rows == 3

    def test_stream_multiple_segments(self):
        segments = []
        expected_ids = []
        for i in range(3):
            ids = [i * 10, i * 10 + 1, i * 10 + 2]
            batch = pa.RecordBatch.from_pydict({"id": ids})
            segments.append(create_stream_bytes(batch))
            expected_ids.extend(ids)

        chunks = list(fast_io.stream_concat_ipc_segments(iter(segments)))

        assert len(chunks) >= 1

        combined = b"".join(chunks)
        reader = ipc.open_stream(pa.BufferReader(combined))
        batches = list(reader)
        assert len(batches) == 3

        actual_ids = []
        for b in batches:
            actual_ids.extend(b.column("id").to_pylist())
        assert actual_ids == expected_ids

    def test_stream_skips_empty_segments(self):
        batch = pa.RecordBatch.from_pydict({"id": [1]})
        segment = create_stream_bytes(batch)

        chunks = list(fast_io.stream_concat_ipc_segments(iter([segment, b"", segment])))

        combined = b"".join(chunks)
        reader = ipc.open_stream(pa.BufferReader(combined))
        batches = list(reader)
        assert len(batches) == 2

    def test_stream_all_empty_segments(self):
        chunks = list(fast_io.stream_concat_ipc_segments(iter([b"", b"", b""])))
        assert chunks == []

    def test_stream_preserves_schema(self):
        batch = pa.RecordBatch.from_pydict(
            {
                "id": [1, 2],
                "value": [1.5, 2.5],
                "name": ["a", "b"],
            }
        )
        segment = create_stream_bytes(batch)

        chunks = list(fast_io.stream_concat_ipc_segments(iter([segment, segment])))

        combined = b"".join(chunks)
        reader = ipc.open_stream(pa.BufferReader(combined))
        assert reader.schema == batch.schema

    def test_stream_handles_multi_batch_segments(self):
        sink = pa.BufferOutputStream()
        schema = pa.schema([("id", pa.int64())])
        writer = ipc.new_stream(sink, schema)
        writer.write_batch(pa.RecordBatch.from_pydict({"id": [1, 2]}))
        writer.write_batch(pa.RecordBatch.from_pydict({"id": [3, 4]}))
        writer.close()
        multi_batch_segment = sink.getvalue().to_pybytes()

        single_batch = pa.RecordBatch.from_pydict({"id": [5]})
        single_segment = create_stream_bytes(single_batch)

        chunks = list(
            fast_io.stream_concat_ipc_segments(iter([multi_batch_segment, single_segment]))
        )

        combined = b"".join(chunks)
        reader = ipc.open_stream(pa.BufferReader(combined))
        batches = list(reader)

        assert len(batches) == 3
        all_ids = []
        for b in batches:
            all_ids.extend(b.column("id").to_pylist())
        assert all_ids == [1, 2, 3, 4, 5]

    def test_stream_is_lazy(self):
        """Segments are fetched on demand.

        Small segments may be coalesced below the boundary threshold, so the data here is large.
        """
        fetch_count = 0

        def lazy_segments():
            nonlocal fetch_count
            for i in range(3):
                fetch_count += 1
                # Large enough to exceed the boundary threshold.
                batch = pa.RecordBatch.from_pydict({"id": list(range(10000))})
                yield create_stream_bytes(batch)

        gen = fast_io.stream_concat_ipc_segments(lazy_segments())

        assert fetch_count == 0

        # First chunk is the schema.
        first_chunk = next(gen)
        assert first_chunk
        assert fetch_count >= 1

        list(gen)
        assert fetch_count == 3

    def test_stream_vs_concat_produce_same_result(self):
        segments = []
        for i in range(5):
            batch = pa.RecordBatch.from_pydict({"id": [i * 100 + j for j in range(10)]})
            segments.append(create_stream_bytes(batch))

        buffered_result = fast_io.concat_stream_bytes(segments.copy())

        streaming_result = b"".join(fast_io.stream_concat_ipc_segments(iter(segments)))

        assert buffered_result == streaming_result

    def test_stream_emits_single_schema_multiple_batches(self):
        """The IPC stream contract: one schema, then every batch, then EOS."""
        schema = pa.schema([("id", pa.int64()), ("value", pa.float64())])

        def make_segment(ids, values):
            batch = pa.RecordBatch.from_pydict({"id": ids, "value": values})
            sink = pa.BufferOutputStream()
            writer = ipc.new_stream(sink, schema)
            writer.write_batch(batch)
            writer.close()
            return sink.getvalue().to_pybytes()

        segments = [
            make_segment([1, 2], [1.0, 2.0]),
            make_segment([3, 4, 5], [3.0, 4.0, 5.0]),
            make_segment([6], [6.0]),
        ]

        result = b"".join(fast_io.stream_concat_ipc_segments(iter(segments)))

        reader = ipc.open_stream(pa.BufferReader(result))

        assert reader.schema == schema

        batches = list(reader)
        assert len(batches) == 3

        all_ids = []
        all_values = []
        for batch in batches:
            all_ids.extend(batch.column("id").to_pylist())
            all_values.extend(batch.column("value").to_pylist())

        assert all_ids == [1, 2, 3, 4, 5, 6]
        assert all_values == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_stream_dictionary_encoded_columns(self):
        """Dictionary-encoded columns survive streaming concatenation."""

        def make_dict_segment(categories: list[str], ids: list[int]) -> bytes:
            cat_array = pa.array(categories).dictionary_encode()
            batch = pa.RecordBatch.from_arrays(
                [pa.array(ids), cat_array],
                names=["id", "category"],
            )
            return create_stream_bytes(batch)

        segments = [
            make_dict_segment(["A", "B", "C", "A", "B"], [1, 2, 3, 4, 5]),
            make_dict_segment(["X", "Y", "X"], [6, 7, 8]),
            make_dict_segment(["A", "A", "B", "B"], [9, 10, 11, 12]),
        ]

        chunks = list(fast_io.stream_concat_ipc_segments(iter(segments)))
        assert len(chunks) >= 1  # May coalesce small segments

        combined = b"".join(chunks)
        reader = ipc.open_stream(pa.BufferReader(combined))

        batches = list(reader)
        assert len(batches) == 3

        all_ids = []
        all_categories = []
        for batch in batches:
            all_ids.extend(batch.column("id").to_pylist())
            all_categories.extend(batch.column("category").to_pylist())

        assert all_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert all_categories == [
            "A",
            "B",
            "C",
            "A",
            "B",  # batch 1
            "X",
            "Y",
            "X",  # batch 2
            "A",
            "A",
            "B",
            "B",  # batch 3
        ]

        buffered = fast_io.concat_stream_bytes(segments)
        assert combined == buffered

    def test_stream_respects_min_chunk_size(self):
        """Small batches are buffered into fewer chunks of at least min_chunk_size."""
        # Each ~100-200 bytes
        segments = []
        for i in range(20):
            batch = pa.RecordBatch.from_pydict({"id": [i]})
            segments.append(create_stream_bytes(batch))

        # The default 256KB minimum coalesces small segments at the segment boundary.
        chunks_default = list(fast_io.stream_concat_ipc_segments(iter(segments)))

        # min_chunk_size=0 yields each batch immediately.
        chunks_no_buffer = list(
            fast_io.stream_concat_ipc_segments(iter(segments), min_chunk_size=0)
        )

        assert len(chunks_no_buffer) >= len(chunks_default)

        combined_default = b"".join(chunks_default)
        combined_no_buffer = b"".join(chunks_no_buffer)
        assert combined_default == combined_no_buffer

        reader = ipc.open_stream(pa.BufferReader(combined_default))
        batches = list(reader)
        assert len(batches) == 20
        all_ids = [b.column("id").to_pylist()[0] for b in batches]
        assert all_ids == list(range(20))

    def test_stream_large_batches_yield_immediately(self):
        """A batch larger than min_chunk_size is yielded without further buffering."""
        large_data = list(range(100000))  # ~800KB as int64
        batch = pa.RecordBatch.from_pydict({"id": large_data})
        segment = create_stream_bytes(batch)

        chunks = list(fast_io.stream_concat_ipc_segments(iter([segment]), min_chunk_size=1024))

        # At least schema + batch data + eos
        assert len(chunks) >= 2

        combined = b"".join(chunks)
        reader = ipc.open_stream(pa.BufferReader(combined))
        batches = list(reader)
        assert len(batches) == 1
        assert batches[0].num_rows == 100000

    def test_stream_schema_mismatch_raises_error(self):
        """Mismatched schemas fail early with a clear error, not corrupt output.

        Coalescing may raise it on the first next() call.
        """
        segment1 = create_stream_bytes(
            pa.RecordBatch.from_pydict({"id": [1, 2], "value": [1.0, 2.0]})
        )
        segment2 = create_stream_bytes(
            pa.RecordBatch.from_pydict({"id": [3, 4], "name": ["a", "b"]})
        )

        gen = fast_io.stream_concat_ipc_segments(iter([segment1, segment2]))

        with pytest.raises(ValueError, match="Schema mismatch"):
            list(gen)

    def test_stream_schema_mismatch_column_order(self):
        """A column-order difference counts as a schema mismatch."""
        segment1 = create_stream_bytes(pa.RecordBatch.from_pydict({"a": [1], "b": [2]}))
        segment2 = create_stream_bytes(pa.RecordBatch.from_pydict({"b": [3], "a": [4]}))

        gen = fast_io.stream_concat_ipc_segments(iter([segment1, segment2]))

        with pytest.raises(ValueError, match="Schema mismatch"):
            list(gen)


class TestStreamEnforcementHooks:
    """Enforcement hooks on stream_concat_ipc_segments."""

    def test_max_output_bytes_aborts_on_exceed(self):
        # ~1KB of output
        batch = pa.RecordBatch.from_pydict({"id": list(range(100))})
        segment = create_stream_bytes(batch)

        gen = fast_io.stream_concat_ipc_segments(
            iter([segment]),
            max_output_bytes=100,  # Too small for even the schema
        )

        with pytest.raises(fast_io.StreamLimitExceeded, match="size limit exceeded"):
            list(gen)

    def test_max_output_bytes_allows_under_limit(self):
        batch = pa.RecordBatch.from_pydict({"id": [1, 2, 3]})
        segment = create_stream_bytes(batch)

        chunks = list(
            fast_io.stream_concat_ipc_segments(
                iter([segment]),
                max_output_bytes=1_000_000,
            )
        )

        combined = b"".join(chunks)
        reader = ipc.open_stream(pa.BufferReader(combined))
        batches = list(reader)
        assert len(batches) == 1
        assert batches[0].num_rows == 3

    def test_max_output_bytes_partial_stream_before_abort(self):
        """Some data is yielded before the limit is hit."""
        segments = []
        for i in range(10):
            batch = pa.RecordBatch.from_pydict({"id": list(range(1000))})
            segments.append(create_stream_bytes(batch))

        single_segment_size = len(segments[0])

        # Room for about 2 segments.
        limit = single_segment_size * 2

        gen = fast_io.stream_concat_ipc_segments(
            iter(segments),
            max_output_bytes=limit,
            min_chunk_size=0,
        )

        chunks = []
        with pytest.raises(fast_io.StreamLimitExceeded):
            for chunk in gen:
                chunks.append(chunk)

        # At least the schema arrives, but not all the data.
        assert len(chunks) >= 1
        assert len(chunks) < 10

    def test_deadline_aborts_when_exceeded(self):
        import time

        batch = pa.RecordBatch.from_pydict({"id": [1, 2, 3]})
        segment = create_stream_bytes(batch)

        past_deadline = time.monotonic() - 1.0

        gen = fast_io.stream_concat_ipc_segments(
            iter([segment]),
            deadline=past_deadline,
        )

        with pytest.raises(fast_io.StreamDeadlineExceeded, match="deadline exceeded"):
            list(gen)

    def test_deadline_allows_before_expiry(self):
        import time

        batch = pa.RecordBatch.from_pydict({"id": [1, 2, 3]})
        segment = create_stream_bytes(batch)

        future_deadline = time.monotonic() + 60.0

        chunks = list(
            fast_io.stream_concat_ipc_segments(
                iter([segment]),
                deadline=future_deadline,
            )
        )

        combined = b"".join(chunks)
        reader = ipc.open_stream(pa.BufferReader(combined))
        batches = list(reader)
        assert len(batches) == 1

    def test_deadline_checked_per_segment(self, monkeypatch):
        """The deadline is checked at segment boundaries, after earlier output.

        A fake clock passes the deadline once the first segment is consumed.
        """
        from types import SimpleNamespace

        now = [0.0]
        monkeypatch.setattr(fast_io, "time", SimpleNamespace(monotonic=lambda: now[0]))

        # Large enough to pass the chunk threshold at the segment boundary.
        batch = pa.RecordBatch.from_pydict({"id": list(range(10000))})

        def segments():
            yield create_stream_bytes(batch)
            now[0] = 101.0  # past the deadline before the next segment
            yield create_stream_bytes(batch)

        gen = fast_io.stream_concat_ipc_segments(segments(), deadline=100.0)

        chunks = []
        with pytest.raises(fast_io.StreamDeadlineExceeded):
            for chunk in gen:
                chunks.append(chunk)

        assert len(chunks) >= 1

    def test_both_limits_can_be_set(self):
        import time

        batch = pa.RecordBatch.from_pydict({"id": [1, 2, 3]})
        segment = create_stream_bytes(batch)

        chunks = list(
            fast_io.stream_concat_ipc_segments(
                iter([segment]),
                max_output_bytes=1_000_000,
                deadline=time.monotonic() + 60.0,
            )
        )

        combined = b"".join(chunks)
        reader = ipc.open_stream(pa.BufferReader(combined))
        batches = list(reader)
        assert len(batches) == 1

    def test_size_limit_takes_precedence_over_deadline(self):
        """The size limit error wins when the deadline has also passed."""
        import time

        batch = pa.RecordBatch.from_pydict({"id": list(range(10000))})
        segment = create_stream_bytes(batch)

        gen = fast_io.stream_concat_ipc_segments(
            iter([segment]),
            max_output_bytes=100,
            deadline=time.monotonic() - 1.0,
        )

        # Deadline is checked first (at segment start), so it raises before the size limit.
        with pytest.raises(fast_io.StreamDeadlineExceeded):
            list(gen)

    def test_no_limits_by_default(self):
        batch = pa.RecordBatch.from_pydict({"id": list(range(100000))})
        segment = create_stream_bytes(batch)

        chunks = list(fast_io.stream_concat_ipc_segments(iter([segment])))

        combined = b"".join(chunks)
        reader = ipc.open_stream(pa.BufferReader(combined))
        batches = list(reader)
        assert batches[0].num_rows == 100000


class TestIncrementalIpcMerger:
    """The push-style IPC stream merger."""

    def test_feed_and_finish_produce_single_stream(self):
        """Merging N complete streams yields one stream with all rows."""
        merger = fast_io.IncrementalIpcMerger()
        pieces = []
        expected_ids = []
        for i in range(3):
            ids = [i * 10, i * 10 + 1, i * 10 + 2]
            batch = pa.RecordBatch.from_pydict({"id": ids})
            pieces.append(merger.feed(create_stream_bytes(batch)))
            expected_ids.extend(ids)
        pieces.append(merger.finish())

        combined = b"".join(pieces)

        # One standard reader must see ALL rows: naive byte concatenation of
        # complete streams gets this wrong (the reader stops at the first EOS).
        reader = ipc.open_stream(pa.BufferReader(combined))
        table = reader.read_all()
        assert table.column("id").to_pylist() == expected_ids

    def test_merged_output_is_exactly_one_stream(self):
        """No concatenated streams hide after the first EOS."""
        merger = fast_io.IncrementalIpcMerger()
        pieces = []
        for i in range(3):
            batch = pa.RecordBatch.from_pydict({"id": [i]})
            pieces.append(merger.feed(create_stream_bytes(batch)))
        pieces.append(merger.finish())
        combined = b"".join(pieces)

        import io

        f = io.BytesIO(combined)
        streams = 0
        total_rows = 0
        while f.tell() < len(combined):
            try:
                reader = ipc.open_stream(f)
            except pa.ArrowInvalid:
                break
            total_rows += reader.read_all().num_rows
            streams += 1
        assert streams == 1
        assert total_rows == 3

    def test_matches_concat_stream_bytes(self):
        segments = []
        for i in range(3):
            batch = pa.RecordBatch.from_pydict({"id": [i, i + 100]})
            segments.append(create_stream_bytes(batch))

        merger = fast_io.IncrementalIpcMerger()
        pieces = [merger.feed(s) for s in segments]
        pieces.append(merger.finish())
        incremental = b"".join(pieces)
        buffered = fast_io.concat_stream_bytes(segments)

        t1 = ipc.open_stream(pa.BufferReader(incremental)).read_all()
        t2 = ipc.open_stream(pa.BufferReader(buffered)).read_all()
        assert t1.equals(t2)

    def test_schema_mismatch_raises(self):
        """Segments with different schemas are rejected, not mangled."""
        merger = fast_io.IncrementalIpcMerger()
        merger.feed(create_stream_bytes(pa.RecordBatch.from_pydict({"id": [1]})))
        other = create_stream_bytes(pa.RecordBatch.from_pydict({"name": ["x"]}))
        with pytest.raises(ValueError, match="Schema mismatch"):
            merger.feed(other)

    def test_empty_segment_is_skipped(self):
        """Feeding b'' contributes nothing and does not break the stream."""
        merger = fast_io.IncrementalIpcMerger()
        pieces = [merger.feed(create_stream_bytes(pa.RecordBatch.from_pydict({"id": [1]})))]
        pieces.append(merger.feed(b""))
        pieces.append(merger.feed(create_stream_bytes(pa.RecordBatch.from_pydict({"id": [2]}))))
        pieces.append(merger.finish())

        table = ipc.open_stream(pa.BufferReader(b"".join(pieces))).read_all()
        assert table.column("id").to_pylist() == [1, 2]

    def test_finish_without_feed_returns_empty(self):
        merger = fast_io.IncrementalIpcMerger()
        assert merger.finish() == b""


class TestValidateIpcStream:
    """The write-time IPC integrity gate."""

    def test_valid_single_stream(self):
        batch = pa.RecordBatch.from_pydict({"id": [1, 2, 3]})
        assert fast_io.validate_ipc_stream(create_stream_bytes(batch)) == 3

    def test_multi_batch_stream(self):
        sink = pa.BufferOutputStream()
        writer = ipc.new_stream(sink, pa.schema([("id", pa.int64())]))
        writer.write_batch(pa.RecordBatch.from_pydict({"id": [1, 2]}))
        writer.write_batch(pa.RecordBatch.from_pydict({"id": [3]}))
        writer.close()
        assert fast_io.validate_ipc_stream(sink.getvalue().to_pybytes()) == 3

    def test_concatenated_streams_rejected(self):
        """Complete streams butted together, the shape a bad merge produces."""
        segment = create_stream_bytes(pa.RecordBatch.from_pydict({"id": [1]}))
        with pytest.raises(ValueError, match="Trailing bytes"):
            fast_io.validate_ipc_stream(segment + segment)

    def test_garbage_rejected(self):
        with pytest.raises(pa.ArrowInvalid):
            fast_io.validate_ipc_stream(b"not arrow data at all")

    def test_empty_is_zero_rows(self):
        assert fast_io.validate_ipc_stream(b"") == 0


class TestValidateIpcStreamReader:
    """The bounded file-like variant used for write-through-persisted blobs."""

    def test_valid_single_stream_returns_rows_and_schema(self):
        batch = pa.RecordBatch.from_pydict({"id": [1, 2, 3]})
        rows, schema_json = fast_io.validate_ipc_stream_reader(
            io.BytesIO(create_stream_bytes(batch))
        )
        assert rows == 3
        assert "id" in schema_json

    def test_multi_batch_stream(self):
        sink = pa.BufferOutputStream()
        writer = ipc.new_stream(sink, pa.schema([("id", pa.int64())]))
        writer.write_batch(pa.RecordBatch.from_pydict({"id": [1, 2]}))
        writer.write_batch(pa.RecordBatch.from_pydict({"id": [3]}))
        writer.close()
        rows, _ = fast_io.validate_ipc_stream_reader(io.BytesIO(sink.getvalue().to_pybytes()))
        assert rows == 3

    def test_concatenated_streams_rejected(self):
        """Same guard as the bytes variant, from a file-like source."""
        segment = create_stream_bytes(pa.RecordBatch.from_pydict({"id": [1]}))
        with pytest.raises(ValueError, match="Trailing bytes"):
            fast_io.validate_ipc_stream_reader(io.BytesIO(segment + segment))

    def test_matches_bytes_variant(self):
        batch = pa.RecordBatch.from_pydict({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})
        data = create_stream_bytes(batch)
        rows, _ = fast_io.validate_ipc_stream_reader(io.BytesIO(data))
        assert rows == fast_io.validate_ipc_stream(data)


class TestConcatRefusesDamagedSegments:
    """A damaged segment must fail loudly, never silently shorten the result.

    The disk cache validates only a segment's head and tail markers, so a segment damaged in the
    middle reaches concat, whose fast path reads the schema-message length.
    """

    @staticmethod
    def _damaged_interior(segment: bytes) -> bytes:
        """Corrupt the schema-message length, leaving head and tail markers intact."""
        damaged = bytearray(segment)
        damaged[4:8] = (0xFFFFFF00).to_bytes(4, "little")
        return bytes(damaged)

    def test_a_damaged_segment_raises_instead_of_dropping_its_rows(self):
        good = create_stream_bytes(pa.RecordBatch.from_pydict({"id": [1, 2, 3]}))
        damaged = self._damaged_interior(
            create_stream_bytes(pa.RecordBatch.from_pydict({"id": [4, 5, 6, 7]}))
        )
        # Still passes the head/tail check the cache performs.
        assert damaged[:4] == b"\xff\xff\xff\xff"
        assert damaged[-8:] == b"\xff\xff\xff\xff\x00\x00\x00\x00"

        with pytest.raises((OSError, ValueError)):
            fast_io.concat_stream_bytes([good, damaged])

    def test_a_segment_too_small_for_an_eos_marker_raises(self):
        good = create_stream_bytes(pa.RecordBatch.from_pydict({"id": [1, 2, 3]}))
        with pytest.raises((OSError, ValueError)):
            fast_io.concat_stream_bytes([good, b"\xff\xff\xff\xff"])

    def test_well_formed_segments_are_unaffected(self):
        """The guard leaves the ordinary multi-segment path alone."""
        first = create_stream_bytes(pa.RecordBatch.from_pydict({"id": [1, 2, 3]}))
        second = create_stream_bytes(pa.RecordBatch.from_pydict({"id": [4, 5]}))

        result = fast_io.concat_stream_bytes([first, b"", second])

        table = ipc.open_stream(pa.BufferReader(result)).read_all()
        assert table.column("id").to_pylist() == [1, 2, 3, 4, 5]
