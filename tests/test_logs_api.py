"""Tests for the log ring buffer + /v1/logs endpoints (observability, B1)."""

import logging

import httpx
import pytest

from strata.config import StrataConfig
from strata.log_buffer import RingBufferLogHandler
from tests.conftest import find_free_port, run_server


def _emit(handler: RingBufferLogHandler, level: int, message: str, **fields) -> None:
    logger = logging.getLogger("test.log_buffer")
    if handler not in logger.handlers:
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    # Use the StructuredLogger kwargs path (info/warning/…) so extra fields land
    # in the JSON entry via structured_data — how production code logs.
    getattr(logger, logging.getLevelName(level).lower())(message, **fields)


class TestRingBufferLogHandler:
    def test_emit_and_read_with_cursor(self):
        buf = RingBufferLogHandler(capacity=100)
        _emit(buf, logging.INFO, "first")
        _emit(buf, logging.INFO, "second")

        result = buf.read()
        assert [e["message"] for e in result["entries"]] == ["first", "second"]
        assert result["cursor"] == 2
        # Each entry is cursor-tagged, monotonically increasing.
        assert [e["cursor"] for e in result["entries"]] == [1, 2]

    def test_since_pages_forward(self):
        buf = RingBufferLogHandler(capacity=100)
        _emit(buf, logging.INFO, "a")
        _emit(buf, logging.INFO, "b")
        result = buf.read(since=1)
        assert [e["message"] for e in result["entries"]] == ["b"]

    def test_level_filter_is_minimum_severity(self):
        buf = RingBufferLogHandler(capacity=100)
        _emit(buf, logging.INFO, "info-msg")
        _emit(buf, logging.WARNING, "warn-msg")
        _emit(buf, logging.ERROR, "error-msg")
        msgs = [e["message"] for e in buf.read(level="warning")["entries"]]
        assert msgs == ["warn-msg", "error-msg"]

    def test_regex_filter_matches_message(self):
        buf = RingBufferLogHandler(capacity=100)
        _emit(buf, logging.INFO, "cache miss for table events")
        _emit(buf, logging.INFO, "cache hit for table users")
        msgs = [e["message"] for e in buf.read(regex="miss")["entries"]]
        assert msgs == ["cache miss for table events"]

    def test_notebook_filter_matches_field(self):
        buf = RingBufferLogHandler(capacity=100)
        _emit(buf, logging.INFO, "nb-scoped", notebook_id="nb-123")
        _emit(buf, logging.INFO, "other")
        msgs = [e["message"] for e in buf.read(notebook="nb-123")["entries"]]
        assert msgs == ["nb-scoped"]

    def test_bad_regex_raises(self):
        buf = RingBufferLogHandler(capacity=100)
        _emit(buf, logging.INFO, "x")
        with pytest.raises(Exception):  # re.error
            buf.read(regex="(unclosed")

    def test_capacity_evicts_oldest(self):
        buf = RingBufferLogHandler(capacity=3)
        for i in range(5):
            _emit(buf, logging.INFO, f"m{i}")
        result = buf.read()
        # Only the last 3 remain; cursor keeps counting past evictions.
        assert [e["message"] for e in result["entries"]] == ["m2", "m3", "m4"]
        assert result["cursor"] == 5


class TestLogsEndpoints:
    def test_get_logs_returns_recent_entries(self, tmp_path):
        port = find_free_port()
        config = StrataConfig(host="127.0.0.1", port=port, cache_dir=tmp_path / "cache")
        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                data = client.get(f"{base_url}/v1/logs").json()
                # The server logs on startup, so the buffer is non-empty.
                assert data["cursor"] > 0
                assert len(data["entries"]) > 0
                entry = data["entries"][0]
                assert {"level", "logger", "message", "cursor"} <= set(entry)
                # Paging: since=<latest> returns nothing new.
                assert (
                    client.get(f"{base_url}/v1/logs?since={data['cursor']}").json()["entries"] == []
                )

    def test_get_logs_bad_regex_is_400(self, tmp_path):
        port = find_free_port()
        config = StrataConfig(host="127.0.0.1", port=port, cache_dir=tmp_path / "cache")
        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(f"{base_url}/v1/logs", params={"regex": "(unclosed"})
                assert resp.status_code == 400

    def test_stream_replays_buffered_entries_as_sse(self, tmp_path):
        port = find_free_port()
        config = StrataConfig(host="127.0.0.1", port=port, cache_dir=tmp_path / "cache")
        with run_server(config) as base_url:
            with httpx.Client(timeout=10.0) as client:
                # since=0 → the tail replays existing buffered entries immediately.
                with client.stream("GET", f"{base_url}/v1/logs/stream") as stream:
                    assert stream.headers["content-type"].startswith("text/event-stream")
                    first_data = None
                    for line in stream.iter_lines():
                        if line.startswith("data:"):
                            first_data = line
                            break
                    assert first_data is not None and first_data.startswith("data:")
