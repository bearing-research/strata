"""The log ring buffer and the /v1/logs endpoints."""

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
    # The StructuredLogger kwargs path puts extra fields in the JSON entry via
    # structured_data, as production code logs.
    getattr(logger, logging.getLevelName(level).lower())(message, **fields)


class TestRingBufferLogHandler:
    def test_emit_and_read_with_cursor(self):
        buf = RingBufferLogHandler(capacity=100)
        _emit(buf, logging.INFO, "first")
        _emit(buf, logging.INFO, "second")

        result = buf.read()
        assert [e["message"] for e in result["entries"]] == ["first", "second"]
        assert result["cursor"] == 2
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
        # Only the last 3 remain; the cursor keeps counting past evictions.
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
                # since=0: the tail replays existing buffered entries immediately.
                with client.stream("GET", f"{base_url}/v1/logs/stream") as stream:
                    assert stream.headers["content-type"].startswith("text/event-stream")
                    first_data = None
                    for line in stream.iter_lines():
                        if line.startswith("data:"):
                            first_data = line
                            break
                    assert first_data is not None and first_data.startswith("data:")


@pytest.fixture
def ring_with_entry(monkeypatch):
    """A fresh server-wide ring buffer holding one known record."""
    import strata.log_buffer as log_buffer

    buf = RingBufferLogHandler(capacity=100)
    _emit(buf, logging.INFO, "tenant-a secret path /data/a")
    monkeypatch.setattr(log_buffer, "_ring_buffer", buf)
    return buf


def _logs_client(tmp_path, **config_kwargs):
    from fastapi.testclient import TestClient

    import strata.server as server_module
    from strata.server import ServerState, app

    config = StrataConfig(
        host="127.0.0.1", port=8765, cache_dir=tmp_path / "cache", **config_kwargs
    )
    original = server_module._state
    server_module._state = ServerState(config)
    try:
        yield TestClient(app)
    finally:
        server_module._state = original


@pytest.fixture
def service_logs_client(tmp_path, ring_with_entry):
    yield from _logs_client(
        tmp_path,
        deployment_mode="service",
        auth_mode="trusted_proxy",
        proxy_token="test-token",
        artifact_dir=tmp_path / "artifacts",
    )


@pytest.fixture
def personal_logs_client(tmp_path, ring_with_entry):
    yield from _logs_client(tmp_path, deployment_mode="personal")


def _principal(scopes: str) -> dict[str, str]:
    return {
        "X-Strata-Proxy-Token": "test-token",
        "X-Strata-Principal": "analyst",
        "X-Tenant-ID": "tenant-b",
        "X-Strata-Scopes": scopes,
    }


class TestLogsScopeGate:
    """The ring buffer holds every tenant's records, so principal auth requires ``admin:*``."""

    # The stream case sends a bad regex so an ungated route 400s instead of tailing forever.
    @pytest.mark.parametrize(
        "path,params", [("/v1/logs", {}), ("/v1/logs/stream", {"regex": "(unclosed"})]
    )
    def test_principal_without_admin_scope_is_refused(self, service_logs_client, path, params):
        resp = service_logs_client.get(
            path, params=params, headers=_principal("notebook:read admin:cache")
        )
        assert resp.status_code == 403
        assert "tenant-a secret" not in resp.text

    def test_admin_scope_reads_entries(self, service_logs_client):
        resp = service_logs_client.get("/v1/logs", headers=_principal("admin:*"))
        assert resp.status_code == 200
        assert [e["message"] for e in resp.json()["entries"]] == ["tenant-a secret path /data/a"]

    def test_admin_scope_passes_the_stream_gate(self, service_logs_client):
        # A bad regex 400s in the handler before the endless tail starts, so 400 means the
        # gate admitted the caller.
        resp = service_logs_client.get(
            "/v1/logs/stream", params={"regex": "(unclosed"}, headers=_principal("admin:*")
        )
        assert resp.status_code == 400

    def test_personal_mode_is_unchanged(self, personal_logs_client):
        resp = personal_logs_client.get("/v1/logs")
        assert resp.status_code == 200
        assert [e["message"] for e in resp.json()["entries"]] == ["tenant-a secret path /data/a"]
