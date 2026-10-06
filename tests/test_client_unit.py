"""Unit tests for ``strata.client`` against a mocked HTTP transport.

Covers the error, retry and parsing paths the live-server integration tests do not reach.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from strata_client.client import (
    Artifact,
    RetryConfig,
    StrataClient,
    _parse_artifact_uri,
)

# --- Helpers ---


def _make_client(handler) -> StrataClient:
    """Build a StrataClient whose requests go to ``handler``."""
    return StrataClient.from_transport(httpx.MockTransport(handler))


def _arrow_ipc_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


# --- RetryConfig ---


class TestRetryConfig:
    """``RetryConfig.calculate_delay``: pure math, no I/O."""

    def test_defaults_are_set(self):
        config = RetryConfig()
        assert config.max_retries >= 0
        assert config.base_delay > 0
        assert config.max_delay >= config.base_delay
        assert config.jitter >= 0

    def test_zeroth_attempt_uses_base_delay(self):
        config = RetryConfig(base_delay=1.0, max_delay=60.0, jitter=0.0)
        assert config.calculate_delay(0) == 1.0

    def test_exponential_growth(self):
        config = RetryConfig(base_delay=1.0, max_delay=1000.0, jitter=0.0)
        # base * 2^attempt; no jitter, for determinism
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(2) == 4.0
        assert config.calculate_delay(3) == 8.0

    def test_max_delay_caps_growth(self):
        config = RetryConfig(base_delay=1.0, max_delay=5.0, jitter=0.0)
        # 2^10 = 1024, but the cap is 5.0
        assert config.calculate_delay(10) == 5.0

    def test_jitter_adds_within_bounds(self):
        config = RetryConfig(base_delay=1.0, max_delay=1000.0, jitter=0.5)
        for _ in range(20):
            delay = config.calculate_delay(0)
            assert 1.0 <= delay <= 1.5


# --- Artifact: properties and URI parsing ---


class TestArtifactProperties:
    """Properties on ``Artifact``: pure formatting, no I/O."""

    def _artifact(self, **kwargs: Any) -> Artifact:
        defaults = {"_client": None, "artifact_id": "abc123", "version": 7}
        return Artifact(**{**defaults, **kwargs})

    def test_uri_format(self):
        assert self._artifact().uri == "strata://artifact/abc123@v=7"

    def test_name_uri_when_named(self):
        assert self._artifact(name="daily_sales").name_uri == "strata://name/daily_sales"

    def test_name_uri_without_name(self):
        assert self._artifact().name_uri is None


class TestParseArtifactUri:
    """``_parse_artifact_uri`` raises ValueError on invalid input."""

    def test_valid_uri(self):
        assert _parse_artifact_uri("strata://artifact/abc@v=3") == ("abc", 3)

    def test_uri_with_complex_id(self):
        # Artifact IDs can contain underscores, dashes, dots.
        assert _parse_artifact_uri("strata://artifact/nb_42_cell_x@v=12") == (
            "nb_42_cell_x",
            12,
        )

    def test_invalid_uri_raises(self):
        with pytest.raises(ValueError, match="Invalid artifact URI"):
            _parse_artifact_uri("http://example.com/foo")

    def test_missing_version_raises(self):
        with pytest.raises(ValueError, match="Invalid artifact URI"):
            _parse_artifact_uri("strata://artifact/abc")

    def test_non_numeric_version_raises(self):
        with pytest.raises(ValueError, match="Invalid artifact URI"):
            _parse_artifact_uri("strata://artifact/abc@v=x")


# --- Artifact: HTTP-backed accessors (info / lineage / dependents) ---


class TestArtifactHttpAccessors:
    """``info`` / ``lineage`` / ``dependents`` call the server."""

    def test_info_returns_decoded_json(self):
        captured: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(str(request.url))
            return httpx.Response(200, json={"state": "ready", "size_bytes": 1234})

        client = _make_client(handler)
        artifact = Artifact(_client=client, artifact_id="abc", version=2)
        info = artifact.info()
        assert info == {"state": "ready", "size_bytes": 1234}
        assert "/v1/artifacts/abc/v/2" in captured[0]

    def test_lineage_sends_only_what_the_route_reads(self):
        """The route walks upstream only, so a ``direction`` parameter was silently ignored."""
        captured: list[httpx.URL] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request.url)
            return httpx.Response(200, json={"nodes": [], "edges": []})

        client = _make_client(handler)
        artifact = Artifact(_client=client, artifact_id="abc", version=2)
        artifact.lineage(max_depth=5)
        url = captured[0]
        assert "/v1/artifacts/abc/v/2/lineage" in str(url)
        assert dict(url.params) == {"max_depth": "5"}

    def test_dependents_passes_the_limit_the_route_reads(self):
        """Dependents are one hop and capped by ``limit``; ``max_depth`` was silently ignored."""
        captured: list[httpx.URL] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request.url)
            return httpx.Response(200, json={"dependents": []})

        client = _make_client(handler)
        artifact = Artifact(_client=client, artifact_id="abc", version=2)
        artifact.dependents(limit=3)
        url = captured[0]
        assert "/v1/artifacts/abc/v/2/dependents" in str(url)
        assert dict(url.params) == {"limit": "3"}


# --- Artifact: to_table / to_pandas / to_polars via cached stream ---


class TestArtifactToTable:
    """``Artifact.to_table`` and friends on the stream-cached path."""

    def test_to_table_from_empty_stream(self):
        artifact = Artifact(_client=None, artifact_id="abc", version=1, _stream_data=b"")
        result = artifact.to_table()
        assert isinstance(result, pa.Table)
        assert result.num_rows == 0
        assert result.num_columns == 0

    def test_to_table_from_stream_bytes(self):
        table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        artifact = Artifact(
            _client=None,
            artifact_id="abc",
            version=1,
            _stream_data=_arrow_ipc_bytes(table),
        )
        result = artifact.to_table()
        assert result.num_rows == 3
        assert result.column("x").to_pylist() == [1, 2, 3]

    def test_to_pandas_from_stream(self):
        table = pa.table({"x": [10, 20]})
        artifact = Artifact(
            _client=None,
            artifact_id="abc",
            version=1,
            _stream_data=_arrow_ipc_bytes(table),
        )
        df = artifact.to_pandas()
        assert list(df["x"]) == [10, 20]

    def test_to_polars_from_stream(self):
        table = pa.table({"x": [100]})
        artifact = Artifact(
            _client=None,
            artifact_id="abc",
            version=1,
            _stream_data=_arrow_ipc_bytes(table),
        )
        polars_df = artifact.to_polars()
        assert polars_df.shape == (1, 1)


# --- StrataClient: top-level endpoints ---


class TestSimpleEndpoints:
    """``health`` / ``metrics`` / ``clear_cache``."""

    def test_health(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/health"
            return httpx.Response(200, json={"status": "healthy"})

        client = _make_client(handler)
        assert client.health() == {"status": "healthy"}

    def test_metrics(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/metrics"
            return httpx.Response(200, json={"counter": 42})

        client = _make_client(handler)
        assert client.metrics() == {"counter": 42}

    def test_clear_cache_posts(self):
        captured: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request.method)
            return httpx.Response(200, json={"cleared": True})

        client = _make_client(handler)
        result = client.clear_cache()
        assert captured == ["POST"]
        assert result == {"cleared": True}


# --- _fetch_stream_with_retry: 429 backoff ---


class TestFetchStreamRetry:
    """``_fetch_stream_with_retry``: 429, backoff, then success."""

    def test_returns_content_on_first_success(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"payload")

        client = _make_client(handler)
        client.retry_config = RetryConfig(max_retries=3, base_delay=0.0, jitter=0.0)
        assert client._fetch_stream_with_retry("/streams/abc").content == b"payload"

    def test_retries_on_429_then_succeeds(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda _: None)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] < 3:
                return httpx.Response(429, headers={"Retry-After": "0"})
            return httpx.Response(200, content=b"after-retry")

        client = _make_client(handler)
        client.retry_config = RetryConfig(max_retries=5, base_delay=0.0, jitter=0.0)
        assert client._fetch_stream_with_retry("/streams/abc").content == b"after-retry"
        assert calls["n"] == 3

    def test_max_retries_exceeded_raises(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda _: None)

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, headers={"Retry-After": "0"})

        client = _make_client(handler)
        client.retry_config = RetryConfig(max_retries=2, base_delay=0.0, jitter=0.0)
        with pytest.raises(httpx.HTTPStatusError):
            client._fetch_stream_with_retry("/streams/abc")

    def test_malformed_retry_after_falls_back_to_calculated_delay(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda _: None)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, headers={"Retry-After": "not-a-number"})
            return httpx.Response(200, content=b"ok")

        client = _make_client(handler)
        client.retry_config = RetryConfig(max_retries=3, base_delay=0.0, jitter=0.0)
        assert client._fetch_stream_with_retry("/streams/abc").content == b"ok"


# --- _fetch_artifact_data_with_wait: state machine ---


class TestFetchWithWait:
    """``_fetch_artifact_data_with_wait``: building, failed and unknown states."""

    def test_failed_state_raises_with_message(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"state": "failed", "error_message": "build broke"},
            )

        client = _make_client(handler)
        with pytest.raises(RuntimeError, match="build broke"):
            client._fetch_artifact_data_with_wait("abc", 1, timeout=10.0)

    def test_building_then_ready_eventually_fetches(self, monkeypatch):
        monkeypatch.setattr("time.sleep", lambda _: None)
        calls = {"status": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/v1/artifacts/abc/v/1"):
                calls["status"] += 1
                state = "building" if calls["status"] < 3 else "ready"
                return httpx.Response(200, json={"state": state})
            # Anything else is the data fetch; return an empty IPC stream.
            return httpx.Response(
                200,
                json={"stream_url": "/v1/streams/x"},
            )

        client = _make_client(handler)
        # Patch the data path to avoid pulling Arrow over the mock.
        monkeypatch.setattr(
            client,
            "_fetch_artifact_data",
            lambda _id, _v: pa.table({"x": [1]}),
        )
        result = client._fetch_artifact_data_with_wait("abc", 1, timeout=30.0)
        assert result.num_rows == 1
        assert calls["status"] >= 3

    def test_unknown_state_falls_back_to_direct_fetch(self, monkeypatch):
        """An unrecognized state falls through to a fetch rather than looping forever."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"state": "some-new-state"})

        client = _make_client(handler)
        monkeypatch.setattr(
            client,
            "_fetch_artifact_data",
            lambda _id, _v: pa.table({"x": [42]}),
        )
        result = client._fetch_artifact_data_with_wait("abc", 1, timeout=5.0)
        assert result.column("x").to_pylist() == [42]

    def test_timeout_during_building_raises(self, monkeypatch):
        # ``time.time`` advances faster than ``time.sleep`` here, so the timeout
        # trips on the second poll.
        clock = [0.0]
        monkeypatch.setattr("time.sleep", lambda _: clock.__setitem__(0, clock[0] + 100))
        monkeypatch.setattr("time.time", lambda: clock[0])

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"state": "building"})

        client = _make_client(handler)
        with pytest.raises(TimeoutError, match="timed out"):
            client._fetch_artifact_data_with_wait("abc", 1, timeout=1.0)


# --- materialize: request shaping ---


class TestMaterializeRequestShape:
    """``materialize`` request body: ``ref`` sent as ``executor``, optional flags, cache hits."""

    def test_ref_is_renamed_to_executor(self):
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "artifact_uri": "strata://artifact/abc@v=1",
                    "hit": True,
                    "state": "ready",
                },
            )

        client = _make_client(handler)
        client.materialize(
            inputs=["file:///x"],
            transform={"ref": "scan@v1", "params": {}},
        )
        body = captured[0]
        # ``ref`` is the public spelling; the server only accepts ``executor``.
        assert "ref" not in body["transform"]
        assert body["transform"]["executor"] == "scan@v1"

    def test_name_and_refresh_round_trip(self):
        captured: list[dict[str, Any]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(json.loads(request.content))
            return httpx.Response(
                200,
                json={
                    "artifact_uri": "strata://artifact/abc@v=1",
                    "hit": False,
                    "state": "ready",
                },
            )

        client = _make_client(handler)
        client.materialize(
            inputs=["file:///x"],
            transform={"executor": "scan@v1", "params": {}},
            name="my_artifact",
            refresh=True,
        )
        body = captured[0]
        assert body["name"] == "my_artifact"
        assert body["refresh"] is True

    def test_cache_hit_returns_artifact_with_hit_flag(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "artifact_uri": "strata://artifact/abc@v=1",
                    "hit": True,
                    "state": "ready",
                },
            )

        client = _make_client(handler)
        artifact = client.materialize(
            inputs=["file:///x"],
            transform={"executor": "scan@v1", "params": {}},
            mode="artifact",  # no stream fetch path
        )
        assert artifact.cache_hit is True
        assert artifact.execution == "cache"
        assert artifact.artifact_id == "abc"
        assert artifact.version == 1
