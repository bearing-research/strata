"""Standalone tests for the ``strata-client`` distribution.

These run with ONLY the client's own dependencies installed (httpx + pyarrow) —
no ``strata`` / server stack. The client-only CI job runs this file in such an
environment to guard the slim-install promise. They also pass in the full dev
env. See docs/internal/design-strata-client.md.
"""

from __future__ import annotations

import sys

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
from strata_client import Filter, FilterOp, RetryConfig, StrataClient, gt


def test_no_server_deps_imported() -> None:
    """Importing the client must not pull the server's heavy stack."""
    for heavy in ("strata", "pyiceberg", "fastapi", "uvicorn", "duckdb", "pydantic"):
        assert heavy not in sys.modules, f"strata_client pulled {heavy}"


def test_default_url_resolution(monkeypatch) -> None:
    monkeypatch.delenv("STRATA_SERVER_URL", raising=False)
    monkeypatch.delenv("STRATA_HOST", raising=False)
    monkeypatch.delenv("STRATA_PORT", raising=False)
    client = StrataClient()
    assert client.config is None
    assert client.base_url == "http://127.0.0.1:8765"
    client.close()


def test_server_url_env_override(monkeypatch) -> None:
    monkeypatch.setenv("STRATA_SERVER_URL", "https://strata.example.com")
    client = StrataClient()
    assert client.base_url == "https://strata.example.com"
    client.close()


def test_filter_constructors() -> None:
    f = gt("amount", 100)
    assert isinstance(f, Filter)
    assert f.column == "amount"
    assert f.op is FilterOp.GT
    assert f.value == 100


def test_fetch_over_mock_transport() -> None:
    """Arrow IPC fetch decodes correctly against a mocked server (no network).

    The full materialize protocol is covered in test_client_unit.py; this just
    confirms the standalone client can do an Arrow round-trip end to end.
    """
    table = pa.table({"x": [1, 2, 3]})
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    ipc_bytes = sink.getvalue().to_pybytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/data"):
            return httpx.Response(
                200,
                content=ipc_bytes,
                headers={"content-type": "application/vnd.apache.arrow.stream"},
            )
        # Artifact status probe — report ready so fetch proceeds to /data.
        return httpx.Response(200, json={"artifact_id": "abc123", "version": 1, "state": "ready"})

    client = StrataClient.from_transport(httpx.MockTransport(handler))
    fetched = client.fetch("strata://artifact/abc123@v=1")
    assert fetched.to_pydict() == {"x": [1, 2, 3]}
    client.close()


def test_retry_config_backoff() -> None:
    rc = RetryConfig(max_retries=2, base_delay=1.0, max_delay=30.0, jitter=0.0)
    assert rc.calculate_delay(0) == 1.0
    assert rc.calculate_delay(1) == 2.0
    assert rc.calculate_delay(10) == 30.0  # capped
