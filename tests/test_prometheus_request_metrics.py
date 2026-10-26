"""``/metrics/prometheus`` counts requests by route template and status class, times them,
and counts failed object-store calls.

The series are process globals, so each test reads them before and after and asserts the
difference.
"""

from __future__ import annotations

import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient
from vestigo.blob_store import S3BlobStore


@pytest.fixture
def client(tmp_path):
    from vestigo.store import reset_artifact_store

    import strata.server as server_module
    from strata.config import StrataConfig
    from strata.server import ServerState, app

    original = server_module._state
    server_module._state = ServerState(
        StrataConfig(
            host="127.0.0.1",
            cache_dir=tmp_path / "cache",
            artifact_dir=tmp_path / "artifacts",
            metadata_db=tmp_path / "meta.sqlite",
        )
    )
    reset_artifact_store()
    try:
        # An unhandled exception is answered 500 rather than raised into the test.
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        server_module._state = original
        reset_artifact_store()


def _value(body: str, series: str) -> float:
    """The sample value of *series* (name plus labels) in an exposition, 0 when absent."""
    match = re.search(rf"^{re.escape(series)} (\S+)$", body, re.MULTILINE)
    return float(match.group(1)) if match else 0.0


def _scrape(client: TestClient) -> str:
    response = client.get("/metrics/prometheus")
    assert response.status_code == 200, response.text
    return response.text


def _requests(method: str, route: str, status: str) -> str:
    return f'strata_http_requests_total{{method="{method}",route="{route}",status="{status}"}}'


TABLE_ROUTE = "/metrics/tables/{table_id}"


def test_a_500_and_a_200_count_under_their_route_templates(client, monkeypatch):
    import strata.server as server_module

    before = _scrape(client)

    def _broken(_table_id):
        raise RuntimeError("metrics backend fell over")

    monkeypatch.setattr(server_module._state.metrics, "get_table_metrics", _broken)
    assert client.get("/metrics/tables/acme.sales.orders").status_code == 500
    assert client.get("/health").status_code == 200

    after = _scrape(client)
    for series in (_requests("GET", TABLE_ROUTE, "5xx"), _requests("GET", "/health", "2xx")):
        assert _value(after, series) - _value(before, series) == 1, series
    # The 500 is not also counted as a success.
    assert _value(after, _requests("GET", TABLE_ROUTE, "2xx")) == _value(
        before, _requests("GET", TABLE_ROUTE, "2xx")
    )
    # The template, never the concrete path, is the label.
    assert "acme.sales.orders" not in after


def test_a_handled_404_counts_as_4xx(client):
    before = _scrape(client)
    assert client.get("/metrics/tables/no.such.table").status_code == 404
    after = _scrape(client)

    series = _requests("GET", TABLE_ROUTE, "4xx")
    assert _value(after, series) - _value(before, series) == 1


def test_a_request_answered_before_routing_is_labelled_unmatched(client):
    from strata.server import app

    before = _scrape(client)
    # A Host the server does not answer to is refused before any route is chosen.
    refused = TestClient(app, base_url="http://rebound.example").get("/some/raw/path-7f3a")
    assert refused.status_code == 400
    after = _scrape(client)

    series = _requests("GET", "unmatched", "4xx")
    assert _value(after, series) - _value(before, series) == 1
    assert "path-7f3a" not in after


def test_a_route_outside_the_api_routers_keeps_its_template(client):
    """FastAPI's own docs page is a plain Starlette route, which leaves no route in the scope."""
    before = _scrape(client)
    assert client.get("/docs").status_code == 200
    after = _scrape(client)

    series = _requests("GET", "/docs", "2xx")
    assert _value(after, series) - _value(before, series) == 1


def test_request_duration_is_a_histogram_per_route(client):
    before = _scrape(client)
    assert client.get("/health").status_code == 200
    after = _scrape(client)

    assert "# TYPE strata_http_request_duration_seconds histogram" in after
    count = 'strata_http_request_duration_seconds_count{route="/health"}'
    inf = 'strata_http_request_duration_seconds_bucket{route="/health",le="+Inf"}'
    assert _value(after, count) - _value(before, count) == 1
    assert _value(after, inf) == _value(after, count)
    assert 'strata_http_request_duration_seconds_sum{route="/health"}' in after
    buckets = re.findall(
        r'^strata_http_request_duration_seconds_bucket\{route="/health",le="([^"]+)"\} (\S+)$',
        after,
        re.MULTILINE,
    )
    counts = [float(value) for _, value in buckets]
    assert counts == sorted(counts), "buckets must be cumulative"


class _RefusingBucket(BaseHTTPRequestHandler):
    """An S3 endpoint that refuses every call, and answers a key named ``missing`` with 404."""

    def _answer(self) -> None:
        length = int(self.headers.get("content-length") or 0)
        if length:
            self.rfile.read(length)
        missing = "missing" in self.path
        if missing and "list-type=2" in self.path:
            # Arrow lists the key as a directory after a 404; nothing is under it either.
            status, body = 200, b"<ListBucketResult><KeyCount>0</KeyCount></ListBucketResult>"
        else:
            status, code = (404, "NoSuchKey") if missing else (403, "AccessDenied")
            body = f"<Error><Code>{code}</Code></Error>".encode()
        if self.command == "HEAD":
            body = b""
        self.send_response(status)
        self.send_header("Content-Type", "application/xml")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_PUT = do_POST = do_HEAD = do_DELETE = _answer

    def log_message(self, format, *args):  # noqa: A002
        return


@pytest.fixture
def refusing_s3():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RefusingBucket)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield S3BlobStore(
            bucket="refusing",
            region="us-east-1",
            endpoint_url=f"http://127.0.0.1:{server.server_address[1]}",
            access_key="key",
            secret_key="secret",
        )
    finally:
        server.shutdown()
        server.server_close()


def _store_errors(operation: str) -> str:
    return f'strata_object_store_errors_total{{operation="{operation}"}}'


def test_a_put_to_a_refusing_bucket_counts_one_put_error(client, refusing_s3):
    before = _scrape(client)
    with pytest.raises(OSError):
        refusing_s3.write_blob("art", 1, b"payload")
    after = _scrape(client)

    assert _value(after, _store_errors("put")) - _value(before, _store_errors("put")) == 1


def test_a_refused_head_propagates_and_counts(client, refusing_s3):
    """Only a missing object is absent; a refusal reaches the caller and the scrape."""
    before = _scrape(client)
    with pytest.raises(OSError, match="ACCESS_DENIED"):
        refusing_s3.blob_exists("art", 1)
    after = _scrape(client)

    assert _value(after, _store_errors("head")) - _value(before, _store_errors("head")) == 1


def test_an_absent_object_is_not_an_error(client, refusing_s3):
    before = _scrape(client)
    assert refusing_s3.open_blob_reader("missing", 1) is None
    assert refusing_s3.blob_size("missing", 1) is None
    after = _scrape(client)

    for operation in ("get", "head"):
        assert _value(after, _store_errors(operation)) == _value(
            before, _store_errors(operation)
        ), operation


def test_every_operation_has_a_series_from_the_start(client):
    body = _scrape(client)
    for operation in ("put", "get", "head", "list", "delete"):
        assert _store_errors(operation) in body
