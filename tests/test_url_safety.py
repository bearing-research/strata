"""A guarded client connects only to an address the guard validated.

Resolving once to check and again to connect lets DNS rebinding reach 127.0.0.1. A recording backend
stands in for sockets, so tests see the exact host and port opened.
"""

from __future__ import annotations

import ssl
from types import SimpleNamespace

import httpcore
import httpx
import pytest
from httpcore._backends.auto import AutoBackend

from strata.url_safety import guarded_async_transport, guarded_transport

_PUBLIC = "93.184.216.34"
_PUBLIC_V6 = "2606:2800:220:1:248:1893:25c8:1946"
_RESPONSE = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"


class _Stream(httpcore.NetworkStream):
    """A connection that answers one canned response and records what it saw."""

    def __init__(self, record: dict):
        self._record = record
        self._unread = _RESPONSE

    def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        chunk, self._unread = self._unread[:max_bytes], self._unread[max_bytes:]
        return chunk

    def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._record.setdefault("sent", b"")
        self._record["sent"] += buffer

    def close(self) -> None:
        self._record["closed"] = True

    def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        self._record["ssl_context"] = ssl_context
        self._record["server_hostname"] = server_hostname
        return self

    def get_extra_info(self, info: str):
        return None


class _AsyncStream(httpcore.AsyncNetworkStream):
    def __init__(self, record: dict):
        self._sync = _Stream(record)

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        return self._sync.read(max_bytes, timeout)

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self._sync.write(buffer, timeout)

    async def aclose(self) -> None:
        self._sync.close()

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        self._sync.start_tls(ssl_context, server_hostname, timeout)
        return self

    def get_extra_info(self, info: str):
        return None


@pytest.fixture
def sockets(monkeypatch):
    """Every TCP connection httpcore opens, as ``(host, port)``; ``refuse`` hosts fail."""

    recorded = SimpleNamespace(connects=[], refuse=set(), record={})

    def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        recorded.connects.append((host, port))
        if host in recorded.refuse:
            raise httpcore.ConnectError(f"{host} unreachable")
        return _Stream(recorded.record)

    async def connect_tcp_async(
        self, host, port, timeout=None, local_address=None, socket_options=None
    ):
        recorded.connects.append((host, port))
        if host in recorded.refuse:
            raise httpcore.ConnectError(f"{host} unreachable")
        return _AsyncStream(recorded.record)

    monkeypatch.setattr(httpcore.SyncBackend, "connect_tcp", connect_tcp)
    monkeypatch.setattr(AutoBackend, "connect_tcp", connect_tcp_async)
    return recorded


def _get(url: str, **guard) -> httpx.Response:
    with httpx.Client(transport=guarded_transport(**guard)) as client:
        return client.get(url)


class TestPinning:
    def test_the_connection_goes_to_the_address_the_check_validated(self, rebinding_dns, sockets):
        """The name is resolved once and the socket opens to that address, not the name."""
        rebinding_dns.answers["rebind.test"] = [[_PUBLIC], ["127.0.0.1"]]

        response = _get("http://rebind.test:8080/data")

        assert response.status_code == 200
        assert sockets.connects == [(_PUBLIC, 8080)]
        assert rebinding_dns.lookups == ["rebind.test"]

    async def test_an_async_client_is_pinned_the_same_way(self, rebinding_dns, sockets):
        rebinding_dns.answers["rebind.test"] = [[_PUBLIC], ["127.0.0.1"]]

        async with httpx.AsyncClient(transport=guarded_async_transport()) as client:
            response = await client.get("http://rebind.test:8080/data")

        assert response.status_code == 200
        assert sockets.connects == [(_PUBLIC, 8080)]
        assert rebinding_dns.lookups == ["rebind.test"]

    @pytest.mark.parametrize("address", ["127.0.0.1", "169.254.169.254", "10.0.0.1", "::1"])
    def test_a_name_that_resolves_to_a_private_address_is_not_connected(
        self, rebinding_dns, sockets, address
    ):
        rebinding_dns.answers["rebind.test"] = [[address]]

        with pytest.raises(httpx.ConnectError, match="non-routable"):
            _get("http://rebind.test/data")

        assert sockets.connects == []

    async def test_an_async_client_refuses_a_private_address(self, rebinding_dns, sockets):
        rebinding_dns.answers["rebind.test"] = [["169.254.169.254"]]

        async with httpx.AsyncClient(transport=guarded_async_transport()) as client:
            with pytest.raises(httpx.ConnectError, match="non-routable"):
                await client.get("http://rebind.test/latest/meta-data/")

        assert sockets.connects == []

    def test_one_private_address_among_public_ones_refuses_the_name(self, rebinding_dns, sockets):
        rebinding_dns.answers["rebind.test"] = [[_PUBLIC, "127.0.0.1"]]

        with pytest.raises(httpx.ConnectError, match="127.0.0.1"):
            _get("http://rebind.test/data")

        assert sockets.connects == []

    def test_an_unreachable_address_falls_through_to_the_next_validated_one(
        self, rebinding_dns, sockets
    ):
        """A host whose IPv6 address has no route still connects over IPv4."""
        rebinding_dns.answers["dual.test"] = [[_PUBLIC_V6, _PUBLIC]]
        sockets.refuse.add(_PUBLIC_V6)

        response = _get("http://dual.test/data")

        assert response.status_code == 200
        assert sockets.connects == [(_PUBLIC_V6, 80), (_PUBLIC, 80)]

    def test_an_allowlisted_name_is_connected_by_name(self, rebinding_dns, sockets):
        """Trusted by name, so not resolved or checked here."""
        rebinding_dns.answers["build.internal"] = [["10.0.0.5"]]  # refused, were it checked

        response = _get("http://build.internal:8000/finalize", allowed_hosts=("build.internal",))

        assert response.status_code == 200
        assert sockets.connects == [("build.internal", 8000)]
        assert rebinding_dns.lookups == []

    def test_allow_local_leaves_the_client_at_httpx_defaults(self):
        assert guarded_transport(allow_local=True) is None
        assert guarded_async_transport(allow_local=True) is None


class TestTheRequestStillNamesTheHost:
    def test_https_verifies_the_certificate_against_the_hostname(self, rebinding_dns, sockets):
        rebinding_dns.answers["files.example.org"] = [[_PUBLIC]]

        _get("https://files.example.org/data.csv")

        assert sockets.connects == [(_PUBLIC, 443)]
        assert sockets.record["server_hostname"] == "files.example.org"
        context = sockets.record["ssl_context"]
        assert context.verify_mode == ssl.CERT_REQUIRED
        assert context.check_hostname is True

    def test_plain_http_sends_the_original_host_header(self, rebinding_dns, sockets):
        rebinding_dns.answers["files.example.org"] = [[_PUBLIC]]

        _get("http://files.example.org/data.csv")

        assert sockets.connects == [(_PUBLIC, 80)]
        head = sockets.record["sent"].split(b"\r\n")
        assert head[0] == b"GET /data.csv HTTP/1.1"
        assert b"Host: files.example.org" in head


class TestProxies:
    def test_a_guarded_client_ignores_the_environments_proxy(
        self, rebinding_dns, sockets, monkeypatch
    ):
        """A proxy resolves the name itself, so the check would cover an unused address."""
        monkeypatch.setenv("HTTP_PROXY", "http://proxy.test:3128")
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy.test:3128")
        rebinding_dns.answers["files.example.org"] = [[_PUBLIC]]

        _get("http://files.example.org/data.csv")

        assert sockets.connects == [(_PUBLIC, 80)]
