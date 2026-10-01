"""Whether a URL is safe for this process to fetch.

Shared by the worker (URLs a manifest hands it) and a notebook's ``@fetch`` (the URL a
cell names): both are untrusted requests aimed at whatever network this process reaches.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlparse

import httpcore
import httpx
from httpcore._backends.auto import AutoBackend

logger = logging.getLogger(__name__)

ALLOWED_URL_SCHEMES = frozenset({"http", "https"})


def web_url_or_none(value: str) -> str | None:
    """*value* if it is an http(s) URL, else ``None``.

    Scheme only, no network: this answers "may I render this as a link", which must not depend
    on DNS. Blocks ``javascript:`` values, which survive HTML escaping; a leading-"http"
    test would also pass ``httpfoo://x``.
    """
    try:
        scheme = urlparse(value).scheme.lower()
    except ValueError:
        return None
    return value if scheme in ALLOWED_URL_SCHEMES else None


def host_is_allowlisted(host: str, allowed_hosts: tuple[str, ...]) -> bool:
    """Whether *host* is named in the allowlist (by name, never by resolved address).

    Suffixes are anchored on a dot: ``.example.com`` matches ``build.example.com``, not
    ``evil-example.com``.
    """
    candidate = host.lower().rstrip(".")
    for entry in allowed_hosts:
        if entry.startswith("."):
            if candidate.endswith(entry) or candidate == entry[1:]:
                return True
        elif candidate == entry:
            return True
    return False


def url_safety_problem(
    url: str,
    field: str,
    *,
    allowed_hosts: tuple[str, ...] = (),
    allow_local: bool = False,
) -> str | None:
    """Why *url* is unsafe to fetch, or ``None``.

    Returned rather than raised so the worker (HTTP 400) and ``@fetch`` (cell error) can each
    report it. Two rules:

    1. Scheme: only http and https.
    2. Address: every address the host resolves to must be public (not loopback,
       link-local including cloud metadata, private, multicast, reserved or unspecified;
       IPv4-mapped IPv6 is unmapped first). A blocklist on internal ranges rather than a host
       allowlist, because signed S3/GCS URLs resolve to many public addresses.

    ``STRATA_WORKER_ALLOWED_HOSTS`` exempts named hosts from the address rule (a server on a
    private address); ``STRATA_WORKER_ALLOW_LOCAL_HOSTS=1`` exempts every host, for tests and
    local dev, and wins when both are set. Allowlisted hosts are not resolved here.

    This check alone is racy under DNS rebinding; fetches close that with
    ``guarded_transport``. It stays in front for the scheme rule and an early, specific error.
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_URL_SCHEMES:
        return (
            f"{field} URL uses disallowed scheme {scheme!r}; "
            f"only {sorted(ALLOWED_URL_SCHEMES)} are accepted."
        )

    host = parsed.hostname
    if not host:
        return f"{field} URL is missing a host: {url!r}"

    # After the host check: the bypass relaxes only the address rule, and it is
    # set on every managed worker, so a hostless URL must still be refused.
    if allow_local:
        return None

    if host_is_allowlisted(host, allowed_hosts):
        # Listed hosts are trusted: whoever controls their resolution was
        # trusted by being listed, so the resolve-then-fetch race doesn't matter.
        return None

    try:
        addresses = _resolve(host)
    except socket.gaierror as exc:
        return f"{field} URL host {host!r} did not resolve: {exc}"

    problem = _address_problem(host, addresses)
    return None if problem is None else f"{field} URL {problem}"


def _resolve(host: str) -> list[str]:
    """Every address *host* resolves to: the one lookup the guard makes."""
    return [str(entry[4][0]) for entry in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)]


def _address_problem(host: str, addresses: list[str]) -> str | None:
    """Why connecting to *host* at *addresses* is unsafe, or ``None``.

    Every address must pass, so a name answering with one public and one private address is
    refused.
    """
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            return f"host {host!r} resolved to non-IP address {address!r}"
        if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
            ip = ip.ipv4_mapped
        if (
            ip.is_loopback
            or ip.is_link_local
            or ip.is_private
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return f"host {host!r} resolves to non-routable address {ip}; refusing to fetch"
    return None


def _validated_addresses(host: str, allowed_hosts: tuple[str, ...]) -> list[str]:
    """The addresses a connection to *host* may use: one lookup, every address checked.

    An allowlisted host comes back as its name, for the socket layer to resolve.

    Raises:
        httpcore.ConnectError: the host did not resolve, or resolved to a refused address
            (surfaces as ``httpx.ConnectError``).
    """
    if host_is_allowlisted(host, allowed_hosts):
        return [host]
    try:
        addresses = _resolve(host)
    except socket.gaierror as exc:
        raise httpcore.ConnectError(f"host {host!r} did not resolve: {exc}") from exc
    problem = _address_problem(host, addresses)
    if problem is not None:
        raise httpcore.ConnectError(problem)
    return addresses


class _GuardedBackend(httpcore.NetworkBackend):
    """Opens a TCP connection only to an address the guard has just validated.

    Only the TCP target changes (an IP literal from the checked lookup); httpcore still uses
    the host name for TLS SNI, certificate verification and the ``Host`` header.
    """

    def __init__(self, allowed_hosts: tuple[str, ...]):
        self._allowed_hosts = allowed_hosts
        self._inner = httpcore.SyncBackend()

    def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.NetworkStream:
        *earlier, last = _validated_addresses(host, self._allowed_hosts)
        # Each address in turn, as socket.create_connection does with a name,
        # so a host whose first address is unreachable (IPv6 without a route)
        # still connects. The last address's error is the one reported.
        for address in earlier:
            try:
                return self._inner.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                logger.debug("Could not connect to %s at %s: %s", host, address, exc)
        return self._inner.connect_tcp(
            last, port, timeout=timeout, local_address=local_address, socket_options=socket_options
        )


class _GuardedAsyncBackend(httpcore.AsyncNetworkBackend):
    """``_GuardedBackend`` for an async client; the lookup runs in a thread."""

    def __init__(self, allowed_hosts: tuple[str, ...]):
        self._allowed_hosts = allowed_hosts
        # What httpcore's async pool connects with when given no backend.
        self._inner = AutoBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[Any] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        *earlier, last = await asyncio.to_thread(_validated_addresses, host, self._allowed_hosts)
        for address in earlier:
            try:
                return await self._inner.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                logger.debug("Could not connect to %s at %s: %s", host, address, exc)
        return await self._inner.connect_tcp(
            last, port, timeout=timeout, local_address=local_address, socket_options=socket_options
        )


def guarded_transport(
    *, allowed_hosts: tuple[str, ...] = (), allow_local: bool = False
) -> httpx.HTTPTransport | None:
    """An httpx transport whose every connection passes the address rule.

    The check and the connection share one lookup, so DNS rebinding cannot swap in
    127.0.0.1, for every request and redirect hop. ``None`` when *allow_local* (httpx
    defaults apply). A guarded client ignores ``HTTPS_PROXY`` and connects directly, since a
    proxy would resolve the name itself; proxy-only deployments must make the proxy their
    egress control.
    """
    if allow_local:
        return None
    ssl_context = httpx.create_ssl_context()
    transport = httpx.HTTPTransport(verify=ssl_context)
    # httpx takes no network backend, so the pool it built is replaced by one
    # with the same TLS context, httpx's default limits, and the guard.
    transport._pool = httpcore.ConnectionPool(
        ssl_context=ssl_context,
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=5.0,
        network_backend=_GuardedBackend(allowed_hosts),
    )
    return transport


def guarded_async_transport(
    *, allowed_hosts: tuple[str, ...] = (), allow_local: bool = False
) -> httpx.AsyncHTTPTransport | None:
    """``guarded_transport`` for an ``httpx.AsyncClient``."""
    if allow_local:
        return None
    ssl_context = httpx.create_ssl_context()
    transport = httpx.AsyncHTTPTransport(verify=ssl_context)
    transport._pool = httpcore.AsyncConnectionPool(
        ssl_context=ssl_context,
        max_connections=100,
        max_keepalive_connections=20,
        keepalive_expiry=5.0,
        network_backend=_GuardedAsyncBackend(allowed_hosts),
    )
    return transport
