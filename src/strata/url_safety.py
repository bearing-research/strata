"""Whether a URL is safe for this process to fetch.

Shared by the worker, which fetches the URLs a manifest hands it, and by a
notebook's ``@fetch``, which fetches the URL a cell names. Both are a request an
untrusted party shaped aimed at whatever network this process can reach.
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

    Scheme only, and no network: this answers "may I make this a link", which
    a page render asks and which must not depend on DNS. ``url_safety_problem``
    is the question a fetch asks and resolves the host.

    A record holds whatever was written into it, so a value can be
    ``javascript:...`` -- which survives HTML escaping and runs on the origin
    of whoever clicks it. Testing for a leading "http" is not the same check:
    ``httpfoo://x`` starts with it and is not a URL.
    """
    try:
        scheme = urlparse(value).scheme.lower()
    except ValueError:
        return None
    return value if scheme in ALLOWED_URL_SCHEMES else None


def host_is_allowlisted(host: str, allowed_hosts: tuple[str, ...]) -> bool:
    """Whether *host* is named in the allowlist.

    Matched on the name, never on the resolved address — that is the whole
    point, since these hosts are trusted *because* an operator named them.

    Suffixes are anchored on a dot, so ``.example.com`` matches
    ``build.example.com`` and not ``evil-example.com``. Getting that wrong is
    silent: the wrong host passes and nothing says so.
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

    Returned rather than raised so a worker (an HTTP 400) and a notebook
    ``@fetch`` (a cell error) can each report it their own way.

    A compromised or buggy orchestrator could hand the worker URLs
    that point at internal services. Two distinct defenses:

    1. **Scheme allowlist** — only http and https. Blocks file://,
       data:, javascript:, ftp:// and any other scheme httpx might
       grow plugin support for.
    2. **Host resolution + IP-range blocklist** — the resolved IP
       must not be loopback, link-local (incl. cloud metadata
       169.254.169.254 / fd00:ec2::254), private, multicast,
       reserved, or unspecified. Hostnames are resolved via
       getaddrinfo and every returned address is checked; a
       hostname that resolves to multiple addresses must have all
       of them in the public range to pass. This rules out both
       direct internal-IP URLs and hostname-based variants
       (e.g. metadata.google.internal). Set
       ``STRATA_WORKER_ALLOW_LOCAL_HOSTS=1`` to bypass the IP check
       (tests / local dev with 127.0.0.1 build servers); production
       deployments leave it unset.

    Allowlist-on-host instead of blocklist-on-host would be more
    restrictive but breaks real signed-URL usage where S3/GCS
    buckets resolve to public IPs across many regions. Blocklist
    on internal ranges is the right tradeoff.

    ``STRATA_WORKER_ALLOWED_HOSTS`` names specific hosts that pass
    the address rule anyway -- for a server on a private address,
    which is the ordinary shape of a managed worker talking to the
    server that dispatched it. It supersedes
    ``STRATA_WORKER_ALLOW_LOCAL_HOSTS``, which relaxes the same rule
    for *every* host and remains for tests and local development
    where 127.0.0.1 really is the target. A deployment that sets
    both gets the wholesale bypass, because that is what it asked
    for; prefer the allowlist in production.

    Caveats:
    * This check alone is racy: a name can resolve to a public
      address here and to 127.0.0.1 when the request connects (DNS
      rebinding). A fetch closes that with ``guarded_transport``,
      whose connections use only addresses validated by the lookup
      that produced them. This check stays in front of it for the
      scheme and host rules and for an early, specific error. An
      allowlisted host does not resolve at all here, and is not
      pinned either: an allowlist entry is trust in whoever controls
      that name's resolution, which is what listing it says.
    * IPv4-mapped IPv6 (``::ffff:127.0.0.1``) is caught — we
      ``unmap()`` before checking.
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

    Every address has to pass, so a name that answers with one public and one
    private address is refused rather than connected to whichever comes first.
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
    """The addresses a connection to *host* may use: what one lookup returned,
    every one of them checked.

    An allowlisted host comes back as its name, for the socket layer to
    resolve, because it is trusted by name (see ``url_safety_problem``).

    Raises:
        httpcore.ConnectError: the host did not resolve, or resolved to an
            address the guard refuses. httpx reports it as ``httpx.ConnectError``.
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

    httpcore passes this the request's host name and, separately, uses that
    name for TLS (SNI and certificate verification) and the ``Host`` header.
    Only the TCP target changes: an IP literal from the checked lookup, so the
    socket layer has no name left to look up again.
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

    For ``httpx.Client(transport=...)``. The check and the connection use the
    same lookup, so a name cannot answer the check with a public address and
    the connection with 127.0.0.1. That holds for every request the client
    makes, each redirect hop included, whatever ``follow_redirects`` says.

    ``None`` when *allow_local*: there is no address rule to hold a
    connection to, and the client keeps httpx's defaults.

    Proxies: a client given a transport does not read ``HTTPS_PROXY`` and the
    like, so a guarded client connects directly. That is deliberate: through a
    proxy, the proxy resolves the name, and what this process checked says
    nothing about where the proxy connects. A deployment that reaches the
    internet only through a proxy has to make that proxy its egress control.
    With *allow_local* set the environment's proxies apply, as before.
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
