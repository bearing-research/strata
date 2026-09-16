"""Whether a URL is safe for this process to fetch.

Shared by the worker, which fetches the URLs a manifest hands it, and by a
notebook's ``@fetch``, which fetches the URL a cell names. Both are a request an
untrusted party shaped aimed at whatever network this process can reach.
"""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

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
    * DNS rebinding race: the IP we resolved here may differ from
      the IP httpx resolves at fetch time. Honest mitigation
      requires resolving once and passing the IP to httpx; left as
      a follow-up because the practical attacker who controls DNS
      already has stronger primitives. An allowlisted host does not
      resolve at all here, so the race does not apply to it -- but
      that is not a stronger position: it means an allowlist entry
      is trust in whoever controls that name's resolution, which is
      what listing it says.
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

    # After the host check, not before it. The bypass used to return here and
    # skipped both, so a URL with no host at all was accepted whenever it was
    # set — which is on every managed worker, since that is the documented way
    # to reach a server on a private address. Only the address rule is meant
    # to be relaxed.
    if allow_local:
        return None

    if host_is_allowlisted(host, allowed_hosts):
        # Named, therefore trusted. This is a statement about names the
        # operator controls, not a general relaxation: the resolve-then-fetch
        # race below stops mattering for these hosts, because whoever controls
        # their resolution was already trusted by being listed.
        return None

    try:
        addrinfo = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return f"{field} URL host {host!r} did not resolve: {exc}"

    for entry in addrinfo:
        sockaddr = entry[4]
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            return f"{field} URL host {host!r} resolved to non-IP address {sockaddr[0]!r}"
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
            return (
                f"{field} URL host {host!r} resolves to "
                f"non-routable address {ip}; refusing to fetch"
            )
    return None
