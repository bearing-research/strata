"""Pure parsing for artifact and name URIs (no store access).

Store-backed resolution belongs to the consumer.

URI grammar, where ``scheme`` is :func:`uri_scheme` (``vestigo`` unless the
consumer set its own with :func:`set_uri_scheme`):
    {scheme}://artifact/{id}@v={version}   -> pinned version
    {scheme}://artifact/{id}               -> latest (version reported as -1)
    {scheme}://name/{name}                 -> named pointer

A URI under any other scheme is not an artifact or name URI: the parsers return
``None`` for it, and nothing records which scheme a parse ran under. A consumer
with its own scheme sets it before anything parses, in the package's
``__init__`` ahead of the modules that parse at import time.
"""

from __future__ import annotations

import re

DEFAULT_URI_SCHEME = "vestigo"

# RFC 3986 scheme: a letter, then letters, digits, "+", "-" or ".".
_SCHEME = re.compile(r"^[a-z][a-z0-9+.\-]*$")

# Sentinel for "latest": callers must resolve the concrete version in the store.
LATEST_VERSION = -1


def _patterns(scheme: str) -> tuple[re.Pattern[str], re.Pattern[str], re.Pattern[str]]:
    prefix = re.escape(scheme) + "://"
    # ``@`` cannot appear in an id, so ``[^@]+`` cleanly stops before ``@v=``.
    return (
        re.compile(rf"^{prefix}artifact/([^@]+)@v=(\d+)$"),
        re.compile(rf"^{prefix}artifact/([^@]+)$"),
        re.compile(rf"^{prefix}name/(.+)$"),
    )


_scheme = DEFAULT_URI_SCHEME
_ARTIFACT_PINNED, _ARTIFACT_LATEST, _NAME = _patterns(_scheme)


def set_uri_scheme(scheme: str) -> None:
    """Use *scheme* in every URI this process parses or writes.

    A consumer sets it once, before any URI is parsed: a parse before that ran
    under the previous scheme and answered ``None`` for the consumer's URIs,
    silently. The scheme is persisted in ``input_versions`` rows and in whatever
    the consumer keeps, so a store written under one scheme is read under the
    same one.
    """
    global _scheme, _ARTIFACT_PINNED, _ARTIFACT_LATEST, _NAME
    if not _SCHEME.match(scheme):
        raise ValueError(f"not a URI scheme: {scheme!r}")
    _scheme = scheme
    _ARTIFACT_PINNED, _ARTIFACT_LATEST, _NAME = _patterns(scheme)


def uri_scheme() -> str:
    """The scheme every artifact and name URI carries in this process."""
    return _scheme


def artifact_uri(artifact_id: str, version: int) -> str:
    """The pinned URI of one version.

    Raises:
        ValueError: *version* is negative; ``LATEST_VERSION`` is a parse result,
            not a version, and the unpinned form has no ``@v=``.
    """
    if version < 0:
        raise ValueError(f"version {version} is not a pinned version")
    return f"{_scheme}://artifact/{artifact_id}@v={version}"


def artifact_uri_prefix() -> str:
    """What every artifact URI starts with."""
    return f"{_scheme}://artifact/"


def name_uri_prefix() -> str:
    """What every name URI starts with."""
    return f"{_scheme}://name/"


def parse_artifact_uri(uri: str) -> tuple[str, int] | None:
    """Parse an artifact URI into ``(artifact_id, version)``.

    ``version`` is ``LATEST_VERSION`` for the unpinned form; returns ``None`` when
    ``uri`` is not an artifact URI.
    """
    match = _ARTIFACT_PINNED.match(uri)
    if match:
        return (match.group(1), int(match.group(2)))

    match = _ARTIFACT_LATEST.match(uri)
    if match:
        return (match.group(1), LATEST_VERSION)

    return None


def parse_name_uri(uri: str) -> str | None:
    """Return the name in ``{scheme}://name/{name}``, or ``None`` when ``uri`` is not one."""
    match = _NAME.match(uri)
    if match:
        return match.group(1)
    return None
