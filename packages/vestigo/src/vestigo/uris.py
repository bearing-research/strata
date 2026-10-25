"""Pure parsing for Strata artifact and name URIs (no store or server access).

Store-backed resolution lives in ``server._resolve_artifact_uri``.

URI grammar:
    strata://artifact/{id}@v={version}   -> pinned version
    strata://artifact/{id}               -> latest (version reported as -1)
    strata://name/{name}                 -> named pointer
"""

from __future__ import annotations

import re

# ``@`` cannot appear in an id, so ``[^@]+`` cleanly stops before ``@v=``.
_ARTIFACT_PINNED = re.compile(r"^strata://artifact/([^@]+)@v=(\d+)$")
_ARTIFACT_LATEST = re.compile(r"^strata://artifact/([^@]+)$")
_NAME = re.compile(r"^strata://name/(.+)$")

# Sentinel for "latest": callers must resolve the concrete version in the store.
LATEST_VERSION = -1


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
    """Return the name in ``strata://name/{name}``, or ``None`` when ``uri`` is not one."""
    match = _NAME.match(uri)
    if match:
        return match.group(1)
    return None
