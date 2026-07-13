"""Pure parsing for Strata artifact and name URIs.

These helpers are the stateless half of URI handling: string → identifiers,
with no artifact-store or server-state access. Store-backed *resolution*
(latest-version lookup, name resolution) lives in ``server._resolve_artifact_uri``,
which builds on these parsers.

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

# Sentinel version meaning "resolve to the latest version" — callers that see
# this must look the concrete version up in the artifact store.
LATEST_VERSION = -1


def parse_artifact_uri(uri: str) -> tuple[str, int] | None:
    """Parse an artifact URI into ``(artifact_id, version)``.

    Returns ``version == LATEST_VERSION`` for the unpinned
    ``strata://artifact/{id}`` form, or ``None`` when ``uri`` is not an
    artifact URI.
    """
    match = _ARTIFACT_PINNED.match(uri)
    if match:
        return (match.group(1), int(match.group(2)))

    match = _ARTIFACT_LATEST.match(uri)
    if match:
        return (match.group(1), LATEST_VERSION)

    return None


def parse_name_uri(uri: str) -> str | None:
    """Parse a name URI (``strata://name/{name}``) into its name.

    Returns ``None`` when ``uri`` is not a name URI.
    """
    match = _NAME.match(uri)
    if match:
        return match.group(1)
    return None
