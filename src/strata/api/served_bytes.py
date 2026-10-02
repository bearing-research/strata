"""The type and headers stored artifact bytes are served under.

A writer declares ``content_type`` freely (``PUT /v1/artifacts``, the import route,
a cell's display output), so serving it as given would let anyone with write
access put a page that runs script on this server's origin.
"""

from __future__ import annotations

# Strata's own content types that a browser renders without running anything, and
# the media type each is served as. Anything else is downloaded as opaque bytes.
_INLINE_MEDIA_TYPES = {
    "image/png": "image/png",
    "arrow/ipc": "application/vnd.apache.arrow.stream",
    "json/object": "application/json",
    "text/markdown": "text/markdown",
}

# No sniffing the bytes into HTML, and no script even if a browser renders them.
_DATA_HEADERS = {"X-Content-Type-Options": "nosniff", "Content-Security-Policy": "sandbox"}


def served_media_type(content_type: str) -> tuple[str, dict[str, str]]:
    """Return ``(media_type, headers)`` for bytes whose stored type is *content_type*."""
    media_type = _INLINE_MEDIA_TYPES.get(content_type)
    if media_type is not None:
        return media_type, dict(_DATA_HEADERS)
    return "application/octet-stream", {**_DATA_HEADERS, "Content-Disposition": "attachment"}


def data_headers() -> dict[str, str]:
    """The headers for a route whose media type is fixed but whose bytes a writer chose."""
    return dict(_DATA_HEADERS)
