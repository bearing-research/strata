"""The type and headers stored artifact bytes are served under, and the response streaming them.

A writer declares ``content_type`` freely (``PUT /v1/artifacts``, the import route,
a cell's display output), so serving it as given would let anyone with write
access put a page that runs script on this server's origin.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from typing import Any, BinaryIO

from starlette.responses import StreamingResponse
from starlette.types import Receive, Scope, Send
from vestigo.blob_store import BLOB_STREAM_CHUNK_BYTES

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


class ClosingStreamingResponse(StreamingResponse):
    """A streaming response that runs ``on_close`` once it is done sending, however it ends.

    A generator's ``finally`` runs only if the body was iterated; a client gone before the
    first chunk never starts it.
    """

    def __init__(self, content: Any, *, on_close: Callable[[], Awaitable[None]], **kwargs: Any):
        super().__init__(content, **kwargs)
        self._on_close = on_close

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            await self._on_close()


class BlobResponse(ClosingStreamingResponse):
    """Stream an artifact's open blob reader, and close it however the response ends."""

    def __init__(
        self,
        reader_cm: AbstractContextManager[BinaryIO],
        *,
        media_type: str,
        headers: Mapping[str, str] | None = None,
        chunk_size: int = BLOB_STREAM_CHUNK_BYTES,
    ) -> None:
        reader = reader_cm.__enter__()

        def chunks() -> Iterator[bytes]:
            while chunk := reader.read(chunk_size):
                yield chunk

        async def close() -> None:
            # A read in progress finishes first: a cancelled thread-pool read is waited for.
            reader_cm.__exit__(None, None, None)

        super().__init__(chunks(), on_close=close, media_type=media_type, headers=headers)
