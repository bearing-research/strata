"""Logs viewer routes (observability, B1).

Read-only, system-scoped access to the recent structured log stream, backed by
the in-memory ring buffer (``strata.log_buffer``). ``GET /v1/logs`` pages by
cursor; ``GET /v1/logs/stream`` tails via SSE (polls the buffer by cursor).

v1 is unauthenticated system-wide read — fine behind a loopback personal
deployment; a scope gate is required before service mode exposes it (design
open-question #4).
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

router = APIRouter(tags=["logs"])

# How often the SSE tail polls the ring buffer for new entries.
_STREAM_POLL_SECONDS = 0.5


def _read(
    since: int, level: str | None, notebook: str | None, regex: str | None, limit: int
) -> dict[str, Any]:
    from strata.log_buffer import get_log_ring_buffer

    ring = get_log_ring_buffer()
    if ring is None:
        return {"entries": [], "cursor": 0}
    try:
        return ring.read(since=since, level=level, notebook=notebook, regex=regex, limit=limit)
    except re.error as exc:
        raise HTTPException(status_code=400, detail=f"invalid regex: {exc}")


@router.get("/v1/logs")
async def get_logs(
    since: int = 0,
    level: str | None = None,
    notebook: str | None = None,
    regex: str | None = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 200,
) -> dict[str, Any]:
    """Return recent structured log entries after ``since`` (cursor), newest last.

    Filters: ``level`` (minimum severity), ``notebook`` (exact ``notebook_id``),
    ``regex`` (matched against the message). Pass the returned ``cursor`` back as
    ``since`` to page forward.
    """
    return _read(since, level, notebook, regex, limit)


@router.get("/v1/logs/stream")
async def stream_logs(
    since: int = 0,
    level: str | None = None,
    notebook: str | None = None,
    regex: str | None = None,
) -> StreamingResponse:
    """Server-Sent Events tail of the log stream for the UI's "Live" mode.

    Emits each new entry as an SSE ``data:`` frame. Reconnect with
    ``?since=<last cursor>`` to resume without gaps.
    """
    # Validate the regex once up front so a bad pattern fails fast with 400
    # rather than inside the stream (where it would just close the connection).
    if regex is not None:
        try:
            re.compile(regex)
        except re.error as exc:
            raise HTTPException(status_code=400, detail=f"invalid regex: {exc}")

    async def event_stream():
        cursor = since
        while True:
            result = _read(cursor, level, notebook, regex, limit=1000)
            for entry in result["entries"]:
                yield f"data: {json.dumps(entry)}\n\n"
            cursor = result["cursor"]
            await asyncio.sleep(_STREAM_POLL_SECONDS)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
