"""Read-only routes over the in-memory structured log ring buffer.

The buffer is server-wide (every tenant's records), so under principal auth both
routes require ``admin:*``; without principal auth they stay open.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from strata.api.dependencies import require_scope

router = APIRouter(tags=["logs"])

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


@router.get("/v1/logs", dependencies=[require_scope("admin:*")])
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
    ``since`` to page forward. Requires ``admin:*`` under principal auth.
    """
    return _read(since, level, notebook, regex, limit)


@router.get("/v1/logs/stream", dependencies=[require_scope("admin:*")])
async def stream_logs(
    since: int = 0,
    level: str | None = None,
    notebook: str | None = None,
    regex: str | None = None,
) -> StreamingResponse:
    """Tail the log stream as Server-Sent Events, one ``data:`` frame per entry.

    Reconnect with ``?since=<last cursor>`` to resume without gaps. Requires
    ``admin:*`` under principal auth.
    """
    # Validate the regex up front so a bad pattern is a 400, not a silently closed stream.
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
