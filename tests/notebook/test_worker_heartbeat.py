"""A pool worker behind a proxy that ends silent requests keeps a long /execute answer sending.

RunPod's proxy ends a request that sends nothing for 125 s. With
``STRATA_WORKER_HEARTBEAT_SECONDS`` set, an answer not ready within the interval
streams spaces, then the status and body it became, which strata-pool unwraps.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from strata.notebook.remote_executor import (
    STATUS_IN_BODY_HEADER,
    _with_heartbeat,
    create_notebook_executor_app,
)

INTERVAL = 0.01


async def _held(release: asyncio.Event, outcome):
    await release.wait()
    if isinstance(outcome, BaseException):
        raise outcome
    return outcome


async def _stream_until_released(response: StreamingResponse, release: asyncio.Event) -> list:
    """Read two heartbeats, let the work finish, then read the rest."""
    chunks = []
    iterator = response.body_iterator.__aiter__()
    chunks.append(await iterator.__anext__())
    chunks.append(await iterator.__anext__())
    release.set()
    async for chunk in iterator:
        chunks.append(chunk)
    return chunks


async def test_an_answer_ready_in_time_is_sent_as_it_is():
    answer = JSONResponse({"success": True}, status_code=201)

    async def quick():
        return answer

    assert await _with_heartbeat(quick(), interval=5.0) is answer


async def test_a_quick_refusal_keeps_its_status():
    async def refuse():
        raise HTTPException(status_code=400, detail="bad manifest")

    with pytest.raises(HTTPException) as exc_info:
        await _with_heartbeat(refuse(), interval=5.0)
    assert exc_info.value.status_code == 400


@pytest.mark.parametrize(
    ("outcome", "status", "content"),
    [
        (JSONResponse({"success": True}), 200, {"success": True}),
        (
            JSONResponse({"detail": "cell timed out"}, status_code=408),
            408,
            {"detail": "cell timed out"},
        ),
        (HTTPException(status_code=502, detail="upload failed"), 502, {"detail": "upload failed"}),
        (RuntimeError("bug"), 500, {"detail": "Internal Server Error"}),
    ],
    ids=["success", "error-response", "http-exception", "unexpected"],
)
async def test_a_long_answer_streams_spaces_then_the_status_it_became(outcome, status, content):
    release = asyncio.Event()

    response = await _with_heartbeat(_held(release, outcome), interval=INTERVAL)

    assert response.status_code == 200
    assert response.headers[STATUS_IN_BODY_HEADER] == "1"
    chunks = await _stream_until_released(response, release)
    assert chunks[:2] == [b" ", b" "]
    # The pool parses the whole body: leading spaces are JSON whitespace.
    answer = json.loads(b"".join(chunks))
    assert answer["status_code"] == status
    assert json.loads(answer["content"]) == content


async def test_a_caller_leaving_mid_stream_does_not_stop_the_execution():
    """As without the stream: the work runs on and cleans up after itself."""
    release = asyncio.Event()
    finished = asyncio.Event()

    async def work():
        await release.wait()
        finished.set()
        return JSONResponse({"success": True})

    response = await _with_heartbeat(work(), interval=INTERVAL)
    iterator = response.body_iterator.__aiter__()
    await iterator.__anext__()
    await iterator.aclose()

    release.set()
    await asyncio.wait_for(finished.wait(), timeout=5.0)


@pytest.mark.parametrize("heartbeat", [None, "30"])
def test_the_pool_route_refuses_a_bad_manifest_with_its_own_status(monkeypatch, heartbeat):
    monkeypatch.setenv("STRATA_WORKER_TOKEN", "t")
    if heartbeat is None:
        monkeypatch.delenv("STRATA_WORKER_HEARTBEAT_SECONDS", raising=False)
    else:
        monkeypatch.setenv("STRATA_WORKER_HEARTBEAT_SECONDS", heartbeat)
    client = TestClient(create_notebook_executor_app())

    response = client.post("/execute", content=b"[]", headers={"Authorization": "Bearer t"})

    assert response.status_code == 400
    assert STATUS_IN_BODY_HEADER not in response.headers


@pytest.mark.parametrize("value", ["0", "-1", "soon"])
def test_a_bad_interval_stops_the_worker_at_startup(monkeypatch, value):
    monkeypatch.setenv("STRATA_WORKER_HEARTBEAT_SECONDS", value)

    with pytest.raises(ValueError):
        create_notebook_executor_app()
