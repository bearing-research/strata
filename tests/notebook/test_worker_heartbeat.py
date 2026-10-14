"""A pool worker behind a proxy that ends silent requests keeps a long /execute answer sending.

RunPod's proxy ends a request that sends nothing for 125 s. With
``STRATA_WORKER_HEARTBEAT_SECONDS`` set, an answer not ready within the interval
streams spaces, then the status and body it became, which strata-pool unwraps.
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

from strata.notebook import remote_executor
from strata.notebook.remote_executor import (
    NOTEBOOK_EXECUTOR_TRANSFORM_REF,
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

    assert await _with_heartbeat(quick(), interval=5.0, orphans=set()) is answer


async def test_a_quick_refusal_keeps_its_status():
    async def refuse():
        raise HTTPException(status_code=400, detail="bad manifest")

    with pytest.raises(HTTPException) as exc_info:
        await _with_heartbeat(refuse(), interval=5.0, orphans=set())
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

    response = await _with_heartbeat(_held(release, outcome), interval=INTERVAL, orphans=set())

    assert response.status_code == 200
    assert response.headers[STATUS_IN_BODY_HEADER] == "1"
    chunks = await _stream_until_released(response, release)
    assert chunks[:2] == [b" ", b" "]
    # The pool parses the whole body: leading spaces are JSON whitespace.
    answer = json.loads(b"".join(chunks))
    assert answer["status_code"] == status
    assert json.loads(answer["content"]) == content


async def _leave_mid_stream(work, orphans: set) -> None:
    response = await _with_heartbeat(work, interval=INTERVAL, orphans=orphans)
    iterator = response.body_iterator.__aiter__()
    await iterator.__anext__()
    await iterator.aclose()


async def test_a_caller_leaving_mid_stream_does_not_stop_the_execution():
    """As without the stream: the work runs on, held by the app until it finishes."""
    release = asyncio.Event()
    orphans: set = set()

    await _leave_mid_stream(_held(release, JSONResponse({"success": True})), orphans)

    assert len(orphans) == 1
    (task,) = orphans
    assert not task.done()
    release.set()
    await asyncio.wait_for(task, timeout=5.0)
    assert orphans == set()


async def test_an_orphan_that_fails_is_logged(caplog):
    """Nobody is left to read its answer, so the worker's log is the only record."""
    release = asyncio.Event()
    orphans: set = set()
    await _leave_mid_stream(_held(release, HTTPException(502, "upload failed")), orphans)
    (task,) = orphans

    with caplog.at_level(logging.ERROR, logger=remote_executor.logger.name):
        release.set()
        with pytest.raises(HTTPException):
            await asyncio.wait_for(task, timeout=5.0)

    assert "failed after its caller left" in caplog.text
    assert "upload failed" in caplog.text


async def test_a_handler_cancelled_before_streaming_cancels_its_work():
    """Without the heartbeat the work is the handler, so it stops with it."""
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def work():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            stopped.set()
            raise

    handler = asyncio.create_task(_with_heartbeat(work(), interval=60.0, orphans=set()))
    await started.wait()
    handler.cancel()

    await asyncio.wait_for(stopped.wait(), timeout=5.0)


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


def test_a_long_job_on_the_pool_route_streams_and_ends_with_its_status(monkeypatch):
    """Through the real route: the body is read before the stream starts, and an input
    download that fails after the first heartbeat comes back as the 502 it became."""
    monkeypatch.setenv("STRATA_WORKER_TOKEN", "t")
    monkeypatch.setenv("STRATA_WORKER_HEARTBEAT_SECONDS", "0.05")
    monkeypatch.setenv("STRATA_WORKER_ALLOW_LOCAL_HOSTS", "1")

    async def slow_missing_input(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.3)
        return httpx.Response(404)

    monkeypatch.setattr(
        remote_executor, "_guarded_transport", lambda: httpx.MockTransport(slow_missing_input)
    )
    manifest = {
        "metadata": {
            "executor_ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
            "params": {
                "source": "y = x",
                "input_specs": {"x": {"uri": "strata://artifact/a@v=1"}},
            },
        },
        "inputs": [{"artifact_id": "a", "version": 1, "url": "http://127.0.0.1:9/input"}],
        "output": {"url": "http://127.0.0.1:9/upload"},
        "finalize_url": "http://127.0.0.1:9/finalize",
    }
    client = TestClient(create_notebook_executor_app())

    response = client.post("/execute", json=manifest, headers={"Authorization": "Bearer t"})

    assert response.status_code == 200
    assert response.headers[STATUS_IN_BODY_HEADER] == "1"
    assert response.content.startswith(b" ")
    answer = response.json()
    assert answer["status_code"] == 502
    assert "Failed to download notebook input" in json.loads(answer["content"])["detail"]


async def test_a_body_slower_than_the_interval_is_read_before_the_stream_starts(monkeypatch):
    """Once streaming, Starlette reads ``receive`` for a disconnect; a manifest still
    arriving then would lose its messages to that listener."""
    monkeypatch.setenv("STRATA_WORKER_HEARTBEAT_SECONDS", "0.05")
    monkeypatch.setenv("STRATA_WORKER_ALLOW_LOCAL_HOSTS", "1")
    monkeypatch.delenv("STRATA_WORKER_TOKEN", raising=False)
    monkeypatch.setattr(
        remote_executor,
        "_guarded_transport",
        lambda: httpx.MockTransport(lambda request: httpx.Response(404)),
    )
    body = json.dumps(
        {
            "metadata": {
                "executor_ref": NOTEBOOK_EXECUTOR_TRANSFORM_REF,
                "params": {
                    "source": "y = x",
                    "input_specs": {"x": {"uri": "strata://artifact/a@v=1"}},
                },
            },
            "inputs": [{"artifact_id": "a", "version": 1, "url": "http://127.0.0.1:9/input"}],
            "output": {"url": "http://127.0.0.1:9/upload"},
            "finalize_url": "http://127.0.0.1:9/finalize",
        }
    ).encode()
    half = len(body) // 2
    arriving = [
        {"type": "http.request", "body": body[:half], "more_body": True},
        {"type": "http.request", "body": body[half:], "more_body": False},
    ]

    async def receive():
        if arriving:
            await asyncio.sleep(0.2)
            return arriving.pop(0)
        await asyncio.Event().wait()  # a connection that stays open

    sent: list[dict] = []

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/execute",
        "raw_path": b"/execute",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"content-type", b"application/json")],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 9),
    }

    await asyncio.wait_for(create_notebook_executor_app()(scope, receive, send), timeout=10.0)

    start = next(m for m in sent if m["type"] == "http.response.start")
    content = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    # The body's arrival is not the job's time, so the quick refusal keeps its status.
    assert start["status"] == 502
    assert "Failed to download notebook input" in json.loads(content)["detail"]


@pytest.mark.parametrize("value", ["0", "-1", "soon"])
def test_a_bad_interval_stops_the_worker_at_startup(monkeypatch, value):
    monkeypatch.setenv("STRATA_WORKER_HEARTBEAT_SECONDS", value)

    with pytest.raises(ValueError):
        create_notebook_executor_app()
