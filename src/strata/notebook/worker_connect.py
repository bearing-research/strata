"""Serve the worker app over an outbound WebSocket to a relay (``strata-worker --connect``).

A worker behind NAT dials out, and the relay presents it at a URL the server
dispatches to as it would to any worker, forwarding each HTTP request over the
socket. The wire format is specified in docs/reference/worker-connect.md.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any
from urllib.parse import unquote

from starlette.types import ASGIApp, Message, Scope
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException
from websockets.typing import Subprotocol

logger = logging.getLogger(__name__)

SUBPROTOCOL = "strata-worker-connect.v1"
# Largest body payload in one binary message; the 4-byte stream id comes on top.
MAX_CHUNK_BYTES = 1024 * 1024
_ID_BYTES = 4
# Bounds a request body held for the app; a full queue stalls the socket's reader, as TCP would.
_BODY_QUEUE_CHUNKS = 16
INITIAL_BACKOFF_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 60.0


# The loop holds only weak references to tasks, and a request outlives a dropped connection.
_TASKS: set[asyncio.Task[None]] = set()


class RelayRefusedError(RuntimeError):
    """The relay turned the worker away; reconnecting would be refused the same way."""


def chunk_message(stream_id: int, data: bytes) -> bytes:
    """A binary message: the stream id, 4 bytes big-endian, then body bytes."""
    return stream_id.to_bytes(_ID_BYTES, "big") + data


class _Stream:
    """One forwarded request: its body on the way in and whether it was abandoned."""

    def __init__(self) -> None:
        self.inbox: asyncio.Queue[Message] = asyncio.Queue(maxsize=_BODY_QUEUE_CHUNKS)
        self.body_done = False
        self.aborted = False

    def drop_body(self) -> None:
        # Getting frees a slot, which wakes a reader blocked putting into a full queue.
        while not self.inbox.empty():
            self.inbox.get_nowait()

    def abort(self) -> None:
        self.aborted = True
        self.drop_body()
        self.inbox.put_nowait({"type": "http.disconnect"})


def _scope(frame: dict[str, Any]) -> Scope:
    path = str(frame.get("path") or "/")
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": str(frame.get("method") or "GET").upper(),
        "scheme": "http",
        "path": unquote(path),
        "raw_path": path.encode("latin-1"),
        "query_string": str(frame.get("query") or "").encode("latin-1"),
        "root_path": "",
        "headers": [
            (str(name).lower().encode("latin-1"), str(value).encode("latin-1"))
            for name, value in frame.get("headers") or []
        ],
        "client": None,
        "server": None,
    }


async def _handle(
    ws: ClientConnection,
    app: ASGIApp,
    stream_id: int,
    frame: dict[str, Any],
    stream: _Stream,
    streams: dict[int, _Stream],
) -> None:
    """Run one request through the app and send its response back as frames."""
    started = False
    finished = False

    async def receive() -> Message:
        if stream.aborted:
            return {"type": "http.disconnect"}
        return await stream.inbox.get()

    async def send(message: Message) -> None:
        nonlocal started, finished
        # As uvicorn does once the client has gone: the app finishes, its output goes nowhere.
        if stream.aborted or finished:
            return
        if message["type"] == "http.response.start":
            started = True
            headers = [
                [name.decode("latin-1"), value.decode("latin-1")]
                for name, value in message.get("headers", [])
            ]
            await ws.send(
                json.dumps(
                    {
                        "type": "response",
                        "id": stream_id,
                        "status": message["status"],
                        "headers": headers,
                    }
                )
            )
        elif message["type"] == "http.response.body":
            body = message.get("body", b"")
            for offset in range(0, len(body), MAX_CHUNK_BYTES):
                await ws.send(chunk_message(stream_id, body[offset : offset + MAX_CHUNK_BYTES]))
            if not message.get("more_body", False):
                finished = True
                await ws.send(json.dumps({"type": "end", "id": stream_id}))

    try:
        try:
            await app(_scope(frame), receive, send)
        except ConnectionClosed:
            return
        except Exception:
            logger.exception("Relayed request %s %s failed", frame.get("method"), frame.get("path"))
        if finished or stream.aborted:
            return
        try:
            if started:
                await ws.send(json.dumps({"type": "abort", "id": stream_id}))
            else:
                await ws.send(
                    json.dumps(
                        {
                            "type": "response",
                            "id": stream_id,
                            "status": 500,
                            "headers": [["content-type", "text/plain; charset=utf-8"]],
                        }
                    )
                )
                await ws.send(chunk_message(stream_id, b"Internal Server Error"))
                await ws.send(json.dumps({"type": "end", "id": stream_id}))
        except ConnectionClosed:
            return
    finally:
        streams.pop(stream_id, None)
        stream.drop_body()


async def serve_connection(ws: ClientConnection, app: ASGIApp) -> None:
    """Serve the requests a relay forwards over *ws* until the connection closes.

    Requests run concurrently, one task each. A request still running when the
    connection drops runs to the end, as one whose client hung up does under uvicorn;
    the server's cancel route is what stops a cell.
    """
    streams: dict[int, _Stream] = {}
    try:
        async for message in ws:
            if isinstance(message, bytes):
                stream = streams.get(int.from_bytes(message[:_ID_BYTES], "big"))
                if stream is not None and not stream.body_done and not stream.aborted:
                    await stream.inbox.put(
                        {"type": "http.request", "body": message[_ID_BYTES:], "more_body": True}
                    )
                continue
            try:
                frame = json.loads(message)
            except ValueError:
                logger.debug("Ignoring a relay frame that is not JSON")
                continue
            if not isinstance(frame, dict):
                continue
            kind = frame.get("type")
            stream_id = frame.get("id")
            if kind == "request":
                # The id must fit the 4-byte chunk header, or framing the response fails.
                if (
                    not isinstance(stream_id, int)
                    or not 0 < stream_id < 2**32
                    or stream_id in streams
                ):
                    logger.debug("Ignoring a request frame with a bad or duplicate id")
                    continue
                stream = _Stream()
                streams[stream_id] = stream
                task = asyncio.create_task(_handle(ws, app, stream_id, frame, stream, streams))
                _TASKS.add(task)
                task.add_done_callback(_TASKS.discard)
                continue
            stream = streams.get(stream_id) if isinstance(stream_id, int) else None
            if stream is None or stream.aborted:
                continue
            if kind == "end" and not stream.body_done:
                stream.body_done = True
                await stream.inbox.put({"type": "http.request", "body": b"", "more_body": False})
            elif kind == "abort":
                stream.abort()
    finally:
        for stream in list(streams.values()):
            stream.abort()


async def run_connect(url: str, app: ASGIApp, *, token: str | None = None) -> None:
    """Hold a connection to the relay at *url*, reconnecting with backoff, until cancelled.

    Raises:
        RelayRefusedError: The relay refused the handshake with a 4xx (other than
            408 or 429), or does not speak this protocol.
    """
    headers = {"Authorization": f"Bearer {token}"} if token else None
    delay = INITIAL_BACKOFF_SECONDS
    while True:
        try:
            async with connect(
                url,
                additional_headers=headers,
                subprotocols=[Subprotocol(SUBPROTOCOL)],
                max_size=MAX_CHUNK_BYTES + _ID_BYTES,
            ) as ws:
                if ws.subprotocol != SUBPROTOCOL:
                    raise RelayRefusedError(
                        f"The relay at {url} did not select the {SUBPROTOCOL} subprotocol"
                    )
                delay = INITIAL_BACKOFF_SECONDS
                logger.info("Connected to the relay at %s", url)
                await serve_connection(ws, app)
            logger.warning("The relay at %s closed the connection", url)
        except InvalidStatus as exc:
            status = exc.response.status_code
            if 400 <= status < 500 and status not in (408, 429):
                raise RelayRefusedError(
                    f"The relay at {url} refused the worker with HTTP {status}"
                ) from exc
            logger.warning("The relay at %s answered HTTP %d", url, status)
        except (OSError, TimeoutError, WebSocketException) as exc:
            logger.warning("Lost the relay at %s: %s", url, exc)
        wait = delay / 2 + random.uniform(0, delay / 2)
        logger.info("Reconnecting to the relay in %.1f s", wait)
        await asyncio.sleep(wait)
        delay = min(delay * 2, MAX_BACKOFF_SECONDS)
