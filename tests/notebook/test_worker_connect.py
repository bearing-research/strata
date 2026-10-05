"""``strata-worker --connect``: a worker that dials out to a relay and binds no port.

The relay here is a test fixture written against docs/reference/worker-connect.md,
not against the worker's code, so these tests check the documented wire format.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Callable
from typing import Any

import httpx
import pytest
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket

from strata.notebook import worker_connect
from strata.notebook.remote_executor import create_notebook_executor_app
from strata.notebook.worker_connect import RelayRefusedError, run_connect, serve_connection
from tests.conftest import find_free_port, wait_for_server

RELAY_TOKEN = "relay-secret"
WORKER_TOKEN = "worker-secret"
_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


# ---- A minimal reference relay (test only; Strata does not ship one) ----


class _Relay:
    """One worker socket at ``/connect``; HTTP under ``/w/`` is forwarded over it."""

    def __init__(self, token: str) -> None:
        self.token = token
        self.loop: asyncio.AbstractEventLoop | None = None
        self.socket: WebSocket | None = None
        self.connections = 0
        self.connected = threading.Event()
        self.reconnected = threading.Event()  # a second worker socket was accepted
        self.forwarded: list[tuple[str, str]] = []
        self._next_id = 0
        self._pending: dict[int, asyncio.Queue[tuple[str, Any]]] = {}
        self.app = Starlette(
            routes=[
                Route("/health", lambda request: JSONResponse({"relay": True})),
                WebSocketRoute("/connect", self._accept),
                Route("/w/{path:path}", self._forward, methods=["GET", "POST"]),
            ]
        )

    async def _accept(self, websocket: WebSocket) -> None:
        offered = websocket.headers.get("sec-websocket-protocol", "")
        if websocket.headers.get("authorization") != f"Bearer {self.token}":
            await websocket.close()  # before accept, so the handshake gets 403
            return
        assert "strata-worker-connect.v1" in offered
        await websocket.accept(subprotocol="strata-worker-connect.v1")
        self.loop = asyncio.get_running_loop()
        self.socket = websocket
        self.connections += 1
        self.connected.set()
        if self.connections >= 2:
            self.reconnected.set()
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    data = message["bytes"]
                    queue = self._pending.get(int.from_bytes(data[:4], "big"))
                    if queue is not None:
                        queue.put_nowait(("chunk", data[4:]))
                else:
                    frame = json.loads(message["text"])
                    queue = self._pending.get(frame["id"])
                    if queue is not None:
                        queue.put_nowait((frame["type"], frame))
        finally:
            if self.socket is websocket:
                self.socket = None
                self.connected.clear()
            for queue in self._pending.values():
                queue.put_nowait(("abort", None))

    async def _forward(self, request: Request) -> Response:
        socket = self.socket
        if socket is None:
            return Response("no worker connected", status_code=502)
        self._next_id += 1
        stream_id = self._next_id
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        self._pending[stream_id] = queue
        path = request.url.path[len("/w") :]
        self.forwarded.append((request.method, path))
        headers = [[k, v] for k, v in request.headers.items() if k not in _HOP_BY_HOP]
        await socket.send_text(
            json.dumps(
                {
                    "type": "request",
                    "id": stream_id,
                    "method": request.method,
                    "path": path,
                    "query": request.url.query,
                    "headers": headers,
                }
            )
        )
        async for chunk in request.stream():
            for offset in range(0, len(chunk), 1024 * 1024):
                piece = chunk[offset : offset + 1024 * 1024]
                await socket.send_bytes(stream_id.to_bytes(4, "big") + piece)
        await socket.send_text(json.dumps({"type": "end", "id": stream_id}))

        kind, frame = await queue.get()
        if kind != "response":
            self._pending.pop(stream_id, None)
            return Response("worker aborted", status_code=502)

        async def _body():
            try:
                while True:
                    kind, item = await queue.get()
                    if kind == "chunk":
                        yield item
                    elif kind == "end":
                        return
                    else:
                        raise RuntimeError("the worker aborted the response")
            finally:
                self._pending.pop(stream_id, None)

        response_headers = {
            k: v for k, v in frame["headers"] if k.lower() not in _HOP_BY_HOP | {"content-length"}
        }
        return StreamingResponse(_body(), status_code=frame["status"], headers=response_headers)

    def drop_worker(self) -> None:
        """Close the worker's socket from the relay's side."""
        assert self.loop is not None and self.socket is not None
        asyncio.run_coroutine_threadsafe(self.socket.close(), self.loop).result(timeout=10)


@pytest.fixture
def relay():
    relay = _Relay(RELAY_TOKEN)
    port = find_free_port()
    server = uvicorn.Server(
        uvicorn.Config(
            relay.app, host="127.0.0.1", port=port, log_level="warning", ws="websockets-sansio"
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    if not wait_for_server(port, thread=thread):
        raise RuntimeError("the relay did not start")
    relay.http_url = f"http://127.0.0.1:{port}/w"  # type: ignore[attr-defined]
    relay.connect_url = f"ws://127.0.0.1:{port}/connect"  # type: ignore[attr-defined]
    try:
        yield relay
    finally:
        server.should_exit = True
        thread.join(timeout=5)


class _WorkerThread:
    """``run_connect`` on its own loop, as a separate worker process would run it."""

    def __init__(self, url: str, app: Any, token: str | None) -> None:
        self.app = app
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()
        self.future = asyncio.run_coroutine_threadsafe(
            run_connect(url, app, token=token), self.loop
        )

    def stop(self) -> None:
        async def _cancel_everything() -> None:
            current = asyncio.current_task()
            tasks = [t for t in asyncio.all_tasks() if t is not current]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        asyncio.run_coroutine_threadsafe(_cancel_everything(), self.loop).result(timeout=30)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)
        self.loop.close()


@pytest.fixture
def connected_worker(relay, monkeypatch):
    """A token-guarded worker connected to the relay; it binds no port."""
    monkeypatch.setenv("STRATA_WORKER_TOKEN", WORKER_TOKEN)
    monkeypatch.setenv("STRATA_WORKER_ALLOW_LOCAL_HOSTS", "1")
    monkeypatch.setattr(worker_connect, "INITIAL_BACKOFF_SECONDS", 0.05)
    worker = _WorkerThread(relay.connect_url, create_notebook_executor_app(), RELAY_TOKEN)
    try:
        assert relay.connected.wait(timeout=30), "the worker never reached the relay"
        yield worker
    finally:
        worker.stop()


# ---- Through the relay, end to end ----


class TestThroughARelay:
    def test_health_and_the_worker_token_pass_through(self, relay, connected_worker):
        """``/health`` is open; ``/v1/*`` still needs the worker's own token, not the relay's."""
        assert httpx.get(f"{relay.http_url}/health", timeout=30).json()["status"] == "healthy"

        refused = httpx.post(
            f"{relay.http_url}/v1/executions/b1/cancel",
            headers={"Authorization": f"Bearer {RELAY_TOKEN}"},
            timeout=30,
        )
        assert refused.status_code == 401

        allowed = httpx.post(
            f"{relay.http_url}/v1/executions/b1/cancel",
            headers={"Authorization": f"Bearer {WORKER_TOKEN}"},
            timeout=30,
        )
        assert allowed.json() == {"build_id": "b1", "cancelled": False}

    def test_the_worker_comes_back_after_the_relay_drops_it(self, relay, connected_worker):
        relay.drop_worker()

        # ``connected`` stays set until the relay reads the disconnect, so wait
        # on the event the second accept sets rather than polling it.
        assert relay.reconnected.wait(timeout=30), "the worker did not reconnect"
        assert relay.connections == 2
        assert relay.connected.wait(timeout=30)
        assert httpx.get(f"{relay.http_url}/health", timeout=30).json()["status"] == "healthy"


def _session(tmp_path, worker_url: str, transport: str, build_server: dict | None):
    from strata.notebook.models import WorkerBackendType, WorkerSpec
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook

    notebook_dir = create_notebook(tmp_path, "Relay Worker")
    add_cell_to_notebook(notebook_dir, "cell1", None)
    add_cell_to_notebook(notebook_dir, "cell2", "cell1")
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    spec: dict[str, Any] = {"url": worker_url, "transport": transport, "token": WORKER_TOKEN}
    if build_server is not None:
        spec["strata_url"] = build_server["base_url"]
        build_server["config"].transforms_config["notebook_workers"] = [
            {"name": "relayed", "backend": "executor", "runtime_id": "r", "config": spec}
        ]
    session.notebook_state.workers = [
        WorkerSpec(name="relayed", backend=WorkerBackendType.EXECUTOR, runtime_id="r", config=spec)
    ]
    return session


def _capture_broadcasts(monkeypatch, on_send: Callable[[dict], None] | None = None) -> list[dict]:
    sent: list[dict] = []

    async def _capture(notebook_id, message):
        sent.append(message)
        if on_send is not None:
            on_send(message)

    monkeypatch.setattr("strata.notebook.ws._broadcast_message", _capture)
    return sent


@pytest.mark.asyncio
async def test_a_pushed_cell_and_its_input_go_through_the_relay(tmp_path, relay, connected_worker):
    """The direct transport: a multipart body in, the output bundle out, both relayed."""
    from strata.notebook.executor import CellExecutor

    session = _session(tmp_path, f"{relay.http_url}/v1/execute", "direct", None)
    cell1, cell2 = session.notebook_state.cells
    cell1.source = "x = 41"
    cell2.source = "y = x + 1"
    cell2.worker = "relayed"
    session.re_analyze_cell(cell1.id)
    session.re_analyze_cell(cell2.id)

    executor = CellExecutor(session)
    assert (await executor.execute_cell(cell1.id, cell1.source)).success
    result = await executor.execute_cell(cell2.id, cell2.source)

    assert result.success, result.error
    assert result.execution_method == "executor"
    assert ("POST", "/v1/execute") in relay.forwarded
    assert result.outputs["y"]["preview"] == 42


@pytest.mark.asyncio
async def test_a_signed_cell_streams_its_console_and_returns_its_result(
    tmp_path, relay, connected_worker, notebook_build_server, monkeypatch
):
    from strata.notebook.executor import CellExecutor

    sent = _capture_broadcasts(monkeypatch)
    session = _session(tmp_path, f"{relay.http_url}/v1/execute", "signed", notebook_build_server)
    cell = session.notebook_state.cells[0]
    cell.worker = "relayed"
    cell.source = "print('over-the-relay')\nx = 7"
    session.re_analyze_cell(cell.id)

    result = await CellExecutor(session).execute_cell(cell.id, cell.source)

    assert result.success, result.error
    assert ("POST", "/v1/execute-manifest") in relay.forwarded
    assert result.outputs["x"]["preview"] == 7
    console = [m for m in sent if m["type"] == "cell_console"]
    assert "".join(m["payload"]["text"] for m in console) == "over-the-relay\n"


@pytest.mark.asyncio
async def test_cancelling_a_relayed_cell_stops_it_on_the_worker(
    tmp_path, relay, connected_worker, notebook_build_server, monkeypatch
):
    """The cancel is a second request over the same socket while the first is in flight."""
    from strata.notebook.executor import CellExecutor

    loop = asyncio.get_running_loop()
    printed = asyncio.Event()

    def _on_send(message: dict) -> None:
        if message["type"] == "cell_console":
            loop.call_soon_threadsafe(printed.set)

    _capture_broadcasts(monkeypatch, _on_send)
    session = _session(tmp_path, f"{relay.http_url}/v1/execute", "signed", notebook_build_server)
    cell = session.notebook_state.cells[0]
    cell.worker = "relayed"
    cell.source = "import time\nprint('started', flush=True)\ntime.sleep(300)\nx = 1"
    session.re_analyze_cell(cell.id)

    run = asyncio.create_task(CellExecutor(session).execute_cell(cell.id, cell.source))
    await asyncio.wait_for(printed.wait(), timeout=120)
    in_flight = connected_worker.app.state.in_flight
    assert len(in_flight) == 1, "the harness is not running on the worker"
    ((build_id, proc),) = in_flight.items()

    run.cancel()
    with pytest.raises(asyncio.CancelledError):
        await run

    assert ("POST", f"/v1/executions/{build_id}/cancel") in relay.forwarded
    for _ in range(3000):
        if proc.returncode is not None:
            break
        await asyncio.sleep(0.01)
    assert proc.returncode is not None, "the harness survived the cancel"


# ---- Handshake and CLI ----


@pytest.mark.asyncio
async def test_a_relay_that_refuses_the_token_stops_the_worker(relay, monkeypatch):
    monkeypatch.delenv("STRATA_WORKER_TOKEN", raising=False)
    with pytest.raises(RelayRefusedError, match="403"):
        await asyncio.wait_for(
            run_connect(relay.connect_url, create_notebook_executor_app(), token="wrong"),
            timeout=30,
        )


def test_the_cli_dials_out_without_binding_a_port(relay, monkeypatch):
    """``--connect`` never starts uvicorn; a refused token exits 1 rather than retrying."""
    from strata.notebook import remote_executor

    def _no_bind(*args, **kwargs):
        raise AssertionError("strata-worker --connect bound a port")

    monkeypatch.setattr(uvicorn, "run", _no_bind)
    monkeypatch.setenv("STRATA_WORKER_CONNECT_TOKEN", "wrong")
    monkeypatch.setenv("STRATA_WORKER_TOKEN", WORKER_TOKEN)

    assert remote_executor.main(["--connect", relay.connect_url]) == 1


def test_the_cli_presents_the_relay_token_from_the_environment(relay, monkeypatch):
    """The token comes from ``STRATA_WORKER_CONNECT_TOKEN`` and is kept from cells."""
    from strata.notebook import remote_executor

    seen: dict[str, Any] = {}

    async def _record(url, app, *, token=None):
        seen["token"] = token

    monkeypatch.setattr(worker_connect, "run_connect", _record)
    monkeypatch.setenv("STRATA_WORKER_CONNECT_TOKEN", RELAY_TOKEN)
    monkeypatch.setenv("STRATA_WORKER_TOKEN", WORKER_TOKEN)

    assert remote_executor.main(["--connect", relay.connect_url]) == 0
    assert seen["token"] == RELAY_TOKEN
    import os

    assert "STRATA_WORKER_CONNECT_TOKEN" not in os.environ, "a cell could read it"


def test_the_cli_refuses_a_non_websocket_url():
    from strata.notebook import remote_executor

    with pytest.raises(SystemExit):
        remote_executor.main(["--connect", "https://relay.example.com/connect"])


# ---- The multiplexer, against a scripted socket ----


class _ScriptedSocket:
    """Feeds frames to ``serve_connection`` and records what it sends back."""

    def __init__(self) -> None:
        self.incoming: asyncio.Queue[str | bytes | None] = asyncio.Queue()
        self.sent: list[str | bytes] = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        message = await self.incoming.get()
        if message is None:
            raise StopAsyncIteration
        return message

    async def send(self, message: str | bytes) -> None:
        self.sent.append(message)

    def request(self, stream_id: int, method: str, path: str, body: list[bytes] = ()) -> None:
        frame = {"type": "request", "id": stream_id, "method": method, "path": path}
        self.incoming.put_nowait(json.dumps({**frame, "query": "", "headers": []}))
        for chunk in body:
            self.incoming.put_nowait(stream_id.to_bytes(4, "big") + chunk)
        self.incoming.put_nowait(json.dumps({"type": "end", "id": stream_id}))

    def response(self, stream_id: int) -> dict[str, Any]:
        out: dict[str, Any] = {"status": None, "body": b"", "chunks": 0, "end": False}
        for message in self.sent:
            if isinstance(message, bytes):
                if int.from_bytes(message[:4], "big") == stream_id:
                    out["body"] += message[4:]
                    out["chunks"] += 1
                continue
            frame = json.loads(message)
            if frame["id"] != stream_id:
                continue
            if frame["type"] == "response":
                out["status"] = frame["status"]
            elif frame["type"] in ("end", "abort"):
                out[frame["type"]] = True
        return out


async def _until(predicate: Callable[[], bool]) -> None:
    for _ in range(3000):
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never held")


@pytest.mark.asyncio
async def test_a_request_refused_unread_does_not_wedge_the_socket(monkeypatch):
    """A 401 answers before the body is read; the body left behind must not stall the reader."""
    monkeypatch.setenv("STRATA_WORKER_TOKEN", WORKER_TOKEN)
    monkeypatch.setattr(worker_connect, "_BODY_QUEUE_CHUNKS", 1)
    socket = _ScriptedSocket()
    serving = asyncio.create_task(serve_connection(socket, create_notebook_executor_app()))
    try:
        socket.request(1, "POST", "/v1/execute", [b"x" * 1024] * 8)
        socket.request(2, "GET", "/health")
        await asyncio.wait_for(_until(lambda: socket.response(2)["end"]), timeout=60)

        assert socket.response(1)["status"] == 401
        assert socket.response(2)["status"] == 200
    finally:
        socket.incoming.put_nowait(None)
        await serving


@pytest.mark.asyncio
async def test_requests_run_concurrently():
    """The first request waits on the second; served one at a time, neither would finish."""
    second_arrived = asyncio.Event()

    async def app(scope, receive, send):
        if scope["path"] == "/first":
            await second_arrived.wait()
        else:
            second_arrived.set()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": scope["path"].encode()})

    socket = _ScriptedSocket()
    serving = asyncio.create_task(serve_connection(socket, app))
    try:
        socket.request(1, "GET", "/first")
        socket.request(2, "GET", "/second")
        await asyncio.wait_for(_until(lambda: socket.response(1)["end"]), timeout=30)
    finally:
        socket.incoming.put_nowait(None)
        await serving

    assert socket.response(1)["body"] == b"/first"
    assert socket.response(2)["body"] == b"/second"


@pytest.mark.asyncio
async def test_an_abort_reaches_the_app_as_a_disconnect():
    received: list[str] = []
    done = asyncio.Event()

    async def app(scope, receive, send):
        while True:
            message = await receive()
            received.append(message["type"])
            if message["type"] == "http.disconnect":
                break
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"late"})
        done.set()

    socket = _ScriptedSocket()
    serving = asyncio.create_task(serve_connection(socket, app))
    try:
        socket.request(1, "POST", "/", [b"part"])
        await _until(lambda: "http.request" in received)
        socket.incoming.put_nowait(json.dumps({"type": "abort", "id": 1}))
        await asyncio.wait_for(done.wait(), timeout=30)
    finally:
        socket.incoming.put_nowait(None)
        await serving

    assert received[-1] == "http.disconnect"
    assert socket.response(1)["status"] is None, "an abandoned response must not be sent"


@pytest.mark.asyncio
async def test_a_large_body_is_split_into_bounded_messages():
    body = b"z" * (2 * worker_connect.MAX_CHUNK_BYTES + 10)

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": body})

    socket = _ScriptedSocket()
    serving = asyncio.create_task(serve_connection(socket, app))
    try:
        socket.request(1, "GET", "/")
        await asyncio.wait_for(_until(lambda: socket.response(1)["end"]), timeout=30)
    finally:
        socket.incoming.put_nowait(None)
        await serving

    response = socket.response(1)
    assert response["body"] == body
    assert response["chunks"] == 3
    assert all(
        len(m) <= worker_connect.MAX_CHUNK_BYTES + 4 for m in socket.sent if isinstance(m, bytes)
    )


@pytest.mark.asyncio
async def test_an_app_that_raises_before_answering_gets_a_500():
    async def app(scope, receive, send):
        raise RuntimeError("boom")

    socket = _ScriptedSocket()
    serving = asyncio.create_task(serve_connection(socket, app))
    try:
        socket.request(1, "GET", "/")
        await asyncio.wait_for(_until(lambda: socket.response(1)["end"]), timeout=30)
    finally:
        socket.incoming.put_nowait(None)
        await serving

    assert socket.response(1)["status"] == 500
