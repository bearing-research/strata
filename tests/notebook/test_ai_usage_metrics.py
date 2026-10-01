"""Model tokens this server used, by tenant, principal and model.

A fake OpenAI-compatible provider returns known token counts, so they pass through the
real streamed and plain completion parsing and out of the Prometheus route.
"""

from __future__ import annotations

import http.server
import json
import threading

import pytest
from fastapi.testclient import TestClient

from strata.auth import set_principal
from strata.notebook.llm.config import LlmConfig
from strata.notebook.llm.usage import reset_llm_usage
from strata.types import Principal


class _Provider(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("stream"):
            chunks = [
                {"model": "m-1", "choices": [{"delta": {"content": "Done."}}]},
                {
                    "model": "m-1",
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                },
                {
                    "model": "m-1",
                    "choices": [],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
                },
            ]
            payload = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            data = payload.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
        else:
            data = json.dumps(
                {
                    "model": "m-2",
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        return None


@pytest.fixture
def provider():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield LlmConfig(base_url=f"http://127.0.0.1:{server.server_address[1]}", api_key="k", model="m")
    server.shutdown()


@pytest.fixture(autouse=True)
def _clean_usage():
    reset_llm_usage()
    yield
    reset_llm_usage()
    set_principal(None)


async def test_usage_is_counted_per_tenant_and_principal_and_exported(tmp_path, provider):
    import strata.server as server_module
    from strata.config import StrataConfig
    from strata.notebook.llm.client import chat_completion, chat_completion_stream
    from strata.server import ServerState, app

    set_principal(Principal(id="ana", tenant="acme"))
    done = []
    for message in ("tidy up", "and again"):
        async for event in chat_completion_stream(provider, [{"role": "user", "content": message}]):
            if event["type"] == "done":
                done.append(event)
    set_principal(Principal(id="ben", tenant="globex"))
    await chat_completion(provider, [{"role": "user", "content": "hi"}])
    set_principal(None)

    # Each stream reports its own total in its done event.
    assert [(e["input_tokens"], e["output_tokens"]) for e in done] == [(7, 3), (7, 3)]

    server_module._state = ServerState(StrataConfig(cache_dir=tmp_path / "cache"))
    text = TestClient(app).get("/metrics/prometheus").text

    assert 'strata_ai_calls_total{tenant="acme",principal="ana",model="m-1"} 2' in text
    assert 'strata_ai_input_tokens_total{tenant="acme",principal="ana",model="m-1"} 14' in text
    assert 'strata_ai_output_tokens_total{tenant="acme",principal="ana",model="m-1"} 6' in text
    assert 'strata_ai_input_tokens_total{tenant="globex",principal="ben",model="m-2"} 5' in text
    assert 'strata_ai_output_tokens_total{tenant="globex",principal="ben",model="m-2"} 2' in text


def test_nothing_is_exported_before_a_model_is_called(tmp_path):
    import strata.server as server_module
    from strata.config import StrataConfig
    from strata.server import ServerState, app

    server_module._state = ServerState(StrataConfig(cache_dir=tmp_path / "cache"))
    text = TestClient(app).get("/metrics/prometheus").text

    assert "strata_ai_" not in text
