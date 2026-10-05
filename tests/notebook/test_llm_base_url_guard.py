"""A notebook's ``[ai] base_url`` on a service-mode server.

A notebook-chosen base_url goes through the ``@fetch`` guard: private and loopback addresses
are refused unless listed in ``notebook_fetch_allowed_hosts``. The operator's ``ai_base_url``
stays trusted. The fake provider listens on 127.0.0.1, so whether it was reached is the test.
"""

from __future__ import annotations

import http.server
import json
import threading
from types import SimpleNamespace

import pytest

from strata.notebook.llm.config import LlmConfig, llm_config_for_session
from strata.notebook.prompt_executor import execute_prompt_cell


class _Provider(http.server.BaseHTTPRequestHandler):
    hits: list[str]

    def do_POST(self):  # noqa: N802
        self.hits.append(self.path)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body.get("stream"):
            chunks = [
                {"model": "m", "choices": [{"delta": {"content": "internal"}}]},
                {
                    "model": "m",
                    "choices": [],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                },
            ]
            data = (
                "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
            ).encode()
            content_type = "text/event-stream"
        elif self.path.endswith("/messages"):
            data = json.dumps(
                {
                    "model": "m",
                    "content": [{"type": "tool_use", "name": "respond", "input": {"n": 1}}],
                    "usage": {"input_tokens": 1, "output_tokens": 1},
                }
            ).encode()
            content_type = "application/json"
        else:
            data = json.dumps(
                {
                    "model": "m",
                    "choices": [{"message": {"content": "internal"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                }
            ).encode()
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        return None


@pytest.fixture
def provider():
    hits: list[str] = []
    handler = type("Provider", (_Provider,), {"hits": hits})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_address[1]}/v1", hits=hits)
    server.shutdown()
    server.server_close()


def _server(
    monkeypatch,
    *,
    mode: str,
    allowed: list[str] | None = None,
    ai_base_url: str | None = None,
) -> None:
    monkeypatch.setattr(
        "strata.server._state",
        SimpleNamespace(
            config=SimpleNamespace(
                deployment_mode=mode,
                notebook_fetch_allowed_hosts=allowed or [],
                ai_base_url=ai_base_url,
                ai_api_key="operator-key" if ai_base_url else None,
                ai_model=None,
            )
        ),
    )


def _session(tmp_path, source: str, ai: dict[str, str]):
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    notebook_dir = create_notebook(tmp_path, "Prompted", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "p1", language="prompt")
    write_cell(notebook_dir, "p1", source)
    lines = "".join(f"{key} = {json.dumps(value)}\n" for key, value in ai.items())
    with open(notebook_dir / "notebook.toml", "a") as f:
        f.write(f"\n[ai]\n{lines}")
    return NotebookSession(parse_notebook(notebook_dir), notebook_dir)


async def _run(session, *, streaming: bool = False):
    config = llm_config_for_session(session)
    assert config is not None

    async def on_delta(_payload):
        return None

    return await execute_prompt_cell(
        session,
        "p1",
        session.notebook_state.cells[0].source,
        config,
        use_cache=False,
        on_delta=on_delta if streaming else None,
    )


@pytest.mark.parametrize("streaming", [False, True])
async def test_service_mode_refuses_a_notebook_base_url_on_a_private_address(
    tmp_path, provider, monkeypatch, streaming
):
    _server(monkeypatch, mode="service")
    session = _session(tmp_path, "Say hi.", {"base_url": provider.url, "api_key": "k"})

    result = await _run(session, streaming=streaming)

    assert result["success"] is False
    assert "base_url" in result["error"]
    assert "STRATA_NOTEBOOK_FETCH_ALLOWED_HOSTS" in result["error"]
    assert provider.hits == []


async def test_the_anthropic_native_path_is_guarded_too(provider, monkeypatch):
    """A schema on an Anthropic base_url posts to ``/messages`` instead."""
    from strata.notebook.llm.client import chat_completion
    from strata.notebook.llm.config import resolve_llm_config

    _server(monkeypatch, mode="service")
    config = resolve_llm_config(
        {"base_url": f"{provider.url}/anthropic", "api_key": "k"},
        SimpleNamespace(deployment_mode="service", notebook_fetch_allowed_hosts=[]),
    )
    assert isinstance(config, LlmConfig)

    with pytest.raises(RuntimeError, match="base_url"):
        await chat_completion(
            config,
            [{"role": "user", "content": "n?"}],
            output_schema={"type": "object", "properties": {"n": {"type": "integer"}}},
        )
    assert provider.hits == []


async def test_a_host_the_operator_allowed_is_reached(tmp_path, provider, monkeypatch):
    _server(monkeypatch, mode="service", allowed=["127.0.0.1"])
    session = _session(tmp_path, "Say hi.", {"base_url": provider.url, "api_key": "k"})

    result = await _run(session)

    assert result["success"] is True, result["error"]
    assert provider.hits == ["/v1/chat/completions"]


async def test_the_operators_base_url_is_trusted(tmp_path, provider, monkeypatch):
    """The operator's ``STRATA_AI_BASE_URL`` is trusted; a notebook naming it adds nothing."""
    _server(monkeypatch, mode="service", ai_base_url=provider.url)
    inherited = _session(tmp_path / "a", "Say hi.", {"model": "m"})
    restated = _session(tmp_path / "b", "Say hi.", {"base_url": provider.url + "/"})

    for session in (inherited, restated):
        result = await _run(session)
        assert result["success"] is True, result["error"]
    assert len(provider.hits) == 2


async def test_personal_mode_is_unchanged(tmp_path, provider, monkeypatch):
    """A local model server on the author's own machine is the ordinary case."""
    _server(monkeypatch, mode="personal")
    session = _session(tmp_path, "Say hi.", {"base_url": provider.url, "api_key": "k"})

    result = await _run(session, streaming=True)

    assert result["success"] is True, result["error"]
    assert provider.hits == ["/v1/chat/completions"]
