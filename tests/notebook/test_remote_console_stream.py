"""A remote cell's console arrives while it runs, not only when it finishes.

Output used to reach the notebook in the result bundle, so a cell dispatched
to a worker was silent for its whole run — most of an hour for a training
loop, and for one that dies at hour three the tail is the whole diagnostic.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from strata.notebook import console_relay
from strata.transforms.signed_urls import URLSigner


@pytest.fixture(autouse=True)
def _clean_relay():
    console_relay._routes.clear()
    console_relay._streamed.clear()
    yield
    console_relay._routes.clear()
    console_relay._streamed.clear()


class TestLogUrlSigning:
    def test_a_log_url_is_signed_and_verifies(self):
        signer = URLSigner(b"secret")
        url = signer.generate_log_url(base_url="http://s", build_id="b1")

        from urllib.parse import parse_qs, urlparse

        params = parse_qs(urlparse(url).query)
        assert signer.verify_log_signature(
            "b1", float(params["expires_at"][0]), params["signature"][0]
        )

    def test_a_finalize_capability_cannot_be_replayed_as_a_log_one(self):
        """Separate ``op``, so one capability is not silently another."""
        from urllib.parse import parse_qs, urlparse

        signer = URLSigner(b"secret")
        finalize = signer.generate_finalize_url(base_url="http://s", build_id="b1").url
        params = parse_qs(urlparse(finalize).query)

        assert not signer.verify_log_signature(
            "b1", float(params["expires_at"][0]), params["signature"][0]
        )

    def test_another_builds_signature_is_refused(self):
        signer = URLSigner(b"secret")
        url = signer.generate_log_url(base_url="http://s", build_id="b1")

        from urllib.parse import parse_qs, urlparse

        params = parse_qs(urlparse(url).query)
        assert not signer.verify_log_signature(
            "b2", float(params["expires_at"][0]), params["signature"][0]
        )

    def test_the_manifest_carries_it(self):
        signer = URLSigner(b"secret")
        manifest = signer.generate_build_manifest(
            base_url="http://s",
            build_id="b1",
            metadata={},
            input_artifacts=[],
            max_output_bytes=1024,
        ).to_dict()

        assert "/v1/builds/b1/log" in manifest["log_url"]


class TestRelayRouting:
    @pytest.mark.asyncio
    async def test_a_chunk_reaches_the_registered_cell(self, monkeypatch):
        sent: list[dict] = []

        async def _capture(notebook_id, message):
            sent.append({"notebook_id": notebook_id, **message})

        monkeypatch.setattr("strata.notebook.ws._broadcast_message", _capture)
        console_relay.register("b1", "nb1", "cell9")

        assert await console_relay.deliver("b1", "stderr", "loss=0.3\n") is True

        assert len(sent) == 1
        assert sent[0]["notebook_id"] == "nb1"
        assert sent[0]["type"] == "cell_console"
        assert sent[0]["payload"] == {
            "cell_id": "cell9",
            "stream": "stderr",
            "text": "loss=0.3\n",
        }

    @pytest.mark.asyncio
    async def test_a_chunk_for_an_unknown_build_is_dropped(self, monkeypatch):
        """A stale worker, or a replica that did not dispatch it.

        Guessing a cell would be worse than showing nothing: the console would
        appear under someone else's cell.
        """
        sent: list = []
        monkeypatch.setattr(
            "strata.notebook.ws._broadcast_message",
            lambda *a, **k: sent.append(a),
        )

        assert await console_relay.deliver("unknown", "stdout", "text") is False
        assert sent == []

    @pytest.mark.asyncio
    async def test_unregistering_stops_delivery(self, monkeypatch):
        sent: list = []
        monkeypatch.setattr(
            "strata.notebook.ws._broadcast_message",
            lambda *a, **k: sent.append(a),
        )
        console_relay.register("b1", "nb1", "cell9")
        console_relay.unregister("b1")

        assert await console_relay.deliver("b1", "stdout", "late") is False
        assert sent == []


class TestNoDoubleDelivery:
    """The finished-run broadcast must not reprint what was streamed.

    The frontend *appends* console text, so re-sending the complete stdout at
    the end would show the whole run a second time underneath itself.
    """

    @pytest.mark.asyncio
    async def test_a_streamed_cell_skips_the_terminal_console_frames(self, monkeypatch):
        from strata.notebook.executor import CellExecutionResult
        from strata.notebook.ws import _broadcast_execution_result

        sent: list[dict] = []

        async def _capture(notebook_id, message):
            sent.append(message)

        monkeypatch.setattr("strata.notebook.ws._broadcast_message", _capture)

        console_relay.register("b1", "nb1", "cell9")
        await console_relay.deliver("b1", "stdout", "epoch 1\n")
        sent.clear()

        result = CellExecutionResult(cell_id="cell9", success=True, stdout="epoch 1\n", stderr="")
        await _broadcast_execution_result("nb1", 5, "cell9", result)

        assert [m["type"] for m in sent] == ["cell_output"], (
            "the streamed console must not be sent again at the end"
        )

    @pytest.mark.asyncio
    async def test_a_cell_that_did_not_stream_still_gets_its_console(self, monkeypatch):
        """Local cells, and workers that ignore the log URL, are unchanged."""
        from strata.notebook.executor import CellExecutionResult
        from strata.notebook.ws import _broadcast_execution_result

        sent: list[dict] = []

        async def _capture(notebook_id, message):
            sent.append(message)

        monkeypatch.setattr("strata.notebook.ws._broadcast_message", _capture)

        result = CellExecutionResult(
            cell_id="cell9", success=True, stdout="hello\n", stderr="oops\n"
        )
        await _broadcast_execution_result("nb1", 5, "cell9", result)

        assert [m["type"] for m in sent] == ["cell_console", "cell_console", "cell_output"]


class TestWorkerTeeing:
    @pytest.mark.asyncio
    async def test_output_is_forwarded_as_it_is_produced(self, tmp_path, monkeypatch):
        """The point of the feature: chunks arrive before the process exits."""
        from strata.notebook import remote_executor

        posted: list[tuple[str, str]] = []
        first_chunk_seen = asyncio.Event()

        async def _fake_post(log_url, stream, text):
            posted.append((stream, text))
            first_chunk_seen.set()

        monkeypatch.setattr(remote_executor, "_post_log_chunk", _fake_post)

        script = tmp_path / "chatty.py"
        script.write_text(
            "import sys,time\n"
            "print('first', flush=True)\n"
            "time.sleep(5)\n"
            "print('second', flush=True)\n"
        )
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        drain = asyncio.create_task(remote_executor._drain(proc, "http://server/log"))
        await asyncio.wait_for(first_chunk_seen.wait(), timeout=10)

        assert proc.returncode is None, "the chunk arrived before the process exited"
        assert ("stdout", "first\n") in posted

        proc.kill()
        await drain

    @pytest.mark.asyncio
    async def test_without_a_log_url_the_output_is_still_collected(self, tmp_path):
        """A worker given no log URL behaves exactly as it did before."""
        from strata.notebook import remote_executor

        script = tmp_path / "quiet.py"
        script.write_text("import sys\nprint('out')\nprint('err', file=sys.stderr)\n")
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await remote_executor._drain(proc, None)

        assert stdout == b"out\n"
        assert stderr == b"err\n"
