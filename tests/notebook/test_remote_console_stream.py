"""A remote cell's console arrives while it runs, not only when it finishes.

For a long training loop that dies, the streamed tail is the whole diagnostic.
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
        """A finalize capability has its own ``op``, so it cannot be replayed as a log one."""
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
        """A chunk for an unknown build (stale worker, other replica) is dropped, never guessed."""
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
    """The finished-run broadcast does not reprint what was streamed.

    The frontend appends console text, so resending everything would show it twice.
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
        await _broadcast_execution_result("nb1", "cell9", result)

        assert [m["type"] for m in sent] == ["cell_output"], (
            "the streamed console must not be sent again at the end"
        )

    @pytest.mark.asyncio
    async def test_a_cell_that_did_not_stream_still_gets_its_console(self, monkeypatch):
        """Local cells, and workers that ignore the log URL, still get their console."""
        from strata.notebook.executor import CellExecutionResult
        from strata.notebook.ws import _broadcast_execution_result

        sent: list[dict] = []

        async def _capture(notebook_id, message):
            sent.append(message)

        monkeypatch.setattr("strata.notebook.ws._broadcast_message", _capture)

        result = CellExecutionResult(
            cell_id="cell9", success=True, stdout="hello\n", stderr="oops\n"
        )
        await _broadcast_execution_result("nb1", "cell9", result)

        assert [m["type"] for m in sent] == ["cell_console", "cell_console", "cell_output"]


def remote_executor_module():
    from strata.notebook import remote_executor

    return remote_executor


class TestWorkerTeeing:
    @pytest.mark.asyncio
    async def test_output_is_forwarded_as_it_is_produced(self, tmp_path, monkeypatch):
        """Chunks arrive before the process exits."""
        from strata.notebook import remote_executor

        posted: list[tuple[str, str]] = []
        first_chunk_seen = asyncio.Event()

        async def _fake_post(client, log_url, stream, text):
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
    async def test_a_real_cell_reaches_the_pipe_the_worker_reads(self, tmp_path):
        """Driven through the real harness, which replaces ``sys.stdout``.

        A capture that does not write through leaves the pipe empty, so the tests above
        could pass while a worker streams nothing.
        """
        import json

        output_dir = tmp_path / "run"
        output_dir.mkdir()
        manifest = output_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "source": "print('from-the-cell')\nx = 1\n",
                    "inputs": {},
                    "output_dir": str(output_dir),
                    "mounts": {},
                    "tables": {},
                    "env": {},
                    "mutation_defines": [],
                    # What the worker sets when it has somewhere to forward to.
                    "stream_console": True,
                }
            )
        )

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "strata.notebook.harness",
            str(manifest),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(remote_executor_module()._drain(proc, None), timeout=60)

        assert b"from-the-cell" in stdout, "a worker would have streamed nothing"
        result = json.loads((output_dir / "harness-result.json").read_text())
        assert result["stdout"] == "from-the-cell\n", "and the bundle still carries it whole"

    @pytest.mark.asyncio
    async def test_a_run_nobody_is_watching_does_not_pay_for_it(self, tmp_path):
        """A local run does not tee output: its reader discards the pipe and uses the result.

        Teeing would hold a second copy of everything printed in the parent's memory.
        """
        import json

        output_dir = tmp_path / "quiet-run"
        output_dir.mkdir()
        manifest = output_dir / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "source": "print('from-the-cell')\nx = 1\n",
                    "inputs": {},
                    "output_dir": str(output_dir),
                    "mounts": {},
                    "tables": {},
                    "env": {},
                    "mutation_defines": [],
                }
            )
        )

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "strata.notebook.harness",
            str(manifest),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(remote_executor_module()._drain(proc, None), timeout=60)

        assert b"from-the-cell" not in stdout, "the cell's output was copied for nobody"
        result = json.loads((output_dir / "harness-result.json").read_text())
        assert result["stdout"] == "from-the-cell\n", "and the result still carries it whole"

    @pytest.mark.asyncio
    async def test_without_a_log_url_the_output_is_still_collected(self, tmp_path):
        """A worker given no log URL still collects output into the result."""
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


class TestWhatStreamingDropped:
    """The final report sends only what streaming did not deliver.

    Forwarding is best effort (timeouts, expired URLs, other replicas), so the end
    sends the undelivered tail, not everything and not nothing.
    """

    @pytest.mark.asyncio
    async def test_the_part_that_never_arrived_is_sent_at_the_end(self, monkeypatch):
        from strata.notebook.executor import CellExecutionResult
        from strata.notebook.ws import _broadcast_execution_result

        sent: list[dict] = []

        async def _capture(notebook_id, message):
            sent.append(message)

        monkeypatch.setattr("strata.notebook.ws._broadcast_message", _capture)
        console_relay.register("b1", "nb1", "cell9")
        await console_relay.deliver("b1", "stdout", "epoch 1\n")
        sent.clear()

        # The worker's remaining chunks never made it; the bundle has them all.
        result = CellExecutionResult(
            cell_id="cell9", success=True, stdout="epoch 1\nepoch 2\nepoch 3\n", stderr=""
        )
        await _broadcast_execution_result("nb1", "cell9", result)

        console = [m for m in sent if m["type"] == "cell_console"]
        assert [m["payload"]["text"] for m in console] == ["epoch 2\nepoch 3\n"]
        assert [m["type"] for m in sent][-1] == "cell_output"

    @pytest.mark.asyncio
    async def test_a_stream_that_arrived_whole_is_not_repeated(self, monkeypatch):
        from strata.notebook.executor import CellExecutionResult
        from strata.notebook.ws import _broadcast_execution_result

        sent: list[dict] = []

        async def _capture(notebook_id, message):
            sent.append(message)

        monkeypatch.setattr("strata.notebook.ws._broadcast_message", _capture)
        console_relay.register("b1", "nb1", "cell9")
        await console_relay.deliver("b1", "stdout", "epoch 1\n")
        await console_relay.deliver("b1", "stderr", "warn\n")
        sent.clear()

        result = CellExecutionResult(
            cell_id="cell9", success=True, stdout="epoch 1\n", stderr="warn\n"
        )
        await _broadcast_execution_result("nb1", "cell9", result)

        assert [m["type"] for m in sent] == ["cell_output"]


class TestForwardingDoesNotHoldTheCell:
    """A server that never answers must not cost the cell its timeout; the bundle is the record."""

    @pytest.mark.asyncio
    async def test_a_log_server_that_never_answers_still_lets_the_cell_finish(self, tmp_path):
        import asyncio as _asyncio

        from strata.notebook.remote_executor import _drain

        class _Hanging:
            """A stand-in for the POST that never comes back."""

            async def post(self, *args, **kwargs):
                await _asyncio.sleep(3600)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

        import strata.notebook.remote_executor as remote_executor

        original = remote_executor.httpx.AsyncClient
        remote_executor.httpx.AsyncClient = lambda *a, **k: _Hanging()
        try:
            proc = await _asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                "import sys\nfor i in range(200): print('line', i)\nsys.stdout.flush()",
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            stdout, _ = await _asyncio.wait_for(
                _drain(proc, "http://server/v1/builds/b1/log"), timeout=60
            )
        finally:
            remote_executor.httpx.AsyncClient = original

        assert stdout.decode().count("line ") == 200, "the bundle keeps the whole console"


class TestWhatIsShownStaysAPrefix:
    """What was streamed stays a prefix of the whole console.

    The final report sends ``text[delivered:]``; dropping the oldest chunk under
    backpressure would lose the start and repeat the end.
    """

    @pytest.mark.asyncio
    async def test_a_burst_the_link_cannot_keep_up_with_keeps_its_beginning(self, tmp_path):
        import asyncio as _asyncio

        import strata.notebook.remote_executor as remote_executor

        posted: list[str] = []
        release = _asyncio.Event()

        async def _slow_post(client, log_url, stream, text):
            await release.wait()
            posted.append(text)

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(remote_executor, "_post_log_chunk", _slow_post)
        try:
            proc = await _asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                # More chunks than the forwarding queue holds, so it fills.
                "print('X' * 4_000_000)",
                stdout=_asyncio.subprocess.PIPE,
                stderr=_asyncio.subprocess.PIPE,
            )
            drain = _asyncio.create_task(_drain_via(remote_executor, proc))
            await _asyncio.sleep(0.2)
            release.set()
            stdout, _ = await drain
        finally:
            monkeypatch.undo()

        shown = "".join(posted)
        assert stdout.decode().startswith(shown), (
            "what the notebook was shown is no longer the beginning of the console"
        )


async def _drain_via(remote_executor, proc):
    return await remote_executor._drain(proc, "http://server/v1/builds/b1/log")
