"""A remote cell's console arrives while it runs, not only when it finishes.

For a long training loop that dies, the streamed tail is the whole diagnostic.
"""

from __future__ import annotations

import asyncio
import sys

import pytest

from strata.notebook import console_relay
from strata.transforms.signed_urls import URLSigner
from tests.conftest import seed_build_targets


@pytest.fixture(autouse=True)
def _clean_relay():
    console_relay._runs.clear()
    console_relay._streamed.clear()
    yield
    console_relay._runs.clear()
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

        assert await console_relay.deliver("b1", "stderr", 0, "loss=0.3\n") is True

        assert len(sent) == 1
        assert sent[0]["notebook_id"] == "nb1"
        assert sent[0]["type"] == "cell_console"
        assert sent[0]["payload"] == {
            "cell_id": "cell9",
            "stream": "stderr",
            "text": "loss=0.3\n",
            "chunk_seq": 0,
        }

    @pytest.mark.asyncio
    async def test_a_chunk_for_an_unknown_build_is_dropped(self, monkeypatch):
        """A chunk for an unknown build (stale worker, other replica) is dropped, never guessed."""
        sent: list = []
        monkeypatch.setattr(
            "strata.notebook.ws._broadcast_message",
            lambda *a, **k: sent.append(a),
        )

        assert await console_relay.deliver("unknown", "stdout", 0, "text") is False
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

        assert await console_relay.deliver("b1", "stdout", 0, "late") is False
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
        await console_relay.deliver("b1", "stdout", 0, "epoch 1\n")
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

    @pytest.mark.asyncio
    async def test_a_failed_run_carries_its_console_on_cell_error(self, monkeypatch):
        """A client replaces the console from ``cell_error``, as from ``cell_output``."""
        from strata.notebook.executor import CellExecutionResult
        from strata.notebook.ws import _broadcast_execution_result

        sent: list[dict] = []

        async def _capture(notebook_id, message):
            sent.append(message)

        monkeypatch.setattr("strata.notebook.ws._broadcast_message", _capture)

        result = CellExecutionResult(
            cell_id="cell9", success=False, stdout="world\n", stderr="Traceback\n", error="boom"
        )
        await _broadcast_execution_result("nb1", "cell9", result)

        (error,) = [m for m in sent if m["type"] == "cell_error"]
        assert error["payload"]["stdout"] == "world\n"
        assert error["payload"]["stderr"] == "Traceback\n"


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

        async def _fake_post(client, log_url, stream, seq, text):
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
    async def test_a_character_split_across_two_reads_is_posted_whole(self, monkeypatch):
        """The posted text is the bundle's decode, chunk by chunk.

        A per-read decode turns the halves of one character into two replacement
        characters, so the live console is garbled and the server counts one more
        character as shown than the bundle has, and the final tail starts late.
        """
        from strata.notebook import remote_executor

        posted: list[tuple[str, int, str]] = []

        async def _fake_post(client, log_url, stream, seq, text):
            posted.append((stream, seq, text))

        monkeypatch.setattr(remote_executor, "_post_log_chunk", _fake_post)

        whole = b"a" * 8191 + "étail".encode()
        proc = _FakeProcess(stdout=[whole[:8192], whole[8192:]], stderr=[])

        stdout, _ = await remote_executor._drain(proc, "http://server/v1/builds/b1/log")

        assert stdout == whole
        texts = [text for _, _, text in posted]
        assert "".join(texts) == "a" * 8191 + "étail"
        assert "�" not in "".join(texts)
        assert sum(len(text) for text in texts) == len(whole.decode())
        assert [seq for _, seq, _ in posted] == list(range(len(posted)))

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
        await console_relay.deliver("b1", "stdout", 0, "epoch 1\n")
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
        await console_relay.deliver("b1", "stdout", 0, "epoch 1\n")
        await console_relay.deliver("b1", "stderr", 0, "warn\n")
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

        async def _slow_post(client, log_url, stream, seq, text):
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
            # Release only once both pipes are read to the end: the queue is then
            # full and every later chunk was dropped, whatever the machine's speed.
            # Watching the drain too lets its exception surface instead of a hang.
            while not drain.done() and not (proc.stdout.at_eof() and proc.stderr.at_eof()):
                await _asyncio.sleep(0.01)
            release.set()
            stdout, _ = await drain
        finally:
            monkeypatch.undo()

        shown = "".join(posted)
        assert 0 < len(shown) < len(stdout), "the burst was not larger than the queue"
        assert stdout.decode().startswith(shown), (
            "what the notebook was shown is no longer the beginning of the console"
        )


async def _drain_via(remote_executor, proc):
    return await remote_executor._drain(proc, "http://server/v1/builds/b1/log")


class _FakePipe:
    """A pipe that answers each ``read`` with the next scripted chunk, then EOF."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, n: int) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""


class _FakeProcess:
    def __init__(self, stdout: list[bytes], stderr: list[bytes]) -> None:
        self.stdout = _FakePipe(stdout)
        self.stderr = _FakePipe(stderr)

    async def wait(self) -> int:
        return 0


def _capture_broadcasts(monkeypatch, signal: asyncio.Event | None = None) -> list[dict]:
    sent: list[dict] = []

    async def _capture(notebook_id, message):
        sent.append({"notebook_id": notebook_id, **message})
        if signal is not None:
            signal.set()

    monkeypatch.setattr("strata.notebook.ws._broadcast_message", _capture)
    return sent


def _console_texts(sent: list[dict]) -> list[str]:
    return [m["payload"]["text"] for m in sent if m["type"] == "cell_console"]


class TestChunkOrder:
    """Each chunk carries a sequence number, and the notebook shows chunks in it, once each."""

    @pytest.mark.asyncio
    async def test_a_repeated_chunk_is_shown_once(self, monkeypatch):
        sent = _capture_broadcasts(monkeypatch)
        console_relay.register("b1", "nb1", "cell9")

        await console_relay.deliver("b1", "stdout", 0, "epoch 1\n")
        await console_relay.deliver("b1", "stdout", 0, "epoch 1\n")

        assert _console_texts(sent) == ["epoch 1\n"]

    @pytest.mark.asyncio
    async def test_an_early_chunk_waits_for_the_one_before_it(self, monkeypatch):
        sent = _capture_broadcasts(monkeypatch)
        console_relay.register("b1", "nb1", "cell9")

        await console_relay.deliver("b1", "stdout", 1, "epoch 2\n")
        assert sent == []
        await console_relay.deliver("b1", "stdout", 0, "epoch 1\n")

        assert _console_texts(sent) == ["epoch 1\n", "epoch 2\n"]
        assert [m["payload"]["chunk_seq"] for m in sent] == [0, 1]

    @pytest.mark.asyncio
    async def test_streams_are_numbered_apart(self, monkeypatch):
        sent = _capture_broadcasts(monkeypatch)
        console_relay.register("b1", "nb1", "cell9")

        await console_relay.deliver("b1", "stdout", 0, "out\n")
        await console_relay.deliver("b1", "stderr", 0, "err\n")

        assert _console_texts(sent) == ["out\n", "err\n"]

    @pytest.mark.asyncio
    async def test_after_a_lost_chunk_the_shown_console_stays_a_prefix(self, monkeypatch):
        """A chunk the worker dropped never arrives; what follows it waits for the final report."""
        from strata.notebook.executor import CellExecutionResult
        from strata.notebook.ws import _broadcast_execution_result

        sent = _capture_broadcasts(monkeypatch)
        console_relay.register("b1", "nb1", "cell9")
        await console_relay.deliver("b1", "stdout", 0, "epoch 1\n")
        await console_relay.deliver("b1", "stdout", 2, "epoch 3\n")
        console_relay.unregister("b1")

        result = CellExecutionResult(
            cell_id="cell9", success=True, stdout="epoch 1\nepoch 2\nepoch 3\n", stderr=""
        )
        await _broadcast_execution_result("nb1", "cell9", result)

        assert "".join(_console_texts(sent)) == "epoch 1\nepoch 2\nepoch 3\n"


class _FedProc:
    """Pipes the test feeds by hand, so which chunks queue up is not left to timing."""

    def __init__(self):
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.returncode = None

    async def wait(self):
        self.returncode = 0
        return 0


class TestWorkerNumbersChunks:
    @pytest.mark.asyncio
    async def test_each_stream_is_numbered_from_zero(self, monkeypatch):
        from strata.notebook import remote_executor

        posted: list[tuple[str, int]] = []
        arrived = asyncio.Event()

        async def _fake_post(client, log_url, stream, seq, text):
            posted.append((stream, seq))
            if len(posted) % 2 == 0:
                arrived.set()

        monkeypatch.setattr(remote_executor, "_post_log_chunk", _fake_post)
        proc = _FedProc()
        drain = asyncio.create_task(remote_executor._drain(proc, "http://server/log"))
        for text in (b"a", b"b"):
            arrived.clear()
            proc.stdout.feed_data(text)
            proc.stderr.feed_data(text)
            await asyncio.wait_for(arrived.wait(), timeout=10)
        proc.stdout.feed_eof()
        proc.stderr.feed_eof()
        await drain

        assert [seq for stream, seq in posted if stream == "stdout"] == [0, 1]
        assert [seq for stream, seq in posted if stream == "stderr"] == [0, 1]

    @pytest.mark.asyncio
    async def test_a_chunk_dropped_for_a_full_queue_leaves_a_gap(self, monkeypatch):
        """The gap is what tells the server not to show anything after it."""
        from strata.notebook import remote_executor

        posted: list[int] = []
        first_posted = asyncio.Event()

        async def _fake_post(client, log_url, stream, seq, text):
            posted.append(seq)
            first_posted.set()

        monkeypatch.setattr(remote_executor, "_post_log_chunk", _fake_post)
        monkeypatch.setattr(remote_executor, "_LOG_QUEUE_CHUNKS", 1)
        monkeypatch.setattr(remote_executor, "_LOG_READ_CHUNK_BYTES", 1)
        proc = _FedProc()
        proc.stderr.feed_eof()
        # Read in one go before the forwarder posts: chunk 0 queues, 1 and 2 find it full.
        proc.stdout.feed_data(b"abc")
        drain = asyncio.create_task(remote_executor._drain(proc, "http://server/log"))
        await asyncio.wait_for(first_posted.wait(), timeout=10)
        proc.stdout.feed_data(b"d")
        proc.stdout.feed_eof()
        stdout, _ = await drain

        assert stdout == b"abcd"
        assert posted == [0, 3]


class TestLateJoiner:
    """A viewer who opens the notebook mid-run sees what the running cell printed so far."""

    @pytest.fixture
    def session(self, tmp_path):
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import add_cell_to_notebook, create_notebook

        notebook_dir = create_notebook(tmp_path, "Late Joiner")
        add_cell_to_notebook(notebook_dir, "cell1", None)
        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        cell = session.notebook_state.cells[0]
        cell.console_stdout = "last run's output\n"
        cell.console_stderr = "last run's warning\n"
        return session

    @pytest.mark.asyncio
    async def test_notebook_sync_carries_the_running_cells_console(self, session, monkeypatch):
        import json

        from strata.notebook.ws import _handle_notebook_sync

        _capture_broadcasts(monkeypatch)
        session.mark_cell_running("cell1")
        console_relay.register("b1", session.id, "cell1")
        await console_relay.deliver("b1", "stdout", 0, "epoch 1\n")
        await console_relay.deliver("b1", "stdout", 1, "epoch 2\n")

        class _Socket:
            def __init__(self):
                self.sent: list[dict] = []

            async def send_text(self, text):
                self.sent.append(json.loads(text))

        socket = _Socket()
        await _handle_notebook_sync(socket, session, session.id)

        (frame,) = socket.sent
        assert frame["type"] == "notebook_state"
        (cell,) = frame["payload"]["cells"]
        assert cell["console_stdout"] == "epoch 1\nepoch 2\n"
        # Nothing on stderr yet this run, and viewers cleared the last run's on `running`.
        assert cell["console_stderr"] == ""

    @pytest.mark.asyncio
    async def test_a_hit_that_replays_no_console_sends_the_kept_one(self, session, monkeypatch):
        """The frame replaces the console with what a resync of the session shows."""
        from types import SimpleNamespace

        import strata.notebook.ws as notebook_ws
        from strata.notebook.executor import CellExecutionResult

        sent = _capture_broadcasts(monkeypatch)
        monkeypatch.setattr(
            notebook_ws,
            "_get_session_manager",
            lambda: SimpleNamespace(get_session=lambda _id: session),
        )
        result = CellExecutionResult(cell_id="cell1", success=True, cache_hit=True)
        await notebook_ws._broadcast_execution_result(session.id, "cell1", result)

        (output,) = [m for m in sent if m["type"] == "cell_output"]
        assert output["payload"]["stdout"] == "last run's output\n"
        assert output["payload"]["stderr"] == "last run's warning\n"

    def test_a_running_local_cell_does_not_serialize_the_last_runs_console(self, session):
        session.mark_cell_running("cell1")
        cell = session.notebook_state.cells[0]
        data = session.serialize_cell(cell)
        assert (data["console_stdout"], data["console_stderr"]) == ("", "")

    @pytest.mark.asyncio
    async def test_a_finished_run_serializes_its_own_console_again(self, session, monkeypatch):
        _capture_broadcasts(monkeypatch)
        console_relay.register("b1", session.id, "cell1")
        await console_relay.deliver("b1", "stdout", 0, "epoch 1\n")
        console_relay.unregister("b1")

        cell = session.notebook_state.cells[0]
        assert session.serialize_cell(cell)["console_stdout"] == "last run's output\n"

    @pytest.mark.asyncio
    async def test_the_buffer_keeps_only_the_end(self, monkeypatch):
        _capture_broadcasts(monkeypatch)
        monkeypatch.setattr(console_relay, "_TAIL_CHARS", 8)
        console_relay.register("b1", "nb1", "cell9")
        await console_relay.deliver("b1", "stdout", 0, "epoch 1\n")
        await console_relay.deliver("b1", "stdout", 1, "epoch 2\n")

        assert console_relay.live_console("nb1", "cell9") == {"stdout": "epoch 2\n"}

    def test_a_cell_that_has_printed_nothing_has_no_live_console(self):
        console_relay.register("b1", "nb1", "cell9")

        assert console_relay.live_console("nb1", "cell9") == {}
        assert console_relay.live_console("nb1", "other") == {}


def _running_build(store, build_id: str = "b1") -> None:
    store.create_build(build_id=build_id, artifact_id="a", version=1, executor_ref="x@v1")
    store.start_build(build_id)


class TestSharedStore:
    """Chunks for a build another node dispatched wait in the shared build store."""

    @pytest.fixture
    def store(self, tmp_path):
        from strata.transforms.build_store import BuildStore

        seed_build_targets(tmp_path, ["a"], [1])
        return BuildStore(tmp_path / "artifacts.sqlite")

    def test_chunks_are_taken_once_in_order(self, store):
        _running_build(store)
        assert store.append_console_chunk("b1", "stdout", 1, "epoch 2\n")
        assert store.append_console_chunk("b1", "stdout", 0, "epoch 1\n")
        assert not store.append_console_chunk("b1", "stdout", 0, "epoch 1\n"), "a retry"

        assert store.take_console_chunks("b1") == [
            ("stdout", 0, "epoch 1\n"),
            ("stdout", 1, "epoch 2\n"),
        ]
        assert store.take_console_chunks("b1") == []

    def test_a_chunk_stored_between_the_read_and_the_delete_is_not_lost(self, store, tmp_path):
        """Another node fills a gap while this one is taking what it read: the next take sees it."""
        from strata.transforms.build_store import BuildStore

        _running_build(store)
        store.append_console_chunk("b1", "stdout", 3, "three\n")
        store.append_console_chunk("b1", "stdout", 5, "five\n")
        other_node = BuildStore(tmp_path / "artifacts.sqlite")
        real_connect = store._get_connection

        class _Rows:
            def __init__(self, rows):
                self._rows = rows

            def fetchall(self):
                return self._rows

        class _Connection:
            """Lands seq 4 from the other node after the read, before the delete."""

            def __init__(self, conn):
                self._conn = conn

            def execute(self, sql, params=()):
                cursor = self._conn.execute(sql, params)
                if sql.lstrip().startswith("SELECT"):
                    rows = cursor.fetchall()
                    assert other_node.append_console_chunk("b1", "stdout", 4, "four\n")
                    return _Rows(rows)
                return cursor

            def __getattr__(self, name):
                return getattr(self._conn, name)

        store._get_connection = lambda: _Connection(real_connect())
        assert store.take_console_chunks("b1") == [
            ("stdout", 3, "three\n"),
            ("stdout", 5, "five\n"),
        ]
        store._get_connection = real_connect

        assert store.take_console_chunks("b1") == [("stdout", 4, "four\n")]

    def test_a_build_that_is_not_running_stores_nothing(self, store):
        """A late or stale worker cannot fill the table for a build nobody will read."""
        _running_build(store)
        store.complete_build("b1", "a", 1)

        assert not store.append_console_chunk("b1", "stdout", 0, "late\n")
        assert not store.append_console_chunk("unknown", "stdout", 0, "stray\n")
        assert store.take_console_chunks("b1") == []
        assert store.take_console_chunks("unknown") == []

    @pytest.mark.asyncio
    async def test_the_dispatching_node_relays_and_then_clears_them(self, store, monkeypatch):
        seen = asyncio.Event()
        sent = _capture_broadcasts(monkeypatch, signal=seen)
        _running_build(store)
        # Received by another node while this one dispatched the build.
        store.append_console_chunk("b1", "stdout", 0, "epoch 1\n")

        async with console_relay.relaying("b1", "nb1", "cell9", shared_store=store, poll_seconds=0):
            await asyncio.wait_for(seen.wait(), timeout=10)
            store.append_console_chunk("b1", "stdout", 5, "arrives as the run ends\n")

        assert _console_texts(sent) == ["epoch 1\n"]
        assert sent[0]["notebook_id"] == "nb1"
        assert store.take_console_chunks("b1") == [], "what is left is cleared at the end"

    @pytest.mark.asyncio
    async def test_a_single_node_does_not_poll(self, store, monkeypatch):
        taken: list[str] = []
        monkeypatch.setattr(store, "take_console_chunks", lambda build_id: taken.append(build_id))

        async with console_relay.relaying("b1", "nb1", "cell9", poll_seconds=0):
            for _ in range(10):
                await asyncio.sleep(0)

        assert taken == []


class TestThroughARealWorker:
    """A remote cell's print reaches the sockets of the session that ran it, numbered."""

    @pytest.fixture
    def session(self, tmp_path, notebook_executor_server, notebook_build_server):
        from strata.notebook.models import WorkerBackendType, WorkerSpec
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import add_cell_to_notebook, create_notebook

        notebook_dir = create_notebook(tmp_path, "Remote Console")
        add_cell_to_notebook(notebook_dir, "cell1", None)
        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.refresh_environment_runtime()
        spec = {
            "url": notebook_executor_server["execute_url"],
            "transport": "signed",
            "strata_url": notebook_build_server["base_url"],
        }
        notebook_build_server["config"].transforms_config["notebook_workers"] = [
            {"name": "remote", "backend": "executor", "runtime_id": "r", "config": spec}
        ]
        session.notebook_state.workers = [
            WorkerSpec(
                name="remote", backend=WorkerBackendType.EXECUTOR, runtime_id="r", config=spec
            )
        ]
        session.notebook_state.worker = "remote"
        cell = session.notebook_state.cells[0]
        cell.worker = "remote"
        cell.source = "print('from-the-worker')\nx = 1"
        session.re_analyze_cell("cell1")
        return session

    @pytest.mark.asyncio
    async def test_the_chunk_is_broadcast_under_the_session_id(self, session, monkeypatch):
        """The sockets are registered under the session id, not the notebook.toml id."""
        from strata.notebook.executor import CellExecutor

        sent = _capture_broadcasts(monkeypatch)

        result = await CellExecutor(session).execute_cell(
            "cell1", session.notebook_state.cells[0].source
        )

        assert result.success, result.error
        console = [m for m in sent if m["type"] == "cell_console"]
        assert console, "nothing was streamed"
        assert {m["notebook_id"] for m in console} == {session.id}
        assert console[0]["payload"]["chunk_seq"] == 0
        assert "".join(m["payload"]["text"] for m in console) == "from-the-worker\n"

    @pytest.mark.asyncio
    async def test_a_multi_node_server_follows_the_shared_store(
        self, session, notebook_build_server, monkeypatch
    ):
        from strata.notebook.executor import CellExecutor

        followed: list[object] = []
        real = console_relay.relaying

        def _recording(build_id, notebook_id, cell_id, shared_store=None, **kwargs):
            followed.append(shared_store)
            return real(build_id, notebook_id, cell_id, shared_store=shared_store, **kwargs)

        monkeypatch.setattr(console_relay, "relaying", _recording)
        monkeypatch.setattr(notebook_build_server["config"], "node_advertised_url", "http://node-a")

        result = await CellExecutor(session).execute_cell(
            "cell1", session.notebook_state.cells[0].source
        )

        assert result.success, result.error
        assert followed == [notebook_build_server["build_store"]]


class TestLogRoute:
    """``POST /v1/builds/{id}/log`` takes the chunk's ``seq`` and, on another node, stores it."""

    @pytest.fixture
    def route(self, tmp_path, monkeypatch):
        from unittest.mock import MagicMock

        from fastapi.testclient import TestClient

        import strata.server as server_module
        from strata.config import StrataConfig
        from strata.server import app
        from strata.transforms.build_store import get_build_store, reset_build_store

        signer = URLSigner(b"secret")
        state = MagicMock()
        state.config = StrataConfig(cache_dir=tmp_path / "cache", artifact_dir=tmp_path / "a")
        state.url_signer = signer
        monkeypatch.setattr(server_module, "_state", state)
        reset_build_store()
        seed_build_targets(tmp_path, ["a"], [1])
        store = get_build_store(tmp_path / "artifacts.sqlite")
        _running_build(store)
        url = signer.generate_log_url(base_url="http://testserver", build_id="b1")
        yield {"client": TestClient(app), "url": url, "state": state, "store": store}
        reset_build_store()

    def test_a_chunk_without_seq_is_refused(self, route):
        response = route["client"].post(route["url"] + "&stream=stdout", content=b"x")

        assert response.status_code == 422

    def test_a_chunk_reaches_the_dispatching_session_with_its_seq(self, route, monkeypatch):
        sent = _capture_broadcasts(monkeypatch)
        console_relay.register("b1", "nb1", "cell9")

        response = route["client"].post(route["url"] + "&stream=stdout&seq=0", content=b"hi\n")

        assert response.status_code == 202
        assert response.json() == {"delivered": True}
        assert [m["payload"]["chunk_seq"] for m in sent] == [0]

    def test_on_another_node_the_chunk_waits_in_the_shared_store(self, route, monkeypatch):
        monkeypatch.setattr(route["state"].config, "node_advertised_url", "http://node-b")

        response = route["client"].post(route["url"] + "&stream=stderr&seq=3", content=b"warn\n")

        assert response.json() == {"delivered": True}
        assert route["store"].take_console_chunks("b1") == [("stderr", 3, "warn\n")]

    def test_a_single_node_drops_a_chunk_for_a_build_it_is_not_running(self, route):
        response = route["client"].post(route["url"] + "&stream=stdout&seq=0", content=b"x")

        assert response.json() == {"delivered": False}
        assert route["store"].take_console_chunks("b1") == []
