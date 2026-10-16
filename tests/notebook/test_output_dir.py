"""``# @output_dir <name>``: a folder a cell fills, kept as files and a manifest output.

What is in the folder when the run ends is the output: each file stored once by digest,
the manifest stored as the variable, and a downstream cell gets the folder back as a local
path. A failed run keeps its files under a failed version that is never a cache hit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from strata.notebook.annotations import parse_annotations
from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

WRITER = (
    "# @output_dir ckpt\n"
    "(ckpt / 'epochs').mkdir()\n"
    "for i in range(2):\n"
    "    (ckpt / 'epochs' / f'e{i}.bin').write_bytes(bytes([i]) * 10)\n"
    "(ckpt / 'final.txt').write_text('done')\n"
)
READER = "seen = sorted(p.relative_to(ckpt).as_posix() for p in ckpt.rglob('*') if p.is_file())\n"


def _session(tmp_path: Path, writer: str = WRITER, reader: str | None = READER):
    nb = create_notebook(tmp_path, "Folders")
    add_cell_to_notebook(nb, "w", None)
    write_cell(nb, "w", writer)
    if reader is not None:
        add_cell_to_notebook(nb, "r", "w")
        write_cell(nb, "r", reader)
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


def _manifest(session, version: int | None = None) -> tuple[dict, str]:
    manager = session.get_artifact_manager()
    artifact_id = manager.cell_artifact_id("w", "ckpt")
    store = manager.artifact_store
    artifact = (
        store.get_latest_version(artifact_id)
        if version is None
        else store.get_artifact(artifact_id, version)
    )
    return json.loads(store.read_blob(artifact.id, artifact.version)), artifact.state


class TestTheAnnotation:
    def test_it_parses(self):
        assert parse_annotations("# @output_dir ckpt\n# @output_dir logs\nx = 1").output_dirs == [
            "ckpt",
            "logs",
        ]

    def test_a_name_that_is_not_an_identifier_is_dropped(self):
        assert parse_annotations("# @output_dir 2ckpt\nx = 1").output_dirs == []

    def test_the_folder_is_a_define_of_its_cell(self, tmp_path):
        session = _session(tmp_path)

        assert "ckpt" in session.notebook_state.get_cell("w").defines
        assert "w" in session.notebook_state.get_cell("r").upstream_ids


class TestAFolderOutput:
    async def test_a_downstream_cell_reads_the_folder(self, tmp_path):
        session = _session(tmp_path)
        executor = CellExecutor(session)

        written = await executor.execute_cell("w", WRITER)
        assert written.success, written.error
        read = await executor.execute_cell("r", READER)

        assert read.success, read.error
        assert read.outputs["seen"]["preview"] == ["epochs/e0.bin", "epochs/e1.bin", "final.txt"]
        manifest, state = _manifest(session)
        assert state == "ready"
        assert manifest["files"]["final.txt"]["size"] == 4

    async def test_a_leaf_folder_is_kept(self, tmp_path):
        """Nothing reads it: a checkpoint is the point of the cell, not a downstream value."""
        session = _session(tmp_path, reader=None)

        assert (await CellExecutor(session).execute_cell("w", WRITER)).success

        manifest, state = _manifest(session)
        assert state == "ready"
        assert sorted(manifest["files"]) == ["epochs/e0.bin", "epochs/e1.bin", "final.txt"]

    async def test_a_second_run_is_a_cache_hit(self, tmp_path):
        session = _session(tmp_path)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("w", WRITER)).success

        again = await executor.execute_cell("w", WRITER)

        assert again.cache_hit

    async def test_a_file_that_did_not_change_is_stored_once(self, tmp_path):
        session = _session(tmp_path, reader=None)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("w", WRITER)).success
        edited = WRITER + "(ckpt / 'extra.txt').write_text('more')\n"
        cell = session.notebook_state.get_cell("w")
        cell.source = edited
        session.re_analyze_cell("w")
        assert (await executor.execute_cell("w", edited)).success

        manifest, _ = _manifest(session)
        manager = session.get_artifact_manager()
        sha = manifest["files"]["final.txt"]["sha256"]
        file_id = manager.cell_artifact_id("w", f"__file__{sha}")
        assert manager.artifact_store.get_latest_version(file_id).version == 1
        assert sorted(manifest["files"]) == [
            "epochs/e0.bin",
            "epochs/e1.bin",
            "extra.txt",
            "final.txt",
        ]

    async def test_a_failed_run_keeps_what_it_wrote_and_is_not_cached(self, tmp_path):
        failing = (
            "# @output_dir ckpt\n"
            "(ckpt / 'e0.bin').write_bytes(b'x' * 5)\n"
            "raise RuntimeError('lost')\n"
        )
        session = _session(tmp_path, writer=failing, reader=None)
        executor = CellExecutor(session)

        result = await executor.execute_cell("w", failing)

        assert not result.success
        # A failed version is not the latest value; it is read by its number.
        manifest, state = _manifest(session, version=1)
        assert state == "failed"
        assert list(manifest["files"]) == ["e0.bin"]
        manager = session.get_artifact_manager()
        file_id = manager.cell_artifact_id("w", f"__file__{manifest['files']['e0.bin']['sha256']}")
        assert manager.artifact_store.read_blob(file_id, 1) == b"x" * 5
        assert not (await executor.execute_cell("w", failing)).cache_hit

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    async def test_a_link_in_the_folder_is_not_kept(self, tmp_path):
        secret = tmp_path / "server-only.txt"
        secret.write_text("secret")
        writer = (
            "# @output_dir ckpt\n"
            "(ckpt / 'mine.txt').write_text('mine')\n"
            f"(ckpt / 'leak.txt').symlink_to({str(secret)!r})\n"
        )
        session = _session(tmp_path, writer=writer, reader=None)

        assert (await CellExecutor(session).execute_cell("w", writer)).success

        manifest, _ = _manifest(session)
        assert list(manifest["files"]) == ["mine.txt"]


class TestNotYet:
    async def test_a_loop_cell_is_refused(self, tmp_path):
        source = "# @loop max_iter=2 carry=n\n# @output_dir ckpt\nn = 1\n"
        session = _session(tmp_path, writer=source, reader=None)

        result = await CellExecutor(session).execute_cell("w", source)

        assert not result.success
        assert "@output_dir does not work in a # @loop cell" in result.error

    async def test_a_worker_that_cannot_be_asked_is_refused_before_it_runs(self, tmp_path):
        source = "# @worker gpu\n# @output_dir ckpt\n(ckpt / 'a').write_text('a')\n"
        session = _session(tmp_path, writer=source, reader=None)
        from strata.notebook.models import WorkerBackendType, WorkerSpec

        session.notebook_state.workers = [
            WorkerSpec(
                name="gpu",
                backend=WorkerBackendType.EXECUTOR,
                config={"url": "http://127.0.0.1:9/v1/execute"},
            )
        ]

        result = await CellExecutor(session).execute_cell("w", source)

        assert not result.success
        assert "could not be asked whether it returns # @output_dir folders" in result.error

    async def test_an_older_worker_is_refused_before_it_runs(self, tmp_path, monkeypatch):
        source = "# @worker gpu\n# @output_dir ckpt\n(ckpt / 'a').write_text('a')\n"
        session = _session(tmp_path, writer=source, reader=None)
        from strata.notebook import workers
        from strata.notebook.models import WorkerBackendType, WorkerSpec

        async def older(worker, feature):
            return False

        monkeypatch.setattr(workers, "worker_advertises", older)
        session.notebook_state.workers = [
            WorkerSpec(
                name="gpu",
                backend=WorkerBackendType.EXECUTOR,
                config={"url": "http://127.0.0.1:9/v1/execute"},
            )
        ]

        result = await CellExecutor(session).execute_cell("w", source)

        assert not result.success
        assert "does not return # @output_dir folders" in result.error

    async def test_run_all_runs_it_on_its_own(self, tmp_path):
        """A batch shares one process and has no run directory per cell to collect from."""
        from strata.notebook.executor import is_cell_batchable

        session = _session(tmp_path)
        executor = CellExecutor(session)

        assert not is_cell_batchable(executor, session.notebook_state.get_cell("w"))
        assert is_cell_batchable(executor, session.notebook_state.get_cell("r"))


class TestTheBundle:
    """A worker returns each folder's files in the bundle, streamed past the member cap."""

    def test_files_larger_than_the_member_cap_round_trip(self, tmp_path, monkeypatch):
        from strata.notebook.output_dirs import OUTPUT_DIRS
        from strata.notebook.remote_bundle import (
            pack_notebook_output_bundle,
            unpack_notebook_output_bundle,
        )

        monkeypatch.setenv("STRATA_NOTEBOOK_MAX_BUNDLE_MEMBER_BYTES", "4096")
        run = tmp_path / "run"
        folder = run / OUTPUT_DIRS / "ckpt"
        (folder / "deep").mkdir(parents=True)
        big = bytes(range(256)) * 40  # 10240 bytes, past the 4096-byte cap
        (folder / "deep" / "big.bin").write_bytes(big)
        (folder / "empty.txt").write_bytes(b"")
        if sys.platform != "win32":
            (folder / "leak").symlink_to(tmp_path / "elsewhere")
        bundle = tmp_path / "bundle.tar"

        pack_notebook_output_bundle(bundle, {"success": True, "variables": {}}, run)
        out = tmp_path / "unpacked"
        unpack_notebook_output_bundle(bundle, out)

        got = out / OUTPUT_DIRS / "ckpt"
        assert (got / "deep" / "big.bin").read_bytes() == big
        assert (got / "empty.txt").read_bytes() == b""
        assert sorted(p.relative_to(got).as_posix() for p in got.rglob("*") if p.is_file()) == [
            "deep/big.bin",
            "empty.txt",
        ]

    @pytest.mark.parametrize("rel", ["../escape.bin", "/abs.bin"])
    def test_a_path_leaving_the_folder_is_refused(self, tmp_path, rel):
        import io
        import tarfile

        from strata.notebook.remote_bundle import SCHEMA_VERSION, unpack_notebook_output_bundle

        bundle = tmp_path / "bundle.tar"
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "success": True,
            "variables": {},
            "stdout_file": "stdout.txt",
            "stderr_file": "stderr.txt",
            "output_dirs": {"ckpt": {rel: 1}},
        }
        with tarfile.open(bundle, "w") as tar:
            for name, data in (
                ("manifest.json", json.dumps(manifest).encode()),
                ("stdout.txt", b""),
                ("stderr.txt", b""),
                (f"dirs/ckpt/{rel}", b"x"),
            ):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

        with pytest.raises(ValueError, match="Invalid bundle output file"):
            unpack_notebook_output_bundle(bundle, tmp_path / "unpacked")
        assert not (tmp_path / "escape.bin").exists()


class TestOnAWorker:
    @pytest.mark.parametrize("transport", ["direct", "signed"])
    async def test_a_worker_returns_the_folder_and_a_local_cell_reads_it(
        self, tmp_path, transport, notebook_executor_server, notebook_personal_server
    ):
        from strata.notebook.models import WorkerBackendType, WorkerSpec

        writer = "# @worker remote\n" + WRITER
        session = _session(tmp_path, writer=writer)
        config = {"url": notebook_executor_server["execute_url"], "transport": transport}
        if transport == "signed":
            config["strata_url"] = notebook_personal_server["base_url"]
        session.notebook_state.workers = [
            WorkerSpec(
                name="remote", backend=WorkerBackendType.EXECUTOR, runtime_id="r", config=config
            )
        ]
        executor = CellExecutor(session)

        written = await executor.execute_cell("w", writer)
        assert written.success, written.error
        assert written.execution_method == "executor"
        read = await executor.execute_cell("r", READER)

        assert read.success, read.error
        assert read.outputs["seen"]["preview"] == ["epochs/e0.bin", "epochs/e1.bin", "final.txt"]

    async def test_a_failed_run_on_a_worker_keeps_what_it_wrote(
        self, tmp_path, notebook_executor_server
    ):
        from strata.notebook.models import WorkerBackendType, WorkerSpec

        failing = (
            "# @worker remote\n# @output_dir ckpt\n"
            "(ckpt / 'e0.bin').write_bytes(b'x' * 5)\n"
            "raise RuntimeError('lost')\n"
        )
        session = _session(tmp_path, writer=failing, reader=None)
        session.notebook_state.workers = [
            WorkerSpec(
                name="remote",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="r",
                config={"url": notebook_executor_server["execute_url"]},
            )
        ]

        result = await CellExecutor(session).execute_cell("w", failing)

        assert not result.success
        manifest, state = _manifest(session, version=1)
        assert state == "failed"
        assert list(manifest["files"]) == ["e0.bin"]


def test_a_folder_name_cannot_leave_the_run_directory(tmp_path, monkeypatch):
    """The parser takes identifiers only; the check holds even if it did not."""
    from strata.notebook import annotations
    from strata.notebook.output_dirs import mount_entries

    real = annotations.parse_annotations

    def loose(source):
        parsed = real(source)
        parsed.output_dirs = ["../escape"]
        return parsed

    monkeypatch.setattr(annotations, "parse_annotations", loose)

    with pytest.raises(ValueError, match="leaves the run directory"):
        mount_entries("x = 1", tmp_path / "run")
    assert not (tmp_path / "escape").exists()


class TestTheUploadUrl:
    def test_it_verifies_and_is_not_a_log_url(self):
        from urllib.parse import parse_qs, urlparse

        from strata.transforms.signed_urls import URLSigner

        signer = URLSigner(b"k" * 32)
        url = signer.generate_output_file_url(base_url="http://s", build_id="b1")
        query = parse_qs(urlparse(url).query)
        expires, signature = float(query["expires_at"][0]), query["signature"][0]

        assert "/v1/builds/b1/output-file" in url
        assert signer.verify_output_file_signature("b1", expires, signature)
        assert not signer.verify_output_file_signature("b2", expires, signature)
        assert not signer.verify_log_signature("b1", expires, signature)

        log = parse_qs(urlparse(signer.generate_log_url(base_url="http://s", build_id="b1")).query)
        assert not signer.verify_output_file_signature(
            "b1", float(log["expires_at"][0]), log["signature"][0]
        )


class TestTheUploadRoute:
    """``POST /v1/builds/{id}/output-file`` stores a file for the run receiving it."""

    @staticmethod
    def _receiver(tmp_path):
        from strata.notebook.output_file_relay import Receiver

        session = _session(tmp_path, reader=None)
        return session, Receiver(
            session.get_artifact_manager(),
            "w",
            upload_dir=tmp_path / "uploads",
            folders=frozenset({"ckpt"}),
        )

    @staticmethod
    def _url(build_id: str, **params: str) -> str:
        from urllib.parse import urlencode

        from strata.server import get_state

        url = get_state().url_signer.generate_output_file_url(
            base_url="http://127.0.0.1", build_id=build_id
        )
        return url.split("http://127.0.0.1", 1)[1] + "&" + urlencode(params)

    def test_a_file_reaches_the_receiving_run(self, tmp_path, notebook_personal_server):
        import httpx

        from strata.notebook.output_file_relay import receiving

        session, receiver = self._receiver(tmp_path)
        base = notebook_personal_server["base_url"]
        with receiving("b1", receiver):
            response = httpx.post(
                base + self._url("b1", folder="ckpt", path="deep/e0.bin"), content=b"x" * 7
            )

        assert response.status_code == 201, response.text
        entry = receiver.received()["ckpt"]["deep/e0.bin"]
        manager = session.get_artifact_manager()
        file_id = manager.cell_artifact_id("w", f"__file__{entry['sha256']}")
        assert manager.artifact_store.read_blob(file_id, entry["version"]) == b"x" * 7
        assert list((tmp_path / "uploads").iterdir()) == []

    @pytest.mark.parametrize(
        ("build", "params", "status"),
        [
            ("b1", {"folder": "ckpt", "path": "../escape"}, 400),
            ("b1", {"folder": "../x", "path": "e0.bin"}, 400),
            ("b1", {"folder": "ckpt", "path": "/abs"}, 400),
            ("other", {"folder": "ckpt", "path": "e0.bin"}, 409),
            ("b1", {"folder": "undeclared", "path": "e0.bin"}, 400),
        ],
    )
    def test_a_bad_upload_is_refused(
        self, tmp_path, notebook_personal_server, build, params, status
    ):
        import httpx

        from strata.notebook.output_file_relay import receiving

        _, receiver = self._receiver(tmp_path)
        base = notebook_personal_server["base_url"]
        with receiving("b1", receiver):
            response = httpx.post(base + self._url(build, **params), content=b"x")

        assert response.status_code == status, response.text
        assert receiver.received() == {}

    def test_a_forged_signature_is_refused(self, tmp_path, notebook_personal_server):
        import httpx

        from strata.notebook.output_file_relay import receiving

        _, receiver = self._receiver(tmp_path)
        url = self._url("b1", folder="ckpt", path="e0.bin").replace("signature=", "signature=0")
        with receiving("b1", receiver):
            response = httpx.post(notebook_personal_server["base_url"] + url, content=b"x")

        assert response.status_code == 403
        assert receiver.received() == {}


class TestStreamingFromAWorker:
    """On the signed transport a worker uploads settled files while the cell runs."""

    @staticmethod
    def _session_on_worker(tmp_path, writer, executor_server, personal_server):
        from strata.notebook.models import WorkerBackendType, WorkerSpec

        session = _session(tmp_path, writer=writer)
        session.notebook_state.workers = [
            WorkerSpec(
                name="remote",
                backend=WorkerBackendType.EXECUTOR,
                runtime_id="r",
                config={
                    "url": executor_server["execute_url"],
                    "transport": "signed",
                    "strata_url": personal_server["base_url"],
                },
            )
        ]
        return session

    async def test_files_arrive_while_it_runs_and_the_result_only_names_them(
        self, tmp_path, monkeypatch, notebook_executor_server, notebook_personal_server
    ):
        # e0 is uploaded, then deleted: the result's list keeps it out of the folder.
        monkeypatch.setenv("STRATA_WORKER_OUTPUT_DIR_POLL_SECONDS", "0.1")
        writer = (
            "# @worker remote\n# @output_dir ckpt\nimport time\n"
            "(ckpt / 'e0.bin').write_bytes(b'0' * 9)\n"
            "(ckpt / 'keep.bin').write_bytes(b'k' * 9)\n"
            "time.sleep(1)\n"
            "(ckpt / 'e0.bin').unlink()\n"
            "(ckpt / 'e1.bin').write_bytes(b'1' * 9)\n"
        )
        session = self._session_on_worker(
            tmp_path, writer, notebook_executor_server, notebook_personal_server
        )
        executor = CellExecutor(session)

        written = await executor.execute_cell("w", writer)

        assert written.success, written.error
        assert sorted(executor._received_output_files["w"]["ckpt"]) == [
            "e0.bin",
            "e1.bin",
            "keep.bin",
        ]
        manifest, state = _manifest(session)
        assert state == "ready"
        assert sorted(manifest["files"]) == ["e1.bin", "keep.bin"]

    async def test_a_lost_worker_leaves_what_it_uploaded(
        self, tmp_path, monkeypatch, notebook_executor_server, notebook_personal_server
    ):
        """The run times out on the worker: no result comes back, the uploads stay.

        Only the uploads: a folder of the same name where the server runs is not the run's.
        """
        stray = tmp_path / "cwd" / "output_dirs" / "ckpt"
        stray.mkdir(parents=True)
        (stray / "stray.bin").write_bytes(b"not the run's")
        monkeypatch.chdir(tmp_path / "cwd")
        monkeypatch.setenv("STRATA_WORKER_OUTPUT_DIR_POLL_SECONDS", "0.1")
        writer = (
            "# @worker remote\n# @timeout 4\n# @output_dir ckpt\nimport time\n"
            "(ckpt / 'e0.bin').write_bytes(b'0' * 9)\n"
            "time.sleep(60)\n"
        )
        session = self._session_on_worker(
            tmp_path, writer, notebook_executor_server, notebook_personal_server
        )

        result = await CellExecutor(session).execute_cell("w", writer)

        assert not result.success
        manifest, state = _manifest(session, version=1)
        assert state == "failed"
        assert list(manifest["files"]) == ["e0.bin"]

    async def test_a_folder_emptied_before_the_end_is_empty(
        self, tmp_path, monkeypatch, notebook_executor_server, notebook_personal_server
    ):
        """Every file sent was deleted: the result lists the folder with none of them."""
        monkeypatch.setenv("STRATA_WORKER_OUTPUT_DIR_POLL_SECONDS", "0.1")
        writer = (
            "# @worker remote\n# @output_dir ckpt\nimport time\n"
            "(ckpt / 'tmp.bin').write_bytes(b't' * 9)\n"
            "time.sleep(1)\n"
            "(ckpt / 'tmp.bin').unlink()\n"
        )
        session = self._session_on_worker(
            tmp_path, writer, notebook_executor_server, notebook_personal_server
        )
        executor = CellExecutor(session)

        written = await executor.execute_cell("w", writer)

        assert written.success, written.error
        assert list(executor._received_output_files["w"]["ckpt"]) == ["tmp.bin"]
        manifest, _ = _manifest(session)
        assert manifest["files"] == {}

    async def test_a_cancelled_run_keeps_what_arrived(
        self, tmp_path, monkeypatch, notebook_executor_server, notebook_personal_server
    ):
        import asyncio

        from strata.notebook import output_file_relay

        monkeypatch.setenv("STRATA_WORKER_OUTPUT_DIR_POLL_SECONDS", "0.1")
        writer = (
            "# @worker remote\n# @output_dir ckpt\nimport time\n"
            "(ckpt / 'e0.bin').write_bytes(b'0' * 9)\n"
            "time.sleep(60)\n"
        )
        session = self._session_on_worker(
            tmp_path, writer, notebook_executor_server, notebook_personal_server
        )
        run = asyncio.create_task(CellExecutor(session).execute_cell("w", writer))
        for _ in range(200):
            if any(r.received() for r in output_file_relay._receivers.values()):
                break
            await asyncio.sleep(0.05)

        run.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run

        manifest, state = _manifest(session, version=1)
        assert state == "failed"
        assert list(manifest["files"]) == ["e0.bin"]


class TestTheStreamer:
    """The worker's poller: never fails the run, and does not resend what is refused."""

    async def test_a_file_gone_between_listing_and_stat_is_skipped(self, tmp_path, monkeypatch):
        from strata.notebook import output_dirs as module
        from strata.notebook.remote_executor import _OutputDirStreamer

        (tmp_path / "ckpt").mkdir()
        (tmp_path / "ckpt" / "here.bin").write_bytes(b"x")
        monkeypatch.setattr(module, "regular_files", lambda folder: ["gone.tmp", "here.bin"])

        states = _OutputDirStreamer(tmp_path, "http://s/u?x=1")._states()

        assert list(states) == [("ckpt", "here.bin")]

    async def test_a_refused_file_is_not_sent_again(self, tmp_path):
        import httpx

        from strata.notebook.remote_executor import _OutputDirStreamer

        (tmp_path / "ckpt").mkdir()
        (tmp_path / "ckpt" / "big.bin").write_bytes(b"x" * 10)
        posts: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            posts.append(str(request.url))
            return httpx.Response(413, json={"detail": "too large"})

        streamer = _OutputDirStreamer(tmp_path, "http://s/u?x=1")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            key, state = next(iter(streamer._states().items()))
            for _ in range(3):
                if streamer._due(key, state):
                    await streamer._upload(client, key, state)

        assert len(posts) == 1


def test_a_result_that_lists_no_files_keeps_none_of_the_uploads(tmp_path):
    """An empty list is an answer (nothing held), unlike no list at all (a lost worker)."""
    import json as json_module

    from strata.notebook.output_dirs import STREAMED

    session = _session(tmp_path, reader=None)
    executor = CellExecutor(session)
    executor._received_output_files["w"] = {
        "ckpt": {"gone.bin": {"sha256": "0" * 64, "size": 1, "version": 1}}
    }
    run = tmp_path / "run"
    run.mkdir()
    (run / STREAMED).write_text(json_module.dumps({}))

    listed_none = executor._stage_output_dir(
        "w", "ckpt", run, "p" * 64, {}, source_hash="", source="", env_hash=""
    )
    lost = executor._stage_output_dir(
        "w", "ckpt", None, "q" * 64, {}, source_hash="", source="", env_hash=""
    )

    manager = session.get_artifact_manager()
    read = lambda staged: json_module.loads(  # noqa: E731
        manager.artifact_store.blob_store.read_blob(staged.artifact_id, staged.version)
    )
    assert read(listed_none)["files"] == {}
    assert list(read(lost)["files"]) == ["gone.bin"]
