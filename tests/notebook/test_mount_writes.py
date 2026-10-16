"""What a cell writes through a read-write mount is recorded as ``__mount__<name>``.

A read-write cell never cache-hits, so the record is the run's history of its side effect:
the files it created or changed, with digests, linked to the inputs it read.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import httpx
import pytest

from strata.notebook.executor import CellExecutor, _worker_mount_writes
from strata.notebook.models import MountMode, MountSpec
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _session(tmp_path: Path, out: Path) -> NotebookSession:
    nb = create_notebook(tmp_path, "Writes")
    add_cell_to_notebook(nb, "c0", None)
    write_cell(nb, "c0", "n = 2")
    add_cell_to_notebook(nb, "c1", "c0")
    write_cell(
        nb,
        "c1",
        f"# @mount out {out.as_uri()} rw\n"
        '(out / "a.txt").write_text("a" * n)\n'
        '(out / "deep").mkdir(exist_ok=True)\n'
        '(out / "deep" / "b.txt").write_text("b")\n',
    )
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


def _record(session: NotebookSession):
    manager = session.get_artifact_manager()
    records = dict(manager.list_cell_artifacts("c1"))
    return manager, records.get("__mount__out")


class TestTheRecord:
    async def test_both_files_are_listed_with_digests(self, tmp_path):
        out = tmp_path / "out"
        session = _session(tmp_path, out)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("c0", "n = 2")).success
        cell = session.notebook_state.get_cell("c1")
        result = await executor.execute_cell("c1", cell.source)
        assert result.success, result.error

        manager, artifact = _record(session)
        assert artifact is not None
        content = json.loads(manager.artifact_store.read_blob(artifact.id, artifact.version))
        assert content == {
            "uri": out.as_uri(),
            "files": [
                {"path": "a.txt", "size": 2, "sha256": _sha(b"aa")},
                {"path": "deep/b.txt", "size": 1, "sha256": _sha(b"b")},
            ],
        }

    async def test_its_lineage_reaches_what_the_run_read(self, tmp_path):
        from strata.services.artifact import ArtifactService

        session = _session(tmp_path, tmp_path / "out")
        executor = CellExecutor(session)
        await executor.execute_cell("c0", "n = 2")
        await executor.execute_cell("c1", session.notebook_state.get_cell("c1").source)
        manager, artifact = _record(session)

        lineage = ArtifactService().build_lineage(
            manager.artifact_store,
            artifact=artifact,
            artifact_id=artifact.id,
            version=artifact.version,
            tenant_filter=None,
            max_depth=5,
        )

        assert any((node.artifact_id or "").endswith("_cell_c0_var_n") for node in lineage.nodes)

    async def test_each_run_adds_a_version_and_keeps_the_earlier(self, tmp_path):
        """The cell never cache-hits, so every run writes, and every run is history."""
        session = _session(tmp_path, tmp_path / "out")
        executor = CellExecutor(session)
        await executor.execute_cell("c0", "n = 2")
        source = session.notebook_state.get_cell("c1").source
        await executor.execute_cell("c1", source)
        manager, first = _record(session)
        await executor.execute_cell("c1", source)
        _, second = _record(session)

        assert second.version == first.version + 1
        assert manager.artifact_store.read_blob(first.id, first.version) is not None


class TestAWorkersReport:
    """A worker syncs its own staging copy, so it reports what was written in its result."""

    async def test_the_worker_returns_the_files(self, tmp_path, monkeypatch):
        from strata.notebook.remote_bundle import unpack_notebook_output_bundle
        from strata.notebook.remote_executor import (
            NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
            create_notebook_executor_app,
        )

        class _Bucket:
            def exists(self, uri):
                return False

            def put(self, local_path, remote_uri, recursive=True):
                pass

        monkeypatch.setitem(
            sys.modules, "fsspec", types.SimpleNamespace(filesystem=lambda p, **k: _Bucket())
        )
        metadata = {
            "protocol_version": NOTEBOOK_EXECUTOR_PROTOCOL_VERSION,
            "source": '(out / "a.txt").write_text("aa")\n(out / "b.txt").write_text("b")\n',
            "inputs": {},
            "mounts": [{"name": "out", "uri": "s3://bucket/out", "mode": "rw"}],
            "env": {},
            "timeout_seconds": 120,
        }
        transport = httpx.ASGITransport(app=create_notebook_executor_app())
        async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
            response = await client.post(
                "/v1/notebook-execute",
                files={"metadata": ("m.json", json.dumps(metadata).encode(), "application/json")},
                timeout=120,
            )
        assert response.status_code == 200, response.text
        bundle = tmp_path / "bundle.tar"
        bundle.write_bytes(response.content)
        result = unpack_notebook_output_bundle(bundle, tmp_path / "unpacked")

        assert result["success"], result.get("error")
        assert result["mount_writes"] == [
            {
                "mount": "out",
                "uri": "s3://bucket/out",
                "files": [
                    {"path": "a.txt", "size": 2, "sha256": _sha(b"aa")},
                    {"path": "b.txt", "size": 1, "sha256": _sha(b"b")},
                ],
            }
        ]

    def test_the_server_keeps_only_reports_for_the_cells_own_rw_mounts(self):
        specs = [
            MountSpec(name="out", uri="s3://bucket/out", mode=MountMode.READ_WRITE),
            MountSpec(name="data", uri="s3://bucket/in", mode=MountMode.READ_ONLY),
        ]
        good = {"mount": "out", "uri": "s3://bucket/out", "files": []}
        report = {
            "mount_writes": [
                good,
                {"mount": "data", "uri": "s3://bucket/in", "files": []},  # read-only
                {"mount": "out", "uri": "s3://elsewhere", "files": []},  # another location
                {"mount": "ghost", "uri": "s3://bucket/x", "files": []},  # not this cell's
                {"mount": "out", "files": "nope"},  # malformed
            ]
        }

        assert [w.model_dump() for w in _worker_mount_writes(report, specs)] == [good]
        assert _worker_mount_writes({}, specs) == []
        assert _worker_mount_writes(None, specs) == []


class TestAnAgentFindsIt:
    async def test_the_lineage_tool_reads_it_and_names_it_when_asked_wrongly(
        self, tmp_path, monkeypatch
    ):
        from strata.notebook import mcp_server

        session = _session(tmp_path, tmp_path / "out")
        executor = CellExecutor(session)
        await executor.execute_cell("c0", "n = 2")
        await executor.execute_cell("c1", session.notebook_state.get_cell("c1").source)
        monkeypatch.setattr(mcp_server, "_live_session", lambda manager, sid: session)

        chain = mcp_server._lineage(None, "sid", "c1", "__mount__out")
        with pytest.raises(ValueError, match="__mount__out") as excinfo:
            mcp_server._lineage(None, "sid", "c1", "missing")

        assert any(step["artifact_id"].endswith("_cell_c0_var_n") for step in chain["steps"])
        assert "__console__" not in str(excinfo.value)
