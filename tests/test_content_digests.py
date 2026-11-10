"""Every artifact version carries a digest, so two machines can compare outputs.

``byte_size`` and row count cannot tell apart two outputs of the same shape.
"""

from __future__ import annotations

import hashlib

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from vestigo.store import ArtifactStore

from tests.conftest import prepared_venv


def _ipc_bytes(values: list[int]) -> bytes:
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, pa.schema([("id", pa.int64())])) as writer:
        writer.write_batch(pa.RecordBatch.from_pydict({"id": values}))
    return sink.getvalue().to_pybytes()


@pytest.fixture
def store(tmp_path):
    return ArtifactStore(tmp_path / "artifacts")


def _store_one(store: ArtifactStore, artifact_id: str, blob: bytes, rows: int = 3, **kwargs):
    version = store.create_artifact(artifact_id, hashlib.sha256(artifact_id.encode()).hexdigest())
    store.write_blob(artifact_id, version, blob)
    return store.finalize_artifact(
        artifact_id, version, '{"fields": []}', rows, len(blob), **kwargs
    )


class TestOnTheWire:
    """The digest has to leave the store, or comparing machines needs one."""

    @pytest.fixture
    def served(self, tmp_path):
        from tests.conftest import run_server_with_context

        artifact_dir = tmp_path / "served"
        with run_server_with_context(tmp_path / "cache", artifact_dir, "personal") as ctx:
            store = ArtifactStore(artifact_dir)
            blob = _ipc_bytes([1, 2, 3])
            upstream = _store_one(store, "rows", blob)
            version = store.create_artifact(
                "model",
                "e" * 64,
                input_versions={"strata://artifact/rows@v=1": "rows@v=1"},
            )
            store.write_blob("model", version, blob)
            store.finalize_artifact("model", version, '{"fields": []}', 3, len(blob))
            yield ctx.base_url, upstream.content_sha256

    def test_the_artifact_record_carries_it(self, served):
        import httpx

        base_url, digest = served

        body = httpx.get(f"{base_url}/v1/artifacts/rows/v/1", timeout=10).json()

        assert body["content_sha256"] == digest

    def test_every_step_of_the_lineage_carries_it(self, served):
        """Each lineage step carries a digest so a diff can say where outputs differ."""
        import httpx

        base_url, digest = served

        body = httpx.get(f"{base_url}/v1/artifacts/model/v/1/lineage", timeout=10).json()

        artifacts = [n for n in body["nodes"] if n["type"] == "artifact"]
        assert len(artifacts) == 2
        assert all(n["content_sha256"] for n in artifacts)
        assert next(n for n in artifacts if n["artifact_id"] == "rows")["content_sha256"] == digest

    def test_a_provenance_hit_carries_it(self, served):
        """A team-cache hit carries the digest so a puller can check the bytes it got."""
        import httpx

        base_url, digest = served

        body = httpx.get(f"{base_url}/v1/artifacts/by-provenance/{'e' * 64}", timeout=10).json()

        assert body["content_sha256"]


class TestRunReport:
    """``strata run --format json``: the report two machines diff."""

    def _notebook(self, tmp_path, value: str):
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        nb = create_notebook(tmp_path, "Digests", initialize_environment=False)
        prepared_venv(nb)
        add_cell_to_notebook(nb, "rows", None, language="python")
        write_cell(nb, "rows", f"rows = {value}\n")
        add_cell_to_notebook(nb, "total", "rows", language="python")
        write_cell(nb, "total", "total = sum(rows)\nprint(total)\n")
        return nb

    def _run(self, nb, capsys):
        import json as _json

        from strata.notebook.cli import run_main

        assert run_main([str(nb), "--no-sync", "--force", "--format", "json"]) == 0
        return _json.loads(capsys.readouterr().out)

    def test_each_cell_reports_its_outputs_and_their_digests(self, tmp_path, capsys):
        payload = self._run(self._notebook(tmp_path, "[1, 2, 3]"), capsys)

        rows = next(c for c in payload["cells"] if c["id"] == "rows")
        assert rows["provenance_hash"]
        assert [o["name"] for o in rows["outputs"]] == ["rows"]
        assert rows["outputs"][0]["content_sha256"]

    def test_two_notebooks_computing_differently_report_different_digests(self, tmp_path, capsys):
        """Both runs are green and similar in duration, but the digests differ."""
        mine = self._run(self._notebook(tmp_path / "mine", "[1, 2, 3]"), capsys)
        yours = self._run(self._notebook(tmp_path / "yours", "[1, 2, 4]"), capsys)

        def digest(payload, cell_id):
            cell = next(c for c in payload["cells"] if c["id"] == cell_id)
            return cell["outputs"][0]["content_sha256"]

        assert {c["status"] for c in mine["cells"]} == {"ok"}
        assert {c["status"] for c in yours["cells"]} == {"ok"}
        assert digest(mine, "rows") != digest(yours, "rows")

    def test_two_runs_of_the_same_notebook_agree_on_the_bytes(self, tmp_path, capsys):
        """A forced rerun moves the version but not the bytes, so digests agree."""
        nb = self._notebook(tmp_path, "[1, 2, 3]")

        first = self._run(nb, capsys)
        second = self._run(nb, capsys)

        def digests(payload):
            return {
                c["id"]: [o["content_sha256"] for o in c.get("outputs", [])]
                for c in payload["cells"]
            }

        assert digests(first)["rows"] == digests(second)["rows"]
        assert digests(first)["rows"] != [None]

    @staticmethod
    def _outputs(payload, cell_id):
        cell = next(c for c in payload["cells"] if c["id"] == cell_id)
        return {o["name"]: o["content_sha256"] for o in cell.get("outputs", [])}

    def test_a_leaf_cell_reports_what_it_defines(self, tmp_path, capsys):
        """The last cell is often the result, and nothing downstream reads it."""
        payload = self._run(self._notebook(tmp_path, "[1, 2, 3]"), capsys)

        total = self._outputs(payload, "total")
        assert total["total"]

    def test_two_notebooks_differ_at_the_leaf(self, tmp_path, capsys):
        mine = self._run(self._notebook(tmp_path / "mine", "[1, 2, 3]"), capsys)
        yours = self._run(self._notebook(tmp_path / "yours", "[1, 2, 4]"), capsys)

        assert self._outputs(mine, "total")["total"] != self._outputs(yours, "total")["total"]

    @staticmethod
    def _run_warm(nb, capsys):
        import json as _json

        from strata.notebook.cli import run_main

        assert run_main([str(nb), "--no-sync", "--format", "json"]) == 0
        return _json.loads(capsys.readouterr().out)

    @staticmethod
    def _cache_hit(payload, cell_id):
        return next(c for c in payload["cells"] if c["id"] == cell_id)["cache_hit"]

    def test_a_cached_leaf_still_reports_its_digest(self, tmp_path, capsys):
        """A warm rerun replays the leaf; the digest is the stored one, the same bytes."""
        nb = self._notebook(tmp_path, "[1, 2, 3]")
        fresh = self._run_warm(nb, capsys)
        # The second run of a new notebook still executes; the third is the warm one.
        self._run_warm(nb, capsys)

        warm = self._run_warm(nb, capsys)

        assert self._cache_hit(warm, "total") is True
        assert self._outputs(warm, "total")["total"] == self._outputs(fresh, "total")["total"]

    def test_a_row_stored_without_a_digest_is_filled_in(self, tmp_path, capsys):
        """Results stored before digests existed still get one, read from their bytes."""
        nb = self._notebook(tmp_path, "[1, 2, 3]")
        first = self._run_warm(nb, capsys)
        self._run_warm(nb, capsys)
        store = ArtifactStore(nb / ".strata" / "artifacts")
        conn = store._get_connection()
        try:
            conn.execute("UPDATE artifact_versions SET content_sha256 = NULL")
            conn.commit()
        finally:
            conn.close()

        warm = self._run_warm(nb, capsys)

        assert self._cache_hit(warm, "rows") is True
        assert self._outputs(warm, "rows")["rows"] == self._outputs(first, "rows")["rows"]
        assert self._outputs(warm, "rows")["rows"] is not None

    def test_the_zip_exports_provenance_carries_the_digests(self, tmp_path, capsys):
        import io
        import json as _json
        import zipfile

        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.snapshot import write_committed_files

        nb = self._notebook(tmp_path, "[1, 2, 3]")
        payload = self._run(nb, capsys)
        session = NotebookSession(parse_notebook(nb), nb)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            write_committed_files(session, archive)

        with zipfile.ZipFile(buffer) as archive:
            cells = _json.loads(archive.read("provenance.json"))["cells"]

        exported = {o["name"]: o["content_sha256"] for o in cells["rows"]["outputs"]}
        assert exported == self._outputs(payload, "rows")
