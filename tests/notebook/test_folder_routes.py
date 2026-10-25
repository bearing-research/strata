"""A cell's ``# @output_dir`` folder over HTTP: its versions, and each file's bytes.

A failed run's version is listed and downloadable, since that is how a crash's
checkpoints are recovered. Only a name the cell declares and a path its manifest
holds are served.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

from strata.notebook import output_dirs
from strata.notebook.dependencies import EnvironmentOperationLog
from strata.notebook.executor import CellExecutor
from strata.notebook.routes import get_session_manager, router
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

CRASH = (
    "# @output_dir ckpt\n"
    "(ckpt / 'epochs').mkdir()\n"
    "(ckpt / 'epochs' / 'e0.pt').write_bytes(b'first' * 3)\n"
    "(ckpt / 'epochs' / 'e1.pt').write_bytes(b'second')\n"
    "raise RuntimeError('lost')\n"
)
SUCCESS = "# @output_dir ckpt\n(ckpt / 'final.pt').write_bytes(b'done')\n"


@pytest.fixture(autouse=True)
def no_uv_sync(monkeypatch):
    """Cells run on the dev interpreter; no venv is built."""
    monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)

    async def _fake_run_uv_command_streaming(*args, **kwargs):
        del args, kwargs
        return SimpleNamespace(
            success=True, error=None, operation_log=EnvironmentOperationLog(command="uv")
        )

    monkeypatch.setattr(
        "strata.notebook.dependencies.run_uv_command_streaming",
        _fake_run_uv_command_streaming,
    )

    async def _noop_start(self):
        pass

    monkeypatch.setattr("strata.notebook.pool.WarmProcessPool.start", _noop_start)


@pytest.fixture
async def client():
    app = FastAPI()
    app.include_router(router)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as http:
        yield http


@pytest.fixture
async def opened(tmp_path):
    """Open a notebook whose one cell ``w`` has *source*; closed after the test."""
    sessions = []

    def _open(source: str):
        nb = create_notebook(tmp_path, "Folders")
        add_cell_to_notebook(nb, "w", None)
        write_cell(nb, "w", source)
        session = get_session_manager().open_notebook(nb)
        sessions.append(session)
        return session

    yield _open
    for session in sessions:
        await asyncio.gather(*get_session_manager().close_session(session.id))


async def _run(session, source: str) -> bool:
    cell = session.notebook_state.get_cell("w")
    cell.source = source
    session.re_analyze_cell("w")
    return (await CellExecutor(session).execute_cell("w", source)).success


def _digest(data: bytes) -> str:
    return f"sha-256=:{base64.b64encode(hashlib.sha256(data).digest()).decode()}:"


class TestTheListing:
    async def test_a_crashed_run_is_listed_failed_with_both_checkpoints(self, client, opened):
        session = opened(CRASH)
        assert not await _run(session, CRASH)

        response = await client.get(f"/v1/notebooks/{session.id}/cells/w/folders/ckpt")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["cell_id"] == "w"
        assert body["name"] == "ckpt"
        [version] = body["versions"]
        assert version["version"] == 1
        assert version["failed"] is True
        assert version["created_at"] > 0
        assert "principal" in version
        assert version["files"] == [
            {
                "path": "epochs/e0.pt",
                "size": 15,
                "sha256": hashlib.sha256(b"first" * 3).hexdigest(),
            },
            {"path": "epochs/e1.pt", "size": 6, "sha256": hashlib.sha256(b"second").hexdigest()},
        ]
        assert version["file_count"] == 2
        assert version["total_bytes"] == 21

    async def test_versions_come_newest_first(self, client, opened):
        session = opened(CRASH)
        assert not await _run(session, CRASH)
        assert await _run(session, SUCCESS)

        response = await client.get(f"/v1/notebooks/{session.id}/cells/w/folders/ckpt")

        versions = response.json()["versions"]
        assert [(v["version"], v["failed"]) for v in versions] == [(2, False), (1, True)]
        assert [f["path"] for f in versions[0]["files"]] == ["final.pt"]

    async def test_a_row_a_crash_left_without_bytes_is_not_listed(self, client, opened):
        from computation_core.provenance import TransformSpec

        session = opened(SUCCESS)
        assert await _run(session, SUCCESS)
        manager = session.get_artifact_manager()
        store = manager.artifact_store
        artifact_id = manager.cell_artifact_id("w", "ckpt")
        spec = TransformSpec(
            executor="notebook/cell@v1",
            params={"content_type": output_dirs.CONTENT_TYPE},
            inputs=[],
        )
        zombie = store.create_artifact(artifact_id, provenance_hash="crash", transform_spec=spec)
        store.fail_artifact(artifact_id, zombie)
        building = store.create_artifact(artifact_id, provenance_hash="run", transform_spec=spec)

        response = await client.get(f"/v1/notebooks/{session.id}/cells/w/folders/ckpt")

        assert zombie != building
        assert [v["version"] for v in response.json()["versions"]] == [1]

    async def test_a_run_still_writing_is_not_listed_though_its_manifest_is_stored(
        self, client, opened
    ):
        session = opened(SUCCESS)
        assert await _run(session, SUCCESS)
        manager = session.get_artifact_manager()
        # Staged as a run does: the manifest's bytes are written while the row is building.
        staged = manager.stage_cell_output(
            cell_id="w",
            variable_name="ckpt",
            blob_data=json.dumps({"cell_id": "w", "files": {}}).encode(),
            content_type=output_dirs.CONTENT_TYPE,
            provenance_hash="run",
        )
        store = manager.artifact_store
        assert store.get_artifact(staged.artifact_id, staged.version).state == "building"
        assert store.read_blob(staged.artifact_id, staged.version) is not None

        response = await client.get(f"/v1/notebooks/{session.id}/cells/w/folders/ckpt")

        assert [v["version"] for v in response.json()["versions"]] == [1]

    async def test_a_name_the_cell_does_not_declare_is_404(self, client, opened):
        session = opened(SUCCESS)
        assert await _run(session, SUCCESS)
        # A variable of the cell, stored like a folder's manifest, is still not a folder.
        manager = session.get_artifact_manager()
        manager.store_cell_output(
            cell_id="w",
            variable_name="other",
            blob_data=json.dumps({"cell_id": "w", "files": {}}).encode(),
            content_type=output_dirs.CONTENT_TYPE,
            provenance_hash="other",
        )

        for name in ("other", "nope"):
            response = await client.get(f"/v1/notebooks/{session.id}/cells/w/folders/{name}")
            assert response.status_code == 404, name

    async def test_an_unknown_cell_is_404(self, client, opened):
        session = opened(SUCCESS)

        response = await client.get(f"/v1/notebooks/{session.id}/cells/ghost/folders/ckpt")

        assert response.status_code == 404


class TestAFile:
    async def test_each_checkpoint_of_a_crashed_run_downloads(self, client, opened):
        session = opened(CRASH)
        assert not await _run(session, CRASH)
        base = f"/v1/notebooks/{session.id}/cells/w/folders/ckpt/v/1/files"

        for path, data in (("epochs/e0.pt", b"first" * 3), ("epochs/e1.pt", b"second")):
            response = await client.get(f"{base}/{path}")

            assert response.status_code == 200, response.text
            assert response.content == data
            assert response.headers["content-length"] == str(len(data))
            assert response.headers["content-digest"] == _digest(data)
            assert response.headers["content-type"] == "application/octet-stream"
            assert response.headers["x-content-type-options"] == "nosniff"

    async def test_a_path_outside_the_manifest_is_404(self, client, opened):
        session = opened(CRASH)
        assert not await _run(session, CRASH)
        base = f"/v1/notebooks/{session.id}/cells/w/folders/ckpt/v/1/files"

        for path in ("epochs", "e0.pt", "epochs/missing.pt"):
            response = await client.get(f"{base}/{path}")
            assert response.status_code == 404, path
        # Encoded, so the client does not resolve the dots before sending.
        for path in ("..%2F..%2Fnotebook.toml", "%2Fetc%2Fpasswd", "epochs%2F..%2Fe0.pt"):
            response = await client.get(f"{base}/{path}")
            assert response.status_code == 404, path

    async def test_a_name_the_cell_does_not_declare_is_404(self, client, opened):
        session = opened(SUCCESS)
        assert await _run(session, SUCCESS)
        # The folder's own manifest, stored again under a name the cell does not declare.
        manager = session.get_artifact_manager()
        store = manager.artifact_store
        manifest = store.get_latest_version(manager.cell_artifact_id("w", "ckpt"))
        manager.store_cell_output(
            cell_id="w",
            variable_name="other",
            blob_data=store.read_blob(manifest.id, manifest.version),
            content_type=output_dirs.CONTENT_TYPE,
            provenance_hash="other",
        )

        for name in ("other", "nope"):
            response = await client.get(
                f"/v1/notebooks/{session.id}/cells/w/folders/{name}/v/1/files/final.pt"
            )
            assert response.status_code == 404, name

    async def test_a_file_the_manifest_names_by_id_is_served_from_there(self, client, opened):
        from computation_core.provenance import TransformSpec

        session = opened(SUCCESS)
        assert await _run(session, SUCCESS)
        manager = session.get_artifact_manager()
        store = manager.artifact_store
        # A copy that landed a file under another id names it; its bytes live only there.
        data = b"landed elsewhere"
        sha = hashlib.sha256(data).hexdigest()
        file_id = f"{manager.cell_artifact_id('w', f'__file__{sha}')}@import=0badf00d"
        spec = TransformSpec(executor="notebook/cell@v1", params={}, inputs=[])
        file_version = store.create_artifact(file_id, provenance_hash="landed", transform_spec=spec)
        store.write_blob(file_id, file_version, data)
        store.finalize_artifact(file_id, file_version, "{}", None, len(data))
        manager.store_cell_output(
            cell_id="w",
            variable_name="ckpt",
            blob_data=json.dumps(
                {
                    "cell_id": "w",
                    "files": {
                        "final.pt": {
                            "sha256": sha,
                            "size": len(data),
                            "version": file_version,
                            "id": file_id,
                        }
                    },
                }
            ).encode(),
            content_type=output_dirs.CONTENT_TYPE,
            provenance_hash="moved",
        )

        response = await client.get(
            f"/v1/notebooks/{session.id}/cells/w/folders/ckpt/v/2/files/final.pt"
        )

        assert response.status_code == 200, response.text
        assert response.content == data
        assert response.headers["content-digest"] == _digest(data)

    async def test_a_version_that_is_not_there_is_404(self, client, opened):
        session = opened(SUCCESS)
        assert await _run(session, SUCCESS)

        response = await client.get(
            f"/v1/notebooks/{session.id}/cells/w/folders/ckpt/v/9/files/final.pt"
        )

        assert response.status_code == 404


class TestAPerVariantCell:
    """Each instance's folder is its own, listed under its variant, a failed one's too."""

    WRITER = (
        "# @per_variant\n# @output_dir ckpt\n"
        "(ckpt / 'k.txt').write_text(str(k))\n"
        "if k == 2:\n    raise RuntimeError('lost')\n"
    )

    @pytest.fixture
    async def fanned_out(self, tmp_path):
        from strata.notebook.writer import set_variant_mode

        nb = create_notebook(tmp_path, "Folders")
        after = None
        for cell_id, source in (
            ("a", "# @variant model a\nk = 1\n"),
            ("b", "# @variant model b\nk = 2\n"),
            ("w", self.WRITER),
        ):
            add_cell_to_notebook(nb, cell_id, after)
            write_cell(nb, cell_id, source)
            after = cell_id
        set_variant_mode(nb, "model", "sweep")
        session = get_session_manager().open_notebook(nb)
        executor = CellExecutor(session)
        for cell_id in ("a", "b"):
            cell = session.notebook_state.get_cell(cell_id)
            assert (await executor.execute_cell(cell_id, cell.source)).success
        result = await executor.execute_cell("w", self.WRITER)
        assert not result.success
        assert "variant 'b' failed" in result.error
        yield session
        await asyncio.gather(*get_session_manager().close_session(session.id))

    async def test_each_variant_is_listed_and_the_failed_one_is_marked(self, client, fanned_out):
        response = await client.get(f"/v1/notebooks/{fanned_out.id}/cells/w/folders/ckpt")

        assert response.status_code == 200, response.text
        listed = {
            (v["variant"], v["version"], v["failed"]): [f["path"] for f in v["files"]]
            for v in response.json()["versions"]
        }
        assert listed == {("a", 1, False): ["k.txt"], ("b", 1, True): ["k.txt"]}

    async def test_a_variants_file_downloads_by_its_variant(self, client, fanned_out):
        base = f"/v1/notebooks/{fanned_out.id}/cells/w/folders/ckpt/v/1/files/k.txt"

        for variant, data in (("a", b"1"), ("b", b"2")):
            response = await client.get(base, params={"variant": variant})
            assert response.status_code == 200, response.text
            assert response.content == data
        for params in ({}, {"variant": "c"}, {"variant": "a@iter=1"}):
            response = await client.get(base, params=params)
            assert response.status_code == 404, params
