"""A ``# @output_dir`` folder copied into another store: its manifest and every file it names.

A folder is a manifest artifact whose files are artifacts of their own, not lineage edges, so a
copy that walked lineage alone left the files behind.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import zipfile

import httpx
import pytest

from strata.artifact.store import ArtifactStore
from strata.artifact.transfer import RemoteStore
from strata.notebook import output_dirs
from strata.notebook.artifact_integration import NotebookArtifactManager

FILES = {"epochs/e0.bin": b"\x00" * 10, "epochs/e1.bin": b"\x01" * 10, "final.txt": b"done"}
PROXY_TOKEN = "folder-proxy-token"


def _folder(root, files, *, provenance="c3" * 32, notebook_id="nb"):
    """Store a folder output of cell ``w`` in a notebook's store, as a run would."""
    manager = NotebookArtifactManager(notebook_id, artifact_dir=root / "notebook")
    run = root / "run"
    folder = run / output_dirs.OUTPUT_DIRS / "ckpt"
    shutil.rmtree(folder, ignore_errors=True)
    for rel, data in files.items():
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(data)
    manifest = output_dirs.collect(manager, "w", run, "ckpt")
    artifact = manager.store_cell_output(
        cell_id="w",
        variable_name="ckpt",
        blob_data=manifest,
        content_type=output_dirs.CONTENT_TYPE,
        provenance_hash=provenance,
        input_versions={},
        source="# @output_dir ckpt",
    )
    return manager, artifact


def _rebuilt(store: ArtifactStore, artifact_id: str, version: int, dest) -> dict[str, bytes]:
    """The folder a store's manifest version describes, rebuilt from that store alone."""
    manifest = store.read_blob(artifact_id, version)
    tenant = store.get_artifact(artifact_id, version).tenant
    output_dirs.materialize(store, artifact_id, manifest, dest, tenant=tenant)
    return {
        path.relative_to(dest).as_posix(): path.read_bytes()
        for path in dest.rglob("*")
        if path.is_file()
    }


def _stored(store: ArtifactStore, artifact_id: str, blob: bytes, *, content_type, tenant=None):
    """A version holding *blob* under *artifact_id*, as an import could leave one."""
    from strata.artifact.provenance import TransformSpec

    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=hashlib.sha256(artifact_id.encode() + blob).hexdigest(),
        transform_spec=TransformSpec(
            executor="notebook/cell@v1", params={"content_type": content_type}, inputs=[]
        ),
        tenant=tenant,
    )
    store.write_blob(artifact_id, version, blob)
    store.finalize_artifact(
        artifact_id, version, "", None, len(blob), hashlib.sha256(blob).hexdigest()
    )
    return store.get_artifact(artifact_id, version)


def _promote(root, url, ref, *, name="team/ckpt", header=None) -> int:
    from strata.artifact.cli import cmd_promote

    return cmd_promote(
        argparse.Namespace(
            ref=ref,
            artifact_dir=str(root / "notebook"),
            to_url=url,
            name=name,
            alias=None,
            tag=None,
            header=header,
            format="human",
            tenant=None,
            max_depth=10,
        )
    )


def _publish(root, url, ref, *, header=None) -> int:
    from strata.artifact.cli import cmd_publish

    return cmd_publish(
        argparse.Namespace(
            ref=ref,
            artifact_dir=str(root / "notebook"),
            format="json",
            title=None,
            author=None,
            here=False,
            into=None,
            to_url=url,
            header=header,
            max_depth=10,
        )
    )


@pytest.fixture
def team(tmp_path):
    """A real server standing in for the team's store."""
    from tests.conftest import run_server_with_context

    with run_server_with_context(tmp_path / "cache", tmp_path / "team", "personal") as ctx:
        yield ctx.base_url, tmp_path / "team"


class TestPromote:
    def test_a_promoted_folder_rebuilds_from_the_store_alone(self, team, tmp_path):
        base, team_dir = team
        _, folder = _folder(tmp_path / "alice", FILES)

        assert _promote(tmp_path / "alice", base, folder.id) == 0

        # Another store instance on the team's directory: nothing of the notebook's.
        store = ArtifactStore(team_dir)
        named = store.resolve_name("team/ckpt")
        assert (named.id, named.version) == (folder.id, folder.version)
        assert _rebuilt(store, named.id, named.version, tmp_path / "out") == FILES

    def test_a_folder_upstream_of_the_promoted_result_travels_too(self, team, tmp_path):
        base, team_dir = team
        manager, folder = _folder(tmp_path / "alice", FILES)
        ref = f"{folder.id}@v={folder.version}"
        score = manager.store_cell_output(
            cell_id="s",
            variable_name="score",
            blob_data=b"[1]",
            content_type="json/object",
            provenance_hash="d4" * 32,
            input_versions={f"strata://artifact/{ref}": ref},
            source="score = [1]",
        )

        assert _promote(tmp_path / "alice", base, score.id) == 0

        store = ArtifactStore(team_dir)
        assert _rebuilt(store, folder.id, folder.version, tmp_path / "out") == FILES

    def test_a_file_the_store_holds_is_not_sent_again(self, team, tmp_path, monkeypatch):
        base, _ = team
        manager, first = _folder(tmp_path / "alice", FILES)
        assert _promote(tmp_path / "alice", base, first.id) == 0
        more = {**FILES, "epochs/e2.bin": b"\x02" * 10}
        _, second = _folder(tmp_path / "alice", more, provenance="e5" * 32)
        sent = []
        real_import = RemoteStore.import_artifact

        def spy(self, record, blob):
            sent.append(record.id)
            return real_import(self, record, blob)

        monkeypatch.setattr(RemoteStore, "import_artifact", spy)

        assert _promote(tmp_path / "alice", base, second.id) == 0

        new_file = output_dirs.file_artifact_id(
            manager, "w", hashlib.sha256(more["epochs/e2.bin"]).hexdigest()
        )
        assert sent == [new_file, second.id]

    def test_a_file_held_at_another_version_is_named_at_that_version(
        self, team, tmp_path, monkeypatch
    ):
        """Stored again locally, a file the store already holds is the same bytes."""
        base, team_dir = team
        manager, first = _folder(tmp_path / "alice", FILES)
        assert _promote(tmp_path / "alice", base, first.id) == 0
        local = manager.artifact_store
        final = output_dirs.file_artifact_id(
            manager, "w", hashlib.sha256(FILES["final.txt"]).hexdigest()
        )
        # A second copy of each file under its id, as two runs storing it at once make.
        monkeypatch.setattr(output_dirs, "_reused", lambda store, artifact_id: None)
        _, second = _folder(tmp_path / "alice", FILES, provenance="e5" * 32)
        assert (
            json.loads(local.read_blob(second.id, second.version))["files"]["final.txt"]["version"]
            == 2
        )

        assert _promote(tmp_path / "alice", base, second.id) == 0

        store = ArtifactStore(team_dir)
        assert store.get_artifact(final, 2) is None
        assert _rebuilt(store, second.id, second.version, tmp_path / "out") == FILES

    def test_a_folder_missing_a_file_is_refused_not_copied_without_it(self, team, tmp_path):
        base, team_dir = team
        manager, folder = _folder(tmp_path / "alice", FILES)
        final = output_dirs.file_artifact_id(
            manager, "w", hashlib.sha256(FILES["final.txt"]).hexdigest()
        )
        manager.artifact_store.delete_artifact(final, 1)

        assert _promote(tmp_path / "alice", base, folder.id) != 0

        assert ArtifactStore(team_dir).get_artifact(folder.id, folder.version) is None

    def test_a_folder_whose_file_lost_its_bytes_is_refused_before_any_file_is_sent(self, tmp_path):
        """The file's row is there and its bytes are not: no file and no manifest is copied."""
        from strata.artifact.transfer import copy_chain

        manager, folder = _folder(tmp_path / "alice", FILES)
        final = output_dirs.file_artifact_id(
            manager, "w", hashlib.sha256(FILES["final.txt"]).hexdigest()
        )
        assert manager.artifact_store.blob_store.delete_blob(final, 1)
        central = ArtifactStore(tmp_path / "central")

        with pytest.raises(ValueError, match="whose bytes are gone"):
            copy_chain(manager.artifact_store, central, folder, 10)

        assert central.list_versions_by_id_prefix("nb_") == []

    def test_a_promoted_folder_pulls_by_id_from_the_stores_server(self, team, tmp_path):
        from strata.artifact.cli import cmd_pull

        base, team_dir = team
        _, folder = _folder(tmp_path / "alice", FILES)
        assert _promote(tmp_path / "alice", base, folder.id) == 0
        out = tmp_path / "pulled"

        def pull() -> int:
            return cmd_pull(
                argparse.Namespace(
                    ref=f"{folder.id}@v={folder.version}",
                    artifact_dir=str(team_dir),
                    to=str(out),
                    tenant=None,
                )
            )

        assert pull() == 0

        assert {
            path.relative_to(out).as_posix(): path.read_bytes()
            for path in out.rglob("*")
            if path.is_file()
        } == FILES
        # Never into a directory holding other files: they would read as the folder's.
        assert pull() == 1


@pytest.fixture
def central(tmp_path):
    """A central store: service mode behind a trusted proxy, one tenant per team."""
    from tests.conftest import run_server_with_context

    with run_server_with_context(
        tmp_path / "central-cache",
        tmp_path / "central",
        "service",
        auth_mode="trusted_proxy",
        proxy_token=PROXY_TOKEN,
        multi_tenant_enabled=True,
        service_writes_enabled=True,
    ) as ctx:
        yield ctx.base_url, tmp_path / "central"


def _listener(tenant: str) -> list[str]:
    return [
        f"X-Strata-Proxy-Token: {PROXY_TOKEN}",
        f"X-Tenant-ID: {tenant}",
        "X-Strata-Principal: alice",
        "X-Strata-Scopes: artifacts:write artifacts:publish",
    ]


class TestPublish:
    def test_the_archive_serves_the_folders_files(self, team, tmp_path, capsys):
        base, _ = team
        _, folder = _folder(tmp_path / "alice", FILES)

        assert _publish(tmp_path / "alice", base, folder.id) == 0

        token = json.loads(capsys.readouterr().out)["token"]
        archive = httpx.get(f"{base}/p/{token}/archive.zip", timeout=30)
        assert archive.status_code == 200
        with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
            held = {
                name.removeprefix("folder/"): bundle.read(name)
                for name in bundle.namelist()
                if name.startswith("folder/")
            }
            record = json.loads(bundle.read("manifest.json"))
        assert held == FILES
        assert {entry["file"]: entry["sha256"] for entry in record["additional_files"]} == {
            f"folder/{rel}": hashlib.sha256(data).hexdigest() for rel, data in FILES.items()
        }
        verified = httpx.get(
            f"{base}/p/{token}/verify",
            params={"sha256": hashlib.sha256(FILES["final.txt"]).hexdigest()},
            timeout=30,
        ).json()
        assert (verified["matches"], verified["file"]) == (True, "folder/final.txt")

    def test_a_folder_missing_a_file_is_refused_with_a_message(self, team, tmp_path, capsys):
        base, team_dir = team
        manager, folder = _folder(tmp_path / "alice", FILES)
        final = output_dirs.file_artifact_id(
            manager, "w", hashlib.sha256(FILES["final.txt"]).hexdigest()
        )
        manager.artifact_store.delete_artifact(final, 1)

        assert _publish(tmp_path / "alice", base, folder.id) == 1

        assert capsys.readouterr().out.startswith(f"Cannot publish: {folder.id}@v=1 is a folder")
        assert ArtifactStore(team_dir).get_artifact(folder.id, folder.version) is None

    def test_an_unreadable_published_manifest_is_answered_as_a_missing_file(self, team, tmp_path):
        """An import can publish any bytes as a folder; its archive is a 404, as for a lost file."""
        base, team_dir = team
        store = ArtifactStore(team_dir)
        lost_file = _stored(
            store,
            "nb_x_cell_w_var_lost",
            json.dumps(
                {"cell_id": "w", "files": {"a": {"sha256": "0" * 64, "size": 1, "version": 1}}}
            ).encode(),
            content_type=output_dirs.CONTENT_TYPE,
        )
        unreadable = _stored(
            store, "nb_x_cell_w_var_junk", b"[1]", content_type=output_dirs.CONTENT_TYPE
        )

        def archive_status(record) -> int:
            token = store.publish_artifact(record.id, record.version).token
            return httpx.get(f"{base}/p/{token}/archive.zip", timeout=30).status_code

        assert archive_status(lost_file) == 404
        assert archive_status(unreadable) == 404

    def test_a_folder_another_tenant_holds_lands_as_this_tenants_copy(self, central, tmp_path):
        """Remapped, each file lands under a fresh id, and the manifest names it there."""
        base, central_dir = central
        _, theirs = _folder(tmp_path / "alice", FILES)
        assert _publish(tmp_path / "alice", base, theirs.id, header=_listener("team-a")) == 0
        # A clone of the same notebook: same ids, one file changed.
        mine_files = {**FILES, "final.txt": b"other"}
        _, mine = _folder(tmp_path / "bob", mine_files, provenance="f6" * 32)

        assert _publish(tmp_path / "bob", base, mine.id, header=_listener("team-b")) == 0

        store = ArtifactStore(central_dir)
        landed = store.find_by_provenance(mine.provenance_hash, "team-b")
        assert landed is not None and landed.id != mine.id
        assert _rebuilt(store, landed.id, landed.version, tmp_path / "out") == mine_files
        # Each file read is this tenant's own copy, not the one team-a holds under its id.
        refs = output_dirs.file_refs(landed.id, store.read_blob(landed.id, landed.version))
        assert all(store.get_artifact(fid, fv).tenant == "team-b" for fid, fv, _ in refs)


class TestReading:
    def test_a_path_leaving_the_folder_is_refused(self, tmp_path):
        store = ArtifactStore(tmp_path / "store")
        manifest = json.dumps(
            {"cell_id": "w", "files": {"../escape": {"sha256": "0" * 64, "size": 1, "version": 1}}}
        ).encode()

        with pytest.raises(ValueError, match="leaves"):
            output_dirs.materialize(
                store, "nb_x_cell_w_var_ckpt", manifest, tmp_path / "out", tenant=None
            )

        assert not (tmp_path / "escape").exists()


class TestRetention:
    """A folder something holds keeps the file versions it names through a sweep.

    The files are not lineage edges, so a walk over edges alone left them collectable.
    """

    @pytest.mark.parametrize("hold", ["publish", "name", "pin"])
    def test_a_held_folders_files_survive_collecting_everything(self, tmp_path, hold):
        from strata.artifact.transfer import copy_chain

        manager, folder = _folder(tmp_path / "alice", FILES)
        central = ArtifactStore(tmp_path / "central")
        copy_chain(manager.artifact_store, central, folder, 10)
        if hold == "publish":
            central.publish_artifact(folder.id, folder.version)
        elif hold == "name":
            central.set_name("team/ckpt", folder.id, folder.version)
        else:
            central.pin_artifact(folder.id, folder.version, "paper")

        swept = central.garbage_collect(max_idle_days=0, collect_latest=True)

        assert swept["deleted_count"] == 0
        assert _rebuilt(central, folder.id, folder.version, tmp_path / "out") == FILES

    def test_a_superseded_file_version_a_pinned_folder_names_survives(self, tmp_path, monkeypatch):
        manager, folder = _folder(tmp_path / "alice", FILES)
        store = manager.artifact_store
        store.pin_artifact(folder.id, folder.version, "paper")
        # Each file stored again under its id, as two runs storing it at once make.
        monkeypatch.setattr(output_dirs, "_reused", lambda store, artifact_id: None)
        _folder(tmp_path / "alice", FILES, provenance="e5" * 32)
        refs = output_dirs.file_refs(folder.id, store.read_blob(folder.id, folder.version))
        assert all(store.get_artifact(fid, fv).state == "superseded" for fid, fv, _ in refs)

        store.garbage_collect(keep_superseded=0)

        assert _rebuilt(store, folder.id, folder.version, tmp_path / "out") == FILES


class TestAnotherTenantsFile:
    """A manifest's bytes are its importer's: naming another tenant's file reads as missing it."""

    @staticmethod
    def _crafted(tmp_path):
        store = ArtifactStore(tmp_path / "central")
        secret = b"team-b's weights"
        sha256 = hashlib.sha256(secret).hexdigest()
        theirs = _stored(
            store,
            f"nb_b_cell_w_var___file__{sha256}",
            secret,
            content_type="application/octet-stream",
            tenant="team-b",
        )
        entry = {"id": theirs.id, "sha256": sha256, "size": len(secret), "version": 1}
        manifest = json.dumps({"cell_id": "w", "files": {"w.bin": entry}}).encode()
        mine = _stored(
            store,
            "nb_a_cell_w_var_ckpt",
            manifest,
            content_type=output_dirs.CONTENT_TYPE,
            tenant="team-a",
        )
        return store, mine, manifest

    def test_a_rebuild_refuses_it(self, tmp_path):
        store, mine, manifest = self._crafted(tmp_path)

        with pytest.raises(FileNotFoundError, match="missing"):
            output_dirs.materialize(store, mine.id, manifest, tmp_path / "out", tenant="team-a")

        assert not (tmp_path / "out" / "w.bin").exists()
        # The file is there: a folder of its own tenant reads it.
        output_dirs.materialize(store, mine.id, manifest, tmp_path / "theirs", tenant="team-b")
        assert (tmp_path / "theirs" / "w.bin").read_bytes() == b"team-b's weights"

    def test_an_archive_refuses_it(self, tmp_path, capsys):
        from strata.artifact.cli import cmd_archive

        _, mine, _ = self._crafted(tmp_path)

        code = cmd_archive(
            argparse.Namespace(
                ref=f"{mine.id}@v={mine.version}",
                token=None,
                artifact_dir=str(tmp_path / "central"),
                title=None,
                author=None,
                max_depth=None,
                to=str(tmp_path / "bundle"),
                force=False,
                tenant="team-a",
            )
        )

        assert code == 1
        assert "missing from the store" in capsys.readouterr().out
        assert not (tmp_path / "bundle" / "folder" / "w.bin").exists()

    def test_a_copy_refuses_it(self, tmp_path):
        from strata.artifact.transfer import copy_chain

        store, mine, _ = self._crafted(tmp_path)
        other = ArtifactStore(tmp_path / "other")

        with pytest.raises(ValueError, match="which is gone"):
            copy_chain(store, other, mine, 10)

        assert other.list_versions_by_id_prefix("nb_") == []


class TestAnUnreadableManifest:
    """An import can hold any bytes under the folder type and any id: one names no file."""

    @pytest.mark.parametrize(
        ("artifact_id", "blob"),
        [
            ("nb_x_cell_w_var_ckpt", b"\xff not json"),
            ("nb_x_cell_w_var_ckpt", b"[1]"),
            ("nb_x_cell_w_var_ckpt", b'{"cell_id": "w", "files": {"a": {"sha256": "0"}}}'),
            (
                "an-id-of-no-cell",
                json.dumps(
                    {"cell_id": "w", "files": {"a": {"sha256": "0" * 64, "size": 1, "version": 1}}}
                ).encode(),
            ),
        ],
    )
    def test_garbage_collection_completes_past_a_held_one(
        self, tmp_path, artifact_id, blob, caplog
    ):
        from strata.artifact.transfer import copy_chain

        manager, folder = _folder(tmp_path / "alice", FILES)
        central = ArtifactStore(tmp_path / "central")
        copy_chain(manager.artifact_store, central, folder, 10)
        central.set_name("team/ckpt", folder.id, folder.version)
        junk = _stored(central, artifact_id, blob, content_type=output_dirs.CONTENT_TYPE)
        central.set_name("team/junk", junk.id, junk.version)

        central.garbage_collect(max_idle_days=0, collect_latest=True)

        assert central.get_artifact(junk.id, junk.version) is not None
        assert _rebuilt(central, folder.id, folder.version, tmp_path / "out") == FILES
        assert f"{artifact_id}@v=1 is not a readable folder manifest" in caplog.text

    def test_a_notebook_prune_keeps_every_folder_file(self, tmp_path):
        """What the unreadable one names is unknown, so no folder file is taken."""
        manager, folder = _folder(tmp_path / "alice", FILES)
        stray = tmp_path / "stray.bin"
        stray.write_bytes(b"named by no manifest")
        sha256 = hashlib.sha256(stray.read_bytes()).hexdigest()
        unnamed = output_dirs.store_received(manager, "w", stray, sha256, 20)
        manager.store_cell_output(
            cell_id="j",
            variable_name="junk",
            blob_data=b"not json",
            content_type=output_dirs.CONTENT_TYPE,
            provenance_hash="a1" * 32,
            input_versions={},
            source="# @output_dir junk",
        )

        manager.prune(keep_superseded=0, min_idle_seconds=0)

        store = manager.artifact_store
        stray_id = output_dirs.file_artifact_id(manager, "w", sha256)
        assert store.get_artifact(stray_id, unnamed["version"]) is not None
        assert _rebuilt(store, folder.id, folder.version, tmp_path / "out") == FILES
