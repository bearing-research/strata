"""Tests for the ``strata artifact`` inspection CLI (list/show/lineage/pull)."""

from __future__ import annotations

import argparse
import json

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from strata.artifact_cli import cmd_lineage, cmd_list, cmd_pull, cmd_show
from strata.artifact_store import ArtifactStore, TransformSpec, reset_artifact_store


def _ipc_bytes(num_rows: int) -> bytes:
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, pa.schema([("id", pa.int64())])) as writer:
        writer.write_batch(pa.RecordBatch.from_pydict({"id": list(range(num_rows))}))
    return sink.getvalue().to_pybytes()


@pytest.fixture
def chain_store(tmp_path):
    """Store with a 3-level chain: model <- features <- scan <- table."""
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    store = ArtifactStore(artifact_dir)

    def make(artifact_id, provenance, executor, inputs, input_versions, rows, name=None):
        store.create_artifact(
            artifact_id,
            provenance,
            transform_spec=TransformSpec(executor=executor, params={}, inputs=inputs),
            input_versions=input_versions,
        )
        store.write_blob(artifact_id, 1, _ipc_bytes(rows))
        store.finalize_artifact(artifact_id, 1, "{}", rows, 100)
        if name:
            store.set_name(name, artifact_id, 1)

    make(
        "scan-1",
        "prov-scan",
        "scan@v1",
        ["file:///wh#db.events"],
        {"file:///wh#db.events": "111222333"},
        100,
        name="demo/raw",
    )
    make(
        "feat-1",
        "prov-feat",
        "feature_eng@v1",
        ["strata://artifact/scan-1@v=1"],
        {"strata://artifact/scan-1@v=1": "scan-1@v=1"},
        50,
    )
    make(
        "model-1",
        "prov-model",
        "train@v1",
        ["strata://artifact/feat-1@v=1"],
        {"strata://artifact/feat-1@v=1": "feat-1@v=1"},
        1,
        name="demo/model",
    )

    yield {"dir": str(artifact_dir), "store": store}
    reset_artifact_store()


def _args(**kwargs) -> argparse.Namespace:
    defaults = {"artifact_dir": None, "format": "human"}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestList:
    def test_human_listing(self, chain_store, capsys):
        rc = cmd_list(_args(artifact_dir=chain_store["dir"], state=None, limit=50))
        out = capsys.readouterr().out
        assert rc == 0
        assert "model-1" in out
        assert "demo/model" in out
        assert "scan-1" in out

    def test_json_listing(self, chain_store, capsys):
        rc = cmd_list(_args(artifact_dir=chain_store["dir"], state=None, limit=50, format="json"))
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        ids = {a["artifact_id"] for a in payload}
        assert {"scan-1", "feat-1", "model-1"} <= ids

    def test_missing_dir_exits_two(self, tmp_path):
        assert cmd_list(_args(artifact_dir=str(tmp_path / "nope"), state=None, limit=5)) == 2


class TestShow:
    def test_show_by_name(self, chain_store, capsys):
        rc = cmd_show(_args(ref="demo/model", artifact_dir=chain_store["dir"]))
        out = capsys.readouterr().out
        assert rc == 0
        assert "model-1@v=1" in out
        assert "train@v1" in out
        assert "demo/model" in out

    def test_show_by_id_version(self, chain_store, capsys):
        rc = cmd_show(_args(ref="feat-1@v=1", artifact_dir=chain_store["dir"]))
        assert rc == 0
        assert "feat-1@v=1" in capsys.readouterr().out

    def test_show_by_bare_id_latest(self, chain_store, capsys):
        rc = cmd_show(_args(ref="scan-1", artifact_dir=chain_store["dir"]))
        assert rc == 0
        assert "scan-1@v=1" in capsys.readouterr().out

    def test_unknown_ref_exits_one(self, chain_store):
        assert cmd_show(_args(ref="ghost", artifact_dir=chain_store["dir"])) == 1


class TestLineage:
    def test_renders_full_chain_to_snapshot(self, chain_store, capsys):
        rc = cmd_lineage(_args(ref="demo/model", artifact_dir=chain_store["dir"], max_depth=10))
        out = capsys.readouterr().out
        assert rc == 0
        # model -> features -> scan -> table @ snapshot, in order
        assert out.index("model-1@v=1") < out.index("feat-1@v=1") < out.index("scan-1@v=1")
        assert "table file:///wh#db.events  @ snapshot 111222333" in out

    def test_json_tree(self, chain_store, capsys):
        rc = cmd_lineage(
            _args(ref="demo/model", artifact_dir=chain_store["dir"], max_depth=10, format="json")
        )
        assert rc == 0
        tree = json.loads(capsys.readouterr().out)
        assert tree["artifact_id"] == "model-1"
        leaf = tree["inputs"][0]["inputs"][0]["inputs"][0]
        assert leaf == {"uri": "file:///wh#db.events", "version": "111222333"}

    def test_max_depth_cuts_recursion(self, chain_store, capsys):
        rc = cmd_lineage(
            _args(ref="demo/model", artifact_dir=chain_store["dir"], max_depth=1, format="json")
        )
        assert rc == 0
        tree = json.loads(capsys.readouterr().out)
        # depth 1: features expanded, scan stays a URI leaf
        feat = tree["inputs"][0]
        assert feat["artifact_id"] == "feat-1"
        assert feat["inputs"][0]["uri"] == "strata://artifact/scan-1@v=1"


class TestPull:
    def test_pull_by_name_to_path(self, chain_store, tmp_path, capsys):
        out_file = tmp_path / "model.arrow"
        rc = cmd_pull(_args(ref="demo/model", artifact_dir=chain_store["dir"], to=str(out_file)))
        assert rc == 0
        table = ipc.open_stream(out_file.read_bytes()).read_all()
        assert table.num_rows == 1

    def test_pull_unknown_exits_one(self, chain_store, tmp_path):
        rc = cmd_pull(_args(ref="ghost", artifact_dir=chain_store["dir"], to=str(tmp_path / "x")))
        assert rc == 1


class TestTenantAgnosticResolution:
    def test_legacy_default_tenant_name_resolves(self, tmp_path, capsys):
        """A name written under legacy '_default' is still findable by the CLI."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        store = ArtifactStore(artifact_dir)
        store.create_artifact("legacy-1", "prov-x", tenant="_default")
        store.write_blob("legacy-1", 1, _ipc_bytes(2))
        store.finalize_artifact("legacy-1", 1, "{}", 2, 100)
        store.set_name("old/model", "legacy-1", 1, tenant="_default")

        rc = cmd_show(_args(ref="old/model", artifact_dir=str(artifact_dir)))
        out = capsys.readouterr().out
        assert rc == 0
        assert "legacy-1@v=1" in out
        reset_artifact_store()

    def _two_tenant_store(self, tmp_path):
        """A store where two tenants publish the same name 'shared/model'."""
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        store = ArtifactStore(artifact_dir)
        for tenant, aid in (("team-a", "a-1"), ("team-b", "b-1")):
            store.create_artifact(aid, f"prov-{aid}", tenant=tenant)
            store.finalize_artifact(aid, 1, "{}", 1, 10)
            store.set_name("shared/model", aid, 1, tenant=tenant)
        return str(artifact_dir)

    def test_ambiguous_name_without_tenant_errors(self, tmp_path, capsys):
        artifact_dir = self._two_tenant_store(tmp_path)
        rc = cmd_show(_args(ref="shared/model", artifact_dir=artifact_dir))
        err = capsys.readouterr().err
        assert rc == 1
        assert "multiple tenants" in err
        reset_artifact_store()

    def test_tenant_flag_disambiguates(self, tmp_path, capsys):
        artifact_dir = self._two_tenant_store(tmp_path)
        rc = cmd_show(_args(ref="shared/model", artifact_dir=artifact_dir, tenant="team-a"))
        out = capsys.readouterr().out
        assert rc == 0
        assert "a-1@v=1" in out
        assert "tenant:    team-a" in out
        reset_artifact_store()

    def test_json_payload_includes_tenant(self, tmp_path, capsys):
        artifact_dir = self._two_tenant_store(tmp_path)
        rc = cmd_show(
            _args(ref="shared/model", artifact_dir=artifact_dir, tenant="team-b", format="json")
        )
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert payload["tenant"] == "team-b"
        assert "principal" in payload
        reset_artifact_store()


class TestAliasRefsAndAudit:
    """CLI alias refs (name@alias) and the audit command (#129)."""

    def test_show_resolves_alias_ref(self, chain_store, capsys):
        store = chain_store["store"]
        store.set_alias("demo/model", "champion", "model-1", 1)

        rc = cmd_show(_args(ref="demo/model@champion", artifact_dir=chain_store["dir"]))
        out = capsys.readouterr().out
        assert rc == 0
        assert "model-1@v=1" in out
        assert "demo/model@champion" in out  # rendered in aliases line

    def test_audit_renders_moves(self, chain_store, capsys):
        from strata.artifact_cli import cmd_audit

        store = chain_store["store"]
        store.set_alias("demo/model", "champion", "feat-1", 1)
        store.set_alias("demo/model", "champion", "model-1", 1)

        rc = cmd_audit(
            _args(ref=None, artifact_dir=chain_store["dir"], name="demo/model", limit=50)
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "alias_set" in out
        assert "demo/model@champion" in out
        # cross-artifact move shows both sides: feat-1@v1 -> model-1@v1
        assert "feat-1@v1 -> model-1@v1" in out

    def test_audit_json(self, chain_store, capsys):
        from strata.artifact_cli import cmd_audit

        store = chain_store["store"]
        store.set_tag("model-1", 1, "auc", "0.9")
        rc = cmd_audit(
            _args(ref=None, artifact_dir=chain_store["dir"], name=None, limit=50, format="json")
        )
        assert rc == 0
        entries = json.loads(capsys.readouterr().out)
        assert any(e["action"] == "tag_set" and e["key"] == "auc" for e in entries)


class TestPublish:
    """``strata artifact publish`` and the disclosure it prints first."""

    def test_publish_lists_every_step_the_link_will_expose(self, chain_store, capsys):
        """Publishing exposes the whole ancestry, not just the artifact.

        That is the transparency being asked for, but it is also the thing a
        researcher can be surprised by — upstream cell source can name private
        dataset paths. So the chain is printed back before the link is used,
        and this pins that it names every step rather than only the one being
        published.
        """
        from strata.artifact_cli import cmd_publish

        rc = cmd_publish(
            _args(ref="demo/model", artifact_dir=chain_store["dir"], title=None, max_depth=10)
        )

        assert rc == 0
        out = capsys.readouterr().out
        assert "/p/" in out
        for step in ("model-1@v=1", "feat-1@v=1", "scan-1@v=1"):
            assert step in out, f"{step} is exposed by the link but was not disclosed"

    def test_unpublish_withdraws_and_is_not_repeatable(self, chain_store, capsys):
        from strata.artifact_cli import cmd_publish, cmd_unpublish

        cmd_publish(
            _args(ref="demo/model", artifact_dir=chain_store["dir"], title=None, max_depth=10)
        )
        token = chain_store["store"].list_publications()[0].token

        assert cmd_unpublish(_args(token=token, artifact_dir=chain_store["dir"])) == 0
        assert "Withdrawn" in capsys.readouterr().out

        assert cmd_unpublish(_args(token=token, artifact_dir=chain_store["dir"])) == 1


class TestPublishAuthor:
    def test_republishing_says_the_author_did_not_take(self, chain_store, capsys):
        """Publishing is idempotent, so a later --author is quietly dropped.

        Printing the usual success banner would report an author that never
        reached the page, with no indication anything was ignored.
        """
        from strata.artifact_cli import cmd_publish

        base = {"ref": "demo/model", "artifact_dir": chain_store["dir"], "max_depth": 10}
        cmd_publish(_args(title=None, author=None, **base))
        capsys.readouterr()

        cmd_publish(_args(title=None, author="F. Li", **base))

        out = capsys.readouterr().out
        assert "already published" in out
        assert "does not change that" in out


class TestArchive:
    """``strata artifact archive`` — the copy that needs no server."""

    @staticmethod
    def _archive(chain_store, tmp_path, **overrides):
        from strata.artifact_cli import cmd_archive

        dest = tmp_path / "bundle"
        args = {
            "ref": "demo/model",
            "artifact_dir": chain_store["dir"],
            "to": str(dest),
            "title": "Figure 3",
            "author": "F. Li",
            "max_depth": 10,
        }
        args.update(overrides)
        assert cmd_archive(_args(**args)) == 0
        return dest

    def test_the_bundle_opens_without_a_server(self, chain_store, tmp_path):
        """No route references: the page has to work from a Zenodo download."""
        dest = self._archive(chain_store, tmp_path)
        html = (dest / "index.html").read_text()

        assert "/p/" not in html, "an archived page must not link at a server"
        assert (dest / "manifest.json").exists()
        assert (dest / "README.md").exists()

    def test_the_recorded_digest_is_the_one_a_reader_computes(self, chain_store, tmp_path):
        """The README tells the reader to run sha256sum and compare.

        If the recorded digest were taken from anything but the bytes actually
        written into the bundle, that instruction would fail for every reader
        who followed it — the one check the bundle offers, broken.
        """
        import hashlib
        import json as jsonlib

        dest = self._archive(chain_store, tmp_path)
        manifest = jsonlib.loads((dest / "manifest.json").read_text())
        payload = next(p for p in dest.iterdir() if p.name.startswith("artifact."))

        assert manifest["content_sha256"] == hashlib.sha256(payload.read_bytes()).hexdigest()
        assert manifest["content_sha256"] in (dest / "README.md").read_text()

    def test_archiving_grants_nobody_access_to_the_server(self, chain_store, tmp_path):
        """Archiving is not publishing, and must not quietly become it.

        A bundle is a file someone chooses to hand over. Minting a live public
        link as a side effect would put the artifact on the network without
        anyone asking for that.
        """
        self._archive(chain_store, tmp_path)

        assert chain_store["store"].list_publications() == []

    def test_it_carries_the_whole_chain_not_just_the_artifact(self, chain_store, tmp_path):
        import json as jsonlib

        dest = self._archive(chain_store, tmp_path)
        manifest = jsonlib.loads((dest / "manifest.json").read_text())

        uris = {n["uri"] for n in manifest["lineage"]["nodes"]}
        assert any("feat-1" in u for u in uris)
        assert any("scan-1" in u for u in uris)

    def test_it_does_not_date_itself_as_a_publication(self, chain_store, tmp_path):
        """Archiving mints no link and serves nothing, so it publishes nothing.

        A bundle reporting a ``published_at`` dates an event that never
        happened — to a reader, and to a machine consuming a deposit that has
        no way to know better.
        """
        import json as jsonlib

        dest = self._archive(chain_store, tmp_path)
        manifest = jsonlib.loads((dest / "manifest.json").read_text())

        assert "publication" not in manifest
        assert "archived_at" in manifest["archive"]
        assert "Archived" in (dest / "index.html").read_text()

    def test_a_payload_is_referenced_not_embedded(self, chain_store, tmp_path):
        """The file sits beside the page, so base64 would only double the size.

        Unbounded, it also turns a large figure into an index.html no browser
        will open, which is the single thing a bundle has to guarantee.
        """
        dest = self._archive(chain_store, tmp_path)

        assert "data:image/png;base64," not in (dest / "index.html").read_text()

    def test_it_refuses_a_non_empty_destination(self, chain_store, tmp_path):
        """`--to .` was a one-keystroke way to clobber someone's README."""
        from strata.artifact_cli import cmd_archive

        dest = tmp_path / "occupied"
        dest.mkdir()
        (dest / "README.md").write_text("someone else's work")

        rc = cmd_archive(
            _args(
                ref="demo/model",
                artifact_dir=chain_store["dir"],
                to=str(dest),
                title=None,
                author=None,
                max_depth=10,
            )
        )

        assert rc == 1
        assert (dest / "README.md").read_text() == "someone else's work"

    def test_an_unfinished_artifact_is_not_archivable(self, chain_store, tmp_path):
        """A half-written blob digests like any other.

        Without the guard the bundle presents truncated bytes as a
        deposit-ready record, with a sha256sum line vouching for the fragment.
        """
        from strata.artifact_cli import cmd_archive

        store = chain_store["store"]
        store.create_artifact("halfway", "f" * 64)
        with store.open_blob_writer("halfway", 1) as writer:
            writer.write(b"partial")

        rc = cmd_archive(
            _args(
                ref="halfway",
                artifact_dir=chain_store["dir"],
                to=str(tmp_path / "bundle"),
                title=None,
                author=None,
                max_depth=10,
            )
        )

        assert rc == 1
        assert not (tmp_path / "bundle").exists()

    def test_the_author_is_credited_when_given(self, chain_store, tmp_path):
        """A local run has no authenticated identity, so this is the only way
        a lone researcher's name reaches the page."""
        dest = self._archive(chain_store, tmp_path)

        assert "F. Li" in (dest / "index.html").read_text()
