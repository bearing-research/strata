"""Copying an artifact into a store on another machine.

Versions must survive: lineage edges are recorded as ``id@v=N``, so fresh versions would orphan the
descendants' edges.
"""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest


def _metadata(artifact_id: str, version: int, provenance: str, **extra) -> dict:
    return {
        "id": artifact_id,
        "version": version,
        "state": "ready",
        "provenance_hash": provenance,
        "created_at": 1.0,
        **extra,
    }


def _post(client: str, metadata: dict, blob: bytes | None = b"x", **params):
    files = {"metadata": ("metadata.json", json.dumps(metadata), "application/json")}
    if blob is not None:
        files["data"] = ("data.bin", blob, "application/octet-stream")
    return httpx.post(f"{client}/v1/artifacts/import", files=files, params=params, timeout=10)


@pytest.fixture
def served_dir(tmp_path):
    return tmp_path / "served"


@pytest.fixture
def client(tmp_path, served_dir):
    """A real server, because the route's auth gate only runs on a real path."""
    from tests.conftest import run_server_with_context

    with run_server_with_context(tmp_path / "cache", served_dir, "personal") as ctx:
        yield ctx.base_url


class TestImport:
    def test_a_record_keeps_its_id_and_version(self, client):
        """Fresh versions would break every edge naming it."""
        body = _post(client, _metadata("fig", 7, "a" * 64)).json()

        assert body["id"] == "fig"
        assert body["version"] == 7
        assert body["written"] is True
        assert body["artifact_uri"] == "strata://artifact/fig@v=7"

    def test_importing_twice_writes_nothing_the_second_time(self, client):
        _post(client, _metadata("fig", 1, "b" * 64))
        body = _post(client, _metadata("fig", 1, "b" * 64)).json()

        assert body["written"] is False

    def test_the_same_computation_under_another_id_resolves_onto_the_first(self, client):
        """One ready row per tenant and provenance, so a second identical computation lands on the
        first and the response must say where.
        """
        _post(client, _metadata("nb_alice_cell_c1_var_rows", 1, "c" * 64))

        body = _post(client, _metadata("nb_bob_cell_c1_var_rows", 1, "c" * 64)).json()

        assert body["id"] == "nb_alice_cell_c1_var_rows"
        assert body["written"] is False
        assert body["remapped"] is True


class TestIdempotencyByCompleteness:
    """A record already here but missing something is completed, not skipped."""

    def test_a_row_whose_bytes_are_missing_is_repaired(self, client, served_dir):
        from strata.artifact_store import ArtifactStore

        _post(client, _metadata("fig", 1, "d" * 64), blob=b"payload")
        store = ArtifactStore(served_dir)
        blob_path = store._blob_path("fig", 1)
        blob_path.unlink()
        assert not store.blob_exists("fig", 1)

        body = _post(client, _metadata("fig", 1, "d" * 64), blob=b"payload").json()

        assert body["written"] is False
        assert store.blob_exists("fig", 1), "a retry must repair what an interruption lost"

    def test_lineage_completes_a_row_that_had_none(self, client, served_dir):
        """The team cache writes results with no inputs and usually gets there first, so promotion
        must fill in their lineage.
        """
        from strata.artifact_store import ArtifactStore

        _post(client, _metadata("rows", 1, "e" * 64))  # as the cache would write it

        edges = json.dumps({"strata://artifact/up@v=1": "up@v=1"})
        _post(client, _metadata("rows", 1, "e" * 64, input_versions=edges))

        store = ArtifactStore(served_dir)
        assert store.get_artifact("rows", 1).input_versions == edges

    def test_existing_lineage_is_never_rewritten(self, client, served_dir):
        """An import may add history. It may not revise it."""
        from strata.artifact_store import ArtifactStore

        original = json.dumps({"strata://artifact/first@v=1": "first@v=1"})
        _post(client, _metadata("rows", 1, "f" * 64, input_versions=original))
        _post(
            client,
            _metadata(
                "rows",
                1,
                "f" * 64,
                input_versions=json.dumps({"strata://artifact/other@v=9": "other@v=9"}),
            ),
        )

        store = ArtifactStore(served_dir)
        assert store.get_artifact("rows", 1).input_versions == original


class TestAnIdTwoComputationsClaim:
    """Notebook ids derive from notebook and cell ids, so two people from one repo send the same id
    for differently edited cells.
    """

    def test_a_different_computation_under_a_held_id_is_refused(self, client, served_dir):
        from strata.artifact_store import ArtifactStore

        assert _post(client, _metadata("shared", 1, "a" * 64), blob=b"ALICE").status_code == 200

        response = _post(client, _metadata("shared", 1, "b" * 64), blob=b"BOBBY")

        assert response.status_code == 409
        assert "different computation" in response.json()["detail"]
        store = ArtifactStore(served_dir)
        assert store.read_blob("shared", 1) == b"ALICE", "and nobody's bytes were replaced"

    def test_remap_lands_it_under_a_fresh_id(self, client, served_dir):
        """Keep both, as for the tenant clash."""
        from strata.artifact_store import ArtifactStore

        _post(client, _metadata("shared", 1, "a" * 64), blob=b"ALICE")

        body = _post(client, _metadata("shared", 1, "b" * 64), blob=b"BOBBY", remap="true").json()

        assert body["written"] is True
        assert body["id"] != "shared"
        store = ArtifactStore(served_dir)
        assert store.read_blob("shared", 1) == b"ALICE"
        assert store.read_blob(body["id"], body["version"]) == b"BOBBY"

    def test_the_same_computation_is_still_a_no_op(self, client):
        """A repeated import is what the id check is for."""
        _post(client, _metadata("shared", 1, "a" * 64), blob=b"ALICE")

        body = _post(client, _metadata("shared", 1, "a" * 64), blob=b"ALICE").json()

        assert body["written"] is False


class TestIntegrity:
    def test_bytes_contradicting_the_declared_digest_are_refused(self, client):
        """Storing them would publish a page whose verify step fails."""
        metadata = _metadata("fig", 1, "g" * 64, content_sha256=hashlib.sha256(b"real").hexdigest())

        response = _post(client, metadata, blob=b"different")

        assert response.status_code == 400
        assert "digest" in response.json()["detail"]

    def test_matching_bytes_are_accepted(self, client):
        payload = b"real"
        metadata = _metadata("fig", 1, "h" * 64, content_sha256=hashlib.sha256(payload).hexdigest())

        assert _post(client, metadata, blob=payload).status_code == 200

    @pytest.mark.parametrize("missing", ["id", "version", "provenance_hash"])
    def test_a_record_missing_an_identifier_is_refused(self, client, missing):
        metadata = _metadata("fig", 1, "i" * 64)
        del metadata[missing]

        assert _post(client, metadata).status_code == 400

    @pytest.mark.parametrize(
        "field",
        [
            {"byte_size": "big"},
            {"byte_size": -1},
            {"row_count": 1.5},
            {"row_count": True},
            {"input_versions": {"strata://artifact/a@v=1": "a@v=1"}},
            {"input_versions": "[1, 2]"},
            {"transform_spec": "not json"},
            {"schema_json": 3},
            {"principal": ["ana"]},
            {"state": "building"},
        ],
    )
    def test_a_field_of_the_wrong_shape_is_refused(self, client, served_dir, field):
        """Stored unchecked, a string ``byte_size`` broke every later sweep with a TypeError."""
        from strata.artifact_store import ArtifactStore

        response = _post(client, _metadata("fig", 1, "k" * 64, **field))

        assert response.status_code == 400
        assert ArtifactStore(served_dir).get_artifact("fig", 1) is None

    def test_well_formed_fields_are_kept(self, client, served_dir):
        from strata.artifact_store import ArtifactStore

        edges = json.dumps({"strata://artifact/a@v=1": "a@v=1"})
        metadata = _metadata(
            "fig", 1, "l" * 64, byte_size=1, row_count=0, input_versions=edges, state="superseded"
        )

        assert _post(client, metadata).status_code == 200
        stored = ArtifactStore(served_dir).get_artifact("fig", 1)
        assert (stored.byte_size, stored.input_versions, stored.state) == (1, edges, "superseded")

    @pytest.mark.parametrize("artifact_id", ["../../../../victim/pwned", "/etc/pwned", "a/b", ".."])
    def test_an_id_that_names_a_path_is_refused(self, client, served_dir, artifact_id):
        """The id from the request becomes a blob key, so a path-like id must not write anywhere."""
        response = _post(client, _metadata(artifact_id, 1, "j" * 64), blob=b"pwned")

        assert response.status_code == 400
        assert not list(served_dir.parent.rglob("*pwned*@v=1.arrow"))


class TestPublishTo:
    """``strata artifact publish --to`` end to end: a notebook's chain sent to a remote server."""

    def test_a_chain_travels_and_the_link_resolves_on_the_far_side(
        self, client, tmp_path, served_dir, capsys
    ):
        import argparse

        from strata.artifact_cli import cmd_publish
        from strata.artifact_store import ArtifactStore
        from strata.notebook.artifact_integration import NotebookArtifactManager
        from strata.services.artifact import ArtifactService

        local = tmp_path / "notebook"
        manager = NotebookArtifactManager("nb", artifact_dir=local)
        upstream = manager.store_cell_output(
            cell_id="c1",
            variable_name="rows",
            blob_data=b"[1]",
            content_type="json/object",
            provenance_hash="a1" * 32,
            input_versions={},
            source="rows = [1]",
        )
        ref = f"{upstream.id}@v={upstream.version}"
        figure = manager.store_cell_output(
            cell_id="c2",
            variable_name="__display__0",
            blob_data=b"PNG",
            content_type="image/png",
            provenance_hash="b2" * 32,
            input_versions={f"strata://artifact/{ref}": ref},
            source="plt.plot(rows)",
        )

        rc = cmd_publish(
            argparse.Namespace(
                ref=figure.id,
                artifact_dir=str(local),
                format="human",
                title="Figure 3",
                author=None,
                here=False,
                into=None,
                to_url=client,
                header=None,
                max_depth=10,
            )
        )
        assert rc == 0
        # unpublish only reaches a local store, so it would answer "No active publication".
        assert "Withdraw it with" not in capsys.readouterr().out

        served = ArtifactStore(served_dir)
        published = served.list_publications()
        assert len(published) == 1, "the grant must be minted where the link resolves"

        # And the chain arrived, resolvable from the far store alone.
        copied = served.get_artifact(figure.id, figure.version)
        assert copied is not None
        lineage = ArtifactService().build_lineage(
            served,
            artifact=copied,
            artifact_id=copied.id,
            version=copied.version,
            tenant_filter=None,
            max_depth=10,
        )
        assert [n.artifact_id for n in lineage.nodes if n.type == "artifact"] == [
            figure.id,
            upstream.id,
        ]

    def test_a_refusal_from_the_far_side_is_reported_not_swallowed(self, tmp_path):
        """A chain half-copied to a store that refused it must report the refusal."""
        import argparse

        from strata.artifact_cli import cmd_publish
        from strata.notebook.artifact_integration import NotebookArtifactManager

        local = tmp_path / "notebook"
        manager = NotebookArtifactManager("nb", artifact_dir=local)
        figure = manager.store_cell_output(
            cell_id="c1",
            variable_name="fig",
            blob_data=b"PNG",
            content_type="image/png",
            provenance_hash="c3" * 32,
            input_versions={},
            source="fig = 1",
        )

        with pytest.raises((RuntimeError, httpx.HTTPError)):
            cmd_publish(
                argparse.Namespace(
                    ref=figure.id,
                    artifact_dir=str(local),
                    format="human",
                    title=None,
                    author=None,
                    here=False,
                    into=None,
                    to_url="http://127.0.0.1:1",  # nothing listening
                    header=None,
                    max_depth=10,
                )
            )
