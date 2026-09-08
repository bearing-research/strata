"""Copying an artifact into a store on another machine.

A chain lives in the store its cells wrote to; a link resolves from the store
the server serves. On a hosted deployment those are different machines, so the
chain has to travel — with its versions intact, because lineage edges are
recorded as ``id@v=N`` and a copy that assigned fresh versions would land
ancestors under numbers the descendants' edges do not name.
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
        """The whole point: fresh versions would break every edge naming it."""
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
        """Two people whose notebooks ran the identical cell.

        The store permits one ready row per tenant and provenance hash, so the
        second import cannot land as itself. It has to say where it went, or
        the caller's descendants name a row this store never received.
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
        """The team cache writes results with no inputs at all.

        It fires on every successful cell while promotion is deliberate, so the
        cache almost always gets there first. Without this, a promoted chain
        resolves exactly one level before reaching an artifact naming no
        inputs.
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


class TestPublishTo:
    """``strata artifact publish --to`` end to end.

    The case the route exists for: a chain in a notebook's own store, published
    to a server on another machine, with the link resolving there.
    """

    def test_a_chain_travels_and_the_link_resolves_on_the_far_side(
        self, client, tmp_path, served_dir
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
        """A chain half-copied to a store that refused it must say so."""
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
