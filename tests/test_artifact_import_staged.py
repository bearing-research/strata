"""The import in two steps: the bytes, then the record that names them.

``PUT /v1/artifacts/import/blobs/{sha256}`` uploads bytes; ``POST /v1/artifacts/import`` then
imports the record whose ``content_sha256`` names them. Large artifacts never sit in memory.
"""

from __future__ import annotations

import hashlib

import httpx
import pytest

from strata.artifact_store import IMPORT_STAGING_TTL_SECONDS, ArtifactStore

PROXY_TOKEN = "import-proxy-token"
BLOB = b"the bytes of a figure"
DIGEST = hashlib.sha256(BLOB).hexdigest()


def _record(artifact_id: str = "fig", version: int = 1, **extra) -> dict:
    return {
        "id": artifact_id,
        "version": version,
        "state": "ready",
        "provenance_hash": hashlib.sha256(artifact_id.encode()).hexdigest(),
        "created_at": 1.0,
        "content_sha256": DIGEST,
        **extra,
    }


def _stage(base: str, blob: bytes = BLOB, digest: str = DIGEST, headers=None):
    return httpx.put(
        f"{base}/v1/artifacts/import/blobs/{digest}", content=blob, headers=headers, timeout=10
    )


def _import(base: str, record: dict, headers=None, **params):
    return httpx.post(
        f"{base}/v1/artifacts/import", json=record, params=params, headers=headers, timeout=10
    )


@pytest.fixture
def served(tmp_path):
    from tests.conftest import run_server_with_context

    artifact_dir = tmp_path / "served"
    with run_server_with_context(tmp_path / "cache", artifact_dir, "personal") as ctx:
        yield ctx.base_url, artifact_dir


class TestTwoSteps:
    def test_the_upload_then_the_record_imports_the_bytes(self, served):
        base, artifact_dir = served

        staged = _stage(base)
        assert staged.status_code == 201, staged.text
        assert staged.json() == {"content_sha256": DIGEST, "byte_size": len(BLOB)}

        imported = _import(base, _record())
        assert imported.status_code == 200, imported.text
        assert imported.json()["written"] is True

        store = ArtifactStore(artifact_dir)
        assert store.read_blob("fig", 1) == BLOB
        assert store.get_artifact("fig", 1).content_sha256 == DIGEST

    def test_the_upload_is_let_go_once_imported(self, served):
        base, artifact_dir = served
        _stage(base)
        _import(base, _record())

        assert ArtifactStore(artifact_dir).open_staged_import(None, DIGEST) is None

    def test_a_retry_after_the_import_went_through_writes_nothing(self, served):
        """The upload is gone by then, and the record needs none."""
        base, _ = served
        _stage(base)
        _import(base, _record())

        again = _import(base, _record())
        assert again.status_code == 200, again.text
        assert again.json()["written"] is False


class TestRefusals:
    def test_bytes_that_do_not_match_the_path_are_not_kept(self, served):
        base, _ = served

        assert _stage(base, blob=b"something else").status_code == 400
        assert _import(base, _record()).status_code == 400

    def test_a_record_with_nothing_uploaded_is_refused_naming_the_upload(self, served):
        base, artifact_dir = served

        response = _import(base, _record())
        assert response.status_code == 400
        assert f"/v1/artifacts/import/blobs/{DIGEST}" in response.json()["detail"]
        assert ArtifactStore(artifact_dir).get_artifact("fig", 1) is None

    def test_a_record_without_its_creation_time_is_a_400(self, served):
        """The column is NOT NULL, so this must be a 400, not a database 500."""
        base, artifact_dir = served
        _stage(base)
        record = _record()
        del record["created_at"]

        assert _import(base, record).status_code == 400
        assert ArtifactStore(artifact_dir).get_artifact("fig", 1) is None

    def test_a_json_import_names_its_bytes(self, served):
        base, _ = served
        record = _record()
        del record["content_sha256"]

        assert _import(base, record).status_code == 400

    @pytest.mark.parametrize("version", [0, -1])
    def test_a_version_below_one_is_refused(self, served, version):
        """Staged uploads wait under version 0, which no artifact has."""
        base, _ = served
        _stage(base)

        assert _import(base, _record(version=version)).status_code == 400

    def test_the_digest_in_the_path_must_be_one(self, served):
        base, _ = served

        assert _stage(base, digest="not-a-digest").status_code == 422


class TestAbandonedUploads:
    def test_an_upload_nothing_imports_is_dropped_a_day_later(self, tmp_path):
        store = ArtifactStore(tmp_path / "store")
        source = tmp_path / "blob"
        source.write_bytes(BLOB)
        other = hashlib.sha256(b"later").hexdigest()

        store.stage_import_blob(None, DIGEST, source, len(BLOB), now=0.0)
        store.stage_import_blob(None, other, source, len(BLOB), now=IMPORT_STAGING_TTL_SECONDS - 1)
        assert store.open_staged_import(None, DIGEST) is not None

        store.stage_import_blob(None, other, source, len(BLOB), now=IMPORT_STAGING_TTL_SECONDS + 1)
        assert store.open_staged_import(None, DIGEST) is None
        assert store.open_staged_import(None, other) is not None


def _headers(tenant: str, principal: str = "publisher", scopes: str = "artifacts:write"):
    return {
        "X-Strata-Proxy-Token": PROXY_TOKEN,
        "X-Strata-Principal": principal,
        "X-Tenant-ID": tenant,
        "X-Strata-Scopes": scopes,
    }


@pytest.fixture
def central(tmp_path):
    """A central store: service mode behind a trusted proxy, one tenant per team."""
    from tests.conftest import run_server_with_context

    artifact_dir = tmp_path / "central"
    with run_server_with_context(
        tmp_path / "cache",
        artifact_dir,
        "service",
        auth_mode="trusted_proxy",
        proxy_token=PROXY_TOKEN,
        multi_tenant_enabled=True,
        service_writes_enabled=True,
        hide_forbidden_as_not_found=False,
    ) as ctx:
        yield ctx.base_url, artifact_dir


class TestInACentralStore:
    def test_the_callers_tenant_is_stamped_whatever_the_record_says(self, central):
        base, artifact_dir = central
        _stage(base, headers=_headers("team-a"))

        response = _import(base, _record(tenant="team-b"), headers=_headers("team-a"))
        assert response.status_code == 200, response.text
        assert ArtifactStore(artifact_dir).get_artifact("fig", 1).tenant == "team-a"

    def test_one_tenants_upload_does_not_satisfy_anothers_import(self, central):
        """A digest is printed on every publication page, so knowing it does not prove holding the
        bytes.
        """
        base, artifact_dir = central
        assert _stage(base, headers=_headers("team-a")).status_code == 201

        response = _import(base, _record(), headers=_headers("team-b"))
        assert response.status_code == 400
        assert ArtifactStore(artifact_dir).get_artifact("fig", 1) is None

    def test_a_version_another_tenant_holds_is_a_409_and_remap_mints_a_fresh_id(self, central):
        base, _ = central
        _stage(base, headers=_headers("team-a"))
        _import(base, _record(), headers=_headers("team-a"))
        _stage(base, headers=_headers("team-b"))

        clash = _import(base, _record(), headers=_headers("team-b"))
        assert clash.status_code == 409

        remapped = _import(base, _record(), headers=_headers("team-b"), remap="true")
        assert remapped.status_code == 200, remapped.text
        assert remapped.json()["remapped"] is True
        assert remapped.json()["id"] != "fig"

    @pytest.mark.parametrize("step", ["stage", "import"])
    def test_without_the_write_scope_nothing_is_taken(self, central, step):
        base, _ = central
        reader = _headers("team-a", scopes="artifacts:read")

        response = (
            _stage(base, headers=reader)
            if step == "stage"
            else _import(base, _record(), headers=reader)
        )
        assert response.status_code == 403


def _chain(directory):
    """A notebook's two-step chain: rows, and a figure drawn from them."""
    from strata.notebook.artifact_integration import NotebookArtifactManager

    manager = NotebookArtifactManager("nb", artifact_dir=directory)
    rows = manager.store_cell_output(
        cell_id="c1",
        variable_name="rows",
        blob_data=b"[1, 2]",
        content_type="json/object",
        provenance_hash="a1" * 32,
        input_versions={},
        source="rows = [1, 2]",
    )
    ref = f"{rows.id}@v={rows.version}"
    figure = manager.store_cell_output(
        cell_id="c2",
        variable_name="__display__0",
        blob_data=b"PNG",
        content_type="image/png",
        provenance_hash="b2" * 32,
        input_versions={f"strata://artifact/{ref}": ref},
        source="plt.plot(rows)",
    )
    return rows, figure


def _publish_to(base: str, local, ref: str, tenant: str | None = None) -> int:
    import argparse

    from strata.artifact_cli import cmd_publish

    scopes = "artifacts:write artifacts:publish"
    headers = [f"{k}: {v}" for k, v in _headers(tenant, scopes=scopes).items()] if tenant else None
    return cmd_publish(
        argparse.Namespace(
            ref=ref,
            artifact_dir=str(local),
            format="human",
            title=None,
            author=None,
            here=False,
            into=None,
            to_url=base,
            header=headers,
            max_depth=10,
        )
    )


def _row_count(artifact_dir) -> int:
    conn = ArtifactStore(artifact_dir)._get_connection()
    try:
        return conn.execute("SELECT COUNT(*) FROM artifact_versions").fetchone()[0]
    finally:
        conn.close()


class TestPublishTo:
    """``strata artifact publish --to`` uses the staged import and remaps a clash."""

    def test_the_bytes_are_staged_and_the_record_posted_as_json(
        self, served, tmp_path, monkeypatch
    ):
        base, artifact_dir = served
        _, figure = _chain(tmp_path / "notebook")
        calls = []
        bodies = []
        real_put, real_post = httpx.put, httpx.post

        def put(url, **kwargs):
            calls.append(("PUT", url.removeprefix(base), sorted(kwargs)))
            bodies.append(kwargs["content"])
            return real_put(url, **kwargs)

        def post(url, **kwargs):
            calls.append(("POST", url.removeprefix(base), sorted(kwargs)))
            return real_post(url, **kwargs)

        monkeypatch.setattr(httpx, "put", put)
        monkeypatch.setattr(httpx, "post", post)

        assert _publish_to(base, tmp_path / "notebook", figure.id) == 0

        imports = [(verb, path, keys) for verb, path, keys in calls if "/import" in path]
        assert [verb for verb, _, _ in imports] == ["PUT", "POST", "PUT", "POST"]
        assert all(path.startswith("/v1/artifacts/import/blobs/") for v, path, _ in imports[::2])
        assert all("json" in keys and "files" not in keys for _, _, keys in imports[1::2])
        # Streamed from a spooled file, never the artifact's bytes held whole.
        assert not any(isinstance(body, bytes) for body in bodies)
        store = ArtifactStore(artifact_dir)
        assert store.read_blob(figure.id, figure.version) == b"PNG"
        assert store.get_artifact(figure.id, figure.version).content_sha256 == (
            hashlib.sha256(b"PNG").hexdigest()
        )

    def test_a_chain_another_tenant_holds_lands_as_this_tenants_copy(self, central, tmp_path):
        """Every edge resolves on the copy and every provenance hash is unchanged."""
        from strata.services.artifact import ArtifactService

        base, artifact_dir = central
        rows, figure = _chain(tmp_path / "notebook")
        assert _publish_to(base, tmp_path / "notebook", figure.id, tenant="team-a") == 0

        assert _publish_to(base, tmp_path / "notebook", figure.id, tenant="team-b") == 0

        store = ArtifactStore(artifact_dir)
        theirs = store.find_by_provenance(figure.provenance_hash, "team-b")
        assert theirs is not None and theirs.id != figure.id
        lineage = ArtifactService().build_lineage(
            store,
            artifact=theirs,
            artifact_id=theirs.id,
            version=theirs.version,
            tenant_filter="team-b",
            max_depth=10,
        )
        nodes = [store.get_artifact(n.artifact_id, n.version) for n in lineage.nodes]
        assert [n.provenance_hash for n in nodes] == [figure.provenance_hash, rows.provenance_hash]
        assert all(n.tenant == "team-b" for n in nodes)
        assert len(store.list_publications("team-a")) == len(store.list_publications("team-b")) == 1

    def test_publishing_the_same_chain_again_writes_nothing(self, central, tmp_path):
        base, artifact_dir = central
        _, figure = _chain(tmp_path / "notebook")
        _publish_to(base, tmp_path / "notebook", figure.id, tenant="team-a")
        _publish_to(base, tmp_path / "notebook", figure.id, tenant="team-b")
        before = _row_count(artifact_dir)

        assert _publish_to(base, tmp_path / "notebook", figure.id, tenant="team-b") == 0

        assert _row_count(artifact_dir) == before == 4
