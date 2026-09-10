"""Who wrote a publication, and what identifies it.

``published_by`` was one free-form string: who made the grant. That is not who
wrote the work — which has an order, an affiliation and an identifier — and it
is not a DOI, which is registered against a deposit that already has to be
reachable and so almost always arrives after the token does. Item 3.
"""

from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from strata.artifact_store import ArtifactStore

ORCID = "0000-0002-1825-0097"


@pytest.fixture
def store(tmp_path):
    store = ArtifactStore(tmp_path / "artifacts")
    version = store.create_artifact("fig", "a" * 64)
    store.write_blob("fig", version, b"PNG")
    store.finalize_artifact("fig", version, '{"fields": []}', 1, 3)
    return store


@pytest.fixture
def served(tmp_path):
    """A running server holding one published artifact."""
    from tests.conftest import run_server_with_context

    artifact_dir = tmp_path / "served"
    with run_server_with_context(tmp_path / "cache", artifact_dir, "personal") as ctx:
        store = ArtifactStore(artifact_dir)
        version = store.create_artifact("fig", hashlib.sha256(b"PNG").hexdigest())
        store.write_blob("fig", version, b"PNG")
        store.finalize_artifact("fig", version, '{"fields": []}', 1, 3)
        yield ctx.base_url, store, version


class TestPublishingWithAuthors:
    def test_they_are_kept_in_the_order_given(self, store):
        """Author order carries meaning, so it is preserved rather than sorted."""
        publication = store.publish_artifact(
            "fig",
            1,
            authors=[{"name": "B. Second"}, {"name": "A. First", "orcid": ORCID}],
        )

        assert [a["name"] for a in publication.authors] == ["B. Second", "A. First"]

    def test_an_author_without_an_orcid_carries_no_empty_one(self, store):
        """Otherwise a missing identifier becomes the string "None" on a page."""
        publication = store.publish_artifact("fig", 1, authors=[{"name": "Solo"}])

        assert publication.authors == ({"name": "Solo"},)

    def test_they_survive_a_round_trip(self, store):
        publication = store.publish_artifact(
            "fig", 1, authors=[{"name": "F. Li", "orcid": ORCID, "affiliation": "Somewhere"}]
        )

        reloaded = store.get_publication(publication.token)

        assert reloaded.authors == publication.authors

    def test_publishing_without_them_changes_nothing(self, store):
        """The property that keeps every link printed before this saying what
        it said: no authors leaves ``published_by`` as the byline."""
        publication = store.publish_artifact("fig", 1, published_by="alice")

        assert publication.authors == ()
        assert publication.published_by == "alice"


class TestPatchingIdentifiers:
    def test_a_doi_can_be_added_after_the_fact(self, store):
        publication = store.publish_artifact("fig", 1)

        updated = store.update_publication_credits(
            publication.token, external_ids=[{"scheme": "doi", "value": "10.5281/zenodo.1"}]
        )

        assert updated.external_ids == ({"scheme": "doi", "value": "10.5281/zenodo.1"},)

    def test_patching_one_field_leaves_the_other_alone(self, store):
        publication = store.publish_artifact("fig", 1, authors=[{"name": "F. Li"}])

        updated = store.update_publication_credits(
            publication.token, external_ids=[{"scheme": "doi", "value": "10.1/x"}]
        )

        assert [a["name"] for a in updated.authors] == ["F. Li"]

    def test_it_cannot_repoint_the_token(self, store):
        """The binding is the whole value of a URL printed in a paper. It is
        not a parameter here and not a column this write names."""
        second = store.create_artifact("other", "b" * 64)
        store.write_blob("other", second, b"OTHER")
        store.finalize_artifact("other", second, '{"fields": []}', 1, 5)
        publication = store.publish_artifact("fig", 1)

        store.update_publication_credits(
            publication.token, external_ids=[{"scheme": "doi", "value": "10.1/x"}]
        )

        reloaded = store.get_publication(publication.token)
        assert (reloaded.artifact_id, reloaded.version) == ("fig", 1)

    def test_an_unknown_token_is_not_silently_a_success(self, store):
        assert store.update_publication_credits("nope", external_ids=[]) is None


class TestThePage:
    def _page(self, store, **kwargs):
        from strata.api.publication_page import render_publication
        from strata.services.artifact import ArtifactService

        publication = store.publish_artifact("fig", 1, **kwargs)
        artifact = store.get_artifact("fig", 1)
        lineage = ArtifactService().build_lineage(
            store, artifact=artifact, artifact_id="fig", version=1, tenant_filter=None, max_depth=5
        )
        return render_publication(
            publication=publication,
            artifact=artifact,
            lineage=lineage,
            content_type="image/png",
            image_src=None,
        )

    def test_authors_appear_with_their_orcid_links(self, store):
        html = self._page(store, authors=[{"name": "F. Li", "orcid": ORCID}, {"name": "B. Second"}])

        assert f"https://orcid.org/{ORCID}" in html
        assert "by <a" in html
        assert "F. Li" in html and "B. Second" in html

    def test_the_byline_falls_back_to_who_published(self, store):
        html = self._page(store, published_by="alice")

        assert "by alice" in html

    def test_no_identifiers_means_no_citation_line(self, store):
        """A citation line saying nothing reads as "there is no DOI for this",
        which is a different claim from nobody having recorded one."""
        assert "Cite as" not in self._page(store)


class TestTheCrate:
    def _crate(self, store, **kwargs):
        from strata.api.provenance_ld import build_crate
        from strata.services.artifact import ArtifactService

        publication = store.publish_artifact("fig", 1, **kwargs)
        store.update_publication_credits(
            publication.token, external_ids=[{"scheme": "doi", "value": "10.5281/zenodo.1"}]
        )
        publication = store.get_publication(publication.token)
        artifact = store.get_artifact("fig", 1)
        lineage = ArtifactService().build_lineage(
            store, artifact=artifact, artifact_id="fig", version=1, tenant_filter=None, max_depth=5
        )
        return build_crate(
            publication=publication,
            artifact=artifact,
            lineage=lineage,
            content_type="image/png",
            payload_id="artifact.png",
            include_descriptor=False,
        )

    def test_the_root_identifier_is_the_resolvable_doi(self, store):
        """A bare "10.5281/zenodo.1" is not something an ingesting repository
        can follow; the URL form is."""
        crate = self._crate(store, authors=[{"name": "F. Li"}])

        root = next(n for n in crate["@graph"] if n["@id"] == "./")
        assert root["identifier"] == "https://doi.org/10.5281/zenodo.1"

    def test_an_author_with_an_orcid_is_identified_by_it(self, store):
        """An ORCID is a persistent identifier for a person, which is exactly
        what an ``@id`` is for — two crates naming the same researcher say so."""
        crate = self._crate(store, authors=[{"name": "F. Li", "orcid": ORCID}])

        root = next(n for n in crate["@graph"] if n["@id"] == "./")
        assert root["author"] == [{"@id": f"https://orcid.org/{ORCID}"}]
        person = next(n for n in crate["@graph"] if n["@id"] == f"https://orcid.org/{ORCID}")
        assert person["@type"] == "Person"
        assert person["name"] == "F. Li"


class TestTheRoute:
    def test_publishing_and_then_patching_shows_up_in_the_record(self, served):
        base_url, store, version = served

        published = httpx.post(
            f"{base_url}/v1/artifacts/fig/v/{version}/publish",
            json={"title": "Figure 1", "authors": [{"name": "F. Li", "orcid": ORCID}]},
            timeout=10,
        ).json()
        patched = httpx.patch(
            f"{base_url}/v1/publications/{published['token']}",
            json={"external_ids": [{"scheme": "doi", "value": "10.5281/zenodo.1"}]},
            timeout=10,
        )

        assert patched.status_code == 200
        assert patched.json()["external_ids"] == [{"scheme": "doi", "value": "10.5281/zenodo.1"}]
        record = httpx.get(f"{base_url}/v1/publications/{published['token']}", timeout=10).json()
        assert record["publication"]["authors"][0]["orcid"] == ORCID
        assert record["publication"]["external_ids"][0]["value"] == "10.5281/zenodo.1"

    def test_an_unresolvable_scheme_is_refused(self, served):
        """A record that accepted any scheme name would produce citation lines
        nobody can follow, and the caller would hear about it from a reader."""
        base_url, store, version = served

        published = httpx.post(
            f"{base_url}/v1/artifacts/fig/v/{version}/publish", json={}, timeout=10
        ).json()
        response = httpx.patch(
            f"{base_url}/v1/publications/{published['token']}",
            json={"external_ids": [{"scheme": "made-up", "value": "x"}]},
            timeout=10,
        )

        assert response.status_code == 422

    def test_patching_an_unknown_token_is_a_404(self, served):
        base_url, _, _ = served

        response = httpx.patch(
            f"{base_url}/v1/publications/not-a-token", json={"authors": []}, timeout=10
        )

        assert response.status_code == 404


class TestTheArchiveManifest:
    def test_it_carries_both(self, store, tmp_path):
        """The escrow bundle lacking the DOI is the reason this is Phase 1."""
        from strata.api.publication_bundle import write_bundle

        publication = store.publish_artifact("fig", 1, authors=[{"name": "F. Li", "orcid": ORCID}])
        store.update_publication_credits(
            publication.token, external_ids=[{"scheme": "doi", "value": "10.5281/zenodo.1"}]
        )
        publication = store.get_publication(publication.token)

        dest = tmp_path / "bundle"
        dest.mkdir()
        write_bundle(store, store.get_artifact("fig", 1), dest, publication=publication)

        manifest = json.loads((dest / "manifest.json").read_text())
        assert manifest["archive"]["authors"][0]["orcid"] == ORCID
        assert manifest["archive"]["external_ids"][0]["value"] == "10.5281/zenodo.1"


class TestMigration:
    def test_a_store_written_before_the_columns_gains_them(self, tmp_path):
        """Nothing is backfilled: ``published_by`` stays the byline for every
        publication made before this, which is what keeps a link printed in a
        paper saying what it said yesterday."""
        store = ArtifactStore(tmp_path / "old")
        version = store.create_artifact("fig", "c" * 64)
        store.write_blob("fig", version, b"PNG")
        store.finalize_artifact("fig", version, '{"fields": []}', 1, 3)
        publication = store.publish_artifact("fig", version, published_by="alice")

        conn = store._get_connection()
        try:
            conn.execute("UPDATE artifact_publications SET authors = NULL, external_ids = NULL")
            conn.commit()
        finally:
            conn.close()

        reloaded = ArtifactStore(tmp_path / "old").get_publication(publication.token)
        assert reloaded.authors == ()
        assert reloaded.external_ids == ()
        assert reloaded.published_by == "alice"
