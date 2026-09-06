"""Publications: opt-in public read grants and the page they resolve to.

The security surface here is unusual for this codebase — these are the only
routes that answer an unauthenticated caller with artifact contents. Most of
what follows guards the *edges* of that: which routes are exempt from auth,
what a revoked token does, and whether user-written text reaches the page
unescaped.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from strata.artifact_store import ArtifactStore


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


def _ready_artifact(store: ArtifactStore, artifact_id: str, payload: bytes) -> int:
    version = store.create_artifact(artifact_id, hashlib.sha256(payload).hexdigest())
    with store.open_blob_writer(artifact_id, version) as writer:
        writer.write(payload)
    store.finalize_artifact(
        artifact_id, version, schema_json="", row_count=0, byte_size=len(payload)
    )
    return version


class TestPublicationGrants:
    def test_publishing_records_the_digest_of_the_published_bytes(self, store):
        """The only checkable integrity claim the page can make.

        Nothing else in the store records a digest of the *content* —
        ``provenance_hash`` covers inputs and transform — so without this the
        page could assert nothing a reader is able to test.
        """
        payload = b"figure-bytes"
        version = _ready_artifact(store, "fig", payload)

        publication = store.publish_artifact("fig", version)

        assert publication.content_sha256 == hashlib.sha256(payload).hexdigest()

    def test_publishing_twice_returns_the_same_token(self, store):
        """Two live URLs for one artifact would make revocation a lie.

        Someone withdrawing "the" link would believe the artifact was no longer
        public while the other token still served it.
        """
        version = _ready_artifact(store, "fig", b"x")

        first = store.publish_artifact("fig", version)
        second = store.publish_artifact("fig", version)

        assert first.token == second.token

    def test_revoking_keeps_the_row_so_the_token_is_never_reissued(self, store):
        """A citation must fail closed, never start resolving to other content."""
        version = _ready_artifact(store, "fig", b"x")
        publication = store.publish_artifact("fig", version)

        assert store.revoke_publication(publication.token) is True
        assert store.revoke_publication(publication.token) is False, "revoke is one-way"

        after = store.get_publication(publication.token)
        assert after is not None, "the token must still resolve, to say 'withdrawn'"
        assert not after.is_active

    def test_revoked_grants_are_out_of_the_listing_by_default(self, store):
        version = _ready_artifact(store, "fig", b"x")
        publication = store.publish_artifact("fig", version)
        store.revoke_publication(publication.token)

        assert store.list_publications() == []
        assert len(store.list_publications(include_revoked=True)) == 1

    def test_an_unknown_artifact_cannot_be_published(self, store):
        with pytest.raises(ValueError, match="not found"):
            store.publish_artifact("nope", 1)

    def test_a_building_artifact_cannot_be_published(self, store):
        """Publishing a half-written artifact would hand out a moving target."""
        store.create_artifact("half", "a" * 64)

        with pytest.raises(ValueError, match="not readable"):
            store.publish_artifact("half", 1)


class TestAuthExemption:
    """Which routes the auth and tenant middleware let through unauthenticated.

    This predicate is the whole boundary. If it widens by accident, routes that
    mint or withdraw grants — or enumerate what a tenant has published — become
    reachable by anyone who can address the server.
    """

    @staticmethod
    def _request(method: str, path: str):
        from types import SimpleNamespace

        return SimpleNamespace(method=method, url=SimpleNamespace(path=path))

    @pytest.mark.parametrize(
        "path",
        [
            "/p/sometoken",
            "/p/sometoken/data",
            "/p/sometoken/verify",
            "/v1/publications/sometoken",
        ],
    )
    def test_public_reads_are_exempt(self, path):
        from strata.server import _is_public_publication_request

        assert _is_public_publication_request(self._request("GET", path))

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("POST", "/v1/artifacts/fig/v/1/publish"),
            ("DELETE", "/v1/publications/sometoken"),
            ("GET", "/v1/publications"),
            ("GET", "/v1/artifacts"),
            ("GET", "/v1/artifacts/fig/v/1/data"),
            ("POST", "/p/sometoken"),
        ],
    )
    def test_everything_else_still_needs_auth(self, method, path):
        from strata.server import _is_public_publication_request

        assert not _is_public_publication_request(self._request(method, path))


class TestPublicationPage:
    """What the rendered page does and does not say."""

    @staticmethod
    def _render(store, *, source: str, title: str | None = None, revoke: bool = False):
        from strata.api.publication_page import render_publication
        from strata.notebook.artifact_integration import NotebookArtifactManager
        from strata.services.artifact import ArtifactService

        manager = NotebookArtifactManager("paper", artifact_dir=store.artifact_dir)
        upstream = manager.store_cell_output(
            cell_id="c1",
            variable_name="rows",
            blob_data=b"[1, 2, 3]",
            content_type="json/object",
            provenance_hash="a" * 64,
            input_versions={},
            source=source,
        )
        ref = f"{upstream.id}@v={upstream.version}"
        figure = manager.store_cell_output(
            cell_id="c2",
            variable_name="__display__0",
            blob_data=b"PNG",
            content_type="image/png",
            provenance_hash="b" * 64,
            input_versions={f"strata://artifact/{ref}": ref},
            source="plt.plot(rows)",
        )
        publication = manager.artifact_store.publish_artifact(
            figure.id, figure.version, title=title
        )
        if revoke:
            manager.artifact_store.revoke_publication(publication.token)
            publication = manager.artifact_store.get_publication(publication.token)

        lineage = ArtifactService().build_lineage(
            manager.artifact_store,
            artifact=figure,
            artifact_id=figure.id,
            version=figure.version,
            tenant_filter=None,
            max_depth=25,
        )
        return render_publication(
            publication=publication,
            artifact=figure,
            lineage=lineage,
            content_type="image/png",
            inline_png=None,
        )

    def test_cell_source_is_escaped(self, store):
        """Cell source is written by whoever used the notebook, and this page is
        served unauthenticated to anyone holding the link."""
        html = self._render(store, source="rows = []  # <script>alert(1)</script>")

        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_the_title_is_escaped(self, store):
        html = self._render(store, source="rows = []", title="<img src=x onerror=1>")

        assert "<img src=x" not in html
        assert "&lt;img src=x" in html

    def test_it_shows_the_chain_not_just_the_artifact(self, store):
        """The upstream step's code is the point — a plot with no visible
        ancestry answers nothing a referee asked."""
        html = self._render(store, source="rows = [1, 2, 3]")

        assert "plt.plot(rows)" in html, "the artifact's own source"
        assert "rows = [1, 2, 3]" in html, "the upstream step's source"

    def test_it_never_claims_the_result_was_verified_or_reproduced(self, store):
        """A green check reads as 'someone reproduced this' to a referee.

        Reproduction needs a re-run, and randomness, thread counts, float
        accumulation order and unavailable input data each break it. A badge
        that can be wrong is worse than no badge, so the page states what it
        checked in words instead.
        """
        html = self._render(store, source="rows = []")

        assert "verified" not in html.lower()
        assert "does <em>not</em> claim the result was reproduced" in html

    def test_a_withdrawn_publication_says_so_rather_than_404ing(self, store):
        """A reader chasing a footnote deserves 'withdrawn', not what reads as
        a typo — and the page says the link was never repointed."""
        html = self._render(store, source="rows = []", revoke=True)

        assert "withdrawn" in html.lower()
        assert "never repointed" in html
        assert "plt.plot(rows)" not in html, "a withdrawn artifact shows no content"


@pytest.fixture
def published_server(tmp_path):
    """A running server with one published artifact.

    Yields ``(base_url, token, payload)``. Uses a real server rather than
    ``TestClient`` because the middleware exemption is half of what is under
    test here, and it only runs on a real request path.
    """
    import httpx

    from tests.conftest import run_server_with_context

    artifact_dir = tmp_path / "artifacts"
    with run_server_with_context(tmp_path / "cache", artifact_dir, "personal") as ctx:
        payload = b"\x89PNG\r\n\x1a\n published bytes"
        store = ArtifactStore(artifact_dir)
        version = _ready_artifact(store, "fig", payload)

        response = httpx.post(
            f"{ctx.base_url}/v1/artifacts/fig/v/{version}/publish",
            json={"title": "Figure 3"},
            timeout=10,
        )
        response.raise_for_status()
        yield ctx.base_url, response.json()["token"], payload


class TestPublicRoutesEndToEnd:
    def test_the_page_and_bytes_are_readable_without_credentials(self, published_server):
        import httpx

        base_url, token, payload = published_server

        page = httpx.get(f"{base_url}/p/{token}", timeout=10)
        assert page.status_code == 200
        assert "Figure 3" in page.text

        data = httpx.get(f"{base_url}/p/{token}/data", timeout=10)
        assert data.status_code == 200
        assert data.content == payload

    def test_verify_compares_against_the_recorded_digest(self, published_server):
        import httpx

        base_url, token, payload = published_server

        verified = httpx.get(f"{base_url}/p/{token}/verify", timeout=10).json()

        assert verified["matches"] is True
        assert verified["actual_sha256"] == hashlib.sha256(payload).hexdigest()

    def test_verify_reports_a_mismatch_when_the_bytes_change(self, published_server, tmp_path):
        """The failure this check exists to catch.

        A digest routine that returns "matches" unconditionally passes the test
        above; only tampering with the blob distinguishes it.
        """
        import httpx

        base_url, token, _ = published_server

        store = ArtifactStore(tmp_path / "artifacts")
        with store.open_blob_writer("fig", 1) as writer:
            writer.write(b"tampered")

        verified = httpx.get(f"{base_url}/p/{token}/verify", timeout=10).json()

        assert verified["matches"] is False
        assert verified["actual_sha256"] != verified["recorded_sha256"]

    def test_revoking_stops_the_bytes_but_keeps_the_explanation(self, published_server):
        import httpx

        base_url, token, _ = published_server

        assert httpx.delete(f"{base_url}/v1/publications/{token}", timeout=10).status_code == 200

        assert httpx.get(f"{base_url}/p/{token}/data", timeout=10).status_code == 410

        page = httpx.get(f"{base_url}/p/{token}", timeout=10)
        assert page.status_code == 200
        assert "withdrawn" in page.text.lower()

    def test_an_unknown_token_is_not_found(self, published_server):
        import httpx

        base_url, _, _ = published_server

        assert httpx.get(f"{base_url}/p/nosuchtoken", timeout=10).status_code == 404
