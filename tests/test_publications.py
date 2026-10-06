"""Publications: opt-in public read grants and the page they resolve to.

These are the only routes that serve artifact contents to an unauthenticated caller. Most tests
guard the edges: auth exemptions, revoked tokens, and user text reaching the page unescaped.
"""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from strata.artifact_store import ArtifactStore, ArtifactVersion


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

        ``provenance_hash`` covers inputs and transform, not content.
        """
        payload = b"figure-bytes"
        version = _ready_artifact(store, "fig", payload)

        publication = store.publish_artifact("fig", version)

        assert publication.content_sha256 == hashlib.sha256(payload).hexdigest()

    def test_publishing_twice_returns_the_same_token(self, store):
        """Two live URLs for one artifact would make revocation a lie."""
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

    This predicate is the whole boundary: if it widens, routes that mint, withdraw or enumerate
    grants become public.
    """

    @staticmethod
    def _request(method: str, path: str, root_path: str = ""):
        from starlette.requests import Request

        scope = {"type": "http", "method": method, "path": root_path + path, "headers": []}
        return Request({**scope, "root_path": root_path, "query_string": b""})

    @pytest.mark.parametrize(
        "path",
        [
            "/p/sometoken",
            "/p/sometoken/data",
            "/p/sometoken/verify",
            "/v1/publications/sometoken",
            "/p/sometoken/embed",
            "/oembed",
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

    @pytest.mark.parametrize("path", ["/p/sometoken", "/oembed", "/v1/publications/sometoken"])
    def test_public_reads_are_exempt_under_a_base_path(self, path):
        """Behind a proxy at a non-root path the request path carries the base."""
        from strata.server import _is_public_publication_request

        assert _is_public_publication_request(self._request("GET", path, "/o/acme/lab"))
        assert not _is_public_publication_request(
            self._request("GET", "/v1/publications", "/o/acme/lab")
        )


class TestPublicationPage:
    """What the rendered page does and does not say."""

    @staticmethod
    def _render(
        store,
        *,
        source: str,
        title: str | None = None,
        revoke: bool = False,
        external_ids: list[dict[str, str]] | None = None,
        upstream_params: dict[str, str] | None = None,
        publish_upstream: bool = False,
    ):
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
            extra_params=upstream_params,
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
        if publish_upstream:
            figure = upstream
        publication = manager.artifact_store.publish_artifact(
            figure.id, figure.version, title=title
        )
        if external_ids is not None:
            manager.artifact_store.update_publication_credits(
                publication.token, external_ids=external_ids
            )
            publication = manager.artifact_store.get_publication(publication.token)
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
            image_src=None,
        )

    def test_a_snapshot_step_says_until_when_its_state_can_be_queried(self, store):
        """A ``# @cache snapshot`` SQL cell upstream records its moment and horizon."""
        html = self._render(
            store,
            source="SELECT * FROM orders",
            upstream_params={
                "sql_snapshot_at": "2026-09-15T10:05:00+00:00",
                "sql_snapshot_valid_until": "2026-09-16T10:05:00+00:00",
            },
        )

        assert "Warehouse state as of</td><td>2026-09-15 10:05 UTC" in html
        assert "Queryable until</td><td>2026-09-16 10:05 UTC" in html

    def test_a_published_snapshot_cell_says_it_of_itself(self, store):
        html = self._render(
            store,
            source="SELECT * FROM orders",
            upstream_params={"sql_snapshot_at": "2026-09-15T10:05:00+00:00"},
            publish_upstream=True,
        )

        this_artifact = html.split("<h2>This artifact</h2>")[1].split("<h2>")[0]
        assert "Warehouse state as of</td><td>2026-09-15 10:05 UTC" in this_artifact
        assert "Queryable until</td><td><span class='note'>not recorded" in this_artifact

    def test_a_step_without_a_snapshot_says_nothing_about_one(self, store):
        assert "Queryable until" not in self._render(store, source="rows = []")

    def test_cell_source_is_escaped(self, store):
        """Cell source is user-written and the page is served unauthenticated."""
        html = self._render(store, source="rows = []  # <script>alert(1)</script>")

        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html

    def test_an_identifier_that_is_not_a_url_is_printed_not_linked(self, store):
        """``url`` takes anything and escaping leaves the scheme alone, on this server's origin."""
        html = self._render(
            store,
            source="rows = []",
            external_ids=[{"scheme": "url", "value": "javascript:alert(1)"}],
        )

        assert "href='javascript:" not in html
        assert 'href="javascript:' not in html
        assert "javascript:alert(1)" in html, "the identifier is still shown"

    def test_a_real_url_is_still_a_link(self, store):
        html = self._render(
            store,
            source="rows = []",
            external_ids=[{"scheme": "url", "value": "https://example.org/paper"}],
        )

        assert "href='https://example.org/paper'" in html

    def test_a_doi_is_still_resolved_through_doi_org(self, store):
        html = self._render(
            store,
            source="rows = []",
            external_ids=[{"scheme": "doi", "value": "10.1000/xyz"}],
        )

        assert "href='https://doi.org/10.1000/xyz'" in html

    def test_the_title_is_escaped(self, store):
        html = self._render(store, source="rows = []", title="<img src=x onerror=1>")

        assert "<img src=x" not in html
        assert "&lt;img src=x" in html

    def test_it_shows_the_chain_not_just_the_artifact(self, store):
        """The upstream step's code is the point; a plot with no ancestry answers nothing."""
        html = self._render(store, source="rows = [1, 2, 3]")

        assert "plt.plot(rows)" in html, "the artifact's own source"
        assert "rows = [1, 2, 3]" in html, "the upstream step's source"

    def test_sibling_variables_from_one_cell_show_their_code_once(self, store):
        """A cell defining several consumed variables contributes one ancestor each.

        Printing each would repeat the cell's code and read as a rendering fault.
        """
        from strata.api.publication_page import render_publication
        from strata.notebook.artifact_integration import NotebookArtifactManager
        from strata.services.artifact import ArtifactService

        manager = NotebookArtifactManager("paper", artifact_dir=store.artifact_dir)
        shared_source = "dose = load()\nresponse = measure(dose)"
        refs = {}
        for index, var in enumerate(("dose", "response")):
            produced = manager.store_cell_output(
                cell_id="load",
                variable_name=var,
                blob_data=b"[]",
                content_type="json/object",
                provenance_hash=str(index) * 64,
                input_versions={},
                source=shared_source,
            )
            ref = f"{produced.id}@v={produced.version}"
            refs[f"strata://artifact/{ref}"] = ref

        figure = manager.store_cell_output(
            cell_id="figure",
            variable_name="__display__0",
            blob_data=b"PNG",
            content_type="image/png",
            provenance_hash="b" * 64,
            input_versions=refs,
            source="plt.plot(dose, response)",
        )
        publication = manager.artifact_store.publish_artifact(figure.id, figure.version)
        lineage = ArtifactService().build_lineage(
            manager.artifact_store,
            artifact=figure,
            artifact_id=figure.id,
            version=figure.version,
            tenant_filter=None,
            max_depth=10,
        )

        html = render_publication(
            publication=publication,
            artifact=figure,
            lineage=lineage,
            content_type="image/png",
            image_src=None,
        )

        assert html.count("response = measure(dose)") == 1, "the shared source was repeated"
        assert "Produced by the same cell as" in html

    def test_it_never_claims_the_result_was_verified_or_reproduced(self, store):
        """A green check reads as "someone reproduced this" to a referee.

        Reproduction needs a re-run and many things break it, so the page states in words what it
        checked.
        """
        html = self._render(store, source="rows = []")

        assert "verified" not in html.lower()
        assert "does <em>not</em> claim the result was reproduced" in html

    def test_it_distinguishes_who_published_from_who_computed(self, store):
        """The byline names the publisher; the table row names who produced the bytes.

        Labelling the second "Author" read as a contradiction ("Published by X" over "Author: not
        recorded").
        """
        html = self._render(store, source="rows = []")

        assert "Computed by" in html
        assert ">Author<" not in html

    def test_a_withdrawn_publication_says_so_rather_than_404ing(self, store):
        """A reader chasing a footnote gets "withdrawn", and that the link was never repointed."""
        html = self._render(store, source="rows = []", revoke=True)

        assert "withdrawn" in html.lower()
        assert "never repointed" in html
        assert "plt.plot(rows)" not in html, "a withdrawn artifact shows no content"


@pytest.fixture
def published_server(tmp_path):
    """A running server with one published artifact; yields ``(base_url, token, payload)``.

    A real server, because the middleware exemption under test only runs on a real request path.
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
        """Only tampering with the blob catches a check that always says "matches"."""
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

        # The JSON record says the same as the page, and no more: no chain, no source.
        record = httpx.get(f"{base_url}/v1/publications/{token}", timeout=10)
        assert record.status_code == 200
        body = record.json()
        assert set(body) == {"publication"}
        assert body["publication"]["token"] == token
        assert body["publication"]["title"] == "Figure 3"
        assert body["publication"]["revoked_at"] is not None

    def test_an_unknown_token_is_not_found(self, published_server):
        import httpx

        base_url, _, _ = published_server

        assert httpx.get(f"{base_url}/p/nosuchtoken", timeout=10).status_code == 404


def _csp_directives(response) -> list[str]:
    return [d.strip() for d in response.headers["content-security-policy"].split(";")]


class TestTheBytesAreServedAsData:
    """A writer declares ``content_type`` freely; the public bytes must never run as a page on
    this server's origin.
    """

    PAYLOAD = b"<script>alert(document.domain)</script>"

    @staticmethod
    def _client(monkeypatch, tmp_path):
        from fastapi.testclient import TestClient

        import strata.server as server_module
        from strata.artifact_store import get_artifact_store, reset_artifact_store
        from strata.config import StrataConfig
        from strata.server import ServerState, app

        artifact_dir = tmp_path / "artifacts"
        reset_artifact_store()
        monkeypatch.setattr(
            server_module,
            "_state",
            ServerState(StrataConfig(artifact_dir=artifact_dir, cache_dir=tmp_path / "cache")),
        )
        return TestClient(app), get_artifact_store(artifact_dir)

    def _published_data(self, monkeypatch, tmp_path, content_type):
        from strata.artifact_store import TransformSpec

        client, store = self._client(monkeypatch, tmp_path)
        version = store.create_artifact(
            "fig",
            hashlib.sha256(self.PAYLOAD).hexdigest(),
            transform_spec=TransformSpec(
                executor="local@v1", params={"content_type": content_type}, inputs=[]
            ),
        )
        with store.open_blob_writer("fig", version) as writer:
            writer.write(self.PAYLOAD)
        store.finalize_artifact(
            "fig", version, schema_json="", row_count=0, byte_size=len(self.PAYLOAD)
        )
        token = store.publish_artifact("fig", version).token
        return client.get(f"/p/{token}/data")

    @pytest.mark.parametrize("content_type", ["text/html", "image/svg+xml", "", "pickle/object"])
    def test_a_type_that_could_render_or_is_unknown_is_downloaded(
        self, monkeypatch, tmp_path, content_type
    ):
        response = self._published_data(monkeypatch, tmp_path, content_type)

        assert response.status_code == 200
        assert response.content == self.PAYLOAD
        assert response.headers["content-type"] == "application/octet-stream"
        assert response.headers["content-disposition"] == "attachment"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "sandbox" in _csp_directives(response)
        # The route's own sandbox policy keeps the server's framing rule too.
        assert "frame-ancestors 'self'" in _csp_directives(response)

    @pytest.mark.parametrize(
        ("content_type", "served_as"),
        [
            ("image/png", "image/png"),
            ("arrow/ipc", "application/vnd.apache.arrow.stream"),
            ("json/object", "application/json"),
            ("text/markdown", "text/markdown; charset=utf-8"),
        ],
    )
    def test_types_strata_produces_are_served_inline(
        self, monkeypatch, tmp_path, content_type, served_as
    ):
        response = self._published_data(monkeypatch, tmp_path, content_type)

        assert response.status_code == 200
        assert response.headers["content-type"] == served_as
        assert "content-disposition" not in response.headers
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "sandbox" in _csp_directives(response)

    def test_the_authenticated_bytes_carry_the_same_guards(self, monkeypatch, tmp_path):
        self._published_data(monkeypatch, tmp_path, "text/html")
        client, _ = self._client(monkeypatch, tmp_path)

        response = client.get("/v1/artifacts/fig/v/1/data")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/vnd.apache.arrow.stream"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "sandbox" in _csp_directives(response)


class TestPublishDestination:
    """Where a grant is written, and whether the caller is told.

    ``--artifact-dir`` says where to read, not where the publication lands; an unnamed destination
    is how a test run once wrote into a developer's ``~/.strata/artifacts``.
    """

    @staticmethod
    def _publish(tmp_path, capsys, **overrides):
        import argparse

        from strata.artifact_cli import cmd_publish
        from strata.notebook.artifact_integration import NotebookArtifactManager

        manager = NotebookArtifactManager("nb", artifact_dir=tmp_path / "source")
        figure = manager.store_cell_output(
            cell_id="c1",
            variable_name="__display__0",
            blob_data=b"PNG",
            content_type="image/png",
            provenance_hash="a" * 64,
            input_versions={},
            source="plt.plot()",
        )
        args = {
            "ref": figure.id,
            "artifact_dir": str(tmp_path / "source"),
            "format": "human",
            "title": None,
            "author": None,
            "here": False,
            "into": None,
            "max_depth": 10,
        }
        args.update(overrides)
        assert cmd_publish(argparse.Namespace(**args)) == 0
        return capsys.readouterr().out, figure

    def test_the_destination_is_always_reported(self, tmp_path, capsys):
        """Even when nothing is copied.

        Without it a caller cannot tell a working link from one their own server will never resolve.
        """
        out, _ = self._publish(tmp_path, capsys, here=True)

        assert "Published into" in out
        assert str(tmp_path / "source") in out

    def test_into_sends_it_somewhere_named(self, tmp_path, capsys):
        from strata.artifact_store import ArtifactStore

        elsewhere = tmp_path / "elsewhere"
        out, figure = self._publish(tmp_path, capsys, into=str(elsewhere))

        assert str(elsewhere) in out
        assert len(ArtifactStore(elsewhere).list_publications()) == 1
        # …and not in the store it was read from.
        assert ArtifactStore(tmp_path / "source").list_publications() == []

    def test_here_keeps_it_in_the_source_store(self, tmp_path, capsys):
        from strata.artifact_store import ArtifactStore

        self._publish(tmp_path, capsys, here=True)

        assert len(ArtifactStore(tmp_path / "source").list_publications()) == 1

    def test_into_and_here_are_mutually_exclusive(self, tmp_path):
        """Naming both a directory and "the source" is a contradiction, so it is refused."""
        from strata.cli import main

        with pytest.raises(SystemExit) as exit_info:
            main(
                [
                    "artifact",
                    "publish",
                    "x",
                    "--here",
                    "--into",
                    str(tmp_path / "somewhere"),
                ]
            )

        assert exit_info.value.code != 0


class TestImportAcrossStores:
    """Copying an artifact into the store that will serve it.

    Cells write to the notebook's ``.strata/artifacts`` while the server serves its
    ``artifact_dir``, so a token minted in the first store 404s unless the chain is copied.
    """

    def test_import_preserves_the_version(self, store, tmp_path):
        """Lineage edges are ``id@v=N`` strings, so a fresh version would orphan them."""
        other = ArtifactStore(tmp_path / "other")
        _ready_artifact(other, "pad", b"a")
        _ready_artifact(other, "pad", b"b")  # so the next id would not be v=1

        version = _ready_artifact(store, "fig", b"x")
        record = store.get_artifact("fig", version)

        assert other.import_artifact(record, b"x").written is True

        imported = other.get_artifact("fig", version)
        assert imported is not None
        assert imported.version == record.version
        assert imported.provenance_hash == record.provenance_hash

    def test_import_is_idempotent(self, store, tmp_path):
        other = ArtifactStore(tmp_path / "other")
        version = _ready_artifact(store, "fig", b"x")
        record = store.get_artifact("fig", version)

        assert other.import_artifact(record, b"x").written is True
        assert other.import_artifact(record, b"x").written is False

    def test_publishing_copies_the_chain_into_the_served_store(self, store, tmp_path, monkeypatch):
        """A token minted here resolves over there, with its whole chain.

        The page shows every upstream step, so copying the artifact alone is not enough.
        """
        from strata.artifact_cli import cmd_publish
        from strata.notebook.artifact_integration import NotebookArtifactManager
        from strata.services.artifact import ArtifactService

        notebook_store = NotebookArtifactManager("nb", artifact_dir=tmp_path / "notebook")
        upstream = notebook_store.store_cell_output(
            cell_id="c1",
            variable_name="rows",
            blob_data=b"[1]",
            content_type="json/object",
            provenance_hash="a" * 64,
            input_versions={},
            source="rows = [1]",
        )
        ref = f"{upstream.id}@v={upstream.version}"
        figure = notebook_store.store_cell_output(
            cell_id="c2",
            variable_name="__display__0",
            blob_data=b"PNG",
            content_type="image/png",
            provenance_hash="b" * 64,
            input_versions={f"strata://artifact/{ref}": ref},
            source="plt.plot(rows)",
        )

        served = ArtifactStore(tmp_path / "served")
        monkeypatch.setattr("strata.artifact_cli._server_store", lambda: served)

        import argparse

        rc = cmd_publish(
            argparse.Namespace(
                ref=figure.id,
                artifact_dir=str(tmp_path / "notebook"),
                format="human",
                title=None,
                author=None,
                here=False,
                max_depth=10,
            )
        )

        assert rc == 0
        published = served.list_publications()
        assert len(published) == 1, "the token must be minted in the store that serves it"

        # The chain came with it, resolvable from the served store alone.
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

    def test_import_deduplicates_against_the_same_computation_under_another_id(
        self, store, tmp_path
    ):
        """A provenance hash does not carry the cell id.

        Identical cells in two notebooks share a hash under different ids, and the store allows one
        ready row per ``(tenant, provenance_hash)``; this must not raise ``IntegrityError``.
        """
        alice = ArtifactStore(tmp_path / "alice")
        bob = ArtifactStore(tmp_path / "bob")
        target = ArtifactStore(tmp_path / "central")

        payload = b"identical"
        a_version = _ready_artifact(alice, "nb_alice_cell_c1_var_rows", payload)
        b_version = _ready_artifact(bob, "nb_bob_cell_c1_var_rows", payload)
        a_record = alice.get_artifact("nb_alice_cell_c1_var_rows", a_version)
        b_record = bob.get_artifact("nb_bob_cell_c1_var_rows", b_version)
        assert a_record.provenance_hash == b_record.provenance_hash

        first = target.import_artifact(a_record, payload)
        second = target.import_artifact(b_record, payload)

        assert first.written is True
        assert second.written is False
        assert second.ref == first.ref, "the second import must resolve to the row already here"

    def test_an_input_that_is_also_an_ancestor_is_copied_before_its_descendants(self, tmp_path):
        """``load -> features(load) -> train(load, features)``, with ``load`` already on the target.

        BFS reaches ``load`` at depth 1 and ``features`` too, so reversed BFS copied ``features``
        first, with an edge naming ``load``'s source id that never landed on the target.
        """
        import json

        from strata.artifact_store import TransformSpec
        from strata.artifact_transfer import copy_chain
        from strata.services.artifact import ArtifactService

        def ready(store, artifact_id, provenance, inputs=None):
            version = store.create_artifact(
                artifact_id,
                provenance,
                TransformSpec("notebook/cell@v1", {"source": artifact_id}, []),
                input_versions=inputs,
            )
            store.write_blob(artifact_id, version, b"x")
            store.finalize_artifact(artifact_id, version, "", 1, 1)
            return f"{artifact_id}@v={version}"

        def edge(ref):
            return {f"strata://artifact/{ref}": ref}

        source = ArtifactStore(tmp_path / "source")
        target = ArtifactStore(tmp_path / "target")
        load = ready(source, "nb_A_cell_load_var_df", "prov-load")
        features = ready(source, "nb_A_cell_feat_var_x", "prov-feat", edge(load))
        ready(source, "nb_A_cell_train_var_m", "prov-train", {**edge(load), **edge(features)})
        # A colleague's notebook already put the same load step on the target.
        theirs = ready(target, "nb_B_cell_load_var_df", "prov-load")

        train = source.get_artifact("nb_A_cell_train_var_m", 1)
        copy_chain(source, target, train, 10)

        landed = target.get_artifact("nb_A_cell_feat_var_x", 1)
        assert json.loads(landed.input_versions) == edge(theirs)
        chain = ArtifactService().build_lineage(
            target,
            artifact=target.get_artifact("nb_A_cell_train_var_m", 1),
            artifact_id="nb_A_cell_train_var_m",
            version=1,
            tenant_filter=None,
            max_depth=25,
        )
        assert "nb_A_cell_load_var_df" not in {n.artifact_id for n in chain.nodes}

    def test_import_does_not_deduplicate_across_tenants(self, tmp_path):
        """Dedup is per tenant: another tenant's row would hand out a ref the caller cannot read."""
        target = ArtifactStore(tmp_path / "central")
        record = ArtifactVersion(
            id="shared",
            version=1,
            state="ready",
            provenance_hash="d" * 64,
            created_at=1.0,
            tenant="acme",
        )
        other_tenant = replace(record, id="shared-other", tenant="globex")

        assert target.import_artifact(record, b"x").written is True
        assert target.import_artifact(other_tenant, b"x").written is True

    def test_an_import_that_dies_before_the_row_leaves_nothing_readable(self, store, tmp_path):
        """The row makes a version readable, so it is committed last.

        A row committed first looks like a finished import, so no retry could repair the missing
        bytes.
        """
        target = ArtifactStore(tmp_path / "central")
        version = _ready_artifact(store, "fig", b"x")
        record = store.get_artifact("fig", version)

        def die(*args, **kwargs):
            raise OSError("disk went away mid-upload")

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(target, "open_blob_writer", die)
            with pytest.raises(OSError):
                target.import_artifact(record, b"x")

        assert target.get_artifact("fig", version) is None, (
            "a failed import must not leave a row claiming to be ready"
        )

        # The retry a live store would make now succeeds.
        assert target.import_artifact(record, b"x").written is True
        reader = target.open_blob_reader("fig", version)
        assert reader is not None
        with reader as handle:
            assert handle.read() == b"x"

    def test_a_shared_upstream_still_resolves_for_the_second_publisher(self, tmp_path, monkeypatch):
        """Two chains over one computation, published into one served store.

        The second publisher's upstream dedups onto the first's row, so the second figure's edge
        must be rewritten to name it.
        """
        from strata.artifact_cli import cmd_publish
        from strata.notebook.artifact_integration import NotebookArtifactManager
        from strata.services.artifact import ArtifactService

        served = ArtifactStore(tmp_path / "served")
        monkeypatch.setattr("strata.artifact_cli._server_store", lambda: served)

        def publish_a_figure(notebook: str, figure_provenance: str):
            manager = NotebookArtifactManager(notebook, artifact_dir=tmp_path / notebook)
            upstream = manager.store_cell_output(
                cell_id="c1",
                variable_name="rows",
                blob_data=b"[1]",
                content_type="json/object",
                # The shared cell: identical source, identical hash, and an id
                # that differs only because the notebook does.
                provenance_hash="a" * 64,
                input_versions={},
                source="rows = [1]",
            )
            ref = f"{upstream.id}@v={upstream.version}"
            figure = manager.store_cell_output(
                cell_id="c2",
                variable_name="__display__0",
                blob_data=b"PNG",
                content_type="image/png",
                provenance_hash=figure_provenance,
                input_versions={f"strata://artifact/{ref}": ref},
                source="plt.plot(rows)",
            )
            rc = cmd_publish(
                argparse.Namespace(
                    ref=figure.id,
                    artifact_dir=str(tmp_path / notebook),
                    format="human",
                    title=None,
                    author=None,
                    here=False,
                    max_depth=10,
                )
            )
            return rc, upstream, figure

        first_rc, alice_upstream, _ = publish_a_figure("alice", "b" * 64)
        second_rc, bob_upstream, bob_figure = publish_a_figure("bob", "c" * 64)

        assert first_rc == 0
        assert second_rc == 0, "the second publisher must not crash on the shared upstream"
        assert len(served.list_publications()) == 2

        assert served.get_artifact(bob_upstream.id, bob_upstream.version) is None, (
            "the shared computation is stored once, under whoever published it first"
        )

        copied = served.get_artifact(bob_figure.id, bob_figure.version)
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
            bob_figure.id,
            alice_upstream.id,
        ], "the second figure's edge must name the row its upstream landed on"


class TestPublicationRetention:
    """A sweep must never collect anything a live token depends on."""

    def _chain(self, store):
        """An upstream, a figure that names it, and a newer upstream version.

        The first upstream version is neither named nor the latest of its id: exactly what the sweep
        collects.
        """
        upstream_v1 = _ready_artifact(store, "rows", b"[1]")
        ref = f"rows@v={upstream_v1}"
        figure_version = store.create_artifact(
            "figure",
            "f" * 64,
            input_versions={f"strata://artifact/{ref}": ref},
        )
        with store.open_blob_writer("figure", figure_version) as writer:
            writer.write(b"PNG")
        store.finalize_artifact("figure", figure_version, schema_json="", row_count=0, byte_size=3)

        publication = store.publish_artifact("figure", figure_version)
        _ready_artifact(store, "rows", b"[1, 2]")  # the re-run: v1 is no longer latest
        return upstream_v1, figure_version, publication

    def test_a_sweep_keeps_the_chain_behind_a_publication(self, store):
        upstream_v1, figure_version, _ = self._chain(store)

        store.garbage_collect(max_idle_days=0)

        assert store.get_artifact("figure", figure_version) is not None
        assert store.get_artifact("rows", upstream_v1) is not None, (
            "collecting an ancestor leaves a live token whose lineage resolves to nothing"
        )

    def test_a_withdrawn_publication_still_protects_its_chain(self, store):
        """A withdrawal keeps its chain readable for audit."""
        upstream_v1, figure_version, publication = self._chain(store)
        store.revoke_publication(publication.token)

        store.garbage_collect(max_idle_days=0)

        assert store.get_artifact("rows", upstream_v1) is not None

    def test_a_sweep_still_collects_an_unpublished_superseded_version(self, store):
        """The protection is publications, not a blanket amnesty."""
        orphan = _ready_artifact(store, "scratch", b"a")
        _ready_artifact(store, "scratch", b"b")  # supersedes it

        store.garbage_collect(max_idle_days=0)

        assert store.get_artifact("scratch", orphan) is None


class TestEmbedding:
    """The card, and the oEmbed endpoint that unfurls a pasted link."""

    def test_the_card_carries_the_link_to_the_provenance(self, published_server):
        """An embed that is only an image defeats its purpose: it must link to the chain."""
        import httpx

        base_url, token, _ = published_server

        card = httpx.get(f"{base_url}/p/{token}/embed", timeout=10)

        assert card.status_code == 200
        assert "See what produced it" in card.text
        assert f"{base_url}/p/{token}" in card.text

    def test_the_card_may_be_framed_anywhere(self, published_server):
        """The default `frame-ancestors 'self'` protects the app view, not the embed card.

        The full page keeps the restrictive default.
        """
        import httpx

        base_url, token, _ = published_server

        card = httpx.get(f"{base_url}/p/{token}/embed", timeout=10)
        page = httpx.get(f"{base_url}/p/{token}", timeout=10)

        assert card.headers["content-security-policy"] == "frame-ancestors *"
        assert page.headers["content-security-policy"] == "frame-ancestors 'self'"

    @pytest.mark.parametrize(
        "path",
        ["/anything/embed", "/notebook/embed", "/a/b/c/embed", "/embed"],
    )
    def test_only_the_real_embed_route_may_be_framed(self, published_server, path):
        """The SPA catch-all serves index.html for any unmatched path.

        A suffix match on "/embed" would let any origin frame `/x/embed#/notebook/<session>`, the
        live notebook app.
        """
        import httpx

        base_url, _, _ = published_server

        response = httpx.get(f"{base_url}{path}", timeout=10)

        assert response.headers["content-security-policy"] == "frame-ancestors 'self'"

    def test_oembed_matches_a_host_written_differently(self, published_server):
        """Case and an explicit default port name the same server."""
        import httpx

        base_url, token, _ = published_server
        loud = base_url.replace("127.0.0.1", "127.0.0.1").upper().replace("HTTP", "http")

        response = httpx.get(f"{base_url}/oembed", params={"url": f"{loud}/p/{token}"}, timeout=10)

        assert response.status_code == 200

    def test_oembed_describes_the_card(self, published_server):
        import httpx

        base_url, token, _ = published_server

        payload = httpx.get(
            f"{base_url}/oembed", params={"url": f"{base_url}/p/{token}"}, timeout=10
        ).json()

        assert payload["version"] == "1.0"
        assert payload["type"] == "rich"
        assert f"/p/{token}/embed" in payload["html"]
        assert payload["width"] > 0 and payload["height"] > 0

    def test_oembed_refuses_a_url_on_another_host(self, published_server):
        """A provider must not describe pages it has never seen."""
        import httpx

        base_url, token, _ = published_server

        response = httpx.get(
            f"{base_url}/oembed",
            params={"url": f"https://example.invalid/p/{token}"},
            timeout=10,
        )

        assert response.status_code == 404

    def test_oembed_says_so_rather_than_serving_empty_xml(self, published_server):
        import httpx

        base_url, token, _ = published_server

        response = httpx.get(
            f"{base_url}/oembed",
            params={"url": f"{base_url}/p/{token}", "format": "xml"},
            timeout=10,
        )

        assert response.status_code == 501

    def test_the_page_advertises_the_oembed_endpoint(self, published_server):
        """Discovery lets a wiki unfurl a pasted link without being told the endpoint."""
        import httpx

        base_url, token, _ = published_server

        page = httpx.get(f"{base_url}/p/{token}", timeout=10).text

        assert "application/json+oembed" in page

    def test_a_withdrawn_publication_has_no_card(self, published_server):
        import httpx

        base_url, token, _ = published_server
        httpx.delete(f"{base_url}/v1/publications/{token}", timeout=10)

        assert httpx.get(f"{base_url}/p/{token}/embed", timeout=10).status_code == 410


class TestRoCrate:
    """The chain as RO-Crate JSON-LD, the form software reads."""

    @staticmethod
    def _crate(store, tmp_path, *, source="rows = [1]"):
        from strata.api.provenance_ld import build_crate
        from strata.notebook.artifact_integration import NotebookArtifactManager
        from strata.services.artifact import ArtifactService

        manager = NotebookArtifactManager("nb", artifact_dir=tmp_path / "nb")
        upstream = manager.store_cell_output(
            cell_id="c1",
            variable_name="rows",
            blob_data=b"[1]",
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
            figure.id, figure.version, title="Figure 3", published_by="F. Li"
        )
        lineage = ArtifactService().build_lineage(
            manager.artifact_store,
            artifact=figure,
            artifact_id=figure.id,
            version=figure.version,
            tenant_filter=None,
            max_depth=10,
        )
        return (
            build_crate(
                publication=publication,
                artifact=figure,
                lineage=lineage,
                content_type="image/png",
                payload_id="artifact.png",
                include_descriptor=True,
            ),
            upstream,
            figure,
        )

    @staticmethod
    def _refs(value):
        if isinstance(value, dict):
            if set(value) == {"@id"}:
                yield value["@id"]
            else:
                for inner in value.values():
                    yield from TestRoCrate._refs(inner)
        elif isinstance(value, list):
            for inner in value:
                yield from TestRoCrate._refs(inner)

    def test_every_reference_resolves_inside_the_crate(self, store, tmp_path):
        """A dangling @id makes the graph useless, and nothing looks wrong until it is walked."""
        crate, _, _ = self._crate(store, tmp_path)
        ids = {entity["@id"] for entity in crate["@graph"]}

        dangling = {
            ref
            for entity in crate["@graph"]
            for ref in self._refs(entity)
            if ref not in ids and not ref.startswith("http")
        }

        assert not dangling, f"references nothing declares: {sorted(dangling)}"

    def test_a_table_input_is_declared_not_just_referenced(self, store, tmp_path):
        """Non-artifact inputs are named by ``object`` and need an entity.

        A figure read from an Iceberg table would otherwise reference an undeclared `@id`; the
        dangling-reference test only has artifact inputs.
        """
        from strata.api.provenance_ld import build_crate
        from strata.notebook.artifact_integration import NotebookArtifactManager
        from strata.services.artifact import ArtifactService

        table_uri = "iceberg://warehouse/test_db.events"
        manager = NotebookArtifactManager("nb", artifact_dir=tmp_path / "tbl")
        figure = manager.store_cell_output(
            cell_id="c1",
            variable_name="__display__0",
            blob_data=b"PNG",
            content_type="image/png",
            provenance_hash="a" * 64,
            input_versions={table_uri: "12345"},
            source="plt.plot(events)",
        )
        publication = manager.artifact_store.publish_artifact(figure.id, figure.version)
        lineage = ArtifactService().build_lineage(
            manager.artifact_store,
            artifact=figure,
            artifact_id=figure.id,
            version=figure.version,
            tenant_filter=None,
            max_depth=10,
        )

        crate = build_crate(
            publication=publication,
            artifact=figure,
            lineage=lineage,
            content_type="image/png",
            payload_id="artifact.png",
            include_descriptor=True,
        )
        ids = {entity["@id"] for entity in crate["@graph"]}
        dangling = {
            ref
            for entity in crate["@graph"]
            for ref in self._refs(entity)
            if ref not in ids and not ref.startswith("http")
        }

        assert table_uri in ids, "the table input is referenced but never declared"
        assert not dangling, f"references nothing declares: {sorted(dangling)}"

    def test_two_versions_of_one_id_get_separate_entities(self, store, tmp_path):
        """JSON-LD flattening merges nodes that share an @id.

        Ids keyed on the artifact alone would collapse two versions into one entity.
        """
        from strata.api.provenance_ld import _action_id, _source_id

        class _Node:
            def __init__(self, version):
                self.artifact_id = "nb_x_cell_c_var_x"
                self.version = version

        assert _source_id(_Node(1)) != _source_id(_Node(2))
        assert _action_id(_Node(1)) != _action_id(_Node(2))

    def test_the_digest_survives_json_ld_expansion(self, store, tmp_path):
        """`sha256` is not an RO-Crate 1.1 term, and undefined terms vanish on expansion."""
        crate, _, _ = self._crate(store, tmp_path)

        context = crate["@context"]

        assert isinstance(context, list), "a bare context string defines no sha256"
        assert any(isinstance(part, dict) and "sha256" in part for part in context), (
            "sha256 has no definition, so a processor drops it"
        )

    def test_an_author_with_a_space_is_a_usable_identifier(self, store, tmp_path):
        """A space is illegal in an IRI, so strict processors would drop `#agent-F. Li`."""
        crate, _, _ = self._crate(store, tmp_path)

        agents = [e for e in crate["@graph"] if e.get("@type") == "Person"]

        assert agents, "the publisher should appear as an agent"
        assert all(" " not in agent["@id"] for agent in agents)

    def test_upstream_steps_are_described_but_not_claimed_as_files(self, store, tmp_path):
        """Upstream data is not in the crate, so listing it under hasPart would be false."""
        crate, upstream, _ = self._crate(store, tmp_path)
        root = next(e for e in crate["@graph"] if e["@id"] == "./")
        upstream_id = f"{upstream.id}@v={upstream.version}"

        described = next(e for e in crate["@graph"] if e["@id"] == upstream_id)

        assert described["@type"] == "CreativeWork"
        assert [part["@id"] for part in root["hasPart"]] == ["artifact.png"]

    def test_each_step_records_its_code_as_the_instrument(self, store, tmp_path):
        crate, _, figure = self._crate(store, tmp_path)

        from strata.api.provenance_ld import _action_id

        class _Ref:
            artifact_id = figure.id
            version = figure.version

        action = next(e for e in crate["@graph"] if e["@id"] == _action_id(_Ref))
        source = next(e for e in crate["@graph"] if e["@id"] == action["instrument"]["@id"])

        assert action["@type"] == "CreateAction"
        assert source["@type"] == "SoftwareSourceCode"
        assert source["text"] == "plt.plot(rows)"

    def test_the_deposited_crate_declares_what_it_conforms_to(self, store, tmp_path):
        """Without the descriptor a repository sees a folder of JSON, not a crate."""
        crate, _, _ = self._crate(store, tmp_path)

        descriptor = next(e for e in crate["@graph"] if e["@id"] == "ro-crate-metadata.json")

        assert descriptor["conformsTo"]["@id"] == "https://w3id.org/ro/crate/1.1"
        assert descriptor["about"]["@id"] == "./"

    def test_source_cannot_break_out_of_the_inline_script(self, store, tmp_path):
        """Cell source sits in a <script> block, where html.escape would not stop injection."""
        import json as jsonlib
        import re

        from strata.api.publication_page import render_publication
        from strata.services.artifact import ArtifactService

        evil = "x = 1  # </script><script>alert(1)</script>"
        crate, _, figure = self._crate(store, tmp_path, source=evil)
        manager_store = ArtifactStore(tmp_path / "nb")
        artifact = manager_store.get_latest_version(figure.id)
        lineage = ArtifactService().build_lineage(
            manager_store,
            artifact=artifact,
            artifact_id=artifact.id,
            version=artifact.version,
            tenant_filter=None,
            max_depth=10,
        )
        publication = manager_store.list_publications()[0]

        html = render_publication(
            publication=publication,
            artifact=artifact,
            lineage=lineage,
            content_type="image/png",
            image_src=None,
            json_ld=jsonlib.dumps(crate),
        )

        block = re.search(r"<script type='application/ld\+json'>(.*?)</script>", html, re.S)
        assert block is not None
        assert "</script>" not in block.group(1)
        # The escaping must leave valid JSON behind, not just safe text.
        assert isinstance(jsonlib.loads(block.group(1).replace("<\\/", "</")), dict)


class TestBadge:
    """The README pill, and the claim its shape invites.

    A shields-style `label | status` badge asserts; these pin the ways it could start saying a
    result was verified.
    """

    def test_it_reports_the_chain_size_and_claims_nothing(self, published_server):
        import httpx

        base_url, token, _ = published_server

        svg = httpx.get(f"{base_url}/p/{token}/badge.svg", timeout=10)

        assert svg.status_code == 200
        assert svg.headers["content-type"].startswith("image/svg+xml")
        assert "provenance" in svg.text
        for claim in ("verified", "valid", "reproduced", "passing", "trusted"):
            assert claim not in svg.text.lower(), f"the badge asserts {claim!r}"

    def test_it_is_not_green(self, published_server):
        """Green means "passing" by badge convention."""
        import httpx

        base_url, token, _ = published_server

        svg = httpx.get(f"{base_url}/p/{token}/badge.svg", timeout=10).text.lower()

        for green in ("#4c1", "brightgreen", "#2ea44f", "#3fb950", "green"):
            assert green not in svg

    def test_a_withdrawn_publication_still_renders_and_says_so(self, published_server):
        """A broken image says only that the server is unwell; the badge keeps rendering."""
        import httpx

        base_url, token, _ = published_server
        httpx.delete(f"{base_url}/v1/publications/{token}", timeout=10)

        svg = httpx.get(f"{base_url}/p/{token}/badge.svg", timeout=10)

        assert svg.status_code == 200
        assert "withdrawn" in svg.text

    def test_the_text_cannot_overflow_its_pill(self, published_server):
        """Widths are estimated from a font the server cannot know renders.

        Every run uses `textLength`, so a bad estimate looks loose or tight but never spills past
        the edge.
        """
        from strata.api.badge import render_badge

        svg = render_badge(label="provenance", value="1234 steps", title="t")

        assert svg.count('textLength="') == 4  # two runs, drawn twice each
        assert 'lengthAdjust="spacingAndGlyphs"' in svg

    def test_the_page_hands_over_a_ready_made_badge_snippet(self, published_server):
        """Nobody assembles a linked badge by hand from three route names."""
        import httpx

        base_url, token, _ = published_server

        page = httpx.get(f"{base_url}/p/{token}", timeout=10).text

        assert f"/p/{token}/badge.svg" in page
        assert "Putting it somewhere" in page


class TestExternalInputs:
    """Bytes a cell fetched from a URL, listed by URL, digest and time."""

    DIGEST = "d" * 64

    # 2026-01-02 03:04 UTC, well before the step that read the bytes.
    RETRIEVED = 1767323040.0

    def _published(self, tmp_path, url: str, *, fetched_at: float | None = None):
        import json

        from strata.notebook.artifact_integration import NotebookArtifactManager
        from strata.services.artifact import ArtifactService

        manager = NotebookArtifactManager("nb", artifact_dir=tmp_path / "fetched")
        figure = manager.store_cell_output(
            cell_id="c1",
            variable_name="__display__0",
            blob_data=b"PNG",
            content_type="image/png",
            provenance_hash="a" * 64,
            input_versions={url: f"sha256:{self.DIGEST}"},
            source="plt.plot(pd.read_csv(zones))",
            extra_params=(
                {"fetched_at": json.dumps({url: fetched_at})} if fetched_at is not None else None
            ),
        )
        publication = manager.artifact_store.publish_artifact(figure.id, figure.version)
        lineage = ArtifactService().build_lineage(
            manager.artifact_store,
            artifact=figure,
            artifact_id=figure.id,
            version=figure.version,
            tenant_filter=None,
            max_depth=10,
        )
        return publication, figure, lineage

    @staticmethod
    def _page(publication, figure, lineage) -> str:
        from strata.api.publication_page import render_publication

        return render_publication(
            publication=publication,
            artifact=figure,
            lineage=lineage,
            content_type="image/png",
            image_src=None,
        )

    def test_the_page_lists_the_url_the_digest_and_the_step_that_read_it(self, tmp_path):
        url = "https://example.org/taxi_zones.csv"
        publication, figure, lineage = self._published(tmp_path, url)

        html = self._page(publication, figure, lineage)
        external = html.split("<h2>External inputs</h2>", 1)[1].split("<h2>", 1)[0]

        assert url in external
        assert self.DIGEST in external
        assert f"{figure.id}@v={figure.version}" in external
        # Named once, as an external input, not again as an upstream step.
        assert html.count(url) == 1

    def test_the_page_says_when_the_bytes_were_retrieved(self, tmp_path):
        """The download can predate the run by months; the run's time would misdate the data."""
        url = "https://example.org/taxi_zones.csv"
        publication, figure, lineage = self._published(tmp_path, url, fetched_at=self.RETRIEVED)

        html = self._page(publication, figure, lineage)
        external = html.split("<h2>External inputs</h2>", 1)[1].split("<h2>", 1)[0]

        assert "Retrieved" in external
        assert "2026-01-02 03:04 UTC" in external

    def test_a_step_with_no_recorded_retrieval_claims_none(self, tmp_path):
        url = "https://example.org/taxi_zones.csv"
        html = self._page(*self._published(tmp_path, url))
        external = html.split("<h2>External inputs</h2>", 1)[1].split("<h2>", 1)[0]

        assert "Retrieved" not in external

    def test_a_hostile_url_is_printed_not_linked(self, tmp_path):
        """The record is not trusted to hold only https."""
        html = self._page(*self._published(tmp_path, "javascript:alert(1)//<script>x</script>"))

        assert "<script>x</script>" not in html
        assert "href='javascript:" not in html

    def test_the_crate_declares_the_url_as_a_file_with_its_digest(self, tmp_path):
        from strata.api.provenance_ld import build_crate

        url = "https://example.org/taxi_zones.csv"
        publication, figure, lineage = self._published(tmp_path, url)

        crate = build_crate(
            publication=publication,
            artifact=figure,
            lineage=lineage,
            content_type="image/png",
            payload_id="artifact.png",
            include_descriptor=True,
        )

        entity = next(e for e in crate["@graph"] if e["@id"] == url)
        assert entity["@type"] == "File"
        assert entity["sha256"] == self.DIGEST
        action = next(e for e in crate["@graph"] if e.get("@type") == "CreateAction")
        assert {"@id": url} in action["object"]

    def test_the_crate_dates_the_retrieval(self, tmp_path):
        from strata.api.provenance_ld import build_crate

        url = "https://example.org/taxi_zones.csv"
        publication, figure, lineage = self._published(tmp_path, url, fetched_at=self.RETRIEVED)

        crate = build_crate(
            publication=publication,
            artifact=figure,
            lineage=lineage,
            content_type="image/png",
            payload_id="artifact.png",
            include_descriptor=True,
        )

        entity = next(e for e in crate["@graph"] if e["@id"] == url)
        assert entity["sdDatePublished"].startswith("2026-01-02T03:04")


class TestWhatPublishingRequires:
    """A publication is the strongest read: anyone with the link gets the bytes."""

    @staticmethod
    def _service(monkeypatch, tmp_path, acl):
        from fastapi.testclient import TestClient

        import strata.server as server_module
        from strata.artifact_store import get_artifact_store, reset_artifact_store
        from strata.config import StrataConfig
        from strata.server import ServerState, app

        artifact_dir = tmp_path / "service-artifacts"
        config = StrataConfig(
            deployment_mode="service",
            auth_mode="trusted_proxy",
            proxy_token="sekrit",
            artifact_dir=artifact_dir,
            cache_dir=tmp_path / "cache",
            acl_config=acl,
        )
        reset_artifact_store()
        monkeypatch.setattr(server_module, "_state", ServerState(config))
        store = get_artifact_store(artifact_dir)
        payload = b"rows"
        version = store.create_artifact(
            "secret",
            hashlib.sha256(payload).hexdigest(),
            transform_spec=_scan_of("file:///wh#test_db.events"),
            tenant="acme",
        )
        with store.open_blob_writer("secret", version) as writer:
            writer.write(payload)
        store.finalize_artifact("secret", version, schema_json="", row_count=1, byte_size=4)
        return TestClient(app), version

    def test_a_denied_artifact_cannot_be_published(self, tmp_path, monkeypatch):
        client, version = self._service(
            monkeypatch,
            tmp_path,
            {"default": "allow", "deny_rules": [{"principal": "*", "tables": ["file:test_db.*"]}]},
        )
        headers = {
            "X-Strata-Proxy-Token": "sekrit",
            "X-Strata-Principal": "intruder",
            "X-Strata-Tenant": "acme",
            "X-Tenant-ID": "acme",
            "X-Strata-Scopes": "artifacts:publish artifacts:read",
        }

        read = client.get(f"/v1/artifacts/secret/v/{version}/data", headers=headers)
        published = client.post(f"/v1/artifacts/secret/v/{version}/publish", headers=headers)

        assert read.status_code in (403, 404)
        assert published.status_code in (403, 404), (
            "a table the caller cannot read was published to anyone with the link"
        )

    def test_an_allowed_artifact_is_still_published(self, tmp_path, monkeypatch):
        client, version = self._service(monkeypatch, tmp_path, {"default": "allow"})
        headers = {
            "X-Strata-Proxy-Token": "sekrit",
            "X-Strata-Principal": "analyst",
            "X-Strata-Tenant": "acme",
            "X-Tenant-ID": "acme",
            "X-Strata-Scopes": "artifacts:publish artifacts:read",
        }

        published = client.post(f"/v1/artifacts/secret/v/{version}/publish", headers=headers)

        assert published.status_code == 200, published.text
        token = published.json()["token"]
        assert client.get(f"/p/{token}/data").status_code == 200


def _scan_of(table_uri: str):
    from strata.artifact_store import TransformSpec

    return TransformSpec(executor="scan@v1", params={"table": table_uri}, inputs=[table_uri])


class TestACitationCannotBeRepointed:
    def test_a_published_version_is_not_deleted_out_from_under_its_link(self, store):
        version = _ready_artifact(store, "nb_abc_cell_c1_var_figure", b"ORIGINAL")
        store.publish_artifact("nb_abc_cell_c1_var_figure", version)

        with pytest.raises(ValueError, match="published"):
            store.delete_artifact("nb_abc_cell_c1_var_figure", version)

        assert store.get_artifact("nb_abc_cell_c1_var_figure", version) is not None

    def test_withdrawing_it_first_allows_the_delete(self, store):
        version = _ready_artifact(store, "fig", b"ORIGINAL")
        publication = store.publish_artifact("fig", version)

        store.revoke_publication(publication.token)

        assert store.delete_artifact("fig", version) is True


class TestWhatTheSweepProtects:
    def test_a_published_figures_inputs_survive_when_the_edge_is_a_name(self, store):
        """A ``@dataset`` or ``strata://name/…`` input records a name edge.

        The sweep must follow it, or the page's chain loses its inputs.
        """
        rows = _ready_artifact(store, "rows", b"[1]")
        store.set_name("team/rows", "rows", rows)
        figure = store.create_artifact(
            "figure",
            "f" * 64,
            input_versions={"strata://name/team/rows": f"rows@v={rows}"},
        )
        with store.open_blob_writer("figure", figure) as writer:
            writer.write(b"png")
        store.finalize_artifact("figure", figure, schema_json="", row_count=0, byte_size=3)
        store.publish_artifact("figure", figure)
        store.set_name("team/rows", "rows", _ready_artifact(store, "rows", b"[2]"))

        store.garbage_collect(max_idle_days=0)

        assert store.get_artifact("rows", rows) is not None, (
            "the published figure's input was collected"
        )
