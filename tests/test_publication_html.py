"""A published HTML result: shown on its page in a sandboxed frame, fetched from other hosts only
when the reader asks.
"""

from __future__ import annotations

import hashlib
from html import escape

import pytest
from vestigo.provenance import TransformSpec

# What plotly's default HTML is like: its library from a CDN, an SVG namespace inline.
CDN_HTML = (
    '<div id="p"></div><script src="https://cdn.plot.ly/plotly-4.1.1.min.js"></script>'
    "<script>Plotly.newPlot('p', [])</script>"
)
SELF_CONTAINED_HTML = (
    "<table><tr><td>Oslo</td></tr></table>"
    '<svg xmlns="http://www.w3.org/2000/svg"><circle r="2"/></svg>'
)


def _client_and_store(monkeypatch, tmp_path, **config):
    from fastapi.testclient import TestClient
    from vestigo.store import get_artifact_store, reset_artifact_store

    import strata.server as server_module
    from strata.config import StrataConfig

    artifact_dir = tmp_path / "artifacts"
    reset_artifact_store()
    monkeypatch.setattr(
        server_module,
        "_state",
        server_module.ServerState(
            StrataConfig(artifact_dir=artifact_dir, cache_dir=tmp_path / "cache", **config)
        ),
    )
    return TestClient(server_module.app), get_artifact_store(artifact_dir)


@pytest.fixture
def client_and_store(monkeypatch, tmp_path):
    return _client_and_store(monkeypatch, tmp_path)


def _published(store, payload: str | bytes, content_type: str = "text/html"):
    blob = payload.encode() if isinstance(payload, str) else payload
    version = store.create_artifact(
        "fig",
        hashlib.sha256(blob).hexdigest(),
        transform_spec=TransformSpec(
            executor="local@v1", params={"content_type": content_type}, inputs=[]
        ),
    )
    with store.open_blob_writer("fig", version) as writer:
        writer.write(blob)
    store.finalize_artifact("fig", version, schema_json="", row_count=0, byte_size=len(blob))
    return store.publish_artifact("fig", version, title="Figure 3")


def _policy(response) -> list[str]:
    return [d.strip() for d in response.headers["content-security-policy"].split(";")]


class TestExternalHosts:
    @pytest.mark.parametrize(
        ("html", "hosts"),
        [
            (CDN_HTML, ["cdn.plot.ly"]),
            (SELF_CONTAINED_HTML, []),
            ("<script src='//cdn.jsdelivr.net/npm/vega@6'></script>", ["cdn.jsdelivr.net"]),
            (
                "import('HTTPS://CDN.Example.org/x.js'); new WebSocket('wss://live.example.org')",
                ["cdn.example.org", "live.example.org"],
            ),
            # As a JSON string carries it, and a template literal.
            ('{"src": "https:\\/\\/cdn.plot.ly\\/x.js"}', ["cdn.plot.ly"]),
            ("fetch(`//cdn.example.org/a`)", ["cdn.example.org"]),
            # One label, and a port, after a scheme.
            ("fetch('http://localhost:8000/data')", ["localhost:8000"]),
            # A comment is no address.
            ("<script>// a.b.c\n</script>", []),
        ],
    )
    def test_it_names_what_showing_it_may_contact(self, html, hosts):
        from strata.api.publication_page import external_hosts

        assert external_hosts(html) == hosts


class TestThePage:
    def test_a_self_contained_result_is_shown_in_its_frame(self, client_and_store):
        client, store = client_and_store
        token = _published(store, SELF_CONTAINED_HTML).token

        page = client.get(f"/p/{token}").text

        assert "<h2>Result</h2>" in page
        assert f"src='http://testserver/p/{token}/view'" in page
        assert "sandbox='allow-scripts allow-popups'" in page
        assert "load it" not in page
        # Framed, not inlined: the result never runs in the page's own document.
        assert "<td>Oslo</td>" not in page

    def test_a_result_naming_other_hosts_waits_for_the_reader(self, client_and_store):
        client, store = client_and_store
        token = _published(store, CDN_HTML).token

        page = client.get(f"/p/{token}").text

        assert "This result loads from <code>cdn.plot.ly</code>" in page
        assert (
            f"<a href='http://testserver/p/{token}/view' target='strata-result'>load it</a>" in page
        )
        frame = page[page.index("<iframe class='result'") :]
        frame = frame[: frame.index("</iframe>")]
        assert "name='strata-result'" in frame
        assert " src=" not in frame

    def test_one_too_large_to_read_per_request_is_not_framed(self, client_and_store, monkeypatch):
        import strata.api.routers.publications as publications

        client, store = client_and_store
        token = _published(store, SELF_CONTAINED_HTML).token
        monkeypatch.setattr(publications, "MAX_FRAMED_HTML_BYTES", len(SELF_CONTAINED_HTML) - 1)

        assert "<iframe" not in client.get(f"/p/{token}").text
        assert client.get(f"/p/{token}/view").status_code == 404

    def test_a_withdrawn_result_shows_no_frame(self, client_and_store):
        client, store = client_and_store
        token = _published(store, SELF_CONTAINED_HTML).token
        store.revoke_publication(token)

        page = client.get(f"/p/{token}").text

        assert "<iframe" not in page


class TestTheView:
    def test_a_self_contained_result_can_fetch_nothing(self, client_and_store):
        client, store = client_and_store
        token = _published(store, SELF_CONTAINED_HTML).token

        response = client.get(f"/p/{token}/view")

        assert response.status_code == 200
        assert response.text == SELF_CONTAINED_HTML
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        policy = _policy(response)
        assert policy[0] == "sandbox allow-scripts allow-popups"
        assert "default-src 'none'" in policy
        assert not [d for d in policy if "http" in d]
        assert "frame-ancestors 'self'" in policy
        assert response.headers["referrer-policy"] == "no-referrer"
        assert response.headers["x-content-type-options"] == "nosniff"

    def test_a_result_naming_other_hosts_may_reach_them(self, client_and_store):
        client, store = client_and_store
        token = _published(store, CDN_HTML).token

        response = client.get(f"/p/{token}/view")

        assert response.status_code == 200
        policy = _policy(response)
        assert "default-src 'none'" in policy
        # The named host, and no other.
        assert [d for d in policy if d.startswith("script-src")] == [
            "script-src 'unsafe-inline' 'unsafe-eval' data: blob: "
            "http://cdn.plot.ly https://cdn.plot.ly"
        ]
        assert "connect-src http://cdn.plot.ly https://cdn.plot.ly" in policy
        assert response.headers["referrer-policy"] == "no-referrer"

    def test_it_may_be_framed_wherever_its_page_may(self, monkeypatch, tmp_path):
        client, store = _client_and_store(
            monkeypatch, tmp_path, embed_frame_ancestors=["https://lab.example.org"]
        )
        token = _published(store, SELF_CONTAINED_HTML).token

        page = client.get(f"/p/{token}")
        view = client.get(f"/p/{token}/view")

        framing = "frame-ancestors 'self' https://lab.example.org"
        assert framing in _policy(page)
        assert framing in _policy(view)

    def test_bytes_that_are_not_utf8_are_served_as_they_are(self, client_and_store):
        client, store = client_and_store
        latin1 = '<meta charset="iso-8859-1"><p>caf\xe9</p>'.encode("latin-1")
        token = _published(store, latin1).token

        response = client.get(f"/p/{token}/view")

        assert response.content == latin1
        assert response.headers["content-type"] == "text/html"

    @pytest.mark.parametrize("content_type", ["image/png", "text/markdown", ""])
    def test_only_an_html_result_has_one(self, client_and_store, content_type):
        client, store = client_and_store
        token = _published(store, SELF_CONTAINED_HTML, content_type).token

        assert client.get(f"/p/{token}/view").status_code == 404

    def test_a_withdrawn_result_is_gone(self, client_and_store):
        client, store = client_and_store
        token = _published(store, SELF_CONTAINED_HTML).token
        store.revoke_publication(token)

        assert client.get(f"/p/{token}/view").status_code == 410

    def test_the_bytes_still_download(self, client_and_store):
        client, store = client_and_store
        token = _published(store, CDN_HTML).token

        response = client.get(f"/p/{token}/data")

        assert response.headers["content-disposition"] == "attachment"
        assert response.content == CDN_HTML.encode()


class TestTheArchive:
    def _bundle(self, store, tmp_path, payload):
        from strata.api.publication_bundle import write_bundle

        publication = _published(store, payload)
        dest = tmp_path / "bundle"
        dest.mkdir()
        write_bundle(
            store, store.get_artifact("fig", publication.version), dest, publication=publication
        )
        return dest

    def test_a_self_contained_result_is_framed_from_its_file(self, client_and_store, tmp_path):
        _, store = client_and_store
        dest = self._bundle(store, tmp_path, SELF_CONTAINED_HTML)

        index = (dest / "index.html").read_text()
        view = (dest / "view.html").read_text()

        assert (dest / "artifact.html").read_text() == SELF_CONTAINED_HTML
        assert "src='view.html'></iframe>" in index
        # No server sends the policy, so the framed copy carries it.
        assert view.startswith(
            '<meta charset="utf-8"><meta http-equiv="Content-Security-Policy" '
            'content="default-src &#x27;none&#x27;;'
        )
        assert view.endswith(SELF_CONTAINED_HTML)
        assert "http:" not in view.split(">", 2)[1]
        assert "https:" not in view.split(">", 2)[1]
        assert "makes no external requests." in (dest / "README.md").read_text()

    def test_a_result_naming_other_hosts_says_so_and_waits(self, client_and_store, tmp_path):
        _, store = client_and_store
        dest = self._bundle(store, tmp_path, CDN_HTML)

        index = (dest / "index.html").read_text()
        view = (dest / "view.html").read_text()

        assert "<a href='view.html' target='strata-result'>load it</a>" in index
        from strata.api.publication_page import result_policy

        assert view.startswith(
            '<meta charset="utf-8"><meta http-equiv="Content-Security-Policy" '
            f'content="{escape(result_policy(["cdn.plot.ly"]))}">'
        )
        assert (
            "makes no external requests until you load the result, which reaches cdn.plot.ly"
            in (dest / "README.md").read_text()
        )

    def test_a_bundle_without_one_has_no_view(self, client_and_store, tmp_path):
        from strata.api.publication_bundle import write_bundle

        _, store = client_and_store
        publication = _published(store, "PNG", "image/png")
        dest = tmp_path / "bundle"
        dest.mkdir()

        written = write_bundle(
            store, store.get_artifact("fig", publication.version), dest, publication=publication
        )

        assert "view.html" not in written
        assert "view.html" not in (dest / "README.md").read_text()

    def test_an_archive_built_by_an_earlier_format_is_rebuilt(self, client_and_store, monkeypatch):
        from strata.api.publication_bundle import BUNDLE_FORMAT, cached_bundle_zip

        _, store = client_and_store
        publication = _published(store, SELF_CONTAINED_HTML)
        artifact = store.get_artifact("fig", publication.version)
        first, _ = cached_bundle_zip(store, artifact, publication=publication)

        monkeypatch.setattr("strata.api.publication_bundle.BUNDLE_FORMAT", BUNDLE_FORMAT + 1)
        second, _ = cached_bundle_zip(store, artifact, publication=publication)

        assert second != first
        assert not first.exists()
