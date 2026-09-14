"""``@fetch``: the bytes a cell reads from a URL, recorded as an input. Item 44."""

from __future__ import annotations

import hashlib
import http.server
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from strata.notebook import fetch as fetch_module
from strata.notebook.annotations import parse_annotations
from strata.notebook.fetch import FetchCache, FetchError, FetchPinMismatch
from strata.notebook.models import FetchSpec


class _Origin:
    """A small HTTP server whose bytes the test can change, and which honours
    ``If-None-Match`` so a conditional check is observable."""

    def __init__(self):
        self.body = b"zone,borough\n1,Queens\n"
        self.requests: list[tuple[str, dict[str, str]]] = []
        self.redirect_to: str | None = None
        origin = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                origin.requests.append((self.path, dict(self.headers)))
                if origin.redirect_to and self.path == "/redirect":
                    self.send_response(302)
                    self.send_header("Location", origin.redirect_to)
                    self.end_headers()
                    return
                etag = '"' + hashlib.sha256(origin.body).hexdigest()[:16] + '"'
                if self.headers.get("If-None-Match") == etag:
                    self.send_response(304)
                    self.send_header("ETag", etag)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("ETag", etag)
                self.send_header("Content-Length", str(len(origin.body)))
                self.end_headers()
                self.wfile.write(origin.body)

            def log_message(self, *args):
                return None

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path: str = "/zones.csv") -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}{path}"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def origin():
    server = _Origin()
    yield server
    server.close()


def _cache(tmp_path: Path, clock=None) -> FetchCache:
    kwargs = {"allowed_hosts": ("127.0.0.1",)}
    if clock is not None:
        kwargs["clock"] = clock
    return FetchCache(tmp_path, **kwargs)


class TestAnnotation:
    def test_the_declaration_parses_with_its_options(self):
        digest = "ab" * 32
        (spec,) = parse_annotations(
            f"# @fetch zones https://x.org/zones.csv sha256={digest} refetch=never\nx = 1"
        ).fetches

        assert (spec.name, spec.url, spec.sha256, spec.refetch) == (
            "zones",
            "https://x.org/zones.csv",
            digest,
            "never",
        )

    @pytest.mark.parametrize(
        "value",
        [
            "zones",
            "1bad https://x.org/a",
            "zones https://x.org/a refetch=sometimes",
            "z u sha256=nothex",
        ],
    )
    def test_a_malformed_declaration_is_ignored(self, value):
        assert parse_annotations(f"# @fetch {value}\nx = 1").fetches == []


class TestTheCache:
    def test_bytes_are_stored_by_digest_under_the_urls_file_name(self, tmp_path, origin):
        fetched = _cache(tmp_path).resolve(FetchSpec(name="zones", url=origin.url()))

        assert fetched.sha256 == hashlib.sha256(origin.body).hexdigest()
        assert fetched.path == tmp_path / ".strata" / "fetch" / fetched.sha256 / "zones.csv"
        assert fetched.path.read_bytes() == origin.body

    def test_within_the_recheck_interval_the_url_is_not_asked_again(self, tmp_path, origin):
        """Staleness recomputes on every edit; a request per keystroke would be
        worse than the problem."""
        now = [1000.0]
        cache = _cache(tmp_path, clock=lambda: now[0])
        spec = FetchSpec(name="zones", url=origin.url())
        cache.resolve(spec)

        now[0] += fetch_module.STALE_CHECK_SECONDS / 2
        cache.resolve(spec)

        assert len(origin.requests) == 1

    def test_after_the_interval_the_check_is_conditional(self, tmp_path, origin):
        now = [1000.0]
        cache = _cache(tmp_path, clock=lambda: now[0])
        spec = FetchSpec(name="zones", url=origin.url())
        first = cache.resolve(spec)

        now[0] += fetch_module.STALE_CHECK_SECONDS + 1
        second = cache.resolve(spec)

        assert second == first
        assert "If-None-Match" in origin.requests[-1][1]

    def test_moved_bytes_are_a_new_digest(self, tmp_path, origin):
        cache = _cache(tmp_path)
        spec = FetchSpec(name="zones", url=origin.url())
        before = cache.resolve(spec)

        origin.body = b"zone,borough\n1,Brooklyn\n"
        after = cache.resolve(spec, max_age=0)

        assert after.sha256 != before.sha256
        assert after.path.read_bytes() == origin.body
        assert before.path.read_bytes() != origin.body, "the old bytes stay where they were"

    def test_never_uses_what_is_cached(self, tmp_path, origin):
        cache = _cache(tmp_path)
        spec = FetchSpec(name="zones", url=origin.url(), refetch="never")
        cache.resolve(spec)
        origin.body = b"changed"

        cache.resolve(spec, max_age=0)

        assert len(origin.requests) == 1

    def test_always_downloads_ignoring_validators(self, tmp_path, origin):
        cache = _cache(tmp_path)
        spec = FetchSpec(name="zones", url=origin.url(), refetch="always")
        cache.resolve(spec)
        cache.resolve(spec)

        assert len(origin.requests) == 2
        assert "If-None-Match" not in origin.requests[-1][1]


class TestPins:
    def test_a_pin_that_matches_needs_no_network(self, tmp_path, origin):
        cache = _cache(tmp_path)
        digest = cache.resolve(FetchSpec(name="zones", url=origin.url())).sha256
        pinned = FetchSpec(name="zones", url=origin.url(), sha256=digest)

        cache.resolve(pinned, max_age=0)

        assert len(origin.requests) == 1
        assert cache.fingerprint(pinned) == f"zones:fetch:{origin.url()}:{digest}"

    def test_bytes_that_differ_from_the_pin_fail_with_both_digests(self, tmp_path, origin):
        pin = "0" * 64
        spec = FetchSpec(name="zones", url=origin.url(), sha256=pin)

        with pytest.raises(FetchPinMismatch) as caught:
            _cache(tmp_path).resolve(spec)

        assert pin in str(caught.value)
        assert hashlib.sha256(origin.body).hexdigest() in str(caught.value)


class TestTheGuard:
    def test_a_private_address_is_refused_unless_its_host_is_named(self, tmp_path, origin):
        with pytest.raises(FetchError, match="non-routable"):
            FetchCache(tmp_path).resolve(FetchSpec(name="zones", url=origin.url()))

        assert origin.requests == []

    def test_a_redirect_is_checked_hop_by_hop(self, tmp_path, origin):
        """A permitted host that redirects to the metadata service is the
        request the guard exists to refuse."""
        origin.redirect_to = "http://169.254.169.254/latest/meta-data/"

        with pytest.raises(FetchError, match="169.254.169.254"):
            _cache(tmp_path).resolve(FetchSpec(name="zones", url=origin.url("/redirect")))

    def test_a_redirect_to_a_permitted_host_is_followed(self, tmp_path, origin):
        origin.redirect_to = origin.url()

        fetched = _cache(tmp_path).resolve(FetchSpec(name="zones", url=origin.url("/redirect")))

        assert fetched.path.read_bytes() == origin.body
        assert [path for path, _ in origin.requests] == ["/redirect", "/zones.csv"]

    def test_only_http_is_fetched(self, tmp_path):
        with pytest.raises(FetchError, match="scheme"):
            _cache(tmp_path).resolve(FetchSpec(name="zones", url="file:///etc/passwd"))

    def test_an_unresolvable_fetch_fingerprints_uniquely_and_never_raises(self, tmp_path):
        spec = FetchSpec(name="zones", url="file:///etc/passwd")
        cache = _cache(tmp_path)

        assert cache.fingerprint(spec) != cache.fingerprint(spec)


class TestInACell:
    def _session(self, tmp_path, source, monkeypatch):
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        monkeypatch.setattr(
            "strata.server._state",
            SimpleNamespace(
                config=SimpleNamespace(
                    deployment_mode="personal", notebook_fetch_allowed_hosts=["127.0.0.1"]
                )
            ),
        )
        nb = create_notebook(tmp_path, "Fetching")
        add_cell_to_notebook(nb, "c1", None)
        write_cell(nb, "c1", source)
        add_cell_to_notebook(nb, "c2", "c1")
        write_cell(nb, "c2", "n = rows")
        return NotebookSession(parse_notebook(nb), nb)

    async def test_the_cell_reads_the_bytes_and_goes_stale_when_they_move(
        self, tmp_path, origin, monkeypatch
    ):
        from strata.notebook.executor import CellExecutor

        source = (
            f"# @fetch zones {origin.url()}\n"
            "rows = len(zones.read_text().splitlines())\n"
            "print(zones.name)"
        )
        session = self._session(tmp_path, source, monkeypatch)

        first = await CellExecutor(session).execute_cell("c1", source)
        assert first.success, first.error
        assert "zones.csv" in first.stdout
        assert session.compute_staleness()["c1"].status.value == "ready"

        again = await CellExecutor(session).execute_cell("c1", source)
        assert again.cache_hit is True

        origin.body = b"zone,borough\n1,Queens\n2,Bronx\n"
        # The recheck interval has passed: staleness asks the URL again.
        monkeypatch.setattr(fetch_module, "STALE_CHECK_SECONDS", 0.0)
        assert session.compute_staleness()["c1"].status.value != "ready"

        moved = await CellExecutor(session).execute_cell("c1", source)
        assert moved.success, moved.error
        assert moved.cache_hit is False

    async def test_a_pin_the_url_no_longer_matches_fails_the_cell_with_both_digests(
        self, tmp_path, origin, monkeypatch
    ):
        from strata.notebook.executor import CellExecutor

        pin = "1" * 64
        source = f"# @fetch zones {origin.url()} sha256={pin}\nrows = 1"
        session = self._session(tmp_path, source, monkeypatch)

        result = await CellExecutor(session).execute_cell("c1", source)

        assert result.success is False
        assert pin in result.error
        assert hashlib.sha256(origin.body).hexdigest() in result.error

    def test_a_fetching_cell_is_not_batched(self, tmp_path, origin, monkeypatch):
        from strata.notebook.executor import CellExecutor, is_cell_batchable

        source = f"# @fetch zones {origin.url()}\nrows = 1"
        session = self._session(tmp_path, source, monkeypatch)
        cell = session.notebook_state.get_cell("c1")

        assert is_cell_batchable(CellExecutor(session), cell) is False
