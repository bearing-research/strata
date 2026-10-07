"""``GET /p/{token}/archive.zip``: the deposit copy, without the store's DSN.

A service with only HTTP access to the central store can produce the bundle without the store's
credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import zipfile
from base64 import b64encode
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest

from strata.artifact_store import ArtifactStore, TransformSpec

TABLE = pa.table({"id": [1, 2, 3], "value": [0.1, 0.2, 0.3]})


def _arrow_bytes() -> bytes:
    buffer = io.BytesIO()
    with ipc.new_stream(buffer, TABLE.schema) as writer:
        writer.write_table(TABLE)
    return buffer.getvalue()


@pytest.fixture
def served(tmp_path):
    """A server holding one published artifact, and the token for it."""
    from tests.conftest import run_server_with_context

    artifact_dir = tmp_path / "served"
    with run_server_with_context(tmp_path / "cache", artifact_dir, "personal") as ctx:
        store = ArtifactStore(artifact_dir)
        payload = _arrow_bytes()
        version = store.create_artifact(
            "fig",
            hashlib.sha256(payload).hexdigest(),
            # A real Arrow artifact records its content type, and that is what
            # names the file inside the bundle.
            transform_spec=TransformSpec(
                executor="notebook/cell@v1", params={"content_type": "arrow/ipc"}, inputs=[]
            ),
        )
        with store.open_blob_writer("fig", version) as writer:
            writer.write(payload)
        store.finalize_artifact("fig", version, schema_json="", row_count=3, byte_size=len(payload))
        publication = store.publish_artifact("fig", version, title="Figure 1")
        yield ctx.base_url, publication, artifact_dir


def _fetch(base_url: str, token: str) -> httpx.Response:
    return httpx.get(f"{base_url}/p/{token}/archive.zip", timeout=30)


class TestACoreResult:
    def test_a_transform_result_is_archived_as_arrow(self, tmp_path):
        """Only notebook cells declare a content type; a ``duckdb_sql`` result was ``.bin``."""
        from strata.api.publication_bundle import write_bundle
        from strata.api.publication_page import content_type_of

        store = ArtifactStore(tmp_path / "artifacts")
        payload = _arrow_bytes()
        version = store.create_artifact(
            "features",
            hashlib.sha256(payload).hexdigest(),
            transform_spec=TransformSpec(
                executor="duckdb_sql@v1", params={"sql": "SELECT 1"}, inputs=[]
            ),
        )
        store.write_blob("features", version, payload)
        store.finalize_artifact(
            "features", version, TABLE.schema.to_string(), TABLE.num_rows, len(payload)
        )
        artifact = store.get_artifact("features", version)
        publication = store.publish_artifact("features", version)
        dest = tmp_path / "bundle"
        dest.mkdir()

        written = write_bundle(store, artifact, dest, publication=publication)

        assert content_type_of(artifact) == "arrow/ipc"
        assert "artifact.arrow" in written
        assert "artifact.parquet" in written


class TestTheBundle:
    def test_it_is_the_same_files_the_cli_writes(self, served, tmp_path):
        """Two implementations of mutually describing files would drift silently, so both must
        match.
        """
        from strata.artifact_cli import cmd_archive

        base_url, publication, artifact_dir = served

        response = _fetch(base_url, publication.token)
        assert response.status_code == 200
        unzipped = tmp_path / "unzipped"
        with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
            bundle.extractall(unzipped)

        by_cli = tmp_path / "by-cli"
        assert (
            cmd_archive(
                argparse.Namespace(
                    ref="fig",
                    artifact_dir=str(artifact_dir),
                    to=str(by_cli),
                    force=False,
                    title="Figure 1",
                    author=None,
                    tenant=None,
                    max_depth=10,
                )
            )
            == 0
        )

        assert sorted(p.name for p in unzipped.iterdir()) == sorted(
            p.name for p in by_cli.iterdir()
        )
        for member in sorted(unzipped.iterdir()):
            twin = by_cli / member.name
            if member.name in ("index.html", "manifest.json", "ro-crate-metadata.json"):
                # These name the publication, and the CLI's copy mints no token. Everything else
                # has to match byte for byte.
                continue
            assert member.read_bytes() == twin.read_bytes(), member.name

    def test_the_payload_bytes_survive_the_round_trip(self, served):
        base_url, publication, _ = served

        with zipfile.ZipFile(io.BytesIO(_fetch(base_url, publication.token).content)) as bundle:
            archived = bundle.read("artifact.arrow")

        assert archived == _arrow_bytes()
        assert hashlib.sha256(archived).hexdigest() == publication.content_sha256

    def test_it_opens_on_index_html(self, served):
        """Someone who unzips this meets the page first, not a .bin."""
        base_url, publication, _ = served

        with zipfile.ZipFile(io.BytesIO(_fetch(base_url, publication.token).content)) as bundle:
            assert bundle.namelist()[0] == "index.html"


def _archive(artifact_dir, to, **overrides):
    """Run ``strata artifact archive`` in-process."""
    from strata.artifact_cli import cmd_archive

    args = dict(
        ref=None,
        token=None,
        artifact_dir=str(artifact_dir),
        to=str(to),
        force=False,
        title=None,
        author=None,
        tenant=None,
        max_depth=10,
    )
    args.update(overrides)
    return cmd_archive(argparse.Namespace(**args))


class TestTheSameBytesEachTime:
    """A depositor checks the zip's digest after upload, so member timestamps must not vary between
    archives of one publication.
    """

    def test_every_member_carries_a_fixed_time_and_mode(self, served):
        base_url, publication, _ = served

        with zipfile.ZipFile(io.BytesIO(_fetch(base_url, publication.token).content)) as bundle:
            for info in bundle.infolist():
                assert info.date_time == (1980, 1, 1, 0, 0, 0), info.filename
                assert info.external_attr == 0o644 << 16, info.filename

    def test_two_fetches_are_one_file(self, served):
        base_url, publication, _ = served

        first = _fetch(base_url, publication.token)
        second = _fetch(base_url, publication.token)
        assert first.content == second.content
        assert first.headers["Content-Digest"] == second.headers["Content-Digest"]

    def test_the_cli_writes_the_zip_the_route_serves(self, served, tmp_path):
        """An archiver without HTTP access to the store produces the same deposit, authors and DOI
        included, since both read the stored record.
        """
        base_url, publication, artifact_dir = served
        ArtifactStore(artifact_dir).update_publication_credits(
            publication.token,
            authors=[{"name": "F. Li", "orcid": "0000-0002-1825-0097"}],
            external_ids=[{"scheme": "doi", "value": "10.5555/figure-1"}],
        )

        served_zip = _fetch(base_url, publication.token)
        by_cli = tmp_path / "deposit.zip"
        assert _archive(artifact_dir, by_cli, token=publication.token) == 0

        assert by_cli.read_bytes() == served_zip.content
        expected = b64encode(hashlib.sha256(by_cli.read_bytes()).digest()).decode()
        assert served_zip.headers["Content-Digest"] == f"sha-256=:{expected}:"
        with zipfile.ZipFile(by_cli) as bundle:
            manifest = bundle.read("manifest.json").decode()
        assert "10.5555/figure-1" in manifest
        assert "0000-0002-1825-0097" in manifest


class TestADeepChain:
    def test_the_archive_walks_as_far_as_the_page(self, tmp_path):
        """The page walked 25 steps and the archive 10, so a long chain lost steps in its zip."""
        import json

        from strata.api.publication_bundle import cached_bundle_zip

        store = ArtifactStore(tmp_path / "artifacts")
        payload = _arrow_bytes()
        inputs = None
        for step in range(12):
            store.create_artifact(
                f"step{step}",
                f"prov-{step}",
                transform_spec=TransformSpec(executor="duckdb_sql@v1", params={}, inputs=[]),
                input_versions=inputs,
            )
            store.write_blob(f"step{step}", 1, payload)
            store.finalize_artifact(f"step{step}", 1, "schema", TABLE.num_rows, len(payload))
            inputs = {f"strata://artifact/step{step}@v=1": f"step{step}@v=1"}
        publication = store.publish_artifact("step11", 1)

        served, _ = cached_bundle_zip(
            store, store.get_artifact("step11", 1), publication=publication
        )

        with zipfile.ZipFile(served) as bundle:
            nodes = json.loads(bundle.read("manifest.json"))["lineage"]["nodes"]
        assert len(nodes) == 12
        # And the CLI by token, with no --max-depth, still writes the same zip.
        out = tmp_path / "deposit.zip"
        assert _archive(store.artifact_dir, out, token=publication.token, max_depth=None) == 0
        assert out.read_bytes() == served.read_bytes()


class TestTheCliByToken:
    def test_the_publication_id_archives_the_same_zip(self, served, tmp_path):
        # The raw token is shown only at mint; the id is what the list routes return.
        base_url, publication, artifact_dir = served
        out = tmp_path / "by-id.zip"

        assert _archive(artifact_dir, out, token=publication.id) == 0
        assert out.read_bytes() == _fetch(base_url, publication.token).content

    def test_a_withdrawn_publication_is_not_archived(self, served, tmp_path):
        _, publication, artifact_dir = served
        ArtifactStore(artifact_dir).revoke_publication(publication.token)

        assert _archive(artifact_dir, tmp_path / "out.zip", token=publication.token) == 1
        assert not (tmp_path / "out.zip").exists()

    def test_an_unknown_token_is_refused(self, served, tmp_path):
        _, _, artifact_dir = served

        assert _archive(artifact_dir, tmp_path / "out.zip", token="not-a-token") == 1

    @pytest.mark.parametrize("which", [{}, {"ref": "fig", "token": "t"}])
    def test_it_takes_a_ref_or_a_token(self, served, tmp_path, which):
        _, _, artifact_dir = served

        assert _archive(artifact_dir, tmp_path / "out.zip", **which) == 1

    def test_a_token_brings_its_own_title(self, served, tmp_path):
        _, publication, artifact_dir = served

        assert _archive(artifact_dir, tmp_path / "out.zip", token=publication.token, title="x") == 1


class TestHeaders:
    def test_the_digest_covers_the_zip_that_was_sent(self, served):
        base_url, publication, _ = served

        response = _fetch(base_url, publication.token)

        expected = b64encode(hashlib.sha256(response.content).digest()).decode()
        assert response.headers["content-digest"] == f"sha-256=:{expected}:"

    def test_it_arrives_as_a_download(self, served):
        base_url, publication, _ = served

        response = _fetch(base_url, publication.token)

        assert response.headers["content-type"] == "application/zip"
        assert publication.token in response.headers["content-disposition"]


class TestRefusals:
    def test_a_withdrawn_publication_hands_over_nothing(self, served):
        """The page still says withdrawn, but handing over the archive would undo the withdrawal."""
        base_url, publication, artifact_dir = served
        ArtifactStore(artifact_dir).revoke_publication(publication.token)

        assert _fetch(base_url, publication.token).status_code == 410
        # And the page is still readable, which is the distinction.
        assert httpx.get(f"{base_url}/p/{publication.token}", timeout=10).status_code == 200

    def test_an_unknown_token_is_a_404(self, served):
        base_url, _, _ = served

        assert _fetch(base_url, "not-a-token").status_code == 404

    def test_upstream_bytes_are_not_in_it(self, served, tmp_path):
        """Publishing shows which steps produced a result; it does not consent to handing over
        upstream datasets. Same rule as ``/p/{token}/data``.
        """
        base_url, publication, _ = served

        with zipfile.ZipFile(io.BytesIO(_fetch(base_url, publication.token).content)) as bundle:
            names = bundle.namelist()

        assert [n for n in names if Path(n).suffix == ".arrow"] == ["artifact.arrow"]


def _on_the_event_loop() -> bool:
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


class TestBuildingIt:
    """Anyone with the link can ask for the zip, so a build must neither stall the server nor
    repeat.
    """

    @staticmethod
    def _count_builds(monkeypatch) -> list[bool]:
        """Record, per build, whether it ran on the event loop's thread."""
        from strata.api import publication_bundle

        builds: list[bool] = []
        real = publication_bundle.bundle_zip

        def spy(*args, **kwargs):
            builds.append(_on_the_event_loop())
            return real(*args, **kwargs)

        monkeypatch.setattr(publication_bundle, "bundle_zip", spy)
        return builds

    def test_it_is_built_off_the_event_loop(self, served, monkeypatch):
        base_url, publication, _ = served
        builds = self._count_builds(monkeypatch)

        assert _fetch(base_url, publication.token).status_code == 200

        assert builds == [False]

    def test_a_second_fetch_reuses_the_first_build(self, served, monkeypatch):
        base_url, publication, _ = served
        builds = self._count_builds(monkeypatch)

        first = _fetch(base_url, publication.token)
        second = _fetch(base_url, publication.token)

        assert len(builds) == 1
        assert first.content == second.content
        assert first.headers["Content-Digest"] == second.headers["Content-Digest"]

    def test_new_credits_rebuild_it_and_replace_the_old_one(self, served, monkeypatch):
        from strata.api.publication_bundle import ARCHIVE_CACHE_DIRNAME

        base_url, publication, artifact_dir = served
        builds = self._count_builds(monkeypatch)
        _fetch(base_url, publication.token)

        ArtifactStore(artifact_dir).update_publication_credits(
            publication.token, external_ids=[{"scheme": "doi", "value": "10.5555/figure-1"}]
        )
        response = _fetch(base_url, publication.token)

        assert len(builds) == 2
        with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
            assert "10.5555/figure-1" in bundle.read("manifest.json").decode()
        cached = list((artifact_dir / ARCHIVE_CACHE_DIRNAME / publication.id).iterdir())
        assert len(cached) == 1

    def test_withdrawing_drops_the_built_copy(self, served):
        from strata.api.publication_bundle import ARCHIVE_CACHE_DIRNAME

        base_url, publication, artifact_dir = served
        _fetch(base_url, publication.token)
        cache_dir = artifact_dir / ARCHIVE_CACHE_DIRNAME / publication.id
        assert any(cache_dir.iterdir())

        revoked = httpx.delete(f"{base_url}/v1/publications/{publication.token}", timeout=10)

        assert revoked.status_code == 200
        assert not cache_dir.exists()

    def test_verify_hashes_off_the_event_loop(self, served, monkeypatch):
        base_url, publication, _ = served
        calls: list[bool] = []
        real = ArtifactStore.blob_digest

        def spy(self, *args, **kwargs):
            calls.append(_on_the_event_loop())
            return real(self, *args, **kwargs)

        monkeypatch.setattr(ArtifactStore, "blob_digest", spy)

        verified = httpx.get(f"{base_url}/p/{publication.token}/verify", timeout=10).json()

        assert verified["matches"] is True
        assert calls == [False]


def _verify(base_url: str, token: str, digest: str) -> httpx.Response:
    return httpx.get(f"{base_url}/p/{token}/verify", params={"sha256": digest}, timeout=30)


def _member_digest(base_url: str, token: str, name: str) -> str:
    with zipfile.ZipFile(io.BytesIO(_fetch(base_url, token).content)) as bundle:
        return hashlib.sha256(bundle.read(name)).hexdigest()


class TestVerifyingAFileFromTheBundle:
    """A reader holds one file of the deposit and asks whether it is this publication's."""

    @pytest.mark.parametrize("name", ["artifact.arrow", "artifact.parquet"])
    def test_either_file_is_accepted(self, served, name):
        base_url, publication, _ = served
        digest = _member_digest(base_url, publication.token, name)

        verified = _verify(base_url, publication.token, digest.upper()).json()

        assert verified["matches"] is True
        assert verified["file"] == name
        assert verified["sha256"] == digest

    def test_a_file_from_elsewhere_is_not(self, served):
        base_url, publication, _ = served

        verified = _verify(base_url, publication.token, hashlib.sha256(b"other").hexdigest()).json()

        assert verified["matches"] is False
        assert verified["file"] is None
        # The stored bytes are still checked and still fine.
        assert verified["actual_sha256"] == verified["recorded_sha256"]

    def test_a_malformed_digest_is_refused(self, served):
        base_url, publication, _ = served

        assert _verify(base_url, publication.token, "abc").status_code == 422


class TestALargeTable:
    def test_only_the_arrow_file_verifies_without_a_parquet_copy(self, served, monkeypatch):
        """Past the cap the archive has no Parquet, so no Parquet digest can name a file of it."""
        from strata.api import publication_bundle

        base_url, publication, _ = served
        parquet = _member_digest(base_url, publication.token, "artifact.parquet")
        publication_bundle.drop_cached_bundles(ArtifactStore(served[2]), publication.token)
        monkeypatch.setattr(
            publication_bundle, "PARQUET_COMPANION_MAX_BYTES", len(_arrow_bytes()) - 1
        )

        assert _verify(base_url, publication.token, parquet).json()["file"] is None
        arrow = hashlib.sha256(_arrow_bytes()).hexdigest()
        assert _verify(base_url, publication.token, arrow).json()["file"] == "artifact.arrow"

    def test_it_is_archived_without_its_parquet_copy(self, served, monkeypatch, tmp_path):
        """The Parquet rendering holds the whole table in memory, so past a fixed size the Arrow
        file stands alone, and the route and the CLI still agree byte for byte.
        """
        from strata.api import publication_bundle

        base_url, publication, artifact_dir = served
        monkeypatch.setattr(
            publication_bundle, "PARQUET_COMPANION_MAX_BYTES", len(_arrow_bytes()) - 1
        )

        response = _fetch(base_url, publication.token)
        by_cli = tmp_path / "deposit.zip"
        assert _archive(artifact_dir, by_cli, token=publication.token) == 0

        with zipfile.ZipFile(io.BytesIO(response.content)) as bundle:
            assert "artifact.parquet" not in bundle.namelist()
            assert "additional_files" not in bundle.read("manifest.json").decode()
            assert bundle.read("artifact.arrow") == _arrow_bytes()
        assert by_cli.read_bytes() == response.content

    def test_at_the_cap_it_still_carries_one(self, served, monkeypatch):
        from strata.api import publication_bundle

        base_url, publication, _ = served
        monkeypatch.setattr(publication_bundle, "PARQUET_COMPANION_MAX_BYTES", len(_arrow_bytes()))

        with zipfile.ZipFile(io.BytesIO(_fetch(base_url, publication.token).content)) as bundle:
            assert "artifact.parquet" in bundle.namelist()
