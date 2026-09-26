"""``GET /p/{token}/archive.zip`` — the deposit copy, without the store's DSN.

The bundle was built by opening the store locally, so a service that only has
HTTP access to the central store had to be handed the store's credentials to
produce one. Item 18.
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


class TestTheBundle:
    def test_it_is_the_same_files_the_cli_writes(self, served, tmp_path):
        """The property the item asks for. Two implementations of a set of
        files that describe each other would drift, and the drift would be
        silent — both would keep producing a bundle."""
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
                # These name the publication, and the CLI's copy is not one —
                # it mints no token. Everything else has to match byte for byte.
                continue
            assert member.read_bytes() == twin.read_bytes(), member.name

    def test_the_payload_bytes_survive_the_round_trip(self, served):
        base_url, publication, _ = served

        with zipfile.ZipFile(io.BytesIO(_fetch(base_url, publication.token).content)) as bundle:
            archived = bundle.read("artifact.arrow")

        assert archived == _arrow_bytes()
        assert hashlib.sha256(archived).hexdigest() == publication.content_sha256

    def test_it_opens_on_index_html(self, served):
        """A person who unzips this meets the page first, not a .bin."""
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
    """A depositor records the digest of the zip they received and checks it
    after the upload. Each member used to carry the modification time of a
    file written moments before into a fresh directory, so two archives of one
    publication differed in bytes, and in digest, while no file inside did."""

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
        """What an archiver without HTTP access to the store produces is the
        same deposit, authors and DOI included, since both read the stored
        record."""
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


class TestTheCliByToken:
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
        """The page still resolves and says withdrawn, because a reader chasing
        a footnote deserves that answer. Handing them the archive anyway would
        undo the withdrawal."""
        base_url, publication, artifact_dir = served
        ArtifactStore(artifact_dir).revoke_publication(publication.token)

        assert _fetch(base_url, publication.token).status_code == 410
        # And the page is still readable, which is the distinction.
        assert httpx.get(f"{base_url}/p/{publication.token}", timeout=10).status_code == 200

    def test_an_unknown_token_is_a_404(self, served):
        base_url, _, _ = served

        assert _fetch(base_url, "not-a-token").status_code == 404

    def test_upstream_bytes_are_not_in_it(self, served, tmp_path):
        """Showing which steps produced a result is transparency; handing over
        the upstream datasets is not the same thing, and is not what publishing
        consented to. Same rule as ``/p/{token}/data``."""
        base_url, publication, _ = served

        with zipfile.ZipFile(io.BytesIO(_fetch(base_url, publication.token).content)) as bundle:
            names = bundle.namelist()

        assert [n for n in names if Path(n).suffix == ".arrow"] == ["artifact.arrow"]
