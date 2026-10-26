"""``verify_artifacts`` checks a blob's bytes only through a validator the application
registered for its declared content type; the ledger knows no format itself."""

from __future__ import annotations

import pytest

from vestigo.provenance import TransformSpec
from vestigo.store import _VALIDATORS, ArtifactStore, register_validator


@pytest.fixture
def store(tmp_path) -> ArtifactStore:
    return ArtifactStore(tmp_path / "artifacts")


@pytest.fixture
def validators(monkeypatch):
    """This test's registrations only; the suite's own (conftest) are restored after."""
    monkeypatch.setattr("vestigo.store._VALIDATORS", dict(_VALIDATORS))


def _stored(store: ArtifactStore, artifact_id: str, data: bytes, content_type: str, rows=None):
    version = store.create_artifact(
        artifact_id,
        f"prov-{artifact_id}",
        TransformSpec(executor="x@v1", params={"content_type": content_type}, inputs=[]),
    )
    store.write_blob(artifact_id, version, data)
    store.finalize_artifact(artifact_id, version, "", rows, len(data))


def _lines(data: bytes) -> int:
    if b"\x00" in data:
        raise ValueError("not text")
    return data.count(b"\n")


@pytest.mark.usefixtures("validators")
class TestARegisteredValidator:
    def test_its_error_is_an_invalid_stream(self, store):
        register_validator("text/lines", _lines)
        _stored(store, "good", b"a\nb\n", "text/lines", rows=2)
        _stored(store, "bad", b"a\x00", "text/lines", rows=1)

        [finding] = store.verify_artifacts()

        assert (finding["artifact_id"], finding["problem"]) == ("bad", "invalid_stream")
        assert finding["detail"] == "not text"

    def test_its_count_is_checked_against_the_recorded_rows(self, store):
        register_validator("text/lines", _lines)
        _stored(store, "short", b"a\n", "text/lines", rows=3)

        [finding] = store.verify_artifacts()

        assert (finding["artifact_id"], finding["problem"]) == ("short", "row_count_mismatch")

    def test_a_type_without_one_is_checked_for_its_digest_only(self, store):
        register_validator("text/lines", _lines)
        _stored(store, "opaque", b"a\x00", "application/octet-stream", rows=7)

        assert store.verify_artifacts() == []
