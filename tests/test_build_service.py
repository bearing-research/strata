"""Unit tests for ``BuildService.assemble_manifest`` — pure, no server/DB.

The signing itself (``generate_build_manifest``) is its own concern and is
stubbed here; these tests cover the service's own logic: resolving a build's
input URIs to ``(artifact_id, version)`` pairs, raising on an unresolvable
input, and assembling the executor metadata.
"""

from types import SimpleNamespace

import pytest

from strata.services.build import _resolve_to_artifact_version, build_service


class _FakeStore:
    def __init__(self, names=None):
        self._names = names or {}  # name -> (id, version)

    def resolve_name(self, name, *, tenant=None):
        hit = self._names.get(name)
        return SimpleNamespace(id=hit[0], version=hit[1]) if hit else None


def _build(**kw):
    return SimpleNamespace(
        build_id="b1",
        artifact_id="OUT",
        version=2,
        executor_ref="duckdb_sql@v1",
        params={"sql": "select 1"},
        tenant_id="t1",
        **kw,
    )


@pytest.fixture
def captured_manifest():
    """A stub signer that captures its ``generate_build_manifest`` kwargs.

    Returns a namespace exposing ``.signer`` (passed to ``assemble_manifest``)
    and ``.calls`` (the captured kwargs).
    """
    calls: dict = {}

    class _FakeSigner:
        def generate_build_manifest(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(
                to_dict=lambda: {"ok": True, "n_inputs": len(kwargs["input_artifacts"])}
            )

    return SimpleNamespace(signer=_FakeSigner(), calls=calls)


def test_resolve_artifact_and_name_uris():
    store = _FakeStore(names={"champion": ("A", 7)})
    assert _resolve_to_artifact_version("strata://artifact/X@v=3", store) == ("X", 3)
    assert _resolve_to_artifact_version("strata://name/champion", store) == ("A", 7)
    assert _resolve_to_artifact_version("strata://name/missing", store) is None
    assert _resolve_to_artifact_version("s3://bucket/t", store) is None


def test_assemble_manifest_resolves_inputs_and_builds_metadata(captured_manifest):
    store = _FakeStore(names={"champ": ("A", 7)})
    build = _build(input_uris=["strata://artifact/X@v=3", "strata://name/champ"])

    result = build_service.assemble_manifest(
        store,
        signer=captured_manifest.signer,
        build=build,
        base_url="http://host",
        max_output_bytes=1000,
        url_expiry_seconds=60.0,
    )

    assert result == {"ok": True, "n_inputs": 2}
    assert captured_manifest.calls["input_artifacts"] == [("X", 3), ("A", 7)]
    assert captured_manifest.calls["metadata"] == {
        "build_id": "b1",
        "artifact_id": "OUT",
        "version": 2,
        "executor_ref": "duckdb_sql@v1",
        "params": {"sql": "select 1"},
    }
    assert captured_manifest.calls["base_url"] == "http://host"


def test_assemble_manifest_raises_valueerror_on_unresolvable_input(captured_manifest):
    store = _FakeStore()
    build = _build(input_uris=["strata://name/nope"])

    with pytest.raises(ValueError, match="Cannot resolve input artifact"):
        build_service.assemble_manifest(
            store,
            signer=captured_manifest.signer,
            build=build,
            base_url="http://host",
            max_output_bytes=1,
            url_expiry_seconds=1.0,
        )


def test_assemble_manifest_handles_no_inputs(captured_manifest):
    build = _build(input_uris=None)
    build_service.assemble_manifest(
        _FakeStore(),
        signer=captured_manifest.signer,
        build=build,
        base_url="http://host",
        max_output_bytes=1,
        url_expiry_seconds=1.0,
    )
    assert captured_manifest.calls["input_artifacts"] == []


def _state(**kw):
    base = dict(error_message=None, completed=False, started=False, artifact_state=None)
    base.update(kw)
    return build_service.derive_build_state(**base)


def test_derive_build_state_precedence():
    # Nothing happened yet.
    assert _state() == "pending"
    # Started but not done.
    assert _state(started=True) == "building"
    # Completed stream, or a ready artifact.
    assert _state(completed=True) == "ready"
    assert _state(artifact_state="ready") == "ready"
    # Failure wins over everything, incl. a ready artifact / completed stream.
    assert _state(error_message="boom", artifact_state="ready") == "failed"
    assert _state(artifact_state="failed", completed=True) == "failed"
    # error_message takes precedence over a started/completed stream too.
    assert _state(error_message="boom", started=True, completed=True) == "failed"
