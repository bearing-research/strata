"""Unit tests for the path-loaded notebook ``strata`` client shim.

The shim runs in the notebook venv (pyarrow + stdlib only) and must not import
strata. Only offline pieces are covered here; the wire protocol is tested live.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest


def _load_shim():
    """Load notebook_client.py by path, as the harness does, so strata is never imported."""
    path = Path("src/strata/notebook/notebook_client.py")
    spec = importlib.util.spec_from_file_location("_nb_client_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def shim():
    return _load_shim()


def test_shim_does_not_import_strata():
    source = Path("src/strata/notebook/notebook_client.py").read_text()
    # No executable strata import (the word appears only in the docstring).
    for line in source.splitlines():
        stripped = line.strip()
        assert not stripped.startswith("import strata")
        assert not stripped.startswith("from strata")


def test_client_surface(shim):
    c = shim.StrataClient(base_url="http://127.0.0.1:8765/")
    assert c.base_url == "http://127.0.0.1:8765"  # trailing slash stripped
    for method in (
        "materialize",
        "put",
        "set_alias",
        "resolve_alias",
        "set_tag",
        "get_tags",
        "resolve_name",
        "set_name",
        "get_registry_audit",
        "promote",
        "close",
    ):
        assert callable(getattr(c, method))
    assert c.close() is None  # no-op, present for drop-in compatibility


def test_parse_artifact_uri(shim):
    assert shim._parse_artifact_uri("strata://artifact/abc-123@v=7") == ("abc-123", 7)


def test_convert_to_arrow_ipc_roundtrip(shim):
    c = shim.StrataClient(base_url="http://x")

    table = pa.table({"x": [1, 2, 3]})
    for data in (table, pd.DataFrame({"x": [1, 2, 3]}), {"x": [1, 2, 3]}):
        ipc_bytes = shim._convert_to_arrow_ipc(data)
        assert isinstance(ipc_bytes, bytes) and ipc_bytes
        art = shim.Artifact(c, "id", 1, stream_data=ipc_bytes)
        assert art.to_arrow().num_rows == 3
        assert art.to_pandas().shape[0] == 3

    # bytes pass through unchanged
    assert shim._convert_to_arrow_ipc(b"raw") == b"raw"

    with pytest.raises(TypeError):
        shim._convert_to_arrow_ipc(object())


def test_artifact_uri_and_empty_stream(shim):
    c = shim.StrataClient(base_url="http://x")
    art = shim.Artifact(c, "id-9", 3, cache_hit=True)
    assert art.uri == "strata://artifact/id-9@v=3"
    assert art.cache_hit is True
    # Empty stream data yields an empty table, not a crash.
    assert shim.Artifact(c, "id", 1, stream_data=b"").to_arrow().num_rows == 0


def test_client_accepts_cell_id(shim):
    c = shim.StrataClient(base_url="http://x", cell_id="cell-42")
    assert c._cell_id == "cell-42"


def test_materialize_rejects_non_stream_mode(shim):
    """mode='artifact' is refused before any request; the slim client cannot poll a build."""
    c = shim.StrataClient(base_url="http://x")
    with pytest.raises(ValueError, match="mode='stream'"):
        c.materialize([], {"executor": "scan@v1", "params": {}}, mode="artifact")


def test_stamp_cell_is_noop_without_cell_or_name(shim):
    # No cell_id, or no name → no stamp attempt, never raises.
    c = shim.StrataClient(base_url="http://x")  # no cell_id
    art = shim.Artifact(c, "id", 1)
    c._stamp_cell(art, name="team/model")  # cell_id missing → no-op
    c2 = shim.StrataClient(base_url="http://x", cell_id="c")
    c2._stamp_cell(art, name=None)  # unnamed → no-op
    # With both, set_tag is attempted over HTTP and the failure is swallowed.
    c2._stamp_cell(art, name="team/model")  # no server → swallowed, no raise


def test_get_bytes_drains_large_response_in_chunks(shim, monkeypatch):
    """``_get_bytes`` reads a streamed response in bounded chunks, intact.

    One large ``resp.read()`` of a live stream let the server's send buffer fill
    and abort the stream. The fake response rejects an all-at-once read.
    """
    import urllib.request

    payload = b"\xa5" * (5 * (1 << 20) + 123)  # >5 MiB, not a chunk multiple

    class _FakeResp:
        def __init__(self, data: bytes) -> None:
            self._buf = data
            self._pos = 0

        def read(self, size: int = -1) -> bytes:
            assert size is not None and size > 0, "client must read bounded chunks, not all-at-once"
            chunk = self._buf[self._pos : self._pos + size]
            self._pos += len(chunk)
            return chunk

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _FakeResp(payload))
    c = shim.StrataClient(base_url="http://test")
    assert c._get_bytes("/v1/streams/abc") == payload


def test_remote_store_headers_attached_to_requests(shim, monkeypatch):
    """With a remote store, auth headers go on every request: JSON, GET bytes and multipart put."""
    import urllib.request

    captured = []

    class _Resp:
        def __init__(self, body=b"{}"):
            self._body = body

        def read(self, size=-1):
            if size and size > 0:
                b, self._body = self._body[:size], self._body[size:]
                return b
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def _fake_urlopen(req, timeout=None):
        captured.append(req)
        return _Resp(b'{"ok": true}')

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    headers = {"X-Strata-Principal": "alice", "X-Tenant-ID": "team-a"}
    c = shim.StrataClient(base_url="http://remote-store", headers=headers)

    # JSON request
    c._request("GET", "/v1/names/team/x")
    # GET-bytes request
    c._get_bytes("/v1/artifacts/a/v/1/data")
    # multipart put
    c._put_multipart("/v1/artifacts", [("data", "d.arrow", "application/octet-stream", b"x")])

    assert len(captured) == 3
    for req in captured:
        assert req.get_header("X-strata-principal") == "alice"
        assert req.get_header("X-tenant-id") == "team-a"


def test_no_headers_by_default(shim, monkeypatch):
    import urllib.request

    captured = []

    class _Resp:
        def read(self, size=-1):
            return b"{}"

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(
        urllib.request, "urlopen", lambda req, timeout=None: captured.append(req) or _Resp()
    )
    c = shim.StrataClient(base_url="http://local")
    c._request("GET", "/v1/names/x")
    assert captured[0].get_header("X-strata-principal") is None


class TestPromote:
    """``strata.promote("rows", name=...)`` from inside a cell.

    It takes the variable name because researchers never see minted artifact ids.
    """

    def _client(self, shim, **overrides):
        kwargs = {
            "base_url": "http://store.example",
            "promote_url": "http://127.0.0.1:8765/v1/notebooks/sess-1",
            "inputs": {"rows": "strata://artifact/nb_rows@v=3"},
        }
        kwargs.update(overrides)
        return shim.StrataClient(**kwargs)

    def _captured(self, client, monkeypatch):
        seen = {}

        def _fake(method, path, body=None):
            seen.update(method=method, path=path, body=body)
            return {"name": body["name"]}

        monkeypatch.setattr(client, "_request", _fake)
        return seen

    def test_a_variable_resolves_to_the_artifact_behind_it(self, shim, monkeypatch):
        client = self._client(shim)
        seen = self._captured(client, monkeypatch)

        client.promote("rows", name="taxi/rows", alias="champion", tags={"stage": "candidate"})

        assert seen["method"] == "POST"
        assert seen["path"] == (
            "http://127.0.0.1:8765/v1/notebooks/sess-1/artifacts/nb_rows/v/3/promote"
        )
        assert seen["body"] == {
            "name": "taxi/rows",
            "tags": {"stage": "candidate"},
            "alias": "champion",
        }

    def test_an_explicit_reference_still_works(self, shim, monkeypatch):
        client = self._client(shim)
        seen = self._captured(client, monkeypatch)

        client.promote("other@v=7", name="taxi/other")

        assert seen["path"].endswith("/artifacts/other/v/7/promote")

    def test_no_team_store_is_answered_here_not_by_a_round_trip(self, shim):
        """Without a team store the client refuses locally, naming the setting, instead of a 409."""
        client = self._client(shim, promote_url=None)

        with pytest.raises(RuntimeError, match="notebook_remote_store_url"):
            client.promote("rows", name="taxi/rows")

    def test_a_name_that_is_not_an_input_says_what_is(self, shim):
        client = self._client(shim)

        with pytest.raises(RuntimeError, match="rows"):
            client.promote("typo", name="taxi/rows")
