"""Unit tests for the path-loaded notebook ``strata`` client shim.

The shim runs in the notebook venv (pyarrow + stdlib only) and must NOT
import strata — it's loaded by file path exactly like the harness loads it.
These tests cover the offline pieces (Arrow IPC, URI parsing, the Artifact
wrapper, surface). The HTTP wire protocol is validated live against a real
server elsewhere.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest


def _load_shim():
    """Load notebook_client.py by path — the way the harness does, so the
    test exercises the real (strata-free) load mechanism."""
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
