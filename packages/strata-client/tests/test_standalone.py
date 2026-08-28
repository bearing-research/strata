"""Standalone tests for the ``strata-client`` distribution.

These run with ONLY the client's own dependencies installed (httpx + pyarrow) —
no ``strata`` / server stack. The client-only CI job runs this file in such an
environment to guard the slim-install promise. They also pass in the full dev
env. See docs/internal/design-strata-client.md.
"""

from __future__ import annotations

import importlib.util
import sys

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from strata_client import Filter, FilterOp, RetryConfig, StrataClient, gt


@pytest.mark.skipif(
    importlib.util.find_spec("strata") is not None,
    reason=(
        "the server (strata) is installed — the no-server-deps guard only applies in "
        "the isolated client-only environment (the CI 'strata-client (no server deps)' "
        "job), where other tests haven't already imported the server"
    ),
)
def test_no_server_deps_imported() -> None:
    """Importing the client must not pull the server's heavy stack."""
    for heavy in ("strata", "pyiceberg", "fastapi", "uvicorn", "duckdb", "pydantic"):
        assert heavy not in sys.modules, f"strata_client pulled {heavy}"


def test_default_url_resolution(monkeypatch) -> None:
    monkeypatch.delenv("STRATA_SERVER_URL", raising=False)
    monkeypatch.delenv("STRATA_HOST", raising=False)
    monkeypatch.delenv("STRATA_PORT", raising=False)
    client = StrataClient()
    assert client.config is None
    assert client.base_url == "http://127.0.0.1:8765"
    client.close()


def test_server_url_env_override(monkeypatch) -> None:
    monkeypatch.setenv("STRATA_SERVER_URL", "https://strata.example.com")
    client = StrataClient()
    assert client.base_url == "https://strata.example.com"
    client.close()


def test_filter_constructors() -> None:
    f = gt("amount", 100)
    assert isinstance(f, Filter)
    assert f.column == "amount"
    assert f.op is FilterOp.GT
    assert f.value == 100


def test_fetch_over_mock_transport() -> None:
    """Arrow IPC fetch decodes correctly against a mocked server (no network).

    The full materialize protocol is covered in test_client_unit.py; this just
    confirms the standalone client can do an Arrow round-trip end to end.
    """
    table = pa.table({"x": [1, 2, 3]})
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    ipc_bytes = sink.getvalue().to_pybytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/data"):
            return httpx.Response(
                200,
                content=ipc_bytes,
                headers={"content-type": "application/vnd.apache.arrow.stream"},
            )
        # Artifact status probe — report ready so fetch proceeds to /data.
        return httpx.Response(200, json={"artifact_id": "abc123", "version": 1, "state": "ready"})

    client = StrataClient.from_transport(httpx.MockTransport(handler))
    fetched = client.fetch("strata://artifact/abc123@v=1")
    assert fetched.to_pydict() == {"x": [1, 2, 3]}
    client.close()


def test_retry_config_backoff() -> None:
    rc = RetryConfig(max_retries=2, base_delay=1.0, max_delay=30.0, jitter=0.0)
    assert rc.calculate_delay(0) == 1.0
    assert rc.calculate_delay(1) == 2.0
    assert rc.calculate_delay(10) == 30.0  # capped


class TestSnapshotPinReachesTheWire:
    """``snapshot_id`` is the reproducibility control: it pins a scan to one
    Iceberg snapshot.

    The arrow and datafusion integrations accepted it, stored it, exposed it
    as a property and documented it as "Pin to specific snapshot" — but their
    ``_build_scan_transform`` did not take the parameter, so it never reached
    the request. A pinned read silently returned the *current* snapshot, and
    because the provenance hash is computed from what was actually sent, it
    recorded the current snapshot too. Nothing downstream could flag the
    divergence. ``duckdb``/``polars``/``pandas`` always did this correctly.
    """

    def test_arrow_puts_the_snapshot_in_the_transform(self) -> None:
        pytest.importorskip("pyarrow")
        from strata_client.integration.arrow import _build_scan_transform

        params = _build_scan_transform(["id"], None, 12345)["params"]
        assert params["snapshot_id"] == 12345

    def test_datafusion_puts_the_snapshot_in_the_transform(self) -> None:
        pytest.importorskip("pyarrow")
        from strata_client.integration.datafusion import _build_scan_transform

        params = _build_scan_transform(["id"], None, 12345)["params"]
        assert params["snapshot_id"] == 12345

    def test_an_unpinned_scan_sends_no_snapshot(self) -> None:
        """Omitting the key is what makes the server read the current snapshot."""
        pytest.importorskip("pyarrow")
        from strata_client.integration.arrow import _build_scan_transform as arrow_build
        from strata_client.integration.datafusion import _build_scan_transform as df_build

        assert "snapshot_id" not in arrow_build(["id"])["params"]
        assert "snapshot_id" not in df_build(["id"])["params"]

    def test_it_matches_the_integrations_that_were_already_correct(self) -> None:
        # This one genuinely needs duckdb: it compares against that builder.
        pytest.importorskip("duckdb")
        from strata_client.integration.arrow import _build_scan_transform as arrow_build
        from strata_client.integration.duckdb import _build_scan_transform as duckdb_build

        assert (
            arrow_build(["id"], None, 999)["params"]["snapshot_id"]
            == duckdb_build(["id"], None, 999)["params"]["snapshot_id"]
        )


class TestOptionalExtrasAreIndependentlyUsable:
    """The package declares four separate extras — duckdb, pandas, polars,
    datafusion — but ``integration/__init__`` imported all five integrations
    eagerly, and importing any submodule runs that file first.

    So ``pip install "strata-client[pandas]"`` then ``from
    strata_client.integration.pandas import scan_to_pandas`` raised
    ``ModuleNotFoundError: No module named 'duckdb'``. Only ``[all]`` worked,
    which made the separate extras misleading.
    """

    def _import_with_blocked(self, blocked: str, statement: str) -> str:
        import subprocess
        import sys
        import textwrap

        code = textwrap.dedent(f"""
            import sys
            class Blocker:
                def find_spec(self, name, path=None, target=None):
                    if name == {blocked!r}:
                        raise ImportError("No module named {blocked!r}")
                    return None
            sys.meta_path.insert(0, Blocker())
            {statement}
            print("OK")
        """)
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd="packages/strata-client/src",
            capture_output=True,
            text=True,
        )
        return proc.stdout.strip() or proc.stderr.strip()

    def test_pandas_integration_imports_without_duckdb(self):
        out = self._import_with_blocked(
            "duckdb", "from strata_client.integration.pandas import scan_to_pandas"
        )
        assert out == "OK", out

    def test_arrow_integration_imports_without_duckdb(self):
        out = self._import_with_blocked(
            "duckdb", "from strata_client.integration.arrow import StrataDataset"
        )
        assert out == "OK", out

    def test_the_re_exports_still_resolve(self):
        pytest.importorskip("duckdb")
        pytest.importorskip("pandas")
        from strata_client.integration import StrataDataset, scan_to_pandas, strata_query

        assert all(x is not None for x in (StrataDataset, scan_to_pandas, strata_query))

    def test_an_unknown_name_still_raises_attribute_error(self):
        import strata_client.integration as integration

        with pytest.raises(AttributeError):
            integration.no_such_export


class TestJsonArtifactRoundTrip:
    """``put_json`` and ``get_json`` disagreed about the encoding.

    ``_dict_to_ipc`` stores a dict *columnar* when every value is an
    equal-length list. ``get_json`` used a different discriminator — "one
    column named ``data`` means it's a JSON blob" — and a dict whose only key
    is ``data`` satisfies both. So the columnar write was read back through
    the blob branch: three strings in, the integer ``1`` out, no error. With
    non-numeric strings it raised ``JSONDecodeError`` from a call that has no
    documented failure mode.

    The encoding is now marked in the schema metadata rather than guessed.
    """

    def _roundtrip(self, payload):
        import json

        from strata_client.client import _dict_to_ipc, _is_json_blob

        table = ipc.open_stream(pa.py_buffer(_dict_to_ipc(payload))).read_all()
        if _is_json_blob(table):
            return json.loads(table.column("data")[0].as_py())
        return table.to_pydict()

    @pytest.mark.parametrize(
        "payload",
        [
            {"data": ["1", "2", "3"]},  # the collision, numeric-looking
            {"data": ["a", "b"]},  # the collision, used to raise
            {"data": ["only"]},  # single row: ambiguous with the blob shape
            {"a": [1, 2], "b": [3, 4]},  # ordinary columnar
            {"nested": {"x": 1}, "n": 5},  # a genuine JSON document
        ],
    )
    def test_it_round_trips(self, payload):
        assert self._roundtrip(payload) == payload

    def test_a_legacy_blob_without_the_marker_still_decodes(self):
        """Artifacts written before the marker existed must keep working."""
        import json

        from strata_client.client import _is_json_blob

        legacy = pa.Table.from_pydict({"data": [json.dumps({"model": "v2"})]})
        assert _is_json_blob(legacy)
        assert json.loads(legacy.column("data")[0].as_py()) == {"model": "v2"}

    def test_a_legacy_columnar_table_is_not_mistaken_for_a_document(self):
        from strata_client.client import _is_json_blob

        assert not _is_json_blob(pa.Table.from_pydict({"a": [1, 2], "b": [3, 4]}))

    def test_a_scanned_table_with_a_data_column_is_not_a_document(self):
        """A scan result that happens to have one column called ``data`` is
        columnar, and multiple rows are what distinguish it from a blob."""
        from strata_client.client import _is_json_blob

        assert not _is_json_blob(pa.Table.from_pydict({"data": [1, 2, 3]}))
