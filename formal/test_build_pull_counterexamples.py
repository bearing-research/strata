"""Replay BuildLease.tla's pull-path counterexample against the HTTP routes.

Passes while the bug exists (asserts the violating outcome).

    uv run pytest formal/ -v
"""

# Fixtures are imported from tests/ and then requested by name.
# ruff: noqa: F811

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pyarrow as pa

from tests.test_pull_model import (  # noqa: F401  (fixtures)
    artifact_store,
    build_store,
    client,
    config,
    create_test_artifact,
    temp_dir,
)


def _ipc(values):
    sink = pa.BufferOutputStream()
    table = pa.table({"x": values})
    with pa.ipc.new_stream(sink, table.schema) as w:
        w.write_table(table)
    return sink.getvalue().to_pybytes()


def _upload(client, manifest, values):
    params = parse_qs(urlparse(manifest["output"]["url"]).query)
    return client.post(
        "/v1/artifacts/upload",
        params={k: v[0] for k, v in params.items()},
        content=_ipc(values),
    )


def test_retired_manifest_upload_is_published_under_the_current_claim(
    client, build_store, artifact_store
):
    """Build_PullPath / NoStaleBytesPublished.

    Re-fetching a manifest retires the previous *finalize* URL (its lease
    token no longer matches), which is what test_pull_model's
    ``test_refetching_a_manifest_retires_the_previous_capability`` checks.
    The previous *upload* URL carries no lease token, though, and stays
    valid. Executor 1's late upload lands in the same (artifact_id, version)
    slot, and executor 2's legitimate finalize publishes executor 1's bytes.
    """
    version = create_test_artifact(artifact_store, "out-pull", finalize=False)
    build_store.create_build(
        build_id="pull-1",
        artifact_id="out-pull",
        version=version,
        executor_ref="duckdb_sql@v1",
        input_uris=[],
        params={},
    )
    first = client.get("/v1/builds/pull-1/manifest").json()
    second = client.get("/v1/builds/pull-1/manifest").json()

    assert _upload(client, second, [2]).status_code == 200
    assert _upload(client, first, [1]).status_code == 200  # retired claim, still accepted

    finalize = client.post(second["finalize_url"].replace("http://testserver", ""))
    assert finalize.status_code == 200

    data = artifact_store.read_blob("out-pull", version)
    assert pa.ipc.open_stream(data).read_all().column("x").to_pylist() == [1]
