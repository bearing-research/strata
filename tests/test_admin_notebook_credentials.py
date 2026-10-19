"""Amber item 88: a host replaces a running server's named credentials and catalogs."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from strata.config import StrataConfig
from strata.iceberg.catalog import PyIcebergCatalog

ROUTE = "/v1/admin/notebook-credentials"
MOUNT_CELL = "# @mount raw s3://lab/raw ro credential=lab\nx = 1"


def _headers(scopes: str | None) -> dict[str, str]:
    headers = {
        "X-Strata-Proxy-Token": "test-token",
        "X-Strata-Principal": "host",
        "X-Tenant-ID": "team-a",
    }
    if scopes:
        headers["X-Strata-Scopes"] = scopes
    return headers


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A service server's state with a real config and planner catalog, and fsspec stopped
    before any network, recording the options each mount would have been read with."""
    import fsspec

    import strata.server as server_module
    from strata.server import app

    config = StrataConfig(
        deployment_mode="service",
        auth_mode="trusted_proxy",
        proxy_token="test-token",
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
        notebook_storage_dir=tmp_path,
        notebook_credentials={"old": {"key": "old-key"}},
    )
    state = MagicMock()
    state.config = config
    state.planner = SimpleNamespace(catalog=PyIcebergCatalog(config))
    monkeypatch.setattr(server_module, "_state", state)

    fetched: list[dict] = []

    def _filesystem(protocol, **options):
        fetched.append(options)
        raise RuntimeError("stop before any network")

    monkeypatch.setattr(fsspec, "filesystem", _filesystem)
    return SimpleNamespace(
        client=TestClient(app, headers=_headers("admin:notebook-credentials")),
        config=config,
        state=state,
        fetched=fetched,
    )


def _session(parent, source: str):
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    parent.mkdir(parents=True, exist_ok=True)
    nb = create_notebook(parent, "nb")
    add_cell_to_notebook(nb, "c1", None)
    write_cell(nb, "c1", source)
    return NotebookSession(parse_notebook(nb), nb)


async def test_a_session_opened_before_the_put_uses_what_it_set_from_the_next_run(tmp_path, server):
    """The upstream "done when": a new name mounts, a removed name fails naming it, and a
    ``${file:...}`` value rewritten on disk is used at the next run."""
    from strata.notebook.executor import CellExecutor

    session = _session(tmp_path / "nb", MOUNT_CELL)
    secret = tmp_path / "lab-key"

    before = await CellExecutor(session).execute_cell("c1", MOUNT_CELL)
    assert "credential 'lab' is not defined" in (before.error or "")

    reply = server.client.put(
        ROUTE, json={"notebook_credentials": {"lab": {"key": f"${{file:{secret}}}"}}}
    )
    assert reply.status_code == 200, reply.text
    keys = []
    for value in ("key-1", "key-2"):
        secret.write_text(f"{value}\n")
        server.fetched.clear()
        await CellExecutor(session).execute_cell("c1", MOUNT_CELL)
        keys.append({options["key"] for options in server.fetched})
    assert keys == [{"key-1"}, {"key-2"}]

    assert server.client.put(ROUTE, json={}).status_code == 200
    server.fetched.clear()
    removed = await CellExecutor(session).execute_cell("c1", MOUNT_CELL)
    assert removed.success is False
    assert "credential 'lab' is not defined" in (removed.error or "")
    assert server.fetched == []


def test_the_five_settings_are_replaced_together(tmp_path, server):
    lab = str(tmp_path / "lab")
    reply = server.client.put(
        ROUTE,
        json={
            "notebook_credentials": {"lab": {"key": "${LAB_KEY}"}},
            "notebook_mount_credentials": {"s3": "lab"},
            "notebook_credential_paths": {"lab": [lab]},
            "notebook_credential_writers": {"lab": ["user_*"]},
            "catalogs": {"lake": {"type": "rest", "uri": "https://catalog.example"}},
        },
    )
    assert reply.status_code == 200, reply.text
    assert server.config.notebook_credentials == {"lab": {"key": "${LAB_KEY}"}}
    assert server.config.notebook_mount_credentials == {"s3": "lab"}
    assert server.config.notebook_credential_paths == {"lab": [lab]}
    assert server.config.notebook_credential_writers == {"lab": ["user_*"]}
    assert server.config.catalogs == {"lake": {"type": "rest", "uri": "https://catalog.example"}}

    # Left out is replaced too; writers go back to unset, where every name writes.
    assert server.client.put(ROUTE, json={"catalogs": {}}).status_code == 200
    assert server.config.notebook_credentials == {}
    assert server.config.notebook_mount_credentials == {}
    assert server.config.notebook_credential_writers is None


def test_reading_them_back_shows_references_and_hides_literal_secrets(server):
    server.client.put(
        ROUTE,
        json={
            "notebook_credentials": {
                "lab": {"key": "${file:/run/creds/lab}", "secret": "literal-secret"}
            },
            "catalogs": {
                "lake": {
                    "type": "sql",
                    "uri": "postgresql://u:pw@db/iceberg",
                    "warehouse": "s3://lake/wh",
                    "s3.access-key-id": "${LAKE_KEY}",
                }
            },
        },
    )

    shown = server.client.get(ROUTE).json()

    assert shown["notebook_credentials"] == {
        "lab": {"key": "${file:/run/creds/lab}", "secret": "<redacted>"}
    }
    assert shown["catalogs"] == {
        "lake": {
            "type": "sql",
            "uri": "<redacted>",
            "warehouse": "s3://lake/wh",
            "s3.access-key-id": "${LAKE_KEY}",
        }
    }
    assert "literal-secret" not in str(shown) and "pw@" not in str(shown)


@pytest.mark.parametrize(
    ("body", "said"),
    [
        ({"notebook_credential_paths": {"lab": ["relative/dir"]}}, "is not an absolute path"),
        (
            {
                "notebook_credentials": {"lab": {"k": "v"}},
                "catalogs": {"lake": {"type": "rest", "credential": "lab"}},
            },
            "set strata_credential instead",
        ),
        ({"catalogs": {"lake": {"type": "sql", "warehouse": "s3://lake/wh"}}}, "no catalog uri"),
    ],
)
def test_a_refused_replacement_changes_nothing(server, body, said):
    reply = server.client.put(ROUTE, json=body)

    assert reply.status_code == 400
    assert said in reply.json()["detail"]
    assert server.config.notebook_credentials == {"old": {"key": "old-key"}}
    assert server.config.catalogs == {}


def test_a_changed_catalog_is_rebuilt_from_its_new_properties(tmp_path, server):
    def lake(uri_path) -> dict:
        return {
            "lake": {
                "type": "sql",
                "uri": f"sqlite:///{uri_path}",
                "warehouse": f"file://{tmp_path / 'wh'}",
            }
        }

    server.client.put(ROUTE, json={"catalogs": lake(tmp_path / "a.db")})
    first = server.state.planner.catalog._get_named_catalog("lake")
    server.client.put(ROUTE, json={"catalogs": lake(tmp_path / "b.db")})
    second = server.state.planner.catalog._get_named_catalog("lake")

    assert first.properties["uri"].endswith("a.db")
    assert second.properties["uri"].endswith("b.db")


def test_the_routes_need_their_scope(server):
    from strata.server import app

    other = TestClient(app, headers=_headers("admin:notebook-workers"))

    assert other.get(ROUTE).status_code == 403
    assert other.put(ROUTE, json={}).status_code == 403
    assert server.config.notebook_credentials == {"old": {"key": "old-key"}}
