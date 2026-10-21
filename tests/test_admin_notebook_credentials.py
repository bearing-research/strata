"""Amber item 88: a host replaces a running server's named credentials and catalogs."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from strata.config import StrataConfig
from strata.iceberg.catalog import PyIcebergCatalog
from tests.notebook import test_mounts
from tests.notebook.test_read_confirm import _asked, windows

ROUTE = "/v1/admin/notebook-credentials"
MOUNT_CELL = "# @mount raw s3://lab/raw ro credential=lab\nx = 1"
# The two restrictions a body must state, here as unrestricted.
OPEN = {"notebook_credential_paths": {}, "notebook_credential_writers": None}


def _body(**settings) -> dict:
    return {**OPEN, **settings}


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
    """The upstream "done when": a new name mounts, a ``${file:...}`` value under a file
    root rewritten on disk is used at the next run with no second call, and a removed
    name fails naming it."""
    from strata.notebook.executor import CellExecutor

    session = _session(tmp_path / "nb", MOUNT_CELL)
    root = tmp_path / "creds"
    root.mkdir()
    server.config.notebook_credential_file_roots = [root]
    secret = root / "lab-key"

    before = await CellExecutor(session).execute_cell("c1", MOUNT_CELL)
    assert "credential 'lab' is not defined" in (before.error or "")

    reply = server.client.put(
        ROUTE, json=_body(notebook_credentials={"lab": {"key": f"${{file:{secret}}}"}})
    )
    assert reply.status_code == 200, reply.text
    keys = []
    for value in ("key-1", "key-2"):
        secret.write_text(f"{value}\n")
        server.fetched.clear()
        await CellExecutor(session).execute_cell("c1", MOUNT_CELL)
        keys.append({options["key"] for options in server.fetched})
    assert keys == [{"key-1"}, {"key-2"}]

    assert server.client.put(ROUTE, json=OPEN).status_code == 200
    server.fetched.clear()
    removed = await CellExecutor(session).execute_cell("c1", MOUNT_CELL)
    assert removed.success is False
    assert "credential 'lab' is not defined" in (removed.error or "")
    assert server.fetched == []


def test_the_six_settings_are_replaced_together(tmp_path, server):
    lab = str(tmp_path / "lab")
    reply = server.client.put(
        ROUTE,
        json={
            "notebook_credentials": {"lab": {"key": "lab-key"}},
            "notebook_mount_credentials": {"s3": "lab"},
            "notebook_credential_paths": {"lab": [lab]},
            "notebook_credential_writers": {"lab": ["user_*"]},
            "notebook_credential_read_cost": {"lab": 0.09},
            "catalogs": {"lake": {"type": "rest", "uri": "https://catalog.example"}},
        },
    )
    assert reply.status_code == 200, reply.text
    assert server.config.notebook_credentials == {"lab": {"key": "lab-key"}}
    assert server.config.notebook_mount_credentials == {"s3": "lab"}
    assert server.config.notebook_credential_paths == {"lab": [lab]}
    assert server.config.notebook_credential_writers == {"lab": ["user_*"]}
    assert server.config.notebook_credential_read_cost == {"lab": 0.09}
    assert server.config.catalogs == {"lake": {"type": "rest", "uri": "https://catalog.example"}}

    # Left out is replaced too; writers go back to unset, where every name writes.
    assert server.client.put(ROUTE, json=_body(catalogs={})).status_code == 200
    assert server.config.notebook_credentials == {}
    assert server.config.notebook_mount_credentials == {}
    assert server.config.notebook_credential_writers is None
    assert server.config.notebook_credential_read_cost == {}


def test_read_costs_read_back_as_set(server):
    """A rate is not a secret: it is shown as it is used."""
    reply = server.client.put(ROUTE, json=_body(notebook_credential_read_cost={"lab": 2}))
    assert reply.status_code == 200, reply.text

    assert reply.json()["notebook_credential_read_cost"] == {"lab": 2.0}
    assert server.client.get(ROUTE).json()["notebook_credential_read_cost"] == {"lab": 2.0}


@pytest.mark.parametrize("rate", ["cheap", "${LAB_RATE}", [0.09], None])
def test_a_rate_that_is_not_a_number_is_refused_and_changes_nothing(server, rate):
    """Refused as startup refuses it, naming the field; a rate is no reference either."""
    server.config.notebook_credential_read_cost = {"old": 0.02}

    reply = server.client.put(
        ROUTE,
        json=_body(
            notebook_credentials={"lab": {"key": "k"}},
            notebook_credential_read_cost={"lab": rate},
        ),
    )

    assert reply.status_code == 422
    assert "notebook_credential_read_cost" in reply.text
    with pytest.raises(ValueError, match="notebook_credential_read_cost"):
        StrataConfig(notebook_credential_read_cost={"lab": rate})
    assert server.config.notebook_credential_read_cost == {"old": 0.02}
    assert server.config.notebook_credentials == {"old": {"key": "old-key"}}


@windows
async def test_a_new_names_rate_makes_a_costly_run_ask_without_a_restart(
    tmp_path, server, monkeypatch
):
    """Amber item 97's done-when, at item 90's level: a name connected through the route
    brings its rate, and a run reading 250 bytes through it at $10^7 per GB ($2.50) asks
    before going past $1. Read from the server's live config, not a copy taken at start."""
    # Imported by module: a Test class imported by name is collected here again.
    files = test_mounts.TestSharedMountCache._files(
        "bucket/data", {"a": b"aa" * 50, "b": b"bbb" * 50}
    )
    test_mounts._install_fake_fsspec(monkeypatch, test_mounts._FakeRemoteFS(files))
    server.config.notebook_mount_cache_dir = tmp_path / "mounts"
    server.config.notebook_confirm_read_cost = 1.0
    session = _session(tmp_path / "nb", "# @mount data s3://bucket/data ro credential=lab\nx = 1")
    lab = {"lab": {"key": "lab-key"}}

    asked = []
    for rates in ({}, {"lab": 10**7}, {}):
        reply = server.client.put(
            ROUTE, json=_body(notebook_credentials=lab, notebook_credential_read_cost=rates)
        )
        assert reply.status_code == 200, reply.text
        asked.append(await _asked(session))

    assert asked == [[], ["c1"], []]


def _put_secret(server, value: str):
    return server.client.put(
        ROUTE, json=_body(notebook_credentials={"lab": {"key": "literal", "secret": value}})
    )


def test_a_variable_reference_is_refused_naming_the_field(tmp_path, server):
    """It would have the server read its environment (the proxy token, say) into a value
    the caller decides how to use, roots or not."""
    server.config.notebook_credential_file_roots = [tmp_path]

    reply = _put_secret(server, "${STRATA_PROXY_TOKEN}")

    assert reply.status_code == 400
    assert "notebook_credentials['lab']['secret'] is a ${VAR} reference" in reply.json()["detail"]
    assert server.config.notebook_credentials == {"old": {"key": "old-key"}}


def test_a_file_reference_with_no_roots_is_refused(tmp_path, server):
    reply = _put_secret(server, f"${{file:{tmp_path / 'secret'}}}")

    assert reply.status_code == 422
    detail = reply.json()["detail"]
    assert str(tmp_path / "secret") in detail and "notebook_credential_file_roots" in detail
    assert "which is empty" in detail
    assert server.config.notebook_credentials == {"old": {"key": "old-key"}}


@pytest.mark.parametrize("where", ["outside", "link-out", "dotdot"])
def test_a_file_reference_outside_the_roots_is_refused(tmp_path, server, where):
    """Judged by where the path resolves: a link inside the root to a file outside, or a
    ``..`` climbing out, is outside."""
    root = tmp_path / "creds"
    root.mkdir()
    outside = tmp_path / "server-secret"
    outside.write_text("server-only")
    (root / "link").symlink_to(outside)
    server.config.notebook_credential_file_roots = [root]
    path = {
        "outside": outside,
        "link-out": root / "link",
        "dotdot": root / ".." / "server-secret",
    }[where]

    reply = _put_secret(server, f"${{file:{path}}}")

    assert reply.status_code == 422
    assert f"{path} is not inside notebook_credential_file_roots" in reply.json()["detail"]
    assert server.config.notebook_credentials == {"old": {"key": "old-key"}}


async def test_a_file_swapped_for_a_link_out_of_the_root_is_not_read(tmp_path, server):
    """The PUT's check is not enough: the file can become a link after it."""
    from strata.notebook.executor import CellExecutor

    root = tmp_path / "creds"
    root.mkdir()
    secret = root / "lab-key"
    secret.write_text("key-1")
    outside = tmp_path / "server-secret"
    outside.write_text("server-only")
    server.config.notebook_credential_file_roots = [root]
    reply = server.client.put(
        ROUTE, json=_body(notebook_credentials={"lab": {"key": f"${{file:{secret}}}"}})
    )
    assert reply.status_code == 200, reply.text
    secret.unlink()
    secret.symlink_to(outside)

    session = _session(tmp_path / "nb", MOUNT_CELL)
    result = await CellExecutor(session).execute_cell("c1", MOUNT_CELL)

    assert result.success is False
    assert "is not inside notebook_credential_file_roots" in (result.error or "")
    assert server.fetched == []


@pytest.mark.parametrize("left_out", ["notebook_credential_paths", "notebook_credential_writers"])
def test_a_body_that_leaves_out_a_restriction_is_refused_and_keeps_it(tmp_path, server, left_out):
    """Left out would have reset it, lifting the restriction without anyone saying so."""
    restricted = {
        "notebook_credentials": {"lab": {"key": "k"}},
        "notebook_credential_paths": {"lab": [str(tmp_path / "lab")]},
        "notebook_credential_writers": {"lab": ["user_*"]},
    }
    assert server.client.put(ROUTE, json=restricted).status_code == 200
    body = {key: value for key, value in restricted.items() if key != left_out}

    reply = server.client.put(ROUTE, json=body)

    assert reply.status_code == 422
    assert left_out in reply.text
    assert server.config.notebook_credential_paths == {"lab": [str(tmp_path / "lab")]}
    assert server.config.notebook_credential_writers == {"lab": ["user_*"]}


def test_reading_them_back_shows_references_and_hides_literal_secrets(server):
    reply = server.client.put(
        ROUTE,
        json=_body(
            notebook_credentials={"lab": {"secret": "literal-secret"}},
            catalogs={
                "lake": {
                    "type": "sql",
                    "uri": "postgresql://u:pw@db/iceberg",
                    "warehouse": "s3://lake/wh",
                    "strata_credential": "lab",
                    "s3.access-key-id": "${LAKE_KEY}",
                    "adls.connection-string": "AccountKey=conn-secret",
                    "header.X-Api-Key": "header-secret",
                }
            },
        ),
    )
    assert reply.status_code == 200, reply.text
    # A reference, as the operator's startup config would hold it.
    server.config.notebook_credentials["lab"]["key"] = "${file:/run/creds/lab}"

    shown = server.client.get(ROUTE).json()

    assert shown["notebook_credentials"] == {
        "lab": {"key": "${file:/run/creds/lab}", "secret": "<redacted>"}
    }
    # Only keys known to hold no secret are shown; a catalog resolves no reference.
    assert shown["catalogs"] == {
        "lake": {
            "type": "sql",
            "uri": "<redacted>",
            "warehouse": "s3://lake/wh",
            "strata_credential": "lab",
            "s3.access-key-id": "<redacted>",
            "adls.connection-string": "<redacted>",
            "header.X-Api-Key": "<redacted>",
        }
    }
    for secret in ("literal-secret", "pw@", "conn-secret", "header-secret"):
        assert secret not in str(shown)


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
    reply = server.client.put(ROUTE, json=_body(**body))

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

    server.client.put(ROUTE, json=_body(catalogs=lake(tmp_path / "a.db")))
    first = server.state.planner.catalog._get_named_catalog("lake")
    server.client.put(ROUTE, json=_body(catalogs=lake(tmp_path / "b.db")))
    second = server.state.planner.catalog._get_named_catalog("lake")

    assert first.properties["uri"].endswith("a.db")
    assert second.properties["uri"].endswith("b.db")


def test_the_routes_need_their_scope(server):
    from strata.server import app

    other = TestClient(app, headers=_headers("admin:notebook-workers"))

    assert other.get(ROUTE).status_code == 403
    assert other.put(ROUTE, json={}).status_code == 403
    assert server.config.notebook_credentials == {"old": {"key": "old-key"}}
