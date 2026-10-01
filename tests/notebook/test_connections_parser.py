"""Tests for [connections.<name>] parsing and round-trip in notebook.toml."""

from __future__ import annotations

import tempfile
import tomllib
from pathlib import Path

import tomli_w

from strata.notebook.models import ConnectionSpec
from strata.notebook.parser import parse_notebook
from strata.notebook.writer import create_notebook, write_notebook_toml


def _write_raw_toml(notebook_dir: Path, body: dict) -> None:
    """Write a TOML body straight to notebook.toml.

    Used to test parser tolerance of hand-edited connection blocks
    (the surface users will actually edit in v1).
    """
    with open(notebook_dir / "notebook.toml", "wb") as f:
        tomli_w.dump(body, f)


def _read_raw_toml(notebook_dir: Path) -> dict:
    with open(notebook_dir / "notebook.toml", "rb") as f:
        return tomllib.load(f)


def test_parse_single_connection():
    with tempfile.TemporaryDirectory() as tmp:
        nb = create_notebook(Path(tmp), "with_conn")
        body = _read_raw_toml(nb)
        body["connections"] = {
            "warehouse": {
                "driver": "postgresql",
                "uri": "postgresql://localhost:5432/dev",
            }
        }
        _write_raw_toml(nb, body)

        state = parse_notebook(nb)
        assert len(state.connections) == 1
        conn = state.connections[0]
        assert conn.name == "warehouse"
        assert conn.driver == "postgresql"
        # Driver-specific keys are preserved via Pydantic extra="allow".
        assert getattr(conn, "uri") == "postgresql://localhost:5432/dev"


def test_parse_multiple_connections_preserves_extras():
    with tempfile.TemporaryDirectory() as tmp:
        nb = create_notebook(Path(tmp), "multi")
        body = _read_raw_toml(nb)
        body["connections"] = {
            "warehouse": {
                "driver": "snowflake",
                "account": "ACME-PROD",
                "warehouse": "ANALYTICS",
                "database": "EVENTS",
                "role": "READER",
                "auth": {
                    "user": "${SNOWFLAKE_USER}",
                    "password": "${SNOWFLAKE_PASSWORD}",
                },
            },
            "local_pg": {
                "driver": "postgresql",
                "uri": "postgresql://localhost:5432/dev",
                "options": {"application_name": "strata-notebook"},
            },
        }
        _write_raw_toml(nb, body)

        state = parse_notebook(nb)
        by_name = {c.name: c for c in state.connections}
        assert set(by_name) == {"warehouse", "local_pg"}

        wh = by_name["warehouse"]
        assert wh.driver == "snowflake"
        assert getattr(wh, "account") == "ACME-PROD"
        assert getattr(wh, "warehouse") == "ANALYTICS"
        assert wh.auth == {
            "user": "${SNOWFLAKE_USER}",
            "password": "${SNOWFLAKE_PASSWORD}",
        }

        pg = by_name["local_pg"]
        assert pg.driver == "postgresql"
        assert pg.options == {"application_name": "strata-notebook"}


def test_invalid_connection_name_is_dropped():
    """Connection names must be valid Python identifiers — invalid keys
    are silently dropped from the parsed state. The annotation_validation
    layer surfaces these as user-visible diagnostics; the parser stays
    permissive so notebooks open even when partially malformed."""
    with tempfile.TemporaryDirectory() as tmp:
        nb = create_notebook(Path(tmp), "bad_name")
        body = _read_raw_toml(nb)
        body["connections"] = {
            "valid_one": {"driver": "sqlite", "path": "/tmp/x.db"},
            "bad name with spaces": {"driver": "sqlite", "path": "/tmp/y.db"},
            "bad-hyphen": {"driver": "sqlite", "path": "/tmp/z.db"},
        }
        _write_raw_toml(nb, body)

        state = parse_notebook(nb)
        names = {c.name for c in state.connections}
        assert names == {"valid_one"}


def test_connection_missing_driver_is_dropped():
    """A [connections.foo] block without ``driver`` is malformed; skip
    rather than raise so the rest of the notebook still opens."""
    with tempfile.TemporaryDirectory() as tmp:
        nb = create_notebook(Path(tmp), "no_driver")
        body = _read_raw_toml(nb)
        body["connections"] = {
            "ok": {"driver": "sqlite", "path": "/tmp/a.db"},
            "missing_driver": {"path": "/tmp/b.db"},
        }
        _write_raw_toml(nb, body)

        state = parse_notebook(nb)
        names = {c.name for c in state.connections}
        assert names == {"ok"}


def test_no_connections_block_yields_empty_list():
    with tempfile.TemporaryDirectory() as tmp:
        nb = create_notebook(Path(tmp), "no_conns")
        state = parse_notebook(nb)
        assert state.connections == []


def test_writer_roundtrip_preserves_connections():
    """Writing a NotebookToml back out preserves the connections block.

    Without this, every save (cell add/remove/reorder, worker change,
    etc.) would silently drop user-defined connections.
    """
    with tempfile.TemporaryDirectory() as tmp:
        nb = create_notebook(Path(tmp), "roundtrip")
        body = _read_raw_toml(nb)
        body["connections"] = {
            "warehouse": {
                "driver": "postgresql",
                "uri": "postgresql://localhost:5432/dev",
                "options": {"application_name": "strata"},
            }
        }
        _write_raw_toml(nb, body)

        state = parse_notebook(nb)
        assert len(state.connections) == 1

        from strata.notebook.models import NotebookToml

        toml_obj = NotebookToml(
            notebook_id=state.id,
            name=state.name,
            cells=[],
            connections=state.connections,
        )
        write_notebook_toml(nb, toml_obj)

        body_after = _read_raw_toml(nb)
        assert "connections" in body_after
        assert body_after["connections"]["warehouse"]["driver"] == "postgresql"
        assert body_after["connections"]["warehouse"]["uri"] == "postgresql://localhost:5432/dev"
        assert body_after["connections"]["warehouse"]["options"] == {"application_name": "strata"}


def test_writer_elides_empty_auth_and_options():
    """A connection with no auth/options should write a tight block
    without empty placeholder dicts cluttering the committed file."""
    with tempfile.TemporaryDirectory() as tmp:
        nb = create_notebook(Path(tmp), "tight")
        from strata.notebook.models import NotebookToml

        toml_obj = NotebookToml(
            notebook_id="nb1",
            name="tight",
            cells=[],
            connections=[ConnectionSpec(name="db", driver="sqlite")],
        )
        write_notebook_toml(nb, toml_obj)
        body = _read_raw_toml(nb)
        assert body["connections"] == {"db": {"driver": "sqlite"}}


def test_writer_elides_connections_block_when_empty():
    with tempfile.TemporaryDirectory() as tmp:
        nb = create_notebook(Path(tmp), "no_conns")
        body = _read_raw_toml(nb)
        assert "connections" not in body


# --- malformed connection preservation ----------------------------------


def test_malformed_connection_block_is_preserved_across_writer_roundtrip():
    """Regression: a malformed [connections.<name>] block must survive
    an unrelated notebook rewrite (cell add, worker change, etc.).
    Without this, a transient typo gets silently erased on the next
    save and the user can't recover what they typed."""
    with tempfile.TemporaryDirectory() as tmp:
        nb = create_notebook(Path(tmp), "malformed")
        body = _read_raw_toml(nb)
        body["connections"] = {
            "good": {
                "driver": "postgresql",
                "uri": "postgresql://localhost:5432/dev",
            },
            "missing_driver": {
                "host": "localhost",
                "port": 5432,
            },
            "bad-name-with-hyphen": {
                "driver": "sqlite",
                "path": "/tmp/x.db",
            },
        }
        _write_raw_toml(nb, body)

        # Parse, then rewrite via the writer (simulates an unrelated
        # save like adding a cell).
        state = parse_notebook(nb)
        from strata.notebook.models import NotebookToml

        toml_obj = NotebookToml(
            notebook_id=state.id,
            name=state.name,
            cells=[],
            connections=state.connections,
            malformed_connections=state.malformed_connections,
        )
        write_notebook_toml(nb, toml_obj)

        body_after = _read_raw_toml(nb)
        assert set(body_after["connections"]) == {
            "good",
            "missing_driver",
            "bad-name-with-hyphen",
        }
        # The malformed bodies are preserved verbatim.
        assert body_after["connections"]["missing_driver"] == {
            "host": "localhost",
            "port": 5432,
        }
        assert body_after["connections"]["bad-name-with-hyphen"] == {
            "driver": "sqlite",
            "path": "/tmp/x.db",
        }


def test_malformed_connection_carries_error_for_diagnostics():
    """The parser tags each malformed block with a human-readable error
    so annotation_validation can surface a helpful diagnostic instead
    of just 'connection unknown'."""
    with tempfile.TemporaryDirectory() as tmp:
        nb = create_notebook(Path(tmp), "errors")
        body = _read_raw_toml(nb)
        body["connections"] = {
            "missing_driver": {"host": "localhost"},
            "bad-name": {"driver": "sqlite"},
        }
        _write_raw_toml(nb, body)

        state = parse_notebook(nb)
        by_name = {m.name: m for m in state.malformed_connections}
        assert "driver" in by_name["missing_driver"].error.lower()
        # bad-name fails Pydantic name-pattern validation
        assert by_name["bad-name"].error  # non-empty


# --- auth scrubbing -----------------------------------------------------


def test_writer_scrubs_literal_auth_values():
    """Regression: the writer must blank `auth.*` values that aren't
    `${VAR}` indirections so secrets never reach disk. The key is
    preserved so the notebook remembers WHICH credentials are
    configured."""
    with tempfile.TemporaryDirectory() as tmp:
        nb = create_notebook(Path(tmp), "scrub")
        from strata.notebook.models import NotebookToml

        toml_obj = NotebookToml(
            notebook_id="nb1",
            name="scrub",
            cells=[],
            connections=[
                ConnectionSpec(
                    name="db",
                    driver="postgresql",
                    auth={
                        "user": "${PGUSER}",
                        "password": "hunter2",  # literal: should be blanked
                    },
                )
            ],
        )
        write_notebook_toml(nb, toml_obj)

        body = _read_raw_toml(nb)
        # Indirection preserved; literal blanked.
        assert body["connections"]["db"]["auth"] == {
            "user": "${PGUSER}",
            "password": "",
        }


def test_writer_scrubs_literal_auth_in_malformed_blocks():
    """Even in malformed connection blocks, literal auth values must
    not survive a writer round-trip — otherwise a typo could leak a
    secret to disk through the malformed-preservation path."""
    with tempfile.TemporaryDirectory() as tmp:
        nb = create_notebook(Path(tmp), "scrub_malformed")
        body = _read_raw_toml(nb)
        body["connections"] = {
            "leaky": {
                # No driver → malformed
                "host": "localhost",
                "auth": {
                    "user": "${PGUSER}",
                    "password": "hunter2",  # literal: must be blanked
                },
            }
        }
        _write_raw_toml(nb, body)

        state = parse_notebook(nb)
        from strata.notebook.models import NotebookToml

        toml_obj = NotebookToml(
            notebook_id=state.id,
            name=state.name,
            cells=[],
            connections=state.connections,
            malformed_connections=state.malformed_connections,
        )
        write_notebook_toml(nb, toml_obj)

        body_after = _read_raw_toml(nb)
        assert body_after["connections"]["leaky"]["auth"] == {
            "user": "${PGUSER}",
            "password": "",
        }


def test_is_auth_indirection_recognizes_var_pattern():
    """Contract test for the regex that distinguishes ${VAR} from
    literals. Lower-case and underscored names are accepted; bare
    $VAR / empty / whitespace forms are rejected."""
    from strata.notebook.writer import is_auth_indirection

    assert is_auth_indirection("${PGPASS}")
    assert is_auth_indirection("${pgpass}")
    assert is_auth_indirection("${_PRIVATE}")
    assert is_auth_indirection("${a1b2}")
    assert not is_auth_indirection("$PGPASS")  # bash form, not supported
    assert not is_auth_indirection("${}")
    assert not is_auth_indirection("hunter2")
    assert not is_auth_indirection("")
    assert not is_auth_indirection("${PGPASS} extra")
    assert not is_auth_indirection(None)
    assert not is_auth_indirection(1234)


def test_parser_keeps_relative_path_verbatim():
    """The parser does NOT resolve relative paths — the on-disk
    value round-trips byte-for-byte through any unrelated edit.

    Codex review fix: an earlier iteration resolved relative paths
    in the parser, which meant editing a connection list through
    the UI rewrote ``path = "analytics.db"`` to a host-specific
    absolute path. The cell executor now resolves at adapter-open
    time instead (see ``cell_executor._resolve_runtime_spec``);
    the on-disk shape is preserved here."""
    from strata.notebook.parser import _parse_connections

    valid, malformed = _parse_connections(
        {"connections": {"warehouse": {"driver": "sqlite", "path": "analytics.db"}}}
    )
    assert malformed == []
    assert valid[0].path == "analytics.db"


def test_parser_keeps_absolute_path_verbatim():
    from strata.notebook.parser import _parse_connections

    valid, _ = _parse_connections(
        {"connections": {"db": {"driver": "sqlite", "path": "/tmp/elsewhere.db"}}}
    )
    assert valid[0].path == "/tmp/elsewhere.db"


def test_relative_path_resolves_at_adapter_open_time(tmp_path):
    """Cell-executor resolution: a relative ``path`` is rewritten
    to an absolute one only in the runtime view handed to the
    adapter. The original spec stays untouched."""
    from strata.notebook.models import ConnectionSpec
    from strata.notebook.sql.cell_executor import _resolve_runtime_spec

    spec = ConnectionSpec(name="db", driver="sqlite", path="analytics.db")
    runtime = _resolve_runtime_spec(spec, tmp_path)
    assert runtime.path == str((tmp_path / "analytics.db").resolve())
    # The original is unchanged, so it round-trips through the writer.
    assert spec.path == "analytics.db"


def test_absolute_path_unchanged_at_adapter_open_time(tmp_path):
    from strata.notebook.models import ConnectionSpec
    from strata.notebook.sql.cell_executor import _resolve_runtime_spec

    spec = ConnectionSpec(name="db", driver="sqlite", path="/tmp/already.db")
    runtime = _resolve_runtime_spec(spec, tmp_path)
    assert runtime.path == "/tmp/already.db"
