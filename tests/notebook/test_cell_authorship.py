"""Which cells an agent wrote, and which a person did.

A notebook had an owner; a cell had nobody. Runs are attributed through the
artifact's ``principal``, but only for remote builds and team-store offers, so
an edit left no record beyond git — and on a server where an agent and a person
both write cells, "which of these did the agent write" had no answer. Item 31.
"""

from __future__ import annotations

import tomllib

import pytest

from strata.notebook.authorship import LOCAL_AUTHOR, clean_author, resolve_author
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


@pytest.fixture
def notebook(tmp_path):
    return create_notebook(tmp_path, "Authored", initialize_environment=False)


def _cells(notebook):
    with open(notebook / "notebook.toml", "rb") as f:
        return {c["id"]: c for c in tomllib.load(f).get("cells", [])}


class TestResolving:
    def test_a_declared_author_is_taken_when_nobody_is_authenticated(self):
        """A personal server authenticates nobody, so the client's claim is the
        only thing available — and it is a claim, which is the honest amount of
        trust on a machine where anyone who can reach the server owns it."""
        assert resolve_author("agent:claude/1") == "agent:claude/1"

    def test_declaring_nothing_is_local(self):
        assert resolve_author(None) == LOCAL_AUTHOR

    def test_an_authenticated_principal_wins_over_a_claim(self, monkeypatch):
        """A client that could name itself where the server authenticates would
        be claiming an identity rather than presenting one."""
        from types import SimpleNamespace

        import strata.auth as auth_module

        monkeypatch.setattr(auth_module, "get_principal", lambda: SimpleNamespace(id="alice"))

        assert resolve_author("agent:pretending-to-be-alice") == "alice"

    def test_a_newline_never_reaches_the_byline(self):
        """This lands in TOML and in a cell view; a control character in the
        middle of a name is not something a caller meant."""
        assert clean_author("ag\nent\t") == "agent"

    def test_an_empty_claim_is_no_claim(self):
        assert clean_author("   ") is None


class TestAdding:
    def test_a_new_cell_records_who_added_it(self, notebook):
        add_cell_to_notebook(notebook, "c1", None, author="agent:claude")

        cell = _cells(notebook)["c1"]
        assert cell["created_by"] == "agent:claude"

    def test_it_also_counts_as_the_last_change(self, notebook):
        """A cell added and not yet edited was last changed by whoever added
        it; an empty ``updated_by`` would read as nobody having touched it."""
        add_cell_to_notebook(notebook, "c1", None, author="agent:claude")

        assert _cells(notebook)["c1"]["updated_by"] == "agent:claude"

    def test_adding_without_an_author_records_nobody(self, notebook):
        """What every cell added before this has."""
        add_cell_to_notebook(notebook, "c1", None)

        assert "created_by" not in _cells(notebook)["c1"]


class TestEditing:
    def test_an_edit_by_someone_else_moves_updated_by(self, notebook):
        add_cell_to_notebook(notebook, "c1", None, author="agent:claude")

        write_cell(notebook, "c1", "x = 1", author="local")

        cell = _cells(notebook)["c1"]
        assert cell["created_by"] == "agent:claude"
        assert cell["updated_by"] == "local"

    def test_editing_your_own_cell_does_not_rewrite_the_toml(self, notebook):
        """Source updates are a runtime concern that never touches committed
        config. Recording the author on every debounced flush would turn typing
        into a stream of commit-worthy diffs, so the write only happens when
        the answer actually changes."""
        add_cell_to_notebook(notebook, "c1", None, author="local")
        toml_path = notebook / "notebook.toml"
        before = toml_path.stat().st_mtime_ns
        original = toml_path.read_bytes()

        for i in range(5):
            write_cell(notebook, "c1", f"x = {i}", author="local")

        assert toml_path.read_bytes() == original
        assert toml_path.stat().st_mtime_ns == before

    def test_the_first_edit_by_a_new_author_writes_once(self, notebook):
        add_cell_to_notebook(notebook, "c1", None, author="local")
        toml_path = notebook / "notebook.toml"

        write_cell(notebook, "c1", "x = 1", author="agent:claude")
        after_first = toml_path.read_bytes()
        write_cell(notebook, "c1", "x = 2", author="agent:claude")

        assert _cells(notebook)["c1"]["updated_by"] == "agent:claude"
        assert toml_path.read_bytes() == after_first

    def test_an_edit_with_no_author_leaves_the_record_alone(self, notebook):
        """A caller with no opinion must not erase one someone else recorded."""
        add_cell_to_notebook(notebook, "c1", None, author="agent:claude")

        write_cell(notebook, "c1", "x = 1")

        assert _cells(notebook)["c1"]["updated_by"] == "agent:claude"


class TestItReachesTheView:
    def test_a_cell_added_through_mcp_shows_its_author(self, tmp_path, monkeypatch):
        """The case the item exists for: an agent's edits are distinguishable
        on a server that authenticates nobody."""
        import asyncio

        from strata.notebook.mcp_server import _add_cell
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession, SessionManager

        nb = create_notebook(tmp_path, "Authored", initialize_environment=False)
        sm = SessionManager()
        session = NotebookSession(parse_notebook(nb), nb)
        sm._sessions[session.id] = session

        view = asyncio.run(_add_cell(sm, session.id, "x = 1", author="agent:claude/sub"))

        assert view["created_by"] == "agent:claude/sub"
        assert view["updated_by"] == "agent:claude/sub"
        assert _cells(nb)[view["id"]]["created_by"] == "agent:claude/sub"
