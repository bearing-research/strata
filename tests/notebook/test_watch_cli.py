"""`strata watch` dispatch — the watch-only TUI attach.

The spectator itself blocks on a live Textual app, so these tests monkeypatch
``run_spectator`` and assert the subcommand routes its target (a local notebook
dir or a ``--session`` id) and server through to it.
"""

from __future__ import annotations

import pytest

from strata.cli import main


@pytest.fixture
def captured(monkeypatch):
    calls: list[dict] = []

    def fake_run_spectator(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr("strata.notebook.tui.cli.run_spectator", fake_run_spectator)
    return calls


def test_watch_by_notebook_dir(captured):
    assert main(["watch", "/tmp/nb", "--server", "http://localhost:9000"]) == 0
    assert captured == [
        {
            "server": "http://localhost:9000",
            "session": None,
            "notebook": "/tmp/nb",
            "user_header": None,
            "user": None,
        }
    ]


def test_watch_by_session(captured):
    assert main(["watch", "--session", "sess-123"]) == 0
    assert captured[0]["session"] == "sess-123"
    assert captured[0]["notebook"] is None


def test_watch_dir_and_session_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        main(["watch", "/tmp/nb", "--session", "sess-123"])
