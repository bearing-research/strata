"""`strata watch` dispatch to the watch-only TUI.

``run_spectator`` is monkeypatched (it blocks on a live Textual app); these assert the target
(notebook dir or ``--session`` id) and server reach it.
"""

from __future__ import annotations

import pytest

from strata.cli import main
from strata.notebook.tui.cli import main as tui_main


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
        }
    ]


def test_watch_by_session(captured):
    assert main(["watch", "--session", "sess-123"]) == 0
    assert captured[0]["session"] == "sess-123"
    assert captured[0]["notebook"] is None


def test_watch_dir_and_session_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        main(["watch", "/tmp/nb", "--session", "sess-123"])


@pytest.fixture
def without_pillow(monkeypatch):
    """Simulate an install without the [tui] extra: PIL cannot be imported."""
    import sys

    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.delitem(sys.modules, "strata.notebook.tui.app", raising=False)


@pytest.mark.parametrize(
    "run",
    [
        lambda: main(["watch", "--session", "sess-123"]),
        lambda: tui_main(["--session", "sess-123"]),
    ],
    ids=["strata watch", "strata-notebook-tui"],
)
def test_missing_tui_extra_names_the_extra(without_pillow, run):
    with pytest.raises(SystemExit) as exc_info:
        run()
    message = str(exc_info.value.code)
    assert "[tui] extra" in message
    assert "'PIL'" in message
    assert "uv tool install 'strata-notebook[tui]'" in message
    assert "\n" not in message
