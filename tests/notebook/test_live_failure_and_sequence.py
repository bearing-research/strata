"""What an attached client is told, and whether it can trust the numbering.

Round 9, all three findings following from round 8.

Persisting a failed upstream's error was only half of it: the WebSocket
broadcasts from the staleness map, and the override that made a standing
failure win was applied to the cell and not to the map. So every other reader
of the session saw `error` while a viewer applying deltas was told `idle` and
kept the table from before the failure until it resynced.

The sequence contract was fixed for the paths the round-8 trace exercised, and
presence was not one of them: it read the counter's current value rather than
allocating, so two focus changes carried the number of whatever was sent before
them. A client deduping on `seq`, as the reference tells it to, dropped both.

And a write cell regenerated each statement from the parse tree before running
it, which is not always the statement the cell declares.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("adbc_driver_sqlite")

from strata.notebook.models import CellStatus  # noqa: E402
from strata.notebook.parser import parse_notebook  # noqa: E402
from strata.notebook.session import NotebookSession  # noqa: E402
from strata.notebook.writer import (  # noqa: E402
    add_cell_to_notebook,
    create_notebook,
    write_cell,
)

GOOD = (
    "# @sql connection=db\n"
    "# @name rev\n"
    "SELECT region, SUM(gross) AS net FROM orders GROUP BY region\n"
)
BAD = GOOD.replace("SUM(gross)", "SUM(missing_live_value)")
CONSUMER = "rows = len(rev)\n{'rows': rows}\n"


class Observer:
    """A connection that only records what the server sends it."""

    def __init__(self) -> None:
        self.raw: list[str] = []

    async def send_text(self, text: str) -> None:
        self.raw.append(text)

    @property
    def sent(self) -> list[dict[str, Any]]:
        return [json.loads(text) for text in self.raw]

    def frames_for(self, cell_id: str) -> list[dict[str, Any]]:
        return [f for f in self.sent if (f.get("payload") or {}).get("cell_id") == cell_id]


def _notebook(tmp_path: Path, cells: list[tuple[str, str, str]]) -> Path:
    db = tmp_path / "orders.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, region TEXT, gross REAL)")
        conn.executemany(
            "INSERT INTO orders VALUES (?,?,?)", [(1, "North", 100.0), (2, "South", 500.0)]
        )
        conn.commit()
    nb = create_notebook(tmp_path / "nb", "round9")
    after = None
    for cell_id, language, source in cells:
        add_cell_to_notebook(nb, cell_id, after_cell_id=after, language=language)
        write_cell(nb, cell_id, source)
        after = cell_id
    toml = nb / "notebook.toml"
    toml.write_text(toml.read_text() + f'\n[connections.db]\ndriver="sqlite"\npath="{db}"\n')
    return nb


def _session(nb: Path) -> Any:
    session = NotebookSession(parse_notebook(nb), nb)
    session._analyze_and_build_dag()
    session.environment_sync_state = "ready"
    return session


async def _run(session: Any, cell_id: str) -> Any:
    from strata.notebook.ws import _ensure_execution_state, execute_cell_and_broadcast

    return await execute_cell_and_broadcast(
        session, cell_id, _ensure_execution_state(session.id), session.id, mode="normal"
    )


@pytest.mark.asyncio
async def test_a_watching_client_is_told_the_dependency_failed(tmp_path):
    """The frames have to agree with every other view of the session."""
    from strata.notebook.ws import _notebook_connections

    nb = _notebook(tmp_path, [("q", "sql", GOOD), ("py", "python", CONSUMER)])
    session = _session(nb)
    await _run(session, "py")

    observer = Observer()
    _notebook_connections.setdefault(session.id, []).append(observer)
    try:
        write_cell(nb, "q", BAD)
        session.reload()
        session._analyze_and_build_dag()
        await _run(session, "py")
    finally:
        _notebook_connections.get(session.id, []).remove(observer)

    statuses = [
        (f.get("payload") or {}).get("status")
        for f in observer.frames_for("q")
        if f.get("type") == "cell_status"
    ]
    assert statuses, "the failed dependency was never mentioned to the client"
    assert statuses[-1] == "error", f"the client was told {statuses[-1]!r}"
    # And the same thing every other reader of the session is told.
    assert session.notebook_state.get_cell("q").status == CellStatus.ERROR


@pytest.mark.asyncio
async def test_presence_frames_carry_their_own_sequence(tmp_path):
    """Focus changes are distinct updates on one connected stream.

    The round-8 sequence test drove runs, syncs and notes; presence was never
    called, and presence was the path still reading the counter rather than
    advancing it.
    """
    from strata.notebook.ws import _handle_cell_focus, _notebook_connections, broadcast_presence

    nb = _notebook(tmp_path, [("q", "sql", GOOD), ("py", "python", CONSUMER)])
    session = _session(nb)

    observer = Observer()
    _notebook_connections.setdefault(session.id, []).append(observer)
    session.presence.join(observer, "reviewer")
    try:
        await broadcast_presence(session.id, session)
        await _handle_cell_focus(
            observer, session, {"cell_id": "q", "author": "reviewer"}, session.id
        )
        await _handle_cell_focus(
            observer, session, {"cell_id": None, "author": "reviewer"}, session.id
        )
    finally:
        _notebook_connections.get(session.id, []).remove(observer)

    seqs = [f["seq"] for f in observer.sent if f["type"] == "presence"]
    assert len(seqs) == 3, f"expected three presence frames, got {seqs}"
    assert len(set(seqs)) == len(seqs), f"presence frames repeated a sequence: {seqs}"
    assert seqs == sorted(seqs), f"presence sequences went backwards: {seqs}"


@pytest.mark.asyncio
async def test_a_write_cell_runs_the_statement_the_cell_declares(tmp_path):
    """A named recursive CTE lost its column list when the statement was
    regenerated from the parse tree, and SQLite refused what came out."""
    db = tmp_path / "w.db"
    sqlite3.connect(db).close()
    nb = create_notebook(tmp_path / "nb", "cte")
    add_cell_to_notebook(nb, "w", None, language="sql")
    write_cell(
        nb,
        "w",
        "# @sql connection=db write=true\n"
        "# @name cte_alias_probe\n"
        "CREATE TABLE IF NOT EXISTS alias_totals(total INTEGER);\n"
        "INSERT INTO alias_totals\n"
        "SELECT SUM(n) FROM (\n"
        "  WITH RECURSIVE counter(n) AS (\n"
        "    VALUES(0)\n"
        "    UNION ALL SELECT n+1 FROM counter WHERE n < 3\n"
        "  )\n"
        "  SELECT n FROM counter\n"
        ");\n",
    )
    toml = nb / "notebook.toml"
    toml.write_text(toml.read_text() + f'\n[connections.db]\ndriver="sqlite"\npath="{db}"\n')

    session = _session(nb)
    result = await _run(session, "w")

    assert result is not None and result.success, result and result.error
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT total FROM alias_totals").fetchall() == [(6,)]


@pytest.mark.asyncio
async def test_a_multi_statement_write_still_runs_each_statement(tmp_path):
    """Splitting on the tokenizer's semicolons has to keep every statement,
    and only the statements: a body ending in one must not run an empty tail."""
    db = tmp_path / "w.db"
    sqlite3.connect(db).close()
    nb = create_notebook(tmp_path / "nb", "multi")
    add_cell_to_notebook(nb, "w", None, language="sql")
    write_cell(
        nb,
        "w",
        "# @sql connection=db write=true\n"
        "# @name multi\n"
        "CREATE TABLE IF NOT EXISTS t(v TEXT);\n"
        "INSERT INTO t VALUES ('a; not a separator');\n"
        "INSERT INTO t VALUES ('b');\n",
    )
    toml = nb / "notebook.toml"
    toml.write_text(toml.read_text() + f'\n[connections.db]\ndriver="sqlite"\npath="{db}"\n')

    session = _session(nb)
    result = await _run(session, "w")

    assert result is not None and result.success, result and result.error
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT v FROM t ORDER BY v").fetchall() == [
            ("a; not a separator",),
            ("b",),
        ]
