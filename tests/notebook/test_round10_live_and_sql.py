"""Three gaps round 10 found, two of them in round 9's own fixes.

Splitting a write body at the tokenizer's semicolons kept any non-whitespace
tail as a statement, so a script ending in ``-- done`` handed the driver a bare
comment and it answered "INTERNAL: (unknown error)" for work it had already
finished.

Broadcasting the failed dependency's *status* was not enough: a status changes
a client's badge and nothing else, so the cell showed an error colour over the
table it produced before the failure, with no error text to read.

And the outbound sequence counter lived on the per-connection execution state,
which the grace-period teardown drops, so reconnecting to a session that was
still open restarted the count at 1. The protocol reference says the counter
resets only when the session closes or the server restarts.
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
BAD = GOOD.replace("SUM(gross)", "SUM(absent_detail_check)")
CONSUMER = "rows = len(rev)\n{'rows': rows}\n"


class Observer:
    """A connection that records only what the server sends it."""

    def __init__(self) -> None:
        self.raw: list[str] = []

    async def send_text(self, text: str) -> None:
        self.raw.append(text)

    @property
    def sent(self) -> list[dict[str, Any]]:
        return [json.loads(text) for text in self.raw]


def _db(tmp_path: Path) -> Path:
    db = tmp_path / "orders.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, region TEXT, gross REAL)")
        conn.executemany(
            "INSERT INTO orders VALUES (?,?,?)", [(1, "North", 100.0), (2, "South", 500.0)]
        )
        conn.commit()
    return db


def _notebook(tmp_path: Path, cells: list[tuple[str, str, str]], db: Path) -> Path:
    nb = create_notebook(tmp_path / "nb", "round10")
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


@pytest.mark.parametrize(
    ("name", "tail"),
    [("line", "-- done\n"), ("block", "/* all\n   done */\n")],
)
@pytest.mark.asyncio
async def test_a_trailing_comment_is_not_a_statement(tmp_path, name, tail):
    """The text after the last semicolon has no tokens, so there is nothing
    there to run."""
    db = tmp_path / "w.db"
    sqlite3.connect(db).close()
    nb = _notebook(
        tmp_path,
        [
            (
                "w",
                "sql",
                "# @sql connection=db write=true\n"
                "# @name probe\n"
                f"CREATE TABLE IF NOT EXISTS probe_{name}(value INTEGER);\n{tail}",
            )
        ],
        db,
    )
    session = _session(nb)

    result = await _run(session, "w")

    assert result is not None and result.success, result and result.error


@pytest.mark.asyncio
async def test_comments_and_quoted_semicolons_survive_together(tmp_path):
    """The reported script: a leading comment, a semicolon inside a string, and
    a trailing comment. Only the trailing comment broke it, and the row still
    has to land with its semicolon intact."""
    db = tmp_path / "w.db"
    sqlite3.connect(db).close()
    nb = _notebook(
        tmp_path,
        [
            (
                "w",
                "sql",
                "# @sql connection=db write=true\n"
                "# @name ledger\n"
                "-- seed the ledger\n"
                "CREATE TABLE IF NOT EXISTS notes(label TEXT);\n"
                "INSERT INTO notes VALUES ('quoted;semicolon');\n"
                "-- Completed this batch.\n",
            )
        ],
        db,
    )
    session = _session(nb)

    result = await _run(session, "w")

    assert result is not None and result.success, result and result.error
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT label FROM notes").fetchall() == [("quoted;semicolon",)]


@pytest.mark.asyncio
async def test_a_failed_dependency_sends_its_error_not_only_its_status(tmp_path):
    """A status alone leaves the client showing the table from before the
    failure, coloured as an error, with nothing to say what went wrong."""
    from strata.notebook.ws import _notebook_connections

    nb = _notebook(tmp_path, [("q", "sql", GOOD), ("py", "python", CONSUMER)], _db(tmp_path))
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

    errors = [
        f
        for f in observer.sent
        if f["type"] == "cell_error" and (f.get("payload") or {}).get("cell_id") == "q"
    ]
    assert errors, "the failed dependency's error never reached the client"
    assert "absent_detail_check" in errors[-1]["payload"]["error"]
    # The cell that was asked for keeps exactly one error frame of its own.
    consumer = [
        f
        for f in observer.sent
        if f["type"] == "cell_error" and (f.get("payload") or {}).get("cell_id") == "py"
    ]
    assert len(consumer) == 1, "the requested cell's error was announced twice"
    assert session.notebook_state.get_cell("q").status == CellStatus.ERROR


@pytest.mark.asyncio
async def test_the_sequence_survives_a_reconnect_after_the_grace_window(tmp_path):
    """The session stayed open the whole time, so the counter has to carry on.

    A client keeping the last number it saw, as the reference tells it to,
    reads a restart at 1 as frames it has already handled.
    """
    from strata.notebook.ws import (
        _ensure_execution_state,
        _tear_down_notebook_state,
        next_notebook_sequence,
    )

    nb = _notebook(tmp_path, [("q", "sql", GOOD)], _db(tmp_path))
    session = _session(nb)
    _ensure_execution_state(session.id)
    for _ in range(5):
        next_notebook_sequence(session.id)
    before = _ensure_execution_state(session.id).sequence
    assert before == 5

    # The last client goes away and the grace window expires.
    await _tear_down_notebook_state(session.id)

    assert next_notebook_sequence(session.id) > before


@pytest.mark.asyncio
async def test_closing_the_session_does_let_the_counter_go(tmp_path):
    """The one boundary the reference gives, so the map does not grow forever."""
    from strata.notebook.session import SessionManager
    from strata.notebook.ws import _ensure_execution_state, next_notebook_sequence

    nb = _notebook(tmp_path, [("q", "sql", GOOD)], _db(tmp_path))
    session = _session(nb)
    manager = SessionManager()
    manager._sessions[session.id] = session
    for _ in range(3):
        next_notebook_sequence(session.id)

    # Through the real close, which is also how the TTL sweep and the
    # max-count eviction end a session. Calling the helper directly would pass
    # while every eviction path still leaked.
    manager.close_session(session.id)

    assert _ensure_execution_state(session.id).sequence == 0


@pytest.mark.asyncio
async def test_a_body_of_only_comments_runs_nothing(tmp_path):
    """It tokenizes to no statements, which is not the same as failing to
    tokenize. Sending the comment is what the driver refused."""
    db = tmp_path / "w.db"
    sqlite3.connect(db).close()
    nb = _notebook(
        tmp_path,
        [("w", "sql", "# @sql connection=db write=true\n# @name nothing\n-- todo\n")],
        db,
    )
    session = _session(nb)

    result = await _run(session, "w")

    assert result is not None and result.success, result and result.error


@pytest.mark.asyncio
async def test_a_statement_after_a_comment_keeps_its_row_count(tmp_path):
    """A trailing comment parses to its own node, so dropping the fragment
    without dropping that node left the two lists a different length and every
    statement's kind fell back to a guess at its text. A guess reads
    ``WITH ... INSERT`` as "WITH", which is not DML, and the run then reports
    no row count for a statement that has one."""
    db = tmp_path / "w.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE totals(n INTEGER)")
        conn.commit()
    nb = _notebook(
        tmp_path,
        [
            (
                "w",
                "sql",
                "# @sql connection=db write=true\n"
                "# @name counted\n"
                "WITH RECURSIVE counter(n) AS (\n"
                "  VALUES(0) UNION ALL SELECT n+1 FROM counter WHERE n < 2\n"
                ")\n"
                "INSERT INTO totals SELECT n FROM counter;\n"
                "-- Completed.\n",
            )
        ],
        db,
    )
    session = _session(nb)

    result = await _run(session, "w")

    assert result is not None and result.success, result and result.error
    # The status row names the kind and the count. "3" alone would also match
    # the table's own "3 cols" header and pass without either being right.
    preview = (result.display_outputs or [{}])[-1].get("preview", "")
    assert "| INSERT | 3 |" in preview, f"kind or row count wrong: {preview!r}"


@pytest.mark.asyncio
async def test_a_failed_dependency_keeps_what_its_result_carried(tmp_path):
    """The frame is built from the upstream's own result, so an install
    suggestion survives. A frame made from cell state could not carry one:
    ``suggest_install`` is on the result and nowhere else."""
    from strata.notebook.ws import _notebook_connections

    nb = _notebook(
        tmp_path,
        [
            ("up", "python", "import definitely_not_installed_round10\nvalue = 1\n"),
            ("down", "python", "doubled = value * 2\n{'doubled': doubled}\n"),
        ],
        _db(tmp_path),
    )
    session = _session(nb)

    observer = Observer()
    _notebook_connections.setdefault(session.id, []).append(observer)
    try:
        await _run(session, "down")
    finally:
        _notebook_connections.get(session.id, []).remove(observer)

    frames = [
        f
        for f in observer.sent
        if f["type"] == "cell_error" and (f.get("payload") or {}).get("cell_id") == "up"
    ]
    assert frames, "the failed dependency's error never reached the client"
    assert frames[-1]["payload"].get("suggest_install") == "definitely_not_installed_round10"


@pytest.mark.asyncio
async def test_typing_elsewhere_does_not_re_announce_a_failure(tmp_path):
    """A source flush broadcasts the whole staleness map, and a standing
    failure is re-stamped on every pass. Announcing the error there would
    replay it every couple of seconds while someone types in another cell."""
    from strata.notebook.ws import (
        _ensure_execution_state,
        _handle_cell_source_update,
        _notebook_connections,
    )

    nb = _notebook(tmp_path, [("q", "sql", GOOD), ("py", "python", CONSUMER)], _db(tmp_path))
    session = _session(nb)
    await _run(session, "py")
    write_cell(nb, "q", BAD)
    session.reload()
    session._analyze_and_build_dag()
    await _run(session, "py")

    observer = Observer()
    _notebook_connections.setdefault(session.id, []).append(observer)
    try:
        await _handle_cell_source_update(
            observer,
            session,
            {"cell_id": "py", "source": CONSUMER + "# typing\n"},
            _ensure_execution_state(session.id),
            session.id,
        )
    finally:
        _notebook_connections.get(session.id, []).remove(observer)

    assert not [f for f in observer.sent if f["type"] == "cell_error"], (
        "a source flush replayed a failure that had already been announced"
    )


@pytest.mark.asyncio
async def test_every_cell_a_chain_broke_gets_its_own_error(tmp_path):
    """Round 11: a chain fails more than once, and each failure is somebody's
    stale result.

    Recording one upstream per run kept only the last one seen. The outer
    materialization sees the *consumer* fail after the inner one saw the cell
    that actually broke, so the client heard about the middle of the chain and
    never about its start, and the cell at fault went on showing its table.
    """
    from strata.notebook.ws import _notebook_connections

    nb = _notebook(
        tmp_path,
        [
            ("q", "sql", GOOD),
            ("mid", "python", "total = len(rev)\ntotal\n"),
            ("rep", "python", "doubled = total * 2\n{'doubled': doubled}\n"),
        ],
        _db(tmp_path),
    )
    session = _session(nb)
    warm = await _run(session, "rep")
    assert warm is not None and warm.success, warm and warm.error

    observer = Observer()
    _notebook_connections.setdefault(session.id, []).append(observer)
    try:
        write_cell(nb, "q", BAD)
        session.reload()
        session._analyze_and_build_dag()
        await _run(session, "rep")
    finally:
        _notebook_connections.get(session.id, []).remove(observer)

    errored = [
        (f.get("payload") or {}).get("cell_id") for f in observer.sent if f["type"] == "cell_error"
    ]
    assert set(errored) == {"q", "mid", "rep"}, f"only these were told about: {errored}"
    # The cell that broke is announced before the one its failure broke.
    assert errored.index("q") < errored.index("mid")
    seqs = [f["seq"] for f in observer.sent]
    assert seqs == sorted(seqs), f"frames went out of order: {seqs}"
    assert len(set(seqs)) == len(seqs), f"frames share a sequence: {seqs}"


@pytest.mark.asyncio
async def test_the_whole_chain_recovers_when_the_query_is_fixed(tmp_path):
    nb = _notebook(
        tmp_path,
        [
            ("q", "sql", GOOD),
            ("mid", "python", "total = len(rev)\ntotal\n"),
            ("rep", "python", "doubled = total * 2\n{'doubled': doubled}\n"),
        ],
        _db(tmp_path),
    )
    session = _session(nb)
    await _run(session, "rep")
    write_cell(nb, "q", BAD)
    session.reload()
    session._analyze_and_build_dag()
    await _run(session, "rep")

    write_cell(nb, "q", GOOD)
    session.reload()
    session._analyze_and_build_dag()
    result = await _run(session, "rep")

    assert result is not None and result.success, result and result.error
    for cell_id in ("q", "mid", "rep"):
        cell = session.notebook_state.get_cell(cell_id)
        assert cell.status == CellStatus.READY, f"{cell_id} is {cell.status}"
        assert cell.error is None, f"{cell_id} still carries {cell.error!r}"


@pytest.mark.asyncio
async def test_a_broken_chain_is_announced_whatever_language_it_runs_through(tmp_path):
    """The guarantee cannot depend on what kind of cell sits in the chain.

    Python and R turn a broken upstream into a failed result through their own
    ``except``; SQL, prompt and loop cells let the error out instead, and an
    escaping exception reached the caller by a route that never looked at what
    the run had found. So a SQL cell in the middle was never recorded and never
    announced, and a SQL cell at the end took the whole run with it.
    """
    from strata.notebook.ws import _notebook_connections

    db = _db(tmp_path)
    nb = _notebook(
        tmp_path,
        [
            ("up", "python", "import definitely_not_installed_round11\nlimit = 1\n"),
            (
                "q",
                "sql",
                "# @sql connection=db\n# @name rev\nSELECT region FROM orders LIMIT :limit\n",
            ),
            ("rep", "python", "seen = len(rev)\n{'seen': seen}\n"),
        ],
        db,
    )
    session = _session(nb)

    observer = Observer()
    _notebook_connections.setdefault(session.id, []).append(observer)
    try:
        await _run(session, "rep")
    finally:
        _notebook_connections.get(session.id, []).remove(observer)

    errored = [
        (f.get("payload") or {}).get("cell_id") for f in observer.sent if f["type"] == "cell_error"
    ]
    assert "up" in errored, f"the cell that broke was never announced: {errored}"
    assert "q" in errored, f"the SQL cell in the chain was never announced: {errored}"
    assert session.notebook_state.get_cell("q").status == CellStatus.ERROR


@pytest.mark.asyncio
async def test_a_sql_target_still_reports_the_upstream_that_broke(tmp_path):
    """A SQL cell asked for directly, whose own upstream is broken. The error
    used to escape as an exception, and the path that catches one never told
    the client anything about the cell that had actually failed."""
    from strata.notebook.ws import _notebook_connections

    nb = _notebook(
        tmp_path,
        [
            ("up", "python", "import definitely_not_installed_round11\nlimit = 1\n"),
            (
                "q",
                "sql",
                "# @sql connection=db\n# @name rev\nSELECT region FROM orders LIMIT :limit\n",
            ),
        ],
        _db(tmp_path),
    )
    session = _session(nb)

    observer = Observer()
    _notebook_connections.setdefault(session.id, []).append(observer)
    try:
        await _run(session, "q")
    finally:
        _notebook_connections.get(session.id, []).remove(observer)

    errored = [
        (f.get("payload") or {}).get("cell_id") for f in observer.sent if f["type"] == "cell_error"
    ]
    assert "up" in errored, f"only the SQL cell was mentioned: {errored}"
    assert session.notebook_state.get_cell("up").status == CellStatus.ERROR
