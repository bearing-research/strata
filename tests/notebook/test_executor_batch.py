"""Integration tests for ``CellExecutor.execute_batch``.

They spawn the real harness subprocess against a real notebook venv.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


def _make_session_with_cells(tmp_path: Path, cells: list[tuple[str, str]]) -> NotebookSession:
    """Build a notebook with the given (cell_id, source) pairs and return a session."""
    notebook_dir = create_notebook(tmp_path, "BatchTest")
    prev: str | None = None
    for cell_id, source in cells:
        add_cell_to_notebook(notebook_dir, cell_id, after_cell_id=prev)
        write_cell(notebook_dir, cell_id, source)
        prev = cell_id
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


def _cell_spec(
    cell_id: str,
    source: str,
    *,
    env: dict[str, str] | None = None,
    mount_manifest: dict[str, dict[str, str]] | None = None,
) -> dict:
    """Build a batch cell-spec for tests."""
    return {
        "cell_id": cell_id,
        "source": source,
        "consumed_vars": [],  # filled in below from the session
        "env": dict(env or {}),
        "mount_manifest": dict(mount_manifest or {}),
        "source_hash": "",
        "env_hash": "",
    }


def _populate_consumed_vars(specs: list[dict], session: NotebookSession) -> list[dict]:
    """Fill consumed_vars on each spec from the session DAG."""
    dag = session.dag
    for spec in specs:
        consumed = dag.consumed_variables.get(spec["cell_id"], set()) if dag else set()
        spec["consumed_vars"] = sorted(consumed)
    return specs


@pytest.mark.asyncio
async def test_batch_executes_two_linear_cells_end_to_end(tmp_path: Path):
    """c1 produces x and c2 reads it; both succeed and persist, so a single-cell re-run of c2
    hits the cache.
    """
    session = _make_session_with_cells(
        tmp_path,
        [
            ("c1", "x = 41\n"),
            ("c2", "y = x + 1\n"),
        ],
    )
    specs = _populate_consumed_vars(
        [_cell_spec("c1", "x = 41\n"), _cell_spec("c2", "y = x + 1\n")],
        session,
    )

    executor = CellExecutor(session)
    result = await executor.execute_batch(specs)

    assert result.completed, (
        f"batch did not complete: end_reason={result.end_reason} "
        f"failed_cell_id={result.failed_cell_id} cell_results={result.cell_results}"
    )
    assert result.end_reason == "complete"
    statuses = {r.cell_id: r.status for r in result.cell_results}
    assert statuses == {"c1": "ok", "c2": "ok"}

    # c1's `x` is consumed by c2, so it is persisted; c2's `y` has no downstream
    # reader, so it is not in consumed_vars.
    c1 = session.notebook_state.get_cell("c1")
    assert "x" in c1.artifact_uris


@pytest.mark.asyncio
async def test_batch_stops_cleanly_on_cell_error(tmp_path: Path):
    """Cell 2 raises; batch ends with cell_error reason and c3 does not run."""
    session = _make_session_with_cells(
        tmp_path,
        [
            ("c1", "x = 1\n"),
            ("c2", "raise RuntimeError('boom')\n"),
            ("c3", "z = 99\n"),
        ],
    )
    specs = _populate_consumed_vars(
        [
            _cell_spec("c1", "x = 1\n"),
            _cell_spec("c2", "raise RuntimeError('boom')\n"),
            _cell_spec("c3", "z = 99\n"),
        ],
        session,
    )

    executor = CellExecutor(session)
    result = await executor.execute_batch(specs)

    assert not result.completed
    assert result.end_reason == "cell_error"
    assert result.failed_cell_id == "c2"

    statuses = {r.cell_id: r.status for r in result.cell_results}
    assert statuses["c1"] == "ok"
    assert statuses["c2"] == "cell_error"
    assert statuses["c3"] == "not_run"

    c2_result = next(r for r in result.cell_results if r.cell_id == "c2")
    assert c2_result.error is not None
    assert "RuntimeError" in (c2_result.traceback or "")


@pytest.mark.asyncio
async def test_batch_blocked_module_export_fails_persist(tmp_path: Path):
    """A consumed top-level lambda (blocked from module export) fails to persist, as in
    single-cell runs: status=persist_failed, not a silent pickle/object store.
    """
    # A top-level lambda is blocked from module export; with a downstream
    # consumer, single-cell mode returns success=False.
    session = _make_session_with_cells(
        tmp_path,
        [
            ("c1", "add = lambda x: x + 1\n"),
            ("c2", "y = add(41)\n"),
        ],
    )
    specs = _populate_consumed_vars(
        [_cell_spec("c1", "add = lambda x: x + 1\n"), _cell_spec("c2", "y = add(41)\n")],
        session,
    )

    executor = CellExecutor(session)
    result = await executor.execute_batch(specs)

    statuses = {r.cell_id: r.status for r in result.cell_results}
    assert statuses["c1"] == "persist_failed", (
        f"c1 should be persist_failed (blocked module export), got {statuses}"
    )
    # c2 must not run: the batch ends on persist failure.
    assert statuses["c2"] == "not_run"
    assert result.end_reason == "persist_failed"
    assert result.failed_cell_id == "c1"

    # c1's `add` must not be stored as a pickle; artifact_uris stays empty
    # because persist was rejected.
    c1 = session.notebook_state.get_cell("c1")
    assert "add" not in c1.artifact_uris, (
        f"add must not be persisted after module-export rejection; "
        f"got artifact_uris={c1.artifact_uris}"
    )


@pytest.mark.asyncio
async def test_batch_cache_hit_skips_execution(tmp_path: Path):
    """Run a 2-cell batch twice; c1's consumed ``x`` persists, so the second batch's c1 hits
    through ``_batch_service_cache_check`` and the harness load path.
    """
    session = _make_session_with_cells(
        tmp_path,
        [
            ("c1", "x = 41\n"),
            ("c2", "y = x + 1\n"),  # consumer of x, making x a consumed_var
        ],
    )
    specs = _populate_consumed_vars(
        [_cell_spec("c1", "x = 41\n"), _cell_spec("c2", "y = x + 1\n")],
        session,
    )

    executor = CellExecutor(session)
    first = await executor.execute_batch(specs)
    assert first.completed
    assert {r.cell_id: r.status for r in first.cell_results} == {"c1": "ok", "c2": "ok"}

    second = await executor.execute_batch(specs)
    assert second.completed, f"second batch failed: {second.end_reason}"
    statuses = {r.cell_id: r.status for r in second.cell_results}
    assert statuses["c1"] == "cache_hit", f"c1 should cache-hit on re-run; got {statuses}"


@pytest.mark.asyncio
async def test_batch_env_overrides_reach_cell(tmp_path: Path):
    """Cell-level env passed via cell_spec is bound when the cell runs."""
    session = _make_session_with_cells(
        tmp_path,
        [
            ("c1", "import os\nvalue = os.environ['STRATA_BATCH_ENV_TEST']\n"),
            ("c2", "y = value + '!'\n"),
        ],
    )
    specs = _populate_consumed_vars(
        [
            _cell_spec(
                "c1",
                "import os\nvalue = os.environ['STRATA_BATCH_ENV_TEST']\n",
                env={"STRATA_BATCH_ENV_TEST": "hello"},
            ),
            _cell_spec("c2", "y = value + '!'\n"),
        ],
        session,
    )

    executor = CellExecutor(session)
    result = await executor.execute_batch(specs)

    assert result.completed, f"batch errored: {result.end_reason}"
    statuses = {r.cell_id: r.status for r in result.cell_results}
    assert statuses == {"c1": "ok", "c2": "ok"}, (
        f"c1 should read the env var; got {statuses} "
        f"(if c1 is cell_error, the env didn't reach the harness)"
    )


@pytest.mark.asyncio
async def test_batch_cached_displays_round_trip(tmp_path: Path):
    """Second batch invocation cache-hits and restores display outputs.

    c1 produces a consumed variable (so the cache check has something to validate) and a
    display output.
    """
    session = _make_session_with_cells(
        tmp_path,
        [
            ("c1", "x = 1\ndisplay(Markdown('first hello'))\n"),
            ("c2", "y = x + 1\n"),
        ],
    )
    specs = _populate_consumed_vars(
        [
            _cell_spec("c1", "x = 1\ndisplay(Markdown('first hello'))\n"),
            _cell_spec("c2", "y = x + 1\n"),
        ],
        session,
    )

    executor = CellExecutor(session)
    first = await executor.execute_batch(specs)
    assert first.completed

    c1_first = next(r for r in first.cell_results if r.cell_id == "c1")
    assert c1_first.display_outputs, (
        f"first run: c1 should have display_outputs, got {c1_first.display_outputs!r}"
    )
    # Post-persist metadata carries an artifact_uri.
    assert c1_first.display_outputs[0].get("artifact_uri"), (
        f"display metadata after persist should carry artifact_uri; "
        f"got {c1_first.display_outputs[0]}"
    )

    second = await executor.execute_batch(specs)
    assert second.completed
    c1_second = next(r for r in second.cell_results if r.cell_id == "c1")
    assert c1_second.status == "cache_hit"
    assert c1_second.display_outputs, (
        f"cache-hit c1 should restore display_outputs; got {c1_second.display_outputs!r}"
    )
    # Cached displays keep the rich metadata (markdown_text / preview / image
    # inline data), not just {content_type, file, artifact_uri}.
    cached = c1_second.display_outputs[0]
    assert cached.get("markdown_text") == "first hello", (
        f"cached display lost markdown_text; got {cached!r}"
    )


@pytest.mark.asyncio
async def test_the_parents_own_work_is_not_charged_to_a_cell(tmp_path: Path, monkeypatch):
    """A cell's timeout is for its own code. While the parent answers the
    harness's cache check the harness is blocked, so that time is not the
    cell's: a slow cache check must not time out a fast cell."""
    session = _make_session_with_cells(tmp_path, [("c1", "x = 1\n")])
    specs = _populate_consumed_vars([_cell_spec("c1", "x = 1\n")], session)

    executor = CellExecutor(session)
    real_cache_check = executor._batch_service_cache_check

    async def slow_cache_check(*args, **kwargs):
        # Past the cell timeout below. The timeout itself is generous: what
        # still counts is the harness's own work (starting Python, running
        # the cell, serializing), which a loaded machine can stretch past 2s.
        await asyncio.sleep(10.0)
        return await real_cache_check(*args, **kwargs)

    monkeypatch.setattr(executor, "_batch_service_cache_check", slow_cache_check)

    result = await executor.execute_batch(specs, cell_timeout_seconds=8.0)

    assert result.completed, (result.end_reason, result.failed_cell_id)
    assert {r.cell_id: r.status for r in result.cell_results} == {"c1": "ok"}


@pytest.mark.asyncio
async def test_per_cell_watchdog_kills_hung_cell(tmp_path: Path):
    """A cell that hangs inside the batch is killed at its per-cell timeout
    instead of consuming the whole batch_timeout_seconds budget.
    Subsequent cells in the batch end up status=not_run.
    """
    session = _make_session_with_cells(
        tmp_path,
        [
            ("c1", "x = 1\n"),
            ("c_hang", "import time\ntime.sleep(60)\n"),
            ("c3", "y = 2\n"),
        ],
    )
    specs = _populate_consumed_vars(
        [
            _cell_spec("c1", "x = 1\n"),
            _cell_spec("c_hang", "import time\ntime.sleep(60)\n"),
            _cell_spec("c3", "y = 2\n"),
        ],
        session,
    )

    executor = CellExecutor(session)
    # c1's harness-side work still counts against it and a loaded run can take
    # it past 2s. 10s is still far below c_hang's 60s sleep and the 600s
    # batch_timeout_seconds, so a pass still means the per-cell kill.
    result = await executor.execute_batch(specs, cell_timeout_seconds=10.0)

    assert not result.completed
    assert result.end_reason == "cell_timeout"
    assert result.failed_cell_id == "c_hang"

    statuses = {r.cell_id: r.status for r in result.cell_results}
    assert statuses["c1"] == "ok"
    assert statuses["c_hang"] == "cell_error"
    assert (
        "timed out"
        in (next(r for r in result.cell_results if r.cell_id == "c_hang").error or "").lower()
    )
    # c3 was never reached: the harness was killed before it started.
    assert statuses.get("c3", "not_run") == "not_run"


@pytest.mark.asyncio
async def test_display_only_cell_is_cacheable(tmp_path: Path):
    """A cell with displays but no consumed variables cache-hits on re-run."""
    session = _make_session_with_cells(
        tmp_path,
        # c2 makes c1's `x` a consumed var (so the batch has something
        # consumed_vars-wise on c1). c2 itself is the display-only cell
        # we're verifying caches.
        [
            ("c1", "x = 1\n"),
            ("c2", "display(Markdown(f'value: {x}'))\n"),
        ],
    )
    specs = _populate_consumed_vars(
        [
            _cell_spec("c1", "x = 1\n"),
            _cell_spec("c2", "display(Markdown(f'value: {x}'))\n"),
        ],
        session,
    )

    executor = CellExecutor(session)
    first = await executor.execute_batch(specs)
    assert first.completed
    c2_first = next(r for r in first.cell_results if r.cell_id == "c2")
    assert c2_first.display_outputs, "c2 should produce display output"

    second = await executor.execute_batch(specs)
    assert second.completed
    c2_second = next(r for r in second.cell_results if r.cell_id == "c2")
    assert c2_second.status == "cache_hit", (
        f"display-only cell should cache-hit on re-run; got {c2_second.status}"
    )


@pytest.mark.asyncio
async def test_leaf_cells_cache_hit_in_a_batch_on_their_console(tmp_path: Path):
    """A leaf's record is its console, empty or not: a silent leaf and a print-only leaf both
    hit on the second Run All, and the printing one gets its stdout back.
    """
    cells = [
        ("c1", "x = 1\n"),
        ("c2", "import os\ny = x + 1\n"),
        ("c3", "print('seen', x)\n"),
    ]
    session = _make_session_with_cells(tmp_path, cells)
    specs = _populate_consumed_vars([_cell_spec(cid, src) for cid, src in cells], session)

    executor = CellExecutor(session)
    first = await executor.execute_batch(specs)
    assert first.completed
    assert {r.cell_id: r.status for r in first.cell_results} == {
        "c1": "ok",
        "c2": "ok",
        "c3": "ok",
    }

    second = await executor.execute_batch(specs)
    assert second.completed, f"second batch failed: {second.end_reason}"
    by_id = {r.cell_id: r for r in second.cell_results}
    assert by_id["c2"].status == "cache_hit", f"silent leaf re-ran: {by_id['c2']}"
    assert by_id["c3"].status == "cache_hit", f"print-only leaf re-ran: {by_id['c3']}"
    assert by_id["c3"].stdout == "seen 1\n"


@pytest.mark.asyncio
async def test_batch_warns_on_inplace_input_mutation_end_to_end(tmp_path: Path):
    """An in-place mutation of an upstream DataFrame warns on the BatchCellResult.

    Exercises the full harness, persist and executor chain, with the aliased form the static
    analyzer cannot recapture.
    """
    src_make = "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})\n"
    src_mutate = "alias = df\nalias.drop(index=[0], inplace=True)\nn = len(df)\n"
    session = _make_session_with_cells(
        tmp_path,
        [("make", src_make), ("mutate", src_mutate)],
    )
    specs = _populate_consumed_vars(
        [_cell_spec("make", src_make), _cell_spec("mutate", src_mutate)],
        session,
    )
    # references drive the harness's runtime mutation detection.
    for spec in specs:
        cell = session.notebook_state.get_cell(spec["cell_id"])
        spec["references"] = sorted(cell.references or [])

    executor = CellExecutor(session)
    result = await executor.execute_batch(specs)

    assert result.completed, (
        f"batch did not complete: end_reason={result.end_reason} "
        f"failed_cell_id={result.failed_cell_id}"
    )
    by_id = {r.cell_id: r for r in result.cell_results}
    # Producer mutated nothing it received; mutator warns about df.
    assert by_id["make"].mutation_warnings == []
    warnings = by_id["mutate"].mutation_warnings
    assert len(warnings) == 1
    assert warnings[0]["var_name"] == "df"


@pytest.mark.asyncio
async def test_an_edit_during_a_batch_does_not_rename_what_ran(tmp_path: Path):
    """A batch runs the sources captured when the partition was built.

    An edit to a cell whose turn has not come is not refused, so outputs must be filed under
    the hash of the source that ran, not the edited one.
    """
    session = _make_session_with_cells(
        tmp_path,
        [("c1", "x = 41\n"), ("c2", "y = x + 1\n"), ("c3", "z = y + 1\n")],
    )
    specs = _populate_consumed_vars(
        [
            _cell_spec("c1", "x = 41\n"),
            _cell_spec("c2", "y = x + 1\n"),
            _cell_spec("c3", "z = y + 1\n"),
        ],
        session,
    )
    executor = CellExecutor(session)
    edited = "y = x + 1000\n"

    async def _edit_the_cell_whose_turn_has_not_come(result) -> None:
        if result.cell_id == "c1":
            session.notebook_state.get_cell("c2").source = edited

    outcome = await executor.execute_batch(
        specs, on_cell_event=_edit_the_cell_whose_turn_has_not_come
    )

    assert outcome.completed, f"batch failed: {outcome.end_reason}"
    assert session.notebook_state.get_cell("c2").source == edited, "the edit never landed"
    ran = await executor._compute_cell_provenance("c2", "y = x + 1\n")
    as_edited = await executor._compute_cell_provenance("c2", edited)
    assert ran.provenance_hash != as_edited.provenance_hash, "the two sources must differ"
    # Each consumed variable is keyed off the cell's hash, so that is where
    # "which source does this output claim to be" actually shows up.
    from strata.notebook.provenance import derive_subkey

    store = session.artifact_manager.artifact_store
    assert store.find_by_provenance(derive_subkey(as_edited.provenance_hash, "y")) is None, (
        "the batch stored its result under the hash of the edited source, so "
        "reopening shows the edit as ready while holding output it never produced"
    )
    assert store.find_by_provenance(derive_subkey(ran.provenance_hash, "y")) is not None, (
        "the result was not stored under the source that actually ran"
    )


@pytest.mark.asyncio
async def test_a_cell_in_a_batch_can_promote_what_it_reads(tmp_path: Path):
    """``strata.promote("x")`` in a batch needs a promote url and the variable-to-artifact map.

    Without the map the client cannot resolve ``x``; without the url it reports no team store,
    though the cell promotes fine on its own.
    """
    import http.server
    import threading

    promoted: list[str] = []

    class _Store(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server's spelling
            promoted.append(self.path)
            body = b'{"ok": true}'
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            return

    server = http.server.HTTPServer(("127.0.0.1", 0), _Store)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        session = _make_session_with_cells(
            tmp_path,
            [
                ("c1", "x = 41\n"),
                ("c2", 'strata.promote("x", name="shared/x")\ny = x + 1\n'),
            ],
        )
        specs = _populate_consumed_vars(
            [
                _cell_spec("c1", "x = 41\n"),
                _cell_spec("c2", 'strata.promote("x", name="shared/x")\ny = x + 1\n'),
            ],
            session,
        )
        for spec in specs:
            spec["strata_url"] = f"http://127.0.0.1:{port}"
            spec["strata_promote_url"] = f"http://127.0.0.1:{port}"

        outcome = await CellExecutor(session).execute_batch(specs)

        assert outcome.completed, f"batch failed: {outcome.end_reason}"
        statuses = {r.cell_id: r.status for r in outcome.cell_results}
        assert statuses["c2"] == "ok", (
            f"the promoting cell failed: "
            f"{[r.error for r in outcome.cell_results if r.cell_id == 'c2']}"
        )
        assert promoted, "the cell never reached the team store"
        assert "/promote" in promoted[0] and "@v=" not in promoted[0], promoted[0]
        assert "_cell_c1_var_x" in promoted[0], (
            f"promoted something other than c1's x: {promoted[0]}"
        )
    finally:
        server.shutdown()


@pytest.mark.asyncio
async def test_nocache_runs_every_time_in_a_batch_too(tmp_path: Path):
    """``# @nocache`` marks an effect the artifact does not capture, so Run All must run the
    cell, not serve it from cache.
    """
    source = "# @nocache\nimport pathlib\nx = 1\n"
    # c3 exists so c2 has a consumer and the control below is an ordinary
    # variable hit, not a leaf's console hit.
    session = _make_session_with_cells(
        tmp_path,
        [("c1", source), ("c2", "y = x + 1\n"), ("c3", "z = y + 1\n")],
    )
    specs = _populate_consumed_vars(
        [
            _cell_spec("c1", source),
            _cell_spec("c2", "y = x + 1\n"),
            _cell_spec("c3", "z = y + 1\n"),
        ],
        session,
    )
    executor = CellExecutor(session)

    first = await executor.execute_batch(specs)
    second = await executor.execute_batch(specs)

    assert first.completed and second.completed
    hits = {r.cell_id: r.cache_hit for r in second.cell_results}
    assert hits["c1"] is False, "a @nocache cell was served from cache inside Run All"
    assert hits["c2"] is True, "the ordinary cell should still cache, or this proves nothing"
