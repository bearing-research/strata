"""Deterministic CI coverage for the agent-notebook eval harness.

Exercises the trajectory classification, the graders, the drivers (replay +
stream-json parsing), and the full replay `run_suite` — with no server, no
venv, and no LLM. The live Claude Code path is on-demand and not covered here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.agent_notebook import graders, runner
from evals.agent_notebook.drivers import (
    ReplayDriver,
    parse_normalized,
    parse_stream_json,
)
from evals.agent_notebook.tasks import TASKS_BY_ID
from evals.agent_notebook.trajectory import ToolEvent, Trajectory

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSCRIPTS = REPO_ROOT / "evals" / "agent_notebook" / "transcripts"


# --- trajectory classification -------------------------------------------------


def test_mcp_tool_name_prefix_is_recognized() -> None:
    # Claude Code v2.1.x keeps the hyphen: mcp__strata-notebook__<tool>. Splitting
    # on __ and keying off the trailing segment tolerates hyphen or underscore.
    assert ToolEvent("mcp__strata-notebook__run_cell").is_notebook_work
    assert ToolEvent("mcp__strata_notebook__run_cell").is_notebook_work
    assert ToolEvent("mcp__strata-notebook__get_notebook").is_notebook_read
    # Reads are neither work nor escape.
    assert not ToolEvent("mcp__strata-notebook__status").is_notebook_work


def test_bash_python_is_an_escape_but_other_bash_is_not() -> None:
    assert ToolEvent("Bash", {"command": "python train.py"}).escape_reason == "bash-python"
    assert ToolEvent("Bash", {"command": "uv run python -c 'print(1)'"}).is_escape
    assert ToolEvent("Bash", {"command": "pytest -q"}).escape_reason == "bash-python"
    assert not ToolEvent("Bash", {"command": "ls -la"}).is_escape
    assert not ToolEvent("Bash", {"command": "git status"}).is_escape
    # A path that merely contains 'python' must not trip the detector.
    assert not ToolEvent("Bash", {"command": "echo $PYTHONPATH"}).is_escape


def test_bash_package_install_is_an_escape() -> None:
    # Managing deps via Bash bypasses add_dependency — the change never lands
    # in the notebook's committed env.
    pip = ToolEvent("Bash", {"command": "pip install requests"})
    uv_add = ToolEvent("Bash", {"command": "uv add pandas"})
    assert pip.escape_reason == "bash-install"
    assert uv_add.escape_reason == "bash-install"
    assert ToolEvent("Bash", {"command": "uv pip install numpy"}).is_escape
    assert not ToolEvent("Bash", {"command": "uv sync"}).is_escape


def test_editing_a_cell_file_directly_is_an_escape() -> None:
    # Writing/Editing a cell's source file bypasses edit_cell/add_cell.
    assert (
        ToolEvent("Write", {"file_path": "/nb/cells/abc123.py"}).escape_reason == "cell-file-edit"
    )
    assert ToolEvent("Edit", {"file_path": "/nb/cells/abc123.py"}).is_escape
    # Editing other files (a data file, the pyproject) is not a notebook bypass.
    assert not ToolEvent("Write", {"file_path": "/nb/data/x.csv"}).is_escape
    assert not ToolEvent("Edit", {"file_path": "/nb/pyproject.toml"}).is_escape


# --- graders -------------------------------------------------------------------


def test_in_tool_rate_all_notebook_is_one() -> None:
    traj = Trajectory(events=[ToolEvent("mcp__strata_notebook__run_cell") for _ in range(3)])
    grade = graders.grade_in_tool(traj)
    assert grade.notebook_work == 3
    assert grade.escapes == 0
    assert grade.in_tool_rate == 1.0
    assert not grade.no_activity


def test_in_tool_rate_mixed_penalizes_escapes() -> None:
    traj = Trajectory(
        events=[
            ToolEvent("mcp__strata_notebook__add_cell"),
            ToolEvent("mcp__strata_notebook__run_cell"),
            ToolEvent("mcp__strata_notebook__run_cell"),
            ToolEvent("Bash", {"command": "python scratch.py"}),
        ]
    )
    grade = graders.grade_in_tool(traj)
    assert grade.notebook_work == 3
    assert grade.escapes == 1
    assert grade.in_tool_rate == 0.75
    assert grade.escape_details == ["bash-python: python scratch.py"]


def test_in_tool_rate_no_activity_flagged() -> None:
    traj = Trajectory(events=[ToolEvent("mcp__strata_notebook__list_notebooks")])
    grade = graders.grade_in_tool(traj)
    assert grade.no_activity
    assert grade.notebook_work == 0


def test_completion_checks_variables_and_run_ok(tmp_path) -> None:
    from strata.notebook.ops import LocalNotebookOps
    from strata.notebook.writer import create_notebook

    nb = create_notebook(tmp_path, "compl", initialize_environment=False)
    ops = LocalNotebookOps(nb)
    ops.add_cell("df = 1\n", language="python")
    ops.add_cell("total_population = 2\n", language="python")

    good = graders.grade_completion(nb, ["df", "total_population"], run_ok=True)
    assert good.passed and good.missing_variables == []

    missing = graders.grade_completion(nb, ["df", "nope"], run_ok=True)
    assert not missing.passed and missing.missing_variables == ["nope"]

    failed_run = graders.grade_completion(nb, ["df"], run_ok=False)
    assert not failed_run.passed


# --- drivers -------------------------------------------------------------------


def test_parse_stream_json_both_shapes_and_result() -> None:
    lines = [
        json.dumps(
            {
                "type": "stream_event",
                "event": {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
            }
        ),
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "ok"},
                        {
                            "type": "tool_use",
                            "name": "mcp__strata_notebook__run_cell",
                            "input": {"cell_id": "c1"},
                        },
                    ]
                },
            }
        ),
        json.dumps({"type": "result", "result": "done", "is_error": False, "subtype": "success"}),
    ]
    traj = parse_stream_json("\n".join(lines))
    assert [e.name for e in traj.events] == ["Bash", "mcp__strata_notebook__run_cell"]
    assert traj.events[1].is_notebook_work
    assert traj.final_text == "done"
    assert traj.ok


def test_parse_stream_json_marks_error_result() -> None:
    line = json.dumps({"type": "result", "result": "boom", "is_error": True})
    assert parse_stream_json(line).ok is False


def test_parse_normalized_roundtrip() -> None:
    obj = {"events": [{"name": "Bash", "arguments": {"command": "python x.py"}}], "ok": True}
    traj = parse_normalized(obj)
    assert traj.events[0].is_escape


def test_replay_driver_reads_committed_transcripts() -> None:
    driver = ReplayDriver(TRANSCRIPTS)
    task = TASKS_BY_ID["build_summary"]
    traj = driver.run(task, Path("/unused"))
    assert traj.work_events()  # it added and ran cells
    assert not traj.escape_events()  # ideal in-tool run


def test_replay_driver_missing_transcript_raises(tmp_path) -> None:
    driver = ReplayDriver(tmp_path)
    with pytest.raises(FileNotFoundError):
        driver.run(TASKS_BY_ID["build_summary"], Path("/unused"))


# --- full replay suite (harness wiring, no server) -----------------------------


def _transcript_backed_tasks() -> list:
    """Tasks that have a committed transcript (hard tasks are live-only)."""
    ids = {p.stem for p in TRANSCRIPTS.glob("*.json")}
    ids |= {p.stem for p in TRANSCRIPTS.glob("*.jsonl")}
    return [TASKS_BY_ID[i] for i in sorted(ids) if i in TASKS_BY_ID]


def test_run_suite_replay_scores_committed_transcripts(tmp_path) -> None:
    driver = ReplayDriver(TRANSCRIPTS)
    tasks = _transcript_backed_tasks()
    assert len(tasks) >= 9  # the core suite is transcript-backed
    results = runner.run_suite(tasks, driver, tmp_path, live=False)

    assert len(results) == len(tasks)
    assert all(r.error is None for r in results)
    # Every committed transcript is an ideal in-tool run: no escapes, rate 1.0.
    assert all(r.in_tool.escapes == 0 for r in results)
    assert all(r.in_tool.in_tool_rate == 1.0 for r in results)

    summary = runner.summarize(results)
    assert summary["errored"] == 0
    assert summary["mean_in_tool_rate"] == 1.0
    # Table renders without raising.
    assert "in-tool" in runner.format_table(results)


def test_run_suite_replay_detects_an_escape(tmp_path) -> None:
    # A transcript where the agent routes around the notebook scores < 1.0.
    bad_dir = tmp_path / "bad"
    bad_dir.mkdir()
    (bad_dir / "extend_dag.json").write_text(
        json.dumps(
            {
                "events": [
                    {"name": "mcp__strata_notebook__list_notebooks"},
                    {
                        "name": "Bash",
                        "arguments": {"command": "uv run python -c 'print(sum(range(100)))'"},
                    },
                ],
                "ok": True,
            }
        )
    )
    results = runner.run_suite(
        [TASKS_BY_ID["extend_dag"]], ReplayDriver(bad_dir), tmp_path, live=False
    )
    assert results[0].in_tool.escapes == 1
    assert results[0].in_tool.in_tool_rate == 0.0


def test_run_suite_repeats_produces_one_result_per_run(tmp_path) -> None:
    driver = ReplayDriver(TRANSCRIPTS)
    tasks = [TASKS_BY_ID["build_summary"], TASKS_BY_ID["extend_dag"]]
    results = runner.run_suite(tasks, driver, tmp_path, live=False, repeats=3)
    assert len(results) == 6  # 2 tasks x 3 repeats
    # The aggregated table has one row per task, not per run.
    body = [ln for ln in runner.format_table(results).splitlines()[2:] if ln.strip()]
    assert len(body) == 2
