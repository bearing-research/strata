"""The uptake runner: arm setup, preflight, turns through ``--resume``, scoring, the
report and its intervals. A fake ``claude`` stands in for the agent."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from evals.agent_notebook.uptake import report, run, stats
from evals.agent_notebook.uptake.__main__ import main
from evals.agent_notebook.uptake.tasks import TASKS_BY_ID

# The runner puts POSIX shell shims and symlinks on the agent's PATH.
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the runner is POSIX only")

LOAD = (
    "PYTHONPATH=src:packages/fleet/src {python} -c"
    " 'from harbor.data import load_events; load_events()'"
)


@pytest.fixture
def claude(tmp_path) -> str:
    """The fake agent, as an executable."""
    path = tmp_path / "fake-claude"
    fake = Path(__file__).with_name("fake_claude.py")
    path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake}" "$@"\n')
    path.chmod(0o755)
    return str(path)


def _trial(tmp_path, claude, monkeypatch, task_id, arm, script, **env):
    monkeypatch.setenv("FAKE_SCRIPT", json.dumps(script))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return run.run_trial(
        TASKS_BY_ID[task_id],
        arm,
        0,
        tmp_path / "trial",
        claude=claude,
        rows=20_000,
        load_seconds=0,
        sync=False,
        pinned={"model": "fake"},
    )


def _answers(task_id: str, tmp_path: Path) -> list[str]:
    """Replies that pass, computed on the same seeded build the trial makes."""
    from evals.agent_notebook.uptake import workspace as ws

    work = ws.build(tmp_path / "answers", rows=20_000)
    replies = []
    for expect in TASKS_BY_ID[task_id].expected(work):
        replies.append(expect.options[0] if hasattr(expect, "options") else f"{expect.value}")
    return replies


class TestATrial:
    def test_its_turns_resume_one_session_and_are_scored(self, tmp_path, claude, monkeypatch):
        replies = _answers("iter_west", tmp_path)
        load = LOAD.format(python=sys.executable)
        script = [[load, replies[0]], [None, replies[1]], [load, "no idea"], [None, replies[3]]]

        trial = _trial(tmp_path, claude, monkeypatch, "iter_west", "A", script)

        assert trial.discarded is None
        calls = [
            json.loads(line)
            for line in (tmp_path / "trial" / "config" / "calls.jsonl").read_text().splitlines()
        ]
        assert "--resume" not in calls[0]
        assert {c[c.index("--resume") + 1] for c in calls[1:]} == {trial.session_id}
        assert [t.passed for t in trial.turns] == [True, True, False, True]
        assert not trial.completed
        # The counter saw both loads, each in its own turn.
        assert [t.loads for t in trial.turns] == [1, 0, 1, 0]
        assert len(trial.loads) == 2
        saved = json.loads((tmp_path / "trial" / "trial.json").read_text())
        assert saved["completed"] is False and saved["strata_calls"] == 0

    def test_an_edit_is_checked_in_the_files(self, tmp_path, claude, monkeypatch):
        fix = (
            f"{sys.executable} -c \"import pathlib; p = pathlib.Path('src/harbor/text.py');"
            " p.write_text(p.read_text().replace('[^a-z0-9]', '[^a-z0-9]+'))\""
        )

        done = _trial(tmp_path, claude, monkeypatch, "edit_fix_slugify", "B", [[fix, "Fixed."]])

        assert done.discarded is None and done.completed
        assert run.uses_strata({"name": "Skill", "input": {"skill": "strata-scratchpad"}})

    def test_the_workspace_is_a_committed_repo(self, tmp_path, claude, monkeypatch):
        _trial(tmp_path, claude, monkeypatch, "neutral_modpow", "A", [[None, "246"]])

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=tmp_path / "trial" / "harbor",
            capture_output=True,
            text=True,
            check=True,
        )
        assert status.stdout == ""


class TestPreflight:
    def test_arm_b_without_its_skill_is_discarded(self, tmp_path, claude, monkeypatch):
        prepare = run.prepare_arm

        def without_skill(arm, config, bin_dir):
            prepare(arm, config, bin_dir)
            shutil.rmtree(config / "skills")

        monkeypatch.setattr(run, "prepare_arm", without_skill)

        trial = _trial(tmp_path, claude, monkeypatch, "neutral_modpow", "B", [[None, "246"]])

        assert trial.discarded == "turn 0: the strata-scratchpad skill is missing"
        assert not trial.turns[0].passed

    def test_a_strata_server_in_any_arm_is_discarded(self, tmp_path, claude, monkeypatch):
        trial = _trial(
            tmp_path,
            claude,
            monkeypatch,
            "neutral_modpow",
            "A",
            [[None, "246"]],
            FAKE_SERVERS="strata",
        )

        assert trial.discarded.startswith("turn 0: a strata MCP server is connected")

    def test_a_resume_that_loses_the_session_is_discarded(self, tmp_path, claude, monkeypatch):
        script = [[None, "x"]] * 4

        trial = _trial(
            tmp_path, claude, monkeypatch, "iter_west", "A", script, FAKE_NEW_SESSION="2"
        )

        assert trial.discarded == "turn 2: the resumed turn started a new session"
        assert len(trial.turns) == 3

    def test_a_turn_that_never_reached_the_model_is_discarded(self, tmp_path, claude, monkeypatch):
        script = [[None, "x"]] * 4

        trial = _trial(tmp_path, claude, monkeypatch, "iter_west", "A", script, FAKE_API_ERROR="1")

        assert trial.discarded == "turn 1: the agent never reached the model: Invalid API key"

    def test_init_events(self):
        assert run.preflight("A", None) == "no init event"
        assert run.preflight("A", {"skills": ["strata-scratchpad"]}) == (
            "the strata-scratchpad skill is loaded"
        )
        assert run.preflight("B", {"skills": ["strata-scratchpad"], "mcp_servers": []}) is None


class TestArmA:
    def test_strata_is_not_found_and_its_commands_are_denied(self, tmp_path):
        config, bin_dir = tmp_path / "config", tmp_path / "bin"
        run.prepare_arm("A", config, bin_dir)
        env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}

        missing = subprocess.run(["sh", "-c", "strata --version"], env=env, capture_output=True)
        assert missing.returncode == 127

        (hook,) = json.loads((config / "settings.json").read_text())["hooks"]["PreToolUse"]
        command = hook["hooks"][0]["command"]

        def decide(bash: str) -> int:
            payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": bash}})
            return subprocess.run(command, shell=True, input=payload, text=True).returncode

        assert decide("strata cell add ./s -c 'x = 1' --run") == 2
        assert decide("cd x && uv run strata ls") == 2
        assert decide("~/.local/bin/strata --version") == 2
        assert decide("python3 -c 'print(1)'") == 0
        assert decide("grep -r stratagem .") == 0

    def test_the_agent_env_leaves_the_eval_process_out(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("VIRTUAL_ENV", "/repo/.venv")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://proxy")

        env = run.agent_env(tmp_path / "c", tmp_path / "b", tmp_path / "l", 20)

        assert "CLAUDECODE" not in env and "VIRTUAL_ENV" not in env
        assert "ANTHROPIC_BASE_URL" not in env and env["ANTHROPIC_API_KEY"] == "k"
        assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "c")
        assert env["PATH"].startswith(str(tmp_path / "b"))


class TestStats:
    def test_wilson_meets_the_designs_bars(self):
        assert stats.wilson(31, 40)[0] >= 0.60 > stats.wilson(30, 40)[0]
        assert stats.wilson(0, 81)[1] <= 0.05 < stats.wilson(1, 81)[1]

    def test_newcombe_matches_the_published_example(self):
        # Newcombe (1998), example (a): 56/70 against 48/80.
        low, high = stats.newcombe(56, 70, 48, 80)

        assert (round(low, 4), round(high, 4)) == (0.0524, 0.3339)

    def test_the_median_ratio_brackets_its_point(self):
        point, low, high = stats.median_ratio([10, 11, 12, 13, 14], [20, 21, 22, 23, 24])

        assert point == 12 / 22 and low <= point <= high < 1


def _saved(arm, task_id, *, completed=True, loads=1, strata=0, discarded=None, wall=10.0):
    task = TASKS_BY_ID[task_id]
    return {
        "task": task_id,
        "kind": task.kind,
        "arm": arm,
        "discarded": discarded,
        "completed": completed,
        "strata_calls": strata,
        "loads": [{}] * loads,
        "wall_s": wall,
        "turns": [{"agent_turns": 3, "cost_usd": 0.5}],
        "pins": {"model": "m"},
    }


class TestTheReport:
    def test_it_scores_per_arm_and_kind_and_flags_a_broken_gate(self):
        trials = [
            _saved("A", "iter_west", loads=4, wall=100),
            _saved("A", "iter_west", loads=4, wall=110, strata=1),
            _saved("A", "iter_west", discarded="turn 0: no init event"),
            _saved("B", "iter_west", loads=1, wall=40, strata=2),
            _saved("B", "iter_west", loads=1, wall=50, strata=1, completed=False),
            _saved("B", "edit_fix_slugify", loads=0, strata=1),
        ]

        summary = report.summarize(trials)

        a = summary["arms"]["A"]["iteration"]
        assert (a["trials"], a["discarded"], a["payoff"]["k"]) == (3, 1, 0)
        b = summary["arms"]["B"]
        assert (b["iteration"]["payoff"]["k"], b["iteration"]["completed"]["k"]) == (2, 1)
        assert summary["false_positive"]["B"]["k"] == 1
        assert summary["versus_control"]["B"]["iteration"]["median_wall_ratio"][0] == 45 / 105
        assert summary["warnings"] == [
            "arm A: 1 of 3 runs discarded",
            "arm A used strata in 1 runs",
        ]
        assert "arm B vs A" in report.render(summary)

    def test_the_cli_runs_trials_once_and_reports(self, tmp_path, claude, monkeypatch, capsys):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("FAKE_SCRIPT", json.dumps([[None, "246"]]))
        monkeypatch.setattr(run, "prepare_workspace", lambda dest, **kw: _small(dest))
        argv = ["run", "--arm", "A", "--out", str(tmp_path / "out"), "--task", "neutral_modpow"]

        assert main([*argv, "--claude", claude]) == 0
        assert main([*argv, "--claude", claude]) == 0
        printed = capsys.readouterr()
        assert printed.out.count("neutral_modpow #0: completed") == 1
        assert printed.err == ""
        # A trial that died partway is left for a person to look at, not rerun.
        (tmp_path / "out" / "A" / "neutral_modpow" / "1").mkdir()
        assert main([*argv, "--claude", claude, "--runs", "2"]) == 0
        assert "an unfinished trial is there" in capsys.readouterr().err

        assert main(["report", str(tmp_path / "out"), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["arms"]["A"]["all"]["completed"]["k"] == 1

    def test_the_cli_needs_an_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        assert main(["run", "--arm", "A", "--out", str(tmp_path)]) == 2


def _small(dest: Path) -> Path:
    from evals.agent_notebook.uptake import workspace as ws

    return ws.build(dest, rows=1_000)
