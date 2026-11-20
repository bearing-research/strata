"""The uptake runner: arm setup, preflight, turns through ``--resume``, scoring, the
report and its intervals. A fake ``claude`` stands in for the agent."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from contextlib import contextmanager, suppress
from pathlib import Path

import httpx
import pytest

from evals.agent_notebook.uptake import report, run, server, stats
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


def _trial(tmp_path, claude, monkeypatch, task_id, arm, script, timeout=600.0, **env):
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
        timeout=timeout,
        pinned={"model": "fake"},
    )


def _gone(pid: int) -> bool:
    """Whether *pid* has exited and been reaped, given a few seconds to."""
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


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
        assert all("stdin-null=True" in call for call in calls)
        assert {c[c.index("--resume") + 1] for c in calls[1:]} == {trial.session_id}
        assert [t.passed for t in trial.turns] == [True, True, False, True]
        assert not trial.completed
        # The counter saw both loads, each in its own turn.
        assert [t.loads for t in trial.turns] == [1, 0, 1, 0]
        assert len(trial.loads) == 2
        saved = json.loads((tmp_path / "trial" / "trial.json").read_text())
        assert saved["completed"] is False and saved["strata_calls"] == 0

    def test_once_scored_it_keeps_its_record_but_not_its_venv(self, tmp_path, claude, monkeypatch):
        venv = tmp_path / "trial" / "harbor" / ".venv"
        build = run.prepare_workspace

        def synced(dest, **kw):
            work = build(dest, **kw)
            (work / ".venv").mkdir()
            return work

        monkeypatch.setattr(run, "prepare_workspace", synced)
        trial = _trial(tmp_path, claude, monkeypatch, "neutral_modpow", "A", [[None, "246"]])

        assert trial.discarded is None
        assert not venv.exists()
        assert (tmp_path / "trial" / "trial.json").exists()
        assert (tmp_path / "trial" / "turn0.jsonl").exists()

    def test_shrinking_removes_venvs_and_artifact_stores_anywhere(self, tmp_path):
        notebook = tmp_path / "scratchpad" / "notebooks" / "scratch"
        rebuilt = [
            tmp_path / "home" / ".strata" / "envs",
            tmp_path / "harbor" / ".venv" / "bin",
            tmp_path / "scratchpad" / "artifacts" / "blobs",
            notebook / ".venv" / "lib",
            notebook / ".strata" / "artifacts",
        ]
        for path in rebuilt:
            path.mkdir(parents=True)
        (notebook / "cells").mkdir()
        (notebook / "cells" / "a1b2c3d4.py").write_text("x = 1\n")
        agents_own = tmp_path / "harbor" / "artifacts"
        agents_own.mkdir()
        shared = tmp_path / "envs" / "abc"
        shared.mkdir(parents=True)
        (tmp_path / "harbor" / "packages").mkdir()
        (tmp_path / "harbor" / "packages" / ".venv").symlink_to(shared)

        run.shrink(tmp_path)

        assert [p for p in rebuilt if p.exists()] == []
        assert not (tmp_path / "harbor" / "packages" / ".venv").is_symlink()
        assert shared.exists() and agents_own.exists()
        assert (notebook / "cells" / "a1b2c3d4.py").read_text() == "x = 1\n"

    def test_a_store_that_will_not_go_is_reported_not_raised(self, tmp_path, capsys):
        store = tmp_path / "harbor" / ".venv"
        store.mkdir(parents=True)
        (store / "lib").mkdir()
        store.chmod(0o500)
        try:
            run.shrink(tmp_path)
        finally:
            store.chmod(0o700)

        assert f"could not remove {store}" in capsys.readouterr().err

    def test_an_edit_is_checked_in_the_files(self, tmp_path, claude, monkeypatch):
        fix = (
            f"{sys.executable} -c \"import pathlib; p = pathlib.Path('src/harbor/text.py');"
            " p.write_text(p.read_text().replace('[^a-z0-9]', '[^a-z0-9]+'))\""
        )

        done = _trial(tmp_path, claude, monkeypatch, "edit_fix_slugify", "B", [[fix, "Fixed."]])

        assert done.discarded is None and done.completed
        assert run.uses_strata({"name": "Skill", "input": {"skill": "strata-scratchpad"}})

    def test_its_answers_are_computed_before_the_agent_touches_the_data(
        self, tmp_path, claude, monkeypatch
    ):
        script = [["rm data/events.parquet", "Tuesday"]]

        trial = _trial(tmp_path, claude, monkeypatch, "plot_daily_events", "A", script)

        assert trial.completed

    def test_stdin_is_never_the_runners(self, tmp_path, claude, monkeypatch):
        # Run from a pipe that never closes, as from another agent's shell.
        read, write = os.pipe()
        saved = os.dup(0)
        os.dup2(read, 0)
        try:
            _trial(tmp_path, claude, monkeypatch, "neutral_modpow", "A", [[None, "246"]])
        finally:
            os.dup2(saved, 0)
            for fd in (saved, read, write):
                os.close(fd)

        (call,) = (tmp_path / "trial" / "config" / "calls.jsonl").read_text().splitlines()
        assert "stdin-null=True" in call

    def test_a_turn_past_its_timeout_is_discarded_and_saved(self, tmp_path, claude, monkeypatch):
        script = [[None, "x"]] * 4

        trial = _trial(
            tmp_path, claude, monkeypatch, "iter_west", "A", script, timeout=2, FAKE_HANG="1"
        )

        assert trial.discarded == "turn 1: timed out after 2 s"
        saved = json.loads((tmp_path / "trial" / "trial.json").read_text())
        assert saved["discarded"] == trial.discarded
        # The turn's whole group went with it, not only the agent's own process.
        child = int((tmp_path / "trial" / "config" / "hang.pid").read_text())
        assert _gone(child)

    def test_every_arms_trial_stops_the_scratchpad_daemon_on_its_project(
        self, tmp_path, claude, monkeypatch
    ):
        stopped = []
        monkeypatch.setattr(server, "stop_daemon", stopped.append)

        _trial(tmp_path, claude, monkeypatch, "neutral_modpow", "B", [[None, "246"]])

        assert stopped == [tmp_path / "trial" / "harbor"]

    def test_it_refuses_a_folder_whose_parents_reach_the_agent(self, tmp_path, claude):
        (tmp_path / "repo").mkdir()
        (tmp_path / "repo" / "CLAUDE.md").write_text("Use the strata scratchpad.")

        with pytest.raises(ValueError, match="CLAUDE.md would reach the agent"):
            run.run_trial(
                TASKS_BY_ID["neutral_modpow"],
                "A",
                0,
                tmp_path / "repo" / "runs" / "t",
                claude=claude,
                sync=False,
            )
        assert run.isolation_problem(tmp_path / "elsewhere" / "t") is None
        assert main(["run", "--arm", "A", "--out", str(tmp_path / "repo" / "runs")]) == 2

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

        def without_skill(arm, config, bin_dir, path):
            prepare(arm, config, bin_dir, path)
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
        run.prepare_arm("A", config, bin_dir, "")
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
        assert decide("strata-notebook --help") == 2
        assert decide("python3 -c 'from strata.notebook import x'") == 2
        assert decide("python3 -c 'print(1)'") == 0
        assert decide("grep -r stratagem .") == 0

    def test_every_strata_command_on_the_path_is_not_found(self, tmp_path):
        tools = tmp_path / "tools"
        tools.mkdir()
        for name in ("strata-notebook", "strata-worker", "uv"):
            (tools / name).write_text("#!/bin/sh\necho ran\n")
            (tools / name).chmod(0o755)
        bin_dir = tmp_path / "bin"
        run.prepare_arm("A", tmp_path / "config", bin_dir, str(tools))
        env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{tools}"}

        def code(name: str) -> int:
            return subprocess.run([name], env=env, capture_output=True).returncode

        assert (code("strata-notebook"), code("strata-worker"), code("uv")) == (127, 127, 0)

    def test_the_agents_path_leaves_out_the_evals_own_env(self, monkeypatch):
        own = str(Path(sys.prefix) / "bin")
        monkeypatch.delenv("VIRTUAL_ENV", raising=False)

        assert run.agent_path(f"{own}{os.pathsep}/usr/bin") == "/usr/bin"

    def test_the_agent_env_leaves_the_eval_process_out(self, tmp_path, monkeypatch):
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("VIRTUAL_ENV", "/repo/.venv")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://proxy")
        monkeypatch.setenv("STRATA_NOTEBOOK_SHARED_ENV_DIR", "/home/me/.strata/envs")

        env = run.agent_env(tmp_path / "c", tmp_path / "b", "/usr/bin", tmp_path / "l", 20)

        assert "CLAUDECODE" not in env and "VIRTUAL_ENV" not in env
        assert "STRATA_NOTEBOOK_SHARED_ENV_DIR" not in env
        assert "ANTHROPIC_BASE_URL" not in env and env["ANTHROPIC_API_KEY"] == "k"
        assert env["CLAUDE_CONFIG_DIR"] == str(tmp_path / "c")
        assert env["PATH"].startswith(str(tmp_path / "b"))


class TestArmCp:
    # As a real Cp turn's init event shows it (Claude Code 2.1.323).
    INIT = {
        "skills": ["debug"],
        "mcp_servers": [{"name": "strata", "status": "connected", "source": "user"}],
        "tools": ["Bash", "mcp__strata__run", "mcp__strata__scratchpad"],
    }

    def test_preflight_wants_its_server_connected_with_both_tools(self):
        assert run.preflight("Cp", self.INIT) is None
        failed = {**self.INIT, "mcp_servers": [{"name": "strata", "status": "failed"}]}
        assert run.preflight("Cp", failed).startswith("want only 'strata' connected")
        plugin = {
            **self.INIT,
            "mcp_servers": [
                *self.INIT["mcp_servers"],
                {"name": "plugin:strata-scratchpad:strata", "status": "connected"},
            ],
        }
        assert run.preflight("Cp", plugin).startswith("want only 'strata' connected")
        deferred = {**self.INIT, "tools": ["Bash", "mcp__strata__run"]}
        assert run.preflight("Cp", deferred) == (
            "the scratchpad tools are not loaded: ['mcp__strata__scratchpad']"
        )
        skilled = {**self.INIT, "skills": ["strata-scratchpad"]}
        assert run.preflight("Cp", skilled) == "the strata-scratchpad skill is loaded"
        assert run.preflight("Cpp", self.INIT) is None
        assert run.preflight("Cpp", deferred) == run.preflight("Cp", deferred)

    def test_its_turns_run_while_its_server_is_up(self, tmp_path, claude, monkeypatch):
        events = []

        @contextmanager
        def fake_server(root, project, agent_env, *, claude, load_log, load_seconds):
            calls = Path(agent_env["CLAUDE_CONFIG_DIR"]) / "calls.jsonl"
            events.append(("up", project.name, calls.exists()))
            yield "http://127.0.0.1:1"
            events.append(("down", len(calls.read_text().splitlines())))

        monkeypatch.setattr(server, "scratchpad_server", fake_server)

        trial = _trial(
            tmp_path,
            claude,
            monkeypatch,
            "neutral_modpow",
            "Cp",
            [[None, "246"]],
            FAKE_SERVERS="strata",
            FAKE_TOOLS="mcp__strata__run,mcp__strata__scratchpad",
        )

        assert trial.discarded is None and trial.turns[0].passed
        assert events == [("up", "harbor", False), ("down", 1)]

    def test_arm_cpp_runs_its_turns_inside_its_stdio_entry(self, tmp_path, claude, monkeypatch):
        events = []

        @contextmanager
        def fake_stdio(home, project, agent_env, *, claude, load_log, load_seconds):
            events.append(("up", home.relative_to(tmp_path).as_posix(), project.name))
            yield
            events.append(("down",))

        monkeypatch.setattr(server, "stdio_scratchpad", fake_stdio)

        trial = _trial(
            tmp_path,
            claude,
            monkeypatch,
            "neutral_modpow",
            "Cpp",
            [[None, "246"]],
            FAKE_SERVERS="strata",
            FAKE_TOOLS="mcp__strata__run,mcp__strata__scratchpad",
        )

        assert trial.discarded is None and trial.turns[0].passed
        assert events == [("up", "trial/home", "harbor"), ("down",)]

    def test_arm_c_runs_its_turns_after_the_install_on_a_home_of_its_own(
        self, tmp_path, claude, monkeypatch
    ):
        events = []

        @contextmanager
        def fake_install(home, project, agent_env, *, claude, load_log, load_seconds):
            bin_dir = Path(agent_env["PATH"].split(os.pathsep)[0])
            events.append(("up", home, agent_env["HOME"], (bin_dir / "strata").exists()))
            yield
            events.append(("down",))

        monkeypatch.setattr(server, "installed_scratchpad", fake_install)

        trial = _trial(
            tmp_path,
            claude,
            monkeypatch,
            "neutral_modpow",
            "C",
            [[None, "246"]],
            FAKE_SERVERS="strata",
            FAKE_TOOLS="mcp__strata__run,mcp__strata__scratchpad",
        )

        assert trial.discarded is None and trial.turns[0].passed
        home = tmp_path / "trial" / "home"
        assert events == [("up", home, str(home), True), ("down",)]
        assert run.preflight("C", self.INIT) is None

    def test_arm_c_without_uvs_dirs_is_discarded_not_raised(self, tmp_path, claude, monkeypatch):
        def no_uv(*args):
            raise subprocess.CalledProcessError(2, ["uv", "cache", "dir"])

        monkeypatch.setattr(server, "cell_env", no_uv)

        trial = _trial(tmp_path, claude, monkeypatch, "neutral_modpow", "C", [[None, "246"]])

        assert trial.discarded.startswith("setup: CalledProcessError") and not trial.turns

    def test_the_install_writes_a_users_config_into_the_trials(self, tmp_path, monkeypatch):
        # As a native install has it: `claude` a link to a file named for its version, and
        # its folder not on the agent's PATH.
        (tmp_path / "versions").mkdir()
        recorder = tmp_path / "versions" / "9.9.9"
        recorder.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{tmp_path}/args"\n')
        recorder.chmod(0o755)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        (bin_dir / "claude").symlink_to(recorder)
        claude = str(bin_dir / "claude")
        monkeypatch.setattr(server, "_warm_home", lambda home: None)
        warmed = []
        monkeypatch.setattr(server, "_warm_scratchpad", lambda *args: warmed.append(args))
        config, home = tmp_path / "config", tmp_path / "home"
        config.mkdir()
        env = {
            **server.host_env(),
            "PATH": f"/usr/bin{os.pathsep}/bin",
            "HOME": str(home),
            "CLAUDE_CONFIG_DIR": str(config),
        }
        kw = {"load_log": tmp_path / "l.jsonl", "load_seconds": 0}

        with server.installed_scratchpad(home, tmp_path, env, claude=claude, **kw):
            args = (tmp_path / "args").read_text().splitlines()
            assert args[:5] == ["mcp", "add-json", "--scope", "user", "strata"]
            assert Path(json.loads(args[5])["command"]).parent == Path(sys.executable).parent
            settings = json.loads((config / "settings.json").read_text())
            assert settings["permissions"]["allow"] == ["mcp__strata__*"]
            assert set(settings["hooks"]) == {"SessionStart", "UserPromptSubmit"}
            assert "mcp__strata__run" in (config / "CLAUDE.md").read_text()
            # The install's record is the trial's, not the runner's home.
            assert (home / ".strata" / "agent-install" / "state.json").exists()
            assert warmed == [(tmp_path, env)]

        recorder.write_text("#!/bin/sh\nexit 1\n")
        fresh = {**env, "HOME": str(tmp_path / "h2")}
        with pytest.raises(server.SetupError, match="strata agent install failed"):
            with server.installed_scratchpad(tmp_path / "h2", tmp_path, fresh, claude=claude, **kw):
                pass
        with pytest.raises(server.SetupError, match="not on PATH"):
            with server.installed_scratchpad(tmp_path / "h3", tmp_path, fresh, claude="nope", **kw):
                pass

    @pytest.mark.parametrize("attached", [True, False])
    def test_c_scratchpad_is_started_loaded_and_stopped_before_the_clock(
        self, tmp_path, monkeypatch, attached
    ):
        from strata import scratch
        from strata.notebook import env as notebook_env

        project, agent_env, steps = tmp_path / "harbor", {"HOME": "trial"}, []
        venv = ".venv-overlay" if attached else ".venv"
        python = str(project / ".strata-scratch" / venv / "bin" / "python")
        failing: set[str] = set()
        running = [False]

        def fake_run(command, *, cwd, env, **kw):
            assert (cwd, env) == (project, agent_env)
            name = "python" if command[1] == "-c" else command[2]
            steps.append([name, command[0]] if name == "python" else [name])
            return subprocess.CompletedProcess(command, int(name in failing), "", "boom")

        monkeypatch.setattr(server.subprocess, "run", fake_run)
        monkeypatch.setattr(
            notebook_env, "attached_identity", lambda home: b"id" if attached else None
        )
        monkeypatch.setattr(server, "stop_daemon", lambda where: steps.append(["stop", where]))
        monkeypatch.setattr(scratch, "find_daemon", lambda anchor: object() if running[0] else None)

        server._warm_scratchpad(project, agent_env)
        assert steps == [["start"], ["python", python], ["stop", project]]

        failing.add("python")
        steps.clear()
        with pytest.raises(server.SetupError, match=f"importing in {venv}"):
            server._warm_scratchpad(project, agent_env)
        assert ["stop", project] not in steps

        failing.clear()
        running[0] = True
        with pytest.raises(server.SetupError, match="its daemon did not stop"):
            server._warm_scratchpad(project, agent_env)

        def busy(anchor):
            raise scratch.NotAnswering("busy")

        monkeypatch.setattr(scratch, "find_daemon", busy)
        with pytest.raises(server.SetupError, match="its daemon did not stop"):
            server._warm_scratchpad(project, agent_env)

    def test_the_stdio_entry_is_strata_mcp_on_the_trials_home(self, tmp_path):
        command = server.stdio_entry_command(
            "claude", tmp_path / "harbor", tmp_path / "home", tmp_path / "l.jsonl", 20.0
        )

        assert command[:6] == ["claude", "mcp", "add-json", "-s", "user", "strata"]
        entry = json.loads(command[6])
        assert Path(entry["command"]).name == "strata" and Path(entry["command"]).exists()
        assert entry["args"] == ["mcp", "--root", str(tmp_path / "harbor")]
        assert entry["env"]["HOME"] == str(tmp_path / "home")
        assert entry["env"]["HARBOR_LOAD_LOG"] == str(tmp_path / "l.jsonl")
        assert (entry["type"], entry["alwaysLoad"]) == ("stdio", True)

    def test_the_stdio_entry_is_written_on_a_warmed_home(self, tmp_path, monkeypatch):
        recorder = tmp_path / "claude"
        recorder.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{tmp_path}/args"\n')
        recorder.chmod(0o755)
        warmed = []
        monkeypatch.setattr(server, "_warm_home", warmed.append)
        project = tmp_path / "harbor"
        kw = {"load_log": tmp_path / "l.jsonl", "load_seconds": 0}

        with server.stdio_scratchpad(
            tmp_path / "home", project, dict(os.environ), claude=str(recorder), **kw
        ):
            assert (tmp_path / "args").read_text().splitlines()[:3] == ["mcp", "add-json", "-s"]
            assert warmed == [tmp_path / "home"]

        with pytest.raises(server.SetupError, match="add-json failed"):
            with server.stdio_scratchpad(
                tmp_path / "h2", project, dict(os.environ), claude="false", **kw
            ):
                pass

    def test_a_warmed_home_has_matplotlibs_font_list(self, tmp_path):
        server._warm_home(tmp_path)

        assert list(tmp_path.rglob("fontlist-*.json"))

    def test_a_daemon_busy_past_the_probe_is_stopped_with_its_group(self, tmp_path, monkeypatch):
        from strata import scratch

        project = tmp_path / "harbor"
        (scratch.scratch_dir(project) / ".strata").mkdir(parents=True)
        # A daemon that ignores SIGTERM, with a cell process that does too.
        stubborn = "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        leader = subprocess.Popen(
            [
                sys.executable,
                "-c",
                stubborn + "import subprocess, sys; "
                f"c = subprocess.Popen([sys.executable, '-c', {stubborn + 'time.sleep(600)'!r}]); "
                "print(c.pid, flush=True); time.sleep(600)",
            ],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            assert leader.stdout is not None
            cell = int(leader.stdout.readline())
            record = scratch.scratch_dir(project) / ".strata" / "server.json"
            record.write_text(json.dumps({"pid": leader.pid}))

            def busy(anchor):
                raise scratch.NotAnswering("not answering")

            monkeypatch.setattr(scratch, "find_daemon", busy)
            server.stop_daemon(project, wait=0.5)

            assert leader.wait(timeout=10) == -signal.SIGKILL
            assert _gone(cell)
        finally:
            with suppress(ProcessLookupError, PermissionError):
                os.killpg(leader.pid, signal.SIGKILL)
            leader.wait()
            leader.stdout.close()

    def test_a_daemon_that_does_not_answer_is_left_and_odd_records_reported(
        self, tmp_path, monkeypatch, capsys
    ):
        from strata import scratch

        project = tmp_path / "harbor"
        (scratch.scratch_dir(project) / ".strata").mkdir(parents=True)
        signalled = []
        monkeypatch.setattr(server, "_end_group", lambda pid, wait: signalled.append(pid))
        record = scratch.scratch_dir(project) / ".strata" / "server.json"
        record.write_text(json.dumps({"pid": 4242}))

        # Gone: its pid may be another process's now.
        monkeypatch.setattr(scratch, "find_daemon", lambda anchor: None)
        server.stop_daemon(project)
        assert signalled == []

        monkeypatch.setattr(scratch, "find_daemon", lambda anchor: object())
        record.write_text("{}")
        server.stop_daemon(project)
        assert "did not stop: KeyError('pid')" in capsys.readouterr().err
        # No scratchpad at all: nothing to stop, no lock file made.
        server.stop_daemon(tmp_path / "untouched")
        assert not (tmp_path / "untouched").exists()

    def test_a_server_that_does_not_come_up_discards_the_trial(self, tmp_path, claude, monkeypatch):
        def broken(*args, **kwargs):
            raise server.SetupError("the server exited (1) waiting for health")

        monkeypatch.setattr(server, "scratchpad_server", broken)

        trial = _trial(tmp_path, claude, monkeypatch, "neutral_modpow", "Cp", [[None, "246"]])

        assert trial.discarded == "setup: the server exited (1) waiting for health"
        assert trial.turns == []
        saved = json.loads((tmp_path / "trial" / "trial.json").read_text())
        assert saved["discarded"] == trial.discarded

    def test_its_server_keeps_everything_under_its_root(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STRATA_ARTIFACT_DIR", "/home/me/.strata/artifacts")
        monkeypatch.setenv("STRATA_MCP_TOOLS", "all")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("PYTHONPATH", "/repo/src")
        monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/repo/.venv")
        root = tmp_path / "srv"

        env = server.server_env(root, 8799, tmp_path / "loads.jsonl", 20.0)

        stored = ("HOME", "STRATA_CACHE_DIR", "STRATA_ARTIFACT_DIR", "STRATA_METADATA_DB")
        stored += ("STRATA_NOTEBOOK_STORAGE_DIR", "STRATA_WORKER_ENV_ROOT")
        assert all(Path(env[name]).is_relative_to(root) for name in stored)
        assert "STRATA_MCP_TOOLS" not in env and "ANTHROPIC_API_KEY" not in env
        assert "PYTHONPATH" not in env and "UV_PROJECT_ENVIRONMENT" not in env
        assert (env["STRATA_MCP_ENABLED"], env["STRATA_MCP_PROFILE"]) == ("true", "scratchpad")
        assert (env["STRATA_HOST"], env["STRATA_PORT"]) == ("127.0.0.1", "8799")
        assert env["HARBOR_LOAD_LOG"] == str(tmp_path / "loads.jsonl")

    def test_the_notebook_is_warmed_under_the_servers_home(self, tmp_path, monkeypatch):
        # matplotlib's first import builds its font list under HOME: warmed under another,
        # each trial's first plot would pay it on the agent's clock.
        calls: list[tuple[list[str], dict[str, str] | None]] = []

        def ran(command, **kwargs):
            calls.append((command, kwargs.get("env")))
            return subprocess.CompletedProcess(command, 0, "", "")

        monkeypatch.setattr(server.subprocess, "run", ran)
        monkeypatch.setenv("HARBOR_LOAD_LOG", str(tmp_path / "loads.jsonl"))
        root = tmp_path / "srv"

        server._prepare_notebook(root, tmp_path / "project")

        (warm_env,) = [e for command, e in calls if command[1:] == ["-c", server._WARM]]
        assert warm_env is not None
        assert warm_env["HOME"] == str(server._home(root))
        # Nor does a load during the warm-up count as the trial's.
        assert "HARBOR_LOAD_LOG" not in warm_env

    @pytest.mark.parametrize(
        "failure",
        [
            KeyError("environment"),
            ValueError("Expecting value"),
            httpx.ReadTimeout("timed out"),
            subprocess.CalledProcessError(2, ["uv", "cache", "dir"]),
        ],
    )
    def test_a_failure_bringing_it_up_is_a_setup_error(self, failure):
        with pytest.raises(server.SetupError, match=type(failure).__name__):
            with server.as_setup_error():
                raise failure

    def test_its_own_refusals_and_other_errors_pass_through(self):
        with pytest.raises(server.SetupError, match="^uv add failed$"):
            with server.as_setup_error():
                raise server.SetupError("uv add failed")
        with pytest.raises(ZeroDivisionError):
            with server.as_setup_error():
                raise ZeroDivisionError

    def test_its_entry_is_user_scope_and_always_loaded(self):
        command = server.mcp_entry_command("claude", "http://127.0.0.1:8799")
        assert command[:6] == ["claude", "mcp", "add-json", "-s", "user", "strata"]
        assert json.loads(command[6]) == {
            "type": "http",
            "url": "http://127.0.0.1:8799/mcp/",
            "alwaysLoad": True,
        }

    def test_what_a_dead_server_started_is_stopped_with_it(self):
        # The leader exits and leaves a child in its group, as a crashed server leaves a pool.
        with subprocess.Popen(
            ["sh", "-c", "sleep 60 & echo $!"],
            stdout=subprocess.PIPE,
            text=True,
            start_new_session=True,
        ) as proc:
            child = int(proc.stdout.readline())
            proc.wait()
            server._stop_group(proc)

        def gone() -> bool:
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                return True
            except PermissionError:
                return False  # macOS: a zombie its new parent has not reaped yet
            return False

        # Its new parent reaps it a moment after the signal.
        deadline = time.monotonic() + 10
        while not gone() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert gone()

    def test_a_group_left_with_zombies_is_stopped_as_one_that_is_gone(self, monkeypatch):
        # macOS answers killpg with EPERM, not ESRCH, while the group's last members wait
        # to be reaped; it raced the server's SIGTERM in CI.
        with subprocess.Popen(["true"], start_new_session=True) as proc:
            proc.wait()
            signals = []

            def killpg(pgid, sig):
                signals.append(sig)
                raise PermissionError(1, "Operation not permitted")

            monkeypatch.setattr(server.os, "killpg", killpg)
            server._stop_group(proc)

        assert signals == [signal.SIGTERM, signal.SIGKILL]


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

    def test_an_arm_is_compared_with_a_on_the_tasks_both_ran(self):
        trials = [
            _saved("A", "iter_west", wall=100),
            _saved("A", "iter_january", wall=1000),
            _saved("B", "iter_west", wall=50),
        ]

        row = report.summarize(trials)["versus_control"]["B"]["iteration"]

        assert row["median_wall_ratio"][0] == 0.5

    def test_a_pin_other_than_strata_that_differs_is_a_warning(self):
        trials = [_saved("A", "iter_west"), _saved("Cp", "iter_west")]
        trials[0]["pins"] = {"claude_code": "2.1.326", "strata_commit": "aaa"}
        trials[1]["pins"] = {"claude_code": "2.1.327", "strata_commit": "bbb"}

        assert report.summarize(trials)["warnings"] == [
            "claude_code differs between trials: 2.1.326 (A), 2.1.327 (Cp)"
        ]

    def test_pins_of_discarded_runs_and_absent_pins_are_no_warning(self):
        trials = [
            _saved("A", "iter_west"),
            _saved("Cp", "iter_west"),
            _saved("Cp", "iter_west", discarded="turn 0: no init event"),
        ]
        trials[0]["pins"] = {"claude_code": "2.1.327", "uv": "0.13.1"}
        trials[1]["pins"] = {"claude_code": "2.1.327"}
        trials[2]["pins"] = {"claude_code": "2.1.326"}

        assert report.summarize(trials)["warnings"] == ["arm Cp: 1 of 2 runs discarded"]

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

    def test_the_cli_runs_several_arms_in_one_seeded_shuffle(
        self, tmp_path, claude, monkeypatch, capsys
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
        monkeypatch.setenv("FAKE_SCRIPT", json.dumps([[None, "246"]]))
        monkeypatch.setattr(run, "prepare_workspace", lambda dest, **kw: _small(dest))
        out = str(tmp_path / "out")
        argv = ["run", "--arm", "A", "--arm", "B", "--out", out, "--task", "neutral_modpow"]

        assert main([*argv, "--runs", "3", "--seed", "7", "--claude", claude]) == 0

        ran = [line.split(":")[0] for line in capsys.readouterr().out.splitlines()]
        assert sorted(ran) == sorted(f"{a} neutral_modpow #{i}" for a in "AB" for i in range(3))
        # Neither arm-major nor the plain round-robin a loop would give.
        arms = [line.split()[0] for line in ran]
        assert arms not in (["A"] * 3 + ["B"] * 3, ["B"] * 3 + ["A"] * 3, ["A", "B"] * 3)
        # The seed alone fixes the order.
        shutil.rmtree(tmp_path / "out")
        assert main([*argv, "--runs", "3", "--seed", "7", "--claude", claude]) == 0
        assert [line.split(":")[0] for line in capsys.readouterr().out.splitlines()] == ran

    def test_the_cli_needs_an_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        assert main(["run", "--arm", "A", "--out", str(tmp_path)]) == 2


def _small(dest: Path) -> Path:
    from evals.agent_notebook.uptake import workspace as ws

    return ws.build(dest, rows=1_000)
