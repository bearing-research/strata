"""The field metric reads Claude Code transcripts into what agents did on this machine."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from evals.agent_notebook import field
from evals.agent_notebook.field import build_report, classify_bash, read_session


@pytest.fixture(autouse=True)
def _eval_runs_under_private_tmp(monkeypatch):
    # The transcripts' projects live in pytest's temp folder, which may be under a real
    # eval root (/private/tmp on macOS); eval runs here live under a root nothing else uses.
    monkeypatch.setattr(field, "_TEMP_ROOTS", ("/eval-runs/",))


@pytest.fixture
def project(tmp_path) -> Path:
    """A project whose package is ``proj``."""
    root = tmp_path / "proj-repo"
    (root / "src" / "proj").mkdir(parents=True)
    (root / "src" / "proj" / "__init__.py").write_text("")
    (root / "pyproject.toml").write_text('[project]\nname = "proj-repo"\n')
    field.project_modules.cache_clear()
    return root


def _kinds(command: str, cwd: str = "/nowhere") -> dict[str, int]:
    return dict(classify_bash(command, cwd))


class TestABashCall:
    def test_inline_python_is_compute(self):
        assert _kinds('python3 -c "print(1 + 1)"') == {"compute": 1}
        assert _kinds("python3 - <<'EOF'\nprint(2)\nEOF") == {"compute": 1}
        assert _kinds("python <<EOF\nprint(2)\nEOF") == {"compute": 1}

    def test_in_the_project_env_importing_its_code(self, project):
        command = 'uv run python -c "import proj; print(proj)"'

        assert _kinds(command, str(project)) == {
            "compute": 1,
            "compute_project_env": 1,
            "compute_imports_project": 1,
        }
        assert (
            _kinds(".venv/bin/python -c 'from proj import x'", str(project))[
                "compute_imports_project"
            ]
            == 1
        )
        # The env, but not the project's code.
        assert "compute_imports_project" not in _kinds(
            'uv run python -c "import json"', str(project)
        )

    def test_the_project_is_where_the_command_cds_to(self, project, tmp_path):
        command = f'cd {project} && uv run python -c "import proj"'

        assert _kinds(command, str(tmp_path))["compute_imports_project"] == 1
        # Run from elsewhere, `proj` is no project's code.
        assert "compute_imports_project" not in _kinds(
            'uv run python -c "import proj"', str(tmp_path)
        )

    def test_a_cd_ends_at_its_separator_or_line(self, project, tmp_path):
        for command in (
            f'cd {project}; uv run python -c "import proj"',
            f"cd {project}\nuv run python - <<'EOF'\nimport proj\nEOF",
        ):
            assert _kinds(command, str(tmp_path))["compute_imports_project"] == 1, command

    def test_a_uv_managed_interpreter_is_not_the_project_env(self):
        command = "~/.local/share/uv/python/cpython-3.12/bin/python3 -c 'print(1)'"

        assert _kinds(command) == {"compute": 1}

    def test_a_script_that_writes_files_is_an_edit_though_it_prints(self):
        command = (
            "python3 - <<'EOF'\nfrom pathlib import Path\np = Path('a.py')\n"
            "p.write_text(p.read_text().replace('a', 'b'))\nprint('ok')\nEOF"
        )

        assert _kinds(command) == {"python_file_edits": 1}
        assert _kinds("python3 -c \"open('a', 'w').write('x')\"") == {"python_file_edits": 1}

    def test_a_json_pipe_is_not_compute(self):
        assert _kinds("gh api x | python3 -c 'import json,sys; print(json.load(sys.stdin))'") == {
            "json_pipes": 1
        }
        assert _kinds("cat a.json | python3 -m json.tool") == {"json_pipes": 1}

    def test_a_cat_heredoc_and_the_scratchpad_cli(self):
        assert _kinds("cat > a.py <<'EOF'\nx = 1\nEOF") == {"cat_heredocs": 1}
        assert _kinds("strata cell add ./scratch -c 'x = 1' --run") == {"cli_work": 1}

    def test_python_not_from_inline_source_is_nothing(self):
        assert _kinds("python3 script.py") == {}
        assert _kinds("uv run pytest tests") == {}


def _transcript(path: Path, cwd: str, *calls: tuple[str, str, dict]) -> Path:
    """A transcript at *path*: a user record, then one assistant tool call per *calls*
    entry of (timestamp, tool name, input)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [{"type": "user", "cwd": cwd, "timestamp": calls[0][0] if calls else ""}]
    for timestamp, name, args in calls:
        records.append(
            {
                "type": "assistant",
                "cwd": cwd,
                "timestamp": timestamp,
                "message": {"content": [{"type": "tool_use", "name": name, "input": args}]},
            }
        )
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


class TestASession:
    def test_its_tool_calls_are_counted(self, tmp_path, project):
        path = _transcript(
            tmp_path / "p" / "s.jsonl",
            str(project),
            ("2026-09-01T10:00:00Z", "Bash", {"command": 'uv run python -c "import proj"'}),
            ("2026-09-01T10:01:00Z", "mcp__strata-notebook__run_cell", {}),
            ("2026-09-01T10:02:00Z", "mcp__codegraph__codegraph_explore", {}),
            ("2026-09-03T09:00:00Z", "Skill", {"skill": "strata-scratchpad"}),
        )

        session = read_session(path)

        assert (session.compute, session.compute_imports_project) == (1, 1)
        assert (session.mcp_calls, session.codegraph, session.skill_triggers) == (1, 1, 1)
        # Dated by its own record, not the session's start.
        assert session.skill_trigger_days == ["2026-09-03"]
        assert not session.subagent

    def test_a_call_runs_from_its_own_records_cwd(self, tmp_path, project):
        path = _transcript(
            tmp_path / "p" / "s.jsonl",
            str(tmp_path),
            ("2026-09-01T10:00:00Z", "Bash", {"command": 'uv run python -c "import proj"'}),
        )
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[-1]["cwd"] = str(project)  # the session cd'd into the project first
        path.write_text("".join(json.dumps(r) + "\n" for r in records))

        assert read_session(path).compute_imports_project == 1

    def test_a_subagents_transcript_is_one(self, tmp_path):
        path = _transcript(tmp_path / "p" / "s" / "subagents" / "agent-1.jsonl", "/w")

        assert read_session(path).subagent


class TestAReport:
    def test_eval_runs_and_sessions_outside_the_window_are_left_out(self, tmp_path):
        python = ("Bash", {"command": "python3 -c 'print(1)'"})
        _transcript(tmp_path / "a" / "1.jsonl", "/w/a", ("2026-09-02T00:00:00Z", *python))
        _transcript(tmp_path / "a" / "2.jsonl", "/w/a", ("2026-08-30T00:00:00Z", *python))
        _transcript(
            tmp_path / "b" / "3.jsonl", "/eval-runs/run-1", ("2026-09-02T00:00:00Z", *python)
        )
        _transcript(
            tmp_path / "a" / "1" / "subagents" / "agent-1.jsonl",
            "/w/a",
            ("2026-09-08T00:00:00Z", *python),
            ("2026-09-08T00:01:00Z", "mcp__strata__run", {}),
        )

        report = build_report(field.sessions(tmp_path), since=date(2026, 9, 1))

        assert (report.sessions, report.subagent_sessions) == (1, 1)
        assert report.eval_sessions_left_out == 1
        assert report.compute == 2
        # Weeks 36 and 37: one session with compute work each; the second used Strata.
        assert report.weekly == {"2026-W36": [1, 0], "2026-W37": [1, 1]}

    def test_it_samples_bypasses_in_the_project_env(self, tmp_path, project):
        commands = [f'uv run python -c "print({i})"' for i in range(5)]
        _transcript(
            tmp_path / "a" / "1.jsonl",
            str(project),
            *[("2026-09-02T00:00:00Z", "Bash", {"command": c}) for c in commands],
        )

        report = build_report(field.sessions(tmp_path), examples=3)

        assert len(report.examples) == 3
        assert set(report.examples) <= set(commands)

    def test_the_cli_prints_text_or_json(self, tmp_path, capsys):
        _transcript(tmp_path / "a" / "1.jsonl", "/w", ("2026-09-02T00:00:00Z", "Bash", {}))

        assert field.main(["--root", str(tmp_path), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["sessions"] == 1
        assert field.main(["--root", str(tmp_path)]) == 0
        assert "sessions: 1 main" in capsys.readouterr().out
