"""The flagship workflow (design §3, P4's acceptance test), scripted against a real daemon in
a project with its own environment and package: load a dataset, answer three questions,
edit a project function one derived cell uses, see only that cell and its downstream rerun,
stop the daemon, and in a new session answer a fourth question without loading again."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from strata import agent_hook, scratch
from strata.cli import main
from strata.notebook.project_env import venv_python

DATA = """\
from pathlib import Path

LOG = Path(__file__).resolve().parents[2] / "calls.log"


def load_events():
    with LOG.open("a") as log:
        log.write("load_events\\n")
    return {"user": ["a", "b", "a", "c", "b", "a"], "value": [3, 5, 2, 8, 1, 4]}
"""

FEATURES = """\
from pathlib import Path

LOG = Path(__file__).resolve().parents[2] / "calls.log"


def totals(events):
    with LOG.open("a") as log:
        log.write("totals\\n")
    out = {{}}
    for user, value in zip(events["user"], events["value"]):
        out[user] = out.get(user, 0) + {term}
    return out
"""


def _project(root: Path) -> None:
    """A project with a ``src`` package and a ``.venv`` that holds it and, through a ``.pth``,
    the test's packages, so nothing is installed."""
    (root / "pyproject.toml").write_text("")
    package = root / "src" / "proj"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "data.py").write_text(DATA)
    (package / "features.py").write_text(FEATURES.format(term="value"))
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    venv = root / ".venv"
    subprocess.run(
        ["uv", "venv", "-q", "--python", sys._base_executable, str(venv)], check=True, env=env
    )
    site = subprocess.run(
        [str(venv_python(venv)), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    (Path(site) / "proj.pth").write_text(f"{root / 'src'}\n{sysconfig.get_path('purelib')}\n")


def _cli(capsys, *args: str) -> dict:
    code = main([*args])
    out = json.loads(capsys.readouterr().out)
    assert code == 0, out
    return out


def _calls(root: Path) -> list[str]:
    return (root / "calls.log").read_text().splitlines()


def _held(capsys, root: str) -> dict[str, dict]:
    """What ``scratchpad`` lists, by cell id."""
    listed = _cli(capsys, "scratch", "show", "--root", root)
    return {cell["cell"]: cell for cell in listed["cells"]}


def _hook(root: Path) -> str:
    """The session-start hook, as Claude Code runs it, with no daemon up."""
    done = subprocess.run(
        [sys.executable, "-I", agent_hook.__file__, "session-start"],
        input=json.dumps({"cwd": str(root)}),
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout


def test_the_flagship_workflow(project, monkeypatch, capsys):
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    _project(project)
    root = str(project)

    def run(source: str, cell: str | None = None) -> dict:
        extra = ["--cell", cell] if cell else []
        out = _cli(capsys, "scratch", "run", "-c", source, *extra, "--root", root)
        assert out["status"] == "ok", out
        return out

    # Session 1: load once, three questions, one of them through a project function.
    load = run("from proj.data import load_events\nevents = load_events()")
    count = run("n_events = len(events['user'])\nn_events")
    peak = run("peak = max(events['value'])\npeak")
    derive = run("from proj.features import totals\nby_user = totals(events)")
    top = run("top = max(by_user, key=by_user.get)\nf'{top}={by_user[top]}'")
    assert (count["value"], peak["value"]) == (6, 8)
    assert top["value"] == "a=9"
    assert _calls(project) == ["load_events", "totals"]
    ids = [out["cell"] for out in (load, count, peak, derive, top)]

    # Edit the function the derived cell uses: it and its downstream go stale, no other.
    features = project / "src" / "proj" / "features.py"
    features.write_text(FEATURES.format(term="1"))
    held = _held(capsys, root)
    assert {cell: held[cell]["status"] for cell in ids} == dict(
        zip(ids, ["ready", "ready", "ready", "stale", "stale"], strict=True)
    )
    assert held[derive["cell"]]["changed_code"] == str(features.resolve())

    # Asking again reruns the derived cell first; the load stays cached.
    again = run("top = max(by_user, key=by_user.get)\nf'{top}={by_user[top]}'", cell=top["cell"])
    assert [step["cell"] for step in again["reran"]] == [derive["cell"]]
    assert again["value"] == "a=3"
    assert not again["cache_hit"]
    for cell, source in [
        (count["cell"], "n_events = len(events['user'])\nn_events"),
        (peak["cell"], "peak = max(events['value'])\npeak"),
    ]:
        assert run(source, cell=cell)["cache_hit"]
    assert _calls(project) == ["load_events", "totals", "totals"]

    # The daemon stops; the next session starts with no daemon and is told what is held.
    assert main(["scratch", "stop", "--root", root]) == 0
    capsys.readouterr()
    assert scratch.find_daemon(project) is None
    told = _hook(project).splitlines()[1:-1]
    assert [line.partition("(cell ")[2][:8] for line in told] == ids, told
    assert all(line.endswith(", cached") for line in told), told

    # Session 2: every cell current, and a fourth question that loads nothing.
    held = _held(capsys, root)
    assert [held[cell]["status"] for cell in ids] == ["ready"] * 5, held
    fourth = run("low = min(events['value'])\nlow")
    assert fourth["value"] == 1
    assert "reran" not in fourth
    assert _calls(project) == ["load_events", "totals", "totals"]
