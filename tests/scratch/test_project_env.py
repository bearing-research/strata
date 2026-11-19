"""A scratchpad runs in its project's environment when it has one (design 4.2)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import sysconfig
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from strata import cli, scratch
from strata.notebook import mcp_server
from strata.notebook.annotation_validation import validate_cell_annotations
from strata.notebook.cell_paths.python import python_harness_command
from strata.notebook.env import ATTACHED_IDENTITY, OVERLAY_DIR, compute_lockfile_hash
from strata.notebook.env_backend import UvBackend, get_backend
from strata.notebook.models import CellState, DiagnosticSeverity
from strata.notebook.parser import parse_notebook
from strata.notebook.project_env import (
    PYTHON_OVERRIDE,
    ProjectEnvBackend,
    detect,
    parse_dry_run,
    reconcile,
    venv_python,
)
from strata.notebook.session import SessionManager
from strata.notebook.writer import _NOTEBOOK_RUNTIME_PACKAGES, create_notebook
from strata.scratch import SCRATCH_DIR

_PYPROJECT = """\
[project]
name = "{name}"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = {deps}
"""


def _pyproject(directory: Path, name: str = "proj", deps: str = "[]", extra: str = "") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "pyproject.toml").write_text(_PYPROJECT.format(name=name, deps=deps) + extra)


def _venv(path: Path) -> Path:
    """An empty venv on the test's base interpreter; its site-packages."""
    subprocess.run(
        ["uv", "venv", "-q", "--python", sys._base_executable, str(path)],
        check=True,
        env={k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"},
    )
    return Path(
        subprocess.run(
            [
                str(venv_python(path)),
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project whose environment holds a module of its own, and reaches the test's
    packages (the runtime ones among them) through a ``.pth``, so nothing is installed."""
    monkeypatch.delenv(PYTHON_OVERRIDE, raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    root = tmp_path / "proj"
    root.mkdir()
    site = _venv(root / ".venv")
    (site / "projmod.py").write_text("VALUE = 7\n")
    (site / "test_packages.pth").write_text(sysconfig.get_path("purelib") + "\n")
    return root


def _record(notebook: Path) -> dict:
    with open(notebook / "notebook.toml", "rb") as f:
        return tomllib.load(f).get("scratchpad_environment")


def _scratchpad(anchor: Path) -> Path:
    return create_notebook(
        anchor,
        "scratch",
        directory=SCRATCH_DIR,
        initialize_environment=False,
        write_gitignore_file=False,
        scratchpad=True,
    )


@pytest.fixture
async def manager():
    """Sessions whose warm processes are released before the test's loop closes."""
    sm = SessionManager()
    yield sm
    for session_id in list(sm._sessions):
        await asyncio.gather(*sm.close_session(session_id))


@pytest.fixture
def _server(monkeypatch):
    config = SimpleNamespace(
        host="0.0.0.0",
        port=8765,
        public_base_url=None,
        public_base_path="",
        principal_auth_enabled=False,
    )
    monkeypatch.setattr(mcp_server, "_server_config", lambda: config)


# --- the dry run ---


def _dry(sync_action="check", changes=(), lock_action="check") -> str:
    return json.dumps(
        {
            "schema": {"version": "preview"},
            "project": {"path": "/w/services/api", "workspace": {"path": "/w"}},
            "sync": {
                "environment": {"path": "/w/.venv"},
                "action": sync_action,
                "changes": list(changes),
            },
            "lock": {"path": "/w/uv.lock", "action": lock_action},
            "dry_run": True,
        }
    )


@pytest.mark.parametrize(
    ("output", "pending"),
    [
        (_dry(), False),
        (_dry(sync_action="create"), True),
        (_dry(changes=[{"name": "six", "action": "installed"}]), True),
        (_dry(lock_action="update"), True),
    ],
)
def test_the_dry_run_names_the_workspace_the_environment_and_whether_it_is_behind(output, pending):
    assert parse_dry_run(output, "", Path("/w/services/api")) == (
        Path("/w"),
        Path("/w/.venv"),
        pending,
    )


def test_without_its_json_the_text_line_names_the_environment_and_a_sync_runs():
    stderr = "Using CPython 3.14.5\nWould create project environment at: .venv\nResolved 3\n"
    assert parse_dry_run("", stderr, Path("/w")) == (Path("/w"), Path("/w/.venv"), True)
    assert parse_dry_run("not json", "nothing useful", Path("/w")) is None


def test_uv_reports_a_members_workspace_and_environment_and_creates_nothing(tmp_path):
    """The fields read, on the uv the suite runs: the JSON is marked preview."""
    root = tmp_path / "w"
    build = '\n[build-system]\nrequires = ["uv_build>=0.8,<1"]\nbuild-backend = "uv_build"\n'
    _pyproject(root, "w", extra='\n[tool.uv.workspace]\nmembers = ["packages/*", "services/*"]\n')
    _pyproject(root / "packages" / "common", "common", extra=build)
    (root / "packages/common/src/common").mkdir(parents=True)
    (root / "packages/common/src/common/__init__.py").write_text("")
    api = root / "services" / "api"
    _pyproject(
        api, "api", deps='["common"]', extra="\n[tool.uv.sources]\ncommon = { workspace = true }\n"
    )
    before = sorted(p.relative_to(root) for p in root.rglob("*"))

    found = detect(api, {})

    assert found.mode == "attached" and found.how == "uv" and found.sync
    assert found.python == venv_python(root / ".venv")
    assert root / "pyproject.toml" in found.watched
    assert sorted(p.relative_to(root) for p in root.rglob("*")) == before


# --- detection ---


def test_a_lock_at_the_workspace_root_makes_it_uv_managed_and_watches_each_member(tmp_path):
    root = tmp_path / "w"
    build = '\n[build-system]\nrequires = ["uv_build>=0.8,<1"]\nbuild-backend = "uv_build"\n'
    _pyproject(
        root,
        "w",
        deps='["lib"]',
        extra='\n[tool.uv.sources]\nlib = { path = "libs/lib", editable = true }\n',
    )
    _pyproject(root / "libs" / "lib", "lib", extra=build)
    subprocess.run(["uv", "lock", "-q", "--offline"], cwd=root, check=True)
    (root / "pyproject.toml").write_text(
        (root / "pyproject.toml").read_text().replace("[tool.uv.sources]", "[tool.other]")
    )
    assert "[tool.uv" not in (root / "pyproject.toml").read_text()

    found = detect(root, {})

    assert (found.mode, found.how) == ("attached", "uv")
    assert root / "libs" / "lib" / "pyproject.toml" in found.watched
    assert root / "uv.lock" in found.watched


def test_a_uv_project_uv_cannot_plan_is_still_its_own_and_its_sync_says_why(tmp_path):
    _pyproject(tmp_path, extra="\n[tool.uv]\n")
    (tmp_path / "uv.lock").write_text("not a lock")
    found = detect(tmp_path, {})
    assert (found.mode, found.how, found.sync) == ("attached", "uv", True)
    assert found.python == venv_python(tmp_path / ".venv")


def test_a_tool_uv_table_alone_makes_it_uv_managed(tmp_path):
    _pyproject(tmp_path, extra="\n[tool.uv]\npackage = false\n")
    found = detect(tmp_path, {})
    assert (found.mode, found.how, found.sync) == ("attached", "uv", True)


def test_a_project_uv_does_not_manage_and_has_no_environment_gets_its_own(tmp_path):
    _pyproject(tmp_path)
    found = detect(tmp_path, {})
    assert (found.mode, found.python) == ("owned", None)
    # The dry run ran, and made nothing: no lock, no environment.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pyproject.toml"]


def test_a_poetry_projects_environment_is_attached_as_it_is(tmp_path):
    _pyproject(tmp_path, extra='\n[tool.poetry]\nname = "proj"\n')
    _venv(tmp_path / ".venv")
    found = detect(tmp_path, {})
    assert (found.mode, found.how, found.sync) == ("attached", ".venv", False)
    assert found.python == venv_python(tmp_path / ".venv")


def test_an_active_environment_is_attached_unless_it_is_stratas_own(tmp_path):
    active = tmp_path / "elsewhere"
    _venv(active)
    found = detect(tmp_path, {"VIRTUAL_ENV": str(active)})
    assert (found.mode, found.how, found.python) == ("attached", "VIRTUAL_ENV", venv_python(active))
    # What `uv run` and uvx set for the process Strata runs in.
    assert detect(tmp_path, {"VIRTUAL_ENV": sys.prefix}).mode == "owned"


def test_the_override_wins_and_is_never_synced(tmp_path):
    _pyproject(tmp_path, extra="\n[tool.uv]\n")
    found = detect(tmp_path, {PYTHON_OVERRIDE: "/opt/py/bin/python"})
    assert (found.mode, found.how, found.sync) == ("attached", PYTHON_OVERRIDE, False)
    assert found.python == Path("/opt/py/bin/python")


# --- the overlay ---


def test_the_overlay_runs_the_projects_code_and_installs_nothing_the_project_has(project):
    notebook = _scratchpad(project)
    overlay = reconcile(notebook, venv_python(project / ".venv"), _NOTEBOOK_RUNTIME_PACKAGES)

    assert not isinstance(overlay, str)
    assert overlay.python == venv_python(notebook / OVERLAY_DIR)
    ran = subprocess.run(
        [
            str(overlay.python),
            "-c",
            "import projmod, pyarrow, sys; print(projmod.VALUE, sys.prefix)",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert ran == ["7", str(notebook / OVERLAY_DIR)]
    state = json.loads((notebook / OVERLAY_DIR / ".strata-overlay.json").read_text())
    assert state["installed"] == []


def test_the_overlay_fills_a_gap_until_the_project_has_it_then_the_projects_wins(tmp_path):
    """Bare names, after the project's site-packages: nothing shadows the project."""
    project_site = _venv(tmp_path / "proj" / ".venv")
    project_python = venv_python(tmp_path / "proj" / ".venv")
    notebook = _scratchpad(tmp_path / "proj")
    where = "import toolz; print(toolz.__file__)"

    first = reconcile(notebook, project_python, ("toolz",))
    assert not isinstance(first, str)
    overlay_site = Path(
        subprocess.run(
            [str(first.python), "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    assert subprocess.run(
        [str(first.python), "-c", where], capture_output=True, text=True, check=True
    ).stdout.startswith(str(overlay_site))

    subprocess.run(
        ["uv", "pip", "install", "-q", "--python", str(project_python), "toolz"], check=True
    )
    second = reconcile(notebook, project_python, ("toolz",))

    assert not isinstance(second, str)
    assert subprocess.run(
        [str(second.python), "-c", where], capture_output=True, text=True, check=True
    ).stdout.startswith(str(project_site))


def test_the_identity_is_the_key_basis_until_the_scratchpad_owns_its_environment(project):
    notebook = _scratchpad(project)
    owned = compute_lockfile_hash(notebook)
    overlay = reconcile(notebook, venv_python(project / ".venv"), _NOTEBOOK_RUNTIME_PACKAGES)
    assert not isinstance(overlay, str)
    attached = compute_lockfile_hash(notebook)
    assert attached != owned

    (notebook / OVERLAY_DIR / ATTACHED_IDENTITY).write_text("another environment")
    assert compute_lockfile_hash(notebook) not in (owned, attached)
    (notebook / OVERLAY_DIR / ATTACHED_IDENTITY).unlink()
    assert compute_lockfile_hash(notebook) == owned


def test_a_reconcile_that_fails_leaves_no_identity_keying_cells(project, tmp_path):
    notebook = _scratchpad(project)
    assert not isinstance(
        reconcile(notebook, venv_python(project / ".venv"), _NOTEBOOK_RUNTIME_PACKAGES), str
    )
    failed = reconcile(notebook, tmp_path / "gone" / "python", _NOTEBOOK_RUNTIME_PACKAGES)
    assert isinstance(failed, str) and "does not exist" in failed
    assert not (notebook / OVERLAY_DIR / ATTACHED_IDENTITY).exists()


# --- the backend ---


def test_a_dependency_change_is_refused_with_the_projects_command(tmp_path):
    _pyproject(tmp_path, extra="\n[tool.uv]\n")
    backend = ProjectEnvBackend(_scratchpad(tmp_path))
    added = backend.add("numpy", timeout=10)
    assert not added.success
    assert "runs in your project's environment" in (added.error or "")
    assert f"`uv add numpy` in {tmp_path}" in (added.error or "")
    assert "`uv remove numpy`" in (backend.remove("numpy", timeout=10).error or "")


def test_a_project_change_or_an_install_means_syncing_again(project):
    _pyproject(project, extra="\n[tool.poetry]\n")
    backend = ProjectEnvBackend(_scratchpad(project), detect(project, {}))
    assert backend.changed()  # never synced
    assert backend.sync(python_version=None, timeout=60).success
    assert not backend.changed()

    (project / "pyproject.toml").write_text((project / "pyproject.toml").read_text() + "\n")
    assert backend.changed()
    assert backend.sync(python_version=None, timeout=60).success
    assert not backend.changed()

    site = next((project / ".venv" / "lib").glob("python*/site-packages"))
    (site / "installed_later.py").write_text("")
    assert backend.changed()


def test_the_record_chooses_the_backend(tmp_path):
    notebook = _scratchpad(tmp_path)
    assert isinstance(get_backend(notebook), UvBackend)
    text = (notebook / "notebook.toml").read_text()
    (notebook / "notebook.toml").write_text(
        text + '\n[scratchpad_environment]\nmode = "attached"\nattach = "/p/bin/python"\n'
    )
    assert isinstance(get_backend(notebook), ProjectEnvBackend)
    assert parse_notebook(notebook).scratchpad


def test_an_overlay_interpreter_starts_the_harness_itself(tmp_path):
    overlay = venv_python(tmp_path / OVERLAY_DIR)
    own = tmp_path / ".venv" / "bin" / "python"
    direct = python_harness_command(Path("h.py"), tmp_path, Path("m.json"), overlay, None)
    assert direct == [str(overlay), "h.py", "m.json"]
    through_uv = python_harness_command(Path("h.py"), tmp_path, Path("m.json"), own, None)
    assert through_uv is not None and through_uv[1:4] == ["run", "--directory", str(tmp_path)]


# --- the session ---


@pytest.mark.asyncio
@pytest.mark.usefixtures("_server")
async def test_a_scratchpad_runs_its_cells_in_its_projects_environment(project, manager):
    sm = manager
    session = sm.open_notebook(_scratchpad(project))

    assert isinstance(session.backend, ProjectEnvBackend)
    assert session.venv_python == venv_python(session.path / OVERLAY_DIR)
    record = _record(session.path)
    assert record == {"mode": "attached", "attach": str(venv_python(project / ".venv"))}

    out = await mcp_server._scratch_run(sm, "import projmod, sys\nprint(projmod.VALUE, sys.prefix)")
    assert out["status"] == "ok", out
    assert out["stdout"].split() == ["7", str(session.path / OVERLAY_DIR)]
    # Never the scratchpad's own environment.
    assert not (session.path / ".venv").exists()
    assert mcp_server._scratchpad(sm)["environment"] == {
        "mode": "attached",
        "project_python": str(venv_python(project / ".venv")),
        "found_by": ".venv",
    }


@pytest.mark.asyncio
@pytest.mark.usefixtures("_server")
async def test_a_project_that_will_not_sync_runs_no_cell(tmp_path, monkeypatch, manager):
    monkeypatch.delenv(PYTHON_OVERRIDE, raising=False)
    monkeypatch.setenv("UV_OFFLINE", "1")
    _pyproject(tmp_path, deps='["strata-no-such-package-anywhere"]', extra="\n[tool.uv]\n")
    sm = manager
    session = sm.open_notebook(_scratchpad(tmp_path))

    assert session.environment_sync_state == "failed"
    assert session.venv_python is None
    assert "strata-no-such-package-anywhere" in (session.environment_sync_error or "")
    out = await mcp_server._scratch_run(sm, "x = 1")
    assert out["status"] == "error"
    assert out["error"].startswith("Notebook environment is not ready. The project's environment")
    assert out["fix"].startswith(f"fix the project's environment in {tmp_path}")
    assert session.notebook_state.cells == []


@pytest.mark.asyncio
@pytest.mark.usefixtures("_server")
async def test_a_changed_project_is_synced_again_before_the_next_run(project, monkeypatch, manager):
    manager.open_notebook(_scratchpad(project))
    synced: list[bool] = []

    async def sync_environment(**_):
        synced.append(True)
        return {}

    session = next(iter(manager._sessions.values()))
    monkeypatch.setattr(session, "sync_environment", sync_environment)
    assert (await mcp_server._scratch_run(manager, "x = 1"))["status"] == "ok"
    assert synced == []
    (project / "pyproject.toml").write_text('[project]\nname = "proj"\nversion = "0.2.0"\n')
    assert (await mcp_server._scratch_run(manager, "y = 2"))["status"] == "ok"
    assert synced == [True]


@pytest.mark.asyncio
async def test_the_first_sync_of_a_deferred_open_finds_the_project(project, manager):
    """How the daemon opens a scratchpad: the sync is a background job."""
    session = manager.open_notebook(_scratchpad(project), defer_initial_venv_sync=True)
    assert isinstance(session.backend, UvBackend)
    await session.environment_jobs.submit(action="sync")
    await session.environment_jobs.wait()
    assert isinstance(session.backend, ProjectEnvBackend)
    assert session.environment_sync_state == "ready"
    assert session.venv_python == venv_python(session.path / OVERLAY_DIR)


def test_a_project_that_loses_its_environment_gives_the_scratchpad_its_own(project):
    session = SessionManager().open_notebook(_scratchpad(project))
    assert (session.path / OVERLAY_DIR).exists()
    import shutil

    shutil.rmtree(project / ".venv")
    session.detect_scratchpad_environment()

    assert isinstance(session.backend, UvBackend)
    assert not (session.path / OVERLAY_DIR).exists()
    assert _record(session.path) == {"mode": "owned"}


# --- cells may not move imports ---


@pytest.mark.parametrize("scratchpad", [True, False])
def test_a_scratchpad_cell_may_not_set_where_imports_resolve(tmp_path, scratchpad):
    notebook = (
        _scratchpad(tmp_path)
        if scratchpad
        else create_notebook(tmp_path, "nb", initialize_environment=False)
    )
    state = parse_notebook(notebook)
    source = "# @env FOO=1\n# @env PYTHONPATH=/elsewhere\nimport x\n"
    cell = CellState(id="c1", source=source)
    errors = [
        d for d in validate_cell_annotations(cell, state) if d.severity == DiagnosticSeverity.ERROR
    ]
    if not scratchpad:
        assert errors == []
        return
    assert [(d.code, d.line) for d in errors] == [("env_moves_imports", 2)]
    assert "`@env PYTHONPATH`" in errors[0].message


# --- the daemon ---


def test_python_reaches_the_daemon_as_given(tmp_path, monkeypatch):
    """Not resolved: a venv's python is often a link to its base interpreter."""
    monkeypatch.delenv(PYTHON_OVERRIDE, raising=False)
    monkeypatch.chdir(tmp_path)
    seen = []
    monkeypatch.setattr(
        scratch,
        "ensure_daemon",
        lambda anchor, progress=None: (
            seen.append(os.environ.get(PYTHON_OVERRIDE))
            or scratch.Daemon(url="u", session_id="s", pid=1, version="v")
        ),
    )
    (tmp_path / "env" / "bin").mkdir(parents=True)
    (tmp_path / "env" / "bin" / "python").symlink_to(sys.executable)
    cli.main(["scratch", "start", "--root", str(tmp_path), "--python", "env/bin/python"])
    assert seen == [str(tmp_path / "env" / "bin" / "python")]


def test_a_daemon_on_another_interpreter_is_replaced_once_idle(tmp_path, monkeypatch):
    from importlib.metadata import version

    running = scratch.Daemon(url="u", session_id="s", pid=1, version=version("strata-notebook"))
    stopped: list[scratch.Daemon] = []
    monkeypatch.setattr(scratch, "find_daemon", lambda anchor: running)
    monkeypatch.setattr(scratch, "_busy", lambda url: False)
    monkeypatch.setattr(scratch, "_stop", stopped.append)
    monkeypatch.setattr(scratch, "_start", lambda anchor, step: (2, "u2", "s2"))

    monkeypatch.delenv(PYTHON_OVERRIDE, raising=False)
    assert scratch.ensure_daemon(tmp_path) is running
    monkeypatch.setenv(PYTHON_OVERRIDE, "/p/bin/python")
    started = scratch.ensure_daemon(tmp_path)

    assert stopped == [running]
    assert (started.url, started.python) == ("u2", "/p/bin/python")
    record = json.loads((tmp_path / SCRATCH_DIR / ".strata" / "server.json").read_text())
    assert record["python"] == "/p/bin/python"


def test_a_busy_daemon_on_another_interpreter_is_refused_naming_both(tmp_path, monkeypatch):
    from importlib.metadata import version

    running = scratch.Daemon(
        url="u", session_id="s", pid=1, version=version("strata-notebook"), python="/a/python"
    )
    monkeypatch.setattr(scratch, "find_daemon", lambda anchor: running)
    monkeypatch.setattr(scratch, "_busy", lambda url: True)
    monkeypatch.setenv(PYTHON_OVERRIDE, "/b/python")
    with pytest.raises(scratch.ScratchError, match="runs cells on /a/python, not /b/python"):
        scratch.ensure_daemon(tmp_path)
    monkeypatch.delenv(PYTHON_OVERRIDE)
    with pytest.raises(scratch.ScratchError, match="not the environment it found"):
        scratch.ensure_daemon(tmp_path)
    monkeypatch.setenv(PYTHON_OVERRIDE, "/a/python")
    assert scratch.ensure_daemon(tmp_path) is running
