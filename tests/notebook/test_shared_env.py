"""One environment per lockfile, shared by the notebooks that have it."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from strata.notebook import shared_env
from strata.notebook.dependencies import EnvironmentOperationLog, _UvCommandResult
from strata.notebook.env import uv_lock_key
from strata.notebook.env_backend import UvBackend, get_backend
from strata.notebook.shared_env import COMPLETE_MARKER, SharedEnvBackend, collect

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the link is a symlink")

_PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"


def _notebook(parent: Path, name: str, extra: str = "") -> Path:
    """A notebook whose project is named after it, as ``strata new`` writes."""
    notebook = parent / name
    notebook.mkdir()
    (notebook / "pyproject.toml").write_text(
        f'[project]\nname = "{name}"\nversion = "0.1.0"\n'
        f'requires-python = ">={_PYTHON}"\ndependencies = []\n{extra}'
    )
    return notebook


def _package(parent: Path) -> Path:
    """A local package, so adding a dependency needs no index."""
    package = parent / "tinydep"
    (package / "tinydep").mkdir(parents=True)
    (package / "tinydep" / "__init__.py").write_text("")
    (package / "pyproject.toml").write_text('[project]\nname = "tinydep"\nversion = "0.1.0"\n')
    return package


def _linked(notebook: Path) -> Path:
    venv = notebook / ".venv"
    assert venv.is_symlink(), f"{venv} is not a link"
    return Path(os.readlink(venv))


def _has_tinydep(env: Path) -> bool:
    return any(env.glob("lib/python*/site-packages/tinydep-*.dist-info"))


@pytest.fixture
def shared(tmp_path, monkeypatch):
    root = tmp_path / "envs"
    config = SimpleNamespace(
        notebook_env_backend="shared",
        notebook_shared_env_dir=root,
        notebook_storage_dir=tmp_path / "notebooks",
    )
    monkeypatch.setattr("strata.server._state", SimpleNamespace(config=config))
    return root


def test_the_server_setting_chooses_the_backend(tmp_path, shared, monkeypatch):
    assert isinstance(get_backend(tmp_path), SharedEnvBackend)
    assert get_backend(tmp_path).root == shared.resolve()

    monkeypatch.setattr(
        "strata.server._state", SimpleNamespace(config=SimpleNamespace(notebook_env_backend="uv"))
    )
    assert isinstance(get_backend(tmp_path), UvBackend)


async def _sync(notebook: Path, streaming: bool):
    backend = get_backend(notebook)
    if streaming:
        return await backend.sync_streaming(python_version=None, timeout=180, on_update=None)
    return await asyncio.to_thread(backend.sync, python_version=None, timeout=180)


@pytest.mark.parametrize("streaming", [False, True], ids=["sync", "streaming"])
async def test_two_notebooks_with_one_lock_share_one_environment_and_the_second_only_links(
    tmp_path, shared, streaming
):
    first, second = _notebook(tmp_path, "first"), _notebook(tmp_path, "second")

    built = await _sync(first, streaming)
    linked = await _sync(second, streaming)

    assert built.success, built.error
    assert linked.success, linked.error
    assert _linked(first) == _linked(second)
    assert _linked(first).parent == shared.resolve()
    assert "uv sync" in built.operation_log.command
    assert linked.operation_log.command == "uv lock", "the second sync installed nothing"
    assert (second / ".venv" / "bin" / "python").exists()


def test_adding_a_package_to_one_notebook_leaves_the_other_untouched(tmp_path, shared):
    from strata.notebook.dependencies import add_dependency

    first, second = _notebook(tmp_path, "first"), _notebook(tmp_path, "second")
    for notebook in (first, second):
        assert get_backend(notebook).sync(python_version=None, timeout=180).success
    before = _linked(second)
    lock_before = (second / "uv.lock").read_bytes()

    added = add_dependency(first, str(_package(tmp_path)))

    assert added.success, added.error
    assert _linked(first) != before
    assert _has_tinydep(_linked(first))
    assert _linked(second) == before
    assert not _has_tinydep(before), "the shared environment was changed in place"
    assert (second / "uv.lock").read_bytes() == lock_before


def test_a_notebook_s_own_venv_is_replaced_by_the_link(tmp_path, shared):
    notebook = _notebook(tmp_path, "legacy")
    (notebook / ".venv" / "bin").mkdir(parents=True)

    assert get_backend(notebook).sync(python_version=None, timeout=180).success
    assert _linked(notebook).parent == shared.resolve()


class TestTheSweep:
    def _two_keys(self, tmp_path):
        """One environment still linked, one abandoned by its notebook."""
        kept, moved = _notebook(tmp_path, "kept"), _notebook(tmp_path, "moved")
        for notebook in (kept, moved):
            assert get_backend(notebook).sync(python_version=None, timeout=180).success
        linked = _linked(kept)
        # Move one notebook to its own lock, then away again, leaving an
        # environment nothing links to.
        assert get_backend(moved).add(str(_package(tmp_path)), timeout=180).success
        orphan = _linked(moved)
        assert get_backend(moved).remove("tinydep", timeout=180).success
        assert _linked(moved) == linked
        return linked, orphan

    def test_an_orphan_past_its_ttl_goes_and_a_linked_environment_stays(self, tmp_path, shared):
        linked, orphan = self._two_keys(tmp_path)

        result = collect(
            shared, ttl_days=7, now=(orphan / COMPLETE_MARKER).stat().st_mtime + 8 * 86400
        )

        assert result.removed == [orphan.name]
        assert result.referenced == [linked.name]
        assert not orphan.exists()
        assert linked.exists()

    def test_an_orphan_inside_its_ttl_stays(self, tmp_path, shared):
        _, orphan = self._two_keys(tmp_path)

        result = collect(shared, ttl_days=7)

        assert result.removed == []
        assert orphan.exists()

    def test_a_reference_from_a_deleted_notebook_does_not_keep_an_environment(
        self, tmp_path, shared
    ):
        import shutil

        notebook = _notebook(tmp_path, "gone")
        assert get_backend(notebook).sync(python_version=None, timeout=180).success
        env = _linked(notebook)
        shutil.rmtree(notebook)

        assert collect(shared, ttl_days=0).removed == [env.name]

    def test_the_cli_runs_the_sweep(self, tmp_path, shared, capsys):
        from strata.cli import _dispatch_env_gc

        _, orphan = self._two_keys(tmp_path)

        assert _dispatch_env_gc(argparse.Namespace(env_dir=str(shared), ttl_days=0.0)) == 0
        assert f"removed {orphan.name}" in capsys.readouterr().out
        assert not orphan.exists()


def test_the_key_names_the_interpreter_build(tmp_path):
    notebook = _notebook(tmp_path, "nb")
    (notebook / "uv.lock").write_text("version = 1\n")
    backend = SharedEnvBackend(notebook, tmp_path / "envs")

    assert backend.key("cpython 3.13.1  macosx-14-arm64") != backend.key(
        "cpython 3.13.2  macosx-14-arm64"
    )


def _lock(
    project: str,
    *,
    six: str = "1.17.0",
    marker: str = "",
    requires_python: str = ">=3.12",
    source: str = 'virtual = "."',
) -> str:
    """A ``uv.lock`` for a notebook *project* depending on six, in uv's name order."""
    dependency = '{ name = "six"' + (f', marker = "{marker}"' if marker else "") + " }"
    root = (
        f'[[package]]\nname = "{project}"\nversion = "0.1.0"\nsource = {{ {source} }}\n'
        f"dependencies = [\n    {dependency},\n]\n\n"
        f'[package.metadata]\nrequires-dist = [{{ name = "six", specifier = ">=1" }}]\n'
    )
    six_entry = (
        f'[[package]]\nname = "six"\nversion = "{six}"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://example/six-{six}.whl", hash = "sha256:{six}" }}]\n'
    )
    entries = sorted([(project, root), ("six", six_entry)])
    header = f'version = 1\nrevision = 3\nrequires-python = "{requires_python}"\n\n'
    return header + "\n".join(entry for _, entry in entries)


class TestTheLockKey:
    """The key names what a lock installs, not which notebook's project it belongs to."""

    def test_notebooks_named_apart_with_the_same_packages_get_one_key(self):
        assert _lock("alpha") != _lock("zeta")
        assert uv_lock_key(_lock("alpha")) == uv_lock_key(_lock("zeta"))

    def test_an_installed_project_and_a_virtual_one_get_one_key(self):
        # The shared environment never installs the project itself.
        assert uv_lock_key(_lock("alpha", source='editable = "."')) == uv_lock_key(_lock("zeta"))

    @pytest.mark.parametrize(
        "changed",
        [
            {"six": "1.16.0"},
            {"marker": "sys_platform == 'linux'"},
            {"requires_python": ">=3.13"},
        ],
        ids=["package-version", "dependency-marker", "requires-python"],
    )
    def test_anything_that_changes_the_install_changes_the_key(self, changed):
        assert uv_lock_key(_lock("alpha", **changed)) != uv_lock_key(_lock("alpha"))

    def test_a_worker_is_sent_the_same_key_for_both(self, tmp_path):
        from strata.notebook.worker_env import _validated, environment_spec

        specs = []
        for name in ("alpha", "zeta"):
            notebook = _notebook(tmp_path, name)
            (notebook / "uv.lock").write_text(_lock(name))
            specs.append(environment_spec(notebook, None))

        assert specs[0]["key"] == specs[1]["key"]
        assert all(_validated(spec)["key"] == spec["key"] for spec in specs)


def _imports_tinydep(notebook: Path) -> bool:
    ran = subprocess.run(
        [str(notebook / ".venv" / "bin" / "python"), "-c", "import tinydep"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return ran.returncode == 0


def test_notebooks_named_apart_with_the_same_dependencies_share_one_environment_and_run(
    tmp_path, shared
):
    package = _package(tmp_path)
    alpha, zeta = _notebook(tmp_path, "alpha"), _notebook(tmp_path, "zeta")

    built = get_backend(alpha).add(str(package), timeout=180)
    linked = get_backend(zeta).add(str(package), timeout=180)

    assert built.success, built.error
    assert linked.success, linked.error
    assert (alpha / "uv.lock").read_text() != (zeta / "uv.lock").read_text()
    assert _linked(alpha) == _linked(zeta)
    assert "uv sync" not in linked.operation_log.command, "the second notebook installed"
    assert _imports_tinydep(alpha)
    assert _imports_tinydep(zeta)


def test_the_shared_environment_does_not_install_a_notebook_s_own_project(tmp_path, shared):
    packaged = _notebook(
        tmp_path,
        "packaged",
        extra='\n[build-system]\nrequires = ["uv_build>=0.8"]\nbuild-backend = "uv_build"\n',
    )
    (packaged / "src" / "packaged").mkdir(parents=True)
    (packaged / "src" / "packaged" / "__init__.py").write_text("")
    plain = _notebook(tmp_path, "plain")

    synced = get_backend(packaged).sync(python_version=None, timeout=180)

    assert synced.success, synced.error
    site_packages = list(_linked(packaged).glob("lib/python*/site-packages/*"))
    assert site_packages, "nothing was installed at all"
    assert not [path for path in site_packages if path.name.startswith("packaged")], (
        "a notebook's own project went into an environment other notebooks link to"
    )
    assert get_backend(plain).sync(python_version=None, timeout=180).success
    assert _linked(plain) == _linked(packaged)


class TestTheSweepOnlyTakesWhatItBuilt:
    """A removed environment is a notebook that cannot run, so the sweep is conservative."""

    def test_a_directory_it_did_not_build_is_left_alone(self, tmp_path, shared):
        theirs = shared / "not-an-environment"
        (theirs / "nested").mkdir(parents=True)
        (theirs / "nested" / "thesis.csv").write_text("data")
        old = time.time() - 30 * 86400
        os.utime(theirs, (old, old))

        result = collect(shared, ttl_days=1)

        assert result.removed == []
        assert (theirs / "nested" / "thesis.csv").read_text() == "data"

    def test_a_half_built_environment_is_left_to_be_built_again(self, tmp_path, shared):
        half = shared / ("0" * 32)
        half.mkdir(parents=True)
        old = time.time() - 30 * 86400
        os.utime(half, (old, old))

        assert collect(shared, ttl_days=1).removed == []
        assert half.exists()

    def test_a_notebook_whose_volume_is_away_keeps_its_environment(self, tmp_path, shared):
        env = shared / ("a" * 32)
        env.mkdir(parents=True)
        (env / COMPLETE_MARKER).touch()
        refs = shared / "refs" / ("a" * 32)
        refs.mkdir(parents=True)
        # A reference this process cannot read says nothing about whether the
        # notebook still links here.
        unreadable = refs / "ref1"
        unreadable.write_text(str(tmp_path / "gone"))
        unreadable.chmod(0o000)
        old = time.time() - 30 * 86400
        os.utime(env / COMPLETE_MARKER, (old, old))

        try:
            result = collect(shared, ttl_days=1)
        finally:
            unreadable.chmod(0o600)

        assert result.removed == []
        assert env.exists()


def test_opening_a_notebook_keeps_its_environment_in_use(tmp_path, shared, monkeypatch):
    """The sweep ages from the last link, so an opened, unsynced notebook still counts as a user."""
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import create_notebook

    notebook = create_notebook(tmp_path / "nb", "nb", initialize_environment=False)
    (notebook / "uv.lock").write_text("version = 1\n")
    session = NotebookSession(parse_notebook(notebook), notebook)
    synced: list[str] = []

    def _sync(self, *, python_version, timeout):
        synced.append("backend")
        from strata.notebook.dependencies import EnvironmentOperationLog, _UvCommandResult

        return _UvCommandResult(
            success=True, error=None, operation_log=EnvironmentOperationLog(command="uv sync")
        )

    monkeypatch.setattr(SharedEnvBackend, "sync", _sync)

    session.ensure_venv_synced()

    assert synced == ["backend"], "the shared backend must do the sync, not a bare uv sync"


def _ok(command: str) -> _UvCommandResult:
    return _UvCommandResult(
        success=True, error=None, operation_log=EnvironmentOperationLog(command=command)
    )


@pytest.fixture
async def building(shared, monkeypatch):
    """Fake uv where every notebook maps to one key whose install waits for ``finish``,
    so a streaming sync holds that key's lock for as long as a test needs."""
    loop = asyncio.get_running_loop()
    state = SimpleNamespace(
        installing=asyncio.Event(), finish=asyncio.Event(), prepared=0, second=asyncio.Event()
    )

    def counted() -> None:
        state.prepared += 1
        if state.prepared == 2:
            state.second.set()

    def prepare(self, python_version):
        self.root.mkdir(parents=True, exist_ok=True)
        # Counted on the loop, so a loop blocked after this call never counts it.
        loop.call_soon_threadsafe(counted)
        return Path(sys.executable), "k", shared_env._key_lock(self.root, "k")

    async def streaming(notebook_dir, args, *, env=None, **kwargs):
        if args[0] == "sync":
            state.installing.set()
            await state.finish.wait()
            python = Path(env["UV_PROJECT_ENVIRONMENT"]) / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.symlink_to(sys.executable)
        return _ok(f"uv {args[0]}")

    monkeypatch.setattr(SharedEnvBackend, "_prepare", prepare)
    monkeypatch.setattr(shared_env, "run_uv_command_streaming", streaming)
    monkeypatch.setattr(
        shared_env, "_run_uv_command", lambda _dir, args, **kw: _ok(f"uv {args[0]}")
    )
    yield state
    state.finish.set()


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "blocking",
    [
        lambda backend: backend.sync(python_version=None, timeout=60),
        lambda backend: backend.add("tinydep", timeout=60),
        lambda backend: backend.remove("tinydep", timeout=60),
    ],
    ids=["sync", "add", "remove"],
)
async def test_a_blocking_sync_on_the_event_loop_is_refused_while_a_streaming_one_builds(
    tmp_path, building, blocking
):
    """The streaming sync releases the key's lock only once the loop runs again, so a
    blocking sync waiting for that lock on the loop would hang it (a timeout here)."""
    first, second = _notebook(tmp_path, "first"), _notebook(tmp_path, "second")
    build = asyncio.create_task(
        get_backend(first).sync_streaming(python_version=None, timeout=60, on_update=None)
    )
    await building.installing.wait()

    with pytest.raises(RuntimeError, match="event loop"):
        blocking(get_backend(second))

    building.finish.set()
    assert (await build).success


@pytest.mark.timeout(30)
async def test_opening_an_imported_notebook_while_its_environment_builds_keeps_the_server_up(
    tmp_path, shared, building, monkeypatch
):
    """On a new service-mode server the import's environment job builds the key, and an
    open of the same notebook waits for that build without stopping the event loop."""
    from fastapi import FastAPI
    from httpx import ASGITransport, AsyncClient

    from strata.notebook.routes import router

    storage = tmp_path / "notebooks"
    storage.mkdir()
    config = SimpleNamespace(
        deployment_mode="service",
        notebook_env_backend="shared",
        notebook_shared_env_dir=shared,
        notebook_storage_dir=storage,
        transforms_config={},
    )
    monkeypatch.setattr("strata.server._state", SimpleNamespace(config=config))

    async def _no_warm_pool(self):
        return None

    monkeypatch.setattr("strata.notebook.pool.WarmProcessPool.start", _no_warm_pool)
    app = FastAPI()
    app.include_router(router)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    ipynb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [{"cell_type": "code", "metadata": {}, "outputs": [], "source": ["x = 1\n"]}],
    }
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        imported = await client.post(
            "/v1/notebooks/import", files={"file": ("demo.ipynb", json.dumps(ipynb))}
        )
        assert imported.status_code == 200, imported.text
        # The import's environment job is installing the key, holding its lock.
        await building.installing.wait()

        opening = asyncio.create_task(
            client.post("/v1/notebooks/open", json={"path": imported.json()["path"]})
        )
        await building.second.wait()
        assert (await client.get("/health")).status_code == 200
        assert not opening.done(), "the open waits for the build it shares"

        building.finish.set()
        opened = await opening

    assert opened.status_code == 200, opened.text
    assert opened.json()["environment"]["sync_state"] == "ready"
    assert opened.json()["session_id"] != imported.json()["session_id"]
