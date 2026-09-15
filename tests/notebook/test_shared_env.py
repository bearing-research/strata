"""One environment per lockfile, shared by the notebooks that have it. Item 5."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from strata.notebook.env_backend import UvBackend, get_backend
from strata.notebook.shared_env import COMPLETE_MARKER, SharedEnvBackend, collect

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the link is a symlink")

_PYPROJECT = f"""[project]
name = "nb"
version = "0.1.0"
requires-python = ">={sys.version_info.major}.{sys.version_info.minor}"
dependencies = []
"""


def _notebook(parent: Path, name: str) -> Path:
    notebook = parent / name
    notebook.mkdir()
    (notebook / "pyproject.toml").write_text(_PYPROJECT)
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
    return backend.sync(python_version=None, timeout=180)


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
