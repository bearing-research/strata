"""Tests for the environment-backend protocol and detection.

The Phase 2 backend split lives at
``src/strata/notebook/env_backend.py``. The detection rules are
documented in ``detect_backend``'s docstring; these tests pin the
behaviour so a future change can't silently flip an attached venv
back into "Strata-managed by uv" mode (or vice versa).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from strata.notebook.env_backend import (
    AttachedBackend,
    BackendDoesNotSupportMutations,
    UvBackend,
    _read_backend_override,
    _venv_has_uv_marker,
    detect_backend,
    get_backend,
)


def _write_pyvenv(notebook_dir: Path, contents: str) -> None:
    """Create a fake .venv/pyvenv.cfg with *contents* under *notebook_dir*."""
    venv = notebook_dir / ".venv"
    venv.mkdir(parents=True, exist_ok=True)
    (venv / "pyvenv.cfg").write_text(contents)


_UV_PYVENV = (
    "home = /Users/x/.local/share/uv/python/cpython-3.12.11-macos-aarch64-none/bin\n"
    "implementation = CPython\n"
    "uv = 0.8.13\n"
    "version_info = 3.12.11\n"
    "include-system-site-packages = false\n"
    "prompt = nb\n"
)

_STDLIB_PYVENV = (
    "home = /opt/homebrew/opt/python@3.13/bin\n"
    "include-system-site-packages = false\n"
    "version = 3.13.3\n"
    "executable = /opt/homebrew/.../python3.13\n"
    "command = /opt/homebrew/.../python3.13 -m venv .venv\n"
)


# ---------------------------------------------------------------------------
# _venv_has_uv_marker — the venv-layer signal


def test_marker_uv_pyvenv_returns_true(tmp_path: Path):
    _write_pyvenv(tmp_path, _UV_PYVENV)
    assert _venv_has_uv_marker(tmp_path) is True


def test_marker_stdlib_pyvenv_returns_false(tmp_path: Path):
    _write_pyvenv(tmp_path, _STDLIB_PYVENV)
    assert _venv_has_uv_marker(tmp_path) is False


def test_marker_missing_pyvenv_returns_false(tmp_path: Path):
    assert _venv_has_uv_marker(tmp_path) is False


def test_marker_tolerates_extra_whitespace(tmp_path: Path):
    """The pyvenv.cfg lines are key=value; we should ignore surrounding
    whitespace and match the bare ``uv`` key. Don't false-positive on
    keys that just *start* with ``uv`` (e.g. a hypothetical
    ``uv_meta = …`` future key wouldn't be a Strata-uv marker)."""
    _write_pyvenv(tmp_path, "  uv  =  0.8.13  \n")
    assert _venv_has_uv_marker(tmp_path) is True


def test_marker_rejects_keys_that_only_prefix_uv(tmp_path: Path):
    _write_pyvenv(tmp_path, "uv_meta = something\n")
    assert _venv_has_uv_marker(tmp_path) is False


# ---------------------------------------------------------------------------
# detect_backend — the layered check


def test_detect_uv_lockfile_present(tmp_path: Path):
    (tmp_path / "uv.lock").write_text("")
    assert detect_backend(tmp_path) == "uv"


def test_detect_uv_lockfile_wins_even_without_venv(tmp_path: Path):
    """Project-level signal survives venv deletion — if the user
    removed .venv/ but kept uv.lock, intent is still uv-managed."""
    (tmp_path / "uv.lock").write_text("")
    assert detect_backend(tmp_path) == "uv"


def test_detect_uv_venv_marker(tmp_path: Path):
    _write_pyvenv(tmp_path, _UV_PYVENV)
    assert detect_backend(tmp_path) == "uv"


def test_detect_attached_for_stdlib_venv(tmp_path: Path):
    _write_pyvenv(tmp_path, _STDLIB_PYVENV)
    assert detect_backend(tmp_path) == "attached"


def test_detect_uv_for_empty_directory(tmp_path: Path):
    """No venv, no lockfile → fresh notebook, default to uv (Strata
    will create one on first sync)."""
    assert detect_backend(tmp_path) == "uv"


# ---------------------------------------------------------------------------
# _read_backend_override — notebook.toml override


def test_override_uv(tmp_path: Path):
    (tmp_path / "notebook.toml").write_text('[strata]\nbackend = "uv"\n')
    assert _read_backend_override(tmp_path) == "uv"


def test_override_attached(tmp_path: Path):
    (tmp_path / "notebook.toml").write_text('[strata]\nbackend = "attached"\n')
    assert _read_backend_override(tmp_path) == "attached"


def test_override_missing_when_unset(tmp_path: Path):
    (tmp_path / "notebook.toml").write_text("[strata]\n")
    assert _read_backend_override(tmp_path) is None


def test_override_missing_when_no_notebook_toml(tmp_path: Path):
    assert _read_backend_override(tmp_path) is None


def test_override_unknown_value_ignored(tmp_path: Path):
    """Garbage values fall back to detection rather than crashing.
    A typo in notebook.toml should not prevent the user from opening
    their work — they'd never recover without command-line access."""
    (tmp_path / "notebook.toml").write_text('[strata]\nbackend = "conda"\n')
    assert _read_backend_override(tmp_path) is None


def test_override_malformed_toml_ignored(tmp_path: Path):
    """Same reasoning as the unknown-value case: corrupt notebook.toml
    falls back to detection."""
    (tmp_path / "notebook.toml").write_text("[strata\nbackend =")
    assert _read_backend_override(tmp_path) is None


def test_override_legacy_environment_section_ignored(tmp_path: Path):
    """The override lives under [strata] specifically -- a user who
    writes it under [environment] (the historical Strata section that
    held legacy runtime metadata) gets detection, not the override.
    The parser strips [environment] aggressively, so honoring it here
    would be a half-supported path that disappears on the next save."""
    (tmp_path / "notebook.toml").write_text('[environment]\nbackend = "attached"\n')
    assert _read_backend_override(tmp_path) is None


# ---------------------------------------------------------------------------
# get_backend — precedence: override > detection


def test_get_backend_returns_uv_by_default(tmp_path: Path):
    assert isinstance(get_backend(tmp_path), UvBackend)


def test_get_backend_routes_attached_pyvenv_to_attached(tmp_path: Path):
    _write_pyvenv(tmp_path, _STDLIB_PYVENV)
    assert isinstance(get_backend(tmp_path), AttachedBackend)


def test_get_backend_override_uv_takes_over_stdlib_venv(tmp_path: Path):
    """User-forced ``backend = "uv"`` wins over the stdlib-venv
    detection so the user can promote a hand-managed venv into a
    uv-managed one."""
    _write_pyvenv(tmp_path, _STDLIB_PYVENV)
    (tmp_path / "notebook.toml").write_text('[strata]\nbackend = "uv"\n')
    assert isinstance(get_backend(tmp_path), UvBackend)


def test_get_backend_override_attached_hands_off_uv_venv(tmp_path: Path):
    """User-forced ``backend = "attached"`` wins over uv markers so
    a uv-created venv can be detached and handed back to the user's
    own tooling."""
    _write_pyvenv(tmp_path, _UV_PYVENV)
    (tmp_path / "uv.lock").write_text("")
    (tmp_path / "notebook.toml").write_text('[strata]\nbackend = "attached"\n')
    assert isinstance(get_backend(tmp_path), AttachedBackend)


# ---------------------------------------------------------------------------
# AttachedBackend — mutation refusal


@pytest.mark.asyncio
async def test_attached_backend_refuses_add(tmp_path: Path):
    backend = AttachedBackend(tmp_path)
    with pytest.raises(BackendDoesNotSupportMutations, match="add"):
        backend.add("requests", timeout=10)
    with pytest.raises(BackendDoesNotSupportMutations, match="add"):
        await backend.add_streaming("requests", timeout=10, on_update=None)


@pytest.mark.asyncio
async def test_attached_backend_refuses_remove(tmp_path: Path):
    backend = AttachedBackend(tmp_path)
    with pytest.raises(BackendDoesNotSupportMutations, match="remove"):
        backend.remove("requests", timeout=10)
    with pytest.raises(BackendDoesNotSupportMutations, match="remove"):
        await backend.remove_streaming("requests", timeout=10, on_update=None)


@pytest.mark.asyncio
async def test_attached_backend_refuses_sync(tmp_path: Path):
    backend = AttachedBackend(tmp_path)
    with pytest.raises(BackendDoesNotSupportMutations, match="sync"):
        backend.sync(python_version=None, timeout=10)
    with pytest.raises(BackendDoesNotSupportMutations, match="sync"):
        await backend.sync_streaming(python_version=None, timeout=10, on_update=None)


def test_attached_backend_python_executable(tmp_path: Path):
    backend = AttachedBackend(tmp_path)
    assert backend.python_executable() == tmp_path / ".venv" / "bin" / "python"
    assert backend.supports_mutations is False
    assert backend.name == "attached"


def test_uv_backend_metadata(tmp_path: Path):
    backend = UvBackend(tmp_path)
    assert backend.python_executable() == tmp_path / ".venv" / "bin" / "python"
    assert backend.supports_mutations is True
    assert backend.name == "uv"


# ---------------------------------------------------------------------------
# Regression: NotebookSession.reload() must re-resolve the backend
#
# Before this was added, ``self.backend`` got set once in ``__init__`` and
# stayed cached. When a user flipped ``[strata] backend = "attached"`` to
# ``"uv"`` in notebook.toml and the session was reused (via
# ``SessionManager.open_notebook(reuse_existing=True)`` → ``reload()``),
# the reused session kept rejecting mutations with 409 -- the new 409
# message told users to set ``backend = "uv"`` but doing so had no effect.


def test_reload_picks_up_backend_override_flip(tmp_path: Path):
    """Switching ``[strata] backend`` from attached to uv on disk must
    take effect after ``reload()`` without requiring session
    destruction."""
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import create_notebook

    notebook_dir = create_notebook(tmp_path, "BackendFlip", initialize_environment=False)
    notebook_toml = notebook_dir / "notebook.toml"
    notebook_toml.write_text(notebook_toml.read_text() + '\n[strata]\nbackend = "attached"\n')

    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    assert session.backend.name == "attached"
    assert session.backend.supports_mutations is False

    # Flip the override and reload.
    contents = notebook_toml.read_text().replace('backend = "attached"', 'backend = "uv"')
    notebook_toml.write_text(contents)
    session.reload()

    assert session.backend.name == "uv"
    assert session.backend.supports_mutations is True


def test_reload_picks_up_attached_when_override_added(tmp_path: Path):
    """The opposite direction: a session that started as uv-managed
    must switch to attached when the user adds an override on disk."""
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import create_notebook

    notebook_dir = create_notebook(tmp_path, "BackendFlipBack", initialize_environment=False)
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    assert session.backend.name == "uv"

    notebook_toml = notebook_dir / "notebook.toml"
    notebook_toml.write_text(notebook_toml.read_text() + '\n[strata]\nbackend = "attached"\n')
    session.reload()

    assert session.backend.name == "attached"
    assert session.backend.supports_mutations is False
