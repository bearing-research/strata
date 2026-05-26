"""Tests for ``_renv_sync`` + the ``[r]`` block schema.

Same two-tier shape as ``test_language_r_analyzer.py``:

- Unit tests monkeypatch ``shutil.which`` + ``subprocess.run`` to cover
  the wrapper's success / failure / timeout / missing-Rscript paths.
- Schema tests pin that the ``[r]`` block round-trips through the
  writer + parser.

The capstone real-renv integration tests land with #59.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

from strata.notebook.models import NotebookToml
from strata.notebook.parser import parse_notebook
from strata.notebook.writer import _renv_sync, create_notebook, write_notebook_toml

# ---------------------------------------------------------------------------
# _renv_sync wrapper
# ---------------------------------------------------------------------------


class TestRenvSyncMissingRscript:
    """When Rscript isn't on PATH, the helper logs + returns False."""

    def test_returns_false_without_rscript(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        # Even with an existing renv.lock, no Rscript → False.
        (tmp_path / "renv.lock").write_text("")
        assert _renv_sync(tmp_path) is False


class TestRenvSyncNoLockfile:
    """Missing ``renv.lock`` is treated as success — nothing to restore."""

    def test_returns_true_when_lockfile_missing(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")
        # No renv.lock in tmp_path.
        called = []

        def fake_run(*args, **kwargs):
            called.append(args)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert _renv_sync(tmp_path) is True
        assert called == [], "should not invoke Rscript when nothing to restore"


class TestRenvSyncSubprocessFailures:
    """Subprocess failures surface as ``False`` + a warning log."""

    def _setup(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")
        (tmp_path / "renv.lock").write_text("")

    def test_timeout(self, monkeypatch, tmp_path):
        self._setup(monkeypatch, tmp_path)

        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="Rscript", timeout=600)

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert _renv_sync(tmp_path) is False

    def test_nonzero_exit(self, monkeypatch, tmp_path):
        self._setup(monkeypatch, tmp_path)

        def fake_run(*args, **kwargs):
            raise subprocess.CalledProcessError(
                returncode=1, cmd="Rscript", stderr=b"renv: package missing"
            )

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert _renv_sync(tmp_path) is False


class TestRenvSyncHappyPath:
    """Rscript exits 0 with a lockfile present → True."""

    def test_returns_true_on_success(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")
        (tmp_path / "renv.lock").write_text("")

        invocations: list[tuple] = []

        def fake_run(args, **kwargs):
            invocations.append((args, kwargs.get("cwd")))
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert _renv_sync(tmp_path) is True
        assert len(invocations) == 1
        args, cwd = invocations[0]
        # Confirm we call renv::restore() with the prompt suppressed —
        # otherwise it'd block on stdin asking for confirmation.
        assert "renv::restore(prompt = FALSE)" in args
        # Confirm cwd is set to the notebook dir; renv resolves the
        # project off cwd.
        assert cwd == str(tmp_path)


class TestRenvSyncDefaultTimeout:
    """Default timeout is generous enough for CRAN compile times."""

    def test_default_timeout_is_at_least_5_minutes(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")
        (tmp_path / "renv.lock").write_text("")

        captured: dict[str, int] = {}

        def fake_run(args, **kwargs):
            captured["timeout"] = kwargs.get("timeout")
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        monkeypatch.setattr(subprocess, "run", fake_run)
        _renv_sync(tmp_path)

        # 5 min minimum lets renv compile a handful of packages from
        # source on platforms without wheels. ``_uv_sync``'s 60s
        # default is fine because uv ships wheels; renv doesn't have
        # that luxury.
        assert captured["timeout"] >= 300


# ---------------------------------------------------------------------------
# [r] block schema
# ---------------------------------------------------------------------------


class TestRBlockSchema:
    """The ``[r]`` block round-trips through writer + parser."""

    def test_default_is_empty_dict(self):
        toml = NotebookToml(notebook_id="n1", name="N1")
        assert toml.r == {}

    def test_round_trip(self, tmp_path: Path):
        notebook_dir = create_notebook(tmp_path, "R Block Test", initialize_environment=False)

        # Round-trip a populated [r] block. Shape mirrors the Python
        # side's [environment]: lockfile hash + sync timestamp + version.
        toml = NotebookToml(
            notebook_id="r-block-test",
            name="R Block Test",
            r={
                "lock_hash": "deadbeef" * 8,
                "last_synced_at": 1717003200000,
                "r_version": "4.4.1",
            },
        )
        write_notebook_toml(notebook_dir, toml)

        state = parse_notebook(notebook_dir)
        assert state.r["lock_hash"] == "deadbeef" * 8
        assert state.r["last_synced_at"] == 1717003200000
        assert state.r["r_version"] == "4.4.1"

    def test_empty_r_block_not_written(self, tmp_path: Path):
        """An empty ``[r]`` block stays out of the file to avoid noise."""
        notebook_dir = create_notebook(tmp_path, "Empty R Block", initialize_environment=False)
        toml = NotebookToml(notebook_id="empty-r", name="Empty R Block", r={})
        write_notebook_toml(notebook_dir, toml)

        text = (notebook_dir / "notebook.toml").read_text(encoding="utf-8")
        # The ``**({"r": ...} if toml.r else {})`` short-circuit in
        # write_notebook_toml drops empty blocks so notebooks without
        # R cells don't accumulate a stub [r] section.
        assert "[r]" not in text
        assert "\nr =" not in text
