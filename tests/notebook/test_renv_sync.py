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
from strata.notebook.writer import (
    _renv_sync,
    create_notebook,
    update_notebook_r_block,
    write_notebook_toml,
)

# ---------------------------------------------------------------------------
# _renv_sync wrapper
# ---------------------------------------------------------------------------


class TestRenvSyncMissingRscript:
    """When Rscript isn't on PATH **and** there's a lockfile, return False."""

    def test_returns_false_without_rscript_when_lockfile_exists(self, monkeypatch, tmp_path):
        """A real restore is needed but R isn't available — surface the failure."""
        monkeypatch.setattr(shutil, "which", lambda name: None)
        (tmp_path / "renv.lock").write_text("")
        assert _renv_sync(tmp_path) is False


class TestRenvSyncNoLockfile:
    """Missing ``renv.lock`` is success regardless of R availability.

    The pre-bootstrap state (notebook has R cells, ``renv::init()``
    hasn't run yet) must not surface as a failure — that's the
    expected initial state, not an error. Particularly important
    when R isn't installed locally: we don't want notebook open to
    fail "renv sync failed" for a user who hasn't even installed R
    yet but is editing R cells via the Vue UI.
    """

    def test_returns_true_when_lockfile_missing_with_rscript(self, monkeypatch, tmp_path):
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")
        # No renv.lock in tmp_path.
        called = []

        def fake_run(*args, **kwargs):
            called.append(args)
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert _renv_sync(tmp_path) is True
        assert called == [], "should not invoke Rscript when nothing to restore"

    def test_returns_true_when_lockfile_missing_without_rscript(self, monkeypatch, tmp_path):
        """The "R not installed" case: still nothing to do, still success.

        Lockfile check runs before ``shutil.which("Rscript")`` so a
        user without R installed who hasn't yet bootstrapped renv
        doesn't get a spurious failure. Review feedback on PR #68.
        """
        monkeypatch.setattr(shutil, "which", lambda name: None)
        # No renv.lock in tmp_path.
        assert _renv_sync(tmp_path) is True


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

    def test_does_not_pass_vanilla_or_no_init_file(self, monkeypatch, tmp_path):
        """``.Rprofile`` must run — it's what activates the project's renv.

        Review feedback on PR #68: ``--no-init-file`` / ``--vanilla``
        flags disable ``.Rprofile``, which is what sources
        ``renv/activate.R`` to put the project library on
        ``.libPaths()``. Without it ``renv::restore()`` can't see the
        project library and tries to install everything into the
        user's default lib — usually failing because that lib doesn't
        even have renv installed.
        """
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")
        (tmp_path / "renv.lock").write_text("")

        captured_args: list[str] = []

        def fake_run(args, **kwargs):
            captured_args.extend(args)
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

        monkeypatch.setattr(subprocess, "run", fake_run)
        _renv_sync(tmp_path)

        assert "--vanilla" not in captured_args
        assert "--no-init-file" not in captured_args


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


class TestUpdateNotebookRBlock:
    """``update_notebook_r_block`` mirrors the partial-update pattern of
    ``update_notebook_env``: read the toml, overwrite just the ``[r]``
    block, write back. Other sections round-trip unchanged."""

    def test_writes_block_when_absent(self, tmp_path: Path):
        notebook_dir = create_notebook(tmp_path, "Update R Block", initialize_environment=False)

        update_notebook_r_block(
            notebook_dir,
            {
                "lock_hash": "cafe" * 16,
                "last_synced_at": 1717003200000,
                "r_version": "4.4.1",
            },
        )

        state = parse_notebook(notebook_dir)
        assert state.r["lock_hash"] == "cafe" * 16
        assert state.r["last_synced_at"] == 1717003200000
        assert state.r["r_version"] == "4.4.1"

    def test_overwrites_existing_block(self, tmp_path: Path):
        notebook_dir = create_notebook(tmp_path, "Overwrite R Block", initialize_environment=False)
        update_notebook_r_block(notebook_dir, {"lock_hash": "old", "last_synced_at": 1})

        update_notebook_r_block(notebook_dir, {"lock_hash": "new", "last_synced_at": 2})

        state = parse_notebook(notebook_dir)
        assert state.r["lock_hash"] == "new"
        assert state.r["last_synced_at"] == 2

    def test_empty_dict_removes_block(self, tmp_path: Path):
        notebook_dir = create_notebook(tmp_path, "Remove R Block", initialize_environment=False)
        update_notebook_r_block(notebook_dir, {"lock_hash": "x", "last_synced_at": 1})

        update_notebook_r_block(notebook_dir, {})

        text = (notebook_dir / "notebook.toml").read_text(encoding="utf-8")
        assert "[r]" not in text

    def test_no_op_when_block_unchanged(self, tmp_path: Path):
        """Calling with identical content leaves the file untouched.

        Mirrors ``update_notebook_env``'s anti-churn invariant: typing
        an API key or hitting a cached ``_renv_sync`` shouldn't bump
        ``updated_at`` on the committed notebook.
        """
        notebook_dir = create_notebook(tmp_path, "Idempotent R Block", initialize_environment=False)
        update_notebook_r_block(notebook_dir, {"lock_hash": "z", "last_synced_at": 1})
        before = (notebook_dir / "notebook.toml").read_bytes()

        update_notebook_r_block(notebook_dir, {"lock_hash": "z", "last_synced_at": 1})
        after = (notebook_dir / "notebook.toml").read_bytes()

        assert before == after, "no-change update must not rewrite the toml"

    def test_does_not_touch_other_sections(self, tmp_path: Path):
        """Updating ``[r]`` leaves ``[env]`` and other blocks unchanged."""
        notebook_dir = create_notebook(tmp_path, "Other Sections", initialize_environment=False)
        toml = NotebookToml(
            notebook_id="multi",
            name="Other Sections",
            env={"DEBUG": "true"},
        )
        write_notebook_toml(notebook_dir, toml)

        update_notebook_r_block(notebook_dir, {"lock_hash": "h", "last_synced_at": 1})

        state = parse_notebook(notebook_dir)
        assert state.env == {"DEBUG": "true"}
        assert state.r["lock_hash"] == "h"


# ---------------------------------------------------------------------------
# Session.ensure_renv_synced — wiring _renv_sync into open
# ---------------------------------------------------------------------------


class TestEnsureRenvSynced:
    """``ensure_renv_synced`` is the session-side hook that runs
    ``_renv_sync`` on open. Pins the three honest user-facing
    behaviours: no-op without a lockfile, [r] block populated on
    success, [r] block left alone on failure."""

    def _make_session(self, notebook_dir):
        from strata.notebook.session import NotebookSession

        state = parse_notebook(notebook_dir)
        return NotebookSession(state, notebook_dir)

    def test_no_op_without_renv_lock(self, tmp_path: Path, monkeypatch):
        """Python-only notebook (no renv.lock): the [r] block stays absent
        and ``_renv_sync`` is never even called.
        """
        from strata.notebook import session as session_module

        called: list[Path] = []

        def _fake_renv_sync(notebook_dir: Path, *, timeout: int = 600) -> bool:
            called.append(notebook_dir)
            return True

        monkeypatch.setattr(session_module, "_renv_sync", _fake_renv_sync)

        notebook_dir = create_notebook(tmp_path, "No R", initialize_environment=False)
        session = self._make_session(notebook_dir)

        session.ensure_renv_synced()

        assert called == [], "_renv_sync must not be called when renv.lock is missing"
        state = parse_notebook(notebook_dir)
        assert state.r == {}

    def test_success_populates_r_block(self, tmp_path: Path, monkeypatch):
        """Real-looking ``renv.lock`` + a ``_renv_sync`` success stub
        produces a populated [r] block with lock_hash + last_synced_at."""
        from strata.notebook import session as session_module

        def _fake_renv_sync(notebook_dir: Path, *, timeout: int = 600) -> bool:
            return True

        monkeypatch.setattr(session_module, "_renv_sync", _fake_renv_sync)
        # Stub the R version probe so the test doesn't depend on
        # whether Rscript is installed on the host.
        monkeypatch.setattr(
            session_module.NotebookSession,
            "_probe_r_version",
            lambda self: "4.4.1",
        )

        notebook_dir = create_notebook(tmp_path, "Has R", initialize_environment=False)
        renv_content = '{"R": {"Version": "4.4.1"}, "Packages": {"arrow": "1.0"}}\n'
        (notebook_dir / "renv.lock").write_text(renv_content, encoding="utf-8")
        session = self._make_session(notebook_dir)

        session.ensure_renv_synced()

        state = parse_notebook(notebook_dir)
        # Hash matches what update_notebook_r_block was handed, so a
        # later open with the same lockfile reads back as a cache hit.
        import hashlib

        assert state.r["lock_hash"] == hashlib.sha256(renv_content.encode()).hexdigest()
        assert state.r["r_version"] == "4.4.1"
        assert isinstance(state.r["last_synced_at"], int)

    def test_failure_leaves_r_block_untouched(self, tmp_path: Path, monkeypatch):
        """``_renv_sync`` returning False (Rscript missing, timeout,
        non-zero exit) must leave the [r] block as-is. Stamping a
        new ``last_synced_at`` after a failure would lie about when
        the env was last good."""
        from strata.notebook import session as session_module

        def _fake_renv_sync(notebook_dir: Path, *, timeout: int = 600) -> bool:
            return False

        monkeypatch.setattr(session_module, "_renv_sync", _fake_renv_sync)

        notebook_dir = create_notebook(tmp_path, "Failing R", initialize_environment=False)
        (notebook_dir / "renv.lock").write_text("{}\n", encoding="utf-8")
        # Pre-seed an [r] block as if a prior successful sync had run.
        update_notebook_r_block(
            notebook_dir,
            {"lock_hash": "previous-good", "last_synced_at": 1},
        )
        session = self._make_session(notebook_dir)

        session.ensure_renv_synced()

        state = parse_notebook(notebook_dir)
        assert state.r["lock_hash"] == "previous-good"
        assert state.r["last_synced_at"] == 1
