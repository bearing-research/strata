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
from strata.notebook.runtime_state import RRuntime, load_runtime_state, save_runtime_state
from strata.notebook.writer import _renv_sync, create_notebook, write_notebook_toml

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


# ---------------------------------------------------------------------------
# Session.ensure_renv_synced — wiring _renv_sync into open
# ---------------------------------------------------------------------------


class TestEnsureRenvSynced:
    """``ensure_renv_synced`` is the session-side hook that runs
    ``_renv_sync`` on open. Pins:

    * No-op without a lockfile (Python-only notebooks pay nothing).
    * Successful sync persists to ``.strata/runtime.json`` (not to
      committed ``notebook.toml`` — runtime state lives in
      ``runtime.json`` so reopens don't churn the committed file).
    * Hash-unchanged reopens short-circuit before spawning
      ``Rscript``.
    * Failed sync leaves the prior runtime entry untouched.
    """

    def _make_session(self, notebook_dir):
        from strata.notebook.session import NotebookSession

        state = parse_notebook(notebook_dir)
        return NotebookSession(state, notebook_dir)

    def _stub_renv_sync(self, monkeypatch, *, ok: bool):
        from strata.notebook import session as session_module

        calls: list[Path] = []

        def _fake_renv_sync(notebook_dir: Path, *, timeout: int = 600) -> bool:
            calls.append(notebook_dir)
            return ok

        monkeypatch.setattr(session_module, "_renv_sync", _fake_renv_sync)
        monkeypatch.setattr(
            session_module.NotebookSession,
            "_probe_r_version",
            lambda self: "4.4.1",
        )
        return calls

    def test_no_op_without_renv_lock(self, tmp_path: Path, monkeypatch):
        """Python-only notebook: ``_renv_sync`` never called, runtime
        ``r`` block stays at the empty default, and the committed
        ``notebook.toml`` is byte-identical before and after.
        """
        calls = self._stub_renv_sync(monkeypatch, ok=True)
        notebook_dir = create_notebook(tmp_path, "No R", initialize_environment=False)
        toml_before = (notebook_dir / "notebook.toml").read_bytes()

        session = self._make_session(notebook_dir)
        session.ensure_renv_synced()

        assert calls == [], "_renv_sync must not be called when renv.lock is missing"
        assert load_runtime_state(notebook_dir).r == RRuntime()
        assert (notebook_dir / "notebook.toml").read_bytes() == toml_before

    def test_success_persists_to_runtime_json(self, tmp_path: Path, monkeypatch):
        """A successful sync writes ``r`` into ``runtime.json``, not into
        the committed ``notebook.toml``. P2 from #87 review: per-session
        timestamps must not bump ``notebook.toml``'s ``updated_at``.
        """
        import hashlib

        self._stub_renv_sync(monkeypatch, ok=True)
        notebook_dir = create_notebook(tmp_path, "Has R", initialize_environment=False)
        renv_content = '{"R": {"Version": "4.4.1"}, "Packages": {"arrow": "1.0"}}\n'
        (notebook_dir / "renv.lock").write_text(renv_content, encoding="utf-8")
        toml_before = (notebook_dir / "notebook.toml").read_bytes()

        session = self._make_session(notebook_dir)
        session.ensure_renv_synced()

        runtime = load_runtime_state(notebook_dir).r
        assert runtime.lock_hash == hashlib.sha256(renv_content.encode()).hexdigest()
        assert runtime.r_version == "4.4.1"
        assert runtime.last_synced_at > 0
        assert runtime.has_lockfile is True

        # The on-disk [r] block stays empty (the committed config has
        # no opinion on runtime state) and notebook.toml is byte-
        # identical — no churn.
        assert parse_notebook(notebook_dir).r == {}
        assert (notebook_dir / "notebook.toml").read_bytes() == toml_before

    def test_hash_unchanged_short_circuits(self, tmp_path: Path, monkeypatch):
        """A second open against the same lockfile must NOT spawn Rscript.

        P3 from #87 review: the session-reuse path now calls
        ``ensure_renv_synced`` on every reopen; the hash short-circuit
        is what keeps that free.
        """
        calls = self._stub_renv_sync(monkeypatch, ok=True)
        notebook_dir = create_notebook(tmp_path, "Cached R", initialize_environment=False)
        renv_content = '{"R": {"Version": "4.4.1"}}\n'
        (notebook_dir / "renv.lock").write_text(renv_content, encoding="utf-8")

        session = self._make_session(notebook_dir)
        session.ensure_renv_synced()
        first_runtime = load_runtime_state(notebook_dir).r
        assert len(calls) == 1

        # Second call — same lockfile bytes — must short-circuit.
        session.ensure_renv_synced()
        second_runtime = load_runtime_state(notebook_dir).r

        assert len(calls) == 1, "second call must not re-spawn _renv_sync"
        # Runtime entry unchanged — the short-circuit doesn't even
        # re-stamp ``last_synced_at``.
        assert first_runtime == second_runtime

    def test_lockfile_edit_triggers_resync(self, tmp_path: Path, monkeypatch):
        """Editing ``renv.lock`` invalidates the hash, so the next call
        runs ``_renv_sync`` and updates the runtime entry."""
        calls = self._stub_renv_sync(monkeypatch, ok=True)
        notebook_dir = create_notebook(tmp_path, "Edited R", initialize_environment=False)
        (notebook_dir / "renv.lock").write_text('{"R": {"Version": "4.4.0"}}\n', encoding="utf-8")

        session = self._make_session(notebook_dir)
        session.ensure_renv_synced()
        first_hash = load_runtime_state(notebook_dir).r.lock_hash

        (notebook_dir / "renv.lock").write_text('{"R": {"Version": "4.4.1"}}\n', encoding="utf-8")
        session.ensure_renv_synced()
        second_hash = load_runtime_state(notebook_dir).r.lock_hash

        assert len(calls) == 2, "lockfile edit must re-run _renv_sync"
        assert first_hash != second_hash

    def test_failure_leaves_runtime_untouched(self, tmp_path: Path, monkeypatch):
        """``_renv_sync`` returning False (Rscript missing, timeout, etc.)
        must leave any prior ``RRuntime`` entry alone — stamping a fresh
        timestamp after a failure would lie about the last-good sync."""
        self._stub_renv_sync(monkeypatch, ok=False)
        notebook_dir = create_notebook(tmp_path, "Failing R", initialize_environment=False)
        (notebook_dir / "renv.lock").write_text("{}\n", encoding="utf-8")

        # Pre-seed the runtime entry as if a prior good sync had run.
        seed = RRuntime(
            lock_hash="previous-good-hash",
            r_version="4.3.0",
            last_synced_at=1,
            has_lockfile=True,
        )
        state = load_runtime_state(notebook_dir)
        state.r = seed
        save_runtime_state(notebook_dir, state)

        session = self._make_session(notebook_dir)
        session.ensure_renv_synced()

        runtime = load_runtime_state(notebook_dir).r
        # The new lockfile has a different hash, so the short-circuit
        # didn't fire — ``_renv_sync`` was actually called and failed.
        # The prior runtime entry must survive untouched.
        assert runtime == seed

    def test_lockfile_removed_clears_runtime_entry(self, tmp_path: Path, monkeypatch):
        """When a notebook removes its ``renv.lock``, the runtime entry
        clears back to the empty default — no phantom hash hangs around."""
        self._stub_renv_sync(monkeypatch, ok=True)
        notebook_dir = create_notebook(tmp_path, "Cleared R", initialize_environment=False)
        (notebook_dir / "renv.lock").write_text("{}\n", encoding="utf-8")

        session = self._make_session(notebook_dir)
        session.ensure_renv_synced()
        assert load_runtime_state(notebook_dir).r.has_lockfile is True

        # User removes renv.lock and reopens — the next ensure clears.
        (notebook_dir / "renv.lock").unlink()
        session.ensure_renv_synced()

        assert load_runtime_state(notebook_dir).r == RRuntime()
