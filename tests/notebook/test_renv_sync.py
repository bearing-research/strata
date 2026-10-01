"""Tests for ``_renv_sync`` + the ``[r]`` block schema.

Same two-tier shape as ``test_language_r_analyzer.py``:

- Unit tests monkeypatch ``shutil.which`` + ``subprocess.run`` to cover
  the wrapper's success / failure / timeout / missing-Rscript paths.
- Schema tests pin that the ``[r]`` block round-trips through the
  writer + parser.

The capstone real-renv integration tests land with #59.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from strata.notebook.dependencies import renv_add, renv_init, renv_process_lock
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
            # Only Rscript: loading the server config outside a server (to
            # see which environment backend is on) lists Pythons through uv.
            if args[1:2] == ["-e"]:
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
# Cross-process renv lock (issue #102)
# ---------------------------------------------------------------------------

# Holds the renv process lock from a real second process, so the
# contention below is genuine cross-process flock contention — a
# second FileLock instance in the same process would also conflict
# on POSIX, but that's an implementation detail we don't want to
# depend on.
_LOCK_HOLDER_SCRIPT = """
import sys
from filelock import FileLock
lock = FileLock(sys.argv[1])
lock.acquire()
print("held", flush=True)
sys.stdin.read()  # block until the parent closes stdin
"""


@contextmanager
def _hold_renv_lock_in_subprocess(notebook_dir: Path):
    """Spawn a child process that holds *notebook_dir*'s renv lock."""
    lock_path = notebook_dir / ".strata" / "renv-process.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [sys.executable, "-c", _LOCK_HOLDER_SCRIPT, str(lock_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "held"
        yield
    finally:
        assert proc.stdin is not None
        proc.stdin.close()
        proc.wait(timeout=30)


class TestRenvCrossProcessLock:
    """renv mutations are mutually exclusive across processes.

    Issue #102: ``renv::restore()`` from a server and ``strata run``
    (or two CLI runs) in separate processes used to hit the same
    ``renv/library`` + ``renv.lock`` concurrently — the
    ``threading.Lock`` in ``dependencies.py`` only serializes within
    one process and renv has no locking of its own.
    """

    def test_lock_file_lives_under_strata_runtime_dir(self, tmp_path):
        """The lock is runtime state — ``.strata/``, never committed."""
        lock = renv_process_lock(tmp_path)
        assert Path(lock.lock_file) == tmp_path / ".strata" / "renv-process.lock"
        assert (tmp_path / ".strata").is_dir()

    def test_renv_sync_blocked_by_other_process_returns_false(self, monkeypatch, tmp_path):
        """Lock held elsewhere → give up after *timeout*, never spawn Rscript."""
        (tmp_path / "renv.lock").write_text("")
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")

        def boom(*args, **kwargs):
            raise AssertionError("Rscript must not run while another process holds the lock")

        with _hold_renv_lock_in_subprocess(tmp_path):
            monkeypatch.setattr(subprocess, "run", boom)
            assert _renv_sync(tmp_path, timeout=1) is False

    def test_renv_sync_releases_lock_between_calls(self, monkeypatch, tmp_path):
        """Back-to-back syncs in one process must not self-deadlock."""
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")
        (tmp_path / "renv.lock").write_text("")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
        )

        assert _renv_sync(tmp_path) is True
        assert _renv_sync(tmp_path) is True

    async def test_renv_add_blocked_by_other_process_fails_cleanly(self, tmp_path):
        """``renv_add`` surfaces a structured failure, not a corrupted env."""
        with _hold_renv_lock_in_subprocess(tmp_path):
            result = await renv_add(tmp_path, "ggplot2", timeout=1)

        assert result.success is False
        assert result.action == "r_add"
        assert result.package == "ggplot2"
        assert "Another process" in (result.error or "")

    async def test_renv_init_blocked_by_other_process_fails_cleanly(self, tmp_path):
        with _hold_renv_lock_in_subprocess(tmp_path):
            result = await renv_init(tmp_path, timeout=1)

        assert result.success is False
        assert result.action == "r_init"
        assert "Another process" in (result.error or "")


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
    * Reopens short-circuit only when both the lockfile hash AND
      the on-disk renv library match. Library deleted out from
      under us → force a real re-sync.
    * Failed sync records the error but preserves the last-good
      ``lock_hash`` / ``r_version`` / ``last_synced_at`` so the UI
      can still show "you had a working env at <T>".
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

    @staticmethod
    def _seed_renv_library(notebook_dir: Path) -> None:
        """Create the ``renv/library`` directory **with at least one
        entry** so ``_renv_library_present`` returns True.

        The fake ``_renv_sync`` doesn't actually install anything;
        the short-circuit path needs the library to look like a real
        renv project's library, not just an empty directory. We
        write a single sentinel directory under
        ``renv/library/<platform>/`` to mirror renv's actual
        layout — the probe stops at the first child entry so we
        don't have to build a full package tree.
        """
        library = notebook_dir / "renv" / "library" / "x86_64-pc-linux-gnu-R-4.4"
        library.mkdir(parents=True, exist_ok=True)
        # Stub a package directory so a future "library must contain
        # a real package" tightening doesn't silently re-break this.
        (library / "stub_package").mkdir(exist_ok=True)

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
        assert runtime.sync_error == ""

        # The on-disk [r] block stays empty (the committed config has
        # no opinion on runtime state) and notebook.toml is byte-
        # identical — no churn.
        assert parse_notebook(notebook_dir).r == {}
        assert (notebook_dir / "notebook.toml").read_bytes() == toml_before

    def test_hash_and_library_match_short_circuits(self, tmp_path: Path, monkeypatch):
        """A second open against the same lockfile + present library
        must NOT spawn Rscript. The session-reuse path calls
        ``ensure_renv_synced`` on every reopen; the short-circuit
        keeps that free."""
        calls = self._stub_renv_sync(monkeypatch, ok=True)
        notebook_dir = create_notebook(tmp_path, "Cached R", initialize_environment=False)
        renv_content = '{"R": {"Version": "4.4.1"}}\n'
        (notebook_dir / "renv.lock").write_text(renv_content, encoding="utf-8")

        session = self._make_session(notebook_dir)
        session.ensure_renv_synced()
        # The fake ``_renv_sync`` doesn't populate ``renv/library/`` on
        # its own (that's the real renv's job); seed it so the
        # short-circuit fires on the second call.
        self._seed_renv_library(notebook_dir)
        first_runtime = load_runtime_state(notebook_dir).r
        assert len(calls) == 1

        session.ensure_renv_synced()
        second_runtime = load_runtime_state(notebook_dir).r

        assert len(calls) == 1, "second call must not re-spawn _renv_sync"
        assert first_runtime == second_runtime

    def test_short_circuit_requires_library_on_disk(self, tmp_path: Path, monkeypatch):
        """Hash match alone is NOT enough — if the project library
        directory has been deleted out from under us, force a real
        ``_renv_sync`` to restore it. Otherwise the runtime metadata
        would survive while the actual library doesn't, and the next
        R cell would fail with "no package called ...".
        """
        import shutil as _shutil

        calls = self._stub_renv_sync(monkeypatch, ok=True)
        notebook_dir = create_notebook(tmp_path, "Library Gone", initialize_environment=False)
        (notebook_dir / "renv.lock").write_text("{}\n", encoding="utf-8")

        session = self._make_session(notebook_dir)
        session.ensure_renv_synced()
        self._seed_renv_library(notebook_dir)
        assert len(calls) == 1

        # User wipes the library directory (manual cleanup, container
        # rebuild, .strata removal that took renv with it).
        _shutil.rmtree(notebook_dir / "renv" / "library")

        session.ensure_renv_synced()

        assert len(calls) == 2, (
            "missing renv/library must trigger a fresh _renv_sync "
            "even when the lockfile hash hasn't changed"
        )

    def test_short_circuit_rejects_empty_library_directory(self, tmp_path: Path, monkeypatch):
        """``renv/library/`` existing as an empty directory does NOT count
        as a healthy library. Pre-fix the probe only checked
        ``.exists()`` — an empty dir (left over from an aborted
        restore, manual cleanup that wiped the contents but not the
        parent, or a test fixture) would pass and the UI would
        report "in sync" while packages are missing.
        """
        calls = self._stub_renv_sync(monkeypatch, ok=True)
        notebook_dir = create_notebook(tmp_path, "Empty Library", initialize_environment=False)
        (notebook_dir / "renv.lock").write_text("{}\n", encoding="utf-8")

        session = self._make_session(notebook_dir)
        session.ensure_renv_synced()
        assert len(calls) == 1

        # Recreate ``renv/library`` as an empty directory — the
        # probe-only-checks-exists() bug would have passed this.
        (notebook_dir / "renv" / "library").mkdir(parents=True, exist_ok=True)

        session.ensure_renv_synced()

        assert len(calls) == 2, (
            "empty renv/library must trigger a fresh _renv_sync; the "
            "directory existing alone is not evidence the project is restored"
        )

    def test_lockfile_edit_triggers_resync(self, tmp_path: Path, monkeypatch):
        """Editing ``renv.lock`` invalidates the hash, so the next call
        runs ``_renv_sync`` and updates the runtime entry."""
        calls = self._stub_renv_sync(monkeypatch, ok=True)
        notebook_dir = create_notebook(tmp_path, "Edited R", initialize_environment=False)
        (notebook_dir / "renv.lock").write_text('{"R": {"Version": "4.4.0"}}\n', encoding="utf-8")

        session = self._make_session(notebook_dir)
        session.ensure_renv_synced()
        self._seed_renv_library(notebook_dir)
        first_hash = load_runtime_state(notebook_dir).r.lock_hash

        (notebook_dir / "renv.lock").write_text('{"R": {"Version": "4.4.1"}}\n', encoding="utf-8")
        session.ensure_renv_synced()
        second_hash = load_runtime_state(notebook_dir).r.lock_hash

        assert len(calls) == 2, "lockfile edit must re-run _renv_sync"
        assert first_hash != second_hash

    def test_failure_preserves_last_good_state(self, tmp_path: Path, monkeypatch):
        """``_renv_sync`` returning False records ``sync_error`` but
        preserves the last-good ``lock_hash`` / ``r_version`` /
        ``last_synced_at``. The UI can then show "last good sync
        was at T, latest attempt failed" instead of losing the prior
        state.
        """
        self._stub_renv_sync(monkeypatch, ok=False)
        notebook_dir = create_notebook(tmp_path, "Failing R", initialize_environment=False)
        (notebook_dir / "renv.lock").write_text("{}\n", encoding="utf-8")

        # Pre-seed the runtime entry as if a prior good sync had run.
        state = load_runtime_state(notebook_dir)
        state.r = RRuntime(
            lock_hash="previous-good-hash",
            r_version="4.3.0",
            last_synced_at=1,
            sync_error="",
        )
        save_runtime_state(notebook_dir, state)

        session = self._make_session(notebook_dir)
        session.ensure_renv_synced()

        runtime = load_runtime_state(notebook_dir).r
        # Last-good fields survive; sync_error populated.
        assert runtime.lock_hash == "previous-good-hash"
        assert runtime.r_version == "4.3.0"
        assert runtime.last_synced_at == 1
        assert runtime.sync_error != ""

    def test_lockfile_removed_clears_runtime_entry(self, tmp_path: Path, monkeypatch):
        """When a notebook removes its ``renv.lock``, the runtime entry
        clears back to the empty default — no phantom hash or error
        hangs around."""
        self._stub_renv_sync(monkeypatch, ok=True)
        notebook_dir = create_notebook(tmp_path, "Cleared R", initialize_environment=False)
        (notebook_dir / "renv.lock").write_text("{}\n", encoding="utf-8")

        session = self._make_session(notebook_dir)
        session.ensure_renv_synced()
        assert load_runtime_state(notebook_dir).r.last_synced_at > 0

        # User removes renv.lock and reopens — the next ensure clears.
        (notebook_dir / "renv.lock").unlink()
        session.ensure_renv_synced()

        assert load_runtime_state(notebook_dir).r == RRuntime()


# ---------------------------------------------------------------------------
# serialize_r_environment_state — payload shape the UI consumes
# ---------------------------------------------------------------------------


class TestSerializeREnvironmentState:
    """``serialize_r_environment_state`` derives ``has_lockfile`` and
    ``sync_state`` from current disk state + runtime metadata so the
    UI sees the truth — particularly important for never-synced and
    failed-sync notebooks where the panel must STILL render so the
    user can see *why* the env is broken.
    """

    def _make_session(self, notebook_dir):
        from strata.notebook.session import NotebookSession

        state = parse_notebook(notebook_dir)
        return NotebookSession(state, notebook_dir)

    def test_no_lockfile_returns_absent(self, tmp_path: Path):
        notebook_dir = create_notebook(tmp_path, "No R", initialize_environment=False)
        payload = self._make_session(notebook_dir).serialize_r_environment_state()
        assert payload["has_lockfile"] is False
        assert payload["sync_state"] == "absent"
        assert payload["sync_error"] is None

    def test_lockfile_but_never_synced_renders_never(self, tmp_path: Path):
        """User adds renv.lock but the sync hasn't run yet. UI must
        still render the R section so they can see the state."""
        notebook_dir = create_notebook(tmp_path, "Pending R", initialize_environment=False)
        (notebook_dir / "renv.lock").write_text('{"R": {"Version": "4.4.1"}}\n', encoding="utf-8")

        payload = self._make_session(notebook_dir).serialize_r_environment_state()

        assert payload["has_lockfile"] is True
        assert payload["sync_state"] == "never"
        assert payload["sync_error"] is None

    def test_failed_sync_surfaces_failed_state(self, tmp_path: Path):
        """A failed sync stays visible via ``sync_state='failed'`` +
        ``sync_error``. The panel must NOT hide — the user needs to
        see *why* their R env is broken (the exact regression Codex
        called out)."""
        notebook_dir = create_notebook(tmp_path, "Broken R", initialize_environment=False)
        (notebook_dir / "renv.lock").write_text("{}\n", encoding="utf-8")
        state = load_runtime_state(notebook_dir)
        state.r = RRuntime(sync_error="renv::restore() failed: Rscript not on PATH")
        save_runtime_state(notebook_dir, state)

        payload = self._make_session(notebook_dir).serialize_r_environment_state()

        assert payload["has_lockfile"] is True
        assert payload["sync_state"] == "failed"
        assert payload["sync_error"] is not None
        assert "Rscript" in payload["sync_error"]

    def test_outdated_when_lockfile_edited_after_last_good_sync(self, tmp_path: Path):
        """User edits renv.lock between syncs: on-disk hash diverges
        from runtime.r.lock_hash, no sync_error yet. UI shows
        'outdated' so the user knows to re-sync."""
        notebook_dir = create_notebook(tmp_path, "Outdated R", initialize_environment=False)
        (notebook_dir / "renv.lock").write_text('{"R": {"Version": "4.4.1"}}\n', encoding="utf-8")
        state = load_runtime_state(notebook_dir)
        state.r = RRuntime(
            lock_hash="0" * 64,
            r_version="4.4.0",
            last_synced_at=1000,
            sync_error="",
        )
        save_runtime_state(notebook_dir, state)

        payload = self._make_session(notebook_dir).serialize_r_environment_state()

        assert payload["sync_state"] == "outdated"
        # Last-good fields survive so the UI can show them.
        assert payload["r_version"] == "4.4.0"

    def test_ok_when_hash_matches(self, tmp_path: Path):
        """Happy path: current renv.lock hash matches runtime.lock_hash,
        no error → ``sync_state='ok'``."""
        import hashlib

        notebook_dir = create_notebook(tmp_path, "Healthy R", initialize_environment=False)
        renv_content = '{"R": {"Version": "4.4.1"}, "Packages": {"arrow": "1.0"}}\n'
        (notebook_dir / "renv.lock").write_text(renv_content, encoding="utf-8")
        lock_hash = hashlib.sha256(renv_content.encode()).hexdigest()
        state = load_runtime_state(notebook_dir)
        state.r = RRuntime(
            lock_hash=lock_hash,
            r_version="4.4.1",
            last_synced_at=2000,
            sync_error="",
        )
        save_runtime_state(notebook_dir, state)

        payload = self._make_session(notebook_dir).serialize_r_environment_state()

        assert payload["sync_state"] == "ok"
        assert payload["current_lock_hash"] == lock_hash
        assert payload["lock_hash"] == lock_hash


# ---------------------------------------------------------------------------
# Who builds R packages
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="the recording Rscript is a shell script")
class TestRCodeRunsAsTheHarnessUser:
    """``renv::restore()`` builds source packages, which runs their configure
    scripts, and every Rscript started in a notebook directory sources the
    notebook's ``.Rprofile``. Where cells run as the harness user, so does
    that code; a service-mode server with no harness user runs none of it.

    The harness user here is the current user, as in ``test_harness_user.py``:
    dropping to someone else needs root. What shows the drop is the identity
    the spawn is given and the environment it is filtered to.
    """

    @pytest.fixture
    def recorded(self, tmp_path, monkeypatch):
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        record = tmp_path / "rscript-run.txt"
        fake = bin_dir / "Rscript"
        target = shlex.quote(str(record))
        fake.write_text(
            f'#!/bin/sh\n{{ echo "uid=$(id -u)"; echo "cwd=$(pwd)"; env; }} > {target}\n'
        )
        fake.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
        # The server's own environment: a home that is not the harness user's,
        # and a credential cells are not given.
        monkeypatch.setenv("HOME", str(tmp_path / "server-home"))
        monkeypatch.setenv("SERVER_ONLY_SECRET", "s3cret")

        def read() -> dict[str, str] | None:
            if not record.exists():
                return None
            lines = record.read_text().splitlines()
            return dict(line.split("=", 1) for line in lines if "=" in line)

        return read

    @staticmethod
    def _server(monkeypatch, *, mode: str, user: str | None) -> None:
        monkeypatch.setattr(
            "strata.server._state",
            SimpleNamespace(
                config=SimpleNamespace(
                    deployment_mode=mode,
                    notebook_harness_user=user,
                    notebook_harness_env_allowlist=["PATH"],
                )
            ),
        )

    @staticmethod
    def _me():
        import pwd

        return pwd.getpwuid(os.getuid())

    def _assert_ran_as_the_harness_user(self, run, notebook_dir):
        assert run is not None, "Rscript was not started"
        assert run["uid"] == str(os.getuid())
        assert run["HOME"] == self._me().pw_dir
        assert run["USER"] == self._me().pw_name
        assert "SERVER_ONLY_SECRET" not in run
        assert Path(run["cwd"]).resolve() == notebook_dir.resolve()

    def test_restore_runs_as_the_harness_user(self, tmp_path, recorded, monkeypatch):
        notebook_dir = tmp_path / "nb"
        notebook_dir.mkdir()
        (notebook_dir / "renv.lock").write_text("{}")
        self._server(monkeypatch, mode="service", user=self._me().pw_name)

        assert _renv_sync(notebook_dir) is True
        self._assert_ran_as_the_harness_user(recorded(), notebook_dir)

    def test_service_mode_without_a_harness_user_does_not_restore(
        self, tmp_path, recorded, monkeypatch, caplog
    ):
        notebook_dir = tmp_path / "nb"
        notebook_dir.mkdir()
        (notebook_dir / "renv.lock").write_text("{}")
        self._server(monkeypatch, mode="service", user=None)

        with caplog.at_level("WARNING", logger="strata.notebook.writer"):
            assert _renv_sync(notebook_dir) is False

        assert recorded() is None
        assert "STRATA_NOTEBOOK_HARNESS_USER" in caplog.text

    @pytest.mark.parametrize("points_into_the_store", [True, False])
    def test_a_library_link_is_handed_over_only_inside_the_shared_store(
        self, tmp_path, recorded, monkeypatch, points_into_the_store
    ):
        """Cell code runs as the harness user, which owns renv/ after a restore
        and can repoint renv/library. The next restore runs as the server and
        hands the link's target to the harness user: anywhere the link names
        would become that user's, so only the shared store's libraries are."""
        from strata.notebook import writer
        from strata.notebook.harness_user import HarnessUser

        me = self._me()
        user = HarnessUser(name=me.pw_name, uid=me.pw_uid, gid=me.pw_gid, home=me.pw_dir)
        store = tmp_path / "envs" / "r"
        (store / "cache").mkdir(parents=True)
        target = store / "abc123" if points_into_the_store else tmp_path / "server-state"
        target.mkdir()
        notebook_dir = tmp_path / "nb"
        (notebook_dir / "renv").mkdir(parents=True)
        (notebook_dir / "renv.lock").write_text("{}")
        (notebook_dir / "renv" / "library").symlink_to(target, target_is_directory=True)
        handed: list[Path] = []
        monkeypatch.setattr(writer, "hand_over", lambda path, _user: handed.append(Path(path)))

        restored = writer._renv_restore_locked(
            notebook_dir,
            timeout=30,
            env={"RENV_PATHS_CACHE": str(store / "cache")},
            harness_user=user,
        )

        assert restored is points_into_the_store
        assert (target.resolve() in [h.resolve() for h in handed]) is points_into_the_store
        if not points_into_the_store:
            assert recorded() is None

    def test_personal_mode_restores_as_the_server(self, tmp_path, recorded, monkeypatch):
        notebook_dir = tmp_path / "nb"
        notebook_dir.mkdir()
        (notebook_dir / "renv.lock").write_text("{}")
        self._server(monkeypatch, mode="personal", user=None)

        assert _renv_sync(notebook_dir) is True
        run = recorded()
        assert run is not None
        assert run["HOME"] == str(tmp_path / "server-home")

    def test_listing_packages_runs_as_the_harness_user(self, tmp_path, recorded, monkeypatch):
        from strata.notebook.dependencies import list_r_packages

        self._server(monkeypatch, mode="service", user=self._me().pw_name)

        list_r_packages(tmp_path)
        self._assert_ran_as_the_harness_user(recorded(), tmp_path)

    def test_listing_packages_is_refused_without_a_harness_user(
        self, tmp_path, recorded, monkeypatch
    ):
        from strata.notebook.dependencies import list_r_packages

        self._server(monkeypatch, mode="service", user=None)

        listing = list_r_packages(tmp_path)

        assert recorded() is None
        assert listing.status == "failed"
        assert "STRATA_NOTEBOOK_HARNESS_USER" in (listing.error or "")

    @pytest.mark.parametrize("user", [None, "me"])
    async def test_installing_a_package_is_refused_where_cells_are_isolated(
        self, tmp_path, recorded, monkeypatch, user
    ):
        """renv writes ``renv.lock`` and ``.Rprofile`` into the notebook
        directory, which the harness user cannot write, so an install cannot
        drop to it; it does not run as the server either."""
        self._server(monkeypatch, mode="service", user=self._me().pw_name if user else None)

        added = await renv_add(tmp_path, "ggplot2", timeout=5)
        initialized = await renv_init(tmp_path, timeout=5)

        assert recorded() is None
        for result in (added, initialized):
            assert result.success is False
            assert "renv.lock" in (result.error or "")
