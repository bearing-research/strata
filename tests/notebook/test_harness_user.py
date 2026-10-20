"""Who a cell runs as, and whether a service-mode host starts it at all.

A cell running as the server's user could read ``/proc/<server pid>/environ``, so a
service-mode server refuses to run cell code on its own host unless cells run as a
separate OS user. Switching users needs root, so here the harness user is the current
user; the root-only check lives in ``tests/test_harness_isolation.py``.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from strata.notebook.harness_user import (
    REFUSAL,
    HarnessUser,
    LocalExecutionRefused,
    _process_table,
    hand_over,
    identity_env,
    leftover_processes,
    prepare_work_dir,
    resolve_harness_user,
    spawn_kwargs,
    sweep_leftovers,
)

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="OS users are POSIX here")


def _server(monkeypatch, *, mode: str, user: str | None = None) -> None:
    monkeypatch.setattr(
        "strata.server._state",
        SimpleNamespace(config=SimpleNamespace(deployment_mode=mode, notebook_harness_user=user)),
    )


def _me() -> str:
    import pwd

    return pwd.getpwuid(os.getuid()).pw_name


def _session(tmp_path, source: str = "x = 1"):
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    notebook_dir = create_notebook(tmp_path, "Isolated")
    add_cell_to_notebook(notebook_dir, "c1", None)
    write_cell(notebook_dir, "c1", source)
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


class TestResolution:
    def test_personal_mode_runs_as_the_server(self, monkeypatch):
        """One person on their own machine has nothing to isolate a cell from."""
        _server(monkeypatch, mode="personal")

        assert resolve_harness_user() is None

    def test_outside_a_server_nothing_is_refused(self, monkeypatch):
        """A CLI run has no server credentials for a cell to read, and falling back to the loaded
        config's default (service) mode would refuse every `strata run`.
        """
        monkeypatch.setattr("strata.server._state", None)

        assert resolve_harness_user() is None

    def test_service_mode_without_a_harness_user_refuses(self, monkeypatch):
        _server(monkeypatch, mode="service")

        with pytest.raises(LocalExecutionRefused) as caught:
            resolve_harness_user()

        # Both ways out, named.
        assert "server-managed worker" in str(caught.value)
        assert "STRATA_NOTEBOOK_HARNESS_USER" in str(caught.value)

    @posix_only
    def test_a_user_that_does_not_exist_is_named(self, monkeypatch):
        _server(monkeypatch, mode="service", user="no-such-user-strata")

        with pytest.raises(LocalExecutionRefused, match="no-such-user-strata"):
            resolve_harness_user()

    @posix_only
    def test_switching_to_someone_else_without_root_is_explained(self, monkeypatch):
        """Rather than the bare PermissionError Popen(user=) would raise."""
        if os.geteuid() == 0:
            pytest.skip("root can switch users")
        _server(monkeypatch, mode="service", user="nobody")

        with pytest.raises(LocalExecutionRefused, match="root"):
            resolve_harness_user()

    @posix_only
    def test_the_configured_user_is_resolved(self, monkeypatch):
        _server(monkeypatch, mode="service", user=_me())

        user = resolve_harness_user()

        assert user is not None
        assert (user.name, user.uid) == (_me(), os.getuid())


@posix_only
class TestSpawnArguments:
    def test_the_identity_travels_with_the_uid(self):
        """Popen(user=) changes the uid and nothing else; a cell told root's
        HOME fails every cache write under ``~``."""
        user = HarnessUser(name="cells", uid=1234, gid=1234, home="/home/cells")

        env = identity_env({"HOME": "/root", "USER": "root", "KEEP": "1"}, user)

        assert env == {"HOME": "/home/cells", "USER": "cells", "LOGNAME": "cells", "KEEP": "1"}

    def test_supplementary_groups_are_only_cleared_by_root(self):
        user = HarnessUser(name="cells", uid=1234, gid=1234, home="/home/cells")

        kwargs = spawn_kwargs(user)

        assert (kwargs["user"], kwargs["group"]) == (1234, 1234)
        assert ("extra_groups" in kwargs) is (os.geteuid() == 0)

    def test_no_user_changes_nothing(self):
        assert spawn_kwargs(None) == {}
        assert identity_env(None, None) is None

    @pytest.mark.skipif(sys.platform != "linux", reason="PR_SET_CHILD_SUBREAPER is Linux's")
    def test_a_daemon_a_run_starts_with_setsid_stays_its_descendant(self):
        """``ray start`` and the like fork, ``setsid`` and exit, orphaning their daemon. The run
        adopts it rather than init, so the sweep holds it while the run lives."""
        import pwd
        import subprocess

        me = HarnessUser(_me(), os.getuid(), os.getgid(), pwd.getpwuid(os.getuid()).pw_dir)
        script = (
            "import os, time\n"
            "r, w = os.pipe()\n"
            "middle = os.fork()\n"
            "if middle == 0:\n"
            "    os.setsid()\n"
            "    mid = os.getpid()\n"
            "    if os.fork() == 0:\n"
            "        while os.getppid() == mid:\n"
            "            time.sleep(0.01)\n"
            "        os.write(w, str(os.getppid()).encode())\n"
            "    os._exit(0)\n"
            "os.waitpid(middle, 0)\n"
            "os.close(w)\n"
            "print(os.getpid(), os.read(r, 64).decode())\n"
        )

        run = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
            **spawn_kwargs(me),
        )

        harness, adopter = run.stdout.split()
        assert adopter == harness


@posix_only
class TestHandOver:
    """Chowning to another user needs root, so ``os.chown`` records what it would change."""

    @staticmethod
    def _record_chowns(monkeypatch, on_chown=None) -> set[tuple[int, int]]:
        changed: set[tuple[int, int]] = set()

        def _chown(path, uid, gid, *, dir_fd=None, follow_symlinks=True):
            st = os.stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
            changed.add((st.st_dev, st.st_ino))
            if on_chown is not None:
                on_chown(st)

        monkeypatch.setattr(os, "chown", _chown)
        return changed

    @staticmethod
    def _ids(*paths) -> set[tuple[int, int]]:
        return {(st.st_dev, st.st_ino) for st in (os.lstat(p) for p in paths)}

    @staticmethod
    def _cells() -> HarnessUser:
        return HarnessUser(name="cells", uid=os.geteuid() + 1, gid=os.getegid(), home="/")

    def test_every_entry_but_a_link_is_handed_over(self, tmp_path, monkeypatch):
        """A symlink is left as it is: the user owns its directory, so it can replace it."""
        outside = tmp_path / "outside"
        outside.mkdir()
        run = tmp_path / "run"
        (run / "a" / "b").mkdir(parents=True)
        (run / "a" / "b" / "f").write_text("x")
        (run / "g").write_text("y")
        (run / "link").symlink_to(outside)
        changed = self._record_chowns(monkeypatch)

        hand_over(run, self._cells())

        tree = [run, run / "a", run / "a" / "b", run / "a" / "b" / "f", run / "g"]
        assert changed == self._ids(*tree)

    def test_a_file_with_another_hard_link_is_not_handed_over(self, tmp_path, monkeypatch, caplog):
        """With ``fs.protected_hardlinks=0`` the user can link a server file into a directory it
        already owns (``renv/``, the renv cache); chowning that entry would hand the file over."""
        secret = tmp_path / "secret"
        secret.write_text("server's")
        run = tmp_path / "run"
        run.mkdir()
        os.link(secret, run / "planted")
        (run / "mine").write_text("x")
        changed = self._record_chowns(monkeypatch)

        hand_over(run, self._cells())

        assert changed == self._ids(run, run / "mine")
        assert "planted" in caplog.text and "hard link" in caplog.text

    def test_a_fifo_and_a_socket_the_user_left_do_not_stop_a_hand_over(self, tmp_path, monkeypatch):
        """A FIFO is opened without blocking; a socket cannot be opened and is left as it is."""
        import socket

        run = tmp_path / "run"
        run.mkdir()
        os.mkfifo(run / "fifo")
        monkeypatch.chdir(run)  # a socket path is capped near 100 bytes
        sock = socket.socket(socket.AF_UNIX)
        sock.bind("sock")
        changed = self._record_chowns(monkeypatch)
        try:
            hand_over(run, self._cells())
        finally:
            sock.close()

        assert changed == self._ids(run, run / "fifo")

    def test_a_directory_swapped_for_a_link_mid_walk_is_not_followed(self, tmp_path, monkeypatch):
        """A harness-user process can rename a directory it owns and put a link in its place
        while the walk is still in a sibling; root would then chown what the link names."""
        outside = tmp_path / "outside"
        (outside / "secret").mkdir(parents=True)
        run = tmp_path / "run"
        for name in ("a", "c"):
            (run / name).mkdir(parents=True)
            (run / name / "entry").write_text(name)
        triggers = {os.lstat(run / name / "entry").st_ino: name for name in ("a", "c")}
        swapped: list[str] = []

        def _attack(st) -> None:
            # Once the walk is inside one directory, swap the other, already listed, one.
            inside = triggers.get(st.st_ino)
            if inside is None or swapped:
                return
            other = "c" if inside == "a" else "a"
            (run / other).rename(tmp_path / f"moved-{other}")
            (run / other).symlink_to(outside)
            swapped.append(other)

        changed = self._record_chowns(monkeypatch, _attack)

        hand_over(run, self._cells())

        assert swapped
        assert not changed & self._ids(outside, outside / "secret")

    def test_a_run_directory_that_is_a_link_hands_over_nothing_it_names(
        self, tmp_path, monkeypatch
    ):
        outside = tmp_path / "outside"
        (outside / "secret").mkdir(parents=True)
        run = tmp_path / "run"
        run.symlink_to(outside)
        changed = self._record_chowns(monkeypatch)

        hand_over(run, self._cells())

        assert not changed & self._ids(outside, outside / "secret")


class TestEveryPlaceCellCodeStarts:
    """Refused on each site, and each refusal reaches the person running it."""

    def test_a_cell_is_refused_with_the_reason(self, tmp_path, monkeypatch):
        session = _session(tmp_path)
        _server(monkeypatch, mode="service")
        from strata.notebook.executor import CellExecutor

        result = asyncio.run(CellExecutor(session).execute_cell("c1", "x = 1"))

        assert result.success is False
        assert result.error is not None and REFUSAL in result.error

    def test_a_cache_hit_is_still_served(self, tmp_path, monkeypatch):
        """A hit starts no cell code, so there is nothing to refuse."""
        from strata.notebook.executor import CellExecutor
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import add_cell_to_notebook, write_cell

        session = _session(tmp_path, "x = 41 + 1")
        # A reader, so c1's value is stored and a rerun can hit it.
        add_cell_to_notebook(session.path, "c2", "c1")
        write_cell(session.path, "c2", "y = x")
        session = NotebookSession(parse_notebook(session.path), session.path)
        session.refresh_environment_runtime()

        _server(monkeypatch, mode="personal")
        first = asyncio.run(CellExecutor(session).execute_cell("c1", "x = 41 + 1"))
        assert first.success, first.error

        _server(monkeypatch, mode="service")
        again = asyncio.run(CellExecutor(session).execute_cell("c1", "x = 41 + 1"))

        assert again.success, again.error
        assert again.cache_hit is True

    @posix_only
    def test_with_a_harness_user_the_cell_runs(self, tmp_path, monkeypatch):
        from strata.notebook.executor import CellExecutor

        session = _session(tmp_path)
        _server(monkeypatch, mode="service", user=_me())
        # The server's HOME is not the harness user's; the cell must be told its own.
        monkeypatch.setenv("HOME", str(tmp_path / "server-home"))
        source = "import os\nprint('HOME=' + os.environ['HOME'])"

        result = asyncio.run(CellExecutor(session).execute_cell("c1", source))

        assert result.success, result.error
        import pwd

        assert f"HOME={pwd.getpwnam(_me()).pw_dir}" in result.stdout

    def test_an_embedded_worker_is_this_host_too(self, tmp_path, monkeypatch):
        """``embedded://`` runs the harness in-place. It is a worker by name and
        not another machine, so it does not count as running cells elsewhere."""
        from strata.notebook.executor import CellExecutor

        session = _session(tmp_path)
        monkeypatch.setattr(
            "strata.server._state",
            SimpleNamespace(
                config=SimpleNamespace(
                    deployment_mode="service",
                    transforms_config={
                        "notebook_workers": [
                            {
                                "name": "gpu-a100",
                                "backend": "executor",
                                "runtime_id": "cuda-12.4",
                                "config": {"url": "embedded://local"},
                            }
                        ]
                    },
                )
            ),
        )
        session.notebook_state.worker = "gpu-a100"

        result = asyncio.run(CellExecutor(session).execute_cell("c1", "x = 1"))

        assert result.success is False
        assert result.error is not None and REFUSAL in result.error

    def test_a_batch_called_directly_reports_the_refusal_on_each_cell(self, tmp_path, monkeypatch):
        from strata.notebook.executor import CellExecutor

        session = _session(tmp_path)
        _server(monkeypatch, mode="service")

        batch = asyncio.run(
            CellExecutor(session)._run_batch(
                [{"cell_id": "c1"}, {"cell_id": "c2"}],
                use_cache=True,
                batch_timeout_seconds=30,
            )
        )

        assert batch.completed is False
        assert [r.status for r in batch.cell_results] == ["cell_error", "cell_error"]
        assert all(REFUSAL in (r.error or "") for r in batch.cell_results)

    def test_an_r_cell_is_refused_before_looking_for_r(self, tmp_path, monkeypatch):
        """Refused whether or not R is installed, so the reason is the refusal
        and not a missing Rscript."""
        from strata.notebook.executor import CellExecutor

        session = _session(tmp_path)
        _server(monkeypatch, mode="service")

        result = asyncio.run(
            CellExecutor(session)._run_r_harness(tmp_path / "manifest.json", timeout_seconds=5)
        )

        assert result["success"] is False
        assert result["error"] == REFUSAL

    def test_the_inspect_repl_is_refused(self, tmp_path, monkeypatch):
        """It evaluates whatever is typed into it."""
        from strata.notebook.inspect_repl import InspectSession

        session = _session(tmp_path)
        _server(monkeypatch, mode="service")

        answer = asyncio.run(InspectSession("c1").start(session))

        assert answer == REFUSAL

    def test_cell_tests_are_refused(self, tmp_path, monkeypatch):
        """A test run imports the cell's source."""
        from strata.notebook.executor import CellExecutor

        session = _session(tmp_path)
        _server(monkeypatch, mode="service")

        result = asyncio.run(
            CellExecutor(session).run_cell_tests("c1", "def test_x(cell):\n    assert True\n")
        )

        assert result.errored == 1
        assert REFUSAL in result.tests[0].message

    def test_no_warm_worker_is_spawned(self, tmp_path, monkeypatch):
        from strata.notebook import pool as pool_module

        spawned = []

        async def _spawn(*args, **kwargs):
            spawned.append(args)
            raise RuntimeError("should not spawn")

        monkeypatch.setattr(pool_module.asyncio, "create_subprocess_exec", _spawn)
        _server(monkeypatch, mode="service")

        asyncio.run(pool_module.WarmProcessPool(tmp_path, pool_size=1)._spawn_warm_process())

        assert spawned == []


class TestLeftoverProcesses:
    """Which harness-user processes a run no longer holds. A table is pid -> (parent, uid,
    session); the server is 100, started by 50 (a supervisor that runs other workers)."""

    CELLS, OTHER = 2000, 3000
    BASE = {1: (0, 0, 1), 50: (1, 0, 50), 100: (50, 0, 50)}

    def _leftovers(self, **procs: tuple[int, int, int]) -> list[int]:
        table = dict(self.BASE)
        table.update({int(pid[1:]): proc for pid, proc in procs.items()})
        return sorted(leftover_processes(table, 100, self.CELLS))

    def test_a_running_harness_and_its_children_are_held(self):
        assert self._leftovers(p200=(100, self.CELLS, 200), p201=(200, self.CELLS, 200)) == []

    def test_an_orphan_a_subreaper_above_the_server_adopted_is_a_leftover(self):
        """``systemd --user`` or ``tini -s`` adopts what an ended run left; the server's
        ancestors hold nothing."""
        assert self._leftovers(p300=(50, self.CELLS, 300), p301=(300, self.CELLS, 300)) == [
            300,
            301,
        ]

    def test_a_setsid_daemon_its_running_harness_adopted_is_held(self):
        """A harness is a subreaper, so what its run orphans stays under it."""
        assert self._leftovers(p200=(100, self.CELLS, 200), p203=(200, self.CELLS, 203)) == []

    def test_an_orphan_still_in_a_running_cells_session_is_held(self):
        """``cmd &`` from a shell the cell ran: init adopts it, the session stays."""
        assert self._leftovers(p200=(100, self.CELLS, 200), p202=(1, self.CELLS, 200)) == []

    def test_a_process_that_left_its_session_is_a_leftover_while_its_cell_runs(self):
        assert self._leftovers(p200=(100, self.CELLS, 200), p203=(1, self.CELLS, 203)) == [203]

    def test_an_orphan_whose_cell_ended_is_a_leftover(self):
        assert self._leftovers(p202=(1, self.CELLS, 200)) == [202]

    def test_a_double_forked_daemon_and_its_children_are_leftovers(self):
        assert self._leftovers(p400=(1, self.CELLS, 399), p401=(400, self.CELLS, 399)) == [
            400,
            401,
        ]

    def test_another_users_orphan_is_left_alone(self):
        assert self._leftovers(p500=(1, self.OTHER, 500)) == []

    def test_a_parent_loop_from_a_reused_pid_is_not_held(self):
        assert self._leftovers(p600=(601, self.CELLS, 600), p601=(600, self.CELLS, 600)) == [
            600,
            601,
        ]


@posix_only
class TestSweep:
    CELLS = 2000

    @pytest.fixture
    def killed(self, monkeypatch) -> list[tuple[int, int]]:
        """Every signal sent, as (pid, signal); a pidfd stands in for its pid."""
        from strata.notebook import harness_user

        sent: list[tuple[int, int]] = []
        pids: dict[int, int] = {}

        def _pidfd_open(pid: int) -> int:
            fd = os.open(os.devnull, os.O_RDONLY)
            pids[fd] = pid
            return fd

        monkeypatch.setattr(harness_user.sys, "platform", "linux")
        monkeypatch.setattr(harness_user.os, "kill", lambda pid, sig: sent.append((pid, sig)))
        monkeypatch.setattr(harness_user.os, "pidfd_open", _pidfd_open, raising=False)
        monkeypatch.setattr(
            harness_user.signal,
            "pidfd_send_signal",
            lambda fd, sig: sent.append((pids[fd], sig)),
            raising=False,
        )
        self._tables(monkeypatch, lambda: {700: (1, self.CELLS, 700)})
        return sent

    @staticmethod
    def _tables(monkeypatch, procs) -> None:
        """Each scan sees init, the server and what ``procs()`` returns then."""
        from strata.notebook import harness_user

        server = os.getpid()
        monkeypatch.setattr(
            harness_user,
            "_process_table",
            lambda: {1: (0, 0, 1), server: (1, 0, server), **procs()},
        )

    def _cells(self, uid: int = CELLS) -> HarnessUser:
        return HarnessUser(name="cells", uid=uid, gid=uid, home="/")

    def test_a_leftover_is_stopped_then_killed(self, killed):
        import signal

        sweep_leftovers(self._cells())

        assert killed == [(700, signal.SIGSTOP), (700, signal.SIGKILL)]

    def test_what_a_leftover_forked_is_stopped_before_anything_is_killed(self, killed, monkeypatch):
        """Killed a pass at a time, what the last pass missed would outlive the sweep."""
        import signal

        def _procs() -> dict[int, tuple[int, int, int]]:
            procs = {700: (1, self.CELLS, 700)}
            if (700, signal.SIGSTOP) in killed:
                procs[701] = (700, self.CELLS, 700)  # forked just before it stopped
            return procs

        self._tables(monkeypatch, _procs)

        sweep_leftovers(self._cells())

        assert killed == [
            (700, signal.SIGSTOP),
            (701, signal.SIGSTOP),
            (700, signal.SIGKILL),
            (701, signal.SIGKILL),
        ]

    def test_a_leftover_that_outforks_the_sweep_is_named_and_what_stopped_is_killed(
        self, killed, monkeypatch, caplog
    ):
        import signal

        from strata.notebook import harness_user

        newest = iter(range(1000, 2000))
        running: list[int] = []

        def _procs() -> dict[int, tuple[int, int, int]]:
            # A new process every scan, so no scan comes up clean.
            running.append(next(newest))
            return {pid: (1, self.CELLS, pid) for pid in range(1000, running[-1] + 1)}

        self._tables(monkeypatch, _procs)

        sweep_leftovers(self._cells())

        stops = [pid for pid, sig in killed if sig == signal.SIGSTOP]
        kills = [pid for pid, sig in killed if sig == signal.SIGKILL]
        assert stops and sorted(kills) == sorted(stops)
        unstopped = len(running) - len(stops)
        passes = harness_user._SWEEP_PASSES
        assert f"{unstopped} process(es) left running as cells kept forking past {passes}" in (
            caplog.text
        )

    def test_a_pid_reused_by_a_running_cell_is_not_signalled(self, killed, monkeypatch):
        """A leftover can exit and its pid go to a new harness between the scan and the pidfd."""
        server = os.getpid()
        scans = iter([{700: (1, self.CELLS, 700)}])
        self._tables(monkeypatch, lambda: next(scans, {700: (server, self.CELLS, 700)}))

        sweep_leftovers(self._cells())

        assert killed == []

    def test_without_pidfds_a_leftover_is_signalled_by_pid(self, killed, monkeypatch):
        """Linux before 5.3."""
        import errno
        import signal

        from strata.notebook import harness_user

        def _enosys(pid: int) -> int:
            raise OSError(errno.ENOSYS, "pidfd_open")

        monkeypatch.setattr(harness_user.os, "pidfd_open", _enosys, raising=False)

        sweep_leftovers(self._cells())

        assert killed == [(700, signal.SIGSTOP), (700, signal.SIGKILL)]

    @pytest.mark.parametrize("uid", ["root", "server"])
    def test_root_or_the_servers_own_user_is_never_swept(self, killed, uid, monkeypatch):
        """Every process of theirs outside the server would be a leftover: a login, init."""
        from strata.notebook import harness_user

        uid = 0 if uid == "root" else os.geteuid()
        server = os.getpid()
        table = {1: (0, 0, 1), server: (1, 0, server), 700: (1, uid, 700)}
        monkeypatch.setattr(harness_user, "_process_table", lambda: table)

        sweep_leftovers(self._cells(uid))

        assert killed == []

    def test_off_linux_nothing_is_swept(self, killed, monkeypatch):
        from strata.notebook import harness_user

        monkeypatch.setattr(harness_user.sys, "platform", "darwin")

        sweep_leftovers(self._cells())

        assert killed == []

    def test_as_pid_1_nothing_is_swept(self, killed, monkeypatch):
        """Every orphan is the server's child there, so none can be told from a cell."""
        from strata.notebook import harness_user

        monkeypatch.setattr(harness_user.os, "getpid", lambda: 1)

        sweep_leftovers(self._cells())

        assert killed == []

    def test_a_directory_is_handed_over_only_after_a_sweep(self, tmp_path, monkeypatch):
        from strata.notebook import harness_user

        order: list[str] = []
        monkeypatch.setattr(harness_user, "sweep_leftovers", lambda user: order.append("sweep"))
        monkeypatch.setattr(harness_user.os, "chown", lambda *a, **k: order.append("chown"))

        hand_over(tmp_path, self._cells(os.geteuid() + 1))

        assert order[0] == "sweep" and "chown" in order

    def test_work_is_handed_over_only_after_a_sweep(self, tmp_path, monkeypatch):
        from strata.notebook import harness_user

        order: list[str] = []
        monkeypatch.setattr(harness_user, "sweep_leftovers", lambda user: order.append("sweep"))
        monkeypatch.setattr(harness_user.os, "chown", lambda *a, **k: order.append("chown"))

        prepare_work_dir(tmp_path, self._cells(os.geteuid() + 1))

        assert order == ["sweep", "chown"]


@pytest.mark.skipif(sys.platform != "linux", reason="reads /proc")
def test_the_process_table_is_read_from_proc_without_the_dead():
    import subprocess
    import time

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        deadline = time.monotonic() + 30
        status = Path(f"/proc/{child.pid}/status")
        while "\tZ" not in status.read_text().split("State:")[1][:3]:
            assert time.monotonic() < deadline
            time.sleep(0.05)

        table = _process_table()  # imported before conftest stubs the module's

        assert table[os.getpid()] == (os.getppid(), os.getuid(), os.getsid(0))
        assert child.pid not in table
    finally:
        child.wait()
