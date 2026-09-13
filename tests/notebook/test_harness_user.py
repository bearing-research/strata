"""Who a cell runs as, and whether a service-mode host starts it at all. Item 49.

The environment allowlist filters what a cell is given; a cell running as the
server's own user could still read ``/proc/<server pid>/environ``. So a
service-mode server refuses to start cell code on its own host unless cells run
as a separate OS user — or on another machine, which is a worker and never
reaches these spawns.

Dropping to a *different* user needs root, which the suite does not have; the
root-only end-to-end check lives in ``tests/test_harness_isolation.py``. Here the
harness user is the current user, which exercises every spawn path without
privileges, and the refusal is checked on every site that starts cell code.
"""

from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

from strata.notebook.harness_user import (
    REFUSAL,
    HarnessUser,
    LocalExecutionRefused,
    identity_env,
    resolve_harness_user,
    spawn_kwargs,
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
    return NotebookSession(parse_notebook(notebook_dir), notebook_dir)


class TestResolution:
    def test_personal_mode_runs_as_the_server(self, monkeypatch):
        """One person on their own machine has nothing to isolate a cell from."""
        _server(monkeypatch, mode="personal")

        assert resolve_harness_user() is None

    def test_outside_a_server_nothing_is_refused(self, monkeypatch):
        """A CLI run has no server credentials for a cell to read — and the
        loaded config's default mode is service, so falling back to it would
        refuse every `strata run`."""
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
