"""Tests for the SSH-worker service layer (establish + register / teardown).

A fake supervisor stands in for the tunnel + provisioning (covered elsewhere), so
these focus on the service's job: registering the tunnel as a `[[workers]]` entry
via the shared ops and reversing it, against a real local notebook session.
"""

from __future__ import annotations

import pytest

from strata.notebook.parser import parse_notebook
from strata.notebook.remote_worker_supervisor import TunnelRecord
from strata.notebook.session import NotebookSession
from strata.notebook.ssh_worker_service import (
    default_worker_name,
    establish_ssh_worker,
    teardown_ssh_worker,
)
from tests.notebook.test_cli import _build_notebook


class FakeSupervisor:
    """Records establish/teardown and hands back a canned tunnel record."""

    def __init__(self) -> None:
        self.established: list[tuple[str, str]] = []
        self.torn_down: list[tuple[str, bool]] = []
        self.present: set[str] = set()

    def establish(self, name, ssh_target, **kwargs):
        self.established.append((name, ssh_target))
        self.present.add(name)
        return TunnelRecord(
            name=name,
            ssh_target=ssh_target,
            local_port=6000,
            remote_port=9000,
            remote_pid=42,
            healthy=True,
            executor_url="http://127.0.0.1:6000/v1/execute",
        )

    def teardown(self, name, *, stop_remote=False):
        self.torn_down.append((name, stop_remote))
        existed = name in self.present
        self.present.discard(name)
        return existed


@pytest.fixture
def session(tmp_path):
    nb = _build_notebook(tmp_path, cells=[("a", "x = 1", None)])
    return NotebookSession(parse_notebook(nb), nb)


def test_default_worker_name_slugifies_host():
    assert default_worker_name("user@gpu-box.internal") == "gpu-box.internal"
    assert default_worker_name("gpu-box") == "gpu-box"
    assert default_worker_name("user@10.0.0.5:22") == "10.0.0.5"
    assert default_worker_name("@") == "remote"


def test_establish_registers_worker_and_sets_default(session):
    sup = FakeSupervisor()
    record = establish_ssh_worker(session, sup, ssh_target="user@gpu-box", set_default=True)
    assert record.executor_url == "http://127.0.0.1:6000/v1/execute"
    assert sup.established == [("gpu-box", "user@gpu-box")]
    # Registered as a direct executor worker, keyed to the target, set default.
    worker = next(w for w in session.notebook_state.workers if w.name == "gpu-box")
    assert worker.backend.value == "executor"
    assert worker.config.url == "http://127.0.0.1:6000/v1/execute"
    assert worker.config.transport == "direct"
    assert worker.runtime_id == "ssh:user@gpu-box"
    assert session.notebook_state.worker == "gpu-box"


def test_establish_honors_explicit_name(session):
    sup = FakeSupervisor()
    establish_ssh_worker(session, sup, ssh_target="user@box", name="gpu")
    assert [w.name for w in session.notebook_state.workers] == ["gpu"]
    assert session.notebook_state.worker is None  # set_default defaulted False


def test_establish_refused_in_service_mode(session, monkeypatch):
    sup = FakeSupervisor()
    monkeypatch.setattr(
        "strata.notebook.workers.notebook_worker_definitions_editable", lambda state: False
    )
    with pytest.raises(PermissionError):
        establish_ssh_worker(session, sup, ssh_target="user@box")
    assert sup.established == []  # never opened an SSH connection


def test_teardown_removes_registration_and_tunnel(session):
    sup = FakeSupervisor()
    establish_ssh_worker(session, sup, ssh_target="user@box", name="gpu", set_default=True)
    assert teardown_ssh_worker(session, sup, "gpu") is True
    assert sup.torn_down == [("gpu", False)]
    assert [w.name for w in session.notebook_state.workers] == []
    assert session.notebook_state.worker is None  # default cleared with the worker


def test_teardown_unregistered_name_still_tears_tunnel(session):
    sup = FakeSupervisor()
    sup.present.add("ghost")  # a tunnel with no notebook registration
    assert teardown_ssh_worker(session, sup, "ghost") is True
    assert sup.torn_down == [("ghost", False)]
