import socket

import pytest

from tests.docker_helpers import (
    REQUIRE_CONTAINERS_ENV,
    docker_daemon_reachable,
    start_container_or_skip,
)


def test_an_absent_daemon_is_unreachable_and_its_check_leaves_no_socket_open(tmp_path, monkeypatch):
    # A socket left open fails this test at its teardown collect.
    monkeypatch.setenv("DOCKER_HOST", f"unix://{tmp_path / 'docker.sock'}")

    assert not docker_daemon_reachable()


def test_a_daemon_that_accepts_connections_is_reachable(monkeypatch):
    with socket.create_server(("127.0.0.1", 0)) as daemon:
        monkeypatch.setenv("DOCKER_HOST", f"tcp://127.0.0.1:{daemon.getsockname()[1]}")

        assert docker_daemon_reachable()


def test_a_closed_port_is_unreachable(monkeypatch):
    with socket.create_server(("127.0.0.1", 0)) as gone:
        port = gone.getsockname()[1]
    monkeypatch.setenv("DOCKER_HOST", f"tcp://127.0.0.1:{port}")

    assert not docker_daemon_reachable()


class _Unpullable:
    def __init__(self):
        self.stopped = False

    def start(self):
        raise RuntimeError("404 Client Error: No such image")

    def stop(self):
        self.stopped = True


def test_a_container_that_cannot_start_skips_where_containers_are_optional(monkeypatch):
    monkeypatch.delenv(REQUIRE_CONTAINERS_ENV, raising=False)
    container = _Unpullable()

    # Any outcome: the other one, raised here, would end this test as itself.
    with pytest.raises(BaseException, match="MinIO container unavailable.*No such image") as out:
        start_container_or_skip(container, label="MinIO")
    assert out.type is pytest.skip.Exception
    assert container.stopped


def test_a_container_that_cannot_start_fails_where_containers_are_required(monkeypatch):
    # The integration job: a skip there would pass a run that never reached the backend.
    monkeypatch.setenv(REQUIRE_CONTAINERS_ENV, "1")
    container = _Unpullable()

    # Any outcome: the other one, raised here, would end this test as itself.
    with pytest.raises(BaseException, match="MinIO container unavailable.*No such image") as out:
        start_container_or_skip(container, label="MinIO")
    assert out.type is pytest.fail.Exception
    assert container.stopped


def test_a_stop_that_fails_too_is_in_the_reason(monkeypatch):
    monkeypatch.delenv(REQUIRE_CONTAINERS_ENV, raising=False)
    container = _Unpullable()

    def stop():
        raise RuntimeError("no such container")

    container.stop = stop

    with pytest.raises(pytest.skip.Exception, match="No such image; stopping it failed too: no"):
        start_container_or_skip(container, label="MinIO")
