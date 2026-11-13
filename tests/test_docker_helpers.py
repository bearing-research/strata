import socket

from tests.docker_helpers import docker_daemon_reachable


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
