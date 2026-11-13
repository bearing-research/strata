from tests.docker_helpers import docker_daemon_reachable


def test_an_absent_daemon_is_unreachable_and_its_check_leaves_no_socket_open(tmp_path, monkeypatch):
    # A socket left open fails this test at its teardown collect.
    monkeypatch.setenv("DOCKER_HOST", f"unix://{tmp_path / 'docker.sock'}")

    assert not docker_daemon_reachable()
