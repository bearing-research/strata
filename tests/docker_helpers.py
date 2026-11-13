"""Whether a Docker daemon answers, for the modules whose tests need one."""

import socket
from contextlib import closing

import docker
from docker.constants import DEFAULT_UNIX_SOCKET, IS_WINDOWS_PLATFORM
from docker.utils import kwargs_from_env


def docker_daemon_reachable() -> bool:
    """Whether the daemon docker-py would use answers a ping.

    A unix socket is tried here first: docker-py leaves the socket of a failed connect to the
    collector, and the suite fails on whichever later test that collection lands in. Windows
    has no unix sockets; docker-py uses a named pipe there.
    """
    base_url = kwargs_from_env().get("base_url") or DEFAULT_UNIX_SOCKET
    if not IS_WINDOWS_PLATFORM and "unix://" in base_url:
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.connect(base_url.split("unix://", 1)[1])
        except OSError:
            return False
    try:
        with closing(docker.from_env()) as client:
            client.ping()
    except Exception:
        return False
    return True
