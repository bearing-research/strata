"""Whether a Docker daemon answers, for the modules whose tests need one."""

import os
import socket
from urllib.parse import urlsplit

_DEFAULT_HOST = (
    "npipe:////./pipe/docker_engine" if os.name == "nt" else "unix:///var/run/docker.sock"
)


def docker_daemon_reachable() -> bool:
    """Whether the daemon ``DOCKER_HOST`` names, or the platform's default, accepts a connection.

    Not docker-py: its unix connect leaves the socket of a failed attempt to the collector, and
    the suite fails on whichever later test that collection lands in.
    """
    url = urlsplit(os.environ.get("DOCKER_HOST") or _DEFAULT_HOST)
    try:
        if url.scheme in ("unix", "http+unix") and hasattr(socket, "AF_UNIX"):
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                probe.connect(url.path)
        elif url.scheme == "tcp" and url.hostname:
            with socket.create_connection((url.hostname, url.port or 2375), timeout=5):
                pass
        elif url.scheme == "npipe":
            return os.path.exists(url.path.replace("/", "\\"))
        else:
            return False
    except OSError:
        return False
    return True
