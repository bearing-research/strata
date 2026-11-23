"""Whether a Docker daemon answers, and starting a container, for the tests that need one."""

import os
import socket
from typing import NoReturn
from urllib.parse import urlsplit

import pytest

# "1" where the containers must start, the integration job: there a skip would pass
# a run that never reached the backend. Any other value leaves them optional.
REQUIRE_CONTAINERS_ENV = "STRATA_TESTS_REQUIRE_CONTAINERS"


def backend_unavailable(reason: str, *, module: bool = False) -> NoReturn:
    """Skip a test (or with ``module``, its module) whose backend is missing; fail it where
    ``REQUIRE_CONTAINERS_ENV`` says the backend must be there."""
    if os.environ.get(REQUIRE_CONTAINERS_ENV) == "1":
        pytest.fail(reason, pytrace=False)
    pytest.skip(reason, allow_module_level=module)


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


def start_container_or_skip(container, *, label: str):
    """Start a testcontainers container; if startup fails, ``backend_unavailable``.

    A failed image pull (a Docker Hub flake, or fork CI without registry secrets) means the
    backend is unavailable: a skip, not a failure, unless ``REQUIRE_CONTAINERS_ENV`` is set.
    Only startup, including the container's wait strategy, is converted; later errors
    propagate. The caller owns ``stop()``.
    """
    try:
        container.start()
    except Exception as exc:
        try:
            container.stop()
        except Exception:
            pass
        backend_unavailable(
            f"{label} container unavailable (Docker image pull/start failed): {exc}"
        )
    return container
