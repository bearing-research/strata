"""Shared pytest fixtures and helpers for Strata tests.

Ports and server polling, Arrow IPC conversion, test-server context managers, and the
base fixtures (temp_warehouse, strata_config, server_with_client).
"""

import io
import os
import shlex
import socket
import sys
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

# Python 3.10 has no datetime.UTC.
try:
    from datetime import UTC
except ImportError:
    UTC = UTC

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
import uvicorn
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import (
    DoubleType,
    LongType,
    NestedField,
    StringType,
)
from strata_client.client import StrataClient

from strata.config import StrataConfig

# Global-state isolation


def prepared_venv(notebook_dir: Path) -> None:
    """Give a notebook a ``.venv`` whose interpreter is this one, for ``--no-sync``.

    ``--no-sync`` refuses a venv without ``.venv/bin/python``. No Windows venv has one, and
    the notebook subsystem is skipped there, so a test needing this is too.
    """
    if os.name == "nt":
        pytest.skip("Strata's venv interpreter path is bin/python; not a Windows venv layout")
    python = notebook_dir / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    if not python.exists():
        # A wrapper, not a symlink: Python started through a symlink looks
        # for pyvenv.cfg beside the symlink, finds none, and comes up as the
        # base interpreter without this venv's packages.
        python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
        python.chmod(0o755)


@pytest.fixture(autouse=True)
def _home_is_a_temp_dir(tmp_path_factory, monkeypatch):
    """Every Strata default under ``~/.strata`` lands in this test's temp dir.

    Each default derives from ``Path.home()``, so this one seam keeps tests that build a
    partial config out of the developer's real store.
    """
    # Its own directory, not under ``tmp_path``: some tests assert that
    # ``tmp_path`` is left empty.
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))


@pytest.fixture(autouse=True)
def _never_publish_into_the_real_store(monkeypatch):
    """Keep ``strata artifact publish`` away from the developer's own store.

    With ``None``, ``cmd_publish`` mints in the store the test opened rather than the
    server's ``~/.strata/artifacts``. The bridge test overrides this.
    """
    monkeypatch.setattr("strata.artifact_cli._server_store", lambda: None)


@pytest.fixture(autouse=True)
def _allow_testclient_host(monkeypatch):
    """The in-process clients send ``Host: testserver`` or ``test``; personal mode refuses both."""
    monkeypatch.setenv("STRATA_ALLOWED_HOSTS", "testserver,test")


@pytest.fixture(autouse=True)
def _reset_process_globals():
    """Reset process-global server state after every test.

    ``strata.server._state``, the tenant registry and the artifact-store singleton are shared
    by every test on an xdist worker, so a leaked value fails an unrelated later test.
    """
    yield
    import strata.server as server_module

    # Drain stream-cleanup tasks: tests that bypass the lifespan (a bare
    # ``TestClient(app)``) would otherwise orphan the TTL tasks, which log "Task was
    # destroyed but it is pending" once their loop closes. This only cancels, so it never raises.
    state = server_module._state
    if state is not None:
        streams = getattr(state, "streams", None)
        if streams is not None:
            streams.shutdown_cleanups()

    server_module._state = None
    from strata.artifact_store import reset_artifact_store
    from strata.metadata_cache import reset_caches
    from strata.rate_limiter import reset_rate_limiter
    from strata.tenant_registry import reset_tenant_registry

    reset_artifact_store()
    reset_tenant_registry()
    # The metadata store is a process global and a no-arg ``get_metadata_store()``
    # returns the installed one, so a test's store would leak into later callers
    # (including ``GET /metrics`` and ``POST /v1/metadata/cleanup``).
    reset_caches()
    # The rate limiter is a process global; a drained bucket left by one test
    # shows up as a spurious 429 in the next.
    reset_rate_limiter()
    # Vended lake credentials are registered per table location by whichever
    # test planned against a named catalog; a later test must not read with them.
    from strata.lake_files import reset as reset_lake_files

    reset_lake_files()
    # The build runner's heartbeat task is bound to the loop that started it; a
    # leftover runner breaks the next lifespan's teardown ("attached to a different loop").
    _reset_transform_singletons()


# Common utility functions


# Chainguard still serves MinIO anonymously (quay.io and Docker Hub no longer do).
# Its free tier only has ``latest``, so pin by digest. To move it:
# ``docker pull cgr.dev/chainguard/minio`` and take the digest from
# ``docker inspect --format '{{index .RepoDigests 0}}'``.
_MINIO_DIGEST = "sha256:bd014394a80898e68c149f2311fdf8d5a2c2f3bb2c33b9327ae6d02b4b065ae1"
MINIO_IMAGE = f"cgr.dev/chainguard/minio@{_MINIO_DIGEST}"


def start_container_or_skip(container, *, label: str, ready=None):
    """Start a testcontainers container, skipping the module if startup fails.

    A failed image pull (a Docker Hub flake, or fork CI without registry secrets) means the
    backend is unavailable: a skip, not a failure. Only startup and the optional ``ready``
    probe are converted; later errors propagate. The caller owns ``stop()``.
    """
    try:
        container.start()
        if ready is not None:
            ready(container)
    except Exception as exc:
        try:
            container.stop()
        except Exception:
            pass
        pytest.skip(f"{label} container unavailable (Docker image pull/start failed): {exc}")
    return container


def _reset_transform_singletons() -> None:
    """Reset build store and runner globals between test servers.

    The build store singleton caches its first db_path, so a later server's builds would land
    in the first server's database and sit in pending.
    """
    from strata.transforms.build_store import reset_build_store
    from strata.transforms.runner import reset_build_runner

    reset_build_store()
    reset_build_runner()


def find_free_port() -> int:
    """Find an available port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(
    port: int, timeout: float = 60.0, thread: threading.Thread | None = None
) -> bool:
    """Poll /health until the server answers; True if ready, False if it died or timed out.

    ``timeout`` (seconds) is a generous hang guard, since a Windows runner under xdist can
    stall. Pass the serving ``thread`` to stop at once if it exits, e.g. on a failed bind.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=1.0)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        if thread is not None and not thread.is_alive():
            return False
        time.sleep(0.1)
    return False


# IPC conversion helpers


def table_to_ipc_bytes(table: pa.Table) -> bytes:
    """Convert an Arrow table to IPC stream bytes."""
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def ipc_bytes_to_table(data: bytes) -> pa.Table:
    """Convert IPC stream bytes to an Arrow table."""
    reader = ipc.open_stream(io.BytesIO(data))
    return reader.read_all()


# Server context managers


@dataclass
class ServerContext:
    """Context for a running test server."""

    config: StrataConfig
    port: int
    base_url: str
    server_instance: uvicorn.Server | None = None
    thread: threading.Thread | None = None


@contextmanager
def run_server(config: StrataConfig, reset_caches: bool = False) -> Iterator[str]:
    """Run a Strata server in a daemon thread via uvicorn.run and yield its base URL.

    ``reset_caches`` resets the global metadata caches first. Raises RuntimeError if the
    server does not answer /health within 20 seconds.
    """
    import strata.server as server_module
    from strata.artifact_store import reset_artifact_store
    from strata.server import ServerState, app

    if reset_caches:
        from strata.metadata_cache import reset_caches as do_reset_caches

        do_reset_caches()

    reset_artifact_store()
    _reset_transform_singletons()

    server_module._state = ServerState(config)

    # A uvicorn.Server handle (not uvicorn.run) so teardown can stop it: orphaned
    # servers each keep a build-runner poll loop, and that load makes busy CI runners
    # refuse connections mid-suite.
    server_config = uvicorn.Config(
        app=app,
        host=config.host,
        port=config.port,
        log_level="error",
        # Match production (server.main): default ws="auto" imports uvicorn's
        # legacy websockets backend (DeprecationWarning; broken on CPython 3.14).
        ws="websockets-sansio",
    )
    server_instance = uvicorn.Server(server_config)
    server_thread = threading.Thread(target=server_instance.run, daemon=True)
    server_thread.start()

    # Generous window: the embedded build-runner poll loop can delay startup well
    # past a few seconds under CI load.
    base_url = f"http://{config.host}:{config.port}"
    for _ in range(200):  # 20 second timeout
        try:
            with httpx.Client() as client:
                resp = client.get(f"{base_url}/health", timeout=1.0)
                if resp.status_code == 200:
                    break
        except Exception:
            pass
        time.sleep(0.1)
    else:
        raise RuntimeError("Server failed to start")

    try:
        yield base_url
    finally:
        # Join generously so the lifespan shutdown (which stops the build runner)
        # completes; abandoned threads leave runners that starve later servers' startup.
        server_instance.should_exit = True
        server_thread.join(timeout=15.0)
        server_module._state = None
        reset_artifact_store()
        _reset_transform_singletons()


@contextmanager
def run_server_with_context(
    cache_dir,
    artifact_dir=None,
    deployment_mode: Literal["personal", "service"] = "personal",
    **config_overrides,
) -> Iterator[ServerContext]:
    """Run a server and yield a ServerContext for graceful shutdown.

    Unlike ``run_server`` it supports ``artifact_dir`` and resets the artifact store on
    cleanup. Raises RuntimeError if the server fails to start.
    """
    from strata import server
    from strata.artifact_store import reset_artifact_store
    from strata.server import ServerState, app

    port = find_free_port()

    # Rate limiting off unless a test asks for it: every fixture server sees one
    # client making bursts of legitimate requests, and the default 20-token burst
    # can 429 a test's own setup requests.
    config_overrides.setdefault("rate_limit_enabled", False)
    config = StrataConfig(
        host="127.0.0.1",
        port=port,
        cache_dir=cache_dir,
        deployment_mode=deployment_mode,
        artifact_dir=artifact_dir,
        **config_overrides,
    )
    reset_artifact_store()
    _reset_transform_singletons()
    server._state = ServerState(config)

    server_config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        # Match production (server.main): default ws="auto" imports uvicorn's
        # legacy websockets backend (DeprecationWarning; broken on CPython 3.14).
        ws="websockets-sansio",
    )
    server_instance = uvicorn.Server(server_config)
    thread = threading.Thread(target=server_instance.run, daemon=True)
    thread.start()

    if not wait_for_server(port, thread=thread):
        raise RuntimeError(
            f"Server failed to start on port {port} "
            f"(serving thread {'still running' if thread.is_alive() else 'exited'})"
        )

    try:
        yield ServerContext(
            config=config,
            port=port,
            base_url=f"http://127.0.0.1:{port}",
            server_instance=server_instance,
            thread=thread,
        )
    finally:
        # Join generously so the lifespan shutdown (stopping the build runner)
        # completes instead of orphaning the thread (see run_server).
        server_instance.should_exit = True
        thread.join(timeout=15.0)
        server._state = None
        reset_artifact_store()
        _reset_transform_singletons()


# Fixtures


@pytest.fixture
def temp_warehouse(tmp_path):
    """A temporary warehouse with a sample Iceberg table; skipped on Windows (pyiceberg paths)."""
    if sys.platform == "win32":
        pytest.skip("pyiceberg + pyarrow LocalFileSystem path handling broken on Windows")
    warehouse_path = tmp_path / "warehouse"
    warehouse_path.mkdir()

    warehouse_uri = warehouse_path.as_uri()
    catalog_db = (warehouse_path / "catalog.db").as_posix()
    catalog = SqlCatalog(
        "strata",
        **{
            "uri": f"sqlite:///{catalog_db}",
            "warehouse": warehouse_uri,
        },
    )

    catalog.create_namespace("test_db")

    # Optional fields to match PyArrow defaults.
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
        NestedField(3, "name", StringType(), required=False),
        NestedField(4, "timestamp", LongType(), required=False),  # Epoch micros
    )

    table = catalog.create_table("test_db.events", schema)

    num_rows = 500
    base_ts = int(datetime(2024, 1, 1, tzinfo=UTC).timestamp() * 1_000_000)
    data = pa.table(
        {
            "id": pa.array(range(num_rows), type=pa.int64()),
            "value": pa.array([float(i * 1.5) for i in range(num_rows)], type=pa.float64()),
            "name": pa.array([f"item_{i}" for i in range(num_rows)], type=pa.string()),
            "timestamp": pa.array(
                [base_ts + i * 3600_000_000 for i in range(num_rows)],  # micros
                type=pa.int64(),
            ),
        }
    )

    table.append(data)

    # Close the fixture's write connection to catalog.db: overlapping with the
    # server's own SqlCatalog connection intermittently trips ``SQLITE_IOERR`` on
    # tmpfs + Python 3.14. The returned ``catalog`` reconnects lazily.
    catalog.engine.dispose()

    return {
        "warehouse_path": warehouse_path,
        "table_uri": f"{warehouse_uri}#test_db.events",
        "catalog": catalog,
        "table": table,
    }


@pytest.fixture
def strata_config(tmp_path):
    """Create a test configuration."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return StrataConfig(cache_dir=cache_dir)


@pytest.fixture
def server_with_client(temp_warehouse, tmp_path):
    """Start a server and yield a dict with ``client``, ``config`` and ``warehouse``."""
    import strata.server as server_module
    from strata.server import ServerState, app

    port = find_free_port()

    config = StrataConfig(
        host="127.0.0.1",
        port=port,
        cache_dir=tmp_path / "cache",
        deployment_mode="personal",
    )

    server_module._state = ServerState(config)

    # uvicorn.Server handle so teardown can stop it (see run_server).
    server_config = uvicorn.Config(
        app=app,
        host=config.host,
        port=config.port,
        log_level="error",
        # Match production (server.main): default ws="auto" imports uvicorn's
        # legacy websockets backend (DeprecationWarning; broken on CPython 3.14).
        ws="websockets-sansio",
    )
    server_instance = uvicorn.Server(server_config)
    server_thread = threading.Thread(target=server_instance.run, daemon=True)
    server_thread.start()

    if not wait_for_server(port, thread=server_thread):
        raise RuntimeError(
            f"Server failed to start on port {port} "
            f"(serving thread {'still running' if server_thread.is_alive() else 'exited'})"
        )

    client = StrataClient(base_url=f"http://127.0.0.1:{port}")

    yield {
        "client": client,
        "config": config,
        "warehouse": temp_warehouse,
    }

    client.close()
    server_instance.should_exit = True
    server_thread.join(timeout=2.0)


# The placeholder artifact ids the build-store tests build against, plus the
# canonical id ``update_build_output`` repoints a deduped build at.
BUILD_TARGET_IDS: tuple[str, ...] = (
    "a",
    "art-123",
    "canonical-artifact",
    *(f"art-{i}" for i in range(4)),
    *(f"artifact-{i}" for i in range(1, 4)),
)


def seed_build_targets(
    artifact_dir: Path,
    artifact_ids: Iterable[str] = BUILD_TARGET_IDS,
    versions: Iterable[int] = range(1, 8),
) -> None:
    """Create the artifact versions that build rows are allowed to reference.

    ``artifact_builds`` has a foreign key on ``(artifact_id, version)`` and production always
    creates the version first. Opening the artifact store also creates ``artifact_versions``
    in the shared database, without which the key cannot resolve.
    """
    from strata.artifact_store import ArtifactStore

    store = ArtifactStore(artifact_dir)
    try:
        for artifact_id in artifact_ids:
            for version in versions:
                created = store.create_artifact(artifact_id, f"seed-{artifact_id}-{version}")
                store.blob_store.write_blob(artifact_id, created, b"seed")
                store.finalize_artifact(artifact_id, created, "{}", 1, 4)
    finally:
        store.close()


class RebindingDNS:
    """Name resolution that answers a name differently on each lookup.

    ``answers[name]`` is a list of address lists; each lookup takes the next and the last
    repeats, and other names go to the real resolver. Patched in as ``socket.getaddrinfo``,
    which both the guard and the socket layer use.
    """

    def __init__(self) -> None:
        self.answers: dict[str, list[list[str]]] = {}
        self.lookups: list[str] = []

    def getaddrinfo(self, host, port, *args, real, **kwargs):
        # anyio passes the name IDNA-encoded, as bytes.
        name = host.decode("ascii") if isinstance(host, bytes) else host
        if name not in self.answers:
            return real(host, port, *args, **kwargs)
        self.lookups.append(name)
        queue = self.answers[name]
        addresses = queue.pop(0) if len(queue) > 1 else queue[0]
        port_number = int(port or 0)
        return [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (address, port_number, 0, 0))
            if ":" in address
            else (socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port_number))
            for address in addresses
        ]


@pytest.fixture
def rebinding_dns(monkeypatch) -> RebindingDNS:
    dns = RebindingDNS()
    real = socket.getaddrinfo
    monkeypatch.setattr(
        socket, "getaddrinfo", lambda *args, **kwargs: dns.getaddrinfo(*args, real=real, **kwargs)
    )
    return dns
