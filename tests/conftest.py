"""Shared pytest fixtures and helpers for Strata tests.

This module provides:
- Common utility functions (find_free_port, wait_for_server, etc.)
- IPC conversion helpers (table_to_ipc_bytes, ipc_bytes_to_table)
- Server context managers for running test servers
- Base fixtures (temp_warehouse, strata_config, server_with_client)
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

    ``--no-sync`` takes the interpreter at ``.venv/bin/python`` and refuses a
    venv without one. An empty ``.venv`` directory used to pass, and the cells
    then ran with whatever ``python`` was on PATH; that fallback is what the
    check removed. No Windows venv has ``bin/python``, and the notebook
    subsystem is skipped there (ci.yml), so a test that needs this is too.
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

    A personal-mode ``StrataConfig`` with no ``artifact_dir`` uses
    ``~/.strata/artifacts``, and the same goes for ``cache_dir``,
    ``metadata_db``, the notebook storage dir, worker envs and the mount
    cache. Dozens of tests build a config with only the fields they care
    about, and the developer's real store had 45,000 rows of their leftovers.
    Each default is computed from ``Path.home()``, so this is the one seam
    that catches all of them, now and for the next config a test writes.
    """
    # Its own directory, not under ``tmp_path``: some tests assert that
    # ``tmp_path`` is left empty.
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))


@pytest.fixture(autouse=True)
def _never_publish_into_the_real_store(monkeypatch):
    """Keep ``strata artifact publish`` away from the developer's own store.

    Publishing resolves the *server's* artifact directory through
    ``StrataConfig.load()`` and copies into it, so a link minted from a
    notebook resolves. Loaded in a test that sets no override, that is
    ``~/.strata/artifacts`` — and a CLI test duly wrote four fixture artifacts
    and two publications into the real one before this existed.

    Returning ``None`` makes ``cmd_publish`` mint in whatever store the test
    opened, which is what every test but the bridge one wants. The bridge test
    monkeypatches this again, and being function-scoped it wins.
    """
    monkeypatch.setattr("strata.artifact_cli._server_store", lambda: None)


@pytest.fixture(autouse=True)
def _reset_process_globals():
    """Nuke process-global server state after every test.

    ``strata.server._state`` (plus the tenant registry and artifact-store
    singletons) are process globals shared by every test on a pytest-xdist
    worker. A test that sets ``_state`` — e.g. a notebook route test configuring
    ``notebook_storage_dir`` — could leak it into an unrelated later test on the
    same worker, which surfaced under xdist as
    ``TestCellIterationsEndpoint`` getting a 400 "must be inside configured
    notebook storage" (it passes serially / at a different worker count because
    test→worker packing differs). Most fixtures already reset on teardown; this
    autouse teardown is the belt-and-suspenders guarantee that no test starts
    with a dirty global, so the suite is safe to run in parallel.
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

    The S3/GCS/Azure mount e2e tests spin up emulator containers (MinIO,
    fake-gcs-server, Azurite). When the Docker image pull times out — a
    common Docker Hub flake, and the norm on dependabot/fork CI that can't
    reach the registry with the right secrets — testcontainers raises during
    ``start()``, turning every test in the module into an ERROR. A pull/start
    failure means the backend is simply unavailable, which is a skip, not a
    failure. Only the startup phase is converted to a skip; anything raised
    after the container is up (i.e. a real test failure) propagates normally.

    ``ready`` is an optional callable run after start (e.g. a ``wait_for_logs``
    readiness probe); a failure there is also treated as "backend unavailable".
    Returns the started container — the caller owns ``stop()``.
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
    """Reset build store + runner globals between test servers.

    The personal-mode lifespan now starts an embedded build runner; the
    build store singleton caches its first db_path forever, so without a
    reset every later test server would create build records in the FIRST
    server's database while its own runner polls a different one — builds
    sit in pending until the wait times out.
    """
    from strata.transforms.build_store import reset_build_store
    from strata.transforms.runner import reset_build_runner

    reset_build_store()
    reset_build_runner()


def find_free_port() -> int:
    """Find an available port on localhost.

    Uses SO_REUSEADDR to avoid "address already in use" errors.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_server(
    port: int, timeout: float = 60.0, thread: threading.Thread | None = None
) -> bool:
    """Wait for server to be ready by polling /health endpoint.

    The timeout is a hang guard, not a statement about how fast startup ought
    to be, so it is generous: a Windows runner under xdist can stall long
    enough to blow a tighter budget while the server is perfectly fine. Pass
    `thread` to keep that generosity from costing anything when the server is
    genuinely broken — a serving thread that has exited (a failed bind being
    the usual reason) will never answer, so we stop immediately instead of
    waiting out the clock and then reporting a timeout that explains nothing.

    Args:
        port: Port the server is running on
        timeout: Maximum time to wait in seconds
        thread: Optional uvicorn serving thread, to fail fast when it dies

    Returns:
        True if server is ready, False if it died or timed out
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
    """Convert Arrow table to IPC stream bytes.

    Args:
        table: PyArrow Table to convert

    Returns:
        Bytes representing the Arrow IPC stream
    """
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def ipc_bytes_to_table(data: bytes) -> pa.Table:
    """Convert IPC stream bytes to Arrow table.

    Args:
        data: Arrow IPC stream bytes

    Returns:
        PyArrow Table
    """
    reader = ipc.open_stream(io.BytesIO(data))
    return reader.read_all()


# Server context managers


@dataclass
class ServerContext:
    """Context for a running test server.

    Attributes:
        config: StrataConfig used by the server
        port: Port the server is running on
        base_url: Base URL for HTTP requests
        server_instance: The uvicorn Server instance (if using uvicorn.Server)
        thread: The thread running the server
    """

    config: StrataConfig
    port: int
    base_url: str
    server_instance: uvicorn.Server | None = None
    thread: threading.Thread | None = None


@contextmanager
def run_server(config: StrataConfig, reset_caches: bool = False) -> Iterator[str]:
    """Run a Strata server in a background thread using uvicorn.run.

    This is the basic server context manager that yields the base URL.
    Server runs as a daemon thread and is killed on exit.

    Args:
        config: StrataConfig with host/port settings
        reset_caches: If True, reset global metadata caches before starting

    Yields:
        Base URL string (e.g., "http://127.0.0.1:8765")

    Raises:
        RuntimeError: If server fails to start within 5 seconds
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
    """Run a server with full context including graceful shutdown.

    This context manager provides more control than run_server():
    - Returns ServerContext with server instance for graceful shutdown
    - Supports artifact_dir configuration
    - Resets artifact store on cleanup

    Args:
        cache_dir: Path for cache directory
        artifact_dir: Optional path for artifact storage
        deployment_mode: "personal" or "service"

    Yields:
        ServerContext with server details

    Raises:
        RuntimeError: If server fails to start
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
    """Create a temporary warehouse with a sample Iceberg table.

    Skipped on Windows: pyiceberg's PyArrowFileIO strips ``file://``
    to a path like ``/C:/...`` which Windows pyarrow LocalFileSystem
    can't resolve. The stack is pyiceberg + pyarrow upstream; working
    around it here would mean bypassing the normal catalog code path.
    Iceberg scanning on Windows is a tier-2 target — skip the tests
    that need a real warehouse.
    """
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
    """Start a server and provide a client.

    Yields a dict with:
        - client: StrataClient connected to the running server
        - config: StrataConfig used by the server
        - warehouse: temp_warehouse dict with table_uri, catalog, etc.
    """
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

    ``artifact_builds`` declares a foreign key on ``(artifact_id, version)``,
    and both production callers create the artifact version first — the server
    at ``server.py`` and the notebook executor, which calls ``create_artifact``
    immediately before ``create_build``. A build row for an artifact that never
    existed is an impossible state, so a build-store test that fabricates one
    needs this to supply the other half.

    Constructing the artifact store also creates ``artifact_versions`` in the
    shared database: a build store opened on a file of its own has no such
    table, and the foreign key cannot even be resolved against it.
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

    ``answers[name]`` is a list of answers, each a list of addresses; every
    lookup of *name* takes the next one and the last repeats. Any other name
    goes to the real resolver, so an IP literal still resolves to itself.
    Patched in as ``socket.getaddrinfo``, the one function both the guard and
    the socket layer resolve with, so it stands for a DNS server an attacker
    controls.
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
