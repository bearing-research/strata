"""Amber item 89: what a cell's inputs weigh, and how much is already on the machine."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from strata.notebook.models import MountMode, MountSpec
from strata.notebook.mount_cache import MountCache
from strata.notebook.mounts import MountResolver, MountWeight, weigh_mount_sync
from tests.notebook import test_mounts
from tests.notebook.test_mounts import _FakeRemoteFS, _install_fake_fsspec

# Through the module: a test class imported by name would be collected here again.
_files = test_mounts.TestSharedMountCache._files


def _mount(uri: str = "s3://bucket/data", **kwargs: object) -> MountSpec:
    return MountSpec(name="data", uri=uri, mode=MountMode.READ_ONLY, **kwargs)  # type: ignore[arg-type]


@pytest.fixture
def remote(monkeypatch) -> _FakeRemoteFS:
    fs = _FakeRemoteFS(_files("bucket/data", {"a": b"aa", "b": b"bbb"}))
    _install_fake_fsspec(monkeypatch, fs)
    return fs


async def test_a_mounts_cached_share_counts_the_objects_the_machine_holds(tmp_path, remote):
    cache = MountCache(tmp_path / "mc")
    resolver = MountResolver(cache_dir=tmp_path / "nb", mount_cache=cache)

    cold = weigh_mount_sync(resolver, _mount(), cache)[1]
    # A view of one file of the prefix brings its object; the other is still remote.
    await resolver.prepare_mounts([_mount("s3://bucket/data/a")])
    partly = weigh_mount_sync(resolver, _mount(), cache)[1]
    await resolver.prepare_mounts([_mount()])
    warm = weigh_mount_sync(resolver, _mount(), cache)[1]

    assert [cold, partly, warm] == [
        MountWeight(bytes=5, cached_bytes=0),
        MountWeight(bytes=5, cached_bytes=2),
        MountWeight(bytes=5, cached_bytes=5),
    ]


async def test_files_with_no_version_are_cached_once_their_view_is_built(tmp_path, monkeypatch):
    """With no etag or mtime a file has no shared object: it is downloaded into the view."""
    files = _files("bucket/data", {"a": b"aa"})
    for info in files.values():
        del info["etag"], info["mtime"]
    _install_fake_fsspec(monkeypatch, _FakeRemoteFS(files))
    cache = MountCache(tmp_path / "mc")
    resolver = MountResolver(cache_dir=tmp_path / "nb", mount_cache=cache)

    before = weigh_mount_sync(resolver, _mount(), cache)[1]
    await resolver.prepare_mounts([_mount()])

    assert [before, weigh_mount_sync(resolver, _mount(), cache)[1]] == [
        MountWeight(bytes=2, cached_bytes=0),
        MountWeight(bytes=2, cached_bytes=2),
    ]


def test_weighing_gives_the_fingerprint_staleness_computes(tmp_path, remote):
    from strata.notebook.mounts import mount_fingerprint_sync

    resolver = MountResolver(cache_dir=tmp_path / "nb", mount_cache=MountCache(tmp_path / "mc"))

    assert weigh_mount_sync(resolver, _mount(), None)[0] == mount_fingerprint_sync(
        resolver, _mount()
    )


@pytest.mark.parametrize(
    ("mount", "weight"),
    [
        # Without the machine's cache, the share is unknown.
        (_mount(), MountWeight(bytes=5, cached_bytes=None)),
        # A pin is not listed, so nothing is known.
        (_mount(pin="v1"), MountWeight(bytes=None, cached_bytes=None)),
        # Not read through a view, or moving nothing: no weight at all.
        (MountSpec(name="data", uri="s3://bucket/data", mode=MountMode.READ_WRITE), None),
    ],
)
def test_what_is_not_known_is_not_guessed(tmp_path, remote, mount, weight):
    resolver = MountResolver(cache_dir=tmp_path / "nb", mount_cache=MountCache(tmp_path / "mc"))

    assert weigh_mount_sync(resolver, mount, None)[1] == weight


def test_a_local_mount_has_no_weight(tmp_path):
    (tmp_path / "data").mkdir()
    resolver = MountResolver(cache_dir=tmp_path / "nb")

    assert weigh_mount_sync(resolver, _mount(f"file://{tmp_path / 'data'}"), None)[1] is None


async def _run_inputs_frames(tmp_path, monkeypatch, *sources: str):
    """Make the cell's provenance for each source in turn, with a socket on its session.

    Returns the ``cell_inputs`` payloads sent after each, and the last provenance.
    """
    from strata.config import StrataConfig
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
    from strata.notebook.ws import _notebook_connections, forget_notebook_execution_state

    class _Socket:
        def __init__(self):
            self.sent: list[dict] = []

        async def send_text(self, text: str) -> None:
            self.sent.append(json.loads(text))

    config = StrataConfig(cache_dir=tmp_path / "cache", notebook_mount_cache_dir=tmp_path / "mc")
    monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: config)
    nb = create_notebook(tmp_path / "nb", "Weights")
    add_cell_to_notebook(nb, "c1", None)
    write_cell(nb, "c1", sources[0])
    session = NotebookSession(parse_notebook(nb), nb)
    executor = CellExecutor(session)
    socket = _Socket()
    _notebook_connections[session.id] = [socket]
    frames = []
    try:
        for source in sources:
            socket.sent.clear()
            prov = await executor._compute_cell_provenance("c1", source)
            frames.append([f["payload"] for f in socket.sent if f["type"] == "cell_inputs"])
    finally:
        _notebook_connections.pop(session.id, None)
        forget_notebook_execution_state(session.id)
    return frames, prov


async def test_a_run_tells_the_notebook_what_its_inputs_weigh(tmp_path, remote, monkeypatch):
    """The frame reaches the session's sockets, with the share in the server's cache."""
    # Another notebook already read one file of the prefix on this machine.
    await MountResolver(
        cache_dir=tmp_path / "other", mount_cache=MountCache(tmp_path / "mc")
    ).prepare_mounts([_mount("s3://bucket/data/b")])

    frames, prov = await _run_inputs_frames(
        tmp_path, monkeypatch, "# @mount data s3://bucket/data ro\nx = 1"
    )

    expected = {
        "kind": "mount",
        "name": "data",
        "uri": "s3://bucket/data",
        "bytes": 5,
        "cached_bytes": 3,
    }
    assert frames == [[{"cell_id": "c1", "machine": "local", "inputs": [expected]}]]
    assert [w.cached_bytes for w in prov.input_weights] == [3]


async def test_only_a_change_is_sent_and_reading_nothing_is_one(tmp_path, remote, monkeypatch):
    """Upstreams that only hit the cache make provenance too: they must stay quiet."""
    mounted = "# @mount data s3://bucket/data ro\nx = 1"

    frames, _ = await _run_inputs_frames(tmp_path, monkeypatch, mounted, mounted, "x = 1", "x = 1")

    assert [[len(p["inputs"]) for p in sent] for sent in frames] == [[1], [], [0], []]


@pytest.mark.skipif(sys.platform == "win32", reason="pyiceberg's local warehouse paths fail")
async def test_a_run_records_what_each_input_moved_and_where(tmp_path, remote, monkeypatch):
    """Amber item 89's done-when: sizes and cached share before, bytes moved after."""
    from strata.config import StrataConfig
    from strata.notebook import reads_index
    from strata.notebook.executor import CellExecutor
    from strata.notebook.models import InputWeight
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
    from tests.notebook import test_fetch, test_table_index

    origin = test_fetch._Origin()
    try:
        table, table_uri = test_table_index._warehouse(tmp_path)
        monkeypatch.setattr(
            "strata.server._state",
            SimpleNamespace(
                config=SimpleNamespace(
                    deployment_mode="personal", notebook_fetch_allowed_hosts=["127.0.0.1"]
                )
            ),
        )
        config = StrataConfig(
            cache_dir=tmp_path / "cache", notebook_mount_cache_dir=tmp_path / "mc"
        )
        monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: config)
        source = (
            "# @mount data s3://bucket/data ro\n"
            f"# @table events {table_uri}\n"
            f"# @fetch zones {origin.url()}\n"
            "n = len(list(data.iterdir())) + len(zones.read_bytes())\n"
        )
        nb = create_notebook(tmp_path / "nb", "Moves")
        add_cell_to_notebook(nb, "c1", None)
        write_cell(nb, "c1", source)
        session = NotebookSession(parse_notebook(nb), nb)
        session.refresh_environment_runtime()

        # One executor, as a session keeps: what the first run moved must not carry over.
        executor = CellExecutor(session)
        first = await executor.execute_cell("c1", source)
        assert first.success, first.error
        before = session.notebook_state.get_cell("c1").inputs
        moved_first = reads_index.load(nb)["c1"]
        hit = await executor.execute_cell("c1", source)
        assert hit.cache_hit is True
        moved_again = reads_index.load(nb)["c1"]
        reopened = parse_notebook(nb).cells[0]
        session.remove_cell("c1")
        after_delete = reads_index.load(nb)
    finally:
        origin.close()

    table.refresh()
    table_bytes = sum(task.file.file_size_in_bytes for task in table.scan().plan_files())
    body = len(origin.body)
    assert before == [
        InputWeight(kind="mount", name="data", uri="s3://bucket/data", bytes=5, cached_bytes=0),
        InputWeight(kind="table", name="events", uri=table_uri, bytes=table_bytes),
        InputWeight(kind="fetch", name="zones", uri=origin.url(), bytes=body, cached_bytes=0),
    ]
    # The second run found everything on the machine, and a reopen still knows it.
    assert [w.cached_bytes for w in reopened.inputs] == [5, None, body]
    assert reopened.inputs_machine == "local"

    def moved(entry):
        return [(i["kind"], i["name"], i["bytes_moved"]) for i in entry["inputs"]]

    assert moved_first["machine"] == "local"
    assert moved(moved_first) == [
        ("mount", "data", 5),
        ("table", "events", None),
        ("fetch", "zones", body),
    ]
    assert moved(moved_again) == [
        ("mount", "data", 0),
        ("table", "events", None),
        ("fetch", "zones", 0),
    ]
    assert after_delete == {}


async def test_a_batched_run_counts_what_its_mounts_downloaded(tmp_path, remote, monkeypatch):
    """Run all batches these two; the batch prepares mounts before any provenance."""
    import asyncio

    from strata.config import StrataConfig
    from strata.notebook import reads_index
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
    from strata.notebook.ws import (
        _ensure_execution_state,
        _handle_notebook_run_all,
        _notebook_connections,
        forget_notebook_execution_state,
    )

    class _Socket:
        async def send_text(self, text: str) -> None:
            return None

    config = StrataConfig(cache_dir=tmp_path / "cache", notebook_mount_cache_dir=tmp_path / "mc")
    monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: config)
    nb = create_notebook(tmp_path / "nb", "Batched")
    add_cell_to_notebook(nb, "c1", None)
    write_cell(nb, "c1", "# @mount data s3://bucket/data ro\nn = len(list(data.iterdir()))")
    add_cell_to_notebook(nb, "c2", "c1")
    write_cell(nb, "c2", "m = n + 1")
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    session._analyze_and_build_dag()
    session.environment_sync_state = "ready"
    socket = _Socket()
    _notebook_connections[session.id] = [socket]
    try:
        state = _ensure_execution_state(session.id)
        await _handle_notebook_run_all(socket, session, state, session.id, {})
        if state.execution_task is not None:
            await asyncio.gather(state.execution_task)
    finally:
        _notebook_connections.pop(session.id, None)
        forget_notebook_execution_state(session.id)

    assert [c.status.value for c in session.notebook_state.cells] == ["ready", "ready"]
    assert [i["bytes_moved"] for i in reads_index.load(nb)["c1"]["inputs"]] == [5]


async def test_a_worker_answers_by_its_own_object_keys(tmp_path, monkeypatch):
    """The keys hash the storage options, and a worker's are not the server's."""
    from datetime import UTC, datetime

    from fastapi.testclient import TestClient

    from strata.notebook.credentials import CredentialResolver
    from strata.notebook.mounts import MountFingerprinter, listing_entries
    from strata.notebook.remote_executor import create_notebook_executor_app

    files = _files("bucket/data", {"a": b"aa", "b": b"bbb"})
    # s3fs lists a datetime: the entries carry it as text, and the keys still agree.
    for info in files.values():
        del info["mtime"]
        info["LastModified"] = datetime(2026, 10, 1, tzinfo=UTC)
    _install_fake_fsspec(monkeypatch, _FakeRemoteFS(files))
    monkeypatch.setenv("STRATA_NOTEBOOK_MOUNT_CACHE_DIR", str(tmp_path / "worker"))
    # The worker's default credential for s3 adds an option the server does not have.
    monkeypatch.setenv("STRATA_NOTEBOOK_CREDENTIALS", json.dumps({"anon": {"anon": "true"}}))
    monkeypatch.setenv("STRATA_NOTEBOOK_MOUNT_CREDENTIALS", json.dumps({"s3": "anon"}))
    worker_reader = MountResolver(
        cache_dir=tmp_path / "w",
        credential_resolver=CredentialResolver(
            {"anon": {"anon": "true"}}, scheme_defaults={"s3": "anon"}
        ),
        mount_cache=MountCache(tmp_path / "worker"),
    )
    # The server lists with no options of its own.
    _, listing = MountFingerprinter.remote_listing_sync("s3", "bucket/data", None)
    assert listing is not None
    body = {
        "mounts": [
            {"spec": spec.model_dump(mode="json"), "entries": listing_entries(listing)}
            for spec in (_mount(), _mount(credential="nope"))
        ]
    }

    with TestClient(create_notebook_executor_app()) as client:
        features = client.get("/health").json()["capabilities"]["features"]
        cold = client.post("/v1/mount-cache/cached", json=body).json()
        await worker_reader.prepare_mounts([_mount("s3://bucket/data/a")])
        partly = client.post("/v1/mount-cache/cached", json=body).json()
        await worker_reader.prepare_mounts([_mount()])
        warm = client.post("/v1/mount-cache/cached", json=body).json()

    assert features["mount_cache_probe"] is True
    # A credential the worker does not have: it cannot tell.
    assert [cold, partly, warm] == [
        {"cached_bytes": [0, None]},
        {"cached_bytes": [2, None]},
        {"cached_bytes": [5, None]},
    ]


@pytest.mark.parametrize("transport", ["direct", "signed"])
async def test_a_remote_run_shows_the_workers_share_and_what_it_moved(
    tmp_path, remote, monkeypatch, notebook_executor_server, notebook_personal_server, transport
):
    from strata.config import StrataConfig
    from strata.notebook import reads_index
    from strata.notebook.executor import CellExecutor
    from strata.notebook.models import WorkerBackendType, WorkerSpec
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    monkeypatch.setenv("STRATA_NOTEBOOK_MOUNT_CACHE_DIR", str(tmp_path / "worker"))
    config = StrataConfig(cache_dir=tmp_path / "cache", notebook_mount_cache_dir=tmp_path / "srv")
    monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: config)
    source = "# @mount data s3://bucket/data ro\nn = len(list(data.iterdir()))"
    nb = create_notebook(tmp_path / "nb", "Remote moves")
    add_cell_to_notebook(nb, "c1", None)
    write_cell(nb, "c1", source)
    session = NotebookSession(parse_notebook(nb), nb)
    worker_config = {"url": notebook_executor_server["execute_url"], "transport": transport}
    if transport == "signed":
        worker_config["strata_url"] = notebook_personal_server["base_url"]
    session.notebook_state.workers = [
        WorkerSpec(
            name="remote", backend=WorkerBackendType.EXECUTOR, runtime_id="r", config=worker_config
        )
    ]
    session.notebook_state.worker = "remote"
    executor = CellExecutor(session)
    cell = session.notebook_state.get_cell("c1")

    first = await executor.execute_cell("c1", source)
    assert first.success, first.error
    assert first.execution_method == "executor"
    shares = [cell.inputs[0].cached_bytes]
    moved = [reads_index.load(nb)["c1"]]
    again = await executor.execute_cell_rerun("c1", source)
    assert again.success, again.error
    shares.append(cell.inputs[0].cached_bytes)
    moved.append(reads_index.load(nb)["c1"])

    # Asked of the worker, not read from the server's own (empty) cache.
    assert shares == [0, 5]
    reported = [(e["machine"], e["inputs"][0]["bytes_moved"]) for e in moved]
    # A signed rerun under the same transport provenance gets the first run's bundle
    # back, report included, so only the first run's figure is the worker's own.
    runs = 1 if transport == "signed" else 2
    assert reported[:runs] == [("remote", 5), ("remote", 0)][:runs]
    assert not (tmp_path / "srv").exists() or not any((tmp_path / "srv").rglob("*.lock"))


@pytest.fixture
def scan_server(tmp_path):
    """A live personal server the cells' ``strata`` client scans through."""
    from strata.config import StrataConfig
    from tests.conftest import find_free_port, run_server

    config = StrataConfig(
        host="127.0.0.1",
        port=find_free_port(),
        cache_dir=tmp_path / "scan-cache",
        artifact_dir=tmp_path / "scan-artifacts",
        deployment_mode="personal",
        rate_limit_enabled=False,
    )
    with run_server(config) as base:
        yield base


def _scan(table_uri: str) -> str:
    return (
        f"# @table events {table_uri}\n"
        "scan = strata.materialize(inputs=[events], transform={'executor': 'scan@v1'})\n"
        "rows = scan.to_arrow().num_rows\n"
        "del scan\n"
    )


def _chunk_bytes(table) -> int:
    """Every column chunk of the snapshot's data files, compressed: a full scan's read."""
    import pyarrow.parquet as pq

    total = 0
    for task in table.scan().plan_files():
        meta = pq.ParquetFile(task.file.file_path.removeprefix("file://")).metadata
        for rg in range(meta.num_row_groups):
            row_group = meta.row_group(rg)
            total += sum(
                row_group.column(i).total_compressed_size for i in range(row_group.num_columns)
            )
    return total


def _table_moved(nb, cell_id: str) -> int | None:
    from strata.notebook import reads_index

    (entry,) = reads_index.load(nb)[cell_id]["inputs"]
    assert entry["kind"] == "table"
    return entry["bytes_moved"]


@pytest.mark.integration
async def test_a_cells_scans_count_what_they_read_from_storage(
    scan_server, tmp_path, temp_warehouse
):
    """The first scan reads the files; a rerun is answered by the stored scan."""
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    source = _scan(temp_warehouse["table_uri"])
    nb = create_notebook(tmp_path / "nb", "Scans")
    add_cell_to_notebook(nb, "c1", None)
    write_cell(nb, "c1", source)
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    executor = CellExecutor(session)

    first = await executor.execute_cell("c1", source)
    assert first.success, first.error
    moved_first = _table_moved(nb, "c1")
    again = await executor.execute_cell_rerun("c1", source)
    assert again.success, again.error

    assert moved_first == _chunk_bytes(temp_warehouse["table"]) > 0
    assert _table_moved(nb, "c1") == 0


@pytest.mark.integration
async def test_a_batched_run_counts_each_cells_scans_apart(scan_server, tmp_path, temp_warehouse):
    """One token for the batch; each cell's own scans count for it."""
    import asyncio

    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
    from strata.notebook.ws import (
        _ensure_execution_state,
        _handle_notebook_run_all,
        _notebook_connections,
        forget_notebook_execution_state,
    )

    class _Socket:
        async def send_text(self, text: str) -> None:
            return None

    uri = temp_warehouse["table_uri"]
    nb = create_notebook(tmp_path / "nb", "Batched scans")
    add_cell_to_notebook(nb, "c1", None)
    write_cell(nb, "c1", _scan(uri))
    add_cell_to_notebook(nb, "c2", "c1")
    write_cell(nb, "c2", _scan(uri).replace("rows =", "rows2 =") + "total = rows + rows2\n")
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    session._analyze_and_build_dag()
    session.environment_sync_state = "ready"
    socket = _Socket()
    _notebook_connections[session.id] = [socket]
    try:
        state = _ensure_execution_state(session.id)
        await _handle_notebook_run_all(socket, session, state, session.id, {})
        if state.execution_task is not None:
            await asyncio.gather(state.execution_task)
    finally:
        _notebook_connections.pop(session.id, None)
        forget_notebook_execution_state(session.id)

    assert [c.status.value for c in session.notebook_state.cells] == ["ready", "ready"]
    # The second cell's scan is the first's stored artifact.
    assert _table_moved(nb, "c1") == _chunk_bytes(temp_warehouse["table"])
    assert _table_moved(nb, "c2") == 0
