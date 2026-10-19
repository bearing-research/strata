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
