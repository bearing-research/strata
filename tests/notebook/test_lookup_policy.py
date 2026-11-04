"""What each cell path does to decide a hit, kept apart on purpose.

The table in ``docs/internal/executor-cleanup-brief.md`` (PR 1) names the policies. Each row is
pinned here at the layer it names, the case they differ on included: a key with a row in the
store and no local canonical output, which Python and R treat as a miss with no team pull, and
batch pulls for.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from vestigo.provenance import TransformSpec, derive_subkey

from strata.notebook.cell_paths import batch, fanout
from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from tests.notebook.conftest import skip_if_no_r

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX cell execution")

PYTHON = ("python", "value = 1\n", "value = 2\n", "seen = value + 1\n")
R = ("r", "value <- 1\n", "value <- 2\n", "seen <- value + 1\n")
LANGUAGES = [
    pytest.param(*PYTHON, id="python"),
    pytest.param(*R, id="r", marks=skip_if_no_r),
]


def _session(tmp_path: Path, *cells: tuple[str, str, str]) -> NotebookSession:
    """A notebook of ``(cell id, language, source)``."""
    nb = create_notebook(tmp_path / "nb", "Lookup")
    after = None
    for cell_id, language, source in cells:
        add_cell_to_notebook(nb, cell_id, after, language=language)
        write_cell(nb, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


async def _run(session: NotebookSession, cell_id: str):
    cell = session.notebook_state.get_cell(cell_id)
    result = await CellExecutor(session).execute_cell(cell_id, cell.source)
    assert result.success, result.error
    return result


def _edit(session: NotebookSession, cell_id: str, source: str) -> None:
    write_cell(session.path, cell_id, source)
    session.notebook_state.get_cell(cell_id).source = source
    session.re_analyze_cell(cell_id)


def _latest(session: NotebookSession, cell_id: str, label: str) -> int | None:
    manager = session.get_artifact_manager()
    latest = manager.artifact_store.get_latest_version(manager.cell_artifact_id(cell_id, label))
    return latest.version if latest is not None else None


async def _key(session: NotebookSession, cell_id: str) -> str:
    cell = session.notebook_state.get_cell(cell_id)
    prov = await CellExecutor(session)._compute_cell_provenance(cell_id, cell.source)
    return prov.provenance_hash


def _row_elsewhere(session: NotebookSession, provenance_hash: str) -> None:
    """A row under *provenance_hash* in this store that is not this cell's canonical output."""
    store = session.get_artifact_manager().artifact_store
    blob = json.dumps(1).encode()
    version = store.create_artifact(
        artifact_id="another_notebook_row",
        provenance_hash=provenance_hash,
        transform_spec=TransformSpec(
            executor="seed@v1", params={"content_type": "json/object"}, inputs=[]
        ),
    )
    store.write_blob("another_notebook_row", version, blob)
    store.finalize_artifact(
        "another_notebook_row",
        version,
        schema_json="{}",
        row_count=0,
        byte_size=len(blob),
        content_sha256=hashlib.sha256(blob).hexdigest(),
    )


@pytest.fixture
def pulls(monkeypatch) -> list[dict]:
    """Every team-store pull a path asks for, the store itself having nothing."""
    calls: list[dict] = []

    async def pull(self, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(CellExecutor, "_pull_from_team_store", pull)
    return calls


class TestPythonAndR:
    """Provenance lookup, then the canonical check; a pull only when the lookup misses."""

    @pytest.mark.parametrize(("language", "first", "second", "reader"), LANGUAGES)
    async def test_a_key_with_no_row_asks_the_team_store(
        self, tmp_path, pulls, language, first, second, reader
    ):
        session = _session(tmp_path, ("v", language, first), ("r", language, reader))

        result = await _run(session, "v")

        assert not result.cache_hit
        assert [call["cell_id"] for call in pulls] == ["v"]

    @pytest.mark.parametrize(("language", "first", "second", "reader"), LANGUAGES)
    async def test_a_key_with_a_row_elsewhere_is_a_miss_with_no_pull(
        self, tmp_path, pulls, language, first, second, reader
    ):
        session = _session(tmp_path, ("v", language, first), ("r", language, reader))
        _row_elsewhere(session, derive_subkey(await _key(session, "v"), "value"))

        result = await _run(session, "v")

        assert not result.cache_hit
        assert pulls == []
        assert _latest(session, "v", "value") == 1

    @pytest.mark.parametrize(("language", "first", "second", "reader"), LANGUAGES)
    async def test_an_unchanged_run_hits_with_no_pull(
        self, tmp_path, pulls, language, first, second, reader
    ):
        session = _session(tmp_path, ("v", language, first), ("r", language, reader))
        await _run(session, "v")
        pulls.clear()

        result = await _run(session, "v")

        assert result.cache_hit
        assert pulls == []

    @pytest.mark.parametrize(("language", "first", "second", "reader"), LANGUAGES)
    async def test_a_revert_promotes_the_variable(
        self, tmp_path, pulls, language, first, second, reader
    ):
        session = _session(tmp_path, ("v", language, first), ("r", language, reader))
        await _run(session, "v")
        _edit(session, "v", second)
        await _run(session, "v")
        _edit(session, "v", first)
        pulls.clear()

        result = await _run(session, "v")

        assert result.cache_hit
        assert pulls == []
        # The first version, written again as latest; nothing ran to make it.
        assert _latest(session, "v", "value") == 3
        assert result.artifact_uri.endswith("@v=3")
        cell = session.notebook_state.get_cell("v")
        assert cell.artifact_uris["value"].endswith("@v=3")
        manager = session.get_artifact_manager()
        latest = manager.artifact_store.get_latest_version(manager.cell_artifact_id("v", "value"))
        assert latest.provenance_hash == derive_subkey(await _key(session, "v"), "value")


class TestPythonLeaves:
    async def test_a_console_only_leaf_is_cached(self, tmp_path, pulls):
        session = _session(tmp_path, ("c", "python", 'print("one")\n'))
        await _run(session, "c")

        result = await _run(session, "c")

        assert result.cache_hit
        assert result.stdout == "one\n"
        assert pulls == []

    async def test_a_console_only_leaf_is_promoted_by_a_revert(self, tmp_path):
        session = _session(tmp_path, ("c", "python", 'print("one")\n'))
        await _run(session, "c")
        _edit(session, "c", 'print("two")\n')
        await _run(session, "c")
        _edit(session, "c", 'print("one")\n')

        result = await _run(session, "c")

        assert result.cache_hit
        assert result.stdout == "one\n"
        assert _latest(session, "c", "__console__") == 3

    async def test_a_revert_brings_back_the_earlier_display_set(self, tmp_path):
        one = 'display(Markdown("one"))\n'
        session = _session(tmp_path, ("d", "python", one))
        await _run(session, "d")
        _edit(session, "d", 'display(Markdown("two"))\ndisplay(Markdown("three"))\n')
        assert len((await _run(session, "d")).display_outputs) == 2
        _edit(session, "d", one)

        result = await _run(session, "d")

        assert result.cache_hit
        assert [d["markdown_text"] for d in result.display_outputs] == ["one"]


@skip_if_no_r
class TestRLeaves:
    async def test_a_console_only_leaf_is_not_cached(self, tmp_path):
        session = _session(tmp_path, ("c", "r", 'cat("one\\n")\n'))
        await _run(session, "c")

        result = await _run(session, "c")

        assert not result.cache_hit
        assert _latest(session, "c", "__console__") is None

    async def test_a_display_leaf_is_cached(self, tmp_path):
        session = _session(tmp_path, ("p", "r", "plot(1)\n"))
        await _run(session, "p")

        result = await _run(session, "p")

        assert result.cache_hit
        assert len(result.display_outputs) == 1

    async def test_a_revert_brings_back_the_earlier_display_set(self, tmp_path):
        session = _session(tmp_path, ("p", "r", "plot(1)\n"))
        await _run(session, "p")
        _edit(session, "p", "plot(1)\nplot(2)\n")
        assert len((await _run(session, "p")).display_outputs) == 2
        _edit(session, "p", "plot(1)\n")

        result = await _run(session, "p")

        assert result.cache_hit
        assert len(result.display_outputs) == 1
        assert _latest(session, "p", "__display__0") == 3

    async def test_a_revert_keeps_the_variable_and_its_displays_together(self, tmp_path):
        first = "value <- 1\nplot(1)\n"
        session = _session(tmp_path, ("v", "r", first), ("r", "r", "seen <- value + 1\n"))
        await _run(session, "v")
        _edit(session, "v", "value <- 2\nplot(1)\nplot(2)\n")
        assert len((await _run(session, "v")).display_outputs) == 2
        _edit(session, "v", first)

        result = await _run(session, "v")

        assert result.cache_hit
        assert len(result.display_outputs) == 1
        assert _latest(session, "v", "value") == 3
        assert _latest(session, "v", "__display__0") == 3


class TestBatch:
    """Canonical materialization; a pull whenever that fails."""

    async def _check(self, session, cell_id, tmp_path):
        return await batch.batch_service_cache_check(
            session,
            CellExecutor(session)._batch_context(),
            cell_id,
            tmp_path,
            executed_sources={},
            provenance={},
            use_cache=True,
        )

    async def test_a_key_with_no_row_asks_the_team_store(self, tmp_path, pulls):
        session = _session(tmp_path, ("v", "python", PYTHON[1]), ("r", "python", PYTHON[3]))

        answer = await self._check(session, "v", tmp_path)

        assert answer["cache_hit"] is False
        assert [call["cell_id"] for call in pulls] == ["v"]

    async def test_a_key_with_a_row_elsewhere_still_asks_the_team_store(self, tmp_path, pulls):
        """Where batch differs from a single run, which takes this for a miss."""
        session = _session(tmp_path, ("v", "python", PYTHON[1]), ("r", "python", PYTHON[3]))
        _row_elsewhere(session, derive_subkey(await _key(session, "v"), "value"))

        answer = await self._check(session, "v", tmp_path)

        assert answer["cache_hit"] is False
        assert [call["cell_id"] for call in pulls] == ["v"]

    async def test_an_edit_is_not_served_the_cells_earlier_output(self, tmp_path, pulls):
        session = _session(tmp_path, ("v", "python", PYTHON[1]), ("r", "python", PYTHON[3]))
        await _run(session, "v")
        _edit(session, "v", PYTHON[2])
        pulls.clear()

        answer = await self._check(session, "v", tmp_path)

        assert answer["cache_hit"] is False
        assert [call["cell_id"] for call in pulls] == ["v"]

    async def test_a_display_is_served_from_its_canonical_id(self, tmp_path, pulls):
        session = _session(tmp_path, ("d", "python", 'display(Markdown("one"))\n'))
        await _run(session, "d")

        answer = await self._check(session, "d", tmp_path)

        assert answer["cache_hit"] is True
        assert [d["markdown_text"] for d in answer["cached_displays"]] == ["one"]

    async def test_a_revert_promotes_the_variable_with_no_pull(self, tmp_path, pulls):
        session = _session(tmp_path, ("v", "python", PYTHON[1]), ("r", "python", PYTHON[3]))
        await _run(session, "v")
        _edit(session, "v", PYTHON[2])
        await _run(session, "v")
        _edit(session, "v", PYTHON[1])
        pulls.clear()

        answer = await self._check(session, "v", tmp_path)

        assert answer["cache_hit"] is True
        assert pulls == []
        assert _latest(session, "v", "value") == 3

    async def test_a_console_only_leaf_is_cached_and_promoted(self, tmp_path, pulls):
        session = _session(tmp_path, ("c", "python", 'print("one")\n'))
        await _run(session, "c")
        assert (await self._check(session, "c", tmp_path))["cache_hit"] is True

        _edit(session, "c", 'print("two")\n')
        await _run(session, "c")
        _edit(session, "c", 'print("one")\n')
        pulls.clear()

        answer = await self._check(session, "c", tmp_path)

        assert answer["cache_hit"] is True
        assert pulls == []
        assert _latest(session, "c", "__console__") == 3


LOOP = "# @loop max_iter={n} carry=state\nstate = state + {step}\n"
SEED = ("s", "python", "state = 0\n")


class TestWholeLoopReuse:
    """The carry and every consumed variable are the cell's canonical outputs; no team tier."""

    def _source(self, n: int = 2, step: int = 1) -> str:
        return LOOP.format(n=n, step=step)

    async def test_a_cold_loop_never_asks_the_team_store(self, tmp_path, pulls):
        session = _session(tmp_path, SEED, ("l", "python", self._source()))

        result = await _run(session, "l")

        assert not result.cache_hit
        assert [call for call in pulls if call["cell_id"] == "l"] == []

    async def test_an_unchanged_loop_is_reused(self, tmp_path, pulls):
        session = _session(tmp_path, SEED, ("l", "python", self._source()))
        await _run(session, "l")

        result = await _run(session, "l")

        assert result.cache_hit
        assert [call for call in pulls if call["cell_id"] == "l"] == []

    async def test_a_revert_promotes_the_carry(self, tmp_path, pulls):
        session = _session(tmp_path, SEED, ("l", "python", self._source()))
        await _run(session, "l")
        _edit(session, "l", self._source(step=2))
        await _run(session, "l")
        _edit(session, "l", self._source())

        result = await _run(session, "l")

        assert result.cache_hit
        assert [call for call in pulls if call["cell_id"] == "l"] == []
        assert _latest(session, "l", "state") == 3

    async def test_a_row_elsewhere_is_not_the_loops_result(self, tmp_path, pulls):
        session = _session(tmp_path, SEED, ("l", "python", self._source()))
        _row_elsewhere(session, derive_subkey(await _key(session, "l"), "state"))

        result = await _run(session, "l")

        assert not result.cache_hit
        assert [call for call in pulls if call["cell_id"] == "l"] == []
        assert _latest(session, "l", "state") == 1


class TestIterationReuse:
    """An iteration has no lookup by key: only ``start_from`` reuses one."""

    def _iteration(self, session, cell_id: str, k: int) -> int | None:
        manager = session.get_artifact_manager()
        latest = manager.artifact_store.get_latest_version(
            manager.cell_artifact_id(cell_id, "state", iteration=k)
        )
        return latest.version if latest is not None else None

    async def test_a_longer_loop_runs_its_first_iterations_again(self, tmp_path):
        session = _session(tmp_path, SEED, ("l", "python", LOOP.format(n=2, step=1)))
        await _run(session, "l")
        assert self._iteration(session, "l", 0) == 1

        _edit(session, "l", LOOP.format(n=3, step=1))
        result = await _run(session, "l")

        assert not result.cache_hit
        assert self._iteration(session, "l", 0) == 2
        assert self._iteration(session, "l", 2) == 1

    async def test_start_from_seeds_a_fork_from_a_stored_iteration(self, tmp_path):
        fork = "# @loop max_iter=1 carry=state start_from=l@iter=1\nstate = state + 10\n"
        session = _session(
            tmp_path, SEED, ("l", "python", LOOP.format(n=3, step=1)), ("f", "python", fork)
        )
        await _run(session, "l")
        versions = [self._iteration(session, "l", k) for k in range(3)]

        result = await _run(session, "f")

        # l's iteration 1 holds 2; nothing of l ran again to get it.
        assert not result.cache_hit
        assert [self._iteration(session, "l", k) for k in range(3)] == versions
        assert _latest_value(session, "f", "state") == 12


def _latest_value(session: NotebookSession, cell_id: str, label: str):
    manager = session.get_artifact_manager()
    store = manager.artifact_store
    latest = store.get_latest_version(manager.cell_artifact_id(cell_id, label))
    return json.loads(store.read_blob(latest.id, latest.version))


class TestEach:
    """One probe of the whole cell's key; team pulls are per instance."""

    def _session(self, tmp_path: Path, source: str) -> NotebookSession:
        raw = tmp_path / "raw"
        raw.mkdir()
        (raw / "a.txt").write_text("a")
        (raw / "b.txt").write_text("b")
        return _session(
            tmp_path,
            ("e", "python", f"# @mount data {raw.as_uri()} ro\n# @each p in data\n{source}"),
            ("r", "python", "seen = sorted(out.values())\n"),
        )

    async def test_a_cold_run_asks_once_per_item(self, tmp_path, pulls):
        session = self._session(tmp_path, "out = p.read_text()\n")

        await _run(session, "e")

        assert sorted(call["variant"] for call in pulls) == sorted(
            {call["variant"] for call in pulls}
        )
        assert len(pulls) == 2

    async def test_nothing_changed_is_one_probe_and_no_pull(self, tmp_path, pulls, monkeypatch):
        session = self._session(tmp_path, "out = p.read_text()\n")
        await _run(session, "e")
        pulls.clear()
        listed: list[str] = []
        real = fanout._list_each_items

        async def list_items(host, context, cell_id, spec, base):
            listed.append(cell_id)
            return await real(host, context, cell_id, spec, base)

        monkeypatch.setattr(fanout, "_list_each_items", list_items)

        result = await _run(session, "e")

        assert result.cache_hit
        assert pulls == []
        assert listed == []

    async def test_a_revert_promotes_the_manifest(self, tmp_path, pulls):
        session = self._session(tmp_path, "out = p.read_text()\n")
        first = session.notebook_state.get_cell("e").source
        await _run(session, "e")
        _edit(session, "e", first.replace("p.read_text()", "p.read_text().upper()"))
        await _run(session, "e")
        _edit(session, "e", first)
        pulls.clear()

        result = await _run(session, "e")

        assert result.cache_hit
        assert pulls == []
        assert _latest(session, "e", "out") == 3


def test_an_inline_display_is_filed_under_its_canonical_id(tmp_path):
    """SQL's displays: the second call reuses the version the first one stored."""
    session = _session(tmp_path, ("v", "python", PYTHON[1]))
    display = {"content_type": "text/markdown", "markdown_text": "one"}
    executor = CellExecutor(session)

    first = executor._store_inline_display_outputs("v", "key", [display])
    second = executor._store_inline_display_outputs("v", "key", [display])

    canonical = session.get_artifact_manager().cell_artifact_id("v", "__display__0")
    assert first[0]["artifact_uri"] == f"strata://artifact/{canonical}@v=1"
    assert second[0]["artifact_uri"] == first[0]["artifact_uri"]


@pytest.mark.parametrize("label", ["value", "__display__0", "__console__"])
def test_a_canonical_id_is_the_one_every_path_stored_under(tmp_path, label):
    session = _session(tmp_path, ("v", "python", PYTHON[1]))
    notebook_id = session.notebook_state.id

    built = session.get_artifact_manager().cell_artifact_id("v", label)

    assert built == f"nb_{notebook_id}_cell_v_var_{label}"
