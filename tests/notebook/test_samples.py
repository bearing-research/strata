"""The bundled samples: copies of ``examples/``, and what opening one does."""

from __future__ import annotations

import argparse
import asyncio
import filecmp
import tomllib
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from strata.notebook import routes, tutorial
from strata.notebook.executor import CellExecutor
from strata.notebook.models import CellStatus
from strata.notebook.parser import parse_notebook
from strata.notebook.samples import SAMPLES, SAMPLES_DIR, copy_sample, get_sample
from strata.notebook.scopes import NOTEBOOK_SCOPE_WRITE, required_scope_for_route
from strata.notebook.session import NotebookSession

EXAMPLES_DIR = Path(__file__).parents[2] / "examples"


# What running an example leaves behind in a checkout; all of it is gitignored.
_RUNTIME = {".strata", ".venv", "work", "uv.lock", "__pycache__"}


def _files(root: Path) -> list[str]:
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and not _RUNTIME & set(p.relative_to(root).parts)
    )


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s.name)
def test_a_bundled_sample_matches_its_example(sample):
    bundled = SAMPLES_DIR / sample.name
    example = EXAMPLES_DIR / sample.name
    files = _files(example)
    assert _files(bundled) == files
    _, mismatch, errors = filecmp.cmpfiles(example, bundled, files, shallow=False)
    assert (mismatch, errors) == ([], [])


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s.name)
def test_a_copy_of_each_sample_reads_as_a_notebook_with_its_cells(sample, tmp_path):
    notebook = parse_notebook(copy_sample(sample, tmp_path))
    assert notebook.cells
    assert all(cell.source.strip() for cell in notebook.cells)


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: s.name)
def test_no_sample_opens_with_a_warning(sample, tmp_path):
    notebook_dir = copy_sample(sample, tmp_path)
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    assert session.dag.shadow_warnings == {}


def test_every_sample_runs_without_a_key():
    """A first run should not stop at a missing API key, and a prompt cell needs one."""
    for sample in SAMPLES:
        toml = tomllib.loads((SAMPLES_DIR / sample.name / "notebook.toml").read_text())
        assert all(cell["language"] != "prompt" for cell in toml["cells"]), sample.name


def test_the_samples_directory_holds_only_listed_samples():
    assert {p.name for p in SAMPLES_DIR.iterdir()} == {s.name for s in SAMPLES}


def test_a_copy_is_a_notebook_of_its_own(tmp_path):
    sample = get_sample("caching")
    first = copy_sample(sample, tmp_path)
    second = copy_sample(sample, tmp_path)

    assert (first.name, second.name) == ("caching", "caching_2")
    ids = [tomllib.loads((d / "notebook.toml").read_text())["notebook_id"] for d in (first, second)]
    shipped = tomllib.loads((SAMPLES_DIR / "caching" / "notebook.toml").read_text())
    assert len({*ids, shipped["notebook_id"]}) == 3
    assert parse_notebook(first).name == sample.title
    assert parse_notebook(second).name == f"{sample.title} 2"


def test_the_listed_title_is_the_notebooks_own(tmp_path):
    for sample in SAMPLES:
        toml = tomllib.loads((SAMPLES_DIR / sample.name / "notebook.toml").read_text())
        assert toml["name"] == sample.title


def test_listing_names_every_sample(capsys):
    args = argparse.Namespace(name=None, server="http://unused")
    assert tutorial.tutorial_main(args) == 0
    out = capsys.readouterr().out
    for sample in SAMPLES:
        assert sample.name in out


def test_an_unknown_sample_is_an_input_error(capsys):
    args = argparse.Namespace(name="nope", server="http://unused")
    assert tutorial.tutorial_main(args) == 2
    assert "no sample named 'nope'" in capsys.readouterr().err


def test_a_running_server_is_reused_and_the_url_printed(monkeypatch, capsys):
    posted = {}
    monkeypatch.setattr(tutorial, "_server_alive", lambda url: True)

    def post(url, body, server_url, *, timeout):
        posted["url"] = url
        return "sid123"

    monkeypatch.setattr(tutorial, "_post_for_session", post)
    monkeypatch.setattr(
        tutorial, "_spawn_server", lambda *a: pytest.fail("a running server is reused")
    )

    args = argparse.Namespace(name="caching", server="http://localhost:9/")
    assert tutorial.tutorial_main(args) == 0
    assert posted["url"] == "http://localhost:9/v1/notebooks/samples/caching/open"
    assert "http://localhost:9/#/notebook/sid123" in capsys.readouterr().out


@pytest.fixture
def sample_client(tmp_path, monkeypatch):
    monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)
    monkeypatch.setattr(routes, "_get_notebook_storage_root", lambda: tmp_path)
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_the_route_lists_what_the_cli_lists(sample_client):
    listed = sample_client.get("/v1/notebooks/samples").json()["samples"]
    assert [row["name"] for row in listed] == [sample.name for sample in SAMPLES]
    assert all(row["title"] and row["description"] for row in listed)


def test_opening_copies_warms_and_returns_a_session(sample_client, tmp_path, monkeypatch):
    warmed = []

    async def warm(python, modules):
        warmed.append(modules)

    monkeypatch.setattr(routes, "_warm_environment", warm)

    body = sample_client.post("/v1/notebooks/samples/caching/open").json()

    assert body["session_id"]
    assert Path(body["path"]) == tmp_path / "caching"
    assert (tmp_path / "caching" / "notebook.toml").is_file()
    assert warmed == [get_sample("caching").warm_imports]


def test_with_no_environment_there_is_nothing_to_warm(sample_client, monkeypatch):
    async def warm(python, modules):
        pytest.fail("nothing to warm")

    opened = routes._open_with_synced_environment

    async def open_without_a_venv(notebook_path, timing):
        session = await opened(notebook_path, timing)
        session.venv_python = None
        return session

    monkeypatch.setattr(routes, "_warm_environment", warm)
    monkeypatch.setattr(routes, "_open_with_synced_environment", open_without_a_venv)
    assert sample_client.post("/v1/notebooks/samples/caching/open").status_code == 200


def test_opening_a_sample_needs_write_like_creating_a_notebook():
    open_sample = required_scope_for_route("POST", "/v1/notebooks/samples/{name}/open")
    assert open_sample == required_scope_for_route("POST", "/v1/notebooks/create")
    assert open_sample == NOTEBOOK_SCOPE_WRITE


def test_a_sample_opened_twice_is_two_notebooks(sample_client, tmp_path):
    first = sample_client.post("/v1/notebooks/samples/caching/open").json()
    second = sample_client.post("/v1/notebooks/samples/caching/open").json()
    assert first["session_id"] != second["session_id"]
    assert first["path"] != second["path"]


def test_an_unknown_sample_is_not_found(sample_client, tmp_path):
    assert sample_client.post("/v1/notebooks/samples/nope/open").status_code == 404
    assert list(tmp_path.iterdir()) == []


def test_warming_imports_into_the_given_interpreter(tmp_path, monkeypatch):
    # A stand-in interpreter: warming must run it, and must survive it failing.
    seen = tmp_path / "seen"
    python = tmp_path / "python"
    python.write_text(f'#!/bin/sh\necho "$2" > {seen}\nexit 3\n')
    python.chmod(0o755)

    asyncio.run(routes._warm_environment(python, ("pandas", "matplotlib.pyplot")))

    assert seen.read_text().strip() == "import pandas, matplotlib.pyplot"


class TestTheTutorialDoesWhatItSays:
    """Its steps, read back from the notebook they describe."""

    @pytest.fixture
    def session(self, tmp_path):
        notebook_dir = copy_sample(get_sample("caching"), tmp_path)
        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
        session.refresh_environment_runtime()
        return session

    @staticmethod
    def _run_all(session) -> dict:
        executor = CellExecutor(session, session.warm_pool)

        async def run():
            results = {}
            for cell in session.notebook_state.cells:
                results[cell.id] = await executor.execute_cell(cell.id, cell.source)
                assert results[cell.id].success, results[cell.id].error
            return results

        return asyncio.run(run())

    @staticmethod
    def _edit(session, old: str, new: str) -> None:
        cell = session.notebook_state.get_cell("summarize")
        assert old in cell.source
        cell.source = cell.source.replace(old, new)

    def test_the_steps(self, session):
        code = ("load", "summarize", "plot")

        first = self._run_all(session)
        assert not any(first[c].cache_hit for c in code)

        # Step 2: nothing runs.
        second = self._run_all(session)
        assert all(second[c].cache_hit for c in code)

        # Step 3: only what depends on the edit goes stale, and only that runs.
        self._edit(session, ".sum()", ".mean()")
        staleness = session.compute_staleness()
        assert staleness["summarize"].status == CellStatus.STALE
        assert staleness["plot"].status == CellStatus.STALE
        assert staleness["load"].status == CellStatus.READY
        third = self._run_all(session)
        assert third["load"].cache_hit
        assert not third["summarize"].cache_hit
        assert not third["plot"].cache_hit

        # And going back is a hit for both.
        self._edit(session, ".mean()", ".sum()")
        fourth = self._run_all(session)
        assert all(fourth[c].cache_hit for c in code)

    def test_its_markdown_is_not_counted_as_cached(self, session):
        first = self._run_all(session)
        second = self._run_all(session)
        # What Run All records for every cell it runs, markdown included.
        for results in (first, second):
            assert not results["step1"].cache_hit
            for cell_id, result in results.items():
                session.record_execution(cell_id, result.duration_ms, result.cache_hit)
        summary = session.get_profiling_summary()
        ids = {row["cell_id"] for row in summary["cell_profiles"]}
        assert ids == {"load", "summarize", "plot"}
        assert summary["cache_hits"] == 3
        assert summary["cache_misses"] == 3

    def test_the_graph_step_4_describes(self, session):
        by_id = {c.id: c for c in session.notebook_state.cells}
        assert by_id["summarize"].upstream_ids == ["load"]
        assert by_id["plot"].upstream_ids == ["summarize"]
