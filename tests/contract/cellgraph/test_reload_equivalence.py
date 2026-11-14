"""A cell reads the same after a reopen as it did in the session that ran it.

A client builds a cell's view from live frames while it watches, and from
``serialize_cell`` of a freshly opened session when it reloads. A field the live
session holds and a reopened one drops vanishes on reload (#1455, #1457, #1425, #1433,
#1401). After every run step of every scenario, this opens a new session on the same
directory and compares each cell's ``serialize_cell`` with the live session's, both
after a staleness pass, as a server's open does.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

if os.name == "nt":
    pytest.skip("the notebook subsystem is not run on Windows", allow_module_level=True)

from tests.contract.cellgraph import record  # noqa: E402

# What one run says about itself and no later process can: the run that served it
# is gone. A reopened cell says how it last got its value through ``last_got``.
PROCESS_LOCAL = {"cache_hit", "execution_method"}


def _reopened(notebook_dir: Any) -> Any:
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession

    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir, fetch_secrets=False)
    session.refresh_environment_runtime()
    session.compute_staleness()
    return session


def _differences(live: Any, reopened: Any) -> dict[str, dict[str, Any]]:
    live.compute_staleness()
    found: dict[str, dict[str, Any]] = {}
    for cell in live.notebook_state.cells:
        other = reopened.notebook_state.get_cell(cell.id)
        assert other is not None, f"{cell.id} is gone after a reopen"
        before, after = live.serialize_cell(cell), reopened.serialize_cell(other)
        differ = {
            key: {"live": before.get(key), "reopened": after.get(key)}
            for key in sorted((set(before) | set(after)) - PROCESS_LOCAL)
            if before.get(key) != after.get(key)
        }
        if differ:
            found[cell.id] = differ
    return found


# What a reopen loses today, found by this test when it was written (2026-11-13). Each is
# a bug to fix in its own PR; a fixed one passes here, and the strict xfail then fails
# until its entry is removed.
KNOWN_RELOAD_GAPS = {
    "cell_level_ref": (
        "a loop cell's artifact_uri is not restored, so the cell after it (@after lp) "
        "reads stale (upstream) after a reopen while the live session reads it ready"
    ),
    "credentials": (
        "a cell whose mount names a credential loses its artifact_uri and artifact_uris "
        "after a reopen"
    ),
    "fanout": (
        "a per_variant cell's artifact_uri and artifact_uris are not restored, so its "
        "consumer reads stale (upstream, unavailable) after a reopen"
    ),
    "mounts": "a cell with a read-write mount loses its artifact_uri and artifact_uris",
    "sql": (
        "a SQL cell that ran reads idle after a reopen and loses its artifact_uris, so "
        "the cell reading its result reads stale (upstream)"
    ),
    "worker": (
        "a cell run on a worker loses remote_worker and remote_transport, which its header shows"
    ),
}


def _scenario(name: str) -> object:
    gap = KNOWN_RELOAD_GAPS.get(name)
    if gap is None:
        return name
    return pytest.param(name, marks=pytest.mark.xfail(reason=gap, strict=True))


@pytest.mark.usefixtures("executor_host_only")
@pytest.mark.parametrize(
    "name", [_scenario(n) for n in record.scenario_names() if n not in record.REQUIRES]
)
def test_every_run_reads_the_same_after_a_reopen(name):
    failures = []
    with record._locked(name):
        for index, (step, notebook_dir, _spy, _run, session) in enumerate(record._steps(name)):
            if session is None:
                continue
            differences = _differences(session, _reopened(notebook_dir))
            if differences:
                failures.append(f"step {index} ({step['op']}): {differences}")

    assert not failures, "\n".join(failures)
