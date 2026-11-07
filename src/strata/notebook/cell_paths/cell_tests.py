"""A cell's unit tests: its source and tests under pytest, with what a run of it reads."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strata.notebook import output_dirs
from strata.notebook.annotations import parse_annotations
from strata.notebook.cell_paths.context import no_interpreter_message
from strata.notebook.graph import analysis_with
from strata.notebook.harness_user import (
    LocalExecutionRefused,
    UnsafeRunFile,
    hand_over,
    identity_env,
    let_through,
    prepare_work_dir,
    resolve_harness_user,
)
from strata.notebook.keys import inactive_variant_refusal
from strata.notebook.layout import WORK_DIR
from strata.notebook.models import (
    CellLanguage,
    CellTestCase,
    CellTestResult,
    MountMode,
    MountSpec,
)
from strata.notebook.mounts import resolve_cell_mounts

if TYPE_CHECKING:
    from strata.notebook.executor import _CellProvenance
    from strata.notebook.executor_host import ExecutorHost

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CellTestsContext:
    """What running a cell's tests asks of the executor, and nothing else."""

    materialize_upstreams: Callable[[str], Awaitable[None]]
    forget_moved: Callable[[str], None]
    compute_provenance: Callable[..., Awaitable[_CellProvenance]]
    moved: Callable[[str], tuple[dict[str, int | None], dict[str, int]]]
    manifest_tables: Callable[..., dict[str, dict[str, Any]]]
    prepare_mounts: Callable[..., Awaitable[dict[str, Any]]]
    resolve_runtime_env: Callable[[str, dict[str, str]], dict[str, str]]
    fanout_variants: Callable[[str], Any]
    load_input_blobs: Callable[..., dict[str, dict[str, Any]]]
    add_dataset_inputs: Callable[..., None]
    harness_env: Callable[..., dict[str, str] | None]


def _variant_test_run(raw: dict[str, Any], variant: str) -> dict[str, Any]:
    """*raw* with each case named as pytest names a parametrized one: ``test_x[variant]``."""
    return {
        **raw,
        "tests": [
            {
                **case,
                "name": f"{case['name']}[{variant}]",
                "nodeid": f"{case['nodeid']}[{variant}]" if case.get("nodeid") else "",
            }
            for case in raw["tests"]
        ],
    }


def _merged_test_runs(raws: list[dict[str, Any]]) -> dict[str, Any]:
    """One result over a cell's test runs: totals summed, cases in run order."""
    if len(raws) == 1:
        return raws[0]
    totals = {
        key: sum(raw.get(key, 0) for raw in raws)
        for key in ("passed", "failed", "errored", "skipped")
    }
    return {**totals, "tests": [case for raw in raws for case in raw["tests"]]}


async def run_cell_tests(
    host: ExecutorHost, context: CellTestsContext, cell_id: str, test_source: str
) -> CellTestResult:
    """Run a Python cell's unit tests and persist the result.

    Tests target the functions/classes the cell defines, which are never
    stored as artifacts, so this re-executes the source with its upstream
    inputs injected and runs ``pytest`` in the notebook venv. It does not go
    through the harness and never touches the artifact cache or cell status.
    """
    # cell_test_runner is stdlib-only, kept beside its single caller.
    from strata.notebook.cell_test_runner import (
        PytestUnavailableError,
        run_cell_tests_in_dir,
    )
    from strata.notebook.runtime_state import persist_cell_test_result

    cell = host.notebook_state.get_cell(cell_id)
    if cell is None:
        raise FileNotFoundError(f"Cell {cell_id} not found")
    venv_python = host.venv_python
    if venv_python is None:
        raise RuntimeError(no_interpreter_message(host))

    source = cell.source
    if host.analysis is not None:
        analysis = analysis_with(cell, host.notebook_state, host.analysis)
        inactive = inactive_variant_refusal(analysis, cell_id)
        if inactive is not None:
            # It has no key to fingerprint its inputs by; nothing is recorded.
            source_hash, test_source_hash, _ = host.cell_test_fingerprint(
                cell_id, source, test_source
            )
            return CellTestResult(
                errored=1,
                tests=[
                    CellTestCase(name="<refused>", nodeid="", outcome="error", message=inactive)
                ],
                cell_source_hash=source_hash,
                test_source_hash=test_source_hash,
                ran_at=int(time.time() * 1000),
            )
    await context.materialize_upstreams(cell_id)
    context.forget_moved(cell_id)

    # The same fetch, mount, dataset, table and env inputs a run of the cell gets.
    # Cell tests are pytest over Python cells.
    annotations = parse_annotations(source, CellLanguage.PYTHON)
    # Every mount: a test reads ``cell.<name>``, which no source analysis sees.
    mount_specs = resolve_cell_mounts([], cell.mounts, annotations.mounts)
    prov = await context.compute_provenance(
        cell_id, source, annotations=annotations, mount_specs=mount_specs
    )
    problem = prov.fetch_error or prov.dataset_error
    if annotations.each is not None:
        problem = "Tests of an @each cell are not supported yet."
    tables: dict[str, dict[str, Any]] = {}
    if problem is None:
        try:
            tables = context.manifest_tables(annotations.tables, prov.table_snapshots)
        except RuntimeError as exc:
            problem = str(exc)
    mount_specs = [
        *mount_specs,
        *(
            MountSpec(name=name, uri=path.resolve().as_uri(), mode=MountMode.READ_ONLY)
            for name, path in prov.fetched.items()
        ),
    ]
    resolved_mounts = await context.prepare_mounts(mount_specs, cell_id)
    # The tests read this listing, which the provenance's may predate.
    prov = prov.over_mounts(resolved_mounts)
    runtime_env = context.resolve_runtime_env(cell_id, annotations.env)

    # Tests run the cell's source, so they are refused or dropped to the harness user
    # like any harness.
    refused: dict[str, Any] | None = None
    harness_user = None
    try:
        harness_user = resolve_harness_user()
        # The cell runs where a run of it does, its notebook's work/.
        prepare_work_dir(host.path, harness_user)
    except (LocalExecutionRefused, UnsafeRunFile) as exc:
        problem = str(exc)
    if problem is not None:
        refused = {
            "passed": 0,
            "failed": 0,
            "errored": 1,
            "skipped": 0,
            "tests": [{"name": "<refused>", "nodeid": "", "outcome": "error", "message": problem}],
        }

    pytest_unavailable = False
    auto_installed: list[str] = []
    with tempfile.TemporaryDirectory(prefix="strata_celltest_") as tmp:
        # A fan-out cell is tested once per variant, binding what that instance reads.
        runs: list[tuple[str | None, Path, dict[str, dict[str, Any]]]] = []
        for index, variant in enumerate(context.fanout_variants(cell_id) or (None,)):
            blob_dir = Path(tmp) / f"inputs-{index}"
            blob_dir.mkdir()
            input_specs = context.load_input_blobs(cell_id, blob_dir, fanout_variant=variant)
            context.add_dataset_inputs(input_specs, prov.datasets, blob_dir)
            runs.append((variant, blob_dir, input_specs))

        # The harness user gets the inputs and, once staged, the run directory, never tmp
        # itself: a process an earlier cell left running could plant links in it.
        let_through(Path(tmp), harness_user)
        for _, blob_dir, _ in runs:
            hand_over(blob_dir, harness_user)

        def _run(index: int) -> dict[str, Any]:
            variant, blob_dir, input_specs = runs[index]
            rundir = Path(tmp) / f"run-{index}"
            # Each ``# @output_dir`` is a fresh folder; nothing a test writes is stored.
            folders = output_dirs.mount_entries(source, rundir)
            raw = run_cell_tests_in_dir(
                rundir=rundir,
                venv_python=venv_python,
                cell_source=source,
                test_source=test_source,
                inputs=input_specs,
                input_dir=blob_dir,
                mounts={
                    **{name: str(rm.local_path) for name, rm in resolved_mounts.items()},
                    **{name: entry["local_path"] for name, entry in folders.items()},
                },
                tables=tables,
                notebook_dir=host.path,
                work_dir=WORK_DIR,
                env=identity_env(context.harness_env(runtime_env), harness_user),
                run_as=harness_user,
            )
            return raw if variant is None else _variant_test_run(raw, variant)

        def _run_all() -> dict[str, Any]:
            return _merged_test_runs([_run(index) for index in range(len(runs))])

        empty_raw: dict[str, Any] = {
            "passed": 0,
            "failed": 0,
            "errored": 0,
            "skipped": 0,
            "tests": [],
        }
        raw: dict[str, Any]
        try:
            raw = refused if refused is not None else await asyncio.to_thread(_run_all)
        except PytestUnavailableError:
            # Auto-provision pytest into the dev group and retry once. Dev tools are excluded
            # from the env hash, so cell caches survive. On failure, fall back to the actionable
            # pytest_unavailable flag.
            from strata.notebook.dependencies import ensure_dev_tool

            install = await asyncio.to_thread(ensure_dev_tool, host.path, "pytest")
            if install.success:
                auto_installed = ["pytest"]
                try:
                    raw = await asyncio.to_thread(_run_all)
                except PytestUnavailableError:
                    pytest_unavailable = True
                    raw = empty_raw
            else:
                logger.warning(
                    "Auto-install of pytest failed for notebook %s: %s",
                    host.path,
                    install.error,
                )
                pytest_unavailable = True
                raw = empty_raw

    # The env hash after the run: a pytest auto-install can change the lockfile.
    input_fingerprint = host.record_cell_test_inputs(
        cell_id,
        prov.input_hashes
        + prov.mount_fingerprints
        + prov.table_fingerprints
        + prov.fetch_fingerprints
        + prov.dataset_fingerprints,
        host.cell_env_hash(cell),
    )
    source_hash, test_source_hash, _ = host.cell_test_fingerprint(cell_id, source, test_source)
    result = CellTestResult(
        passed=raw["passed"],
        failed=raw["failed"],
        errored=raw["errored"],
        skipped=raw.get("skipped", 0),
        tests=[CellTestCase(**t) for t in raw["tests"]],
        cell_source_hash=source_hash,
        test_source_hash=test_source_hash,
        input_fingerprint=input_fingerprint,
        ran_at=int(time.time() * 1000),
        pytest_unavailable=pytest_unavailable,
        auto_installed=auto_installed,
    )

    persist_cell_test_result(host.path, cell_id, result.model_dump())
    cell.test_result = result
    cell.test_source = test_source
    if refused is None:
        # The tests read the cell's inputs onto this machine, as a run of it does.
        from strata.notebook import reads_index

        live = [c.id for c in host.notebook_state.cells]
        reads_index.record(host.path, cell_id, prov.reads(*context.moved(cell_id)), live)
    return result
