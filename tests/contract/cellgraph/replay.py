"""Replay a recorded cellgraph trace through cellgraph, with Strata's vocabulary.

For each step the notebook is rebuilt from the scenario's files and the step's
edits, with no session and no run, and analyzed by ``cellgraph.analyze`` under the
``legacy-1`` formats. Keys are computed from what the snapshot recorded: the
upstream outputs it published and the input components it resolved. Every
comparison returns a mismatch naming the scenario, the step, the cell and the
first part that differs.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cellgraph as cg

from strata.notebook import vocabulary
from tests.contract.cellgraph import record

# cellgraph's codes for a check Strata names after its own annotation.
CODE_IN_STRATA = {"seed_unknown_cell": "loop_start_from_unknown"}
# Structural errors are compared through ``dag_error``, not per cell.
STRUCTURAL = cg.analysis.STRUCTURAL_CODES

_RUN_OPS = {"run_all", "run_cell"}
_KIND_PREFIX = {"mount": "", "table": "table:", "fetch": "fetch:", "dataset": "dataset:"}
_KIND_RESOLVED = {"mount": "mounts", "table": "tables", "fetch": "fetches", "dataset": "datasets"}


@dataclass
class Mismatch:
    scenario: str
    step: int
    cell: str | None
    part: str
    expected: Any
    actual: Any

    def __str__(self) -> str:
        where = f"{self.scenario} steps[{self.step}]" + (f" cell {self.cell}" if self.cell else "")
        return (
            f"{where}: {self.part}\n"
            f"  recorded: {json.dumps(self.expected, default=str)[:400]}\n"
            f"  cellgraph: {json.dumps(self.actual, default=str)[:400]}"
        )


def notebooks(name: str, workdir: Path) -> Iterator[tuple[int, dict[str, Any], Any]]:
    """``(index, step, notebook state)`` for every step, rebuilt from files alone."""
    from strata.notebook.parser import parse_notebook

    source = record.SCENARIOS / name
    # The path the trace was recorded at: it is in mount URIs, as written.
    recorded_dir = str(record.ROOT / name)
    notebook_dir = workdir / name
    shutil.copytree(source, notebook_dir, ignore=shutil.ignore_patterns("steps.json"))
    for path in notebook_dir.rglob("*"):
        if path.suffix in record._TEXT_SUFFIXES and path.is_file():
            path.write_text(path.read_text().replace(record.PLACEHOLDER, recorded_dir))
    steps = json.loads(
        (source / "steps.json").read_text().replace(record.PLACEHOLDER, recorded_dir)
    )
    for index, step in enumerate(steps):
        record.apply_file_op(step, notebook_dir)
        yield index, step, parse_notebook(notebook_dir)


def analyze(state: Any) -> cg.Analysis:
    """``cellgraph.analyze`` over a parsed notebook, as Strata's facade will call it."""
    cells = [
        cg.Cell(
            id=cell.id,
            source=cell.source,
            language=cell.language.value,
            meta={
                "language": cell.language.value,
                "mounts": list(cell.mounts),
                "notebook_mounts": list(state.mounts),
                "connections": list(state.connections),
            },
        )
        for cell in state.cells
    ]
    groups = cg.GroupConfig(
        modes=dict(state.variant_modes), active=dict(state.variant_active_selections)
    )
    return cg.analyze(
        cells,
        vocabulary.strata_vocabulary(),
        groups,
        normalization=cg.NORMALIZE_LEGACY_1,
        digest=cg.DIGEST_LEGACY_1,
    )


# --- what the snapshot recorded, as cellgraph's host inputs -------------------------


class _Ledger:
    """Provenance hashes the trace names, by artifact, across the steps so far.

    A cell-level ``artifact_uri`` is recorded without its hash. The executor files a
    variable's artifact under ``derive_subkey(<run key>, <var>)``, so the hash is
    derived from the key of the run that published that version.
    """

    def __init__(self) -> None:
        self.hashes: dict[str, str] = {}
        self.published_by: dict[str, set[str]] = {}

    def learn(self, snapshot: dict[str, Any]) -> None:
        for cell_id, cell in snapshot["cells"].items():
            published = cell["published"]
            for entry in [*published["artifact_uris"].values(), *published["instances"].values()]:
                if entry is not None:
                    self.hashes[entry["artifact"]] = entry["provenance_hash"]
            run = cell.get("run") or {}
            uri = published["artifact_uri"]
            if uri and run.get("provenance_hash"):
                artifact = uri.rsplit("/", 1)[-1]
                if artifact not in self.hashes:
                    var = artifact.split("_var_", 1)[1].split("@", 1)[0]
                    self.hashes[artifact] = cg.derive_subkey(run["provenance_hash"], var)
            if run.get("provenance_hash"):
                self.published_by.setdefault(cell_id, set()).add(run["provenance_hash"])


def _ref(entry: dict[str, Any] | None) -> cg.OutputRef | cg.Unavailable:
    if entry is None:
        return cg.Unavailable("the artifact is not in the store")
    return cg.OutputRef(cg.Provenance(entry["provenance_hash"]), entry["content_sha256"])


def published_from(snapshot: dict[str, Any], ledger: _Ledger):
    def published(cell_id: str) -> cg.Published:
        recorded = snapshot["cells"][cell_id]["published"]
        cell_ref = None
        if recorded["artifact_uri"]:
            artifact = recorded["artifact_uri"].rsplit("/", 1)[-1]
            known = ledger.hashes.get(artifact)
            cell_ref = (
                cg.OutputRef(cg.Provenance(known))
                if known is not None
                else cg.Unavailable("no recorded hash")
            )
        return cg.Published(
            outputs={var: _ref(entry) for var, entry in recorded["artifact_uris"].items()},
            instances={
                tuple(key.split("@variant=", 1)): _ref(entry)
                for key, entry in recorded["instances"].items()
            },
            cell_ref=cell_ref,
        )

    return published


class _Resolver:
    """Each input's component from the snapshot's ``resolved`` lists, by name."""

    def __init__(self, recorded: dict[str, Any]):
        self.recorded = recorded
        self.used: set[str] = set()

    def __call__(self, spec: cg.Input) -> cg.Resolved | cg.Unavailable:
        prefix = f"{spec.name}:{_KIND_PREFIX[spec.kind]}"
        found = [c for c in self.recorded[_KIND_RESOLVED[spec.kind]] if c.startswith(prefix)]
        if len(found) != 1:
            return cg.Unavailable(f"{len(found)} recorded components for {spec.kind} {spec.name}")
        self.used.add(found[0])
        return cg.Resolved(identity=None, fingerprint=None, legacy=found[0])

    def unused(self) -> list[str]:
        everything = [c for kind in _KIND_RESOLVED.values() for c in self.recorded[kind]]
        return [c for c in everything if c not in self.used]


# --- the comparisons ---------------------------------------------------------------


def _error_codes(recorded: dict[str, Any], cell_id: str) -> set[str]:
    diagnostics = recorded["cells"][cell_id]["analysis"]["diagnostics"]
    return {code for code, severity, _line in diagnostics if severity == "error"}


def _strata_blocks(recorded: dict[str, Any], cell_id: str) -> bool:
    """Whether Strata would refuse *cell_id*: no DAG, or an error on it or upstream."""
    if recorded["dag_error"] is not None:
        return True
    stack, seen = [cell_id], set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        if _error_codes(recorded, current):
            return True
        stack.extend(recorded["cells"][current]["analysis"]["upstream_ids"])
    return False


def _refusals(recorded: dict[str, Any]) -> dict[str, bool]:
    """Per cell a run step ran: True if refused before it started, False if it ran.

    A refusal names one of the cell's own error diagnostics (``error_code``; a fetch
    that fails mid-run records a code too, not a diagnostic), and a cell skipped
    because a refused upstream did not run is refused too. A cell that ran has a provenance
    hash. Any other failure (an error with no code: a run that failed, or a single
    cell run whose upstream was refused) says nothing either way, and is left out.
    """
    runs = {cid: c["run"] for cid, c in recorded["cells"].items() if c.get("run")}
    upstream: dict[str, set[str]] = {}
    for source, to, _var in (recorded["dag"] or {}).get("edges", []):
        upstream.setdefault(to, set()).add(source)
    refused = {
        cid
        for cid, run in runs.items()
        if run["status"] == "error" and run.get("error_code") in _error_codes(recorded, cid)
    }

    def after_refusal(cell_id: str) -> bool:
        stack, seen = [cell_id], set()
        while stack:
            for parent in upstream.get(stack.pop(), ()):
                if parent in refused:
                    return True
                if parent not in seen:
                    seen.add(parent)
                    stack.append(parent)
        return False

    found: dict[str, bool] = {}
    for cell_id, run in runs.items():
        if cell_id in refused or (run["status"] == "skipped" and after_refusal(cell_id)):
            found[cell_id] = True
        elif "provenance_hash" in run:
            found[cell_id] = False
    return found


def _dag(analysis: cg.Analysis) -> dict[str, Any]:
    graph = analysis.graph
    return {
        "edges": [[e.from_cell_id, e.to_cell_id, e.variable] for e in graph.edges],
        "topological_order": list(graph.topological_order),
        "variant_groups": [
            {
                "group": g.group,
                "mode": g.mode,
                "members": [list(m) for m in g.members],
                "active": g.active_cell_id,
            }
            for g in graph.variant_groups
        ],
        "inactive_cells": sorted(graph.inactive_cells),
        "fanout_cells": {cid: p.group for cid, p in sorted(graph.fanout_cells.items())},
        "consumed_variables": {
            cid: sorted(names) for cid, names in sorted(graph.consumed_variables.items())
        },
    }


def _first_difference(expected: Any, actual: Any, path: str) -> str | None:
    return record.first_difference(expected, actual, path)


def _expected_plan(recorded: dict[str, Any], target: str) -> list[str]:
    upstream: dict[str, set[str]] = {}
    for source, to, _var in recorded["dag"]["edges"]:
        upstream.setdefault(to, set()).add(source)
    seen = {target}
    stack = [target]
    while stack:
        for parent in upstream.get(stack.pop(), ()):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    status = {cid: cell["staleness"]["status"] for cid, cell in recorded["cells"].items()}
    steps = [
        cid
        for cid in recorded["dag"]["topological_order"]
        if cid in seen and cid != target and status[cid] != "ready"
    ]
    return [*steps, target]


@dataclass
class Report:
    mismatches: list[Mismatch]
    # (step, cell): recorded keys of a cell cellgraph does not plan, so it gives none.
    unplannable_with_keys: list[tuple[int, str]]
    keyed: int
    host_keyed: int


def replay(name: str, workdir: Path) -> Report:
    trace = json.loads((record.TRACES / f"{name}.json").read_text())
    mismatches: list[Mismatch] = []
    unplannable_with_keys: list[tuple[int, str]] = []
    keyed = host_keyed = 0
    ledger = _Ledger()

    def miss(step: int, cell: str | None, part: str, expected: Any, actual: Any) -> None:
        mismatches.append(Mismatch(name, step, cell, part, expected, actual))

    for (index, step, state), recorded in zip(
        notebooks(name, workdir), trace["steps"], strict=True
    ):
        assert step == recorded["step"]
        analysis = analyze(state)
        previously_published = {cid: set(hs) for cid, hs in ledger.published_by.items()}
        ledger.learn(recorded)
        refusals = _refusals(recorded)

        # Structure.
        structural = analysis.structural_errors
        if recorded["dag_error"] is not None:
            messages = sorted({d.message for d in structural})
            if recorded["dag_error"] not in messages:
                miss(index, None, "dag_error", recorded["dag_error"], messages)
        elif structural:
            miss(index, None, "dag_error", None, sorted({d.message for d in structural}))
        if recorded["dag"] is not None:
            actual_dag = _dag(analysis)
            path = _first_difference(recorded["dag"], actual_dag, "dag")
            if path is not None:
                miss(
                    index,
                    None,
                    path,
                    record._lookup(recorded, path),
                    record._lookup({"dag": actual_dag}, path),
                )

        for cell_id, expected in recorded["cells"].items():
            ca = analysis.cells[cell_id]
            names = {
                "defines": ca.defines,
                "references": ca.references,
                "builtin_references": ca.builtin_references,
            }
            if recorded["dag"] is not None:
                names["upstream_ids"] = analysis.graph.cell_upstream.get(cell_id, [])
            for part, actual in names.items():
                if list(actual) != expected["analysis"][part]:
                    miss(index, cell_id, f"analysis.{part}", expected["analysis"][part], actual)

            ours = sorted(
                [CODE_IN_STRATA.get(d.code, d.code), d.severity.value, d.line]
                for d in ca.diagnostics
                if d.code not in STRUCTURAL
            )
            theirs = sorted(expected["analysis"]["diagnostics"])
            if ours != theirs:
                miss(index, cell_id, "analysis.diagnostics", theirs, ours)

            refused = refusals.get(cell_id)
            if refused is not None and refused == ca.plannable:
                miss(index, cell_id, "plannable", not refused, ca.plannable)

            if ca.plannable and recorded["dag"] is not None:
                status = {cid: c["staleness"]["status"] for cid, c in recorded["cells"].items()}
                plan = analysis.plan(cell_id, ready=lambda cid: status[cid] == "ready")
                wanted = _expected_plan(recorded, cell_id)
                if plan != wanted:
                    miss(index, cell_id, "plan", wanted, plan)

            if "keys" not in expected:
                continue
            if not ca.plannable:
                # Strata keys a cell it will not run; cellgraph does not key it at all.
                if not _strata_blocks(recorded, cell_id):
                    miss(index, cell_id, "plannable", True, False)
                unplannable_with_keys.append((index, cell_id))
                continue
            keys = expected["keys"]
            if ca.definition_key != keys["definition_key"]:
                miss(
                    index, cell_id, "keys.definition_key", keys["definition_key"], ca.definition_key
                )
            resolver = _Resolver(expected["resolved"])
            resolve = dict.fromkeys(_KIND_RESOLVED, resolver)
            published = published_from(recorded, ledger)
            result = analysis.execution_key(
                cell_id, published=published, resolve=resolve, env_hash=keys["env_hash"]
            )
            if isinstance(result, cg.HostKeyed):
                host_keyed += 1
                if ca.language not in ("sql", "prompt"):
                    miss(index, cell_id, "keys", "keyed", "host-keyed")
                continue
            if ca.language in ("sql", "prompt"):
                miss(index, cell_id, "keys", "host-keyed", type(result).__name__)
                continue
            keyed += 1
            _compare_key(miss, index, cell_id, "keys", keys, result)
            if resolver.unused():
                miss(index, cell_id, "keys.resolved", resolver.unused(), "no input reads them")
            for instance, instance_keys in sorted(keys.get("instances", {}).items()):
                instance_result = analysis.execution_key(
                    cell_id,
                    instance=instance,
                    published=published,
                    resolve=resolve,
                    env_hash=keys["env_hash"],
                )
                _compare_key(
                    miss,
                    index,
                    cell_id,
                    f"keys.instances.{instance}",
                    instance_keys,
                    instance_result,
                )

            run = expected.get("run")
            if (
                step["op"] in _RUN_OPS
                and run
                and run["status"] == "ok"
                and isinstance(result, cg.Keyed)
            ):
                seen_before = previously_published.get(cell_id, set())
                if run["cache_hit"] and result.digest not in seen_before:
                    miss(index, cell_id, "cache decision", "hit", "a key never published before")
                if not run["cache_hit"] and result.digest in seen_before and not ca.nocache:
                    miss(index, cell_id, "cache decision", "miss", "a key published before")

    return Report(mismatches, unplannable_with_keys, keyed, host_keyed)


def _compare_key(miss, index, cell_id, part, recorded_keys, result) -> None:
    if not isinstance(result, cg.Keyed):
        miss(
            index,
            cell_id,
            part,
            recorded_keys["execution_key"],
            f"{type(result).__name__}: {getattr(result, 'reason', '')}",
        )
        return
    if result.digest == recorded_keys["execution_key"]:
        return
    components = sorted(
        [value for _, value in result.parts["upstream"]] + [c for _, _, c in result.parts["inputs"]]
    )
    path = _first_difference(recorded_keys["input_hashes"], components, f"{part}.input_hashes")
    if path is not None:
        miss(index, cell_id, path, recorded_keys["input_hashes"], components)
    elif result.parts["definition_key"] != recorded_keys.get(
        "source_hash", result.parts["definition_key"]
    ):
        miss(
            index,
            cell_id,
            f"{part}.source_hash",
            recorded_keys["source_hash"],
            result.parts["definition_key"],
        )
    else:
        miss(index, cell_id, f"{part}.execution_key", recorded_keys["execution_key"], result.digest)
