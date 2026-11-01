"""Record what Strata does, step by step, on each cellgraph scenario.

A scenario is a notebook directory plus ``steps.json``. The recorder copies it to a
work directory, applies each step through the code paths Strata itself uses (the cell
writers, ``strata run``, ``LocalNotebookOps.run_cell``), and after every step opens the
notebook in a fresh session and records the graph, each cell's keys and their parts,
what each external input resolved to, what each cell published and its staleness.

    uv run python -m tests.contract.cellgraph.record            # check every trace
    uv run python -m tests.contract.cellgraph.record --update   # rewrite the traces

A ``file://`` mount's URI is in its key as written, so every scenario is recorded at
the same path, :data:`ROOT`/<name>, on every machine; scenario files write it as
``@NB@``. Every step runs with the notebook directory as the working directory, where
``StrataConfig`` finds the scenario's ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import dataclasses
import hashlib
import http.server
import io
import itertools
import json
import os
import shlex
import shutil
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

HERE = Path(__file__).resolve().parent
SCENARIOS = HERE / "scenarios"
TRACES = HERE / "traces"
TRACE_FORMAT = 1

# Design 13.1 replaces the causality map, so the exact gate leaves it out.
OUTSIDE_EXACT_GATE = "causality"

# Not a temporary directory: its path differs by machine, and it would enter the keys.
ROOT = Path("/tmp/strata-cellgraph")
# Scenario text files name the notebook directory as this.
PLACEHOLDER = "@NB@"
_TEXT_SUFFIXES = {".py", ".r", ".R", ".sql", ".prompt", ".toml"}
_FIXED_MTIME_NS = 1_700_000_000 * 10**9
# The fetch scenario's URLs are in its keys, so its server listens on a fixed port.
FETCH_PORT = 47613
_SKIP_DIRS = {".strata", ".venv", "work"}

# What a scenario needs from the machine beyond the dev environment.
REQUIRES = {"r_directives": "Rscript"}


def scenario_names() -> list[str]:
    return sorted(p.parent.name for p in SCENARIOS.glob("*/steps.json"))


def missing_requirement(name: str) -> str | None:
    """Why *name* cannot be recorded on this machine, or None."""
    needed = REQUIRES.get(name)
    if needed is not None and shutil.which(needed) is None:
        return f"{needed} is not on PATH"
    return None


# --- environment -------------------------------------------------------------------


def _prepared_venv(notebook_dir: Path) -> None:
    """A ``.venv`` whose interpreter is this one, as ``--no-sync`` takes it."""
    python = notebook_dir / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)


def _pin_mtimes(notebook_dir: Path) -> None:
    """Give every file one mtime, so a ``file://`` listing depends on names and sizes."""
    for root, dirs, files in os.walk(notebook_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        for name in files:
            os.utime(Path(root) / name, ns=(_FIXED_MTIME_NS, _FIXED_MTIME_NS))


class _Counter:
    """Deterministic stand-ins for what Strata and pyiceberg draw at random."""

    def __init__(self, scenario: str) -> None:
        self._scenario = scenario
        self._nonces = itertools.count()
        self._snapshots = itertools.count(1)

    def nonce(self) -> str:
        n = next(self._nonces)
        return hashlib.sha256(f"cellgraph:{self._scenario}:nonce:{n}".encode()).hexdigest()

    def snapshot_id(self) -> int:
        return 1000 + next(self._snapshots)


class _Spy:
    """Records the SQL and prompt keys their executors compute, calling through."""

    def __init__(self) -> None:
        self.cell: str | None = None
        self.calls: dict[str, list[dict[str, Any]]] = {}

    def entry(self, real):
        spy = self

        async def wrapper(session, cell_id, *args, **kwargs):
            spy.cell = cell_id
            try:
                return await real(session, cell_id, *args, **kwargs)
            finally:
                spy.cell = None

        return wrapper

    def key(self, name: str, real):
        spy = self

        def wrapper(**kwargs):
            result = real(**kwargs)
            if spy.cell is not None:
                spy.calls.setdefault(spy.cell, []).append(
                    {"function": name, "inputs": _plain(kwargs), "key": result}
                )
            return result

        return wrapper


def _plain(value: Any) -> Any:
    """*value* as JSON: bytes as base64, dataclasses as dicts, sequences as lists."""
    if isinstance(value, bytes):
        return {"base64": base64.b64encode(value).decode("ascii")}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    return value


class _FetchServer:
    """Serves a scenario's ``serve/`` directory on :data:`FETCH_PORT`."""

    def __init__(self, root: Path) -> None:
        directory = str(root)

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=directory, **kwargs)

            def log_message(self, *args):
                return None

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", FETCH_PORT), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@contextlib.contextmanager
def _recording(scenario: str, notebook_dir: Path, registry: Path) -> Iterator[_Spy]:
    """The process state every step of one scenario runs in."""
    counter = _Counter(scenario)
    spy = _Spy()
    saved_env = dict(os.environ)
    saved_cwd = Path.cwd()
    for key in list(os.environ):
        if key.startswith("STRATA_") or key.lower().endswith("_proxy"):
            del os.environ[key]
    os.environ.update(
        {
            # @dataset reads this server's store; never the user's ~/.strata.
            "STRATA_ARTIFACT_DIR": str(registry),
            "STRATA_DEPLOYMENT_MODE": "personal",
            "STRATA_NOTEBOOK_WARM_POOL_SIZE": "0",
            "STRATA_LOG_LEVEL": "ERROR",
        }
    )

    from strata.notebook import cli, prompt_executor
    from strata.notebook.executor import CellExecutor
    from strata.notebook.sql import cell_executor as sql_cell_executor

    def _harness_command(self, manifest_path: Path, venv_python: Path, harness_user):
        # As the harness tests start it: this interpreter, not ``uv run``.
        return [str(venv_python), str(self.harness_path), str(manifest_path)]

    patches = [
        mock.patch("strata.notebook.nonce.unresolved_nonce", counter.nonce),
        mock.patch("pyiceberg.table.metadata._generate_snapshot_id", counter.snapshot_id),
        mock.patch.object(CellExecutor, "_harness_command", _harness_command),
        mock.patch.object(
            sql_cell_executor,
            "compute_sql_provenance_hash",
            spy.key("compute_sql_provenance_hash", sql_cell_executor.compute_sql_provenance_hash),
        ),
        mock.patch.object(
            prompt_executor,
            "compute_prompt_provenance_hash",
            spy.key(
                "compute_prompt_provenance_hash", prompt_executor.compute_prompt_provenance_hash
            ),
        ),
        mock.patch.object(
            prompt_executor,
            "execute_prompt_cell",
            spy.entry(prompt_executor.execute_prompt_cell),
        ),
        mock.patch.object(
            sql_cell_executor, "execute_sql_cell", spy.entry(sql_cell_executor.execute_sql_cell)
        ),
        mock.patch.object(cli, "_quiet_notebook_logs", contextlib.nullcontext),
    ]
    server = _FetchServer(notebook_dir / "serve") if (notebook_dir / "serve").is_dir() else None
    os.chdir(notebook_dir)
    try:
        with contextlib.ExitStack() as stack:
            for patch in patches:
                stack.enter_context(patch)
            stack.enter_context(_llm_stub())
            yield spy
    finally:
        os.chdir(saved_cwd)
        os.environ.clear()
        os.environ.update(saved_env)
        if server is not None:
            server.close()


@contextlib.contextmanager
def _llm_stub() -> Iterator[None]:
    """The prompt cells' model: echoes the model name and the rendered prompt."""
    from strata.notebook import prompt_executor
    from strata.notebook.llm import LlmCompletionResult

    async def chat_completion(config, messages, **kwargs):
        prompt = messages[-1]["content"] if messages else ""
        return LlmCompletionResult(
            content=f"[{config.model}] {prompt}",
            input_tokens=1,
            output_tokens=1,
            model=config.model,
        )

    async def chat_completion_stream(config, messages, **kwargs):
        result = await chat_completion(config, messages, **kwargs)
        yield {"type": "done", "result": result}

    with (
        mock.patch.object(prompt_executor, "chat_completion", chat_completion),
        mock.patch.object(prompt_executor, "chat_completion_stream", chat_completion_stream),
    ):
        yield


# --- steps -------------------------------------------------------------------------


def _set_dotted(data: dict[str, Any], dotted: str, value: Any) -> bool:
    *parents, leaf = dotted.split(".")
    node = data
    for part in parents:
        node = node.setdefault(part, {})
    if value is None:
        return node.pop(leaf, None) is not None
    if node.get(leaf) == value:
        return False
    node[leaf] = value
    return True


def _iceberg_catalog(notebook_dir: Path, name: str):
    import tomllib

    from pyiceberg.catalog.sql import SqlCatalog

    with open(notebook_dir / "pyproject.toml", "rb") as f:
        properties = tomllib.load(f)["tool"]["strata"]["catalogs"][name]
    return SqlCatalog(name, **{k: v for k, v in properties.items() if k != "type"})


def _rows(rows: list[int]):
    import pyarrow as pa

    return pa.table({"id": pa.array(rows, type=pa.int64())})


def _seed_dataset(registry: Path, name: str, value: Any) -> None:
    """A named version of *value* in the registry ``@dataset`` reads."""
    from vestigo.provenance import TransformSpec
    from vestigo.store import get_artifact_store

    store = get_artifact_store(registry)
    blob = json.dumps(value).encode()
    artifact_id = name.replace("/", "-")
    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=hashlib.sha256(blob + artifact_id.encode()).hexdigest(),
        transform_spec=TransformSpec(
            executor="seed@v1", params={"content_type": "json/object"}, inputs=[]
        ),
    )
    store.write_blob(artifact_id, version, blob)
    store.finalize_artifact(
        artifact_id,
        version,
        schema_json="{}",
        row_count=0,
        byte_size=len(blob),
        content_sha256=hashlib.sha256(blob).hexdigest(),
    )
    store.set_name(name, artifact_id, version)


async def _run_all(notebook_dir: Path) -> tuple[dict[str, Any], Any]:
    """``strata run --no-sync --format json``: its report, and the session it ran in."""
    from strata.notebook import cli
    from strata.notebook.executor import CellExecutor

    sessions = []
    real_init = CellExecutor.__init__

    def init(self, session, *args, **kwargs):
        sessions.append(session)
        real_init(self, session, *args, **kwargs)

    args = argparse.Namespace(
        path=str(notebook_dir), force=False, no_sync=True, timeout=None, format="json", quiet=True
    )
    out = io.StringIO()
    with contextlib.redirect_stdout(out), mock.patch.object(CellExecutor, "__init__", init):
        await cli._run_async(args)
    report = json.loads(out.getvalue())
    results = {
        entry["id"]: {
            key: entry[key]
            for key in ("status", "cache_hit", "error_code", "reason", "provenance_hash", "stdout")
            if entry.get(key) is not None
        }
        for entry in report["cells"]
    }
    return results, sessions[0] if sessions else None


async def _run_cell(notebook_dir: Path, cell_id: str) -> tuple[dict[str, Any], Any]:
    """``LocalNotebookOps.run_cell`` (``strata cell run``): its result, and its session."""
    from strata.notebook.ops import LocalNotebookOps

    ops = LocalNotebookOps(notebook_dir)
    try:
        ops.use_existing_environment()
        result = await ops.run_cell(cell_id)
    finally:
        await ops.aclose()
    entry = {"status": result.status, "cache_hit": result.cache_hit}
    if result.error_code:
        entry["error_code"] = result.error_code
    if result.stdout:
        entry["stdout"] = result.stdout
    return {cell_id: entry}, ops._session


def _apply(
    step: dict[str, Any], notebook_dir: Path, registry: Path
) -> tuple[dict[str, Any], Any] | None:
    """Apply one step; a run step returns its results and the session it ran in."""
    from strata.notebook import writer

    op = step["op"]
    if op == "validate":
        return None
    if op == "edit_cell":
        writer.write_cell(notebook_dir, step["cell"], step["source"])
    elif op == "add_cell":
        writer.add_cell_to_notebook(
            notebook_dir, step["cell"], step.get("after"), language=step.get("language", "python")
        )
        writer.write_cell(notebook_dir, step["cell"], step["source"])
    elif op == "remove_cell":
        writer.remove_cell_from_notebook(notebook_dir, step["cell"])
    elif op == "write_file":
        path = notebook_dir / step["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(step["text"])
    elif op == "remove_file":
        (notebook_dir / step["path"]).unlink()
    elif op == "toml_set":
        writer._apply_notebook_toml_update(
            notebook_dir, lambda data: _set_dotted(data, step["key"], step.get("value"))
        )
    elif op == "set_variant_active":
        writer.set_variant_active(notebook_dir, step["group"], step["name"])
    elif op == "set_variant_mode":
        writer.set_variant_mode(notebook_dir, step["group"], step["mode"])
    elif op == "set_env":
        if step.get("value") is None:
            os.environ.pop(step["name"], None)
        else:
            os.environ[step["name"]] = step["value"]
    elif op == "iceberg_create":
        catalog = _iceberg_catalog(notebook_dir, step["catalog"])
        namespace, _ = step["table"].split(".")
        catalog.create_namespace_if_not_exists(namespace)
        table = catalog.create_table(step["table"], _rows([]).schema)
        if step.get("rows"):
            table.append(_rows(step["rows"]))
    elif op == "iceberg_append":
        catalog = _iceberg_catalog(notebook_dir, step["catalog"])
        catalog.load_table(step["table"]).append(_rows(step["rows"]))
    elif op == "seed_dataset":
        _seed_dataset(registry, step["name"], step["value"])
    elif op == "run_all":
        return asyncio.run(_run_all(notebook_dir))
    elif op == "run_cell":
        return asyncio.run(_run_cell(notebook_dir, step["cell"]))
    else:
        raise ValueError(f"unknown step op {op!r}")
    return None


# --- snapshot ----------------------------------------------------------------------


def _artifact(store: Any, artifact_id: str, version: int) -> dict[str, Any] | None:
    record = store.get_artifact(artifact_id, version)
    if record is None:
        return None
    return {
        "artifact": f"{artifact_id}@v={version}",
        "provenance_hash": record.provenance_hash,
        "content_sha256": record.content_sha256,
    }


def _published(session: Any, cell: Any, variants: list[str]) -> dict[str, Any]:
    manager = session.get_artifact_manager()
    store = manager.artifact_store
    outputs = {}
    for var, uri in sorted(cell.artifact_uris.items()):
        artifact_id, version = session._parse_artifact_uri(uri)
        outputs[var] = _artifact(store, artifact_id, version)
    instances = {}
    for var in sorted(set(cell.defines) | set(cell.artifact_uris)):
        for variant in variants:
            latest = store.get_latest_version(
                manager.cell_artifact_id(cell.id, var, variant=variant)
            )
            if latest is not None:
                instances[f"{var}@variant={variant}"] = _artifact(store, latest.id, latest.version)
    return {"artifact_uris": outputs, "artifact_uri": cell.artifact_uri, "instances": instances}


def _keys(session: Any, cell: Any, variants: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """The cell's keys and their parts, and what its external inputs resolved to.

    Each part comes from the call ``compute_staleness`` makes for it.
    """
    from vestigo.provenance import derive_subkey

    from strata.notebook.provenance import (
        cell_provenance_inputs,
        compute_provenance_hash,
        compute_source_hash,
    )

    source_hash = compute_source_hash(cell.source)
    env_hash = session.cell_env_hash(cell)
    outside = session._outside_world_for(cell)
    external = [
        outside.mount_fingerprints,
        outside.table_fingerprints,
        outside.fetch_fingerprints,
        outside.dataset_fingerprints,
    ]
    resolved = {
        "mounts": outside.mount_fingerprints,
        "has_rw_mount": outside.has_rw_mount,
        "tables": outside.table_fingerprints,
        "fetches": outside.fetch_fingerprints,
        "datasets": outside.dataset_fingerprints,
    }

    def key(upstream: list[str]) -> dict[str, Any]:
        inputs = cell_provenance_inputs(upstream, *external)
        return {
            # As assembled, but sorted: their order follows set iteration, which varies
            # by process, and the key sorts them anyway.
            "upstream_hashes": sorted(upstream),
            "input_hashes": sorted(inputs),
            "execution_key": compute_provenance_hash(inputs, source_hash, env_hash),
        }

    keys: dict[str, Any] = {
        "definition_key": source_hash,
        "source_hash": source_hash,
        "env_hash": env_hash,
        **key(session._collect_input_hashes(cell.id)),
    }
    if variants:
        instances = {}
        for variant in variants:
            instance = key(session._collect_input_hashes(cell.id, variant=variant))
            instance["execution_key"] = derive_subkey(
                instance["execution_key"], f"variant={variant}"
            )
            instances[variant] = instance
        keys["instances"] = instances
    return keys, resolved


def _dag(session: Any) -> dict[str, Any] | None:
    dag = session.dag
    if dag is None:
        return None
    return {
        "edges": [[e.from_cell_id, e.to_cell_id, e.variable] for e in dag.edges],
        "topological_order": list(dag.topological_order),
        "variant_groups": [
            {
                "group": g.group,
                "mode": g.mode,
                "members": [list(m) for m in g.members],
                "active": g.active_cell_id,
            }
            for g in dag.variant_groups
        ],
        "inactive_cells": sorted(dag.inactive_cells),
        "fanout_cells": {cid: p.group for cid, p in sorted(dag.fanout_cells.items())},
        "consumed_variables": {
            cid: sorted(names) for cid, names in sorted(dag.consumed_variables.items())
        },
    }


def _snapshot(
    notebook_dir: Path, spy: _Spy, run: dict[str, Any] | None, session: Any
) -> dict[str, Any]:
    """The notebook as *session* sees it; a freshly opened session when it is None."""
    from strata.notebook.languages import get_language_executor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession

    if session is None:
        session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.compute_staleness()
    fanout = session.dag.fanout_cells if session.dag is not None else {}
    cells: dict[str, Any] = {}
    for cell in session.notebook_state.cells:
        variants = [name for name, _ in fanout[cell.id].variants] if cell.id in fanout else []
        entry: dict[str, Any] = {
            "language": cell.language.value,
            "analysis": {
                "defines": list(cell.defines),
                "references": list(cell.references),
                "builtin_references": list(cell.builtin_references),
                "upstream_ids": list(cell.upstream_ids),
                "diagnostics": [
                    [d.code, d.severity.value, d.line] for d in cell.annotation_diagnostics
                ],
            },
            "staleness": {
                "status": cell.status.value,
                "reasons": [r.value for r in (cell.staleness.reasons if cell.staleness else [])],
            },
            "published": _published(session, cell, variants),
        }
        if not get_language_executor(cell.language).skips_execution_provenance:
            entry["keys"], entry["resolved"] = _keys(session, cell, variants)
            if get_language_executor(cell.language).has_alternate_cache_scheme:
                entry["keys"]["host_keyed"] = spy.calls.get(cell.id) if run is not None else None
        if run is not None and cell.id in run:
            entry["run"] = run[cell.id]
        cells[cell.id] = entry
    return {
        "dag": _dag(session),
        "dag_error": session.dag_error,
        "cells": cells,
        OUTSIDE_EXACT_GATE: {
            cid: _plain(dataclasses.asdict(chain)) for cid, chain in session.causality_map.items()
        },
    }


# --- recording ---------------------------------------------------------------------


def record(name: str) -> dict[str, Any]:
    """Record scenario *name* in a fresh copy under :data:`ROOT`; return its trace."""
    source = SCENARIOS / name
    notebook_dir = ROOT / name
    registry = ROOT / f"{name}-registry"
    shutil.rmtree(notebook_dir, ignore_errors=True)
    shutil.rmtree(registry, ignore_errors=True)
    shutil.copytree(source, notebook_dir, ignore=shutil.ignore_patterns("steps.json"))
    for path in notebook_dir.rglob("*"):
        if path.suffix in _TEXT_SUFFIXES and path.is_file():
            path.write_text(path.read_text().replace(PLACEHOLDER, str(notebook_dir)))
    steps = json.loads((source / "steps.json").read_text().replace(PLACEHOLDER, str(notebook_dir)))
    _prepared_venv(notebook_dir)
    registry.mkdir(parents=True)

    recorded = []
    with _recording(name, notebook_dir, registry) as spy:
        for step in steps:
            spy.calls = {}
            ran = _apply(step, notebook_dir, registry)
            run, session = ran if ran is not None else (None, None)
            _pin_mtimes(notebook_dir)
            recorded.append({"step": step, **_snapshot(notebook_dir, spy, run, session)})
    trace = {"trace_format": TRACE_FORMAT, "scenario": name, "steps": recorded}
    _check_no_machine_paths(trace)
    shutil.rmtree(notebook_dir, ignore_errors=True)
    shutil.rmtree(registry, ignore_errors=True)
    return trace


def _check_no_machine_paths(trace: dict[str, Any]) -> None:
    """A path of this machine in a trace would make it differ on the next one."""
    text = json.dumps(trace)
    for path in {HERE.parents[2], Path.home(), Path(sys.executable).parent, ROOT.resolve()}:
        if str(path) != str(ROOT) and str(path) in text:
            raise AssertionError(f"the trace names {path}, a path of this machine")


def dumps(trace: dict[str, Any]) -> str:
    return json.dumps(trace, indent=1, sort_keys=True) + "\n"


def without_ungated(trace: Any) -> Any:
    """*trace* without the parts outside the exact gate."""
    return {
        **trace,
        "steps": [{k: v for k, v in s.items() if k != OUTSIDE_EXACT_GATE} for s in trace["steps"]],
    }


def first_difference(expected: Any, actual: Any, path: str = "") -> str | None:
    """The path of the first place *actual* differs from *expected*, e.g. ``steps[3].cells``."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            here = f"{path}.{key}" if path else key
            if key not in expected or key not in actual:
                return here
            found = first_difference(expected[key], actual[key], here)
            if found is not None:
                return found
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        for index, (e, a) in enumerate(zip(expected, actual, strict=False)):
            found = first_difference(e, a, f"{path}[{index}]")
            if found is not None:
                return found
        if len(expected) != len(actual):
            return f"{path}[{min(len(expected), len(actual))}]"
        return None
    return None if expected == actual else path


def _lookup(trace: Any, path: str) -> Any:
    node = trace
    for part in path.replace("[", ".[").split("."):
        if not part:
            continue
        if part.startswith("["):
            index = int(part[1:-1])
            node = node[index] if isinstance(node, list) and index < len(node) else "<absent>"
        else:
            node = node.get(part, "<absent>") if isinstance(node, dict) else "<absent>"
    return node


def describe_difference(expected: dict[str, Any], actual: dict[str, Any]) -> str | None:
    """The first gated difference, with both values; None when the traces match."""
    path = first_difference(without_ungated(expected), without_ungated(actual))
    if path is None:
        return None
    return (
        f"first difference at {path}\n"
        f"  recorded: {json.dumps(_lookup(expected, path))[:400]}\n"
        f"  now:      {json.dumps(_lookup(actual, path))[:400]}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="scenarios (default: all)")
    parser.add_argument("--update", action="store_true", help="rewrite traces/<name>.json")
    args = parser.parse_args(argv)
    failed = 0
    for name in args.names or scenario_names():
        reason = missing_requirement(name)
        if reason is not None:
            print(f"{name}: skipped ({reason})")
            continue
        trace = record(name)
        target = TRACES / f"{name}.json"
        if args.update:
            target.write_text(dumps(trace))
            print(f"{name}: written")
            continue
        if not target.exists():
            print(f"{name}: no recorded trace")
            failed += 1
            continue
        difference = describe_difference(json.loads(target.read_text()), trace)
        print(f"{name}: {'matches' if difference is None else difference}")
        failed += difference is not None
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
