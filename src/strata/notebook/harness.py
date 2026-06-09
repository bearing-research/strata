"""Harness script that runs inside the notebook subprocess.

This script receives a manifest JSON file (path as argv[1]), executes cell source,
captures stdout/stderr, and serializes outputs.

It runs in the notebook's venv, so it only has access to the notebook's dependencies.
It cannot ``import strata`` — instead it loads ``serializer.py`` from the same
directory via ``importlib.util``.
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# orjson writes ~3-10× faster than stdlib and serializes datetime,
# numpy scalars, Decimal, and UUID natively — exactly the types that
# previously truncated manifest.json mid-write under stdlib json. We
# bake it into every generated notebook pyproject.toml, so it's
# guaranteed to be importable in the venv this harness runs in.
import orjson


def _load_local_module(filename: str, module_name: str):
    """Load a sibling module by absolute file path."""
    module_path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_ser = _load_local_module("serializer.py", "_nb_serializer")
_immut = _load_local_module("immutability.py", "_nb_immutability")
_display = _load_local_module("display/runtime.py", "_nb_display_runtime")


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: str) -> dict:
    with open(manifest_path, "rb") as f:
        return orjson.loads(f.read())


# ---------------------------------------------------------------------------
# Input deserialization
# ---------------------------------------------------------------------------


def deserialize_inputs(manifest: dict) -> dict[str, Any]:
    """Deserialize input variables listed in the manifest."""
    output_dir = Path(manifest.get("output_dir", "/tmp/strata_output"))
    inputs = {}

    for var_name, spec in manifest.get("inputs", {}).items():
        content_type = spec.get("content_type", "")
        file_name = spec.get("file", "")
        if not file_name:
            print(f"Warning: no file path for input {var_name}", file=sys.stderr)
            continue

        full_path = output_dir / file_name
        if not full_path.exists():
            print(f"Warning: input file not found: {full_path}", file=sys.stderr)
            continue

        try:
            inputs[var_name] = _ser.deserialize_value(content_type, full_path)
        except _ser.StrataRArtifactError as e:
            # R-only payload from an upstream R cell. Re-raise with the
            # variable name attached so the cell fails loudly instead
            # of silently leaving `var_name` undefined and triggering
            # an unhelpful NameError further down.
            raise _ser.StrataRArtifactError(e.file_path, variable_name=var_name) from e
        except Exception as e:
            print(f"Error deserializing {var_name}: {e}", file=sys.stderr)

    return inputs


def _exec_with_display(source: str, namespace: dict) -> Any | None:
    """Execute source; if the last statement is a bare expression, eval and return it."""
    import ast as _ast

    try:
        tree = _ast.parse(source)
    except SyntaxError:
        exec(source, namespace)  # noqa: S102
        return None

    if not tree.body:
        return None

    last = tree.body[-1]
    if isinstance(last, _ast.Expr):
        if len(tree.body) > 1:
            mod = _ast.Module(body=tree.body[:-1], type_ignores=[])
            _ast.fix_missing_locations(mod)
            exec(compile(mod, "<cell>", "exec"), namespace)  # noqa: S102
        expr = _ast.Expression(body=last.value)
        _ast.fix_missing_locations(expr)
        result = eval(compile(expr, "<cell>", "eval"), namespace)  # noqa: S307
        return result if result is not None else None
    else:
        exec(source, namespace)  # noqa: S102
        return None


def inject_mounts(manifest: dict, namespace: dict) -> None:
    """Inject mount paths as Path variables into the cell namespace.

    Each mount becomes a ``pathlib.Path`` bound to the mount's local path.
    Read-only mounts are verified to exist; read-write mounts are created
    if missing.
    """
    mounts = manifest.get("mounts", {})
    for mount_name, spec in mounts.items():
        local_path = Path(spec.get("local_path", ""))
        if local_path and local_path.exists():
            namespace[mount_name] = local_path
        elif spec.get("mode") == "rw":
            local_path.mkdir(parents=True, exist_ok=True)
            namespace[mount_name] = local_path
        else:
            print(
                f"Warning: mount '{mount_name}' path does not exist: {local_path}",
                file=sys.stderr,
            )


def inject_tables(manifest: dict, namespace: dict) -> None:
    """Inject lake table inputs into the cell namespace.

    Each ``@table`` declaration becomes two variables: ``<name>`` — the
    table URI string — and ``<name>_snapshot`` — the snapshot id the
    executor resolved at provenance time, so the cell can scan exactly the
    snapshot its provenance recorded.
    """
    tables = manifest.get("tables", {})
    for table_name, spec in tables.items():
        namespace[table_name] = spec.get("uri", "")
        namespace[f"{table_name}_snapshot"] = spec.get("snapshot_id")


def inject_client(manifest: dict, namespace: dict) -> Any:
    """Inject an ambient ``strata`` client bound to the server URL.

    Returns the client so the caller can close it after the cell runs —
    the warm pool reuses the process, so a leaked ``httpx.Client`` would
    accumulate sockets. Returns ``None`` when no ``strata_url`` is set.
    Must be called before the namespace is snapshotted for output capture
    so ``strata`` is treated as an injected input, not a cell output.
    """
    url = manifest.get("strata_url")
    if not url:
        return None
    from strata.client import StrataClient

    client = StrataClient(base_url=url)
    namespace["strata"] = client
    return client


def close_client(client: Any) -> None:
    """Close an injected ambient client, swallowing teardown errors."""
    if client is None:
        return
    try:
        client.close()
    except Exception:
        pass


@contextmanager
def apply_env_overrides(manifest: dict):
    """Apply manifest-scoped environment overrides for one execution."""
    overrides = {str(key): str(value) for key, value in manifest.get("env", {}).items()}
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def execute_cell(
    source: str,
    inputs: dict,
    mutation_defines: list[str] | None = None,
    loop_until_expr: str | None = None,
) -> tuple[dict, list[Any], str, str, list[dict], dict[str, Any] | None]:
    """Execute a cell and return its outputs, displays, captured streams, mutations, and loop state.

    ``mutation_defines`` lists variable names that the analyzer marked as
    in-place mutations (``df["col"] = ...``). These are always serialized
    as outputs even when the cell's execution preserved ``id()`` — the
    identity check alone would miss them and downstream cells would get
    the stale pre-mutation artifact.

    ``loop_until_expr``, when supplied, is a Python expression evaluated
    in the cell namespace after the body runs. The returned ``loop_state``
    dict carries ``until_reached`` (truthy result of the expression) and,
    on failure to compile/evaluate, an ``error`` field. When the caller
    passes ``None``, ``loop_state`` is ``None`` and no loop work happens.
    """
    namespace = dict(inputs)
    display_capture = _display.DisplayCapture()
    display_capture.install(namespace)
    mutation_set = set(mutation_defines or [])

    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        namespace_before = set(namespace.keys())
        input_identities = {name: id(namespace[name]) for name in namespace_before}
        input_snapshots = _immut.snapshot_inputs(namespace, list(namespace_before))

        with display_capture.capture_side_effects():
            _display_value = _exec_with_display(source, namespace)

        loop_state: dict[str, Any] | None = None
        if loop_until_expr is not None:
            loop_state = _eval_loop_until(loop_until_expr, namespace)

        _skip = {"__builtins__", "__name__", "__doc__", "__package__"}
        new_vars: dict[str, Any] = {}
        for name, value in namespace.items():
            if name.startswith("_") or name in _skip:
                continue
            if (
                name not in namespace_before
                or id(value) != input_identities.get(name)
                or name in mutation_set
            ):
                new_vars[name] = value

        display_values = display_capture.resolve(_display_value)

        mutation_warnings = list(_immut.detect_mutations(namespace, input_snapshots))
        return (
            new_vars,
            display_values,
            stdout_capture.getvalue(),
            stderr_capture.getvalue(),
            mutation_warnings,
            loop_state,
        )

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _eval_loop_until(expr: str, namespace: dict[str, Any]) -> dict[str, Any]:
    """Evaluate ``@loop_until`` in the cell namespace.

    Returns a dict with ``until_reached`` (bool) and, on any failure to
    compile or evaluate, an ``error`` field carrying a short message.
    Evaluation failures do not abort cell execution — they bubble up as
    a termination signal the executor can surface to the user.
    """
    try:
        code = compile(expr, "<loop_until>", "eval")
    except SyntaxError as exc:
        return {
            "until_reached": False,
            "error": f"@loop_until syntax error: {exc.msg}",
        }

    try:
        result = eval(code, namespace)  # noqa: S307 — predicate is declared by the user
    except Exception as exc:
        return {
            "until_reached": False,
            "error": f"@loop_until evaluation failed: {type(exc).__name__}: {exc}",
        }

    return {"until_reached": bool(result)}


# ---------------------------------------------------------------------------
# Batch execution (run-all single-process mode)
# ---------------------------------------------------------------------------
#
# Batched run-all exec's many cells in one Python process with a shared
# namespace. Parent owns all strata-side bookkeeping (provenance, cache,
# persist) and asks this harness to run cell bodies + serialize outputs.
# Communication uses two pipes:
#   - frame_out  — harness → parent (events + requests, line-delimited JSON)
#   - resp_in    — parent → harness (responses to cache_check / persist)
# Blob bytes travel as files in ``output_dir/<cell_id>/{var_name}{ext}``,
# never inline in the JSON. See issue #26 for full design.


_MISSING = object()


def _send_frame(stream: Any, frame_type: str, payload: dict) -> None:
    """Write one length-line JSON frame to the parent and flush."""
    line = orjson.dumps({"type": frame_type, "payload": payload}) + b"\n"
    stream.write(line)
    stream.flush()


def _recv_response(stream: Any) -> dict:
    """Read one JSON response line from the parent."""
    line = stream.readline()
    if not line:
        raise RuntimeError("Batch response pipe closed unexpectedly")
    return orjson.loads(line)


def _run_one_batched_cell(
    cell: dict,
    namespace: dict,
    output_dir: Path,
    frame_out: Any,
    resp_in: Any,
) -> tuple[str, str | None]:
    """Execute one cell within a batch. Returns (status, failed_reason).

    status ∈ {"ok", "cell_error", "persist_failed"}. failed_reason names
    the trigger when status != "ok" (matches batch_end's "reason" field).
    """
    cell_id = cell["cell_id"]
    source = cell["source"]
    consumed_vars: list[str] = list(cell.get("consumed_vars") or [])
    cell_env: dict = cell.get("env") or {}
    mount_manifest: dict = cell.get("mount_manifest") or {}
    table_manifest: dict = cell.get("table_manifest") or {}
    source_hash: str = cell.get("source_hash", "")
    env_hash: str = cell.get("env_hash", "")

    cell_output_dir = output_dir / cell_id
    cell_output_dir.mkdir(parents=True, exist_ok=True)

    _send_frame(frame_out, "cell_start", {"cell_id": cell_id})

    # Save existing namespace bindings under each mount name so we can
    # restore on exit — protects any pre-existing user variable with the
    # same name as a mount.
    mount_names = list(mount_manifest.keys())
    table_names = [injected for name in table_manifest for injected in (name, f"{name}_snapshot")]
    ambient_names = ["strata"] if cell.get("strata_url") else []
    previous_bindings: dict[str, Any] = {
        name: namespace.get(name, _MISSING) for name in mount_names + table_names + ambient_names
    }
    inject_mounts({"mounts": mount_manifest}, namespace)
    inject_tables({"tables": table_manifest}, namespace)
    ambient_client = inject_client(cell, namespace)

    try:
        with apply_env_overrides({"env": cell_env}):
            # Cache check — parent decides hit/miss.
            _send_frame(frame_out, "cache_check", {"cell_id": cell_id})
            response = _recv_response(resp_in)

            if response.get("cache_hit"):
                # Load cached outputs into namespace. Parent has already
                # materialized blobs into cell_output_dir before responding.
                cached_outputs: dict = response.get("cached_outputs") or {}
                for var_name, spec in cached_outputs.items():
                    content_type = spec.get("content_type", "")
                    file_name = spec.get("file", "")
                    if not content_type or not file_name:
                        continue
                    blob_path = cell_output_dir / file_name
                    if not blob_path.exists():
                        continue
                    try:
                        namespace[var_name] = _ser.deserialize_value(content_type, blob_path)
                    except Exception as exc:
                        print(
                            f"Warning: failed to load cached {var_name} for {cell_id}: {exc}",
                            file=sys.stderr,
                        )
                _send_frame(
                    frame_out,
                    "cell_output",
                    {
                        "cell_id": cell_id,
                        "cache_hit": True,
                        "outputs": cached_outputs,
                        "display_outputs": response.get("cached_displays") or [],
                    },
                )
                return ("ok", None)

            # Cache miss — execute the cell body.
            #
            # ``DisplayCapture.install`` uses ``setdefault`` (display/runtime.py
            # L56) so once a key exists in the namespace it sticks. Single-cell
            # mode gets away with this because each cell runs in a fresh
            # process. In batch mode, namespace persists across cells; clear
            # the per-cell display keys so the new capture's handlers actually
            # install.
            for _display_key in ("display", "Markdown"):
                namespace.pop(_display_key, None)
            display_capture = _display.DisplayCapture()
            display_capture.install(namespace)
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            try:
                try:
                    with display_capture.capture_side_effects():
                        display_value = _exec_with_display(source, namespace)
                except Exception as exc:
                    _send_frame(
                        frame_out,
                        "cell_error",
                        {
                            "cell_id": cell_id,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                            "stdout": stdout_capture.getvalue(),
                            "stderr": stderr_capture.getvalue(),
                        },
                    )
                    return ("cell_error", "cell_error")

                # Success: serialize consumed vars + displays.
                #
                # Known gap (see issue #26 round-7): instances of in-batch-
                # defined classes serialize as ``pickle/object`` rather than
                # ``module/cell-instance``. The single-cell flow tags the
                # class via the parent's ``_write_module_export_outputs``
                # *before* harness serialization. In batch we'd need to
                # either pre-classify here or post-rewrite parent-side.
                # Deferred to PR-b3 — affects content-type only; the
                # pickled value still round-trips correctly.
                outputs: dict[str, Any] = {}
                for var_name in consumed_vars:
                    if var_name not in namespace:
                        continue
                    try:
                        outputs[var_name] = _ser.serialize_value(
                            namespace[var_name], cell_output_dir, var_name
                        )
                    except Exception as exc:
                        outputs[var_name] = {
                            "error": str(exc),
                            "type": type(namespace[var_name]).__name__,
                        }

                display_values = display_capture.resolve(display_value)
                serialized_displays: list[dict[str, Any]] = []
                for idx, display in enumerate(display_values):
                    try:
                        # ``__display__N`` is the variable-name convention the
                        # serializer recognizes as display output for content-
                        # type detection (serializer.py ``_is_display_variable_name``).
                        # ``display_N`` would be classified as a regular pickle.
                        meta = _ser.serialize_value(display, cell_output_dir, f"__display__{idx}")
                        serialized_displays.append(meta)
                    except Exception:
                        # Display serialization errors don't abort the cell;
                        # matches single-cell behavior.
                        pass
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            # Persist request — parent writes to artifact store.
            _send_frame(
                frame_out,
                "persist",
                {
                    "cell_id": cell_id,
                    "outputs": outputs,
                    "display_outputs": serialized_displays,
                    "stdout": stdout_capture.getvalue(),
                    "stderr": stderr_capture.getvalue(),
                    "source_hash": source_hash,
                    "env_hash": env_hash,
                },
            )
            ack = _recv_response(resp_in)
            if not ack.get("ok"):
                return ("persist_failed", "persist_failed")
            return ("ok", None)
    finally:
        # Close the ambient client (the batch process is reused across
        # cells, so it must not leak sockets), then restore mount/table/
        # client name bindings — discard any cell-introduced ones, rebind
        # whatever was there before.
        close_client(ambient_client)
        for name, previous in previous_bindings.items():
            if previous is _MISSING:
                namespace.pop(name, None)
            else:
                namespace[name] = previous


def execute_batch(
    cells: list[dict],
    upstream_inputs: dict,
    output_dir: Path,
    frame_out: Any,
    resp_in: Any,
) -> None:
    """Execute a sequence of cells in one Python process.

    ``cells`` is the per-cell list ``[{cell_id, source, consumed_vars,
    env, mount_manifest, source_hash, env_hash}, ...]`` in notebook order.
    ``upstream_inputs`` seeds the shared namespace from artifacts of
    non-batched upstream cells (same ``{var: {content_type, file}}``
    shape ``deserialize_inputs`` reads).

    Streams frames over ``frame_out`` and reads responses from
    ``resp_in``. Returns after emitting ``batch_end``; the caller is
    responsible for closing the pipes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Seed namespace from upstream artifacts. Done inline (rather than
    # via deserialize_inputs) so a single R-only artifact doesn't
    # abort the whole subprocess pre-``cell_start`` — the parent would
    # then see ``subprocess_died`` instead of the actionable
    # StrataRArtifactError. Per-variable failures are stashed and
    # surfaced as a ``cell_error`` on the first cell that references
    # the tainted variable.
    namespace: dict[str, Any] = {}
    tainted_inputs: dict[str, _ser.StrataRArtifactError] = {}
    for var_name, spec in (upstream_inputs or {}).items():
        content_type = spec.get("content_type", "")
        file_name = spec.get("file", "")
        if not file_name:
            print(f"Warning: no file path for input {var_name}", file=sys.stderr)
            continue
        full_path = output_dir / file_name
        if not full_path.exists():
            print(f"Warning: input file not found: {full_path}", file=sys.stderr)
            continue
        try:
            namespace[var_name] = _ser.deserialize_value(content_type, full_path)
        except _ser.StrataRArtifactError as exc:
            tainted_inputs[var_name] = _ser.StrataRArtifactError(
                exc.file_path, variable_name=var_name
            )
        except Exception as exc:
            print(f"Error deserializing {var_name}: {exc}", file=sys.stderr)

    for cell in cells:
        blocker = _first_tainted_reference(cell["source"], tainted_inputs)
        if blocker is not None:
            _emit_tainted_cell_error(cell["cell_id"], blocker, frame_out)
            _send_frame(
                frame_out,
                "batch_end",
                {"reason": "cell_error", "failed_cell_id": cell["cell_id"]},
            )
            return

        status, reason = _run_one_batched_cell(cell, namespace, output_dir, frame_out, resp_in)
        if status != "ok":
            _send_frame(
                frame_out,
                "batch_end",
                {"reason": reason, "failed_cell_id": cell["cell_id"]},
            )
            return

    _send_frame(frame_out, "batch_end", {"reason": "complete"})


def _first_tainted_reference(
    source: str,
    tainted_inputs: dict[str, _ser.StrataRArtifactError],
) -> _ser.StrataRArtifactError | None:
    """Return the first tainted-upstream error that a cell's source references.

    Word-boundary match — keeps ``fit`` from spuriously triggering on
    ``unfit_data`` or string literals like ``"fit_score"``. AST-walk
    would be more precise, but a single regex pass over the source
    catches the cases the reviewer cares about (the consumer using
    the bare name as an identifier).
    """
    if not tainted_inputs:
        return None
    import re

    for var_name, exc in tainted_inputs.items():
        if re.search(rf"\b{re.escape(var_name)}\b", source):
            return exc
    return None


def _emit_tainted_cell_error(
    cell_id: str,
    exc: _ser.StrataRArtifactError,
    frame_out: Any,
) -> None:
    """Emit cell_start + cell_error for a cell blocked on a tainted upstream.

    Mirrors the frame shape ``_run_one_batched_cell`` would emit on a
    body-level exception, so consumers of the frame protocol don't
    need a separate branch for this case.
    """
    _send_frame(frame_out, "cell_start", {"cell_id": cell_id})
    _send_frame(
        frame_out,
        "cell_error",
        {
            "cell_id": cell_id,
            "error": str(exc),
            "traceback": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
            "stdout": "",
            "stderr": "",
        },
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def batch_main() -> None:
    """Batch-mode CLI entry. Invoked as ``python harness.py --batch <manifest>``.

    Reads the cell list and upstream inputs from the manifest. Pipe file
    descriptors come via ``STRATA_BATCH_FRAME_FD`` (write) and
    ``STRATA_BATCH_RESP_FD`` (read); output dir from
    ``STRATA_BATCH_OUTPUT_DIR``. Calls ``execute_batch`` and exits.
    """
    if len(sys.argv) < 3:
        print("Usage: harness.py --batch <manifest_path>", file=sys.stderr)
        sys.exit(1)

    manifest_path = sys.argv[2]
    manifest = load_manifest(manifest_path)

    frame_fd = int(os.environ["STRATA_BATCH_FRAME_FD"])
    resp_fd = int(os.environ["STRATA_BATCH_RESP_FD"])
    output_dir = Path(os.environ["STRATA_BATCH_OUTPUT_DIR"])

    # Open the inherited fds as buffered file objects. closefd=True so they
    # close when this process exits.
    frame_out = os.fdopen(frame_fd, "wb")
    resp_in = os.fdopen(resp_fd, "rb")

    execute_batch(
        cells=manifest.get("cells", []),
        upstream_inputs=manifest.get("upstream_inputs", {}),
        output_dir=output_dir,
        frame_out=frame_out,
        resp_in=resp_in,
    )

    # execute_batch returns after emitting batch_end; flush + close the
    # frame pipe so the parent's reader sees EOF promptly.
    frame_out.flush()
    frame_out.close()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        batch_main()
        return

    if len(sys.argv) < 2:
        print("Usage: harness.py <manifest_path>", file=sys.stderr)
        sys.exit(1)

    manifest_path = sys.argv[1]
    manifest: dict = {}
    stdout_text = ""
    stderr_text = ""
    ambient_client: Any = None

    try:
        manifest = load_manifest(manifest_path)
        source = manifest.get("source", "")
        output_dir = Path(manifest.get("output_dir", "/tmp/strata_output"))

        inputs = deserialize_inputs(manifest)
        inject_mounts(manifest, inputs)
        inject_tables(manifest, inputs)
        ambient_client = inject_client(manifest, inputs)
        loop_config = manifest.get("loop") or {}
        loop_until_expr = loop_config.get("until_expr") if isinstance(loop_config, dict) else None
        with apply_env_overrides(manifest):
            (
                outputs,
                display_values,
                stdout_text,
                stderr_text,
                mutation_warnings,
                loop_state,
            ) = execute_cell(
                source,
                inputs,
                mutation_defines=manifest.get("mutation_defines") or [],
                loop_until_expr=loop_until_expr,
            )

        serialized: dict[str, Any] = {}
        for var_name, value in outputs.items():
            try:
                serialized[var_name] = _ser.serialize_value(value, output_dir, var_name)
            except Exception as e:
                serialized[var_name] = {"error": str(e), "type": type(value).__name__}

        serialized_displays: list[dict[str, Any]] = []
        for index, value in enumerate(display_values):
            try:
                serialized_display = _ser.serialize_value(
                    value,
                    output_dir,
                    f"__display__{index}",
                )
            except Exception as e:
                serialized_display = {"error": str(e), "type": type(value).__name__}
            serialized_displays.append(serialized_display)

        if serialized_displays:
            serialized["_"] = serialized_displays[-1]

        result = {
            "success": True,
            "variables": serialized,
            "displays": serialized_displays,
            "stdout": stdout_text,
            "stderr": stderr_text,
            "mutation_warnings": mutation_warnings,
        }
        if loop_state is not None:
            result["loop"] = loop_state

    except Exception as e:
        # sys.exit(1) raises SystemExit, which triggers the finally
        # block — the parent gets exit=1 plus a result.json with
        # success=False. The exit code is informational; the parent
        # only checks for the presence + contents of result.json.
        result = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "variables": {},
            "stdout": stdout_text,
            "stderr": stderr_text,
            "mutation_warnings": [],
        }
        sys.exit(1)

    finally:
        close_client(ambient_client)
        # Write the harness output to a *separate* filename so it
        # can't be confused with the input manifest the parent wrote.
        # The previous "manifest.json" overlap forced a fragile
        # ``"success" not in result`` heuristic on the parent side to
        # tell "harness crashed before finally" from "this is still
        # the input we wrote" — fragile because adding any field
        # named ``success`` to the input manifest would silently
        # mask real crashes.
        #
        # Hyphen in the filename guarantees no collision with the
        # variable files: variables get the JSON-serialized output
        # written at ``output_dir / <var_name>.json``, and "harness-
        # result" can't be a Python identifier (so it can't be a
        # variable name).
        result_path = Path(manifest.get("output_dir", "/tmp/strata_output")) / "harness-result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        # orjson handles datetime, Decimal, numpy, UUID natively;
        # OPT_NON_STR_KEYS covers the rare case of non-string dict
        # keys in previews. default=str catches anything orjson can't
        # encode so previews never truncate the manifest mid-write.
        with open(result_path, "wb") as f:
            f.write(
                orjson.dumps(
                    result,
                    option=orjson.OPT_INDENT_2
                    | orjson.OPT_SERIALIZE_NUMPY
                    | orjson.OPT_NON_STR_KEYS,
                    default=str,
                )
            )


if __name__ == "__main__":
    main()
