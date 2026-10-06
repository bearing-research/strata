"""Harness script that runs a cell inside the notebook subprocess.

Takes a manifest JSON path as argv[1], executes the cell source, captures
stdout/stderr and serializes outputs. Runs in the notebook's venv and cannot
``import strata``; ``serializer.py`` in the same directory is loaded via
``importlib.util``.
"""

from __future__ import annotations

import importlib.util
import io
import os
import platform
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# orjson serializes datetime, numpy scalars, Decimal and UUID natively (stdlib
# json truncated manifest.json on them). Every generated pyproject includes it.
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
_client_mod = _load_local_module("notebook_client.py", "_nb_client")

# Harness-injected, not user inputs: excluded from mutation fingerprinting.
_AMBIENT_NAMES = frozenset({"strata", *_display.DISPLAY_HELPER_NAMES})


# --- Manifest I/O ---


def load_manifest(manifest_path: str) -> dict:
    with open(manifest_path, "rb") as f:
        return orjson.loads(f.read())


# --- Input deserialization ---


def _deserialize_one(var_name: str, spec: dict, output_dir: Path) -> Any:
    """Deserialize one ``{content_type, file}`` spec, or ``_MISSING`` if the file is absent."""
    file_name = spec.get("file", "")
    if not file_name:
        print(f"Warning: no file path for input {var_name}", file=sys.stderr)
        return _MISSING
    full_path = output_dir / file_name
    if not full_path.exists():
        print(f"Warning: input file not found: {full_path}", file=sys.stderr)
        return _MISSING
    try:
        # A module/cell export may carry injected upstream values its defs close
        # over; deserialize those and hydrate the synthetic module.
        injected_specs = spec.get("injected")
        if spec.get("content_type", "") == "module/cell" and injected_specs:
            injected: dict[str, Any] = {}
            for inj_name, inj_spec in injected_specs.items():
                value = _deserialize_one(f"{var_name}__inj__{inj_name}", inj_spec, output_dir)
                if value is not _MISSING:
                    injected[inj_name] = value
            return _ser.deserialize_cell_module_with_injection(full_path, injected)
        return _ser.deserialize_value(spec.get("content_type", ""), full_path)
    except _ser.StrataPrecisionError as e:
        # Attach the variable name; otherwise this surfaces as a bare NameError.
        raise _ser.StrataPrecisionError(
            e.stored_dtype, e.reconstructed_dtype, variable_name=var_name
        ) from e
    except _ser.StrataRArtifactError as e:
        # R-only payload: attach the variable name so the cell fails loudly instead
        # of hitting an unhelpful NameError later.
        raise _ser.StrataRArtifactError(e.file_path, variable_name=var_name) from e
    except Exception as e:
        print(f"Error deserializing {var_name}: {e}", file=sys.stderr)
        return _MISSING


_MISSING = object()


def deserialize_inputs(manifest: dict) -> dict[str, Any]:
    """Deserialize input variables listed in the manifest.

    A normal input is a single ``{content_type, file}`` spec. A sweep-group
    input is ``{"kind": "sweep_dict", "variants": {name: spec}}`` and binds to a
    ``{variant_name: value}`` dict.
    """
    output_dir = Path(manifest.get("output_dir", "/tmp/strata_output"))
    inputs: dict[str, Any] = {}

    for var_name, spec in manifest.get("inputs", {}).items():
        if isinstance(spec, dict) and spec.get("kind") == "sweep_dict":
            bundle: dict[str, Any] = {}
            for variant_name, variant_spec in spec.get("variants", {}).items():
                value = _deserialize_one(var_name, variant_spec, output_dir)
                if value is not _MISSING:
                    bundle[variant_name] = value
            inputs[var_name] = bundle
            continue

        value = _deserialize_one(var_name, spec, output_dir)
        if value is not _MISSING:
            inputs[var_name] = value

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
    """Bind each mount as a ``pathlib.Path`` in the cell namespace.

    Read-only mounts must exist; read-write mounts are created if missing.
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
    """Inject ``@table`` inputs: ``<name>`` (the URI) and ``<name>_snapshot`` (the resolved id)."""
    tables = manifest.get("tables", {})
    for table_name, spec in tables.items():
        namespace[table_name] = spec.get("uri", "")
        namespace[f"{table_name}_snapshot"] = spec.get("snapshot_id")


def inject_client(manifest: dict, namespace: dict) -> Any:
    """Inject an ambient ``strata`` client bound to the server URL.

    Returns the client so the caller can close it, or ``None`` when no
    ``strata_url`` is set. Call before the namespace is snapshotted, so
    ``strata`` counts as an injected input rather than a cell output.
    """
    url = manifest.get("strata_url")
    if not url:
        return None
    # Path-loaded, not ``import strata``: the notebook venv has only pyarrow + stdlib.
    cell_id = manifest.get("strata_cell_id") or manifest.get("cell_id")
    # Auth headers when the client targets a remote shared store (empty locally).
    headers = manifest.get("strata_headers") or None
    # Lets ``strata.promote("rows")`` name an input the way the cell reads it.
    input_uris = {
        name: spec.get("uri", "")
        for name, spec in (manifest.get("inputs") or {}).items()
        if isinstance(spec, dict) and spec.get("uri")
    }
    client = _client_mod.StrataClient(
        base_url=url,
        cell_id=cell_id,
        headers=headers,
        promote_url=manifest.get("strata_promote_url") or None,
        inputs=input_uris,
    )
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
    stdout_capture: io.StringIO | None = None,
    stderr_capture: io.StringIO | None = None,
) -> tuple[dict, list[Any], str, str, list[dict], dict[str, Any] | None]:
    """Execute a cell; return outputs, displays, captured streams, mutations and loop state.

    ``mutation_defines`` (in-place mutations such as ``df["col"] = ...``) are always
    serialized, even when ``id()`` is unchanged. With ``loop_until_expr`` set, the
    expression is evaluated after the body and ``loop_state`` carries
    ``until_reached`` (or ``error``); otherwise ``loop_state`` is ``None``.
    Captured text is returned only on success, so a caller that needs the output of
    a cell that raised passes its own ``stdout_capture`` / ``stderr_capture``.
    """
    namespace = dict(inputs)
    display_capture = _display.DisplayCapture()
    display_capture.install(namespace)
    mutation_set = set(mutation_defines or [])

    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout_capture = io.StringIO() if stdout_capture is None else stdout_capture
    stderr_capture = io.StringIO() if stderr_capture is None else stderr_capture

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        namespace_before = set(namespace.keys())
        input_identities = {name: id(namespace[name]) for name in namespace_before}
        input_snapshots = _immut.snapshot_inputs(
            namespace, [n for n in namespace_before if n not in _AMBIENT_NAMES]
        )

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

        # Only inputs mutated in place AND not exported: an exported mutation reaches
        # downstream correctly.
        mutation_warnings = list(
            _immut.detect_mutations(namespace, input_snapshots, exported_names=set(new_vars))
        )
        # Outputs sharing a mutable object decouple once stored as separate artifacts
        # (e.g. an optimizer over a model's parameters).
        mutation_warnings.extend(_immut.detect_shared_mutable_outputs(new_vars))
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

    Returns ``until_reached`` (bool) and, if compiling or evaluating fails, an
    ``error`` message, which the parent reports as the cell's failure.
    """
    try:
        code = compile(expr, "<loop_until>", "eval")
    except SyntaxError as exc:
        return {
            "until_reached": False,
            "error": f"@loop_until syntax error: {exc.msg}",
        }

    try:
        result = eval(code, namespace)  # noqa: S307 (the user declares the predicate)
    except Exception as exc:
        return {
            "until_reached": False,
            "error": f"@loop_until evaluation failed: {type(exc).__name__}: {exc}",
        }

    return {"until_reached": bool(result)}


# --- Batch execution (run-all single-process mode) ---
#
# Many cells exec in one process with a shared namespace. The parent owns
# provenance, cache and persist; this harness runs bodies and serializes.
# Pipes: frame_out (harness -> parent, line-delimited JSON) and resp_in
# (parent -> harness, cache_check / persist responses). Blobs travel as files
# in ``output_dir/<cell_id>/{var_name}{ext}``, never inline.


_MISSING = object()


def build_env_identity() -> str:
    """Which interpreter, on which machine, produced these bytes.

    Recorded on every artifact, not hashed: folding the platform into the key would
    kill cross-machine (Mac locally, Linux in CI) cache hits. Computed here because
    this is the process that ran the cell. Not normalised: ``arm64`` and
    ``aarch64`` stay distinct, since nothing here can vouch they are portable.
    """
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        f"{platform.python_implementation().lower()}-{version}-{sys.platform}-{platform.machine()}"
    )


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
    """Execute one cell within a batch; return ``(status, failed_reason)``.

    ``status`` is ``"ok"``, ``"cell_error"`` or ``"persist_failed"``;
    ``failed_reason`` matches ``batch_end``'s ``reason`` field.
    """
    cell_id = cell["cell_id"]
    source = cell["source"]
    consumed_vars: list[str] = list(cell.get("consumed_vars") or [])
    references: list[str] = list(cell.get("references") or [])
    cell_env: dict = cell.get("env") or {}
    mount_manifest: dict = cell.get("mount_manifest") or {}
    table_manifest: dict = cell.get("table_manifest") or {}
    source_hash: str = cell.get("source_hash", "")
    env_hash: str = cell.get("env_hash", "")

    cell_output_dir = output_dir / cell_id
    cell_output_dir.mkdir(parents=True, exist_ok=True)

    _send_frame(frame_out, "cell_start", {"cell_id": cell_id})

    # Saved so a user variable named like a mount is restored on exit.
    mount_names = list(mount_manifest.keys())
    table_names = [injected for name in table_manifest for injected in (name, f"{name}_snapshot")]
    ambient_names = ["strata"] if cell.get("strata_url") else []
    previous_bindings: dict[str, Any] = {
        name: namespace.get(name, _MISSING) for name in mount_names + table_names + ambient_names
    }
    inject_mounts({"mounts": mount_manifest}, namespace)
    inject_tables({"tables": table_manifest}, namespace)
    ambient_client = None

    try:
        with apply_env_overrides({"env": cell_env}):
            # Parent decides hit/miss.
            _send_frame(frame_out, "cache_check", {"cell_id": cell_id})
            response = _recv_response(resp_in)

            # After the response, which carries every input's artifact uri (needed by
            # ``strata.promote("rows")``). Before the namespace snapshot below, so
            # ``strata`` counts as an injected input, not a cell output.
            ambient_client = inject_client(
                {**cell, "inputs": response.get("input_uris") or {}}, namespace
            )

            if response.get("cache_hit"):
                # Parent has already materialized blobs into cell_output_dir.
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
                        # A leaf's replayed console, as single-cell returns it on a hit.
                        "stdout": response.get("stdout", ""),
                        "stderr": response.get("stderr", ""),
                    },
                )
                return ("ok", None)

            # Cache miss: execute the body.
            #
            # ``DisplayCapture.install`` uses ``setdefault``, and the namespace persists
            # across batch cells, so clear the display keys or the new handlers never install.
            for _display_key in _display.DISPLAY_HELPER_NAMES:
                namespace.pop(_display_key, None)
            display_capture = _display.DisplayCapture()
            display_capture.install(namespace)
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            # Batch can't recapture in-place input mutations (the DAG is static), so
            # this is warn-only.
            input_snapshots = _immut.snapshot_inputs(namespace, references)
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

                # Catches mutations the static analyzer can't see (aliases, helper-fn
                # mutation, bare mutators).
                mutation_warnings = list(_immut.detect_mutations(namespace, input_snapshots))
                mutation_warnings.extend(
                    _immut.detect_shared_mutable_outputs(
                        {vn: namespace[vn] for vn in consumed_vars if vn in namespace}
                    )
                )

                # Known gap: instances of in-batch-defined classes serialize as
                # ``pickle/object``, not ``module/cell-instance`` (single-cell tags the class
                # in the parent before serialization). Content type only; the value
                # round-trips.
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
                written = [
                    (namespace[name], payload)
                    for name, payload in outputs.items()
                    if name in namespace
                ]
                serialized_displays: list[dict[str, Any]] = []
                for idx, display in enumerate(display_values):
                    try:
                        # Applies the ``__display__N`` naming the serializer detects (``display_N``
                        # would be a plain pickle) and reuses the payload when the display is one of
                        # the variables just written.
                        meta = _ser.serialize_display_value(display, cell_output_dir, idx, written)
                        serialized_displays.append(meta)
                    except Exception:
                        # Display serialization errors don't abort the cell, as in single-cell.
                        pass
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

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
                    "mutation_warnings": mutation_warnings,
                    "build_env": build_env_identity(),
                },
            )
            ack = _recv_response(resp_in)
            if not ack.get("ok"):
                return ("persist_failed", "persist_failed")
            return ("ok", None)
    finally:
        # The batch process is reused across cells, so close the client (no leaked
        # sockets) and restore pre-cell name bindings.
        close_client(ambient_client)
        for name, previous in previous_bindings.items():
            if previous is _MISSING:
                namespace.pop(name, None)
            else:
                namespace[name] = previous


def _seed_upstream_namespace(
    upstream_inputs: dict,
    output_dir: Path,
    namespace: dict[str, Any],
    tainted_inputs: dict[str, Exception],
) -> None:
    """Load a batch's non-batched upstream artifacts into the shared namespace.

    Done inline (not via ``deserialize_inputs``) so one unreadable artifact (e.g.
    R-only) does not kill the subprocess before ``cell_start``. Per-variable
    failures surface as a ``cell_error`` on the first cell that references them.
    """
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
        except _ser.StrataPrecisionError as exc:
            tainted_inputs[var_name] = _ser.StrataPrecisionError(
                exc.stored_dtype, exc.reconstructed_dtype, variable_name=var_name
            )
        except _ser.StrataRArtifactError as exc:
            tainted_inputs[var_name] = _ser.StrataRArtifactError(
                exc.file_path, variable_name=var_name
            )
        except Exception as exc:
            print(f"Error deserializing {var_name}: {exc}", file=sys.stderr)


def execute_batch(
    cells: list[dict],
    upstream_inputs: dict,
    output_dir: Path,
    frame_out: Any,
    resp_in: Any,
) -> None:
    """Execute a sequence of cells in one Python process.

    ``cells`` is ``[{cell_id, source, consumed_vars, env, mount_manifest,
    source_hash, env_hash}, ...]`` in notebook order. ``upstream_inputs`` seeds the
    shared namespace (same shape ``deserialize_inputs`` reads). Streams frames to
    ``frame_out``, reads responses from ``resp_in``, and returns after
    ``batch_end``; the caller closes the pipes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    namespace: dict[str, Any] = {}
    tainted_inputs: dict[str, Exception] = {}

    # Deserializing imports value libraries, and some read config once at import
    # (jax and JAX_ENABLE_X64), configuring them for the whole one-process batch.
    # Apply only entries common to every cell: a value one cell sets and another
    # doesn't is ambiguous. The notebook-level ``[env]`` is the case that matters.
    batch_envs = [dict(cell.get("env") or {}) for cell in cells]
    shared_env = {
        key: value
        for key, value in (batch_envs[0] if batch_envs else {}).items()
        if all(env.get(key) == value for env in batch_envs)
    }

    with apply_env_overrides({"env": shared_env}):
        _seed_upstream_namespace(upstream_inputs, output_dir, namespace, tainted_inputs)

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
    tainted_inputs: dict[str, Exception],
) -> Exception | None:
    """Return the first tainted-upstream error that a cell's source references.

    A word-boundary regex (not an AST walk), so ``fit`` does not match
    ``unfit_data``.
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
    """Emit ``cell_start`` + ``cell_error`` for a cell blocked on a tainted upstream.

    Same frame shape as a body-level exception in ``_run_one_batched_cell``.
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


# --- Entry point ---


def batch_main() -> None:
    """Batch-mode entry: ``python harness.py --batch <manifest>``.

    Pipe fds come from ``STRATA_BATCH_FRAME_FD`` (write) and
    ``STRATA_BATCH_RESP_FD`` (read); the output dir from ``STRATA_BATCH_OUTPUT_DIR``.
    """
    if len(sys.argv) < 3:
        print("Usage: harness.py --batch <manifest_path>", file=sys.stderr)
        sys.exit(1)

    manifest_path = sys.argv[2]
    manifest = load_manifest(manifest_path)

    frame_fd = int(os.environ["STRATA_BATCH_FRAME_FD"])
    resp_fd = int(os.environ["STRATA_BATCH_RESP_FD"])
    output_dir = Path(os.environ["STRATA_BATCH_OUTPUT_DIR"])

    frame_out = os.fdopen(frame_fd, "wb")
    resp_in = os.fdopen(resp_fd, "rb")

    execute_batch(
        cells=manifest.get("cells", []),
        upstream_inputs=manifest.get("upstream_inputs", {}),
        output_dir=output_dir,
        frame_out=frame_out,
        resp_in=resp_in,
    )

    # Flush + close so the parent's reader sees EOF promptly.
    frame_out.flush()
    frame_out.close()


class _Tee(io.StringIO):
    """Captures the cell's output and also writes it through to the wrapped stream.

    The result manifest needs the full text, and a worker streams a running cell's
    console by reading this process's stdout pipe. Flushed on every write so the
    console is live.
    """

    def __init__(self, stream: Any) -> None:
        super().__init__()
        self._stream = stream

    def write(self, text: str) -> int:
        # The capture is the result's source, so a broken pipe must not lose it.
        try:
            self._stream.write(text)
            self._stream.flush()
        except (ValueError, OSError):
            pass
        return super().write(text)


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
    # Owned here, not in execute_cell: a raising cell never returns its captured
    # streams, and the print trail matters most on failure.
    # Plain unless the manifest asks to write through: that doubles memory in
    # the reader and only helps when the reader forwards output somewhere.
    stdout_buffer: io.StringIO = io.StringIO()
    stderr_buffer: io.StringIO = io.StringIO()
    ambient_client: Any = None

    try:
        manifest = load_manifest(manifest_path)
        if manifest.get("stream_console"):
            stdout_buffer = _Tee(sys.stdout)
            stderr_buffer = _Tee(sys.stderr)
        source = manifest.get("source", "")
        output_dir = Path(manifest.get("output_dir", "/tmp/strata_output"))

        # Before deserializing, not just around the body: a library that reads
        # config once at import (jax and JAX_ENABLE_X64) would otherwise take the
        # server's env and silently downcast float64 inputs to float32.
        with apply_env_overrides(manifest):
            inputs = deserialize_inputs(manifest)
            inject_mounts(manifest, inputs)
            inject_tables(manifest, inputs)
            ambient_client = inject_client(manifest, inputs)
            loop_config = manifest.get("loop") or {}
            loop_until_expr = (
                loop_config.get("until_expr") if isinstance(loop_config, dict) else None
            )
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
                stdout_capture=stdout_buffer,
                stderr_capture=stderr_buffer,
            )
            # Reported here: inputs deserialize before console capture starts.
            if _ser.x64_was_enabled_here():
                stderr_text += _ser.X64_NOTE

        serialized: dict[str, Any] = {}
        for var_name, value in outputs.items():
            try:
                serialized[var_name] = _ser.serialize_value(value, output_dir, var_name)
            except Exception as e:
                serialized[var_name] = {"error": str(e), "type": type(value).__name__}

        # Lets the display loop reuse a payload when the last expression is a variable.
        written = [
            (value, serialized[name]) for name, value in outputs.items() if name in serialized
        ]
        serialized_displays: list[dict[str, Any]] = []
        for index, value in enumerate(display_values):
            try:
                serialized_display = _ser.serialize_display_value(
                    value,
                    output_dir,
                    index,
                    written,
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
            "build_env": build_env_identity(),
        }
        if loop_state is not None:
            result["loop"] = loop_state

    except Exception as e:
        # SystemExit still runs the finally block; the parent checks result.json, not the exit code.
        result = {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "variables": {},
            "stdout": stdout_buffer.getvalue() or stdout_text,
            "stderr": stderr_buffer.getvalue() or stderr_text,
            "mutation_warnings": [],
        }
        sys.exit(1)

    finally:
        close_client(ambient_client)
        # A separate name from the input manifest, so the parent can tell a crash
        # from unread input. The hyphen can't be a Python identifier, so it can't
        # collide with a variable's ``<var_name>.json``.
        result_path = Path(manifest.get("output_dir", "/tmp/strata_output")) / "harness-result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        # default=str catches what orjson can't encode, so the write never truncates.
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
