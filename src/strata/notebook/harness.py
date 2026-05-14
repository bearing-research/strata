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
_display = _load_local_module("display_runtime.py", "_nb_display_runtime")


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
# Entry point
# ---------------------------------------------------------------------------


def main():
    if len(sys.argv) < 2:
        print("Usage: harness.py <manifest_path>", file=sys.stderr)
        sys.exit(1)

    manifest_path = sys.argv[1]
    manifest: dict = {}
    stdout_text = ""
    stderr_text = ""

    try:
        manifest = load_manifest(manifest_path)
        source = manifest.get("source", "")
        output_dir = Path(manifest.get("output_dir", "/tmp/strata_output"))

        inputs = deserialize_inputs(manifest)
        inject_mounts(manifest, inputs)
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
