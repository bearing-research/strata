#!/usr/bin/env python3
"""Pool worker script that runs in the notebook subprocess.

Single-shot: the worker imports common deps, sends a ``ready`` signal,
reads exactly one manifest path from stdin, executes it, prints the
JSON result, and exits. The parent (``WarmProcessPool``) kills the
worker after every cell to preserve per-cell isolation (sys.path,
os.environ, and module-state cannot leak between cells), so any
reuse loop here would be unreachable.

This script runs in the notebook's venv and cannot ``import strata``.
Serialization is delegated to ``serializer.py`` in the same directory,
loaded via ``importlib.util``.
"""

import importlib.util
import io
import os
import sys
import tomllib
from contextlib import contextmanager
from pathlib import Path
from typing import Any

# orjson ships as a required dep in every notebook pyproject.toml we
# generate, so it's guaranteed to be importable in the venv this
# pool worker runs in. Native datetime / numpy / Decimal support
# means we don't have to paper over exotic types at the application
# level like stdlib json forced us to.
import orjson


def _dumps_result(result: dict) -> str:
    """Encode a harness result for stdout.

    default=str catches anything orjson can't encode (previews are
    display-only, so stringifying exotic values is safe). Returns str
    rather than bytes so callers can use ``print(..., flush=True)``.
    """
    return orjson.dumps(
        result,
        option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NON_STR_KEYS,
        default=str,
    ).decode("utf-8")


# ---------------------------------------------------------------------------
# Load the shared serializer
# ---------------------------------------------------------------------------


def _load_local_module(filename: str, module_name: str):
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
# Warm-up helpers
# ---------------------------------------------------------------------------


def parse_common_imports() -> list[str]:
    """Parse pyproject.toml to find packages to pre-import."""
    try:
        notebook_dir = Path(sys.argv[1])
        pyproject_path = notebook_dir / "pyproject.toml"
        if not pyproject_path.exists():
            return []

        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        dependencies = data.get("project", {}).get("dependencies", [])
        result = []
        for dep in dependencies:
            name = dep.split("[")[0].split(";")[0]
            for sep in (">=", "==", "!=", "~=", "<", ">"):
                name = name.split(sep)[0]
            result.append(name.strip())
        return result
    except Exception:
        return []


def warm_imports(imports: list[str]) -> None:
    for module_name in imports:
        try:
            __import__(module_name)
        except (ImportError, ModuleNotFoundError):
            pass


# ---------------------------------------------------------------------------
# Cell execution (mirrors harness.py logic)
# ---------------------------------------------------------------------------


def _exec_with_display(source: str, namespace: dict) -> Any | None:
    """Execute source; if the last statement is a bare expression, eval and return it."""
    import ast as _ast

    try:
        tree = _ast.parse(source)
    except SyntaxError:
        exec(source, namespace)
        return None

    if not tree.body:
        return None

    last = tree.body[-1]
    if isinstance(last, _ast.Expr):
        if len(tree.body) > 1:
            mod = _ast.Module(body=tree.body[:-1], type_ignores=[])
            _ast.fix_missing_locations(mod)
            exec(compile(mod, "<cell>", "exec"), namespace)
        expr = _ast.Expression(body=last.value)
        _ast.fix_missing_locations(expr)
        result = eval(compile(expr, "<cell>", "eval"), namespace)
        return result if result is not None else None
    else:
        exec(source, namespace)
        return None


@contextmanager
def _apply_env_overrides(manifest: dict):
    """Apply manifest-scoped environment overrides for one worker execution."""
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


def _inject_mounts(manifest: dict, namespace: dict[str, Any]) -> None:
    """Inject prepared mount paths into the warm worker namespace."""
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


def execute_harness(manifest: dict) -> dict:
    """Execute a cell manifest and return the result dict."""
    source = manifest.get("source", "")
    inputs = manifest.get("inputs", {})
    output_dir = Path(manifest.get("output_dir", ""))

    namespace: dict[str, Any] = {}
    display_capture = _display.DisplayCapture()

    # Deserialize inputs
    for var_name, spec in inputs.items():
        content_type = spec.get("content_type", "")
        file_name = spec.get("file", "")
        if not file_name:
            continue
        full_path = output_dir / file_name
        if not full_path.exists():
            continue
        try:
            namespace[var_name] = _ser.deserialize_value(content_type, full_path)
        except Exception as exc:
            print(f"Error deserializing {var_name}: {exc}", file=sys.stderr)

    _inject_mounts(manifest, namespace)
    display_capture.install(namespace)

    namespace_before = set(namespace.keys())
    input_identities = {name: id(namespace[name]) for name in namespace_before}
    input_snapshots = _immut.snapshot_inputs(namespace, list(namespace_before))
    mutation_set = set(manifest.get("mutation_defines") or [])

    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    sys.stdout = stdout_buf
    sys.stderr = stderr_buf

    _skip = {"__builtins__", "__name__", "__doc__", "__package__"}

    try:
        with _apply_env_overrides(manifest):
            with display_capture.capture_side_effects():
                _display_value = _exec_with_display(source, namespace)

        sys.stdout = old_stdout
        sys.stderr = old_stderr

        variables: dict[str, Any] = {}
        for key, value in namespace.items():
            if key.startswith("_") or key in _skip:
                continue
            if (
                key not in namespace_before
                or id(value) != input_identities.get(key)
                or key in mutation_set
            ):
                try:
                    variables[key] = _ser.serialize_value(value, output_dir, key)
                except Exception as e:
                    variables[key] = {"content_type": "error", "error": str(e)}

        display_values = display_capture.resolve(_display_value)
        serialized_displays: list[dict[str, Any]] = []
        for index, value in enumerate(display_values):
            try:
                serialized_displays.append(
                    _ser.serialize_value(value, output_dir, f"__display__{index}")
                )
            except Exception:
                continue

        mutation_warnings = list(_immut.detect_mutations(namespace, input_snapshots))

        return {
            "success": True,
            "variables": {
                **variables,
                **({"_": serialized_displays[-1]} if serialized_displays else {}),
            },
            "displays": serialized_displays,
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "error": None,
            "mutation_warnings": mutation_warnings,
        }

    except Exception as e:
        import traceback

        sys.stdout = old_stdout
        sys.stderr = old_stderr
        return {
            "success": False,
            "variables": {},
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            "mutation_warnings": [],
        }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    try:
        imports = parse_common_imports()
        warm_imports(imports)

        print("ready", flush=True)

        line = sys.stdin.readline()
        if not line:
            return
        manifest_path = line.strip()
        if not manifest_path:
            return

        try:
            with open(manifest_path, "rb") as f:
                manifest = orjson.loads(f.read())
            result = execute_harness(manifest)
            print(_dumps_result(result), flush=True)
        except Exception as e:
            print(
                _dumps_result(
                    {
                        "success": False,
                        "variables": {},
                        "stdout": "",
                        "stderr": "",
                        "error": f"Pool worker error: {e}",
                        "mutation_warnings": [],
                    }
                ),
                flush=True,
            )

    except Exception as e:
        print(f"fatal: {e}", file=sys.stderr, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
