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

# Sentinel for "input could not be deserialized" (distinct from a real None).
_MISSING = object()
_immut = _load_local_module("immutability.py", "_nb_immutability")
_display = _load_local_module("display/runtime.py", "_nb_display_runtime")
_client_mod = _load_local_module("notebook_client.py", "_nb_client")


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


def _inject_tables(manifest: dict, namespace: dict[str, Any]) -> None:
    """Inject ``@table`` lake inputs into the warm worker namespace.

    Mirrors ``harness.inject_tables``: each declaration becomes ``<name>``
    (the table URI) and ``<name>_snapshot`` (the resolved snapshot id). The
    standalone harness path injects both mounts and tables; the warm pool
    worker must too, or an ``@table`` cell run through the pool fails with
    ``NameError`` for the injected variable.
    """
    tables = manifest.get("tables", {})
    for table_name, spec in tables.items():
        namespace[table_name] = spec.get("uri", "")
        namespace[f"{table_name}_snapshot"] = spec.get("snapshot_id")


def _inject_client(manifest: dict, namespace: dict[str, Any]) -> Any:
    """Inject an ambient ``strata`` client into the warm worker namespace.

    Mirrors ``harness.inject_client``. Returns the client so the caller
    can close it — the warm worker process is reused across cells, so a
    leaked ``httpx.Client`` would accumulate sockets. ``None`` if no
    ``strata_url`` is set.
    """
    url = manifest.get("strata_url")
    if not url:
        return None
    # Path-loaded, not ``import strata`` — the warm worker runs in the
    # notebook venv (pyarrow + stdlib only); see notebook_client.py.
    cell_id = manifest.get("strata_cell_id") or manifest.get("cell_id")
    # Auth headers when the client targets a remote shared store (empty locally).
    # The warm pool is the default WS path, so dropping these broke auth / tenant
    # scoping for shared-store cells. Mirror harness.inject_client.
    headers = manifest.get("strata_headers") or None
    client = _client_mod.StrataClient(base_url=url, cell_id=cell_id, headers=headers)
    namespace["strata"] = client
    return client


def _close_client(client: Any) -> None:
    """Close an injected ambient client, swallowing teardown errors."""
    if client is None:
        return
    try:
        client.close()
    except Exception:
        pass


def execute_harness(manifest: dict) -> dict:
    """Execute a cell manifest and return the result dict."""
    source = manifest.get("source", "")
    inputs = manifest.get("inputs", {})
    output_dir = Path(manifest.get("output_dir", ""))

    namespace: dict[str, Any] = {}
    display_capture = _display.DisplayCapture()

    old_stdout, old_stderr = sys.stdout, sys.stderr
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    _skip = {"__builtins__", "__name__", "__doc__", "__package__"}
    ambient_client: Any = None

    try:
        # Deserialize inputs. Done inside the try/except so a
        # StrataRArtifactError (R-only RDS artifact consumed by a
        # Python cell) produces the same structured error result as
        # any other failure — previously this loop ran outside the
        # try and a swallowed RDS error regressed to ``NameError: fit``
        # once the cell body ran.
        def _deser_one(spec: dict, var_name: str) -> Any:
            """Deserialize one ``{content_type, file}`` spec, or ``_MISSING``."""
            file_name = spec.get("file", "")
            if not file_name:
                return _MISSING
            full_path = output_dir / file_name
            if not full_path.exists():
                return _MISSING
            try:
                return _ser.deserialize_value(spec.get("content_type", ""), full_path)
            except _ser.StrataRArtifactError as exc:
                raise _ser.StrataRArtifactError(exc.file_path, variable_name=var_name) from exc
            except Exception as exc:
                print(f"Error deserializing {var_name}: {exc}", file=sys.stderr)
                return _MISSING

        for var_name, spec in inputs.items():
            # Sweep-group input → {variant_name: value} dict (parity with
            # harness.deserialize_inputs; a pooled @worker cell would otherwise
            # never bind the var and crash with NameError).
            if isinstance(spec, dict) and spec.get("kind") == "sweep_dict":
                bundle: dict[str, Any] = {}
                for variant_name, variant_spec in spec.get("variants", {}).items():
                    value = _deser_one(variant_spec, var_name)
                    if value is not _MISSING:
                        bundle[variant_name] = value
                namespace[var_name] = bundle
                continue
            value = _deser_one(spec, var_name)
            if value is not _MISSING:
                namespace[var_name] = value

        _inject_mounts(manifest, namespace)
        _inject_tables(manifest, namespace)
        ambient_client = _inject_client(manifest, namespace)
        display_capture.install(namespace)

        namespace_before = set(namespace.keys())
        input_identities = {name: id(namespace[name]) for name in namespace_before}
        input_snapshots = _immut.snapshot_inputs(namespace, list(namespace_before))
        mutation_set = set(manifest.get("mutation_defines") or [])

        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

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
    finally:
        _close_client(ambient_client)


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
